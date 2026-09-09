from __future__ import annotations

import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.config import get_settings
from app.uploads import parse_multipart
from rag.constants import SUPPORTED_SUFFIXES


STATIC_DIR = Path(__file__).resolve().parent / "web_static"
logger = logging.getLogger(__name__)


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class RagWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_web_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    from app.uploads import get_upload_store

    get_upload_store().recover()
    server = RagWebServer((host, port), WebRequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


class WebRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGWeb/1.0"

    def do_GET(self) -> None:
        self._run_request(self._dispatch_get)

    def do_POST(self) -> None:
        self._run_request(self._dispatch_post)

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_static_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            self._send_json(self._status_payload())
        elif parsed.path == "/api/documents":
            self._handle_documents()
        elif parsed.path == "/api/upload-jobs":
            from app.uploads import get_upload_store

            self._send_json({"jobs": get_upload_store().snapshot()})
        elif parsed.path.startswith("/api/answer-jobs/"):
            self._handle_answer_job(parsed.path.rsplit("/", 1)[-1])
        elif parsed.path.startswith("/static/"):
            static_path = STATIC_DIR / parsed.path.removeprefix("/static/")
            self._send_static_file(static_path, self._content_type(static_path))
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")

    def _dispatch_post(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/ask":
            self._handle_ask()
        elif parsed.path == "/api/documents/upload":
            self._handle_document_upload()
        elif parsed.path == "/api/answer-jobs":
            self._handle_create_answer_job()
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")

    def _run_request(self, handler: Callable[[], None]) -> None:
        try:
            handler()
        except RequestError as exc:
            self._send_error(exc.status, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected from %s", self.client_address[0])
        except Exception:
            logger.exception("Unhandled Web request error for %s", self.path)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_ask(self) -> None:
        from app.service import answer_question

        payload = self._read_json()
        question_value = payload.get("question", "")
        if not isinstance(question_value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "question must be a string")
        question = question_value.strip()

        if not question:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Question is required")

        settings = get_settings()
        top_k, similarity_threshold, retrieval_method = self._read_retrieval_params(payload)
        result = answer_question(
            question,
            settings,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            retrieval_method=retrieval_method,
        )
        self._send_json(
            {
                "question": question,
                "retrieval_method": result["retrieval_method"],
                "answer": result["answer"],
                "vector_context": result.get("vector_context", []),
                "vector_hits": result.get("vector_hits", []),
            }
        )

    def _handle_create_answer_job(self) -> None:
        from app.answer_jobs import get_answer_store

        payload = self._read_json()
        question_value = payload.get("question", "")
        if not isinstance(question_value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "question must be a string")
        try:
            top_k, similarity_threshold, retrieval_method = self._read_retrieval_params(payload)
            job = get_answer_store().create(
                question_value,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                retrieval_method=retrieval_method,
            )
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(job, status=HTTPStatus.ACCEPTED)

    def _handle_answer_job(self, job_id: str) -> None:
        from app.answer_jobs import get_answer_store

        try:
            job = get_answer_store().get(job_id)
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        self._send_json(job)

    def _handle_documents(self) -> None:
        from app.service import list_indexed_documents

        settings = get_settings()
        if not (settings.chroma_persist_dir / "chroma.sqlite3").is_file():
            self._send_json({"documents": []})
            return
        self._send_json({"documents": list_indexed_documents(settings)})

    def _handle_document_upload(self) -> None:
        from app.uploads import get_upload_store

        settings = get_settings()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from exc
        if content_length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Upload body is required")
        if content_length > settings.max_upload_bytes:
            self.close_connection = True
            raise RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Upload body is too large",
            )

        body = self.rfile.read(content_length)
        try:
            files = parse_multipart(
                self.headers.get("Content-Type", ""),
                body,
                supported_suffixes=SUPPORTED_SUFFIXES,
                max_files=settings.max_upload_files,
            )
            job = get_upload_store().create_uploaded(files, settings.data_dir)
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except RuntimeError as exc:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc

        self._send_json(job, status=HTTPStatus.ACCEPTED)

    def _status_payload(self) -> dict[str, object]:
        settings = get_settings()
        supported_files = []
        if settings.data_dir.exists():
            supported_files = [
                str(path)
                for path in sorted(settings.data_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ]

        return {
            "has_openai_api_key": bool(settings.openai_api_key),
            "data_dir": str(settings.data_dir),
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "embedding_base_url": settings.embedding_base_url or settings.openai_base_url,
            "ocr_model": settings.ocr_model,
            "ocr_max_pages": settings.ocr_max_pages,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_method": settings.retrieval_method,
            "similarity_threshold": settings.similarity_threshold,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_upload_files": settings.max_upload_files,
            "max_upload_jobs": settings.max_upload_jobs,
            "openai_base_url": settings.openai_base_url,
            "supported_files": supported_files,
            "chroma_exists": (settings.chroma_persist_dir / "chroma.sqlite3").is_file(),
        }

    def _read_json(self) -> dict[str, object]:
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "application/json":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from exc
        if content_length < 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
        if content_length > get_settings().max_request_bytes:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")
        if content_length == 0:
            return {}

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return payload

    @staticmethod
    def _read_bool(payload: dict[str, object], key: str) -> bool:
        value = payload.get(key, False)
        if not isinstance(value, bool):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{key} must be a boolean")
        return value

    @staticmethod
    def _read_retrieval_params(
        payload: dict[str, object],
    ) -> tuple[int | None, float | None, str | None]:
        retrieval_method = payload.get("retrieval_method")
        if "retrieval_method" in payload and retrieval_method not in ("vector", "bm25"):
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "retrieval_method must be vector or bm25"
            )
        top_k = payload.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise RequestError(HTTPStatus.BAD_REQUEST, "top_k must be an integer")
            if top_k <= 0 or top_k > 100:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "top_k must be between 1 and 100"
                )
        similarity_threshold = payload.get("similarity_threshold")
        if similarity_threshold is not None:
            if isinstance(similarity_threshold, bool) or not isinstance(
                similarity_threshold, (int, float)
            ):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "similarity_threshold must be a number",
                )
            if not 0.0 <= similarity_threshold <= 1.0:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "similarity_threshold must be between 0 and 1",
                )
        return top_k, similarity_threshold, retrieval_method

    def _send_json(self, payload: dict[str, object], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static_file(self, path: Path, content_type: str) -> None:
        resolved = path.resolve()
        if not resolved.is_file() or STATIC_DIR.resolve() not in resolved.parents:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".css":
            return "text/css; charset=utf-8"
        if suffix == ".js":
            return "application/javascript; charset=utf-8"
        return "application/octet-stream"

    def _send_security_headers(self) -> None:
        self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
