from __future__ import annotations

import hmac
import json
import logging
import re
import signal
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from app.access import UserContext, known_groups, load_access_policy
from app.config import get_settings
from app.uploads import parse_multipart
from rag.constants import SUPPORTED_SUFFIXES
from rag import markdown_cache
from rag.index_manifest import read_index_embedding_model


STATIC_DIR = Path(__file__).resolve().parent / "web_static"
logger = logging.getLogger(__name__)


def _actor_name(user: UserContext | None) -> str:
    """Identity recorded in the audit trail and as a conversation owner."""

    from app.conversations import owner_of

    return owner_of(user)


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class RagWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: object, client_address: object) -> None:
        """连接级异常走 logger，别把 traceback 喷进 stderr。

        线程数无上限，每个连接又都有读超时——公网上一堆扫描器只把连接开一半就
        晾着，超时回收是常态。这些不是「程序出错」，默认的 traceback 只会把
        launchd 的日志刷满。
        """

        logger.info("连接异常已回收 (%s): %s", client_address, sys.exc_info()[1])


def run_web_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    from app.uploads import get_upload_store

    # 常驻服务的可观测性基本只有这个日志：launchd 把 stderr 收进
    # StandardErrorPath。不配 basicConfig 的话 root logger 是 WARNING 级且没有
    # handler，logger.info 会被无声丢掉——包括下面的停机信息、连接回收和
    # 客户端断开记录。basicConfig 已有 handler 时是空操作，所以 CLI 重复调用无害。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    get_upload_store().recover()
    server = RagWebServer((host, port), WebRequestHandler)

    # 收到 SIGTERM/SIGINT 走 shutdown()，而不是让默认行为当场结束进程：先停止
    # 接受新连接、把手上的请求放完。shutdown() 与 serve_forever() 不能在同一个
    # 线程里互相等，所以丢给一个短命线程。
    #
    # 注意别把这里当成「保护索引写入」——索引跑在 ThreadPoolExecutor（uploads.py）
    # 里，它的 worker 是非 daemon，解释器退出时 concurrent.futures 的 atexit 会
    # join 它们，所以排队中的入库本来就会跑完。真正的坑是**退出可能很久**：那些
    # 排队任务要先跑完，超过 launchd 默认 20s 的 ExitTimeOut 就会被 SIGKILL，
    # 那才是真的硬杀。所以部署时 plist 里必须调大 ExitTimeOut（见 deploy/）。
    def _stop(signum: int, _frame: object) -> None:
        logger.info("收到信号 %s，正在停止…", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for _signal in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_signal, _stop)

    try:
        server.serve_forever()
    finally:
        server.server_close()
        logger.info("服务已停止")


class WebRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGWeb/1.0"

    # 连接空闲超过这个秒数就断开。线程数无上限，不设超时的话，只把连接开一半
    # 晾着就能一直占着线程。按「每次读」计时，所以 100MB 上传只要持续有数据就
    # 不会被打断，只能挡住真正的停滞连接。
    timeout = 60

    def do_GET(self) -> None:
        self._run_request(self._dispatch_get)

    def do_POST(self) -> None:
        self._run_request(self._dispatch_post)

    def do_DELETE(self) -> None:
        self._run_request(self._dispatch_delete)

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_static_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            # 顶栏状态条对所有用户都要显示，所以是「认证即可」而非仅管理员。
            # 走 _current_user 而不是 _require_admin：开放模式下它返回 None 且
            # 不报错，本地单机不受影响。
            self._send_json(self._status_payload(self._current_user(get_settings())))
        elif parsed.path == "/api/me":
            self._handle_me()
        elif parsed.path == "/api/config":
            self._handle_get_config()
        elif parsed.path == "/api/audit":
            self._handle_audit_log(parsed.query)
        elif parsed.path == "/api/staging":
            self._handle_staging_list()
        elif parsed.path == "/api/conversations":
            self._handle_conversations()
        elif parsed.path.startswith("/api/conversations/"):
            self._handle_conversation_detail(
                parsed.path.removeprefix("/api/conversations/")
            )
        elif parsed.path == "/api/documents":
            self._handle_documents()
        elif parsed.path.startswith("/api/documents/"):
            self._handle_document_detail(
                parsed.path.removeprefix("/api/documents/"), parsed.query
            )
        elif parsed.path == "/api/upload-jobs":
            from app.uploads import get_upload_store

            self._require_admin()
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
        elif parsed.path == "/api/config":
            self._handle_update_config()
        elif parsed.path == "/api/models":
            self._handle_list_models()
        elif parsed.path == "/api/documents/upload":
            self._handle_document_upload()
        elif parsed.path.startswith("/api/documents/"):
            doc_id, _, action = parsed.path.removeprefix("/api/documents/").partition("/")
            if action != "access":
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            self._handle_document_access(doc_id)
        elif parsed.path.startswith("/api/staging/"):
            item_id, _, action = parsed.path.removeprefix("/api/staging/").partition("/")
            if action not in ("approve", "reject"):
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            self._handle_staging_decision(item_id, action)
        elif parsed.path == "/api/answer-jobs":
            self._handle_create_answer_job()
        elif parsed.path == "/api/conversations":
            self._handle_create_conversation()
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")

    def _dispatch_delete(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/documents/"):
            self._handle_document_delete(parsed.path.removeprefix("/api/documents/"))
        elif parsed.path.startswith("/api/conversations/"):
            self._handle_conversation_delete(
                parsed.path.removeprefix("/api/conversations/")
            )
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")

    def _require_trusted_proxy(self) -> None:
        """校验反向代理共享密钥（设置了才生效）。

        身份最终来自 X-User-ID 头，所以任何能连上后端端口的人都能冒充任何人。
        把后端绑在回环地址是第一道防线；这个密钥是第二道——后端端口万一被意外
        暴露（改错绑定、容器网络、SSH 转发），没有密钥仍然进不来。

        这是纵深防御，**不是**认证本身：没设密钥时直接放行，本地开发无感。
        """

        expected = get_settings().trusted_proxy_secret
        if not expected:
            return
        supplied = self.headers.get("X-Proxy-Secret", "")
        # 比较 bytes 而不是 str：str 版 compare_digest 遇到非 ASCII 会抛
        # TypeError，而 header 是按 latin-1 解出来的，攻击者发一个高位字节
        # （`X-Proxy-Secret: caf\xe9`）就能把 401 顶成 500 并往日志里灌 traceback。
        # 失败方向仍然是关闭的，只是没必要留这个噪音。
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise RequestError(HTTPStatus.UNAUTHORIZED, "请求未经过可信代理")

    def _run_request(self, handler: Callable[[], None]) -> None:
        try:
            self._require_trusted_proxy()
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
        settings = get_settings()
        user = self._current_user(settings)
        question_value = payload.get("question", "")
        if not isinstance(question_value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "question must be a string")
        question = question_value.strip()

        if not question:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Question is required")

        top_k, similarity_threshold, retrieval_method, candidate_pool = (
            self._read_retrieval_params(payload)
        )
        result = answer_question(
            question,
            settings,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            retrieval_method=retrieval_method,
            candidate_pool=candidate_pool,
            user=user,
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

    def _handle_get_config(self) -> None:
        from app.config import runtime_config_view

        settings = get_settings()
        self._require_admin()
        self._send_json({**runtime_config_view(), **self._index_model_info(settings)})

    def _handle_update_config(self) -> None:
        from app.config import runtime_config_view, update_runtime_config

        self._require_admin()
        settings = get_settings()
        updates = self._read_config_updates(self._read_json())
        try:
            update_runtime_config(updates)
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json({**runtime_config_view(), **self._index_model_info(settings)})

    def _handle_list_models(self) -> None:
        from app.config import endpoint_credentials
        from app.model_catalog import list_models

        self._require_admin()
        payload = self._read_json()
        target = payload.get("target")
        if target not in ("llm", "embedding"):
            raise RequestError(
                HTTPStatus.BAD_REQUEST, "target must be llm or embedding"
            )

        stored_base_url, stored_api_key = endpoint_credentials(target)
        raw_base_url = payload.get("base_url")
        base_url = (
            stored_base_url
            if raw_base_url is None
            else self._clean_base_url(raw_base_url, "base_url")
        )
        raw_api_key = payload.get("api_key")
        # An empty key means "use the saved one"; the UI never echoes secrets.
        api_key = (
            raw_api_key.strip()
            if isinstance(raw_api_key, str) and raw_api_key.strip()
            else stored_api_key
        )
        self._send_json({"target": target, **list_models(base_url, api_key)})

    def _handle_create_answer_job(self) -> None:
        from app.answer_jobs import get_answer_store
        from app.conversations import get_conversation_store, owner_of

        payload = self._read_json()
        settings = get_settings()
        user = self._current_user(settings)
        question_value = payload.get("question", "")
        if not isinstance(question_value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "question must be a string")

        conversation_id = payload.get("conversation_id")
        if conversation_id is not None:
            if not isinstance(conversation_id, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "conversation_id must be a string"
                )
            conversation_id = conversation_id.strip() or None
        # 不校验的话，A 能传 B 的 id，让 worker 把内容写进 B 的对话。
        if conversation_id and not get_conversation_store().owns(
            conversation_id, owner_of(user)
        ):
            raise RequestError(
                HTTPStatus.FORBIDDEN, "对话不存在或不属于当前用户"
            )

        try:
            top_k, similarity_threshold, retrieval_method, candidate_pool = (
                self._read_retrieval_params(payload)
            )
            job = get_answer_store().create(
                question_value,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                retrieval_method=retrieval_method,
                candidate_pool=candidate_pool,
                user=user,
                conversation_id=conversation_id,
            )
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(job, status=HTTPStatus.ACCEPTED)

    def _handle_answer_job(self, job_id: str) -> None:
        from app.answer_jobs import get_answer_store

        user = self._current_user(get_settings())
        try:
            job = get_answer_store().get_for_user(job_id, user)
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except PermissionError as exc:
            raise RequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
        self._send_json(job)

    def _handle_documents(self) -> None:
        from app.access import known_groups, load_access_policy
        from app.service import list_indexed_documents

        settings = get_settings()
        user = self._current_user(settings)
        policy = load_access_policy(settings.access_policy_file)
        is_admin = policy is not None and user is not None and policy.is_admin(user)
        access = {
            "enabled": policy is not None,
            "manageable": is_admin,
            "groups": sorted(known_groups(settings.access_policy_file))
            if policy is not None
            else [],
        }

        if not (settings.chroma_persist_dir / "chroma.sqlite3").is_file():
            self._send_json({"documents": [], "access": access})
            return
        self._send_json(
            {
                "documents": list_indexed_documents(settings, user, admin=is_admin),
                "access": access,
            }
        )

    def _handle_document_access(self, raw_doc_id: str) -> None:
        from app.service import set_document_access

        user = self._require_admin()
        doc_id = unquote(raw_doc_id).strip()
        group = self._read_json().get("group", "")
        if not isinstance(group, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "group must be a string")
        try:
            updated = set_document_access(
                get_settings(), doc_id, group, _actor_name(user)
            )
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, "Document not found") from exc
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(
            {
                "doc_id": doc_id,
                "access_group": group.strip(),
                "updated_chunks": updated,
            }
        )

    def _handle_audit_log(self, query: str) -> None:
        from app.audit import audit_log_path, read_entries

        self._require_admin()
        params = parse_qs(query)
        limit = self._read_query_int(
            params, "limit", default=100, minimum=1, maximum=1000
        )
        self._send_json(
            {
                "path": str(audit_log_path()),
                "entries": read_entries(limit),
            }
        )

    def _handle_conversations(self) -> None:
        from app.conversations import get_conversation_store, owner_of

        user = self._current_user(get_settings())
        conversations = get_conversation_store().list_for(owner_of(user))
        self._send_json({"conversations": conversations})

    def _handle_create_conversation(self) -> None:
        from app.conversations import get_conversation_store, owner_of

        user = self._current_user(get_settings())
        conversation = get_conversation_store().create(owner_of(user))
        self._send_json(conversation, status=HTTPStatus.CREATED)

    def _handle_conversation_detail(self, raw_id: str) -> None:
        from app.conversations import get_conversation_store, owner_of

        user = self._current_user(get_settings())
        conversation_id = unquote(raw_id).strip()
        if not conversation_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "conversation_id is required")
        try:
            payload = get_conversation_store().get(conversation_id, owner_of(user))
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, "Conversation not found") from exc
        except PermissionError as exc:
            raise RequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
        self._send_json(payload)

    def _handle_conversation_delete(self, raw_id: str) -> None:
        from app.conversations import get_conversation_store, owner_of

        user = self._current_user(get_settings())
        conversation_id = unquote(raw_id).strip()
        if not conversation_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "conversation_id is required")
        try:
            get_conversation_store().delete(conversation_id, owner_of(user))
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, "Conversation not found") from exc
        except PermissionError as exc:
            raise RequestError(HTTPStatus.FORBIDDEN, str(exc)) from exc
        self._send_json({"id": conversation_id, "deleted": True})

    def _handle_staging_list(self) -> None:
        from app.staging import get_staging_store

        settings = get_settings()
        self._require_admin()
        policy = load_access_policy(settings.access_policy_file)
        self._send_json(
            {
                "items": get_staging_store().snapshot(),
                "groups": sorted(known_groups(settings.access_policy_file))
                if policy is not None
                else [],
            }
        )

    def _handle_staging_decision(self, raw_item_id: str, action: str) -> None:
        from app.service import approve_staged_upload, reject_staged_upload

        user = self._require_admin()
        item_id = unquote(raw_item_id).strip()
        if not item_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "item_id is required")
        payload = self._read_json()
        actor = _actor_name(user)
        settings = get_settings()

        try:
            if action == "approve":
                group = payload.get("group", "")
                if not isinstance(group, str):
                    raise RequestError(HTTPStatus.BAD_REQUEST, "group must be a string")
                group = group.strip()
                if group and group not in known_groups(settings.access_policy_file):
                    raise RequestError(
                        HTTPStatus.BAD_REQUEST, f"未知的访问组：{group}"
                    )
                item = approve_staged_upload(settings, item_id, group, actor)
            else:
                reason = payload.get("reason", "")
                item = reject_staged_upload(
                    item_id, actor, reason if isinstance(reason, str) else ""
                )
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise RequestError(HTTPStatus.CONFLICT, str(exc)) from exc
        self._send_json(item)

    def _upload_permission(self, settings) -> tuple[UserContext | None, bool]:
        """Admins index directly; everyone else submits to the staging queue.

        顺序是承重的：`_current_user` 必须在 `policy is None` 之前调用。要求策略
        却读不到策略时它会抛 503，于是下面那个「开放模式直接入库」的分支走不到——
        否则匿名者就能往向量库写东西。重排这两行会重新打开那个缺口。
        """

        policy = load_access_policy(settings.access_policy_file)
        user = self._current_user(settings)
        if policy is None:
            return user, True
        return user, user is not None and policy.is_admin(user)

    def _handle_document_upload(self) -> None:
        from app.uploads import get_upload_store

        settings = get_settings()
        user, is_admin = self._upload_permission(settings)
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
            files, fields = parse_multipart(
                self.headers.get("Content-Type", ""),
                body,
                supported_suffixes=SUPPORTED_SUFFIXES,
                max_files=settings.max_upload_files,
            )
            structurize = fields.get("structurize") == "on"
            if is_admin:
                job = get_upload_store().create_uploaded(
                    files, settings.data_dir, structurize, _actor_name(user)
                )
            else:
                # No write access: park it for an admin to approve.
                from app import audit
                from app.staging import get_staging_store

                job = get_staging_store().add(files, _actor_name(user), structurize)
                job["staged"] = True
                audit.record(
                    "staging.submitted",
                    actor=_actor_name(user),
                    item_id=job["id"],
                    files=job["filenames"],
                    structurize=structurize,
                )
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except RuntimeError as exc:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc

        self._send_json(job, status=HTTPStatus.ACCEPTED)

    def _handle_document_delete(self, raw_doc_id: str) -> None:
        from app.service import delete_indexed_documents

        user = self._require_admin()
        doc_id = unquote(raw_doc_id).strip()
        if not doc_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "doc_id is required")
        deleted_chunks = delete_indexed_documents(
            get_settings(), [doc_id], _actor_name(user)
        )
        if not deleted_chunks:
            raise RequestError(HTTPStatus.NOT_FOUND, "Document not found")
        self._send_json({"doc_id": doc_id, "deleted_chunks": deleted_chunks})

    def _handle_document_detail(self, raw_doc_id: str, query: str) -> None:
        from app.service import get_indexed_document

        settings = get_settings()
        user = self._current_user(settings)
        doc_id = unquote(raw_doc_id).strip()
        if not doc_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "doc_id is required")

        params = parse_qs(query)
        offset = self._read_query_int(params, "offset", default=0, minimum=0)
        limit = self._read_query_int(
            params, "limit", default=50, minimum=1, maximum=200
        )
        focus_chunk_id = (params.get("focus") or [""])[0].strip()[:200]
        try:
            document = get_indexed_document(
                settings,
                doc_id,
                user,
                offset,
                limit,
                focus_chunk_id or None,
            )
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, "Document not found") from exc
        except PermissionError as exc:
            raise RequestError(HTTPStatus.UNAUTHORIZED, str(exc)) from exc
        self._send_json(document)

    @staticmethod
    def _read_query_int(
        params: dict[str, list[str]],
        key: str,
        *,
        default: int,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        values = params.get(key)
        if not values or values[0] == "":
            return default
        try:
            value = int(values[0])
        except ValueError as exc:
            raise RequestError(
                HTTPStatus.BAD_REQUEST, f"{key} must be an integer"
            ) from exc
        if value < minimum or (maximum is not None and value > maximum):
            bounds = (
                f"{minimum} and {maximum}" if maximum is not None else f"{minimum}+"
            )
            raise RequestError(
                HTTPStatus.BAD_REQUEST, f"{key} must be between {bounds}"
            )
        return value

    @staticmethod
    def _index_model_info(settings) -> dict[str, object]:
        """Which embedding model built the stored vectors, and whether it still matches."""

        index_model = read_index_embedding_model(settings.chroma_persist_dir)
        return {
            "index_embedding_model": index_model,
            "embedding_model_mismatch": bool(index_model)
            and index_model != settings.embedding_model,
        }

    def _handle_me(self) -> None:
        """当前身份，前端靠它决定「设置 / 审计」导航显不显示。

        dev_proxy 也提供这个路由，但它会连全部账号名一起返回、供本地切换身份用。
        这里**故意不回那份列表**：它既不该外泄，前端拿到还会渲染出「切换身份」
        菜单——点下去跳的 /login 只有 dev_proxy 有，生产环境根本不存在。

        开放模式（没有策略文件）算管理员，与 _require_admin / _status_payload
        保持一致：那种状态下后端本来就什么都允许，前端显示管理员入口才是诚实的。
        """

        settings = get_settings()
        user = self._current_user(settings)
        policy = load_access_policy(settings.access_policy_file)
        self._send_json(
            {
                "user": user.user_id if user else None,
                "users": [],
                "is_admin": policy is None
                or (user is not None and policy.is_admin(user)),
            }
        )

    def _status_payload(self, user: UserContext | None = None) -> dict[str, object]:
        settings = get_settings()
        # 绝对路径、data_dir 下的完整文件名清单、内网端点只给管理员：公网上这些
        # 足够画出这台机器的结构（路径里还带本机用户名）。它们只被设置页用到，
        # 而设置页本来就仅管理员可见。模型名与检索参数留给所有人——问答页要用。
        #
        # 没有策略文件时视为管理员，保持本地单机的既有行为。公网那侧不靠这里
        # 兜底：REQUIRE_ACCESS_POLICY 会让「没有策略」根本无法启动。
        policy = load_access_policy(settings.access_policy_file)
        admin = policy is None or (user is not None and policy.is_admin(user))

        supported_files = []
        if admin and settings.data_dir.exists():
            supported_files = [
                str(path)
                for path in sorted(settings.data_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ]

        return {
            "has_openai_api_key": bool(settings.openai_api_key),
            "data_dir": str(settings.data_dir) if admin else None,
            "chroma_persist_dir": str(settings.chroma_persist_dir) if admin else None,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "embedding_base_url": (
                settings.embedding_base_url or settings.openai_base_url
            ) if admin else None,
            "ocr_model": settings.ocr_model,
            "ocr_max_pages": settings.ocr_max_pages,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_method": settings.retrieval_method,
            "candidate_pool_size": settings.candidate_pool_size,
            "similarity_threshold": settings.similarity_threshold,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_upload_files": settings.max_upload_files,
            "max_upload_jobs": settings.max_upload_jobs,
            "openai_base_url": settings.openai_base_url if admin else None,
            "supported_files": supported_files,
            "chroma_exists": (settings.chroma_persist_dir / "chroma.sqlite3").is_file(),
            "mineru_configured": bool(settings.mineru_api_token),
            "structurize_provider": settings.structurize_provider,
            "markdown_cache": markdown_cache.info() if admin else {},
            **self._index_model_info(settings),
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

    def _current_user(self, settings) -> UserContext | None:
        """请求身份。策略文件缺失时是**失败即关闭**，不是放行。

        没配策略 = 开放模式，本地单机靠它省掉登录。但同样的分支在公网就是灾难：
        把策略文件删掉或改名，`_require_admin` 与上传鉴权会一起静默失效，匿名者
        就能读写 API Key、直接往向量库写数据。所以公网部署要把
        REQUIRE_ACCESS_POLICY 打开，让这种状态直接拒绝服务而不是降级成全开放。
        """

        policy = load_access_policy(settings.access_policy_file)
        if policy is None:
            if settings.require_access_policy:
                raise RequestError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "服务要求访问策略，但策略文件不存在或不可读",
                )
            return None
        user_id = self.headers.get("X-User-ID", "").strip()
        if not user_id:
            raise RequestError(HTTPStatus.UNAUTHORIZED, "Authenticated user is required")
        try:
            return policy.user_context(user_id)
        except PermissionError as exc:
            raise RequestError(HTTPStatus.UNAUTHORIZED, str(exc)) from exc

    def _require_admin(self) -> UserContext | None:
        settings = get_settings()
        policy = load_access_policy(settings.access_policy_file)
        # 先取身份：_current_user 负责「要求策略却没有策略」时的拒绝，
        # 所以这里不能再用 `policy is not None` 把整条判断短路掉。
        user = self._current_user(settings)
        if policy is not None and (user is None or not policy.is_admin(user)):
            raise RequestError(HTTPStatus.FORBIDDEN, "需要管理员权限")
        return user

    @staticmethod
    def _read_bool(payload: dict[str, object], key: str) -> bool:
        value = payload.get(key, False)
        if not isinstance(value, bool):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{key} must be a boolean")
        return value

    MAX_MODEL_NAME_LENGTH = 128
    MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")

    @staticmethod
    def _clean_base_url(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} must be a string")
        value = value.strip()
        if value and not value.lower().startswith(("http://", "https://")):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{field} must start with http:// or https://",
            )
        return value

    @classmethod
    def _clean_model_name(cls, value: object, field: str) -> str:
        if not isinstance(value, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} must be a string")
        value = value.strip()
        if not value:
            return ""
        if len(value) > cls.MAX_MODEL_NAME_LENGTH or not cls.MODEL_NAME_PATTERN.match(value):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{field} 只能包含字母、数字和 . _ : / @ + -",
            )
        return value

    @classmethod
    def _read_config_updates(cls, payload: dict[str, object]) -> dict[str, str]:
        updates: dict[str, str] = {}
        for field in ("openai_base_url", "embedding_base_url"):
            if field in payload:
                updates[field] = cls._clean_base_url(payload[field], field)
        for field in ("openai_api_key", "embedding_api_key"):
            if field not in payload:
                continue
            value = payload[field]
            if not isinstance(value, str):
                raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} must be a string")
            value = value.strip()
            # An empty key means "keep the existing one": the UI never echoes secrets.
            if value:
                updates[field] = value
        for field in ("llm_model", "embedding_model"):
            if field in payload:
                value = cls._clean_model_name(payload[field], field)
                if value:
                    updates[field] = value
        if "mineru_api_token" in payload:
            value = payload["mineru_api_token"]
            if not isinstance(value, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "mineru_api_token must be a string"
                )
            value = value.strip()
            if value:
                updates["mineru_api_token"] = value
        if payload.get("clear_mineru_api_token") is True:
            # Secrets are never echoed back, so "blank" has to mean "keep";
            # disabling MinerU needs an explicit flag.
            updates["mineru_api_token"] = ""
        if "mineru_model_version" in payload:
            value = payload["mineru_model_version"]
            if value not in ("pipeline", "vlm"):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "mineru_model_version must be pipeline or vlm",
                )
            updates["mineru_model_version"] = str(value)
        if "structurize_provider" in payload:
            value = payload["structurize_provider"]
            if value not in ("markitdown", "mineru"):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "structurize_provider must be markitdown or mineru",
                )
            updates["structurize_provider"] = str(value)
        return updates

    @staticmethod
    def _read_retrieval_params(
        payload: dict[str, object],
    ) -> tuple[int | None, float | None, str | None]:
        retrieval_method = payload.get("retrieval_method")
        if "retrieval_method" in payload and retrieval_method not in (
            "vector", "bm25", "hybrid",
        ):
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                "retrieval_method must be vector, bm25 or hybrid",
            )
        candidate_pool = payload.get("candidate_pool")
        if candidate_pool is not None:
            if isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "candidate_pool must be an integer"
                )
            if not 20 <= candidate_pool <= 500:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "candidate_pool must be between 20 and 500",
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
        return top_k, similarity_threshold, retrieval_method, candidate_pool

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
        # Without this the browser heuristically caches the ES modules, so a
        # code change keeps running the old file until a hard refresh.
        self.send_header("Cache-Control", "no-cache")
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
