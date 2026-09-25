from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from http import HTTPStatus
from threading import Lock
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from app.access import (
    PUBLIC_GROUP,
    UserContext,
    known_groups,
    load_access_policy,
)
from app.web_auth import (
    DEFAULT_AUTH_FILE,
    DEFAULT_SESSION_TTL,
    create_session,
    load_auth_config,
    verify_password,
    verify_session,
)
from app.config import get_settings
from app.uploads import parse_multipart
from rag.constants import SUPPORTED_SUFFIXES
from rag import markdown_cache
from rag.index_manifest import read_index_embedding_model


STATIC_DIR = Path(__file__).resolve().parent / "web_static"
logger = logging.getLogger(__name__)

# 运行中的代码版本（进程启动时算一次并缓存），前端展示用。
from app.system_qa import CODE_VERSION  # noqa: E402

LOGIN_FAILURE_WINDOW = 15 * 60
LOGIN_FAILURE_LIMIT = 8
_login_failures: dict[str, list[float]] = {}
_login_failures_lock = Lock()


def _actor_name(user: UserContext | None) -> str:
    """Identity recorded in the audit trail and as a conversation owner."""

    from app.conversations import owner_of

    return owner_of(user)


DEFAULT_KS = (1, 3, 5)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _read_k_list(value: object) -> list[int]:
    """评估用的 K 列表。

    非法值直接报 400，而不是悄悄退回默认——那样两次跑批的 cutoff 不同，快照对比
    就没意义了，而且从输出上看不出来。
    """

    if value is None:
        return list(DEFAULT_KS)
    if not isinstance(value, list):
        raise RequestError(HTTPStatus.BAD_REQUEST, "ks must be a list of integers")
    ks: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "ks must be positive integers")
        ks.append(item)
    return sorted(set(ks)) or list(DEFAULT_KS)


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

    def do_DELETE(self) -> None:
        self._run_request(self._dispatch_delete)

    def _dispatch_get(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            settings = get_settings()
            policy = load_access_policy(settings.access_policy_file)
            if policy is not None and not self._session_user_id(settings):
                self._send_redirect("/login")
                return
            self._send_static_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path == "/login":
            self._send_static_file(
                STATIC_DIR / "login.html", "text/html; charset=utf-8"
            )
        elif parsed.path == "/logout":
            self._handle_logout_redirect()
        elif parsed.path == "/api/me":
            self._handle_me()
        elif parsed.path == "/api/status":
            self._send_json(self._status_payload())
        elif parsed.path == "/api/system-stats":
            self._handle_system_stats()
        elif parsed.path == "/api/knowledge-bases":
            self._handle_knowledge_bases()
        elif parsed.path == "/api/config":
            self._handle_get_config()
        elif parsed.path == "/api/audit":
            self._handle_audit_log(parsed.query)
        elif parsed.path == "/api/eval":
            self._handle_eval()
        elif parsed.path == "/api/staging":
            self._handle_staging_list()
        elif parsed.path == "/api/users":
            self._handle_users_list()
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
        elif parsed.path == "/api/login":
            self._handle_login()
        elif parsed.path == "/api/logout":
            self._handle_logout()
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
        elif parsed.path == "/api/users":
            self._handle_users_create()
        elif parsed.path.startswith("/api/users/"):
            self._handle_users_update(parsed.path.removeprefix("/api/users/"))
        elif parsed.path.startswith("/api/eval/"):
            self._handle_eval_action(parsed.path.removeprefix("/api/eval/"))
        else:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")

    def _dispatch_delete(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/users/"):
            self._handle_users_delete(parsed.path.removeprefix("/api/users/"))
        elif parsed.path.startswith("/api/documents/"):
            self._handle_document_delete(parsed.path.removeprefix("/api/documents/"))
        elif parsed.path.startswith("/api/conversations/"):
            self._handle_conversation_delete(
                parsed.path.removeprefix("/api/conversations/")
            )
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
        # updated：本次真正写进去的字段名，前端拿它和「这次想改的字段」对比，
        # 有没生效的能当场暴露出来。
        self._send_json(
            {
                **runtime_config_view(),
                "updated": sorted(updates),
                **self._index_model_info(settings),
            }
        )

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
        payload = self._read_json()
        groups = payload.get("groups")
        if isinstance(groups, list):
            if not all(isinstance(group, str) for group in groups):
                raise RequestError(HTTPStatus.BAD_REQUEST, "groups must be strings")
        elif isinstance(payload.get("group"), str):
            # 兼容旧前端：单组 `{"group": "hr"}`。
            groups = [payload["group"]]
        else:
            raise RequestError(HTTPStatus.BAD_REQUEST, "groups must be a list")
        try:
            updated = set_document_access(
                get_settings(), doc_id, groups, _actor_name(user)
            )
        except KeyError as exc:
            raise RequestError(HTTPStatus.NOT_FOUND, "Document not found") from exc
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(
            {
                "doc_id": doc_id,
                "access_groups": sorted(
                    {str(group).strip() for group in groups if str(group).strip()}
                ),
                "updated_chunks": updated,
            }
        )

    def _handle_users_list(self) -> None:
        from app.accounts import list_users

        self._require_admin()
        self._send_json({"users": list_users(get_settings())})

    def _handle_users_create(self) -> None:
        from app.accounts import create_user

        user = self._require_admin()
        payload = self._read_json()
        try:
            created = create_user(
                get_settings(),
                payload.get("username", ""),
                groups=payload.get("groups", []),
                is_admin=bool(payload.get("is_admin", False)),
                password=payload.get("password", ""),
                actor=_actor_name(user),
            )
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(created)

    def _handle_users_update(self, raw_username: str) -> None:
        from app.accounts import update_user

        user = self._require_admin()
        payload = self._read_json()
        try:
            updated = update_user(
                get_settings(),
                unquote(raw_username),
                groups=payload.get("groups"),
                is_admin=payload.get("is_admin"),
                password=payload.get("password"),
                clear_password=bool(payload.get("clear_password", False)),
                actor=_actor_name(user),
            )
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json(updated)

    def _handle_users_delete(self, raw_username: str) -> None:
        from app.accounts import delete_user

        user = self._require_admin()
        try:
            delete_user(get_settings(), unquote(raw_username), actor=_actor_name(user))
        except ValueError as exc:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        self._send_json({"deleted": unquote(raw_username)})

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

    def _handle_knowledge_bases(self) -> None:
        """当前知识库，以及策略里定义了但本实例不服务的其它知识库。

        **一个实例只服务一个知识库**：`knowledge_base_id` 是进程级设置，检索时拿它
        当 where 过滤条件。所以策略里的其它 kb 只能如实说成「定义了、但本实例不
        服务」——把它们列成「可用知识库」会让人以为切过去就能查到东西。

        需要已认证身份（列表随身份而变）；开放模式下 `_current_user` 返回 None 且
        不报错，所以不会 500。
        """

        settings = get_settings()
        user = self._current_user(settings)
        current = settings.knowledge_base_id
        policy = load_access_policy(settings.access_policy_file)

        if policy is None:
            self._send_json(
                {
                    "current": current,
                    "group": PUBLIC_GROUP,
                    "accessible": True,
                    "policy_enabled": False,
                    "other_knowledge_bases": [],
                }
            )
            return

        group = policy.knowledge_base_group(current)
        self._send_json(
            {
                "current": current,
                "group": group,
                # 调用者不属于这个库的访问组时，检索会返回空——值得明说，否则
                # 表现出来就是「莫名其妙什么都搜不到」
                "accessible": user is not None and group in user.groups,
                "policy_enabled": True,
                "other_knowledge_bases": sorted(
                    kb_id for kb_id in policy.knowledge_bases if kb_id != current
                ),
            }
        )

    # --- 评估（全部仅管理员）-------------------------------------------------

    def _handle_eval(self) -> None:
        """问题集与标注。标注已经重新锚定到当前索引，失效的会标出来。"""

        from app.eval_store import read_questions
        from app.service import anchor_eval_labels

        self._require_admin()
        settings = get_settings()
        questions = read_questions()
        anchored = anchor_eval_labels(settings, questions) if questions else []
        self._send_json({"questions": anchored, "total": len(anchored)})

    def _handle_eval_action(self, action: str) -> None:
        handlers = {
            "import": self._eval_import,
            "label": self._eval_label,
            "retrieve": self._eval_retrieve,
            "run": self._eval_run,
        }
        handler = handlers.get(action)
        if handler is None:
            raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
        handler()

    def _eval_import(self) -> None:
        from app.eval_store import import_first_turn_questions, read_questions

        self._require_admin()
        self._send_json(
            {
                "imported": import_first_turn_questions(),
                "total": len(read_questions()),
            }
        )

    def _eval_label(self) -> None:
        from app.eval_store import fingerprints_for, upsert_question
        from app.service import resolve_chunks

        self._require_admin()
        payload = self._read_json()
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise RequestError(HTTPStatus.BAD_REQUEST, "question is required")

        chunk_ids = _string_list(payload.get("relevant_chunk_ids"))
        doc_ids = _string_list(payload.get("relevant_doc_ids"))
        resolved = resolve_chunks(get_settings(), chunk_ids)
        kept = list(resolved)
        entry = upsert_question(
            question,
            relevant_chunk_ids=kept,
            relevant_doc_ids=doc_ids,
            fingerprints=fingerprints_for(list(resolved.values()), kept),
            note=str(payload.get("note") or ""),
        )
        # 取不到的 id 要回给前端：是拼错了、还是索引重建过导致编号变了，
        # 标注的人得当场知道，而不是让标注静悄悄地少一块。
        self._send_json(
            {
                "id": entry["id"],
                "relevant_chunk_ids": kept,
                "unresolved": [c for c in chunk_ids if c not in resolved],
            }
        )

    def _eval_retrieve(self) -> None:
        from app.service import retrieve_question

        user = self._require_admin()
        payload = self._read_json()
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise RequestError(HTTPStatus.BAD_REQUEST, "question is required")

        top_k, threshold, method, pool = self._read_retrieval_params(payload)
        hits = retrieve_question(
            question,
            get_settings(),
            retrieval_method=method,
            top_k=top_k,
            similarity_threshold=threshold,
            candidate_pool=pool,
            user=user,
        )
        self._send_json({"hits": hits})

    def _eval_run(self) -> None:
        from app.eval_runner import run_batch
        from app.eval_store import read_questions
        from app.service import anchor_eval_labels

        user = self._require_admin()
        payload = self._read_json()
        ks = _read_k_list(payload.get("ks"))
        top_k, threshold, method, pool = self._read_retrieval_params(payload)

        settings = get_settings()
        batch = run_batch(
            settings,
            anchor_eval_labels(settings, read_questions()),
            ks=ks,
            method=method,
            top_k=top_k,
            threshold=threshold,
            pool=pool,
            user=user,
        )
        # 只回检索结果 id，不回 hit 正文：那些按题去 /api/eval/retrieve 取，
        # 一次跑批塞进几十道题的正文会让响应白白变大。
        self._send_json(
            {
                "summary": batch["summary"],
                "labelled": batch["labelled"],
                "total": batch["total"],
                "stale_total": batch["stale_total"],
                "questions": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "retrieved": row["retrieved"],
                        "relevant": row["relevant"],
                        "stale": row["stale"],
                        "elapsed_ms": round(row["elapsed"] * 1000, 1),
                    }
                    for row in batch["rows"]
                ],
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
        """Admins index directly; everyone else submits to the staging queue."""

        policy = load_access_policy(settings.access_policy_file)
        user = self._current_user(settings)
        if policy is None:
            return user, True
        return user, user is not None and policy.is_admin(user)

    def _handle_document_upload(self) -> None:
        from app import audit
        from app.uploads import get_upload_store

        settings = get_settings()
        # 认证/权限失败不算一次文档写入尝试，不写进向量库审计。拿到身份之后发生的
        # 请求解析、入队、暂存错误才是「上传失败」。
        user, is_admin = self._upload_permission(settings)
        actor = _actor_name(user)
        filenames: list[str] = []
        structurize = False
        stage = "request"

        try:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "Invalid Content-Length"
                ) from exc
            if content_length <= 0:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Upload body is required")
            if content_length > settings.max_upload_bytes:
                self.close_connection = True
                raise RequestError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "Upload body is too large",
                )

            body = self.rfile.read(content_length)
            stage = "multipart_parse"
            files, fields = parse_multipart(
                self.headers.get("Content-Type", ""),
                body,
                supported_suffixes=SUPPORTED_SUFFIXES,
                max_files=settings.max_upload_files,
            )
            filenames = [filename for filename, _ in files]
            structurize = fields.get("structurize") == "on"
            if is_admin:
                stage = "queue"
                job = get_upload_store().create_uploaded(
                    files, settings.data_dir, structurize, actor
                )
            else:
                # No write access: park it for an admin to approve.
                from app.staging import get_staging_store

                stage = "staging"
                job = get_staging_store().add(files, actor, structurize)
                job["staged"] = True
                audit.record(
                    "staging.submitted",
                    actor=actor,
                    item_id=job["id"],
                    files=job["filenames"],
                    structurize=structurize,
                )
        except RequestError as exc:
            audit.record(
                "ingest.files",
                actor=actor,
                knowledge_base_id=settings.knowledge_base_id,
                files=filenames,
                structurize=structurize,
                embedding_model=settings.embedding_model,
                **audit.failure_fields(exc, stage),
            )
            raise
        except ValueError as exc:
            audit.record(
                "ingest.files",
                actor=actor,
                knowledge_base_id=settings.knowledge_base_id,
                files=filenames,
                structurize=structurize,
                embedding_model=settings.embedding_model,
                **audit.failure_fields(exc, stage),
            )
            raise RequestError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except RuntimeError as exc:
            audit.record(
                "ingest.files",
                actor=actor,
                knowledge_base_id=settings.knowledge_base_id,
                files=filenames,
                structurize=structurize,
                embedding_model=settings.embedding_model,
                **audit.failure_fields(exc, stage),
            )
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc
        except Exception as exc:
            # 未预期的磁盘/解析异常仍交给 _run_request 统一变成 500，但先留下失败审计。
            audit.record(
                "ingest.files",
                actor=actor,
                knowledge_base_id=settings.knowledge_base_id,
                files=filenames,
                structurize=structurize,
                embedding_model=settings.embedding_model,
                **audit.failure_fields(exc, stage),
            )
            raise

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
            "candidate_pool_size": settings.candidate_pool_size,
            "similarity_threshold": settings.similarity_threshold,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_upload_files": settings.max_upload_files,
            "max_upload_jobs": settings.max_upload_jobs,
            "openai_base_url": settings.openai_base_url,
            "supported_files": supported_files,
            "chroma_exists": (settings.chroma_persist_dir / "chroma.sqlite3").is_file(),
            "mineru_configured": bool(settings.mineru_api_token),
            "structurize_provider": settings.structurize_provider,
            "code_version": CODE_VERSION,
            "markdown_cache": markdown_cache.info(),
            **self._index_model_info(settings),
        }

    def _handle_system_stats(self) -> None:
        """极简服务器资源快照（CPU/内存/磁盘/进程/向量服务），供页头状态条轮询。"""

        settings = get_settings()
        from app.system_stats import collect_system_stats

        self._send_json(
            collect_system_stats(
                settings.embedding_base_url or settings.openai_base_url,
                data_dir=str(settings.data_dir),
            )
        )

    def _handle_me(self) -> None:
        """Expose the current session identity and its policy groups."""

        settings = get_settings()
        policy = load_access_policy(settings.access_policy_file)
        if policy is None:
            self._send_json({"user": None, "users": [], "is_admin": False})
            return

        user_id = self._session_user_id(settings)
        if not user_id:
            raise RequestError(HTTPStatus.UNAUTHORIZED, "Authenticated user is required")
        try:
            user = policy.user_context(user_id)
        except PermissionError as exc:
            raise RequestError(HTTPStatus.UNAUTHORIZED, str(exc)) from exc

        # Basic Auth identities are managed by the proxy; do not offer in-app
        # switching between accounts. The frontend only needs the current user.
        self._send_json(
            {
                "user": user.user_id,
                "users": [{"id": user.user_id, "groups": sorted(user.groups)}],
                "is_admin": policy.is_admin(user),
            }
        )

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

    def _session_user_id(self, settings) -> str:
        auth = load_auth_config(Path(settings.web_auth_file))
        secret = str(auth.get("secret", ""))
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        token = cookie["rag_session"].value if "rag_session" in cookie else ""
        return verify_session(token, secret) if token and secret else ""

    def _current_user(self, settings) -> UserContext | None:
        policy = load_access_policy(settings.access_policy_file)
        if policy is None:
            return None
        user_id = self._session_user_id(settings)
        if not user_id:
            raise RequestError(HTTPStatus.UNAUTHORIZED, "Authenticated user is required")
        try:
            return policy.user_context(user_id)
        except PermissionError as exc:
            raise RequestError(HTTPStatus.UNAUTHORIZED, str(exc)) from exc

    def _handle_login(self) -> None:
        settings = get_settings()
        payload = self._read_json()
        username = payload.get("username", "")
        password = payload.get("password", "")
        if not isinstance(username, str) or not isinstance(password, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "username and password are required")

        client_ip = getattr(self, "client_address", ("unknown", 0))[0]
        attempt_key = f"{client_ip}:{username}"
        now = time.monotonic()
        with _login_failures_lock:
            failures = [
                stamp for stamp in _login_failures.get(attempt_key, [])
                if now - stamp < LOGIN_FAILURE_WINDOW
            ]
            _login_failures[attempt_key] = failures
            if len(failures) >= LOGIN_FAILURE_LIMIT:
                retry_after = max(1, int(LOGIN_FAILURE_WINDOW - (now - failures[0])))
                raise RequestError(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    f"尝试次数过多，请 {retry_after} 秒后再试",
                )

        auth = load_auth_config(Path(settings.web_auth_file))
        users = auth.get("users") if isinstance(auth.get("users"), dict) else {}
        record = users.get(username) if isinstance(users.get(username), dict) else None
        encoded = str(record.get("password_hash", "")) if record else ""
        if not encoded or not verify_password(password, encoded):
            with _login_failures_lock:
                _login_failures.setdefault(attempt_key, []).append(time.monotonic())
            raise RequestError(HTTPStatus.UNAUTHORIZED, "用户名或密码错误")

        policy = load_access_policy(settings.access_policy_file)
        if policy is None or username not in policy.users:
            raise RequestError(HTTPStatus.FORBIDDEN, "该账号未配置网页权限")

        ttl = int(auth.get("session_ttl_seconds", DEFAULT_SESSION_TTL))
        token = create_session(username, str(auth.get("secret", "")), ttl)
        secure = self.headers.get("X-Forwarded-Proto", "").lower() == "https"
        cookie = (
            f"rag_session={token}; Path=/; HttpOnly; SameSite=Lax; "
            f"Max-Age={ttl}" + ("; Secure" if secure else "")
        )
        user = policy.user_context(username)
        with _login_failures_lock:
            _login_failures.pop(attempt_key, None)
        self._send_json(
            {
                "user": user.user_id,
                "users": [{"id": user.user_id, "groups": sorted(user.groups)}],
                "is_admin": policy.is_admin(user),
            },
            headers={"Set-Cookie": cookie},
        )

    def _handle_logout(self) -> None:
        expired = "rag_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
        self._send_json(
            {"ok": True},
            headers={"Set-Cookie": expired, "Clear-Site-Data": '"cookies", "storage"'},
        )

    def _handle_logout_redirect(self) -> None:
        expired = "rag_session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/login")
        self.send_header("Set-Cookie", expired)
        self.send_header("Clear-Site-Data", '"cookies", "storage"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _require_admin(self) -> UserContext | None:
        settings = get_settings()
        policy = load_access_policy(settings.access_policy_file)
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
        from app.config import RUNTIME_CONFIG_FIELDS

        # 前端发了后端不认识的字段（通常是前后端版本不一致）：明确报错而不是
        # 静默丢弃——否则用户以为存上了、实际没生效。
        known = set(RUNTIME_CONFIG_FIELDS) | {"clear_mineru_api_token"}
        unknown = set(payload) - known
        if unknown:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"Unknown config field(s): {', '.join(sorted(unknown))}",
            )

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
        if "audio_transcribe_base_url" in payload:
            updates["audio_transcribe_base_url"] = cls._clean_base_url(
                payload["audio_transcribe_base_url"], "audio_transcribe_base_url"
            )
        if "audio_transcribe_api_key" in payload:
            value = payload["audio_transcribe_api_key"]
            if not isinstance(value, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "audio_transcribe_api_key must be a string"
                )
            value = value.strip()
            if value:
                updates["audio_transcribe_api_key"] = value
        if "audio_transcribe_model" in payload:
            # 模型名会回显，用户清空即为「禁用视频转写」。
            updates["audio_transcribe_model"] = cls._clean_model_name(
                payload["audio_transcribe_model"], "audio_transcribe_model"
            )
        if "audio_transcribe_provider" in payload:
            value = payload["audio_transcribe_provider"]
            if value not in ("openai", "aliyun"):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST,
                    "audio_transcribe_provider must be openai or aliyun",
                )
            updates["audio_transcribe_provider"] = str(value)
        for field in ("aliyun_access_key_id", "aliyun_access_key_secret", "aliyun_appkey"):
            if field in payload:
                value = payload[field]
                if not isinstance(value, str):
                    raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} must be a string")
                value = value.strip()
                if value:
                    updates[field] = value
        if "aliyun_region" in payload:
            value = payload["aliyun_region"]
            if not isinstance(value, str) or not value.strip():
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "aliyun_region must be a string"
                )
            updates["aliyun_region"] = value.strip()
        if "aliyun_oss_bucket" in payload:
            value = payload["aliyun_oss_bucket"]
            if not isinstance(value, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "aliyun_oss_bucket must be a string"
                )
            updates["aliyun_oss_bucket"] = value.strip()
        if "aliyun_oss_endpoint" in payload:
            updates["aliyun_oss_endpoint"] = cls._clean_base_url(
                payload["aliyun_oss_endpoint"], "aliyun_oss_endpoint"
            )
        if "aliyun_oss_prefix" in payload:
            value = payload["aliyun_oss_prefix"]
            if not isinstance(value, str):
                raise RequestError(
                    HTTPStatus.BAD_REQUEST, "aliyun_oss_prefix must be a string"
                )
            updates["aliyun_oss_prefix"] = value.strip()
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

    def _send_json(
        self,
        payload: dict[str, object],
        status: int = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
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
