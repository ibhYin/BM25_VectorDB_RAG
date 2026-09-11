#!/usr/bin/env python3
"""Dev-only proxy that injects X-User-ID so the browser can use the ACL-enabled API.

The RAG app is served directly; identity defaults to RAG_USER and can be
switched in-app via /login?user=ID (stored in a cookie). /api/me exposes the
effective user and the policy's user list to the frontend.
ponytail: trusts a plaintext cookie and forwards to 127.0.0.1:8000. Local
development convenience only; real deployments need an authenticating proxy.
"""

import json
import os
from pathlib import Path
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BACKEND = os.environ.get("RAG_BACKEND", "http://127.0.0.1:8000")
PORT = int(os.environ.get("RAG_PROXY_PORT", "8010"))
DEFAULT_USER = os.environ.get("RAG_USER", "alice")
POLICY_FILE = Path(os.environ.get("RAG_POLICY", "config/access_policy.json"))
COOKIE_NAME = "rag_user"

SKIP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-encoding"}


def load_policy() -> dict:
    try:
        payload = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_users() -> dict[str, list[str]]:
    return {
        str(user_id): [str(group) for group in value.get("groups", [])]
        for user_id, value in load_policy().get("users", {}).items()
        if isinstance(value, dict)
    }


def is_admin(user_id: str) -> bool:
    return user_id in (load_policy().get("admins") or [])


def identity(user_id: str) -> str:
    users = load_users()
    if user_id in users:
        return user_id
    return DEFAULT_USER if DEFAULT_USER in users else ""


class DevProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cookie_user(self) -> str:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        user_id = cookie[COOKIE_NAME].value if COOKIE_NAME in cookie else ""
        return identity(user_id)

    def _api_me(self) -> None:
        users = load_users()
        payload = {
            "user": self._cookie_user() or None,
            "default_user": DEFAULT_USER if DEFAULT_USER in users else None,
            "is_admin": is_admin(self._cookie_user()),
            "users": [
                {"id": user_id, "groups": groups}
                for user_id, groups in sorted(users.items())
            ],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, cookie: str = "") -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/login?"):
            user_id = ""
            for part in self.path.split("?", 1)[1].split("&"):
                key, _, value = part.partition("=")
                if key == "user":
                    user_id = value
            if user_id in load_users():
                self._redirect("/", f"{COOKIE_NAME}={user_id}; Path=/; HttpOnly")
            else:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Unknown user: {user_id}")
            return
        if self.path == "/logout":
            self._redirect("/", f"{COOKIE_NAME}=; Path=/; Max-Age=0")
            return
        if self.path == "/api/me":
            self._api_me()
            return
        self._forward(self._cookie_user())

    def do_POST(self) -> None:
        self._forward(self._cookie_user())

    def do_DELETE(self) -> None:
        self._forward(self._cookie_user())

    def _forward(self, user_id: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in SKIP_HEADERS
        }
        headers["X-User-ID"] = user_id
        request = Request(BACKEND + self.path, data=body, headers=headers, method=self.command)
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as exc:
            response = exc
        except URLError as exc:
            self.send_error(502, str(exc))
            return

        payload = response.read()
        self.send_response(response.code)
        for key, value in response.headers.items():
            if key.lower() not in SKIP_HEADERS and key.lower() != "content-length":
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dev-proxy] {self.command} {self.path}")


if __name__ == "__main__":
    print(f"Dev proxy: http://127.0.0.1:{PORT} -> {BACKEND}")
    ThreadingHTTPServer(("127.0.0.1", PORT), DevProxy).serve_forever()
