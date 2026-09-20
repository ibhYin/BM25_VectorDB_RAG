"""Password hashing and signed session tokens for web logins."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_AUTH_FILE = Path("config/web_auth.json")
DEFAULT_SESSION_TTL = 12 * 60 * 60
ITERATIONS = 390_000


def load_auth_config(path: Path = DEFAULT_AUTH_FILE) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def hash_password(password: str, *, salt: str | None = None, iterations: int = ITERATIONS) -> str:
    raw_salt = bytes.fromhex(salt) if salt else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), raw_salt, iterations)
    return f"pbkdf2_sha256${iterations}${raw_salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_text)
        )
        return hmac.compare_digest(actual, expected)
    except (TypeError, ValueError):
        return False


def _encode(payload: dict[str, Any], secret: str) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).rstrip(b"=")
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return body.decode("ascii") + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def create_session(user_id: str, secret: str, ttl_seconds: int = DEFAULT_SESSION_TTL) -> str:
    return _encode({"sub": user_id, "exp": int(time.time()) + int(ttl_seconds)}, secret)


def verify_session(token: str | None, secret: str) -> str | None:
    if not isinstance(token, str) or not token:
        return None
    try:
        body, signature = token.split(".", 1)
        expected = hmac.new(
            secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256
        ).digest()
        actual = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        if not hmac.compare_digest(actual, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if payload.get("exp", 0) < int(time.time()):
            return None
        user_id = payload.get("sub")
        return str(user_id) if user_id else None
    except (KeyError, TypeError, ValueError):
        return None
