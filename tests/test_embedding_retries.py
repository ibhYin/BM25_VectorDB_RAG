"""Exercise the real embedding client against an in-memory rate-limited API."""
import asyncio
from email.utils import formatdate
import time

import httpx
import pytest
from langchain_openai import OpenAIEmbeddings
from openai import BadRequestError, RateLimitError

from rag import vector_store


@pytest.fixture
def endpoint(monkeypatch):
    clock = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        clock["sleeps"].append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(time, "sleep", sleep)

    def build(handler, batch_size=1):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(vector_store, "OpenAIEmbeddings", lambda **kwargs: OpenAIEmbeddings(
            http_client=client, http_async_client=async_client, **kwargs))
        return vector_store.get_embeddings("test-model", "test-key", "https://embedding.test/v1", batch_size)

    return clock, build


def success(request):
    import json
    texts = json.loads(request.content)["input"]
    return httpx.Response(200, json={"data": [
        {"index": i, "embedding": [1.0, 2.0]} for i, _ in enumerate(texts)
    ]})


def limited(code="AccountRateLimitExceeded", headers=None):
    return httpx.Response(429, headers=headers, json={"error": {
        "code": code, "message": "Requests are too frequent.", "type": "TooManyRequests"}})


def test_minute_rate_limit_recovers_without_repeating_completed_batches(endpoint):
    clock, build = endpoint
    seen = []

    def handler(request):
        seen.append(request.content)
        if len(seen) == 1 or clock["now"] >= 60:
            return success(request)
        return limited()

    embeddings = build(handler)
    assert embeddings.embed_documents(["first", "second"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert sum(b'first' in body for body in seen) == 1
    assert 60 <= clock["now"] < 140


@pytest.mark.parametrize("headers,seconds", [
    ({"Retry-After": "30"}, 30),
    ({"retry-after-ms": "12000"}, 12),
])
def test_retry_uses_server_delay(endpoint, headers, seconds):
    clock, build = endpoint
    calls = []

    def handler(request):
        calls.append(request)
        return limited(headers=headers) if len(calls) == 1 else success(request)

    assert build(handler).embed_query("query") == [1.0, 2.0]
    assert clock["sleeps"] == [seconds]


def test_retry_after_http_date(endpoint, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    test_retry_uses_server_delay(endpoint, {
        "Retry-After": formatdate(1_700_000_090, usegmt=True)}, 90)


@pytest.mark.parametrize("value", ["broken", "nan", "inf", "-1"])
def test_invalid_retry_after_falls_back_to_backoff(endpoint, value):
    clock, build = endpoint
    calls = []

    def handler(request):
        calls.append(request)
        return limited(headers={"Retry-After": value}) if len(calls) == 1 else success(request)

    assert build(handler).embed_query("query") == [1.0, 2.0]
    assert 5 <= clock["sleeps"][0] <= 6


def test_excessive_server_cooldown_surfaces_original_error(endpoint):
    clock, build = endpoint
    with pytest.raises(RateLimitError, match="AccountRateLimitExceeded"):
        build(lambda request: limited(headers={"Retry-After": "3600"})).embed_query("query")
    assert clock["sleeps"] == []


@pytest.mark.parametrize("status", [408, 409, 500, 503])
def test_other_transient_http_failures_still_retry(endpoint, status):
    clock, build = endpoint
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "temporary"}}) if len(calls) == 1 else success(request)

    assert build(handler).embed_query("query") == [1.0, 2.0]
    assert len(calls) == 2


def test_persistent_rate_limit_is_bounded_and_keeps_provider_error(endpoint):
    clock, build = endpoint
    calls = []

    def handler(request):
        calls.append(request)
        return limited()

    with pytest.raises(RateLimitError, match="AccountRateLimitExceeded"):
        build(handler).embed_query("query")
    assert len(calls) == 7
    assert len(clock["sleeps"]) == 6


@pytest.mark.parametrize("code", ["QuotaExceeded", "AccountQuotaExceeded", "SetLimitExceeded", "insufficient_quota"])
def test_quota_errors_are_not_retried(endpoint, code):
    clock, build = endpoint
    with pytest.raises(RateLimitError):
        build(lambda request: limited(code)).embed_query("query")
    assert clock["sleeps"] == []


def test_invalid_input_is_not_retried(endpoint):
    clock, build = endpoint
    with pytest.raises(BadRequestError):
        build(lambda request: httpx.Response(400, json={"error": {
            "code": "InvalidParameter", "message": "invalid input"}})).embed_query("query")
    assert clock["sleeps"] == []


def test_async_embedding_uses_the_same_retry_policy(endpoint):
    clock, build = endpoint
    calls = []

    def handler(request):
        calls.append(request)
        return limited(headers={"Retry-After": "30"}) if len(calls) == 1 else success(request)

    embeddings = build(handler)

    async def sleep(seconds):
        clock["sleeps"].append(seconds)
        clock["now"] += seconds

    # Tenacity uses asyncio.sleep for coroutine wrappers; no real delay in tests.
    if hasattr(embeddings.async_client.create, "retry"):
        embeddings.async_client.create.retry.sleep = sleep
    assert asyncio.run(embeddings.aembed_query("query")) == [1.0, 2.0]
    assert clock["sleeps"] == [30]
