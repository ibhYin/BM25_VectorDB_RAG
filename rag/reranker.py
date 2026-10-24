"""SiliconFlow text reranking. Return original hits on a failed request."""
from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from time import perf_counter

import requests

from app.config import Settings, rerank_endpoint_url


logger = logging.getLogger(__name__)


def rerank_hits(question: str, hits: list[dict], settings: Settings, top_k: int) -> tuple[list[dict], dict]:
    started = perf_counter()
    detail = {
        "enabled": settings.rerank_enabled, "model": settings.rerank_model,
        "pool_size": settings.rerank_pool_size, "input_count": len(hits),
        "score_filter_enabled": settings.rerank_score_filter_enabled,
        "score_threshold": settings.rerank_score_threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_id": "", "provider_request_id": "", "http_status": None,
        "error_type": "", "error_code": "", "error_message": "",
    }

    def finish(rows, status, reason=""):
        detail.update(status=status, reason=reason, output_count=len(rows),
                      elapsed_ms=round((perf_counter() - started) * 1000))
        return rows, detail

    if not settings.rerank_enabled:
        return finish(hits[:top_k], "skipped", "disabled")
    if not hits:
        return finish([], "skipped", "no_candidates")
    if len(hits) == 1 and not settings.rerank_score_filter_enabled:
        return finish(hits[:top_k], "skipped", "single_candidate")

    detail["request_id"] = uuid.uuid4().hex
    texts = [str(hit.get("text") or "") for hit in hits]
    secrets = [str(getattr(settings, name)) for name in type(settings).model_fields
               if any(part in name for part in ("api_key", "secret", "token", "appkey")) and getattr(settings, name)]

    def safe_id(value):
        text = str(value or "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", text):
            return ""
        if any(secret in text for secret in secrets) or (question and question in text) or any(body and body in text for body in texts):
            return ""
        return text

    error_type = "invalid_request"
    message = "重排服务未配置密钥或模型"
    try:
        if not settings.rerank_api_key or not settings.rerank_model:
            raise ValueError(message)
        if not question.strip() or any(not text.strip() for text in texts):
            message = "重排问题或候选正文为空"
            raise ValueError(message)
        body = {"model": settings.rerank_model, "query": question, "documents": texts,
                "top_n": len(texts), "return_documents": False}
        if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > 1_048_576:
            message = "重排输入超过 1 MiB 请求预算"
            raise ValueError(message)
        error_type, message = "invalid_response", "重排服务返回的数据不完整或格式错误"
        # ponytail: one request, no SDK/retries; a different provider needs explicit protocol support.
        with requests.post(rerank_endpoint_url(settings.rerank_url), json=body, headers={
            "Authorization": f"Bearer {settings.rerank_api_key}",
            "X-Trace-Id": detail["request_id"],
        }, timeout=settings.rerank_timeout, allow_redirects=False) as response:
            detail["http_status"] = response.status_code
            detail["provider_request_id"] = safe_id(response.headers.get("x-siliconcloud-trace-id"))
            if response.status_code != 200:
                error_type, message = {
                    400: ("invalid_request", "请求参数或输入长度不符合服务要求"),
                    401: ("authentication", "API 密钥无效"),
                    403: ("permission", "重排服务拒绝访问，请检查模型权限"),
                    404: ("not_found", "重排接口或模型不存在"),
                    429: ("rate_limit", "重排服务限流或额度不足"),
                }.get(response.status_code, ("provider_error", "重排服务暂时不可用"))
                try:
                    error = response.json()
                    if isinstance(error, dict):
                        detail["error_code"] = safe_id(error.get("code"))
                except ValueError:
                    pass
                raise ValueError(message)
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise ValueError(message)
            detail["provider_request_id"] = detail["provider_request_id"] or safe_id(data.get("id"))
            results = data["results"]
            if len(results) != len(hits):
                raise ValueError(message)
            seen = set()
            ranked = []
            for item in results:
                if not isinstance(item, dict):
                    raise ValueError(message)
                index, score = item.get("index"), item.get("relevance_score")
                if (type(index) is not int or not 0 <= index < len(hits) or index in seen
                    or isinstance(score, bool) or not isinstance(score, (float, int))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError(message)
                seen.add(index)
                ranked.append((index, float(score)))

        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        ordered = []
        for rank, (index, score) in enumerate(ranked, 1):
            hit = hits[index]
            ordered.append({**hit, "retrieval_rank": hit.get("rank"),
                            "fusion_rank": hit.get("rank") if hit.get("retrieval_method") == "hybrid" else None,
                            "rerank_rank": rank, "rerank_score": score})
        # Keep only references/scores in diagnostics, never candidate text or the query.
        detail["candidates"] = [{key: row.get(key) for key in (
            "chunk_id", "doc_id", "title", "retrieval_rank", "fusion_rank", "rerank_rank", "rerank_score"
        )} for row in ordered]
        if settings.rerank_score_filter_enabled:
            ordered = [row for row in ordered if row["rerank_score"] >= settings.rerank_score_threshold]
        output = [{**row, "rank": rank} for rank, row in enumerate(ordered[:top_k], 1)]
        detail["filtered_count"] = len(hits) - len(ordered)
        return finish(output, "completed")
    except requests.Timeout:
        error_type, message = "timeout", "重排请求超时"
    except requests.exceptions.JSONDecodeError:
        error_type, message = "invalid_response", "重排服务返回了无效 JSON"
    except requests.RequestException:
        error_type, message = "network", "无法连接重排服务"
    except ValueError:
        pass

    detail.update(error_type=error_type, error_message=message)
    output, detail = finish(hits[:top_k], "degraded", "fallback")
    logger.warning("Rerank fallback %s", json.dumps(detail, ensure_ascii=False))
    return output, detail
