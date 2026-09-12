#!/usr/bin/env python3
"""诊断 MinerU Token：直接调用 API 并显示详细错误。"""

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# 加载配置
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import get_settings

settings = get_settings()
token = settings.mineru_api_token

if not token:
    print("❌ 未配置 MINERU_API_TOKEN")
    sys.exit(1)

print(f"✅ Token 已配置: {token[:15]}...{token[-10:]}")
print(f"   长度: {len(token)} 字符")
print()

# 尝试最简单的 API 调用
API_BASE = "https://mineru.net/api/v4"
url = f"{API_BASE}/file-urls/batch"
payload = {
    "files": [{"name": "test.pdf", "data_id": "test"}],
    "model_version": settings.mineru_model_version,
}

request = Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "rag-test/1.0",
    },
)

print(f"正在调用: POST {url}")
print(f"请求头: Authorization: Bearer {token[:10]}...")
print()

try:
    with urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
        print(f"✅ HTTP {response.status}")
        print(json.dumps(body, indent=2, ensure_ascii=False))
except HTTPError as exc:
    print(f"❌ HTTP {exc.code} {exc.reason}")
    try:
        error_body = exc.read().decode("utf-8")
        error_data = json.loads(error_body)
        print(json.dumps(error_data, indent=2, ensure_ascii=False))
    except Exception:
        print(error_body if error_body else "(无响应体)")
    print()
    if exc.code == 403:
        print("可能的原因:")
        print("  1. Token 已过期或被吊销 → 去 mineru.net/apiManage 重新生成")
        print("  2. Token 没有精准解析 API 权限 → 检查账户权限")
        print("  3. 账户欠费或配额用尽 → 检查 mineru.net 账户余额")
    sys.exit(1)
except URLError as exc:
    print(f"❌ 网络错误: {exc}")
    sys.exit(1)
