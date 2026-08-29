"""Small manual connectivity check; this is not part of the pytest suite."""

from openai import AuthenticationError, OpenAI, OpenAIError

from app.config import get_settings


def mask_key(api_key: str) -> str:
    if len(api_key) <= 12:
        return "***"
    return f"{api_key[:7]}...{api_key[-4:]}"


def main() -> int:
    settings = get_settings()
    api_key = settings.openai_api_key.strip()
    base_url = settings.openai_base_url.strip()

    if not api_key:
        print("OPENAI_API_KEY is empty. Please set it in .env.")
        return 2

    print(f"Testing OPENAI_API_KEY: {mask_key(api_key)}")
    if base_url:
        print(f"Using OPENAI_BASE_URL: {base_url}")

    kwargs = {"base_url": base_url} if base_url else {}
    client = OpenAI(api_key=api_key, timeout=60, max_retries=2, **kwargs)
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": "请只回复 OK"},
            ],
            temperature=0,
            max_tokens=8,
        )
    except AuthenticationError as exc:
        print("Invalid API key.")
        print(f"Error: {exc}")
        return 1
    except OpenAIError as exc:
        print("API request failed.")
        print(f"Error: {exc}")
        return 1

    content = response.choices[0].message.content
    print("API key is valid.")
    print(f"Model response: {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
