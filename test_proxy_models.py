"""
Probe which model names are currently usable on the configured proxy.
"""

from openai import OpenAI

from test_support import get_api_key, get_proxy_base_url, init_env, short_text


init_env()

API_KEY = get_api_key()
BASE_URL = get_proxy_base_url()

TEST_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro",
    "gemini-2.0-pro-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-3.1-flash-image-preview",
    "gemini-2.0-flash-image-preview",
    "限时特价/gemini-2.0-flash",
    "限时特价/gemini-1.5-flash",
    "优质gemini/gemini-2.0-flash",
    "官转gemini/gemini-2.0-flash",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet",
]


def test_model(model_name: str) -> tuple[bool, str]:
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        return True, short_text(response.choices[0].message.content, limit=80)
    except Exception as e:
        error_msg = str(e)
        if "无可用渠道" in error_msg:
            return False, "无可用渠道"
        if "not found" in error_msg.lower():
            return False, "模型不存在"
        return False, short_text(error_msg, limit=120)


def main() -> None:
    print("=" * 70)
    print("Proxy model probe")
    print(f"Proxy: {BASE_URL}")
    print(f"API key set: {'yes' if API_KEY else 'no'}")
    print("=" * 70)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return

    working_models: list[str] = []

    for model in TEST_MODELS:
        print(f"\nTesting: {model}")
        success, result = test_model(model)
        if success:
            print(f"  [OK] {result}")
            working_models.append(model)
        else:
            print(f"  [X] {result}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if working_models:
        print(f"Found {len(working_models)} working models:")
        for model in working_models:
            print(f"  - {model}")
    else:
        print("No working models were found")


if __name__ == "__main__":
    main()
