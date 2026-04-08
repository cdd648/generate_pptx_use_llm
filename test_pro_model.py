"""
Quick probe for the gemini-3.1-pro-preview model on the configured proxy.
"""

from openai import OpenAI

from test_support import get_api_key, get_proxy_base_url, init_env, short_text


init_env()

API_KEY = get_api_key()
BASE_URL = get_proxy_base_url()
MODEL_NAME = "gemini-3.1-pro-preview"


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
        return False, short_text(e, limit=120)


def main() -> None:
    print("=" * 60)
    print(f"Model probe: {MODEL_NAME}")
    print("=" * 60)
    print(f"Proxy: {BASE_URL}")
    print(f"API key set: {'yes' if API_KEY else 'no'}")

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return

    success, result = test_model(MODEL_NAME)
    if success:
        print(f"\n[OK] {result}")
    else:
        print(f"\n[X] {result}")


if __name__ == "__main__":
    main()
