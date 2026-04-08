"""
Test the third-party proxy API (vectorengine.ai style).
"""

import os

from openai import OpenAI

from test_support import (
    find_test_image,
    get_api_key,
    get_proxy_base_url,
    image_to_data_uri,
    init_env,
    parse_proxy_image_response,
    read_image_bytes,
    short_text,
)


init_env()

API_KEY = get_api_key()
BASE_URL = get_proxy_base_url()
TEXT_MODEL = os.getenv("TEST_PROXY_TEXT_MODEL", "gemini-2.0-flash")
VISION_MODEL = os.getenv("TEST_PROXY_VISION_MODEL", TEXT_MODEL)
IMAGE_EDIT_MODEL = os.getenv(
    "TEST_PROXY_IMAGE_EDIT_MODEL",
    "gemini-3.1-flash-image-preview",
)
TEST_IMAGE_PATH = find_test_image()


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def test_proxy_text() -> bool:
    print("=" * 60)
    print("Test 1: Proxy text generation")
    print("=" * 60)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return False

    try:
        client = make_client()
        print(f"\nModel: {TEXT_MODEL}")
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": "Hello, are you working?"}],
        )
        reply = response.choices[0].message.content
        print("[OK] Text generation succeeded")
        print(f"Reply: {short_text(reply)}")
        return True
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


def test_proxy_vision() -> bool:
    print("\n" + "=" * 60)
    print("Test 2: Proxy vision analysis")
    print("=" * 60)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return False
    if TEST_IMAGE_PATH is None:
        print("[X] No test image was found under examples/")
        return False

    try:
        client = make_client()
        print(f"\nImage: {TEST_IMAGE_PATH}")
        print(f"Model: {VISION_MODEL}")

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly"},
                        {"type": "image_url", "image_url": {"url": image_to_data_uri(TEST_IMAGE_PATH)}},
                    ],
                }
            ],
        )

        description = response.choices[0].message.content
        print("[OK] Vision analysis succeeded")
        print(f"Description: {short_text(description)}")
        return True
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


def test_proxy_image_edit() -> bool:
    print("\n" + "=" * 60)
    print("Test 3: Proxy image edit")
    print("=" * 60)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return False
    if TEST_IMAGE_PATH is None:
        print("[X] No test image was found under examples/")
        return False

    try:
        client = make_client()
        image_bytes = read_image_bytes(TEST_IMAGE_PATH)

        print(f"\nImage: {TEST_IMAGE_PATH}")
        print(f"Image size: {len(image_bytes)} bytes")
        print(f"Model: {IMAGE_EDIT_MODEL}")

        response = client.chat.completions.create(
            model=IMAGE_EDIT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Remove all text from this image"},
                        {"type": "image_url", "image_url": {"url": image_to_data_uri(TEST_IMAGE_PATH)}},
                    ],
                }
            ],
            extra_body={"response_modalities": ["image"]},
        )

        image_data = parse_proxy_image_response(
            response.choices[0].message.content,
            response,
        )
        if image_data:
            print(f"[OK] Image edit succeeded, returned {len(image_data)} bytes")
            return True

        print("[X] Image edit returned no image data")
        print(f"Content preview: {short_text(response.choices[0].message.content)}")
        return False
    except Exception as e:
        error_msg = str(e)
        print(f"[X] Image edit failed: {error_msg}")
        if "not supported" in error_msg.lower() or "unsupported" in error_msg.lower():
            print("Hint: this proxy does not support image editing for the selected model")
        return False


def main() -> None:
    print("Third-party proxy API test tool")
    print(f"Base URL: {BASE_URL}")
    print(f"API key set: {'yes' if API_KEY else 'no'}")
    print(f"Test image: {TEST_IMAGE_PATH if TEST_IMAGE_PATH else 'not found'}")
    print()

    results = [
        ("Text generation", test_proxy_text()),
        ("Vision analysis", test_proxy_vision()),
        ("Image edit", test_proxy_image_edit()),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results:
        status = "[OK]" if success else "[X]"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
