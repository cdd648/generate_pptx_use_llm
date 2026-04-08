"""
Compare official Gemini API image editing with the proxy API.
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
TEST_IMAGE_PATH = find_test_image()

OFFICIAL_TEXT_MODEL = os.getenv("TEST_OFFICIAL_TEXT_MODEL", "gemini-2.0-flash")
OFFICIAL_IMAGE_MODEL = os.getenv(
    "TEST_OFFICIAL_IMAGE_MODEL",
    "gemini-3.1-flash-image-preview",
)
PROXY_TEXT_MODEL = os.getenv("TEST_PROXY_TEXT_MODEL", "gemini-2.0-flash")
PROXY_IMAGE_MODEL = os.getenv(
    "TEST_PROXY_IMAGE_EDIT_MODEL",
    "gemini-3.1-flash-image-preview",
)


def test_native_gemini() -> bool:
    print("=" * 60)
    print("Test 1: Official Gemini API")
    print("=" * 60)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return False
    if TEST_IMAGE_PATH is None:
        print("[X] No test image was found under examples/")
        return False
    if API_KEY.startswith("sk-"):
        print("[!] The configured key looks like a proxy key; official API calls may fail")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=API_KEY)

        print(f"\nText model: {OFFICIAL_TEXT_MODEL}")
        text_response = client.models.generate_content(
            model=OFFICIAL_TEXT_MODEL,
            contents="Reply with OK only.",
        )
        print(f"[OK] Official text generation succeeded: {short_text(text_response.text)}")

        image_bytes = read_image_bytes(TEST_IMAGE_PATH)
        mime_type = "image/png" if TEST_IMAGE_PATH.suffix.lower() == ".png" else "image/jpeg"
        print(f"Image model: {OFFICIAL_IMAGE_MODEL}")
        print(f"Image: {TEST_IMAGE_PATH}")

        response = client.models.generate_content(
            model=OFFICIAL_IMAGE_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text="Remove all text from this image"),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data is not None:
                    print(
                        "[OK] Official image edit succeeded, returned "
                        f"{len(part.inline_data.data)} bytes"
                    )
                    return True

        print("[X] Official image edit returned no image data")
        print(f"Response preview: {short_text(response)}")
        return False
    except Exception as e:
        print(f"[X] Official Gemini API failed: {e}")
        return False


def test_proxy_api() -> bool:
    print("\n" + "=" * 60)
    print("Test 2: Proxy API")
    print("=" * 60)

    if not API_KEY:
        print("[X] GEMINI_API_KEY is not set")
        return False
    if TEST_IMAGE_PATH is None:
        print("[X] No test image was found under examples/")
        return False

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        print(f"\nText model: {PROXY_TEXT_MODEL}")
        try:
            text_response = client.chat.completions.create(
                model=PROXY_TEXT_MODEL,
                messages=[{"role": "user", "content": "Reply with OK only."}],
            )
            print(
                "[OK] Proxy text generation succeeded: "
                f"{short_text(text_response.choices[0].message.content)}"
            )
        except Exception as text_error:
            print(f"[!] Proxy text generation failed: {text_error}")

        print(f"Image model: {PROXY_IMAGE_MODEL}")
        print(f"Image: {TEST_IMAGE_PATH}")
        response = client.chat.completions.create(
            model=PROXY_IMAGE_MODEL,
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
            print(f"[OK] Proxy image edit succeeded, returned {len(image_data)} bytes")
            return True

        print("[X] Proxy image edit returned no image data")
        print(f"Content preview: {short_text(response.choices[0].message.content)}")
        return False
    except Exception as e:
        print(f"[X] Proxy API failed: {e}")
        return False


def main() -> None:
    print("Image editing capability test")
    print(f"Base URL: {BASE_URL}")
    print(f"API key set: {'yes' if API_KEY else 'no'}")
    print(f"Test image: {TEST_IMAGE_PATH if TEST_IMAGE_PATH else 'not found'}")
    print()

    results = [
        ("Official Gemini API", test_native_gemini()),
        ("Proxy API", test_proxy_api()),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results:
        status = "[OK]" if success else "[X]"
        print(f"{name}: {status}")

    print("\nNotes:")
    print("- If the official API fails with API_KEY_INVALID, the configured key is not a Google key.")
    print("- If the proxy API fails while the official API passes, the proxy is the blocker.")


if __name__ == "__main__":
    main()
