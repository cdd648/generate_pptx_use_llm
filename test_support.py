import base64
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_PROXY_BASE_URL = "https://api.vectorengine.ai/v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def init_env() -> None:
    load_dotenv(ROOT / ".env")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


def get_proxy_base_url() -> str:
    return os.getenv("GEMINI_BASE_URL", DEFAULT_PROXY_BASE_URL).rstrip("/")


def find_test_image() -> Path | None:
    examples_dir = ROOT / "examples"
    if not examples_dir.exists():
        return None

    for path in sorted(examples_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def read_image_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


def image_to_data_uri(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(read_image_bytes(image_path)).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_proxy_image_response(content, response=None) -> bytes | None:
    if isinstance(content, str) and "base64" in content:
        match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
        if match:
            return base64.b64decode(match.group(1))

    if isinstance(content, list) and content and isinstance(content[0], dict):
        image_obj = content[0].get("image")
        if isinstance(image_obj, dict):
            if image_obj.get("base64"):
                return base64.b64decode(image_obj["base64"])

    if response is not None and getattr(response, "data", None):
        first = response.data[0]
        if getattr(first, "b64_json", None):
            return base64.b64decode(first.b64_json)

    return None


def short_text(value, limit: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    if not text:
        return "<empty>"
    return text[:limit]
