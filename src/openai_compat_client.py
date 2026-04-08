"""
OpenAI 兼容客户端 - 用于调用第三方 OpenAI 格式的 API 代理

当用户配置了 GEMINI_BASE_URL 时，使用此模块通过 OpenAI SDK 调用代理服务。
支持的功能：
  - 文本生成（chat completions）
  - 图片视觉分析（vision / 多模态）
  - 图片生成（image generation）
  - 图片编辑（image editing）
"""

import os
import time
import base64
import httpx

from openai import OpenAI, APIError, APIStatusError, APITimeoutError


# 缓存的客户端实例和对应的配置
_client = None
_cached_api_key = None
_cached_base_url = None

# 可重试的 HTTP 状态码列表
_RETRYABLE_STATUS_CODES = {502, 503, 504}

# 最大重试次数和重试间隔（秒）
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2


def _get_base_url() -> str | None:
    """获取配置的 Base URL，自动去除末尾斜杠"""
    url = os.getenv("GEMINI_BASE_URL")
    if url:
        return url.rstrip("/")
    return None


def get_client() -> OpenAI:
    """
    获取或创建 OpenAI 客户端实例

    通过环境变量配置：
        GEMINI_API_KEY: API 密钥
        GEMINI_BASE_URL: 第三方代理的基础 URL（必须设置才使用此客户端）

    Returns:
        OpenAI: 配置好的 OpenAI 客户端实例
    """
    global _client, _cached_api_key, _cached_base_url

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = _get_base_url()

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    if not base_url:
        raise ValueError("GEMINI_BASE_URL not set - cannot use OpenAI compat client without proxy URL")

    # 配置变化时重新创建客户端
    if _client is None or api_key != _cached_api_key or base_url != _cached_base_url:
        _client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(timeout=300.0),
        )
        _cached_api_key = api_key
        _cached_base_url = base_url

    return _client


def reset_client():
    """
    重置客户端缓存，强制下次调用时创建新实例
    """
    global _client, _cached_api_key, _cached_base_url
    _client = None
    _cached_api_key = None
    _cached_base_url = None


def _call_with_retry(callable_fn):
    """
    对 API 调用封装自动重试逻辑，处理 502/503/504 等瞬态服务端错误

    Args:
        callable_fn: 无参数的可调用对象，执行实际的 API 调用

    Returns:
        callable_fn 的返回值

    Raises:
        APIStatusError: HTTP 错误，重试耗尽后抛出
    """
    last_exception = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return callable_fn()
        except APIStatusError as e:
            last_exception = e
            status_code = e.status_code if hasattr(e, 'status_code') else 500
            if status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * attempt
                time.sleep(delay)
                continue
            raise
        except (APITimeoutError, APIError):
            raise
    raise last_exception


def _image_to_base64_data_uri(image_path: str) -> str:
    """
    将图片文件转换为 base64 data URI 格式，用于 OpenAI Vision API

    Args:
        image_path: 图片文件路径

    Returns:
        str: base64 编码的 data URI 字符串
    """
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def generate_text(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    使用指定模型生成文本内容（OpenAI chat completions 格式）

    Args:
        model: 模型名称，例如 gemini-2.0-flash（代理会做模型映射）
        system_prompt: 系统提示词，定义模型角色和行为
        user_prompt: 用户输入的文本提示词

    Returns:
        str: 模型生成的文本响应
    """
    client = get_client()

    def _do_call():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    return _call_with_retry(_do_call)


def generate_text_with_images(
    model: str, system_prompt: str, user_prompt: str, image_paths: list[str]
) -> str:
    """
    发送文本和图片到视觉模型进行分析（OpenAI Vision / 多模态格式）

    Args:
        model: 视觉分析模型名称
        system_prompt: 系统提示词
        user_prompt: 用户输入的文本提示词
        image_paths: 图片文件路径列表

    Returns:
        str: 模型对图片分析后的文本响应
    """
    client = get_client()

    # 构建 content 数组：先放所有图片，再放文字提示
    content_parts = []
    for img_path in image_paths:
        data_uri = _image_to_base64_data_uri(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    content_parts.append({
        "type": "text",
        "text": user_prompt,
    })

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    def _do_call():
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content

    return _call_with_retry(_do_call)


def generate_image(model: str, prompt: str) -> bytes | None:
    """
    使用图片生成模型生成图片（OpenAI images/generations 格式）

    注意：并非所有第三方代理都支持图片生成功能

    Args:
        model: 图片模型名称
        prompt: 生成图片的文本描述

    Returns:
        bytes | None: 生成的图片二进制数据（base64 解码后），不支持则返回 None
    """
    client = get_client()

    def _do_call():
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        if response.data and response.data[0].b64_json:
            return base64.b64decode(response.data[0].b64_json)
        # 如果返回的是 URL，尝试下载
        if response.data and response.data[0].url:
            img_resp = httpx.get(response.data[0].url, timeout=60)
            return img_resp.content
        return None

    try:
        return _call_with_retry(_do_call)
    except Exception:
        # 图片生成可能不被支持，静默失败
        return None


def edit_image(
    model: str, prompt: str, image_path: str
) -> bytes | None:
    """
    发送图片和编辑提示到图片编辑模型（OpenAI images/edits 格式）

    注意：并非所有第三方代理都支持图片编辑功能

    Args:
        model: 图片编辑模型名称
        prompt: 编辑指令，描述如何修改图片
        image_path: 原始图片路径

    Returns:
        bytes | None: 编辑后的图片二进制数据，不支持则返回 None
    """
    client = get_client()

    def _do_call():
        with open(image_path, "rb") as f:
            image_file = f.read()

        response = client.images.edit(
            model=model,
            prompt=prompt,
            image=image_file,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        if response.data and response.data[0].b64_json:
            return base64.b64decode(response.data[0].b64_json)
        if response.data and response.data[0].url:
            img_resp = httpx.get(response.data[0].url, timeout=60)
            return img_resp.content
        return None

    try:
        return _call_with_retry(_do_call)
    except Exception:
        # 图片编辑可能不被支持，静默失败
        return None
