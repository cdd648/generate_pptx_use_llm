"""
Google Gemini 原生 API 客户端

使用 google-genai SDK 直接调用 Google 官方 Gemini API。
当未配置 GEMINI_BASE_URL 时，由 gemini_client.py 路由到此模块。
"""

import os
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


# 缓存的客户端实例和对应的配置（用于检测配置变化）
_client = None
_cached_api_key = None
_cached_base_url = None

# 可重试的 HTTP 状态码列表
_RETRYABLE_STATUS_CODES = {502, 503, 504}

# 最大重试次数和重试间隔（秒）
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2


def get_client() -> genai.Client:
    """
    获取或创建 Gemini API 原生客户端实例

    通过环境变量配置：
        GEMINI_API_KEY: API 密钥
        GEMINI_BASE_URL: 可选，API 基础 URL（默认使用 Google 官方端点）

    Returns:
        genai.Client: 配置好的客户端实例
    """
    global _client, _cached_api_key, _cached_base_url

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    # 如果配置发生变化，重新创建客户端
    if _client is None or api_key != _cached_api_key or base_url != _cached_base_url:
        if base_url:
            _client = genai.Client(
                api_key=api_key,
                http_options={"base_url": base_url},
            )
        else:
            _client = genai.Client(api_key=api_key)

        _cached_api_key = api_key
        _cached_base_url = base_url

    return _client


def reset_client():
    """
    重置客户端缓存，强制下次调用 get_client() 时创建新实例
    """
    global _client, _cached_api_key, _cached_base_url
    _client = None
    _cached_api_key = None
    _cached_base_url = None


def _call_with_retry(callable_fn, model: str):
    """
    对 Gemini API 调用封装自动重试逻辑，处理 502/503/504 等瞬态服务端错误

    Args:
        callable_fn: 无参数的可调用对象，执行实际的 API 调用
        model: 当前使用的模型名称（用于日志）

    Returns:
        callable_fn 的返回值

    Raises:
        ClientError: 客户端错误（4xx），不重试直接抛出
        ServerError: 服务端错误（5xx）重试耗尽后抛出
    """
    last_exception = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return callable_fn()
        except ServerError as e:
            last_exception = e
            status_code = e.status_code if hasattr(e, 'status_code') else 500
            if status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * attempt
                time.sleep(delay)
                continue
            raise
        except ClientError as e:
            raise
    raise last_exception


def generate_text(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    使用指定模型生成文本内容（含自动重试）

    Args:
        model: 模型名称，例如 gemini-2.0-flash
        system_prompt: 系统提示词
        user_prompt: 用户输入的文本提示词

    Returns:
        str: 模型生成的文本响应
    """
    client = get_client()

    def _do_call():
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            ),
        )
        return response.text

    return _call_with_retry(_do_call, model)


def generate_text_with_images(
    model: str, system_prompt: str, user_prompt: str, image_paths: list[str]
) -> str:
    """
    发送文本和图片到 Vision 模型进行分析（含自动重试）

    Args:
        model: Vision 分析模型名称
        system_prompt: 系统提示词
        user_prompt: 用户输入的文本提示词
        image_paths: 图片文件路径列表

    Returns:
        str: 模型对图片分析后的文本响应
    """
    client = get_client()
    parts = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    parts.append(types.Part.from_text(text=user_prompt))

    def _do_call():
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
        )
        return response.text

    return _call_with_retry(_do_call, model)


def generate_image(model: str, prompt: str) -> bytes | None:
    """
    使用图片生成模型生成图片（含自动重试）

    Args:
        model: 图片模型名称
        prompt: 生成图片的文本描述

    Returns:
        bytes | None: 生成的图片二进制数据
    """
    client = get_client()

    def _do_call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        return None

    return _call_with_retry(_do_call, model)


def edit_image(
    model: str, prompt: str, image_path: str
) -> bytes | None:
    """
    发送图片和编辑提示到图片编辑模型（含自动重试）

    Args:
        model: 图片编辑模型名称
        prompt: 编辑指令
        image_path: 原始图片路径

    Returns:
        bytes | None: 编辑后的图片二进制数据
    """
    client = get_client()
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type=mime),
        types.Part.from_text(text=prompt),
    ]

    def _do_call():
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        return None

    return _call_with_retry(_do_call, model)
