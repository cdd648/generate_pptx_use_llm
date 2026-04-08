"""
Gemini API 客户端 - 统一入口，自动路由到正确的后端

路由逻辑：
  - 配置了 GEMINI_BASE_URL → 使用 OpenAI 兼容客户端（第三方代理）
  - 未配置 GEMINI_BASE_URL  → 使用 google-genai 原生客户端（Google 官方 API）

对外暴露统一的接口函数，上层代码无需关心底层实现。
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _use_openai_compat() -> bool:
    """
    检测是否应该使用 OpenAI 兼容模式

    Returns:
        bool: True 表示使用 OpenAI 兼容客户端（第三方代理），False 表示使用原生 Gemini 客户端
    """
    return bool(os.getenv("GEMINI_BASE_URL"))


def reset_client():
    """
    重置所有客户端缓存

    同时重置原生 Gemini 客户端和 OpenAI 兼容客户端的缓存实例，
    确保下次调用时使用最新的配置（API Key、Base URL 等）
    """
    if _use_openai_compat():
        from src.openai_compat_client import reset_client as _reset_oai
        _reset_oai()
    else:
        from src.gemini_native_client import reset_client as _reset_gemini
        _reset_gemini()


def generate_text(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    使用指定模型生成文本内容（自动选择后端）

    Args:
        model: 模型名称
        system_prompt: 系统提示词，定义模型角色和行为
        user_prompt: 用户输入的文本提示词

    Returns:
        str: 模型生成的文本响应
    """
    if _use_openai_compat():
        from src.openai_compat_client import generate_text as _fn
    else:
        from src.gemini_native_client import generate_text as _fn
    return _fn(model, system_prompt, user_prompt)


def generate_text_with_images(
    model: str, system_prompt: str, user_prompt: str, image_paths: list[str]
) -> str:
    """
    发送文本和图片到视觉模型进行分析（自动选择后端）

    Args:
        model: 视觉分析模型名称
        system_prompt: 系统提示词
        user_prompt: 用户输入的文本提示词
        image_paths: 图片文件路径列表

    Returns:
        str: 模型对图片分析后的文本响应
    """
    if _use_openai_compat():
        from src.openai_compat_client import generate_text_with_images as _fn
    else:
        from src.gemini_native_client import generate_text_with_images as _fn
    return _fn(model, system_prompt, user_prompt, image_paths)


def generate_image(model: str, prompt: str) -> bytes | None:
    """
    使用图片生成模型生成图片（自动选择后端）

    Args:
        model: 图片模型名称
        prompt: 生成图片的文本描述

    Returns:
        bytes | None: 生成的图片二进制数据
    """
    if _use_openai_compat():
        from src.openai_compat_client import generate_image as _fn
    else:
        from src.gemini_native_client import generate_image as _fn
    return _fn(model, prompt)


def edit_image(
    model: str, prompt: str, image_path: str
) -> bytes | None:
    """
    发送图片和编辑提示到图片编辑模型（自动选择后端）

    Args:
        model: 图片编辑模型名称
        prompt: 编辑指令
        image_path: 原始图片路径

    Returns:
        bytes | None: 编辑后的图片二进制数据
    """
    if _use_openai_compat():
        from src.openai_compat_client import edit_image as _fn
    else:
        from src.gemini_native_client import edit_image as _fn
    return _fn(model, prompt, image_path)
