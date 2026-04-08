"""
Streamlit 前端应用 - 信息图转 PPTX 生成器

功能：
  - 上传单张或多张信息图图片
  - 配置 Gemini API 参数和模型选项
  - 实时显示处理进度
  - 生成后提供 PPTX 文件下载

用法：
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from generate_pptx import (
    DEFAULT_MODEL,
    IMAGE_EDIT_MODEL,
    analyze_image_text,
    build_slide_from_image,
    remove_text_from_image,
)
from src.gemini_client import reset_client
from google.genai.errors import ClientError as GeminiClientError, ServerError as GeminiServerError

try:
    from openai import APIStatusError as OpenAIStatusError, APIConnectionError as OpenAIConnectionError, APITimeoutError as OpenAITimeoutError
except ImportError:
    # openai 包未安装时使用占位类，避免导入错误
    class OpenAIStatusError(Exception): pass
    class OpenAIConnectionError(Exception): pass
    class OpenAITimeoutError(Exception): pass


def _init_page_config():
    """初始化页面基本配置（标题、布局、主题等）"""
    st.set_page_config(
        page_title="信息图转PPTX生成器",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📊 信息图转 PPTX 生成器")
    st.markdown("""
    上传信息图图片，使用 **Gemini AI** 自动分析文本元素、擦除文字并生成可编辑的 **PPTX �灯片**。
    """)


def _render_sidebar():
    """
    渲染侧边栏配置面板

    Returns:
        dict: 包含所有用户配置参数的字典
    """
    with st.sidebar:
        st.header("⚙️ 配置")

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Google Gemini API 密钥，也可通过环境变量 GEMINI_API_KEY 设置",
        )

        base_url = st.text_input(
            "API Base URL（可选）",
            value=os.getenv("GEMINI_BASE_URL", ""),
            help="第三方 API 端点，留空则使用 Google 官方端点",
        )

        st.divider()

        model = st.text_input(
            "文本分析模型",
            value=DEFAULT_MODEL,
            help="用于分析图片中文本元素位置的模型",
        )

        image_edit_model = st.text_input(
            "图片编辑模型",
            value=IMAGE_EDIT_MODEL,
            help="用于擦除图片上文字的模型",
        )

        st.divider()

        skip_text = st.checkbox(
            "跳过文本叠加层",
            value=False,
            help="勾选后仅使用原图作为背景，不添加可编辑文本框",
        )

        keep_clean = st.checkbox(
            "保留干净背景图",
            value=False,
            help="保留 Gemini 擦除文字后的干净背景图片",
        )

        st.divider()

        # 连接测试按钮
        if st.button("🔗 测试 API 连接", use_container_width=True):
            with st.spinner("正在测试连接..."):
                try:
                    # 用当前输入的字段值构建临时配置（config 尚未返回，使用局部变量）
                    _test_config = {
                        "api_key": api_key,
                        "base_url": base_url,
                        "model": model,
                        "image_edit_model": image_edit_model,
                        "skip_text": skip_text,
                        "keep_clean": keep_clean,
                    }
                    _apply_env(_test_config)
                    from src.gemini_client import generate_text
                    test_resp = generate_text(
                        model=model,
                        system_prompt="你是一个测试助手",
                        user_prompt="回复'OK'即可，不要输出其他内容",
                    )
                    st.success(f"✅ 连接成功！模型响应: {test_resp[:50]}")
                except (GeminiClientError, OpenAIStatusError) as e:
                    status_code = getattr(e, 'status_code', None) or 400
                    if status_code == 401:
                        st.error(f"❌ **API Key 无效或已过期** (HTTP {status_code})\n\n请检查你的 API Key 是否正确。")
                    elif status_code == 400:
                        st.error(f"❌ **请求参数错误** (HTTP {status_code})\n\n可能原因：模型名称不支持、请求格式不兼容等。")
                    else:
                        st.error(f"❌ **API 错误** (HTTP {status_code})\n\n{e}")
                except (GeminiServerError, OpenAIConnectionError, OpenAITimeoutError) as e:
                    status_code = getattr(e, 'status_code', None) or 502
                    endpoint = base_url or "Google 官方端点"
                    if status_code in (502, 503, 504):
                        st.error(f"""
❌ **服务端错误** (HTTP {status_code})

当前端点: `{endpoint}`

**可能原因：**
- 第三方代理服务暂时不可用或过载
- 网络连接不稳定
- 代理服务器无法访问上游 API

**建议操作：**
1. 稍后重试
2. 检查代理服务是否正常运行
3. 尝试切换到其他 API 端点
""")
                    else:
                        st.error(f"❌ **服务/网络错误**: {e}")
                except Exception as e:
                    st.error(f"❌ 连接失败: {e}")

        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "image_edit_model": image_edit_model,
            "skip_text": skip_text,
            "keep_clean": keep_clean,
        }


def _validate_config(config: dict) -> tuple[bool, str]:
    """
    验证用户配置是否完整有效

    Args:
        config: 用户配置字典

    Returns:
        tuple[bool, str]: (是否有效, 错误信息)
    """
    if not config["api_key"]:
        return False, "请输入 Gemini API Key"
    if not config["model"]:
        return False, "请指定文本分析模型"
    if not config["image_edit_model"]:
        return False, "请指定图片编辑模型"
    return True, ""


def _apply_env(config: dict):
    """
    将用户输入的配置写入环境变量，并重置 Gemini 客户端缓存

    每次应用配置时强制重置客户端，确保使用最新的 API Key 和 Base URL
    """
    os.environ["GEMINI_API_KEY"] = config["api_key"]
    if config["base_url"]:
        os.environ["GEMINI_BASE_URL"] = config["base_url"]
    elif "GEMINI_BASE_URL" in os.environ:
        del os.environ["GEMINI_BASE_URL"]

    # 重置客户端缓存，强制下次调用时创建新实例
    reset_client()


def _process_single_image_with_progress(
    image_path: str,
    config: dict,
    status_container,
) -> str | None:
    """
    处理单张图片，并在 Streamlit 状态容器中展示实时进度

    Args:
        image_path: 待处理的图片路径
        config: 用户配置参数
        status_container: Streamlit 状态容器对象，用于输出进度日志

    Returns:
        str | None: 处理成功返回生成的 pptx 路径，失败返回 None
    """
    import json
    import re
    from pptx import Presentation
    from pptx.util import Inches, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    model = config["model"]
    image_edit_model = config["image_edit_model"]
    skip_text = config["skip_text"]
    keep_clean = config["keep_clean"]

    text_elements = None
    clean_image_path = image_path

    if not skip_text:
        try:
            with status_container.status("🔍 **Step 1/3**: 分析图片中的文本元素...", expanded=True):
                st.write(f"正在调用 `{model}` 分析图片...")
                data = analyze_image_text(image_path, model)
                text_elements = data.get("text_elements", [])
                graphic_elements = data.get("graphic_elements", None)
                st.success(f"✅ 提取到 **{len(text_elements)}** 个文本元素")
                if text_elements:
                    elem_preview = "\n".join(
                        f"- 「{e['text']}」({e.get('left_pct',0)}%, {e.get('top_pct',0)}%)"
                        for e in text_elements[:5]
                    )
                    if len(text_elements) > 5:
                        elem_preview += f"\n- ... 共 {len(text_elements)} 个"
                    st.code(elem_preview, language=None)

            with status_container.status("🧹 **Step 2/3**: 使用 Gemini 擦除图片文字...", expanded=True):
                if keep_clean:
                    base, ext = os.path.splitext(image_path)
                    clean_output = f"{base}_clean.png"
                else:
                    fd, clean_output = tempfile.mkstemp(suffix=".png")
                    os.close(fd)

                st.write(f"正在调用 `{image_edit_model}` 擦除文字...")
                remove_text_from_image(
                    image_path,
                    clean_output,
                    text_elements,
                    graphic_elements,
                    image_edit_model,
                )
                st.success("✅ 文字已擦除，干净背景图已生成")
                clean_image_path = clean_output

        except Exception as e:
            status_container.error(f"❌ 处理失败: {e}")
            st.exception(e)
            clean_image_path = image_path
            text_elements = None

    with status_container.status("📄 **Step 3/3**: 构建 PPTX 幻灯片...", expanded=True):
        fd_out, output_path = tempfile.mkstemp(suffix=".pptx")
        os.close(fd_out)

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        build_slide_from_image(prs, clean_image_path, text_elements)

        prs.save(output_path)
        st.success("✅ PPTX 已生成！")

        if clean_image_path != image_path and not keep_clean:
            try:
                os.unlink(clean_image_path)
            except OSError:
                pass

        return output_path


def _process_multiple_images_with_progress(
    image_paths: list[str],
    config: dict,
    status_container,
) -> str | None:
    """
    处理多张图片，逐张处理后合并为一个 PPTX 文件

    Args:
        image_paths: 待处理的图片路径列表（按顺序）
        config: 用户配置参数
        status_container: Streamlit 状态容器对象

    Returns:
        str | None: 成功返回 pptx 路径，失败返回 None
    """
    from pptx import Presentation
    from pptx.util import Inches

    model = config["model"]
    image_edit_model = config["image_edit_model"]
    skip_text = config["skip_text"]
    keep_clean = config["keep_clean"]

    total = len(image_paths)
    all_clean_paths = []

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for i, img_path in enumerate(image_paths):
        with status_container.status(
            f"🖼️ **处理第 {i + 1}/{total} 张图片**: `{Path(img_path).name}`", expanded=True
        ):
            text_elements = None
            clean_image_path = img_path

            if not skip_text:
                try:
                    st.write(f"  🔍 分析文本元素...")
                    data = analyze_image_text(img_path, model)
                    text_elements = data.get("text_elements", [])
                    graphic_elements = data.get("graphic_elements", None)
                    st.write(f"  ✅ 提取到 {len(text_elements)} 个文本元素")

                    st.write(f"  🧹 擦除文字...")
                    if keep_clean:
                        base, ext = os.path.splitext(img_path)
                        clean_output = f"{base}_clean.png"
                    else:
                        fd, clean_output = tempfile.mkstemp(suffix=".png")
                        os.close(fd)

                    remove_text_from_image(
                        img_path, clean_output,
                        text_elements, graphic_elements, image_edit_model,
                    )
                    clean_image_path = clean_output
                    all_clean_paths.append(clean_output)
                    st.write(f"  ✅ 文字已擦除")

                except Exception as e:
                    st.warning(f"  ⚠️ 处理失败，使用原图: {e}")
                    clean_image_path = img_path

            st.write(f"  📄 构建幻灯片...")
            build_slide_from_image(prs, clean_image_path, text_elements)
            st.write(f"  ✅ 第 {i + 1} 页完成")

            progress_val = (i + 1) / total
            status_container.progress(progress_val, text=f"整体进度: {i+1}/{total}")

    fd_out, output_path = tempfile.mkstemp(suffix=".pptx")
    os.close(fd_out)
    prs.save(output_path)

    if not keep_clean:
        for p in all_clean_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return output_path


def _render_upload_area():
    """
    渲染文件上传区域

    Returns:
        tuple[list, list]: (上传的文件对象列表, 保存后的临时文件路径列表)
    """
    uploaded = st.file_uploader(
        "📁 上传信息图图片（支持 JPG/PNG/BMP，可多选）",
        type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"],
        accept_multiple_files=True,
        help="支持同时上传多张图片，将按上传顺序生成多页 PPTX",
    )

    if not uploaded:
        return [], []

    saved_paths = []
    temp_dir = tempfile.mkdtemp(prefix="img2pptx_")

    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        save_path = os.path.join(temp_dir, f.name)
        with open(save_path, "wb") as out:
            out.write(f.read())
        saved_paths.append(save_path)

        col_idx = i % min(len(uploaded), 4)
        with cols[col_idx]:
            st.image(save_path, caption=f.name, width="auto")

    st.info(f"已上传 **{len(uploaded)}** 张图片，点击下方按钮开始生成 PPTX")
    return uploaded, saved_paths


def _render_result(output_path: str | None, config: dict):
    """
    渲染处理结果区域，提供下载按钮

    Args:
        output_path: 生成的 PPTX 文件路径，为 None 表示未生成
        config: 用户配置（用于显示文件名）
    """
    if output_path is None:
        return

    st.divider()
    st.header("✅ 生成完成！")

    col_download, col_info = st.columns([1, 2])

    with col_download:
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 下载 PPTX 文件",
                data=f,
                file_name=f"幻灯片_{int(time.time())}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True,
            )

    with col_info:
        file_size = os.path.getsize(output_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.1f} MB"
        st.markdown(f"""
        - 📦 文件大小：**{size_str}**
        - 🤖 文本分析模型：`{config['model']}`
        - 🎨 图片编辑模型：`{config['image_edit_model']}`
        - 📝 文本叠加：{'❌ 已关闭' if config['skip_text'] else '✅ 已开启'}
        """)


def main():
    """Streamlit 应用主入口"""
    _init_page_config()
    config = _render_sidebar()

    # 先渲染上传区域，让用户可以看到上传按钮
    uploaded_files, image_paths = _render_upload_area()

    # 验证配置，如果不完整则显示警告但不阻止界面渲染
    valid, err_msg = _validate_config(config)
    if not valid:
        st.sidebar.warning(err_msg)
        if not image_paths:
            st.info("👆 请先在侧边栏输入 API Key，然后上传图片")
            st.stop()
    else:
        _apply_env(config)

    if not image_paths:
        st.info("👆 请先在上方上传图片文件")
        st.stop()

    if st.button("🚀 开始生成 PPTX", type="primary", use_container_width=True):
        progress_area = st.container()
        with progress_area:
            status = st.status("⏳ 准备中...", expanded=True)

            start_time = time.time()

            try:
                if len(image_paths) == 1:
                    output_path = _process_single_image_with_progress(
                        image_paths[0], config, status
                    )
                else:
                    output_path = _process_multiple_images_with_progress(
                        image_paths, config, status
                    )

                elapsed = time.time() - start_time
                if output_path:
                    status.update(
                        state="complete",
                        label=f"✅ 全部完成！耗时 {elapsed:.1f}s",
                    )
                    _render_result(output_path, config)
                else:
                    status.update(state="error", label="❌ 生成失败")

            except Exception as e:
                elapsed = time.time() - start_time
                status.update(
                    state="error",
                    label=f"❌ 处理出错（耗时 {elapsed:.1f}s）",
                )

                # 针对不同错误类型给出更友好的诊断信息
                if isinstance(e, (GeminiServerError, OpenAIConnectionError, OpenAITimeoutError)):
                    status_code = getattr(e, 'status_code', None) or 500
                    endpoint = config.get("base_url") or "Google 官方端点"
                    if status_code in (502, 503, 504):
                        st.error(f"""
### 服务端错误 HTTP {status_code}

当前 API 端点: **{endpoint}**

**这是代理服务器返回的错误，可能原因：**
- 🔴 第三方代理服务暂时不可用或过载
- 🟡 代理服务器无法连接到上游 API

**建议操作：**
1. **稍后重试** — 可能是临时性故障
2. 检查代理服务状态是否正常
3. 尝试清空 Base URL 使用 Google 官方直连
4. 联系代理服务提供方确认可用性
""")
                    else:
                        st.error(f"❌ 服务/网络错误 (HTTP {status_code}): {e}")
                elif isinstance(e, (GeminiClientError, OpenAIStatusError)):
                    status_code = getattr(e, 'status_code', None) or 400
                    if status_code == 401:
                        st.error("❌ **API Key 无效或已过期** — 请检查侧边栏中输入的 API Key 是否正确")
                    elif status_code == 400:
                        st.error("❌ **请求参数错误** — 可能模型名称不支持，请检查模型配置")
                    elif status_code == 429:
                        st.error("❌ **请求频率超限** — 请稍后重试")
                    else:
                        st.error(f"❌ 客户端错误 (HTTP {status_code}): {e}")
                else:
                    st.error(f"发生错误: {e}")

                st.exception(e)


if __name__ == "__main__":
    main()
