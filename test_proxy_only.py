"""
测试第三方代理 API (vectorengine.ai)
用于验证代理是否支持图片编辑功能
"""
import os
import sys
import base64
from dotenv import load_dotenv

load_dotenv()

# 测试配置
TEST_IMAGE_PATH = "examples/俄罗斯地理和文化/01.jpg"
API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://api.vectorengine.ai/v1"

def test_proxy_text():
    """测试代理的文本生成功能"""
    print("=" * 60)
    print("测试 1: 代理文本生成")
    print("=" * 60)
    
    if not API_KEY:
        print("[X] 错误: 未设置 GEMINI_API_KEY")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        print("\n调用模型: gemini-2.0-flash")
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
        )
        reply = response.choices[0].message.content
        print(f"[OK] 文本生成成功!")
        print(f"回复: {reply[:100]}...")
        return True
        
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False


def test_proxy_image_edit():
    """测试代理的图片编辑功能"""
    print("\n" + "=" * 60)
    print("测试 2: 代理图片编辑")
    print("=" * 60)
    
    if not API_KEY:
        print("[X] 错误: 未设置 GEMINI_API_KEY")
        return False
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[X] 测试图片不存在: {TEST_IMAGE_PATH}")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        # 读取测试图片
        print(f"\n读取图片: {TEST_IMAGE_PATH}")
        with open(TEST_IMAGE_PATH, "rb") as f:
            image_data = f.read()
        print(f"图片大小: {len(image_data)} 字节")
        
        # 测试图片编辑
        print("\n调用模型: gemini-3.1-flash-image-preview")
        print("提示词: Remove all text from this image")
        
        try:
            response = client.images.edit(
                model="gemini-3.1-flash-image-preview",
                prompt="Remove all text from this image",
                image=image_data,
                n=1,
            )
            
            # 检查响应
            if response.data and len(response.data) > 0:
                if response.data[0].b64_json:
                    img_bytes = base64.b64decode(response.data[0].b64_json)
                    print(f"[OK] 图片编辑成功! 返回 {len(img_bytes)} 字节")
                    return True
                elif response.data[0].url:
                    print(f"[OK] 图片编辑成功! 返回 URL: {response.data[0].url}")
                    return True
            
            print("[X] 图片编辑未返回数据")
            print(f"响应: {response}")
            return False
            
        except Exception as e:
            error_msg = str(e)
            print(f"[X] 图片编辑失败: {error_msg}")
            
            # 分析错误
            if "model" in error_msg.lower() or "not found" in error_msg.lower():
                print("\n>>> 模型名称可能不正确")
                print(">>> 建议尝试其他模型名称:")
                print("    - gemini-2.0-flash-exp")
                print("    - gemini-2.0-pro")
                print("    - 或咨询代理提供商支持的模型列表")
            
            if "not supported" in error_msg.lower() or "unsupported" in error_msg.lower():
                print("\n>>> 该代理不支持图片编辑功能")
                print(">>> 建议: 使用「跳过文本叠加层」选项")
            
            return False
            
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False


def test_proxy_vision():
    """测试代理的视觉分析功能"""
    print("\n" + "=" * 60)
    print("测试 3: 代理视觉分析")
    print("=" * 60)
    
    if not API_KEY:
        print("[X] 错误: 未设置 GEMINI_API_KEY")
        return False
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[X] 测试图片不存在: {TEST_IMAGE_PATH}")
        return False
    
    try:
        from openai import OpenAI
        import base64
        
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        # 读取图片并转为 base64
        print(f"\n读取图片: {TEST_IMAGE_PATH}")
        with open(TEST_IMAGE_PATH, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode('utf-8')
        
        print("调用模型: gemini-2.0-flash (vision)")
        
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        
        description = response.choices[0].message.content
        print(f"[OK] 视觉分析成功!")
        print(f"描述: {description[:100]}...")
        return True
        
    except Exception as e:
        print(f"[X] 错误: {e}")
        return False


def main():
    print("第三方代理 API 测试工具")
    print(f"代理地址: {BASE_URL}")
    print(f"API Key: {'已设置' if API_KEY else '未设置'}")
    print()
    
    # 运行测试
    results = []
    
    results.append(("文本生成", test_proxy_text()))
    results.append(("视觉分析", test_proxy_vision()))
    results.append(("图片编辑", test_proxy_image_edit()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, success in results:
        status = "[OK] 通过" if success else "[X] 失败"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 60)
    print("结论与建议")
    print("=" * 60)
    
    text_ok = results[0][1]
    vision_ok = results[1][1]
    edit_ok = results[2][1]
    
    if text_ok and vision_ok and edit_ok:
        print("[OK] 所有功能正常!")
        print("代理完全支持该应用的所有功能")
    elif text_ok and vision_ok and not edit_ok:
        print("[INFO] 代理支持文本和视觉分析，但不支持图片编辑")
        print("建议: 在 Streamlit 中勾选「跳过文本叠加层」选项")
        print("      这样可以直接用原图生成 PPTX")
    elif text_ok and not vision_ok:
        print("[WARN] 代理仅支持基础文本功能")
        print("该应用可能无法正常工作")
    else:
        print("[X] 代理连接失败")
        print("请检查 API Key 和代理地址是否正确")


if __name__ == "__main__":
    main()
