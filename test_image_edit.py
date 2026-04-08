"""
测试图片编辑功能
用于验证 gemini-3.1-flash-image-preview 模型是否可用
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 测试配置
TEST_IMAGE_PATH = "examples/俄罗斯地理和文化/01.jpg"  # 使用示例图片
API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://api.vectorengine.ai/v1"  # 你提供的代理地址

def test_native_gemini():
    """测试 Google 官方 Gemini API"""
    print("=" * 60)
    print("测试 1: Google 官方 Gemini API")
    print("=" * 60)
    
    if not API_KEY:
        print("❌ 错误: 未设置 GEMINI_API_KEY 环境变量")
        return False
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=API_KEY)
        
        # 测试文本生成
        print("\n1. 测试文本生成...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello",
        )
        print(f"✅ 文本生成成功: {response.text[:50]}...")
        
        # 测试图片编辑
        print("\n2. 测试图片编辑 (gemini-3.1-flash-image-preview)...")
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"❌ 测试图片不存在: {TEST_IMAGE_PATH}")
            return False
            
        with open(TEST_IMAGE_PATH, "rb") as f:
            img_bytes = f.read()
        
        parts = [
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text="Remove all text from this image"),
        ]
        
        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=parts,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        
        # 检查是否返回图片
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                print(f"✅ 图片编辑成功！返回 {len(part.inline_data.data)} 字节")
                return True
        
        print("❌ 图片编辑未返回图片数据")
        print(f"响应内容: {response}")
        return False
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_proxy_api():
    """测试第三方代理 API"""
    print("\n" + "=" * 60)
    print("测试 2: 第三方代理 API (vectorengine.ai)")
    print("=" * 60)
    
    if not API_KEY:
        print("❌ 错误: 未设置 GEMINI_API_KEY 环境变量")
        return False
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        # 测试文本生成
        print("\n1. 测试文本生成...")
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(f"✅ 文本生成成功: {response.choices[0].message.content[:50]}...")
        
        # 测试图片编辑
        print("\n2. 测试图片编辑 (gemini-3.1-flash-image-preview)...")
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"❌ 测试图片不存在: {TEST_IMAGE_PATH}")
            return False
        
        with open(TEST_IMAGE_PATH, "rb") as f:
            image_file = f.read()
        
        response = client.images.edit(
            model="gemini-3.1-flash-image-preview",
            prompt="Remove all text from this image",
            image=image_file,
            n=1,
        )
        
        if response.data and response.data[0].b64_json:
            import base64
            img_data = base64.b64decode(response.data[0].b64_json)
            print(f"✅ 图片编辑成功！返回 {len(img_data)} 字节")
            return True
        elif response.data and response.data[0].url:
            print(f"✅ 图片编辑成功！返回 URL: {response.data[0].url}")
            return True
        else:
            print("❌ 图片编辑未返回图片数据")
            print(f"响应: {response}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    print("图片编辑功能测试工具")
    print(f"测试图片: {TEST_IMAGE_PATH}")
    print(f"API Key: {'已设置' if API_KEY else '未设置'}")
    print()
    
    # 运行测试
    results = []
    
    # 测试官方 API
    results.append(("Google 官方 API", test_native_gemini()))
    
    # 测试代理 API
    results.append(("vectorengine.ai 代理", test_proxy_api()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")
    
    print("\n建议:")
    print("- 如果官方 API 通过但代理失败，说明代理不支持图片编辑")
    print("- 如果都失败，说明 gemini-3.1-flash-image-preview 模型已下线")
    print("- 建议勾选「跳过文本叠加层」选项作为替代方案")


if __name__ == "__main__":
    main()
