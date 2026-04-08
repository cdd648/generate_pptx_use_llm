"""
测试 gemini-3.1-pro-preview 模型
"""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://api.vectorengine.ai/v1"

def test_model(model_name):
    """测试模型"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        return True, response.choices[0].message.content[:50]
        
    except Exception as e:
        return False, str(e)[:100]


def main():
    print("=" * 60)
    print("测试 gemini-3.1-pro-preview")
    print("=" * 60)
    
    model = "gemini-3.1-pro-preview"
    print(f"\n测试: {model}")
    success, result = test_model(model)
    
    if success:
        print(f"[OK] 可用 - {result}")
        print("\n可以在 Streamlit 中使用这个模型作为文本分析模型")
    else:
        print(f"[X] 失败 - {result}")
        print("\n该模型在此代理下不可用")


if __name__ == "__main__":
    main()
