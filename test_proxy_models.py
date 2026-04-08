"""
测试代理支持的不同模型名称格式
尝试找到可用的模型
"""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://api.vectorengine.ai/v1"

# 尝试不同的模型名称格式
TEST_MODELS = [
    # 标准 Gemini 模型
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro",
    "gemini-2.0-pro-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    
    # 图片编辑模型
    "gemini-3.1-flash-image-preview",
    "gemini-2.0-flash-image-preview",
    
    # 可能的分组前缀格式
    "限时特价/gemini-2.0-flash",
    "限时特价/gemini-1.5-flash",
    "优质gemini/gemini-2.0-flash",
    "官转gemini/gemini-2.0-flash",
    
    # OpenAI 格式
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet",
]

def test_model(model_name):
    """测试单个模型是否可用"""
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
        error_msg = str(e)
        # 提取关键错误信息
        if "无可用渠道" in error_msg:
            return False, "无可用渠道"
        elif "not found" in error_msg.lower():
            return False, "模型不存在"
        else:
            return False, error_msg[:100]


def main():
    print("=" * 70)
    print("测试代理支持的模型列表")
    print(f"代理: {BASE_URL}")
    print("=" * 70)
    
    working_models = []
    
    for model in TEST_MODELS:
        print(f"\n测试: {model}")
        success, result = test_model(model)
        
        if success:
            print(f"  [OK] 可用 - {result}")
            working_models.append(model)
        else:
            print(f"  [X] 失败 - {result}")
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    if working_models:
        print(f"\n找到 {len(working_models)} 个可用模型:")
        for m in working_models:
            print(f"  - {m}")
        
        print("\n建议:")
        print("1. 在 Streamlit 侧边栏使用上述可用模型名称")
        print("2. 文本分析模型和图片描述模型可以使用这些可用模型")
        print("3. 图片编辑功能可能需要特定的模型支持")
    else:
        print("\n[X] 没有找到可用的模型")
        print("\n可能原因:")
        print("- 你的 API Key 余额不足")
        print("- 代理服务商暂时无可用资源")
        print("- 需要联系代理服务商确认支持的模型列表")


if __name__ == "__main__":
    main()
