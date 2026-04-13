"""DeepSeek API 客户端"""
import os
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def get_client() -> OpenAI:
    key = DEEPSEEK_API_KEY
    if not key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=key)


def test_connection():
    """测试 API 连通性"""
    client = get_client()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
        max_tokens=10,
    )
    reply = response.choices[0].message.content.strip()
    print(f"DeepSeek API 连通测试: {reply}")
    print(f"模型: {response.model}")
    print(f"Token 用量: {response.usage}")
    return True


if __name__ == "__main__":
    test_connection()
