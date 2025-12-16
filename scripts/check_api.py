#!/usr/bin/env python3
"""快速测试OpenAI API - 极简版本"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("=" * 60)
print("🧪 快速测试 OpenAI API")
print("=" * 60)

# 检查API Key
if not OPENAI_API_KEY:
    print("❌ 未找到 OPENAI_API_KEY")
    print("💡 请在 .env 文件中配置: OPENAI_API_KEY=your_key")
    exit(1)

print(f"✅ API Key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")

# 检查代理
proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if proxy:
    print(f"🌐 代理: {proxy}")
else:
    print("⚠️  未配置代理")

# 测试连接
print("\n🔄 测试连接...")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=10
    )
    
    print("\n" + "=" * 60)
    print("✅ 成功！OpenAI API 连接正常")
    print("=" * 60)
    print(f"响应: {response.choices[0].message.content}")
    print(f"模型: {response.model}")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 失败！无法连接 OpenAI API")
    print("=" * 60)
    print(f"错误: {str(e)}")
    
    if "Connection" in str(e) or "timeout" in str(e).lower():
        print("\n💡 解决方案: 配置代理")
        print("在 .env 中添加:")
        print("HTTP_PROXY=http://127.0.0.1:7890")
        print("HTTPS_PROXY=http://127.0.0.1:7890")
    
    print("=" * 60)
    exit(1)

