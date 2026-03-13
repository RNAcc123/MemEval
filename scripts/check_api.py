#!/usr/bin/env python3
"""Quick OpenAI API smoke test (minimal version)."""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("=" * 60)
print("🧪 Quick OpenAI API smoke test")
print("=" * 60)

# Check API key
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found")
    print("💡 Please configure it in .env: OPENAI_API_KEY=your_key")
    exit(1)

print(f"✅ API Key: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")

# Check proxy settings
proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if proxy:
    print(f"🌐 Proxy: {proxy}")
else:
    print("⚠️  Proxy is not configured")

# Test connectivity
print("\n🔄 Testing connectivity...")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=10
    )
    
    print("\n" + "=" * 60)
    print("✅ Success! OpenAI API connection is working")
    print("=" * 60)
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ Failed! Unable to connect to OpenAI API")
    print("=" * 60)
    print(f"Error: {str(e)}")
    
    if "Connection" in str(e) or "timeout" in str(e).lower():
        print("\n💡 Possible fix: configure a proxy")
        print("Add the following to .env:")
        print("HTTP_PROXY=http://127.0.0.1:7890")
        print("HTTPS_PROXY=http://127.0.0.1:7890")
    
    print("=" * 60)
    exit(1)

