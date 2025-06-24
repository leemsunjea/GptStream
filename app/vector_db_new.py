# app/vector_db.py

import os
import asyncio
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import List

# 환경변수에서 OPENAI_API_KEY를 읽어옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...your-key...")
client = OpenAI(api_key=OPENAI_API_KEY)

async def stream_chat(messages: List[ChatCompletionMessageParam]) -> str:
    """OpenAI API를 이용한 스트림 챗봇 함수"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=False
            )
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        return f"[ERROR] 챗봇 응답 실패: {e}"
