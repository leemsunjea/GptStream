# app/vector_db.py

import os
import asyncio
import requests
from openai import OpenAI
from typing import List, Any

# 환경변수에서 OPENAI_API_KEY를 읽어옴
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_n8n_prompt(user_input: str, history: list) -> Any:
    """n8n에 user_input과 history를 보내고, messages를 받아옴"""
    payload = {
        "user_input": user_input,
        "chat_history": history
    }
    response = requests.post(
        "https://sunjea1149.app.n8n.cloud/webhook/prompt-rag",
        json=payload
    )
    return response.json()

async def stream_chat(user_input: str, history: list) -> str:
    """n8n에서 messages를 받아 OpenAI API로 전달하는 함수"""
    try:
        # n8n에서 messages 받아오기
        n8n_result = fetch_n8n_prompt(user_input, history)
        messages = n8n_result.get("messages", [])

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
