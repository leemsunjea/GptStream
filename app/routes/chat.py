# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store  # vector 연동
from openai import OpenAI
from sqlalchemy import select

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")
    chat_id = data.get("chatId", "default")

    # 이전 대화 기록 불러오기
    async with async_session() as session:
        previous_chats = await session.execute(
            select(ChatHistory).where(ChatHistory.chat_id == chat_id).order_by(ChatHistory.created_at)
        )
        chat_history = previous_chats.scalars().all()

    # 대화 기록을 시스템 프롬프트에 추가
    history_text = "\n".join([f"User: {chat.user_message}\nBot: {chat.bot_response}" for chat in chat_history])
    system_prompt = (
        f"이것은 {chat_id} 챗봇입니다. 이전 대화 기록을 참고하여 답변하세요.\n\n"
        f"대화 기록:\n{history_text}\n\n"
        "사용자의 질문에 답변해주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    # OpenAI API 호출
    response = await client.chat_completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    # 응답 저장
    bot_response = ""
    async for chunk in response:
        if "content" in chunk.choices[0].delta:
            bot_response += chunk.choices[0].delta.content

    async with async_session() as session:
        async with session.begin():
            session.add(ChatHistory(chat_id=chat_id, user_message=message, bot_response=bot_response))

    return StreamingResponse(event_stream(bot_response), media_type="text/event-stream")

async def event_stream(bot_response):
    yield f"data: {bot_response}\n"

@router.post("/chat/update_system_prompt")
async def update_system_prompt(request: Request):
    data = await request.json()
    new_prompt = data.get("prompt", "")
    chat_id = "system_prompt"  # 고정된 chat_id 사용

    if not new_prompt:
        return {"error": "프롬프트 내용이 비어 있습니다."}

    # 새로운 프롬프트 저장
    async with async_session() as session:
        async with session.begin():
            session.add(ChatHistory(chat_id=chat_id, user_message=new_prompt, bot_response=""))

    # 이전 프롬프트 불러오기
    async with async_session() as session:
        previous_prompts = await session.execute(
            select(ChatHistory).where(ChatHistory.chat_id == chat_id).order_by(ChatHistory.created_at)
        )
        prompts = previous_prompts.scalars().all()

    prompt_texts = [prompt.user_message for prompt in prompts]

    return {"prompts": prompt_texts}