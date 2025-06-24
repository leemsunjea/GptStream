# app/vector_db.py

import os
import asyncio
import requests
from openai import OpenAI
from typing import List, Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data: Dict = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    async def event_stream():
        content = await stream_chat(message, history)
        yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
