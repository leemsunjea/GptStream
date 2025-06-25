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
    # HTTP 요청에 타임아웃을 추가하는 것이 좋습니다.
    response = requests.post(
        "https://sunjea1149.app.n8n.cloud/webhook/prompt-rag",
        json=payload,
        timeout=10 # 타임아웃을 초 단위로 추가
    )
    response.raise_for_status() # 4xx 또는 5xx HTTP 오류 발생 시 예외 발생
    return response.json()

async def stream_chat_response(user_input: str, history: list):
    """n8n에서 messages를 받아 OpenAI API로 전달하는 함수 (실시간 스트리밍 지원)"""
    try:
        n8n_result = fetch_n8n_prompt(user_input, history)
        messages = n8n_result.get("messages", [])

        # !!!!! 중요: 디버그 메시지는 콘솔에만 출력하고, 클라이언트 스트림에는 포함하지 않습니다. !!!!!
        print(f"[DEBUG] n8n에서 받은 메시지: {messages}")

        # ... (구조 검증 로직) ...

        response_stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        # OpenAI에서 청크가 도착하는 즉시 이를 yield합니다.
        for chunk in response_stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content # OpenAI가 보내는 작은 텍스트 조각을 즉시 yield

    except requests.exceptions.RequestException as req_err:
        print(f"n8n 요청 실패: {req_err}")
        yield f"정보를 가져오는 중 오류가 발생했습니다 (n8n 통신 오류): {req_err}"
    except Exception as e:
        print(f"챗봇 응답 실패: {e}")
        yield f"챗봇에서 예상치 못한 오류가 발생했습니다: {e}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    data: Dict = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    async def event_generator():
        async for chunk in stream_chat_response(message, history):
            # 각 청크를 'data: [내용]\n\n' 형식의 완전한 SSE 이벤트로 만듭니다.
            yield f"data: {chunk}\n\n" # 서버는 이 줄을 yield하는 즉시 클라이언트로 보냅니다.
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")