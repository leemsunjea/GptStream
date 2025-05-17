import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 템플릿/정적파일 경로
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
# app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# 홈 페이지
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 챗봇 스트리밍 응답
@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")

    async def event_stream():
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.get("content")
                if content:
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)  # ⭐ flush 보장
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")