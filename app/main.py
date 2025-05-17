import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)  # ✅ 이 줄 반드시 필요

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data: Dict = await request.json()
        message = data.get("message", "")
        logger.info(f"Received chat request: {message[:50]}...")  # 긴 메시지는 잘라서 로깅

        def event_stream():
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": message}],
                    stream=True
                )
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield f"data: {content}\n\n"
                yield "data: [DONE]\n\n"
                logger.info("Chat stream completed successfully")
            except Exception as e:
                logger.error(f"Error in chat stream: {str(e)}")
                raise

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise