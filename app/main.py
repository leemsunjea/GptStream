import os
import asyncio
import json
from fastapi import FastAPI, Request, Depends, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import openai
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import numpy as np

from database import get_db, engine, Base
from faiss_manager import FaissManager
from logger import setup_logger, log_to_db

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI()
logger = setup_logger()
faiss_manager = FaissManager()

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
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# WebSocket 연결 관리
connected_clients = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 클라이언트로부터의 메시지 처리
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.remove(websocket)

# 홈 페이지
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 로그 조회 API
@app.get("/logs")
async def get_logs(db: Session = Depends(get_db)):
    logs = db.query(Log).order_by(Log.timestamp.desc()).limit(100).all()
    return logs

# FAISS 검색 API
@app.post("/search")
async def search_vectors(query_vector: list, k: int = 5):
    try:
        distances, indices = faiss_manager.search(np.array(query_vector), k)
        return {"distances": distances.tolist(), "indices": indices.tolist()}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise

# 챗봇 스트리밍 응답
@app.post("/chat/stream")
async def chat_stream(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    message = data.get("message", "")

    # 로그 저장
    log_to_db(db, "INFO", f"User message: {message}")

    async def event_stream():
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.get("content")
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)

        # 응답 완료 후 로그 저장
        log_to_db(db, "INFO", f"Bot response: {full_response}")
        
        # WebSocket을 통해 로그 전송
        for client in connected_clients:
            try:
                await client.send_text(json.dumps({
                    "type": "log",
                    "message": f"User: {message}\nBot: {full_response}"
                }))
            except Exception as e:
                logger.error(f"WebSocket send error: {str(e)}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")