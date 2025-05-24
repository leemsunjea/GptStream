# app/main.py

import os
import uuid
from fastapi import FastAPI, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_async_session, init_db, get_or_create_session_id, SESSION_COOKIE_NAME
from app.routes import home, chat, vector
from db.models import Base
import asyncio
from contextlib import asynccontextmanager

# 애플리케이션 수명 주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 초기화
    await init_db()
    print("✅ 애플리케이션 초기화 완료")
    yield
    # 앱 종료 시 정리
    print("🛑 애플리케이션 종료 중...")

# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 미들웨어
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    # 세션 ID 가져오기 또는 생성
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 요청 처리
    response = await call_next(request)
    
    # 새로운 세션 ID가 생성된 경우 쿠키 설정
    if not request.cookies.get(SESSION_COOKIE_NAME):
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            httponly=True,
            max_age=30 * 24 * 60 * 60,  # 30일
            samesite="lax"
        )
    
    return response

# DB 의존성 (세션 ID 기반)
async def get_db(request: Request):
    session_id = get_or_create_session_id(request)
    session_factory = await get_async_session(session_id)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

@app.get("/session-test")
async def test_session(request: Request):
    """세션 테스트용 엔드포인트"""
    session_id = get_or_create_session_id(request)
    return {
        "session_id": session_id,
        "message": "세션이 정상적으로 설정되었습니다.",
        "data_dir": os.path.join(os.getenv("DATA_DIR", "/tmp/gptstream"), "users", session_id)
    }

# 에러 핸들러
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "내부 서버 오류가 발생했습니다."}
    )

@app.exception_handler(504)
async def gateway_timeout_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=504,
        content={"success": False, "detail": "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."}
    )

# 기존 라우터 등록
app.include_router(home.router)
app.include_router(chat.router)
app.include_router(vector.router)  # 추가