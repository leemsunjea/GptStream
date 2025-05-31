# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session, engine, Base
from app.routes import home, chat, vector
from app.vector_db import load_faiss_and_docstore
from db.models import UserPreference
from sqlalchemy import select
from fastapi.responses import JSONResponse

app = FastAPI()

DEFAULT_SYSTEM_PROMPTS = [
    "You are a friendly and helpful AI assistant.",
    "You are a professional AI assistant that provides concise and accurate information.",
    "You are a creative AI assistant that can help with brainstorming and writing."
]

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    print("🚀 애플리케이션 시작 중...")
    
    # 1. 데이터베이스 테이블 생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ 데이터베이스 테이블 생성 완료")
    
    # 2. 기본 시스템 프롬프트 추가
    async with async_session() as session:
        stmt = select(UserPreference).where(UserPreference.user_id == "default_system")
        result = await session.execute(stmt)
        existing_default = result.scalars().first()

        if not existing_default and DEFAULT_SYSTEM_PROMPTS:
            default_prompt = UserPreference(
                user_id="default_system",
                system_prompt=DEFAULT_SYSTEM_PROMPTS[0]
            )
            session.add(default_prompt)
            await session.commit()
            print("✅ 기본 시스템 프롬프트 추가 완료")
        else:
            print("ℹ️ 기본 시스템 프롬프트가 이미 존재함")

    # 3. FAISS 인덱스 및 문서 저장소 로드
    await load_faiss_and_docstore()
    print("✅ FAISS 인덱스 로드 완료")

async def get_db():
    """데이터베이스 세션 의존성"""
    async with async_session() as session:
        yield session

@app.get("/db-test")
async def test_db_connection(db: AsyncSession = Depends(get_db)):
    """데이터베이스 연결 테스트"""
    try:
        result = await db.execute(select(1))
        return {"status": "success", "message": "데이터베이스 연결 성공"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"데이터베이스 연결 실패: {str(e)}"}
        )

@app.exception_handler(504)
async def gateway_timeout_handler(request, exc):
    """게이트웨이 타임아웃 핸들러"""
    return JSONResponse(
        status_code=504,
        content={"detail": "요청 처리 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."}
    )

# 라우터 등록
app.include_router(home.router)
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(vector.router, prefix="/vector", tags=["Vector"])
