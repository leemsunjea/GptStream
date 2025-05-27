# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session, engine, Base
from app.routes import home, chat, vector # vector 라우터 명시적 import
from app.vector_db import load_faiss_and_docstore
from db.models import UserPreference # documents, embeddings는 Base.metadata를 통해 인식됨
from sqlalchemy import select, func, text as sql_text # text 추가
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

# 기본 시스템 프롬프트 목록
DEFAULT_SYSTEM_PROMPTS = [
    "You are a friendly and helpful AI assistant.",
    "You are a professional AI assistant that provides concise and accurate information.",
    "You are a creative AI assistant that can help with brainstorming and writing."
]

@app.on_event("startup")
async def unified_startup_event(): # 하나의 startup 함수로 통합
    # 1. 데이터베이스 테이블 생성 (스키마가 없거나 정의와 다를 경우 적용)
    async with engine.begin() as conn:
        # 개발 중이고 스키마 변경으로 인해 테이블을 완전히 재생성해야 할 때만 다음 줄의 주석을 해제합니다.
        # 주의: 이 작업을 수행하면 해당 테이블의 모든 데이터가 삭제됩니다.
        # await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        print("✅ 데이터베이스 테이블 생성/확인 완료.")

    # 2. (주의!) 개발/테스트 시에만 기존 문서 및 임베딩 데이터 초기화 (현재 활성화됨)
    # 프로덕션 환경이나 데이터 보존이 필요하다면 아래 블록을 주석 처리하거나 삭제해야 합니다.
    async with async_session() as session:
        async with session.begin():
            # 명시적으로 테이블 객체를 import 하여 사용
            from db.models import embeddings as embeddings_table_obj, documents as documents_table_obj
            # await session.execute(embeddings_table_obj.delete())
            # await session.execute(documents_table_obj.delete())
            # await session.commit() # begin() 컨텍스트 매니저가 자동으로 커밋/롤백 처리
            # print("⚠️ 기존 문서 및 임베딩 데이터 초기화 완료 (서버 시작 시마다 실행됨).")
            print("ℹ️ 문서 및 임베딩 데이터 초기화 코드가 주석 처리되었습니다.")

    # 3. 기본 시스템 프롬프트 추가 (중복 방지)
    async with async_session() as session:
        stmt = select(UserPreference).where(UserPreference.user_id == "default_system")
        result = await session.execute(stmt)
        existing_default_user_pref = result.scalars().first()

        if not existing_default_user_pref:
            if DEFAULT_SYSTEM_PROMPTS:
                default_prompt_entry = UserPreference(
                    user_id="default_system",
                    system_prompt=DEFAULT_SYSTEM_PROMPTS[0]
                )
                session.add(default_prompt_entry)
                await session.commit()
                print(f"✅ 기본 시스템 프롬프트 ('{DEFAULT_SYSTEM_PROMPTS[0]}') 추가 완료.")
            else:
                print("⚠️ DEFAULT_SYSTEM_PROMPTS 목록이 비어 있어, 기본 시스템 프롬프트를 추가할 수 없습니다.")
        else:
            print(f"ℹ️ 기존 기본 시스템 프롬프트 ('{existing_default_user_pref.system_prompt}')가 이미 존재합니다.")

    # 4. FAISS 인덱스 및 문서 저장소 로드
    # 이 시점에는 테이블이 존재하고, (선택적으로) 데이터가 초기화되었으며, 기본 프롬프트가 준비된 상태입니다.
    await load_faiss_and_docstore()
    print("✅ FAISS 인덱스 및 문서 저장소 로드 완료.")


# DB 의존성
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/db-test")
async def test_db_connection(db: AsyncSession = Depends(get_db)): # 함수명 test -> test_db_connection (test는 pytest와 충돌 가능성)
    try:
        # SQLAlchemy 2.0에서는 text()를 사용하여 문자열 SQL을 실행하는 것이 권장됩니다.
        result = await db.execute(sql_text("SELECT 1"))
        if result.scalar_one() == 1:
            return {"db_connection": "success"}
        else:
            # 이 경우는 거의 발생하지 않지만, 방어적으로 추가
            return {"db_connection": "failed", "detail": "Query did not return 1"}
    except Exception as e:
        return {"db_connection": "error", "detail": str(e)}


@app.exception_handler(504)
async def gateway_timeout_handler(request, exc):
  return JSONResponse(
    status_code=504,
    content={"success": False, "logs": ["서버 타임아웃 발생"], "detail": "Gateway Timeout"}
  )

# 라우터 등록
app.include_router(home.router)
app.include_router(chat.router)
app.include_router(vector.router) # app.routes.vector.router가 여기서 올바르게 포함됩니다.
# app.include_router(router) # 이 줄은 app.routes.vector.router를 의미하며, 위에서 명시적으로 포함했으므로 중복. 제거.