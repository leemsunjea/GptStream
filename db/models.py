# models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, LargeBinary, Table, MetaData # Table, MetaData, LargeBinary 추가
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)  # 사용자 ID 필드 추가
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", String, nullable=False, index=True), # user_id 컬럼 추가 및 인덱스 설정
    Column("pdf_name", String),
    Column("page_number", Integer),
    Column("content", Text),
    Column("title", String, nullable=True), # 문서 제목
    Column("summary", Text, nullable=True), # 문서 요약
    Column("response_style", Text, nullable=True), # 응답 스타일
    Column("created_at", DateTime, server_default=func.now()) # 문서 업로드 시간
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", String, nullable=False, index=True), # user_id 컬럼 추가 및 인덱스 설정
    Column("document_id", Integer, ForeignKey("documents.id")),
    Column("embedding", LargeBinary),
)

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    system_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<UserPreference(user_id='{self.user_id}', system_prompt='{self.system_prompt[:20]}...')>"

# ChatHistory 모델에 user_id 외래 키 제약 조건 및 관계 설정 (선택 사항이지만 권장)
# class ChatHistory(Base):
#     __tablename__ = "chat_history"
#     # ... 기존 컬럼들 ...
#     user_id = Column(String, ForeignKey("user_preferences.user_id"), nullable=False) # ForeignKey 추가
#     user = relationship("UserPreference") # 관계 설정