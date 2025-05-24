# models.py
from sqlalchemy import Column, Integer, Text, DateTime, func, Table, String, ForeignKey, LargeBinary, MetaData
from db.database import Base  # 수정: db에서가 아니라 db.database에서 import

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String(64), nullable=False, index=True),  # 세션 ID 추가
    Column("pdf_name", String(255), nullable=False),  # NOT NULL 제약조건 추가
    Column("page_number", Integer, nullable=False),   # NOT NULL 제약조건 추가
    Column("content", Text, nullable=False),          # NOT NULL 제약조건 추가
    Column("created_at", DateTime, server_default=func.now()),  # 생성일시 추가
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("session_id", String(64), nullable=False, index=True),  # 세션 ID 추가
    Column("embedding", LargeBinary, nullable=False),  # NOT NULL 제약조건 추가
    Column("created_at", DateTime, server_default=func.now()),  # 생성일시 추가
)