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
    Column("pdf_name", String),
    Column("page_number", Integer),
    Column("content", Text),
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer, ForeignKey("documents.id")),
    Column("embedding", LargeBinary),
)