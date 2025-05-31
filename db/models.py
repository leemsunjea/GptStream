# db/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, LargeBinary, Table, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()
metadata = MetaData()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    system_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("pdf_name", String),
    Column("page_number", Integer),
    Column("content", Text),
    Column("title", String, nullable=True),
    Column("summary", Text, nullable=True),
    Column("response_style", Text, nullable=True),
    Column("created_at", DateTime, server_default=func.now())
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("document_id", Integer, ForeignKey("documents.id")),
    Column("embedding", LargeBinary),
)