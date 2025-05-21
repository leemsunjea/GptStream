# models.py
from sqlalchemy import Column, Integer, Text, DateTime, func
from db.database import Base  # 수정: db에서가 아니라 db.database에서 import

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())