import logging
import sys
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from database import Base

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(10))
    message = Column(Text)
    faiss_search_results = Column(Text, nullable=True)

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def log_to_db(db, level: str, message: str, faiss_results: str = None):
    """데이터베이스에 로그를 저장합니다."""
    log_entry = Log(
        level=level,
        message=message,
        faiss_search_results=faiss_results
    )
    db.add(log_entry)
    db.commit()
    return log_entry 