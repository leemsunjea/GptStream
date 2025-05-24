# db/sync_database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# Use a synchronous database URL for Alembic
SYNC_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db").replace(
    "+aiosqlite", ""
)  # Remove async part for sync operations

# Create synchronous engine
engine = create_engine(SYNC_DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for models
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
