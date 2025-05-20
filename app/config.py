# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

settings = Settings()