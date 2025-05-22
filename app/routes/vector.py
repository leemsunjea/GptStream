import unicodedata
import re
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
import os
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np
from app.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.database import async_session, get_async_engine

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

dimension = 1536  # OpenAI Embedding 차원
index = None
doc_store = []

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')

def safe_filename(name):
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name

upload_dir = Path("/tmp/temp_uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = safe_filename(file.filename)
        file_path = upload_dir / f"temp_{filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        doc = fitz.open(str(file_path))
        page_count = 0
        async with async_session() as session:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    # 1. 문서 저장
                    result = await session.execute(
                        insert(documents).values(
                            pdf_name=filename,
                            page_number=i,
                            content=text
                        ).returning(documents.c.id)
                    )
                    doc_id = result.scalar()
                    # 2. 임베딩 저장
                    embedding = get_embedding(text)
                    await session.execute(
                        insert(embeddings).values(
                            document_id=doc_id,
                            embedding=embedding.tobytes()
                        )
                    )
                    page_count += 1
            await session.commit()

        os.remove(file_path)
        return {"status": "uploaded and indexed", "page_count": page_count}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}