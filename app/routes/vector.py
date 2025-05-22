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

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

dimension = 1536  # OpenAI Embedding 차원
index = faiss.IndexFlatL2(dimension)
doc_store = []  # 텍스트 원문 저장소

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')

# 안전한 파일명 생성 함수
def safe_filename(name):
    name = unicodedata.normalize("NFC", name)  # 한글 정규화
    name = re.sub(r"[^\w.\-]", "_", name)      # 한글, 영문, 숫자, .-_ 외 문자 제거
    return name

upload_dir = Path("temp_uploads")
upload_dir.mkdir(exist_ok=True)

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = safe_filename(file.filename)
        file_path = upload_dir / f"temp_{filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        doc = fitz.open(str(file_path))
        for page in doc:
            text = page.get_text()
            if text.strip():
                embedding = get_embedding(text)
                index.add(np.array([embedding]))
                doc_store.append(text)

        os.remove(file_path)
        print("저장된 페이지 수:", len(doc_store))
        print("faiss index 벡터 수:", index.ntotal)
        return {
            "status": "uploaded and indexed",
            "page_count": len(doc_store),
            "vector_count": index.ntotal
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@router.get("/search")
def search(q: str):
    query_embedding = get_embedding(q)
    D, I = index.search(np.array([query_embedding]), k=1)
    if I[0][0] < len(doc_store):
        return {"result": doc_store[I[0][0]]}
    return {"result": "No result found"}