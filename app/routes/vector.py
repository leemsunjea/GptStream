from fastapi import APIRouter, UploadFile, File
import os
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dimension = 1536  # OpenAI Embedding 차원
index = faiss.IndexFlatL2(dimension)
doc_store = []  # 텍스트 원문 저장소

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)

    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text()
        if text.strip():
            embedding = get_embedding(text)
            index.add(np.array([embedding]))
            doc_store.append(text)

    os.remove(file_path)
    return {"status": "uploaded and indexed"}

@router.get("/search")
def search(q: str):
    query_embedding = get_embedding(q)
    D, I = index.search(np.array([query_embedding]), k=1)
    if I[0][0] < len(doc_store):
        return {"result": doc_store[I[0][0]]}
    return {"result": "No result found"}