# app/vector_db.py
import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

dimension = 1536
index = faiss.IndexFlatL2(dimension)
doc_store = []

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')