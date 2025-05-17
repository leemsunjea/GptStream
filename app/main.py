from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data: Dict = await request.json()
    message = data.get("message", "")

    def event_stream():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")