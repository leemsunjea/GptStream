# app/routes/chat.py

import os
import asyncio
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse
import openai
import faiss
from app.config import settings
from db.models import ChatHistory
from db.database import get_async_session, get_user_session, get_user_vector_dir, get_or_create_session_id
from openai import OpenAI

# 전역 변수 대신 세션별 인덱스와 문서 저장소를 관리할 딕셔너리
session_stores: Dict[str, dict] = {}

def get_or_create_session_store(session_id: str) -> dict:
    """세션 ID에 해당하는 저장소를 가져오거나 생성"""
    if session_id not in session_stores:
        # 새 세션 저장소 초기화
        session_stores[session_id] = {
            "index": None,  # FAISS 인덱스
            "doc_store": []  # 문서 저장소
        }
        
        # 저장된 인덱스 로드 시도
        vector_dir = get_user_vector_dir(session_id)
        index_path = os.path.join(vector_dir, "faiss.index")
        doc_store_path = os.path.join(vector_dir, "doc_store.pkl")
        
        try:
            if os.path.exists(index_path) and os.path.exists(doc_store_path):
                session_stores[session_id]["index"] = faiss.read_index(index_path)
                with open(doc_store_path, 'rb') as f:
                    session_stores[session_id]["doc_store"] = pickle.load(f)
        except Exception as e:
            print(f"[WARNING] 세션 저장소 로드 실패: {e}")
    
    return session_stores[session_id]

def save_session_store(session_id: str):
    """세션 저장소를 파일로 저장"""
    if session_id in session_stores:
        vector_dir = get_user_vector_dir(session_id)
        os.makedirs(vector_dir, exist_ok=True)
        
        index_path = os.path.join(vector_dir, "faiss.index")
        doc_store_path = os.path.join(vector_dir, "doc_store.pkl")
        
        try:
            if session_stores[session_id]["index"] is not None:
                faiss.write_index(session_stores[session_id]["index"], index_path)
            with open(doc_store_path, 'wb') as f:
                pickle.dump(session_stores[session_id]["doc_store"], f)
        except Exception as e:
            print(f"[ERROR] 세션 저장소 저장 실패: {e}")

# 임베딩 함수는 vector_db.py에서 가져옴
async def get_embedding_async(text: str) -> Optional[List[float]]:
    """텍스트를 임베딩으로 변환"""
    try:
        response = await asyncio.to_thread(
            openai.Embedding.create,
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 실패: {e}")
        return None

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")
    
    # 세션 ID 가져오기
    session_id = get_or_create_session_id(request)
    session_store = get_or_create_session_store(session_id)
    
    # 세션별 인덱스와 문서 저장소
    index = session_store["index"]
    doc_store = session_store["doc_store"]
    
    # 🔍 질문을 벡터화하고 관련 문단 검색
    context_text = ""
    referenced_docs = []
    
    if index is not None and index.ntotal > 0:
        query_embedding = await get_embedding_async(message)
        if query_embedding is None:
            print("[ERROR] 쿼리 임베딩 생성 실패")
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        print(f"[DEBUG] FAISS 인덱스 벡터 개수: {index.ntotal}")
        try:
            D, I = index.search(np.array([query_embedding], dtype='float32'), k=3)  # 상위 3개 문서 검색
            print(f"[DEBUG] 검색 결과 인덱스: {I}, 거리: {D}")
            
            if I is not None and len(I[0]) > 0:
                referenced_docs = [doc_store[i] for i in I[0] if 0 <= i < len(doc_store)]
                print(f"[DEBUG] 검색된 문서 개수: {len(referenced_docs)}")
                if len(referenced_docs) > 0:
                    print(f"[DEBUG] 첫 번째 검색된 문서 내용: {referenced_docs[0]}")
                else:
                    print("[DEBUG] 검색된 문서가 없습니다.")
            else:
                print("[DEBUG] 검색 결과가 없습니다.")
                
        except Exception as e:
            print(f"[ERROR] FAISS 검색 중 오류 발생: {e}")
            return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        print("[DEBUG] 인덱스에 문서가 없습니다.")

    # 문서 검색 후 context_text 생성
    context_text = "\n\n".join(referenced_docs) if referenced_docs else ""

    # 개선된 시스템 프롬프트
    # 아래 코드로 대체:
    global new_system_prompt
    if 'new_system_prompt' not in globals():
        new_system_prompt = ""  # 사용자로부터 받은 새로운 시스템 프롬프트

    # 이전 대화 기록을 가져오기 위한 전역 변수
    global chat_history
    if 'chat_history' not in globals():
        chat_history = []

    # 이전 대화 기록 추가
    chat_history.append({"role": "user", "content": message})

    if context_text or new_system_prompt:
        system_prompt = (
            "다음은 사용자가 업로드한 문서에서 검색된 내용입니다. 이 내용을 기반으로 사용자의 질문에 답변해주세요. "
            "만약 내용이 질문에 답변하기에 충분하지 않다면, 그 사실을 명시하세요. "
            "또한 답변에 사용된 문서의 특정 부분을 반드시 언급하세요.\n\n"
            "문서 내용:\n" + context_text + "\n\n"
            "" + new_system_prompt + ""
            "답변에서 줄바꿈은 '\n'으로 표시하세요."
        )
    else:
        system_prompt = (
            "업로드된 문서가 없으니 일반 챗봇처럼 답변해주세요. "
            "답변에서 줄바꿈은 '\n'으로 표시하세요."
            "문서 내용:\n" + context_text + "\n\n"
            "추가된 시스템 프롬프트:\n" + new_system_prompt + "\n\n"
        )

    # 이전 대화 기록을 시스템 메시지에 추가
    messages = [{"role": "system", "content": system_prompt}] + chat_history

    async def event_stream():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)

            # 봇 응답을 대화 기록에 추가
            chat_history.append({"role": "assistant", "content": full_response})

            # 참조 문서 출력
            if referenced_docs:
                yield f"\n\ndata: [참고한 문단]\n\n"
                for idx, doc in enumerate(referenced_docs, 1):
                    yield f"data: [문단 {idx}]\n{doc}\n\n"

            yield "data: \n\n[DONE]\n\n"

            # 채팅 기록 저장 (비동기)
            async def save_chat_history():
                try:
                    async with get_async_session(session_id) as session:
                        chat = ChatHistory(
                            session_id=session_id,  # 세션 ID 사용
                            message=message,
                            response=full_response
                        )
                        session.add(chat)
                        await session.commit()
                        
                        # 세션 저장소 저장
                        save_session_store(session_id)
                except Exception as e:
                    print(f"[ERROR] 채팅 기록 저장 실패: {e}")
            
            # 비동기로 채팅 기록 저장
            asyncio.create_task(save_chat_history())

            # DB 저장
            # async with async_session() as session:
            #     chat = ChatHistory(
            #         user_message=message,
            #         bot_response=full_response
            #     )
            #     session.add(chat)
            #     await session.commit()
        except Exception as e:
            import traceback
            print("오류 발생:", e)
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/chat/add_prompt")
async def add_prompt(request: Request):
    data = await request.json()
    new_prompt = data.get("prompt", "")

    if not new_prompt:
        return {"error": "프롬프트가 비어 있습니다."}

    # 새로운 시스템 프롬프트를 전역 변수로 저장
    global new_system_prompt
    previous_prompt = new_system_prompt if 'new_system_prompt' in globals() else "(없음)"
    new_system_prompt = new_prompt

    # 로그 출력
    print("[DEBUG] 이전 프롬프트:", previous_prompt)
    print("[DEBUG] 새로운 프롬프트:", new_system_prompt)

    return {"success": True, "chatHistory": new_system_prompt}