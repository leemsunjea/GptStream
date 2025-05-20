FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (asyncpg 사용 위해 libpq-dev 필요)
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 종속성 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ 'app' 폴더 내부 파일을 현재 경로에 복사 (중첩 방지)
COPY ./app .   

# PYTHONPATH 설정 → 'from db.database' 등 가능
ENV PYTHONPATH=/app

# 실행 디렉토리 그대로 유지
WORKDIR /app

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]