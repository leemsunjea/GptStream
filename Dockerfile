FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (asyncpg 사용 위해 libpq-dev 필요)
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 종속성 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사 (db, app, static 등 포함됨)
COPY . .

# PYTHONPATH 설정 → 'from db.database' 등 가능
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 8000

# 실행 명령 (타임아웃 3분 = 180초로 설정)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "180"]