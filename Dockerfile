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

# 환경변수 파일 복사 (Cloudtype에서 자동 주입 가능 시 생략 가능)
COPY .env .env

# 앱 소스 복사
COPY ./app ./app

# 앱 실행 디렉토리 변경 (main.py가 app/main.py라면)
WORKDIR /app/app

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]