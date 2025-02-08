# <코드 6> Dockerfile
FROM python:3.9-slim

WORKDIR /app

# requirements.txt 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . /app

EXPOSE 5000
CMD ["python", "main.py"]
