import os
from dotenv import load_dotenv  # python-dotenv 임포트

# .env 파일 로드 (백업용)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # 환경 변수에서 API 키 로드
# API 키 직접 설정 (test2.py와 동일하게)

# 디버깅을 위한 API 키 확인 (앞부분과 뒷부분만 출력)
print(f"API 키 설정됨: {GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}")