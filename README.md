# 챗봇 인터페이스 프로젝트

이 프로젝트는 Django를 기반으로 한 챗봇 인터페이스로, 게시판(app: board)과 사용자(app: user) 기능, 그리고 PDF 처리 및 LLM 연동 기능을 포함합니다.

## 폴더 구조
(위의 <표 1> 참조)

## 실행 방법
1. Python 가상환경 생성 후 활성화
2. `pip install -r requirements.txt` 실행
3. `python manage.py migrate`
4. `python manage.py runserver`

Codespace 환경에서는 devcontainer.json 파일을 활용하거나, Python 버전을 맞춰 실행하세요.
