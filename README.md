# 챗봇 인터페이스 프로젝트

이 프로젝트는 Django를 기반으로 한 챗봇 인터페이스로, 게시판(app: board)과 사용자(app: user) 기능, 그리고 PDF 처리 및 LLM 연동 기능을 포함합니다.

## 폴더 구조
(위의 <표 1> 참조)

## 어드민 생성하는 법
```bash
python manage.py createsuperuser
```
## 실행 방법
1. Python 가상환경 생성 후 활성화
2. `pip install -r requirements.txt` 실행
3. `python manage.py migrate`
4. `python manage.py runserver`

Codespace 환경에서는 devcontainer.json 파일을 활용하거나, Python 버전을 맞춰 실행하세요.


## chromadb와 sqlite 버전 오류 해결 방법
1. manage.py파일에 아래 코드 추가
```python
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3
print("현재 sqlite3 버전:", sqlite3.sqlite_version)  # 버전 확인용
```
2. 위 방법으로 해결 안될 시 삭제 후 재설치
```bash
pip uninstall pysqlite3
```
pysqlite3 제거를 위해 **y**입력 

```bash
pip install pysqlite3-binary
```