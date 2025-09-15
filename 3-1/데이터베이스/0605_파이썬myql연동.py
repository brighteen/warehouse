import pymysql

# conn 변수를 데이터베이스 연결 객체로 사용
conn = pymysql.connect(
    host='127.0.0.1', # 데이터베이스 호스트 - 본인 컴퓨터이므로, 127.0.0.1 사용
    port=3306, # MySQL 포트 번호
    user='root', # MySQL 사용자 이름
    password='1234', # 사용자 비밀번호(ppt 17쪽 참조)
    db='market_db', # 접속할 데이터베이스 이름
    charset='utf8mb4') # 문자 인코딩 설정

# db에서 sql문을 실행하고, 실행한 결과를 받는 것 / cursor 라는 이름으로 정의
cursor = conn.cursor()

# data = [
# ('kim', '김도영', 'kim@naver.com', '2003'),
# ('son', '손흥민', 'son@gmail.com', '1992'),
# ('ryu', '류현진', 'ryu@naver.com', '1987')
# ]

# id = input("아이디를 입력하세요: ")
# userName = input("이름을 입력하세요: ")
# email = input("이메일을 입력하세요: ")
# birthYear = input("출생연도를 입력하세요(ex: 2003): ")

# sql = "select * from member"
# sql = "CREATE TABLE userTable(id char(4), userName char(15), email char(20), birthYear int)"
# sql = "INSERT INTO userTable VALUES ('kim', '김도영', 'kim@naver.com', '2003')"
sql = "select * from userTable"
# sql = "INSERT INTO userTable VALUES (%s, %s, %s, %s)"
# cursor.executemany(sql, data)

# cursor.execute(sql, (id, userName, email, birthYear)) # SQL 쿼리 실행
cursor.execute(sql) # SQL 쿼리 실행

# conn.commit() # 변경 사항을 데이터베이스에 반영

results = cursor.fetchall() # 실행 결과를 모든 행을 가져옴
for row in results:
    print(row) # 각 행을 출력

cursor.close() # cursor 닫기
conn.close() # 데이터베이스 연결 닫기