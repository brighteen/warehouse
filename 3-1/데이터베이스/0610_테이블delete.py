import pymysql

# 데이터베이스 연결 (conn 생성)
conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='1234',
    db='market_db',
    charset='utf8mb4'
)

cursor = conn.cursor()

# 삭제할 회원의 ID를 입력받기
id = input("삭제할 회원의 ID를 입력: ")

# SQL DELETE 쿼리 작성 및 실행
sql = "DELETE FROM userTable WHERE id = %s"
cursor.execute(sql, (id,))

# 변경사항 반영
conn.commit()

# 커서와 연결 종료
cursor.close()
conn.close()