import pymysql

# 데이터베이스 연결
conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='1234',
    db='market_db',
    charset='utf8mb4'
)
cursor = conn.cursor()

# # 회원 id 입력받기
# user_id = input("수정할 회원의 id를 입력하세요: ")

# # 새 회원 정보 입력받기
# new_name = input("새로운 이름을 입력하세요: ")
# new_email = input("새로운 이메일을 입력하세요: ")
# new_birthyear = input("새로운 출생연도를 입력하세요 (예: 2003): ")

# # UPDATE 쿼리 실행
# sql = "UPDATE userTable SET userName=%s, email=%s, birthYear=%s WHERE id=%s"
# cursor.execute(sql, (new_name, new_email, new_birthyear, user_id))

# # 변경 내용 저장
# conn.commit()

# 수정 결과 확인
print("\n[수정 후 userTable 전체 조회]")
cursor.execute("SELECT * FROM userTable")
results = cursor.fetchall()
for row in results:
    print(row)

# 자원 해제
cursor.close()
conn.close()