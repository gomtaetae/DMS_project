# MySQL 연결 테스트용으로 크게 신경쓰지 않아도된다.
# 나중에 쓸일이 있을까하고 작성해본것

import pymysql

db = pymysql.connect(host="127.0.0.1",user="root",password="1234", db="dms", charset="utf8")

#cursor는 Connection으로부터 Cursor를 생성하고, db의 sql문을 실행하고 조회된 결과를 가져오는 역할을 한다.
cursor = db.cursor()

sql = "select * from users"

# execute : cursor 객체에 sql문을 실행, 지금은 user table을 불러왔다.
# -executemany : cursor 객체에 동일한 sql문에 파라미터를 변경하여 실행한다.
# - executescript : cursor객체에 세미콜론으로 구분된 여러줄의 sql문을 실행한다.
cursor.execute(sql)

# db 데이터 가져오기 cursor의 메소드
# cursor.fetchall() #모든 행 가져오기
# cursor.fetchone() # 하나의 행만 가져오기
# cursor.fetchmany() # n개의 데이터 가져오기 (n)이렇게 넣어주면된다.
rows = cursor.fetchall()
print(rows)

# DB에 생성, 삭제, 수정등 데이터 변동이 생기면 commit()으로 저장을 해줘야한다.
db.commit()

db.close()
