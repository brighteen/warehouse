from flask import Flask, render_template, request, redirect, session, flash
import pandas as pd
import numpy as np  # NaN 값 처리를 위한 라이브러리

app = Flask(__name__)
app.secret_key = "secret_key"  # 세션 관리를 위한 키

# CSV 파일 경로
CSV_FILE = r"test_1\user_credentials.csv"

# 🔹 CSV 파일 불러오기
def load_csv():
    try:
        df = pd.read_csv(CSV_FILE, encoding="utf-8")
        df["data"] = df["data"].fillna("")  # NaN 값을 빈 문자열로 대체
        print("CSV Columns:", df.columns)  # 디버깅용
        return df
    except Exception as e:
        print("CSV 읽기 오류:", e)
        return None

# 🔹 로그인 검증 함수
def authenticate_user(username, password):
    df = load_csv()
    if df is None:
        return None

    user_data = df[(df["username"] == username) & (df["password"] == password)]
    if not user_data.empty:
        return user_data.iloc[0]  # 로그인 성공 시 사용자 정보 반환
    return None  # 로그인 실패

# 🔹 로그인 페이지
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user_info = authenticate_user(username, password)

        if user_info is not None:
            session["username"] = username
            session["role"] = "admin" if username.startswith("admin") else "user"
            return redirect("/dashboard")
        else:
            flash("❌ 아이디 또는 패스워드를 찾을 수 없습니다!", "error")

    return render_template("login.html")

# 🔹 대시보드 페이지 (Admin & User 분리)
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect("/")
    
    df = load_csv()
    if df is None:
        return "CSV 파일을 불러올 수 없습니다."

    username = session["username"]
    user_query = ""
    user_response = ""

    # 사용자가 쿼리를 입력했을 때
    if request.method == "POST":
        user_query = request.form["query_input"]

        # Admin이면 해당 사용자의 데이터도 함께 출력
        if session["role"] == "admin":
            user_data = df.loc[df["username"] == username, "data"].values
            data_text = f" | Data: {user_data[0]}" if len(user_data) > 0 else " | Data: 없음"
            user_response = f"쿼리: {user_query}{data_text}"
        else:
            user_response = f"쿼리: {user_query}"

    # 모든 사용자 데이터를 가져옴
    all_users = df["username"].tolist()
    all_data = df[["username", "data"]].to_dict(orient="records")

    if session["role"] == "admin":
        return render_template("admin.html", username=username, all_users=all_users, all_data=all_data, query=user_query, response=user_response)

    else:
        return render_template("user.html", username=username, all_data=all_data, query=user_query, response=user_response)

# 🔹 Admin 기능 (모든 사용자의 데이터 추가, 수정, 삭제 가능)
@app.route("/admin_edit", methods=["POST"])
def admin_edit():
    if "username" not in session or session["role"] != "admin":
        return redirect("/")

    df = load_csv()
    if df is None:
        return "CSV 파일을 불러올 수 없습니다."

    target_user = request.form.get("target_user", "")
    
    # 데이터 추가
    if "add_data" in request.form:
        new_data = request.form["add_data"]
        if target_user in df["username"].values:
            current_data = df.loc[df["username"] == target_user, "data"].values[0]
            updated_data = f"{current_data}, {new_data}" if current_data else new_data
            df.loc[df["username"] == target_user, "data"] = updated_data

    # 데이터 수정
    elif "edit_old_data" in request.form and "edit_new_data" in request.form:
        old_data = request.form["edit_old_data"]
        new_data = request.form["edit_new_data"]
        df.loc[(df["username"] == target_user) & (df["data"].str.contains(old_data, na=False)), "data"] = df["data"].str.replace(old_data, new_data, regex=False)

    # 데이터 삭제
    elif "delete_data" in request.form:
        delete_data = request.form["delete_data"]
        df.loc[df["username"] == target_user, "data"] = df["data"].str.replace(f"{delete_data}, ", "", regex=False).str.replace(f", {delete_data}", "", regex=False).str.replace(delete_data, "", regex=False)

    df.to_csv(CSV_FILE, index=False)
    return redirect("/dashboard")

# 🔹 User 기능 (데이터 추가 가능)
@app.route("/user_add", methods=["POST"])
def user_add():
    if "username" not in session or session["role"] != "user":
        return redirect("/")

    df = load_csv()
    if df is None:
        return "CSV 파일을 불러올 수 없습니다."

    username = session["username"]
    new_data = request.form["add_data"]

    current_data = df.loc[df["username"] == username, "data"].values[0]
    updated_data = f"{current_data}, {new_data}" if current_data else new_data
    df.loc[df["username"] == username, "data"] = updated_data

    df.to_csv(CSV_FILE, index=False)
    return redirect("/dashboard")

# 🔹 로그아웃 기능
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
