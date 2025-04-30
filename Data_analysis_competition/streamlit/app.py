import streamlit as st
from dropout_analysis import run_analysis

st.title("🎓 중도탈락 분석 리포트 생성기")

year = st.selectbox("분석할 연도를 선택하세요:", [2021, 2022, 2023])
uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

if st.button("📊 리포트 생성하기") and uploaded_file:
    # 파일 저장
    input_path = f"data/{year}_uploaded.xlsx"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    html_path = run_analysis(year, input_path)
    st.success("✅ 분석 완료!")
    st.markdown(f"📄 [리포트 열기]({html_path})", unsafe_allow_html=True)
