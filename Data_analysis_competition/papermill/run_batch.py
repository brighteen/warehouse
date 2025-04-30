import os
import papermill as pm

os.makedirs("results", exist_ok=True)

for year in [2021, 2022, 2023]:
    out_path = f"results/{year}_output.ipynb"

    # 1. 분석 실행
    pm.execute_notebook(
        "dropout_template.ipynb",
        out_path,
        parameters={
            "year": year,
            "input_file": f"data/{year}_중도탈락 학생 현황 (대학).xlsx"
        }
    )

    # 2. HTML 리포트 생성
    os.system(f"jupyter nbconvert --to html --no-input {out_path}")
