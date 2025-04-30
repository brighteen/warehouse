# dropout_analysis.py
def run_analysis(year, input_path, output_dir="results"):
    import papermill as pm
    import os

    os.makedirs(output_dir, exist_ok=True)

    input_file = input_path
    output_ipynb = f"{output_dir}/{year}_output.ipynb"
    output_html = f"{output_dir}/{year}_output.html"

    pm.execute_notebook(
        "dropout_template.ipynb",
        output_ipynb,
        parameters={"year": year, "input_file": input_file}
    )

    os.system(f"jupyter nbconvert --to html --no-input {output_ipynb}")

    return output_html
