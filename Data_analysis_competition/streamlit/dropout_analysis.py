# dropout_analysis.py
import os
import papermill as pm

def run_analysis(year, input_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    template_path = os.path.join(os.path.dirname(__file__), "dropout_template.ipynb")
    output_ipynb = os.path.join(output_dir, f"{year}_output.ipynb")
    output_html = os.path.join(output_dir, f"{year}_output.html")

    pm.execute_notebook(
        template_path,
        output_ipynb,
        parameters={"year": year, "input_file": input_path}
    )

    os.system(f"jupyter nbconvert --to html --no-input {output_ipynb}")
    return output_html
