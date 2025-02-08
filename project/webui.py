# <코드 5> webui.py
import gradio as gr
import requests

def custom_chatbot_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG 기반 챗봇 데모")

        user_query = gr.Textbox(label="질문을 입력하세요")
        output_box = gr.Textbox(label="응답", interactive=False)

        def send_query_to_api(query):
            url = "http://localhost:5000/api/query"
            payload = {"query": query}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    return response.json().get("answer", "")
                else:
                    return f"오류: {response.text}"
            except Exception as e:
                return f"에러: {str(e)}"

        submit_btn = gr.Button("질문 보내기")
        submit_btn.click(fn=send_query_to_api, inputs=user_query, outputs=output_box)

    return demo

if __name__ == "__main__":
    demo = custom_chatbot_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
