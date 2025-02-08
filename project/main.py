# <코드 1> main.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# RAG 시스템 초기화
rag_system = RAGPipeline(db_path="chroma_db", openai_api_key=openai_api_key)

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    answer = rag_system.generate_answer(user_query)
    return jsonify({"answer": answer})

# <코드 1> main.py (예시)

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    return jsonify({"answer": f"테스트 답변: {user_query}"})

@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, Flask is working!"})

if __name__ == "__main__":
    print("Flask is starting on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)

