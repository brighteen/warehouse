from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: PromptRequest):
    responses = [
        f"AI 응답: {request.prompt} 🤖",
        f"생각 중... {request.prompt} ⏳",
        f"이런 대답은 어때요? {request.prompt} 🧐"
    ]
    return {"response": random.choice(responses)}

# 백엔드 실행 명령어: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
