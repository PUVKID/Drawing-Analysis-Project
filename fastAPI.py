from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/gpt")
async def chat_with_gpt(data: ChatRequest):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": data.message}]
    )
    return {"response": response["choices"][0]["message"]["content"]}

# 실행 코드
# cd "C:\Users\PUVKID\Desktop\심화캡스톤"
# uvicorn fastAPI:app --reload --host 127.0.0.1 --port 8000"""

# curl로 테스트
# curl -X POST http://127.0.0.1:8000/gpt -H "Content-Type: application/json" -d "{\"message\":\"안녕 GPT\"}"