# list_finetune_jobs.py

import os
from openai import OpenAI

# ① API 키 읽어오기 (환경변수 또는 직접 하드코딩)
api_key = os.getenv("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"]

# ② 클라이언트 초기화
client = OpenAI(api_key=api_key)

# ③ Fine‑tune Job 리스트 가져오기
jobs = client.fine_tuning.jobs.list()

# ④ 결과 출력
print("🛠️ Fine-tune jobs:")
for job in jobs.data:
    print(f"- ID: {job.id}, Model: {job.model}, Status: {job.status}")
