from openai import OpenAI
import os

# ① 클라이언트 생성
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # 본인의 API 키

# ② JSONL 파일 업로드
upload_resp = client.files.create(
    file=open(r"C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\train_prepared.jsonl", "rb"),
    purpose="fine-tune"
)
training_file_id = upload_resp.id
print("✅ Training file uploaded →", training_file_id)

# ③ 파인튜닝 작업 생성
ft_response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",   # ✅ Chat 모델 이름
    method={                      # ✅ 하이퍼파라미터 설정
        "type": "supervised",
        "supervised": {
            "hyperparameters": {
                "n_epochs": "auto"
            }
        }
    },
    suffix="drawing_counsel_v1"    # fine‑tuned 모델 꼬리표
)
print("🚀 Fine‑tune job created →", ft_response.id)
