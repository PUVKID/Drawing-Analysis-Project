import pandas as pd
import json
from pathlib import Path

# ① CSV 경로
BASE = Path(r"C:\Users\PUVKID\Desktop\심화캡스톤")
csv_file = BASE / "finetune_samples.csv"
jsonl_file = BASE / "train_prepared.jsonl"

# ② CSV 로드
df = pd.read_csv(csv_file, encoding="utf-8-sig")

# ③ JSONL 생성
with open(jsonl_file, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        prompt     = str(row["prompt"])
        completion = str(row["completion"])
        # (권장) 구분자 추가
        if not prompt.endswith("\n\n###"):
            prompt += "\n\n###"
        if not completion.endswith(" ###"):
            completion = completion.strip() + " ###"
        record = {"prompt": prompt, "completion": completion}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ JSONL 생성 완료:", jsonl_file)
