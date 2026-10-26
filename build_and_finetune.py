# build_and_finetune.py

import pandas as pd, json, subprocess, os, sys, shutil, pathlib

CSV       = r"C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\finetune_samples.csv"
JSONL     = "train_prepared.jsonl"
MODEL_BASE= "gpt-3.5-turbo"
N_EPOCH   = 1000
SUFFIX    = "drawing_counsel_v1"

# 1) CSV → JSONL
df = pd.read_csv(CSV, encoding="utf-8-sig")
with open(JSONL, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        # prompt/completion 컬럼명으로 접근
        prompt    = row["prompt"].rstrip()       # ex: "door_large ->"
        completion= row["completion"].rstrip()   # ex: " 문이 ... ###"
        msg = [
            {"role":"system",    "content":"당신은 미술치료 상담사입니다."},
            {"role":"user",      "content": prompt},
            {"role":"assistant", "content": completion}
        ]
        f.write(json.dumps({"messages": msg}, ensure_ascii=False) + "\n")
print("✅ JSONL 저장 →", JSONL)

# 2) openai CLI 경로 확인
cli = shutil.which("openai") or shutil.which("openai.exe")
if not cli:
    sys.exit("❌ openai CLI가 설치되지 않았습니다. `pip install --upgrade openai` 후 다시 실행")

# 3) 데이터 검증 & 토크나이즈 (여기서 -y 제거!)
print("🔧 데이터 검증 & 토크나이즈 중…")
try:
    res = subprocess.run(
        [cli, "tools", "fine_tunes.prepare_data", "-f", JSONL],
        check=True, capture_output=True, text=True
    )
    print(res.stdout)
except subprocess.CalledProcessError as e:
    print("❌ prepare_data 오류 발생:")
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)
    sys.exit(1)

# 4) 파인튜닝 생성
prepared = pathlib.Path(JSONL).with_name("train_raw_prepared.jsonl")
cmd = [
    cli, "fine_tunes.create",
    "-m", MODEL_BASE,
    "-t", str(prepared),
    "-n", SUFFIX,
    "--n_epochs", str(N_EPOCH)
]
print("🚀 Fine‑tune 시작…")
subprocess.run(cmd, check=True)
print("✅ Fine‑tune 제출 완료!")
