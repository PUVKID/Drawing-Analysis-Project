import pandas as pd
from pathlib import Path

# ① CSV 저장 위치
BASE_DIR = Path(r"C:\\Users\\PUVKID\\Desktop\\심화캡스톤")
OUT_CSV  = BASE_DIR / "finetune_samples.csv"

# ② 실제 학습 샘플 (prompt엔 반드시 구분자 '->'를 포함)
samples = [
    {"prompt":"door_large ->",    "completion":"문이 지나치게 크게 그려져 사회적 인정이나 수용에 지나치게 의존하고 타인 관계에 예민합니다. ###"},
    {"prompt":"door_open ->",     "completion":"열린 문은 외부로부터 정서적 따뜻함을 받고자 하는 강렬한 욕망을 시사합니다. ###"},
    {"prompt":"door_small ->",    "completion":"작은 문은 관계를 원하지만 두려움이 있어 수동적·내성적 태도를 보입니다. ###"},
    {"prompt":"door_none ->",     "completion":"문을 생략하면 자신만의 세계에 고립되고 타인의 접근을 저항하는 경향이 있습니다. ###"},
    {"prompt":"window_many ->",   "completion":"창문이 많으면 타인과 관계 맺고자 하는 욕구가 크며 동시에 노출 욕구를 암시합니다. ###"},
    {"prompt":"window_few ->",    "completion":"창문 수가 적으면 심리적 퇴행이나 대인관계 위축을 시사합니다. ###"},
    {"prompt":"roof_large ->",    "completion":"지붕이 크면 현실 좌절을 공상으로 보상하려는 경향이 강합니다. ###"},
    {"prompt":"chimney_smoke ->", "completion":"연기가 나는 굴뚝은 가정 내 정서적 긴장이나 불화를 암시합니다. ###"},
    {"prompt":"wall_broken ->",   "completion":"허물어지거나 견고하지 못한 벽은 자아 강도가 취약하고 불안을 느낄 수 있음을 보여줍니다. ###"},
    {"prompt":"tree_many ->",     "completion":"집 주변에 큰 나무가 많으면 불안정감을 방어하기 위해 자신 주변에 방어막을 세우고 싶어 합니다. ###"},
]

# ③ CSV로 저장 (UTF-8 BOM 포함)
df = pd.DataFrame(samples)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("✅ finetune_samples.csv 생성 완료 →", OUT_CSV)
