# train_yolo_dataset.py
"""
1. yolo_dataset/ 구조를 점검해 없으면 val 폴더를 train 복사본으로 생성
2. yolo_dataset.yaml 을 절대경로로 자동 작성/수정
3. Ultralytics YOLOv8 모델 재학습
"""

import os, shutil, time
from pathlib import Path
from ultralytics import YOLO

# ───────────────────────── 설정 ─────────────────────────
DATA_DIR  = Path(r"C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\yolo_single_object_dataset")   # ★ 데이터 폴더 절대경로
WEIGHTS   = "yolov8n.pt"        # 또는 기존 best.pt
RUN_NAME  = "v8_finetune"
IMG_SIZE  = 512
EPOCH_CHUNK = 50       # 단계별 에폭
N_CHUNKS    = 30      # 단계 수 (총 EPOCH_CHUNK × N_CHUNKS 에폭)
BATCH     = 16
WORKERS      = 8          # DataLoader 워커 수 (CPU 코어 활용)
CACHE        = False       # 전체 데이터셋을 RAM에 캐싱
NAMES     = ["house","window","door","tree","person",
    "cloud","sun","wall","chimney","smoke","roof","fence",
    "shadow","ground_line","grass","tulip","daisy","blueprint",
    "face","eye","nose","mouth","shoulder","leg","foot","button",
    "clothes","underwear","branch","leaf","crown","root","apple"]   # 클래스 이름 리스트
# ────────────────────────────────────────────────────────


def ensure_val_split():
    """images/val, labels/val 폴더가 없으면 train을 그대로 복사해 만듦"""
    img_train = DATA_DIR / "images" / "train"
    lbl_train = DATA_DIR / "labels" / "train"
    img_val   = DATA_DIR / "images" / "val"
    lbl_val   = DATA_DIR / "labels" / "val"

    if not img_val.exists():
        print("⚠️  images/val 이 없어 train 세트를 복사해 만듭니다.")
        shutil.copytree(img_train, img_val)

    if not lbl_val.exists():
        print("⚠️  labels/val 이 없어 train 세트를 복사해 만듭니다.")
        shutil.copytree(lbl_train, lbl_val)


def make_yaml():
    yaml_path = DATA_DIR / "yolo_dataset.yaml"
    yaml_text = f"""# AUTO‑GENERATED
path: {DATA_DIR.as_posix()}
train: images/train
val: images/val
nc: {len(NAMES)}
names: {NAMES}
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def train():
    ensure_val_split()
    yaml_file = make_yaml()
    model = YOLO(WEIGHTS)

    # ── 학습 실행 ───────────────────────────
    for chunk_idx in range(N_CHUNKS):
        target_epochs = EPOCH_CHUNK * (chunk_idx + 1)
        resume_flag   = chunk_idx > 0
        print(f"🚀 단계 {chunk_idx+1}/{N_CHUNKS} → 총 {target_epochs} 에폭까지 학습 "
              f"(resume={resume_flag})")
        model.train(
            data=str(yaml_file),
            epochs=target_epochs,
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=WORKERS,
            cache=CACHE,
            patience=0,              # 조기 중단 비활성화
            project="drawing_yolo_run",
            name=RUN_NAME,
            exist_ok=True,
            cos_lr=True,
            
            lr0=0.001,          # learning rate 낮춤
            lrf=0.01,  
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            optimizer='SGD',
            momentum=0.9,
            weight_decay=5e-4, 
        )     

    # 가장 성능이 좋았던 가중치(.pt) 경로 출력
    best_pt = model.trainer.best
    print("✅ 학습 완료 →", best_pt)


if __name__ == "__main__":
    assert DATA_DIR.exists(), f"{DATA_DIR} 가 존재하지 않습니다."
    train()