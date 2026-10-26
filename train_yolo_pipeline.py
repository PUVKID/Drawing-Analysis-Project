import os
import cv2
import random
from ultralytics import YOLO

# 1. 경로 설정
IMG_SOURCE = "C:\\Users\\PUVKID\\Desktop\\심화캡스톤\\A_2D_digital_compilation_of_twelve_child-like_cray.png.png"
IMG_DIR = "generated/images/train"
LABEL_DIR = "generated/labels/train"
MODEL_PATH = "models/yolov8n.pt"
YAML_PATH = "yolo_dataset.yaml"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# 2. 이미지 12개로 분할 저장
def split_image_to_patches(img_path, out_folder):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    cols, rows = 3, 4
    crop_w = w // cols
    crop_h = h // rows
    count = 0
    for y in range(rows):
        for x in range(cols):
            crop = img[y * crop_h:(y + 1) * crop_h, x * crop_w:(x + 1) * crop_w]
            filename = os.path.join(out_folder, f"img_{count:03d}.jpg")
            cv2.imwrite(filename, crop)
            count += 1
    print(f"[✓] {count}장 이미지 저장 완료.")

# 3. 자동 라벨링 (예: 정중앙 문 + 지붕 + 나무 라벨 넣기)
def generate_dummy_labels(image_folder, label_folder):
    class_map = {0: "door", 1: "roof", 2: "tree"}
    for img_file in os.listdir(image_folder):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(image_folder, img_file)
            h, w = cv2.imread(img_path).shape[:2]
            label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:
                # 예시 바운딩 박스들 (문, 지붕, 나무)
                f.write(f"0 0.5 0.7 0.2 0.3\n")  # door 중앙
                f.write(f"1 0.5 0.3 0.5 0.2\n")  # roof 위쪽
                f.write(f"2 0.8 0.65 0.15 0.35\n")  # tree 오른쪽
    print(f"[✓] 라벨링 {len(os.listdir(label_folder))}장 완료.")

# 4. data.yaml 파일 생성
def create_yaml():
    with open(YAML_PATH, "w") as f:
        f.write(f"path: generated\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/train\n")  # 동일하게 사용
        f.write(f"nc: 3\n")
        f.write(f"names: ['door', 'roof', 'tree']\n")
    print("[✓] yolo_dataset.yaml 생성 완료.")

# 5. YOLO 학습 시작
def train_yolo():
    model = YOLO(MODEL_PATH)
    model.train(
        data=YAML_PATH,
        epochs=30,
        imgsz=640,
        batch=8,
        project='drawing_yolo_run',
        name='v8_autotrain',
        exist_ok=True
    )

# 실행
if __name__ == "__main__":
    split_image_to_patches(IMG_SOURCE, IMG_DIR)
    generate_dummy_labels(IMG_DIR, LABEL_DIR)
    create_yaml()
    train_yolo()
