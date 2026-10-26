# synthetic_drawings_generator.py
import random, math, os, cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ─────────────── ① 기본 설정 ───────────────
OUT_DIR        = Path("yolo_dataset")
CLASSES        = [
    "house","window","door","tree","person",
    "cloud","sun","wall","chimney","smoke","roof","fence",
    "shadow","ground_line","grass","tulip","daisy","blueprint",
    "face","eye","nose","mouth","shoulder","leg","foot","button",
    "clothes","underwear","branch","leaf","crown","root","apple"
]
IMG_SIZE       = 640
N_TRAIN, N_VAL = 50000, 50000
random.seed(42)

# 폴더 구조
for split in ["train", "val"]:
    (OUT_DIR/"images"/split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"labels"/split).mkdir(parents=True, exist_ok=True)

# ─────────────── ② 도형 그리기 util ───────────────
def rand_color():
    return tuple(int(c) for c in np.random.randint(30, 256, 3))

# 기존 함수들...
def draw_house(img, cx, cy, scale):
    w = int(120*scale); h = int(90*scale)
    x1, y1 = cx-w//2, cy-h//2
    x2, y2 = cx+w//2, cy+h//2
    cv2.rectangle(img, (x1, y1), (x2, y2), rand_color(), -1)
    pts = np.array([[x1, y1], [x2, y1], [cx, y1-int(40*scale)]])
    cv2.fillPoly(img, [pts], rand_color())
    return (0, cx, cy+h*0.05, w, h*1.1)

def draw_window(img, hx, hy, scale):
    w = int(30*scale)
    x1, y1 = hx - w//2, hy - w//2
    x2, y2 = x1 + w, y1 + w
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), -1)
    return (1, (x1+x2)//2, (y1+y2)//2, w, w)

def draw_door(img, hx, hy, scale):
    w = int(35*scale); h = int(55*scale)
    x1 = hx - w//2; y1 = hy + int(20*scale)
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), -1)
    return (2, (x1+x2)//2, (y1+y2)//2, w, h)

def draw_tree(img, cx, cy, scale):
    trunk_w = int(25*scale); trunk_h = int(100*scale)
    x1, y1 = cx-trunk_w//2, cy
    cv2.rectangle(img, (x1, y1), (x1+trunk_w, y1+trunk_h), (120,80,40), -1)
    rad = int(80*scale)
    cv2.circle(img, (cx, cy), rad, (0,160,0), -1)
    return (3, cx, cy+rad//2, rad*1.8, rad*2.2)

def draw_person(img, cx, cy, scale):
    h = int(90*scale)
    cv2.circle(img, (cx, cy), int(15*scale), (255,255,255), -1)
    cv2.line(img, (cx, cy+15), (cx, cy+50), (0,0,0), 2)
    cv2.line(img, (cx, cy+25), (cx-15, cy+35), (0,0,0), 2)
    cv2.line(img, (cx, cy+25), (cx+15, cy+35), (0,0,0), 2)
    cv2.line(img, (cx, cy+50), (cx-10, cy+80), (0,0,0), 2)
    cv2.line(img, (cx, cy+50), (cx+10, cy+80), (0,0,0), 2)
    return (4, cx, cy+40, 35*scale, h)

# ─────────────── ③ 신규 클래스 그리기 함수들 ───────────────
def draw_cloud(img, cx, cy, scale):
    rs = [int(30*scale), int(40*scale), int(35*scale)]
    for dx, r in zip([-40, 0, 40], rs):
        cv2.circle(img, (cx+dx, cy+int(10*scale)), r, (200,200,200), -1)
    w, h = max(rs)*2, max(rs)*2
    return (5, cx, cy, w, h)

def draw_sun(img, cx, cy, scale):
    r = int(30*scale)
    cv2.circle(img, (cx, cy), r, (0,200,200), -1)
    for angle in range(0,360,45):
        x2 = int(cx + math.cos(math.radians(angle))*(r+10*scale))
        y2 = int(cy + math.sin(math.radians(angle))*(r+10*scale))
        cv2.line(img, (cx,cy),(x2,y2),(0,200,200),2)
    return (6, cx, cy, r*2, r*2)

def draw_wall(img, cx, cy, scale):
    w = int(200*scale); h = int(100*scale)
    x1, y1 = cx-w//2, cy-h//2
    cv2.rectangle(img, (x1,y1),(x1+w,y1+h),(150,150,150),-1)
    return (7, cx, cy, w, h)

def draw_chimney(img, cx, cy, scale):
    w = int(20*scale); h = int(60*scale)
    x1, y1 = cx-w//2, cy-h
    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(100,100,100),-1)
    return (8, cx, cy-h//2, w, h)

def draw_smoke(img, cx, cy, scale):
    for i in range(3):
        r = int(10*scale*(1+i*0.3))
        cv2.circle(img, (cx+random.randint(-5,5), cy-int(i*15*scale)), r, (200,200,200), -1)
    return (9, cx, cy-15*scale, r*2, r*2)

def draw_roof(img, cx, cy, scale):
    w = int(140*scale); h = int(60*scale)
    pts = np.array([[cx-w//2, cy],[cx+w//2, cy],[cx, cy-int(h)]])
    cv2.fillPoly(img, [pts], (rand_color()))
    return (10, cx, cy-int(h//2), w, h)

def draw_fence(img, cx, cy, scale):
    w = int(200*scale); h = int(40*scale)
    x1, y1 = cx-w//2, cy
    for i in range(0, w, int(20*scale)):
        cv2.line(img, (x1+i,y1),(x1+i,y1+int(h)),(100,50,0),2)
    return (11, cx, cy+int(h/2), w, h)

def draw_shadow(img, cx, cy, scale):
    w = int(100*scale); h = int(20*scale)
    cv2.ellipse(img,(cx,cy+10), (w//2,h//2), 0,0,360,(50,50,50),-1)
    return (12, cx, cy+10, w, h)

def draw_ground_line(img, cx, cy, scale):
    y = cy
    cv2.line(img,(0,y),(IMG_SIZE,y),(0,0,0),2)
    return (13, IMG_SIZE//2, y, IMG_SIZE, 2)

def draw_grass(img, cx, cy, scale):
    for i in range(20):
        x = cx + random.randint(-50,50)
        cv2.line(img,(x,cy),(x,cy-int(20*scale)),(0,160,0),2)
    return (14, cx, cy-10, 100, 20)

def draw_tulip(img, cx, cy, scale):
    cv2.line(img,(cx,cy),(cx,cy-int(30*scale)),(0,100,0),2)
    cv2.circle(img,(cx,cy-int(40*scale)),int(10*scale),(200,50,50),-1)
    return (15, cx, cy-10, 20, 40)

def draw_daisy(img, cx, cy, scale):
    for a in range(0,360,45):
        x = int(cx+math.cos(math.radians(a))*int(12*scale))
        y = int(cy+math.sin(math.radians(a))*int(12*scale))
        cv2.circle(img,(x,y),int(5*scale),(255,255,0),-1)
    cv2.circle(img,(cx,cy),int(7*scale),(200,150,0),-1)
    return (16, cx, cy, 25, 25)

def draw_blueprint(img, cx, cy, scale):
    w = int(100*scale); h = int(100*scale)
    x1, y1 = cx-w//2, cy-h//2
    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,0,150),2)
    # 그리드
    for i in range(1,4):
        xi = x1 + i*(w//4)
        yi = y1 + i*(h//4)
        cv2.line(img,(xi,y1),(xi,y1+h),(0,0,150),1)
        cv2.line(img,(x1,yi),(x1+w,yi),(0,0,150),1)
    return (17, cx, cy, w, h)

def draw_face(img, cx, cy, scale):
    r = int(20*scale)
    cv2.circle(img,(cx,cy),r,(255,220,200),-1)
    return (18, cx, cy, r*2, r*2)

def draw_eye(img, cx, cy, scale):
    dx = int(10*scale)
    cv2.circle(img,(cx-dx,cy),int(5*scale),(0,0,0),-1)
    cv2.circle(img,(cx+dx,cy),int(5*scale),(0,0,0),-1)
    return (19, cx, cy, dx*2, int(5*scale))

def draw_nose(img, cx, cy, scale):
    pts = np.array([[cx,cy-int(5*scale)],[cx-int(3*scale),cy+int(5*scale)],[cx+int(3*scale),cy+int(5*scale)]])
    cv2.fillPoly(img,[pts],(200,150,150))
    return (20, cx, cy, int(6*scale), int(10*scale))

def draw_mouth(img, cx, cy, scale):
    cv2.ellipse(img,(cx,cy+int(10*scale)), (int(10*scale),int(5*scale)),0,0,180,(0,0,0),2)
    return (21, cx, cy+int(10*scale), int(20*scale), int(10*scale))

def draw_shoulder(img, cx, cy, scale):
    cv2.line(img,(cx-int(20*scale),cy-int(5*scale)),(cx+int(20*scale),cy-int(5*scale)),(0,0,0),3)
    return (22, cx, cy-int(5*scale), int(40*scale), int(3*scale))

def draw_leg(img, cx, cy, scale):
    cv2.line(img,(cx,cy),(cx,cy+int(30*scale)),(0,0,0),3)
    return (23, cx, cy+int(15*scale), int(3*scale), int(30*scale))

def draw_foot(img, cx, cy, scale):
    cv2.ellipse(img,(cx,cy+int(35*scale)), (int(10*scale),int(5*scale)),0,0,360,(0,0,0),-1)
    return (24, cx, cy+int(35*scale), int(20*scale), int(10*scale))

def draw_button(img, cx, cy, scale):
    for i in range(3):
        cv2.circle(img,(cx, cy+int(i*10*scale)),int(3*scale),(0,0,0),-1)
    return (25, cx, cy+int(10*scale), int(6*scale), int(20*scale))

def draw_clothes(img, cx, cy, scale):
    w,h = int(50*scale), int(60*scale)
    cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),rand_color(),-1)
    return (26, cx, cy, w, h)

def draw_underwear(img, cx, cy, scale):
    w,h = int(40*scale), int(30*scale)
    cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),(255,200,200),-1)
    return (27, cx, cy, w, h)

def draw_branch(img, cx, cy, scale):
    length = int(60*scale)
    x2,y2 = cx+int(length*0.7), cy-int(length*0.7)
    cv2.line(img,(cx,cy),(x2,y2),(100,50,0),3)
    return (28, cx+int(length*0.35), cy-int(length*0.35), length, 3)

def draw_leaf(img, cx, cy, scale):
    r = int(10*scale)
    cv2.ellipse(img,(cx,cy),(r,int(r*0.6)),45,0,360,(0,200,0),-1)
    return (29, cx, cy, r*2, int(r*1.2))

def draw_crown(img, cx, cy, scale):
    pts = np.array([[cx-int(30*scale),cy],[cx,cy-int(20*scale)],[cx+int(30*scale),cy]])
    cv2.fillPoly(img,[pts],(200,200,0))
    return (30, cx, cy-int(10*scale), int(60*scale), int(20*scale))

def draw_root(img, cx, cy, scale):
    for i in range(3):
        cv2.line(img,(cx,cy+int(i*10*scale)),(cx+random.randint(-20,20),cy+int((i+1)*15*scale)),(100,50,0),2)
    return (31, cx, cy+int(15*scale), int(40*scale), int(30*scale))

def draw_apple(img, cx, cy, scale):
    r = int(8*scale)
    cv2.circle(img,(cx,cy),r,(0,0,255),-1)
    return (32, cx, cy, r*2, r*2)

# ─────────────── ④ 이미지 생성 루프 ───────────────
def norm(bb):
    _, x, y, w, h = bb
    return (bb[0], x/IMG_SIZE, y/IMG_SIZE, w/IMG_SIZE, h/IMG_SIZE)

def make_split(num, split):
    for idx in tqdm(range(num), desc=split):
        img = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
        labels = []
        # 기존 기본 객체
        labels.append(draw_tree(img, random.randint(120,200), random.randint(100,160), random.uniform(0.8,1.2)))
        hx, hy, sc = random.randint(350,500), random.randint(260,320), random.uniform(0.9,1.2)
        labels.append(draw_house(img, hx, hy, sc))
        labels.append(draw_window(img, hx-25, hy-10, sc*0.8))
        labels.append(draw_window(img, hx+25, hy-10, sc*0.8))
        labels.append(draw_door(img, hx, hy+10, sc))
        labels.append(draw_person(img, random.randint(460,580), random.randint(260,300), random.uniform(0.8,1.1)))

        # ── 신규 객체들 랜덤 추가 ──
        for fn, count in [
            (draw_cloud, random.randint(0,2)),
            (draw_sun, random.randint(0,1)),
            (draw_wall, random.randint(0,1)),
            (draw_chimney, random.randint(0,1)),
            (draw_smoke, random.randint(0,2)),
            (draw_roof, random.randint(0,1)),
            (draw_fence, random.randint(0,1)),
            (draw_shadow, 1),
            (draw_ground_line, 1),
            (draw_grass, random.randint(1,3)),
            (draw_tulip, random.randint(0,2)),
            (draw_daisy, random.randint(0,2)),
            (draw_blueprint, random.randint(0,1)),
            (draw_face, random.randint(0,1)),
            (draw_eye, random.randint(0,1)),
            (draw_nose, random.randint(0,1)),
            (draw_mouth, random.randint(0,1)),
            (draw_shoulder, random.randint(0,1)),
            (draw_leg, random.randint(0,2)),
            (draw_foot, random.randint(0,2)),
            (draw_button, random.randint(0,2)),
            (draw_clothes, random.randint(0,1)),
            (draw_underwear, random.randint(0,1)),
            (draw_branch, random.randint(0,2)),
            (draw_leaf, random.randint(0,3)),
            (draw_crown, random.randint(0,1)),
            (draw_root, random.randint(0,2)),
            (draw_apple, random.randint(0,2)),
        ]:
            for _ in range(count):
                cx, cy, scale = random.randint(50,590), random.randint(50,590), random.uniform(0.5,1.2)
                labels.append(fn(img, cx, cy, scale))

        # 저장
        img_path   = OUT_DIR/f"images/{split}/{idx:04d}.jpg"
        label_path = OUT_DIR/f"labels/{split}/{idx:04d}.txt"
        cv2.imwrite(str(img_path), img)
        with open(label_path, "w") as f:
            for bb in labels:
                f.write(" ".join(f"{v:.6f}" for v in norm(bb)) + "\n")

make_split(N_TRAIN, "train")
make_split(N_VAL,   "val")

# data.yaml
(OUT_DIR/"yolo_dataset.yaml").write_text(f"""
path: {OUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
""", encoding="utf-8")
print("★ Synthetic dataset 생성 완료 →", OUT_DIR.resolve())