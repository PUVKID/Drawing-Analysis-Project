# synthetic_single_object_generator.py
import random, math, os, cv2, math
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ─────────────── ① 기본 설정 ───────────────
OUT_DIR        = Path("yolo_single_object_dataset")
CLASSES        = [
    "house","window","door","tree","person",
    "cloud","sun","wall","chimney","smoke","roof","fence",
    "shadow","ground_line","grass","tulip","daisy","blueprint",
    "face","eye","nose","mouth","shoulder","leg","foot","button",
    "clothes","underwear","branch","leaf","crown","root","apple"
]
IMG_SIZE       = 640
N_PER_CLASS    = 10000  # 각 클래스별 생성 장수
RANDOM_SEED    = 42
# ────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 폴더 구조 생성
for split in ["train", "val"]:
    (OUT_DIR/"images"/split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR/"labels"/split).mkdir(parents=True, exist_ok=True)

# ─────────────── ② 공통 유틸 ───────────────
def rand_color():
    """30~255 사이의 랜덤 컬러 리턴"""
    return tuple(int(c) for c in np.random.randint(100,256,3))

def with_random_colors(func):
    """
    draw_xxx(img, *args, **kwargs) 호출 시
    fill_color, edge_color 두 개의 랜덤 색상을 자동 생성해서
    func 에 넘겨 줍니다.
    """
    def wrapper(img, *args, **kwargs):
        fill_color = rand_color()
        edge_color = rand_color()
        return func(img, *args,
                    fill_color=fill_color,
                    edge_color=edge_color,
                    **kwargs)
    return wrapper

def normalize(bb):
    """YOLO 형식 (cls, x_center, y_center, w, h) → 모두 [0,1] 로 정규화"""
    cls,x,y,w,h = bb
    return (cls, x/IMG_SIZE, y/IMG_SIZE, w/IMG_SIZE, h/IMG_SIZE)

# ─────────────── ③ 클래스별 draw 함수 ───────────────
@with_random_colors
def draw_house(img, cx, cy, scale, fill_color, edge_color):
    w, h = int(120*scale), int(90*scale)
    x1, y1 = cx-w//2, cy-h//2
    x2, y2 = x1+w, y1+h
    # 본체
    cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
    # 지붕 (삼각형)
    pts = np.array([[x1, y1], [x2, y1], [cx, y1-int(40*scale)]])
    cv2.fillPoly(img, [pts], edge_color)
    return (0, cx, cy+h*0.05, w, h*1.1)

@with_random_colors
def draw_window(img, cx, cy, scale, fill_color, edge_color):
    w = int(30*scale); h = w
    x1, y1 = cx-w//2, cy-h//2
    x2, y2 = x1+w, y1+h
    # 유리 채움
    margin = max(2, int(2*scale))
    cv2.rectangle(img,
                  (x1+margin, y1+margin),
                  (x2-margin, y2-margin),
                  fill_color, -1)
    # 프레임
    cv2.rectangle(img, (x1, y1), (x2, y2), edge_color, thickness=margin)
    # 격자
    cv2.line(img, ((x1+x2)//2, y1+margin), ((x1+x2)//2, y2-margin), edge_color, margin)
    cv2.line(img, (x1+margin, (y1+y2)//2), (x2-margin, (y1+y2)//2), edge_color, margin)
    return (1, cx, cy, w, h)

@with_random_colors
def draw_door(img, cx, cy, scale, fill_color, edge_color):
    w, h = int(35*scale), int(55*scale)
    x1, y1 = cx-w//2, cy-h//2
    x2, y2 = x1+w, y1+h
    frame_th = max(2, int(2*scale))
    # 문틀
    cv2.rectangle(img, (x1, y1), (x2, y2), edge_color, thickness=frame_th)
    # 문판
    cv2.rectangle(img,
                  (x1+frame_th, y1+frame_th),
                  (x2-frame_th, y2-frame_th),
                  fill_color, -1)
    # 패널 디테일
    mid_y = (y1+y2)//2
    mid_x = (x1+x2)//2
    cv2.line(img, (x1+frame_th, mid_y), (x2-frame_th, mid_y), edge_color, thickness=frame_th//2)
    cv2.line(img, (mid_x, y1+frame_th), (mid_x, y2-frame_th), edge_color, thickness=frame_th//2)
    # 손잡이
    knob_r = max(2, int(2*scale))
    cv2.circle(img, (x2-frame_th-knob_r*2, mid_y), knob_r, edge_color, -1)
    return (2, cx, cy, w, h)

@with_random_colors
def draw_tree(img, cx, cy, scale, fill_color, edge_color):
    tw, th = int(25*scale), int(100*scale)
    x1, y1 = cx-tw//2, cy
    # 줄기
    cv2.rectangle(img, (x1, y1), (x1+tw, y1+th), fill_color, -1)
    # 수관
    rad = int(80*scale)
    cv2.circle(img, (cx, cy), rad, edge_color, -1)
    return (3, cx, cy+rad//2, rad*1.8, rad*2.2)

@with_random_colors
def draw_person(img, cx, cy, scale, fill_color, edge_color):
    # 머리
    head_r = int(15*scale)
    cv2.circle(img, (cx, cy), head_r, fill_color, -1)
    # 몸통
    body_h = int(40*scale); body_w = int(20*scale)
    cv2.ellipse(img,
                (cx, cy+head_r+body_h//2),
                (body_w//2, body_h//2),
                0, 0, 360, edge_color, -1)
    # 팔다리
    thick = max(2, int(2*scale))
    sh_y = cy+head_r
    # 팔
    cv2.line(img, (cx-body_w//2, sh_y), (cx-body_w//2-int(25*scale), sh_y+int(10*scale)), (0,0,0), thick)
    cv2.line(img, (cx+body_w//2, sh_y), (cx+body_w//2+int(25*scale), sh_y+int(10*scale)), (0,0,0), thick)
    # 다리
    hip_y = cy+head_r+body_h
    cv2.line(img, (cx-int(body_w*0.3), hip_y), (cx-int(body_w*0.3), hip_y+int(30*scale)), (0,0,0), thick)
    cv2.line(img, (cx+int(body_w*0.3), hip_y), (cx+int(body_w*0.3), hip_y+int(30*scale)), (0,0,0), thick)
    # bbox
    w = max(body_w, head_r*2) + int(25*scale)
    h = head_r*2 + body_h + int(30*scale)
    return (4, cx, cy+head_r+(body_h//2), w, h)

@with_random_colors
def draw_cloud(img, cx, cy, scale, fill_color, edge_color):
    rs = [int(30*scale), int(40*scale), int(35*scale)]
    for dx, r in zip([-40,0,40], rs):
        shade = random.randint(220, 255)
        cv2.circle(img, (cx+dx, cy+int(10*scale)), r, (shade,shade,shade), -1)
    size = max(rs)*2
    return (5, cx, cy, size, size)

def draw_sun(img, cx, cy, scale):
    """
    태양: 채우기 노랑, 광선은 주황
    """
    r = int(30 * scale)
    fill_color   = (0, 255, 255)   # Yellow
    ray_color    = (0, 165, 255)   # Orange-ish
    # 원 그리기
    cv2.circle(img, (cx, cy), r, fill_color, -1)
    # 광선 그리기
    for ang in range(0, 360, 45):
        x2 = int(cx + math.cos(math.radians(ang)) * (r + 10 * scale))
        y2 = int(cy + math.sin(math.radians(ang)) * (r + 10 * scale))
        cv2.line(img, (cx, cy), (x2, y2), ray_color, 2)
    return (6, cx, cy, r * 2, r * 2)

@with_random_colors
def draw_wall(img, cx, cy, scale, fill_color, edge_color):
    w, h = int(200*scale), int(100*scale)
    x1, y1 = cx-w//2, cy-h//2; x2,y2 = x1+w,y1+h
    cv2.rectangle(img,(x1,y1),(x2,y2), fill_color, -1)
    # 벽돌 가로줄
    row_h = max(int(20*scale),5)
    shade = max(fill_color[0]-20,0)
    for yy in range(y1+row_h, y2, row_h):
        cv2.line(img, (x1,yy),(x2,yy),(shade,shade,shade),1)
    # 세로줄
    col_w = max(int(50*scale),10)
    for i,row in enumerate(range(y1,y2,row_h)):
        off = col_w//2 if i%2 else 0
        xs = x1-off
        while xs < x2:
            x = xs+col_w
            if x1<x<x2: cv2.line(img,(x,row),(x,row+row_h),(shade,shade,shade),1)
            xs += col_w
    return (7, cx, cy, w, h)

@with_random_colors
def draw_chimney(img, cx, cy, scale, fill_color, edge_color):
    w, h = int(20*scale), int(60*scale)
    x1, y1 = cx-w//2, cy-h; x2,y2 = x1+w,y1+h
    cv2.rectangle(img,(x1,y1),(x2,y2), fill_color, -1)
    # 벽돌라인
    bh = max(int(15*scale),4)
    mortar = tuple(max(c-30,0) for c in edge_color)
    for yy in range(y1+bh,y2,bh):
        cv2.line(img,(x1,yy),(x2,yy), mortar,1)
    bw = max(int(10*scale),6)
    for i,yy in enumerate(range(y1,y2,bh)):
        off = bw//2 if i%2 else 0
        xs=x1-off
        while xs<x2:
            x=xs+bw
            if x1<x<x2: cv2.line(img,(x,yy),(x,yy+bh),mortar,1)
            xs += bw
    # 상단 캡
    cap_h = max(int(8*scale),3)
    cv2.rectangle(img,(x1-2,y1-cap_h),(x2+2,y1), mortar, -1)
    return (8, cx, cy-h//2, w, h+cap_h)

@with_random_colors
def draw_smoke(img, cx, cy, scale, fill_color, edge_color):
    for i in range(3):
        r = int(10*scale*(1+i*0.3))
        cv2.circle(img,(cx+random.randint(-5,5), cy-int(i*15*scale)), r, fill_color, -1)
    return (9, cx, cy-int(15*scale), r*2, r*2)

@with_random_colors
def draw_roof(img, cx, cy, scale, fill_color, edge_color):
    """지붕: 삼각형 + 테두리"""
    w, h = int(140*scale), int(60*scale)
    pts = np.array([[cx-w//2, cy], [cx+w//2, cy], [cx, cy-h]])
    cv2.fillPoly(img, [pts], fill_color)
    cv2.polylines(img, [pts], isClosed=True, color=edge_color, thickness=2)
    return (10, cx, cy - h//2, w, h)

@with_random_colors
def draw_fence(img, cx, cy, scale, fill_color, edge_color):
    """울타리: 세로 기둥 + 가로 대들보"""
    w, h = int(200*scale), int(40*scale)
    x1, y1 = cx - w//2, cy
    # 기둥
    for i in range(0, w, int(20*scale)):
        px = x1 + i
        cv2.rectangle(img, (px-2, y1), (px+2, y1+h), fill_color, -1)
    # 대들보
    cv2.rectangle(img, (x1, y1+h//3), (x1+w, y1+h//3+4), fill_color, -1)
    # 테두리
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), edge_color, thickness=2)
    return (11, cx, cy + h//2, w, h)

@with_random_colors
def draw_shadow(img, cx, cy, scale, fill_color, edge_color):
    """그림자: 타원 모양"""
    w, h = int(100*scale), int(20*scale)
    cv2.ellipse(img, (cx, cy+10), (w//2, h//2), 0, 0, 360, fill_color, -1)
    return (12, cx, cy + 10, w, h)

@with_random_colors
def draw_ground_line(img, cx, cy, scale, fill_color, edge_color):
    """지면선: 직선 한 줄"""
    y = cy
    cv2.line(img, (0, y), (img.shape[1], y), edge_color, thickness=2)
    return (13, img.shape[1]//2, y, img.shape[1], 2)

@with_random_colors
def draw_grass(img, cx, cy, scale, fill_color, edge_color):
    """잔디: 여러 줄기로 이루어진 풀밭"""
    for _ in range(20):
        x = cx + random.randint(-50, 50)
        y_end = cy - int(20*scale)
        cv2.line(img, (x, cy), (x + random.randint(-3,3), y_end), fill_color, thickness=2)
    return (14, cx, cy-10, 100, 20)

@with_random_colors
def draw_tulip(img, cx, cy, scale, fill_color, edge_color):
    """
    튤립: 줄기 + 3개의 겹쳐진 꽃잎으로 보다 튤립 모양을 구현합니다.
    - fill_color: 꽃잎 채우기 색
    - edge_color: 꽃잎 테두리 및 줄기 색
    """
    # 1) 줄기 그리기
    stem_thickness = max(2, int(2 * scale))
    stem_top_y = cy - int(30 * scale)
    cv2.line(img, (cx, cy), (cx, stem_top_y), edge_color, thickness=stem_thickness)

    # 2) 꽃잎 치수
    petal_w = int(14 * scale)
    petal_h = int(20 * scale)

    # 3) 중앙 꽃잎
    cv2.ellipse(
        img,
        (cx, stem_top_y),
        (petal_w, petal_h),
        0,    # 회전 각도
        0, 180,           # 위쪽 반원
        fill_color,
        -1
    )
    cv2.ellipse(
        img,
        (cx, stem_top_y),
        (petal_w, petal_h),
        0,
        0, 180,
        edge_color,
        thickness=2
    )

    # 4) 왼쪽 꽃잎 (15도 기울기)
    left_center = (cx - int(10 * scale), stem_top_y + int(5 * scale))
    cv2.ellipse(
        img,
        left_center,
        (petal_w, petal_h),
        15,
        0, 180,
        fill_color,
        -1
    )
    cv2.ellipse(
        img,
        left_center,
        (petal_w, petal_h),
        15,
        0, 180,
        edge_color,
        thickness=2
    )

    # 5) 오른쪽 꽃잎 (-15도 기울기)
    right_center = (cx + int(10 * scale), stem_top_y + int(5 * scale))
    cv2.ellipse(
        img,
        right_center,
        (petal_w, petal_h),
        -15,
        0, 180,
        fill_color,
        -1
    )
    cv2.ellipse(
        img,
        right_center,
        (petal_w, petal_h),
        -15,
        0, 180,
        edge_color,
        thickness=2
    )

    # 6) YOLO bounding box 계산
    box_w = petal_w * 2 + int(10 * scale)
    box_h = int(30 * scale) + petal_h
    # 클래스 ID 15, 중심 x,y 그대로, 너비/높이 반환
    return (15, cx, cy - 10, box_w, box_h)

@with_random_colors
def draw_daisy(img, cx, cy, scale, fill_color, edge_color):
    """데이지: 흰색 꽃잎 + 노란 중심"""
    for ang in range(0,360,45):
        x = int(cx + math.cos(math.radians(ang))*int(12*scale))
        y = int(cy + math.sin(math.radians(ang))*int(12*scale))
        cv2.circle(img, (x, y), int(5*scale), fill_color, -1)
    cv2.circle(img, (cx, cy), int(7*scale), edge_color, -1)
    return (16, cx, cy, 25, 25)

@with_random_colors
def draw_blueprint(img, cx, cy, scale, fill_color, edge_color):
    """청사진: 채움 + 그리드"""
    w, h = int(100*scale), int(100*scale)
    x1, y1 = cx-w//2, cy-h//2
    # 배경
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), fill_color, -1)
    # 외곽선
    cv2.rectangle(img, (x1, y1), (x1+w, y1+h), edge_color, 2)
    # 그리드
    for i in range(1,4):
        xi = x1 + i*(w//4)
        yi = y1 + i*(h//4)
        cv2.line(img,(xi,y1),(xi,y1+h),edge_color,1)
        cv2.line(img,(x1,yi),(x1+w,yi),edge_color,1)
    return (17, cx, cy, w, h)

@with_random_colors
def draw_face(img, cx, cy, scale, fill_color, edge_color):
    """얼굴: 살구톤 채움 + 눈•입 디테일"""
    r = int(20*scale)
    cv2.circle(img, (cx, cy), r, fill_color, -1)
    ex, ey = int(r*0.5), int(r*0.2)
    cv2.circle(img, (cx-ex, cy-ey), int(3*scale), edge_color, -1)
    cv2.circle(img, (cx+ex, cy-ey), int(3*scale), edge_color, -1)
    cv2.ellipse(img, (cx, cy+ey), (int(8*scale), int(4*scale)), 0, 0, 180, edge_color, 2)
    return (18, cx, cy, r*2, r*2)

@with_random_colors
def draw_eye(img, cx, cy, scale, fill_color, edge_color):
    """눈: 양쪽에 원 두 개"""
    dx = int(10 * scale)
    r = int(5 * scale)
    cv2.circle(img, (cx - dx, cy), r, fill_color, -1)
    cv2.circle(img, (cx + dx, cy), r, fill_color, -1)
    cv2.circle(img, (cx - dx, cy), r, edge_color, 1)
    cv2.circle(img, (cx + dx, cy), r, edge_color, 1)
    return (19, cx, cy, dx*2, r*2)

@with_random_colors
def draw_nose(img, cx, cy, scale, fill_color, edge_color):
    """코: 삼각형"""
    pts = np.array([
        [cx, cy - int(5 * scale)],
        [cx - int(3 * scale), cy + int(5 * scale)],
        [cx + int(3 * scale), cy + int(5 * scale)]
    ])
    cv2.fillPoly(img, [pts], fill_color)
    cv2.polylines(img, [pts], isClosed=True, color=edge_color, thickness=1)
    return (20, cx, cy, int(6 * scale), int(10 * scale))

@with_random_colors
def draw_mouth(img, cx, cy, scale, fill_color, edge_color):
    """입: 반원"""
    r_x = int(10 * scale)
    r_y = int(5 * scale)
    y = cy + int(10 * scale)
    cv2.ellipse(img, (cx, y), (r_x, r_y), 0, 0, 180, fill_color, -1)
    cv2.ellipse(img, (cx, y), (r_x, r_y), 0, 0, 180, edge_color, 1)
    return (21, cx, y, r_x*2, r_y*2)

@with_random_colors
def draw_shoulder(img, cx, cy, scale, fill_color, edge_color):
    """어깨: 반원형"""
    w = int(40 * scale)
    cv2.ellipse(img, (cx, cy), (w//2, int(5*scale)), 0, 0, 180, fill_color, -1)
    cv2.ellipse(img, (cx, cy), (w//2, int(5*scale)), 0, 0, 180, edge_color, 2)
    return (22, cx, cy, w, int(5 * scale))

@with_random_colors
def draw_leg(img, cx, cy, scale, fill_color, edge_color):
    """다리: 직선"""
    length = int(30 * scale)
    cv2.line(img, (cx, cy), (cx, cy + length), edge_color, thickness=3)
    return (23, cx, cy + length//2, 3 * scale, length)

@with_random_colors
def draw_foot(img, cx, cy, scale, fill_color, edge_color):
    """발: 타원"""
    r_x, r_y = int(10 * scale), int(5 * scale)
    y = cy + int(35 * scale)
    cv2.ellipse(img, (cx, y), (r_x, r_y), 0, 0, 360, fill_color, -1)
    cv2.ellipse(img, (cx, y), (r_x, r_y), 0, 0, 360, edge_color, 1)
    return (24, cx, y, r_x*2, r_y*2)

@with_random_colors
def draw_button(img, cx, cy, scale, fill_color, edge_color):
    """단추: 세 개의 작은 원"""
    r = int(3 * scale)
    for i in range(3):
        y = cy + int(i * 10 * scale)
        cv2.circle(img, (cx, y), r, fill_color, -1)
        cv2.circle(img, (cx, y), r, edge_color, 1)
    return (25, cx, cy + 10 * scale, r*2, r*2 * 3)

@with_random_colors
def draw_clothes(img, cx, cy, scale, fill_color, edge_color):
    """
    옷: 여러 스타일
    - T-shirt, Dress, Pants 중 랜덤 선택
    """
    style = random.choice(['tshirt', 'dress', 'pants'])
    if style == 'tshirt':
        # 티셔츠: 몸통 + 반팔
        body_w, body_h = int(50*scale), int(60*scale)
        x1, y1 = cx - body_w//2, cy - body_h//2
        x2, y2 = x1 + body_w, y1 + body_h
        # 몸통
        cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
        # 소매
        sleeve_w, sleeve_h = int(15*scale), int(20*scale)
        cv2.rectangle(img, (x1-sleeve_w, y1), (x1, y1+sleeve_h), fill_color, -1)
        cv2.rectangle(img, (x2, y1), (x2+sleeve_w, y1+sleeve_h), fill_color, -1)

    elif style == 'dress':
        # 드레스: 상단은 몸통, 하단은 삼각형 모양
        top_w, top_h = int(40*scale), int(40*scale)
        x1, y1 = cx - top_w//2, cy - int(10*scale)
        # 상단 몸통
        cv2.rectangle(img, (x1, y1), (x1+top_w, y1+top_h), fill_color, -1)
        # 하단 삼각 치마
        pts = np.array([
            [x1, y1+top_h],
            [x1+top_w, y1+top_h],
            [cx, y1+top_h+int(40*scale)]
        ])
        cv2.fillPoly(img, [pts], fill_color)

    else:  # pants
        # 바지: 두 개의 다리 모양
        leg_w, leg_h = int(20*scale), int(40*scale)
        # 왼쪽 다리
        lx1 = cx - int(leg_w*1.1)
        ly1 = cy - leg_h//2
        cv2.rectangle(img, (lx1, ly1), (lx1+leg_w, ly1+leg_h), fill_color, -1)
        # 오른쪽 다리
        rx1 = cx + int(leg_w*0.1)
        ry1 = ly1
        cv2.rectangle(img, (rx1, ry1), (rx1+leg_w, ry1+leg_h), fill_color, -1)

    # 테두리
    # 전체 바운딩 박스
    w = int(50*scale)
    h = int(60*scale + 20*scale)
    x1b, y1b = cx - w//2, cy - h//2
    x2b, y2b = x1b + w, y1b + h
    cv2.rectangle(img, (x1b, y1b), (x2b, y2b), edge_color, thickness=2)

    return (26, cx, cy, w, h)


@with_random_colors
def draw_underwear(img, cx, cy, scale, fill_color, edge_color):
    """
    속옷: 여러 스타일
    - briefs, boxers, bra 중 랜덤 선택
    """
    style = random.choice(['briefs', 'boxers', 'bra'])

    if style == 'briefs':
        # 브리프: 허리띠 + 몸통
        w, h = int(40*scale), int(30*scale)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
        belt_h = max(2, int(3*scale))
        cv2.rectangle(img, (x1, y1), (x2, y1 + belt_h), edge_color, -1)

    elif style == 'boxers':
        # 박서: 허리 + 넓은 몸통 + 다리 끼임
        w, h = int(50*scale), int(30*scale)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
        # 다리 끼임
        cut = int(10*scale)
        cv2.rectangle(img,
                      (x1 + cut, y2 - cut),
                      (x1 + 2*cut, y2),
                      edge_color, -1)
        cv2.rectangle(img,
                      (x2 - 2*cut, y2 - cut),
                      (x2 - cut, y2),
                      edge_color, -1)

    else:  # bra
        # 브라: 두 컵 + 중앙 밴드
        cup_r = int(12*scale)
        # 컵 위치
        left_cx  = cx - cup_r*1
        right_cx = cx + cup_r*1
        cv2.circle(img, (left_cx,  cy), cup_r, fill_color, -1)
        cv2.circle(img, (left_cx,  cy), cup_r, edge_color, 2)
        cv2.circle(img, (right_cx, cy), cup_r, fill_color, -1)
        cv2.circle(img, (right_cx, cy), cup_r, edge_color, 2)
        # 밴드
        band_h = max(3, int(5*scale))
        cv2.rectangle(img,
                      (left_cx,  cy - band_h//2),
                      (right_cx, cy + band_h//2),
                      edge_color, -1)
        # bounding box 크기
        w = int((right_cx - left_cx) + cup_r)
        h = int(cup_r*2)

    # 반환할 bounding box
    return (27, cx, cy, w, h)

@with_random_colors
def draw_branch(img, cx, cy, scale, fill_color, edge_color):
    """가지: 선"""
    length = int(60 * scale)
    x2, y2 = cx + int(length * 0.7), cy - int(length * 0.7)
    cv2.line(img, (cx, cy), (x2, y2), edge_color, 3)
    return (28, (cx+x2)//2, (cy+y2)//2, length, 3)

@with_random_colors
def draw_leaf(img, cx, cy, scale, fill_color=None, edge_color=None):
    # radius in x and y
    r = int(10 * scale)
    r_y = int(r * 0.6)
    # filled leaf
    cv2.ellipse(
        img,
        (cx, cy),
        (r, r_y),
        45, 0, 360,
        fill_color,
        -1
    )
    # optional outline
    if edge_color is not None:
        cv2.ellipse(
            img,
            (cx, cy),
            (r, r_y),
            45, 0, 360,
            edge_color,
            2
        )
    # return YOLO bbox: class 29
    return (29, cx, cy, r * 2, r_y * 2)

@with_random_colors
def draw_crown(img, cx, cy, scale, fill_color, edge_color):
    """왕관: 삼각형"""
    pts = np.array([[cx-int(30*scale), cy], [cx, cy-int(20*scale)], [cx+int(30*scale), cy]])
    cv2.fillPoly(img, [pts], fill_color)
    cv2.polylines(img, [pts], isClosed=True, color=edge_color, thickness=2)
    return (30, cx, cy-int(10*scale), int(60*scale), int(20*scale))

@with_random_colors
def draw_root(img, cx, cy, scale, fill_color, edge_color):
    """뿌리: 여러 선"""
    for i in range(3):
        y2 = cy + int((i+1)*15*scale)
        x2 = cx + random.randint(-20, 20)
        cv2.line(img, (cx, cy+int(i*10*scale)), (x2, y2), edge_color, 2)
    return (31, cx, cy + 15*scale, 40*scale, 30*scale)

def draw_apple(img, cx, cy, scale):
    """
    사과: 채우기 빨강, 테두리 진한 빨강
    """
    r = int(8 * scale)
    fill_color   = (0, 0, 255)   # Red
    border_color = (0, 0, 150)   # Dark red
    # 채우기
    cv2.circle(img, (cx, cy), r, fill_color, -1)
    # 테두리
    cv2.circle(img, (cx, cy), r, border_color, 1)
    return (32, cx, cy, r * 2, r * 2)

# 매핑
FUNC_MAP = {name: globals()[f"draw_{name}"] for name in CLASSES}

# ─────────────── ④ 단일 객체 생성 루프 ───────────────
def make_single_object_dataset(split):
    for cls_idx, cls_name in enumerate(CLASSES):
        fn = FUNC_MAP[cls_name]
        for i in tqdm(range(N_PER_CLASS), desc=f"{split}/{cls_name}"):
            img = np.full((IMG_SIZE,IMG_SIZE,3),255,np.uint8)
            cx = IMG_SIZE//2 + random.randint(-50,50)
            cy = IMG_SIZE//2 + random.randint(-50,50)
            scale = random.uniform(0.8,1.2)
            bb = fn(img,cx,cy,scale)
            base = f"{cls_idx:02d}_{i:05d}"
            cv2.imwrite(str(OUT_DIR/f"images/{split}/{base}.jpg"), img)
            with open(OUT_DIR/f"labels/{split}/{base}.txt","w") as f:
                f.write(" ".join(f"{v:.6f}" for v in normalize(bb)))

if __name__ == "__main__":
    make_single_object_dataset("train")
    make_single_object_dataset("val")
    # data.yaml
    with open(OUT_DIR/"yolo_dataset.yaml","w",encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR.resolve().as_posix()}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    print("★ Single-object dataset 생성 완료 →", OUT_DIR.resolve())
