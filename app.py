import os
from pathlib import Path
from datetime import datetime
import cv2
import gradio as gr
from ultralytics import YOLO
from fpdf import FPDF
from openai import OpenAI

# ────────────────────────────
# 설정
# ────────────────────────────
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ["OPENAI_API_KEY"]
)
YOLO_MODEL_PATH = r"C:\Users\PUVKID\Desktop\심화캡스톤\drawing_yolo_run\v8_finetune\weights\last.pt"
PROJECT_DIR = Path(r"C:\Users\PUVKID\Desktop\심화캡스톤")
INPUT_DIR = PROJECT_DIR / "input"
RESULT_DIR = PROJECT_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)

TEMP_IMG_PATH = INPUT_DIR / "temp.jpg"
PDF_OUTPUT_PATH = RESULT_DIR / "report.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")

# ────────────────────────────
# 초기화
# ────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
model = YOLO(YOLO_MODEL_PATH)

# ────────────────────────────
# Helper: PDF 리포트 생성
# ────────────────────────────
def build_pdf(summary: str, interpretation: str):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Malgun", "", str(FONT_PATH))

    pdf.add_page()
    pdf.set_font("Malgun", size=14)
    pdf.cell(0, 10, "그림 분석 리포트", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Malgun", size=11)
    pdf.multi_cell(0, 6, summary)
    pdf.ln(4)

    pdf.set_font("Malgun", size=12)
    pdf.cell(0, 8, "GPT 해석", ln=True)
    pdf.set_font("Malgun", size=11)
    pdf.multi_cell(0, 6, interpretation)
    pdf.ln(4)

    # 원본 그림 삽입
    pdf.image(str(TEMP_IMG_PATH), x=10, w=90)
    pdf.output(str(PDF_OUTPUT_PATH))

# ────────────────────────────
# 분석 + GPT 상담 메시지
# ────────────────────────────
def analyze_and_suggest(image):
    # 1) 입력 저장 (RGB->BGR 변환)
    cv2.imwrite(str(TEMP_IMG_PATH), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 2) YOLO 추론
    results = model(str(TEMP_IMG_PATH), conf=0.05, imgsz=640)

    # 3) 탐지 성공/실패 요약 + 타임스탬프
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    has_det = len(results[0].boxes.data) > 0
    status = "탐지 결과: 성공" if has_det else "탐지 결과: 실패"
    analysis_summary = f"[분석 시각] {now}\n\n{status}"

    # 4) GPT 해석 (gpt-4o-mini 사용)
    prompt = (
        "당신은 따뜻하고 다정한 미술치료 상담사입니다.\n\n"
        "다음은 그림 분석 요약입니다:\n\n"
        f"{analysis_summary}\n\n"
        "그림 분석을 더 구체적으로 해주고 그림 분석을 최대한 사람에게 공감을 해주는 식으로 해줘.이 결과를 바탕으로 내담자에게 건넬 수 있는 부드럽고 공감 어린 질문이나 코멘트를 10~15 문장으로 작성해주세요."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 따뜻하고 다정한 미술치료 상담사입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        auto_msg = resp.choices[0].message.content
    except Exception as e:
        auto_msg = f"GPT 호출 오류: {e}"

    # 5) PDF 작성
    build_pdf(analysis_summary, auto_msg)

    # 6) 결과 반환 (원본 RGB 이미지)
    return image, str(PDF_OUTPUT_PATH), auto_msg

# ────────────────────────────
# 추가 GPT 상담
# ────────────────────────────
def ask_gpt(user_input):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 따뜻하고 다정한 미술치료 상담사입니다."},
                {"role": "user", "content": user_input}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"GPT 호출 오류: {e}"

# ────────────────────────────
# Gradio UI
# ────────────────────────────
with gr.Blocks(title="그림 심리 분석 & 상담 챗봇") as demo:
    gr.Markdown("## 🎨 그림 심리 분석 & 상담 챗봇")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="numpy", label="그림 업로드")
            btn = gr.Button("분석 및 상담 메시지 생성")
            res_img = gr.Image(type="numpy", label="분석 결과")
            res_pdf = gr.File(label="PDF 리포트 다운로드")
            res_msg = gr.Textbox(label="AI 자동 상담 메시지")

        with gr.Column():
            chat_in = gr.Textbox(lines=2, placeholder="상담 내용을 입력하세요")
            chat_out = gr.Textbox(label="GPT 상담 결과")
            chat_btn = gr.Button("상담하기")

    btn.click(analyze_and_suggest, inputs=img_in, outputs=[res_img, res_pdf, res_msg])
    chat_btn.click(ask_gpt, inputs=chat_in, outputs=chat_out)

if __name__ == "__main__":
    demo.launch()
