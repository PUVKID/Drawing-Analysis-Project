# 📚 기본

YOLO 그림 탐지, 텍스트 질문, PDF 기록을 연결하는 기본 앱입니다.

- [근거 자료](../../app.py)
- [전체 보관 기록](../VERSIONS.md) · [프로젝트 소개](../../README.md)

## 🧩 주요 내용

- analyze_and_suggest는 업로드 그림의 탐지 결과를 분석·제안 흐름으로 전달합니다.
- ask_gpt는 텍스트 대화, build_pdf는 대화 결과의 PDF 생성을 담당합니다.
- 학습용 자료 준비와 YOLO 학습은 별도 스크립트로 구성했습니다.

## 🔄 자료 구성

그림 분석과 대화 기능을 구현한 기본 앱입니다.

## 📍 관련 자료

- [app.py](../../app.py) · `def analyze_and_suggest` · 63
- [app.py](../../app.py) · `def ask_gpt` · 104
- [train_yolo_dataset.py](../../train_yolo_dataset.py) · `def train` · 59

셀 번호는 0부터, 소스 행 번호와 PDF 쪽 번호는 1부터 셉니다. 
