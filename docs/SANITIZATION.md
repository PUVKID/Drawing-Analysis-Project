# 공개 사본의 제한된 변환

원본 파일과 최초 공개 준비본은 수정하지 않았습니다. 승인된 환경변수 변환과 공개 제외만 적용했습니다. 의존성이나 실험 알고리즘은 추가·수정하지 않았습니다.

| 환경변수 | 준비할 값 |
|---|---|
| `OPENAI_API_KEY` | 사용자 본인의 서비스 API 키 |
| `PDF_OUTPUT_PATH` | 보고서를 저장할 출력 폴더 경로 |
| `YOLO_MODEL_PATH` | 허가된 모델 파일의 경로 |

## 실제 변경

- `app.py`: api_key_literal_to_environment
- `config.py`: api_key_literal_to_environment, model_or_output_path_literal_to_environment, required_os_import_added
- `fastAPI.py`: api_key_literal_to_environment
- `fine_tuning.py`: api_key_literal_to_environment
- `list_finetuning_jops.py`: api_key_literal_to_environment

## 검사 범위

UTF-8, 파일 SHA-256, 추가 원본 두 번의 안정성 읽기, Python AST 및 가능한 노트북 셀 문법, 노트북 출력·첨부·메타데이터 제거, 비밀정보·개인정보 후보 탐지를 검사했습니다. 기존 매직 명령 셀 등 일반 Python 구문이 아닌 셀은 실행하지 않았습니다. 인증 정보·입력 자료가 없으므로 실험 결과와 전체 실행은 확인하지 않았습니다.

최초 프로젝트 명세 SHA-256: `c1bba4887a8b03f1cce53262d02de377c73b53f4ef5c4a48334a1c8446de37f7`.
