# 사용법

적용: 학습·평가·포팅·분석 CLI · 기준: 2026-09-28

[환경 설정](setup.md) 후 호스트 저장소 루트에서 아래 명령으로 진입한다.
학습·평가·ONNX 변환·분석은 학습 컨테이너의 `/workspace`에서 실행한다.
표의 `<값>`은 사용할 경로·값으로 바꾼다.

```bash
docker compose exec dair-train bash
```

## 학습

| 구분 | 명령어 | 주요 옵션 | 결과·목적 |
|---|---|---|---|
| 학습 | `uv run python3 tools/train_custom_trainer.py -w <exp.yaml> -d 0` | `--cfg_options epochs=100 batch=32 workers=4` | `best.pt`, `last.pt`, 학습 지표 |
| 분산 학습 | `bash tools/dist_train_custom.sh 2 -w <exp.yaml>` | 첫 번째 인자: GPU 수 | 여러 GPU로 학습 |

출력: `runs/detect/<PRODUCT>/<Camera>/<Version>/`

## 평가·추론

| 구분 | 명령어 | 주요 옵션 | 결과·목적 |
|---|---|---|---|
| PT 평가 | `uv run python3 tools/validate.py -w <exp.yaml> --weight <best.pt> -d 0` | `-a <dataset.yaml>`, `-b 16` | Precision·Recall·mAP |
| ONNX 평가 | `uv run python3 tools/customEval.py -w <exp.yaml> --eval onnx` | `--onnx-model <model.onnx>`, `--conf 0.01`, `--iou-thr 0.45` | `evaluation.json`, 평가 그래프 |
| 보드 결과 평가 | `uv run python3 tools/customEval.py -w <exp.yaml> --eval board --pred <output.json>` | `--save-dir <출력>`, `--dry-run` | 보드 검출 결과와 GT 비교 |
| 추론·시각화 | `uv run python3 tools/infer.py -w <model.pt 또는 model.onnx> -s <이미지 경로>` | `--conf 0.25`, `--iou 0.45`, `--save-dir runs/infer` | 검출 박스 이미지 |

| 도구 | 추가 옵션 | 목적 |
|---|---|---|
| `customEval.py` | `--eval onnx board` | 두 평가를 함께 실행 |
| `customEval.py` | `--mode ap precision recall f1_score cm` | 산출 지표 선택 |
| `customEval.py` | `--save-pred` | `prediction.json` 저장 |
| `customEval.py` | `--dry-run` | 해석된 설정·출력 경로 확인 |
| `infer.py` | `--color bgr --scale byte --resize stretch` | BGR·0~255·stretch 입력 지정 |

## 포팅

입력·설정·보드 배포·검증 기준은 [NPU 문서](npu.md)를 따른다.

ONNX 변환은 학습 컨테이너에서 실행한다.

```bash
EXPORT_DIR=runs/export/corner_v01
mkdir -p "$EXPORT_DIR"
cp -n configs/releases/BSD_F606/Corner_Truck/v01/weight/weights/best.pt "$EXPORT_DIR/best.pt"
uv run python3 tools/export_onnx.py \
  --target novatek -w "$EXPORT_DIR/best.pt" -i 320 -y 320 -s "$EXPORT_DIR/novatek.onnx"
```

| 옵션 | 값·목적 |
|---|---|
| `--target` | `novatek`: raw 2출력 / `sigmastar`: decoded xyxy 단일 출력 / `dms`: pose 3출력 |
| `-w` | 원본 `.pt` |
| `-i`, `-y` | 입력 너비·높이. `-y` 생략 시 너비와 동일 |
| `-s` | 출력 `.onnx` 경로 |
| `-o` | ONNX opset. 기본 Novatek/DMS 13, Sigmastar 12 |
| `--simplify` | Novatek ONNX 단순화 |

보드 변환은 호스트의 `ultralytics_projects` 루트에서 실행한다.

| 구분 | 명령어 | 결과·목적 |
|---|---|---|
| Novatek 환경 시작 | `docker compose up -d npu-convert` | 변환 컨테이너 시작 |
| Novatek 전체 변환 | `docker compose exec npu-convert bash novatek/convert/run_pipeline.sh <model_dir> <test.jpg> 0.45` | Nova 변환 → 양자화·컴파일 → 시뮬레이션 → 비교 이미지 |
| Novatek 재시뮬레이션 | `docker compose exec npu-convert bash novatek/convert/run_simulator.sh <model_dir> <test.jpg>` | 기존 `.bin` 시뮬레이션 |
| Sigmastar 소스 반영 | `bash docker/sigmastar/run.sh 'bash converter/sync_to_sdk.sh'` | 변환·전후처리 코드 SDK 반영 |
| Sigmastar 전체 변환 | `bash docker/sigmastar/run.sh 'bash converter/run.sh data/input/models/model.onnx yolov8_xyxy_320x320 0.45 data/input/calibration data/input/simulation prepared_bgr'` | 후처리 결합 → 양자화·컴파일 → `.img` 및 Float/Fixed/Offline 결과 |

- Novatek 입력: `novatek/convert/<model_dir>/`에 ONNX·보정 이미지·`gen_config.txt`·`sim_config.txt` 준비.
- Novatek 출력: `runs/npu/novatek/<model_dir>/nvtai_tool/output/<PATTERN>/sdk/*.bin`
- Novatek 클래스명 지정: `run_pipeline.sh` 명령 끝에 `--names bus car motorcycle person truck traffic_light` 추가.
- Sigmastar 입력: `prepared_bgr` 사용 시 이미지를 모델 입력 해상도로 미리 crop·resize.
- Sigmastar 출력: `sigmastar/data/output/<model_name>/img/*.img`
- Sigmastar `0.45`: 시각화 임계값. 모델 내부 NMS 임계 변경 옵션은 아니다.

## 분석

입출력·처리 규칙은 [분석 규격](analysis.md)을 따른다.

| 구분 | 명령어 | 결과·목적 |
|---|---|---|
| ROI·정답 필터 | `uv run python3 tools/analyze_crop.py --src <원본> --dst <출력> --crop 428 8 1072 1072 --min-keep 0.5` | crop 이미지·라벨, `spec.json`, `report.json` |
| 예측 공간 분석 | `uv run python3 tools/analyze_heatmap.py --gt gt.json --pred pred.json --tag baseline --thresholds car=0.3 motorcycle=0.4 person=0.5 --output-dir analysis` | `analysis/analysis_baseline.json`: 10×10 집계·구역·크기·클래스별 지표 |

| 도구 | 옵션 | 목적 |
|---|---|---|
| `analyze_crop.py` | `--crop X Y W H` | crop 영역 |
| `analyze_crop.py` | `--min-keep 0.5` | crop 후 박스 최소 잔존율 |
| `analyze_crop.py` | `--min-size 40 --size-metric min_side` | 짧은 변 기준 최소 40px |
| `analyze_crop.py` | `--min-size-per-class "1:47,3:37"` | 클래스별 최소 박스 크기 |
| `analyze_crop.py` | `--edge-mode lens --lens CX CY RX RY` | 렌즈 영역 기준 필터 |
| `analyze_crop.py` | `--edge-margin 5` | 가장자리 마진 5% |
| `analyze_crop.py` | `--clip-ellipse --mask-outside` | 타원 경계로 라벨 자르기·외부 픽셀 마스킹 |
| `analyze_crop.py` | `--drop-empty` | 정답이 남지 않은 이미지 제외 |
| `analyze_heatmap.py` | `--gt`, `--pred` | COCO GT·예측 JSON, bbox는 절대 좌표 xywh |
| `analyze_heatmap.py` | `--thresholds 클래스=기준값 ...` | 클래스별 예측 score 하한 |
| `analyze_heatmap.py` | `--tag`, `--output-dir` | 분석명·출력 디렉터리 |
