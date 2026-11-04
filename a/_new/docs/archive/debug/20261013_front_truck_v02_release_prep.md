# Front_Truck v02(symmetric_scratch) 릴리스 준비 — 인수인계 (2026-10-13 KST)

적용: truck_dashcam `BSD_F606/Front_Truck` v02 후보 `v02_symmetric_scratch` · 기준: 2026-10-13

이전 경위(v01 릴리스 ONNX 불일치, ONNX→Board 손실 조사, YUV420·ReLU·보정 이미지 실험)는
[20260929_front_truck_v01_onnx_mismatch.md](20260929_front_truck_v01_onnx_mismatch.md)에 있다.
그 문서의 2026-10-07 절은 기존 체크아웃(`feat/tools-analysis-board-porting`)에 **커밋되지 않은 수정**으로만 있다.

## 결정

- 최종 Front 후보는 `v02_symmetric_scratch`(symmetric_yolov8n, 무작위 초기화, SiLU, SGD lr0 0.01, 500 epoch, best ep485)다.
- v01 대비 비교는 같은 조건 수치로 한다. v01 MANIFEST Board 0.704 는 릴리스 번들의 옛 보드 덤프 기준이고,
  배포 `.bin` 을 1280x1024·임계값 0.001 빌드로 다시 재면 0.672 다. 대시보드는 사용자 요청으로 MANIFEST 기록값만 비교한다.

## 평가 결과 (customEval, ROI [320,0,1280,1024], GT 50,371, IoU 0.5)

| AP50 | v01 MANIFEST | v01 재측정 | v02 scratch |
|---|---:|---:|---:|
| ONNX (conf 0.01, PC 후처리) | 0.697 | 0.697 | 0.731 |
| ONNX A (conf 0.001) | — | 0.705 | 0.738 |
| ONNX B (보드식 후처리 근사) | — | 0.689 | 0.723 |
| Board (1280x1024, 임계값 0.001 빌드) | 0.704 | 0.672 | 0.713 |
| traffic_light Board | 0.592 | 0.573 | 0.649 |

- 결과 파일: `runs/detect/BSD_F606/Front_Truck/{v01,v02_symmetric_scratch}/eval/`
- 대시보드(재측정 제외, MANIFEST 기록만 비교): https://claude.ai/artifact/H9QwYA8AcJZ7EjG2WhqRXD ,
  원본 `runs/analysis/front_truck_dashboard/index.html`

### lr0 실험 (원격 192.168.103.162 에서 학습, 이 저장소로 복사)

| lr0 | ONNX A | ONNX B | B traffic_light |
|---|---:|---:|---:|
| 0.003 | 0.732 | 0.716 | 0.652 |
| 0.005 | 0.736 | 0.718 | 0.657 |
| 0.01 (scratch) | 0.738 | 0.723 | 0.677 |
| 0.015 | 0.745 | 0.727 | 0.682 |
| 0.03 (483/500 학습 중 스냅샷) | 0.734 | 0.719 | 0.653 |

lr015 의 +0.004(B)는 재학습 흔들림 범위와 가까워 확정 개선으로 보지 않았다. 결과: `runs/detect/BSD_F606/Front_Truck/v02_symmetric_scratch_lr*/eval/`.
lr03 은 원격 학습 완료 후 다시 받아 export·평가해야 한다.

## 릴리스 산출물 위치

### NPU `.bin` · ONNX · config

| 파일 | 원본 (원격 192.168.103.162, 보드 측정·임계값 추출에 사용) | 이 서버 재현본 (2026-10-01) |
|---|---|---|
| 기준 경로 | `~/Workspace/jdh/ultralytics_projects/novatek/convert/front_truck_stage4_v02_symmetric_scratch_20260917/nvtai_tool/` | `novatek/convert/front_truck_stage4repro_v02_symmetric_scratch_20261001/` |
| `nvt_model.bin` | `output/…/sdk/nvt_model.bin` md5 `df076799` | `runs/npu/novatek/front_truck_stage4repro_v02_symmetric_scratch_20261001/nvtai_tool/output/…/sdk/nvt_model.bin` md5 `df076799` (바이트 동일) |
| 입력 ONNX | `input/model/customer/…/deploy_float.onnx` md5 `67df2873` (= 원격 `weights/best_split.onnx`) | `runs/detect/BSD_F606/Front_Truck/v02_symmetric_scratch/weights/best_novatek.onnx` md5 `481f4e55` |
| gen_config / sim_config | `config/…/cnn25/{gen,sim}_config.txt` (`a57354eb` / `2c1a9124`) | `{gen,sim}_config.txt` (`97371d43` / `e0c0f5d1`) |

- 두 ONNX 는 가중치 127/127·노드 235·입출력이 같고 파일 바이트만 다르다. 두 config 는 경로 표기와 주석만 다르다(보정 720장 동일).

### 클래스별 임계값

- 원격 `runs/debug/1091_nv12_csc_candidates/board_thresholds_v02_symmetric_scratch.json` (2026-09-23, `tools/calc_class_thresholds.py`, stage4 보드 예측)
- optimal: bus 0.248, car 0.137, motorcycle 0.187, person 0.186, truck 0.186, traffic_light 0.408
- 같은 덤프의 customEval 값: `runs/detect/BSD_F606/Front_Truck/v02_symmetric_scratch/eval/board_stage4/evaluation.json` `metrics.thresholds`

### v01 형식 `analysis/` 폴더

`runs/analysis/front_truck_v02_symmetric_scratch/`

- `analysis/`: NAS `Front_Truck/v01/analysis/` 와 같은 구성
  - `metrics_front_{onnx,board}.{json,txt}` (edge/center 포함), `thresholds_detail.json`
  - `thr_tool_front_{onnx,board}.json`, `F1conf_front_*.png`
  - `iputest/output.json` (= `eval/board_stage4_src/output.json`)
  - `errorcases/` 48장, `errorcases_montage.jpg`
- `SOURCES.md`: 파일별 출처와 정의. v01 산출물을 v01 예측으로 재현해 정의를 확인했다.
  - 영역: ROI 가로 3등분, 박스 중심 x 기준. edge = 좌·우 1/3.
  - 영역 P/R/F1: 그 영역 자체의 optimal threshold 기준.
  - errorcases: thr_tool optimal 이상 예측, 이미지별 FP+FN 상위 48장. GT 빨강, 예측 흰색.
- `_scripts/`: 생성 스크립트와 입력 파일(`gt_roi_front.json`, `pred_front_onnx.json`)
- `compare_errorcases_v01_v02/`: v01 errorcases 47장에서 v01 대 v02 비교 그림
  - `improved/` 35장, `worse/` 10장, `improved_top6_montage.jpg`, `README.md`
  - 오류 797 → 663 (−17%), FN 662 → 545, FP 135 → 118

### v01 대 v02 분석 요약 (두 `analysis/` 폴더 비교)

- ONNX mAP 0.697 → 0.731 (+0.035). 6개 클래스 모두 개선, traffic_light +0.066. optimal threshold 기준 FN −1,774 (−12%), FP −463.
- 영역별 ONNX: edge +0.042, center +0.032.
- Board 차이는 v01 기준값에 따라 다르다. MANIFEST 0.704 대비 +0.009, 같은 조건 재측정 0.672 대비 +0.041.
  v01 MANIFEST 는 Board(0.704)가 ONNX(0.697)보다 높아 측정 조건이 달랐을 가능성이 크다.
- 공통 약점: 오류 상위 이미지는 두 모델 모두 INDOOR·4K 야간 교차로의 작은 객체 미검출(FN)이다.

## 데이터셋 (Dashcam/Front_traffic_light, 6 클래스)

| | 이미지 | 인스턴스 |
|---|---:|---:|
| Train | 45,254 (배경 2) | 307,329 |
| Validation | 6,076 | 61,588 (ROI 필터 후 50,371) |

- Train 은 이 서버에 없다. 원격 `~/Workspace/data/Front_traffic_light/Train`(`Dashcam/Front_traffic_light` 은 심볼릭 링크)에서 학습했다.
- `/mnt/HDD_1/BSD_F606_FRONT/dataset/Train/labels` 는 학습 라벨과 바이트 동일(45,254개)이다.
- NAS `01_DATASET/97_RELEASE_BEFORE/rework_traffic_light/train/gt` 는 변환 전 원본이다.
  - 형식: `클래스명,x1,y1,x2,y2` 픽셀, 12 클래스
  - 변환 규칙: rider→person, bicycle→motorcycle, other_vehicle→truck 로 합치고 license_plate·rubber_cone·face 를 빼면 45,254개 전부 일치
  - NAS 에만 있는 JP 야간 14개는 학습에 없다.

## 남은 일

1. 릴리스 번들
   - `configs/experiments/BSD_F606/Front_Truck/v02_symmetric_scratch/` → `configs/releases/BSD_F606/Front_Truck/v02/` 복사
   - `network/` 에 `.bin`·ONNX·config, `analysis/` 에 위 폴더
   - MANIFEST 작성 (v01 MANIFEST 형식, `nas_source`·metrics·issues)
2. NAS 업로드: `/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v02/{analysis,network,weight}`
3. 비교 그림 문서 첨부. 첨부할 문서는 미정이다.
4. ONNX→Board traffic_light 잔여 손실(약 −0.02) 조사 후보
   - TL score 분포 B 대 Board
   - 같은 GT 기준 confidence 변화
   - bbox 위치 변화
   - cls 출력 고정밀 양자화(`spec_demarc_tensors`)
5. lr03 원격 학습 완료본 수신·평가

## 미검증

- v01 MANIFEST Board 0.704 의 측정 조건. 옛 덤프의 입력 해상도·빌드 임계값을 모른다.
- ONNX B 후처리는 보드 소스를 따른 근사다. 보드 int16 출력 2장에서 검출의 약 75% 만 일치했다.
- 비교 그림·analysis 는 ONNX 예측 기준이다. Board 예측 기준 errorcases 는 만들지 않았다.
