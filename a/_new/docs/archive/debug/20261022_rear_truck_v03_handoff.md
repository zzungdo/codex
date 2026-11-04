# Rear_Truck v03 준비 — 162 서버 인수인계 (2026-10-22 KST)

적용: `BSD_F606/Rear_Truck` · 기준: 2026-10-22 · 원 서버: 원래 작업 PC(`/home/ncn/Workspace/jdh/ultralytics_projects_new`)
이 문서만 읽고 162 서버(`ncn-ws40-02`, `~/Workspace/jdh/ultralytics_projects-board-porting`)에서 바로 학습·평가를 시작할 수 있게 쓴다.

## 지금 상황 한 줄

v02 릴리스(E1)는 끝났고, v03 는 **데이터를 먼저 고치는** 방향이다. CnR·vignetting 을 원본에서 재생성했고(CnR_v2·vignetting_v2),
원 서버 GPU0 에서 그 데이터로 v03 스크래치 학습이 돌고 있다(2026-10-22 14:37 시작, 약 42시간). 162 서버는 **v02 와 같은 구성**으로 추가 REAR 학습을 하려고 세팅했다.

## 진행 중인 학습 (2026-10-22 17:00 기준)

| 서버 | 실험 | GPU | 상태 | 비교 대상 |
|---|---|:---:|---|---|
| 원 서버 | `v03_symmetric_scratch` (CnR_v2·vignetting_v2, 스크래치 500 ep, patience 100) | 0 | 10-22 14:37 시작, 28/500, 에포크당 약 5분 → **10-24 오전 8시 전후 종료 예상**. ep27 val mAP50 0.729 (E2b 같은 시점 0.725) | E2b (데이터만 다름), v02 |
| 162 서버 | `v01_symmetric_ft_1t` (1T 만) | 0 | 사용자가 162 서버 Claude 에서 시작 (10-22 오후) | E0 |
| 162 서버 | `v01_symmetric_ft_1t_bgv3` (1T + 배경 v3) | 1 | 위와 같음 | E1 = v02 |

- 원 서버 컨테이너 `rear_v03_symmetric_scratch`, 로그 `runs/detect/BSD_F606/Rear_Truck/v03_symmetric_scratch.train.log`, 진행 확인은 `results.csv`.
- 원 서버 GPU1 은 비어 있다.

### 학습 종료 후 할 일
1. v03: `best.pt` → ONNX export → customEval A/B (전체·leak·clean·truck1t) → E2b 와 비교. 공간 분석·FP 평가셋(배경 v2 정제 2,184장)도.
2. v03 NPU 변환 후 보드 덤프는 사용자가 측정 → Board 평가.
3. 162 1T 실험: E0 의 1T 부분 평가를 먼저 하고, 두 모델을 1T 부분·전체·FP 평가셋으로 대조군과 비교.
4. 결과를 이 문서와 v03 개발 문서(Claude Docs)에 반영.

### 아직 결정하지 않은 것
- 오인식(차선분리대·볼라드를 person ≥0.5): 하드 네거티브 수집 착수 여부. 사용자가 본 장면 위치를 받아서 시작하기로 했다.
- val 검수(기준서 → 후보 추출 → 전수), 세션 hold-out 도입 여부.
- 보드 카메라 LDC 모드, NPU 입력 해상도·지연 예산 (입력 종횡비 실험의 전제).
- CnR_v2 min-keep 을 0.3 으로 맞출지 (현재 0.35 유지).

## 162 서버에 세팅된 것

| 구분 | 경로 (저장소 기준) | 비고 |
|---|---|---|
| 저장소 | `~/Workspace/jdh/ultralytics_projects-board-porting` | 커밋 `7e095ba` (v02 학습 당시와 같음), 브랜치 `feat/tools-analysis-board-porting` |
| `.env` | 저장소 루트 | `DATA_ROOT=/home/ncn/Workspace/data` → 컨테이너 `/dataset` |
| 데이터 | `~/Workspace/data/1T_REAR/` | v02 구성: pure(train·val), CnR, vignetting, new_labeling, background_rear_v3 + **v03 데이터 CnR_v2, vignetting_v2**. `README.md` 에 폴더별 설명 |
| v02 데이터셋 yaml | `configs/datasets/1t_rear_cnr_postprocess_bg_v3.yaml` | git 에 없음 (원 서버에서 복사) |
| v02 학습 설정 | `configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/exp.yaml` | = 릴리스 `configs/releases/BSD_F606/Rear_Truck/v02/exp.yaml` |
| v03 데이터셋 yaml | `configs/datasets/1t_rear_cnrv2_vigv2_bg_v3.yaml` | v02 yaml 에서 CnR → CnR_v2, vignetting → vignetting_v2 두 줄만 다름 |
| 그 밖의 설정 | `.../v01_symmetric_ft_nobg` (E0), `.../v01_symmetric_scratch_bgv3_pat100` (E2b), `.../v03_symmetric_scratch` | v03 스크래치 설정 = E2b 에서 데이터셋만 바꾼 것. **원 서버에서 이미 학습 중이므로 같은 이름으로 또 돌리지 않는다** |
| 출발 가중치 | `runs/detect/BSD_F606/Rear_Truck/v01_symmetric/weights/best.pt` | md5 `b1a2f287c8a5d479503a9bc329761581` (NAS Rear_Truck/v01 과 같음) |
| 비교 모델 | `runs/detect/BSD_F606/Rear_Truck/{v01_symmetric,v01_symmetric_ft_bgv3,v01_symmetric_scratch_bgv3_pat100}/weights/` | best.pt + best_novatek.onnx |
| 보드 덤프 | 각 모델 폴더 `board/` | v01 은 보드에서 0.01 로 잘린 덤프(이미지별 JSON 배열이 이어 붙은 형식). E1·E2b 는 0.001 빌드 + `_score001` 사본 |
| 평가 GT·설정 | `runs/detect/BSD_F606/Rear_Truck/_eval_cfg/` | `ipu_gt_coco_format.json`(6,152장), `gt_variants/ipu_gt_{leak,clean,truck1t}.json`, `exp_{full,leak,clean,truck1t}.yaml`. truck1t = iputest 중 TRUCK1T 3,894장 |
| 분석 스크립트 | `runs/analysis/spatial_prep.py`, `plot_spatial.py`, `rear_truck_thresholds/` | 공간 분석·임계값 |

## 바로 할 수 있는 일

### v02 와 같은 구성으로 학습

설정을 **새 실험 이름 폴더로 복사**한 뒤 바꿀 것만 바꾼다. 결과 폴더는 설정 경로에서 자동으로 정해진다 (`configs/experiments/<P>/<C>/<V>/exp.yaml` → `runs/detect/<P>/<C>/<V>/`).

```bash
cd ~/Workspace/jdh/ultralytics_projects-board-porting
V=v02_repro            # 새 실험 이름
mkdir -p configs/experiments/BSD_F606/Rear_Truck/$V
cp configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/exp.yaml configs/experiments/BSD_F606/Rear_Truck/$V/
docker compose run -d --no-deps --name rear_$V dair-train bash -lc \
  "uv run --frozen python3 tools/train_custom_trainer.py -w configs/experiments/BSD_F606/Rear_Truck/$V/exp.yaml -d 0 \
   > runs/detect/BSD_F606/Rear_Truck/$V.train.log 2>&1"
```

- 확인: 로그 첫 부분에 `100309 images, 4737 backgrounds` 가 나오면 v02 와 같은 데이터다 (배경 v3 4,724 + 빈 라벨 13).
- v02 레시피: v01 best.pt 에서 파인튜닝, SGD lr0 1e-3, lrf 0.01, cosine, warmup 1, 100 epoch / patience 30, batch 48, 320x320, mosaic 0.3. 원 서버 기준 약 9시간 (89 epoch 에서 멈춤, best ep59).
- 스크래치 레시피는 E2b 설정 (lr0 0.01, 500 epoch, **patience 100**, warmup 3). 약 40시간. patience 15 로 하면 ep~200 에서 조기종료돼 낮게 나온다 (E2 0.760 대 E2b 0.785).

### 평가 (customEval, 6,152장 iputest)

```bash
R=runs/detect/BSD_F606/Rear_Truck; M=<실험 이름>; C=$R/_eval_cfg/exp_full.yaml   # leak / clean 도 같은 방식
docker compose run --rm --no-deps dair-train bash -lc "
uv run --frozen python3 tools/customEval.py -w $C --eval onnx --onnx-model $R/$M/weights/best_novatek.onnx --conf 0.001 --save-pred --save-dir $R/$M/eval/onnx_A_conf0001 &&
uv run --frozen python3 tools/customEval.py -w $C --eval onnx --onnx-model $R/$M/weights/best_novatek.onnx --conf 0.001 --save-pred --save-dir $R/$M/eval/onnx_B_boardpp \
  -c pred.post_proc.from_=projects.metrics.board_postproc pred.post_proc.import_=board_postproc_v8"
```

- ONNX 는 먼저 export 해야 한다: `tools/export_onnx.py --target novatek -i 320 -y 320` (출력 2개: `output0 [1,5,2100]` cls, `[1,4,2100]` DFL).
- A = PC 기본 후처리, B = 보드 NMS 근사(차량 묶음 + IoM 0.6 / IoU 0.3). Board = 보드 덤프를 `--eval board --pred <덤프> --conf 0.001` 로.
- 보드 덤프는 사용자가 직접 측정한다 (NPU 변환 → 보드). 덤프 형식: `name`·`bbox`(xyxy)·`category`(대문자)·`score`.
- **두 mAP 코어(customEval / ultralytics)는 섞어 비교하지 않는다 (D-020).** 릴리스 수치는 customEval.
- 재학습만으로도 ±0.004 흔들린다. 그 안의 차이는 개선으로 보지 않는다.

## 기준 수치 (customEval mAP50, 6,152장)

| 모델 | 설명 | ONNX A | ONNX B | Board |
|---|---|---:|---:|---:|
| v01_symmetric | 이전 릴리스 (스크래치) | 0.780 | 0.764 | 0.751 (score≥0.01) |
| **E1 = v02** | v01 파인튜닝 + 배경 v3 | 0.787 | 0.772 | 0.763 (0.759 score≥0.01) |
| E0 | E1 에서 배경만 뺀 대조군 | 0.787 | 0.771 | 0.763 |
| E2b | 스크래치 + 배경 v3, 500 ep | 0.785 | 0.769 | 0.758 |

- clean 부분집합(train 과 클립이 안 겹치는 2,991장) Board: v01 0.716 / E1 0.725 / E0 0.722 / E2b 0.716. leak 쪽은 모든 모델이 +0.08~0.09 높다 → 모델 순위는 같다.
- 보드 클래스별 optimal 임계값(E1): bus 0.291 / car 0.500 / motorcycle 0.250 / person 0.273 / truck 0.251.
- 배경 v2 정제 FP 평가셋(2,184장, ONNX, 클래스별 optimal): 이미지당 FP v01 0.109 → v02 0.020.

## 확인된 사실 (반복 조사 불필요)

- **val/iputest**: val 7,894장 중 62% 가 train 과 같은 클립. 날짜·차량까지 독립인 것은 14장뿐. 장면 구성도 다르다
  (iputest car 21% / bus 9% / truck 10%, train car 42% / bus 3% / truck 6%; 5T 쪽은 정차 주차장 클립이 iputest 의 9%).
  iputest = val 에서 DPI 144 JPEG 1,742장을 뺀 집합.
- **보드 NMS**: 차량 묶음 + IoM 0.6 규칙이 가려진 트럭을 지워 truck 이 −0.05. 보드 규칙은 바꿀 수 없다.
- **공간 분석 (Board)**: REAR 는 가장자리 recall 0.577 대 중앙 0.784, FN 의 60% 가 가장자리. 특히 **맨 윗줄 recall 0.19~0.63** (화면 위쪽 끝에 걸린 먼 차량).
- **CnR**: 원본 LDC0 에서 x=270 y=0 1380x1050 crop → 1920x1080, 잔존율 필터 없음 → 화면 끝 가짜 띠 라벨 563개. 저화질(양자화 평균 29) 재저장.
- **vignetting**: type1 270 0 1380 910 / type2 270 0 1380 1080, min-keep 0.45 (6,734/6,734 재현). 역시 저화질 재저장.
- **작은 박스**: 최소 크기 기준을 적용하면 대부분 실제 원거리 객체가 빠진다 → 적용하지 않음.
- **작은 person 라벨** (pure, 짧은 변 < 20px, 4,097개): 200개 블라인드 판정에서 사람 아님 0개. 모델이 못 잡는 이유는 크기다.
- **new_labeling**: 아주 먼 주차 차량까지 라벨링돼 있다 (다른 하위셋과 기준 차이). 오류는 아님.
- **효과 없음으로 확인된 것**: lr0 미세 조정, ReLU, YUV420 미세조정 (Front), 변환 percentile 변경·16비트 보호 (Rear, SDK 가 설정 무시), low_cam 추가 (E3).

## 원 서버에만 있는 것

| 항목 | 원 서버 경로 |
|---|---|
| CnR·vignetting 원본 | `/mnt/HDD_1/BSD_F606_REAR/new_dataset/org_dataset/train/`, `/mnt/HDD_1/BSD_F606_REAR/new_new_dataset/vignetting/` |
| v03 스크래치 학습 (진행 중) | `runs/detect/BSD_F606/Rear_Truck/v03_symmetric_scratch/`, 로그 `.../v03_symmetric_scratch.train.log`, 대조군 E2b |
| 배경 v2 FP 평가셋 | `/home/ncn/Workspace/data/1T_REAR/background_rear_v2/train` + 제외 목록 `runs/analysis/rear_bg_v2_review/excluded_images.json` |
| 재생성·점검 스크립트와 그림 | `runs/analysis/rear_truck_label_check/` (`cnr_regen/`, `vig_regen/`, 작은 박스 그림), `runs/analysis/rear_person_label_check/` |
| 인수인계 원본 | `docs/archive/debug/20261013_rear_truck_bgv3_training.md` (v02), `20261013_front_truck_v02_release_prep.md` (Front v02) |
| v03 개발 문서 (Claude Docs) | https://claude.ai/artifact/DLkBRcDmwVq5KdeY8HfT9E |

CnR_v2·vignetting_v2 는 이 서버에도 복사돼 있다 (각 폴더 `README.md` 에 생성 조건). v03 데이터로 새 실험을 할 때는
`1t_rear_cnrv2_vigv2_bg_v3.yaml` 을 쓰는 설정을 새 실험 이름으로 만든다. 로그에 `100309 images, 4740 backgrounds` 가 나오면 v03 데이터다.

## 162 서버 실험 후보 판단 (2026-10-22, 사용자와 검토)

사용자가 제안한 네 실험을 비판적으로 본 결과다. 같은 조건 재학습이 ±0.004 흔들리고 평가셋 판별력도 약하므로(위 val/iputest 참고),
**기대 효과가 그보다 작거나 결과가 결정을 바꾸지 못하는 실험은 하지 않는다.**

| 실험 | 판정 | 이유 |
|---|---|---|
| ① 화질만 비교 (CnR_v2·vignetting_v2 이미지 + 기존 라벨) | **보류 (별로)** | 1920x1080 → 320x320 축소에서 JPEG 흔적은 거의 사라진다. 차이가 잡음 안일 가능성이 높고, 어떤 결과든 고화질 유지 결정은 그대로다 |
| ② 1T 만 학습 | **스크래치는 별로 / v03 에서 1T 파인튜닝은 진행 후보** | 1T 는 train 의 45%(약 43,400장). 5T 를 빼면 같은 클래스의 일반화가 줄어 떨어질 위험이 더 크다. iputest 1T 부분(3,894장, `exp_truck1t.yaml`)도 작고 train 과 겹친다. v03 가 끝나면 그 가중치에서 1T 데이터로만 파인튜닝(v02 레시피, 몇 시간)하는 쪽이 이길 가능성이 있는 유일한 형태 |
| ③ crop 없이 원본으로 학습 | **보류 (별로)** | CnR 원본은 LDC0(평가셋은 LDC1/LDC2), vignetting 원본은 검은 원 테두리가 크다. 실제 입력과 더 다른 데이터라 결과 해석이 애매하다. "CnR·vignetting 이 도움이 되나"가 궁금하면 두 하위셋을 **빼고** 학습하는 쪽이 정확하다. 먼저 **실제 보드 카메라의 LDC 모드**를 확인할 것 |
| ④ min-size 로 작은 객체 라벨 제거 | **안 됨** | `analyze_crop.py --min-size` 는 라벨만 지우고 픽셀은 남긴다 → 보이는 먼 객체를 배경으로 가르치고, 평가 GT 에는 남아 recall 이 떨어진다. 하려면 제품 요구(몇 m 밖은 검출 불필요) 확정 + 학습 무시/마스킹(학습기 수정) + 평가도 같은 기준이 모두 필요 |

**우선할 것**
1. 원 서버 v03 결과를 기다린다 (2026-10-24 새벽 전후 예상). 나머지 해석의 기준이다.
2. 그동안 162 서버: 하드 네거티브 수집(아래, GPU 는 추론만), 보드 카메라 LDC 모드와 NPU 입력 해상도·지연 예산 확인.
3. v03 완료 후 ② 1T 파인튜닝. 비교는 v03 와 `exp_truck1t.yaml` 로.
4. 작은 객체가 문제라면 라벨 제거가 아니라 **입력 종횡비·해상도**(16:9 를 320x320 으로 눌러 사람 폭이 2~3px 이 된다)가 직접적인 해결책이다. NPU 예산 안에서 가능한지부터 본다.

## 1T 파인튜닝 실험 (2026-10-22 준비 완료, 실행 전)

v02(E1)와 **같은 출발점(v01 best.pt)·같은 레시피**에서 학습 데이터만 1T 로 좁힌다. 기존 E1·E0 가 그대로 대조군이 된다.

| 모델 | 출발 가중치 | 학습 데이터 | 배경 v3 | 전체 A / B / Board | 1T 부분 A / B / Board |
|---|---|---|:---:|---|---|
| v01 릴리스 | (스크래치) | 1T+5T 95,586 | ✗ | 0.780 / 0.764 / 0.751 | 0.755 / 0.744 / 0.736 |
| E1 = v02 릴리스 | v01 best.pt | 1T+5T 95,586 | ✓ | 0.787 / 0.772 / 0.763 | 0.764 / 0.754 / 0.748 |
| E0 | v01 best.pt | 1T+5T 95,586 | ✗ | 0.787 / 0.771 / 0.763 | 미측정 → 먼저 잴 것 |
| **`v01_symmetric_ft_1t`** (GPU0) | v01 best.pt | **1T 43,410** | ✗ | — | — |
| **`v01_symmetric_ft_1t_bgv3`** (GPU1) | v01 best.pt | **1T 43,410** | ✓ 4,724 | — | — |

- 비교: `ft_1t` 대 E0, `ft_1t_bgv3` 대 E1 (각각 5T 포함 여부만 다름). 두 비교가 같은 방향이면 믿을 만하다. `ft_1t` 대 `ft_1t_bgv3` = 1T 에서의 배경 효과.
- 판단: iputest 1T 부분(`_eval_cfg/exp_truck1t.yaml`)에서 대조군보다 ±0.004 이상 높아야 효과. 전체 점수는 5T 때문에 떨어질 수 있다. **1T 집중은 오인식 대책이 아니므로** 배경 FP 평가셋도 함께 본다.
- 데이터: v02 구성(기존 CnR·vignetting) 중 1T 만. 목록 `/dataset/1T_REAR/_lists/train_1t_{only,bgv3}.txt` (설명 `_lists/README.md`),
  yaml `configs/datasets/1t_rear_1t_{only,bgv3}.yaml`. 1T 판정 = 파일명 토큰 `TRUCK1T` 또는 `1T`. 컨테이너에서 목록·라벨 짝·5T 0장 확인함.
- 설정: `configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_ft_1t{,_bgv3}/exp.yaml` — E1 설정과 `overrides.data` 한 줄만 다르다.
- E0 가중치·보드 덤프도 이 서버 `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_nobg/` 에 있다.

실행 (두 GPU 동시):
```bash
cd ~/Workspace/jdh/ultralytics_projects-board-porting; R=runs/detect/BSD_F606/Rear_Truck
docker compose run -d --no-deps --name rear_v01_symmetric_ft_1t dair-train bash -lc \
  "uv run --frozen python3 tools/train_custom_trainer.py -w configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_ft_1t/exp.yaml -d 0 > $R/v01_symmetric_ft_1t.train.log 2>&1"
docker compose run -d --no-deps --name rear_v01_symmetric_ft_1t_bgv3 dair-train bash -lc \
  "uv run --frozen python3 tools/train_custom_trainer.py -w configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_ft_1t_bgv3/exp.yaml -d 1 > $R/v01_symmetric_ft_1t_bgv3.train.log 2>&1"
```
- 시작 확인 (로그의 train Scanning 줄): `ft_1t` → `43410 images, 11 backgrounds`, `ft_1t_bgv3` → `48134 images, 4735 backgrounds`
  (11 = 원래 빈 라벨인 1T 이미지: pure 2, CnR 2, vignetting 1, new_labeling 6).
- 라벨 캐시는 목록 첫 하위셋에 생긴다: `ft_1t` → `vignetting/train/labels.cache`, `ft_1t_bgv3` → `new_labeling/train/labels.cache`. 두 실험이 같은 캐시를 덮어쓰지 않게 시작 하위셋을 다르게 했다.
- 예상 소요: v02 가 원 서버에서 89 epoch 약 9시간. 데이터가 절반이라 더 짧을 것.

## 다음 후보 작업 (미정)

- 오검출: 차선분리대·볼라드를 person 으로 0.5 이상 잡는 사례가 남아 있다. 주행 영상에서 person≥0.5 프레임을 뽑아 판정 → 하드 네거티브 배경(bg v4, 학습용)과 분리대 FP 평가셋(평가용)을 **영상 단위로 나눠** 만든다. v03 와 효과를 섞지 않도록 다음 학습에 넣는다.
- val 검수: 기준서를 train 관행에서 먼저 만들고, 원본 GT 는 두고 val_v2 GT 를 따로 만든다. 모델 예측과 GT 비교로 후보를 먼저 뽑는다.
- 세션 단위 hold-out: (날짜×차량) 63개 세션 중 20% 를 떼는 안이 분포가 가장 비슷했다(최대 세션 비중 27%). 도입 시 train 이 20% 줄고 기준 모델 재학습이 필요하다.

## 주의

- GPU 는 `-d <번호>` 로만 지정한다. `CUDA_VISIBLE_DEVICES` 를 같이 주면 ultralytics 가 덮어써서 다른 GPU 에서 돈다 (2026-10-03 실제 발생).
- `workers <= 4`. 컨테이너 안 파이썬은 `uv run` 으로 실행한다 (`/usr/bin/python3` 에는 torch 가 없다).
- 학습 설정 키는 `exp.yaml` 의 `overrides` 가 이긴다. 바꾸려면 설정 파일을 고친다.
- 데이터셋 경로를 바꾸면 그 폴더의 `labels.cache` 를 지운다 (첫 학습에서 다시 만든다).
- 라벨 없는 이미지는 배경으로 학습된다 (빈 라벨 파일도 같음). 배경을 추가할 때는 객체가 없는지 먼저 검수한다.
- `configs/` 의 Rear 설정과 이 문서는 git 에 커밋돼 있지 않다. 커밋하려면 `master` 가 아닌 브랜치에서, 사람이 push 한다.
