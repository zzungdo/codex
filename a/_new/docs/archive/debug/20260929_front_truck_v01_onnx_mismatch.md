# Front_Truck v01 릴리스 — ONNX 와 best.pt 불일치 (2026-09-29 KST)

적용: truck_dashcam `configs/releases/BSD_F606/Front_Truck/v01` 번들 · 기준: 2026-09-29

v01 릴리스 번들의 `network/260706_BSD-F606A_Front_v01.onnx`(이하 v01 릴리스 ONNX)는 같은 번들의
`weight/weights/best.pt`(이하 v01 릴리스 `best.pt`)에서 export 된 모델이 아니다. 두 파일은 다른
가중치이며, 같은 평가 조건에서 mAP 가 0.025 다르다. 번들의 학습 기록(`exp.yaml` · `args.yaml` ·
`best.pt`)은 v01 릴리스 ONNX·보드 수치를 만든 모델의 기록이 아닐 가능성이 높다.

> **제품 구분** — 이 문서의 대상은 **truck_dashcam** 인 `BSD_F606/Front_Truck` 이다. NAS
> `BSD_F606/Front/`(v01~v08)는 **car_dashcam** 제품의 산출물·결과이며 별개 제품이다. 아래에서
> car_dashcam 산출물을 인용할 때는 매번 `car_dashcam Front vNN` 으로 표기한다. 두 제품은
> Front_Truck v01 의 `pretrained_weight.pt` 가 car_dashcam Front v08 `best.pt` 라는 점으로만
> 연결된다(미세조정 시작점).

## 결론

| 항목 | 내용 |
|---|---|
| v01 릴리스 ONNX 가중치 | 번들 `best.pt` · `last.pt` · `pretrained_weight.pt` 어느 것에서도 export 되지 않았다 |
| v01 릴리스 ONNX 계보 | `pretrained_weight.pt`(= car_dashcam Front v08 `best.pt`)와 뿌리가 같고, 그보다 훨씬 멀리 학습됐다 |
| 출처 | 미확인. 추정: MANIFEST `mycfg-unreliable` 의 GA 튜닝 학습 `Front_v01_sym_tuned` |
| 성능 | 같은 `customEval` 조건에서 v01 릴리스 ONNX 0.697 · v01 릴리스 `best.pt` export 0.722 |
| 영향 | v01 대비 개선을 말할 때 기준선을 둘 중 무엇으로 할지 명시해야 한다 |

## 환경

- 저장소 `ultralytics_projects_new`, 브랜치 `feat/tools-analysis-board-porting` (HEAD `7e095ba`)
- 컨테이너 `dair-train`: Python 3.12, torch 2.5.1+cu121, ultralytics 8.4.6, `uv.lock` frozen
- export: `tools/export_onnx.py --target novatek -i 640 -y 512` (opset 13, simplify 없음)
- 평가: `tools/customEval.py -w configs/releases/BSD_F606/Front_Truck/v01/exp.yaml --eval onnx`
  — ROI `[320,0,1280,1024]`, GT 61,588 → 50,371, `conf` 0.01, `iou_thr` 0.45

## 검증 1 — 가중치 동일성 (코드 확인)

ONNX initializer 127개를 순서대로 `np.array_equal` 비교했다.

| 비교 대상 (v01 릴리스 ONNX 기준) | 일치 텐서 |
|---|---|
| 번들 `best.pt` → export | 1 / 127 |
| 번들 `last.pt` → export | 1 / 127 |
| 번들 `pretrained_weight.pt` → export | 1 / 127 |
| car_dashcam Front v05 · v06 `nt98530_640x512.onnx` | 0 / 127 |
| car_dashcam Front v07 `nt98530_640x512.onnx` · v08 `best.onnx` | 1 / 127 |
| (방법 검증) `v02_symmetric` `best_split.onnx` vs 같은 `best.pt` 재export | **127 / 127** |

그래프는 동일하다 — 노드 235개, op 종류 동일, 입력 `[1,3,512,640]`, 출력 `[1,6,6720]`·`[1,4,6720]`,
producer pytorch 2.5.1. md5: v01 릴리스 ONNX `d623cbdc…`, v01 릴리스 `best.pt` 재export `029db152…`.

## 검증 2 — 가중치 유사도 (코드 확인)

원소 1,000개 이상인 59개 레이어의 상관계수와 상대 차이(`‖a−b‖/‖a‖`) 중앙값.

| 비교 | 상관 평균 (최소) | 상대 차이 |
|---|---|---|
| v01 릴리스 ONNX vs `best.pt` · `last.pt` · `pretrained` · 재학습 2종 | 0.875 (0.730) | 0.54~0.55 |
| v01 릴리스 ONNX vs car_dashcam Front v06 · v07 | 0.00~0.04 | 1.42 |
| v01 릴리스 `best.pt` vs `pretrained` | 0.999 (0.997) | 0.04 |
| v01 릴리스 `best.pt` vs 재학습 `v01_asrun` · `v02` | 0.9995 (0.999) | 0.03 |

`lr0` 0.0001 미세조정 모델들은 `pretrained` 와 0.999 로 붙어 있다. v01 릴리스 ONNX 는 같은 뿌리이나
0.875 까지 벗어났다 — 더 큰 학습률 또는 더 긴 학습. GA 후보의 `lr0` 0.0047(47배)과 부합하지만
가중치가 없어 확정할 수 없다.

## 검증 3 — 평가 수치 (ONNX 추론, 같은 조건)

| 모델 | bus | car | moto | person | truck | TL | mAP | mF1 |
|---|---|---|---|---|---|---|---|---|
| v01 릴리스 ONNX | 0.706 | 0.805 | 0.668 | 0.691 | 0.692 | 0.622 | **0.697** | 0.731 |
| v01 릴리스 `best.pt` 재export | 0.728 | 0.809 | 0.713 | 0.737 | 0.707 | 0.638 | **0.722** | — |
| 재학습 `v01_asrun` (640x480) | 0.727 | 0.810 | 0.715 | 0.741 | 0.707 | 0.639 | 0.723 | 0.752 |
| 재학습 `v02` (640x512) | 0.730 | 0.815 | 0.720 | 0.742 | 0.699 | 0.640 | 0.724 | 0.753 |

v01 릴리스 ONNX 0.697 · 보드 0.704 는 MANIFEST `metrics.deployment` 기록과 이 저장소에서 재현
일치했다(보드 `traffic_light` 0.594 vs 기록 0.592). 검출 수는 v01 릴리스 `best.pt` 재export 95,840,
v01 릴리스 ONNX 107,044.

공간·크기 분석(`analyze_heatmap.py`, ROI 좌표 변환, 각자 optimal 임계값)에서 v01 릴리스 `best.pt` 가
전 구역·전 크기에서 앞섰다. 차이가 큰 곳은 가까운 객체(ROI r6~r7 recall +0.04~0.06)와 위쪽 FP
(r1~r3 213 → 38). 수평선 부근(r4)만 recall −0.013. ROI 좌우 끝 열의 FP 는 두 모델이 같다.
산출물: `runs/analysis/front_v01_compare/`.

## 부수 확인

- `pretrained_weight.pt` 는 car_dashcam Front v08 `best.pt`
  (NAS `BSD_F606/Front/v08/weight/weights/best.pt`)와 md5 동일
  (`2fc1d3cc…`). 체크포인트 `train_args.model` 은 `/results/FrontDAIRv8/train3/weights/best.pt`,
  epoch 17.
- 해상도 효과: 같은 레시피를 640x480(`v01_asrun`)·640x512(`v02`)로 재학습한 차이는 +0.001.
  v01 릴리스 ONNX → `v02` 의 +0.028 은 해상도가 아니라 v01 릴리스 ONNX 가중치 차이에서 온다.
- 재현성: `v02` 두 번 학습 0.724 · 0.725, v01 레시피 재학습 0.723 vs v01 릴리스 `best.pt` 0.722.

## 변환 조건 — truck_dashcam Front_Truck v01 릴리스 `.bin` vs v02 `.bin`

v02 `.bin`은 이전 저장소 `ultralytics_projects/novatek/convert/front_truck_v02_symmetric_20260909/`
에서 2026-09-09 에 만들어졌다. 보드 결과 `sym_1280x1024_output.json`(ROI 국소 좌표, x 0~1280)을 만든
`.bin`이 이것(`32a76557…`)인지는 사용자 기억으로만 확인됐다.

| 요인 | v01 릴리스 `.bin` | v02 `.bin` (`32a76557…`, 4,758,792 B) |
|---|---|---|
| 입력 ONNX | v01 릴리스 ONNX (가중치 출처 미확인) | `best_split.onnx` (md5 `73f3cb77…`) |
| gen_config · sim_config | 기준 | 경로 키 4개씩만 다름 — 나머지 줄은 동일 |
| 보정 이미지 | `nt98530_d224k_front/calib` 500장 — 현존하지 않음 | Validation 원본 1920x1080 500장 |
| SDK | 미확인 (`profile.txt` 없음) | v02.05.2503042 |

설정은 v01 릴리스 `network/{gen,sim}_config.txt`를 복사해 경로만 바꾼 것으로 보인다. 두 설정 모두
`preproc/in` 1920x1080 → `preproc/resize` 640x512 이므로 보정은 ROI crop 없이 전체 프레임으로
이뤄진다(보드 추론은 ROI 1280x1024). 보드 Δ(보드 − ONNX) 차이(+0.007 vs −0.016)의 후보는 입력
ONNX와 보정 이미지다. v02 보정 이미지는 평가에 쓴 Validation과 같은 집합이다.

## 변환 환경 재현 (C) — 2026-09-29

대상: truck_dashcam `BSD_F606/Front_Truck` v02 변환. car_dashcam `BSD_F606/Front/`와 다른 제품이다.
입력(`deploy_float.onnx`, config, 보정 500장, 시험 이미지)을 `novatek/convert/front_truck_v02_symmetric_20260909/`
로 그대로 복사하고 `run_pipeline.sh`(wrapper)로 컴파일 → 시뮬 → 시각화를 실행했다.

| 환경 | 2026-09-09 (이전 컨테이너) | 2026-09-29 (새 컨테이너) |
|---|---|---|
| base 이미지 | 순정 `ubuntu:20.04` (dair-yolo/npu, 2026-08-03 빌드) | Novatek `ubuntu20p04:202407013` |
| 수동 설치 | 2026-08-05 컨테이너 안: apt `libpython3.8`, pip onnx 1.17.0 · onnxruntime 1.16.3 · onnxsim 0.4.36 · numpy 1.24.4 · protobuf 5.29.6 | 없음 (이미지 제공) |
| `python3` | 3.8.10 | 3.10.11 (onnx 1.16.1, onnxruntime 1.16.3, onnxsim 0.4.35, numpy 1.26.4) |
| `libprotobuf.so.17` | `/opt/pb36` 소스 빌드 | 이미지 `/lib` |
| SDK | v02.05.2503042 (같은 추출본) | v02.05.2503042 |

| 산출물 | 결과 |
|---|---|
| `deploy.onnx` (Nova ONNX) | md5 일치 `b7a66120…` |
| `nvt_model.bin` | 바이트 일치 `32a76557…` (다른 바이트 0) |
| `quan_info_bitinfo.json` · `quant_input_output_bitinfo.json` · `siminfo.bin` · `process_bit_info.txt` | md5 일치 |
| 시뮬 cosine (이미지 1장 `…_0003.jpg`) | cls 0.9837 / dfl 0.9864 — 동일 |
| Golden vs Sim 검출 (conf 0.45) | 9 vs 8, Sim 에서 `traffic_light 0.525` 1건 누락 |

새 컨테이너를 처음 만들었을 때 순서대로 `libpython3.8.so.1.0` 없음 → `No module named 'onnx'`로 멈췄다.
Dockerfile 에 두 의존성이 없었고 이전 컨테이너는 수동 설치로 돌았다(`docker diff`, `/var/log/dpkg.log`).
결정은 D-027. 시각화는 `npu_manifest.read_imgsz()`가 `preproc/in`을 읽어 anchor 42,420개로 죽었고
`preproc/resize` 우선으로 고쳐 6,720개로 동작했다. 같은 수정이 이전 저장소 `66a2698`(2026-08-06,
`feat/BSD_F606-REAR_Truck-v01` 브랜치에만 있고 master 미병합)에 있었다. SDK 시뮬레이션까지만 확인했고
보드 실행은 하지 않았다.

## SDK v02.05 · v02.06 비교

- v02.06 배포본(`250808_…_release_exe.7z`)의 릴리스 노트는 v02.05 와 같은 파일이다(md5 `249d0390…`).
- Convertor User Guide v1.11(v02.06) 개정 이력에 CNN25(NT98530) 동작 변경 항목은 없다. 추가 항목은
  NT98533 지원, CNN30 전용·신규 정밀도 옵션(`easy_quant`, `smoothquant` 등), 신규 CLI 옵션이다.
- 실측 근거는 `docs/npu.md`(Corner 1모델·4장, cosine 소수 6자리 일치)뿐이다.
- 사내 이미지 `nvt_converter:250818`(NAS `03_SDK/novatek/NPU/nvt_converter/`)에 들어 있는 SDK 도 v02.05.2503042 다.

## 미검증

- `.bin`(`260706_BSD-F606A_Front_v01.bin`)이 v01 릴리스 ONNX 에서 변환됐는지 — 날짜·보드 수치로
  정황만 있다. SDK 시뮬레이터로 `.bin` 과 ONNX 출력을 비교해야 확정된다. v02 `.bin`은 C 재현으로
  환경 재현성을 확인했으나 v01 릴리스 `.bin`은 보정 이미지가 없어 재현할 수 없다.
- v01 릴리스 ONNX 의 원 가중치·설정. 검색 범위: NAS_2 `02_Trained_weights` 파일명
  (`BSD-F606A` · `BSD_F606A` · `Front_v01` · `260706`), car_dashcam Front v01~v08 ONNX 지문.
  미검색: NAS_2 `99_Project&Tools`, NAS_1, SSD_1, 원 학습 호스트(`/home/ncn/Developments/…`).

## 후속

- 비교 기준선 명시: v01 릴리스 ONNX 기준 0.697(보드 0.704) 또는 v01 릴리스 `best.pt` 기준 0.722.
- MANIFEST `Front_Truck/v01` `issues` 반영은 보류(2026-09-29 결정).

## 2026-10-07 Front_Truck ONNX → Board gap 후속 분석

대상: truck_dashcam `BSD_F606/Front_Truck`. 하이퍼파라미터(lr0) 튜닝은 PC-A 에서 별도 진행 중이며 이 절의
범위가 아니다.

### 기준 모델과 평가 조건

- 기준 모델: `v02_symmetric_scratch` (symmetric_yolov8n, from-scratch, SiLU, SGD lr0 0.01, 500 epoch, best
  epoch 485, 학습 중 val mAP50 0.790 / mAP50-95 0.566)
- 보드 `.bin`: 보정 720장 stage4 레시피 `df076799` — 이 저장소에서 바이트 재현 확인
  (`novatek/convert/front_truck_stage4repro_v02_symmetric_scratch_20261001/`)
- 평가: `customEval`, 릴리스 v01 `exp.yaml`(ROI `[320,0,1280,1024]`, GT 50,371), conf 0.001.
  보드는 1280x1024 입력·임계값 0.001 빌드 덤프
- A = 기존 PC 후처리(`basic_postproc_v8`). B = 보드식 후처리 근사(`projects/metrics/board_postproc.py`:
  bus·car·truck 통합 NMS, IoU >= 0.3 또는 IoM >= 0.6, 묶음별 상위 256)

### A / B / 실제 Board

| run | A AP50 | B AP50 | A→B | Board AP50 | B→Board |
|---|---:|---:|---:|---:|---:|
| v02_symmetric_scratch | 0.738 | 0.723 | −0.015 | 0.713 | −0.010 |
| v02_symmetric_bg | 0.735 | 0.719 | −0.016 | 0.707 | −0.012 |
| v02_symmetric | 0.732 | 0.717 | −0.014 | 0.709 | −0.008 |
| v01_repro3 | 0.731 | 0.715 | −0.016 | 0.705 | −0.010 |
| v01 릴리스 ONNX (배포 `.bin`) | 0.705 | 0.689 | −0.016 | 0.672 | −0.016 |

- A·B 의 모델 순위는 같고, 실제 Board 순위와도 거의 같다(bg·v02_symmetric 0.002 차이만 뒤바뀜).
  B 는 Board 절대값 예측에는 A 보다 낫지만 모델 선택 순위는 바꾸지 않는다.
- v01 릴리스 기록의 Board 0.704 는 같은 조건(1280x1024)으로 재현되지 않는다(재측정 0.672).
- **traffic_light 손실이 가장 크다.** scratch 기준 B→Board 클래스별 gap: TL −0.028, 나머지 −0.005~−0.009.

### YUV420 입력 영향 (학습 없음)

1091장 실측 보드 NPU 입력(`board_npu_input.bin`)은 JPEG 내부 YCbCr 4:2:0 과 사실상 같았다(최대 차이 1).
BT.601 full-range + 2x2 평균 색 축소로 근사(`projects/metrics/board_preproc.py`)해 같은 ONNX 에 입력만 바꿨다.

| scratch | AP50 | AP50-95 | TL AP50 |
|---|---:|---:|---:|
| Original + B | 0.7230 | 0.4696 | 0.677 |
| YUV420 + B | 0.7229 | 0.4694 | 0.671 (−0.007) |

전체 영향은 없고 TL 만 −0.007 — B→Board TL gap(−0.028)의 약 1/4. YUV420 은 주요 원인이 아니다.

### 효과 없음으로 확인한 것 (반복하지 않는다)

| 시도 | 결과 |
|---|---|
| YUV420 강건성 미세조정 (`v02_symmetric_scratchft_yuv{000,025,050,100}`, scratch best.pt 에서 lr0 1e-4·100 epoch) | YUV420 입력 시 TL 하락이 control −0.005 → yuv100 −0.004. 잡음 수준. 변환·Board 생략 |
| 추가 미세조정 자체 (yuv000 control) | val mAP50 0.789 — 출발점 0.790 과 같음 |
| ReLU (`v02_symmetric_scratch_relu`) | ONNX A 0.729 / B 0.713 (SiLU 0.738 / 0.723), TL B 0.646 (−0.031). Board 미측정 |
| 보정 이미지 변경 | v01 배포 `.bin` 0.672 대 720장 재컴파일 0.674. scratch 500장 원본·ROI 보정은 시뮬 cosine 은 개선(0.977→0.988)됐으나 Board(하한 0.116 빌드) 차이 ±0.002 |

### 한계

- B 후처리는 보드 소스(`post_proc_baseline`/`nms_proc_baseline`)를 따른 **근사**다. 보드 int16 출력 2장에
  적용해 보드 덤프와 비교했을 때 검출의 약 75% 만 일치했고, 그 텐서가 보드 출력인지 시뮬 출력인지는 미확인이다.
- YUV420 근사의 색 복원(4:2:0 → 4:4:4)은 최근접 복제로 가정했다(NUE2 방식 미확인).

### 다음 조사 후보 (TL 잔여 손실 약 −0.02)

1. TL confidence score 분포 — B 대 Board
2. 같은 GT 객체 기준 ONNX ↔ Board confidence 변화
3. TL bbox/regression 위치 변화
4. INT8 양자화 — cls 출력 텐서 고정밀(`spec_demarc_tensors`) 지정 시 시뮬·Board 비교
5. 전처리/디코더 — PC 사전 crop 640x512 입력 실행에서 car·truck·bus 는 약 0.02 회복, TL 은 하락
   (하한 0.1218 빌드, `v02_symmetric_scratch/eval/board_precrop640/`)

### 운영 메모

- 학습 GPU 는 `-d <번호>` 로만 지정한다. ultralytics 가 `-d` 값으로 `CUDA_VISIBLE_DEVICES` 를 덮어써
  `-e CUDA_VISIBLE_DEVICES=1 … -d 0` 은 GPU0 에서 돈다(2026-10-03 queue 에서 실제 발생).
- 실험용 코드는 새 파일로만 추가했다: `projects/metrics/board_{postproc,preproc}.py`,
  `projects/custom_utils/yuv420.py`, `projects/experiments/yuv420_robustness.py`,
  `tools/train_symmetric_yuv420.py`, `tests/test_yuv420_robustness.py`.

## 2026-10-12 v01 Board 0.704 의 출처 — 릴리스 ONNX 가 아니라 `best.pt` 계열

대상: truck_dashcam `BSD_F606/Front_Truck` v01. 릴리스 기록은 ONNX 0.697 · Board 0.704 로 Board 가 ONNX 보다
높다. 같은 조건의 다른 측정은 모두 Board < ONNX 였으므로 출처를 확인했다.

### 재계산 — 기록 수치 자체는 재현된다

평가: `tools/customEval.py -w configs/releases/BSD_F606/Front_Truck/v01/exp.yaml` (ROI `[320,0,1280,1024]`,
GT 50,371, conf 0.01, IoU 0.45, 같은 `cat_map`).

| 입력 | md5 | 기록 | 재계산 | 결과 |
|---|---|---:|---:|---|
| NAS `v01/network/260706_BSD-F606A_Front_v01.onnx` | `d623cbdc…` | 0.697 | 0.697 | `v01/eval/onnx_recheck_20261012/` |
| NAS `v01/analysis/iputest/output.json` (2026-07-07) | `9e61d489…` | 0.704 | 0.7045 | `v01/eval/board_recheck_20261012/` |

두 파일 모두 저장소 `configs/releases/BSD_F606/Front_Truck/v01/` 사본과 md5 가 같다. 클래스별 AP 차는 ONNX
traffic_light 0.617 → 0.622 하나뿐이고, 9-28 저장소 평가(`v01/eval/onnx`)와는 검출 107,044 건까지 같다 —
7월 당시 평가 도구와의 차이로 본다. Board 쪽 score 하한은 7월 덤프·10-01 재실행·v02 덤프 모두 약 0.001 로 같았다.

### 검출 일치도 — 어느 ONNX 에서 나온 덤프인가

방법: 보드 덤프의 score >= 0.3 검출마다 같은 이미지·클래스에서 IoU >= 0.7 인 ONNX 검출(conf 0.01)이 있는 비율과
score 차 중앙값. SDK 시뮬레이터는 기존 `.bin` 을 입력으로 받는 래퍼 경로가 없어 쓰지 않았다.

| 보드 덤프 | 릴리스 ONNX `d623…` | `best.pt` 재export (`v01_reexport`) | v02 ONNX |
|---|---|---|---|
| v01 7월 덤프 (0.704) | 0.927 / 0.053 | **0.975 / 0.023** | 0.929 / 0.031 |
| v01 10-01 재실행 `board_stage4_shipped` (0.672) | **0.977 / 0.036** | 0.934 / 0.077 | 0.915 / 0.079 |
| v02 `board_stage4` (0.713, 대조군) | 0.900 / 0.059 | 0.925 / 0.036 | **0.979 / 0.029** |

값은 일치율 / |Δscore| 중앙값. 대조군이 자기 ONNX 와 0.979 이므로 같은 모델이면 0.97 대가 나온다.

### 결론

- **7월 덤프(0.704)는 `best.pt` 계열 `.bin` 의 결과다.** 릴리스 ONNX(`d623…`)와는 다른 모델 수준(0.927)으로만 맞는다.
  릴리스 기록은 서로 다른 가중치의 ONNX 결과와 Board 결과를 한 표에 적었다.
- 10-01 재실행은 릴리스 ONNX 와 맞는다. 이 실행의 `.bin` 은 이름(`release_v01_1280x1024`)으로만 배포본이라 기록돼
  있고 해시는 남아 있지 않다. NAS `v01/network/260706_BSD-F606A_Front_v01.bin` 은 md5 `619b83d9…`.
- 짝을 맞추면 모두 Board < ONNX 다.

| 모델 | ONNX | Board | 차 |
|---|---:|---:|---:|
| v01 릴리스 ONNX / 배포 `.bin` | 0.697 | 0.672 | −0.025 |
| v01 `best.pt` 계열 | 0.722 | 0.704 | −0.018 |
| v02 `v02_symmetric_scratch` | 0.731 | 0.713 | −0.018 |

### 미검증 갱신

- 위 "미검증" 첫 항목(`.bin` 이 v01 릴리스 ONNX 에서 변환됐는지)은 검출 일치도로 **배포 `.bin` = 릴리스 ONNX 계열**
  로 정황이 강해졌다. 10-01 재실행 `.bin` 의 해시가 NAS `.bin` 과 같은지는 여전히 미확인이다.
- 7월 덤프를 만든 `.bin` 파일 자체는 찾지 못했다. 실제 출하된 것이 어느 쪽인지 확인이 필요하다.
