# BSD_F606 Rear_Truck — ONNX vs NVT(NPU) 양자화 손실 실측 + v01 재현 모델 비교

- **작성**: 2026-08-06 (KST)
- **base commit**: `66a2698` (`fix: read network input size from resize config, not pre-resize camera resolution`)
- **대상**: `configs/experiments/BSD_F606/Rear_Truck/v01` (DAIR, 320x320), 대조군 `ultralytics_projects-main` 체크아웃의 production 모델
- **목적**: 사용자가 던진 두 질문에 답한다 —
  1. ONNX(float) → NVT(NPU 양자화) 사이의 손실이 얼마나 되는지, 그리고 그 손실이 모델마다 비슷한지
  2. 지금 이 repo(통합 파이프라인)에서 재현 학습한 모델과, 이전에 통합 안 된 환경(`ultralytics_projects-main`)에서 만든 production 모델의 성능이 얼마나 다른지

> **⚠ Target 개정 공지 (2026-08-19)**: 이 문서 본문(§"최종 결론" 포함)의 "10% 이내 달성"·
> "목표 달성" 서술은 **당시(2026-08-12) target 기준**이다. acceptance target이 이후
> `≤1%`로 상향됐고(`docs/guides/rear_truck_train_validate_nvt_pipeline.md` §13,
> `docs/decisions.md` 해당 D-번호), 새 target 기준으로는 이 문서에서 측정된 5개 모델
> 전부(최선인 `v01_symmetric` -7.1% 포함) 미달이다. 이 문서 본문은 조사 당시 기록을 그대로
> 보존하고 있으므로 값 자체는 고치지 않았다 — 현재 유효한 target과의 비교는 항상 위 두 문서를
> 기준으로 판단할 것.

---

## 질문 1 — 양자화 손실이 얼마나 되었는가, 이전과 비슷한가

**같은 6,152장 GT, 보드와 동일한 dual-threshold NMS로 맞춰서 비교** (NMS 알고리즘 차이는 배제한 순수 양자화 손실).

> **왜 6,152장인가 (2026-08-12 추가)**: Rear_Truck 원본 validation split은 7,894장이다. 이 중
> 1,742장은 physical NVT Board에서 처리되지 않아 board inference 결과를 얻지 못했다(원인은
> 확인되지 않았다 — 추측하지 않는다). 남은 `7,894 - 1,742 = 6,152`장이 Float ONNX와 Board
> 양쪽에서 결과를 얻을 수 있는 공통 집합이며, 임의로 sampling한 subset이 아니다. 아래 표의
> quantization loss는 이 6,152장 Float ONNX ↔ 동일 6,152장 Board 사이에서만 계산한 것이고,
> §2(전체 7,894장 기준 Internal Float validation) 수치와 board 결과를 직접 비교해서 loss를
> 계산하면 안 된다.

| 모델 | Float ONNX mAP@0.5:0.95 | 보드(NVT) mAP@0.5:0.95 | 절대 손실 | 상대 손실 |
|---|---|---|---|---|
| **Production** (`ultralytics_projects-main`, `best_REAR_320x320_DAIR_postprocess.onnx`) | 0.521 | 0.435 | **-0.086** | **-16.5%** |
| **v01 재현** (이 repo, `runs/detect/BSD_F606/Rear_Truck/v01/weights/best.onnx`) | 0.543 | 0.363 | **-0.180** | **-33.1%** |

**결론: 아니요, 비슷하지 않습니다.** v01의 양자화 손실이 production의 **약 2배**입니다. "같은 아키텍처니까 비슷하게 손실 날 것"이라는 가정은 깨졌습니다.

`docs/specs/qat_mixed_precision.md`가 자체적으로 정해둔 기준(">2% 드랍이면 post-training calibration으로 부족, 풀 mixed-precision QAT 필요")을 두 모델 다 크게 넘습니다 — v01은 더 심각합니다.

---

## 질문 2 — 통합 파이프라인(v01) vs 이전 비통합 환경(production) 모델 성능 차이

| 비교 | Production | v01 재현 | 차이 |
|---|---|---|---|
| **Float ONNX** mAP@0.5:0.95 (dual-NMS) | 0.521 | **0.543** | **+0.022 (+4.2%)** |
| **보드(NVT)** mAP@0.5:0.95 | 0.435 | **0.363** | **-0.072 (-16.6%)** |

**결론: 비대칭적입니다.**
- **Float 기준으로는 v01(이 repo 재현 학습)이 오히려 더 좋습니다** — `exp.yaml`에 `status: unvalidated`, `GA 미실시`라고 적혀 있는데도 production보다 +4.2% 높습니다. **학습 파이프라인 자체는 문제가 없습니다.**
- **보드(양자화 후) 기준으로는 v01이 훨씬 나쁩니다** — -16.6%.
- 즉 문제는 학습이 아니라 **NPU 변환 설정**에 있습니다. 지금 `rear_truck_v01_320`에 쓰인 calibration(724장)·`spec_demarc_tensors`는 production 모델(`rear_dair_pp_320`) 기준으로 튜닝된 걸 그대로 재사용한 것이라, v01의 실제 weight/activation 분포와 안 맞는 것으로 보입니다.

### 근거 — 시뮬레이터 cosine 비교 (같은 이미지 `0126.jpg`, 같은 config)

Production은 backbone(`model.0`~`model.9`)이 거의 무손실인데, v01은 **`model.2`, `model.4`의 첫 C2f 블록부터** 이미 눈에 띄게 나쁩니다.

| 레이어 | Production | v01 | 차이 |
|---|---|---|---|
| `Conv__model.2_m.0_cv2` | 0.996 | 0.966 | -0.031 |
| `Mul__model.2_m.0_cv2_act` | 0.998 | 0.962 | -0.036 |
| `Conv__model.4_m.0_cv1` | 0.992 | 0.968 | -0.024 |
| `Mul__model.4_m.1_cv1_act` | 0.979 | 0.950 | -0.029 |

| `[OUT]` 텐서 | Production | v01 | 차이 |
|---|---|---|---|
| DFL Softmax | 0.9630 | 0.9185 | -0.045 |
| Sigmoid `[OUT]` | 0.9815 | 0.9423 | -0.039 |
| dfl_conv `[OUT]` | 0.9888 | 0.9719 | -0.017 |

---

## 부록 — v01 calibration 튜닝 시도 (둘 다 실패)

`rear_dair_pp_320`(production)에서 효과가 없었던 `spec_demarc_tensors` 확장과는 별개로, v01의 backbone-early 손실을 잡아보려고 `precision/calib_mode`를 바꿔봤다 (0126.jpg 기준):

| calib_mode | Sigmoid `[OUT]` | dfl_conv `[OUT]` | Softmax(DFL) |
|---|---|---|---|
| **4 = PERCENTILE(0.999)** — 기존, 최선 | **0.9423** | **0.9719** | **0.9185** |
| 0 = MAX | 0.7638 | 0.9189 | 0.8251 |
| 2 = KLD | 0.7127 | 0.9178 | 0.8287 |

MAX·KLD 둘 다 PERCENTILE보다 훨씬 나쁘다 (특히 분류 branch가 급락). MAX는 outlier 하나에 scale 전체가 끌려가는 것으로 보이고, KLD도 이 모델 분포엔 안 맞는다. **결론: calib_mode 를 바꾸는 방향은 틀렸다** — 원래 PERCENTILE(0.999)이 셋 중 최선이며, 현재 `rear_truck_v01_320/nvtai_tool/config/REAR_TRUCK_V01_320/cnn25/gen_config.txt` 는 이 상태(baseline)로 복구되어 있다.

## 2026-08-10~11 추가 조사

위 "열린 질문"의 각 후보를 실제로 시도한 기록이다. base commit 이후 `66a2698`~`f661751` 사이 커밋들과 이 문서 작성 이후의 로컬 미커밋 변경(`dair.py`, `configs/experiments/BSD_F606/Rear_Truck/v02/`)을 포함한다.

### 1. calibration 이미지 재구성 (열린 질문 #2)

기존 calibration(500장은 production용 재사용, 724장은 `new_1T_REAR`)을 까보니 724장은 **영상 7개에서 0.5초 간격으로 뽑은 연속 프레임** — 장수만 많고 실제 다양성은 매우 낮았다.

`/mnt/NAS_1/video`의 주간 26개·야간 21개 영상(1920x1080, 실제 배포 도메인)에서 영상당 20장씩 균등 샘플링 → **940장**으로 재구성 (`novatek/convert/rear_truck_v01_320/nvtai_tool/input/data/REAR_TRUCK_V01_320/video_calib_jpg/`).

| calibration | cls `[OUT]` cosine | dfl `[OUT]` cosine |
|---|---|---|
| 500장(production 재사용) + 3텐서 | 0.9427 | 0.9691 |
| 724장(연속프레임) + 9텐서(historical) | 0.9423 | 0.9719 |
| **940장(다양 샘플링) + 9텐서(historical)** | 0.9486 | 0.9672 |

다양성을 개선해도 개선폭은 미미했다 — calibration 다양성은 이번 gap의 주 원인이 아니었다.

### 2. calib_mode 외 추가 파라미터 (전부 실패/역효과)

| 시도 | cls cosine | dfl cosine | 결과 |
|---|---|---|---|
| `precision/percentile` 0.995 (열린 질문 #1) | 0.9178 | 0.9630 | **악화** → 0.999 유지 |
| Front 모델 실 production config 그대로 이식 (CLE off, `cal_refine_iter=0`, `bnscale_en=0`) | 0.9323 | 0.9533 | **악화** |
| 538칩 mixed-precision sensitivity 분석 (`compiler.V30`, `--mp-method 0/1`) | - | - | **정보 획득 실패** — 원본 538 config를 못 구해 우리 530 config를 이식했으나 `sorted_sensitivity.json`이 생성되지 않음. `process_bit_info.txt`의 raw bitdepth 표는 최종 출력 2곳만 16bit로 표시해 유용한 신호 없음 |

538 분석용 wrapper(`novatek/convert/run_compiler_538.sh`)는 남겨뒀지만 실제 유효한 sensitivity 데이터는 못 얻었다.

### 3. 돌파구 — 실측 기반 `spec_demarc_tensors` (model.2/4)

538 분석 대신, 이미 갖고 있는 시뮬레이터 `layer_diff_*.txt`(모든 171개 레이어의 cosine)를 직접 정렬해서 v01 자신의 최약 레이어를 찾았다. `model.2`, `model.4`의 C2f 블록(위 §질문2 표와 동일 지점)이 가장 나빴다.

**변경**: 기존 출력 3텐서(`dfl Softmax`, `Sigmoid`, `dfl conv`)에, 실측 최약 6개를 추가한 9텐서로 교체:
```
Conv_/model.2/m.0/cv2/conv/Conv_output_0_Y, Mul_/model.2/m.0/cv2/act/Mul_output_0_Y,
Mul_/model.4/m.0/cv1/act/Mul_output_0_Y, Conv_/model.4/m.1/cv1/conv/Conv_output_0_Y,
Mul_/model.4/m.1/cv1/act/Mul_output_0_Y, Mul_/model.4/cv2/act/Mul_output_0_Y
```

| | cls `[OUT]` | dfl `[OUT]` |
|---|---|---|
| 이전 최선(9텐서 historical, 940장) | 0.9486 | 0.9672 |
| **model.2/4 실측 기반 9텐서** | **0.9869** | **0.9809** |

**6,152장 전체 board mAP로 재확인** (`runs/_eval_scratch/configs/board/board_v01_9tensor_cfg.yaml`,
2026-08-12 재정리 후 경로 — 원래 위치는 `runs/_eval_scratch/board_v01_9tensor_cfg.yaml`이었다):

| | float mAP50-95 | 보드 mAP50-95 | 손실 |
|---|---|---|---|
| production | 0.521 | 0.435 | -16.5% |
| v01 (이전, 9텐서 historical) | 0.543 | 0.368 | -32.2% |
| **v01 (model.2/4 실측 9텐서)** | **0.543** | **0.435** | **-19.9%** |

**v01의 board mAP가 production과 정확히 동일해졌다.** 계산 방식(threshold 0.45 적용, class merge 여부) 때문이 아님도 별도로 확인함 — threshold를 0.45로 올려도 상대 격차는 그대로(-16.6%→-19.9%, 오히려 소폭 확대).

### 4. 그러나 — 자원 비용 2배, 근본 해결 아님

`gentool/profile.txt` 실측(둘 다 940장 calibration 기준):

| | spec_demarc_tensors | 추론시간 | DRAM 대역폭 | 모델 크기 |
|---|---|---|---|---|
| production (9개, **model.18/19/21/22** = 헤드 근처) | 9개 | 6.25ms | 35.0MB | 4.70MB |
| v01 3텐서(baseline) | 3개 | 6.37ms | 34.9MB | 4.71MB |
| **v01 model.2/4 9텐서** | 9개 | **12.25ms** | **78.5MB** | **9.39MB** |

**같은 개수(9개)의 고정밀 텐서인데 비용이 2배 다르다** — production이 보호하는 텐서는 헤드 근처(후반부)라 16bit 전파가 그 뒤 몇 레이어로 끝나는데, 우리가 보호하는 model.2/4는 backbone 초반부라 그 뒤 거의 전체 네트워크가 16bit 경로로 확장된다. Rear_Truck의 실사용 목표가 10~15fps로, 이는 하나의 NPU를 Front/Side/DMS 등 다른 모델과 나눠 쓰는 데서 오는 예산이라 — **이 2배 비용은 가볍게 넘길 사안이 아니다.**

**결론: v01의 board mAP를 production과 동일하게 만들긴 했지만, weight/activation 자체의 양자화 민감도를 고친 게 아니라 하드웨어 자원을 2배 써서 보완한 것이다.** 근본 원인은 미해결.

### 5. 코드 레벨 원인 조사 — `ultralytics_projects-main`(production 학습 환경) vs 현재 repo

production은 `ultralytics_projects-main`(별도 체크아웃)에서 학습됐다. 학습에 관여하는 파일들을 전부 diff:

| 파일 | 실질 차이 |
|---|---|
| `loss.py`, `augment.py`(rear 경로), `symmetric_yolov8.yaml`(아키텍처), `metric.py`, `cfg.py`, `dair.py`(val/model) | **없음** — whitespace/import 정리, docstring 추가뿐 |
| `tal.py` | `bg_idx` 버그 — production은 원래 `self.bg_idx = num_classes`가 있어 문제없었는데, 이 repo로 옮기며 그 줄이 삭제됨. 오늘 세션 초반에 고친 것과 동일 건(`a713f73`) — 수정 후 동작은 production과 동일해짐 |
| **`train/dair.py`** | **`TensorSpeakerOverlayAug`(대시보드/스피커 오버레이, DAIR augmented-view용)** — production은 **주석 처리(비활성)**, v01은 **활성 상태로 학습됨**. 유일하게 실질적으로 다른 학습 코드 |

`simota_dair.py`(170줄 diff)는 우리 config(`assigner` 기본값 `tal`, `dair_method_mode: or`)가 애초에 그 경로를 안 타서 무관.

### 6. 진행 중 — v02 ablation 학습

`TensorSpeakerOverlayAug`를 주석 처리(production과 동일하게)하고, 그 외 하이퍼파라미터는 v01과 완전히 동일하게 재학습 시작.

- 코드: `projects/src/models/yolo/detect/train/dair.py` (미커밋)
- config: `configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml` (미커밋)
- 실행: `python3 tools/train_custom_trainer.py -w configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml -d 0`, GPU 0, `docker compose exec -d`로 완전 분리 실행
- 상태: 진행 중 (2026-08-11 기준 134/500 epoch, ~13.7시간 경과) — ※ 당시 시점 기록. v02는
  이후 epoch333에서 patience로 정상 종료됐다. 최종 결과는 문서 최하단 "최종 결론" 참조.

**검증 계획**: v02 학습 완료 → float 성능 확인 → 3텐서(baseline, 저비용) 그대로 NPU 변환·board mAP 측정. production 수준(-16.5%대) 손실이 **자원 비용 증가 없이** 나오면 이 augmentation이 root cause였다는 강한 증거. 여전히 -30%대면 이 가설은 기각, QAT 재검토로.

## 다음 단계 (갱신, 2026-08-11)

※ **당시 시점(2026-08-11) 기록이다.** 아래 3개 항목 모두 이후 실측으로 해소됐다 — v02 완료,
float/board mAP 확보, 가설 지지 확인까지 끝났다. 최종 결과와 결론은 문서 최하단
"최종 결론 (2026-08-12)"을 참조할 것. 이 섹션은 당시의 계획을 그대로 보존한다.

1. **v02 학습 완료 대기** — 완료되면 float mAP, board mAP(3텐서 baseline)를 v01/production과 나란히 비교
2. v02가 가설을 반박하면 → D-010(QAT 보류) 재검토. 두 모델 다 mAP drop > 2% 기준을 크게 넘는다는 게 이미 근거로 확보됨
3. v02가 가설을 지지하면 → `TensorSpeakerOverlayAug` 비활성을 dair 학습 기본값으로 반영할지 논의 (다른 제품/카메라 학습에도 영향 범위 있음)

## 관련 파일

- 평가 config/스크립트(2026-08-12 `_eval_scratch` 재정리 후 경로): `runs/_eval_scratch/configs/board/{board_cfg,board_v01_cfg,board_v01_940calib_cfg,board_v01_9tensor_cfg,board_thr045_cfg,board_v01_940calib_thr045_cfg}.yaml`, `runs/_eval_scratch/configs/onnx/{onnx_float_cfg,onnx_float_dualnms_cfg,onnx_v01_dualnms_cfg}.yaml`, `runs/_eval_scratch/tools/merge_board_dump.py`
- NPU 변환: `novatek/convert/rear_dair_pp_320/` (production), `novatek/convert/rear_truck_v01_320/` (v01, model.2/4 9텐서 + 940장 calibration 현재 상태), `novatek/convert/rear_truck_v01_320_frontstyle/` (Front 이식 실험), `novatek/convert/rear_truck_v01_320_538analysis_538/` (538 분석, 유효 데이터 없음), `novatek/convert/run_compiler_538.sh`
- 평가 로그: `runs/_eval_scratch/results/{board,onnx_float,onnx_float_dualnms,board_v01,onnx_v01_dualnms,board_v01_940calib,board_v01_9tensor,board_thr045,board_v01_940calib_thr045}/*/eval.log`
- v02 ablation: `configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml`, `projects/src/models/yolo/detect/train/dair.py`, 학습 로그 `runs/detect/train_v02.log`

## spec_demarc_tensors 최소 subset 탐색 (2026-08-11, v02와 동시 진행 중 일시 중단)

model.2/4 실측 9텐서(3 historical + 6 measured)로 board mAP를 production 수준(0.435)까지 회복시켰지만,
latency/DRAM/model size가 baseline 대비 약 2배(12.25ms/78.5MB/9.39MB vs 6.37ms/34.9MB/4.71MB)로 늘어난
상태다. 목적은 이 6개 measured 텐서(model.2 C2f 2개 + model.4 C2f 4개) 중 실제로 accuracy 회복에 필요한
최소 subset을 찾아 자원 비용을 줄이는 것.

### 1. 자동화 인프라

`novatek/convert/rear_truck_v01_320_ablation/`에 원본 `rear_truck_v01_320/`을 건드리지 않는 격리된
실험 구조를 만들었다:

- `experiment_configs/tensors.json` — H1-H3(historical output 3텐서, 변경 안 함) / T1-T6(model.2/4
  measured 6텐서) 이름을 `gen_config.txt`의 현재 `spec_demarc_tensors` 라인에서 그대로 확인해 기록.
  H1=`Softmax_/model.22/dfl/Softmax_output_0_Y`, H2=`Sigmoid_output0_Y`,
  H3=`Conv_/model.22/dfl/conv/Conv_output_0_Y`. T1=`Conv_/model.2/m.0/cv2/conv/...`,
  T2=`Mul_/model.2/m.0/cv2/act/...`, T3=`Mul_/model.4/m.0/cv1/act/...`,
  T4=`Conv_/model.4/m.1/cv1/conv/...`, T5=`Mul_/model.4/m.1/cv1/act/...`, T6=`Mul_/model.4/cv2/act/...`
  (본문 §3 목록과 동일).
- `experiment_configs/round1_backward_elim.json` — T1~T6 중 하나씩 제거한 6개 후보(+ 재컴파일 없이
  기존 결과를 재사용하는 `full9`).
- `experiment_configs/round2_coarse_AB.json` — model.2 대 model.4 중 어느 쪽이 accuracy/cost에 더
  기여하는지 빠르게 보기 위한 coarse 2종: A=`H1-H3+T3-T6`(model.2 전체 제거), B=`H1-H3+T1-T2`(model.4
  전체 제거).
- `experiment_configs/all_pending.json` — 위 9개를 합친 목록(한 커맨드로 실행 가능하게).
- `run_quant_subset_search.py` — 실험별로 `nvtai_tool/config/.../gen_config.txt`의
  `spec_demarc_tensors` 라인만 교체해서 별도 디렉터리에 쓰고(대용량 calibration/model 입력은
  `rear_truck_v01_320/nvtai_tool/input`을 symlink로 공유), `docker exec`로 npu-convert 컨테이너 안에서
  `run_compiler.sh`/`run_simulator.sh`를 실행한 뒤 `gentool/profile.txt`(latency/DRAM/model size)와
  `simtool/layer_diff.txt`(텐서별 cosine)를 파싱해 CSV에 누적한다. `--prep-only`(compile 없이 config만
  생성)와 `--force`(이미 있는 결과 재실행) 플래그가 있어 리소스 경쟁 중에도 준비 작업만 안전하게 할 수
  있음.

### 2. 실행을 일시 중단한 이유 — v02 학습과의 리소스 경쟁 실측

`drop_T1`(T1만 제거) 컴파일을 시작했더니, 원래(현재 9텐서 상태 컴파일 시) 약 19분 걸리던 컴파일이
**2시간 20분+**이 지나도 끝나지 않았다(`gentool/cle_nvt_model.onnx`/`legal`까지만 생성되고
`profile.txt` 미생성). 원인은 v02 학습(`dair-train` 컨테이너)이 CPU를 약 770~800% 쓰고 있어 32코어를
컴파일러(약 1100~1500%)와 나눠 쓰는 경쟁 — `vmstat` 기준 컨텍스트 스위치가 초당 100만 회 이상.

`runs/detect/BSD_F606/Rear_Truck/v02/results.csv`로 실측한 epoch당 소요 시간:

| epoch | 시간(초) | 비고 |
|---|---|---|
| 166→170 (컴파일 시작 전) | 368~370초/epoch | baseline |
| 170→171 | 443.5초 (+20%) | drop_T1 컴파일 시작(약 epoch170 종료 시점)과 겹침 |
| 171→172 | 488.6초 (+33%) | |
| 172→173 | 488.1초 (+32%) | 회복 기미 없이 지속 |

329 epoch가 남은 상태에서 이 슬로우다운이 계속되면 v02 완료가 약 9시간 늦어질 수 있다고 판단, 사용자
확인 후 **drop_T1 컴파일 프로세스를 kill**(`compiler.nvtai`, `run_compiler.sh`)하고 미완성 산출물
(`nvtai_tool/output`, `tmp`)을 정리했다. kill 직후 epoch174가 8.5~8.6it/s로 즉시 회복되는 것을
확인함 — v02 학습 자체나 GPU 설정은 전혀 건드리지 않았다.

**결정**: NVT subset 탐색의 실제 compile/simulate 실행은 v02 학습이 끝날 때까지 전면 중단한다. 그
전까지는 config 준비(`--prep-only`)와 기존 데이터 기반 분석만 진행.

### 3. 기존 layer_diff만으로 본 6개 텐서 중요도 (가설, 미확정 — 실제 제거 실험 전)

현재 9텐서(전체 보호) 상태의 `layer_diff.txt`(0126.jpg 기준)에서 T1~T6 자신의 cosine:

| 텐서 | 위치 | cosine (보호된 상태) |
|---|---|---|
| T2 | model.2/m.0/cv2/act | 0.960230 |
| T1 | model.2/m.0/cv2/conv | 0.966988 |
| T5 | model.4/m.1/cv1/act | 0.971573 |
| T3 | model.4/m.0/cv1/act | 0.976738 |
| T4 | model.4/m.1/cv1/conv | 0.978989 |
| T6 | model.4/cv2/act | 0.979592 |

**중요한 관찰**: 이 값들이 §질문2의 원래 unprotected 값(T1 0.966, T2 0.962)과 거의 동일하다 — 즉
16bit로 "보호"해도 그 텐서 **자기 자신**의 cosine은 거의 개선되지 않는다. 개선은 다운스트림
(`Add`/`Concat`으로 합류하는 지점, 그리고 최종 `[OUT]`)에서 나타난다 — 보호 지점 이후의 오차 재확산
(re-quantization으로 인한 오차 누적)을 막아주는 것이 실제 효과로 보인다. 또한 model.6/model.8처럼
**보호되지 않은** 다른 C2f 블록의 `m.0` 브랜치도 로컬 cosine이 비슷하게 나쁘다(예:
`Mul__model.8_m.0_cv1_act` = 0.956766) — 즉 "C2f m.0 브랜치의 act 노드가 로컬에서 나쁘다"는 것 자체는
model.2/4만의 특징이 아니라 이 네트워크 전반의 특징이다. model.2/4가 선택된 이유는 로컬 cosine이
아니라, §질문2에서 이미 확인했듯 **이 두 지점의 오차가 나머지 네트워크 전체(모델.5~22)로 전파되며
최종 cls/dfl cosine을 가장 크게 끌어내렸기 때문**이다.

이로부터 추론 가능한 것(단, 실제 제거 실험 없이는 확정 불가):

- **model.2(T1,T2)가 model.4(T3-T6)보다 그래프상 더 이르다** (layer ~10-19 vs ~38-73, 총 363
  layer 중). 보호 안 하면 오차가 전파될 후속 레이어 수가 더 많아 accuracy 영향이 더 크고, 반대로
  보호하면 16bit 경로가 그만큼 더 오래(더 많은 레이어) 전파되어 **비용도 model.4보다 더 크게
  기여할 가능성**이 높다 — 즉 model.2 쪽이 accuracy에도, latency/DRAM 비용에도 둘 다 더 큰 영향을
  준다는 가설.
- model.2 내에서는 T2(act, 0.960)가 T1(conv, 0.967)보다 로컬 cosine이 더 나쁘고, model.4 내에서는
  T5(0.972)가 나머지(T3,T4,T6 ≥0.977)보다 나쁘다 — 다운스트림 merge 지점 cosine도 model.2의
  `Add__model.2_m.0`(0.982)이 model.4의 동등 지점들(`Add__model.4_m.0` 0.995,
  `Add__model.4_m.1` 0.996)보다 뚜렷이 낮다. 이 역시 model.2 쪽이 더 "안 고쳐지는" 오차를 가지고
  있다는 신호.
- **위 가설을 실제로 가르는 것은 `round2_coarse_AB.json`의 A(model.2 전체 제거)/B(model.4 전체
  제거) 비교다** — 로컬 cosine만으로는 "제거했을 때 최종 cls/dfl이 얼마나 떨어지는지"를 알 수
  없으므로, v02 완료 후 리소스 경쟁 없이 이 2건부터 실행해 확인해야 한다.

### 4. v02 완료 후 재개 순서

※ **당시 시점(2026-08-11) 계획.** v02가 가설을 지지(-18.8%)하면서 아래 순서 중 2번(subset
탐색 본격 실행)은 실행하지 않았다. 이후 v01_symmetric이 저비용 3텐서 그대로 -7.1%를 달성해
model.2/4 subset 탐색의 실사용 후보 선정 우선순위는 더 낮아졌다 — 완전히 폐기하지는 않고 선택적
후속 연구로 남긴다. 최종 결론은 문서 최하단 참조.

1. v02 float mAP 확인, 3텐서(baseline) NVT 변환 + 동일 6,152장 board mAP 확인 (§"다음 단계" 항목1과
   동일) — augmentation root-cause 가설이 맞으면(-16%대로 회복) subset 탐색 자체의 필요성이 줄어듦.
2. v02가 v01처럼 -30%대로 깨지면(가설 반박) → 위에서 준비해둔
   `python3 novatek/convert/rear_truck_v01_320_ablation/run_quant_subset_search.py experiment_configs/round2_coarse_AB.json`
   부터 실행(2건, model.2 vs model.4 기여도 분리) → 그 결과로 `round1_backward_elim.json`의 나머지
   5개(`drop_T2`~`drop_T6`, `drop_T1`은 재실행 필요)를 우선순위대로 실행.
3. 매 컴파일 전후로 `runs/detect/BSD_F606/Rear_Truck/v02/results.csv`(또는 그 시점의 후속 학습)의
   epoch당 시간을 다시 확인 — 위 표의 368~370초 baseline보다 눈에 띄게(20%+) 느려지면 즉시 중단.

## v02 convergence 분석 + weight 분포 정적 분석 (2026-08-11, GPU/NVT 미사용)

drop_T1 kill 이후 NVT compile은 전면 중단한 상태에서, v02 학습 프로세스를 전혀 건드리지 않고
`results.csv`만 읽는 분석과 onnx weight 정적 분석(단일 스레드, `nice -n 19`)만 수행했다.

### 1. v01 convergence 궤적 (251 epoch, patience=15로 종료)

`configs/experiments/BSD_F606/Rear_Truck/v01/exp.yaml`은 `epochs: 500, patience: 15`인데 실제로는
**epoch251에서 종료**됐다 — best epoch=236(fitness=0.1·mAP50+0.9·mAP50-95 기준, mAP50-95=0.55756),
그 뒤 15epoch 동안 개선이 없어 조기 종료. 즉 v01도 500까지 가지 않았다.

epoch별 "최종 best(0.55756) 대비 도달률":

| epoch | mAP50-95 | 최종 best 대비 |
|---|---|---|
| 25 | 0.48406 | 86.82% |
| 50 | 0.51581 | 92.51% |
| 75 | 0.52888 | 94.86% |
| 100 | 0.53665 | 96.25% |
| 125 | 0.54263 | 97.32% |
| 150 | 0.54799 | 98.28% |
| **175** | **0.55093** | **98.81%** |
| 200 | 0.55430 | 99.42% |
| 225 | 0.55637 | 99.79% |
| 236 (best) | 0.55756 | 100.00% |
| 251 (종료) | 0.55731 | 99.96% |

best의 X% 최초 도달 epoch: 90%→epoch36, 95%→epoch77, 97%→epoch121, 98%→epoch144, 99%→epoch182,
99.5%→epoch206. 즉 v01은 epoch175(현재 v02와 같은 지점)에 이미 최종 성능의 98.81%에 도달해
있었고, 그 뒤 61epoch(175→236)를 더 돌려서 얻은 추가 이득은 절대값 +0.00663(상대 1.19%p)뿐이었다.

### 2. v01 vs v02 epoch-aligned 비교 (25epoch 간격)

| epoch | v01 train_loss_tot | v02 train_loss_tot | v01 P | v02 P | v01 R | v02 R | v01 mAP50 | v02 mAP50 | v01 mAP50-95 | v02 mAP50-95 | diff(v02-v01) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 2.94767 | 2.93729 | 0.7871 | 0.8031 | 0.6140 | 0.6021 | 0.71027 | 0.70920 | 0.48406 | 0.48405 | -0.00001 |
| 50 | 2.79141 | 2.78371 | 0.8297 | 0.8356 | 0.6382 | 0.6411 | 0.74198 | 0.74400 | 0.51581 | 0.51526 | -0.00055 |
| 75 | 2.71737 | 2.70759 | 0.8381 | 0.8438 | 0.6512 | 0.6547 | 0.75425 | 0.75663 | 0.52888 | 0.52841 | -0.00047 |
| 100 | 2.67335 | 2.66025 | 0.8473 | 0.8428 | 0.6560 | 0.6659 | 0.76141 | 0.76339 | 0.53665 | 0.53476 | -0.00189 |
| 125 | 2.64279 | 2.63080 | 0.8500 | 0.8519 | 0.6604 | 0.6647 | 0.76557 | 0.76665 | 0.54263 | 0.54090 | -0.00173 |
| 150 | 2.61471 | 2.59954 | 0.8513 | 0.8565 | 0.6620 | 0.6690 | 0.76960 | 0.77003 | 0.54799 | 0.54363 | -0.00436 |
| 175 | 2.59647 | 2.58181 | 0.8528 | 0.8556 | 0.6672 | 0.6705 | 0.77323 | 0.77272 | 0.55093 | 0.54622 | -0.00471 |

train loss는 v02가 v01보다 매 지점 살짝 낮은데(더 잘 fit), mAP50-95는 반대로 v02가 v01보다
꾸준히 낮고 그 격차가 epoch50 이후 커지는 추세(-0.0005→-0.0047)다. Precision/Recall은 둘 다
비슷하거나 v02가 약간 높다 — 즉 augmentation 제거가 train loss·P/R엔 (약간) 유리하지만 float
mAP50-95엔 지금까지는 오히려 약간 불리하게 보인다. 최근 20epoch(156-175) slope는 v01
+0.000116/epoch, v02 +0.000066/epoch — v02가 지금 v01보다 더 느리게 올라가는 중.

**v02 현재(epoch175)=0.54622는 v01의 최종 best(0.55756)의 97.97% 수준** — 아직 진행 중이므로
이는 하한선이다.

### 3. v02는 아직 plateau가 아님, 그러나 정체 조짐

v02 최근 10/20/50epoch 개선량은 각각 +0.00093 / +0.00172 / +0.00532 — 전부 양수라 아직 상승
중이지만, 같은 구간 v01(+0.00012 / +0.00027 / +0.00304 — 주의: v01은 이미 지나간 175-251 구간이라
숫자가 작음)과 직접 비교는 의미가 크지 않고, **같은 epoch 구간(156-175)의 slope 비교가 더
정확**한데 v02(+0.000066)가 v01(+0.000116)의 약 57% 수준이다. v01 precedent(같은 patience=15
config)로 보면 v02도 200~260 epoch 구간 어딘가에서 patience로 자동 종료될 가능성이 높다 — 500까지
갈 필요는 v01 사례상 낮아 보이지만, **patience가 있으므로 수동으로 멈출 필요는 없다.**

### 4. production vs v01 model.2/4 Conv weight 분포 (ONNX initializer, numpy만 사용)

`runs/_eval_scratch/reference/best_REAR_320x320_DAIR_postprocess.onnx`(production, 2026-08-12
재정리 후 경로) vs
`runs/detect/BSD_F606/Rear_Truck/v01/weights/best.onnx`(v01)에서 `model.2.*`/`model.4.*` conv
weight 10개씩을 비교(dair-train 컨테이너 안에서 `OMP_NUM_THREADS=1`, `nice -n 19`, 1초 이내 종료 —
v02 학습에 영향 없음).

T1~T6가 속한 conv weight의 absmax 비율(v01/production, 클수록 v01이 더 넓게/outlier에 가깝게
퍼져 있음):

| 텐서 | weight | production absmax | v01 absmax | 비율 | production std | v01 std | 비율 |
|---|---|---|---|---|---|---|---|
| T1,T2 (model.2/m.0/cv2) | `model.2.m.0.cv2.conv.weight` | 1.508 | 4.126 | **2.74x** | 0.151 | 0.360 | **2.39x** |
| T4,T5 (model.4/m.1/cv1) | `model.4.m.1.cv1.conv.weight` | 0.432 | 0.819 | 1.90x | 0.066 | 0.090 | 1.37x |
| T3 (model.4/m.0/cv1) | `model.4.m.0.cv1.conv.weight` | 1.268 | 1.739 | 1.37x | 0.117 | 0.169 | 1.45x |
| T6 (model.4/cv2) | `model.4.cv2.conv.weight` | 2.073 | 2.749 | 1.33x | 0.127 | 0.146 | 1.15x |

channel별 outlier(median의 5배 초과)도 v01 쪽에 3곳(`model.2.cv2`, `model.4.cv2`,
`model.4.m.0.cv2`)이 있고 production은 0곳 — 특히 `model.4.m.0.cv2.conv.weight`는
max/median 채널 비율이 production 2.05 → v01 5.70(**2.78x**)로 뚜렷한 채널 outlier가 새로
생겼다(단, 이 conv 자체는 spec_demarc 6개 중엔 없음 — m.0.cv2의 output 쪽, T1/T2가 붙은 conv와는
다른 conv).

**결론**: T1/T2(model.2/m.0/cv2)가 6개 중 weight 분포상으로도 production 대비 가장 크게
벌어져 있다(absmax 2.74x, std 2.39x) — §3의 layer_diff 기반 가설(model.2가 model.4보다 더
critical)과 독립적인 근거로 일치한다. T4/T5가 두 번째로 크고, T3·T6는 상대적으로 덜 벌어져 있어
제거 우선순위가 가장 높은 후보(=먼저 빼볼 후보)로 보인다 — 단, 실제 검증은 §3에서 정리한 대로
round2 coarse A/B + backward elimination 실험(v02 완료 후)으로 확정해야 한다.

### 5. v02 mid-training(epoch176) weight snapshot과 3-way 비교

TensorSpeakerOverlayAug 제거가 weight 분포를 production 쪽으로 되돌리는지 확인하기 위해 v02의
현재 best 스냅샷(epoch176, `ema` — 이 시점엔 `ckpt['model']`이 None이라 `ckpt['ema']` 사용)을
읽었다. **안전 절차**: `best.pt`가 두 번의 `stat` 사이(2초 간격) mtime/size가 그대로임을 확인한
뒤 `cp --preserve=all`로 스크래치 복사본을 만들고 md5 일치를 검증하고서 그 복사본만 읽었다(원본
`.pt`/학습 프로세스는 전혀 열지 않음). 로드는 `dair-train` 컨테이너 안에서
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`/`nice -n 19`/`taskset -c 0`/CPU-only로 실행,
전후로 `results.csv` epoch 시간(175→176→177: 373.0s, 371.9s — baseline 그대로)과 학습 로그의
it/s(8.2~8.7it/s, 정상)를 확인해 영향이 없었음을 검증했다. 분석 후 스크래치 스크립트/복사본은
삭제함.

**중요한 주의사항**: v01의 수치는 **학습이 끝나고 patience로 종료된 최종(epoch236/251, LR
anneal 완료) 상태**인데 v02는 **아직 epoch176, 500-epoch 스케줄 기준 LR이 덜 죹아든
mid-training 상태**다. weight의 절대 스케일(absmax·std·mean)은 학습이 진행될수록/LR이
줄어들수록 자연스럽게 안정화(대체로 축소)되는 경향이 있어, "v02(mid-training) vs
v01(fully-converged)"의 절대 스케일 비교는 augmentation 효과와 training-stage 효과가 뒤섞여
있다 — **이 비교만으로 절대 스케일 지표를 root-cause 증거로 쓰지 않는다.** 반대로 채널
outlier 개수·max/median 채널 비율처럼 "그 텐서 내에서 어느 채널이 유별나게 큰가"를 보는
상대적/분포-모양 지표는 학습 스케일 자체보다 덜 민감하므로 더 신뢰할 수 있는 신호로 본다.

| weight | 지표 | production | v01(최종) | v02(epoch176) | 비고 |
|---|---|---|---|---|---|
| `model.2.m.0.cv2.conv.weight` (T1/T2) | absmax | 1.508 | 4.126 | **2.807** | v01→production 방향으로 약 50% 이동 |
| | std | 0.151 | 0.360 | 0.402 | production 반대 방향으로 소폭 이동(-20%) — mid-training confound 가능 |
| | max/median 채널 ratio | 3.514 | 3.067 | **1.718** | production보다도 더 균등해짐(outlier 방향 아님) |
| | outlier 채널(>5x median) | 0 | 0 | 0 | 원래 outlier 없던 텐서 |
| `model.4.m.0.cv1.conv.weight` (T3) | absmax | 1.268 | 1.739 | 3.900 | v01보다도 더 커짐(스케일 지표, mid-training 영향 큼) |
| | max/median 채널 ratio | 2.063 | 2.078 | 1.448 | v01/production 둘보다 균등 |
| `model.4.m.1.cv1.conv.weight` (T4/T5) | absmax | 0.432 | 0.819 | 2.844 | v01보다도 더 커짐 |
| | max/median 채널 ratio | 1.480 | 1.986 | **1.610** | v01→production 방향으로 이동 |
| `model.4.cv2.conv.weight` (T6) | absmax | 2.072 | 2.748 | 6.453 | v01보다도 더 커짐 |
| | max/median 채널 ratio | 4.319 | 5.153 | 2.758 | production보다도 더 균등해짐 |
| | outlier 채널(>5x median) | 0 | **1** | **0** | **v01의 outlier 채널이 v02에서 사라짐** |
| `model.2.cv2.conv.weight` (outlier 발견분) | absmax | 4.041 | 8.159 | 4.930 | v01→production 방향 78% 이동 |
| | outlier 채널(>5x median) | 0 | **1** | **0** | **사라짐** |
| `model.4.m.0.cv2.conv.weight` (outlier 발견분) | absmax | 1.234 | 2.806 | 4.363 | v01보다도 더 커짐 |
| | max/median 채널 ratio | 2.052 | 5.698 | 1.852 | v01→production 방향, production보다도 균등 |
| | outlier 채널(>5x median) | 0 | **1** | **0** | **사라짐** |

**요약**:
- **채널 outlier 지표(가장 신뢰하는 지표)는 뚜렷하게 회복됐다** — v01에서 새로 생긴 outlier
  채널 3곳(`model.2.cv2`, `model.4.cv2`, `model.4.m.0.cv2`) 전부 v02에서 **완전히 사라져
  production과 동일(0곳)**해졌다. max/median 채널 비율도 대부분 v01보다 낮아졌고(더 균등),
  T1/T2(`model.2.m.0.cv2`)는 production보다도 더 균등해졌다.
- **절대 스케일(absmax/std)은 지표별로 방향이 갈린다** — T1/T2의 absmax는 production 방향으로
  절반쯯 이동했지만, T3/T4-T5/T6의 absmax·std는 v01보다도 더 커졌다. 다만 위에서 밝혔듯 이건
  v02가 아직 mid-training(epoch176, LR 안 죽음)이라는 confound가 강하게 껴 있어서, 지금 시점의
  절대 스케일 수치로 "더 나빠졌다"고 결론 내릴 수 없다 — v01의 epoch176 시점 스냅샷이 없어
  직접 대조가 불가능하다(v01은 best.pt/last.pt만 저장, 중간 epoch 체크포인트 없음).

**판단 (A/B/C 중, 2026-08-11 시점)**: **B — 일부만 복귀.** (※ 이후 board mAP 실측으로
해소됨 — 최종 결론 참조.) 채널 outlier·상대적 분포 모양 지표는 뚜렷하게
production 쪽으로 회복되어 augmentation이 root cause 중 하나일 가능성을 지지하지만, 절대
스케일 지표는 mid-training confound 때문에 지금 판단할 수 없다(v01보다 커진 것이 augmentation
때문인지 단순히 학습이 덜 끝나서인지 구분 불가). **NVT 결과가 나오기 전까지는 이 weight-level
근거만으로 root cause를 확정하지 않는다** — 결정적 검증은 v02 완료 후 §4의 체크리스트(float
mAP → 3텐서 baseline NVT 변환 → board mAP)다. 참고로 채널-outlier 회복이 실제로 유의미하다면,
v02가 완전히 수렴한 뒤(§2 분석상 epoch200~260 근처 예상) 같은 분석을 다시 돌려 절대 스케일도
안정화된 상태에서 재확인하는 것이 좋다 — mid-training confound가 사라진 상태의 비교가 될 것이다.

## v02 Float Validation Baseline (2026-08-12)

v02 학습이 epoch333에서 patience(15)로 정상 종료됐다(§ "6. 진행 중 — v02 ablation 학습" 참조,
`results.csv` 기준 F1@0.5 best epoch≈315, 313~317이 거의 평탄한 plateau). NPU 변환에 들어가기
전에, 학습 로그(`results.csv`)에 찍힌 수치를 그대로 베이스라인으로 삼지 않고 `best.pt`를 실제
validation 파이프라인으로 다시 돌려 float baseline을 확정한다.

### 1. 기존 validation 파이프라인 조사

`model_name: dair`로 학습된 모델(v01/v02)에 맞는 표준 검증 경로를 찾으려고 `tools/` 를 확인했다:

- **`tools/validate_custom.py`** (`README.md`가 "평가" 명령으로 안내하는 스크립트) — `CustomDetectionValidator`를
  **하드코딩**해서 쓴다(`model_name`을 보고 분기하지 않음). `dair` 모델은 `dycrop` 데이터셋으로
  학습되는데, 독립적인 x/y 게인 박스 스케일링(`DairDetectionValidator.scale_preds`)이 없는
  `CustomDetectionValidator`로 돌리면 (rear는 crop 비율이 1.0이라도 1920x1080 → 320x320 리사이즈
  자체가 x/y 비율이 다르므로) 박스 스케일링이 어긋날 위험이 있다 — v01/v02에는 부적합.
- **`tools/eval.py`** — `DairDetectionTrainer`를 직접 쓰는, dair 전용으로 보이는 스크립트였지만
  실행해보니 그대로는 동작하지 않는다: (a) `overrides`에 `model` 키를 안 채운 채 트레이너를
  생성해서 ultralytics `BaseTrainer.__init__`이 `self.args.model=None`에서 곧바로 크래시하고,
  (b) `trainer.testset`을 참조하는데 이 프로젝트의 `DetectionTrainer` 계열은 애초에 `self.testset`
  속성을 만들지 않는다(`self.data = self.get_dataset()`이 dict 하나만 리턴 — tuple unpack 대상이
  아님). 즉 **미완성/미사용 코드**로 보인다.

두 스크립트 다 그대로 쓸 수 없어서, 학습 중 매 epoch 내부적으로 쓰이던 것과 **동일한 경로**
(`DairDetectionTrainer.get_dataloader` → `build_dycrop_yolo_dataset`로 val 데이터셋 빌드 →
`DairDetectionValidator`로 forward+메트릭 계산 — `ultralytics/engine/trainer.py`의
`_setup_train`이 `self.test_loader`/`self.validator`를 만드는 방식을 그대로 재현)를 짧은 1회성
스크립트로 재현해서 돌렸다. 스크립트는 재사용 목적이 아니라 컨테이너에서 실행 후 삭제했다 —
로직은 아래 "검증 실행 방법"에 그대로 남긴다.

### 2. 검증 조건

| 항목 | 값 |
|---|---|
| Weight | `runs/detect/BSD_F606/Rear_Truck/v02/weights/best.pt` |
| Config (하이퍼파라미터/데이터 소스) | `configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml` |
| Data yaml | `configs/datasets/1t_rear_cnr_postprocess.yaml` (`val: pure/val/images`) |
| Val 이미지 수 | 7,894장 (`/dataset/1T_REAR/pure/val/labels.cache`) |
| imgsz | 320 (exp.yaml `overrides.imgsz`, 학습과 동일) |
| conf / iou | 미지정 → ultralytics 기본값(`conf=None`→val 시 0.001 상당, `iou=0.7`) — exp.yaml에 별도 지정 없음, 학습 중 내부 val과 동일 조건 |
| batch (val) | 64 (= 학습 `batch=32` × 2, `BaseTrainer._setup_train`의 non-obb 관례를 그대로 재현) |
| Validator | `DairDetectionValidator` (`dair=True`, 학습 때 내부 검증과 동일 클래스) |
| plots / save_json | False / False (검증용 1회성 실행이라 부산물 최소화) |

이 조건은 v02가 학습 내내 매 epoch 내부적으로 검증받던 것과 완전히 동일한 파이프라인/데이터/설정이다
— 학습 로그의 epoch333 값(P=0.8595, R=0.6820, mAP50=0.7823, mAP50-95=0.5570)과 지금 재측정한
best.pt 값이 다른 것은 정상이다(`best.pt`는 fitness가 가장 좋았던 시점, epoch333은 마지막 epoch).

**주의**: 이 표의 mAP는 project 표준 val-split(표준 ultralytics NMS) 기준이다. 이 문서 상단
"질문 1/2"에서 쓴 production/v01 float 수치(0.521 / 0.543)는 **ONNX + 보드와 동일한
dual-threshold NMS, 별도의 6,152장 GT 세트**(이 6,152장의 유래는 "질문 1" 절 상단 참고)로
측정한 것이라 서로 다른 methodology다 — 이
표의 v02 수치를 위 표의 production/v01 수치와 직접 비교하지 말 것. NPU 변환 후 board와 비교할
때는 이번에도 dual-NMS/ONNX 경로로 float를 다시 재는 것이 apples-to-apples 원칙에 맞는다.

### 3. 검증 실행 방법 (컨테이너 내부, `dair-train`)

```python
from pathlib import Path
from projects.custom_utils.cfg import get_my_cfg
from projects.src.models.yolo.detect.train import DairDetectionTrainer

cfg = get_my_cfg("configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml")
kwargs = {"task": "detect", "mode": "val", "model": "runs/detect/BSD_F606/Rear_Truck/v02/weights/best.pt"}

trainer = DairDetectionTrainer(overrides=kwargs, custom_cfg=cfg)
trainer.save_dir = Path("runs/detect/_verify/v02_float_baseline")  # v02 학습 결과물과 분리된 경로
trainer.args.plots = False
trainer.setup_model()

val_batch = trainer.batch_size * 2  # ultralytics BaseTrainer._setup_train과 동일 관례
trainer.test_loader = trainer.get_dataloader(
    trainer.data.get("val") or trainer.data.get("test"), batch_size=val_batch, rank=-1, mode="val"
)
validator = trainer.get_validator()
validator.save_dir = trainer.save_dir
results = validator(model=trainer.model)
```

### 4. 결과

- **검증 날짜**: 2026-08-12
- **best.pt 학습 epoch**: 체크포인트에 `epoch: -1`로 기록돼 있어(최종 저장 시 ultralytics가
  리셋하는 값) 정확한 epoch 번호는 알 수 없음. `results.csv`의 F1@0.5 plateau(epoch 313~317,
  최고점 315)와 체크포인트 `date`(2026-08-11T17:37:25, 마지막 epoch333 완료 시각 18:39보다
  이름)로 보아 이 plateau 구간에서 저장된 것으로 추정.

| 지표 | 값 |
|---|---|
| Precision | 0.8606 |
| Recall | 0.6783 |
| mAP50 | 0.7801 |
| mAP50-95 | **0.5560** |

클래스별(순서: bus, car, motorcycle, person, truck):

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| bus | 2440 | 4397 | 0.931 | 0.808 | 0.884 | 0.705 |
| car | 5111 | 9058 | 0.760 | 0.708 | 0.769 | 0.566 |
| motorcycle | 4779 | 8690 | 0.870 | 0.609 | 0.742 | 0.456 |
| person | 5074 | 15324 | 0.822 | 0.574 | 0.686 | 0.413 |
| truck | 3549 | 5173 | 0.919 | 0.693 | 0.819 | 0.640 |

v01 float(학습 종료 시 로그 기준 mAP50-95≈0.558)과 거의 동일한 수준 — TensorSpeakerOverlayAug
비활성화가 float 성능을 떨어뜨리지 않았음을 확인. 가설 검증의 다음 단계(3텐서 baseline NPU 변환
→ board mAP)로 넘어갈 수 있다.

### 5. NPU 비교 테이블 (골격 — NPU 값은 변환 후 채움)

| Version | Model | Float mAP50 | Float mAP50-95 | NPU mAP50 | NPU mAP50-95 | Δ mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| v02 | best.pt | 0.7801 | 0.5560 | N/A | N/A | N/A |

**주의**: 이 표는 project 표준 val-split(7,894장, 위 §2 조건) 기준이며, **board mAP는 이 methodology로
측정되지 않았고 앞으로도 측정하지 않는다** — board와의 apples-to-apples 비교는 항상 아래 "v02 Float
ONNX Baseline" 섹션의 dual-NMS/6,152장 수치(§5, §5-1)를 쓴다. NPU 열은 "아직 측정 안 함(TBD)"이
아니라 "이 methodology로는 측정 대상이 아님(N/A)"이 정확한 의미다 — 이 표는 그대로 유지하고
대체하지 않는다.

## v02 Float ONNX Baseline (2026-08-12, production/v01과 동일 methodology)

목적: 이 문서 상단 "질문 1/2"에서 production(-16.5%)·v01(-33.1%) 양자화 손실을 비교했던 것과
**완전히 같은 조건**(같은 6,152장 GT, 같은 전처리, 같은 dual-threshold NMS, 같은 평가
스크립트/config 포맷)으로 v02의 Float ONNX baseline을 확보한다. NPU(board) 변환 전 단계이며,
위의 7,894장 `DairDetectionValidator` 결과(§ "v02 Float Validation Baseline")와는 별개로
유지한다 — 서로 대체하지 않음.

### 1. 기존 methodology 확인 (재사용, 신규 제작 없음)

`onnx_v01_dualnms_cfg.yaml`/`onnx_float_dualnms_cfg.yaml`(production)을 그대로 확인해 동일한
필드만 모델 경로를 바꿔 재사용했다:

| 항목 | production/v01과 동일 여부 | 값 |
|---|---|---|
| GT | 동일 | `runs/_eval_scratch/ipu_gt` (6,152장, `cxywhn` 포맷) |
| 전처리 | 동일 | `projects.metrics.utils.basic_preproc` (`RGB: false`) |
| NMS | 동일 | `projects.metrics.utils.basic_postproc_v8_dual_nms` (`min_overlap_thres=0.6`, `union_overlap_thres=0.3`) — 보드의 온디바이스 dual-threshold NMS와 동일 |
| 추론 함수 | 동일 | `projects.metrics.utils.basic_inference_v8` |
| 평가 스크립트 | 동일 | `tools/customEval.py <cfg>` → `projects.metrics.evaluator.Evaluator` |
| classes | 동일 | `[bus, car, motorcycle, person, truck]` |
| config | **신규 파일**(내용은 기존 것 그대로 복사, model_path/save_dir만 교체) | `runs/_eval_scratch/configs/onnx/onnx_v02_dualnms_cfg.yaml` (2026-08-12 재정리 후 경로) |

기존 `onnx_float_dualnms_cfg.yaml`(production), `onnx_v01_dualnms_cfg.yaml`은 읽기만 했고
수정하지 않았다.

### 2. v02 ONNX export

기존 v01 ONNX(`runs/detect/BSD_F606/Rear_Truck/v01/weights/best.onnx`)의 메타데이터
(`opset=13, simplify=False, nms=False, skip_postprocess=True, imgsz=[320,320]`)를 먼저 확인하고,
같은 옵션으로 `tools/export_custom.py`를 실행해 동일한 그래프 구조(출력 shape `(1,5,2100)`,
`(1,4,2100)` — DFL 디코드 포함, NMS 미포함, dual-NMS는 평가 config에서 외부 적용)를 재현했다:

```bash
python3 tools/export_custom.py \
  -w runs/detect/BSD_F606/Rear_Truck/v02/weights/best.pt \
  -i 320 -y 320 \
  -s runs/detect/BSD_F606/Rear_Truck/v02/weights/best.onnx
```

`best.pt`/`last.pt`/`results.csv` 등 기존 v02 학습 결과물은 건드리지 않았고, `best.onnx`만
같은 `weights/` 폴더에 새로 추가했다(v01과 동일한 배치 관례).

### 3. 평가 실행

```bash
# 실행 당시 경로. 2026-08-12 재정리 후 현재 경로는
# runs/_eval_scratch/configs/onnx/onnx_v02_dualnms_cfg.yaml
python3 tools/customEval.py runs/_eval_scratch/onnx_v02_dualnms_cfg.yaml
```

결과: `runs/_eval_scratch/results/onnx_v02_dualnms/onnx_v02_dualnms_cfg/20260812_011808/eval.log`
(기존 production/v01 로그는 그대로 있고, v02용 로그만 새로 생성됨).

### 4. 결과

| 모델 | Float ONNX mAP@0.5 | Float ONNX mAP@0.5:0.95 |
|---|---:|---:|
| **Production** (`ultralytics_projects-main`) | 0.731 | **0.521** |
| **v01** (이 repo) | 0.750 | **0.543** |
| **v02** (TensorSpeakerOverlayAug 비활성) | 0.748 | **0.536** |

클래스별(v02):

| Class | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| bus | 0.866 | 0.695 |
| car | 0.742 | 0.549 |
| motorcycle | 0.738 | 0.454 |
| person | 0.664 | 0.397 |
| truck | 0.730 | 0.585 |
| **all** | **0.748** | **0.536** |

**세 모델 모두 동일 methodology**(같은 6,152장 GT, 같은 전처리/NMS, 같은 평가 스크립트)로 측정됐다
— 직접 비교 가능하다.

v02는 v01보다 float mAP@0.5:0.95가 소폭 낮다(0.536 vs 0.543, -0.007)만, 여전히 production보다는
높다(+0.015). TensorSpeakerOverlayAug 비활성화가 float 성능을 production 수준으로 깎아내리지는
않았다 — §"6. 진행 중 — v02 ablation 학습"의 가설(양자화 손실이 이 augmentation 때문이라면,
float 성능은 유지되면서 NPU 변환 후에만 production 수준(-16.5%대)의 손실로 좁혀져야 한다)을
검증할 수 있는 상태가 됐다.

### 5. NPU(board) 비교 테이블 — apples-to-apples 골격 (NPU 값은 변환 후 채움)

| Version | Model | Float mAP50 (dual-NMS, 6152장) | Float mAP50-95 (dual-NMS, 6152장) | NPU mAP50 | NPU mAP50-95 | Δ mAP50-95 | Δ% |
|---|---|---:|---:|---:|---:|---:|---:|
| Production | `best_REAR_320x320_DAIR_postprocess.onnx` | 0.731 | 0.521 | — | 0.435 | -0.086 | -16.5% |
| v01 | `runs/detect/.../v01/weights/best.onnx` | 0.750 | 0.543 | — | 0.363 | -0.180 | -33.1% |
| **v02** | `runs/detect/.../v02/weights/best.onnx` | **0.748** | **0.536** | **0.678** | **0.435** | **-0.101** | **-18.8%** |

production/v01의 NPU mAP50(IoU@50) 값은 이 문서 상단 §질문1의 원 실험에서 mAP@0.5:0.95만
기록해뒀고 mAP50은 따로 남기지 않아 `—`로 비워둠(필요하면 기존 `board_cfg.yaml`/`board_v01_cfg.yaml`
로그를 다시 파싱해서 채울 수 있음, 지금 단계에서는 하지 않음).

### 5-1. v02 3-tensor board mAP 실측 결과 (2026-08-12, 사용자가 실제 보드에서 추론)

사용자가 3텐서 baseline NVT 모델(`novatek/convert/rear_truck_v02_320/`)을 실제 보드에 넣어 6,152장을
추론한 덤프(`runs/_eval_scratch/board_dumps/v02/v02_output.json`, 2026-08-12 재정리 후 경로,
이미지당 JSON array concatenated — production/v01과
동일 형식)를 가져왔다. `merge_board_dump.py`로 병합 후(`ipu_pred_merged_v02_3tensor.json`),
`board_v01_9tensor_cfg.yaml`과 동일한 구조(`gt`=`ipu_gt` 6,152장, `pred.rep: ipu_test`, 같은
`mode`/`classes`)의 `board_v02_3tensor_cfg.yaml`로 `tools/customEval.py` 평가.

- **Float ONNX mAP50-95**: 0.536 (§4)
- **Board NPU mAP50**: 0.678
- **Board NPU mAP50-95**: 0.435
- **절대 loss**: -0.101
- **상대 loss**: -18.8%
- **Production loss**: -16.5%
- **기존 v01(3텐서 baseline) loss**: -33.1%

**TensorSpeakerOverlayAug 가설 판정 — 지지(강함).** v02의 손실(-18.8%)은 기존 v01의 3텐서
baseline(-33.1%)에서 벌어졌던 격차(-16.6%p)를 거의 다 좁혀(-2.3%p 차이만 남기고) production
수준(-16.5%)에 근접했다. **model.2/4 9텐서 보호(자원 2배) 없이, 저비용 3텐서 baseline 그대로**
이 수준의 손실이 나왔다는 것이 핵심 — §"4. 그러나 — 자원 비용 2배, 근본 해결 아님"에서 확인했던
"board mAP는 맞췄지만 자원을 2배 써서 억지로 맞춘 것"이라는 문제를 우회할 수 있는 경로임을
시사한다. TensorSpeakerOverlayAug(DAIR augmented-view 후보 중 하나)가 activation 분포를
양자화에 불리한 방향으로 왜곡시켜 v01의 큰 손실에 유의미하게 기여했다는 가설이 실측으로 지지됨.

남은 -2.3%p 격차(production 대비)의 원인은 미확인 — 완전히 동일한 0은 아니므로 다른 부차적
요인(학습 seed, augmentation 조합 순서 등)이 소폭 남아있을 가능성은 있으나, 자원비용 없이 이 정도
수준까지 좁혀졌다는 것 자체가 이 조사의 원래 목표(10% 이내 손실)에는 아직 못 미치지만 근본
원인 규명 측면에서는 충분히 강한 증거다.

### 6. 관련 파일 (추가)

- 신규 config: `runs/_eval_scratch/configs/onnx/onnx_v02_dualnms_cfg.yaml` (2026-08-12 재정리 후 경로)
- 신규 export: `runs/detect/BSD_F606/Rear_Truck/v02/weights/best.onnx`
- 신규 평가 로그: `runs/_eval_scratch/results/onnx_v02_dualnms/onnx_v02_dualnms_cfg/20260812_011808/{eval.log,coco_prediction.json,F1_Confidence.png}`

## Rear_Truck 외부 학습 비교 — v01_dair / v01_symmetric (2026-08-12)

다른 학습 PC에서 이미 완료된 두 개의 추가 실험을 이 저장소로 결과만 복사해와서 검증한다:

- **v01_dair** (`configs/experiments/BSD_F606/Rear_Truck/v01_dair/exp.yaml`) — production 원본
  레시피(`mycfg.yaml` 출처: NAS의 production `weight/mycfg.yaml`)를 이 파이프라인으로 재현한 것.
  `model_name: dair`.
- **v01_symmetric** (`configs/experiments/BSD_F606/Rear_Truck/v01_symmetric/exp.yaml`) — v01_dair와
  같은 데이터/해상도/no-crop 조건이지만 `model_name: symmetric`(DAIR 없음)으로 바꾼 병렬 비교용.

두 config 모두 `overrides.data: configs/datasets/bsd_rear_truck.yaml`을 가리키는데, 이 파일과 그
원본 데이터가 이 PC에는 없다(다른 PC에서 2026-08-10에 새로 만든 것). 사용자 확인: 두 학습은 **이
저장소의 기존 `v01`(→ `configs/datasets/1t_rear_cnr_postprocess.yaml`, `/dataset/1T_REAR`)과
동일한 데이터·동일한 train/val split**을 쓴다 — 그래서 내부 validation은 원본 config 파일을
건드리지 않고 **런타임에 `data` 오버라이드만** 이 yaml로 주고 재현했다(아래 §1).

### 1. 실제 구성 확인 (코드/config 기준, 이름으로 추측하지 않음)

| | v01_dair | v01_symmetric |
|---|---|---|
| 학습 종료 | epoch 206 (500 중, patience=15로 추정 정지) | epoch 407 (500 중, patience=15로 추정 정지) |
| `best.pt` 무결성 | `torch.load` 정상, 손상 없음 | `torch.load` 정상, 손상 없음 |
| Trainer | `DairDetectionTrainer` | `SymmetricTrainer` (`super().__init__()` 미호출, BaseTrainer 로직을 직접 재구현) |
| Validator (get_validator 반환값, 코드 확인) | `DairDetectionValidator` — 독립 x/y gain `scale_boxes` 있음 | `SymmetricDetectionValidator` — **표준 ultralytics 균일 gain** `scale_boxes` (독립 x/y gain 없음) |
| dataset 빌드 | `build_dycrop_yolo_dataset` | `build_dycrop_yolo_dataset` (동일 — symmetric도 dycrop 사용, 표준 `build_yolo_dataset` 호출은 주석처리돼 있음) |
| architecture | 동일: `projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml` (`SymmetricDetect` head, 3.01M params) |
| dair_params | mode=or, λ=0.1, γ=0.5 → dual-forward(원본+aug) 로스 | 없음 → `self.dair_params=None`, validator의 dair-aug 분기 비활성, loss_names도 `_m/_o` 단일 로 (results.csv 헤더로 확인) |
| optimizer / lr0 | AdamW / 0.001 | SGD / 0.01 (from-scratch 기본값), `cos_lr=true` |
| batch | 32 | 48 |
| imgsz / crop | 둘 다 320, crop 없음(`crop_width_ratio=1920/1920`, `crop_height_ratio=1080/1080`) — v01/v02와 동일 |

**REAR는 1920x1080 → 320x320 비정방 리사이즈**라 x/y 게인이 다르다 — `SymmetricDetectionValidator`가
독립 게인 없이 표준 `scale_boxes`를 쓰는 점은 v01_symmetric의 내부 mAP 수치에 체계적 편향을 만들
수 있는 잠재 요인이다(자체 학습 내부 검증 방식 그대로 재현한 것이므로 이 자체는 "버그"가 아니라
기존 코드의 특성). ONNX+dual-NMS 평가(§3)는 두 모델 다 같은 외부 GT/전처리를 쓰므로 이 차이의
영향을 받지 않는다.

### 2. Internal Float Validation (각 모델의 학습 내부 pipeline 재현, data만 v01과 동일하게 오버라이드)

| | v01_dair | v01_symmetric |
|---|---|---|
| Weight | `runs/detect/BSD_F606/Rear_Truck/v01_dair/weights/best.pt` | `runs/detect/BSD_F606/Rear_Truck/v01_symmetric/weights/best.pt` |
| Trainer/Validator | `DairDetectionTrainer` / `DairDetectionValidator` | `SymmetricTrainer` / `SymmetricDetectionValidator` |
| Data yaml (런타임 오버라이드) | `configs/datasets/1t_rear_cnr_postprocess.yaml` | `configs/datasets/1t_rear_cnr_postprocess.yaml` |
| Val 이미지 수 | 7,894장 | 7,894장 |
| imgsz | 320 | 320 |
| batch (val) | 64 (=32×2) | 96 (=48×2) |
| conf / iou | 기본값(미지정) | 기본값(미지정) |
| Precision | 0.8469 | 0.8645 |
| Recall | 0.6644 | 0.6947 |
| mAP50 | 0.7646 | 0.7985 |
| **mAP50-95** | **0.5414** | **0.5681** |

결과는 `runs/detect/_verify/v01_dair_float_baseline/`, `runs/detect/_verify/v01_symmetric_float_baseline/`에
저장(원본 학습 결과 폴더는 건드리지 않음). 검증 스크립트는 v02와 동일한 방식(트레이너를 직접
구성해 `test_loader`/`validator`만 학습 내부와 동일하게 재현)으로 실행 후 삭제.

### 3. ONNX export + apples-to-apples Float 평가 (6,152장, dual-NMS)

`tools/export_custom.py -i 320 -y 320`(opset=13, simplify=False, nms=False, skip_postprocess=True —
기존 v01/v02 ONNX 메타데이터와 대조해 확인한 값 그대로)로 각 `best.pt` → `best.onnx`.
`runs/_eval_scratch/configs/onnx/onnx_v01_dualnms_cfg.yaml`(2026-08-12 재정리 후 경로)을 복사해
`model_path`/`save_dir`만 바꾼
`onnx_v01_dair_dualnms_cfg.yaml`, `onnx_v01_symmetric_dualnms_cfg.yaml`로 평가(GT/전처리/dual-NMS
파라미터 전부 동일, 기존 production/v01/v02 config·결과는 미수정).

| Model | Float ONNX mAP50 | Float ONNX mAP50-95 |
|---|---:|---:|
| Production | 0.731 | 0.521 |
| 기존 v01 | 0.750 | 0.543 |
| v02 | 0.748 | 0.536 |
| **v01_dair** | **0.734** | **0.531** |
| **v01_symmetric** | **0.774** | **0.561** |

클래스별(mAP50-95 / mAP50):

| Class | v01_dair | v01_symmetric |
|---|---:|---:|
| bus | 0.700 / 0.860 | 0.700 / 0.872 |
| car | 0.535 / 0.722 | 0.567 / 0.764 |
| motorcycle | 0.445 / 0.725 | 0.485 / 0.777 |
| person | 0.387 / 0.654 | 0.430 / 0.706 |
| truck | 0.589 / 0.711 | 0.622 / 0.749 |
| **all** | **0.531 / 0.734** | **0.561 / 0.774** |

**v01_symmetric이 지금까지 측정된 모든 Rear_Truck float 결과 중 가장 높다** (production/v01/v02
전부 상회). DAIR 유무가 float 성능에 미치는 영향이 이 데이터셋에서는 오히려 음(-)의 방향으로
보이지만, NPU 양자화 손실까지는 아직 미측정 — board는 사용자가 직접 실측할 예정이라 이 세션에서는
진행하지 않음.

### 4. 비교 표 — 최종 (v01_dair board 실측 반영, 2026-08-12)

| Model | Internal Float mAP50-95 | Float ONNX mAP50-95 (6152) | NPU mAP50-95 | Quant Loss | Latency | DRAM | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production | — | 0.521 | 0.435 | -16.5% | 6.25ms | 35.0MB | 4.70MB |
| 기존 v01 (3텐서 baseline) | — | 0.543 | 0.363 | -33.1% | 6.37ms | 34.9MB | 4.71MB |
| v02 | 0.5560 | 0.536 | 0.435 | -18.8% | 6.368ms | 36.54MB | 4.71MB |
| **v01_dair** | **0.5414** | **0.531** | **0.421** | **-20.7%** | 6.368ms | 36.54MB | 4.71MB |
| **v01_symmetric** | **0.5681** | **0.561** | **0.521** | **-7.1%** | 6.368ms | 36.54MB | 4.71MB |

Production/기존 v01의 "Internal Float mAP50-95"는 이 문서 어디에도 실측 기록이 없어 `—`로 남김
(production은 이 저장소의 학습 파이프라인으로 학습된 적이 없어 "내부 validation" 개념 자체가
없고, 기존 v01은 이 문서가 float 기준으로 항상 dual-NMS/6152장 수치만 써왔음).

> **※ 당시 시점 기록 (2026-08-12 오전) — 이후 실측으로 대체됨**: v01_dair board를 측정하지
> 않기로 한 판단 근거는 다음과 같았다. v01_dair는 DAIR 사용·`TensorSpeakerOverlayAug`(추정)
> ON·동일 architecture/dataset/split·batch 32·AdamW·lr0=0.001·동일 DAIR 파라미터 계열로,
> 독립적인 새 ablation이 아니라 **기존 v01과 사실상 같은 recipe 계열의 반복 학습**이다(위
> "버전 비교" §1 표 참조). 기존 v01이 이미 동일한 조건(DAIR, OverlayAug ON, 동일 recipe)에서
> 3텐서 board 양자화 손실 -33.1%를 실측해뒀으므로, v01_dair에 대해 같은 축의 board 측정을
> 반복하는 것은 이번 조사의 핵심 결론에 추가 정보를 주지 않는다고 판단해 우선순위를 낮췄다 —
> "아직 측정 안 함(TBD)"이 아니라 의도적으로 생략한 것이었다.

**실측 결과 (2026-08-12 오후)**: 사용자가 v01_dair의 실제 board 덤프를 추가로 확보해 위 판단과
별개로 측정을 진행했다. 1차 덤프는 v02 덤프와 6,152장 전체 detection이 픽셀 단위로 완전히
동일해(§5-1의 v01_symmetric 1차 무효 덤프와 같은 유형의 stale-model 패턴) 무효 처리하고
재덤프를 요청했고, 2차 재덤프는 다른 4개
모델(production/v01/v02/v01_symmetric) 어느 것과도 md5·detection 내용이 겹치지 않음을
확인(jaccard overlap ≈ 0)한 뒤 평가했다. 결과: **Board mAP50 0.669 / mAP50-95 0.421**, Float
ONNX(0.531) 대비 절대손실 -0.110, **상대손실 -20.7%**.

**판단이 틀렸다는 뜻은 아니지만, 가정이 부정확했다**: 위 생략 판단은 "v01_dair가 기존 v01과
사실상 같은 축이라 board 손실도 -33.1%대일 것"을 전제로 우선순위를 낮춘 것이었다. 실측
결과(-20.7%)는 그 전제와 달리 기존 v01(-33.1%)보다 12.4%p 더 낫고, 오히려 v02(-18.8%)에 근접한
값이다. 즉 "동일 recipe 계열"이라는 이유만으로 board 손실까지 동일할 것이라고 가정하는 것은
근거가 부족했다 — 같은 recipe라도 서로 다른 학습 run(weight) 사이에 board 양자화 손실이 이 정도
(수십 %p 단위) 갈릴 수 있다는 것 자체가 새로운 관찰이다(§"최종 결론" §6, 선택적 후속 연구의
"seed 반복 실험" 항목 참조).

### 5. v01_dair / v01_symmetric 3-tensor NVT 컴파일 + 시뮬레이션 (2026-08-12)

v02와 완전히 동일한 3텐서(H1/H2/H3만, model.2/4 9텐서 보호 없음) baseline 조건으로, 각각 독립
디렉토리에서 순차적으로(동시 실행 안 함) compile + simulate 수행. calibration(940장 video_calib_jpg),
`calib_mode=4`/`percentile=0.999`/`cle_en=1`/`cal_refine_iter=5` 등 `gen_config.txt`의 나머지 항목은
`rear_truck_v02_320`과 바이트 단위로 동일 — `[path/model_dir]`과 `spec_demarc_tensors`(3개, 값 동일)만
패턴명에 맞게 존재. production/기존 v01/v02 디렉토리는 읽기만 하고 전혀 수정하지 않았다.

| | v01_dair | v01_symmetric | (참고) v02 |
|---|---:|---:|---:|
| Latency | 6.368ms | 6.368ms | 6.368ms |
| DRAM bandwidth | 36.54MB | 36.54MB | 36.54MB |
| Model size | 4.71MB | 4.71MB | 4.71MB |
| Softmax(dfl) cosine | 0.9264 | **0.9850** | 0.9058 |
| Sigmoid[OUT] cosine | 0.9515 | **0.9824** | 0.9236 |
| dfl_conv[OUT] cosine | 0.9740 | **0.9933** | 0.9664 |

리소스(latency/DRAM/model size)는 셋 다 완전히 동일 — 3텐서 baseline이라 ONNX weight 차이와
무관하게 비용이 고정됨(예상대로). **cosine은 v01_symmetric(DAIR 없음)이 셋 중 가장 높고,
v02(DAIR + TensorSpeakerOverlayAug 비활성)가 가장 낮다** — DAIR 학습 자체가 quantization
sensitivity에 영향을 주는 요인일 수 있음을 시사하는 새로운 단서. 다만 이 단일 이미지 cosine은
board mAP(6,152장)로 확정하기 전까지는 참고 지표일 뿐이다.

> **업데이트 (2026-08-12) — 위 참고 지표 경고가 실제로 맞아떨어짐**: v01_dair의 board mAP가
> 이후 실측됐다(§4). cosine 서열은 `v01_dair(0.926/0.952/0.974) > v02(0.906/0.924/0.966)`였지만,
> 실제 board 양자화 손실은 `v01_dair(-20.7%)`가 `v02(-18.8%)`보다 **더 나빴다** — cosine이
> 예측한 순서와 실제 board mAP 순서가 뒤바뀐 사례다. 이 문서와 SOP
> (`docs/guides/rear_truck_train_validate_nvt_pipeline.md` §8)가 처음부터 강조했던 "single-image
> cosine은 진단용 신호일 뿐 최종 성능 지표가 아니다"라는 원칙을 실측으로 재확인한 것으로 기록한다.

### 5-1. v01_symmetric 3-tensor board mAP 실측 결과 (2026-08-12, 재덤프)

첫 번째 덤프(`v01_symmetric_output.json`, 최초본)는 `v02_output.json`과 6,152장 전부 detection이
픽셀 단위로 완전히 동일해(md5도 다르고 "time" 필드만 달랐음) 보드에 v01_symmetric NVT 모델이
실제로 플래싱되지 않은 것으로 판단, 사용자에게 재덤프를 요청했다. **재덤프본**(md5/크기/detection
개수 전부 v02와 다름을 확인 후 진행)을 v02와 동일한 절차(`merge_board_dump.py` →
`board_v01_symmetric_3tensor_cfg.yaml`, `board_v01_9tensor_cfg.yaml`/`board_v02_3tensor_cfg.yaml`과
동일 구조 — `gt`=`ipu_gt` 6,152장, `pred.rep: ipu_test`)로 평가했다.

- **Float ONNX mAP50-95**: 0.561 (§3)
- **Board NPU mAP50**: 0.751
- **Board NPU mAP50-95**: 0.521
- **절대 loss**: -0.040
- **상대 loss**: **-7.1%**
- **Production loss**: -16.5% / **기존 v01(3텐서) loss**: -33.1% / **v02(3텐서) loss**: -18.8%

**v01_symmetric이 지금까지 측정된 모든 Rear_Truck 모델 중 양자화 손실이 가장 작다** — production의
절반 이하, 이 조사의 원래 목표였던 "10% 이내" 손실을 처음으로 달성했다. Float 성능도 가장 높고
(0.561), 양자화 손실도 가장 작다 — 3텐서 저비용 baseline, model.2/4 9텐서 보호 없이 나온 결과다.

**해석**: `TensorSpeakerOverlayAug` 비활성화만으로도(v02) 손실이 -33.1%→-18.8%로 크게 좋아졌는데,
DAIR 자체를 아예 쓰지 않은 v01_symmetric은 -7.1%까지 더 좋아졌다. **DAIR dual-forward 학습
방식(또는 그에 딸린 augmented-view 파이프라인 전체)이 양자화 민감도에 TensorSpeakerOverlayAug보다
더 근본적인 영향을 줄 가능성**이 이번 실측으로 새롭게 제기된다 — 다만 optimizer/lr/batch도 함께
다르므로(README.md "버전 비교" 표 참조) DAIR 유무 하나만의 효과로 단정할 수는 없다.

> **※ 당시 시점 기록 (2026-08-12 오전) — 이후 실측으로 대체됨**: 이 시점에는 v01_dair는 기존
> v01과 사실상 같은 recipe 계열(DAIR·OverlayAug ON 추정·동일 하이퍼파라미터)이라는 이유로 board
> 평가를 의도적으로 생략했었다 — 기존 v01의 -33.1% 실측이 이미 그 축을 대표한다고 봤기 때문이다.
> 그리고 위에서 "cosine 서열(v01_symmetric > v01_dair > v02)과도 방향이 일치한다"고 적었었다.

**업데이트 (2026-08-12 오후)**: v01_dair board가 실제로 측정됐다(§4) — 결과는 **-20.7%**로, 기존
v01(-33.1%)과 같은 값이 아니라 오히려 v02(-18.8%)에 더 가까웠다. 즉 위 cosine 서열이 예측한
순서(v01_dair가 v02보다 낮은 손실)와 실제 순서가 반대였고, "v01_dair ≈ v01"이라는 생략 판단의
전제도 실측과 어긋났다. DAIR 유무를 optimizer/lr/batch까지 통제한 상태로 더 엄밀히 분리하려면
별도의 strict ablation이 필요하다는 결론(문서 최하단 "최종 결론"의 선택적 후속 연구 참조)은
그대로 유효하며, 이번 결과는 오히려 그 필요성을 더 뒷받침한다 — 같은 recipe 계열로 분류했던
v01과 v01_dair 사이에서도 board 손실이 12.4%p나 갈렸다는 것은 seed/run-to-run 변동성이 이
데이터셋에서 결코 작지 않을 수 있음을 시사한다.

### 6. 관련 파일

- config (복사, 원본 미수정): `configs/experiments/BSD_F606/Rear_Truck/{v01_dair,v01_symmetric}/exp.yaml`
- 신규 ONNX: `runs/detect/BSD_F606/Rear_Truck/{v01_dair,v01_symmetric}/weights/best.onnx`
- 신규 eval config: `runs/_eval_scratch/configs/onnx/onnx_v01_{dair,symmetric}_dualnms_cfg.yaml` (2026-08-12 재정리 후 경로)
- 신규 eval 로그: `runs/_eval_scratch/results/onnx_v01_dair_dualnms/.../eval.log`,
  `runs/_eval_scratch/results/onnx_v01_symmetric_dualnms/.../eval.log`
- 내부 validation 결과(scratch, 학습 결과와 분리): `runs/detect/_verify/v01_dair_float_baseline/`,
  `runs/detect/_verify/v01_symmetric_float_baseline/`
- 신규 NVT 변환(3텐서 baseline, production/기존v01/v02와 독립):
  `novatek/convert/rear_truck_v01_dair_320/`, `novatek/convert/rear_truck_v01_symmetric_320/`
- comparison 이미지: 각 디렉토리의
  `nvtai_tool/output/sim_comparison_KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC1_PGL_DVR_NC22015_0126.jpg`
- board mAP용 v02 덤프: `runs/_eval_scratch/board_dumps/v02/v02_output.json` →
  `runs/_eval_scratch/merged/v02/ipu_pred_merged_v02_3tensor.json` (2026-08-12 재정리 후 경로),
  eval config `runs/_eval_scratch/configs/board/board_v02_3tensor_cfg.yaml`
- board mAP용 v01_symmetric 덤프(재덤프본): `runs/_eval_scratch/board_dumps/v01_symmetric/v01_symmetric_output.json` →
  `runs/_eval_scratch/merged/v01_symmetric/ipu_pred_merged_v01_symmetric_3tensor.json`
  (2026-08-12 재정리 후 경로), eval config
  `runs/_eval_scratch/configs/board/board_v01_symmetric_3tensor_cfg.yaml`
- board mAP용 v01_dair 덤프(2차 재덤프본 — 1차는 v02와 detection 100% 동일해 무효 처리, 파일
  자체를 재덤프본으로 덮어씀): `runs/_eval_scratch/board_dumps/v01_dair/v01_dair_output.json` →
  `runs/_eval_scratch/merged/v01_dair/ipu_pred_merged_v01_dair_3tensor.json`, eval config
  `runs/_eval_scratch/configs/board/board_v01_dair_3tensor_cfg.yaml`

---

## 최종 결론 (2026-08-12)

이 문서 전체(위 모든 절)는 조사 과정 그대로 보존한 기록이다 — 아래는 처음 읽는 사람이 문서
전체를 뒤지지 않고도 최종 상태를 파악할 수 있도록 정리한 요약이며, 과거 절의 내용을 대체하지
않는다. 이 조사에서 필수로 계획했던 실험(v01 원인 조사 → v02 ablation → v01_dair/v01_symmetric
병렬 비교, 각각의 float/board 실측)은 이 시점 기준 **모두 완료됐다**.

### 1. 기존 문제 — v01은 float가 아니라 양자화가 문제였다

`v01`(이 repo 재현)은 Float ONNX에서 production보다 높았다(0.543 vs 0.521, +4.2%) — 학습
파이프라인 자체는 문제가 없었다. 그런데 NPU(3텐서 baseline)에서는 `0.543 → 0.363`으로
**-33.1%** 무너져, production의 손실(-16.5%)의 약 2배였다. 즉 이건 단순 학습 성능 문제가 아니라
**quantization sensitivity 문제**였다(§질문1, §질문2).

### 2. `TensorSpeakerOverlayAug` — 큰 기여 요인으로 실측 확인됨

production과 v01의 학습 코드를 전부 diff해서 찾은 유일한 실질적 차이(§5)인
`TensorSpeakerOverlayAug`(DAIR augmented-view 후보 중 하나)를 비활성화한 `v02`는, 3텐서 baseline
그대로(model.2/4 9텐서 보호 없이) `0.536 → 0.435`로 손실이 **-18.8%**까지 회복됐다(§v02 Float
ONNX Baseline §5-1). 기존 v01의 -33.1%에서 격차(-16.6%p)를 거의 다 좁혔다 — **이 augmentation이
v01의 심각한 양자화 민감도에 크게 기여한 주요 요인 중 하나로 확인됐다.**

### 3. 그러나 OverlayAug 제거만으로는 완전히 해결되지 않았다

v02도 여전히 -18.8% 손실로, 이 조사의 목표였던 "10% 이내"에는 못 미쳤고 production(-16.5%)과도
완전히 같지 않다(-2.3%p 차이). **`TensorSpeakerOverlayAug`가 유일한 root cause라고는 결론 내리지
않는다.**

### 4. `v01_symmetric` — 지금까지 가장 우수한 결과, 최초로 10% 이내 달성

다른 학습 PC에서 수행된 병렬 비교 실험 `v01_symmetric`(DAIR 미사용, `SymmetricTrainer`)은 같은
저비용 3텐서 baseline에서 `0.561 → 0.521`, **-7.1%** 손실을 기록했다(§ "Rear_Truck 외부 학습
비교" §5-1). Float 성능(0.561)도, board mAP(0.521)도 지금까지 측정된 모든 Rear_Truck 모델 중
가장 높고, 손실률도 가장 작다 — **이 조사의 원래 목표(Float→NPU 상대 손실 10% 이내)를 처음
달성한 결과**다.

### 5. 현재 해석 — DAIR을 root cause로 확정하지 않는다

> **※ 당시 시점 기록 (2026-08-12) — 아래 "강한 후보/단서" 항목은 2026-08-20 실측으로 반박됨**:
> 이 절 작성 시점에는 `v01_symmetric_matched`(strict ablation)의 board 결과가 없어 "DAIR
> dual-forward 학습 방식 자체가 근본 원인일 가능성"을 유효한 가설로 남겨뒀었다. §7("최종 결론
> 갱신 — v01_symmetric_matched board 실측 반영, 2026-08-20")에서 이 가설이 실측으로 **반대
> 방향임이 확정**됐다 — 최신 결론은 그쪽을 볼 것. 아래는 그 이전 시점의 추론 과정 기록이다.

여섯 모델의 순위는 명확하다: `v01_symmetric(-7.1%) > production(-16.5%) > v02(-18.8%) >
v01_dair(-20.7%) > v01_symmetric_matched(-28.15%) > 기존 v01(-33.1%)`. 다만 `v01_symmetric`은
`v01_dair`/`v01`과 DAIR 유무 외에도 Trainer(`SymmetricTrainer` vs `DairDetectionTrainer`)·loss
구조·Validator(`SymmetricDetectionValidator`의 균일 gain `scale_boxes` vs
`DairDetectionValidator`의 독립 x/y gain)·optimizer(SGD vs AdamW)·lr0(0.01 vs 0.001)·cos_lr(true
vs false)·batch(48 vs 32)가 동시에 다르다(README.md "버전 비교" 표). 따라서:

- **정확한 결론**: `v01_symmetric`의 학습 구성은 현재까지 비교한 Rear_Truck 모델 중 Float
  성능과 NVT 양자화 내성 모두 가장 우수했다.
- **강한 후보/단서 (당시 시점 기록 — 이후 반박됨)**: DAIR dual-forward 학습 방식(또는 그에 딸린
  augmented-view 파이프라인 전체)이 `TensorSpeakerOverlayAug`보다 더 근본적으로 양자화 민감도에
  영향을 줄 가능성이 있다 — DAIR을 쓰지 않은 v01_symmetric이 가장 낮은 손실을 보인 것은 여전히
  유효한 근거다. **(2026-08-20 갱신: `v01_symmetric_matched`가 하이퍼파라미터를 `v01_dair`와
  동일하게 맞추고 DAIR만 뺐더니 오히려 -28.15%로 `v01_dair`(-20.7%)보다 더 나빠졌다 — "DAIR을
  빼면 나아진다"는 이 가설과 반대 방향. §7 참조.)**
- **주의 — 단일 이미지 cosine으로 board 순위를 예단하지 않는다**: 시뮬레이터 cosine 서열은
  `v01_symmetric(0.985/0.982/0.993) > v01_dair(0.926/0.952/0.974) > v02(0.906/0.924/0.966)`였지만,
  실제 board 손실 순위는 `v01_symmetric > v02 > v01_dair`로 **v01_dair와 v02의 순서가
  뒤바뀌었다**(§4, §"Rear_Truck 외부 학습 비교" §5 업데이트 참조). cosine은 v01_symmetric이
  가장 우수하다는 신호는 맞게 예측했지만 나머지 둘의 순서는 예측하지 못했다.
- **확정하지 않는 것 (당시 시점 기록)**: "DAIR이 root cause다"라고는 쓰지 않는다. optimizer/lr/batch 등 다른 축이
  섞여 있어 원인을 완전히 분리하지 못했다. 또한 v01과 v01_dair는 "사실상 같은 recipe 계열"로
  분류됐음에도 board 손실이 -33.1% vs -20.7%로 12.4%p 차이가 났다 — 같은 recipe 축 안에서도
  run-to-run 변동성이 상당할 수 있다는 뜻이며, 이는 DAIR/OverlayAug 축의 효과 크기를 해석할 때
  같이 고려해야 하는 불확실성이다(§6, 선택적 후속 연구의 "seed 반복 실험" 참조). **이 "확정하지
  않는다"는 태도 자체는 §7에서도 유지된다 — 다만 이제는 optimizer/lr/batch 축이 DAIR 유무 축보다
  압도적으로 크다(+21.05%p vs -7.45%p)는 것까지는 실측으로 확정됐다.**

### 6. `v01_dair` board — 실측 완료, "생략" 판단은 철회됨

> **※ 당시 시점 기록 (2026-08-12 오전)**: 처음에는 `v01_dair`가 DAIR 사용·
> `TensorSpeakerOverlayAug` ON(추정)·동일 architecture/dataset/split·AdamW·lr0=0.001·batch 32로
> 기존 v01과 사실상 같은 recipe 계열의 반복 학습이라 판단해, 기존 v01의 -33.1% 실측이 이미 그
> 축을 대표한다고 보고 board 평가를 **의도적으로 생략**했었다 — "TBD"가 아니라 확정된 결정으로
> 문서화했었다.

**이후 사용자가 실제 board 덤프를 추가로 제공해(§4) 이 판단과 무관하게 실측이 이뤄졌다.** 결과는
**-20.7%**로, "기존 v01과 같은 값일 것"이라는 생략 판단의 전제와 달리 v01(-33.1%)보다 뚜렷이
낮고 v02(-18.8%)에 더 가까웠다. 즉 이 항목은 더 이상 "생략"이 아니라 **실측 완료** 상태다 —
위 §5의 최종 순위와 요약 표에 반영했다. 애초의 생략 판단 자체가 잘못된 절차였다고 보지는 않는다
(당시 가진 정보로는 합리적인 우선순위 판단이었다) — 다만 그 판단이 깔고 있던 가정("같은 recipe
계열이면 board 손실도 비슷할 것")이 실측과 어긋났다는 점은 기록해둔다.

### 7. 최종 결론 갱신 — `v01_symmetric_matched` board 실측 반영 (2026-08-20)

§5에서 미확정으로 남겨뒀던 질문("`v01_symmetric`이 좋은 이유가 DAIR 제거 때문인지, 다른
하이퍼파라미터 때문인지")에 대한 답이 `v01_symmetric_matched`(§"v01_symmetric_matched ablation
(2026-08-19)" §6, §6-1)의 board 실측으로 나왔다. 전체 상세 도출 과정은 그 절 참조, 여기서는
"최종 결론"에 필요한 요지만 정리한다.

**핵심 실험**: `v01_symmetric_matched`는 `v01_dair`와 optimizer=AdamW·lr0=0.001·batch=32·
cos_lr=false까지 완전히 동일하게 맞추고 Trainer만 `symmetric`(DAIR dual-forward 없음)으로 바꾼
strict ablation이다. 즉 `v01_dair`와의 유일한 통제된 차이는 "DAIR 사용 여부"다(단, `dair_params`
Select 풀에 딸려오는 `TensorSpeakerOverlayAug`도 구조적으로 함께 빠진다는 잔여 confound는
있음 — 아래 참조).

**결과 (2026-08-20 board 실측 확정)**:

| | v01_dair (DAIR 사용) | v01_symmetric_matched (DAIR 미사용, 나머지 동일) |
|---|---:|---:|
| Board mAP50-95 | **0.421** | 0.393 |
| Quant loss | **-20.7%** | -28.15% |

**DAIR을 빼는 것 자체는 quantization을 개선하지 않았다 — 오히려 7.45%p 악화시켰다.** 이는 §5에서
유효한 가설로 남겨뒀던 "DAIR dual-forward 학습 방식이 근본 원인일 가능성"과 **반대 방향**이다.

반면 `v01_symmetric_matched → v01_symmetric`(하이퍼파라미터만 SGD/lr0=0.01/batch=48/cos_lr=true로
변경, DAIR 미사용은 유지) 구간은 quant loss가 -28.15% → -7.1%로 **+21.05%p 개선**됐다 — DAIR
축(-7.45%p 악화)의 거의 3배 크기다.

**확정된 결론**: `v01_symmetric`이 `v01_dair`보다 quantization을 훨씬 잘 버틴 주된 이유는 **DAIR을
안 써서가 아니라 optimizer/lr0/batch/cos_lr 차이 때문**이다. DAIR 유무 축만 따로 떼어보면
"중립"이 아니라 **DAIR을 쓰는 쪽이 오히려 유리한 방향**이라는, §5의 "강한 후보/단서"와 정반대되는
사실이 이번 실측으로 새로 확정됐다.

**아직 분리되지 않은 것**: `v01_symmetric_matched`는 symmetric 트레이너의 DAIR 모드 자체가 이
repo에서 크래시하는 미완성 경로라(§"v01_symmetric_matched ablation" §1) `dair_params`(및 그
Select 풀에 포함된 `TensorSpeakerOverlayAug`)를 의도적으로 뺀 게 아니라 뺄 수밖에 없었다. 따라서
위 "DAIR을 쓰는 쪽이 유리하다"는 관측이 DAIR dual-forward 메커니즘 자체 때문인지, 그 메커니즘에
구조적으로 딸려오는 `TensorSpeakerOverlayAug`(§2에서 이미 "큰 악영향 요인"으로 확인된 바로 그
augmentation) 부재 때문인지는 이 실험 하나로는 분리되지 않는다 — 후자라면 §2의 결론과 정반대
방향의 새로운 퍼즐이 생긴다. strict DAIR on/off ablation(하이퍼파라미터·augmentation 전부 동일,
DAIR만 on/off)이 여전히 필요하다(선택적 후속 연구 참조).

**acceptance target 관련**: 이 실험은 D-019(2026-08-19, ≤1% 상향) 이전에 설계된 것이라 새 target
달성 여부와는 무관하다 — `v01_symmetric_matched`(-28.15%)도 새 target(≤1%)에는 당연히 크게
못 미친다. 이 실험의 가치는 target 달성이 아니라 위 원인 분리에 있다.

### 요약 표

| Model | Float ONNX mAP50-95 | Board NPU mAP50-95 | Absolute Loss | Relative Loss |
|---|---:|---:|---:|---:|
| Production | 0.521 | 0.435 | -0.086 | -16.5% |
| 기존 v01 (3텐서 baseline) | 0.543 | 0.363 | -0.180 | -33.1% |
| v02 | 0.536 | 0.435 | -0.101 | -18.8% |
| v01_dair | 0.531 | 0.421 | -0.110 | -20.7% |
| v01_symmetric_matched (strict DAIR ablation, §7) | 0.547 | 0.393 | -0.154 | -28.15% |
| **v01_symmetric** | **0.561** | **0.521** | **-0.040** | **-7.1%** |

| Model | Latency | DRAM | Model Size |
|---|---:|---:|---:|
| Production (9텐서, 헤드 근처) | 6.25ms | 35.0MB | 4.70MB |
| 기존 v01 3-tensor baseline | 6.37ms | 34.9MB | 4.71MB |
| 기존 v01 model.2/4 9-tensor workaround (§4 "그러나 — 자원 비용 2배") | 12.25ms | 78.5MB | 9.39MB |
| v02 3-tensor | 6.368ms | 36.54MB | 4.71MB |
| v01_dair 3-tensor | 6.368ms | 36.54MB | 4.71MB |
| v01_symmetric_matched 3-tensor | 6.36765ms | 36.54MB (36,540,852 bytes) | 4.71MB |
| v01_symmetric 3-tensor | 6.368ms | 36.54MB | 4.71MB |

**9-tensor workaround 행과 3-tensor baseline 행을 혼동하지 말 것** — 9-tensor는 board mAP를
production 수준으로 맞추긴 했지만(§"3. 돌파구") 자원 비용이 약 2배라 근본 해결이 아니었다(§"4.
그러나"). 반면 v02/v01_dair/v01_symmetric은 전부 **3-tensor baseline 그대로**(model.2/4 보호 없이)
얻은 결과다.

### 선택적 후속 연구 (optional future work — 이번 조사의 완료 조건과는 분리)

이번 조사의 필수 실험은 모두 끝났다. 아래는 결론을 더 다듬고 싶을 때 고려할 수 있는 선택 사항이며,
지금 하지 않아도 현재 결론은 유효하다:

- **DAIR 유무 strict ablation**: optimizer/lr/batch/cos_lr을 완전히 동일하게 맞춘 상태에서
  DAIR on/off만 바꾼 재학습 — §5의 "현재 해석"에서 확정하지 못한 DAIR 자체의 순수 기여를
  분리하기 위함. **완료(2026-08-20, board 실측 포함) — `v01_symmetric_matched`, §7 "최종 결론
  갱신" 및 "v01_symmetric_matched ablation (2026-08-19)" §6/§6-1 참조.** 결과: DAIR을 빼는 것
  자체는 오히려 손실을 7.45%p 악화시켰다 — "DAIR이 근본 원인"이라는 가설과 반대 방향. 단,
  `TensorSpeakerOverlayAug`가 symmetric 트레이너에서 구조적으로 함께 빠지는 잔여 confound가
  있어, 이제는 **"하이퍼파라미터·augmentation 전부 동일, DAIR만 on/off"인 진짜 strict
  ablation**이 새로운 후속 과제로 남는다(symmetric 트레이너의 DAIR 모드 크래시 버그를 먼저
  고쳐야 함).
- **seed 반복 실험**: 지금까지의 비교는 각 조건당 1회 학습 — 결과가 seed 변동성 범위 안인지 확인.
  "같은 recipe 계열"로 분류했던 v01(-33.1%)과 v01_dair(-20.7%)가 서로 다른 학습 run으로 이만큼
  (12.4%p) 차이가 난 것이 확인돼(§6) 이 실험의 우선순위가 이전보다 높아졌다.
- **model.2/4 spec_demarc_tensors subset 탐색** (`novatek/convert/rear_truck_v01_320_ablation/`,
  §"spec_demarc_tensors 최소 subset 탐색" 참조): v01_symmetric이 저비용 3텐서 그대로 -7.1%를
  달성했으므로 실사용 후보 선정 관점에서 **우선순위가 크게 낮아졌다** — 삭제하지 않고 과거
  조사/선택적 후속 연구로 남긴다.
- **QAT 재검토** (D-010): production조차 -16.5%로 애초 기준(">2% 드랍이면 QAT 필요")을 넘는다는
  근거는 이미 확보돼 있다 — 10% 이내를 프로덕션 배포 기준으로 확정한다면 QAT 필요성을 다시 논의할
  수 있다. 단, v01_symmetric의 -7.1%는 이미 이 기준을 만족한다.

## v01_symmetric_matched ablation (2026-08-19)

위 "선택적 후속 연구"의 "DAIR 유무 strict ablation" 항목 실행분. `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_matched/`가
바로 이 실험이다 — **학습·Float export·NVT compile/simulate까지는 2026-08-13에 이미 완료돼
있었으나, 이 문서에 기록되지 않은 채 방치돼 있었다.** 아래는 새로 실행한 것 없이 기존 산출물을
읽어 정리한 기록이다. Board(물리 NPU) 실측만 빠져 있다.

### 1. 실험 설계 확인

`configs/experiments/BSD_F606/Rear_Truck/v01_symmetric_matched/exp.yaml`(`_meta_.status:
unvalidated`) 헤더 주석과 `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_dualnms_cfg.yaml`
주석이 설계 의도를 적어뒀다: v01_dair와 optimizer/lr0/lrf/batch/cos_lr/augmentation까지 동일하게
맞추고 트레이너만 symmetric으로 바꾼 strict 비교 실험. (exp.yaml은 `RESULTS.md`를 두 차례
참조하지만 그 파일은 repo 어디에도 존재하지 않는다 — 작성 예정이었던 것으로 보이나 실제로는
안 만들어졌다.)

`args.yaml` 실측으로 확인:

| 항목 | v01_symmetric (원본) | v01_dair | v01_symmetric_matched |
|---|---|---|---|
| optimizer | SGD | AdamW | **AdamW** |
| lr0 | 0.01 | 0.001 | **0.001** |
| batch | 48 | 32 | **32** |
| cos_lr | true | false | **false** |
| patience | 200 | 15 | **15** |
| Trainer | SymmetricTrainer | DairDetectionTrainer | **SymmetricTrainer** |

즉 트레이너(Symmetric, DAIR dual-forward 없음)만 원본 `v01_symmetric`과 같고, 나머지 축은 전부
`v01_dair`에 맞춰져 있다.

**단, exp.yaml 헤더가 스스로 명시하는 잔여 confound가 하나 있다**: `v01_dair`는 `dair_params`
(Select 증강 풀 — `TensorSpeakerOverlayAug` 포함, §2에서 이미 큰 기여 요인으로 확인된 바로 그
augmentation)가 활성화돼 있다(`dair_params: {dair_method_mode: or, ...}`). `v01_symmetric_matched`는
이 augmentation을 **의도적으로 뺀 게 아니라 뺄 수밖에 없었다** — 2026-08-12 스모크 테스트에서
symmetric 트레이너에 `dair_params`를 그대로 넣으면 DAIR consistency loss 경로에서
`RuntimeError: expand(...)`로 즉시 크래시하는 게 확인됐다(`projects/src/utils/loss.py:428`,
`fraction=0.01 epochs=1`로 재현). 즉 symmetric 트레이너의 DAIR 모드 자체가 이 repo에서 검증된
적 없는 미완성 경로라, "하이퍼파라미터만 다르고 나머지는 완전히 동일"이 아니라 **"하이퍼파라미터는
동일, `TensorSpeakerOverlayAug`를 포함한 배치 단위 augmentation 축은 여전히 v01_dair에만 있음"**
상태로 실험이 진행됐다.

### 2. 학습 완료 확인

`runs/detect/BSD_F606/Rear_Truck/v01_symmetric_matched/`: `results.csv` 마지막 행 = epoch 222
(patience=15로 종료), `weights/best.pt`·`last.pt` 존재. `weights/best.onnx`도 존재(2026-08-13
14:38 — `tools/export_custom.py -i 320 -y 320`, `onnx_v01_dualnms_cfg.yaml`과 동일 export
설정으로 생성된 것으로 config 주석에 기록됨).

### 3. Float ONNX 평가 (기존 실행분, 2026-08-13 05:39)

`runs/_eval_scratch/results/onnx_v01_symmetric_matched_dualnms/onnx_v01_symmetric_matched_dualnms_cfg/20260813_053917/eval.log`,
GT 6,152장·dual-threshold NMS — 다른 모델들과 완전히 동일한 조건.

```
precision mode
       class  IoU@50-95     IoU@50     IoU@75
--------------------------------------------------
         bus      0.702      0.862      0.797
         car      0.557      0.740      0.605
  motorcycle      0.460      0.745      0.478
      person      0.412      0.691      0.426
       truck      0.607      0.731      0.666
         all      0.547      0.754      0.594
```

**Float ONNX mAP50-95 = 0.547, mAP50 = 0.754.**

### 4. NVT 3-tensor compile + simulator (기존 실행분, 2026-08-13 14:5x)

`novatek/convert/rear_truck_v01_symmetric_matched_320/`에 다른 모델들과 동일한 3-tensor baseline
(`spec_demarc_tensors`: Softmax(dfl)/Sigmoid/dfl_conv 3개만 지정)로 compile·simulate·
visualize(`sim_comparison_*.jpg`)까지 이미 실행되어 있다.

- **Model size**: `nvt_model.bin` = 4,711,336 bytes (4.71MB) — 다른 3-tensor baseline들과 동일
  (아키텍처가 같으므로 예상대로).
- **Latency/DRAM (2026-08-21 재확인, 실측)**: `run_compiler.sh rear_truck_v01_symmetric_matched_320`
  재실행 stdout에서 직접 확인 — `Total time(ms): 6.36765`, `Total Bw(bytes): 36540852`
  (36.54MB) — 다른 3-tensor baseline들과 완전히 동일.

`simtool/layer_diff_1_..._0126.jpg.txt`(다른 모델들과 동일 이미지)에서 `spec_demarc_tensors` 3개
텐서를 직접 파싱:

```
Softmax__model.22_dfl_Softmax_output_0_Y   cosine( 0.912626 )
Sigmoid_output0_Y                          cosine( 0.947291 )
Conv__model.22_dfl_conv_Conv_output_0_Y    cosine( 0.949030 )
```

### 5. 종합 비교표

| Model | Float mAP50-95 | Softmax(dfl) cosine | Sigmoid cosine | dfl_conv cosine | Board mAP50-95 | Quant loss |
|---|---:|---:|---:|---:|---:|---:|
| v01_symmetric | 0.561 | 0.985 | 0.982 | 0.993 | 0.521 | -7.1% |
| v01_dair | 0.531 | 0.926 | 0.952 | 0.974 | 0.421 | -20.7% |
| **v01_symmetric_matched** | **0.547** | **0.913** | **0.947** | **0.949** | **0.393** | **-28.15%** |
| v02 | 0.536 | 0.906 | 0.924 | 0.966 | 0.435 | -18.8% |
| 기존 v01 | 0.543 | — | — | — | 0.363 | -33.1% |

**Board 실측 확정 (2026-08-20)**: `runs/_eval_scratch/board_dumps/v01_symmetric_matched/v01_symmetric_matched_output.json`
(6,152장, stale-model 검증 완료 — 다른 5개 모델 dump와 jaccard overlap 전부 0.0000~0.0001 수준,
우연 일치 8~9건뿐). `merge_board_dump.py` → `board_v01_symmetric_matched_3tensor_cfg.yaml` →
`customEval.py` (`runs/_eval_scratch/results/board_v01_symmetric_matched_3tensor/.../20260820_041811/eval.log`)
결과: Board mAP50-95 = **0.393**, Float ONNX 0.547 대비 절대손실 -0.154, 상대손실 **-28.15%**.
**`v01_dair`(-20.7%)보다도 나쁘다** — 5개 모델 중 기존 `v01`(-33.1%) 다음으로 손실이 크다.

### 6. 해석 (2026-08-20 실측 반영 — 더 이상 잠정 아님)

`v01_symmetric_matched`는 DAIR dual-forward도 없고 `TensorSpeakerOverlayAug`도 (뺀 게 아니라
**애초에 못 넣어서**) 없는데도, hyperparameter를 `v01_dair` 것으로 바꾸자마자 시뮬레이터 cosine이
`v01_symmetric`(0.985/0.982/0.993)에서 `v01_dair`/`v02` 수준(0.91~0.95대)으로 **완전히
무너졌다** — `v02`(DAIR 트레이너 유지, augmentation만 제거)보다도 딱히 낫지 않다(Softmax는
`v02`보다 근소 우위, dfl_conv는 오히려 열세).

이는 §2에서 이미 확인된 "`TensorSpeakerOverlayAug`가 큰 기여 요인"이라는 결과와 **모순되지
않으면서 한 걸음 더 나아간다**: 그 augmentation과 DAIR dual-forward 학습 방식을 **둘 다** 제거한
상태에서도, hyperparameter(optimizer=AdamW/lr0=0.001/batch=32/cos_lr=false)만 맞추면 quantization
민감도가 DAIR 계열 수준으로 돌아온다. 즉 §5 "현재 해석"이 확정하지 않았던 질문 — *"v01_symmetric이
좋은 이유가 DAIR/augmentation 제거 때문인지 다른 hyperparameter 때문인지"* — 에 대해,
**hyperparameter(optimizer/lr0/batch/cos_lr) 쪽에 힘을 싣는 증거가 하나 더 늘었다.**

**`v01_dair`와의 직접 비교(하이퍼파라미터가 완전히 같은 유일한 쌍)는 실측으로도 방향이 반대로
확정됐다** — 이 실험이 원래 답하려던 질문("DAIR을 빼면 나아지는가?")에 대한 답은 **아니다,
오히려 나빠진다**이다:

| | v01_dair | v01_symmetric_matched | 우세 |
|---|---:|---:|---|
| Float mAP50-95 | 0.531 | **0.547** | matched (+0.016) |
| Softmax(dfl) cosine | **0.926** | 0.913 | v01_dair |
| Sigmoid cosine | **0.952** | 0.947 | v01_dair |
| dfl_conv cosine | **0.974** | 0.949 | v01_dair (뚜렷) |
| **Board mAP50-95** | **0.421** | 0.393 | **v01_dair** |
| **Quant loss (실측)** | **-20.7%** | **-28.15%** | **v01_dair (7.45%p 더 우수)** |

Float 성능은 DAIR을 뺀 쪽(matched)이 살짝 낫지만, cosine과 **실측 board 결과 둘 다** DAIR을
쓴 쪽(`v01_dair`)이 낫다. 하이퍼파라미터가 완전히 동일한 이 쌍만 놓고 보면 **"DAIR을 빼는 것
자체가 quantization을 개선한다"는 근거는 없다 — 오히려 확실히 악화시킨다(-7.45%p).** 위
문단의 "hyperparameter가 유력하다"는 해석과 모순되지 않는다 — `v01_symmetric`의 절대적 우위는
여전히 hyperparameter 축(+21.05%p, §6-1)이 압도적으로 크게 설명한다. 다만 "DAIR 유무" 축을
독립적으로 떼어보면 **중립이 아니라 DAIR 쪽이 약간 유리한 방향**이라는 게 이번 실측으로
새로 확정된 사실이다 — 두 축이 서로 다른 방향을 가리키는 것이지, 같은 방향이 아니다(§6-1
갱신 참조).

**확정된 것과 남은 것**: (1) board 실측으로 위 표의 손실(%) 자체는 더 이상 잠정이 아니다. (2)
§1에서 밝힌 잔여 confound(`dair_params`/`TensorSpeakerOverlayAug`가 `v01_dair`에는 있고
`v01_symmetric_matched`에는 구조적으로 없음)는 여전히 남아있다 — 즉 "DAIR 쪽이 유리하다"는
관측이 DAIR dual-forward 메커니즘 자체 때문인지, 아니면 그 메커니즘에 딸려오는
`TensorSpeakerOverlayAug`(§2에서 이미 큰 악영향 요인으로 확인된 바로 그 augmentation) 부재
때문인지는 이 실험 하나로는 분리되지 않는다 — 후자라면 오히려 "그 augmentation이 없으면
손해"라는, §2의 결론과 정반대 방향의 새로운 퍼즐이 생긴다. strict DAIR on/off ablation(§"선택적
후속 연구")이 여전히 필요한 이유다.

### 6-1. 세 모델을 두 단계로 분해 (`v01_dair` → `v01_symmetric_matched` → `v01_symmetric`) — board 실측으로 확정

`v01_symmetric_matched`가 두 원본 사이의 중간 지점 역할을 하므로, `v01_dair`(-20.7%)와
`v01_symmetric`(-7.1%) 사이의 격차를 "DAIR 유무만 바뀐 구간"과 "hyperparameter만 바뀐 구간"
둘로 쪼갤 수 있다. **2026-08-20 board 실측으로 cosine 기반 잠정치를 실제 quant loss(%)로
교체한다:**

| | dair→matched (**DAIR만 제거**, hyperparameter 동일) | matched→symmetric (**hyperparameter만 변경**, DAIR 계속 미사용) |
|---|---:|---:|
| Float mAP50-95 | +0.016 | +0.014 |
| Softmax(dfl) cosine | -0.013 | +0.072 |
| **Board mAP50-95** | **-0.028** (0.421→0.393) | **+0.128** (0.393→0.521) |
| **Quant loss (%p, 실측)** | **-7.45%p 악화** (-20.7%→-28.15%) | **+21.05%p 개선** (-28.15%→-7.1%) |

**cosine이 예측한 방향(둘 다)은 실측과 일치했다** — DAIR-제거 구간은 소폭이 아니라 상당히
악화됐고(-7.45%p), hyperparameter 구간은 압도적으로 개선됐다(+21.05%p, 전체 격차 13.6%p의
1.5배 이상 — 즉 hyperparameter 구간이 DAIR-제거 구간의 손해를 상쇄하고도 남는다). **결론
(확정)**: `v01_symmetric`이 `v01_dair`보다 quantization을 훨씬 잘 버틴 이유는 **DAIR을 안 써서가
아니라 optimizer/lr0/batch/cos_lr 차이 때문이다** — 이 방향은 확정이다. 다만 DAIR 유무 축 자체는
"중립"이 아니라 **DAIR을 쓰는 쪽이 오히려 유리한 방향**이라는, 애초 가설(DAIR이 quantization에
나쁘다)과 반대되는 새 사실이 board 실측으로 추가됐다 — §6 하단의 confound(TensorSpeakerOverlayAug)
논의 참조.

### 7. 남은 작업 (2026-08-21 기준, 전부 완료)

1. ~~Board 실측~~ — **완료 (2026-08-20)**. §5/§6/§6-1 참조.
2. ~~Latency/DRAM 재확인~~ — **완료 (2026-08-21)**. §4 참조 (`Total time(ms): 6.36765`,
   `Total Bw(bytes): 36540852`).
3. ~~"최종 결론" §5 갱신 + README.md 반영~~ — **완료**. "최종 결론" §7("최종 결론 갱신 —
   v01_symmetric_matched board 실측 반영, 2026-08-20") 및
   `configs/experiments/BSD_F606/Rear_Truck/README.md` 참조.

남은 것은 이 실험 자체가 아니라 §6 하단에서 확인된 **잔여 confound**(`TensorSpeakerOverlayAug`가
symmetric 트레이너에서 구조적으로 함께 빠짐) 해소용 후속 과제뿐이며, 그건 "선택적 후속 연구"
목록에 별도로 기록돼 있다(symmetric 트레이너의 DAIR 모드 크래시 버그 수정이 선행 조건).
