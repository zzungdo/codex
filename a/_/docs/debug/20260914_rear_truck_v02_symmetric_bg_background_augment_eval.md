# Rear_Truck `v02_symmetric_bg` — 배경(무객체) 이미지 추가 학습 평가 (2026-09-14)

- **배경**: `v01_symmetric`이 릴리즈 후보로 확정된 뒤(`docs/debug/20260902_rear_truck_release_candidate_selection.md`),
  주말(2026-09-11 금 17:16 ~ 2026-09-14 월 08:00) 유휴 GPU 시간을 활용해 `v01_symmetric`에
  배경 이미지(`/mnt/NAS_1/01_DATASET/98_RELEASE_back_up/BackGround/CAR/REAR`, 3,772장, 라벨 없음)를
  추가하면 오탐(특히 §3-2에서 확인된 `v02_dair`류의 person/car precision 저하 패턴)이 줄어드는지
  탐색적으로 확인한 실험.
- **결론(선반영)**: **도움이 안 됐다.** Float 절대 성능·quantization loss 둘 다 `v01_symmetric`보다
  나쁘고, 공식 acceptance target(≤1%, std+ultra mAP50)도 미달이다. `v01_symmetric`의 릴리즈 확정
  결정(20260902 문서)은 바뀌지 않는다 — 이 문서는 그 결정에 대한 새로운 반증도, 강화도 아니고
  "배경 데이터를 더해본 별도 탐색"의 기록이다.

---

## 0. 실험 정의 (SOP §0)

| 항목 | 값 |
|---|---|
| 실험 이름 | `v02_symmetric_bg` |
| 목적/가설 | 배경(무객체) 이미지 추가로 오탐(false positive)이 줄어드는지 확인 |
| baseline 모델 | `v01_symmetric`(현재 릴리즈 확정본) |
| baseline 대비 변경점 | **(1) data**: `1t_rear_cnr_postprocess.yaml` → `1t_rear_cnr_postprocess_bg.yaml`(배경 3,772장 추가, 95,586→99,358장) / **(2) epochs**: 500 → 600 (주말 가용시간에 맞춘 조정, 하이퍼파라미터 튜닝 목적 아님) — **두 가지가 동시에 바뀌어 순수 ablation이 아니다(§6 참조)** |
| dataset / split | `configs/datasets/1t_rear_cnr_postprocess_bg.yaml`, val은 `v01_symmetric`과 동일(`pure/val`, 7,894장) |
| architecture | `projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml` (동일) |
| Trainer / Validator | `SymmetricTrainer` (동일) |
| imgsz / batch | 320 / 48 (동일) |
| optimizer / lr0 / lrf / cos_lr | SGD / 0.01 / 0.01 / true (동일 — 단 epochs가 달라져 cos_lr 감쇠 곡선 자체는 다름) |
| DAIR 여부 | 없음(symmetric, 동일) |
| seed / pretrained | 동일(0 / true) |
| epochs / patience | 600 / 15 (patience는 동일) |
| 기타 augmentation | 동일 (config: `configs/experiments/BSD_F606/Rear_Truck/v02_symmetric_bg/exp.yaml`) |

---

## 1. 학습

2026-09-11 17:16 KST 시작 → epoch 318에서 EarlyStopping(patience=15, best epoch 303) →
2026-09-12 20:07 KST 종료, **26.7시간**(계획한 62.7시간의 42%만 사용).

`best.pt` internal val(7,894장, PyTorch 직접, `SymmetricTrainer`의 최종 val 그대로 — SOP §4에
해당):

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bus | 0.936 | 0.793 | 0.877 | 0.690 |
| car | 0.724 | 0.705 | 0.761 | 0.563 |
| motorcycle | 0.835 | 0.651 | 0.761 | 0.461 |
| person | 0.833 | 0.592 | 0.710 | 0.425 |
| truck | 0.927 | 0.667 | 0.801 | 0.609 |
| **all** | **0.851** | **0.681** | **0.782** | **0.550** |

---

## 2. ONNX export (SOP §5)

`tools/export_symmetric.py -i 320 -y 320 --skip_postprocess` — `onnx.checker` PASS, input
`images[1,3,320,320]`, output `output0[1,5,2100]`(cls) + `433[1,4,2100]`(raw DFL box). 체크리스트
전부 통과.

---

## 3. Float ONNX 평가, 6,152장 (SOP §6)

GT/전처리는 `v01_symmetric`과 완전히 동일(`runs/_eval_scratch/ipu_gt`). std NMS(`iou_thres=0.45`)와
dual-NMS(board 재현, `min_overlap_thres=0.6/union_overlap_thres=0.3`) 둘 다, conf=0.001
(`tools/debug/inf_conf001.py`), pycocotools(`projects/metrics/evaluator.py`)와 ultralytics
(`tools/debug/ultralytics_style_ap.py`, `ap_per_class`) 둘 다로 계산(4-combo, §7 참조).

Config: `runs/_eval_scratch/configs/onnx/onnx_v02_symmetric_bg_{stdnms,conf001_dualnms}_cfg.yaml`.

---

## 4. NVT conversion (SOP §7, 3-tensor baseline)

`novatek/convert/rear_truck_v02_symmetric_bg_320/`(신규 디렉토리, 기존 디렉토리 복사/수정 없음).
calibration 이미지(`input/data/REAR_TRUCK_V01_320/`)와 `input/model` 트리는 `rear_truck_v02_320`과
하드링크 공유(`cp -al`, 기존 관례와 동일), 새 pattern(`REAR_TRUCK_V02_SYMMETRIC_BG_320`)만 추가.
소스 ONNX와 `deploy_float.onnx`의 md5 일치 확인(`a2347f63e7aba8e83be3bf6c6985000a`).

| 항목 | 값 | v02 baseline 참고값 |
|---|---:|---:|
| Latency | 6.368 ms | 6.368 ms |
| DRAM bandwidth | 36.54 MB | 36.54 MB |
| Model size | 4.71 MB | 4.71 MB |

아키텍처가 동일하므로 예상대로 v02와 사실상 동일 — regression 없음.

---

## 5. NVT simulator (SOP §8, 진단용)

단일 이미지(`KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC1_PGL_DVR_NC22015_0126.jpg`) cosine:

| Tensor | `v02_symmetric_bg` | `v01_symmetric` | `v02_dair` |
|---|---:|---:|---:|
| Softmax(dfl) | **0.9938** | 0.9850 | 0.9857 |
| Sigmoid [OUT] | **0.9713** | 0.9824 | 0.9876 |
| dfl_conv [OUT] | **0.9976** | 0.9933 | 0.9945 |

Softmax·dfl_conv는 오히려 더 좋지만 **Sigmoid(classification score)만 뚜렷이 낮다** — 진단
신호일 뿐 최종 판정 근거는 아니지만(SOP 주의사항), §7의 실측 board 손실과 방향이 일치한다.

---

## 6. Board 실측 (SOP §9~11)

물리 보드 덤프(사용자 제공, `runs/_eval_scratch/board_dumps/v02_symmetric_bg/v02_symmetric_bg_output.json`,
6,152장, 211,968 detections) → `merge_board_dump.py` → 병합.

**Stale-model 체크(SOP §9, 필수)**: 기존 9개 모델(`v01`/`v01_dair`/`v02`/`v02_dair`/
`v01_symmetric`/`v01_symmetric_matched`/`v02_symmetric_sgd_lr0001`/`v02_symmetric_adamw_lr0001`/
`production`) 병합 dump와 detection-key(`name,bbox,score,category`) jaccard overlap 전부
**0.0000** — stale 아님, 정상 신규 결과.

Config: `runs/_eval_scratch/configs/board/board_v02_symmetric_bg_3tensor_cfg.yaml`.

---

## 7. Quantization loss (SOP §12, 4-combo)

| Combo | Float mAP50 / mAP50-95 | Board mAP50 / mAP50-95 | loss(mAP50) | loss(mAP50-95) |
|---|---:|---:|---:|---:|
| dual+pycoco | 0.761 / 0.543 | 0.734 / 0.507 | -3.55% | -6.63% |
| std+pycoco | 0.762 / 0.538 | 0.734 / 0.507 | -3.67% | -5.76% |
| dual+ultra | 0.782 / 0.567 | 0.767 / 0.543 | -1.88% | -4.25% |
| **std+ultra(공식 acceptance)** | **0.779 / 0.559** | **0.767 / 0.543** | **-1.55%** | **-2.83%** |

---

## 8. 기존 모델과 비교 (SOP §13~14)

공식 acceptance 지표(mAP50, std+ultra) 기준:

| Model | Float mAP50 | Board mAP50 | Quant loss | 판정(≤1%) |
|---|---:|---:|---:|---|
| `v01_symmetric`(릴리즈 확정본) | 0.796 | 0.799 | **+0.41%** | **통과** |
| `v02_dair` | (미상, §1 참조) | (미상) | -1.22% | 미달 |
| **`v02_symmetric_bg`** | **0.779** | **0.767** | **-1.55%** | **미달** |

**`v02_symmetric_bg`는 절대 성능(Float mAP50 -1.7pt)과 quant loss(-1.55% vs +0.41%) 둘 다
`v01_symmetric`보다 나쁘고, `v02_dair`보다도 근소하게 더 나쁘다.** 배경 데이터로 오탐이 줄었는지
직접 검증하지는 않았지만(클래스별 precision/recall 분해는 이번 범위 밖 — 필요하면 후속으로),
적어도 이 지표들에서는 이점이 드러나지 않는다.

**matched control 경고(SOP §0/§14)**: 이 실험은 배경 데이터 추가와 **동시에** epochs를
500→600으로 바꿔 cos_lr 감쇠 곡선 자체도 달라졌다(주말 시간 제약 때문 — `configs/experiments/
BSD_F606/Rear_Truck/v02_symmetric_bg/exp.yaml` 헤더 코멘트에 이미 기록된 우려). 따라서 **"배경
데이터 자체의 순수 효과"와 "LR 스케줄 차이 효과"가 이 한 번의 실험으로 분리되지 않는다.** 배경
데이터만의 효과를 알고 싶다면 `v01_symmetric`과 epochs/cos_lr까지 완전히 동일하게 맞춘 재학습이
필요하다(이번 범위 밖).

---

## 9. 결론 및 다음 단계

- **릴리즈 후보 결정에는 영향 없음** — `v01_symmetric`이 여전히 유일하게 target을 통과하는
  모델이다(`docs/debug/20260902_rear_truck_release_candidate_selection.md`의 결정 유지).
- **배경 데이터 추가는 이번 조합(+ epochs 600)에서는 이점이 없었다.** 오탐 감소라는 원래 가설을
  뒷받침하는 증거를 찾지 못했다(단, 클래스별 precision 분해는 하지 않았음 — 필요하면 후속 확인).
- 배경 데이터의 순수 효과가 궁금하면, epochs/cos_lr을 `v01_symmetric`과 완전히 맞춘 재학습으로
  재검증할 것. 이번 문서·산출물 범위에서는 하지 않는다.

## 파일 위치

- Config: `configs/experiments/BSD_F606/Rear_Truck/v02_symmetric_bg/exp.yaml`,
  `configs/datasets/1t_rear_cnr_postprocess_bg.yaml`
- 학습 산출물: `runs/detect/BSD_F606/Rear_Truck/v02_symmetric_bg/`
- NVT: `novatek/convert/rear_truck_v02_symmetric_bg_320/`
- 평가: `runs/_eval_scratch/{configs/onnx,configs/board,board_dumps,merged}/v02_symmetric_bg*`
