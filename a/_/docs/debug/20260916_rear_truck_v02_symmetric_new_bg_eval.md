# Rear_Truck `260914_v02_symmetric_new_bg` — 새 배경셋 재시도 평가 (2026-09-15/16)

- **배경**: `v01_symmetric`이 릴리즈 후보로 확정된 뒤(`docs/debug/20260902_rear_truck_release_candidate_selection.md`),
  이전 배경 데이터 실험(`v02_symmetric_bg`, 구 배경셋 `BackGround/CAR/REAR` 3,772장 + epochs 600 —
  `docs/debug/20260914_rear_truck_v02_symmetric_bg_background_augment_eval.md`)이 도움이 안 됐던 것을
  이어받아, **완전히 다른 배경셋**(`/mnt/NAS_1/1T_REAR_BG_jpg`, 2,210장, 라벨 없음)으로 재시도한 실험.
  이전 배경셋(`background_rear/`)은 여기서 전혀 참조하지 않는다(사용자 지시, 2026-09-14 — 두 배경셋을
  섞지 않고 완전히 교체).
- **matched control 개선**: `v02_symmetric_bg`는 배경 데이터 추가와 동시에 epochs를 500→600으로 바꿔
  순수 ablation이 아니었다(§0/§8 경고). 이번엔 epochs를 `v01_symmetric`과 동일한 500으로 유지해 그
  confound를 줄였다.
- **결론(선반영)**: **이번에도 도움이 안 됐다.** internal val·Float ONNX(6,152장, std+ultra) 절대
  성능이 `v01_symmetric`보다 전 지표에서 낮고, board 실측 없이도(§4) 이미 Float 단계에서 뒤처진다 —
  `v02_symmetric_bg`와 동일한 패턴 재확인. **board 실측(§5)은 아직 수행되지 않았다** — 이 문서는
  거기까지만의 중간 기록이다.

---

## 0. 실험 정의 (SOP §0)

| 항목 | 값 |
|---|---|
| 실험 이름 | `260914_v02_symmetric_new_bg` |
| 목적/가설 | 배경(무객체) 이미지 추가로 오탐(false positive)이 줄어드는지 확인 (새 배경셋, matched epochs) |
| baseline 모델 | `v01_symmetric`(현재 릴리즈 확정본) |
| baseline 대비 변경점 | **data만 변경**: `1t_rear_cnr_postprocess.yaml` → `1t_rear_cnr_postprocess_bg_v2.yaml`(배경 2,210장 추가) — epochs/patience(500/15) 등 나머지는 `v01_symmetric`과 완전히 동일, 조정하지 않음 |
| dataset / split | `configs/datasets/1t_rear_cnr_postprocess_bg_v2.yaml`, val은 `v01_symmetric`과 동일(7,894장) |
| architecture | `projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml` (동일) |
| Trainer / Validator | `SymmetricTrainer` (동일) |
| imgsz / batch | 320 / 48 (동일) |
| optimizer / lr0 / lrf / cos_lr | SGD / 0.01 / 0.01 / true (동일) |
| DAIR 여부 | 없음(symmetric, 동일) |
| seed / pretrained | 동일(0 / true) |
| epochs / patience | 500 / 15 (동일 — `v02_symmetric_bg`와 달리 조정하지 않음) |
| exp config | `configs/experiments/BSD_F606/Rear_Truck/260914_v02_symmetric_new_bg/exp.yaml` |

---

## 1. 학습

2026-09-14 시작 → epoch 249에서 EarlyStopping(patience=15, best epoch 234) →
2026-09-15 07:00 KST 종료, 19.211시간 소요.

`best.pt` internal val(7,894장, PyTorch 직접, `SymmetricTrainer`의 최종 val 그대로 — SOP §4):

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bus | 0.935 | 0.771 | 0.877 | 0.694 |
| car | 0.734 | 0.716 | 0.769 | 0.567 |
| motorcycle | 0.836 | 0.636 | 0.758 | 0.458 |
| person | 0.834 | 0.591 | 0.710 | 0.425 |
| truck | 0.888 | 0.653 | 0.781 | 0.597 |
| **all** | **0.845** | **0.673** | **0.779** | **0.548** |

`v01_symmetric` internal val 대비(같은 7,894장 val split, `results.csv` mAP50 best epoch=390 기준
— `v01_symmetric`은 학습 로그가 남아있지 않아 클래스별 분해는 불가, all row만 비교):

| | v01_symmetric | new_bg | diff |
|---|---:|---:|---:|
| Precision | 0.863 | 0.845 | -0.018 |
| Recall | 0.695 | 0.673 | -0.022 |
| mAP50 | 0.799 | 0.779 | -0.020 |
| mAP50-95 | 0.568 | 0.548 | -0.020 |

**internal val 단계에서 이미 전 지표 하락** — `v02_symmetric_bg`와 같은 방향.

---

## 2. ONNX export (SOP §5)

`tools/export_symmetric.py -w runs/detect/BSD_F606/Rear_Truck/260914_v02_symmetric_new_bg/weights/best.pt
-i 320 -y 320 --skip_postprocess -s .../weights/best.onnx` — onnx.checker PASS, input
`images[1,3,320,320]`, output `output0[1,5,2100]`(cls) + `433[1,4,2100]`(raw DFL box). 2026-09-15.

---

## 3. Float ONNX 평가, 6,152장 (SOP §6)

GT/전처리는 `v01_symmetric`과 완전히 동일(`runs/_eval_scratch/ipu_gt`). 공식 acceptance 조합만 수행
(std NMS `iou_thres=0.45`, conf=0.001 `tools/debug/inf_conf001.py`, ultralytics `ap_per_class`
— std+ultra). dual-NMS/pycocotools 조합은 board dump 확보 후 필요시 추가.

Config: `runs/_eval_scratch/configs/onnx/onnx_v02_symmetric_new_bg_stdnms_conf001_cfg.yaml`.

**std+ultra, 6,152장**:

| | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| bus | 0.903 | 0.818 | 0.877 | 0.706 |
| car | 0.721 | 0.730 | 0.767 | 0.567 |
| motorcycle | 0.877 | 0.656 | 0.767 | 0.475 |
| person | 0.844 | 0.593 | 0.704 | 0.418 |
| truck | 0.958 | 0.686 | 0.754 | 0.611 |
| **all** | **0.861** | **0.697** | **0.774** | **0.555** |

`v01_symmetric` Float(std+ultra, 20260902 문서 §3-7)과 비교:

| | v01_symmetric | new_bg | diff |
|---|---:|---:|---:|
| mAP50 | 0.796 | 0.774 | **-0.022** |
| mAP50-95 | 0.579 | 0.555 | **-0.024** |

Float 단계에서도 절대 성능이 뚜렷이 낮다 — internal val과 같은 방향/크기.

---

## 4. NVT conversion (SOP §7, 3-tensor baseline)

`novatek/convert/rear_truck_v02_symmetric_new_bg_320/`(신규 디렉토리). calibration 이미지
(`input/data/REAR_TRUCK_V01_320/`)와 `input/model` 트리는 `rear_truck_v02_symmetric_bg_320`과
하드링크 공유(`cp -al`, 기존 관례와 동일), 이전 pattern(`REAR_TRUCK_V02_SYMMETRIC_BG_320`)의
model/config/output은 제거하고 신규 pattern(`REAR_TRUCK_V02_SYMMETRIC_NEW_BG_320`)만 추가.
소스 ONNX와 `deploy_float.onnx`의 md5 일치 확인(`b3d9421a9904ef7cd7172e9180447b77`).

| 항목 | 값 | v02 baseline 참고값 |
|---|---:|---:|
| Latency | 6.368 ms | 6.368 ms |
| DRAM bandwidth | 36.54 MB | 36.54 MB |
| Model size | 4.71 MB | 4.71 MB |

아키텍처가 동일하므로 예상대로 baseline과 사실상 동일 — regression 없음.

Board binary: `novatek/convert/rear_truck_v02_symmetric_new_bg_320/nvtai_tool/output/REAR_TRUCK_V02_SYMMETRIC_NEW_BG_320/sdk/nvt_model.bin`
(4,711,336 bytes, md5 `eb70cabaa8550f7887d71256253d569e`).

---

## 5. NVT simulator (SOP §8, 진단용)

단일 이미지(`KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC1_PGL_DVR_NC22015_0126.jpg`) cosine:

| Tensor | `new_bg` | `v01_symmetric` | `v02_symmetric_bg` | `v02_dair` |
|---|---:|---:|---:|---:|
| Softmax(dfl) | 0.9904 | 0.9850 | 0.9938 | 0.9857 |
| Sigmoid [OUT] | **0.9642** | 0.9824 | 0.9713 | 0.9876 |
| dfl_conv [OUT] | 0.9956 | 0.9933 | 0.9976 | 0.9945 |

Softmax·dfl_conv는 준수하지만 **Sigmoid(classification score) cosine이 지금까지 비교한 모델 중
가장 낮다** — 진단 신호일 뿐 최종 판정 근거는 아니지만(SOP 주의사항), §1/§3의 절대 성능 하락과
방향이 일치한다.

---

## 6. Board 실측 — **미수행 (다음 단계)**

물리 보드 덤프가 아직 없다. 확보되면:

1. `runs/_eval_scratch/board_dumps/v02_symmetric_new_bg/v02_symmetric_new_bg_output.json`에 보존
2. stale-model 체크(SOP §9, 필수 — 기존 모델들과 detection-key jaccard overlap 확인)
3. `merge_board_dump.py` → 병합
4. `runs/_eval_scratch/configs/board/board_v02_symmetric_new_bg_3tensor_cfg.yaml` 작성
5. quantization loss(std+ultra, 공식 acceptance) 계산 → `v01_symmetric`(+0.41%)과 최종 비교

---

## 7. 중간 결론

**Float 단계까지의 절대 성능 하락(mAP50 -2.2pt, mAP50-95 -2.4pt, std+ultra 6,152장 기준)과
single-image cosine 진단(Sigmoid 0.9642, 지금까지 최저)이 모두 같은 방향을 가리킨다** —
`v02_symmetric_bg`와 마찬가지로 이번 새 배경셋 재시도도 `v01_symmetric` 대비 개선을 보여주지
못했다. Board 실측 없이 최종 판정(quant loss, 공식 acceptance)을 내릴 수는 없지만, Float 단계에서
이미 baseline보다 2pt 이상 뒤처지는 모델이 quantization을 거쳐 그 격차를 뒤집을 가능성은 낮다.
`v01_symmetric`의 릴리즈 확정 결정은 바뀌지 않는다.
