# FRONT Golden Reference for REAR ≤1%

## 1. 목적

Rear_Truck의 Float ONNX → NPU 변환 시 mAP50-95 relative loss를 ≤1%로 줄이기 위해,
과거 실제 Release된 BSD_F606 Front_Truck v01을 Golden Reference로 삼아
"FRONT는 왜 ONNX → NVT/NPU 변환 이후에도 성능이 거의 유지되었는가"를 역추적하고,
그 원인 중 REAR에 적용 가능한 요소를 찾는다.

FRONT 자체 개선은 목적이 아니다. FRONT config를 REAR에 그대로 이식하는 것도 목적이 아니다
(과거 시도에서 오히려 성능이 악화된 기록이 있음).

## 2. REAR 현재 상태

- Acceptance target: Float ONNX → Board NPU **mAP50-95 relative loss ≤ 1%**
- relative loss 계산식: `(Board mAP50-95 / Float ONNX mAP50-95 - 1) × 100`
- 현재 best 모델: `v01_symmetric`
  - Float ONNX mAP50-95 = 0.561
  - Board NPU mAP50-95 = 0.521
  - relative loss = **-7.1%** (target 미달)
- 기존에 이미 조사된 변수: calibration dataset, calib_mode, percentile, spec_demarc_tensors,
  tensor 보호, 학습 recipe, DAIR/symmetric, augmentation 등
  (`docs/debug/20260806_rear_truck_v01_npu_quant_gap.md` 및 관련 debug 문서 참조)

## 3. FRONT 조사 범위

이번 세션의 FRONT는 다음 두 경로로만 고정한다 — 다른 버전/카메라/SoC로 검색을 확장하지 않았다.

- NAS Root: `/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01`
- Dataset Root: `/mnt/HDD_1/BSD_F606_FRONT/dataset`
- Target NPU: **NT98530**로 고정 (NT98529 표기는 과거 문서 오기로 취급, 별도 조사 안 함)
- PT→ONNX source checkpoint 재현/재탐색은 하지 않음 (기존 0814 조사에서 이미 결론 완료)

## 4. FRONT Release Spec

- Version: v01 / Date: 2026-07-06 / Type: YOLOv8-Nano / Color Space: BGR / Task: Object Detection
- Target NPU: NT98530
- Classes (6): bus, car, motorcycle, person, truck, traffic_light
- Image preprocessing (Release 문서 기재 → 실측으로 확인됨, 6절 참조):
  - Crop ROI: x 320~1600, y 0~1024 (1280×1024)
  - Inference size: 640×512 → ONNX input `[1,3,512,640]`

## 5. Reference ONNX

- path: `.../Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx`
- size: 12,071,501 bytes / mtime: 2026-07-07 10:27:08
- sha256: `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0` (직접 재계산, 기존 문서와 일치, 세션 내 재확인)
- opset 13, ir_version 7, producer pytorch 2.5.1
- input: `images [1,3,512,640]` float32
- outputs: `output0 [1,6,6720]` (Sigmoid 직결, `/model.22/Sigmoid`), `433 [1,4,6720]` (DFL conv+reshape, `/model.22/dfl/Reshape_1` — dist2bbox 디코드 이전, postprocess 미포함)
- node count 235, initializer count 127
- op_type 분포: Conv 64, Sigmoid 57, Mul 56, Constant 14, Split 8, Add 6, Concat 15, MaxPool 3, Resize 2, Reshape 8, Transpose 1, Softmax 1 (DFL)
- Stored PT (best.pt/last.pt/pretrained_weight.pt)는 Release ONNX와 대응하지 않음 — 기존 결론(`20260814_front_truck_reference_onnx_vs_best_pt.md`, `..._source_checkpoint_search.md`) 그대로 유지, 이번 세션에서 재검증하지 않음

## 6. Dataset

- root: `/mnt/HDD_1/BSD_F606_FRONT/dataset`
- Train: images 45,254 / labels 45,254 (cxywhn 정규화)
- Validation: images 6,076 / labels 6,076 (cxywhn) / gt 6,076 (픽셀 `class x1 y1 x2 y2`, ROI-crop 좌표계로 보임)
- class count: 6 (Release 문서와 일치)
- Day/Night(29,299/22,031)이 어느 스냅샷(Train/Val/전체) 기준인지 **UNKNOWN** — Train+Val 합계(51,330)와 불일치

## 7. Evaluation Pipeline

- ROI crop `[320.0, 0.0, 1280.0, 1024.0]`은 Release 문서 기재값과 `analysis/metrics_front_*.txt/json`의 `roi` 필드가 정확히 일치함으로 실측 확인
- resize: 1280×1024 → 640×512 (gen_config.txt `preproc/resize`와 ONNX input 일치)
- color: YUV420 입력(`preproc/in/fmt=2`) → BGR 출력(`preproc/out_fmt=3`)
- normalization/mean subtraction: 비활성 (`normalize/en=0`, `meansub/en=0`)
- dtype: UINT8 (`sim_config.txt src/bintype=1`)
- evaluator: `projects/metrics/evaluator.py` 계열 포맷(sensitive/optimal/insensitive)으로 보이나, 정확한 실행 스크립트/config는 NAS 트리에서 찾지 못함
- 정확히 몇 장(6,076 전체 vs subset)을 evaluation에 썼는지는 **미확정** (추정: 6,076 전체)

## 8. Float ONNX Performance

### mAP50
- all: **0.6965** (Release 문서) / 0.697 (실측 `metrics_front_onnx.json` 재계산)
- per-class: bus 0.706, car 0.805, motorcycle 0.668, person 0.691, truck 0.692, traffic_light 0.617

### Precision / Recall (Release 문서 기재, NAS analysis 외부 문서 — 사용자 제공, 2026-08-20 추가)

| Class | mAP50 | Precision | Recall |
|---|---|---|---|
| bus | 0.706 | 0.83 | 0.672 |
| car | 0.805 | 0.845 | 0.781 |
| motorcycle | 0.668 | 0.782 | 0.639 |
| person | 0.691 | 0.805 | 0.633 |
| truck | 0.692 | 0.806 | 0.679 |
| traffic_light | 0.617 | 0.81 | 0.588 |
| **all** | **0.6965** | **0.813** | **0.665** |

이 표는 NAS `analysis/` 산출물이 아니라 사용자가 보유한 원본 Release 문서에서 온 값이며, mAP50 컬럼은
이 세션에서 이미 확인한 값과 완전히 일치한다. Precision/Recall은 NAS analysis 트리에서는 확인되지
않았던 추가 정보다.

### mAP50-95
- Release 문서/NAS metric 파일 자체에는 여전히 **기록 없음** — `analysis/`의 모든 metric 파일이 `AP` 단일 필드(`iou_tp: 0.5`, 즉 mAP50)만 가짐.
- **실험 0 재현치 (2026-08-20, `runs/debug/front_golden_reference_1pct/map50_95_baseline/`)**: onnxruntime으로 Release ONNX를 직접 추론(ROI crop→640×512 resize→BGR/무정규화, dual-threshold NMS로 board와 동일 방식 정렬) — **mAP50-95 = 0.4615**. 재현된 mAP50(0.7177)은 문서값(0.697)과 +0.021 차이로 근접(방법론 정합성의 sanity check). NMS 파라미터(min_overlap=0.6/union_overlap=0.3)는 REAR 문서에서 가져온 값으로 FRONT 자체 확인은 아님 — 근사치로 취급.

## 9. NVT / Board Performance

### mAP50
- all: **0.704** (Release 문서 및 `metrics_front_board.json` 재확인 일치)
- per-class: bus 0.696, car 0.780, motorcycle 0.725, person 0.742, truck 0.687, traffic_light 0.592
- `analysis/iputest/output.json`: board raw detection dump 151,136건 (`bbox/score/name/category/time`) — 온디바이스 NMS/threshold를 거친 최종 detection으로 보이나, 어느 이미지 subset에 대한 것인지와 `metrics_front_board.*`로 이어지는 정확한 파이프라인 스크립트는 확인하지 못함

### mAP50-95
- NAS 자료 자체에는 여전히 UNKNOWN(9절과 동일 이유).
- **실험 0 재현치**: `analysis/iputest/output.json`(151,136 detection, 6,075/6,076장 커버)을 GT(ROI로 클리핑, 8절 참조)와 매칭 — **mAP50-95 = 0.5091**. 재현된 mAP50(0.7293)도 문서값(0.704)과 +0.025 차이로 근접.
- GT를 ROI 경계로 클리핑하지 않고 1차 시도했을 때는 Board mAP50=0.583으로 문서값과 크게 벗어났다 — 네트워크가 보지 못하는 ROI 밖 GT를 그대로 두면 거짓음성으로 잡혀 과소평가됨을 확인 (클리핑이 왜 필요한지의 근거).

## 10. Quantization Loss

### mAP50
- absolute: 0.704 − 0.697 = **+0.007** (Board가 Float보다 높음)
- relative: (0.704 / 0.697 − 1) × 100 = **+1.0%**

### mAP50-95
- **실험 0 재현치**: absolute = 0.5091 − 0.4615 = **+0.0476**, relative = **+10.31%** — Board가 Float보다 높다(REAR가 acceptance 기준으로 삼는 지표에서, FRONT는 손실이 아니라 개선).
- 이로써 **"FRONT가 mAP50-95 기준으로도 ≤1% target을 만족하는가"라는 24절의 핵심 UNKNOWN이 해소되었다**: 최선의 근사 재현 기준으로 FRONT는 ≤1%를 만족할 뿐 아니라 손실 자체가 없다. mAP50(+1.0%)과 같은 방향, 더 큰 폭(+10.31%)으로 나타나 mAP50이 mAP50-95 손실을 가리고 있었다는 우려는 기각된다(오히려 반대 방향으로 강화됨).
- **단서**: 이 수치는 dual-NMS 파라미터(REAR에서 가져옴, FRONT 자체 재확인 아님)와 GT 클리핑 임계값 등 몇 가지 가정 위에서 나온 최선의 근사치다 — mAP50 sanity check(재현 0.7177/0.7293 vs 문서 0.697/0.704, 둘 다 +0.02대 오차로 근접)로 방법론 정합성은 확인했으나, 정확한 수치(+10.31%)보다는 "손실이 없거나 오히려 개선"이라는 정성적 결론에 더 높은 확신을 둔다.

## 11. Calibration Dataset

- gen_config가 가리키는 경로: `/home/ncn/Developments/data/weights/_pipeline_inputs/nt98530_d224k_front/calib/jpg` (+ `ref_img_list.txt`) — 원본 컴파일 머신의 로컬 경로로, 이 host/NAS/repo 어디에도 **존재하지 않음** (0814 문서에서 이미 확인, 재검증 안 함)
- image count: 500 (`ref_data/num=500`, gen_config.txt)
- 실제 파일 내용/순서: **UNKNOWN** (원본 세트 자체가 없어 확인 불가)

## 12. Calibration Preprocessing

- gen_config `[PREPROCESS]` 섹션과 동일할 것으로 추정 (7절과 동일 크롭/리사이즈/색공간/정규화 설정 공유 구조)
- calibration 전용 별도 전처리 차이가 있었는지는 원본 세트 부재로 **확인 불가**

## 13. NVT Production Config

(gen_config.txt 원문 직접 확인)

- compiler: Novatek nvtai_tool (정확한 tool/버전 문자열 없음)
- compiler version: **UNKNOWN**
- chip: 530 (gen_config 경로에 명시, ONNX input과 일치)
- calib_mode: **4** (PERCENTILE류로 추정, REAR 문서의 `calib_mode 4 = PERCENTILE(0.999)`와 동일 코드값)
- percentile: gen_config에 명시적 수치 없음 → **UNKNOWN**
- cle_en: **0** (비활성)
- cal_refine_iter: **0**
- bnscale_en: **0**
- precision: 별도 mixed/high-precision 지정 없음, 16bit 보호 텐서 0개

## 14. spec_demarc_tensors

- gen_config.txt 원문에서 **전체 주석 처리 — 실제로는 미사용**
- 주석으로만 남은 후보: `Conv_/model.22/cv2.0/...` 등 head 근처 6개 텐서
- 즉 FRONT Release는 REAR의 v01_symmetric이 사용 중인 3-tensor(H1-H3, head 근처) 보호보다도 **더 적은(0개) 보호**로 운영되었다

## 15. Compiler

- Release 당시 compiler 버전: **UNKNOWN** (문자열 기록 없음)
- 로컬 재변환 시 생성된 `MODEL_BIN_INFO/VERSION = 2052503042`는 모델 바이너리 포맷 버전이며 컴파일러 툴체인 버전이 아님 — Release 당시와 현재 로컬 컴파일러가 동일한지 확인 불가 (0814 문서에서도 미확인 상태로 남음)
- 따라서 이후 비교에서 compiler version 차이는 별도 변수로 남겨둔다

## 16. Simulator Layer Analysis

- Release NVT bin 자체의 simulator dump(golden_tensor/sim_layer/layer_diff)는 NAS 트리에 **없음** (compile 중간 산출물 미보존)
- 참고 가능한 자료: 동일 Release ONNX + 동일 gen_config를 대체 calibration(Validation 앞 500장)으로 로컬 재변환한 실험(`novatek/convert/front_truck_v01_reference_onnx_local_reconvert_20260814/simtool/layer_diff.txt`)을 실제로 읽음 — **이 값은 Release bin이 아닌 근사치**임에 유의:
  - Softmax(dfl)[OUT] cosine: 0.981978
  - Sigmoid[OUT] cosine: 0.976482
  - dfl_conv[OUT] cosine: 0.992463
  - model.2 블록: 0.9903 ~ 0.9956 범위
  - model.4 블록: 0.9848 ~ 0.9959 범위 (`model.4/m.0/cv1/act` 0.9848, `model.4/m.1/cv1/act` 0.9855가 국소 최저)
- REAR가 겪는 backbone-early cosine 저하(0.96대)까지 떨어지는 지점은 FRONT(근사치)에서 발견되지 않음

## 17. Reference NVT Reproduction

- Reference NVT model: `network/260706_BSD-F606A_Front_v01.bin`, sha256 `fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b` (직접 재계산, 문서와 일치), size 4,758,800 bytes
- latency/DRAM: Release bin 자체 프로파일 로그 없음 → **UNKNOWN**
  - 참고(대체 calibration 로컬 재변환본): latency 19.628ms, DRAM 103,244,064 bytes, model size 4,758,792 bytes — Release bin과 크기 8바이트 차이뿐이나 이 값을 Release 실측치로 대체할 수 없음
- 결론: 원본 500장 calibration set 부재로 **Release NVT bin을 수치적으로 동일하게 재현하는 것은 현재 불가능**. 구조/설정값 확인은 거의 완전하지만 수치 재현에는 근본적 한계가 있다.

## 18. First Divergence

| 단계 | 상태 |
|---|---|
| Reference ONNX | PASS (구조 확인, sha256 일치) |
| Evaluation pipeline | PARTIAL — ROI/resize/color 확인, 정확한 실행 스크립트는 미확인 |
| Calibration dataset | **FAIL (NOT FOUND)** — 원본 500장 세트 자체가 host에 없음 |
| Calibration preprocessing | UNKNOWN (원본 부재로 확인 불가) |
| NVT config | PASS (gen_config.txt 원문 확인) |
| Compiler version | UNKNOWN/DIFFERENT (버전 문자열 없음) |
| Reference NVT reproduction | **FAIL** — 로컬 재변환 bin의 sha256이 Release bin과 다름 |
| Simulator raw output | PARTIAL — 근사치만 존재 (Release bin 자체 값 없음) |
| IPU-test | PASS — output.json 6,075/6,076장 커버 확인, GT와 이미지명 매칭 성공(1장만 board 미검출), bbox가 full-frame 좌표임을 좌표 범위로 확인 |
| Board/Float mAP50-95 (실험 0, 2026-08-20) | **PASS (근사 재현)** — mAP50 sanity check 통과(재현 0.7177/0.7293 vs 문서 0.697/0.704, 오차 +0.02대). mAP50-95: Float 0.4615, Board 0.5091, relative **+10.31%** (손실 없음, 오히려 개선) |

**FIRST DIVERGENCE: Calibration dataset (원본 500장 세트 NOT FOUND)** — 이 지점부터 Release를 수치적으로 완전히 재현하는 경로가 막힌다. 다만 mAP50/mAP50-95 수치와 config 값 확인에는 영향이 없었고, 실험 0(2026-08-20)으로 핵심 미확정 사항(FRONT의 mAP50-95 기준 quantization loss)이 해소되어, 원인 분석(19~21절)과 이번 세션의 핵심 전제 모두 유효하게 확인되었다.

## 19. Why FRONT Is Quantization-Robust

| 후보 | 분류 | 근거 |
|---|---|---|
| spec_demarc_tensors (tensor 보호) | **UNLIKELY as root cause** | FRONT는 0개 보호로도 mAP50 손실이 거의 없음(+1.0%) — REAR는 3-tensor 보호에도 -7.1% |
| calib_mode / percentile | LIKELY 부차적 | REAR와 동일 계열 값(calib_mode 4) 사용 추정, 차별화 요인으로 보기 어려움 |
| CLE / cal_refine_iter / bnscale | UNLIKELY | 전부 비활성(0) — 별도 최적화 없이도 손실이 작음 |
| Weight/Activation distribution (학습된 표현 자체의 quantization robustness) | **LIKELY (근거 있음, 확정은 아님)** | 근사치 layer cosine 기준 model.2/4 backbone-early가 REAR보다 뚜렷이 높음(0.984~0.998 vs REAR 0.96대) — config가 아니라 weight/activation 분포 자체가 강건함을 시사 |
| Calibration dataset 차이 | UNKNOWN | 원본 세트 부재로 직접 비교 불가 |
| Compiler version 차이 | UNKNOWN | 버전 미확인 |
| Final output tensor cosine | **설명력 부족** | FRONT의 최종 output cosine(0.976~0.982)이 REAR(0.982~0.985, 보호 상태)보다 낮거나 비슷한데도 FRONT가 mAP50 손실이 적음 — output cosine 단독으로는 최종 mAP 손실을 설명하지 못함 |

## 20. FRONT vs REAR v01_symmetric

| 항목 | FRONT (Release) | REAR v01_symmetric |
|---|---|---|
| Input shape | [1,3,512,640] | (REAR 문서 참조, 별도 확인 필요) |
| Float mAP50 | 0.697 (문서) / 0.7177 (실험0 재현) | 0.774 |
| Float mAP50-95 | 0.4615 (실험0 재현치) | 0.561 |
| Board/NPU mAP50-95 | 0.5091 (실험0 재현치) | 0.521 |
| Relative loss (mAP50-95) | **+10.31%** (실험0 재현치 — 손실 없음, 개선) | **-7.1%** |
| Relative loss (mAP50) | +1.0% (문서) / +1.6% (실험0 재현) | (별도 확인 필요) |
| spec_demarc_tensors | 0개 (전체 주석) | 3개 (H1-H3, head 근처) |
| calib_mode | 4 | 4 (REAR 문서 기준 동일 계열) |
| CLE / cal_refine_iter / bnscale | 0 / 0 / 0 | (REAR 실험별로 다름, 문서 참조) |
| backbone-early cosine (근사치) | 0.984~0.998 | ~0.96대 |
| final output cosine (근사치) | 0.976~0.982 | 0.982~0.985 |

**mAP50-95 기준 직접 비교 (실험 0, 2026-08-20 결과 반영)**: FRONT는 mAP50-95 기준으로도 Float→NPU 손실이 없다(오히려 +10.31% 개선) — REAR(-7.1% 손실)와 정반대 방향이다. 이 격차의 크기(+10.31% vs -7.1%, 총 17.4%p 차이)는 12~19절에서 확인한 config 차이(REAR의 spec_demarc_tensors 3개 vs FRONT 0개, backbone-early cosine 격차)만으로 설명하기에는 매우 크다 — weight/activation distribution 자체의 근본적 차이(19절 가설)가 이 정도 격차의 주된 원인일 가능성이 높다는 쪽으로 무게가 실린다.

## 21. Transferable Findings

1. **tensor 보호(spec_demarc_tensors)가 많다고 quantization robustness가 보장되지 않는다** — FRONT는 0개로도 mAP50 손실이 거의 없다. REAR의 3-tensor 보호가 -7.1%인 것은 보호 텐서 수의 문제가 아닐 가능성이 높다.
2. **CLE/cal_refine_iter/bnscale 같은 컴파일러 옵션이 핵심 변수가 아닐 수 있다** — FRONT는 전부 비활성 상태로 운영됨.
3. FRONT의 backbone-early cosine이 REAR보다 뚜렷이 높다는 것은, 원인이 **calibration/config가 아니라 학습된 weight/activation 분포**에 있을 가능성을 뒷받침한다 — 이는 REAR 자체 문서가 이미 제기한 가설과 방향이 일치한다.
4. 단, final output cosine만 보면 FRONT가 REAR보다 낫지 않다 — 즉 "cosine이 높으면 mAP가 유지된다"는 단순 일대일 관계는 아니며, 어떤 layer/tensor의 cosine이 최종 mAP와 실제로 연결되는지는 추가 검증이 필요하다.
5. FRONT의 mAP50 기준 결과만으로 REAR의 mAP50-95 문제에 대한 결론을 내리는 것은 위험하다 — 반드시 같은 지표(mAP50-95)로 재비교해야 한다(22절 1번 실험).

## 22. Recommended REAR Experiments

**실험 0 (선행, REAR 아님 — FRONT 자체 검증)**: `analysis/iputest/output.json`(151,136 detection) + Validation GT를 REAR와 동일한 IoU-multi-threshold evaluator에 넣어 FRONT Float/Board의 실제 mAP50-95를 산출. NVT recompile 불필요, 기존 board dump 재사용. PASS 기준: 재현된 mAP50이 기존 문서값(0.697/0.704)에 근접. 이 결과가 나와야 19~21절의 가설을 mAP50-95 기준으로 검증할 수 있다.

이후 REAR는 반드시 **one-variable-at-a-time**으로 설계 (FRONT config 일괄 이식 금지 — 과거 실패 사례 있음):

- **Experiment A**: spec_demarc_tensor 1개만 변경 (예: 3개 중 1개를 FRONT처럼 비활성화하거나 위치 변경)
- **Experiment B**: cal_refine_iter만 변경
- **Experiment C**: 특정 head tensor precision만 변경
- **Experiment D**: calibration preprocessing만 변경
- **Experiment E**: QAT (23절 참조)

각 실험은 Float ONNX mAP50-95, Board NPU mAP50-95, absolute/relative loss, latency, DRAM, model size를 함께 평가하며 다른 변수는 고정한다.

## 23. QAT / Mixed Precision 필요성

- 현재 REAR best(v01_symmetric)가 -7.1%인 상황에서, FRONT 분석 결과 PTQ 설정(calib_mode/CLE/보호 텐서)만으로는 격차의 근본 원인을 설명하지 못할 가능성이 있다(19절 "Weight/Activation distribution" 가설).
- 따라서 PTQ 기반 실험(22절 A~D)이 ≤1%에 도달하지 못할 경우, QAT/mixed precision을 정식 후보로 올린다.
- 이번 세션에서는 FRONT golden reference 분석이 아직 mAP50-95 기준으로 완결되지 않았으므로(실험 0 필요), QAT를 바로 시작하지 않는다.

## 24. UNKNOWN

- Release 500장 calibration set 원본 (경로 자체가 이 host에 없음)
- Release 당시 compiler 버전
- Release NVT bin의 latency/DRAM 실측치 (프로파일 로그 미보존)
- Release NVT bin의 layer-level simulator cosine (compile 중간산출물 미보존, 로컬 재변환값은 근사치일 뿐)
- `iputest/output.json` → `metrics_front_board.*` (문서 기재 mAP50 0.704) 산출에 쓰인 정확한 evaluation 스크립트/이미지 subset (실험 0은 독립적으로 Evaluator를 구성해 근사 재현했을 뿐, 원본 스크립트 자체는 여전히 못 찾음)
- Day(29,299)/Night(22,031) 구성이 어느 dataset 스냅샷(Train/Val/전체) 기준인지
- ~~FRONT Float/Board mAP50-95~~ → **실험 0(2026-08-20)으로 근사 해소**: Float 0.4615 / Board 0.5091 / relative +10.31% (dual-NMS 파라미터 등 일부 가정에 의존한 근사치, 6절 참조)

## 25. Final Conclusion

FRONT Release는 REAR의 v01_symmetric보다 **더 적은 quantization 보호 장치**(spec_demarc_tensors 0개, CLE/cal_refine_iter/bnscale 전부 비활성)로 운영되었음에도 mAP50 기준 손실이 거의 없고(+1.0%), **실험 0(2026-08-20) 재현 결과 mAP50-95 기준으로도 손실이 없다 — 오히려 +10.31% 개선**이다. REAR(-7.1%)와 정반대 방향이며 그 격차(17.4%p)는 config 차이만으로 설명하기엔 크다. 근사치 layer cosine 분석과 함께, 이 강건함이 컴파일러 설정보다 **학습된 weight/activation 분포 자체**에서 온다는 가설을 뒷받침한다.

이번 조사의 최종 답:
- **CONFIRMED**: FRONT는 REAR보다 적은 보호 장치로 운영되며, backbone-early quantization error가 REAR보다 낮다.
- **CONFIRMED (근사치, 2026-08-20 실험 0)**: FRONT는 mAP50-95 기준으로도 Float→NPU 손실이 없다(+10.31%, 손실이 아니라 개선) — 이번 세션의 핵심 전제("FRONT를 Golden Reference로 삼아 REAR를 개선한다")가 확인되었다. 단, dual-NMS 파라미터·GT 클리핑 임계값 등 일부 가정에 의존한 근사치이며, mAP50 sanity check(재현값이 문서값과 +0.02대로 근접)로 방법론 정합성만 확인했지 byte-identical 재현은 아니다.
- **LIKELY**: 이 robustness의 원인은 config가 아니라 weight/activation 분포다 — 격차의 크기(17.4%p)가 config 차이만으로 설명하기엔 너무 크다는 점이 이를 뒷받침한다.
- **UNKNOWN**: 정확히 어떤 weight/activation 통계(채널별 분산, outlier 비율 등)가 이 차이를 만드는지는 이번 세션에서 정량화하지 못했다 — REAR one-variable-at-a-time 실험(22절)으로 좁혀가야 한다.

**다음 단계**: 실험 0이 완료되었으므로, 22절의 REAR one-variable-at-a-time 실험(A~E)을 순서대로 설계·실행한다. Experiment A(spec_demarc_tensor 1개 변경)부터 시작해 tensor 보호가 실제로 핵심 변수가 아님을 REAR에서도 직접 확인하는 것을 권장한다 — FRONT가 보여준 "보호 없이도 강건함"이 REAR에도 적용되는지가 가장 저비용으로 검증 가능한 다음 가설이다.

## REAR Experiment A — v01_symmetric spec_demarc 3-tensor ablation (진행 중, 2026-08-20 중단)

### 목적

REAR v01_symmetric baseline(Float mAP50-95=0.561, Board mAP50-95=0.521, relative loss=-7.1%)이 쓰는 3개
spec_demarc_tensors(H1/H2/H3)를 one-variable-at-a-time으로 하나씩 제거해, 각 tensor가 실제로 quantization
loss를 줄이는 데 기여하는지 확인한다. FRONT config 전체 이식이 아니라, "보호 텐서가 많을수록 좋다"는
가설을 REAR 자체 데이터로 검증하는 것이 목적.

### Baseline

- 경로: `novatek/convert/rear_truck_v01_symmetric_320/nvtai_tool/config/REAR_TRUCK_V01_SYMMETRIC_320/cnn25/gen_config.txt`
  (원본, 이번 실험에서 전혀 수정하지 않음)
- spec_demarc_tensors 원문: `Softmax_/model.22/dfl/Softmax_output_0_Y,Sigmoid_output0_Y,Conv_/model.22/dfl/conv/Conv_output_0_Y`
  - H1 = `Softmax_/model.22/dfl/Softmax_output_0_Y`
  - H2 = `Sigmoid_output0_Y`
  - H3 = `Conv_/model.22/dfl/conv/Conv_output_0_Y`
- calib_mode=4, percentile=0.999, cle_en=1, cal_refine_iter=5, bnscale_en=1, calibration 940장(`video_calib_jpg`)
- 기존 기록: nvt_model.bin 4,711,336 bytes, Float mAP50-95=0.561, Board mAP50-95=0.521, relative loss=-7.1%
  (`docs/debug/20260806_rear_truck_v01_npu_quant_gap.md` 기존 기록 재확인, 재컴파일하지 않음)

### 실험 설계 (A1/A2/A3)

새 디렉터리: `novatek/convert/rear_truck_v01_symmetric_3tensor_ablation/{drop_H1,drop_H2,drop_H3}/`
(baseline 원본과 완전히 분리, input/은 baseline의 ONNX+calibration을 symlink로 참조해 byte-identical 보장)

| Config | spec_demarc_tensors (diff 결과) | 제거된 tensor |
|---|---|---|
| A1 (drop_H1) | `Sigmoid_output0_Y,Conv_/model.22/dfl/conv/Conv_output_0_Y` | H1 |
| A2 (drop_H2) | `Softmax_/model.22/dfl/Softmax_output_0_Y,Conv_/model.22/dfl/conv/Conv_output_0_Y` | H2 |
| A3 (drop_H3) | `Softmax_/model.22/dfl/Softmax_output_0_Y,Sigmoid_output0_Y` | H3 |

config diff 검증 결과: 세 실험 모두 baseline 대비 **spec_demarc_tensors 한 줄만** 다름 (직접 `diff` 재확인 완료,
2026-08-20). calibration/calib_mode/percentile/CLE/cal_refine_iter/bnscale/compiler·chip 설정은 전부 동일.

### 진행 상태 (2026-08-20 기준, 사용자 지시로 중단됨)

| 실험 | Compile | Simulator | Board 평가 | 비고 |
|---|---|---|---|---|
| A1 (drop_H1) | ✅ 완료 (13:21 KST) | ❌ 미실행 | ❌ 미실행 | `nvt_model.bin` 생성됨 (4,711,336 bytes, baseline과 동일 크기) |
| A2 (drop_H2) | ⛔ 시작 안 함 | — | — | 사용자 지시로 중단, output 디렉터리 비어있음 |
| A3 (drop_H3) | ⛔ 시작 안 함 | — | — | 사용자 지시로 중단, output 디렉터리 비어있음 |

- A1의 `nvt_model.bin`은 그대로 보존 중 (삭제하지 않음, 사용자 확인 대기)
- 실행 중이던 compile 프로세스 없음 — 완전히 안전하게 정지된 상태
- baseline 원본은 전혀 수정/재컴파일하지 않음

### 재개 시 다음 순서 (TODO)

1. A1 `nvt_model.bin`을 유지할지 폐기할지 결정
2. A1 simulator 실행 (baseline과 동일 image 사용) → H1/H2/H3 + Softmax/Sigmoid/dfl_conv cosine, `profile.txt`에서 latency/DRAM 확인, `nvt_model.bin` SHA256 계산
3. A2 (drop_H2) compile → simulator
4. A3 (drop_H3) compile → simulator
5. 세 실험 모두 simulator 결과 확보 후, 사용자 지시에 따라 board 후보 선정 (6,152장 GT, dual-threshold NMS, 기존 evaluator와 동일 조건)
6. 최종 결과표(Config / Protected tensors / Float mAP50-95 / Board mAP50-95 / Relative loss / Latency / DRAM / Size) 완성

### 결론

**아직 없음 — 실험 미완료.** A1 compile만 끝났고 시뮬레이터/보드 평가 전 단계에서 사용자 지시로 중단됨.
H1/H2/H3 각각의 기여도에 대한 판정은 이 실험이 끝나야 가능하다.
