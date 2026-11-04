# v01_symmetric 평가 방식 차이 검증 — quantization loss가 -7.3%인가 ~0%인가

- **작성**: 2026-08-24 (KST)
- **대상**: `configs/experiments/BSD_F606/Rear_Truck/v01_symmetric` (Float ONNX
  `260821_best.onnx` vs 실제 board 덤프, 동일 6,152장)
- **관련 기존 문서**: `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`(정본, v01_symmetric의
  -7.1%/-7.3% quantization loss가 처음 기록된 문서). **이 문서는 그 문서를 수정하지 않는다** —
  같은 모델에 대해 "평가 방식(evaluator/NMS)을 바꾸면 loss 숫자가 얼마나 달라지는가"만 별도로
  검증한 독립 기록이다.

## 1. 목적

같은 v01_symmetric, 같은 6,152장 GT, 같은 Float ONNX(`260821_best.onnx`), 같은 board 덤프
(`ipu_pred_merged_v01_symmetric_3tensor.json`)인데, **평가 파이프라인(NMS 종류 + AP 계산
엔진)에 따라** quantization loss가 `-7.3%`(이 프로젝트의 기존 `customEval.py`/pycocotools
기준)에서 `~-0.6%`(ultralytics 자체 AP 계산 방식 기준)까지 크게 달라지는 현상을 확인했다
(이전 세션 채팅 기록). 이 문서는 그 원인을 코드 근거로 분리 검증한다.

## 2. 현재 확인된 두 평가 결과 (검증 이전, 이전 세션에서 나온 값)

### ultralytics 방식 (표준 단일-IoU NMS + `ultralytics.utils.metrics.ap_per_class`)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| golden(float) | 0.8644 | 0.7214 | 0.7965 | 0.5785 |
| board | 0.8698 | 0.7003 | 0.7998 | 0.5749 |

mAP50-95 차이: `-0.0036 (-0.62%)`

### 기존 customEval.py / pycocotools 방식 (dual-threshold NMS)

| | mAP50 | mAP50-95 |
|---|---:|---:|
| golden(float) | 0.778 | 0.562 |
| board | 0.751 | 0.521 |

mAP50-95 차이: `-0.041 (-7.30%)`

## 3. 현재 의문 / 가설

- `-7.3%`를 순수 quantization loss라고 확정할 수 있는가?
- ultralytics 비교는 **golden은 raw output → 표준 NMS를 거치고, board는 이미 온디바이스에서
  NMS까지 끝난 덤프를 추가 NMS 없이 그대로 쓰는 비대칭 구조** — 이 비대칭이 `-0.62%`라는
  숫자에 영향을 줬을 수 있다. 따라서 `-0.62%`도 아직 "순수 quantization loss"로 확정하지 않는다.
- 두 가설을 분리해야 한다: (a) NMS 종류 차이, (b) evaluator(AP 계산 엔진) 차이 — 이 문서의
  핵심 실험은 이 둘을 최소 실험으로 분리하는 것이다.

## 4. 실제 파이프라인 비교 (코드 근거)

### 4-1. Float ONNX(golden) 후처리 — 코드에 실재하는 두 가지 경로

| 항목 | `basic_postproc_v8_dual_nms` (board 재현용) | `basic_postproc_v8` (표준) |
|---|---|---|
| 위치 | `projects/metrics/utils.py:258-378` | `projects/metrics/utils.py:97-224` |
| model output 상태 | raw (cls sigmoid + box distance, 2-tensor, `skip_postprocess=True` export) | 동일 (2-tensor) |
| decode | `dist2bbox`(anchor 기반, ltrb→xyxy)+stride, `utils.py:268-276,319` | `dist2bbox`+stride, `utils.py:98-108,141` (로직 동일) |
| confidence filtering | `confidence_thres` 파라미터로 필터(`utils.py:326`) — **호출부 `basic_inference_v8`가 0.01을 하드코딩**(`utils.py:247`), config로 override 불가(위치 인자로 이미 명시돼 `**kwargs`로 다시 넘기면 `TypeError`, 이번 세션에서 실측 확인) | 동일 구조, 동일 하드코딩(`basic_inference_v8` 공용 호출부) |
| NMS 종류 | dual-threshold, `cnn_nt98530_post_process.cpp`의 `nms()/_rect_overap()` 포팅(`utils.py:260-266`) — `min_overlap=inter/min(area_a,area_b)`, `union_overlap=inter/union` | 표준 단일 IoU, `cv2.dnn.NMSBoxes`(`utils.py:187`) |
| NMS threshold | `min_overlap_thres=0.6`, `union_overlap_thres=0.3`(board와 동일하게 맞춘 값, config에서 지정) | `iou_thres`(기본 0.45, config `onnx_v01_symmetric_stdnms_conf001_cfg.yaml`에서 0.45로 명시) |
| class-aware 여부 | **class-aware** — `for cls in np.unique(labels): ... nms_dual_thresh(...)`(`utils.py:348-355`), 클래스별로 개별 NMS 수행 | **class-agnostic** — `cv2.dnn.NMSBoxes`는 클래스 라벨을 받지 않음(`utils.py:187`), 전체 detection을 한 번에 NMS |
| max detection | 없음(제한 없음) | 없음(제한 없음) |
| box 좌표 변환 | 네트워크 해상도(320x320) → 원본(1920x1080), `x_factor/y_factor` 배율(`utils.py:322-323,334-337`) | 동일 로직(`utils.py:146-147,164-168`) |
| 반환 형식 | `[xywh_box(int32), score, label]` 리스트 | 동일 |

**핵심**: 두 함수는 decode 로직은 사실상 동일하고, **NMS 알고리즘(dual-threshold vs 단일 IoU)과
class-aware 여부만 다르다.**

### 4-2. Board 덤프 — 후처리 여부

- 원본 dump(`board_dumps/v01_symmetric/v01_symmetric_output.json`)는 이미지당 1개 JSON array가
  concatenated된 형태, 각 detection은 `{bbox(xyxy, pixel), score, name, category, time}` —
  **이미 보드 온디바이스 NMS를 거친 최종 detection**이다.
- `merge_board_dump.py`(`runs/_eval_scratch/tools/merge_board_dump.py`)는 단순 병합 + 카테고리명
  리매핑(`MOTOR`→`motorcycle`)만 하고, **추가 필터링/NMS/confidence threshold를 전혀 적용하지
  않는다**(코드 전문 확인, `if/threshold` 분기 없음).
- **board의 "pre-NMS raw" 덤프는 이 repo/NAS 어디에도 존재하지 않는다** — `docs/debug/` 전체
  검색(`grep -rl "온디바이스 NMS\|pre-NMS"`) 결과 board 산출물은 항상 "이미 최종 NMS를 거친
  detection"으로만 기록돼 있다(`20260819_front_golden_reference_for_rear_1pct.md:104` 등).
  즉 §5의 "우선순위 1(float raw와 board pre-NMS raw를 동일 파이프라인으로 비교)"은 **데이터
  자체가 없어 원천적으로 불가능**하다 — 이번 검증도 우선순위 2/3으로 진행한다.

### 4-3. Evaluator — 두 AP 계산 엔진

| 항목 | `projects/metrics/evaluator.py` (customEval.py) | `ultralytics.utils.metrics` |
|---|---|---|
| 라이브러리 | **실제 pycocotools** — `from pycocotools.coco import COCO`(`evaluator.py:16`), `customCOCOeval(COCOeval)`(`custom_coco_eval.py:5,8`) | `ap_per_class` + `BaseValidator.match_predictions`(자체 구현, pycocotools 미사용) |
| customCOCOeval이 바꾼 것 | `accumulate()`만 오버라이드(`custom_coco_eval.py:9`) — `iouThrs`/`areaRng`/`maxDets` 등 `Params`는 pycocotools 기본값 그대로(코드 확인, `__init__` 오버라이드 없음) | — |
| IoU threshold 목록 | pycocotools 기본 `iouThrs = np.linspace(0.5, 0.95, 10)` (10개, `custom_coco_eval.py:23`의 `T=len(p.iouThrs)`가 기본 Params 사용) | `torch.linspace(0.5, 0.95, 10)` — 이번 검증 스크립트에서 동일하게 명시(`tools/debug/ultralytics_style_ap.py`) |
| areaRng | 기본 4종(all/small/medium/large) 중 `evaluator.py:216,228`에서 인덱스 `0`("all") 고정 사용 | 없음(면적 구분 없이 전부 평가) — **차이점, 단 이 데이터셋에서 area=all 사용 시 실질적 영향은 확인 안 됨** |
| maxDets | 기본 `[1, 10, 100]` 중 인덱스 `-1`(=100) 사용(`evaluator.py:216,228`) — **이번 세션에서 board 이미지당 detection 수 실측(최대 60, 6,152장 전수 확인)으로 100 제한에 걸리는 이미지가 0개임을 확인, 원인에서 배제** | 제한 없음(전부 사용) |
| AP 적분 방식 | pycocotools 표준 101-point 보간(스톡 알고리즘, `custom_coco_eval.py`가 건드리지 않음) | `ultralytics.utils.metrics.compute_ap` — **동일하게 101-point 보간**("101-point interp (COCO)", 코드 주석과 함께 확인) |
| 매칭(TP/FP 판정) 알고리즘 | pycocotools `COCOeval.evaluateImg`(스톡, 이미지·IoU threshold별 그리디 매칭) | `BaseValidator.match_predictions`(자체 그리디 매칭, `iou >= threshold`인 것 중 IoU 내림차순으로 GT-detection 1:1 매칭) — **알고리즘이 다르다(둘 다 그리디지만 구현이 별개), 이번 세션에서 그 이상 세부 비교는 하지 않음** |
| class 처리 | `get_class_id()`(`evaluator.py:114-118`) — 문자열 클래스명을 `self.classes.index(name.lower())`로 매핑, 숫자면 그대로 | class id를 정수로 직접 사용(`CLS2ID` dict, `tools/debug/ultralytics_style_ap.py`) — 매핑 자체는 동일 순서(bus=0...truck=4) 확인 |

## 5. 수행한 검증

### 5-1. 최소 실험 설계 — 2×2 isolation

기존 실험(dual-NMS golden vs board, customEval.py)은 그대로 두고, **NMS 축**(dual-threshold ↔
표준 단일-IoU)과 **evaluator 축**(pycocotools ↔ ultralytics ap_per_class)을 각각 하나씩만
바꾼 4개 셀을 구성했다. Board는 이미 최종 NMS를 거친 고정값이라 NMS 축의 영향을 받지 않는다
(§4-2) — board 열은 evaluator 축에서만 바뀐다.

|  | Eval = pycocotools(customEval.py) | Eval = ultralytics `ap_per_class` |
|---|---|---|
| **NMS = dual-threshold**(board 재현) | (a) 기존 결과, §2 그대로 | (b) 신규 |
| **NMS = 표준 단일-IoU**(`basic_postproc_v8`, iou=0.45) | (c) 신규 | (d) 기존 결과, §2 그대로 |

모든 셀에서 golden 쪽 `confidence_thres=0.001`로 통일(호출부 하드코딩 우회, §4-1 참조 —
`tools/debug/inf_conf001.py` 재사용/신규 작성, `confidence_thres=0.01`→`0.001`만 바꾼 복제본).

### 5-2. 재사용한 기존 결과 (재실행 안 함)

- (a) golden: `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_conf001_dualnms_cfg.yaml`
  실행 결과, `coco_prediction.json`
  (`runs/_eval_scratch/results/onnx_v01_symmetric_conf001_dualnms/.../20260824_051433/`) —
  이전 세션에서 이미 생성된 파일을 그대로 읽었다.
- (d) golden/board: 이전 세션에서 실행한 `tools/debug/ultralytics_style_ap.py`(당시엔
  job-tmp 스크립트, 이번 세션에서 `tools/debug/`에 정리해 영구 반영) 결과값을 그대로 채택.

### 5-3. 신규 실행 (최소, 재추론 없음 또는 6,152장 1회만)

1. **(b) 기존 dual-NMS golden 예측을 ultralytics로 재평가** — 새 추론 없음, 기존
   `coco_prediction.json`을 읽어 GT와 매칭만 다시 수행:
   ```
   docker exec ultralytics_projects-dair-train-1 bash -lc \
     'cd /workspace && uv run python3 tools/debug/ultralytics_style_ap.py \
       --pred_format coco \
       --pred_path runs/_eval_scratch/results/onnx_v01_symmetric_conf001_dualnms/onnx_v01_symmetric_conf001_dualnms_cfg/20260824_051433/coco_prediction.json'
   ```
2. **(c) 표준 NMS golden을 pycocotools로 평가** — 6,152장 1회 추론 필요(기존에 없던 조합),
   새 config만 추가(`onnx_v01_symmetric_stdnms_conf001_cfg.yaml`, 기존 `basic_postproc_v8`
   재사용 — 새 후처리 코드 없음):
   ```
   docker exec ultralytics_projects-dair-train-1 bash -lc \
     'cd /workspace && uv run python3 tools/customEval.py \
       runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_stdnms_conf001_cfg.yaml'
   ```
3. **board maxDets 가설 검증** — board raw dump(6,152장)의 이미지당 detection 개수 분포를
   직접 집계(재추론 없음, 기존 JSON 파싱만): 최대 60건/이미지, 100건 초과 이미지 0개 —
   pycocotools `maxDets=100` 제한이 원인일 가능성을 배제.

### 5-4. 신규 추가 파일

- `tools/debug/ultralytics_style_ap.py` — coco 형식(`coco_prediction.json`) 또는 board raw
  병합 json을 입력받아 ultralytics `ap_per_class`/`match_predictions`로 6,152장 GT와 매칭해
  mAP를 구하는 재사용 가능한 진단 스크립트.
- `tools/debug/inf_conf001.py` — `basic_inference_v8`의 `confidence_thres=0.01` 하드코딩만
  `0.001`로 바꾼 복제본(원본 미수정, config로 override 불가능해서 필요).
- `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_stdnms_conf001_cfg.yaml` — 표준 NMS +
  conf=0.001 golden 평가용 config(gitignore 대상, 재현용으로 기록).

기존 결과/덤프/config는 어느 것도 삭제·수정하지 않았다.

## 6. 결과

| 셀 | NMS(golden) | Evaluator | golden mAP50-95 | board mAP50-95 | 절대 손실 | 상대 손실 |
|---|---|---|---:|---:|---:|---:|
| (a) | dual-threshold | pycocotools | 0.562 | 0.521 | -0.041 | **-7.30%** |
| (b) | dual-threshold | ultralytics ap_per_class | 0.5864 | 0.5749 | -0.0115 | **-1.96%** |
| (c) | 표준 단일-IoU | pycocotools | 0.558 | 0.521 | -0.037 | **-6.63%** |
| (d) | 표준 단일-IoU | ultralytics ap_per_class | 0.5785 | 0.5749 | -0.0036 | **-0.62%** |

행/열로 분리한 효과 크기:

- **NMS 축 효과** (evaluator 고정, dual→표준으로 바꿨을 때 golden 자체의 변화):
  - pycocotools 안에서: 0.562→0.558 (-0.004, golden 대비 -0.71%)
  - ultralytics 안에서: 0.5864→0.5785 (-0.0079, golden 대비 -1.35%)
  - → **loss%(golden 대비 board 차이)에 대한 영향은 -7.30%→-6.63%(0.67%p), -1.96%→-0.62%
    (1.34%p) — 두 경우 모두 1.5%p 이내.**
- **Evaluator 축 효과** (NMS 고정, pycocotools→ultralytics로 바꿨을 때):
  - dual-threshold 안에서: -7.30%→-1.96% (**5.34%p**)
  - 표준 NMS 안에서: -6.63%→-0.62% (**6.01%p**)
  - board 자체의 mAP50-95도 evaluator에 따라 0.521→0.5749로 **크게 다르다(+0.0539,
    +10.3%)** — 완전히 동일한 board 예측·동일 GT인데도 evaluator만 바꿔서 이만큼 차이가 난다.

**maxDets=100 가설은 기각**: board 이미지당 detection 수는 최대 60건(6,152장 전수 확인,
p99=42건)으로, pycocotools의 `maxDets=100` 제한에 걸리는 이미지가 0개다. 두 evaluator 간
board mAP 차이(0.521 vs 0.5749)는 이 제한 때문이 아니다.

## 7. 결론

- **NMS(dual-threshold vs 표준 단일-IoU) 차이는 loss%에 미치는 영향이 작다** (1.5%p 이내,
  §6). golden에 board와 다른 NMS를 썼다는 비대칭 자체는 실재하지만, 그 비대칭이 만드는 손실
  숫자 차이는 크지 않다는 것이 확인됐다.
- **evaluator(AP 계산 엔진) 차이가 훨씬 크다** (5~6%p, §6) — `-7.3%`와 `~-0.6%`이라는 두 숫자
  사이 격차의 **주된 원인은 NMS가 아니라 evaluator 선택**이다. 이는 golden뿐 아니라 **동일한
  board 예측·동일 GT를 놓고도 evaluator만 바꾸면 mAP50-95가 0.521→0.5749로 바뀐다**는 사실로
  뒷받침된다.
- **다만 "왜" pycocotools(COCOeval)와 ultralytics(`ap_per_class`)가 동일 입력에서 이만큼 다른
  숫자를 내는지, 그 알고리즘 내부 차이는 이번 세션에서 완전히 규명하지 못했다.** 둘 다 표준
  101-point COCO 보간을 쓰고(§4-3, 코드로 확인), `maxDets=100` 제한도 원인이 아님을 확인했으니
  (§6), 남은 차이는 **이미지·IoU threshold별 TP/FP 그리디 매칭 알고리즘 자체의 구현 차이**
  (pycocotools `evaluateImg` vs `BaseValidator.match_predictions`)로 추정되나, 이 추정은
  **코드를 직접 대조해 검증하지 않았으므로 확정이 아니다.**

### 질문별 답

| 질문 | 답 |
|---|---|
| 기존 `-7.3%`가 실제 quantization loss인가? | **부분적으로 확인됨** — NMS 비대칭으로 인한 왜곡은 1.5%p 이내로 작다는 것은 확인했으나, evaluator 선택에 따라 같은 조건에서 `-0.62%`까지도 나오므로 `-7.3%`를 유일한 "정답"으로 확정할 수 없다. |
| NMS/postprocess 차이가 주요 원인인가? | **아니다** (반증됨, §6) — 영향은 1.5%p 이내. |
| evaluator 차이가 주요 원인인가? | **그렇다** (확인됨, §6) — 5~6%p 차이를 만든다. |
| 실제 순수 quantization loss를 계산할 수 있는가? | **현재로서는 확정 불가** — 어느 evaluator가 "board 실제 배포 성능"을 더 정확히 반영하는지에 대한 근거가 아직 없다. |
| 현재 데이터만으로 계산 불가능한가? | **evaluator 선택에 대한 근거가 없다는 점에서는 그렇다.** NMS 축은 데이터로 답이 나왔지만(§6), evaluator 축은 "어느 게 맞는 지표인가"라는 방법론적 판단이 추가로 필요하다. |

## 8. 아직 확인되지 않은 사항

- pycocotools `COCOeval.evaluateImg`와 `BaseValidator.match_predictions`의 그리디 매칭
  알고리즘을 라인 단위로 직접 대조하지 않았다 — evaluator 차이의 진짜 알고리즘적 원인은
  추정일 뿐 확정이 아니다.
- board의 class-aware/class-agnostic NMS 여부(온디바이스 NMS가 클래스별인지 전체 통합인지)는
  이번 문서에서 재확인하지 않았다 — `merge_board_dump.py`는 그대로 통과시킬 뿐이라 board
  자체의 NMS 방식은 하드웨어/펌웨어 코드에 의존하며 이 세션에서 재검증 대상이 아니었다.
  (참고: `basic_postproc_v8_dual_nms`는 board의 `cnn_nt98530_post_process.cpp` NMS를
  포팅한 것이라는 기존 문서 코멘트가 있으나, 그 포팅이 100% 정확한지도 이번 세션에서
  재검증하지 않았다.)
- "어느 evaluator가 실제 배포 성능(board 단독 운영 시 체감 성능)을 더 정확히 반영하는가"는
  순수 기술 검증을 넘는 방법론/제품 판단이라 이 문서의 범위 밖으로 남긴다.
- areaRng(pycocotools의 all/small/medium/large 구분)가 이 데이터셋에서 실질적으로 어떤
  영향을 주는지는 "all" 고정값만 썼을 뿐 별도로 확인하지 않았다.

## 9. 다음 단계 (제안, 실행하지 않음)

- pycocotools `evaluateImg`(그리디 매칭 부분)과 ultralytics `match_predictions`를 같은 극소
  합성 예시(GT 2~3개, 예측 3~5개)에 나란히 돌려서 TP/FP 판정이 정확히 어느 지점에서 갈리는지
  단위 테스트 수준으로 재현.
- board의 실제 온디바이스 NMS가 class-aware인지 확인할 수 있는 벤더 문서/코드가 있는지 확인.
- "어떤 evaluator를 이 프로젝트의 공식 acceptance 기준으로 쓸지"에 대한 의사결정 — 이 결정에
  따라 `docs/decisions.md`에 D-번호로 기록 필요(현재 D-019의 `≤1%` 기준이 어떤 evaluator
  기준인지도 함께 명확히 할 필요가 있음, 이번 문서에서는 판단하지 않음).

## 10. 네 모델 종합 비교 (mAP50-95) — 2026-08-24 추가

이 문서의 2×2 검증 방법을 v01_dair / v02 / v01_symmetric_matched 세 모델에도 동일하게
적용한 결과(상세: `docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`)를
종합한 표. 원래 `v01_symmetric_matched` 문서의 §8이었으나, 네 모델을 한눈에 비교하는
내용이라 이 문서(기준 문서) 하단으로 옮겼다.

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra | NMS축 효과 | Evaluator축 효과 |
|---|---:|---:|---:|---:|---:|---:|
| v01_symmetric | -7.30% | -1.96% | -6.63% | -0.62% | ~0.7~1.3%p | ~5.3~6.0%p |
| v01_dair | -21.01% | -16.32% | -19.66% | -14.62% | ~1.35~1.70%p | ~4.7~5.0%p |
| v02 | -19.14% | -14.80% | -18.08% | -13.44% | ~1.06~1.36%p | ~4.3~4.6%p |
| **v01_symmetric_matched** | **-28.42%** | **-27.00%** | **-27.36%** | **-25.62%** | **~1.06~1.38%p** | **~1.4~1.7%p** |

## 10-1. 네 모델 종합 비교 (mAP50, 참고) — 2026-08-24 추가

원래 `v01_symmetric_matched` 문서의 §8-1이었던 표. `mAP50` 기준 relative loss(%),
`docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`에 각 모델의 산출
과정이 기록돼 있다.

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra | Evaluator축 효과 |
|---|---:|---:|---:|---:|---:|
| v01_symmetric | -3.47% | **+0.26%** | -3.72% | **+0.41%** | ~3.7~4.1%p |
| v01_dair | -9.72% | -6.03% | -8.48% | -4.66% | ~3.7~3.8%p |
| v02 | -10.08% | -6.80% | -9.60% | -6.24% | ~3.3~3.4%p |
| v01_symmetric_matched | -11.35% | -10.06% | -10.64% | -9.29% | ~1.3%p |

mAP50 기준에서는 v01_symmetric이 ultralytics 평가 시 손실이 아예 **역전(+0.26~0.41%,
board가 golden보다 높음)**되지만, v01_symmetric_matched는 evaluator를 바꿔도 여전히 -9~11%대
손실이 남는다 — mAP50-95와 같은 방향의 결론이다.

**네 모델 종합 결론**: evaluator 축 효과의 크기(%p)는 모델마다 다르다(symmetric 5.3~6.0%p,
dair/v02 4.3~5.0%p, symmetric_matched 1.4~1.7%p) — "evaluator 효과는 항상 일정한 크기로
나타난다"는 가정은 성립하지 않는다. 반면 **NMS 축 효과는 네 모델 모두에서 일관되게 작다
(0.7~1.7%p)** — 이 부분만은 공통적으로 유지된다. v01_symmetric_matched는 이름·트레이너가
v01_symmetric과 같음에도 실측 패턴은 정반대(evaluator 효과 최소, 손실 잔존 최대)로 나타나,
"이름/트레이너가 같으면 같은 패턴일 것"이라는 가정이 틀렸음을 보여준다.
