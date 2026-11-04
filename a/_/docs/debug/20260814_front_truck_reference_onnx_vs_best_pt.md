# Front_Truck Reference ONNX vs best.pt Export 비교

작성일: 2026-08-14. 이 문서는 독립적인 신규 조사다 — `runs/detect/BSD_F606/Front_Truck/standard/validation.md`의
연장이 아니며, 그 문서의 결론을 전제하지 않는다. 기존 산출물(`weights/best_split.onnx`,
`weights/deployment_260706.onnx`, `runs/debug/onnx_compare/*`)은 이번 비교의 근거로 쓰지 않았고,
모든 수치는 이번에 새로 측정했다.

## 최종 판정 요약

| 항목 | 결과 |
|---|---|
| Level 1 — Graph structure | **SAME** |
| Level 2 — Initializer (parameter) | **DIFFERENT** (127개 중 126개) |
| Level 3 — Raw output (동일 입력) | **DIFFERENT** |
| Level 4 — Decoded detection (동일 postprocess) | **DIFFERENT** |
| Level 5 — Validation metric (동일 pipeline) | **DIFFERENT** (방향은 일관, 폭은 mAP 기준 상대적으로 작음) |

**결론: 현재 `best.pt`를 ONNX로 export한 결과는 NAS reference ONNX와 동일하지 않다.**
그래프 구조(노드 종류·개수·순서·연결)는 완전히 같지만, 그래프에 저장된 파라미터(가중치/바이어스)가
첫 레이어부터 이미 다른, **서로 다른 모델**이다.

## 목적

다음 질문 하나에 답한다: *`runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt`를 이 repo의
export 코드로 ONNX 변환하면, NAS의 실제 배포 ONNX(`260706_BSD-F606A_Front_v01.onnx`)와 같은 모델/같은
결과가 나오는가?* 다른 체크포인트 후보 탐색, NPU 변환, quantization은 이번 범위에 포함하지 않는다.

## 비교 대상

### Reference ONNX (A, 읽기 전용 — 절대 수정하지 않음)

- 원본 경로: `/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx`
- size: 12,071,501 bytes
- mtime: `2026-07-07 10:27:08 +0900`
- md5: `d623cbdc6ad13519646b6bda53e879a0`
- sha256: `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0`
- 분석/실행용 읽기 전용 사본: `runs/debug/front_truck_reference_compare/reference_NAS.onnx` (md5 동일 확인됨)

### Source best.pt (B)

- 원본 경로: `/home/ncn/Workspace/jdh/ultralytics_projects/runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt`
- size: 6,233,891 bytes
- mtime: `2026-08-13 15:49:55 +0900`
- md5: `412673ce9a17f52ee40fee87e680dbb4`
- sha256: `f55b8b915149b496e439836b4d53637251725198189f9881ad33e34b418c3cbb`
- export용 사본: `runs/debug/front_truck_reference_compare/source_best.pt` (원본과 별개 파일, md5 동일 확인됨)

### Newly exported ONNX (이번에 새로 생성)

- `runs/debug/front_truck_reference_compare/exported_from_best.onnx`
- size: 12,071,501 bytes
- sha256: `e56e5aa8be7007bd6940177df0233d8cb3e3a12635d4af91b732c4c237808ad1`

기존 `weights/best.onnx`, `weights/best_split.onnx`, `weights/deployment_260706.onnx`는 이번 작업
중 한 번도 열거나 export 대상으로 쓰지 않았다 — 작업 전/후 파일 목록·mtime 변화 없음을 확인했다.

## 실행 환경

- git commit: `66a2698c06916d3dd6204760b54789781540b944` (작업 트리에 이전 세션의 미커밋 변경 다수 존재 — 이번
  조사와 무관한 기존 변경사항)
- 컨테이너: `ultralytics_projects-dair-train-1`
- Python: 3.12.13
- PyTorch: 2.5.1+cu121 (CUDA 12.1 available)
- onnx: 1.21.0
- onnxruntime: 1.26.0 (providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider)
- ultralytics: 8.4.6

## best.pt → ONNX export 방법

1단계(아래)에서 reference ONNX의 실제 출력이 `output0 [1,6,6720]`(클래스, Sigmoid 직결) /
`433 [1,4,6720]`(박스, DFL 적분까지만·dist2bbox 디코드 이전)로 **분리된 구조임을 먼저 확인**한 뒤,
이 구조를 재현하는 `--skip_postprocess` 옵션으로 export했다 — 추측이 아니라 확인 후 결정.

```bash
mkdir -p runs/debug/front_truck_reference_compare
cp runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt \
   runs/debug/front_truck_reference_compare/source_best.pt

uv run python3 tools/export_symmetric.py \
  -w runs/debug/front_truck_reference_compare/source_best.pt \
  --skip_postprocess
# -> runs/debug/front_truck_reference_compare/source_best.onnx 생성됨
mv runs/debug/front_truck_reference_compare/source_best.onnx \
   runs/debug/front_truck_reference_compare/exported_from_best.onnx
```

export 조건: input 640×512(imgsz=640, imgsz_y=512, `export_symmetric.py` 기본값), opset=13,
simplify=False, batch=1(export 기본), `skip_postprocess=True`. `.pt` 원본과 같은 위치(`weights/`)의
파일은 전혀 건드리지 않았다(사본에서 export → 항상 사본과 같은 디렉터리에 결과 생성).

## 1. ONNX interface / graph 비교

스크립트: `tools/debug/reference_compare_graph_and_initializers.py` (읽기 전용, 신규 작성)

```bash
uv run python3 tools/debug/reference_compare_graph_and_initializers.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --out_dir runs/debug/front_truck_reference_compare
```

| 항목 | Reference | Exported |
|---|---|---|
| input | `images [1,3,512,640]` float32 | `images [1,3,512,640]` float32 |
| output | `output0 [1,6,6720]`, `433 [1,4,6720]` | `output0 [1,6,6720]`, `433 [1,4,6720]` |
| opset | 13 | 13 |
| ir_version | 7 | 7 |
| producer | pytorch 2.5.1 | pytorch 2.5.1 |
| node count | 235 | 235 |
| initializer count | 127 | 127 |
| op_type 분포 | Conv 64, Sigmoid 57, Mul 56, Constant 14, Split 8, Add 6, Concat 15, MaxPool 3, Resize 2, Reshape 8, Transpose 1, Softmax 1 | 위와 완전히 동일 |
| node별 op_type/attributes/input·output 개수 불일치 | 0건 | — |

**Level 1 판정: SAME.** interface, node 수·타입·순서·attributes, op_type 분포 전부 일치한다
(node-by-node로 op_type·attribute·input/output 개수까지 비교했고 불일치 0건).

## 2. Initializer 전수 비교

같은 스크립트가 함께 생성: `runs/debug/front_truck_reference_compare/initializer_compare.{csv,json}`

```text
reference initializer count: 127
exported initializer count: 127
common: 127
exact identical: 1
different: 126
reference only: []
exported only: []
```

최초로 다른 initializer(그래프 선언 순서 기준, 그래프의 맨 첫 Conv 레이어): `model.0.conv.bias`
(max_abs_diff=0.942), `model.0.conv.weight`(max_abs_diff=0.0297).

max_abs_diff 상위 5개:

| initializer | max_abs_diff | mean_abs_diff |
|---|---|---|
| `model.9.cv2.conv.bias` | 4.492 | 1.015 |
| `model.22.cv3.0.1.conv.bias` | 3.126 | 0.631 |
| `model.22.cv3.1.1.conv.weight` | 2.536 | 0.053 |
| `model.22.cv3.1.1.conv.bias` | 2.526 | 0.485 |
| `model.22.cv3.0.1.conv.weight` | 2.029 | 0.087 |

**Level 2 판정: DIFFERENT.** 127개 중 126개가 다르고, 차이는 부동소수점 잡음 수준이 아니라
bias 기준 최대 4.5 단위의 실질적인 값 차이다. **그래프 진입점(`model.0`)부터 이미 다르므로
"뒤로 갈수록 누적되는 차이"가 아니라 처음부터 다른 파라미터다.**

## 3. 동일 입력 raw output 비교

스크립트: `tools/debug/reference_compare_raw_output.py` (신규). 고정 입력 10종을 `runs/debug/front_truck_reference_compare/inputs/`에 `.npy`로 저장(zeros/ones/half=127.5/random-seed20260814/impulse/실제 validation 이미지 5장 — `real_0`~`real_4`, index 0/100/500/1000/3000). CPUExecutionProvider, graph optimization OFF/기본 둘 다 실행.

```bash
uv run python3 tools/debug/reference_compare_raw_output.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --inputs_dir runs/debug/front_truck_reference_compare/inputs \
  --out_json runs/debug/front_truck_reference_compare/raw_output_compare.json
```

대표 결과(실제 이미지, optimization OFF 기준 — optimization ON과의 차이는 6번째 소수점 수준으로 무시 가능):

| 입력 | output0(cls) max_abs_diff | 433(box) max_abs_diff |
|---|---|---|
| zeros | 0.00140 | 6.847 |
| ones | 0.00144 | 6.828 |
| half(127.5) | 0.000156 | 6.850 |
| random(seed 20260814) | 0.0289 | 6.082 |
| impulse | 0.00138 | 6.846 |
| real_0 | 0.950 | 8.876 |
| real_1 | 0.926 | 8.387 |
| real_2 | 0.704 | 8.399 |
| real_3 | 0.875 | 10.006 |
| real_4 | 0.780 | 8.294 |

모든 입력·모든 atol(0, 1e-8, 1e-7, 1e-6, 1e-5)에서 `allclose=False`. **입력이 완전히 다른데도
(zeros vs 실제 이미지) box 출력 차이가 항상 6~10 범위로 유지** — 이는 실행 경로/최적화 차이가
아니라 저장된 파라미터 차이에서 오는, 입력에 무관한 차이라는 뜻이다.

**Level 3 판정: DIFFERENT.**

## 4. 최종 detection 비교

스크립트: `tools/debug/reference_compare_detections.py` (신규). 두 onnx의 raw output(`output0`,
`433`)에 **완전히 동일한 디코드**(`ultralytics.utils.tal.dist2bbox`/`make_anchors`, stride 8/16/32,
`output0`는 이미 sigmoid 완료된 값 그대로 사용 — 중복 sigmoid 방지) + **완전히 동일한 NMS**
(`ultralytics.utils.nms.non_max_suppression`, conf=0.05, iou=0.7, max_det=300)를 적용한 뒤
class+IoU≥0.5 기준으로 그리디 매칭했다.

```bash
uv run python3 tools/debug/reference_compare_detections.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --inputs runs/debug/front_truck_reference_compare/inputs/real_{0,1,2,3,4}.npy \
  --conf 0.05 --iou 0.7 \
  --out_json runs/debug/front_truck_reference_compare/detection_compare.json
```

| 이미지 | reference 검출 수 | exported 검출 수 | matched | only-reference | only-exported | matched conf_diff(mean/max) | matched bbox 최대차(mean/max, px) |
|---|---|---|---|---|---|---|---|
| real_0 | 18 | 21 | 17 | 1 | 4 | 0.106 / 0.422 | 2.66 / 9.68 |
| real_1 | 21 | 18 | 17 | 4 | 1 | 0.055 / 0.172 | 2.78 / 14.52 |
| real_2 | 3 | 2 | 2 | 1 | 0 | 0.160 / 0.275 | 4.14 / 4.56 |
| real_3 | 18 | 24 | 15 | 3 | 9 | 0.184 / 0.691 | 1.76 / 6.00 |
| real_4 | 23 | 22 | 17 | 6 | 5 | 0.083 / 0.286 | 3.55 / 14.95 |

같은 물체로 매칭된 detection끼리도 confidence가 평균 0.05~0.18(최대 0.69), bbox 좌표가 평균
1.8~4.1px(최대 15px) 차이나고, 이미지마다 한쪽에만 잡히는 detection이 다수 존재한다(예: real_3은
9개가 exported에만 존재).

**Level 4 판정: DIFFERENT.** raw tensor 차이가 NMS 이후에도 사라지지 않고 실제 detection
결과(개수·confidence·bbox)에 그대로 드러난다.

## 5. Full validation metric 비교

두 onnx를 완전히 동일한 평가 pipeline(`tools/validate_symmetric_jdh.py`, 배포 ROI crop 위치
`--roi_crop`, resize 640×512 `--data_args imgsz_y=512`, conf=0.001/iou=0.7/batch=1 — onnx라 batch
고정 필요)으로 전체 validation set(6,076장)에 대해 이번에 새로 실행했다. 기존 문서의 수치는
사용하지 않았다.

```bash
uv run python3 tools/validate_symmetric_jdh.py \
  -w runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --roi_crop --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1 plots=False
# -> runs/detect/val55

uv run python3 tools/validate_symmetric_jdh.py \
  -w runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --roi_crop --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1 plots=False
# -> runs/detect/val56
```

pycocotools 엔진(`ncn_metric` 계열 `Evaluator`) 기준 AP:

| 클래스 | Reference AP@50 | Exported AP@50 | Δ | Reference AP@50-95 | Exported AP@50-95 | Δ |
|---|---|---|---|---|---|---|
| bus | 0.783 | 0.806 | +0.023 | 0.573 | 0.656 | +0.083 |
| car | 0.863 | 0.882 | +0.019 | 0.635 | 0.705 | +0.070 |
| motorcycle | 0.701 | 0.760 | +0.059 | 0.404 | 0.482 | +0.078 |
| person | 0.703 | 0.753 | +0.050 | 0.396 | 0.491 | +0.095 |
| truck | 0.776 | 0.794 | +0.018 | 0.561 | 0.638 | +0.077 |
| traffic_light | 0.624 | 0.644 | +0.020 | 0.309 | 0.334 | +0.025 |
| **all** | **0.742** | **0.773** | **+0.031** | **0.480** | **0.551** | **+0.071** |

**Level 5 판정: DIFFERENT.** exported가 6클래스 전부에서 일관되게 reference보다 높다(mAP50 기준
+0.018~+0.059, mAP50-95 기준 +0.025~+0.095 — mAP50-95 쪽 격차가 더 크다). "비슷한 편이지만
동일하지 않다"는 방향인데, **mAP가 어느 정도 근접한다고 해서 Level 2·3·4에서 이미 확정된
"다른 모델"이라는 판정이 바뀌지는 않는다** — mAP는 수천 개 detection에 걸친 통계치라 파라미터가
달라도 우연히 비슷한 값에 수렴할 수 있고, 실제로 그런 경우로 보인다.

## 6. 최초 divergence 분석

스크립트: `tools/debug/reference_compare_intermediate.py` (신규). reference 그래프의 node 출력
이름을 20개씩 묶어(chunk) `graph.output`에 추가한 임시 in-memory 사본으로 순서대로 비교, 최초로
달라지는 지점에서 중단.

```bash
uv run python3 tools/debug/reference_compare_intermediate.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --input_npy runs/debug/front_truck_reference_compare/inputs/real_0.npy \
  --out_json runs/debug/front_truck_reference_compare/intermediate_compare.json
```

```text
node 0   Conv   /model.0/conv/Conv
  input:  exact same (둘 다 images 텐서, 동일 .npy)
  weight/bias(model.0.conv.weight/bias): different (2단계에서 이미 확인)
  output(/model.0/conv/Conv_output_0): different (array_equal=False, max_abs_diff=11.586, mean_abs_diff=0.444)
```

**최초 divergence node = node 0(그래프의 첫 Conv) 자체.** 뒤 레이어로 갈수록 차이가 누적되는 게
아니라, 네트워크 진입점부터 이미 다른 파라미터로 계산되고 있다.

## 최종 판정

```text
항목                         결과
------------------------------------------------
Graph structure             SAME
Initializer                 DIFFERENT (127개 중 126개, node 0부터)
Raw output                  DIFFERENT (입력 무관하게 box 출력 차 6~10)
Decoded detection            DIFFERENT (conf/bbox 실질적 차이, only-in-one 다수)
Validation metric            DIFFERENT (방향 일관 +0.02~+0.10, 그러나 근접)
```

**결론: 현재 `best.pt`를 ONNX로 export한 결과는 NAS reference ONNX와 동일하지 않다.**

그래프 구조는 완전히 같은 export 방식(`--skip_postprocess`)으로 재현되지만, 그 안에 담긴 파라미터
(가중치·바이어스)는 그래프의 첫 레이어부터 이미 다른 값이다. 이 차이는 raw output, decode 이후의
실제 detection, 전체 validation mAP까지 일관되게(그리고 항상 같은 방향으로) 나타난다. 즉 지금
`weights/best.pt`는 이 NAS reference ONNX를 만든 것과 **같은 파라미터를 가진 체크포인트가 아니다.**
(어떤 체크포인트가 진짜 소스인지 추적하는 것은 이번 조사의 범위 밖이다.)

## 생성한 스크립트 / 산출물

신규 스크립트(전부 읽기 전용 — 원본 onnx/.pt 미수정, `tools/debug/`):

- `reference_compare_graph_and_initializers.py` — interface/node/attribute/initializer 비교
- `reference_compare_raw_output.py` — 고정 입력 raw output 비교(CPU, optimization on/off)
- `reference_compare_detections.py` — 동일 decode+NMS 적용 후 detection 비교
- `reference_compare_intermediate.py` — 최초 divergence node 탐색

신규 산출물(전부 `runs/debug/front_truck_reference_compare/`, 기존 `runs/debug/onnx_compare/`와
분리):

```text
reference_NAS.onnx              NAS 원본의 읽기 전용 사본 (md5 원본과 동일 확인)
source_best.pt                  best.pt의 사본
exported_from_best.onnx         source_best.pt를 --skip_postprocess로 export한 결과
graph_compare.json
initializer_compare.csv / .json
raw_output_compare.json
detection_compare.json
intermediate_compare.json
inputs/{zeros,ones,half,random_seed20260814,impulse,real_0..4}.npy
```

전체 validation 실행 결과는 이 repo의 표준 위치(`runs/detect/val55`(reference),
`runs/detect/val56`(exported))에 그대로 남아 있다.

## 재현 명령어

```bash
docker compose exec dair-train bash -lc '
mkdir -p runs/debug/front_truck_reference_compare/inputs

# 0단계: 원본 정보는 위 "비교 대상" 절 참고 (md5/sha256는 호스트에서 확인)

# 2단계: export
cp runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt \
   runs/debug/front_truck_reference_compare/source_best.pt
uv run python3 tools/export_symmetric.py \
  -w runs/debug/front_truck_reference_compare/source_best.pt --skip_postprocess
mv runs/debug/front_truck_reference_compare/source_best.onnx \
   runs/debug/front_truck_reference_compare/exported_from_best.onnx

# 3-4단계: graph/initializer 비교
uv run python3 tools/debug/reference_compare_graph_and_initializers.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --out_dir runs/debug/front_truck_reference_compare

# 5단계: raw output 비교 (inputs/*.npy 준비 후)
uv run python3 tools/debug/reference_compare_raw_output.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --inputs_dir runs/debug/front_truck_reference_compare/inputs \
  --out_json runs/debug/front_truck_reference_compare/raw_output_compare.json

# 6단계: detection 비교
uv run python3 tools/debug/reference_compare_detections.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --inputs runs/debug/front_truck_reference_compare/inputs/real_{0,1,2,3,4}.npy \
  --conf 0.05 --iou 0.7 \
  --out_json runs/debug/front_truck_reference_compare/detection_compare.json

# 7단계: full validation metric (전체 6,076장)
uv run python3 tools/validate_symmetric_jdh.py \
  -w runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --roi_crop --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1 plots=False
uv run python3 tools/validate_symmetric_jdh.py \
  -w runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --roi_crop --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1 plots=False

# 8단계: 최초 divergence
uv run python3 tools/debug/reference_compare_intermediate.py \
  --reference runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/front_truck_reference_compare/exported_from_best.onnx \
  --input_npy runs/debug/front_truck_reference_compare/inputs/real_0.npy \
  --out_json runs/debug/front_truck_reference_compare/intermediate_compare.json
'
```
