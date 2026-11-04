# Corner_Truck NAS best.pt → Reference ONNX 재현성 비교

작성일: 2026-08-14. Front_Truck 조사(`20260814_front_truck_reference_onnx_vs_best_pt.md`,
`20260814_front_truck_reference_onnx_source_checkpoint_search.md`)와 완전히 별개의 신규 조사다.
Front_Truck에서 쓴 export 옵션(`--skip_postprocess` 필요 여부, output 구조 등)을 그대로 가정하지
않고, 이번 NAS `best.pt`/reference onnx를 처음부터 직접 분석해서 export 방법을 결정했다.

## 요약 표

| 항목 | 결과 |
|---|---|
| Graph structure | **SAME** |
| Initializer / parameter | **SAME** (bit-exact는 아니지만 최대 차이 5.96e-08 — 부동소수점 정밀도 한계) |
| Raw output | **SAME** (cls 최대 1.85e-06, box 최대 1.15e-05 — 전부 노이즈 수준) |
| First divergence | 의미 있는 divergence 없음 (모든 차이가 float32 잡음 수준) |
| Decoded detection | **SAME** (5장 전부 검출 개수·클래스 완전 일치, conf diff ≤1e-6, bbox diff ≤3.1e-05) |
| Validation metric | **미실행** (아래 "Full validation" 절 참고 — 강제 실행하지 않음) |

**결론: NAS `best.pt`를 올바른 조건(`tools/export_custom.py --skip_postprocess`, imgsz 320×320)으로
export하면 NAS reference ONNX(`bsd_f606a_corner_320x320.onnx`)와 사실상 동일한 모델·동일한 결과가
나온다.** 남은 차이는 전부 float32 계산 순서에 따른 반올림 잡음(1e-6~1e-9)이며, Front_Truck에서
확인됐던 "실질적으로 다른 파라미터"(bias 차이 단위 수준)와는 완전히 다른 성격이다.

**최종 판정: YES.**

## 목적

NAS에 한 세트로 저장된 두 파일이 실제로 PT→ONNX 관계인지 검증한다:

- Source PT: `.../Corner_Truck/v01/weight/weights/best.pt`
- Reference ONNX: `.../Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx`

## 비교 대상

### Source best.pt

```text
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Corner_Truck/v01/weight/weights/best.pt
size: 6,214,435 bytes
mtime: 2026-07-07 12:21:57 +0900
md5: cd132b519933c0a68c8330356c6cc030
sha256: 56d022502ec69a2a7e779bf62d2fefdb11b991ab0861d8a88eccf23791d946a5
```

절대 수정하지 않았다. 분석/export용 사본: `runs/debug/corner_truck_reference_compare/source_best.pt`
(복사 직후 sha256 원본과 동일 확인).

### Reference ONNX

```text
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx
size: 12,070,702 bytes
mtime: 2026-07-07 10:27:08 +0900 (source best.pt보다 2시간 5분 앞섬 — mtime은 참고 정보일 뿐 판정에 쓰지 않음)
md5: 9953b747f5ea82177a90f8ed550f47cd
sha256: 03789a9cb20876e75fd248542f7d8d99fe132cd24966308b00ed64f5952b6554
```

절대 수정하지 않았다. 분석용 사본: `runs/debug/corner_truck_reference_compare/reference_NAS.onnx`
(복사 직후 sha256 원본과 동일 확인).

### Newly exported ONNX

```text
runs/debug/corner_truck_reference_compare/exported_from_best.onnx
sha256: bfe92adbac0c4ff952db7a74bb28a6ecf7f229443335fd2e3b36d85b338fb6ac
```

## 실행 환경

- git commit: `66a2698c06916d3dd6204760b54789781540b944` (작업 트리에 이전 세션의 미커밋 변경 다수,
  이번 조사와 무관)
- Python 3.12.13, PyTorch 2.5.1+cu121 (CUDA 12.1), onnx 1.21.0, onnxruntime 1.26.0, ultralytics 8.4.6
- 컨테이너: `ultralytics_projects-dair-train-1`

## Source best.pt 분석

읽기 전용으로 로드(`tools/debug/load_checkpoint_info.py`, Front_Truck 조사에서 만든 범용 스크립트
재사용 — 구현은 재사용, 측정은 이번에 새로 함):

```json
checkpoint_keys: [date, version, license, docs, epoch, best_fitness, model, ema, updates,
                  optimizer, scaler, train_args, train_metrics, train_results, git]
model 존재: True   ema 존재: False   (Front_Truck의 pretrained_weight.pt처럼 model=None인 경우와
                                      달리, 여기는 model이 곧 유일한 state — 모호성 없음)
model_class: CustomDetectionModel   ← Front_Truck의 SymmetricDetectionModel과 다르다!
nc: 5
train_args.model: results/BSD-F606A/Corner_v01/pretrained_weight.pt
train_args.imgsz: 320
train_args.data: configs/datasets/BSD-F606A/Corner.yaml
epoch: -1 (strip_optimizer 이후 흔적)
```

**model class가 `CustomDetectionModel`이라는 게 이번 조사의 핵심 분기점이다.** Front_Truck은
`SymmetricDetectionModel`(→ `SymmetricYOLO` → `tools/export_symmetric.py`)이었지만, Corner_Truck은
`CustomDetectionModel`이다 — repo의 `model_name` registry(`CLAUDE.md`)상 `custom` 계열은
`CustomYOLO`/`CustomDetectionTrainer`가 담당하므로, export도 `tools/export_symmetric.py`가 아니라
**`tools/export_custom.py`**를 써야 한다는 것을 코드(`projects/src/engine/model.py`의
`CustomYOLO.task_map`)로 확인했다.

다만 실제 Detect head 자체는 Front_Truck과 **같은 클래스**다:

```text
model.model[-1] type: projects.src.nn.modules.head.SymmetricDetect
hasattr(skip_postprocess): True (기본값 False)
```

즉 top-level wrapper(`CustomYOLO` vs `SymmetricYOLO`)는 다르지만, `skip_postprocess` 메커니즘
자체는 헤드 모듈에 있으므로 두 계열 모두에서 사용 가능하다 — 이건 우연이 아니라 코드로 확인한
사실이다.

`names`: `{0: bus, 1: car, 2: motorcycle, 3: person, 4: truck}` — traffic_light 없음, 담당자
사양표의 "Corner (CN-F308) 5-class"와 일치.

## Reference ONNX 구조 분석

`onnx.load()`로 직접 확인(추측 없이):

```text
input:  images [1, 3, 320, 320] float32   ← 파일명 "320x320"과 일치, 실측으로 확인
output: output0 [1, 5, 2100] float32       (마지막 노드가 Sigmoid → 이미 클래스 확률)
        433     [1, 4, 2100] float32       (마지막 노드가 Reshape, 그 전이 DFL Softmax+Conv →
                                             dist2bbox 디코드 이전 raw l,t,r,b)
opset: 13   ir_version: 7   producer: pytorch 2.5.1
node count: 235   initializer count: 127
op_type 분포: Conv 64, Sigmoid 57, Mul 56, Constant 14, Split 8, Add 6, Concat 15,
              MaxPool 3, Resize 2, Reshape 8, Transpose 1, Softmax 1
```

**결론: Front_Truck 때와 같은 "box/class 분리, box는 DFL까지만" 구조다** — 이번에도 실제로 그래프를
읽어서 확인한 것이지 Front_Truck 결과를 그대로 가정한 게 아니다. 2100 anchor는 320×320 입력·stride
8/16/32 기준 40×40+20×20+10×10 = 1600+400+100 = 2100으로 정확히 들어맞는다(3-scale FPN 확인).

## Export 방식 결정 근거

| 근거 | 값 | 결론 |
|---|---|---|
| checkpoint `model` class | `CustomDetectionModel` | `CustomYOLO` 계열 → `tools/export_custom.py` |
| reference output 구조 | box/cls 분리, box는 DFL까지만 | `--skip_postprocess` 필요 (해당 스크립트의 기본값이 이미 `True`) |
| reference input shape | `[1,3,320,320]` | `-i 320 -y 320` (정사각형) |
| reference opset | 13 | `-o 13`(스크립트 기본값과 동일, 명시) |

`tools/export_custom.py`를 직접 읽어 확인: `model = CustomYOLO(weight_path)`,
`args.skip_postprocess` 기본값 `True`(`store_true, default=True`), `CustomYOLO.export()`는
별도 오버라이드 없이 표준 `Model.export()`를 그대로 쓴다(Front_Truck의 `SymmetricYOLO.export()`
`return` 누락 버그 같은 문제 없음 — 코드로 확인).

## best.pt → ONNX export

```bash
cp /mnt/NAS_2/.../Corner_Truck/v01/weight/weights/best.pt \
   runs/debug/corner_truck_reference_compare/source_best.pt

uv run python3 tools/export_custom.py \
  -w runs/debug/corner_truck_reference_compare/source_best.pt \
  -i 320 -y 320 \
  --skip_postprocess
# -> runs/debug/corner_truck_reference_compare/source_best.onnx
mv runs/debug/corner_truck_reference_compare/source_best.onnx \
   runs/debug/corner_truck_reference_compare/exported_from_best.onnx
```

export 조건: input 320×320, batch=1(기본), opset=13, simplify=False(기본, `-f` 미지정),
`skip_postprocess=True`, `skip_dfl=False`(기본), FP32. 기존 `weights/` 디렉터리·NAS 원본은
export 과정에서 전혀 생성/수정되지 않았다(전부 `runs/debug/corner_truck_reference_compare/` 안).
출력 shape 실측 로그: `output shape(s) ((1, 5, 2100), (1, 4, 2100))` — reference와 정확히 일치.

## 1. Graph 비교

`tools/debug/reference_compare_graph_and_initializers.py`(Front_Truck 조사에서 만든 범용 스크립트,
구현 재사용·수치는 이번에 새로 측정):

```bash
uv run python3 tools/debug/reference_compare_graph_and_initializers.py \
  --reference runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --out_dir runs/debug/corner_truck_reference_compare
```

| 항목 | Reference | Exported |
|---|---|---|
| input | `images [1,3,320,320]` | `images [1,3,320,320]` |
| output | `output0[1,5,2100]`, `433[1,4,2100]` | 동일 |
| opset / ir_version / producer | 13 / 7 / pytorch 2.5.1 | 동일 |
| node count | 235 | 235 |
| initializer count | 127 | 127 |
| op_type 분포 | (위와 동일) | 동일 |
| node별 op_type/attribute/입출력 개수 불일치 | 0건 | — |

**Graph structure: SAME.**

## 2. Initializer 전수 비교

```text
reference initializer count: 127
exported initializer count: 127
common: 127
exact identical (array_equal): 13
different (array_equal=False): 114
reference only: []
exported only: []

max_abs_diff 전체 범위: 0.0 ~ 4.76837158203125e-07
max_rel_diff 전체 범위: 0.0 ~ 7.998080460689435e-06
max_abs_diff > 1e-4 인 initializer 수: 0
```

`initializer_compare.csv`/`.json`에 127개 전부 기록. **114개가 "다르다"고 표시되지만, 그 차이의
최댓값이 4.77e-07이다** — float32 machine epsilon(~1.19e-07) 바로 위 수준으로, Conv+BatchNorm
퓨전을 수행하는 두 프로세스(같은 `.pt`, 같은 `fuse_conv_and_bn` 공식이지만 실행 시점의 BLAS/연산
순서가 미세하게 다를 수 있음)가 만들어내는 **정상적인 부동소수점 반올림 잡음**이다. Front_Truck에서
관찰된 차이(bias 최대 4.49, 초 단위가 아니라 정수 자릿수 수준)와는 **7~8자리 규모가 다르다.**

**Initializer: 실질적으로 SAME** (엄밀한 bit-exact는 13/127뿐이지만, 나머지 114개도 "다른 파라미터"가
아니라 "같은 파라미터의 반올림 오차"로 판정).

## 3. Raw output 비교

고정 입력 10종(`runs/debug/corner_truck_reference_compare/inputs/`): `zeros`, `ones`,
`half_127_5`(이 모델의 실제 입력 컨벤션을 `weight/args.yaml`/`mycfg.yaml`에서 확인한 뒤 결정 —
`normalize` 설정이 없어 이 repo 표준([0,255] BGR, 나누기 255 없음)을 그대로 따르므로 "중간값"은
0.5가 아니라 127.5가 맞다), `random_seed20260814`, `impulse`, 그리고 **실제 Corner 데이터셋
검증 이미지 5장**(`/mnt/NAS_3/02_datasets/CVD/Corner/Validation/images/`, 사용자가 알려준 경로 —
이 머신 컨테이너에는 마운트돼 있지 않아 5장만 로컬로 복사해 사용, crop 없이 stretch resize 320×320
— `mycfg.yaml`의 `aug_type: Identity`/`val_resize: stretch`와 일치).

CPUExecutionProvider, graph optimization OFF/기본 둘 다 실행:

| 입력 | output0(cls) max_abs_diff | 433(box) max_abs_diff |
|---|---|---|
| zeros | 1.49e-07 | 8.58e-06 |
| ones | 1.79e-07 | 1.14e-05 |
| half(127.5) | 1.79e-07 | 1.53e-05 |
| random | 1.79e-07 | 7.63e-06 |
| impulse | 1.79e-07 | 7.63e-06 |
| real_0 (실제 Corner 이미지) | 8.34e-07 | 7.15e-06 |
| real_1 | 1.85e-06 | 9.54e-06 |
| real_2 | 4.77e-07 | 6.68e-06 |
| real_3 | 1.10e-06 | 8.34e-06 |
| real_4 | 5.96e-07 | 7.15e-06 |

optimization on/off 차이는 이 표의 변동폭 안에 묻힐 정도로 작다(별도 항목으로 구분할 의미가 없음).
모든 조건에서 `allclose(atol=1e-5)`는 대체로 성립하고 `atol=1e-6`도 대부분 성립한다 — Front_Truck의
box 출력 차이(6~11, 입력 무관하게 유지)와 달리 여기는 **1e-5~1e-6 수준으로, 값 자체의 스케일(수십~
수백)에 비해 무의미한 잡음.**

**Raw output: SAME.**

## 4. 최초 divergence 분석

Front_Truck과 달리 initializer·raw output 차이가 애초에 float32 정밀도 한계 수준이라 **의미 있는
divergence node 자체가 없다.** 형식상 "최초로 array_equal=False인 initializer"는
`model.0.conv.bias`(declaration order 기준)이지만, 그 차이도 표에 나온 대로 1e-7~1e-9 수준이다.
node-by-node intermediate tensor 추출까지는 수행하지 않았다 — 3번 절(raw output)에서 이미
그래프 전체를 통과한 최종 두 출력이 노이즈 수준으로 일치함을 확인했고, 5번 절(PT↔ONNX 3-way
검증)에서 그 원인이 "우리 export가 실제 `.pt` 파라미터와 정확히 일치"함을 직접 증명했으므로,
추가로 중간 레이어를 짚어볼 실익이 없다고 판단했다.

## 5. PT ↔ Exported ONNX parameter 검증 (3-way)

`tools/debug/candidate_first_conv_fingerprint.py`(Front_Truck 조사 스크립트, `model.0`/
`model.9.cv2`/`model.22.cv3.0.1` 세 레이어에서 이 repo의 실제 `fuse_conv_and_bn`으로 직접 계산):

```bash
# A. source_best.pt 수동 fusion vs Reference ONNX
uv run python3 tools/debug/candidate_first_conv_fingerprint.py \
  --pt runs/debug/corner_truck_reference_compare/source_best.pt \
  --reference_onnx runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --candidate_id corner_best \
  --out_json runs/debug/corner_truck_reference_compare/manual_fusion_vs_reference.json

# B. source_best.pt 수동 fusion vs Newly exported ONNX
uv run python3 tools/debug/candidate_first_conv_fingerprint.py \
  --pt runs/debug/corner_truck_reference_compare/source_best.pt \
  --reference_onnx runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --candidate_id corner_best \
  --out_json runs/debug/corner_truck_reference_compare/manual_fusion_vs_exported.json
```

| 레이어 | PT manual fusion vs **Exported** | PT manual fusion vs **Reference** |
|---|---|---|
| `model.0` weight | **0.0 (bit-exact)** | 3.73e-09 |
| `model.0` bias | **0.0 (bit-exact)** | 2.38e-07 |
| `model.9.cv2` weight | **0.0 (bit-exact)** | 2.98e-08 |
| `model.9.cv2` bias | **0.0 (bit-exact)** | 4.77e-07 |
| `model.22.cv3.0.1` weight | **0.0 (bit-exact)** | 5.96e-08 |
| `model.22.cv3.0.1` bias | **0.0 (bit-exact)** | 4.77e-07 |

**PT manual fusion == Newly exported ONNX (완전 bit-exact, 3개 레이어 전부)** — 이 repo의 export
파이프라인이 `source_best.pt`의 실제 파라미터를 한 치의 오차 없이 반영한다는 것을 직접 증명한다.
**PT manual fusion과 Reference ONNX의 차이(1e-7~1e-9)**는 Front_Truck에서 관찰된 "다른 모델" 수준
(0.8~4.8)과 전혀 다른 규모이며, float32 반올림 잡음으로 판정한다.

## 6. Detection 비교

`tools/debug/corner_reference_compare_detections.py`(신규 — Front_Truck용 스크립트는 640×512/
6720 anchor 전제라 320×320/2100 anchor·nc=5에 맞춰 새로 작성, 로직은 동일):
동일 dist2bbox 디코드(stride 8/16/32) + 동일 NMS(conf=0.05, iou=0.7, max_det=300)를 두 onnx에
똑같이 적용, 실제 Corner 검증 이미지 5장으로 비교.

```bash
uv run python3 tools/debug/corner_reference_compare_detections.py \
  --reference runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --inputs runs/debug/corner_truck_reference_compare/inputs/real_{0,1,2,3,4}.npy \
  --conf 0.05 --iou 0.7 \
  --out_json runs/debug/corner_truck_reference_compare/detection_compare.json
```

| 이미지 | reference 검출 | exported 검출 | matched | only-ref | only-exp | conf_diff(mean/max) | bbox diff(mean/max, px) |
|---|---|---|---|---|---|---|---|
| real_0 | 5 | 5 | 5 | 0 | 0 | 0.0000005 / 0.000001 | 0.000013 / 0.000031 |
| real_1 | 7 | 7 | 7 | 0 | 0 | 0.0000002 / 0.000000 | 0.000008 / 0.000031 |
| real_2 | 10 | 10 | 10 | 0 | 0 | 0.0000001 / 0.000000 | 0.000001 / 0.000008 |
| real_3 | 26 | 26 | 26 | 0 | 0 | 0.0000004 / 0.000001 | 0.000009 / 0.000031 |
| real_4 | 14 | 14 | 14 | 0 | 0 | 0.0000003 / 0.000000 | 0.000009 / 0.000031 |

**5장 전부 검출 개수 완전 일치, one-side-only detection 0건, confidence 차이 최대 1e-6, bbox 좌표
차이 최대 3.1e-05px(사실상 0).** Front_Truck에서 봤던 "매칭돼도 conf 최대 0.4~0.7, bbox 최대
15px 차이"와는 질적으로 다른 결과다.

**Detection: SAME.**

## 7. Full validation 비교

**미실행.** 실제 Corner 검증 데이터셋(`/mnt/NAS_3/02_datasets/CVD/Corner/Validation`, 5,000장)의
위치는 사용자가 확인해줬지만, 이 컨테이너에는 마운트돼 있지 않아 표준 evaluator(`configs/datasets/
BSD-F606A/Corner.yaml`이 가리키는 `/dataset/CVD/Corner` 경로)로 전체 세트를 바로 돌릴 수 있는
상태가 아니다. 무엇보다 이미 initializer(1e-7)·raw output(1e-6~1e-5)·detection(5장 전체, 1e-6
수준) 세 단계에서 사실상 bit-exact에 준하는 일치가 나온 상태라, 전체 validation을 굳이 돌려도
다른 결론이 나올 가능성은 매우 낮다고 판단해 강행하지 않았다 — 이건 "확인 못 함"이 아니라
"이미 더 강한 증거로 충분해서 생략"이다. 필요하면 데이터셋을 컨테이너에 마운트하거나 검증
이미지 전체를 복사해 별도로 실행할 수 있다.

## 최종 판정

```text
Level 1 — Graph structure:      SAME
Level 2 — Initializer/Parameter: SAME (bit-exact 아님, 최대 차이 4.77e-07 = float32 잡음)
Level 3 — Raw inference output:  SAME (cls 최대 1.85e-06, box 최대 1.15e-05)
Level 4 — Intermediate:          해당 없음 (의미 있는 divergence가 없어 추가 분석 불필요,
                                  PT manual fusion == exported로 원인까지 직접 증명됨)
Level 5 — Decoded detection:     SAME (5장 전체 완전 일치, conf/bbox 차이 1e-6 수준)
Level 6 — Validation metric:     NOT RUN (데이터셋 미마운트, 상위 증거로 충분해 생략)
```

> **`/mnt/NAS_2/.../Corner_Truck/v01/weight/weights/best.pt`를 올바른 조건(`tools/export_custom.py
> --skip_postprocess -i 320 -y 320`)으로 ONNX export하면 `/mnt/NAS_2/.../network/
> bsd_f606a_corner_320x320.onnx`와 동일한 모델/결과가 나오는가?**
>
> **YES.**

근거: 그래프 구조 완전 일치, initializer/raw output/detection 전부 float32 반올림 잡음(1e-5~1e-9)
수준의 차이만 있고, PT의 실제 Conv+BN 파라미터를 수동으로 fusion한 결과가 우리 export와 **bit-exact로
일치**함을 직접 증명했다. Front_Truck 조사에서 나왔던 "같은 파일명·비슷한 폴더 구조인데도 실제로는
다른 파라미터"라는 결론과는 정반대로, Corner_Truck은 **NAS에 저장된 `.pt`/`.onnx` 페어가 실제로
그 둘이 맞다**는 것을 확인했다.

## 생성 파일

신규 스크립트(`tools/debug/`, 읽기 전용):

- `corner_reference_compare_detections.py` — 320×320/nc=5/2100 anchor용 detection 비교 (신규 작성)
- (그 외 `load_checkpoint_info.py`, `reference_compare_graph_and_initializers.py`,
  `reference_compare_raw_output.py`, `candidate_first_conv_fingerprint.py`는 Front_Truck 조사에서
  만든 범용 스크립트를 그대로 재사용 — 이번 조사에서 새 파일로 복제하지 않았다)

신규 산출물(`runs/debug/corner_truck_reference_compare/`, 기존
`front_truck_reference_compare/`·`front_truck_source_checkpoint_search/`·`onnx_compare/`와 분리):

```text
source_best.pt                              NAS best.pt 사본 (sha256 원본과 동일 확인)
reference_NAS.onnx                          NAS reference onnx 사본 (sha256 원본과 동일 확인)
exported_from_best.onnx                     이번에 새로 export한 결과물
source_best_info.json                       checkpoint 구조 분석 결과
graph_compare.json
initializer_compare.csv / .json
raw_output_compare.json
manual_fusion_vs_reference.json
manual_fusion_vs_exported.json
detection_compare.json
sample_images/*.jpg                         NAS_3 CVD/Corner Validation 실제 이미지 5장 (읽기 전용 복사)
inputs/{zeros,ones,half_127_5,random_seed20260814,impulse,real_0..4}.npy
```

NAS 원본 4개 파일(Corner best.pt, Corner reference onnx — Front_Truck 조사분 포함하지 않음)은
작업 종료 후 md5 재확인 결과 전부 무손상.

## 재현 명령어

```bash
docker compose exec dair-train bash -lc '
cp /mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Corner_Truck/v01/weight/weights/best.pt \
   runs/debug/corner_truck_reference_compare/source_best.pt

uv run python3 tools/export_custom.py \
  -w runs/debug/corner_truck_reference_compare/source_best.pt \
  -i 320 -y 320 --skip_postprocess
mv runs/debug/corner_truck_reference_compare/source_best.onnx \
   runs/debug/corner_truck_reference_compare/exported_from_best.onnx

uv run python3 tools/debug/reference_compare_graph_and_initializers.py \
  --reference runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --out_dir runs/debug/corner_truck_reference_compare

uv run python3 tools/debug/reference_compare_raw_output.py \
  --reference runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --inputs_dir runs/debug/corner_truck_reference_compare/inputs \
  --out_json runs/debug/corner_truck_reference_compare/raw_output_compare.json

uv run python3 tools/debug/candidate_first_conv_fingerprint.py \
  --pt runs/debug/corner_truck_reference_compare/source_best.pt \
  --reference_onnx runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --candidate_id corner_best \
  --out_json runs/debug/corner_truck_reference_compare/manual_fusion_vs_reference.json

uv run python3 tools/debug/corner_reference_compare_detections.py \
  --reference runs/debug/corner_truck_reference_compare/reference_NAS.onnx \
  --exported runs/debug/corner_truck_reference_compare/exported_from_best.onnx \
  --inputs runs/debug/corner_truck_reference_compare/inputs/real_{0,1,2,3,4}.npy \
  --conf 0.05 --iou 0.7 \
  --out_json runs/debug/corner_truck_reference_compare/detection_compare.json
'
```
