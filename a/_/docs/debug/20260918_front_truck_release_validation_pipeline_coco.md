# BSD_F606 Front_Truck — Release 후보 검증 파이프라인 (COCO 평가판, canonical 상태 문서)

> 이 문서는 `docs/debug/20260917_front_truck_release_validation_pipeline.md`(구 문서, 개발
> 과정 전체 이력 + Ultralytics 기반 진단 지표)와는 별도로, **공식 `pycocotools.COCOeval`
> 기준 현재 canonical 상태**만을 빠르게 파악할 수 있도록 유지하는 문서다. 시행착오·superseded
> 결과·시간순 누적 서술은 구 문서와 git history에 남아 있으므로 여기서는 반복하지 않는다.
> 완료되지 않은 항목은 추측으로 채우지 않고 `TBD`로 남긴다.

## 1. 목적

- 현재 존재하는 8개 Front_Truck 후보 모델의 동일 조건 성능을 **공식 COCOeval 기준**으로
  확립한다.
- `PT → ONNX → NVT → Board` 각 단계에서 성능이 어떻게 변하는지, canonical geometry
  (1280x1024→640x512)로 확인된 값만으로 추적한다.
- Release 후보 선정의 근거 자료를 확보한다.

## 2. Canonical Metric / Geometry Policy

**Metric**: 공식 `pycocotools.cocoeval.COCOeval`(무수정), `iouType='bbox'`, IoU
`0.50:0.05:0.95`, 기본 파라미터 변경 없음. 보고 지표는 `AP50`(`stats[1]`),
`AP50:95`(`stats[0]`), `AP75`(`stats[2]`). Ultralytics `ap_per_class`(P/R/F1, best-F1
confidence 1개 지점)는 **auxiliary**로 병기하되 COCO AP와 같은 산식으로 취급하지 않는다.

**Board Geometry** — 이 문서에서 Board canonical 결과로 인정하는 조건은 다음
하나뿐이다:

```
1920x1080 (source)
  → ROI (x=320, y=0, w=1280, h=1024)
  → 1280x1024
  → resize 640x512
  → NPU
```

과거 일부 Board 실행이 `1600x1024→640x512`(non-canonical) 조건으로 수행된 사실이
확인되어, 해당 결과는 이 문서의 canonical 비교에서 **제외**했다(구체적 수치는 구
문서/git history 참고). 이 문서의 Board 성능표에는 **1280x1024→640x512로 확인된
결과만** 기재한다.

Board dump가 crop-local 좌표(잘라낸 영역의 좌상단이 원점)로 나오는 경우:

```
x_full = x_crop + 320
y_full = y_crop
```

로 변환해 GT(full-frame)와 비교한다. 이 +320 offset은 성능을 조작하는 것이 아니라
crop-local 예측과 full-frame GT의 좌표계를 맞추는 것뿐이며, 매 dump마다 GT와 직접
대조해 좌표계(offset 0 vs 320)를 확인한 뒤에만 적용한다(추측 금지).

## 3. Pipeline 현재 상태

| Stage | 내용 | 상태 |
|---|---|---|
| 1 | PyTorch `best.pt` COCO validation | 8/8 완료 |
| 2 | `best.pt` → Split ONNX export | 8/8 완료 |
| 3 | Split ONNX Float COCO validation | 8/8 완료 |
| 4 | NVT compile | release_v01(재컴파일 안 함, 실제 shipped bin 사용) + 7개 candidate 전부 완료 |
| 5 | NVT/INT8 Simulator 1-image sanity | 6/8 완료(`v02_symmetric_bg_scratch` 미실행), release_v01은 실제 shipped bin이라 BLOCKED |
| 6 | Board canonical(1280x1024) validation | **7/8 완료**(`release_v01`, `v01_repro3`, `v02_symmetric`, `v02_symmetric_bg`, `v02_symmetric_scratch`, `v02_symmetric_bg_scratch`, `v02_dair_lr01`), 나머지 1개(`v02_dair`) TBD |
| 7 | 최종 비교 및 Release 후보 선정 | 예정 |

## 4. 대상 모델 (8개)

| run | 핵심 차이 |
|---|---|
| release_v01 | 기존 release baseline |
| v01_repro3 | release_v01 재현 실험 |
| v02_symmetric | symmetric + `imgsz_y=512`(해상도 정합 fix) |
| v02_symmetric_bg | v02_symmetric + 배경 이미지 5,769장 추가 |
| v02_symmetric_scratch | symmetric, 랜덤 초기화(scratch), SGD lr0=0.01 |
| v02_dair | dair, 랜덤 초기화(scratch), lr0=0.0001 |
| v02_dair_lr01 | dair, 랜덤 초기화(scratch), lr0=0.01 |
| v02_symmetric_bg_scratch | v02_symmetric_scratch + v02_symmetric_bg의 배경 데이터셋 조합 |

`release_v01_current720_experiment`는 candidate가 아니라 §13의 controlled
experiment이며 위 8개 수에 포함하지 않는다.

## 5. Stage 1 — PyTorch `best.pt` COCO Validation

각 run의 기존 `weights/best.pt`를 그대로 로드(재학습 없음), deployment ROI
`(320,0,1280,1024)`를 `--valid_xy 320,0`으로 적용, 입력 640x512, conf=0.001, IoU=0.7.

| run | AP50 | AP50:95 | AP75 |
|---|---:|---:|---:|
| release_v01 | 0.745 | 0.488 | 0.504 |
| v01_repro3 | 0.744 | 0.488 | 0.505 |
| v02_symmetric | 0.746 | 0.490 | 0.507 |
| v02_symmetric_bg | 0.749 | 0.491 | 0.509 |
| v02_symmetric_scratch | 0.755 | 0.492 | 0.508 |
| v02_dair | 0.296 | 0.143 | 0.121 |
| v02_dair_lr01 | 0.727 | 0.475 | 0.492 |
| v02_symmetric_bg_scratch | 0.749 | 0.488 | 0.504 |

ranking(AP50:95 기준): `v02_symmetric_scratch(0.492) > v02_symmetric_bg(0.491) >
v02_symmetric(0.490) > release_v01(0.488) ≈ v01_repro3(0.488) ≈
v02_symmetric_bg_scratch(0.488) > v02_dair_lr01(0.475) > v02_dair(0.143)`.
`v02_symmetric_bg_scratch`(scratch+배경 조합)는 각 단독 효과(scratch 0.492,
bg 0.491)보다 오히려 낮다 — 두 변경이 단순 합산되지 않는다.

### Ultralytics auxiliary (참고)

| run | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| release_v01 | 0.866 | 0.677 | 0.781 | 0.560 |
| v01_repro3 | 0.871 | 0.675 | 0.780 | 0.560 |
| v02_symmetric | 0.873 | 0.676 | 0.783 | 0.562 |
| v02_symmetric_bg | 0.869 | 0.682 | 0.784 | 0.565 |
| v02_symmetric_scratch | 0.875 | 0.689 | 0.791 | 0.566 |
| v02_dair | 0.462 | 0.302 | 0.306 | 0.151 |
| v02_dair_lr01 | 0.866 | 0.662 | 0.762 | 0.546 |
| v02_symmetric_bg_scratch | 0.871 | 0.680 | 0.784 | 0.562 |

## 6. Stage 2 — Split ONNX Export

Novatek 배포가 요구하는 형식은 output 2개(cls `[1,6,6720]` sigmoid-only, bbox
`[1,4,6720]` DFL 적분까지, decode 전)인 split ONNX다. `tools/export_symmetric.py
-p/--skip_postprocess` 옵션으로 export한다(opset 13, 640x512, simplify=False). 선택
근거와 시행착오(merged vs split 판단 과정, symlink 이슈 등)는 구 문서 §7 참고.

canonical ONNX 정책:

| run | canonical ONNX |
|---|---|
| release_v01 | `weights/260706_BSD-F606A_Front_v01.onnx` (실제 shipped, 재-export 금지) |
| 나머지 7개(v01_repro3 / v02_symmetric / v02_symmetric_bg / v02_symmetric_scratch / v02_dair / v02_dair_lr01 / v02_symmetric_bg_scratch) | 각 run의 `weights/best_split.onnx` |

7개 candidate 전부 export 성공(input `[1,3,512,640]`, output `[1,6,6720]`+`[1,4,6720]`
동일 확인).

## 7. `release_v01` PT/ONNX Provenance

`release_v01`의 로컬 PT 체크포인트와 실제 shipped ONNX가 같은 학습 산출물인지 직접
검증했다.

| checkpoint | md5 | epoch | canonical COCO (AP50/AP50:95/AP75) | → fresh split ONNX (동일 flow로 재export) |
|---|---|---:|---|---|
| `best.pt` | `412673ce9a17f52ee40fee87e680dbb4` | 40 | 0.745 / 0.488 / 0.504 | **0.745 / 0.488 / 0.504** (무손실) |
| `last.pt` | `82703f978d8c45ca55bf05d905fffa2c` | 50 | 0.743 / 0.486 / 0.503 | **0.743 / 0.486 / 0.503** (무손실) |
| 실제 shipped `260706_BSD-F606A_Front_v01.onnx` | `d623cbdc6ad13519646b6bda53e879a0` | — | **0.717 / 0.434 / 0.452** | (해당 없음, 이미 ONNX) |

**PT→fresh ONNX export 자체는 두 체크포인트 모두 완전 무손실**이다(다른 7개 candidate와
동일한 패턴). 그런데 fresh ONNX와 실제 shipped ONNX의 initializer(가중치) 비교 결과:

- `best.pt` 기반 fresh ONNX vs shipped: 127개 중 **1개만** 값이 완전히 일치(학습되지
  않는 DFL 고정 버퍼) — 나머지 126개는 값이 다름.
- `last.pt` 기반 fresh ONNX vs shipped: **동일하게 1/127**만 일치.
- 그래프 구조(node 이름·순서·op_type·attribute·opset)는 shipped와 fresh 100% 동일 —
  export 설정/파이프라인 차이는 없다. 즉 유일한 차이는 가중치 값이다.

**결론**: 실제 shipped ONNX는 현재 저장소의 `best.pt`에서도 `last.pt`에서도 export된
것이 아니다. 지금 저장소에 이 두 체크포인트 외 다른 epoch 체크포인트는 로컬/NAS
원본 어디에도 없으며(NAS 원본 md5 대조로 확인), shipped ONNX의 원본 체크포인트와
export provenance는 **복구 불가**로 남긴다.

> ⚠️ **Stage 3(§8)의 `release_v01` 행에서 PT(0.745) vs shipped ONNX(0.717) delta
> -0.028은 export loss가 아니다** — `best.pt→fresh ONNX`가 0.745→0.745로 무손실임을
> 위 표에서 직접 확인했다. PT와 shipped ONNX가 애초에 다른 provenance이기 때문에
> 생기는 차이다.

## 8. Stage 3 — Split ONNX Float COCO Validation

`tools/validate_symmetric_split_onnx.py`(batch=1 고정, `--valid_xy 320,0`)로 canonical
ONNX를 Stage 1과 동일 조건에서 평가.

| run | PT AP50 | ONNX AP50 | ΔAP50 | PT AP50:95 | ONNX AP50:95 | ΔAP50:95 | PT AP75 | ONNX AP75 | ΔAP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| release_v01 [^1] | 0.745 | 0.717 | -0.028 | 0.488 | 0.434 | -0.054 | 0.504 | 0.452 | -0.052 |
| v01_repro3 | 0.744 | 0.744 | 0.000 | 0.488 | 0.488 | 0.000 | 0.505 | 0.504 | -0.001 |
| v02_symmetric | 0.746 | 0.746 | 0.000 | 0.490 | 0.490 | 0.000 | 0.507 | 0.507 | 0.000 |
| v02_symmetric_bg | 0.749 | 0.749 | 0.000 | 0.491 | 0.491 | 0.000 | 0.509 | 0.509 | 0.000 |
| v02_symmetric_scratch | 0.755 | 0.755 | 0.000 | 0.492 | 0.492 | 0.000 | 0.508 | 0.508 | 0.000 |
| v02_dair | 0.296 | 0.296 | 0.000 | 0.143 | 0.143 | 0.000 | 0.121 | 0.121 | 0.000 |
| v02_dair_lr01 | 0.727 | 0.727 | 0.000 | 0.475 | 0.475 | 0.000 | 0.492 | 0.492 | 0.000 |
| v02_symmetric_bg_scratch | 0.749 | 0.748 | -0.001 | 0.488 | 0.488 | 0.000 | 0.504 | 0.504 | 0.000 |

[^1]: `release_v01`은 로컬 `best.pt`가 아니라 실제 shipped ONNX를 평가한 값 — §7 참고,
export loss가 아니라 provenance 차이.

**결론**: `release_v01`을 제외한 7개 candidate는 PT→ONNX export가 COCO 기준으로도
사실상 무손실(모든 delta ≤0.001).

## 9. Historical Release Reference

과거 release 문서에 기록된 `release_v01` On-Board 성능(원본 평가 조건/도구는 복구되지
않음, 참고용으로만 인용):

| class | ONNX | NVT_BIN |
|---|---:|---:|
| bus | 0.706 | 0.696 |
| car | 0.805 | 0.780 |
| motorcycle | 0.668 | 0.725 |
| person | 0.691 | 0.742 |
| truck | 0.692 | 0.687 |
| traffic_light | 0.617 | 0.592 |
| **all** | **0.697** | **0.704** |

- 이 기록의 정확한 평가 evaluator/전처리/threshold는 확인되지 않는다 — 표 제목이나
  칩 세대에 대한 추가 해석(예: 어떤 하드웨어였는지, ONNX 열이 무엇을 측정한 것인지)은
  이번 세션에서 검증하지 못했으므로 **UNKNOWN — historical evaluation provenance not
  fully recovered**로 남긴다.
- **historical `ONNX=0.697`을 현재 canonical ONNX baseline으로 쓰지 않는다.** 현재
  canonical ONNX baseline은 실제 shipped ONNX를 현재 canonical COCO pipeline으로
  평가한 **AP50=0.717, AP50:95=0.434, AP75=0.452**(§8)다.
- **historical `NVT_BIN=0.704`도 현재 canonical Board 결과로 쓰지 않는다.** 현재
  canonical Board 결과(§12)는 **AP50=0.672, AP50:95=0.393, AP75=0.404**다. 두 값
  (0.704 vs 0.672)의 차이는 존재하며, historical 실행의 provenance/runtime 조건이
  충분히 남아있지 않아 원인은 확정하지 않는다 — 0.704를 틀린 값으로 단정하지도 않고,
  현재 값과 동일시하지도 않는다. 순수 historical reference로만 유지한다.

## 10. Stage 4 — NVT Compile

`release_v01`은 재컴파일하지 않고 실제 shipped `.bin`(size 4,758,800, md5
`619b83d9bb519be69f1b9f3b37ee0a4c`)을 reference로 사용한다. 나머지 7개 candidate는
동일 compiler/calibration(720장) 조건에서 전부 compile 성공(size 4,758,792,
`[1,6,6720]`+`[1,4,6720]` 구조 유지). `v02_symmetric_bg_scratch`는 2026-09-21에
compile 완료(md5 `e985e254eec63de7920c8719aba20dcc`). 상세 이력·symlink 이슈 등은
구 문서 §9 참고.

## 11. Stage 5 — NVT/INT8 Simulator 1-image Sanity

전체 dataset 평가가 아니라 대표 1장에 대한 Float vs INT8 tensor 정합성 spot-check다
(전체 6,076장 시뮬레이터 평가는 모델당 약 50시간 소요돼 불가능 — 구 문서 §10 참고).

| run | cls cosine | bbox cosine | 결과 |
|---|---:|---:|---|
| v01_repro3 | 0.9932 | 0.9898 | PASS |
| v02_symmetric | 0.9942 | 0.9895 | PASS |
| v02_symmetric_bg | 0.9947 | 0.9898 | PASS |
| v02_symmetric_scratch | 0.9936 | 0.9922 | PASS |
| v02_dair | 0.9796 | 0.9978 | PASS |
| v02_dair_lr01 | 0.9956 | 0.9870 | PASS |
| v02_symmetric_bg_scratch | TBD | TBD | 미실행 |

`release_v01`은 실제 shipped bin이라 `siminfo.bin`(compile 시에만 생성)이 없어
BLOCKED/NOT TESTED(§13의 controlled experiment BIN은 재컴파일했으므로 시뮬레이터
결과가 있음 — §13 참고).

## 12. Stage 6 — Canonical Board COCO Validation

### Canonical Board Policy

- **1280x1024→640x512로 실행된 결과만** 인정한다.
- crop-local 좌표면 `x+320`으로 full-frame GT와 맞춘다(§2).
- non-canonical geometry로 확인된 과거 결과는 이 표에서 제외한다.

### Canonical Board Results

| run | ONNX AP50 | Board AP50 | ΔAP50 | Board AP50:95 | Board AP75 | status |
|---|---:|---:|---:|---:|---:|---|
| *Historical Release Record* (§9, 참고용 baseline) | *0.697* | *0.704* | *+0.007 (+1.0%)* | *N/A* | *N/A* | **canonical 아님** — evaluator/전처리 provenance 불명, 8개 집계·수렴 결론에 미포함 |
| release_v01 | 0.717 | **0.672** | -0.045 (-6.3%) | 0.393 | 0.404 | canonical 1280x1024 완료 |
| v01_repro3 | 0.744 | **0.704** | -0.040 (-5.4%) | 0.456 | 0.472 | canonical 1280x1024 완료 |
| v02_symmetric | 0.746 | **0.708** | -0.038 (-5.1%) | 0.460 | 0.475 | canonical 1280x1024 완료 |
| v02_symmetric_bg | 0.749 | **0.707** | -0.042 (-5.6%) | 0.459 | 0.474 | canonical 1280x1024 완료 |
| v02_symmetric_scratch | 0.755 | **0.712** | -0.043 (-5.7%) | 0.453 | 0.463 | canonical 완료 |
| v02_dair | 0.296 | TBD | TBD | TBD | TBD | board dump 대기 |
| v02_dair_lr01 | 0.727 | **0.690** | -0.037 (-5.1%) | 0.441 | 0.451 | canonical 1280x1024 완료 |
| v02_symmetric_bg_scratch | 0.748 | **0.702** | -0.046 (-6.1%) | 0.449 | 0.460 | canonical 1280x1024 완료(Stage 5 simulator는 생략) |

**7/8 완료.** canonical geometry로 정정된 일곱 값(release_v01 -6.3%, v02_symmetric_bg_scratch
-6.1%, v02_symmetric_bg -5.6%, v02_symmetric_scratch -5.7%, v01_repro3 -5.4%, v02_symmetric -5.1%,
v02_dair_lr01 -5.1%)이 전부 같은 범위(-5.1%~-6.3%)로 수렴한다 — `symmetric` 계열뿐 아니라
`dair` 계열까지 포함해서도 동일 패턴이다. `v01_repro3`의 과거 non-canonical 결과(-16.7%, 제거됨)가 canonical
재실행으로 이 범위에 들어온 것은, **모델/아키텍처마다 ONNX→Board 손실 크기가 크게
다르다는 이전 인상이 실제로는 board 입력 geometry 버그(1600x1024 vs 1280x1024)에서
비롯된 것이었고, 학습 조건 차이가 원인이 아니라는 근거**로 볼 수 있다. 나머지 3개
(v02_symmetric, v02_symmetric_bg, v02_dair)가 canonical로 확인되면 이 결론을 재검증한다.

**Historical baseline과의 대비**: historical record는 ONNX→Board가 오히려 +1.0%(손실
없음, 소폭 개선)였다. 지금 canonical 5개는 전부 -5.1%~-6.3%로, historical baseline과
**뚜렷한 격차**가 있다. 이 격차의 원인(evaluator/GT/board 실행 조건 차이 등)은 §9에서
계속 조사 중이며 아직 확정되지 않았다 — historical `0.704`가 정확히 어떤 조건에서
나온 값인지 모르는 채로 지금 canonical 결과를 그 값에 맞추려 하지 않는다.

### Canonical Board per-class

**release_v01**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.668 | 0.434 |
| car | 0.763 | 0.508 |
| motorcycle | 0.682 | 0.353 |
| person | 0.689 | 0.364 |
| truck | 0.662 | 0.435 |
| traffic_light | 0.571 | 0.267 |

Ultralytics auxiliary: P=0.782, R=0.658, F1=0.714, mAP50=0.702, mAP50-95=0.424.

**v01_repro3**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.695 | 0.491 |
| car | 0.789 | 0.569 |
| motorcycle | 0.730 | 0.423 |
| person | 0.738 | 0.458 |
| truck | 0.682 | 0.495 |
| traffic_light | 0.593 | 0.298 |

Ultralytics auxiliary: P=0.823, R=0.666, F1=0.736, mAP50=0.739, mAP50-95=0.495.

**v02_symmetric**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.701 | 0.497 |
| car | 0.792 | 0.576 |
| motorcycle | 0.734 | 0.428 |
| person | 0.742 | 0.462 |
| truck | 0.682 | 0.496 |
| traffic_light | 0.599 | 0.297 |

Ultralytics auxiliary: P=0.824, R=0.669, F1=0.739, mAP50=0.741, mAP50-95=0.497.

**v02_symmetric_bg**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.703 | 0.497 |
| car | 0.788 | 0.571 |
| motorcycle | 0.728 | 0.426 |
| person | 0.744 | 0.465 |
| truck | 0.679 | 0.490 |
| traffic_light | 0.600 | 0.303 |

Ultralytics auxiliary: P=0.827, R=0.664, F1=0.736, mAP50=0.743, mAP50-95=0.499.

**v02_symmetric_scratch**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.695 | 0.484 |
| car | 0.786 | 0.561 |
| motorcycle | 0.722 | 0.407 |
| person | 0.747 | 0.464 |
| truck | 0.677 | 0.479 |
| traffic_light | 0.646 | 0.322 |

**v02_symmetric_bg_scratch**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.693 | 0.491 |
| car | 0.790 | 0.563 |
| motorcycle | 0.720 | 0.409 |
| person | 0.738 | 0.455 |
| truck | 0.669 | 0.480 |
| traffic_light | 0.604 | 0.296 |

Ultralytics auxiliary: P=0.817, R=0.670, F1=0.735, mAP50=0.735, mAP50-95=0.486.

**v02_dair_lr01**

| class | AP50 | AP50:95 |
|---|---:|---:|
| bus | 0.695 | 0.489 |
| car | 0.781 | 0.550 |
| motorcycle | 0.714 | 0.412 |
| person | 0.727 | 0.446 |
| truck | 0.676 | 0.478 |
| traffic_light | 0.549 | 0.268 |

Ultralytics auxiliary: P=0.831, R=0.649, F1=0.728, mAP50=0.722, mAP50-95=0.476.

나머지 1개 모델(v02_dair)은 canonical Board 결과가 나오는 대로 이 절에 추가한다.

## 13. `release_v01_current720_experiment` — Controlled Experiment

`release_v01`의 Board 성능이 **모델 자체 특성**인지 **과거 calibration(500장)/과거
compiler 조건** 때문인지 분리하기 위한 실험. 실제 shipped `.bin`은 건드리지 않고,
별도 staging에서 실제 shipped **ONNX**만 현재 720장 calibration + 현재 compiler로
재컴파일했다. Candidate가 아니므로 §4/§12/§14 집계에 포함하지 않는다.

| 항목 | 값 |
|---|---|
| source ONNX | 실제 shipped `260706_BSD-F606A_Front_v01.onnx`, md5 `d623cbdc6ad13519646b6bda53e879a0` |
| 실험 BIN | size 4,758,792, md5 `98e057941393f4080e9093c2b148efbf` |
| 실제 shipped BIN(비교 대상, 무수정) | size 4,758,800, md5 `619b83d9bb519be69f1b9f3b37ee0a4c` |
| 1-image simulator sanity | cls cosine 0.9739, bbox cosine 0.9925 → PASS |
| Board(canonical 1280x1024) | **AP50=0.674, AP50:95=0.397, AP75=0.408** |

실제 shipped BIN의 canonical Board 결과(0.672/0.393, §12)와 이 실험 BIN의 결과
(0.674/0.397)는 **거의 동일**하다.

**결론**: 현재 720 calibration + 현재 compiler로 재컴파일해도 actual shipped BIN과
Board 성능이 거의 동일했다 — **calibration 데이터 수/compiler 차이가 release_v01의
Board 성능을 설명한다는 가설은 근거가 약해졌다.**

## 14. End-to-End Canonical Comparison

| Model | PT AP50 | ONNX AP50 | Board AP50 | ONNX→Board Δ | Board AP50:95 | Status |
|---|---:|---:|---:|---:|---:|---|
| *Historical Release Record* (§9, baseline 참고용) | *N/A* | *0.697* | *0.704* | *+0.007 (+1.0%)* | *N/A* | **canonical 아님** — evaluator/provenance 불명 |
| release_v01 [^2] | 0.745 | 0.717 | 0.672 | -0.045 (-6.3%) | 0.393 | Board canonical 완료 |
| v01_repro3 | 0.744 | 0.744 | 0.704 | -0.040 (-5.4%) | 0.456 | Board canonical 완료 |
| v02_symmetric | 0.746 | 0.746 | 0.708 | -0.038 (-5.1%) | 0.460 | Board canonical 완료 |
| v02_symmetric_bg | 0.749 | 0.749 | 0.707 | -0.042 (-5.6%) | 0.459 | Board canonical 완료 |
| v02_symmetric_scratch | 0.755 | 0.755 | 0.712 | -0.043 (-5.7%) | 0.453 | Board canonical 완료 |
| v02_dair | 0.296 | 0.296 | TBD | TBD | TBD | Board dump 대기 |
| v02_dair_lr01 | 0.727 | 0.727 | 0.690 | -0.037 (-5.1%) | 0.441 | Board canonical 완료 |
| v02_symmetric_bg_scratch | 0.749 | 0.748 | 0.702 | -0.046 (-6.1%) | 0.449 | Board canonical 완료(Stage 5 생략) |

[^2]: `release_v01`의 PT(0.745)는 로컬 `best.pt` 평가값이고 ONNX(0.717)는 실제 shipped
ONNX 평가값 — 서로 다른 provenance(§7)이므로 이 행의 PT→ONNX 차이를 export loss로
해석하지 않는다. `best.pt→fresh export ONNX`는 0.745→0.745로 무손실이었다(§7).

`ONNX→Board Δ`는 COCO-to-COCO 델타만이다(Ultralytics 수치와 혼합하지 않음).
`release_v01_current720_experiment`는 candidate가 아니므로 이 표에서 제외한다(§13
참고). Historical Release Record 행은 baseline 비교를 위해 맨 위에 병기했을 뿐,
canonical 8개 집계에는 포함하지 않는다 — historical(+1.0%)과 canonical 5개
(-5.1%~-6.3%)의 격차는 §12/§9에서 계속 조사 중이다.

## 15. Float ONNX Postprocess-Only A/B (vcap_iputest baseline postprocess emulation, source-matched)

§12의 ONNX→Board AP50 손실(v01_repro3 -5.4% 등)이 **양자화/NPU 실행 자체**가 아니라
**후처리(NMS/threshold) 로직 차이**만으로 어느 정도 설명되는지 분리하기 위한 실험.
NVT/Board/quantization은 전혀 개입하지 않는 **순수 Float ONNX** 비교다 — 동일 ONNX
raw output(cls/bbox, 1회 inference)에 서로 다른 두 후처리만 적용해서 비교한다.

### 조건 (모든 run 공통, 아래 표에 run이 추가돼도 이 조건은 고정)

- 데이터셋: canonical validation 전체 6,076장(`/dataset/Front_traffic_light/Validation`)
- Preprocessing/ROI: canonical `(320,0,1600,1024)→640x512`, `cv2.INTER_LINEAR`,
  BGR `[0,255]` float32, 정규화 없음(D-012/D-013) — Stage 1/3(§5/§8)과 동일 파이프라인
- GT: `/dataset/Front_traffic_light/Validation/labels`, ROI 중심점 기준 필터(§2 canonical
  geometry와 동일 관례)
- Evaluator: 공식 `pycocotools.COCOeval`, `iouType='bbox'`(§2 정책과 동일)
- ONNX raw tensor는 **A/B가 완전히 공유**(이미지당 inference 1회) — 따라서 delta는
  전적으로 후처리 로직 차이만 반영하고, inference 재현성/랜덤성 요인은 없다.

**A — 기존 PC 후처리** (`visualize_sim.py`의 `make_anchors`/`decode`, §11에서도 재사용된
바로 그 코드): conf=0.001, per-class NMS IoU=0.7.

**B — "vcap_iputest baseline postprocess emulation (source-matched)"** ⚠️ 실제
production Board runtime 그 자체는 아니며, 확인된 `post_proc_baseline`/`nms_proc_baseline`
소스 로직을 최대한 그대로 재현한 것이다:
- anchor별 top-1 class 선택(전 클래스 중 최고 score 1개만 사용)
- conf=0.001
- CAR/BUS/TRUCK은 하나의 vehicle NMS bucket으로 묶어 경쟁시키되, 살아남은 박스의
  최종 class는 원래 top-1 class를 유지(bucket label로 바뀌지 않음)
- bucket별 score 내림차순 정렬 후, **NMS 이전에 bucket당 최대 256개로 truncate**
- greedy NMS: 같은 bucket 내에서 IoU≥0.3 **또는** IoM(intersection over min-area)≥0.6이면
  suppress
- NMS 이후 별도의 global 256 truncate는 없음
- bbox는 이미 DFL 적분이 끝난 ONNX l/t/r/b를 그대로 사용
  (`x1=(anchor_x-l)·stride, y1=(anchor_y-t)·stride, x2=(anchor_x+r)·stride, y2=(anchor_y+b)·stride`)
  — sigmoid/DFL/INT16 dequant 중복 적용 없음

구현: `float_postproc_B_corrected.py`. 결과 파일: `float_postproc_ABcorrected_result_full6076_{run}.json`(7개, `v02_dair` 제외).

### 결과 — A(기존 PC 후처리) / B(vcap_iputest baseline emulation, source-matched) / 실제 Board 3자 비교

Board 실측값은 §12 canonical Board 표에서 그대로 가져온 것(재측정 아님). `ΔA→Board`는
§12/§14와 동일한 "ONNX→Board 전체 손실", `ΔB→Board`는 **postprocess를 B로 맞춘 뒤 남는
잔여 손실**이다 — `ΔB→Board`가 `ΔA→Board`보다 훨씬 작다면, 전체 손실의 대부분이
후처리 차이만으로 설명된다는 뜻이다(단, B는 emulation이라 확정적 귀속은 아님).

**AP50**

| run | A | B | Board(실측) | ΔA→Board | ΔA→Board % | ΔB→Board | ΔB→Board % |
|---|---:|---:|---:|---:|---:|---:|---:|
| release_v01 [^3] | 0.7170 | 0.6873 | 0.672 | -0.0450 | -6.28% | -0.0153 | **-2.23%** |
| v01_repro3 | 0.7439 | 0.7139 | 0.704 | -0.0399 | -5.36% | -0.0099 | **-1.39%** |
| v02_symmetric | 0.7464 | 0.7169 | 0.708 | -0.0384 | -5.15% | -0.0089 | **-1.24%** |
| v02_symmetric_bg | 0.7490 | 0.7185 | 0.707 | -0.0420 | -5.61% | -0.0115 | **-1.60%** |
| v02_symmetric_scratch | 0.7556 | 0.7224 | 0.712 | -0.0436 | -5.77% | -0.0104 | **-1.44%** |
| v02_dair_lr01 | 0.7274 | 0.7014 | 0.690 | -0.0374 | -5.14% | -0.0114 | **-1.63%** |
| v02_symmetric_bg_scratch | 0.7485 | 0.7159 | 0.702 | -0.0465 | -6.21% | -0.0139 | **-1.94%** |

(v02_dair는 요청에 따라 제외.)

[^3]: `release_v01`은 로컬 `best.pt`가 아니라 실제 shipped ONNX
(`runs/detect/BSD_F606/Front_Truck/release_v01/weights/260706_BSD-F606A_Front_v01.onnx`,
md5 `d623cbdc6ad13519646b6bda53e879a0`, §7과 동일 provenance)를 사용했다 — 처음엔 이
파일이 로컬에 없다고 판단했으나(`configs/releases/.../network/`만 확인) 실제로는
`runs/detect/.../weights/`에 있었다. A=0.7170은 §8의 shipped ONNX 평가값(0.717)과 일치.

**AP50:95 / AP75** (참고, 델타는 AP50만 강조 — §12 관례와 동일)

| run | A AP50:95 | B AP50:95 | Board AP50:95 | A AP75 | B AP75 | Board AP75 |
|---|---:|---:|---:|---:|---:|---:|
| release_v01 | 0.4339 | 0.4123 | 0.393 | 0.4527 | 0.4265 | 0.404 |
| v01_repro3 | 0.4877 | 0.4654 | 0.456 | 0.5046 | 0.4786 | 0.472 |
| v02_symmetric | 0.4896 | 0.4674 | 0.460 | 0.5072 | 0.4811 | 0.475 |
| v02_symmetric_bg | 0.4911 | 0.4684 | 0.459 | 0.5093 | 0.4826 | 0.474 |
| v02_symmetric_scratch | 0.4923 | 0.4682 | 0.453 | 0.5077 | 0.4790 | 0.463 |
| v02_dair_lr01 | 0.4751 | 0.4538 | 0.441 | 0.4924 | 0.4651 | 0.451 |
| v02_symmetric_bg_scratch | 0.4884 | 0.4648 | 0.449 | 0.5045 | 0.4766 | 0.460 |

detection 수 (conf=0.001 무필터 기준, Board 실제 detection 수와는 후처리 자체가 달라
직접 비교 의미 없음, 참고용):

| run | A n_det | B n_det |
|---|---:|---:|
| release_v01 | 607,806 | 192,968 |
| v01_repro3 | 429,369 | 162,854 |
| v02_symmetric | 437,114 | 165,835 |
| v02_symmetric_bg | 407,582 | 155,435 |
| v02_symmetric_scratch | 438,431 | 159,519 |
| v02_dair_lr01 | 468,541 | 160,178 |
| v02_symmetric_bg_scratch | 439,043 | 160,520 |

class별 AP50 (A/B, run별 — Board per-class는 §12 참고):

**release_v01**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.738 | 0.684 | -0.054 |
| car | 0.819 | 0.771 | -0.048 |
| motorcycle | 0.687 | 0.683 | -0.004 |
| person | 0.702 | 0.699 | -0.003 |
| truck | 0.734 | 0.676 | -0.058 |
| traffic_light | 0.623 | 0.611 | -0.012 |

**v01_repro3**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.762 | 0.701 | -0.061 |
| car | 0.828 | 0.783 | -0.045 |
| motorcycle | 0.735 | 0.734 | -0.001 |
| person | 0.750 | 0.746 | -0.004 |
| truck | 0.747 | 0.696 | -0.051 |
| traffic_light | 0.640 | 0.624 | -0.016 |

**v02_symmetric**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.768 | 0.709 | -0.059 |
| car | 0.829 | 0.788 | -0.041 |
| motorcycle | 0.738 | 0.736 | -0.002 |
| person | 0.751 | 0.751 | 0.000 |
| truck | 0.749 | 0.688 | -0.061 |
| traffic_light | 0.644 | 0.629 | -0.015 |

**v02_symmetric_bg**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.771 | 0.710 | -0.061 |
| car | 0.831 | 0.790 | -0.041 |
| motorcycle | 0.742 | 0.737 | -0.005 |
| person | 0.756 | 0.754 | -0.002 |
| truck | 0.749 | 0.687 | -0.062 |
| traffic_light | 0.646 | 0.632 | -0.014 |

**v02_symmetric_scratch**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.767 | 0.703 | -0.064 |
| car | 0.833 | 0.791 | -0.042 |
| motorcycle | 0.739 | 0.731 | -0.008 |
| person | 0.759 | 0.754 | -0.005 |
| truck | 0.748 | 0.682 | -0.066 |
| traffic_light | 0.686 | 0.674 | -0.012 |

**v02_dair_lr01**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.761 | 0.706 | -0.055 |
| car | 0.824 | 0.782 | -0.042 |
| motorcycle | 0.722 | 0.723 | +0.001 |
| person | 0.739 | 0.740 | +0.001 |
| truck | 0.743 | 0.684 | -0.059 |
| traffic_light | 0.576 | 0.572 | -0.004 |

**v02_symmetric_bg_scratch**

| class | A | B | Δ(B-A) |
|---|---:|---:|---:|
| bus | 0.765 | 0.706 | -0.059 |
| car | 0.831 | 0.789 | -0.042 |
| motorcycle | 0.738 | 0.731 | -0.007 |
| person | 0.752 | 0.748 | -0.004 |
| truck | 0.745 | 0.674 | -0.071 |
| traffic_light | 0.660 | 0.648 | -0.012 |

**관찰**:
- corrected-B는 old-B(NMS 이후 global 256-limit)와 **7개 run 전부에서 사실상 동일**하다
  — AP50/AP50:95/AP75가 6개 run은 완전히 동일, `v02_symmetric_scratch`만 AP50
  0.7228→0.7224로 미세하게 다르다(class person 1건 수준). detection 수도 전체 대비
  0.03~0.09%만 감소했다.
- 즉 **256-limit을 어디에 적용하는지(NMS 전/후, bucket 단위/global 단위)는 결과에
  거의 영향이 없다** — 이미지당 vehicle bucket 후보 수가 애초에 256에 거의 도달하지
  않는 조건이기 때문이다.
- 따라서 **A→B 손실의 주요 원인은 256-limit이 아니라 CAR/BUS/TRUCK vehicle bucket
  NMS + IoU≥0.3 OR IoM≥0.6 조건 쪽으로 보는 것이 더 타당**하다(observational). 실제로
  손실은 CAR/BUS/TRUCK(bucket NMS 적용 대상)에 집중되고, motorcycle/person(bucket
  미적용)은 거의 무손실이며, 이 패턴은 `dair` 계열(v02_dair_lr01)에서도 동일하게
  나타나 아키텍처와 무관한 후처리 자체의 효과로 보인다.
- **Board 실측값이 확인된 7개 run의 corrected-B→Board 잔여 손실은 -1.24%~-2.23%**로
  좁게 수렴한다(release_v01 -2.23%, v01_repro3 -1.39%, v02_symmetric -1.24%,
  v02_symmetric_bg -1.60%, v02_symmetric_scratch -1.44%, v02_dair_lr01 -1.63%,
  v02_symmetric_bg_scratch -1.94%) — 원래 ONNX→Board 손실(-5.1%~-6.3%)의 대부분이
  postprocess를 B로 맞추는 것만으로 사라진다.
- 다만 이 잔여(-1.24%~-2.23%)를 **전처리(JPEG/NV12)나 NPU 실행 손실로 아직 확정하지
  않는다** — B가 실제 production Board runtime과 완전히 같다는 보장이 없는 상태에서
  남은 값이므로, "잔여 = 전처리/NPU 손실"이라는 등식은 성립을 가정하지 않은 채로 둔다.

### 재현/확장 방법 (다른 모델 추가 시)

두 스크립트를 순서대로 사용(둘 다 `runs/debug/1091_nv12_csc_candidates/`, `RUN_NAME`을
인자로 받음 — `ONNX_PATH`는 자동으로 `runs/detect/BSD_F606/Front_Truck/{RUN_NAME}/weights/`
아래에서 결정):
1. `float_postproc_ab.py {RUN_NAME} [onnx_filename]` — A 계산 + GT 캐시 생성
   (`float_gt_full6076_{RUN_NAME}.json`, `float_postproc_AB_result_full6076_{RUN_NAME}.json`
   에 A가 저장됨).
2. `float_postproc_B_corrected.py {RUN_NAME} [onnx_filename]` — 캐시된 GT/A를 재사용하고
   corrected-B만 새로 계산, 최종 결과를 `float_postproc_ABcorrected_result_full6076_{RUN_NAME}.json`에
   저장(`A_existing_pc_postproc`/`B_old_emulation_per_raw_class_cap`은 옛 캐시 참고용,
   canonical로 쓸 값은 `B_corrected_per_nms_bucket_cap`).

다음 모델 진행 시 위 표에 행만 추가하면 된다. `release_v01`은 shipped ONNX가 별도
provenance(§7)이므로 포함 시 그 사실을 명시할 것.

## 16. Remaining TBD / 다음 단계

- **§15 Float ONNX postprocess-only A/B, corrected-B 기준 7/8 완료, Board 실측 대조도
  7/7 완료**(release_v01, v01_repro3, v02_symmetric, v02_symmetric_bg,
  v02_symmetric_scratch, v02_dair_lr01, v02_symmetric_bg_scratch). `v02_dair`만 요청에
  따라 제외. Vehicle bucket NMS + IoU/IoM 조건에서만 손실이 크게 나오는 패턴과
  "corrected-B→Board 잔여 손실(-1.24%~-2.23%)이 A→Board 전체 손실(-5.1%~-6.3%)보다
  훨씬 작다"는 패턴이 Board 실측이 있는 7개 run 전부에서 재현됨.
- **Board canonical 결과 미확보(1/8)**: `v02_dair` — 1280x1024→640x512 canonical
  geometry로 처음부터 실행 필요.
- 7개(release_v01 -6.3%, v02_symmetric_bg_scratch -6.1%, v02_symmetric_bg -5.6%,
  v02_symmetric_scratch -5.7%, v01_repro3 -5.4%, v02_symmetric -5.1%, v02_dair_lr01 -5.1%)가
  비슷한 범위로 수렴한 것은 "모델별 quantization loss 차이가 크다"는 이전 인상이 board
  geometry 버그 때문이었다는 근거다(§12) — `symmetric`/`dair` 두 아키텍처 계열 모두
  포함. 나머지 1개(v02_dair)로 재검증 필요.
- `v02_symmetric_bg_scratch`의 Stage 5(simulator)는 결국 생략된 채 Board가 먼저
  실행됨 — 필요시 추후 보완.
- §9의 historical `ONNX=0.697` 평가 provenance는 복구되지 않음 — 추가 자료가 나오면
  재조사.
- §9의 historical `NVT_BIN=0.704` vs 현재 canonical Board(0.672) 차이의 원인은
  미확정 — `release_v01_1280x1024_output.json` 재현성 조사가 별도로 필요하면 진행.
- 8개 candidate 전부 canonical Board 결과가 모이면 Stage 7(최종 비교 및 Release 후보
  선정) 착수.
