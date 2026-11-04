# Corner_Truck v01 — Reference ONNX 재현성과 NVT 검증

> 원문: `20260814_corner_truck_reference_onnx_vs_nas_best_pt.md`(CV), `20260814_corner_truck_v01_nvt_calibration_simulator_comparison.md`(CN).
> 공통 재현 환경/evaluator 정의는 [90_reproduction_reference.md](90_reproduction_reference.md) 참조.
> Front_Truck에서 쓴 export 전제(스크립트/output 구조)를 그대로 가정하지 않고, 이번 NAS
> `best.pt`/reference onnx를 처음부터 직접 분석해 독립적으로 export 방법을 결정한 조사다.

## 0. 핵심 결론

**Corner_Truck NAS `best.pt`를 올바른 조건(`tools/export_custom.py --skip_postprocess -i 320
-y 320`)으로 export하면 NAS reference ONNX(`bsd_f606a_corner_320x320.onnx`)와 사실상 동일한
모델·동일한 결과가 나온다 — YES.** 이는 [Front_Truck](02_front_reference_and_reproduction.md)에서
확인된 "그래프는 같지만 파라미터가 다른 모델"과 **정반대** 결과다 — Corner_Truck은 NAS에
저장된 `.pt`/`.onnx` 페어가 실제로 그 관계가 맞다. 남은 차이는 전부 float32 계산 순서에 따른
반올림 잡음(1e-9~1e-5 수준)이며, Front_Truck에서 확인됐던 "실질적으로 다른 파라미터"(bias
차이가 정수 단위)와는 **7~8자리 규모가 다르다** — "그래프 동일"과 "파라미터 근사 동일/
bit-exact"를 혼동하지 말 것: 이 문서의 판정은 bit-exact(엄밀 동일)가 아니라 **float32 잡음
수준의 근사 동일**이다.

## 1. Reference ONNX vs best.pt 재현성 (CV, 2026-08-14)

**비교 대상**:
- Source `best.pt`: `.../Corner_Truck/v01/weight/weights/best.pt`, 6,214,435 bytes,
  mtime 2026-07-07 12:21:57, md5 `cd132b519933c0a68c8330356c6cc030`,
  sha256 `56d022502ec69a2a7e779bf62d2fefdb11b991ab0861d8a88eccf23791d946a5`.
- Reference ONNX: `.../Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx`,
  12,070,702 bytes, mtime 2026-07-07 10:27:08(best.pt보다 2시간 5분 앞섬 — mtime은 참고
  정보일 뿐 판정에 쓰지 않음), md5 `9953b747f5ea82177a90f8ed550f47cd`,
  sha256 `03789a9cb20876e75fd248542f7d8d99fe132cd24966308b00ed64f5952b6554`.

**Checkpoint 분석(핵심 분기점)**: `model_class = CustomDetectionModel`(**Front_Truck의
`SymmetricDetectionModel`과 다름**) — repo의 `model_name` registry상 `custom` 계열은
`CustomYOLO`/`CustomDetectionTrainer`가 담당하므로 export도 `tools/export_symmetric.py`가
아니라 **`tools/export_custom.py`**를 써야 함을 코드(`CustomYOLO.task_map`)로 확인. 단
`model.model[-1]`(Detect head) 자체는 Front_Truck과 **같은 클래스**(`SymmetricDetect`,
`skip_postprocess` 속성 보유) — top-level wrapper는 다르지만 `skip_postprocess` 메커니즘은
헤드 모듈에 있어 두 계열 모두 사용 가능(코드로 확인, 우연 아님). `names`: 5class
{bus,car,motorcycle,person,truck} — traffic_light 없음(Corner CN-F308 5-class 사양과 일치).
`train_args.data: configs/datasets/BSD-F606A/Corner.yaml`, `imgsz:320`.

**Reference ONNX 구조(실측)**: input `images[1,3,320,320]`, output `output0[1,5,2100]`
(Sigmoid 직결, cls) + `433[1,4,2100]`(Reshape 직전이 DFL Softmax+Conv, dist2bbox 이전 raw
l,t,r,b) — Front_Truck과 같은 "box/cls 분리, box는 DFL까지만" 구조(이번에도 그래프를 직접
읽어 확인, Front_Truck 결과를 가정하지 않음). opset13/ir_version7/node235/initializer127.
2100 anchor = 320×320 입력·stride 8/16/32 기준 40×40+20×20+10×10=1600+400+100=2100(3-scale
FPN 확인).

**Export 방식**: `tools/export_custom.py -w <best.pt> -i 320 -y 320 --skip_postprocess`
(스크립트 기본값이 이미 `skip_postprocess=True`, `CustomYOLO.export()`는 표준
`Model.export()` 그대로 사용 — Front_Truck의 `SymmetricYOLO.export()` return 누락 버그
같은 문제 없음, 코드로 확인). 출력 shape 실측 `((1,5,2100),(1,4,2100))` — reference와 정확히
일치.

**6단계 판정 결과**:

| Level | 항목 | 결과 |
|---|---|---|
| 1 | Graph structure(235 node, op_type 분포) | **SAME**(node-by-node 불일치 0건) |
| 2 | Initializer(127개) | **SAME**(bit-exact는 13/127뿐이나, 나머지 114개도 max_abs_diff 전체 범위 0.0~4.77e-07 — float32 machine epsilon(~1.19e-07) 바로 위 수준, `max_abs_diff>1e-4`인 항목 0개) |
| 3 | Raw output(10종 고정 입력 + 실제 Corner 검증 이미지 5장) | **SAME**(cls 최대 1.85e-06, box 최대 1.15e-05 — 값 스케일(수십~수백) 대비 무의미한 잡음) |
| 4 | 중간 레이어(intermediate) | **해당 없음** — 의미 있는 divergence 자체가 없어 추가 분석 불필요(§5의 3-way 검증으로 원인까지 직접 증명) |
| 5 | Decoded detection(동일 decode+NMS, conf=0.05/iou=0.7/max_det=300) | **SAME**(5장 전부 검출 개수·클래스 완전 일치, one-side-only 0건, conf_diff 최대 1e-6, bbox diff 최대 3.1e-05px) |
| 6 | Full validation metric | **미실행**(데이터셋 미마운트, 상위 5단계 증거로 충분해 강행하지 않음 — "확인 못 함"이 아니라 "생략") |

**PT ↔ Exported ONNX 3-way 검증(핵심 증거)**: `model.0`/`model.9.cv2`/`model.22.cv3.0.1`
3개 레이어에서 `fuse_conv_and_bn`으로 직접 계산한 PT manual fusion과 대조:

| 레이어 | PT manual fusion vs **Exported** | PT manual fusion vs **Reference** |
|---|---:|---:|
| `model.0` weight/bias | **0.0(bit-exact)** | 3.73e-09 / 2.38e-07 |
| `model.9.cv2` weight/bias | **0.0(bit-exact)** | 2.98e-08 / 4.77e-07 |
| `model.22.cv3.0.1` weight/bias | **0.0(bit-exact)** | 5.96e-08 / 4.77e-07 |

**PT manual fusion == Newly exported ONNX(완전 bit-exact, 3개 레이어 전부)** — 이 repo의
export 파이프라인이 `source_best.pt`의 실제 파라미터를 한 치의 오차 없이 반영함을 직접
증명. PT manual fusion과 Reference ONNX의 차이(1e-7~1e-9)는 Front_Truck에서 관찰된 "다른
모델" 수준(0.8~4.8)과 전혀 다른 규모.

## 2. NVT Calibration/Compile/Simulator 검증 (CN, 2026-08-14)

**입력**: Float ONNX = §1에서 검증된 `exported_from_best.onnx`(재-export 안 하고 재사용).
gen_config는 NAS 원본(`.../Corner_Truck/v01/network/gen_config.txt`)을 복사, path 항목만
수정(그 외 precision/preprocessing 값 전부 원본 그대로): `ref_data/num=300`,
`spec_demarc_tensors` 주석 처리(미사용), `cal_refine_iter=0`, `cle_en=0`, `in_hp
bnscale/eltwise/conv_op=0/0/0`, `percentile` 설정 없음. sim_config는 NAS에 원본이 없어
320×320 입력에 맞춰 **신규 작성**.

**Calibration**: `/mnt/NAS_1/jpg`(원본, 읽기 전용)에서 `rsync -a --include="*.jpg"`로 로컬
격리 디렉토리에 복사 — **300장**(원본에 섞여 있던 `.db` 파일 1개는 제외, 약 128MB),
gen_config의 `ref_data/num=300`과 정확히 일치.

**Compile 결과**(`corner_truck_v01_320_quant_compare_20260814`, input sha256
`bfe92adbac0c4ff952db7a74bb28a6ecf7f229443335fd2e3b36d85b338fb6ac`): Latency **6.24694ms**,
DRAM **35,045,388 bytes(약 33.4MB)**, Model size **4,703,564 bytes(약 4.70MB)**. 신규
`nvt_model.bin` sha256 `80778a2cbb66f9f372e8e6618b3a33b14ce0448912ca68fdc37160556b2b8769`.
**NAS 기존 `nvt_model.bin`과 4 bytes 차이**(NAS 4,703,568 vs 신규 4,703,564), SHA256도 다름
— 차이는 매우 작으나 정확한 원인은 binary-level 추가 분석 없이 확정하지 않음(미확정으로
남김).

**Simulator 결과**(테스트 이미지 `KR_20240621_TRUCK1T_CORNER_140CM_RIGHT_205FISH_CROP_
NC22015_0992.jpg`):

| Tensor | Cosine |
|---|---:|
| Softmax(dfl) | 0.987600 |
| Sigmoid[OUT] | 0.996472 |
| dfl_conv[OUT] | 0.994926 |

**전체 171개 tensor 통계**: mean cosine 0.9837, min cosine 0.9604. cosine<0.999인 텐서
163개, <0.995인 텐서 147개, <0.990인 텐서 132개. **Lowest cosine Top 10**(전부 neck/head
영역): `Mul__model.19_act_Mul` 0.9604, `Mul__model.21_m.0_cv2_act_Mul` 0.9621,
`Mul__model.18_m.0_cv2_act_Mul` 0.9624, `Conv__model.21_m.0_cv2_conv_Conv` 0.9640,
`Conv__model.19_conv_Conv` 0.9648, `Mul__model.22_cv3.1_cv3.1.0_act_Mul` 0.9651,
`Conv__model.22_cv3.1_cv3.1.1_conv_Conv` 0.9690, `Split__model.8_Split` 0.9691,
`Mul__model.18_m.0_cv1_act_Mul` 0.9703, `Conv__model.21_m.0_cv1_conv_Conv` 0.9711.

**해석**: 주요 detection output 관련 텐서는 0.988~0.996 cosine — quantization으로 인한
numerical difference는 존재하나(전체 평균 0.984, 일부 neck/head는 0.960까지 하락), **cosine
값만으로 실제 Board mAP 손실을 판단할 수 없다**(Rear_Truck 조사에서 반복 확인된 원칙과
동일) — Board 성능은 실제 Board validation으로 확인해야 한다.

## 3. 기존 Corner Board 결과 (과거 baseline, 이번 신규 BIN 아님)

이번에 새로 생성한 `nvt_model.bin`을 보드에서 측정한 결과가 **아니라**, 과거 기존 Corner
Board 결과다 — 혼동 금지.

| Class | ONNX(Float) | NVT BIN(Board) | Delta(NVT-ONNX) |
|---|---:|---:|---:|
| bus | 0.284 | 0.416 | +0.132 |
| car | 0.393 | 0.414 | +0.021 |
| motorcycle | 0.549 | 0.549 | 0.000 |
| person | 0.571 | 0.523 | -0.048 |
| truck | 0.504 | 0.504 | 0.000 |
| **all** | **0.460** | **0.481** | **+0.021** |

Corner는 전체(all) mAP 기준으로는 quantization 손실이 없다(오히려 +0.021, board가 float보다
높음) — 단 클래스별 편차는 존재(bus +0.132, person -0.048). **이 표의 지표 정의(mAP50 인지
mAP50-95인지, NMS/evaluator 종류)는 원문에 명시돼 있지 않다 — 확인되지 않은 채로 남긴다.**

## 4. 현재 상태 — 신규 NVT BIN Board 검증 (미완료)

이번에 새로 생성한 `nvt_model.bin`(§2)에 대해서는 **아직 Board 측정을 수행하지 않았다.**
다음 단계로 제안된 절차: 실제 NT98529/NT98530 Board에 배포 → 기존 Float ONNX와 동일한
validation dataset/전처리/crop·resize/BGR·RGB/confidence threshold/NMS IoU/class mapping/
postprocess/evaluator 조건으로 Board inference 생성 → `Float ONNX mAP vs 새 NVT BIN Board
mAP`를 class별/all 기준 비교(가능하면 과거 Corner ONNX/NVT Board 성능을 만들 때 쓴 것과
동일한 evaluation pipeline 사용). **Full validation(§1 Level 6)도 미실행** — 실제 Corner
검증 데이터셋(`/mnt/NAS_3/02_datasets/CVD/Corner/Validation`, 5,000장)이 이 컨테이너에
마운트돼 있지 않았다.

## 최종 상태 요약 (원문 그대로)

```
Corner_Truck v01
Float ONNX: 정상 | Calibration: 300 images | NVT Compile: PASS
Latency: 6.24694 ms | DRAM: 33.4 MB | Model size: 4.70 MB
Simulator: PASS | Softmax(dfl): 0.987600 | Sigmoid: 0.996472 | dfl_conv: 0.994926
New nvt_model.bin: 생성 완료 | New NVT BIN Board validation: NOT RUN
기존 Board baseline: ONNX 0.460 | NVT BIN 0.481 | Delta +0.021
```
