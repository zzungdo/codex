# Front_Truck v01 — Reference ONNX 재현성과 릴리즈 산출물 조사

> 원문: `20260814_front_truck_reference_onnx_vs_best_pt.md`(VB), `20260814_front_truck_reference_onnx_source_checkpoint_search.md`(SC),
> `20260814_front_truck_reference_onnx_nvt_board_prepare.md`(BP), `20260814_front_truck_reference_onnx_local_nvt_reconversion.md`(LR),
> `20260819_front_golden_reference_for_rear_1pct.md`(FG, Front 자체 데이터 부분), `20260810_npu_pattern_name_board.md`(PN, Front v01_repro 보드 실측 부분).
> Rear로 전이된 가설/실험은 [01_rear_quantization_and_release.md](01_rear_quantization_and_release.md) §9 참조.
> Front 학습 실험 계보(standard/v01_repro3/v02_*)는 [03_front_training_experiments.md](03_front_training_experiments.md) 참조 — 이 문서는 **NAS 배포 산출물(Release ONNX/BIN) 재현성** 조사만 다룬다.
> 공통 재현 환경·evaluator 정의는 [90_reproduction_reference.md](90_reproduction_reference.md) 참조.

## 0. 핵심 결론 요약

- **NAS reference ONNX(`260706_BSD-F606A_Front_v01.onnx`)와, 현재 repo의 `best.pt`
  (`runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt`)를 이 repo의 export 코드로
  변환한 ONNX는 그래프 구조는 완전히 같지만 파라미터는 다르다.** 즉 **같은 그래프, 다른
  가중치** — 같은 모델이 아니다(§1).
- **source `.pt` checkpoint는 조사한 범위(Front_Truck/v01, 확장해도 Front_Truck 전체) 내에서
  찾지 못했다(NOT FOUND) — 아직 검색하지 않은 범위(Front_Truck 외 다른 제품 라인, 총 62개
  `.pt`)가 있으며, 그 범위에 대해서는 "없다"고 결론 내리지 않는다**(§2).
- **Release calibration 500장 원본 세트는 host/NAS/repo 어디에도 없다(NOT FOUND)** — 로컬
  재변환은 대체 calibration(Front 도메인 Validation 앞 500장)으로 진행됐고, converting
  코드 자체의 재현성과 calibration 데이터 차이 효과가 완전히 분리되지 않는다(§3).
- Release NVT bin은 NAS에서 **provenance 확인 후 그대로 사용(FOUND, 신규 compile 없음)**한
  경로와, 같은 ONNX를 **로컬에서 새로 재변환**한 경로 두 가지가 별도로 존재한다 — 혼동 금지
  (§3, §4).
- Front golden reference 조사(FG)의 근사 재현치에 따르면 Front Release는 **mAP50-95
  기준으로도 Float→NPU 손실이 없다(오히려 +10.31% 개선)** — 단 이 수치는 dual-NMS
  파라미터(Rear에서 가져온 값)·GT 클리핑 임계값 등 가정에 의존한 **근사치**다(§5).
- pattern-name 이슈 수정 후 Front_Truck v01 재현 컴파일의 실제 보드 실측(별도 재현
  스트림, `v01_repro`)은 **mAP50-95 기준 -29.5% 손실**을 보였다(§6) — 이는 §5의 Release
  bin 자체 실측(+10.31%)과 **다른 재현 스트림**이므로 직접 비교하지 말 것(조건 차이는 §6
  참조).

## 1. Reference ONNX vs best.pt export 비교 (VB, 2026-08-14)

**비교 대상**:
- Reference ONNX(A, 읽기 전용): `.../Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx`,
  12,071,501 bytes, mtime 2026-07-07 10:27:08, md5 `d623cbdc6ad13519646b6bda53e879a0`,
  sha256 `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0`.
- Source `best.pt`(B): `runs/detect/BSD_F606/Front_Truck/standard/weights/best.pt`,
  6,233,891 bytes, mtime 2026-08-13 15:49:55, md5 `412673ce9a17f52ee40fee87e680dbb4`,
  sha256 `f55b8b915149b496e439836b4d53637251725198189f9881ad33e34b418c3cbb`.
- Export 방법: 먼저 reference ONNX 출력이 `output0[1,6,6720]`(cls, Sigmoid 직결)+
  `433[1,4,6720]`(box, DFL 적분까지·dist2bbox 디코드 이전)로 분리된 구조임을 실측 확인한 뒤
  이 구조를 재현하는 `tools/export_symmetric.py --skip_postprocess`로 export
  (input 640×512, opset13, simplify=False, batch=1).

**5단계 판정 결과**:

| Level | 항목 | 결과 |
|---|---|---|
| 1 | Graph structure(node 235개, op_type 분포, attributes) | **SAME**(node-by-node 불일치 0건) |
| 2 | Initializer(127개) | **DIFFERENT**(127개 중 126개, 첫 Conv `model.0.conv.{weight,bias}`부터 다름) |
| 3 | Raw output(고정 입력 10종: zeros/ones/half/random/impulse/실제이미지5장) | **DIFFERENT**(입력 무관하게 box(`433`) 출력 차 항상 6~10 범위 — 실행경로 차이가 아니라 파라미터 차이) |
| 4 | Decoded detection(동일 decode+NMS, conf=0.05/iou=0.7/max_det=300) | **DIFFERENT**(matched conf_diff mean 0.055~0.184/max 0.17~0.69, bbox 차 mean 1.8~4.1px/max 15px, only-in-one 다수) |
| 5 | Full validation metric(6,076장, pycocotools, ROI crop+640×512) | **DIFFERENT하나 방향 일관** — exported가 6클래스 전부에서 reference보다 높음(mAP50 +0.018~+0.059, mAP50-95 +0.025~+0.095, all: mAP50 0.742→0.773/+0.031, mAP50-95 0.480→0.551/+0.071) |

**최초 divergence node**: 그래프의 **첫 Conv(node 0, `/model.0/conv/Conv`) 자체**
(output max_abs_diff=11.586, mean_abs_diff=0.444) — 뒤로 갈수록 누적되는 차이가 아니라
네트워크 진입점부터 이미 다른 파라미터로 계산된다.

**결론**: 그래프 구조(노드 종류·개수·순서·연결)는 완전히 같지만, 저장된 파라미터는 첫
레이어부터 이미 다른 **서로 다른 모델**이다. mAP가 방향 일관되게(그리고 어느 정도 근접하게)
나온다고 해서 이 판정이 바뀌지 않는다 — mAP는 수천 detection에 걸친 통계치라 파라미터가
달라도 우연히 비슷한 값에 수렴할 수 있다.

## 2. Source checkpoint 탐색 (SC, 2026-08-14)

판정 기준은 파일명·mtime·mAP 유사도가 아니라 **ONNX initializer의 parameter 값 자체**
(model.0, model.9.cv2, model.22.cv3.0.1 등의 fused conv weight/bias fingerprint).

**Search Scope A(`Front_Truck/v01`)** — 발견된 `.pt` 3개, 전부 unique SHA256:

| candidate | path | model_class | model.0 weight max_abs_diff | model.0 bias max_abs_diff | 판정 |
|---|---|---|---:|---:|---|
| C001 | `weight/pretrained_weight.pt`(ema state만 존재) | SymmetricDetectionModel | 0.0514 | 0.9193 | CLEAR_MISMATCH |
| C002 | `weight/weights/last.pt` | SymmetricDetectionModel | 0.0278 | 0.8261 | CLEAR_MISMATCH |
| C003 | `weight/weights/best.pt`(= 현재 repo `best.pt`, sha256 `f55b8b91...`) | SymmetricDetectionModel | 0.0297 | 0.9420 | CLEAR_MISMATCH |

**Search Scope B(`Front_Truck` 전체)**: `Front_Truck`에는 `v01` 디렉토리 하나뿐이라 Scope A와
동일한 3개 candidate로 귀결 — 새 후보 없음.

**탐색을 멈춘 이유**: 조사 도중 **사용자가 "Front_Truck이 아니면 아마 관계 없을 것"이라고
확인**해줘서, `BSD_F606` 안의 다른 제품 라인(`Front` v01–v08, `DMS`, `Rear` v01–v04, `Side`
v01–v04, `Corner_Truck`, `Rear_Truck`, `Side_Truck` — 총 62개 `.pt`, 파일 목록·SHA256만
`scopeC_sha256_list.txt`에 기록)으로는 확장하지 않았다. 사양표 근거(6클래스=traffic_light
포함 모델은 Front_Truck 계열뿐)는 있으나 **이번 조사에서 실측 검증한 사실이 아니라 사용자
확인 + 기존 문서 사양표를 따른 것** — "확인됨"이 아니라 "탐색하지 않음"으로 기록.

**부수 발견**: `Front_Truck/v01/weight/pretrained_weight.pt`(C001)의 SHA256(`83e51540...`)이
`BSD_F606/Front/v08/weight/weights/best.pt`와 **byte-identical** — Front_Truck 학습이
`Front/v08`의 best weight를 초기값으로 썼다는 뜻(C001의 `train_args.model` 계보와도 일치).
단 C001 자체가 이미 CLEAR_MISMATCH라 이 계보(`Front/v01~v08`)는 더 따라가지 않았다.

**최종 결론**: `Source checkpoint: NOT FOUND` — 조사한 범위(`Front_Truck/v01` = `Front_Truck`
전체, 3개 `.pt`, `model`/`ema` state 포함) **안에서** 어느 후보도 reference ONNX의 첫 Conv
레이어조차 재현하지 못한다. **탐색하지 않은 범위(BSD_F606 내 다른 62개 `.pt`, `05_Yolov8_nano`
상위 다른 SoC/제품군)는 확인되지 않은 것이지 "없다"는 뜻이 아니다** — 삭제/축소하지 말 것.

## 3. Release 산출물(ONNX/BIN) 로컬 재변환 (LR/BP, 2026-08-14)

**BP — 기존 NVT artifact 탐색(신규 compile 없이 기존 것을 그대로 사용하는 경로)**:
`Front_Truck/v01/network/`에 `.onnx`와 `.bin`이 함께 존재, 동일 basename
(`260706_BSD-F606A_Front_v01.*`), 생성 시각 2초 차이(onnx 10:27:08, bin 10:27:06), gen_config의
input size(640×512)가 ONNX와 정확히 일치 — **FOUND로 판단, 신규 compile 진행하지 않음.**
`nvt_model.bin`: 4,758,800 bytes, sha256 `fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b`,
md5 `619b83d9bb519be69f1b9f3b37ee0a4c`. Board 실행 자체는 BP에서 수행하지 않음.

**LR — 별도로 로컬에서 새로 재변환한 경로(현재 converting pipeline 자체를 검증하기 위함)**:
같은 source ONNX(`.../v01/network/260706_..._v01.onnx`)를 `npu-convert` 컨테이너
(`run_compiler.sh --chip 530`)로 재컴파일. Release gen_config(input 640×512, chip530,
`ref_data/num=500`, `calib_mode=4`, `cle_en=0`, `cal_refine_iter=0`, `spec_demarc_tensors`
전체 주석 처리=미사용, `in_hp bnscale/eltwise/conv_op=0/0/0`, `sim_config src/bintype=1`)를
path 항목만 바꿔 그대로 사용(diff로 path 외 무변경 확인).

**Calibration 500장 — NOT FOUND**: gen_config가 가리키는 원래 경로
(`/home/ncn/Developments/data/weights/_pipeline_inputs/nt98530_d224k_front/calib/jpg`)는 원본
컴파일 머신의 로컬 경로로 현재 host/NAS/repo 어디에도 없다(Front_Truck NAS 폴더, NAS_1/NAS_3,
로컬 repo, 컨테이너 `/dataset` 마운트 전부 확인). **대체 calibration**으로 `/dataset/Dashcam/
Front_traffic_light/Validation/images`(Front_Truck과 클래스 목록 정확히 일치하는 Front 도메인
데이터셋)에서 파일명 정렬 후 앞 500장을 사용 — `ref_data/num=500`과 개수는 맞췄으나 **Release와
동일 세트가 아니므로, 신규 BIN과 Release BIN의 차이가 converting code 때문인지 calibration
데이터 차이 때문인지 완전히 분리되지 않는다**(이 조사의 명시된 제약).

**결과**:

| 항목 | NAS Release BIN | Local Reconverted BIN |
|---|---|---|
| Size | 4,758,800 bytes | 4,758,792 bytes(8 bytes 차이) |
| SHA256 | `fa795cb4...ca00b` | `d5727e65...26028`(다름) |
| Latency/DRAM | (프로파일 로그 미보존, UNKNOWN) | 19.628ms / 103,244,064 bytes |
| `cmp` byte diff | — | 3,136 bytes(공통 4,758,792 bytes 구간 기준), 첫 diff는 byte 26 |

SHA256이 다르다고 conversion 실패로 판단하지 않는다(calibration 데이터 자체가 다르고, NVT
binary는 compiler metadata/calibration 통계를 포함할 수 있음). Simulator sanity check(신규
Local BIN만, 기존 Release bin은 compile 중간 산출물 부재로 재실행 불가): Softmax(dfl)
cosine 0.981978, Sigmoid cosine 0.976482, dfl_conv cosine 0.992463, 171개 tensor diff 로그
생성·NaN/Inf 없음 — **171개 텐서 raw output 전체를 골든과 대조하는 전면 검증은 아니고,
board 투입 전 sanity check 수준**이다.

**Board 측정 및 판정 기준(제안, 이 문서 자체에서는 board 미실행)**: Case A(Release
mAP50≈0.704 vs Local reconvert 0.70~0.71 근처, 클래스별도 유사) → converting pipeline이
Release를 재현 가능, 코드 자체 문제 가능성 낮음. Case B(예: 0.63 등 크게 다름) → calibration
데이터 차이를 먼저 분리한 뒤 gen_config/compiler version/wrapper/precision/preprocessing
순으로 조사.

## 4. Front Release Spec — 재현 조건 정리 (FG §4~§7)

- Version v01 / 2026-07-06 / YOLOv8-Nano / BGR / Object Detection, Target NPU NT98530(NT98529
  표기는 과거 문서 오기로 취급, 별도 조사 안 함).
- Classes(6): bus, car, motorcycle, person, truck, **traffic_light**.
- 전처리(Release 문서 기재 → 실측 확인): Crop ROI x 320~1600, y 0~1024(1280×1024) → resize
  640×512 → ONNX input `[1,3,512,640]`.
- Reference ONNX: `.../v01/network/260706_BSD-F606A_Front_v01.onnx`, 12,071,501 bytes,
  mtime 2026-07-07 10:27:08, sha256 `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0`
  (§1과 동일 파일, 2026-08-19 세션에서 재계산해도 일치 확인). opset13/ir_version7/pytorch2.5.1,
  outputs `output0[1,6,6720]`(Sigmoid 직결) + `433[1,4,6720]`(DFL conv+reshape, dist2bbox
  이전). op_type 분포: Conv64/Sigmoid57/Mul56/Constant14/Split8/Add6/Concat15/MaxPool3/
  Resize2/Reshape8/Transpose1/Softmax1(DFL).
- Dataset: `/mnt/HDD_1/BSD_F606_FRONT/dataset` — Train 45,254장, **Validation 6,076장**(cxywhn
  라벨 + 픽셀 GT, ROI-crop 좌표계로 보임). Day/Night(29,299/22,031)이 어느 스냅샷 기준인지는
  **UNKNOWN**(Train+Val 합계 51,330과 불일치).
- Evaluation pipeline: ROI crop `[320.0,0.0,1280.0,1024.0]`이 Release 문서값과
  `analysis/metrics_front_*.txt/json`의 `roi` 필드와 정확히 일치(실측 확인). resize 1280×1024
  → 640×512. color: YUV420 입력(`preproc/in/fmt=2`) → BGR 출력. normalize/meansub 둘 다
  비활성. dtype UINT8. **정확한 실행 스크립트/config는 NAS 트리에서 찾지 못함** — 정확히
  몇 장(6,076 전체 vs subset)을 평가에 썼는지도 **미확정**(추정: 6,076 전체).

## 5. Float/Board Performance — Release 문서값 vs 근사 재현치 (FG §8~§10, §17~§18)

### mAP50 (Release 문서 기재, NAS `analysis/metrics_front_*.json` 재계산과 일치)

| Class | Float mAP50 | Float P | Float R | Board mAP50 |
|---|---:|---:|---:|---:|
| bus | 0.706 | 0.83 | 0.672 | 0.696 |
| car | 0.805 | 0.845 | 0.781 | 0.780 |
| motorcycle | 0.668 | 0.782 | 0.639 | 0.725 |
| person | 0.691 | 0.805 | 0.633 | 0.742 |
| truck | 0.692 | 0.806 | 0.679 | 0.687 |
| traffic_light | 0.617 | 0.81 | 0.588 | 0.592 |
| **all** | **0.6965**(문서, 실측 재계산 0.697) | 0.813 | 0.665 | **0.704**(문서 및 `metrics_front_board.json` 재확인 일치) |

Precision/Recall 열은 NAS analysis 트리가 아니라 **사용자가 보유한 원본 Release 문서**에서
온 값(2026-08-20 추가) — mAP50 열은 이 세션 확인값과 일치하나 P/R은 NAS analysis에서는
확인되지 않았던 추가 정보다.

### mAP50-95 — Release 문서/NAS metric 파일에 기록 자체가 없음(모든 `analysis/` metric 파일이
`AP` 단일 필드, `iou_tp:0.5`만 가짐) → **실험 0(2026-08-20)** 근사 재현:

- Float ONNX(onnxruntime 직접 추론, ROI crop→640×512→BGR/무정규화, dual-threshold NMS로
  board와 정렬 — **NMS 파라미터(min_overlap=0.6/union_overlap=0.3)는 REAR 문서에서 가져온
  값으로 FRONT 자체 확인은 아님, 근사치로 취급**): **mAP50-95 = 0.4615**. sanity check:
  재현 mAP50 0.7177 vs 문서값 0.697(+0.021, 근접).
- Board(`analysis/iputest/output.json`, 151,136 detection, 6,075/6,076장 커버, GT를 ROI
  경계로 클리핑 — 클리핑 없이 1차 시도 시 Board mAP50=0.583으로 문서값과 크게 벗어남, ROI
  밖 GT를 거짓음성으로 잘못 셈했기 때문임을 확인해 클리핑 근거 확보): **mAP50-95 = 0.5091**.
  sanity check: 재현 mAP50 0.7293 vs 문서값 0.704(+0.025, 근접).
- **Quantization loss(mAP50-95, 실험0 재현치)**: absolute = 0.5091 − 0.4615 = **+0.0476**,
  relative = **+10.31%**(Board가 Float보다 높음 — 손실이 아니라 개선). mAP50 기준(+1.0%,
  문서 0.704/0.697)과 같은 방향, 더 큰 폭으로 재확인 — "mAP50이 mAP50-95 손실을 가리고
  있었다"는 우려는 기각(반대로 강화).
- **이 수치의 신뢰 수준**: dual-NMS 파라미터·GT 클리핑 임계값 등 가정에 의존한 **최선의
  근사치**다. mAP50 sanity check(오차 +0.02대)로 방법론 정합성은 확인했으나, "정확한
  수치(+10.31%)"보다 "손실이 없거나 오히려 개선"이라는 **정성적 결론**에 더 높은 확신을 둔다.

## 6. Calibration/Config/Compiler — Release 재현 한계 (FG §11~§18)

- **Calibration dataset(500장)**: gen_config 경로가 원본 컴파일 머신 로컬 경로라 **NOT
  FOUND**(§3과 동일 결론, 별도 재검증 안 함). 실제 파일 내용/순서는 UNKNOWN.
- **Calibration 전용 전처리**: gen_config `[PREPROCESS]`와 동일할 것으로 추정(§4와 크롭/리사이즈/
  색공간/정규화 설정 공유 구조)이나, 원본 세트 부재로 별도 확인 불가.
- **NVT Production Config**(gen_config.txt 원문 확인): compiler 정확한 버전 문자열 없음
  (UNKNOWN), chip 530, calib_mode **4**(PERCENTILE류로 추정, Rear 문서의 `calib_mode 4 =
  PERCENTILE(0.999)`와 동일 코드값이나 percentile 수치 자체는 gen_config에 명시 없어
  UNKNOWN), cle_en **0**, cal_refine_iter **0**, bnscale_en **0**, 16bit 보호 텐서 0개.
- **spec_demarc_tensors**: gen_config.txt 원문에서 **전체 주석 처리 — 실제로는 미사용**(0개
  보호). Rear의 v01_symmetric이 쓰는 3-tensor(H1-H3, 헤드 근처) 보호보다도 적다.
- **Compiler version**: Release 당시 버전 UNKNOWN. 로컬 재변환 시 생성된
  `MODEL_BIN_INFO/VERSION=2052503042`는 모델 바이너리 포맷 버전이지 컴파일러 툴체인 버전이
  아님 — Release 당시와 현재 로컬 컴파일러가 동일한지 확인 불가.
- **Simulator Layer Analysis**: Release NVT bin 자체의 simulator dump는 NAS에 없음(compile
  중간 산출물 미보존). 참고 가능한 값은 §3(LR)의 **로컬 재변환(대체 calibration) 결과** —
  Softmax(dfl) 0.981978/Sigmoid 0.976482/dfl_conv 0.992463, model.2 블록 0.9903~0.9956,
  model.4 블록 0.9848~0.9959(`model.4/m.0/cv1/act` 0.9848, `model.4/m.1/cv1/act` 0.9855가
  국소 최저) — **이 값은 Release bin이 아닌 근사치**임에 유의. Rear가 겪는 backbone-early
  cosine 저하(0.96대)까지 떨어지는 지점은 Front(근사치)에서 발견되지 않았다.
- **Reference NVT Reproduction**: Reference NVT model sha256
  `fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b`, size 4,758,800 bytes.
  latency/DRAM은 Release bin 자체 프로파일 로그가 없어 UNKNOWN(참고: 로컬 재변환본
  latency 19.628ms, DRAM 103,244,064 bytes, size 4,758,792 bytes — Release bin과 크기
  8바이트 차이뿐이나 이 값을 Release 실측치로 대체할 수 없음). **결론: 원본 500장
  calibration set 부재로 Release NVT bin을 수치적으로 동일하게 재현하는 것은 현재
  불가능** — 구조/설정값 확인은 거의 완전하지만 수치 재현에는 근본적 한계.

**First Divergence 표(FG §18)**: Reference ONNX(PASS) → Evaluation pipeline(PARTIAL, 정확한
실행 스크립트 미확인) → **Calibration dataset(FAIL, NOT FOUND)** → Calibration
preprocessing(UNKNOWN) → NVT config(PASS) → Compiler version(UNKNOWN/DIFFERENT) →
**Reference NVT reproduction(FAIL, sha256 다름)** → Simulator raw output(PARTIAL, 근사치만) →
IPU-test(PASS, output.json 6,075/6,076장 커버, 1장만 board 미검출) → Board/Float
mAP50-95(**PASS, 근사 재현**, §5 참조). **FIRST DIVERGENCE: Calibration dataset** — 이 지점부터
수치적으로 완전한 재현 경로가 막히나, mAP 수치·config 확인에는 영향이 없었고 실험0으로 핵심
미확정 사항(Front의 mAP50-95 기준 quantization loss)은 해소됐다.

## 7. 별도 재현 스트림 — pattern-name 수정 후 v01_repro 보드 실측 (PN, 2026-08-10)

이 절은 §1~§6의 "NAS Release 산출물 재현" 스트림과 **다른, 독립적인 재현 스트림**이다:
`configs/releases/BSD_F606/Front_Truck/v01`을 이 repo에서 **재학습(`v01_repro`)**한 뒤
NPU로 재컴파일한 것으로, source `.pt`/calibration 세트도 §1~§6과 다르다. 상세 조사 과정
(pattern-name 대소문자/길이 문제, 시뮬레이터로는 검출 불가능한 보드 인식 실패)은
[90_reproduction_reference.md](90_reproduction_reference.md) §pattern-name 참조 — 이 절은
그 문서에 포함된 **Front 보드 실측 quantization loss** 데이터만 인용한다.

pattern-name을 고치고 보드에서 정상 인식된 뒤(`bsd_f606_front_truck_v01`), val set
**6,076장**(board 결과 기준 "6,075장"으로도 표기됨 — 원문 그대로 병기) 전체를 GT와 직접
비교. 전처리: board bbox는 crop 로컬 좌표(1280×1024)라 GT(원본 1920×1080)와 맞추려면
x에 +320(ROI x-offset). golden 비교 기준은 **640×512**(배포 해상도, 학습 해상도 640×480이
아님 — train-deploy-size-mismatch 축과 동일 이슈)로 `best.onnx`를 새로 추론해 생성. GT
필터링: ROI(320,0,1600,1024) 안에 중심이 있는 GT만 사용(전체 61,587개 → 필터 후 50,371개).

**결과(val 6,075장, 640×512 동일 조건)**:

| | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| golden(float, `best.onnx`) | 0.806 | 0.697 | 0.782 | 0.567 |
| 보드(NPU 양자화) | 0.737 | 0.600 | 0.655 | 0.400 |
| 손실 | -0.069 | -0.097 | **-0.127 (-16.2%)** | **-0.167 (-29.5%)** |

클래스별로는 person·traffic_light 손실이 가장 큼(mAP50 -0.19/-0.18) — 작은/미세한 형태의
객체가 양자화에 더 취약한 전형적 패턴. car/truck/bus(큰 물체)는 손실이 상대적으로 적음
(-0.08 안팎). **미해결**: 이 실측은 `gen_config.txt`가 원본 v01 그대로(`calib_mode=4`,
`in_hp` 전부 0)인 상태에서 나온 수치 — 가이드 권장 기본값(`calib_mode=1` + `in_hp` 전부 1 +
`spec_demarc_tensors` 지정, 이번 컴파일에는 미적용)을 적용하면 손실이 줄어들지 **아직 확인
안 됨**.

**§5 Release bin 실측(+10.31% 개선)과 이 절(v01_repro 보드 실측, -29.5% 손실)을 혼동하지
말 것** — 서로 다른 학습 checkpoint, 다른 calibration/config(§5는 원본 Release 500장 세트를
찾지 못해 미실측/근사치, 이 절은 v01_repro의 실제 재컴파일본을 실제 보드에 올려 직접 실측),
다른 gen_config 상태(§5는 Release 원본 그대로 cle_en=0/spec_demarc 0개, 이 절도 원본 v01
그대로이나 재학습된 다른 weight 기준)에서 나온 별개의 수치다.
