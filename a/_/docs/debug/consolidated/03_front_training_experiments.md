# Front_Truck v01 계열 학습 실험 비교 (living document)

> 원문: `20260909_front_truck_v01_experiment_comparison.md`(전체, 2026-09-09 작성 · 2026-09-14 갱신 포함).
> NAS Release ONNX/BIN 자체의 재현성 조사는 [02_front_reference_and_reproduction.md](02_front_reference_and_reproduction.md) 참조 — 이 문서는 **이 repo 안에서의 학습 실험 계보와 지표**만 다룬다.
> 공통 재현 환경/evaluator 정의는 [90_reproduction_reference.md](90_reproduction_reference.md) 참조.

## 0. 문서 성격

원문 자체가 "living document"로 설계됐다 — 새 실험이 추가되면 정의/계보/지표표에 행을
덧붙이는 방식. 이 문서도 같은 구조를 유지한다. **이 세션(들)에서 재학습·재검증은 수행하지
않았다** — 전부 기존 산출물(`runs/`, `configs/releases`, `configs/experiments`,
`MANIFEST.yaml`) 재정리·대조다.

## 1. 실험 정의

| 실험 | 시작 가중치 | Trainer | 해상도 | epochs(patience) | data |
|---|---|---|---|---|---|
| `standard` | pretrained(NAS 번들 원본) | symmetric | 640×480 | 50 요청, **실제 완료 epoch 불명**(`results.csv` DRM 암호화) | `BSD-F606A/Front.yaml`(legacy 표기, 물리적으로 `dashcam_front_traffic_light`와 동일) |
| `v01_repro3` | pretrained(=standard와 동일 가중치, 사용자 확인) | symmetric | 640×480 | 50 완료 | `dashcam_front_traffic_light.yaml`(canonical) |
| `v02_symmetric` | pretrained(release 경로 표기, 위와 동일 NAS 정본) | symmetric | **640×512**(`imgsz_y=512`, A1) | 50 완료 | 동일 |
| `v02_dair_lr01`(2026-09-14 추가) | **from-scratch**(아키텍처 yaml, 랜덤 초기화) | **dair**(dual-forward, `dair_params`: mode=or,λ=0.1,γ=0.5) | 640×512 | **300 완주**(patience=200) | 동일 |
| `v02_symmetric_bg`(2026-09-14 추가) | pretrained(v02_symmetric과 동일) | symmetric | 640×512 | **422에서 조기종료**(예산 450, patience=200) | `dashcam_front_traffic_light_bg.yaml`(train에 배경 이미지 5,769장 추가, val 동일) |
| `v02_symmetric_scratch`(2026-09-14 추가) | **from-scratch** | symmetric | 640×512 | **500 완주**(patience=15) | `dashcam_front_traffic_light.yaml`(원본, `_bg` 아님) |

### `standard` — baseline 상세

- `configs/releases/BSD_F606/Front_Truck/v01`의 **as-run 원본 번들**(`weight/`)을 담고 있으며,
  `MANIFEST.run.origin_checkout`(다른 checkout, `/home/ncn/Developments/adassw/sources/
  ultralytics_projects`)에서 **학습된** 원본이다.
- **이 repo 안 사본의 출처(2026-09-09 정정)**: 폴더 내 파일들의 mtime이 전부
  **2026-08-13 15:49 KST 단일 시각**(NAS 원본은 2026-07-07~08)으로 통일돼 있어, 평범한 `cp`로
  이 시점에 새로 복사됐다는 뜻이다. `standard/pretrained_weight.pt`의 sha256
  (`83e51540d24a9c0f32deabc14304462287752974554a24697f845fe3735b363f`)이 **현재 NAS 원본과
  정확히 일치**함을 직접 대조로 확인(2026-09-09) — "학습이 실행된 위치"(다른 checkout)와
  "이 repo 사본의 출처"(`configs/releases/.../v01` = NAS 번들)는 별개 질문이다.
- ⚠ 이 폴더의 `mycfg.yaml`은 **신뢰 불가**(GA 튜닝 값 `lr0=0.0047` 등을 담고 있으나, 같은
  폴더 `args.yaml`의 실제 `lr0`은 `0.0001`) — 실제 학습 레시피는 항상 `args.yaml` 기준으로
  읽어야 한다.
- `weights/`의 `best.onnx`/`best_split.onnx`/`deployment_260706.onnx`/
  `260706_BSD-F606A_Front_v01.onnx`는 8/13~8/31 날짜의 **이후 디버깅 세션**(§2 reference ONNX
  조사)에서 추가된 파일 — 원본 번들 구성물이 아니다.
- **`pretrained_weight.pt`(파인튜닝 시작점)의 실제 정체(2026-09-09, `torch.load` 메타데이터
  직접 확인)**: 단순 초기화가 아니라 **이전 학습의 epoch 17 체크포인트**(`best_fitness=0.5287`,
  mAP50=0.6995/mAP50-95=0.5097)였다. 그 이전 학습 근거: `save_dir: /results/FrontDAIRv8/
  dashboard_aug_add_data/train3` — **DAIR 트레이너 계열**(`val/box_org`/`val/box_aug`/
  `val/box_reg` 필드가 DAIR dual-forward consistency loss 분해와 일치). 그 체크포인트조차
  `model: /results/FrontDAIRv8/train3/weights/best.pt`라는, **한 세대 더 이전의 다른
  체크포인트**에서 파인튜닝됐다(경로만 남고 파일 없어 더 추적 불가). `data:
  projects/data_yaml/TLDS_rework_add.yaml` — 지금 쓰는 데이터셋과 **다른 데이터셋**("TLDS"
  계열, yaml 자체는 이 repo에 없어 클래스 구성 확인 불가). `batch=64`, `device='0,1'`,
  `amp=True`(현재 exp.yaml의 `amp:false`, D-006과 다름 — D-006 이전 시점으로 보여 모순 아님).
  이 시점 자체 성능(mAP50 0.6995/mAP50-95 0.5097)은 `standard`의 최종 성능(mAP50 0.774/
  mAP50-95 0.554)보다 훨씬 낮음 — v01/`standard` 학습이 **+0.075 mAP50의 실질적 개선**을
  만들었다(단 데이터셋 자체가 바뀌었을 가능성이 있어, 이 개선폭이 순수 fine-tune 효과인지
  task/data 전환 효과인지는 분리되지 않는다). 이 계보는 §3-4의 "train loss가 초반 40 epoch
  내내 거의 안 움직인다"는 관찰과 정합적이다(이미 어느 정도 수렴된 모델을 낮은 lr0=0.0001로
  살짝 더 맞추는 상황이었을 가능성).

### `v01_repro3` — canonical release recipe 재현

`configs/releases/BSD_F606/Front_Truck/v01/exp.yaml`(canonical)을 재현. `weight_path`는
"번들 실물이 이 clone에 없어 다른 clone(`ultralytics_projects-main`)의 원본을 복사해옴"이라는
mycfg 주석이 있으나, **사용자 확인(2026-09-09): `standard`의 원본 `pretrained_weight.pt`와
동일 가중치** — 두 실험의 시작점은 같다. `workers:4`(CLAUDE.md 호스트 제약, `standard`의
8과 다름 — 의도된 정본화 차이). 해상도 640×480(`standard`와 동일 조건). epochs 50 완료
(`results.csv` 정상, 암호화 없음).

### `v02_symmetric` — 해상도 수정(A1)

`based_on: configs/releases/.../v01/exp.yaml` + `data_args.imgsz_y:512`(A1 수정,
`20260729_config_canonicalization.md`) — 학습 해상도를 배포 해상도(640×512)와 일치. `weight_path`가
release 경로를 직접 가리킴(로컬엔 release 산출물이 상주하지 않는 정책 — 실행 시점에만 NAS에서
임시로 가져옴). NAS 원본 sha256 `83e51540...`이 정본. **한계**: 학습 당시 실제 로딩된 임시
사본은 이미 삭제돼 이 정본과 byte 단위 직접 대조는 불가(NAS 원본이 그 이후 바뀌지 않았다는
전제하의 간접 확인).

### `v02_dair_lr01` — DAIR, from-scratch, lr0 상향(2026-09-14)

**이전 시도 실패 기록**(mycfg.yaml 자체 문서화, 옛 config/run은 로컬에 남아있지 않음): `weight_path`만
아키텍처 yaml로 바꾸고 `lr0=0.0001`을 그대로 둔 채 50 epoch 학습 → **mAP50 0.306/mAP50-95
0.151에 정체**(from-scratch에 비해 lr가 지나치게 낮았음). 수정: `lr0`/`lrf` 0.0001→**0.01**
(Rear_Truck/`v02_dair`가 같은 from-scratch+DAIR 조합에서 lr0=0.01로 epoch5에 mAP50 0.638까지
도달한 선례를 채용), `epochs` 50→**300**(patience=200, 전부 소진). 나머지(batch=48,
optimizer=SGD, `imgsz_y=512`, `aug_type:front`, `eval_roi_crop`, 증강 하이퍼파라미터)는 동일.

### `v02_symmetric_bg` — 배경 이미지 추가(2026-09-14)

`v02_symmetric`과 `weight_path`·하이퍼파라미터 전부 동일, **`data`만 교체**
(`dashcam_front_traffic_light_bg.yaml`, train에 배경/객체 없는 이미지 5,769장 추가, val은
동일) — 배경 데이터 추가 효과만 분리해서 보기 위한 설계. `epochs:450`(mycfg.yaml: "주말 학습
시간 예산 63시간에 맞춰 산정"), `patience:200`. **상태(사용자 확인)**: 학습 로그에 `"422
epochs completed in 54.094 hours."` — 450 예산을 다 채운 게 아니라 **epoch 422에서 조기
종료**됐다. `results.csv` mAP50 최고점은 epoch 219(0.78486, 215/220/87/214도 0.7846~0.7849
범위로 거의 동률 — epoch 87부터 이미 사실상 정점권), 219+200≈419로 patience 종료 시점(422)과
정확히 들어맞는다 — **"epoch 219(best) 근방에서 이미 수렴 판정이 났고 이후 200 epoch 연속
개선이 없어 자동 종료"**가 정확한 설명이지 "450 미완주"가 아니다. 세 신규 실험 중 실제로
patience 조기 종료가 발동한 것은 이 실험뿐이다.

### `v02_symmetric_scratch` — from-scratch(2026-09-14)

`v02_symmetric`과 달리 `weight_path`가 아키텍처 yaml(from-scratch). `lr0`/`lrf` 0.0001→
**0.01**, `epochs` 50→**500**, `patience` 200→**15**(Rear_Truck/`v01_symmetric`의
from-scratch 예산 채용 — mycfg.yaml에 "`v02_dair`의 lr 실수를 명시적으로 피하려 했다"고
기록됨). `data: dashcam_front_traffic_light.yaml`(원본, `_bg` 아님), `imgsz_y=512`,
`aug_type:front`, `eval_roi_crop` — Front 고유 설정은 Rear 값으로 덮지 않고 유지. `results.csv`
501행(epoch 1~500) — **500 epoch 전부 완주**, patience=15 조기 종료 발동 안 함.

## 2. 계보(lineage)

```
standard (원본 as-run, 다른 checkout, workers=8, data=BSD-F606A/Front.yaml, 640x480)
  │  (정본화: workers→4, data path 정리)
  ▼
configs/releases/BSD_F606/Front_Truck/v01/exp.yaml  (canonical, pretrained_weight.pt)
  ├─(그대로 재현, 동일 시작 가중치)──────────▶ v01_repro3      [640x480, from pretrained]
  ├─(A1: data_args.imgsz_y=512 추가)────────▶ v02_symmetric    [640x512, from pretrained]
  └─(data만 배경 5,769장 추가)──────────────▶ v02_symmetric_bg [640x512, from pretrained, epochs 450예산]

symmetric_yolov8.yaml (아키텍처, 랜덤 초기화 — pretrained 계열과 별개 갈래)
  ├─(model_name=symmetric, lr0=0.01, epochs=500, patience=15)──▶ v02_symmetric_scratch
  └─(model_name=dair, lr0=0.01, epochs=300)─────────────────────▶ v02_dair_lr01
```

시작 가중치가 `standard`=`v01_repro3`로 확인됐고, `v02_symmetric`도 같은 release 경로를
가리키므로 **`v01_repro3` vs `v02_symmetric`은 해상도만 다른 대조군 관계**다.
`v02_symmetric_scratch`/`v02_dair_lr01`은 사전학습 가중치 없이 아키텍처 yaml에서 바로
시작하는 별개 갈래 — trainer(symmetric vs dair)만 다르고 나머지는 최대한 맞춰 직접 대조하도록
설계됐다.

## 3. 지표 비교 (2026-09-09, `best.pt` 공식 재검증)

**측정 조건**(공통 조건 ID로 취급, [90](90_reproduction_reference.md) 참조): ultralytics
native `ap_per_class`, 표준 단일-IoU NMS, `conf=0.001`, `iou=0.7`, 학습 때와 동일한 dycrop
crop 1280×1024→해당 해상도 리사이즈, 전체 **6,076장/53,262 인스턴스**.

> **초기 함정(2026-09-09 정정)**: 최초 작성 시 `v01_repro3`/`v02_symmetric` 행은
> `results.csv`의 epoch 50(=`last.pt`) 값을 썼는데, `best.pt`와 `last.pt`의 sha256이 서로
> 달라(별개 체크포인트) epoch 50이 실제 배포 대상인 `best.pt`와 다르다는 게 확인됐다. 아래는
> 세 실험 모두 `best.pt`를 직접 재검증한, 정정된 값이다. `standard`는 재실행하지 않고 기존
> `MANIFEST.yaml`/`validation.md` 기록(2026-08-13 측정, 동일 명령·동일 `conf=0.001` 기본값)을
> 그대로 인용했다 — 이미 같은 방법으로 측정된 값이라 재실행 불필요.

| | `standard`(baseline) | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| 해상도(학습=검증) | 640×480 | 640×480 | 640×512 |
| 값 출처 | `MANIFEST.yaml` `pytorch_val_baseline`(2026-08-13) | `best.pt` 재검증(2026-09-09) | `best.pt` 재검증(2026-09-09) |
| P/R | 0.869/0.670 | 0.867/0.670 | 0.876/0.677 |
| **mAP50** | **0.774** | **0.774** | **0.783** |
| **mAP50-95** | **0.554** | **0.554** | **0.562** |

클래스별(mAP50/mAP50-95): bus 0.809/0.660 · 0.808/0.660 · 0.818/0.669 — car 0.886/0.715 ·
0.887/0.716 · 0.892/0.724 — motorcycle 0.759/0.479 · 0.758/0.480 · 0.765/0.485 — person
0.752/0.494 · 0.752/0.494 · 0.761/0.503 — truck 0.795/0.644 · 0.795/0.644 · 0.803/0.651 —
traffic_light 0.641/0.330 · 0.642/0.331 · 0.657/0.340 (열 순서: standard/v01_repro3/v02_symmetric).

**관찰**: `v01_repro3`는 `standard`와 소수점 셋째 자리까지 사실상 동일(±0.001 이내) — 재현
성공. `v02_symmetric`은 전 클래스 개선(all ΔmAP50 +0.009, ΔmAP50-95 +0.008) — 단 이 개선의
원인(해상도를 512로 맞춘 것 자체 vs 그 해상도로 재학습한 것)은 §3.1에서 분리한다.

## 3.1 640×512로 통일한 비교 (해상도 confound 제거) — 핵심 결과

§3은 각 실험을 **자기 학습 해상도**로 검증한 값이라 480 vs 512 비교가 섞여 있었다. 세
`best.pt` 전부를 **640×512로 통일**해서 재검증(`standard`/`v01_repro3`는 학습 때 보지 못한
해상도로 재평가 — 완전 conv 구조라 실행은 되나 학습 해상도와는 다름에 유의).

| | `standard` | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| 학습 해상도(참고) | 640×480 | 640×480 | **640×512** |
| 검증 해상도(통일) | 640×512 | 640×512 | 640×512 |
| P/R | 0.869/0.678 | 0.872/0.677 | 0.876/0.677 |
| **mAP50** | **0.781** | **0.781** | **0.783** |
| **mAP50-95** | **0.560** | **0.560** | **0.562** |

클래스별(mAP50/mAP50-95, 640×512 통일): bus 0.812/0.665 · 0.812/0.663 · 0.818/0.669 — car
0.890/0.723 · 0.891/0.724 · 0.892/0.724 — motorcycle 0.764/0.482 · 0.765/0.483 · 0.765/0.485
— person 0.762/0.499 · 0.762/0.500 · 0.761/0.503 — truck 0.800/0.648 · 0.801/0.649 ·
0.803/0.651 — traffic_light 0.658/0.342 · 0.659/0.342 · 0.657/0.340.

**결론**: 해상도를 512로 통일하면 `standard`/`v01_repro3`(480 학습, 512로 재평가만)와
`v02_symmetric`(512로 직접 재학습)이 **거의 차이 없음**(all mAP50 0.781 vs 0.783 +0.002,
mAP50-95 0.560 vs 0.562 +0.002) — person·traffic_light는 오히려 `v02_symmetric`이 근소하게
낮다. **즉 §3에서 본 "+0.009/+0.008 개선"의 대부분은 512 재학습 효과가 아니라 "480→512로
검증 해상도만 바꿔도 나오는 효과"였다** — crop(1280×1024, 종횡비 1.25)을 480(1.333)이 아니라
512(1.25)로 리사이즈해 종횡비 왜곡이 줄어드는 것만으로 이득의 대부분(+0.007/+0.006)이 생겼고,
실제 512 재학습의 추가 기여는 **+0.002/+0.002 수준으로 매우 작다.** 배포 스펙이 640×512
하나뿐이므로 512로 학습한 `v02_symmetric` 채택 자체는 여전히 타당하나, "해상도를 맞춰
재학습하면 크게 개선된다"가 아니라 **"애초에 480 학습이 512 배포 대비 갖던 손실 대부분이
검증 해상도 문제였다"**가 정확한 해석이다.

## 3.2 ONNX 변환 (2026-09-09)

세 `best.pt` 전부 `imgsz=640/imgsz_y=512/opset=13/simplify=False`로 병합(`best.onnx`,
디코딩 완료 단일 텐서)/분리(`best_split.onnx`, box/cls 분리, DFL 적분 전 — NPU 배포 구조와
동일) 두 변형 export 완료.

> **함정(실제로 겪음)**: `-p`(`--skip_postprocess`) 유무와 무관하게 export 직후 파일명은
> 항상 `best.onnx`이고 `-s`는 export 후 **이름만 바꿔 옮기는 것**이다. "병합 먼저 → 분리를
> `-s`로 나중에" 순서로 하면 분리 export가 방금 만든 병합 버전을 그대로 덮어쓴 뒤 옮겨져
> **병합 버전이 사라진다**(`v02_symmetric`에서 실제 발생 → 재실행으로 복구). 두 변형을 다
> 만들 땐 분리 버전을 먼저 `-s`로 옮기고 병합 버전을 마지막에(기본 이름 그대로)만들 것.

파일 크기: `best.onnx` standard 12,208,704B(08-14)/v01_repro3 12,208,716B(09-09)/
v02_symmetric 12,208,716B(09-09). `best_split.onnx` standard 12,071,501B(08-14)/v01_repro3
12,071,513B(09-04)/v02_symmetric 12,071,513B(09-09) — ±16B(메타데이터 버전 차이 수준),
graph/initializer 단위 비교는 **아직 하지 않음**.

**검증 — `best.onnx`=`best_split.onnx`(디코딩 전 상태) bit-exact 확인**: 고정 랜덤 입력
(seed=0)으로 PyTorch/ONNX Runtime 비교, `best_split.onnx` raw 출력을 모델 자신의
`decode_bboxes`로 복원해 `best.onnx`와 대조 — **세 모델 전부 box/cls 절대오차 정확히 0.0**
(PyTorch vs ONNX Runtime 자체의 일반적 export 노이즈는 별개로, box 최대 5.8e-4~1.4e-3, cls
최대 ~2e-7~8e-7 수준). `best_split.onnx`는 다른 모델이 아니라 같은 연산을 디코드/concat
이전에 끊어낸 것뿐임을 확인.

**NPU 변환(v02_symmetric, v01_repro3, 2026-09-09 완료)** — `novatek/convert/
front_truck_{v02_symmetric,v01_repro3}_20260909/`. calibration 500장·gen_config(chip530,
640×512, calib_mode4, cle_en0)는 §2(02번 문서)의 `front_truck_v01_reference_onnx_local_
reconvert_20260814`(NAS reference ONNX 재변환 때 쓴 것)를 그대로 재사용, 모델만 이번
`best_split.onnx`로 교체.

| | `v02_symmetric` | `v01_repro3` |
|---|---|---|
| nvt_model.bin | 4,758,792 B, DRAM 4,915,200 B, 연산시간 19.628ms | 4,758,792 B(동일 크기), DRAM 4,915,200 B |
| Simulate(**calibration 이미지 1장 기준**) | Sigmoid(cls) 0.9837, dfl_conv(box) 0.9864, Softmax(dfl 중간) 0.9717 | Sigmoid(cls) 0.9855, dfl_conv(box) 0.9858 |
| Visualize(같은 1장) | Golden 11개 vs NPU Sim 10개(최저 confidence truck 0.253 1개 누락, traffic_light 0.525→0.389로 가장 크게 하락) | Golden 10개 vs NPU Sim 10개(전부 매칭, 누락 없음) |

**부수 발견**: `novatek/convert/visualize_sim.py`가 cls 출력 자르기에서 클래스 수를 **5로
하드코딩**(Rear_Truck류 5클래스 기준)해뒀던 버그를 발견·수정(Front_Truck 6클래스에서
`ValueError` 발생 → golden cls 크기에 맞춰 자르도록 수정, `load_sim()`에 `golden_cls_size`
인자 추가, 커밋 대상).

**범위 — 명확히 한정**: 위는 **calibration 이미지 1장** 기준 compile+simulate+visualize일
뿐이다. **Board 실측이나 전체 데이터셋 기준 quantization loss(mAP)는 이 시점에 존재하지
않는다** — 별도 후속 작업(board dump 필요, §4).

## 3.3 `best.onnx` 전체 데이터셋 재검증 — PT vs ONNX (2026-09-09)

§3.2는 랜덤 입력 1개로 "같은 연산인가"만 확인한 것 — 여기서는 실제 정확도 지표가 export로
달라지는지 전체 6,076장/53,262 인스턴스로 재검증.

**(정정) `standard`의 ONNX 열은 우리가 만든 export가 아니라 NAS에 실제 배포된 onnx로
교체했다** — §1(02번 문서 §1)에서 우리 export(`best.pt`→`best.onnx`)와 NAS 실제 배포 onnx
(`260706_BSD-F606A_Front_v01.onnx`, sha256 `a4e15dea...`, 2026-09-09 NAS 재대조로 지금도
동일 확인)가 **서로 다른 가중치**(126/127 불일치)임이 드러났으므로, "export가 정확도를
깎는가"엔 우리 export를, "실제 제품에 뭐가 올라가 있나"엔 NAS 실제 onnx를 따로 봐야 한다.
NAS 실제 onnx는 박스/클래스 분리+디코딩 전 상태라 `tools/validate_symmetric_jdh.py`
(`SymmetricDetectionValidatorJdh`)로 디코딩해 읽었다 — 표의 P/R/mAP는 동일 엔진
(`ap_per_class`) 기준으로 PT/우리 export 열과 비교 가능하다.

| | standard(PT) | **standard(ONNX, NAS 실제 배포)** | v01_repro3(PT→ONNX 우리export) | v02_symmetric(PT→ONNX 우리export) |
|---|---|---|---|---|
| P | 0.869 | **0.847** | 0.872→0.871 | 0.876→0.875 |
| R | 0.678 | **0.646** | 0.677→0.677 | 0.677→0.677 |
| **mAP50** | 0.781 | **0.744** | 0.781→0.781 | 0.783→0.783 |
| **mAP50-95** | 0.560 | **0.484** | 0.560→0.560 | 0.562→0.562 |

(참고: `standard`의 **우리 export** `best.onnx`는 P/R/mAP50/mAP50-95 = 0.869/0.678/0.781/
0.560으로 PT와 동일 — export 자체는 무손실. `v01_repro3`/`v02_symmetric`은 NAS에 배포된
적이 없어 대응하는 "NAS 실제" 값이 없다.)

클래스별(mAP50/mAP50-95): bus 0.812/0.665, **NAS실제 0.786/0.580**, v01_repro3 0.812/0.663→
0.812/0.664, v02_symmetric 0.818/0.669→0.818/0.669 — car 0.890/0.723, **0.867/0.645**,
0.891/0.724→0.891/0.724, 0.892/0.724→0.892/0.724 — motorcycle 0.764/0.482, **0.704/0.407**,
0.765/0.483→0.765/0.483, 0.765/0.485→0.765/0.485 — person 0.762/0.499, **0.708/0.402**,
0.762/0.500→0.762/0.500, 0.761/0.503→0.761/0.503 — truck 0.800/0.648, **0.782/0.566**,
0.801/0.649→0.801/0.649, 0.803/0.651→0.804/0.650 — traffic_light 0.658/0.342, **0.619/0.304**,
0.659/0.342→0.659/0.342, 0.657/0.340→0.657/0.340.

**결론(정정)**: `v01_repro3`/`v02_symmetric`은 (우리끼리의) PT↔ONNX export 차이가 여전히
표시 자릿수 안에서 0 — 이 결론은 유지. 그러나 **`standard`는 다르다** — 우리 export(=PT와
동일)와 **NAS 실제 배포 onnx**를 같은 조건·엔진으로 비교하면 **all ΔmAP50 -0.037, ΔmAP50-95
-0.076**로 뚜렷이 갈린다(6클래스 전부 일관 하락, motorcycle/person이 가장 큼 -0.06~-0.10).
이는 export 손실이 아니라 §1(02문서)에서 확인한 "애초에 다른 가중치"라는 사실이 정확도
수치로 드러난 것 — 이 repo의 `best.pt`는 NAS에 배포된 모델보다 (이 표준 프로토콜 기준으로는)
더 높은 mAP을 내는, **분명히 다른 모델**이다.

## 3.4 Loss/mAP 추이 분석 (2026-09-09)

`v01_repro3`/`v02_symmetric`의 `results.csv`(1~50 epoch) 분석.

- **Train loss**: epoch 1→36 동안 사실상 정체(v01_repro3 0.840→0.839, v02_symmetric
  0.829→0.830). **epoch 41 급락(v01_repro3 0.798, v02_symmetric 0.790)은 실제 학습 개선이
  아니라 `close_mosaic:10`(마지막 10 epoch 마자이크 비활성) 아티팩트** — 더 쉬운 이미지로
  바뀌어 loss가 떨어진 것.
- **Val loss**: 같은 구간에서 오히려 소폭 악화(v01_repro3 epoch36 0.88379(최저)→epoch50
  0.88647, v02_symmetric epoch36 0.87415(최저)→epoch50 0.87711) — mosaic을 끈 뒤 남은
  10 epoch이 train엔 맞지만 val 일반화엔 도움 안 됨(경증 과적합).
- **mAP 정점 후 하락**: v01_repro3 mAP50 정점 epoch37(0.77376)→epoch50(0.77244, -0.0013).
  v02_symmetric 정점 epoch40~41(0.78284/0.78278)→epoch50(0.78122, -0.0016).
- **LR 스케줄**: `cos_lr` + `lr0=0.0001`(원본 as-run, ultralytics 기본값 0.01의 1/100) —
  epoch50 시점 LR **1.09e-7**, 마지막 5~10 epoch은 가중치가 거의 안 움직임.
- **결론(비판적 판단)**: "꾸준히 감소 vs 정체" → **정체**. mosaic-on 구간(1~40) 내내 평평,
  epoch41 급락은 일반화 개선 아님. val loss/mAP 기준 **epoch 37~41 근방이 사실상 최적점**,
  나머지 epoch은 mAP을 0.001~0.002 깎음. "이어서 더 학습해도 되는가" → **근거 없음**(LR이
  이미 코사인 스케줄 끝까지 소진, val mAP 이미 하락 추세에서 멈춤 — resume해도 변화 없고,
  LR을 다시 올리면 별도의 새 실험). §1의 pretrained_weight.pt 계보(이미 이전 DAIR 학습의
  epoch17 체크포인트)가 이 정체를 정합적으로 설명. "처음부터 epoch을 높게 잡았으면 나았을까"
  → **근거상 아니오**(epoch을 늘려도 코사인 LR 감쇠가 늘어날 뿐, 상대적으로 높은 LR
  구간(epoch6~11, lr≈9e-5)에서도 train loss가 거의 안 움직였음 — "덜 돌려서 못 배웠다"가
  아니라 "lr0=0.0001 자체가 이미 수렴에 가까운 시작점 대비 낮다"가 데이터와 더 부합. 과거
  `v01_ga_candidate.yaml`(미검증)이 `lr0=0.0047`(47배 높음)을 썼던 이력 참고 — epoch 수보다
  epoch=35~40 조기 종료 또는 lr0 상향 실험이 더 근거 있는 다음 스텝).

## 3.5 정성적 영상 비교 — GT 없음 (2026-09-04 / 2026-09-09)

§3~§3.4는 정지 이미지셋(GT 있음) 기준. 여기서는 실제 주행 영상에서 detection이 시각적으로
어떻게 나오는지 `standard`/`v01_repro3`/`v02_symmetric` 3모델을 비교(**정량 지표 없음, 육안
검토용**).

**방법**: ROI crop(320,0,1280,1024) → 640×512 squash resize(dycrop 학습과 동일) → BGR, `/255`
없음 → 표준 단일-IoU NMS(iou=0.45), **confidence_thres=0.25**(§3의 golden 평가용 conf=0.001과
다름 — 저신뢰 박스가 육안 검토에 방해돼 시각 검토 전용값 사용). `standard`는 배포 onnx
(`weights/deployment_260706.onnx`, §3.3에서 NAS 실제 배포본과 그래프 100% 일치 확인됨),
`v01_repro3`/`v02_symmetric`은 각자의 `best_split.onnx`(§3.2에서 bit-exact 확인).

**산출물**: 1차(2026-09-04) `standard` vs `v01_repro3` 단독 영상 14클립(INF 11 + EVT 3) —
`runs/video_infer/front_infer_baseline/`, `front_infer_v01_repro3/`. 2차(2026-09-09) 3-패널
합성 12클립(day/night × city/highway/tollgate/FVSA/sunflare/rainy) —
`runs/video_infer/front_infer_compare/`(`*_compare.mp4`).

**상태 — 미검토**: 영상 자체(1차14+2차12클립)는 생성돼 있으나 **재생해서 정성적 소견(오탐/
누락 패턴, 조건별 차이)을 기록한 사람이 아직 없다** — §4에 다음 확인 사항으로 남음.

## 3.6 신규 실험 3종 추가 (2026-09-14)

세 실험의 `best.pt`를 §3.1과 동일 조건(640×512, `conf=0.001`, `iou=0.7`, 전체 6,076장/
53,262 인스턴스)으로 재검증. `v02_symmetric_bg`는 **완료된 결과**(450 예산 중 epoch 422
조기종료, 정점 epoch 219 부근).

| | standard | v01_repro3 | v02_symmetric | **v02_symmetric_bg** | **v02_symmetric_scratch** | **v02_dair_lr01** |
|---|---|---|---|---|---|---|
| 시작 가중치 | pretrained | pretrained(=standard) | pretrained | pretrained | **from-scratch** | **from-scratch** |
| trainer | symmetric | symmetric | symmetric | symmetric | symmetric | **dair** |
| epochs | — | 50 | 50 | **422(조기종료,예산450)** | **500(완주)** | **300(완주)** |
| P/R | 0.869/0.678 | 0.872/0.677 | 0.876/0.677 | 0.878/0.677 | **0.881/0.686** | 0.861/0.663 |
| **mAP50** | 0.781 | 0.781 | 0.783 | 0.785 | **0.790** | 0.761 |
| **mAP50-95** | 0.560 | 0.560 | 0.562 | 0.564 | **0.566** | 0.546 |

클래스별(mAP50/mAP50-95, v02_symmetric/v02_symmetric_bg/v02_symmetric_scratch/v02_dair_lr01):
bus 0.818/0.669 · 0.819/0.672 · 0.815/0.672 · 0.814/0.663 — car 0.892/0.724 · 0.892/0.726 ·
0.893/0.724 · 0.884/0.709 — motorcycle 0.765/0.485 · 0.768/0.487 · 0.765/0.479 · 0.747/0.472
— person 0.761/0.503 · 0.765/0.505 · 0.765/0.506 · 0.745/0.488 — truck 0.803/0.651 ·
0.802/0.650 · 0.805/0.651 · 0.799/0.642 — **traffic_light 0.657/0.340 · 0.662/0.346 ·
0.696/0.363 · 0.580/0.301**.

**관찰(비판적 판단)**:
- **`v02_symmetric_scratch`(from-scratch, 500epoch)가 6개 모델 중 all/거의 전 클래스에서
  최고 mAP** — traffic_light 0.696/0.363(기존 최고 v02_symmetric 0.657/0.340 대비 크게 앞섬).
  from-scratch가 pretrained 계열보다 나은 것은 **직관에 반해 비판적으로 볼 필요**: (1) 500
  epoch(다른 모델의 10배)이라는 훨씬 큰 예산이 가장 유력한 원인 후보 — "from-scratch가
  근본적으로 낫다"는 결론 아님, (2) pretrained 계열(DAIR lineage, §1)의 원 데이터/태스크가
  현재 타깃(6클래스, traffic_light 포함)과 달라 전이 효과가 제한됐을 가능성, (3) 두 요인
  분리에는 같은 epoch 예산으로 맞춘 대조 실험이 필요 — **아직 없음**.
- **`v02_symmetric_bg`(배경 추가)는 `v02_symmetric` 대비 전 클래스 소폭 개선**(mAP50/
  mAP50-95 각 +0.002) — **이건 완료된 결과이자 세 신규 실험 중 가장 신뢰도 높은 비교**
  (기존 3개 실험과 epoch 예산 차이가 크지 않음 — 사실상 epoch219 근방에서 이미 수렴).
- **`v02_dair_lr01`(DAIR, from-scratch, 300epoch)은 세 신규 실험 중 유일하게 기존
  baseline(standard 0.781/0.560)보다도 낮음** — all mAP50 0.761(-0.020), traffic_light
  0.580(기존 최저 standard의 0.657보다도 낮음), person 0.745/0.488(전체 중 최저)에서 두드러짐.
  DAIR consistency loss가 이 데이터/클래스 구성에서 symmetric 단독보다 수렴이 더디거나
  어려울 수 있으나, **epoch 예산(300 vs scratch symmetric의 500)이 다르다는 점도 함께
  고려해야 함** — 같은 epoch 재비교 전까지 "DAIR이 더 나쁘다" 단정은 이르다.
- **공통 한계**: `v02_dair_lr01`(300 완주)·`v02_symmetric_scratch`(500 완주)는 patience가
  발동하지 않아(예산 전부 소진) **실제 정점 위치를 아직 모른다** — 기존 3개 실험(50epoch)이
  중반(30~40)에 이미 정점 후 정체했던 패턴을 감안하면 이 둘도 훨씬 이른 시점에 이미 수렴했을
  가능성이 있다. `v02_symmetric_bg`만 예외(patience 발동, 정점 epoch219 확인됨) — 즉 세
  신규 실험의 epoch 예산(300/422/500) 자체가 "학습 방식 효과"와 뒤섞인 confound이나,
  `v02_symmetric_bg`만은 유효한 비교다.

## 4. 알려진 한계 / 다음 확인 사항

- `standard` 자체 학습 곡선은 복원 불가(`results.csv` DRM 암호화) — baseline 행은 재학습이
  아니라 이 repo에서 `best.pt`를 별도 재평가해 만든 값(단, `v01_repro3`/`v02_symmetric`과
  동일 도구/설정이라 비교엔 문제없음).
- `standard`의 배포 지표(`MANIFEST` onnx mAP 0.6965/board 0.704, [02](02_front_reference_and_reproduction.md)
  §4~§5 참조)와 이 문서 §3 PyTorch val 값은 파이프라인이 달라(전처리·후처리·평가 엔진)
  **직접 비교 대상이 아니다** — `standard/validation.md`가 이미 그 격차를 규명함. §3 값들은
  PyTorch val 계열 내에서만 비교할 것.
- **NPU 변환·시뮬레이션은 calibration 이미지 1장 기준까지만 완료**(§3.2) — **board 실측이나
  전체 데이터셋 기준 quantization loss(mAP)는 아직 없다.** Rear_Truck 계열에서 확립된 방법
  (board dump → `merge_board_dump.py` → `ultralytics_style_ap.py`, [01](01_rear_quantization_and_release.md)
  §3.4 참조)을 그대로 적용 가능하나 미실행.
- **정성적 영상 비교(§3.5) 육안 검토 — 미완료.** 영상(1차14+2차12클립)은 이미 생성돼 있으나
  재생 후 오탐/누락 패턴, 조건별(야간·역광·우천·톨게이트) 차이를 정리한 소견은 아직 없다.
- **(2026-09-14) `v02_dair_lr01`/`v02_symmetric_scratch`의 epoch 예산 불일치**: 300(완주)/
  500(완주) vs 기존 3개 실험의 50 — §3.6 관찰("from-scratch가 낫다"/"DAIR이 나쁘다") 모두
  epoch 예산 차이와 분리되지 않았다. patience 미발동이라 실제 정점 위치 불명(§3.4 방식으로
  확인 필요). `v02_symmetric_bg`는 patience 발동으로 정점(epoch219) 확인 완료 — 예외.
- `v02_dair_lr01`/`v02_symmetric_bg`/`v02_symmetric_scratch`는 ONNX 변환·NPU 변환·영상 정성
  비교(§3.2/§3.5) **어느 것도 아직 하지 않았다.**

## 5. 향후 실험 추가 규칙

새 Front_Truck v01 계열 실험이 추가되면: (1) §1에 실험 정의(경로/정체/`weight_path`/`data`/
해상도/epochs) 추가, (2) §2 계보 다이어그램에 위치 추가, (3) §3 표에 행 추가 — 열 순서
(해상도/출처/P/R/mAP50/mAP50-95) 유지, 값의 출처(학습 내장 val vs 공식 재검증)를 반드시 명시.
