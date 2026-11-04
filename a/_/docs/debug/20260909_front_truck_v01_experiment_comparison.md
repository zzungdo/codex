# BSD_F606 Front_Truck v01 계열 실험 비교 — standard / v01_repro3 / v02_symmetric

- **작성**: 2026-09-09 (KST)
- **목적**: `runs/detect/BSD_F606/Front_Truck/` 아래 존재하는 v01 계열 실험 3개
  (`standard`, `v01_repro3`, `v02_symmetric`)의 정체·계보·지표를 한 곳에 확정해 기록하고,
  앞으로 추가되는 v01 계열 실험을 이 문서에 이어서 채워 넣는다. **living document** —
  새 실험이 생기면 §1에 정의를 추가하고 §3 표에 행을 추가한다.
- **범위**: 이 문서는 기존 산출물(`runs/`, `configs/releases`, `configs/experiments`,
  `MANIFEST.yaml`) 재정리·대조이며, 이 세션에서 재학습·재검증은 수행하지 않았다.
- **전제 확인 (사용자, 2026-09-09)**: `v01_repro3`의 시작 가중치(`weight_path`)는
  `standard`(baseline)와 **동일 가중치**다 — 아래 §2 계보의 근거.

---

## 1. 실험 정의

### `standard` — baseline (release v01 as-run 원본, 이 repo에서 학습되지 않음)

- 경로: `runs/detect/BSD_F606/Front_Truck/standard/`
- 정체: `configs/releases/BSD_F606/Front_Truck/v01`의 **as-run 원본 번들**(`weight/` 폴더)
  **내용**을 담고 있다. `args.yaml`의 `workers: 8`, `save_dir: results/BSD-F606A/Front_v01_sym/train1`이
  `MANIFEST.yaml`의 `as_run_differs`(workers as_run=8, save_dir as_run=동일 경로)와 정확히
  일치 — `MANIFEST.run.origin_checkout`(`/home/ncn/Developments/adassw/sources/ultralytics_projects`,
  이 repo와 다른 checkout)에서 **학습된** 원본이다.
- **이 repo 안에서의 실제 출처(2026-09-09, 사용자 확인 + mtime/sha256 근거로 정정)**: 이
  폴더의 파일들은 원본 학습 시점 타임스탬프를 보존한 채(rsync -a 등) 옮겨진 게 아니라, **전부
  2026-08-13 15:49 KST 단일 시각**으로 mtime이 통일돼 있다 — NAS 원본(`weight/` 파일들은
  2026-07-07~08)과 다른 값이라 평범한 `cp`로 이 시점에 **새로 복사돼 배치**됐다는 뜻이다. 이
  시점은 `standard/validation.md`(2026-08-13 작성) 조사와 일치한다. 흔적: `configs/releases/
  BSD_F606/Front_Truck/v01`(exp.yaml/MANIFEST.yaml는 2026-08-03에 이미 커밋돼 있었다) 경로가
  이 번들의 정식 위치이고, 그 `weight/`가 NAS `nas_source`에서 온 것 — 실제로 `standard/
  pretrained_weight.pt`의 sha256(`83e51540d24a9c0f32deabc14304462287752974554a24697f845
  fe3735b363f`)이 **현재 NAS 원본과 정확히 일치**한다(2026-09-09 직접 대조). 즉 "다른
  checkout에서 학습됨"은 **학습이 실행된 위치**에 대한 사실이고, "이 repo에 있는 사본이 어디서
  왔나"는 별개 질문인데 그 답은 `configs/releases/BSD_F606/Front_Truck/v01`(=NAS 번들)이다.
- `weight_path`(원본 pretrained): `results/BSD-F606A/Front_v01/pretrained_weight.pt`
  (학습 당시 그 checkout에서의 상대 경로 표기 — 파일 자체는 위에서 확인했듯 NAS 번들의
  `pretrained_weight.pt`와 동일하다)
- `data`: `configs/datasets/BSD-F606A/Front.yaml` (legacy 경로 표기 — 물리적으로는
  `dashcam_front_traffic_light`와 동일 데이터로 MANIFEST가 식별)
- 해상도: 640×480 (`imgsz_y` 미지정 → 기본값, 크롭 1280×1024)
- epochs: `args.yaml`상 50 요청. **완료 여부 불명** — `results.csv`가 DRM 암호화(`MANIFEST.yaml`
  `results-csv-drm` 이슈)라 학습 곡선·실제 완료 epoch을 복원할 수 없다.
- ⚠ 이 폴더의 `mycfg.yaml`은 **신뢰 불가**(`MANIFEST.yaml` `mycfg-unreliable` 이슈) — GA
  튜닝 값(`lr0=0.0047` 등)을 담고 있으나 같은 폴더 `args.yaml`의 실제 `lr0`은 `0.0001`이다.
  이 폴더 안에서도 실제 학습 레시피는 **`args.yaml` 기준**으로 읽어야 한다.
- `weights/` 안 `best.onnx`/`best_split.onnx`/`deployment_260706.onnx`/
  `260706_BSD-F606A_Front_v01.onnx`는 8/13~8/31 날짜로 이후 디버깅 세션(reference ONNX
  조사)에서 추가된 파일 — 원본 번들 구성물이 아니다.
- **`pretrained_weight.pt`(파인튜닝 시작점)의 실제 정체 (2026-09-09, `torch.load`로 체크포인트
  메타데이터 직접 확인)**: 단순 초기화 가중치가 아니라 **이전 학습의 epoch 17 체크포인트**
  (`best_fitness=0.5287`)였다. 그 이전 학습의 `train_args`에서 드러난 사실:
  - `save_dir: /results/FrontDAIRv8/dashboard_aug_add_data/train3` — **DAIR 트레이너 계열**.
    `train_metrics`에 `val/box_org`/`val/box_aug`/`val/box_reg`(원본 view/증강 view/consistency
    정규화) 필드가 있는데, 이는 DAIR의 dual-forward consistency loss 분해와 정확히 일치 —
    원출처가 DAIR였다는 강한 근거.
  - `model: /results/FrontDAIRv8/train3/weights/best.pt` — 이 체크포인트조차 **한 세대 더
    이전의 다른 체크포인트**에서 파인튜닝됐다. 그 이전 체크포인트는 경로만 남아있고 파일이
    없어 더 추적 불가.
  - `data: projects/data_yaml/TLDS_rework_add.yaml` — 지금 이 repo가 쓰는
    `dashcam_front_traffic_light.yaml`/`BSD-F606A/Front.yaml`과 **다른 데이터셋**("TLDS" 계열,
    "rework_add" = 재작업+데이터 추가로 추정). 원본 yaml은 이 repo에 없어 클래스 구성 등은
    확인 불가.
  - `batch=64`, `device='0,1'`(듀얼 GPU), `amp=True` — 지금 exp.yaml의 `amp: false`(D-006)와
    다르다. 이 학습은 D-006(AMP→NaN 발견)보다 이른 시점으로 보여 모순은 아니다.
  - 이 시점 자체 성능: `mAP50=0.6995, mAP50-95=0.5097`(fitness 0.5287) — `standard`의 최종
    성능(mAP50=0.774, mAP50-95=0.554)보다 훨씬 낮다. 즉 v01/`standard` 학습 자체가
    **+0.075 mAP50의 실질적 개선**을 만들어냈다(데이터셋이 바뀌었을 가능성이 있어 이 개선폭이
    순수 fine-tune 효과인지 task/data 전환 효과인지는 분리 불가).
  - **§3-4 loss 분석과의 연결**: 파인튜닝 시작점이 완전 초기화가 아니라 이미 한 차례 이상
    fine-tune된 DAIR 체크포인트였다는 사실은, v01/`standard` 자체 학습(§3-4)에서 관찰된
    "train loss가 초반 40 epoch 내내 거의 안 움직인다"는 현상과 정합적이다 — 이미 어느 정도
    수렴된 모델을 낮은 LR(`lr0=0.0001`)로 살짝만 더 맞추는 상황이었을 가능성이 높다.

### `v01_repro3` — standard 가중치에서 canonical release recipe 재현

- 경로: `runs/detect/BSD_F606/Front_Truck/v01_repro3/`
- 정체: `configs/releases/BSD_F606/Front_Truck/v01/exp.yaml`(**canonical**, `standard`의
  as-run 값을 사람이 정리한 정본)을 재현한 학습. `mycfg.yaml`: `variant: reproducibility-test`,
  `source: configs/releases/BSD_F606/Front_Truck/v01/exp.yaml`.
- `weight_path`: `configs/experiments/BSD_F606/Front_Truck/v01_repro/weight/pretrained_weight.pt`
  — mycfg 주석상 "번들 실물이 이 clone에 없어 다른 clone(`ultralytics_projects-main`)의 원본을
  복사해옴". **사용자 확인(2026-09-09): 이 파일은 `standard`의 원본 pretrained_weight.pt와
  동일 가중치** — 두 실험의 시작점은 같다.
- `data`: `configs/datasets/dashcam_front_traffic_light.yaml` (canonical 경로 — `standard`와
  물리적으로 동일 데이터셋)
- `workers: 4`(CLAUDE.md 호스트 제약 반영, `standard`의 8과 다름 — 의도된 정본화 차이)
- 해상도: 640×480 (`data_args` 없음 → 기본값, `standard`와 동일 조건)
- epochs: 50 완료 (`results.csv` 정상, 암호화 없음)

### `v02_symmetric` — 해상도 수정(A1, imgsz_y=512)

- 경로: `runs/detect/BSD_F606/Front_Truck/v02_symmetric/`
- 정체: `configs/experiments/BSD_F606/Front_Truck/v02_symmetric/exp.yaml`.
  `based_on: configs/releases/BSD_F606/Front_Truck/v01/exp.yaml` + `data_args.imgsz_y: 512`
  (A1 수정, `docs/debug/20260729_config_canonicalization.md`) — 학습 해상도를 배포
  해상도(640×512)와 일치시킨 버전.
- `weight_path`: `configs/releases/BSD_F606/Front_Truck/v01/weight/pretrained_weight.pt`
  (release 번들 경로 표기 — `v01_repro3`처럼 대체 clone 사본이 아니라 release 경로를 직접
  가리킴). **NAS 정본 확인(2026-09-09)**: 프로젝트 정책상 release 산출물은 로컬에 상주하지
  않고 실행 시점에만 NAS에서 임시로 가져와 쓰므로, 지금 로컬 `configs/releases/.../v01/`엔
  `weight/`가 없다(정책대로 정상). NAS 원본(`/mnt/NAS_2/.../BSD_F606/Front_Truck/v01/weight/
  pretrained_weight.pt`)의 sha256은 `83e51540d24a9c0f32deabc14304462287752974554a24697f845
  fe3735b363f` — 이 값이 정본이다. **한계**: `v02_symmetric` 학습 당시 실제 로딩된 임시
  사본은 이미 삭제돼 이 NAS 정본과 byte 단위로 직접 대조할 수는 없다(사본 보존 정책상 원천적
  불가) — NAS 원본이 그 이후 바뀌지 않았다는 전제하의 간접 확인이다.
- `data`: `configs/datasets/dashcam_front_traffic_light.yaml`, `workers: 4`
- 해상도: **640×512** (`imgsz_y: 512` 명시) — `v01_repro3` 대비 유일한 의도된 차이
- epochs: 50 완료

### `v02_dair_lr01` — DAIR, from-scratch, lr0 상향 (2026-09-14 추가)

- 경로: `runs/detect/BSD_F606/Front_Truck/v02_dair_lr01/`, config: `configs/experiments/
  BSD_F606/Front_Truck/v02_dair_lr01/exp.yaml`
- 정체: `model_name: dair`(`DairDetectionTrainer`, dual-forward consistency,
  `dair_params`: mode=or, λ=0.1, γ=0.5). `weight_path`가 **체크포인트가 아니라 아키텍처
  yaml**(`projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml`) — 즉 **완전
  from-scratch**(랜덤 초기화), 사전학습 가중치 없음.
- **이전 시도 실패 기록(mycfg.yaml 자체 문서화, `configs/experiments/BSD_F606/Front_Truck/
  v02_dair`는 로컬에 config/run 둘 다 남아있지 않음 — 사용자 서술만 확인 가능)**:
  `weight_path`만 아키텍처 yaml로 바꾸고 `lr0=0.0001`을 그대로 둔 채 50 epoch 학습 →
  mAP50 0.306/mAP50-95 0.151에 정체(from-scratch에 비해 lr가 지나치게 낮았음).
- 수정: `lr0`/`lrf`를 `0.0001`→**`0.01`**로 상향(Rear_Truck/v02_dair가 같은 from-scratch+DAIR
  조합에서 lr0=0.01로 epoch5 mAP50 0.638까지 도달한 선례를 그대로 채용), `epochs`
  50→**300**. 나머지(batch=48, optimizer=SGD, `data_args.imgsz_y=512`, `aug_type: front`,
  `eval_roi_crop`, 증강 하이퍼파라미터)는 전부 동일.
- `data`: `configs/datasets/dashcam_front_traffic_light.yaml` (v01/v02 계열과 동일)
- `patience: 200` — 300 epoch 전부 소진(조기 종료 없음)
- `best.pt` ≠ `last.pt`(sha256 다름 — 다른 모델들과 같은 패턴, 마지막 epoch이 실제 best가 아님)

### `v02_symmetric_bg` — symmetric, 배경 이미지 추가 (2026-09-14 추가)

- 경로: `runs/detect/BSD_F606/Front_Truck/v02_symmetric_bg/`, config: `configs/experiments/
  BSD_F606/Front_Truck/v02_symmetric_bg/exp.yaml`
- 정체: `v02_symmetric`과 `weight_path`(release v01 pretrained_weight.pt)·하이퍼파라미터
  전부 동일, **`data`만 교체**: `dashcam_front_traffic_light.yaml` →
  `dashcam_front_traffic_light_bg.yaml`(train에 배경/객체 없는 이미지 5,769장 추가, val은
  동일) — 배경 데이터 추가 효과만 분리해서 보기 위한 설계(mycfg.yaml 자체 서술).
- `epochs: 450`(mycfg.yaml: "주말 학습 시간 예산 63시간에 맞춰 산정"), `patience: 200`
- **상태 — 완료, `patience=200` 조기 종료 (사용자 확인, 2026-09-14)**: 학습 로그에
  `"422 epochs completed in 54.094 hours."`로 명시됨 — 450 epoch 예산을 다 채운 게 아니라
  **epoch 422에서 조기 종료**됐다. `results.csv` 자체 mAP50 최고점은 epoch 219(0.78486,
  거의 동률로 215/220/87/214도 0.7846~0.7849 범위 — epoch 87부터 이미 사실상 정점권,
  그 뒤로는 등락만 있었다는 뜻), 219+200≈419로 patience 종료 시점(422)과 정확히 들어맞는다.
  즉 **"epoch 219(best) 근방에서 이미 수렴 판정이 났고, 이후 200 epoch 연속 개선이 없어
  자동 종료됐다"**가 정확한 설명 — 450 미완주가 아니라 **정상적인 조기 종료로 완료된
  결과**다. 세 신규 실험 중 실제로 patience 조기 종료가 발동한 건 이 실험뿐이다
  (`v02_symmetric_scratch`는 500/500, `v02_dair_lr01`은 300/300 전부 완주).
- `best.pt` ≠ `last.pt`(위 최고점 epoch 219 부근이 `best.pt`일 가능성이 높음 — 정확한 epoch은
  `results.csv`만으로는 fitness_mode=f1_50 기준 특정 안 됨)

### `v02_symmetric_scratch` — symmetric, from-scratch (2026-09-14 추가)

- 경로: `runs/detect/BSD_F606/Front_Truck/v02_symmetric_scratch/`, config: `configs/
  experiments/BSD_F606/Front_Truck/v02_symmetric_scratch/exp.yaml`
- 정체: `v02_symmetric`과 달리 `weight_path`가 **아키텍처 yaml**(from-scratch, 사전학습
  없음). `lr0`/`lrf`: `0.0001`→**`0.01`**, `epochs`: 50→**500**, `patience`:
  200→**15**(Rear_Truck/v01_symmetric의 from-scratch 예산을 그대로 채용 — mycfg.yaml에
  `known_pitfall`로 `v02_dair`의 lr 실수를 명시적으로 피하려 했다고 기록돼 있다).
- `data`: `dashcam_front_traffic_light.yaml`(원본, `_bg` 아님), `data_args.imgsz_y=512`,
  `aug_type: front`, `eval_roi_crop` — Front 고유 설정은 Rear 값으로 덮지 않고 그대로 유지.
- `results.csv` 501행(epoch 1~500) — **500 epoch 전부 완주**, `patience=15` 조기 종료
  발동 안 함.
- `best.pt` ≠ `last.pt`

---

## 2. 계보 (lineage)

```
standard (원본 as-run, 다른 checkout, workers=8, data=BSD-F606A/Front.yaml, 640x480)
  │  (정본화: workers→4, data path 정리)
  ▼
configs/releases/BSD_F606/Front_Truck/v01/exp.yaml  (canonical, pretrained_weight.pt)
  ├─(그대로 재현, 동일 시작 가중치)──────────▶ v01_repro3      [640x480, from pretrained]
  ├─(A1: data_args.imgsz_y=512 추가)────────▶ v02_symmetric    [640x512, from pretrained]
  └─(data만 배경 5,769장 추가)──────────────▶ v02_symmetric_bg [640x512, from pretrained, epochs 450]

symmetric_yolov8.yaml (아키텍처, 랜덤 초기화 — pretrained 계열과 별개 갈래)
  ├─(model_name=symmetric, lr0=0.01, epochs=500, patience=15)──▶ v02_symmetric_scratch
  └─(model_name=dair, lr0=0.01, epochs=300)─────────────────────▶ v02_dair_lr01
```

시작 가중치가 `standard`=`v01_repro3`로 확인됐고(§전제 확인), `v02_symmetric`도 표기상 같은
release 경로를 가리키므로, **`v01_repro3` vs `v02_symmetric`은 해상도만 다른 대조군 관계**로
볼 수 있다. `v02_symmetric_scratch`/`v02_dair_lr01`은 위 계열과 달리 **사전학습 가중치 없이
아키텍처 yaml에서 바로 시작**하는 별개 갈래 — trainer(symmetric vs dair)만 다르고 나머지
설정은 최대한 맞춰 두 트레이너를 직접 대조할 수 있게 설계됐다.

---

## 3. 지표 비교 (2026-09-09, `best.pt` 공식 재검증 완료)

**(2026-09-09 갱신)** 최초 작성 시 `v01_repro3`/`v02_symmetric` 행은 `results.csv`의 epoch 50
(=`last.pt`) 값을 썼는데, 이후 `best.pt`와 `last.pt`의 sha256이 서로 달라(둘 다 별개
체크포인트) epoch 50이 실제 배포 대상인 `best.pt`와 다르다는 게 확인됐다. 아래는 그 문제를
바로잡아 **세 실험 모두 `best.pt`를, `standard`와 동일한 방법(ultralytics native `ap_per_class`,
표준 단일-IoU NMS, `conf=0.001`, `iou=0.7`, 학습 때와 동일한 dycrop crop 1280×1024→해당
해상도 리사이즈)으로, 전체 6,076장/53,262 인스턴스에 대해 직접 재검증**한 결과다.

실행 명령 (컨테이너: `dair-train`, 둘 다 `runs/detect/val*`에 원본 저장):

```bash
# v01_repro3 (640x480, standard와 동일 해상도)
uv run python3 tools/validate_symmetric.py \
  -w runs/detect/BSD_F606/Front_Truck/v01_repro3/weights/best.pt \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --cfg_options conf=0.001 iou=0.7

# v02_symmetric (640x512, imgsz_y=512 학습 해상도에 맞춤)
uv run python3 tools/validate_symmetric.py \
  -w runs/detect/BSD_F606/Front_Truck/v02_symmetric/weights/best.pt \
  -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7
```

`standard` 행은 재실행하지 않고 기존 `MANIFEST.yaml`/`validation.md` 기록(동일 명령·동일
conf=0.001 default)을 그대로 인용했다 — 이미 같은 방법으로 측정된 값이라 재실행이 불필요.

| | `standard` (baseline) | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| 해상도 | 640×480 | 640×480 | 640×512 |
| 값의 출처 | `MANIFEST.yaml` `pytorch_val_baseline` (2026-08-13 측정) | `validate_symmetric.py` on `best.pt` (2026-09-09 재검증) | `validate_symmetric.py` on `best.pt` (2026-09-09 재검증) |
| P | 0.869 | 0.867 | 0.876 |
| R | 0.670 | 0.670 | 0.677 |
| **mAP50** | **0.774** | **0.774** | **0.783** |
| **mAP50-95** | **0.554** | **0.554** | **0.562** |

### 클래스별 (mAP50 / mAP50-95)

| 클래스 | `standard` | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| bus | 0.809 / 0.660 | 0.808 / 0.660 | 0.818 / 0.669 |
| car | 0.886 / 0.715 | 0.887 / 0.716 | 0.892 / 0.724 |
| motorcycle | 0.759 / 0.479 | 0.758 / 0.480 | 0.765 / 0.485 |
| person | 0.752 / 0.494 | 0.752 / 0.494 | 0.761 / 0.503 |
| truck | 0.795 / 0.644 | 0.795 / 0.644 | 0.803 / 0.651 |
| traffic_light | 0.641 / 0.330 | 0.642 / 0.331 | 0.657 / 0.340 |

**관찰**:
- `v01_repro3`는 `standard`와 **소수점 셋째 자리까지 사실상 동일**(all 6클래스 ±0.001 이내) —
  동일 시작 가중치·동일 해상도·동일 데이터·동일 `best.pt` 검증으로, 재현이 정확히 성공했음이
  최종 확인됐다. (최초 epoch-50 프록시 값 mAP50 0.77244/mAP50-95 0.55219도 이 결과와
  0.002 이내로 가까웠지만, 이번 `best.pt` 직접 검증이 정본이다.)
- `v02_symmetric`은 전 클래스에서 일관되게 개선 — all 기준 ΔmAP50 **+0.009**, ΔmAP50-95
  **+0.008**. 하지만 이 개선의 원인(해상도를 512로 맞춘 것 자체 vs 그 해상도로 재학습한 것)은
  아직 안 갈렸다 — §3-1에서 해상도를 통일해 분리한다.

---

## 3-1. 640×512로 통일한 비교 (해상도 confound 제거) — **핵심 결과**

§3은 각 실험을 **자기 학습 해상도**로 검증한 값이라 `standard`/`v01_repro3`(480)와
`v02_symmetric`(512)이 서로 다른 해상도를 비교하는 셈이었다. 그 confound를 없애기 위해
**세 `best.pt` 전부를 640×512로 통일**해서 다시 검증했다(`standard`/`v01_repro3`는 학습
때 보지 못한 해상도로 재평가하는 것 — 완전 conv 구조라 실행은 되지만 학습 해상도와는
다르다는 점에 유의).

```bash
# standard, v01_repro3 공통 — v02_symmetric은 기존 §3 값 재사용(이미 512로 검증됨)
uv run python3 tools/validate_symmetric.py \
  -w <best.pt> -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7
```

| | `standard` | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| 학습 해상도(참고) | 640×480 | 640×480 | **640×512** |
| 검증 해상도(통일) | 640×512 | 640×512 | 640×512 |
| P | 0.869 | 0.872 | 0.876 |
| R | 0.678 | 0.677 | 0.677 |
| **mAP50** | **0.781** | **0.781** | **0.783** |
| **mAP50-95** | **0.560** | **0.560** | **0.562** |

### 클래스별 (mAP50 / mAP50-95, 640×512 통일)

| 클래스 | `standard` | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| bus | 0.812 / 0.665 | 0.812 / 0.663 | 0.818 / 0.669 |
| car | 0.890 / 0.723 | 0.891 / 0.724 | 0.892 / 0.724 |
| motorcycle | 0.764 / 0.482 | 0.765 / 0.483 | 0.765 / 0.485 |
| person | 0.762 / 0.499 | 0.762 / 0.500 | 0.761 / 0.503 |
| truck | 0.800 / 0.648 | 0.801 / 0.649 | 0.803 / 0.651 |
| traffic_light | 0.658 / 0.342 | 0.659 / 0.342 | 0.657 / 0.340 |

**결론 — §3의 미해결 질문이 풀렸다**: 해상도를 512로 통일해서 비교하니 `standard`/
`v01_repro3`(480에서 학습된 원본 가중치를 512로 재평가만 한 것)와 `v02_symmetric`(512로
직접 재학습한 것)이 **all 기준 mAP50 0.781 vs 0.783(+0.002), mAP50-95 0.560 vs 0.562
(+0.002)로 거의 차이가 없다.** person·traffic_light는 오히려 `v02_symmetric`이 근소하게
낮다.

**즉 §3에서 관찰한 "v02_symmetric의 +0.009/+0.008 개선"은 대부분 "512로 재학습한 효과"가
아니라 "480→512로 검증 해상도만 바꿔도 나오는 효과"였다** — crop(1280×1024, 종횡비 1.25)을
480(1.333)이 아니라 512(1.25)로 리사이즈해 종횡비 왜곡이 줄어드는 것만으로 이미 이득의
대부분(+0.007/+0.006, §3)이 생겼고, 512 해상도로 실제 재학습한 추가 효과는 **+0.002 mAP50 /
+0.002 mAP50-95 수준으로 매우 작다.** 배포 스펙이 640×512 하나뿐이므로 재학습 자체가
불필요하다는 뜻은 아니지만(추론 시 항상 512로 들어오므로 512로 학습한 `v02_symmetric`을
채택하는 것 자체는 여전히 타당하다), "해상도를 맞춰 재학습하면 baseline 대비 상당히
개선된다"는 것이 아니라 **"애초에 480 학습이 512 배포 대비 갖고 있던 손실 대부분이 검증
해상도 문제였고, 재학습의 실질 기여는 작다"**는 게 정확한 해석이다.

---

## 3-2. ONNX 변환 (2026-09-09)

세 `best.pt` 전부 동일 설정(`imgsz=640`, `imgsz_y=512`, `opset=13`, `simplify=False`)으로
ONNX 변환을 완료했다 — 배포 스펙(640×512)에 맞춘 변환이라 `standard`/`v01_repro3`(학습은
480)도 512 shape로 나온다.

```bash
# 병합(단일 텐서, 디코딩 완료) — 기본 동작
uv run python3 tools/export_symmetric.py -w <best.pt>          # -> <weights_dir>/best.onnx

# 분리(box/cls 분리, DFL 적분 전 — NPU 배포 구조와 동일)
uv run python3 tools/export_symmetric.py -w <best.pt> -p -s <weights_dir>/best_split.onnx
```

⚠ **함정(이번에 직접 겪음)**: `-p`(`--skip_postprocess`) 유무와 무관하게 기본 출력 파일명은
항상 `best.onnx`다 — `-s`는 export가 끝난 뒤 그 파일을 **이름만 바꿔 옮기는 것**이다. 그래서
"병합 먼저 → 분리를 `-s best_split.onnx`로 나중에" 순서로 실행하면, 분리 export가 먼저
`best.onnx`(방금 만든 병합 버전)를 그대로 덮어쓴 뒤에야 `best_split.onnx`로 옮겨져 **병합
버전이 통째로 사라진다**(`v02_symmetric`에서 실제로 발생 → 병합 export를 다시 돌려 복구).
두 변형을 다 만들 땐 **분리 버전을 먼저 `-s`로 옮기고, 병합 버전을 마지막에(기본 이름 그대로)
만들 것**.

| | `standard` | `v01_repro3` | `v02_symmetric` |
|---|---|---|---|
| `best.onnx`(병합) | 12,208,704 B (08-14) | 12,208,716 B (09-09) | 12,208,716 B (09-09) |
| `best_split.onnx`(분리) | 12,071,501 B (08-14) | 12,071,513 B (09-04) | 12,071,513 B (09-09) |

세 모델의 파일 크기가 거의 동일(±16 B, opset/메타데이터 버전 차이 수준)해 그래프 구조가
일관되게 나왔다는 정황 근거는 있으나, **graph/initializer 단위 비교는 아직 하지 않았다** —
필요하면 `tools/debug/reference_compare_graph_and_initializers.py`(2026-08-14 조사에서 쓴
도구)로 세 ONNX를 상호 대조할 수 있다.

**검증 — `best.onnx` = `best_split.onnx`(디코딩 전 상태) 확인 (2026-09-09)**: 같은 랜덤 입력
(`torch.manual_seed(0)`, `rand(1,3,512,640)*255`)을 PyTorch(`best.pt`)와 두 ONNX
(`best.onnx`/`best_split.onnx`, ONNX Runtime `CPUExecutionProvider`)에 넣고, `best_split.onnx`의
raw 출력(box: DFL 적분까지, cls: sigmoid까지)을 모델 자신의 `decode_bboxes(dfl_raw, anchors) *
strides`로 복원해 `best.onnx` 출력과 직접 대조했다. 실측값(절대오차, `abs(a-b)` 기준):

| 모델 | 비교 | box max\|diff\| | box mean\|diff\| | cls max\|diff\| |
|---|---|---|---|---|
| standard | PyTorch 자체(merged vs decoded-split) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 (mean 0.0) |
| standard | **best.onnx vs best_split.onnx(decoded)** | **0.000000e+00** | **0.000000e+00** | **0.000000e+00** |
| standard | (참고) PyTorch merged vs best.onnx | 1.434326e-03 | — | 1.937151e-07 |
| v01_repro3 | PyTorch 자체(merged vs decoded-split) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| v01_repro3 | **best.onnx vs best_split.onnx(decoded)** | **0.000000e+00** | **0.000000e+00** | **0.000000e+00** |
| v01_repro3 | (참고) PyTorch merged vs best.onnx | 8.087158e-04 | — | 2.328306e-07 |
| v02_symmetric | PyTorch 자체(merged vs decoded-split) | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| v02_symmetric | **best.onnx vs best_split.onnx(decoded)** | **0.000000e+00** | **0.000000e+00** | **0.000000e+00** |
| v02_symmetric | (참고) PyTorch merged vs best.onnx | 5.798340e-04 | — | 7.748604e-07 |

세 모델 전부 **box/cls 절대오차 정확히 0.0** — `best_split.onnx`는 `best.onnx`와 다른 모델이
아니라, 같은 연산을 마지막 디코드/concat 이전에 끊어낸 것뿐임을 확인했다. "(참고)" 행은
merged/split 비교가 아니라 **PyTorch(`best.pt`) vs ONNX Runtime(`best.onnx`) 간의 일반적인
export 노이즈**로, box 기준 최대 5.8e-4~1.4e-3, cls 기준 최대 ~2e-7~8e-7 수준이며 merged/split
차이와는 무관하다(연산자 구현 차이에서 오는 부동소수점 오차).

**`v02_symmetric` NPU 변환 완료 (2026-09-09)** — `novatek/convert/front_truck_v02_symmetric_20260909/`.
calibration 500장·`gen_config.txt`/`sim_config.txt` 설정(chip=530, 640×512, calib_mode=4,
cle_en=0)은 `front_truck_v01_reference_onnx_local_reconvert_20260814`(NAS reference ONNX
재변환 때 쓴 것과 동일 제품·카메라·해상도) 것을 그대로 재사용, 모델만 이번에 만든
`best_split.onnx`로 교체. `run_compiler.sh` → `run_simulator.sh` → `visualize_sim.py` 순서로
직접 실행(compile→simulate까지 120초를 넘겨 백그라운드로 완료 확인).

| 단계 | 결과 |
|---|---|
| Compile | `nvt_model.bin` 4,758,792 B, DRAM 4,915,200 B, 연산시간 19.628ms |
| Simulate (calibration 이미지 1장) | Sigmoid(cls) cosine **0.9837**, Conv/dfl/conv(box) cosine **0.9864**, Softmax(dfl 중간) cosine 0.9717 |
| Visualize | Golden(float) 11개 탐지 vs NPU Sim(양자화) 10개 탐지 — 대부분 위치·confidence 거의 일치, 최저 confidence 탐지(truck 0.253) 1개만 누락, traffic_light가 0.525→0.389로 가장 크게 하락 |

**부수 발견 — `novatek/convert/visualize_sim.py` 버그 수정**: cls 출력을 자르는 코드가 클래스
수를 **5로 하드코딩**(`golden_dfl.size // 4 * 5`)해뒀던 게 Front_Truck의 6클래스(bus/car/
motorcycle/person/truck/traffic_light)에서 `ValueError: shapes (40320,) and (33600,) not
aligned`로 죽었다 — Rear_Truck류 5클래스 모델 기준으로 짜여있던 코드로 보인다. golden cls
크기에 맞춰 자르도록 수정(`load_sim()`에 `golden_cls_size` 인자 추가). 이 파일은 gitignore
대상이 아니라 커밋 대상.

**범위**: calibration 이미지 1장 기준 compile+simulate+visualize까지만 — Board 실측이나 전체
데이터셋 기준 quantization loss(mAP)는 별도 후속 작업(다른 모델들처럼 board dump 필요).

**`v01_repro3` NPU 변환 완료 (2026-09-09)** — `novatek/convert/front_truck_v01_repro3_20260909/`,
동일 절차·동일 calibration/설정(위와 동일 템플릿 재사용).

| 단계 | 결과 |
|---|---|
| Compile | `nvt_model.bin` 4,758,792 B(동일 크기), DRAM 4,915,200 B |
| Simulate (calibration 이미지 1장) | Sigmoid(cls) cosine **0.9855**, Conv/dfl/conv(box) cosine **0.9858** |
| Visualize | Golden 10개 vs NPU Sim 10개 — **전부 매칭, 누락 없음**(`v02_symmetric`은 1개 누락) |

### 두 모델 NPU 변환 비교 (같은 calibration 이미지 1장 기준)

| | `v01_repro3` | `v02_symmetric` |
|---|---|---|
| nvt_model.bin 크기 | 4,758,792 B | 4,758,792 B (동일) |
| Sigmoid(cls) cosine | 0.9855 | 0.9837 |
| dfl_conv(box) cosine | 0.9858 | 0.9864 |
| 탐지 매칭 (10~11개 중) | 10/10 매칭 | 10/11 매칭 (1개 누락) |

단일 이미지 기준이라 우열을 단정할 근거는 아니다(둘 다 0.98대 중반으로 비슷한 수준) —
전체 calibration/board 기준 비교가 있어야 의미 있는 결론이 된다.

---

## 3-3. `best.onnx` 전체 데이터셋 재검증 — PT vs ONNX (2026-09-09)

§3-2는 랜덤 입력 1개로 "두 ONNX가 같은 연산인가"만 확인한 것이었다. 여기서는 실제 정확도
지표(P/R/mAP)가 export로 인해 달라지는지, `tools/validate_symmetric.py`로 전체 6,076장/
53,262 인스턴스에 대해 `best.onnx`를 §3-1과 동일 조건(640×512, `conf=0.001`, `iou=0.7`)으로
재검증해 `best.pt`(§3-1) 값과 직접 비교했다. (`best_split.onnx`는 `validate_symmetric.py`가
못 읽는 형식이라 대상에서 제외 — §3-2에서 이미 `best.onnx`와 bit-exact 동일함을 증명했으므로
`best_split.onnx`의 정확도도 수학적으로 동일하다.)

```bash
uv run python3 tools/validate_symmetric.py \
  -w <weights_dir>/best.onnx -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1   # 고정 shape ONNX라 batch=1 필수
```

**(2026-09-09 정정) `standard`의 ONNX 열은 우리가 만든 export가 아니라 NAS에 실제 배포된
onnx로 교체했다.** §3-2에서 우리 export(`best.pt`→`best.onnx`)와 NAS 실제 배포 onnx
(`network/260706_BSD-F606A_Front_v01.onnx`, sha256 `a4e15dea...`, 2026-09-09 NAS 직접
대조로 지금도 동일 확인)가 **서로 다른 가중치**(126/127 파라미터 불일치)임이 드러났으므로,
"export가 정확도를 깎아먹지 않는가"를 보려면 **우리 export**를, "실제 제품에 뭐가 올라가
있는가"를 보려면 **NAS 실제 onnx**를 따로 봐야 한다 — 아래 표는 후자다.

NAS 실제 onnx는 박스/클래스가 분리되고 박스가 디코딩 전(DFL 적분까지만) 상태로 나와
`validate_symmetric.py`가 못 읽는다 — `tools/validate_symmetric_jdh.py`
(`SymmetricDetectionValidatorJdh`, `projects/src/models/yolo/detect/val/symmetric_jdh.py`의
`_decode_split_box`)로 그 분리 구조를 디코딩해서 읽었다. 표에 나온 P/R/mAP는 이 검증기도
그대로 물려받는 **ultralytics native `ap_per_class`** 계산이라(추가로 pycocotools 교차검증도
같이 돌지만 그건 별도 지표), PT/우리 export 열과 **동일 엔진 기준으로 비교 가능**하다.

```bash
uv run python3 tools/validate_symmetric_jdh.py \
  -w <NAS 실제 onnx 경로> -a configs/datasets/dashcam_front_traffic_light.yaml -d 0 \
  --data_args imgsz_y=512 --cfg_options conf=0.001 iou=0.7 batch=1
```

| | standard(PT) | standard(ONNX, **NAS 실제 배포**) | v01_repro3(PT) | v01_repro3(ONNX, 우리 export) | v02_symmetric(PT) | v02_symmetric(ONNX, 우리 export) |
|---|---|---|---|---|---|---|
| P | 0.869 | **0.847** | 0.872 | 0.871 | 0.876 | 0.875 |
| R | 0.678 | **0.646** | 0.677 | 0.677 | 0.677 | 0.677 |
| **mAP50** | 0.781 | **0.744** | 0.781 | 0.781 | 0.783 | 0.783 |
| **mAP50-95** | 0.560 | **0.484** | 0.560 | 0.560 | 0.562 | 0.562 |

(참고: `standard`의 **우리 export** `best.onnx` 값은 P/R/mAP50/mAP50-95 = 0.869/0.678/0.781/
0.560으로 PT와 동일 — §3-2에서 이미 확인한 export 무손실을 여기서도 같은 결론으로
재확인했다. `v01_repro3`/`v02_symmetric`은 NAS에 배포된 적이 없어 대응하는 "NAS 실제" 값이
없다.)

### 클래스별 (mAP50 / mAP50-95)

| 클래스 | standard(PT) | **standard(ONNX, NAS 실제)** | v01_repro3(PT→ONNX, 우리 export) | v02_symmetric(PT→ONNX, 우리 export) |
|---|---|---|---|---|
| bus | 0.812/0.665 | **0.786/0.580** | 0.812/0.663 → 0.812/0.664 | 0.818/0.669 → 0.818/0.669 |
| car | 0.890/0.723 | **0.867/0.645** | 0.891/0.724 → 0.891/0.724 | 0.892/0.724 → 0.892/0.724 |
| motorcycle | 0.764/0.482 | **0.704/0.407** | 0.765/0.483 → 0.765/0.483 | 0.765/0.485 → 0.765/0.485 |
| person | 0.762/0.499 | **0.708/0.402** | 0.762/0.500 → 0.762/0.500 | 0.761/0.503 → 0.761/0.503 |
| truck | 0.800/0.648 | **0.782/0.566** | 0.801/0.649 → 0.801/0.649 | 0.803/0.651 → 0.804/0.650 |
| traffic_light | 0.658/0.342 | **0.619/0.304** | 0.659/0.342 → 0.659/0.342 | 0.657/0.340 → 0.657/0.340 |

**결론 (정정)**: `v01_repro3`/`v02_symmetric`은 (우리끼리의) PT↔ONNX export 차이가 여전히
표시 자릿수 안에서 사실상 0이다 — 이 부분 결론은 그대로 유지. 하지만 **`standard`는 얘기가
다르다** — 우리 export(=PT와 동일)와 **NAS에 실제 배포된 onnx**를 같은 조건·같은 엔진으로
비교하면 **all 기준 ΔmAP50 -0.037, ΔmAP50-95 -0.076**로 뚜렷하게 차이 난다(6클래스 전부
일관되게 하락, motorcycle/person이 가장 크게 -0.06~-0.10). 이건 export 손실이 아니라
§3-2에서 확인한 "**애초에 다른 가중치**"라는 사실이 실제 정확도 수치로도 드러난 것이다 —
이 repo가 갖고 있는 `best.pt`는 NAS에 배포된 모델보다 (이 표준 center-crop·non-ROI 프로토콜
기준으로는) 더 높은 mAP을 내는, **분명히 다른 모델**이다.

---

## 3-4. Loss/mAP 추이 분석 — `v01_repro3`/`v02_symmetric` (2026-09-09)

`results.csv` 전체(1~50 epoch)를 뜯어 학습이 꾸준히 개선 중이었는지, 정체였는지, 추가 학습이나
epoch 상향이 도움이 될지 검토했다.

### Train loss — 40 epoch 내내 정체, epoch 41 급락은 실제 학습 개선이 아님

| epoch | v01_repro3 train/box | v02_symmetric train/box |
|---|---|---|
| 1 | 0.840 | 0.829 |
| 16 | 0.840 | 0.830 |
| 36 | 0.839 | 0.830 |
| **41** | **0.798** | **0.790** |
| 50 | 0.792 | 0.784 |

epoch 1→36까지 35 epoch 동안 train loss가 사실상 안 움직인다. epoch 41의 급락은 exp.yaml의
`close_mosaic: 10`(마지막 10 epoch, 즉 `50-10=40` 이후 mosaic 증강 비활성) 때문 — 더 쉬운
이미지로 바뀌어 loss가 떨어지는 **augmentation 스위치 아티팩트**이지, 모델이 갑자기 더 잘
학습된 게 아니다.

### Val loss — 같은 구간에서 오히려 소폭 악화

| epoch | v01_repro3 val/box | v02_symmetric val/box |
|---|---|---|
| 31 | 0.88397 | 0.87424 |
| 36 | 0.88379(최저점 근처) | 0.87415(최저점) |
| 41 | 0.88403 | 0.87441 |
| 50 | **0.88647** | **0.87711** |

train loss는 급락했는데 val loss는 오히려 상승 — mosaic을 끈 뒤 남은 10 epoch이 train 분포엔
더 잘 맞지만 val 일반화엔 도움이 안 됐다는 신호(경증 과적합 패턴).

### mAP — 중반 정점 후 하락 (§3-1에서 이미 발견한 것과 동일 원인)

- `v01_repro3`: mAP50 정점 **epoch 37 (0.77376)** → epoch 50 (0.77244), **-0.0013**
- `v02_symmetric`: mAP50 정점 **epoch 40~41 (0.78284/0.78278)** → epoch 50 (0.78122), **-0.0016**

### LR 스케줄 — epoch 50 시점 이미 거의 0

`cos_lr` + `lr0=0.0001`(원본 as-run 값, ultralytics 기본값 0.01의 1/100)이라 epoch 50의 LR은
**1.09e-7**. 마지막 5~10 epoch은 가중치가 거의 안 움직이는 상태로 진행됐다.

### 결론 (비판적 판단)

- **꾸준히 감소 vs 정체 → 정체.** train loss는 mosaic-on 구간(1~40) 내내 평평했고, epoch 41의
  급락은 실제 일반화 개선이 아니라 augmentation 난이도 변화다. val loss/mAP 기준으로는
  **epoch 37~41 근방이 사실상 최적점**이었고 나머지 epoch은 mAP을 0.001~0.002 깎아먹었다.
- **이어서 더 학습해도 괜찮은가 → 근거 없음.** LR이 이미 코사인 스케줄 끝(≈0)까지 소진됐고,
  val mAP은 이미 하락 추세에서 멈췄다. 지금 상태(LR≈0)에서 resume하면 사실상 변화가 없고,
  LR을 다시 올려 재개하면 "이어서 학습"이 아니라 별도의 새 실험이다.
  - **§1의 pretrained_weight.pt 계보 조사로 보강된 해석**: 시작 가중치 자체가 이미 이전
    DAIR 학습의 epoch 17 체크포인트였다 — 완전 초기화가 아니라 어느 정도 수렴된 모델을
    낮은 LR로 살짝 더 맞추는 상황이었을 가능성이 높고, 이는 초반 40 epoch의 train loss 정체와
    정합적이다.
- **처음부터 epoch을 높게 잡았으면 나았을까 → 근거상 아니오.** `epochs`를 늘리면 코사인 LR
  감쇠 곡선이 늘어나 초반 LR이 더 오래 유지되지만, 이미 **상대적으로 높은 LR 구간(epoch
  6~11 근방, lr≈9e-5)에서도** train loss가 거의 안 움직였다 — "덜 돌려서 못 배웠다"가 아니라
  "이 lr0=0.0001 자체가 이미 수렴에 가까운 시작점 대비 낮아서 움직임이 적다"는 설명이 데이터와
  더 잘 맞는다. `v01_ga_candidate.yaml`(미검증)이 `lr0=0.0047`(47배 높음)을 썼던 이력이
  있다 — epoch 수를 올리는 것보다 **epoch=35~40 근방 조기 종료(patience 활용) 또는 lr0을 올린
  별도 실험**이 더 근거 있는 다음 스텝이다.

---

## 3-5. 정성적 영상 비교 — GT 없음, 실제 주행 영상 (2026-09-04 / 2026-09-09)

§3~§3-4는 GT가 있는 정지 이미지셋(6,076장) 기준 정량 지표였다. 여기서는 실제 주행 영상에서
detection이 시각적으로 어떻게 나오는지 `standard`/`v01_repro3`/`v02_symmetric` 세 모델을
직접 비교했다 — **정량 지표 없이 육안 검토용**이다.

### 방법

`tools/debug/video_infer_front_standard.py`(단일 모델) / `video_infer_front_compare.py`
(3모델 동시, 나란히 합성) — 배포 전처리와 동일 조건:
- ROI crop `(320, 0, 1280, 1024)` (원본 1920×1080에서 배포와 같은 영역만 사용,
  `configs/releases/BSD_F606/Front_Truck/v01/MANIFEST.yaml` 기준)
- crop 후 640×512로 letterbox 없는 단순(squash) resize — dycrop 학습과 동일
- BGR 유지, `/255` 없음
- 표준 단일-IoU NMS(`iou_thres=0.45`), **`confidence_thres=0.25`**(§3의 golden 평가용
  `conf=0.001`과 다름 — 저신뢰 박스가 너무 많아 육안 검토에 방해되어 시각 검토 전용
  기본값을 따로 씀)
- `video_infer_front_compare.py`는 프레임을 한 번만 디코딩해 3개 세션에 동시에 넣고 바로
  옆으로 이어붙여 쓴다(decode 1회 + encode 1회) — 모델별로 따로 돌려 나중에 합치는 방식
  (decode/encode 각 4회) 대비 video I/O 왕복이 줄어 더 빠르다(추론 연산량 자체는 동일).
- `standard`는 배포 onnx(`weights/deployment_260706.onnx`, §3-3에서 NAS 실제 배포본과
  그래프 100% 일치 확인됨), `v01_repro3`/`v02_symmetric`은 각자의 `best_split.onnx`
  (§3-2에서 `best.onnx`와 bit-exact 동일 확인됨) 사용.

### 산출물

| 라운드 | 대상 모델 | 클립 구성 | 위치 |
|---|---|---|---|
| 1차(2026-09-04) | `standard` vs `v01_repro3` 각각 단일 영상(합성 없음) | 실주행 로그 14클립(INF 11개 + EVT 3개, `20260903_*_channel_0`) | `runs/video_infer/front_infer_baseline/`, `runs/video_infer/front_infer_v01_repro3/` |
| 2차(2026-09-09) | `standard`/`v01_repro3`/`v02_symmetric` 3-패널 나란히 합성 | 조건별 큐레이션 12클립(day/night × city/highway/tollgate/FVSA/sunflare/rainy) | `runs/video_infer/front_infer_compare/`(`*_compare.mp4`) |

2차 클립 원본은 `runs/video_infer/_input/new_front_video_infer/`에 있다. `front_infer_baseline/`에는
2차 클립 중 3개(day_city, day_FVSA×2)에 대한 `standard` 단독 영상도 추가로 남아있다
(`*_standard.mp4`, compare 영상 만들기 전 baseline 확인용으로 보인다).

### 상태 — **미검토**

영상 자체는(1차 14클립 + 2차 12클립) 이미 생성돼 있으나, 실제로 재생해서 정성적 소견(오탐/누락
패턴, 조건별 차이 등)을 기록한 사람은 아직 없다 — §4에 다음 확인 사항으로 남긴다.

---

## 3-6. 신규 실험 3종 추가 — `v02_dair_lr01`/`v02_symmetric_bg`/`v02_symmetric_scratch` (2026-09-14)

§1에 정의된 세 실험의 `best.pt`를 §3-1과 동일 조건(640×512, `tools/validate_symmetric.py`,
ultralytics native, `conf=0.001`, `iou=0.7`, 전체 6,076장/53,262 인스턴스)으로 재검증했다.
`v02_symmetric_bg`는 **완료된 결과**다(§1 정정 — 450 epoch 예산 중 epoch 422에서
`patience=200` 조기 종료, mAP 정점은 epoch 219 부근).

| | standard | v01_repro3 | v02_symmetric | **v02_symmetric_bg** | **v02_symmetric_scratch** | **v02_dair_lr01** |
|---|---|---|---|---|---|---|
| 시작 가중치 | pretrained | pretrained(=standard) | pretrained | pretrained | **from-scratch** | **from-scratch** |
| trainer | symmetric | symmetric | symmetric | symmetric | symmetric | **dair** |
| epochs | — | 50 | 50 | **422(조기종료, 예산 450)** | **500(완주)** | **300(완주)** |
| P | 0.869 | 0.872 | 0.876 | 0.878 | **0.881** | 0.861 |
| R | 0.678 | 0.677 | 0.677 | 0.677 | **0.686** | 0.663 |
| **mAP50** | 0.781 | 0.781 | 0.783 | 0.785 | **0.790** | 0.761 |
| **mAP50-95** | 0.560 | 0.560 | 0.562 | 0.564 | **0.566** | 0.546 |

### 클래스별 (mAP50 / mAP50-95)

| 클래스 | v02_symmetric | v02_symmetric_bg | v02_symmetric_scratch | v02_dair_lr01 |
|---|---|---|---|---|
| bus | 0.818 / 0.669 | 0.819 / 0.672 | 0.815 / 0.672 | 0.814 / 0.663 |
| car | 0.892 / 0.724 | 0.892 / 0.726 | 0.893 / 0.724 | 0.884 / 0.709 |
| motorcycle | 0.765 / 0.485 | 0.768 / 0.487 | 0.765 / 0.479 | 0.747 / 0.472 |
| person | 0.761 / 0.503 | 0.765 / 0.505 | 0.765 / 0.506 | 0.745 / 0.488 |
| truck | 0.803 / 0.651 | 0.802 / 0.650 | 0.805 / 0.651 | 0.799 / 0.642 |
| **traffic_light** | 0.657 / 0.340 | **0.662 / 0.346** | **0.696 / 0.363** | **0.580 / 0.301** |

### 관찰

- **`v02_symmetric_scratch`(from-scratch, 500 epoch)가 지금까지의 6개 모델 중 all/클래스별
  거의 전 항목에서 최고 mAP** — 특히 `traffic_light`에서 0.696/0.363으로 기존 최고
  (`v02_symmetric` 0.657/0.340) 대비 크게 앞선다. pretrained 가중치에서 시작한 계열들보다
  from-scratch가 더 나은 결과를 낸 것은 **직관에 반하는 결과라 비판적으로 볼 필요가 있다**
  — (1) 500 epoch(다른 모델의 10배)이라는 훨씬 큰 학습 예산 차이가 가장 유력한 원인
    후보이지 "from-scratch가 근본적으로 더 낫다"는 결론은 아니다, (2) pretrained 계열(DAIR
    lineage, §1)의 원 데이터/태스크가 지금 타깃(6클래스, traffic_light 포함)과 달랐다는
    점(§1 `standard` 항목)이 오히려 전이 효과를 제한했을 가능성, (3) 두 요인을 분리하려면
    같은 epoch 예산으로 맞춘 대조 실험이 필요하다 — 아직 없음.
- **`v02_symmetric_bg`(배경 이미지 추가)는 `v02_symmetric` 대비 전 클래스 소폭 개선**
  (mAP50 +0.002, mAP50-95 +0.002) — **이건 완료된 결과다**(patience=200으로 epoch 422에서
  정상 조기 종료, 정점은 epoch 219 부근). 배경 이미지 5,769장 추가가 다른 조건 변경 없이
  이 정도 개선을 냈다는 뜻 — 세 신규 실험 중 유일하게 기존 3개 실험(50 epoch)과 epoch 예산
  차이가 크지 않은(사실상 epoch 219 근방에서 이미 수렴) 비교라 가장 신뢰도 높은 결과다.
- **`v02_dair_lr01`(DAIR, from-scratch, 300 epoch)은 세 신규 실험 중 유일하게 기존
  baseline(`standard` 0.781/0.560)보다도 낮다** — all mAP50 0.761(-0.020), 특히
  `traffic_light`(0.580, 기존 최저였던 `standard`의 0.657보다도 더 낮음)와 `person`
  (0.745/0.488, 전체 중 최저)에서 두드러진다. DAIR의 dual-forward consistency loss가
  이 데이터·클래스 구성에서 symmetric 단독 학습보다 수렴이 더디거나 어렵다는 뜻일 수
  있으나, epoch 예산(300 vs scratch symmetric의 500)이 다르다는 점도 함께 고려해야 한다 —
  같은 epoch에서 다시 비교하기 전까지는 "DAIR이 더 나쁘다"고 단정하기는 이르다.
- **공통 한계**: `v02_dair_lr01`(300, 완주)·`v02_symmetric_scratch`(500, 완주)는 실제
  정점이 어디였는지 아직 안 봤다(§3-4 방식으로 확인 필요) — 기존 3개 실험(50 epoch)이
  중반(epoch 30~40)에 이미 정점을 찍고 정체했던 패턴을 감안하면, 이 둘도 전체 예산을 다
  쓰지 않고 훨씬 이른 시점에 이미 수렴했을 가능성이 있다. `v02_symmetric_bg`는 예외 —
  patience=200이 실제로 발동해(epoch 219 정점 → 422 종료) 정점 위치가 이미 확인됐다. 즉
  세 신규 실험의 epoch 예산(300/422/500) 자체가 "학습 방식의 효과"와 뒤섞인 confound라는
  점은 여전히 남아있지만, `v02_symmetric_bg`만은 정점 위치가 확인된 유효한 비교다.

### `v02_symmetric_bg`/`v02_symmetric_scratch` NPU 변환 (2026-09-14)

§3-2와 동일 절차(calibration/설정은 `front_truck_v01_reference_onnx_local_reconvert_20260814`
템플릿 재사용, `best_split.onnx`만 교체)로 두 모델 전부 compile→simulate→visualize 완료.
`best_split.onnx`가 이 시점까지 없어서 새로 export했다 — export 순서 함정(§3-2)이 이번에도
재발해(분리 export가 기존 병합 `best.onnx`를 덮어씀) `best.onnx`를 다시 만들었다.

| | `v02_symmetric_bg` | `v02_symmetric_scratch` |
|---|---|---|
| `novatek/convert/` 경로 | `front_truck_v02_symmetric_bg_20260914/` | `front_truck_v02_symmetric_scratch_20260914/` |
| nvt_model.bin | 4,758,792 B (동일) | 4,758,792 B (동일) |
| Sigmoid(cls) cosine | 0.9834 | 0.9778 |
| dfl_conv(box) cosine | 0.9870 | 0.9858 |
| Golden vs NPU Sim 탐지 수 | 11 vs 11 | 11 vs 11 |

### 다섯 모델 NPU 변환 종합 비교 (같은 calibration 이미지 1장 기준)

| | `v01_repro3` | `v02_symmetric` | `v02_symmetric_bg` | `v02_symmetric_scratch` |
|---|---|---|---|---|
| Sigmoid(cls) cosine | 0.9855 | 0.9837 | 0.9834 | **0.9778(최저)** |
| dfl_conv(box) cosine | 0.9858 | 0.9864 | 0.9870 | 0.9858 |

**관찰**:
- `v02_symmetric_bg`는 탐지 11/11 전부 매칭됐지만 **traffic_light confidence가 0.535→0.266로
  절반 이하 하락** — 다른 클래스는 대체로 골든과 근접(±0.05 이내)한데 traffic_light만 유독
  크게 흔들린다. 기존 `v01_repro3`/`v02_symmetric`(§3-2)에서도 traffic_light이 가장 낮은
  cosine/confidence 안정성을 보였던 패턴과 일치.
- `v02_symmetric_scratch`는 cls cosine이 다섯 모델 중 **가장 낮다(0.9778)** — 실제 탐지에서도
  드러난다: golden엔 있는 `traffic_light 0.315` 탐지가 NPU Sim 목록에서 사라지고, 대신 이미
  `truck 0.500`으로 잡힌 것과 **동일 좌표`[6, 305, 68, 358]`가 `car 0.281`로 중복 분류**돼
  나타난다 — 같은 박스가 두 클래스로 동시에 임계값을 넘는, 다른 네 모델에서는 안 보이던
  현상. from-scratch 학습이 §3-6에서는(PT 기준) 최고 mAP을 냈지만, **양자화 이후에는 가장
  불안정한 모습을 보인다는 점은 주목할 만하다** — PT 정확도와 양자화 강건성이 반드시 같은
  방향으로 가지 않는다는 것을 보여주는 사례.
- 다섯 모델 모두 cosine 자체는 0.977~0.987 범위로 큰 차이는 아니라, 이 관찰 자체는 calibration
  이미지 1장 기준 정성적 신호였다 — **전체 6,076장 보드 실측으로 확정됨, 아래 절 참조**:
  `v02_symmetric_scratch`의 mAP50-95 relative loss는 **−25.3%**로, Front `standard`(+10.3%)는
  물론 Rear `v01_symmetric`(−7.1%)보다도 훨씬 크다.

### 왜 `v02_symmetric_scratch`가 양자화에 더 취약한가 — 가중치 분포 직접 확인 (2026-09-14)

추측이 아니라 실측했다. `best.pt`의 state_dict에서 stem(`0.conv`, BN 융합)·box head 마지막
conv(`22.cv2.0.2`, BN 없음)·cls head 마지막 conv(`22.cv3.0.2`, BN 없음)의 가중치 분포를
직접 비교:

| | `v01_repro3`(fine-tuned) | `v02_symmetric`(fine-tuned) | **`v02_symmetric_scratch`(from-scratch)** |
|---|---|---|---|
| stem abs_max | 0.758 | 0.757 | **1.370 (+81%)** |
| stem std | 0.293 | 0.293 | **0.413 (+41%)** |
| cls head abs_max | 0.822 | 0.821 | **0.955 (+16%)** |
| box head abs_max | 0.606 | 0.606 | **0.802 (+32%)** |

fine-tuned 두 모델은(같은 lineage라 예상대로) 사실상 동일한 분포다. `v02_symmetric_scratch`는
전 레이어에서 일관되게 더 넓다 — 특히 **가장 먼저 양자화되는 stem 레이어의 범위가 거의
2배**다.

**해석**: NPU INT8 양자화는 레이어의 값 범위(min~max)를 256단계에 고정 매핑한다 — 범위가
넓을수록 한 단계가 나타내는 실제 값 폭(step size)이 커져 정밀도 손실이 커진다.
`v02_symmetric_scratch`는 stem부터 이미 손실 폭이 크게 시작해 뒤로 누적 전파됐을 가능성이
높다 — §3-6에서 관찰한 최저 cls cosine(0.9778)·traffic_light 탐지 소실/중복 분류 현상과
정합적이다.

**왜 분포가 넓어졌는가(추정)**: fine-tuned 계열은 이미 여러 세대 학습을 거쳐 정리된 분포에서
시작해 `lr0=0.0001`로 50 epoch만 살짝 다듬은 반면, `v02_symmetric_scratch`는 랜덤 초기화에서
`lr0=0.01`로 500 epoch 전체를 학습했다 — 가중치를 억제할 사전 제약 없이 오래 학습하면
자연스럽게 분포가 넓어지는 경향과 부합한다(`weight_decay=0.0005`는 둘 다 동일해 이 차이를
막지 못했다).

**미해결 — 다음에 시도해볼 수 있는 것**: 지금 쓴 `gen_config.txt`는 v01 릴리즈 설정을 그대로
재사용한 것이라 `precision/cle_en=0`(Cross-Layer Equalization 꺼짐), `spec_demarc_tensors`
(특정 텐서 고정밀도 보호)도 주석 처리로 비활성이다. CLE는 정확히 이런 레이어별 스케일
불균형을 완화하는 기법이라 `v02_symmetric_scratch`에 켜보면 도움이 될 가능성이 있다 —
Rear_Truck 쪽의 `docs/debug/20260819_front_golden_reference_for_rear_1pct.md`(H1/H2/H3
tensor 보호 실험)를 참고할 수 있다. 아직 시도 안 함.

> **⚠️ 정정 (2026-09-14, 아래 `v02_symmetric_bg` 보드 실측 절 참고)**: 이 절의 가설("넓은
> weight dynamic range → 양자화 취약")은 파인튜닝 모델인 `v02_symmetric_bg`가 weight 분포는
> 좁으면서도(`v01_repro3`/`v02_symmetric`과 거의 동일) board 양자화 손실은 scratch와 동급
> (−25.56% vs −25.28%)으로 나오면서 **반박됐다**. 위 표의 상대 비교(fine-tuned가 scratch보다
> 좁다) 자체는 사실이지만, 그 폭이 board mAP 손실의 원인이라는 인과 해석은 성립하지 않는다.

### `v02_symmetric_scratch` 실제 보드 정량 검증 — 전체 6,076장 mAP (2026-09-14)

위 가중치 분포 분석은 calibration 이미지 1장 기준 cosine 유사도(§3-6 NPU 변환 절)에 근거한
추정이었다. 사용자가 확보한 실제 보드(NPU) 추론 raw dump
(`runs/_eval_scratch/v02_FRONT_symmetric_scratch_output.json`, 6,076장 전체, 147,890건
detection)로 실제 mAP 손실을 직접 측정했다.

**방법론**: `docs/debug/20260819_front_golden_reference_for_rear_1pct.md`(§8-10, `standard`/v01
릴리즈의 Float-vs-Board mAP50-95 재현)와 동일한 파이프라인 재사용 — 동일 GT(Validation
6,076장, ROI `[320,0,1280,1024]`로 clip), 동일 dual-threshold NMS(`basic_postproc_v8_dual_nms`,
`min_overlap=0.6`/`union_overlap=0.3`, REAR NT98530 postprocess 포팅), 동일 evaluator
(`projects.metrics.evaluator.Evaluator`, pycocotools 기반). Float ONNX는 `best_split.onnx`
(box: DFL 적분 후 dist2bbox 디코드 전 — 보드와 동일 구조)를 onnxruntime으로 직접 추론.
스크립트: `runs/debug/front_golden_reference_1pct/v02_symmetric_scratch_quant_loss/run_eval.py`.

**주의 — 보드 dump 좌표계가 `standard`와 다름**: `standard`(v01) 보드 dump는 full-frame 좌표였지만
(§18 "IPU-test ... bbox가 full-frame 좌표임을 확인"), 이번 `v02_symmetric_scratch` dump는
**ROI crop-relative 좌표**(원점이 crop 좌상단 `(320,0)`)였다. 같은 이미지의 GT와 대조해 확인함:

| 항목 | GT(full-frame) | GT − (320,0) | 보드 dump 원본 | 판정 |
|---|---|---|---|---|
| car | [851, 576, 1501, 898] | [531, 576, 1181, 898] | [536, 578, 1175, 900] | 일치(수 px 오차) |
| car(좌측 절반만 ROI 안) | [8, 580, 679, 913] | [0(clip), 580, 359, 913] | [0, 580, 357, 913] | 일치 |
| truck | [1675, 414, 1861, 762] | [1355, 414, 1541, 762] | [1362, 427, 1540, 753] | 일치 |

처음 full-frame 좌표 그대로 GT와 매칭했을 때 Board mAP50 = 0.0012(사실상 0) — 좌표계 불일치
때문이었고, crop offset `(320,0)`을 더해 보정한 뒤 정상적인 값이 나왔다. 이 좌표계 차이가
왜 생겼는지(보드 dump 스크립트 차이인지, 이번 테스트만의 설정인지)는 확인하지 않았다 —
**향후 다른 모델의 신규 보드 dump를 받으면 좌표계를 먼저 GT와 대조 확인할 것.**

**결과**:

| | mAP50 | mAP50-95 |
|---|---|---|
| Float ONNX (`best_split.onnx`, dual-NMS) | 0.7565 | 0.5382 |
| Board(NPU, 실측 dump) | 0.6282 | 0.4021 |
| absolute loss | −0.128 | **−0.136** |
| relative loss | −16.9% | **−25.28%** |

클래스별 mAP50-95 (Float → Board):

| class | Float | Board | Δ |
|---|---|---|---|
| bus | 0.640 | 0.478 | −0.162 |
| car | 0.670 | 0.520 | −0.150 |
| motorcycle | 0.460 | 0.341 | −0.119 |
| person | 0.488 | 0.349 | −0.139 |
| truck | 0.616 | 0.435 | −0.181 |
| traffic_light | 0.356 | 0.290 | −0.066 |

**판정**: 전 클래스 예외 없이 손실 — 추측이 아니라 확인된 사실이다. 그리고 이 손실 폭은 지금까지
이 프로젝트에서 관측된 어떤 사례보다 크다:

| 모델 | mAP50-95 relative loss |
|---|---|
| Front `standard`(v01 release, 2026-08-20 재현) | **+10.3%** (손실 없음, 오히려 개선) |
| Rear `v01_symmetric`(acceptance target ≤1%, 미달) | −7.1% |
| **Front `v02_symmetric_scratch`** | **−25.3%** |

`v02_symmetric_scratch`는 Float 단독 정확도로는 §3-6에서 가장 우수한 모델이었지만, 보드
배포 기준으로는 지금까지 확인된 것 중 양자화 손실이 가장 크다. **주의**: 가중치 분포
가설("from-scratch의 넓은 dynamic range가 원인")이 이 수치만으로 확정된 것은 아니다 —
바로 아래 절(`v02_symmetric_bg` 보드 실측)에서 이 가설이 반박된다. `v02_symmetric_scratch`를
그대로 배포 후보로 쓰는 것은 권장하지 않는다.

산출물: `runs/debug/front_golden_reference_1pct/v02_symmetric_scratch_quant_loss/`
(`run_eval.py`, `board_dump_merged.json`, `gt_coco_clipped.json`, `board_pred_coco.json`,
`float_pred_coco.json`, `summary.json`).

### `v02_symmetric_bg` 실제 보드 정량 검증 — 가중치 분포 가설 반박 (2026-09-14)

사용자가 `v02_symmetric_bg`의 실제 보드 dump도 확보했다
(`runs/_eval_scratch/v02_FRONT_symmetric_bg_output.json` — 원래 파일명은
`v02_symmetric_bg_output.json`이었으나, `runs/_eval_scratch/board_dumps/v02_symmetric_bg/`의
**REAR_Truck** `v02_symmetric_bg`(동명이인, `docs/debug/20260914_rear_truck_v02_symmetric_bg_
background_augment_eval.md` 참고)와 헷갈릴 수 있어 사용자가 FRONT로 rename함. 6,076장 중
6,075장 커버, 142,749건
detection — 1장은 보드에서 detection 자체가 없었음). 위와 완전히 동일한 파이프라인
(`run_eval.py`를 ONNX/보드 dump 경로만 바꿔 재사용 — 좌표계도 scratch와 동일한 crop-relative임을
첫 detection으로 재확인: `[0,578,360,910]` score 0.975 vs scratch의 거의 동일한
`[0,580,357,913]` score 0.958, 같은 GT 차량에 대응)로 측정했다.

**결과**:

| | mAP50 | mAP50-95 |
|---|---|---|
| Float ONNX (`best_split.onnx`, dual-NMS) | 0.7489 | 0.5351 |
| Board(NPU, 실측 dump) | 0.6157 | 0.3983 |
| relative loss | −17.8% | **−25.56%** |

`v02_symmetric_scratch`(−25.28%)와 사실상 동일한 손실 폭이다. 그런데 `v02_symmetric_bg`는
**from-scratch가 아니라 `v01_repro3`/`v02_symmetric`과 같은 계보의 파인튜닝 모델**이다
(`weight_path: configs/releases/.../v01/weight/pretrained_weight.pt`, §1 참고) — 배경 이미지
추가 + epochs 600(patience 200으로 epoch 422 조기종료)만 다르다.

**가중치 분포 재확인** (네 모델 `best.pt`를 동일 코드로 한 번에 재계산 — 이전 계산과 절대값
스케일은 다르게 나오지만 네 모델 간 상대 비교는 유효):

| | `v01_repro3` | `v02_symmetric` | **`v02_symmetric_bg`** | `v02_symmetric_scratch` |
|---|---|---|---|---|
| stem abs_max(BN 융합) | 0.106 | 0.105 | **0.108** | 0.212 |
| stem std | 0.033 | 0.033 | **0.035** | 0.040 |
| cls head abs_max | 0.822 | 0.821 | **0.790** | 0.955 |
| box head abs_max | 0.605 | 0.606 | **0.606** | 0.802 |

`v02_symmetric_bg`의 가중치 분포는 `v01_repro3`/`v02_symmetric`과 거의 동일하다(cls head는
오히려 더 좁다) — `v02_symmetric_scratch`처럼 넓지 않다. **즉 가중치 dynamic range가 넓지
않은 모델도 board에서 scratch와 맞먹는 양자화 손실(−25.56%)을 낸다** — §3-6 위쪽 절에서 세운
"from-scratch의 넓은 weight dynamic range가 양자화 취약성의 원인"이라는 가설은 **반박된다**.
weight 분포만으로는 이 두 모델의 손실을 설명할 수 없다.

**두 모델(`bg`, `scratch`)이 공유하는 것 — 재조사 대상**:
- 둘 다 `standard`/`v01_repro3`/`v02_symmetric`(50 epoch 수준)보다 **훨씬 긴 학습**
  (`bg`=422 epoch, `scratch`=500 epoch) — weight가 아니라 **activation 분포**가 학습이
  길어지며 calibration 500장과 어긋나게 drift했을 가능성.
- 둘 다 2026-09-14에 새로 학습된 모델 — 같은 `gen_config.txt`(v01 release calibration 설정
  그대로 재사용, `cle_en=0`, `spec_demarc_tensors` 비활성)를 그대로 썼다는 점은 이미 알려진
  사실이지만, calibration 500장 자체가 이 두 모델의 실제 activation 분포를 얼마나 잘 대표하는지는
  확인한 적이 없다.
- `v02_symmetric_bg`만의 추가 변수: 배경(객체 없음) 이미지 학습 데이터 추가 — 배경 이미지가
  많아지면 학습된 feature map의 활성화 패턴 자체가 바뀔 수 있고, 이것이 objects가 있는
  calibration 이미지만으로 캘리브레이션할 때 대표성 문제를 만들 수 있다(추정, 미검증).

**결론**: 원인은 아직 미해결이다. weight 분포는 기각되었고, activation 분포/calibration 대표성
쪽으로 무게가 옮겨갔지만 직접 측정한 적은 없다 — layer별 activation range나 simulator의
per-layer cosine(§3-6 NPU 변환 절, 1장 기준)을 다시 볼 필요가 있다.

산출물: `runs/debug/front_golden_reference_1pct/v02_symmetric_bg_quant_loss/`
(`run_eval.py`, `board_dump_merged.json`, `gt_coco_clipped.json`, `board_pred_coco.json`,
`float_pred_coco.json`, `summary.json`).

## 3-7. 최종 정리표 — ONNX(Float) vs Board(NPU), 5개 모델 (2026-09-14)

5개 모델을 **동일 엔진**(ultralytics `ap_per_class`, ROI-crop `[320,0,1280,1024]` +
dual-threshold NMS 후처리, pred conf≥0.25)으로 통일해 재계산했다. `v02_symmetric`은 Float
ONNX만 새로 계산했고(같은 dual-NMS ROI-crop 조건으로 `best_split.onnx` 추론, §3-6과 동일
파이프라인) Board는 실제 보드 dump가 없어 N/A다. `v01_repro3`는 사용자가 보드 dump를 두 번
확보해 아래처럼 갱신됐다:

- **1차 dump**(`v01_repro3_FRONT_output.json`, 6,076장 중 **6,075장**만 커버, **87,324건**
  detection) — **2026-09-14 오후, 이후 아래 2차로 대체됨(수치는 역사적 기록으로만 유지)**.
- **2차 dump**(`new_v01_repro3_FRONT_output.json`, **6,076장 전체 커버**, **154,570건**
  detection — 1차의 거의 2배) — **현재 채택된 값**. 1차보다 훨씬 완전한 실행으로 보이며,
  1차가 왜 detection이 적고 1장을 놓쳤는지는 **`확인 필요`**(재실행 사유는 사용자만 알고
  있음, 이 세션에서는 원인을 조사하지 않음).

두 경우 모두 `runs/debug/front_golden_reference_1pct/v01_repro3_quant_loss/`에서 병합/평가했고
(1차 산출물은 `*_old_20260914_run1.json`으로 백업), scratch/bg와 동일한 crop-relative bbox
좌표계임을 GT 대조로 재확인했다. 스크립트: `runs/debug/front_golden_reference_1pct/
{float_only.py, final_summary.py}`.

| 모델 | 구분 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline(standard) | ONNX | 0.861 | 0.654 | 0.782 | 0.558 |
| | Board | 0.874 | 0.667 | 0.796 | **0.614** |
| v01_repro3 | ONNX | 0.865 | 0.688 | 0.806 | 0.630 |
| | Board(2차, 채택) | 0.787 | 0.598 | 0.697 | **0.499** |
| | ~~Board(1차, 대체됨)~~ | ~~0.831~~ | ~~0.516~~ | ~~0.672~~ | ~~0.441~~ |
| v02_symmetric | ONNX | 0.864 | 0.691 | 0.807 | 0.632 |
| | Board | – | – | – | N/A(보드 dump 없음) |
| v02_symmetric_bg | ONNX | 0.866 | 0.691 | 0.808 | 0.634 |
| | Board | 0.795 | 0.603 | 0.700 | **0.508** |
| v02_symmetric_scratch | ONNX | 0.865 | 0.702 | 0.815 | 0.635 |
| | Board | 0.794 | 0.620 | 0.711 | **0.510** |

**ONNX→Board 변화(mAP50-95)**: baseline 0.558→0.614(**+10.0%**, 손실 없음) / `v01_repro3`
0.630→0.499(**−20.8%**, 2차 dump 기준) / `v02_symmetric_bg` 0.634→0.508(**−19.9%**) /
`v02_symmetric_scratch` 0.635→0.510(**−19.7%**) — **2차 dump로 v01_repro3가 더 이상 이상치가
아니라 bg/scratch와 거의 같은 수준으로 정리됐다.**

**주의 — 절대값이 §3-6의 pycocotools 기반 quant-loss 수치와 다르다**: §3-6(`v02_symmetric_scratch`
−25.3%, `v02_symmetric_bg`−25.6%, `v01_repro3` **−26.4%**(2차 dump 기준, pycocotools),
baseline +10.3%)는 `projects.metrics.evaluator.Evaluator`(pycocotools `COCOeval` 기반)로
계산한 반면, 이 표는 ultralytics 자체 `ap_per_class`로 계산했다 — 두 엔진 사이에 절대 수치
차이가 있다는 것은 이미 알려진 사실이다(`docs/debug/20260824_v01_symmetric_eval_gap.md`).
**방향과 상대적 크기 순서(baseline만 견고, 나머지 셋은 서로 비슷하게 큰 손실)는 두 엔진
모두에서 일치한다.**

**Float ONNX 단독 순위**(mAP50-95): scratch(0.635) > bg(0.634) ≈ v02_symmetric(0.632) >
v01_repro3(0.630) > baseline(0.558) — 최근 재학습들이 baseline보다 Float 정확도는 명확히
높다. 하지만 실제 배포 기준(Board)에서 안전한 것은 baseline뿐이다.

### 재해석 — 이 저장소 학습 vs NAS 실배포 (v01_repro3 2차 보드 dump로 정리, 2026-09-14)

2차(완전한) dump로 재평가한 결과 `v01_repro3`는 더 이상 유독 나쁜 이상치가 아니다 — 오히려
결론이 더 단순하고 일관되게 정리된다:

- **weight 분포 가설(§3-6)은 여전히 반박된다.** `v01_repro3`(narrow weight, `v02_symmetric`과
  거의 동일 분포), `v02_symmetric_bg`(narrow, 파인튜닝), `v02_symmetric_scratch`(wide,
  from-scratch) — **셋의 weight dynamic range가 서로 크게 다른데도 Board 손실은 거의 동일한
  −20~26% 대역**에 모인다. weight 분포 폭이 손실 크기를 결정하지 않는다는 근거가 오히려
  강해졌다.
- **가장 단순하고 일관된 설명**: **이 저장소에서 직접 학습한 모델은 전부**(`v01_repro3`/
  `v02_symmetric_bg`/`v02_symmetric_scratch` — 레시피가 서로 다른 세 갈래: 짧은 재현 파인튜닝,
  긴 파인튜닝+배경데이터, from-scratch) **비슷한 크기의 큰 양자화 손실**을 보이고, 유일하게
  견고한 것은 **이 저장소 밖에서 만들어진 NAS 실배포 바이너리**(`standard`, +10%)뿐이다.
  `standard`의 정확한 학습 레시피/환경은 이미 §1에서 추적 불가능하다고 결론 낸 바 있다(NAS
  release ONNX ≠ 로컬 재현 가능한 어떤 `.pt`와도 일치하지 않음) — 이 저장소의 학습 파이프라인
  (프레임워크 버전, augmentation, 또는 calibration 대표성 등 무엇인지는 아직 모름)이 원본이
  가진 양자화 강건성의 핵심 요인을 재현하지 못하고 있을 가능성이 여전히 가장 유력하다.
- **미해결**: activation 분포/calibration 대표성 가설(§3-6)은 아직 직접 검증되지 않았다 —
  `standard`(NAS 실배포) vs `v01_repro3`(로컬 재현)의 layer별 activation range/simulator
  cosine 직접 대조가 다음으로 가장 저비용, 고정보량 실험이라는 판단은 유지된다.

산출물: `runs/debug/front_golden_reference_1pct/v01_repro3_quant_loss/`
(`run_eval.py`, `board_dump_merged.json`(2차, 채택), `gt_coco_clipped.json`, `board_pred_coco.json`
(2차), `float_pred_coco.json`, `summary.json`(2차) + `*_old_20260914_run1.json`(1차 백업)),
`final_5model_summary.json`(갱신).

산출물: `runs/debug/front_golden_reference_1pct/final_5model_summary.json`,
`{v01_repro3,v02_symmetric}_float_only/{gt_coco_clipped,float_pred_coco}.json`.

---

## 4. 알려진 한계 / 다음 확인 사항

- ~~`v01_repro3`/`v02_symmetric` 공식 재검증~~ — **해소 (2026-09-09)**. §3 표가 세 실험 모두
  `best.pt` 기준 동일 방법 재검증 값이다.
- `standard` 자체 학습 곡선은 복원 불가(`results.csv` DRM) — baseline 행은 재학습이 아니라
  이 repo에서 `best.pt`를 별도로 재평가해 만든 값이다(단, 그 재평가 자체는 §3의 `v01_repro3`/
  `v02_symmetric`과 동일한 도구·설정으로 이미 이뤄져 있어 비교에는 문제 없다).
- ~~§3 "재학습 효과 vs 재평가 효과" 분리~~ — **해소 (2026-09-09, §3-1)**. 640×512로 통일해
  검증한 결과, `v02_symmetric`의 실질적 재학습 효과는 +0.002 mAP50/+0.002 mAP50-95 수준으로
  작고, §3에서 본 개선의 대부분은 검증 해상도 차이(480→512)에서 온 것이었다.
- ~~`v02_symmetric`의 `weight_path` sha256 확인~~ — **사실상 해소 (2026-09-09, §1)**.
  `standard/pretrained_weight.pt`(2026-08-13에 `configs/releases/.../v01`=NAS 번들 경로에서
  독립적으로 복사된 사본)의 sha256이 **현재 NAS 원본과 정확히 일치**함을 직접 대조로 확인했다
  — 즉 이 NAS 파일은 최소 2026-08-13부터 지금까지 바뀌지 않았다. `v02_symmetric`의
  `weight_path`도 같은 NAS 번들 경로를 가리키므로, 학습 당시 로컬 임시 사본 자체와의 byte
  대조는 여전히 불가능하지만(이미 삭제됨), 세 실험 모두 같은 불변 NAS 정본에서 나왔다는
  확신은 이 교차 확인으로 충분히 높아졌다.
- `standard`의 배포 지표(`MANIFEST` onnx mAP 0.6965/board 0.704)와 §3 PyTorch val 값은
  파이프라인이 달라(전처리·후처리·평가 엔진) 직접 비교 대상이 아니다 — 이미
  `standard/validation.md`가 그 격차를 규명했다(§3의 값들은 어디까지나 PyTorch val 계열
  내에서만 비교할 것).
- ~~ONNX 변환~~ — **해소 (2026-09-09, §3-2)**. 세 `best.pt` 모두 `best.onnx`(병합)/
  `best_split.onnx`(분리) 확보. **미해결로 남는 것**: NPU 변환·시뮬레이션(`novatek/convert/
  run_pipeline.sh`)은 아직 아무 것도 실행하지 않았다 — `standard`의 2026-08-14 조사는 이번
  `best_split.onnx`가 아니라 NAS reference ONNX 기준이었다는 점에 유의(§3-2 "다음 단계" 참고).
- **정성적 영상 비교(§3-5) 육안 검토 — 미완료.** 1차(baseline vs v01_repro3, 14클립)·2차
  (3모델 side-by-side, 조건별 12클립) 영상은 이미 생성돼 있으나, 재생해서 오탐/누락 패턴이나
  조건별(야간·역광·우천·톨게이트 등) 차이를 정리한 소견은 아직 없다. 다음 세션에서 검토 후
  이 문서에 추가할 것.
- `v01_repro3`/`v02_symmetric`의 NPU 변환(§3-2)은 calibration 이미지 **1장** 기준
  compile+simulate+visualize까지만 완료됐고, **board 실측이나 전체 데이터셋 기준
  quantization loss(mAP)는 아직 없다** — Rear_Truck 계열에서 확립된 방법(board dump →
  `merge_board_dump.py` → `tools/debug/ultralytics_style_ap.py`, `docs/debug/20260902_
  rear_truck_release_candidate_selection.md` 및 후속 debug 문서 참고)을 그대로 적용할 수
  있다.
- ~~`v02_symmetric_bg` 학습 완료 여부~~ — **해소 (2026-09-14, 사용자 확인)**. 진행 중이
  아니라 `patience=200`으로 epoch 422에서 정상 조기 종료(로그: `"422 epochs completed in
  54.094 hours."`), mAP 정점은 epoch 219 부근 — §3-6 값은 완료된 최종 결과다.
- **(2026-09-14 추가) `v02_dair_lr01`/`v02_symmetric_scratch`의 epoch 예산 불일치** —
  300(완주)/500(완주) vs 기존 3개 실험의 50이라, §3-6 관찰의 "from-scratch가 더 낫다"/
  "DAIR이 더 나쁘다" 결론 모두 epoch 예산 차이와 분리되지 않았다. 이 둘은 patience 조기
  종료가 발동하지 않아(전체 예산 소진) 실제 정점 위치를 아직 모른다 — §3-4 방식으로 각
  loss/mAP 곡선을 봐서 확인 필요. (`v02_symmetric_bg`는 patience가 발동해 정점 위치
  epoch 219로 이미 확인됨 — 위 항목 참고.)
- ~~`v02_symmetric_bg`/`v02_symmetric_scratch` ONNX·NPU 변환~~ — **해소 (2026-09-14,
  §3-6)**. 둘 다 compile→simulate→visualize 완료. `v02_symmetric_scratch`는 cls cosine이
  다섯 모델 중 최저(0.9778)이고 golden에 있던 traffic_light 탐지가 사라지는 대신 다른 박스가
  중복 분류되는 현상이 관찰됐다.
- ~~`v02_symmetric_scratch`/`v02_symmetric_bg`/`v01_repro3` board 실측 mAP~~ — **해소
  (2026-09-14, §3-6/§3-7 "실제 보드 정량 검증"/"재해석" 절들)**. 셋 다 사용자가 확보한 실제
  보드 dump(6,076장 전체)로 확인 — pycocotools 엔진 기준 v01_repro3 **−26.4%**(2차, 완전한
  dump로 갱신됨 — 1차 dump는 6,075/6,076장·87,324건만 커버해 −37.8%로 과대평가됐었음, §3-7
  참고), scratch −25.3%, bg −25.56% — 셋 다 비슷한 수준. Front `standard`(NAS 실배포, +10.3%)·
  Rear `v01_symmetric`(−7.1%) 모두보다 나쁘다. `v02_symmetric`만 아직 보드 dump 없음.
- **(2026-09-14 추가, 재오픈) 양자화 취약성의 근본 원인 — 미해결.** 가중치 분포 가설
  (from-scratch의 넓은 dynamic range)은 weight 분포가 서로 다른 세 모델(narrow인
  v01_repro3/bg, wide인 scratch)이 전부 비슷한 손실을 보이면서 반박됐다. 현재 가장 유력한
  설명(§3-7 "재해석" 절): **이 저장소에서 직접 학습한 모델은 전부(v01_repro3/bg/scratch,
  레시피가 서로 다른 세 갈래) 비슷하게 큰 손실을, 저장소 밖에서 만들어진 NAS 실배포
  바이너리(standard)만 견고함을 보인다** — `standard`의 정확한 학습 레시피/환경 자체가 이미
  추적 불가능하다는 게 §1의 기존 결론과 맞물린다. 다음 실험: `standard`(NAS 실배포) vs
  `v01_repro3`(로컬 재현)의 layer별 activation
  range/simulator cosine을 직접 대조 — 아직 하지 않았다.
- `v02_dair_lr01`은 아직 ONNX/NPU 변환·영상 정성 비교(§3-2/§3-5) 어느 것도 하지 않았다.

## 5. 향후 실험 추가 규칙

새 Front_Truck v01 계열 실험이 추가되면:
1. §1에 실험 정의(경로/정체/`weight_path`/`data`/해상도/epochs) 추가
2. §2 계보 다이어그램에 위치 추가
3. §3 표에 행 추가 — 열 순서(해상도/출처/P/R/mAP50/mAP50-95) 유지, 값의 출처(학습 내장 val
   vs `validate_symmetric.py` 공식 재검증)를 반드시 명시
