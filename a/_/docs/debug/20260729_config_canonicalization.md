# 설정 정본화 — 확인 필요 사항 (2026-07-29 KST)

`configs/` 를 기준으로 BSD_F606 카메라별 정본을 만드는 작업에서 나온 미결 항목을 모았다.
릴리스 8개를 기록했고 그중 7개에 정본(`exp.yaml`)이 있다. 각 항목은 해당 `MANIFEST.yaml` 의
`issues` 에도 같은 id 로 들어가 있다. 이 문서는 그 전체 목록 + 담당자 확인이 필요한 질문이다.

작업 산출물:

```
configs/datasets/          정제 3 + legacy/ 17
configs/experiments/       BSD_F606 3 (미출하 v02 · GA 후보) + legacy/ 26
configs/releases/          릴리스 8개 — exp.yaml 7 + MANIFEST 8
                           (+ args_dump 2, best_hyperparameters 1)
configs/archive/           exp_yaml 165 + data_yaml 22
```

번들 산출물(가중치·ONNX·NPU 바이너리·analysis)은 repo 에 두지 않는다 — 각 MANIFEST 의
`nas_source` 에서 가져온다. `configs/releases` 는 172 KB, repo 전체 14 MB.

디렉토리 키는 NAS(`/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/`)를 미러링한다
— 제품 `BSD_F606`, 카메라 `*_Truck`. 우리 번들은 NAS 원본의 사본이고 설정 파일 md5 가 동일함을
확인했다. `configs/` 로 처음 들여온 4개 디렉토리는 `BSD_F606/` 의 `*_Truck` 서브셋이었다.
초기에 쓴 `BSD-F606A/Corner` 식 키는 이 정렬로 철회했다.

---

## 미결 사항 요약

작업 중 해결한 것과 남은 것을 한눈에. 상세는 각 항목 절 참조.

### 담당자 판단이 필요한 것 (미결 27건)

| id | 항목 | 필요한 판단 |
|---|---|---|
| **A1** | 학습·배포 해상도 불일치 — **수정본은 만들었고 검증했다** | Corner v02 / Front v02 를 **본학습해 출하본을 교체할지.** 미학습 상태이며 출하본은 여전히 v01 |
| **D1** | Corner 학습 crop 1280x1016 vs 사양 ROI 1072x1072 | 의도였나. **v02 본학습 전에 결정해야 한다** — 나중에 고치면 학습을 두 번 돌린다 |
| **C1** | Front 정본이 둘 (`v01` as-run / `v01_ga_candidate`) | 어느 쪽을 최종으로 둘지. GA 후보를 살릴 이유가 있는지 |
| **B1** | `results.csv` **4건** DRM 암호화 (Corner_Truck·Front_Truck v01 + Front/v08·Side/v04) | DRM 없는 원본을 받을 수 있나. 지표 대조가 막혀 재현 검증의 병목이다 |
| **B2** | `side_total.yaml` 병합본 부재 | 병합 상대가 무엇이었나. 현 정본은 구성 요소 하나뿐 |
| **B3** | Rear_Truck 설정 부재 (NAS 원본도 동일) — `BSD_F606/Rear/v04` 는 **정본화 완료** | Rear_Truck 자체는 설정 없이 둘지. 사양표 어느 행인지 |
| **C2** | Side `mycfg.weight_path` ≠ `args.model` | `side_v04_total.pt` 가 별도로 존재하나 |
| **D2** | Corner 이미지 1920x1072 vs 사양 1920x1080 | 이미 8px 잘린 데이터인가 (ROI y 오프셋 이중 적용 위험) |
| **D3** | Corner board mAP > onnx mAP (+0.021) | 두 평가의 전처리가 동일했나 |
| **D4** | Front `sim_config` 이 `nt98530_d224k_front` calib 참조 | calib 이미지가 이 제품 것이 맞나 |
| **E1** | `eval_roi_crop`·`val_resize` 소비처 0건 | 평가 도구가 어디 있나. repo 로 들여올지 |
| **E2** | `aug_type: Identity` 가 열거값 아님 | 값 검증을 추가할지 (동작은 의도와 같음) |
| **E3** | Side `gen_config` 의 `${width}`/`${height}` 미치환 | NPU 입력 크기 기록 보강 |
| **F1** | `gen_config`/`sim_config` 절대경로 현존 안 함 | NPU 재컴파일 시 경로 교체 |
| **F3** | 번들 내부 레이아웃 비통일 — 번들을 repo 에 두지 않게 되어 **영향 축소** | NAS 쪽 통일 여부는 별건 |
| **G1~G4·G6·G7** | 사양표 미기재 칸 6건 | 값 확인 |
| **G5** | 제품명 표기 3종 (`BSD_F606`/`BSD-F606`/`BSD-F606A`) | `F606A` 가 개정판인지 별 제품인지 |
| **G8** | NAS 미정본화 카메라 — **BSD_F606 은 8개 완료**. 남은 것은 `BSD_F606/{Front v01..v07, Rear v01..v03, Side v01..v03, DMS v01..v03}`, `D22-4K/{Front,Rear}`, `Sigmastar_SAV83x/CVD_*` | 어디까지 넓힐지 |
| **H1** | ~~`projects/` 설정 트리~~ — **해소.** `projects/` 에 설정이 0개다 (`CVD-H210/` 5개만 G9 로 남음) | — |
| **H4** | `dashcam_front.yaml` 이 class 2 를 `bike` 로 적음 | 기존 파일을 `motorcycle` 로 정정할지 |
| **H5** | ~~`SIDE_v04.yaml` 스키마 드리프트~~ — **해소.** 기계적 변환으로 `configs/releases/BSD_F606/Side/v04/exp.yaml` 이 됐고 원본은 `args_dump.yaml` 로 보존 | — |
| **I1** | `--cfg_options` 우선순위 — **수정 완료** | **D-번호로 기록할지.** 문서화된 인터페이스의 의미를 바꾼 변경이라 D-018 후보. `decisions.md` 가 append-only 여서 임의 추가하지 않고 D-017 본문 메모로만 걸어 뒀다 |
| **F4** | `git_commit: unknown` (학습 체크아웃 현존 안 함) | 복원 불가 — 정보성 기록 |
| **G9** | `projects/CVD-H210/` 5개 — NAS 위치가 `Sigmastar_SAV83x/CVD_H210_{사선,저상}` 로 **다른 SoC** | `configs/releases/` 에 SoC 레벨을 추가할지 (구조 변경, 별건으로 합의) |
| **G10** | `Front/v08/network/binary_SAV837/` 은 Sigmastar SAV837 산출물 (`.sim` 12 MB×3) | NT98530 대상인 이 repo 범위 밖 — 기록만 |
| **G11** | `Side_Truck/241106` 이 `opt_yolox_CVD_F410` — YOLOX 계열이고 제품도 CVD_F410 | 이 repo(YOLOv8 fork) 정본 대상인지 |
| **G12** | `DMS2_v01`(260513) 이 `DMS/v04`(251202) 보다 나중 | DMS2 가 별 계열인지 후속 버전인지 |
| **C3** | `DMS/DMS2_v01` 에 `mycfg`·`args.yaml` 이 둘 다 없다 — GA 튜닝 산출물이라 정본 미작성 | `tune_results.csv` 와 `best.pt` 의 저장된 args 를 읽어 복원할지 |
| **D5** | 신규 5개 릴리스 전부 `data` 추적 불가 — `exp.yaml` 의 `data` 를 비웠다 | ~/Workspace/data 기준 재현 (아래 우선순위 참조) |

### 구조 재편 (2026-07-29, 담당자 요구 반영)

요구사항이 "로컬 실험 중 **실험↔실험**, **실험↔릴리스**가 구분되면 된다" 로 정리되면서 초기
분류를 폐기했다. `configs/exp/` 는 실험이 아니라 **릴리스 레시피**였고 — 출하 v01 을 `args.yaml`
에서 역생성한 것 — 그래서 `configs/models/` 와 같은 대상을 다른 이름으로 가리키고 있었다.

| 재편 후 | 내용 |
|---|---|
| `configs/experiments/<P>/<C>/<V>/exp.yaml` | 실험. 미출하 `v02` 2개와 `v01_ga_candidate` 가 여기로 |
| `configs/releases/<P>/<C>/<V>/` | 출하·동결. `exp.yaml` + `MANIFEST.yaml` + 산출물 |
| `configs/archive/` | 죽은 실험 |

두 트리가 **동일 형태**라 승격이 디렉토리 복사가 된다. `models/` 이름은 버렸다(담은 것은
릴리스이고 `projects/src/cfg/models/` 와 겹쳤다). 미출하 v02 를 실험으로 재분류하면서 "산출물
없는 빈 릴리스 디렉토리" 문제도 사라졌다.

**출력 경로 파생** — `cfg.py` 의 `derive_run_dir` 이
`configs/{experiments,releases}/<P>/<C>/<V>/exp.yaml` → `runs/detect/<P>/<C>/<V>` 로 파생한다.
도구 8개 11곳에서 `apply_cfg_options` 직후 호출. 설정에서 `project`/`name`/`save_dir` 을 전부
제거했다. 트레이너 계열 차이(custom 은 `overrides.project`/`name`, symmetric 은 최상위
`save_dir`)를 헬퍼가 흡수하므로 양쪽이 같은 위치에 떨어진다.

검증 — 6개 설정 전부 파생 결과를 측정하고, 두 계열(`custom` Corner v02, `symmetric` Front v02)
로 간이학습을 돌려 `runs/detect/BSD_F606/{Corner,Front}_Truck/v02` 에 실제로 떨어지는 것을
확인했다 (exit 0, 각 1에폭).

**archive 2차** — 잔류 47개 중 참조 0건인 **22개**를 추가 이관했다 (앞서 basename 부분일치로
`live` 로 과다 집계됐던 것들). 잔류 25개 = 코드 참조 12 + D-010 전이 보존 9 + `search_space` 4.
`_base_` 충돌 0건.


### 이 작업에서 해결한 것

| id | 항목 | 결과 |
|---|---|---|
| A1 (코드·설정) | 해상도 불일치 수정본 | Corner v02 · Front v02 신설. 실제 간이학습으로 각각 320x320 / 640x512 확인 |
| F2 | Rear 버전 표기 | NAS 원본 미러링으로 해소 (임의 명명 철회) |
| H2 | 모델 yaml 두 곳 분산 | `projects/src/cfg/models/audio/` 로 통합, `projects/cfg/` 제거 |
| H3 | README 입력 사양 표 | 2행 → 사양표 12행 전체로 확장 |
| I1 | `--cfg_options` 가 설정에 덮임 | `apply_cfg_options` 헬퍼 + 도구 9개 13곳. 실제 학습에서 `epochs=1` 반영 확인 |
| I2 | 상대 `project` 경로 겹침 | 정본 3개의 `runs/` 접두어 제거 |
| I3 | symmetric `parse_model` 회귀 | `custom.py` 패턴으로 복구 → **D-017** |
| I4 | `v02` → `v022` 네이밍 충돌 | 실행 이름을 문자로 끝나게 (`v01_train`) |

---


## 학습 재현 검증 — 진행 우선순위

정본이 "실제로 그 모델을 만드는가" 를 확인하려면 릴리스마다 네 가지가 필요하다.

1. **데이터셋** — `exp.yaml` 의 `data` 가 채워지고 경로가 resolve 될 것
2. **시작 가중치** — `weight_path`. NAS 번들에서 가져와야 한다
3. **설정이 도는가** — 로더 통과 + 간이학습 완주
4. **지표가 맞는가** — 기록된 지표와 대조. `results.csv` 가 DRM 이면 여기서 막힌다

현재 상태를 이 네 축으로 놓으면 순서가 정해진다.

| 릴리스 | 데이터셋 | 가중치 | 설정 | 지표 |
|---|---|---|---|---|
| Corner_Truck v01 → **v02** | ✅ CVD/Corner | NAS | ✅ 간이학습 확인 | ❌ DRM |
| Front_Truck v01 → **v02** | ✅ Dashcam/Front_traffic_light | NAS | ✅ 간이학습 확인 | ❌ DRM |
| Side_Truck v01 | ⚠ CVD/Side (병합 상대 불명) | NAS | ✅ 로더 통과 | ✅ results.csv 판독 가능 |
| Rear/v04 | ❌ | NAS | ✅ 로더 통과 | ✅ 1000ep 지표 있음 |
| Front/v08 | ❌ (값 2개 상충) | NAS | ✅ 로더 통과 | ❌ DRM |
| Side/v04 | ❌ | ❌ stock yolov8n | ✅ 로더 통과 | ❌ DRM |
| DMS/v04 | ❌ | ❌ 아키텍처 yaml | ✅ 로더 통과 | ✅ 20ep 지표 있음 |
| DMS/DMS2_v01 | ❌ | NAS best.pt | ❌ 정본 없음 | tune_results.csv |
| Rear_Truck, Side_Truck/241106 | ❌ | — | ❌ 설정 없음 | — |

### 재현 검증 절차 (Corner→Front→Side 에서 확립)

세 릴리스를 돌려보며 확정한 순서다. 각 항목은 앞 릴리스에서 놓쳐서 추가된 것이다.

1. **기준선을 우리 val 셋으로 만든다.** 원본 `weight/weights/best.pt` 를 정본과 같은 조건으로
   평가한다. `results.csv` 가 DRM 이어도(Corner·Front) 이 방법은 통한다.
   *Side 에서 배운 것* — `results.csv` 가 판독 가능해도 그 곡선을 그대로 쓰면 안 된다. 그 val 셋은
   `side_total` 의 것이고 우리 것(`CVD/Side` 14,267장)과 다르다. `results.csv` 에는 val 크기 열이
   없어 사후 확인도 불가하다. 곡선은 **형태 참고용**으로만 쓴다.

2. **시작 가중치도 같이 평가한다.** `weight_path` 자체의 성능을 재서 학습이 기여한 폭을 분리한다.
   *Front 에서 배운 것* — 시작점이 mAP50 0.7685, 기준선이 0.774 였다. 50에폭 전체가 0.006 을
   올린 셈이라, 학습을 안 해도 "재현 성공(±0.01)" 으로 판정된다. 6시간을 써도 설정 검증력이 없다.
   `lr0 == lrf` 이고 값이 작으면(Front 는 0.0001) 미세조정 런이라는 신호다.

3. **해상도는 dataset 레벨에서 train·val 각각 측정한다.** 배치 플롯으로 판정하지 않는다.
   *`val/symmetric.py` 누락을 놓친 이유* — `plot_images` 는 letterbox 이후 크기를 그리므로
   `train_batch*.jpg` 와 `val_batch*.jpg` 가 같은 크기로 나와 차이가 가려진다.

4. **1에폭 실측으로 전체 소요를 추정한다.** `fraction` 을 줄인 간이학습은 캐시 효과가 섞여
   외삽이 부정확하다. 실측값: Corner 3.7분, Front 7.3분, Side 8.6분 (모두 GPU 1장).

5. **판정 기준을 먼저 정한다.** 단일 지표 임계값(mAP50-95 ±0.01 등)은 2번에서 시작점이 이미
   그 안에 들어오지 않는지 확인한 뒤에만 의미가 있다.

**GPU 수를 바꾸면 재현이 아니다.** 원본 3종 모두 단일 GPU(`device: '1'`) 로 학습됐다. ultralytics
`BaseTrainer` 에 SyncBatchNorm 이 없어(컨테이너에서 확인) DDP 는 GPU 당 배치로 BN 통계를 계산하고,
`find_unused_parameters=True` 로 gradient reduction 순서도 바뀐다. `batch` 는 총량이라
`batch_size // world_size` 로 쪼개진다. 2 GPU 는 새 모델을 만들 때(v02 등) 쓴다.

### P0 — Corner v02 · Front v02 본학습

네 축이 유일하게 모두 갖춰진 둘이다. **재현 검증이면서 동시에 A1(해상도 불일치) 수정의 산출물**
이라, 성공하면 출하본을 교체할 후보가 나온다. 지표 대조는 DRM 때문에 v01 과 직접 비교가 안 되므로,
"v01 재현" 이 아니라 "수정된 설정으로 새 v02 를 만든다" 로 목표를 잡는 편이 정확하다.

선행 결정 — **D1(Corner crop 비율)**. 사양 ROI 1072x1072 와 학습 crop 1280x1016 이 다르다.
v02 에서 함께 고칠지 지금 정해야 한다. 나중에 고치면 학습을 두 번 돌린다.

### P1 — 데이터셋 동정 (GPU 불필요, 저비용)

v01 3개에서 통했던 방법을 그대로 쓸 수 있다 — 번들의 렌더링된 검증 이미지·errorcase 파일명을
`~/Workspace/data` 에서 역추적하는 것이다. NAS 번들에 재료가 있다.

| 릴리스 | 재료 | 파일 수 |
|---|---|---|
| Front/v08 | `analysis/IPU_Test/` | 2,555 |
| Rear/v04 | `analysis/IPU_Test/` | 1,200 |
| Side/v04 | `analysis/IPU_Test/` | 1,084 |

Rear/v04 는 지표까지 판독 가능하므로 **P1 중에서도 먼저**다 — 데이터셋만 확정되면 네 축이 모두
채워지는 유일한 신규 릴리스다.

DMS/v04 는 `analysis/` 가 비어 있어 이 방법이 안 통한다. `weight/labels.jpg` ·
`labels_correlogram.jpg` 와 Dashcam/DMS_stage{1,2} 대조가 남은 수단이다.

### P2 — DRM `results.csv` 원본 재취득

4건(Corner_Truck·Front_Truck v01, Front/v08, Side/v04). 이것 없이는 그 릴리스들의 **지표 축을
채울 수 없다** — 완료 에폭조차 모른다. P0·P1 과 독립이고 담당자 외부 조치가 필요하다.

### P3 — DMS 계열

`DMS/v04` 는 데이터셋(`sun_light_3px_fine3.yaml`, 경로 없는 파일명)과 시작 가중치(아키텍처 yaml)
둘 다 미해결이다. `DMS/DMS2_v01` 은 정본 자체가 없어 `tune_results.csv` · `best.pt` 의 저장된
args 를 읽는 작업이 선행한다.

### P4 — 설정 부재 릴리스

`Rear_Truck`, `Side_Truck/241106`. 후자는 YOLOX·CVD_F410 이라 이 repo 대상인지부터 확인해야 한다
(G11).

### 공통 선행 작업

- **번들 가져오기 절차** — 지금은 각자 `cp -r <nas_source> configs/releases/<...>/` 다.
  반복되면 스크립트로 만드는 편이 낫다. NAS_2 는 읽기 전용 WebDAV 마운트라 대용량 복사가 느리다
  (신규 5개 합계 901 MB).
- **`data` 가 빈 정본은 그대로 돌지 않는다.** 의도된 실패다 — 조용히 엉뚱한 데이터로 학습하는
  것보다 `FileNotFoundError` 가 낫다. 재현 시 `--cfg_options data=<경로>` 로 넘기거나
  `configs/experiments/` 에 사본을 만들어 채운다 (I1 수정으로 CLI 가 이제 이긴다).

## A. 출하 모델에 영향 — 우선 판단 필요

### A1. 학습·배포 해상도 불일치 (출하 3종 중 2종)

| 카메라 | 설정 | 학습 실측 (W×H) | 배포 ONNX/NPU | 일치 |
|---|---|---|---|---|
| Side | `imgsz 320` + `data_args.imgsz_y 320` | 320×320 | 320×320 | ✅ |
| Corner | `imgsz [320,320]`, `imgsz_y` 없음 | **320×480** | 320×320 | ❌ |
| Front | `imgsz 640`, `imgsz_y` 없음 | **640×480** | 640×512 | ❌ |

원인 — `imgsz_y` 기본값이 480 (`projects/src/data/dataset/dynamic_crop.py:38`).
`imgsz` 가 int 면 `imgsz = (imgsz_y, imgsz)` 가 된다 (같은 파일 `:153`).
Corner 의 `[320,320]` 은 학습 시 ultralytics `check_imgsz(max_dim=1)` 이 정수 320 으로 강제해
("`'train' and 'val' imgsz must be an integer`" 경고) int 분기를 탄다.

측정 근거 — `train_batch0.jpg` 실측: Corner 1280×1920 (4×4 그리드, 셀 320×480),
Front 1920×1440 (셀 640×480, `plot_images` 의 `max_size` 1920 로 0.75 축소),
Side 1280×1280 (셀 320×320). ONNX graph input 직접 파싱: Front `[1,3,512,640]`,
Corner·Side `[1,3,320,320]`. `check_imgsz` 는 컨테이너의 ultralytics 8.4.6 으로 확인.

Front 은 crop(1280×1024, 사양과 일치)을 480 높이로 리사이즈하므로 종횡비가 1.25 → 1.333 으로
왜곡된다.

**고치는 방법이 카메라마다 다르다.**

- **Corner — 설정 수정 완료 (2026-07-29).**
  `configs/experiments/BSD_F606/Corner_Truck/v02/exp.yaml` 신설. `data_args.imgsz_y: 320` 명시 +
  `imgsz` 를 정수 320 으로 (리스트면 `check_imgsz` 가 경고와 함께 변환한다).
  `CustomDetectionTrainer` 가 `data_args` 를 실제로 전달하므로(`custom.py:135`) 설정만으로
  해결된다.

  **검증 2단계 모두 통과.**
  1. dataset 직접 생성 후 샘플 텐서 측정 — v01 `320x480`, v02 `320x320`
     (`imgsz_y` 각각 480 / 320).
  2. 간이학습 (`fraction=0.003`, GPU 0, 3.6분) — 완주 후 `train_batch*.jpg` 와
     `val_batch*.jpg` 전 12장이 모두 `1280x1280` (4x4 그리드, 셀 **320x320**). v01 이었다면
     `1280x1920` (셀 320x480) 이 나온다. 학습 경로와 val 경로(`LetterBox`) 양쪽 확인.
     산출물: `runs/detect/runs/BSD_F606/Corner_Truck/v02/` (경로가 겹친 이유는 아래 I2).
  as-run 기록인 `v01.yaml` 은 손대지 않았다 — 그 파일이 출하 `best.pt` 를 만든 설정이라는
  MANIFEST 기록을 유지해야 한다. **v02 는 미학습이고 출하본은 여전히 v01 이다.**
- **Front — 코드 + 설정 수정 완료 (2026-07-29).** 설정만으로는 고칠 수 없었다.
  `SymmetricTrainer.build_dataset` 이 `data_args` 를 아예 넘기지 않아 `DEFAULT_DYCROP_ARGS` 가
  강제되고 그 안에도 `imgsz_y` 가 없었다. 두 가지를 했다.
  1. `projects/src/models/yolo/detect/train/symmetric.py` `build_dataset` 에
     `data_args=self.exp_params.get('data_args', {})` 추가 — `custom.py:135` 와 동일하게.
     `data_args` 가 없는 설정에는 `{}` 가 넘어가 동작이 바뀌지 않는다 (v01 이 그대로 640x480
     인 것으로 확인).
  2. `configs/experiments/BSD_F606/Front_Truck/v02/exp.yaml` 신설 — `data_args.imgsz_y: 512`.
     crop 1280x1024 의 종횡비 1.25 가 보존된다. 하이퍼파라미터는 v01 과 차이 없음을 대조 확인.

  **검증** — dataset 측정에서 v01 `640x480` / v02 `640x512`. 실제 간이학습
  (`epochs=1 fraction=0.004`) 완주, exit 0, 배치 플롯 9장 전부 `1920x1536` (종횡비 **1.2500**).
  640x512 의 4x4 그리드 2560x2048 이 `plot_images` 의 `max_size` 1920 으로 축소된 값이다.
  640x480 이었다면 1920x1440 (1.3333) 이 나온다.
  **v02 는 미학습이고 출하본은 여전히 v01 이다.**

  단, 이 학습을 돌리기까지 별도의 기존 회귀를 먼저 고쳐야 했다 — 아래 I3.

**❓ 확인 필요**
1. 이 불일치가 의도였나, 사고였나? (Corner 는 수정본을 만들어 뒀으나 학습 여부는 미결)
2. 사고라면 재학습 범위는? Corner v02 를 학습해 출하본을 교체할지, Front 는 어디까지.
3. Front 을 고칠 때 `SymmetricTrainer` 에 `data_args` 전달을 추가할지, `DEFAULT_DYCROP_ARGS`
   에 `imgsz_y` 를 넣을지 — 후자는 Front 외 symmetric 사용처 전체에 영향.
4. Corner v02 학습 시 D1(crop 비율)도 함께 고칠지. D1 도 설정만으로 가능하지만 "의도였나" 가
   미결이라 v02 에서는 v01 그대로 뒀다.

---

## I. 간이학습에서 새로 드러난 코드 결함 (2026-07-29)

A1 검증용 간이학습을 돌리다 발견했다. 둘 다 설정 정본화와 무관하게 이미 존재했던 것이다.

### I1. `--cfg_options` 가 설정 파일에 조용히 덮인다

`tools/train_custom_trainer.py:78-80` 이 `kwargs.update(cfg_options)` 로 CLI 값을 반영한 뒤,
`CustomDetectionTrainer.__init__` (`projects/src/models/yolo/detect/train/custom.py:33`) 가
`overrides.update(self.custom_cfg.overrides)` 로 **설정 파일의 overrides 를 다시 덮어쓴다.**
그래서 설정의 `overrides` 에 있는 키는 CLI 로 바꿀 수 없다. `SymmetricTrainer.__init__`
(`symmetric.py:68`) 도 같다.

측정 — `--cfg_options epochs=1 batch=32 workers=4 fraction=0.003` 로 실행했더니:

| 키 | 설정에 있나 | 지정 | 실제 |
|---|---|---|---|
| `epochs` | 있음 (50) | 1 | **50** |
| `batch` | 있음 (128) | 32 | **128** (300장 → 3 배치) |
| `workers` | 있음 (4) | 4 | 4 (구분 안 됨) |
| `fraction` | 없음 | 0.003 | 0.003 ✅ |

README 는 `--cfg_options epochs=50 batch=64 lr0=0.001` 을 override 수단으로 안내하는데,
그 예시의 세 키가 모두 설정에 흔히 있는 키라 **문서대로 써도 동작하지 않는다.**

**수정 완료 (2026-07-29)** — CLI 가 이기게 했다. `projects/custom_utils/cfg.py` 에
`apply_cfg_options(cfg, cfg_options)` 헬퍼를 두고, 설정을 로드해 `custom_cfg` 로 넘기는 도구
9개 파일 13곳에서 호출한다 (`train_custom_trainer`, `train_symmetric_detector`,
`train_consistency_trainer`, `train_kd`, `train_qat`, `validate_custom`, `eval`,
`run_finetune_pipeline` 4 stage, `run_finetune_pipeline_v2`).
트레이너의 `overrides.update(...)` 는 직접 호출 경로를 보호하므로 그대로 뒀다.

**부수 이득** — 트레이너가 `save_dir/mycfg.yaml` 로 저장하는 as-run 기록에 CLI 값이 포함된다.
`mycfg.yaml` 이 `args.yaml` 과 갈라지는 구조적 원인 하나가 없어진다 (C1 의 근거 2·3 이 그 부류).

**적용하지 않은 4곳** — `train_tuner_tlds.py`(trainer partial 이 주석 처리돼 `custom_cfg` 를
넘기지 않음), `validate_symmetric.py`(`get_my_cfg` 자체가 주석), `nova_qat_pipeline.py`(자체
`PipelineContext` 기구, D-010 보류), `tune_custom_trainer.py`(`custom_cfg` 가 경로 문자열이라
`CustomTuner` 가 파일을 다시 읽는다 — 다만 `CustomTuner` 는 서브프로세스로
`train_custom_trainer.py` 를 부르므로 간접 해결된다).

**❓ 확인 필요** — **D-번호로 기록할지.** 문서화된 인터페이스(`--cfg_options`)의 의미를 바꾼
변경이라 D-018 후보다. `docs/decisions.md` 가 append-only 라 임의로 행을 추가하지 않고 D-017
본문 메모로만 걸어 뒀다.

### I2. 경로 조각 2개 이상인 상대 `project` 는 경로가 겹친다

ultralytics 8.4.6 `get_save_dir` 는 `project` 의 경로 조각이 2개 이상이면 "nested" 로 보고
`RUNS_DIR/<task>/` 를 앞에 붙인다. 컨테이너에서 실측:

| `project` | `save_dir` |
|---|---|
| `runs/BSD_F606/Corner_Truck` | `runs/detect/runs/BSD_F606/Corner_Truck/<name>` ← 겹침 |
| `BSD_F606/Corner_Truck` | `runs/detect/BSD_F606/Corner_Truck/<name>` |
| (없음) | `runs/detect/<name>` |

`train_custom_trainer.py:91-96` 에 이 이중 prefix 를 피하려고 상대 `project` 를 절대화하는
우회가 있고 주석도 그렇게 적혀 있으나, **I1 때문에 무력화된다** — 우회는 `kwargs` 에
적용되고 그 뒤 트레이너가 설정의 `project` 로 다시 덮는다.

정본 3개(`Corner_Truck/{v01,v02}`, `Side_Truck/v01`)가 `project: runs/BSD_F606/...` 로
적혀 있어 겹친 경로를 만들었다. **수정 완료** — 앞의 `runs/` 를 떼어 `BSD_F606/<Camera>` 로
바꿨다 (각 줄에 사유 주석).

### I3. `SymmetricDetectionModel` 이 fork `parse_model` 을 쓰지 않아 symmetric 학습 불가

`tools/train_symmetric_detector.py` 가 `KeyError: 'SymmetricDetect'`
(ultralytics `nn/tasks.py:1612` `parse_model`) 로 즉시 죽었다. v01·v02 양쪽 동일하므로 이번
정본화와 무관한 기존 회귀다.

fork 에는 `SymmetricDetect`/`DMSPose` 를 아는 자체 `parse_model`
(`projects/src/nn/tasks/parse_util.py`) 이 있고 `custom.py`/`dair.py`/`kd_draft.py` 는 모두
`DetectionModel.__init__` 을 우회해 그것을 직접 호출한다. `symmetric.py` 만
`super().__init__(cfg, ch, nc, verbose)` 를 불러 ultralytics 의 `parse_model` 로 떨어졌다.
Front_Truck v01 출하 가중치가 존재하므로 학습 당시에는 동작했고 이후 회귀한 것으로 보인다
(어느 커밋인지는 특정하지 않았다).

**수정 완료** — `custom.py` 와 동일한 패턴으로 교체. 상세는 `docs/decisions.md` 의 **D-017**.

### I4. 재실행 시 `v02` → `v022` 로 붙어 버전처럼 읽힌다

`increment_path(path, exist_ok=False, sep="")` 이므로 이름 뒤에 구분자 없이 숫자가 붙는다.
컨테이너 실측:

| 이름 | 재실행 1 | 2 | 3 |
|---|---|---|---|
| `v02` | `v022` | `v023` | `v024` |
| `train` | `train2` | `train3` | `train4` |
| `v02_train` | `v02_train2` | `v02_train3` | `v02_train4` |

ultralytics 표준 동작(`train` → `train2`)이지만, 이름이 숫자로 끝나면 `v022` 가 "버전 022" 로
읽혀 릴리스 버전과 혼동된다. 실제로 Front 간이학습 산출물이 `v022` 에 들어갔다.

**수정 완료** — 정본 5개의 실행 이름을 문자로 끝나게 바꿨다: `name: v01` → `v01_train`
(Corner·Side), `save_dir: .../v01` → `.../v01_train` (Front). `v01_ga_candidate` 는 이미 문자로
끝나 그대로 뒀다.

---

## B. 복원 불가 — 원본 재취득 필요

### B1. `results.csv` DRM 암호화 (2건)

`configs/releases/BSD_F606/{Corner,Front}_Truck/v01/weight/results.csv` 가 암호화되어 있다.
헤더: `NX5200803SECURE FILE : This file is encrypted.`
번들 219개 파일 중 이 둘만 해당된다 (Side 는 정상 CSV, 150 에폭).

영향 — 학습 곡선과 **실제 완료 에폭을 알 수 없다**. 두 MANIFEST 의 `epochs_completed` 가
`unknown` 인 이유다.

**❓ 확인 필요** — DRM 걸리지 않은 원본을 받을 수 있나?

### B2. `side_total.yaml` 병합본 부재

Side v01 이 참조한 `projects/data_yaml/side_total.yaml` 이 삭제됐고, 그 병합 데이터셋도 이
호스트에 없다.

- v01 의 `val_batch0_labels.jpg` 에 렌더링된 검증 이미지가 `1t_KR_20230314_TRUCK1T_...` —
  차종 접두어가 붙어 있다. 접두어 붙은 파일은 데이터 루트 전체에 **0건**.
- `tools/merge_datasets.py` 는 접두어를 붙이지 않는다 (stem 그대로 복사) → 다른 방법으로 만든 것.
- 접두어 없는 원본이 `/dataset/CVD/Side/validation/images/` 에 4,894건 존재 →
  정본은 `configs/datasets/cvd_side.yaml` (= `/dataset/CVD/Side`) 를 가리킨다.

**❓ 확인 필요** — 병합 상대가 무엇이었나? 현재 정본으로 재학습하면 v01 과 데이터 구성이 다르다.

### B3. Rear 설정 자체가 없음

`configs/releases/BSD_F606/Rear_Truck/` 에는 `binary/nvt_model.bin`,
`weight/{modified_nano,pruned_nano}.onnx` 3개뿐이다. `mycfg.yaml`·`args.yaml`·학습 산출물이
전혀 없어 `configs/` 에서 정본을 복원할 수 없다. ONNX 입력은 둘 다 `[1,3,320,320]`.

**NAS 원본에서도 동일하다** — `Novatec_NT98530/BSD_F606/Rear_Truck/` 에 같은 3개 파일뿐이다.
복사 누락이 아니라 원본에 설정이 없다.

다만 NAS 에는 별도로 `BSD_F606/Rear/{v01..v04}` 가 있고 **v03·v04 에는 `mycfg.yaml` +
`args.yaml` 이 있다**. 이쪽은 `Rear_Truck` 과 다른 카메라(사양표의 BSD-F606 Rear, 480×256)다.

**❓ 확인 필요**
1. `Rear_Truck` 은 설정 없이 ONNX·bin 만 남은 상태로 둘지, 학습 기록을 따로 받을지.
2. `BSD_F606/Rear/v04` (설정 있음) 를 정본화 대상에 넣을지 — 넣는다면 `Rear_Truck` 과 별개
   카메라로 다룬다.
3. 사양표의 BSD-F606 Rear 는 480×256 인데 `Rear_Truck` ONNX 는 320×320 이라 서로 다른 세대로
   보인다. `Rear_Truck` 이 사양표 어느 행에 해당하나?

---

## C. 설정 신뢰성

### C1. Front `mycfg.yaml` 은 손으로 조립됨 → 정본 2개, 최종 미결

출하 `weights/best.pt` 를 만든 설정이 아니다. 근거 4건:

| # | 근거 |
|---|---|
| 1 | YAML 파싱 실패 — `mycfg.yaml:12`, `eval_roi_crop` 이 `replace_tl_score` 리스트 안으로 잘못 들여쓰기 |
| 2 | override 29개 중 **19개**가 `args.yaml` 과 다르다 (GA 튜닝값 vs 실제 쓰인 기본값 — `lr0` 0.0047 vs 0.0001, `degrees` 12.85 vs 0.0, `mosaic` 0.58 vs 1.0 …) |
| 3 | `save_dir` 이 `Front_v01_sym_tuned`, `args.yaml` 은 `Front_v01_sym` |
| 4 | `eval_roi_crop` 이 `[428,8,1072,1072]` — **Corner v01 의 ROI** 와 동일. Front 실제는 `[320,0,1280,1024]`. Corner 설정에서 복사돼 들어왔다 |

파일 안 주석 "실제 학습 복원 (GA trial 15/35)" 도 재구성임을 시사한다.

그래서 정본을 둘로 나눴다 — `v01.yaml` (실제 학습본, `args.yaml` 역생성) /
`v01_ga_candidate.yaml` (GA 후보, 구문 오류만 수정, **학습 산출물 없음**).

**❓ 확인 필요** — 어느 쪽을 최종 정본으로 둘까? 참고로 D-007 은 GA 파라미터가 25ep horizon 에
overfit 한다고 결론냈고 D-008 은 production 을 baseline200 으로 정했다. 이 GA 후보를 살릴 이유가
있는지.

### C2. Side `weight_path` 불일치

`mycfg.yaml` 의 `weight_path` 는 `/workspace/weights/side_v04_total.pt` 인데 같은 런의
`args.yaml` `model` 은 `/workspace/weights/preweight.pt` 이고, 번들 실물은 후자뿐이다.
정본은 `preweight.pt` 를 쓴다.

**❓ 확인 필요** — `side_v04_total.pt` 가 별도로 존재하나? Side 도 Front 처럼 `mycfg` 가
사후 편집됐을 가능성.

---

### C3. `DMS/DMS2_v01` 은 대조할 설정이 없다

번들에 `mycfg.yaml` 도 `args.yaml` 도 없다. 남은 것은 GA 튜닝 산출물뿐이다 —
`weight/{best_hyperparameters.yaml, tune_results.csv, best.pt, 260513_ga_dms2.onnx}`.
`projects/BSD-F606/DMS2_v01.yaml` 은 그 `best_hyperparameters.yaml` 과 내용이 동일하며
(`_fitness: 0.607588`, `_run: final_02_from_gen03_ind03`), 전체 설정이 아니라 하이퍼파라미터
조각이다. 튜닝된 키는 DMS pose 부위별 가중치(`kpt_w_*`, `shape_w_*`)와 우리 fork 의 커스텀 증강
확률(`mask_occlusion`, `sunglasses_occlusion`) 이라 이 튜닝은 fork 로 돌았다 — stock ultralytics
로 돈 `DMS/v04` 와 다르다.

정본을 만들면 `imgsz`·`epochs`·`batch`·`data` 를 전부 추정해야 한다. Front_Truck v01 을 신뢰할 수
없게 만든 손조립을 새로 만드는 셈이라 **만들지 않았다.**

**❓ 확인 필요** — `tune_results.csv` 와 `best.pt` 에 저장된 args 를 읽어 복원할지.

## D. 사양 대조 미해결

기준은 담당자 제공 제품 사양표 (부록 참조). ROI 표기 주의 — 사양표와 README 는
`(x1,y1,x2,y2)`, 설정의 `eval_roi_crop` 은 `(x, y, w, h)` 다. 같은 영역을 다르게 적는다.
(예: Corner 사양 `428, 8, 1500, 1080` = 설정 `[428, 8, 1072, 1072]`)

### D1. Corner 학습 crop 이 사양 ROI 와 다르다

`data_args` 에 crop 비율이 없어 `DynamicCropWrapper` 기본값(1280/1920 · 1024/1080)이 적용된다.
1920×1072 입력에서 학습 crop 은 1280×1016 (random 위치)인데 사양 ROI 는 (428,8) 에서
1072×1072. 사양에 맞추려면 `crop_width_ratio: 1072/1920`, `crop_height_ratio: 1072/1080`.

**❓ 확인 필요** — 의도였나?

### D2. Corner 데이터셋 이미지 높이가 사양과 다르다

실측 `1920×1072`, 사양 원본은 `1920×1080`. 사양 ROI 의 y 범위 8..1080 이 이미지 높이를 8px
넘는다. 평가 도구가 repo 밖이라 처리 방식을 알 수 없다.

**❓ 확인 필요** — 데이터셋 이미지가 이미 8px 잘려 있나? 그렇다면 ROI y 오프셋 8 이 이중 적용될
위험.

### D3. Corner 는 board 가 onnx 보다 좋다

`mAP` onnx 0.460 → board 0.481 (+0.021), `mF1` 0.554 → 0.566. 양자화·컴파일을 거친 온보드가
float ONNX 보다 좋게 나오는 것은 통상과 반대 방향이다. Front 도 같은 방향이지만 폭이 작다
(0.697 → 0.704).

**❓ 확인 필요** — 두 평가의 전처리(ROI, 리사이즈, 정규화)가 동일했나?

### D4. Front `sim_config.txt` 가 다른 제품 calib 을 참조

`test_img_dir` 이 `nt98530_d224k_front` — D22-4K Front 계열 이름이다. `src` 크기는 640×512 로
이 모델과 맞아 디렉토리 이름만 재사용된 것으로 보이나 확인되지 않았다.

**❓ 확인 필요** — calib 이미지가 BSD-F606A Front 것이 맞나?

---

### D5. 신규 5개 릴리스 전부 `data` 추적 불가

`exp.yaml` 의 `data` 를 비웠다. 학습 당시 문자열은 각 MANIFEST 의 `as_run_differs.data` 에 남겼다.

| 릴리스 | 학습 당시 `data` | 상태 |
|---|---|---|
| Front/v08 | `FRONT/TLDS_rework_add.yaml` (mycfg) / `configs/data/BSD-F606/FRONT_v01.yaml` (args) | 두 값이 상충하고 둘 다 현존하지 않음 |
| Rear/v04 | `/ultralytics/projects/data_yaml/FP_448_256/add_org.yaml` | `FP_448_256/` 가 이 repo 이력에 없음 |
| Side/v04 | `./ultralytics/cfg/datasets/BSD_PP_sec_add_back_rear.yaml` | 이 repo 에 없음 |
| DMS/v04 | `sun_light_3px_fine3.yaml` | 경로 없는 파일명 |
| DMS2_v01 | 없음 | 하이퍼파라미터 조각뿐 |

**❓ 확인 필요** — `~/Workspace/data` 기준 재현은 별건으로 미뤘다 (담당자 지시, 2026-07-29).
방법과 순서는 위 "학습 재현 검증 — 진행 우선순위" 의 P1 참조.

## E. 죽은 설정 키

| id | 내용 |
|---|---|
| E1 | `exp_params` 의 `eval_roi_crop` · `val_resize` 는 repo 내 소비처 **0건**. metrics 파일이 ROI 를 기록하므로 평가 도구가 repo 밖에 있다. `DynamicCropWrapper` 가 `**kwargs` 로 흡수해(`projects/src/data/dynamic_crop.py:47`) 크래시는 없다 |
| E2 | `aug_type: Identity` (Corner) 는 열거값(`front`/`rear`/`side`)이 아니라 `else` 분기로 `Identity()` 가 된다 (`projects/src/data/augment.py:483-493`). 결과는 의도와 같지만 값 검증에 걸리지 않는다 |
| E3 | Side `network/gen_config.txt` 의 `[preproc/resize/width]`·`[height]` 가 치환되지 않은 템플릿 변수 `${width}`·`${height}` 다. NPU 입력 크기가 번들에 기록돼 있지 않다 |

**❓ 확인 필요** — E1 의 평가 도구는 어디 있나? 정본이 이 키를 계속 들고 갈지, 평가 도구를
repo 로 들여올지.

---

## F. 경로 잔재

| id | 내용 |
|---|---|
| F1 | `gen_config.txt` / `sim_config.txt` 의 절대경로가 `/home/ncn/Developments/data/weights/...` 로 현존하지 않는다 (model_dir, ref_img_dir, out_dir, test_img_dir). NPU 재컴파일 시 교체 필요 |
| F2 | ~~Rear 버전 표기 없음~~ — **해소.** NAS 원본도 `Rear_Truck/{binary,weight}` 로 버전 레벨이 없어 그대로 미러링했다 (임의 명명 철회) |
| F3 | 번들 내부 레이아웃이 통일되지 않았다 — v01 은 `network/`, Rear_Truck·Side_Truck/241106 은 `binary/`. `weight/` 안에 학습 산출물과 가중치가 섞여 있다 (`weight/weights/best.pt`). **NAS 원본도 동일**하므로 통일하면 NAS 와의 기계적 대응이 깨진다 |
| F4 | 학습이 이뤄진 체크아웃(`/home/ncn/Developments/adassw/sources/ultralytics_projects`)이 현존하지 않아 세 카메라 모두 `git_commit: unknown` |

**❓ 확인 필요** — F2 의 Rear 실제 버전. F3 번들 내부 통일을 할지.

---

## G. 사양표 자체의 미기재 항목

담당자 표에 `확인 필요` 로 남은 칸 + 대조 중 생긴 의문.

| 항목 | 내용 |
|---|---|
| G1 | CVD-H210/F410 Side — 입력 해상도·crop 크기 미기재 |
| G2 | BSD-F606 적재함 — 입력 해상도·crop·네트워크 입력 미기재 (클래스는 Person, 데이터 COCO) |
| G3 | D22-4K Side — 입력 해상도·ROI·crop 미기재 |
| G4 | BSD-F606 DMS 2 — 모델이 `Yolov8-pose-nano 확인 필요` 로 적혀 있다 |
| G5 | 제품명 표기가 **세 가지**다 — NAS 디렉토리는 `BSD_F606`, 사양표는 `BSD-F606`, `network/` 안 파일명과 `gen_config` 는 `BSD-F606A` (`bsd_f606a_corner_320x320.onnx`, `260706_BSD-F606A_Front_v01.onnx`). NAS 전체에서 `F606A` 는 디렉토리 키로는 한 번도 쓰이지 않고 파일명에만 나온다. `F606A` 가 하드웨어 개정판 라벨인지 별 제품인지 불명 |
| G8 | NAS 에 우리가 정본화하지 않은 카메라가 있다 — `BSD_F606/{Front(v01..v08), Rear(v01..v04), Side(v01..v04), DMS(v01..v04, DMS2_v01)}`, `D22-4K/{Front(v1,v2), Rear(v1,v2)}`. `*_Truck` 계열만 `configs/` 로 들어왔다. 정본화 범위를 여기까지 넓힐지 |
| G6 | Corner 변형이 둘 — `CN-F308` 5클래스, `CL-F308` 6클래스(**Wheel** 포함). 우리 Corner v01 은 5클래스라 CN 계열로 보이나 명시 필요 |
| G7 | BSD-F606A Side 의 사양 행이 없다. 우리 v01 은 320×320, crop 없음(비율 1.0) |
| G9 | `projects/CVD-H210/` 5개(`CVD_1t`, `CVD_5t`, `DIAGONAL_v04`, `EPC_v01`, `LOW_v04`)의 NAS 위치가 `Sigmastar_SAV83x/CVD_H210_{사선,저상}`·`CVD_F410_공인인증` 으로 **다른 SoC** 다. 현재 `configs/releases/` 는 `<제품>/<카메라>/<버전>` 이라 SoC 레벨이 없다. 추가하면 구조 변경이므로 별건으로 합의했다 (2026-07-29) |
| G10 | `Front/v08/network/binary_SAV837/` 은 Sigmastar SAV837 산출물이다 (`.sim` 12 MB×3 포함). 이 repo 는 Novatek NT98530 대상이라 범위 밖 — 기록만 남긴다 |
| G11 | `Side_Truck/241106` 의 onnx 가 `241106_opt_yolox_CVD_F410_ssv_refine_modified.onnx` 다 — YOLOX 계열이고 제품 표기도 CVD_F410 이다. NAS 상위에 `03_YoloX-nano` 가 따로 있으므로 이 repo(YOLOv8 fork) 정본 대상이 아닐 수 있다 |
| G12 | `DMS2_v01` 의 onnx 는 `260513_ga_dms2.onnx`(2026-05-13) 이고 `DMS/v04` 바이너리는 `BSD_F606_DMS_251202.bin`(2025-12-02) 이다. DMS2 가 v04 보다 나중이므로 후속 버전인지 별 계열인지 불명 |

---

## H. 구조 정리 잔여 (7단계)

| id | 내용 |
|---|---|
| H1 | `projects/` 에 설정 트리 4개 병존. **부분 해소** — `exp_yaml/` 190개 중 143개와 `coco_yaml/` 19개 전체를 `configs/archive/` 로 옮겼다 (잔류 47개 = live 39 + `search_space` 4 + D-010 4). **남은 미결**: `data_yaml/` 39개(이 호스트에서 5개만 resolve), `BSD-F606/`·`CVD-H210/` 10개. 완전 중복 yaml 8쌍 중 archive 로 간 것 외 잔여 |
| H2 | ~~아키텍처 yaml 두 곳 분산~~ — **해소.** 다만 처음 판단과 결론이 다르다. `projects/src/cfg/models/` 는 사고가 아니라 CLAUDE.md:55 가 규정한 ultralytics 내부 미러이고 참조가 48건(archive 후 20건 생존)이라 그대로 두었다. 실제 불일치는 ncnvox **코드**는 `projects/src/ncnvox/` 인데 **모델 yaml** 만 `projects/cfg/models/` 에 떨어져 있던 것 → `projects/src/cfg/models/audio/` 로 이동해 모델 yaml 10개가 한 곳에 모였다. `projects/cfg/` 는 제거. 갱신한 참조: `tests/test_ncnvox_cmd_crnn.py:28`, `tools/train_ncnvox.py:6`, `README.md:228`. `docs/{changelog,technical_doc,completion_report}.md` 의 구 경로는 과거 상태를 적은 로그라 그대로 뒀다. 검증: pytest 56 passed / 2 skipped, ruff 14 (둘 다 기준선) |
| H3 | ~~README 입력 사양 표가 2행뿐~~ — **해소.** 그 2행은 사양표의 **D22-4K 행과 동일**한데 제품명 없이 "Front (4K)" 로만 적혀 있었다. 제품 열을 붙여 12행 전체로 확장하고, `(x1,y1,x2,y2)` 대 `eval_roi_crop` 의 `(x,y,w,h)` 표기 차이 경고와 데이터셋 공유 관계를 함께 넣었다 |
| H4 | `configs/datasets/legacy/dashcam_front.yaml` 은 class 2 를 `bike` 로 적는데 산출물 기록은 모두 `motorcycle` 이다. 새 정본은 `motorcycle` 을 쓴다 |
| H5 | ~~`projects/BSD-F606/SIDE_v04.yaml` 이 `mycfg` 스키마가 아니라 ultralytics `args.yaml` 원본 덤프(플랫 106키)~~ — **해소.** `DMS_v04.yaml`(108키)도 같았다. 둘 다 stock ultralytics 런이었다 — fork 전용 키가 0개다. 기계적 변환(DEFAULT_CFG 와 값이 다른 키만 남김: 106→5, 108→32)으로 `configs/releases/BSD_F606/{Side,DMS}/v04/exp.yaml` 이 됐고, 변환 전 원본은 같은 디렉토리의 `args_dump.yaml` 로 보존했다. "같은 디렉토리에서 `-w` 가 두 스키마를 번갈아 받는" 상태는 `projects/BSD-F606/` 이 없어지면서 사라졌다 |

---

## 부록 — 담당자 제공 제품 사양표 (2026-07-29)

ROI 는 `(x1,y1,x2,y2)`. 설정의 `eval_roi_crop` 은 `(x,y,w,h)` 라 표기가 다르다.

| 제품 | 카메라 | 입력 | ROI (x1,y1,x2,y2) | Crop | 네트워크 입력 | 모델 | 클래스 | 데이터셋 |
|---|---|---|---|---|---|---|---|---|
| CVD-H210/F410 | Corner (CN-F308) | 1920×1080 | 428, 8, 1500, 1080 | 1072×1072 | 320×320 | YoloX-nano, Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | CVD_5t |
| CVD-H210/F410 | Corner (CL-F308) | 1920×1080 | 172, 0, 1782, 1072 | 1576×1072 | 320×320 | YoloX-nano, Yolov8-nano | Bus, Car, Motorcycle, Person, Truck, **Wheel** | CVD_5t |
| CVD-H210/F410 | Rear | 1920×1080 | – | 1920×1080 | 320×320 | YoloX-nano, Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | CVD\Rear |
| CVD-H210/F410 | Side | 확인 필요 | – | 확인 필요 | 320×320 | YoloX-nano | Bus, Car, Motorcycle, Person, Truck | CVD\Side |
| BSD-F606 | Front | 1920×1080 | 320, 0, 1600, 1024 | 1280×1024 | 640×512 | Yolov8-nano | Bus, Car, Motorcycle, Person, Truck, Traffic_light | DashCAM\Front_traffic_light |
| BSD-F606 | Rear | 1920×1080 | 0, 56, 1920, 1080 | 1920×1024 | 480×256 | Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | DashCAM\Rear |
| BSD-F606 | DMS 1 | Face 기준 | x, y, x+960, y+960 | 960×960 | 320×320 | Yolov8-pose-nano | Face + 70 Keypoints | DashCAM\DMS_stage1 |
| BSD-F606 | DMS 2 | Face 기준 | x, y, x+960, y+960 | 960×960 | 320×320 | Yolov8-pose-nano (확인 필요) | Face + 70 Kpts, bottle, cigarette, phone, mouth | DashCAM\DMS_stage2 |
| BSD-F606 | 적재함 | 확인 필요 | – | 확인 필요 | 확인 필요 | Yolov8n | Person | COCO |
| D22-4K | Front | 3840×2160 | 480, 444, 3360, 1980 | 2880×1536 | 480×256 | Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | DashCAM\Front |
| D22-4K | Rear | 2560×1440 | 240, 160, 2320, 1440 | 2080×1280 | 416×256 | Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | DashCAM\Rear |
| D22-4K | Side | 확인 필요 | 확인 필요 | 확인 필요 | 320×320 | Yolov8-nano | Bus, Car, Motorcycle, Person, Truck | DashCAM\Side |

데이터셋이 제품 간 공유된다 — `DashCAM\Rear` 는 BSD-F606 Rear 와 D22-4K Rear 가,
`DashCAM\Front_traffic_light` 는 BSD-F606 Front 와 BSD-F606A Front 가, `CVD\Side` 는
CVD-H210/F410 Side 와 BSD-F606A Side 가 쓴다. `configs/datasets/` 를 제품 키가 아니라
데이터셋 키로 둔 이유다.
