# NPU 툴체인 디렉터리 재구성 계획

**상태**: **Phase A 이행 완료** (2026-09-07, `42b169a` · D-025) · Phase B 보류 (§6 결정 3 이 트리거)
**작성**: 2026-09-04 v1 · 2026-09-07 v2, `master` `14051b4`
**선행**: D-010 (QAT 보류 — `novatek/nmqt/` 보존 근거) · D-016 (D-014 사이클 결론; 본문에
NPU 변환은 wrapper 로만 돌리고 컴파일 단계를 건너뛰지 않는다는 교훈이 있다. CLAUDE.md 가
"skipping a middle step fails silently (D-016)" 로 인용하는 근거)

**목적은 칩이 늘 때 저장소 최상위가 늘지 않게 하는 것이다.** 지금은 칩 하나가 최상위
디렉터리 하나를 차지하고, 두 칩의 규약이 서로 다르다. NAS_2 `03_SDK/` 에는 `allwinner` ·
`nextchip` · `rockchip` SDK 가 이미 있다 — 세 번째 칩은 가설이 아니다.

**v1 에서 바뀐 것.** v1 은 쉬운 단계를 쉽다고 하고 어려운 단계를 얼버무렸다. 코드 대조에서
틀린 주장 셋(1단계 "코드 변경 없이" · 3단계 "단순 이동" · "칩마다 다른 건 출력 규약 하나")과
D-016 제목 오인용을 고쳤고, 비어 있던 정의(`<model_dir>` 매니페스트 · 헤드×칩 2축 어댑터 ·
docker 배치 · N1 비교 대상)를 채웠다. 이행은 **지금 할 것(Phase A)** 과 **다음 칩이 정해지면
할 것(Phase B)** 으로 갈랐다.

---

## 1. 요구 분석

### 1.1 실측 (2026-09-04, 2026-09-07 재확인)

| 경로 | 용량 | 내용 |
|---|---:|---|
| `novatek/` | 2.0G | |
| `novatek/convert/` | 1.3G | 래퍼 5개(`run_{pipeline,compiler,simulator}.sh` · `visualize_{sim,dms_sim}.py`, 40K) + **모델별 작업 디렉터리 6개** — 입력과 출력이 한 디렉터리에 섞여 있다 (§2.1) |
| `novatek/nmqt/` | 719M | NMQT 툴킷 추출본. D-010 보류 QAT 용, **보존 대상** |
| `novatek/novatek_release/` | 62M | 릴리스 번들 (중복 tar.gz 621M 은 2026-09-04 제거) |
| `novatek/tool/` | 4.0K | **파일 0개.** SDK 는 트리 밖으로 옮겨졌는데 컨테이너를 띄울 때마다 빈 스텁이 재생성된다 (P3) |
| `sigmastar/` | 2.2G | 8,281 파일. 통째로 gitignore |
| `sigmastar/data/` | 1.1G | 변환 입출력 (`input/{models,images}` · `output/`) |
| `sigmastar/reference/` | 758M | 참조 모델 (yolov8 · yolox) |
| `sigmastar/sdk/` | 322M | SGS IPU SDK — NAS_2 `03_SDK/sigmastar/` 에 원본 있음 |
| `sigmastar/converter/` | 100K | 스크립트 9개. **`/workspace` 절대경로 하드코딩 32곳 / 10파일** (docker 포함) |
| `sigmastar/docker/` | 20K | `Dockerfile`(ubuntu:18.04) · `build.sh` · `run.sh` |
| `sigmastar/docs/` | 8.0K | `SETUP_GUIDE.md` |

**스크립트는 168K 인데 트리는 4.2G 다** (novatek 40K + sigmastar 128K). 나머지는 벤더 SDK ·
참조 자산 · 변환 산출물이다.

### 1.2 무엇이 문제인가

| # | 문제 | 근거 | 이 계획에서 |
|---|---|---|---|
| P1 | 칩마다 최상위 디렉터리가 하나씩 는다 | `novatek/` · `sigmastar/` | 해결 (§2) |
| P2 | **`/workspace` 가 칩마다 다른 것을 가리킨다** | compose 는 저장소 루트를, `sigmastar/docker/run.sh` 는 `sigmastar/` 를 `/workspace` 로 마운트. 같은 절대경로가 두 의미 | 해결 (§2.3, Phase B) |
| P3 | SDK 마운트 지점이 소스 트리 안 | `docker-compose.yml:60` 이 `${NOVATEK_SDK_DIR}` → `/workspace/novatek/tool`. `./:/workspace` 와 겹쳐 빈 스텁이 재생성된다. `run_compiler.sh:13,18` · `run_simulator.sh:14,19` 가 그 아래 경로를 리터럴로 박아 놓았다 | 해결 (Phase A) |
| P4 | 변환 산출물이 소스 트리 안 | `novatek/convert/<model>/nvtai_tool/output/` 1.3G, `sigmastar/data/output/`. `runs/` 규약 밖 | 해결 (Phase A) |
| P5 | 변환 단계명이 칩마다 다르다 | compile·simulate·visualize vs convert·calibrate·simulate | **해결하지 않는다.** 단계는 벤더 도구가 정하는 것이라 칩 고유로 둔다. 통일하는 것은 진입점(`pipeline.sh`)과 마지막 `report` 만이다 |
| P6 | docker 설정이 이원화 | compose `npu-convert` 서비스 vs `sigmastar/docker/{build,run}.sh` 독립. 베이스도 다르다 (ubuntu:20.04 vs 18.04) | 해결 (§2.3 — 서비스 둘, 마운트 형태 하나) |
| P7 | export 가 두 축으로 흩어져 있다 | **헤드 축**: `export_custom.py`(Detect·SymmetricDetect, 293줄) · `export_dms.py`(DMSPose kpts, 207줄) · `export_symmetric.py`(38줄). **칩 축**: `export_sigmastar.py`(다른 브랜치). 한 칩에 export 가 셋이고, 두 번째 칩이 네 번째 파일을 더했다 | 해결 (§2.2, Phase B) |
| P8 | 문서가 흩어져 있다 | `docs/npu_{deploy,porting}_guide.md`(다른 브랜치) vs `sigmastar/docs/SETUP_GUIDE.md` | 해결 (Phase B) |

P7·P8 이 가리키는 `export_sigmastar.py` · `npu_*_guide.md` 는 `docs/npu-deploy-guide-per-chip`
브랜치에 있고 `master` 에는 아직 없다. **이 계획은 그 브랜치 머지를 전제한다.**

### 1.3 기능 요구

| # | 요구 |
|---|---|
| R1 | 칩을 추가해도 저장소 최상위가 변하지 않는다 |
| R2 | 칩 하나가 무엇을 제공해야 하는지 **한 곳에 계약으로** 적혀 있고, `<model_dir>` 의 내용이 정의돼 있다 |
| R3 | 벤더 SDK 와 변환 산출물이 소스 트리 밖에 있다 |
| R4 | `/workspace` · `/opt/npu-sdk` 가 어느 칩 컨테이너에서도 같은 것을 가리킨다 |
| R5 | export 의 **헤드 축과 칩 축이 직교한다** — 헤드를 하나 추가해도 칩별 파일을 건드리지 않고, 칩을 하나 추가해도 헤드별 파일을 건드리지 않는다 |

### 1.4 비기능 요구

| # | 요구 | 확인 방법 (§5) |
|---|---|---|
| N1 | **변환 결과 불변** | Novatek: `nvt_model.bin` · `cle_nvt_model.onnx` 바이트 동일. Sigmastar: 캘리브레이션이 벤더 도구 내부라 바이트 동일을 보장할 수 없다 — 고정 이미지 세트 시뮬 검출을 IoU 0.9 · conf 차 0.01 이내로 비교 |
| N2 | 각 단계가 독립적으로 되돌려진다 | 단계 = 커밋 1개. Phase B 3·4 단계는 경로 수정이 수십 곳이라 "단순 revert" 가 아니라 **되돌림 커밋** 이다 |

---

## 2. 목표 구조

```
npu/
  README.md              공통 규약 · 새 칩 추가 계약 · <model_dir> 매니페스트
  common/
    export/              헤드 로딩 + 칩 규약 적용 (§2.2)
    report/              시뮬 결과 비교·시각화 공통부
  novatek/
    pipeline.sh          compile → simulate → report
    steps/
    contract.py          칩 출력 규약 (§2.2)
    Dockerfile           ubuntu:20.04 + protobuf 3.6.1 (현 docker/Dockerfile.npu)
    README.md            SDK 취득(NAS_2 03_SDK/novatek) · 알려진 함정
  sigmastar/
    pipeline.sh          convert → calibrate → simulate → report
    steps/
    contract.py
    Dockerfile           ubuntu:18.04 (현 sigmastar/docker/Dockerfile)
    README.md
```

`chips/` 같은 분류 계층을 두지 않는다 — 이 저장소에 전례가 없고(`projects/` · `configs/`
모두 네임스페이스 아래가 평평하다), `npu/` 아래가 `common/` 하나와 칩 N개뿐이라 계층이 정보를
더하지 않는다. 디렉터리명은 벤더명이다 — CNN25·NT98530·SGS·sav83x·CVD-F410 이 문서마다
섞여 있어 세대명을 고르면 그 혼선을 이름에 박게 된다. 한 벤더에 NPU 세대가 둘 이상 필요해지면
(`novatek_release` 번들명에 CNN25 와 CNN30 이 같이 있다) 그때 `novatek/` 안에서 가른다.

### 2.1 `<model_dir>` 매니페스트 — 계약 1번의 실체

v1 은 `pipeline.sh <model_dir>` 를 "동일 시그니처" 라 했지만 `<model_dir>` 가 무엇인지
정의하지 않았다. 지금 두 칩은 다른 것을 받는다.

| | Novatek (실측 `corner_truck_v01_320/`) | Sigmastar (`run.sh`) |
|---|---|---|
| 모델 | `model/best.onnx` | onnx 파일 경로 (인자) |
| 칩 설정 | `gen_config.txt` · `sim_config.txt` | 후처리 모듈명 인자 (`yolov8_xyxy_320x320`) |
| 캘리브 이미지 | `cal_img/` + `make_cal_img.py` | `/workspace/data/input/images` 고정 |
| **출력** | **같은 디렉터리** `nvtai_tool/{config,input,output,tmp}` · `baseline_v1/` | `/workspace/data/output/<name>/` |

정의:

```
<model_dir>/                 입력만. 추적 여부는 configs 트리 규약을 따른다
  model.onnx                 npu/common/export 가 낸 파일
  cal/                       캘리브레이션 이미지
  chip.yaml                  칩 고유 설정 — novatek: gen_config/sim_config 로 변환되는 키,
                             sigmastar: post_proc 모듈명. 키 스키마는 npu/<vendor>/README.md
runs/npu/<vendor>/<model_dir 이름>/     출력만. pipeline.sh 가 여기에 쓴다
```

`pipeline.sh <model_dir>` 는 입력을 읽고 `runs/npu/` 에 쓴다. **입력 디렉터리에는 쓰지 않는다.**
Novatek 의 `baseline_v1/` 같은 결과 스냅샷은 `runs/npu/.../snapshots/<name>/` 로 간다.

### 2.2 export — 헤드 × 칩 두 축

v1 은 "칩마다 다른 것은 출력 규약 하나" 라 했다. 틀렸다. 이미 한 칩에 헤드별 export 가 셋이다.

| 축 | 값 | 무엇이 다른가 |
|---|---|---|
| **헤드** | `Detect` · `SymmetricDetect` · `DMSPose` | 모델 클래스(`CustomYOLO` · `SymmetricYOLO` · `DMSPoseYOLO`, `projects/src/engine/model.py` 에 이미 있다), 출력 텐서 개수·의미(kpts 유무), 후처리 디코딩 |
| **칩** | Novatek · Sigmastar | 후처리를 그래프에 얼마나 넣는가, 박스 포맷, 정규화, 채널 배치 |

두 칩의 규약 차이는 `export_sigmastar.py` docstring 이 정확히 적었다 — Novatek 은 `--skip_postprocess`
로 raw 2-출력(cls sigmoid · DFL conv)을 내고, Sigmastar 는 그래프 안에서 ① `xyxy` ② `/imgsz`
정규화 ③ `max_score` 채널 추가 ④ `permute(0,2,1)` 까지 마친 단일 `[1,N,C]` 를 낸다.
**이 넷은 전부 헤드 출력에 대한 후처리 변환이고 헤드가 무엇인지와 무관하다.** 그래서 직교가 가능하다.

```
npu/common/export/
  load.py        헤드 → 모델 클래스 (기존 세 export 스크립트의 로딩 부분을 모은다)
  export.py      export(model, contract) — 헤드의 export-mode 출력에 contract 를 적용해 onnx 로
npu/<vendor>/contract.py
  Contract(skip_postprocess: bool, box_format: 'cxcywh'|'xyxy', normalize: bool,
           max_score_channel: bool, layout: 'BCA'|'BAC', opset: int)
```

헤드가 규약을 만족시킬 수 없으면(예: DMSPose kpts 를 Sigmastar `[1,N,C]` 단일 출력에 넣는
방법이 아직 없다) `NotImplementedError` 로 **즉시** 실패한다 — CLAUDE.md 의 예외 규약대로다.
조용히 잘못된 onnx 를 내는 것이 지금의 실패 모드다 ("변환은 exit 0 으로 끝난다").

기존 `tools/export_{custom,dms,symmetric,sigmastar}.py` 는 Phase B-5 이후 `npu/common/export`
를 호출하는 얇은 진입점으로 줄이거나 제거한다. `export_custom.py` 의 FX-QAT 재구성 경로(`--fx_qat`)
는 D-010 보류 대상이라 그대로 두고 옮기지 않는다.

### 2.3 docker — 서비스 둘, 마운트 형태 하나

베이스가 다르다(ubuntu:20.04 vs 18.04, SGS SDK 는 18.04 를 요구). 이미지를 하나로 합치지
않는다. **합치는 것은 마운트 형태다.**

```yaml
# docker-compose.yml
npu-novatek:
  build: { context: ., dockerfile: npu/novatek/Dockerfile }
  volumes:
    - ./:/workspace
    - ${NOVATEK_SDK_DIR:?}:/opt/npu-sdk:ro
npu-sigmastar:
  build: { context: ., dockerfile: npu/sigmastar/Dockerfile }
  volumes:
    - ./:/workspace
    - ${SIGMASTAR_SDK_DIR:?}:/opt/npu-sdk:ro
```

어느 칩 컨테이너에서도 `/workspace` = 저장소 루트, `/opt/npu-sdk` = 그 칩의 SDK 다 (R4).
스크립트는 `$NPU_SDK`(=`/opt/npu-sdk`) 만 참조한다. `sigmastar/docker/{build,run}.sh` 는 compose
서비스로 대체돼 사라진다. 현 compose 가 `:?` 를 피한 이유(파일 전체 보간으로 `dair-train`
까지 파싱 실패)는 그대로 유효하므로, 실제로는 기본값 + command 검사 패턴을 유지한다 — 기본값은
`/tmp/<vendor>-sdk-unset` 처럼 **트리 밖** 이어야 한다. 이것이 P3 의 재발 방지다.

### 2.4 트리 밖으로 나가는 것

| 대상 | 이동처 | 근거 |
|---|---|---|
| 벤더 SDK | `<VENDOR>_SDK_DIR` → `/opt/npu-sdk` | P3. NAS_2 `03_SDK/{novatek,sigmastar}/` 에 원본 |
| 변환 산출물 | `runs/npu/<vendor>/<model>/` | P4. `runs` ignore 규칙이 그대로 먹는다 |
| `sigmastar/reference/` 758M · `sigmastar/data/` 1.1G | NAS 취득 또는 `runs/npu/` | 참조 모델은 산출물이다 |
| `novatek/nmqt/` 719M | **§6 결정 1** | D-010 보존 대상 |

---

## 3. 새 칩 추가 계약

**새 칩 = `npu/<vendor>/` 디렉터리 하나 + compose 서비스 하나. 저장소 최상위는 건드리지 않는다.**

| # | 항목 | 정의 |
|---|---|---|
| 1 | `pipeline.sh <model_dir>` | §2.1 매니페스트를 읽고 `runs/npu/<vendor>/` 에 쓴다. 내부 단계는 칩 고유 |
| 2 | `contract.py` | §2.2 `Contract` 인스턴스. 만족시킬 수 없는 헤드는 `NotImplementedError` |
| 3 | `Dockerfile` | `/workspace` = 저장소 루트, `/opt/npu-sdk` = SDK 를 전제 |
| 4 | `<VENDOR>_SDK_DIR` | `.env.example` 등재, compose 서비스 `npu-<vendor>` 가 `/opt/npu-sdk` 로 마운트 |
| 5 | `README.md` | NAS 취득 경로, `chip.yaml` 키 스키마, 알려진 함정 |
| 6 | `steps/report` 연결 | 시뮬 결과를 `npu/common/report/` 가 읽는 형식으로 내놓는다 |

---

## 4. 이행

### Phase A — 완료 (2026-09-07, `42b169a`)

두 단계 다 **코드 변경이 있었다.** v1 의 "코드 변경 없이" 는 틀렸다. 두 단계가 래퍼의 같은
경로 해석 블록을 고치므로 한 커밋으로 갔다.

| 단계 | 작업 | 실제 작업량 | 효과 |
|---|---|---|---|
| A-1 | Novatek 래퍼의 입출력 분리 — 출력을 `runs/npu/novatek/<model>/` 로 | `run_compiler.sh` · `run_simulator.sh` · `visualize_sim.py` 확인, `visualize_dms_sim.py` 같은 패턴으로 추정: `WORK_DIR="$SCRIPT_DIR/$MODEL_DIR"` 를 입력/출력 둘로 가른다. 기존 작업 디렉터리 6개는 `nvtai_tool/` · `baseline_v1/` 만 `runs/npu/` 로 옮긴다 | 1.3G 가 트리에서 빠진다 (P4) |
| A-2 | SDK 마운트를 `/opt/npu-sdk` 로 | compose 1줄 + `TOOL_ROOT` · `PROTOBUF_LIB` 리터럴 4곳(compiler 2 · simulator 2)을 `$NPU_SDK` 기반으로 | 빈 스텁 재생성이 멈춘다 (P3) |

A 는 `novatek/convert/` 위치를 바꾸지 않는다. 디렉터리 이동은 B 에서 한 번에 한다.

### Phase B — 다음 칩이 정해지면

세 번째 칩이 실제로 정해지기 전에는 하지 않는다. `chips/` 를 뺀 것과 같은 논리 — 현재 없는
필요를 위해 구조를 미리 두지 않는다. 다만 NAS_2 에 후보 셋이 있으므로 "영원히 안 한다" 가 아니라
**트리거가 있는 보류** 다.

| 단계 | 작업 | 실제 작업량 | 되돌리기 |
|---|---|---|---|
| B-1 | `<model_dir>` 매니페스트 + `pipeline.sh` 규약 확정, `npu/README.md` | 문서 + Novatek `pipeline.sh` 가 `chip.yaml` 을 읽어 `gen_config.txt` 를 생성 | revert |
| B-2 | `novatek/convert/` → `npu/novatek/` | 이동 + A-1·A-2 를 거친 래퍼의 `$SCRIPT_DIR` 참조 점검 | 되돌림 커밋 |
| B-3 | `sigmastar/converter` · `docker` → `npu/sigmastar/`, `/workspace` 통일 | **32곳 / 10파일 하드코딩 수정.** `sigmastar/docker/{build,run}.sh` 는 compose 서비스로 대체. 계획 중 가장 큰 편집 | 되돌림 커밋 |
| B-4 | compose 에 `npu-novatek` · `npu-sigmastar` (§2.3) | 서비스 2개, 기본값 트리 밖 | revert |
| B-5 | export 2축 분리 (§2.2) | `npu/common/export/{load,export}.py` + `contract.py` 2개. 기존 4개 스크립트를 얇은 진입점으로. **설계가 필요한 유일한 단계** — DMSPose × Sigmastar 조합은 `NotImplementedError` 로 시작한다 | revert |
| B-6 | 문서 통합 | `docs/npu_*_guide.md` 의 칩별 절을 `npu/<vendor>/README.md` 로, 공통 절을 `npu/README.md` 로 | revert |

v1 의 순서에서 바꾼 것: sigmastar 이동(B-3)을 매니페스트(B-1)와 compose(B-4) 사이에 두었다 —
32곳을 고칠 때 한 번에 `$NPU_SDK` · `runs/npu/` · 매니페스트 규약으로 바꾸는 것이 두 번
고치는 것보다 낫다.

---

## 5. 검증 (N1)

**Phase A 에서 실제로 돌려 확정한 사실이다.** v2 초안이 두 가지를 잘못 적었고 여기서 고친다.

| 칩 | 기준 | 비교 대상 | 판정 | 상태 |
|---|---|---|---|---|
| Novatek | `corner_truck_v01_320` 입력을 **새로 컴파일한** `nvt_model.bin` | 이행 전후 `sdk/nvt_model.bin` | **바이트 동일** | Phase A 통과 (`5a8043a0…`) |
| Sigmastar | `sigmastar/reference/yolov8/network/onnx/CVD_xyxy_refine.onnx` | 고정 이미지 세트의 시뮬 검출 | 박스 IoU ≥ 0.9, conf 차 ≤ 0.01, 개수 동일 | Phase B-3 에서 |

### v2 초안의 오류 두 가지

**1. `cle_nvt_model.onnx` 는 컴파일 단계 산물이 아니다.** `run_compiler.sh` 만 돌리면 12개
파일 7.4MB 가 나온다 (`debug/model_param/quant/` 3, `gentool/` 8, `sdk/nvt_model.bin` 1).
`cle_nvt_model.onnx` 와 `simtool/` 은 시뮬레이터가 만든다 — 전체 파이프라인 후에는 820개
154MB 다. 컴파일 단계 검증 대상은 `sdk/nvt_model.bin` 하나다.

**2. 저장된 출력을 기준선으로 쓸 수 없다.** `corner_truck_v01_320/nvtai_tool/output/` 에
들어 있던 `nvt_model.bin`(`9ae8c067…`)은 **현재 SDK 로 재현되지 않는다** — 구 툴체인 산물이다.
같은 입력을 `.env` 가 가리키는 SDK v02.06 으로 새로 컴파일하면 `5a8043a0…` 이 나오고, 이는
`corner_v0206/` 에 저장된 출력과 바이트 동일하다 (두 디렉터리의 `deploy_float.onnx` ·
`deploy.onnx` 해시가 같다). 즉 **컴파일러는 결정적이며**, 기준선은 저장물이 아니라 이행 직전에
새로 만든 빌드여야 한다.

`baseline_v1/nvt_model.bin`(`50a84b00…`)도 세 값 중 어느 것과도 다르다 — 또 다른 시점의
스냅샷이다.

### 릴리스 수치의 출처 (2026-09-07 확인)

`tools/customEval.py --eval board` 의 Corner 0.481 이 어느 빌드에서 나왔는지 추적했다.
**결론: 이 저장소 밖에서 만들어졌고, `novatek/convert/corner_truck_v01_320/` 로는 재현되지
않는다.**

| 대상 | 크기 | sha256(앞 16) |
|---|---:|---|
| 릴리스 `network/nvt_model.bin` (NAS_2) | 4,703,568 | `77392745…` |
| `corner_truck_v01_320` 저장 출력 (구 SDK) | — | `9ae8c067…` |
| SDK v02.06 재컴파일 = `corner_v0206` 저장 출력 | 3,947,956 | `5a8043a0…` |
| `baseline_v1` 스냅샷 | — | `50a84b00…` |

네 값이 모두 다르다. 릴리스 `gen_config.txt` 가 이유를 보여준다 — 경로가 이 저장소가 아니다.

```
[path/model_dir]   = /home/ncn/Developments/data/weights/nt98530/novatek/bsd_f606a_corner/v2/…
[path/ref_img_dir] = /home/ncn/Developments/data/weights/_pipeline_inputs/bsd_f606_1t_corner/calib/jpg
```

**캘리브레이션 이미지 세트가 다르다.** 양자화 결과가 달라지므로 바이너리가 다른 것은 당연하다.
ONNX 도 다르다 — 릴리스 `bsd_f606a_corner_320x320.onnx`(`03789a9c…`)와 변환 디렉터리의
`deploy_float.onnx`(`743c7619…`)는 **크기가 12,070,702 로 같고 해시만 다르다**. 같은 모델을
다시 export 한 것으로 보이나 확증은 아니다.

MANIFEST 가 "board 가 onnx 보다 mAP +0.021 — 통상과 반대 방향이라 원인 미확인 (D3)" 라고
남긴 이상 현상이 있는데, **캘리브레이션 세트 차이가 후보 설명이다.** 단정하지 않는다 — 보드
바이너리와 로컬 onnx 평가가 서로 다른 캘리브레이션을 거쳤다는 사실만 확인됐다.

B-5 는 export 를 바꾸므로 그 직후 이 비교를 한 번 더 돈다.

---

## 6. 정해야 할 것

| # | 결정 | 선택지 | 필요 시점 |
|---|---|---|---|
| 1 | `novatek/nmqt/` 719M 의 처지 | (a) `npu/novatek/nmqt/` 로 이동, 계속 gitignore (b) 트리에서 빼고 NAS_2 `03_SDK/novatek/` 취득으로 돌림. D-010 이 보존을 요구하는 것은 **코드·config** 이고 툴킷 바이너리는 재취득 가능하다 | B-2 |
| 2 | `sigmastar/` 의 추적 범위 | (a) `converter/` `docker/` `docs/` 128K 만 추적 (b) 계속 통째로 gitignore. 계약(§3)이 `npu/sigmastar/` 를 요구하므로 (a) 가 자연스럽다 | B-3 |
| 3 | 다음 칩 | NAS_2 에 `allwinner` `nextchip` `rockchip`. **어느 것이 다음인지가 Phase B 의 트리거다** | Phase B 착수 조건 |
| 4 | `docs/npu-deploy-guide-per-chip` 머지 | P7·P8 이 그 브랜치 파일을 가리킨다. 먼저 머지되지 않으면 B-5·B-6 의 대상이 `master` 에 없다 | B-5 전 |
| 5 | 릴리스 `Corner_Truck/v01` 의 재현 경로 | 출하 바이너리가 저장소 밖(`~/Developments/…`)에서 다른 캘리브레이션 세트로 만들어졌다 (§5). (a) 그 캘리브 세트를 NAS 에서 찾아 `corner_truck_v01_320/` 에 복원해 재현 가능하게 만든다 (b) 재현 불가를 MANIFEST 에 명시하고 다음 릴리스부터 저장소 안에서 만든다. **(b) 를 하더라도 다음 릴리스 규약은 정해야 한다** | 다음 Corner 릴리스 전 |

---

## 7. 범위 밖

- `projects/src/quantization/` · `quant_tools/` · `tools/nova_qat_pipeline.py` — D-010 보류.
  `export_custom.py --fx_qat` 경로도 옮기지 않는다 (§2.2).
- `configs/releases/**/network/` 의 NPU 바이너리 — `MANIFEST` 규약이 자리를 정한다.
- `sigmastar/.claude/` · `sigmastar/.rollback/` — 별도 하위 프로젝트의 잔재. B-3 에서 옮기지
  않고 버린다 (`.rollback/` 은 2026-06 스냅샷 4개, 53M).
