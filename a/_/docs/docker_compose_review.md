# Docker Compose 구성 검토

- **프로젝트명**: DAIR-YOLO
- **작성일**: 2026-07-27 (KST)
- **성격**: 설계 검토 문서 (실제 Dockerfile / compose 파일은 후속 사이클 산출)
- **확정 방향** (사용자 결정):
  1. NPU 컴파일러 툴체인은 **별도 변환 서비스로 분리** (학습 서비스와 이미지 격리)
  2. 이미지 빌드는 **소스 마운트 + uv** 방식 (editable, 코드 수정 즉시 반영)
  3. 이번 산출은 **검토 문서만**

---

## 1. 목적 / 범위

수동으로 운영하던 개발 컨테이너(`deployment_guide.md`의 "Docker 내 root git 작업" 언급)를
**docker compose로 정식화**한다. 대상 워크로드:

- **학습·평가·ONNX export** — PyTorch 2.5.1+cu121, 2×RTX 4090, uv 기반
- **NPU 변환** — Novatek CNN25 컴파일러/시뮬레이터 (protobuf 3.6.1 로컬 빌드 의존, GPU 불필요)

두 워크로드는 의존성 성격이 완전히 달라(전자=CUDA/torch, 후자=C++/protobuf) **service 2개로 분리**한다.

---

## 2. 호스트 환경 (compose 적용 준비 완료)

| 항목 | 값 | 비고 |
|------|-----|------|
| Docker | 29.6.2 | Compose v2 내장 (`docker compose`) |
| **default-runtime** | **`nvidia`** | `/etc/docker/daemon.json`에 이미 설정 → GPU 접근 기본 활성 |
| GPU | 2× RTX 4090 24GB | |
| NVIDIA Container Toolkit | 설치됨 (`nvidia-ctk`) | |
| docker data-root | `/data/docker` | 이미지·볼륨 저장 위치 (루트 파티션 아님) |
| repo 위치 | `/home/ncn/Workspace/jychoi/ultralytics_projects` | |
| 호스트 데이터셋 후보 | `/home/ncn/Workspace/data` (lpr-*), 그 외 `/dataset`은 미확인 | §5 참조 |

> default-runtime이 이미 nvidia이므로 compose에 `runtime: nvidia`를 명시하지 않아도 GPU가 붙는다.
> 다만 이식성·명시성을 위해 `deploy.resources`로 GPU를 선언하는 것을 권장 (§6).

---

## 3. 서비스 구성안 (2 service)

### 3.1 `dair-train` — 학습 / 평가 / export

| 항목 | 값 |
|------|-----|
| base image | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (torch cu121 휠은 CUDA 자체 번들 → runtime로 충분) |
| **python** | **3.12 필수** — `uv.lock` 이 `onnxruntime-gpu 1.24.3`(cp311/cp312 전용) 을 핀. base 의 py3.10 대신 **`UV_PYTHON=3.12`** 로 uv 관리형 CPython 3.12 사용 (base 무관). ✅ 실증: py3.12/torch2.5.1+cu121/cuda True/2GPU |
| 시스템 패키지 | `build-essential`(pycocotools), `libgl1 libglib2.0-0`(opencv), `git`, `curl` |
| 패키지 매니저 | `uv` (astral) — **`UV_INSTALL_DIR=/usr/local/bin`** 로 설치(non-root user 1000 접근). `uv sync`로 `/opt/venv` 생성, torch는 `[tool.uv.sources]`의 cu121 인덱스 라우팅 |
| workdir | `/workspace` (repo 마운트 지점 — data_yaml의 `/workspace/...` 절대경로와 정합) |
| 실행 | 상시 서비스 아님. `command: sleep infinity` 후 `docker compose exec`로 학습 명령 투입하거나, one-shot `run` |
| GPU | 2장 모두 노출 (§6) |
| shm | `shm_size: '16gb'` 이상 (DataLoader workers RAM 공유 — 없으면 bus error) |

### 3.2 `npu-convert` — Novatek NPU 컴파일/시뮬 (분리)

| 항목 | 값 |
|------|-----|
| base image | **`ubuntu:20.04`** (SDK 바이너리가 ubuntu20 빌드 → glibc/ABI 일치. GPU 불필요, CPU 실행) |
| 시스템 패키지 | `build-essential`, `autoconf`, `libtool`, `curl` (protobuf 3.6.1 소스 빌드용) |
| 특수 의존성 | **protobuf 3.6.1**을 로컬 prefix로 빌드 → `libprotobuf.so.17` (SDK tarball 미포함, requirements.txt 절차 그대로). opencv는 tarball 동봉. `run_compiler.sh`가 `LD_LIBRARY_PATH` 자동 설정 |
| 바이너리 주입 | SDK 툴체인(`compiler.nvtai`/`simulator.nvtai`)은 **볼륨 마운트로 주입** (이미지에 굽지 않음). 출처·추출법 §8.1 |
| workdir | `/workspace` |
| GPU | 불필요 (`runtime` 명시로 nvidia 상속 피하려면 CPU-only 확인 후 조정) |

> **필요 구성요소 상세는 §8.1** — SDK 툴체인 전체(현재 부재) + protobuf 3.6.1 + per-model 변환 디렉토리. 마운트 소스(SDK 배포본 위치)만 확보하면 됨.

---

## 4. 이미지 빌드 방식 — 소스 마운트 + uv

```
repo(host)  ──(bind mount, rw)──►  /workspace (container)
                                     └─ uv sync  →  /workspace/.venv (컨테이너 전용)
```

- 코드는 마운트 → `pip install -e .` / editable 효과를 마운트가 대체 (수정 즉시 반영)
- **주의**: `.venv`는 호스트에 남기지 말 것. 호스트/컨테이너 glibc·아키텍처 차이로 깨짐
  → `.venv`를 **named volume**으로 덮어써 host bind에서 분리 (`- dair-venv:/workspace/.venv`)
  → 또는 `UV_PROJECT_ENVIRONMENT=/opt/venv`로 마운트 밖에 생성
- torch/torchvision은 `[tool.uv.sources]`가 cu121 인덱스로 라우팅 → base CUDA 버전(12.1)과 일치 확인
- `ultralytics==8.4.6` 고정 유지 (fork 정합 — pyproject 주석 참조)

---

## 5. 볼륨 마운트 — 정규화 완료 (2026-07-27)

`configs/datasets/legacy/**/*.yaml`(총 38개)의 `path:`를 **2계층 규약으로 정규화 완료** → 상대경로·`/workspace/dataset`·`/mnt/HDD_2` 분산 제거. 이제 **소스 데이터 마운트가 1줄**로 축소되었다.

### 데이터 경로 규약 (SSOT)
| 계층 | 정규 위치 | 성격 | 예 |
|------|-----------|------|-----|
| **소스 데이터셋** | `/dataset/<name>` | 외부 원본 (호스트 `~/Workspace/data` 마운트) | `/dataset/CVD/CVD_1t`, `/dataset/DMS_stage_1`, `/dataset/260401_4K_merged` |
| **생성 데이터** | `/workspace/data/<name>` | 스크립트 생성 (repo rw 마운트, `.gitignore data/`) | `/workspace/data/ncnvox_cmd`, `/workspace/data/ncnvox_kws` |
| **더미** | `/workspace/<dummy>` | `create_*_dummy_dataset.py`가 컨테이너 생성 | `/workspace/dummy_dataset`, `/workspace/dms_dummy_dataset/*` |

정규화 내역: 소스 8개(`CVD_1t/5t`, `dashcam_test`, `dms_stage1/2`, `FRONT/260401_4K_merged`, `REAR/260401_QHD_REAR`, `FRONT/TLDS_rework_add`) + 생성 3개(`ncnvox_cmd`, `ncnvox_kws_full`, `ncnvox_kws`) = **11개 yaml `path` 수정**. 상대경로 잔존 0, 38개 파싱 통과.

### 마운트 (compose `volumes`) — 1줄로 축소
```yaml
volumes:
  - ./:/workspace                          # repo (소스 + 생성 데이터 /workspace/data + runs/)
  - dair-venv:/workspace/.venv             # venv 격리 (named volume)
  - /home/ncn/Workspace/data:/dataset:rw   # 전 소스 데이터셋 — 단 1줄 (rw: ultralytics .cache 허용)
# YOLO_DATASETS_DIR 불필요 — 상대경로 0개
```
- 소스 데이터셋은 전부 `/dataset/<name>` → 호스트 `~/Workspace/data/<name>`. 예: `~/Workspace/data/CVD/CVD_1t`, `~/Workspace/data/DMS_stage_1`, `~/Workspace/data/260401_4K_merged`
- 생성 데이터(ncnvox)·더미는 repo 마운트(`/workspace`) 안 → 별도 마운트 불필요, 쓰기 가능·영속
- `runs/`도 repo 마운트에 포함 (gitignore라 오염 없음)

**주의**:
1. compose는 `~` 확장 안 함 → **절대경로** `/home/ncn/Workspace/data` 또는 `.env` `${DATA_ROOT}`
2. `/dataset`는 **`rw`** (결정) — ultralytics가 `<split>.cache`를 데이터셋 폴더에 씀. 원본 실수 변경 위험은 감수
3. **실물 미배치**: 현재 `~/Workspace/data`엔 `lpr-*` 3개뿐. CVD·DMS·260401_* 등은 `/dataset/<name>` = `~/Workspace/data/<name>` 구조로 학습 전 배치 필요 (별도 작업)
4. `FRONT/TLDS_rework_add` → `/dataset/TLDS_rework_add`로 정규화(구 `/mnt/HDD_2/...DashCAM/Front/`). 해당 실물을 `~/Workspace/data/TLDS_rework_add`로 옮기거나 심볼릭 링크 필요

---

## 6. GPU 노출

default-runtime이 nvidia라 최소 설정으로도 붙지만, 명시 권장:

```yaml
# dair-train service
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all          # 또는 device_ids: ['0','1']
          capabilities: [gpu]
```
- 단독 GPU 학습은 `-d 0`, 분산은 `dist_train_custom.sh 2` — 컨테이너에 2장 노출 후 스크립트 인자로 제어
- `npu-convert`는 GPU 불필요 → CPU-only 확인 후 `deploy.resources` 생략 (default-runtime nvidia 상속만 주의)

---

## 7. 기타 런타임 설정

| 설정 | 값 | 이유 |
|------|-----|------|
| `shm_size` | `16gb`+ | DataLoader multiprocessing 공유메모리 (기본 64MB로는 bus error) |
| `ulimits.memlock` | `-1` | pinned memory |
| `workers` | **≤ 4 유지** | 호스트 RAM 125GB 제약 (README 「알려진 버그 / 제약」) — 컨테이너 mem_limit 설정 시에도 준수 |
| `.dockerignore` | 필요 | `runs/`, `data/`, `dataset/`, `.git/`, `*.pt`, `*.onnx`, `.venv`, `novatek/tool` 제외 (빌드 컨텍스트 축소) |
| 사용자 | non-root 고려 | 현재 컨테이너 root 작업 → 호스트 `.git/logs` Permission denied 발생 이력. `user: "1000:1000"` 또는 entrypoint chown |
| `env` | `PYTHONPATH=/workspace` | dist 스크립트가 이미 설정하나 exec 편의상 서비스 env로도 |

---

## 8. 확인 완료 / 결정 사항

| # | 항목 | 결정 |
|---|------|------|
| 2 | `/dataset` 실경로 | **`~/Workspace/data`** → `/dataset`으로 마운트 (§5) |
| 3 | `../../data` resolve | **`~/Workspace/data`** → `/workspace/data`로 마운트, 이중 마운트로 흡수 (§5) |
| 4 | 컨테이너 UID/GID | **권장대로 non-root** — `user: "1000:1000"`(호스트 ncn) 매핑, git push 권한·산출물 소유권 문제 회피 |
| 5 | `runs/` 출력 위치 | **repo 디렉토리 내**(`./runs`) — repo 마운트에 자연 포함, 별도 볼륨 불필요 |
| 1 | NPU 툴 바이너리 | **출처 확정** — NAS SDK zip의 `Tool/ubuntu20/release.tar.bz2` (§8.1). 마운트 매핑 검증 완료 |

### 8.1 NPU 변환 서비스(`npu-convert`)에 필요한 것 (#1 파악 결과)

변환 wrapper(`novatek/convert/run_compiler.sh`·`run_simulator.sh`)가 **하드코딩으로 요구**하는 경로:
`TOOL_ROOT = /workspace/novatek/tool/release/release/ai_tool/novatek/novaic/toolchain`

**(A) Novatek CNN25 SDK 툴체인** — `novatek/tool/` 트리 전체. **출처 확정**:
```
NAS: /mnt/NAS_2/03_SDK/novatek/NPU/250307_Novaic_CNN25_v02.05.2503042_CNN30_v04.00.250304.04_release_exe.zip
 └─ .../Tool/ubuntu20/release.tar.bz2   ← 이 tarball이 툴체인 본체 (ubuntu20 빌드)
```
추출 매핑 (**wrapper 하드코딩 경로와 정확히 일치 검증됨**):
```
tar xjf release.tar.bz2 -C <sdk_extract>/novatek/tool/
 → <sdk_extract>/novatek/tool/release/release/ai_tool/novatek/novaic/toolchain/closeprefix/bin/compiler.nvtai
compose: - <sdk_extract>/novatek/tool:/workspace/novatek/tool
```
tarball의 top이 `release/release/...` 이중 구조라 `novatek/tool/`에 풀면 wrapper의 `TOOL_ROOT=/workspace/novatek/tool/release/release/ai_tool/novatek/novaic/toolchain` 과 그대로 맞는다. 구성:
| 구성요소 | 경로(TOOL_ROOT 기준) | 용도 |
|---------|---------------------|------|
| 컴파일러 | `closeprefix/bin/compiler.nvtai` | ONNX → nvt_model.bin |
| 시뮬레이터 | `closeprefix/bin/simulator.nvtai` | 양자화 정합 검증 |
| ONNX 프론트엔드 | `closeprefix/bin/compiler/frontend/onnx-onnx/` (+ `eff_eval_lib/`) | onnx2nova 변환 |
| 엔진 DLL | `closeprefix/bin/compiler/eng_dll/530/` | chip **530**(NT98530=CNN25) |
| 시뮬 DLL | `closeprefix/bin/simulator/emu530/` | 시뮬레이터 백엔드 |
| 벤더 opencv | `repo/proprietary/opencv/lib` | 이미지 I/O |

**(B) protobuf 3.6.1** — `novatek/tool/deps/protobuf/lib`(`libprotobuf.so.17`). SDK에 미포함이면 이미지 빌드 시 소스빌드(requirements.txt 절차), 포함이면 마운트.

**(C) 변환 대상 per-model 디렉토리** — `novatek/convert/<model>/`(gitignore `novatek/convert/*/`). 학습 서비스가 export한 산출물:
- `nvtai_tool/input/model/customer/<PATTERN>/deploy_float.onnx` — 입력 ONNX(raw export)
- `nvtai_tool/input/data/<PATTERN>/` — calibration/test JPG + `test_img_list.txt`
- `nvtai_tool/` — gen_config / sim_config
- 출력: `nvtai_tool/output/<PATTERN>/nvt_model.bin`

**남은 준비 작업 (blocker 아님, 사전 1회)**: NAS zip에서 `Tool/ubuntu20/release.tar.bz2`를 호스트로 추출 → `<sdk_extract>/novatek/tool/`에 풀어두기. 그 경로를 `npu-convert`에 `- <sdk_extract>/novatek/tool:/workspace/novatek/tool` 로 바인드 마운트. wrapper가 `/workspace/novatek/tool` 절대경로를 쓰므로 이 마운트 지점은 고정.

> `npu-convert`는 SDK 툴체인(대용량·바이너리)을 **이미지에 굽지 않고 마운트**로 주입 → 이미지는 protobuf 3.6.1 빌드 환경만 담아 가볍게 유지. protobuf만 이미지 빌드, opencv·컴파일러·시뮬레이터는 tarball 동봉분 사용.

---

## 9. 다음 단계 (승인 시)

1. NAS zip → `Tool/ubuntu20/release.tar.bz2` 추출·배치 (사전 1회, §8.1 A) — 모든 확인 항목 해소됨, blocker 없음
2. `tasks/todo.md` 에 작업 계획 작성
3. 산출: `Dockerfile.train`, `Dockerfile.npu`, `docker-compose.yml`, `.dockerignore`
4. 검증 — 빌드·`pytest tests/`·최소 학습 1 step·NPU wrapper 실행
5. `docs/deployment_guide.md` compose 섹션 반영, README 갱신 여부 확인
