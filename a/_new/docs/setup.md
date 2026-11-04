# 실행 환경 설정

적용: 학습·평가·분석 및 Novatek·Sigmastar 변환 환경 · 기준: 2026-09-29

## 환경

호스트는 Linux와 Docker Compose, GPU 작업에는 NVIDIA 드라이버·컨테이너 GPU 실행 환경이 필요하다.
실제 이미지·마운트 설정은 [docker-compose.yml](../docker-compose.yml), 의존성은
[pyproject.toml](../pyproject.toml)과 [uv.lock](../uv.lock)이 기준이다.

| 환경 | 실행 위치 | Python·주요 의존성 |
|---|---|---|
| 학습·평가·ONNX·분석 | `dair-train`, `/workspace` = 저장소 루트 | Python 3.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.6 |
| Novatek 변환 | `npu-convert`, `/workspace` = 저장소 루트 | 별도 SDK 환경, `/opt/npu-sdk` 마운트 |
| Sigmastar 변환 | `docker/sigmastar/run.sh`, `/workspace` = 호스트 `sigmastar/` | Python 3.7, 32bit 런타임, SGS SDK |

Ultralytics 버전은 고정한다. 서로 다른 Python·ABI를 쓰는 SDK 환경의 의존성을 학습 환경에 합치지 않는다.

## 환경변수

호스트 저장소 루트에서 최초 한 번 복사하고 실제 경로를 편집한다. 기존 `.env`가 있으면 유지한다.

```bash
cp -n .env.example .env
```

| 변수 | 목적·조건 |
|---|---|
| `DATA_ROOT` | 호스트 데이터셋 루트 → 학습 컨테이너 `/dataset`; 절대경로 |
| `NOVATEK_SDK_DIR` | Novatek 추출본 → `/opt/npu-sdk`; 하위에 `release/`가 있어야 함. 현재 호스트: `/home/ncn/Workspace/jdh/novatek_sdk/novatek/tool` (CNN25 v02.05.2503042) |
| `UID`, `GID` | 호스트 사용자와 산출물 소유권 정합; `.env`에 숫자로 기록 |
| `SGS_IMAGE` | Sigmastar wrapper 이미지; 기본 `sgs-convert:latest` |
| `SGS_TTY` | Sigmastar wrapper 터미널 옵션; 비대화 실행은 `-i` |

Compose는 `~`를 확장하지 않는다. `NOVATEK_SDK_DIR` 미설정은 Novatek 서비스 실행을 막지만
학습 서비스만 실행하는 것은 가능하다. SDK 추출 경로와 보드별 버전은 [NPU 문서](npu.md)를 따른다.

## 학습 환경 실행

호스트 저장소 루트:

```bash
docker compose build dair-train
docker compose up -d dair-train
docker compose exec dair-train uv sync --frozen --extra dev
docker compose exec dair-train bash
```

컨테이너 `/workspace`:

```bash
uv run python3 tools/train_custom_trainer.py --help
uv run pytest tests/ -q -rs
uv run ruff check .
```

| 항목 | 결과·목적 |
|---|---|
| `uv sync --frozen --extra dev` | lock 기준 설치, pytest·Ruff 포함 |
| `/opt/venv` | `dair-venv` 볼륨에 유지되는 학습 가상환경 |
| `/workspace/runs/` | 호스트 `runs/`와 공유되는 실행 산출물 |
| `--help` | CLI 실행 확인; 데이터·학습·보드 검증을 대신하지 않음 |

장시간 작업 전 짧은 실행으로 데이터·GPU·출력 경로를 확인한다.
기존 듀얼 GPU 호스트에서는 DataLoader `workers <= 4`를 사용한다.
테스트의 skip 사유와 기존 lint 오류를 확인하고 실행 결과를 기록한다.

## NPU 환경 실행

Novatek 변환 이미지는 Novatek 제공 이미지 `ubuntu20p04:202407013` 위에 만든다
([Dockerfile](../docker/novatek/Dockerfile)). SDK 바이너리가 요구하는 `libpython3.8`·`libprotobuf.so.17`과
Nova ONNX 변환 스크립트가 쓰는 Python 3.10·onnx·onnxruntime·onnxsim을 이 이미지가 제공한다.
순정 `ubuntu:20.04`에서 시작하면 컴파일러가 시작하지 않는다.

호스트 저장소 루트, 최초 한 번 NAS 이미지를 로드한다:

```bash
docker load -i /mnt/NAS_2/03_SDK/novatek/NPU/ubuntu20p04_202407013.tar
```

```bash
docker compose build npu-convert
docker compose up -d npu-convert
bash docker/sigmastar/build.sh
```

Sigmastar SDK를 `sigmastar/sdk/SGS_IPU_SDK_vQ_0.2.1/`에 준비한 뒤 소스를 반영한다.

```bash
bash docker/sigmastar/run.sh 'bash converter/sync_to_sdk.sh'
```

모델 변환은 [사용법](usage.md#포팅), 입력·설정·산출물 규격은 [NPU 문서](npu.md)를 따른다.

## 경로·권한 제약

- 데이터 YAML의 `path`는 `/dataset/<데이터셋>`을 쓴다. 호스트 NAS 경로를 넣지 않는다.
- `DashCAM`(NAS), `Dashcam`(로컬 설정) 등 대소문자를 실제 데이터·설정과 일치시킨다.
- 소스와 산출물은 호스트와 공유된다. 실행 사용자와 파일 소유권을 확인한다.
- SDK 누락·버전·공유 라이브러리 문제는 [NPU 문제 해결](npu.md#미확인-항목과-문제-해결)을 확인한다.
