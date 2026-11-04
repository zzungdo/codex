# 개발·릴리스 절차

적용: 실험 설정·학습 모델·코드 변경의 개발과 릴리스 · 기준: 2026-09-28

브랜치 → 데이터·설정 → 학습·평가 → 포팅·검증 → 릴리스 기록 → 병합 순서로 진행한다.
명령·옵션은 [사용법](usage.md), 환경은 [설정 문서](setup.md), 상세 보드 절차는 [NPU 문서](npu.md)를 따른다.

## 작업 브랜치

호스트 저장소 루트, 작업 트리가 정리된 상태에서 로컬 기준 브랜치로 새 브랜치를 만든다.

```bash
git switch -c feat/BSD_F606-Front_Truck-v03 master
```

기준 브랜치가 `main`인 저장소에서는 `master`를 `main`으로 바꾼다.
이미 작업 중인 브랜치가 있으면 그 브랜치를 사용한다.
기준 브랜치에 직접 커밋하지 않는다.

| 대상 | 이름 |
|---|---|
| 모델 작업 | `<type>/<PRODUCT>-<Camera>-<Version>` |
| 일반 작업 | `<type>/<topic>` |
| type | `fix`, `feat`, `refactor`, `test`, `docs`, `chore` |

## 데이터·설정 준비

| 위치 | 역할 |
|---|---|
| `/mnt/NAS_3/02_datasets/` | 원본 데이터셋 보관 |
| `/mnt/NAS_2/02_Trained_weights/` | 기존 릴리스 가중치 원본·백업; 현재 작업에서는 읽기 전용 |
| 호스트 `DATA_ROOT` | 작업용 데이터셋 → 컨테이너 `/dataset` |
| `configs/models/` | 모델 구조 YAML |
| `configs/datasets/` | 데이터셋 경로·분할·클래스 정의 |
| `configs/experiments/<P>/<C>/<V>/exp.yaml` | 수정 가능한 실험 |
| `configs/releases/<P>/<C>/<V>/` | 동결된 릴리스 설정·MANIFEST |
| `configs/archive/` | 현재 실행에서 참조하지 않는 과거 설정 |

데이터 반입 전 여유 공간과 실제 마운트를 확인한다.
ROI·crop·resize는 실험의 `exp_params`에 두고, 같은 데이터셋을 여러 제품에서 사용할 수 있게 한다.
예시의 버전 디렉터리는 새 작업용 이름으로 바꾼다. 이미 존재하는 버전에는 덮어쓰지 않는다.

호스트 저장소 루트:

```bash
mkdir -p configs/experiments/BSD_F606/Front_Truck/v03
cp -n configs/releases/BSD_F606/Front_Truck/v01/exp.yaml \
  configs/experiments/BSD_F606/Front_Truck/v03/exp.yaml
```

복사한 설정의 모델·데이터·클래스·ROI·학습 옵션을 편집한다.
지원 `model_name`은 custom, symmetric, dair, dms_pose이며
[MODEL_REGISTRY](../projects/src/engine/model.py)가 기준이다. 미등록 이름은 오류다.

## 학습·평가

[학습 환경](setup.md#학습-환경-실행)의 컨테이너 `/workspace`에서 실행한다.

```bash
uv run python3 tools/train_custom_trainer.py \
  -w configs/experiments/BSD_F606/Front_Truck/v03/exp.yaml -d 0
```

출력은 `runs/detect/BSD_F606/Front_Truck/v03/`다.
파일명이 `exp.yaml`이고 설정이 정규 실험·릴리스 경로에 있을 때 run 디렉터리가 유도된다.
다른 경로에서는 `project/name`을 명시하고 실제 출력 경로를 확인한다.

학습 결과의 `exp.yaml`에는 해석된 설정이 저장된다.
`_meta_`, `_resolved_`, `_run_`처럼 밑줄로 시작하는 최상위 키는 기록용이다.
다음 버전 설정으로 복사하면 출력 경로가 새 위치에서 유도되며, 원본 설정의 주석·근거는 별도로 보존한다.

| 평가 | 목적·산출물 |
|---|---|
| `validate.py` | 학습 중·직후 PT 검증 |
| `customEval.py` | 릴리스 ONNX·보드 결과 평가, `evaluation.json`과 그래프 |
| `customEval.py --dry-run` | 해석된 경로·옵션·파생 출처 확인 |
| `customEval.py --save-pred` | 예측 JSON 보관 |

두 평가 코어의 수치는 직접 비교하지 않는다.
`eval:`의 GT·ROI·클래스는 실험·데이터셋 설정에서 파생되며 명시값이 우선한다.
각 variant는 `rep`를 가져야 한다.
`conf`는 예측 생성 임계, `cm_conf`는 혼동행렬 기준이며 서로 다르다.
보드와 ONNX의 화각·좌표·클래스·필터 조건은 [NPU 평가 규칙](npu.md#평가와-검증-기준)을 따른다.

평가 결과에는 지표, GT 수, `num_dropped_by_cat_map`, 실행 파라미터와 판정을 확인한다.
`eval.target`을 지정한 경우 목표 미달은 종료 코드 1로 알린다.

## 검증 기록

학습 컨테이너 `/workspace`:

```bash
uv run pytest tests/ -q -rs
uv run ruff check .
```

전체 ONNX 기준 평가가 필요한 경우:

```bash
EVAL_BASELINE_FULL=1 uv run pytest tests/test_eval_baseline.py -q -rs
```

| 변경 범위 | 추가 검증 |
|---|---|
| 학습·resume | 대표 설정과 실제 데이터로 짧은 학습·재개 |
| 모델·export | PT 로딩, 입력·출력 규약, ONNX Runtime 수치 비교 |
| 전처리·평가 | crop·좌표·클래스 및 고정 데이터 지표 |
| NPU | SDK 모드별 덤프·검출 비교, 실물 보드 별도 확인 |

기록에는 날짜·커밋·의존성 lock·입력 자산·설정·장치·결과를 남긴다.
자산이 없어 skip된 검사는 `not_run`으로 구분한다.
테스트 개수나 이전 측정값을 현재 합격의 근거로 대신 사용하지 않는다.

## 릴리스 기록

검증한 설정을 `configs/releases/<P>/<C>/<V>/`로 복사하고 `MANIFEST.yaml`을 작성한다.
산출물은 `analysis/`, `network/`, `weight/`에 배치한다.
대상 버전이 이미 있으면 새 버전을 만든다. 릴리스 `exp.yaml`은 동결한다.

| MANIFEST 항목 | 기록할 내용 |
|---|---|
| `nas_source`, `config`, `dataset` | 원본 위치, 실험·데이터 설정 |
| `run`, `recorded` | 실제 실행·기록 날짜, 커밋·환경 |
| `artifacts`, `npu` | PT·ONNX·보드 파일과 해시·변환 조건 |
| `metrics` | 데이터·평가기·필터 조건이 있는 지표 |
| `as_run_differs`, `issues` | 설정 차이, 출처 부재, 미검증 항목 |

이 표는 기록 항목 안내이며 모든 키가 필수인 고정 스키마는 아니다.
설정 출처가 없으면 추정 `exp.yaml`을 만들지 않고 MANIFEST에 `no-config` 등 사유를 남긴다.
기존 릴리스와의 차이·확인 근거는 `_meta_`에도 기록한다.

## 커밋·병합

검증 결과와 변경 내용을 검토한 뒤 커밋한다.
메시지는 `type: English title`, 본문은 변경 내용·이유를 영어로 쓴다.
타입은 fix → feat → refactor → test → docs → chore 순서로 처음 해당하는 것을 사용한다.
기존 D-019의 공동 작성자 표기와 병합 규칙은 [설계 결정 기록](archive/decisions.md)을 따른다.

작업 결과는 브랜치에서 검토한다. master/main 병합·태그·리모트 푸시는 별도 릴리스 작업이다.
에이전트는 실제 리모트 푸시를 실행하지 않으며 담당자가 수행한다.
