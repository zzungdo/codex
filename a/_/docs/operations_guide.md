# 운영 가이드 — dev workflow

브랜치 → 데이터 → 개발 → 평가 → 릴리즈 → 머지 순. 2026-07-30 실측, 근거 경로 병기.

관련: [CLAUDE.md](../CLAUDE.md) · [diagrams.md](diagrams.md) · [decisions.md](decisions.md) ·
[technical_doc.md](technical_doc.md) ·
[debug/20260729_config_canonicalization.md](debug/20260729_config_canonicalization.md)

```
NAS_3 데이터셋 ─rsync─▶ ~/Workspace/data ─mount─▶ /dataset
                                                     │
[1] 브랜치 ─▶ [2] 데이터 ─▶ [3] 개발 ─▶ [4] 평가 ─▶ [5] 릴리즈 ─▶ [6] 머지 ─▶ tag
                            configs/     runs/       configs/       master
                            experiments/             releases/
                                                     + NPU 변환
                                                        ▲
                                          NAS_2 릴리스 가중치 원본 (ro)
```

---

## 0. 전제 — 마운트와 환경

| 마운트 | 종류 | 여유 | 역할 |
|---|---|---|---|
| `/mnt/NAS_3` | NFS rw, 28T | **493G (99%)** | 데이터셋 원본 · 영상 |
| `/mnt/NAS_2` | WebDAV **ro**, 1.3T | 509G | **릴리스 가중치 원본** |
| `/mnt/NAS_1` | NFS rw, 70T | 6.2T | 저장소 미참조 |
| `~/Workspace/data` | 로컬 NVMe | 1.2T | 작업 사본 → 컨테이너 `/dataset` |

**원본이 두 곳으로 갈라져 있다.** 데이터셋은 NAS_3, 릴리스 가중치는 NAS_2다 —
`MANIFEST.nas_source` 10개 전부가 NAS_2를 가리키고 NAS_3 `03_weights/` 에는 Vueroid만 있다.

**`.env` 가 없다.** compose 인라인 기본값으로 `dair-train` 은 동작하지만
`NOVATEK_SDK_DIR` 이 `/tmp/novatek-sdk-unset` 으로 떨어져 **`npu-convert` 는 실패한다.**
NPU 변환 전에 `cp .env.example .env` 후 SDK 경로 수정. compose는 `~` 를 확장하지 않는다.

호스트에 venv가 없다. 실행은 컨테이너 또는 `uv run`. `python`은 없고 `python3`만 있다.

---

## 1. 브랜치 따기

규약은 D-018. `master` 에 직접 커밋하지 않는다.

```bash
git switch master && git pull
git switch -c feat/BSD_F606-Front_Truck-v03
```

| 대상 | 이름 |
|---|---|
| 모델 작업 | `<type>/<PRODUCT>-<Camera>-<Version>` — configs 경로와 대응 |
| 그 외 | `<type>/<topic>` |

`type` 은 커밋 타입 6종과 같은 어휘 (`fix` `feat` `refactor` `test` `docs` `chore`).

> 원격(GitLab)에 브랜치 보호 규칙이 없다 — 이 규약은 합의이지 강제가 아니다.

---

## 2. 데이터 가져오기

### NAS_3 레이아웃

```
/mnt/NAS_3/
  01_videos/     eu · kr · multi · usa · youtube
  02_datasets/   DashCAM · CVD · BSD · Vueroid · Research · _staging   ← 정본
  03_weights/    Vueroid                                               (DAIR 없음)
  99_temp/
```

| 디렉토리 | 내용 |
|---|---|
| `DashCAM/` | `DMS_stage1` `DMS_stage2` `Front` `Front_traffic_light` `Rear` `Side` |
| `CVD/` | `Corner` `CVD_1t` `CVD_5t` `Rear` `Side` |
| `BSD/` | `Side` |
| `Vueroid/` | `lpr-{cls,det,det,gen,ocr,pose,pose}-v*` 7개 |

### 반입

**디렉토리명 대소문자가 다르다.** NAS는 `DashCAM`, 로컬·config는 `Dashcam`.
그대로 복사하면 `/dataset/Dashcam/...` 이 해석되지 않는다.

```bash
df -h ~/Workspace/data /mnt/NAS_3                    # 양쪽 여유 확인

rsync -a --info=progress2 /mnt/NAS_3/02_datasets/DashCAM/Front/ \
                          ~/Workspace/data/Dashcam/Front/       # 이름 교정

find ~/Workspace/data/Dashcam/Front/Train/images -type f | wc -l   # 라벨 수와 대조
```

하위 디렉토리 대소문자는 데이터셋마다 다르다 — `CVD/Corner` 는 `Train/Validation`,
`CVD/Side` 는 `train/validation`. **각 `configs/datasets/*.yaml` 의 주석이 실측 기준이다.**

### 현재 로컬 사본

| | NAS_3 | 로컬 |
|---|---|---|
| Dashcam | `DashCAM` 150G | `Dashcam` 151G |
| CVD | `CVD` 135G | `CVD` 135G |
| Vueroid | 7개 버전 | 4개 버전 — **갈라져 있다** (부록 B) |

총 약 507 GB. `xision3-smoke/` 515M 는 NAS에 없다.

### 컨테이너 경로

`docker-compose.yml:27` 이 `DATA_ROOT` → `/dataset`. **`configs/datasets/*.yaml` 의
`path:` 는 항상 컨테이너 경로를 쓴다** (`path: /dataset/CVD/Corner`).

---

## 3. 코드·모델 개발

### configs 네 트리 — 위치가 지위를 결정한다

| 트리 | 지위 |
|---|---|
| `configs/datasets/` | 물리 데이터셋 1개 = 파일 1개. `path` + `names` 만. 제품이 아니라 데이터셋 단위 |
| `configs/experiments/` | 미출하. 자유롭게 추가·삭제 |
| `configs/releases/` | 출하 완료, 동결. + `MANIFEST.yaml` |
| `configs/archive/` | **참조 금지** |

ROI·crop·입력 크기는 데이터셋 yaml이 아니라 experiments/releases 의 `exp_params` 에 둔다 —
같은 데이터셋을 여러 제품이 다른 crop으로 쓴다.

`configs/datasets/` 정본 3개 + `legacy/` 17개. 이 호스트에서 **20개 중 8개만 경로가
해석된다** — 정본 3개 + `legacy/{CVD_1t, CVD_5t, dashcam_front, dms_stage1, dms_stage2}`.

### 출력 경로는 config 경로에서 파생된다

`derive_run_dir` (`projects/custom_utils/cfg.py:115`):

```
configs/{experiments,releases}/<P>/<C>/<V>/exp.yaml → runs/<task>/<P>/<C>/<V>/
```

**이 트리 안의 config에는 `project`/`name`/`save_dir` 를 넣지 않는다.**

3중 gate — 하나라도 실패하면 조용히 no-op이 되어 `runs/detect/train2` 로 떨어진다.

1. 파일명이 정확히 `exp.yaml`
2. `configs/experiments` 또는 `configs/releases` 하위 (`cfg.py:112`)
3. 경로 조각 4개 이상 (`<P>/<C>/<V>/<file>`)

`configs/experiments/legacy/` 는 중첩이 깊어 gate 3에서 걸러진다 — 의도된 동작.

### 절차

```bash
mkdir -p configs/experiments/BSD_F606/Front_Truck/v03
cp configs/releases/BSD_F606/Front_Truck/v01/exp.yaml \
   configs/experiments/BSD_F606/Front_Truck/v03/exp.yaml
# model_name · dataset_name · overrides · exp_params 수정

python3 tools/train_custom_trainer.py \
    -w configs/experiments/BSD_F606/Front_Truck/v03/exp.yaml -d 0
# → runs/detect/BSD_F606/Front_Truck/v03/
```

제약:

- `-w` = `--cfg_path` 별칭 (train/dist/val 공용).
- **`model_name` 은 `.get(key, (CustomYOLO, CustomDetectionTrainer))` 조회다** — 미등록 키는
  실패하지 않고 다른 아키텍처를 학습한다. 등록 키는 `consi`(≠`consistency`) 포함 7종,
  전체 표는 [diagrams.md](diagrams.md).
- 듀얼 GPU 호스트 `workers <= 4` (RAM 125 GB).
- Front 4K GA 튜닝은 `amp: false` (D-006).
- 30초 넘는 실행은 예상 소요를 먼저 알린다. GA는 `--max-hours 168` 단위.

### 코드 변경 시

```bash
docker compose run --rm --no-deps dair-train bash -lc 'uv sync --extra dev && uv run pytest tests/ -q'
docker compose run --rm --no-deps dair-train bash -lc 'uv run ruff check .'
```

기준선: `56 passed, 2 skipped` / ruff **14건**(문서화된 미수정 항목 — 이 숫자가 늘면 회귀).

---

## 4. 평가

```bash
python3 tools/validate_custom.py -w configs/experiments/BSD_F606/Front_Truck/v03/exp.yaml
```

- 형제 릴리스의 `MANIFEST.yaml` `metrics` 블록과 대조한다.
- NPU 시뮬레이터 결과는 PyTorch val 기준선과 대조한다 (변환 후, 5절).
- 코드 변경이 학습·변환에 닿으면 테스트만으로 부족하다 — 1-step 실제 실행을 함께 돌린다.
- 위 `validate_custom.py` 명령은 일반적인 예시다. 모델이 custom Trainer/Validator를 요구하면
  (DAIR·Symmetric 등) 이 명령만으로 부족할 수 있다 — 그런 경우 해당 모델의 상세 SOP를
  우선한다. 예: Rear_Truck →
  [`docs/guides/rear_truck_train_validate_nvt_pipeline.md`](guides/rear_truck_train_validate_nvt_pipeline.md).

---

## 5. 릴리즈

### 승격 — 복사 + MANIFEST

```bash
cp -r configs/experiments/BSD_F606/Front_Truck/v03/ \
      configs/releases/BSD_F606/Front_Truck/v03/
# MANIFEST.yaml 작성
# 산출물 → configs/releases/<P>/<C>/<V>/{analysis,network,weight}/  (NAS 레이아웃과 동일)
```

**승격 후 `exp.yaml` 은 다시 손대지 않는다.** 그것이 "무엇이 출하됐는지"의 기록인 이유다.
수정이 필요하면 `configs/experiments/` 에 새 버전을 만든다.

### MANIFEST 키 (10개 파일 관측)

| 키 | 빈도 | 내용 |
|---|---|---|
| `nas_source` `config` `artifacts` `issues` | 10 | 원본 경로 · 정본/데이터셋 · 파일 목록 · 미해결 |
| `run` `recorded` `npu` `dataset` | 8 | 실행 · 기록 시점 · NPU · 데이터셋 |
| `as_run_differs` | 7 | 실제 학습과 정본의 차이 |
| `metrics` | 5 | 지표 |

전 필드 필수가 아니다. 설정 기록이 없어 정본을 만들 수 없으면 MANIFEST만 두고 `exp.yaml` 을
만들지 않는다 (`BSD_F606/Rear_Truck`: `version: null`, `issues: no-config`).

`exp.yaml` 에는 `_meta_` 로 출처를 남긴다 — `source` · `verified_against` · `deviations`.

### NPU 변환

```bash
# ONNX export
python3 tools/export_custom.py ...          # 또는 export_dms.py / export_symmetric.py

# compile → simulate → visualize (wrapper 만 쓴다)
novatek/convert/run_pipeline.sh <model_dir> [test_img] [conf]
```

- `model_dir` 이름이 대문자로 변환돼 `--pattern-name` 이 된다 (`front_260401` → `FRONT_260401`).
- 시뮬레이터는 `nvtai_tool/input/data/<PATTERN>/test_img_list.txt` 를 읽는다 — 없으면
  compile은 성공하고 simulate만 실패한다.
- **중간 단계를 건너뛰면 조용히 실패한다 (D-016).** SDK 명령을 인라인하지 않는다.
- 모델별로 별도의 apples-to-apples Float↔Board 평가 절차가 있을 수 있다. Rear_Truck은 위 3단계
  compile/simulate 이후 카메라 전용 절차를 따른다 —
  [`docs/guides/rear_truck_train_validate_nvt_pipeline.md`](guides/rear_truck_train_validate_nvt_pipeline.md) 참고.

### git 추적 범위

`configs/releases/` 실측 42 MB. 8개 카메라 중 **3개만 산출물을 갖고 있다** —
`Front_Truck` 18M · `Side_Truck` 12M · `Corner_Truck` 12M. 나머지 5개는 텍스트뿐이라
필요하면 `MANIFEST.nas_source`(NAS_2)에서 받아온다.

`.gitignore:42-47` 이 텍스트만 추적한다 (`MANIFEST.yaml` `exp.yaml` `args_dump.yaml`
`best_hyperparameters.yaml`). `runs/` 는 전체 untracked.

> 릴리스 디렉토리에 산출물이 있는데 `git status` 가 비어 있는 것은 정상이다.
> `git status --ignored --short` 로 확인한다.

---

## 6. 머지

```bash
uv run ruff check . && pytest    # 컨테이너에서 (3절)
git switch master && git pull
git merge --no-ff feat/BSD_F606-Front_Truck-v03 -m "merge: <무엇을>"
git tag -a v0.2.0 -F <notes>     # 릴리스를 동반할 때만
```

- 병합 커밋은 `merge:` 로 쓴다. **`merge:` 는 커밋 타입 6종에 추가하지 않는다** — 병합
  전용이며 `--no-merges` 로 체인지로그에서 제외되므로 내용 손실이 없다.
- 커밋 메시지: `type: English title`, 본문 English, scope 없음,
  `Co-Authored-By: Claude <model> <noreply@anthropic.com>` 로 닫는다.
- 타입은 `fix` → `feat` → `refactor` → `test` → `docs` → `chore` 순으로 처음 맞는 것.
- **`git push` 는 에이전트에게 금지돼 있다** (`.claude/settings.json` deny). 사람이 실행한다.

```bash
git push origin master --follow-tags
```

---

## 부록 A. `.claude` 커맨드

`.claude/commands/<group>/<name>.md` → `/<group>:<name>`. 18개.

### 워크플로 단계별

| 단계 | 커맨드 |
|---|---|
| 3 개발 | `/project:plan` `/project:diagram` `/refactor:simplify` `/refactor:cleanup` `/refactor:dead-code` |
| 4 평가 | `/project:review` `/test:coverage` `/test:integration` |
| 5 릴리즈 | `/git:changelog` `/git:release` `/docs:codemap` `/docs:reference` `/docs:api` |
| 6 머지 | `/git:commit` |
| 상시 | `/workflow:checkpoint` `/workflow:wrap-up` `/workflow:orchestrate` `/devops:dockerfile` |

### 실행 방식

| 방식 | 수 | 성질 |
|---|---|---|
| inline | 9 | frontmatter 없음. 현재 대화 컨텍스트에서 실행 — 진행 중 작업에 붙여 쓰기 좋다 |
| fork | 9 | 별도 컨텍스트의 서브에이전트. **대화 맥락을 물려받지 않으니 인자로 명시한다** |
| background | 1 | `/project:review` — 결과가 나중에 알림으로 도착 |

fork 커맨드와 담당 에이전트 (`.claude/agents/` 에 6개 전부 존재):

| 에이전트 | 커맨드 |
|---|---|
| `git-workflow-manager` | `/git:commit` `/git:changelog` `/git:release` |
| `documentation-engineer` | `/docs:codemap` `/docs:reference` |
| `api-documentation-specialist` | `/docs:api` |
| `code-reviewer` | `/project:review` |
| `refactoring-specialist` | `/refactor:dead-code` |
| `test-architect` | `/test:coverage` |

### 인자

`$ARGUMENTS` 같은 치환 문법은 18개 파일 어디에도 없다. 본문이 산문으로 지시하므로
**자연어로 넘기고, 생략하면 대화 맥락에서 대상을 추론한다.**

| 커맨드 | 인자 |
|---|---|
| `/project:diagram` | 종류 (`flowchart` `classDiagram` `erDiagram` `sequenceDiagram` `graph` `stateDiagram-v2`) |
| `/project:review` | 모듈 경로 · 기능 이름 · PR 번호 |
| `/test:integration` | 대상 모듈·파일 |
| `/workflow:orchestrate` | 워크플로 명세 |

### 파일을 쓰는 커맨드

`/workflow:checkpoint` → `.claude/checkpoints/<timestamp>.md` + `CLAUDE.md` 세션 노트,
`/workflow:wrap-up` → `CLAUDE.md`. **둘 다 추적 파일 `CLAUDE.md` 를 고친다** — 실행 후
`git diff CLAUDE.md` 로 확인한다. `.claude/checkpoints/` 는 첫 실행 시 생성된다.

### 훅 13개 · 금지 명령

`.claude/settings.json` → `.claude/hooks/scripts/`.
`PreToolUse(Bash)`: `smart-approve` `commit-guard` `pre-push-check` /
`(Edit|Write)`: `secret-scanner` / `(Write)`: `block-md-creation` /
`PostToolUse(Bash)`: `suggest-compact` / `SessionStart`: `session-start` `context-loader` /
`SessionEnd` `PreCompact` `Stop` `Notification` `UserPromptSubmit`: 각 1개.

`block-md-creation.py` 는 이름과 달리 차단하지 않는다 (docstring: "Advisory, despite the
name it inherited") — 통상 문서 경로 밖 `.md` 에 메모만 남긴다.

`permissions.deny`: `git push`(모든 형태) · `git reset --hard` · `git checkout --` ·
`git stash` · `git clean` · `ruff format` · `rm *dataset*` · `rm *runs*` ·
`docker compose down -v`.

---

## 부록 B. 확인된 불일치 — 기록만, 미조치

1. **`.claude/` 가 git으로 공유되지 않는다.** `.gitignore` 가 아니라
   `.git/info/exclude:9` 로 제외 — 클론마다 따로이고 복제되지 않는다. 커맨드 18개와 훅
   13개가 다른 클론에 전달되지 않고 git 백업도 없다. 공유하려면 그 줄을 지워야 하는데,
   `settings.json` 의 훅 경로가 절대경로 하드코딩이라 다른 호스트에서 그대로는 안 된다.
2. **`.env` 없음** → `npu-convert` 실패 (0절).
3. **로컬이 NAS_3의 부분집합이 아니다.** `Vueroid/` 가 갈라져 로컬 `lpr-det-v3` ·
   `lpr-pose-v3` 가 NAS에 백업되지 않았다. NAS_3 여유 493 GB뿐이라 올리려면 용량 확보 선행.
   NAS_3에만 있는 것: `BSD/` `Research/` `_staging/`.
4. **`NAS_3` 문자열이 저장소 어디에도 없다** (yaml·md·py·yml 전수, 이 문서 제외).
   NAS_3 ↔ 로컬 연결은 코드·설정에 기록이 없는 수동 규약이다.
5. **대소문자 함정이 저장소 안에서 발현돼 있다** —
   `configs/datasets/legacy/unlabeled.yaml` 이 `/dataset/DashCam/Front`(대문자 `CAM`)를
   가리켜 해석 실패.
6. **CLAUDE.md 수치 2건이 낡아 2026-07-30에 정정됨** — `legacy/` 39→17,
   `configs/releases/` 250MB→42MB.
7. **`model_name` 표 오류 3건이 2026-07-30에 정정됨** — `consistency`→`consi`,
   `dair` 의 상속 관계, 누락된 `dms_pose`/`qat`/`fx_qat`.
