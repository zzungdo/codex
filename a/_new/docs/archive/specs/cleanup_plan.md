# 파일·구조 정리 계획

**상태**: **8단계 전부 완료** (2026-08-04, 브랜치 `feat/eval-consolidation`). 결정 기록은 **D-023**.
구현 중 드러난 것은 §7 에 있다.
**작성**: 2026-08-04
**입력 세 갈래** — (1) 수동 조사(도구 인벤토리·참조 수·경로 해석), (2) `/refactor:dead-code`
커맨드 실행 결과(브랜치 `chore/dead-code-removal`, 4커밋), (3) D-020·D-021·D-022 작업 중
부수적으로 드러난 것들.

**신규 기능은 범위 밖이다.** CI·`Makefile`·`check_dataset.py` 같은 신설 제안은 별도
사안으로 뺐다. 이 계획은 **이미 있는 파일**만 다룬다.

---

## 0. 제약 — 정리는 정보를 잃지 않아야 한다

> **① 지우기 전에 그 정보가 어디에 남는지 확인한다.** D-022 가 이 절차로 두 개의 실질
> 손실을 잡았다 — `deviations` 6건 중 3건이 MANIFEST 에 없었고, 실험 트리 파일 하나는
> MANIFEST 자체가 없었다. 먼저 지웠다면 사라졌다.
>
> **② 두 평가 채널의 수치가 바뀌지 않는다.** Corner board 0.481 / onnx 0.460,
> Front board 0.704 / onnx 0.697, GT-in-ROI 21,742 / 50,371.
>
> **③ 동결 릴리스의 학습 키를 건드리지 않는다.** D-020·D-022 의 예외 조건을 유지한다.
>
> **④ D-010 이 보존을 지시한 것은 지우지 않는다** — QAT 코드·config. KD 도 같은 성격이다.

**정리의 성공 기준은 "줄 수가 줄었다"가 아니라 "실행하면 실패하는 것이 없어졌다"다.**

### 0.1 회귀 확인 명령

```bash
docker compose run --rm --no-deps -e EVAL_BASELINE_FULL=1 dair-train \
    bash -lc 'uv run pytest tests/ -q && uv run ruff check .'
# → 116 passed, 2 skipped / Found 9 errors

for C in Corner_Truck Front_Truck; do
  python3 tools/customEval.py -w configs/releases/BSD_F606/$C/v01/exp.yaml --eval board
done
# → 0.481 / 0.704
```

---

## 1. 입력 세 갈래가 서로 무엇을 잡았는가

겹친 것이 **1건**뿐이다. 두 방식이 사각지대가 다르다.

| | 자동 분석 (`/refactor:dead-code`) | 수동 조사 |
|---|---|---|
| 대상 | Python AST — 심볼·바인딩·도달성 | 파일·설정·문서 경계 |
| 잡은 것 | 미참조 모듈 1 · 미참조 심볼 6 · 미사용 바인딩 10 · 주석코드 4블록 · **`*-` 오타 1** | 깨진 `.sh` 런처 2 · 깨진 기본 config 4 · 문서-실재 불일치 3 · 트리 규모 |
| 못 잡는 것 | `.sh`, yaml, 문서, 문자열 안의 경로 | 문장 내부, 심볼 단위 도달성 |

**겹침** — `projects/model_overrides/simota.py`. 수동 조사는 "참조 0건"까지만 보고 판단을
보류했고, 커맨드는 **왜** 0건인지 규명했다: `SimOTAAssigner` 라는 동명 클래스가
`simota_gpt.py` 에도 있고 실사용 경로(`loss.py:120`)는 `simota_dair` → `simota_gpt` 체인을
탄다. 삭제된 모듈은 11개월간 동명 클래스에 가려져 도달 불가였다. 독립 확인했다.

**판단이 갈린 것** — MongoDB 분기(`custom_tuner.py:64,116`). 수동 조사는 "`pymongo` 미선언
→ 도달 불가"로 정리 후보에 넣었고 커맨드는 "선택 의존성에 게이트된 폴백이지 dead code 가
아니다"로 남겼다. **커맨드가 맞다** — `if self.mongodb:` 는 런타임 분기이고, 고칠 대상은
코드가 아니라 `pyproject.toml` 의 미선언이다. 이 계획은 커맨드 판단을 따른다.

**교훈** — 두 방식을 함께 돌려야 한다. 어느 하나로는 절반만 보인다.

---

## 2. 정리 대상 — 근거 실측

### 2.1 확실히 깨진 것 (실행하면 반드시 실패) — 최우선

가장 값싸고 가장 확실한 정리다. 근거가 "대상 파일이 없다"이므로 판단 여지가 없다.

| # | 대상 | 근거 |
|---|---|---|
| a | `tools/dist_train.sh` | 호출 대상 `tools/train.py` **부재** |
| b | `tools/dist_train_tuner.sh` | 호출 대상 `tools/train_tuner.py` **부재** |
| c | `tools/validate_custom.py:12,13` 기본값 | `/workspace/ultralytics_projects/configs/...` — **구 레이아웃 절대경로.** 현재 컨테이너는 `/workspace` 에 마운트한다. 인자 없이 실행하면 즉시 실패하고, 이것은 CLAUDE.md 워크플로 5단계에 실린 문서화된 명령이다 |
| d | `tools/eval.py:11` 기본값 | `configs/experiments/legacy/eval_test.yaml` 부재 |
| e | `tools/validate_symmetric.py:10` 기본값 | `configs/datasets/legacy/TLDS_rework.yaml` 부재 |

`dist_*.sh` 7개 중 6개가 PORT·대상만 다른 동일 보일러플레이트다(정규화 해시 `383554e0`;
`dist_train_custom.sh` 만 다름). a·b 를 지우면 5개가 남고 전부 대상이 실존한다.

**처분** — a·b 삭제. c·d·e 는 유효 경로로 교체하거나 `required=True` 로 바꿔 **조용한 실패를
명시적 실패로** 바꾼다.

### 2.2 문서와 실재의 어긋남

| # | 대상 | 근거 |
|---|---|---|
| f | `CLAUDE.md:85` `projects/exp_yaml/` | **디렉토리가 존재하지 않는다.** "Still referenced by tools/ defaults" 라고 적혀 있다 |
| g | `CLAUDE.md:87` `projects/data_yaml/` | 존재하지 않는다. "has FRONT/, REAR/ subdirs" |
| h | `CLAUDE.md` Clarity 절의 `loss.py:101` | 이중으로 낡았다 — 실제 위치는 112 였고, 커맨드가 `f02168a` 에서 **그 줄을 삭제**했다 |

f·g 는 Project Structure 절이 실재하지 않는 트리를 안내하는 상태다. `projects/cfg/` 도 없으나
그것은 이관 기록이 있다.

### 2.3 import 만으로 실행되는 스크립트

| # | 대상 | 근거 |
|---|---|---|
| i | `tools/finding_rider.py` (58줄) | `__main__` 가드 **0**, `/dataset/260129_4K_FRONT_RELEASE_ONLY/...` 하드코딩 |
| j | `tools/predict_symmetric.py` (5줄) | 가드 0, `/result/YOLO26/...` 하드코딩. 사실상 스니펫 |

**import 만 해도 하드코딩 경로로 동작을 시도한다.** 커맨드가 entry-point 점검에서
`finding_rider.py` 의 `FileNotFoundError` 가 수정 전후 바이트 동일함을 확인했다 — 즉 이
실패는 이 계획 이전부터 있던 것이다.

가드 있으나 하드코딩이 있는 것: `cvt_FHD.py` · `resize.py` · `drawing_labels.py` ·
`rewrite_label_norm.py`(절대경로 17건).

**처분** — 가드 추가 또는 `tools/oneoff/` 로 격리. 삭제는 권하지 않는다(일회성 변환 이력).

### 2.4 미참조 모듈 3개 (`simota.py` 는 커맨드가 이미 처리)

| # | 대상 | 줄 | 비고 |
|---|---|---|---|
| k | `projects/optimizers/muon.py` | — | import 참조 0 |
| l | `projects/custom_utils/similar_images_remover.py` | — | 참조 0. ruff 미수정 3건(W291/W293)의 출처 |
| m | `projects/src/models/yolo/kd/train/dinov3.py` | — | 참조 0. KD 계열 |

**처분에 사람 확인이 필요하다.** `simota.py` 는 동명 클래스에 가려진 것이 규명돼 삭제가
정당했지만, 이 셋은 "안 쓰는 것"과 "쓸 예정인 것"이 구분되지 않는다. `dinov3.py` 는 KD
계열이라 D-010 의 보존 정신이 적용될 수 있다.

### 2.5 실행 불가 / 미선언 의존성

| # | 대상 | 근거 | 처분 |
|---|---|---|---|
| n | `nmqt` 참조 3파일 | `pyproject.toml` 선언 **0** → import 부터 실패 | **선언하거나 코드를 내린다.** 코드 삭제는 D-010 기각이라 별도 D-번호 |
| o | `pymongo` (`custom_tuner.py`) | 선언 0. 코드는 폴백 분기로 정상 | **선언을 추가**한다. 코드는 그대로 (§1 판단) |
| p | `cls_weights` | 소비처 0. 동결 릴리스 `Side_Truck/v01` 에만 존재 | **"미구현" 문서화.** 제거는 동결을 깬다 |

### 2.6 규모 큰 트리 — 삭제하지 않는다

| 트리 | 파일 | 크기 | 실측 |
|---|---|---|---|
| `configs/archive/` | 207 | 1.0 MB | 외부 참조 22건이 **전부 문서 언급**, 코드 참조 0 |
| `configs/experiments/legacy/` | 26 | 168 KB | `tools/*.py` 기본값·`run_finetune_pipeline.py` 의 `_STAGE*_CFG` 가 참조 |
| `configs/datasets/legacy/` | 17 | 80 KB | — |

**`configs/archive/README.md` 가 이미 처분을 규정하고 있다** — "삭제하지 않고 남긴 이유는
실험 이력이 커밋 메시지보다 파일에 더 잘 남아 있어서다." 이 계획은 그 결정을 따른다.

**중요 — 경로 미해석은 삭제 근거가 아니다.** `configs/datasets/` 20개 중 12개가 존재하지
않는 경로를 가리키지만, 그중 **10개는 참조 2~9건**이다.

```
legacy/FRONT/260225_4K_FRONT.yaml   참조 9
legacy/REAR/260401_REAR.yaml        참조 6
legacy/FRONT/260401_4K_merged.yaml  참조 6
legacy/unlabeled.yaml               참조 3
legacy/ncnvox_{cmd,kws,kws_full}    참조 0   ← 데이터가 33 GB 라 이 호스트에 미동기
```

데이터가 다른 호스트에 있거나 아직 rsync 하지 않은 것이다. **호스트 의존이지 죽은 config 가
아니다.** 처분은 삭제가 아니라 그 사실을 명시하는 것이다.

### 2.7 살아 있으나 미문서화 — 삭제 아님, 등재 대상

참조 0건이지만 정상 도구다. `argparse` 를 갖고 하드코딩이 없다.

| 대상 | 역할 |
|---|---|
| `tools/prepare_sim_eval_imgs.py` | **sim 평가 워크플로 1단계** (`prepare_sim_eval_imgs → run_sim_batch → eval_det_txt`). D-020 §5 가 "미문서화 워크플로"로 적어둔 항목 |
| `tools/get_label_stats.py` | 라벨 통계 |
| `tools/data_analysis.py` (355줄) | 데이터 분석 |
| `tools/sample_val_balanced.py` | val 클래스 균형 샘플링 |
| `tools/predict_to_json.py` | 예측 → COCO json |

**참조 0건을 삭제 근거로 쓰면 이것들이 사라진다.** 이 계획은 이 다섯을 README 명령표에
등재하는 것으로 처분한다.

### 2.8 커맨드가 남긴 동작 변경 1건

| # | 대상 | 근거 |
|---|---|---|
| q | `projects/src/models/yolo/detect/train/kd_draft.py:181` | `self.d_loss *- distill_weight` — `*=` 오타. 다음 줄이 `self.d_loss` 를 그대로 더하므로 **cosine 증류 가중치가 전혀 적용되지 않는다.** `__init__.py:5` 가 export 하므로 실행 경로에 있다 |

커맨드가 리팩터링 커밋에 넣지 않은 것은 맞는 판단이다 — 동작 변경이다. `fix:` 로 분리한다.
KD 는 미사용 실험 잔존물이라 우선순위는 낮다.

---

## 3. 단계 — 위험 낮은 순, 각 단계 끝에 §0.1

### ☑ 0. 브랜치 정리 — small

- **What**: `chore/dead-code-removal` 의 4커밋 처리를 정한다.
- **현황**: 커맨드가 `feat/eval-consolidation` 에서 분기해 그 16커밋을 업고 있다
  (master 대비 20 ahead). 푸시되지 않았다.
- **선택** — (a) `feat/eval-consolidation` 으로 병합해 한 줄기로 만든다, (b) 독립 유지
  (`evaluator.py`·`utils.py` 두 hunk 는 cherry-pick 충돌).
- **권장** — **(a)**. 두 브랜치가 같은 파일을 만지고 있어 분리 유지 비용이 크다.
- **Test**: 병합 후 §0.1.

### ☑ 1. 깨진 것 정리 (2.1) — small

- **What**: a·b 삭제, c·d·e 기본값 교체.
- **How**: c 는 `/workspace/ultralytics_projects/` 접두를 떼어 저장소 상대경로로. d·e 는
  존재하는 config 로 바꾸거나 `required=True`. **조용한 실패를 명시적 실패로 바꾸는 것이
  목적이므로 `required=True` 를 우선 검토**한다.
- **Test**: 5개 스크립트를 인자 없이 실행해 (i) 유효 경로로 동작하거나 (ii) 명확한 인자
  누락 오류를 내는지. 지금은 존재하지 않는 파일을 열려다 실패한다.

### ☑ 2. 문서-실재 불일치 (2.2) — small

- **What**: f·g·h.
- **How**: `CLAUDE.md` Project Structure 에서 없는 두 트리를 제거하고, Clarity 절의
  `loss.py:101` 예시를 살아 있는 예로 교체하거나 문장을 다시 쓴다.
- **Test**: CLAUDE.md 가 언급하는 모든 경로의 실재를 한 번 훑는다.

### ☑ 3. 미문서화 도구 등재 (2.7) — small

- **What**: 5개를 README 명령표에. sim 평가 3단 워크플로를 `operations_guide.md` 4절에.
- **Test**: 문서의 명령을 복사해 `--help` 가 답하는지.

### ☑ 4. import 시 실행 차단 (2.3) — medium

- **What**: i·j 에 `__main__` 가드 추가, 또는 `tools/oneoff/` 신설 후 이동.
- **How**: **가드 추가를 권장**한다 — 디렉토리를 옮기면 `tools/` 를 훑는 문서·스크립트가
  깨지고, 가드만으로 "import 해도 안전"이 달성된다.
- **Test**: 전 스크립트 import 스모크. 커맨드가 확인한 51/59 → 개선 여부.

### ☑ 5. 미선언 의존성 (2.5 n·o) — small

- **What**: `pymongo` 선언 추가(o). `nmqt` 는 결정 사항이라 문서에 미선언 사실만 명시(n).
- **How**: `pymongo` 는 GA 튜닝 선택 기능이므로 `[project.optional-dependencies]` 의
  별도 extra 로. 기본 의존성에 넣지 않는다.
- **Test**: `uv sync --extra <새 extra>` 후 `custom_tuner` import.

### ☑ 6. `cls_weights` 문서화 (2.5 p) — small

- **What**: "미구현" 표기. `Side_Truck/v01/MANIFEST.yaml` 의 `issues` 에 등재.
- **Test**: 없음(문서).

### ☑ 7. `kd_draft.py:181` 오타 (2.8) — small

- **What**: `*-` → `*=`. **`fix:` 커밋으로 분리.**
- **위험**: KD 학습의 손실 크기가 바뀐다. KD 로 학습된 출하 모델은 없다(README 가 `kd_draft`
  를 미사용으로 표기).
- **Test**: KD 경로에 테스트가 없다. 최소 `epochs=1` 스모크가 필요하고, 그것이
  불가하면 **수정을 보류하고 주석으로 표기**하는 것도 정당한 선택이다.

### ☑ 8. 미참조 모듈 3개 (2.4) — 결정 후 착수

- **What**: k·l·m.
- **선행 조건**: 사람 확인. `dinov3.py` 는 D-010 정신이 걸린다.

### ☑ 9. 결정 기록 — small

- **Where**: `docs/decisions.md` (D-023).
- **What**: 정리 원칙 — **참조 0건은 삭제 근거가 아니다**(2.7 이 그 반례), **경로 미해석도
  삭제 근거가 아니다**(2.6), 자동·수동 두 분석을 함께 돌린다(§1).

---

## 4. 위험

| 위험 | 단계 | 대응 |
|---|---|---|
| 참조 0건을 근거로 살아 있는 도구를 삭제 | 3·8 | 2.7 이 반례 5건. `argparse` 보유·하드코딩 없음·워크플로 소속을 함께 본다 |
| 경로 미해석을 근거로 config 삭제 | 2.6 | 12개 중 10개가 참조 2~9건. 호스트 의존이지 죽은 것이 아니다 |
| 기본값을 `required=True` 로 바꿔 기존 호출을 깨뜨림 | 1 | 기본값이 **이미 존재하지 않아** 지금도 실패한다. 깨질 호출이 없다 |
| `kd_draft` 오타 수정이 KD 학습 결과를 바꿈 | 7 | 의도된 변경이나 테스트가 없다. 스모크 불가 시 보류가 정당 |
| dead-code 브랜치 병합이 평가 작업과 충돌 | 0 | `evaluator.py`·`utils.py` 두 hunk. (a) 병합이면 충돌 없음 |
| D-010 보존 지시 위반 | 5·8 | `nmqt`·KD 코드는 지우지 않는다. 문서에서만 내린다 |

---

## 5. 결정이 필요한 것

| # | 항목 | 권장 |
|---|---|---|
| 5-1 | `chore/dead-code-removal` 처리 | **(a) `feat/eval-consolidation` 에 병합** |
| 5-2 | 미참조 모듈 3개(`muon`·`similar_images_remover`·`dinov3`) | 사람 확인 필요. `dinov3` 는 보존 쪽 |
| 5-3 | `kd_draft.py:181` 오타 | 스모크가 가능하면 `fix:`, 불가하면 주석 표기 후 보류 |
| 5-4 | `nmqt` 3파일 | 선언 추가 대 코드 제거. 후자는 D-010 기각 |
| 5-5 | 깨진 기본값을 `required=True` 로 바꿀지 | 권장. 조용한 실패를 없앤다 |

---

## 6. 인수 기준

| 항목 | 조건 |
|---|---|
| 두 채널 수치 | Corner 0.481 / 0.460, Front 0.704 / 0.697 무변화 |
| pytest | `116 passed, 2 skipped` 이상 (`EVAL_BASELINE_FULL=1`) |
| ruff | 9건 이하 |
| 깨진 런처 | 0개 — 모든 `dist_*.sh` 의 대상이 실존 |
| 깨진 기본값 | 0개 — 인자 없이 실행 시 유효 동작 또는 명확한 오류 |
| CLAUDE.md | 언급하는 모든 경로가 실재 |
| import 안전성 | `tools/*.py` 전체가 import 만으로 파일시스템에 접근하지 않음 |
| 정보 손실 | 삭제한 항목마다 정보가 어디에 남는지 커밋 메시지에 근거 기재 |

---

## 7. 구현 기록 (2026-08-04)

결정 기록은 **D-023**. 8단계 중 **0~6·9 완료**, 7·8 은 결정 대기.

### 7.1 계획이 놓쳤던 것 — 죽은 체인의 호출자

`dist_train_tuner.sh` 를 지우려는데 **`train_tuner_iter.py:90` 이 그것을 subprocess 로
부르고 있었다.** 계획은 "대상이 없으니 반드시 실패"까지만 봤고 호출자를 확인하지 않았다.

```
train_tuner_iter.py:90  →  dist_train_tuner.sh  →  train_tuner.py (부재)
```

체인 전체가 죽어 있었으므로 삭제가 상황을 악화시키지는 않지만, 참조를 매달린 채 두면 안
된다. `dist_train_custom.sh` 로 재연결했다 — `${@:2}` 로 인자를 `train_custom_trainer.py`
에 그대로 넘기고 그쪽이 `--cfg_path`·`-d` 를 받으므로 호출 형태가 바뀌지 않는다.
**GA 튜닝 전체 실행으로는 검증하지 못했다(168시간 단위)** — 코드에 그 사실을 적었다.

**교훈** — "대상이 없다"를 삭제 근거로 쓸 때 **호출자도 함께 본다.**

### 7.2 5단계는 조치가 필요 없었다 — 분석 오류 정정 2건

**① MongoDB 분기는 우리 것이 아니다.** 계획 §2.5(o)가 `pymongo` 선언 추가를 지시했으나
틀렸다. `self.mongodb`·`_save_to_mongodb`·`_sync_mongodb_to_csv` 는 **전부 upstream
`ultralytics.engine.tuner.Tuner`** 것이고, upstream 이 `_connect` 안에서
`check_requirements("pymongo")` 로 필요 시 설치한다. 우리 `custom_tuner.py` 는 상속해
호출만 한다. **선언할 것이 없다.**

이것은 `/refactor:dead-code` 커맨드의 판단("선택 의존성 폴백")이 맞았고 제 수동 분석이
틀렸던 지점이다. 커맨드는 "폴백"이라고만 했고, 왜 폴백인지(upstream 소유)까지 확인하니
조치 자체가 사라졌다.

**② `nmqt` 미선언은 이미 기록돼 있었다.** `CLAUDE.md:278` 이
"need the `nmqt` toolkit, which is not a declared dependency and is not needed while QAT is
shelved (D-010)" 로 적고 있다. 새로 할 일이 없었다.

### 7.3 `CLAUDE.md` Project Structure 교체

없는 두 트리(`projects/exp_yaml/`·`projects/data_yaml/`)를 실재하는 넷으로 바꿨다.

```
projects/
  src/              fork modules mirroring ultralytics internals
  metrics/          offline and board evaluation — canonical accuracy path (D-020)
  custom_utils/     config loading (cfg.py), coordinate conversion, plotting, label tooling
  model_overrides/  assigner variants (SimOTA). simota_gpt.py holds the class actually used
  optimizers/       optimizer variants. muon.py has no importer
```

`model_overrides` 와 `optimizers` 행에 **정리 조사 결과를 함께 적었다** — SimOTA 실사용
클래스 위치(동명 클래스 혼동이 `simota.py` 를 11개월 방치시킨 원인)와 `muon.py` 무참조
사실이다. 다음 사람이 같은 조사를 반복하지 않게 하는 것이 목적이다.

### 7.4 검증 결과

| 항목 | 결과 |
|---|---|
| Corner board / onnx | **0.481 / 0.460** 무변화 |
| Front board / onnx | **0.704 / 0.697** 무변화 |
| pytest | **116 passed, 2 skipped** (`EVAL_BASELINE_FULL=1`) |
| ruff | **9** 유지 |
| 깨진 런처 | **0** — 남은 `dist_*.sh` 5개 전부 대상 실존 |
| 깨진 기본값 | **0** — 3개 스크립트가 명확한 인자 누락 오류 + `--help` 정상 |
| import 안전성 | `tools/*.py` 전체 파일시스템 오류 **0건** (이전 2건) |
| CLAUDE.md 경로 | 토큰 32개 전수 확인, 실재하지 않는 것 0 |
| `finding_rider` 스모크 | 합성 라벨에서 person 3 → rider 7, 나머지 클래스 통과 |

### 7.5 남은 것

| 항목 | 상태 |
|---|---|
| 7단계 `kd_draft.py:181` `*-` 오타 | **결정 대기.** 동작 변경이고 KD 경로에 테스트가 없다. KD 로 학습된 출하 모델은 없다 |
| 8단계 미참조 모듈 3개 | **결정 대기.** `dinov3.py` 는 KD 계열이라 D-010 정신이 걸린다 |
| `runs/detect/.../v01_smoke` | 스모크 산출물. `runs/` 삭제가 에이전트에게 거부돼 사람이 지운다 |

### 7.6 7·8단계 — 결정과 근거 (2026-08-05)

**7단계 `kd_draft.py:181` — 수정했다.** 계획은 "스모크 가능하면 fix, 불가하면 보류"였고
스모크는 **불가**했다(KD config `kd_test.yaml` 의 `overrides.data` 가 없는
`TLDS_rework.yaml` 을 가리킨다). 그런데 조사해 보니 보류보다 수정이 맞았다 —

> `KnowledgeDistllationDetectionTrainer` 는 `__init__.py` 가 export 만 하고
> **인스턴스화 지점이 0건**이다. `tools/train_kd.py` 는 `kd/train` 의 `DraftKDTrainer` 를
> 쓰고, `_MODEL_REGISTRY` 에도 KD 키가 없다.

즉 **현재 아무것도 이 코드를 돌리지 않으므로 수정이 회귀를 낼 수 없다.** 되살릴 때
조용히 틀린 것보다 맞는 편이 낫다. `*-` → `*=` 로 고치고, 검증 불가 사실과 그 이유를
코드 주석에 남겼다.

**8단계 미참조 모듈 3개 — 셋을 다르게 처분했다.** "참조 0건"이라는 같은 신호였지만 실체가
전혀 달랐다.

| 모듈 | 실측 | 처분 |
|---|---|---|
| `kd/train/dinov3.py` | **0 바이트.** 저장소 유일한 빈 `.py`. 내용도 참조도 0 | **삭제** |
| `custom_utils/similar_images_remover.py` | 226줄 완성 코드. 그런데 **결함 2개로 쓸 수 없는 상태** | **수정** |
| `optimizers/muon.py` | 373줄 완성된 Muon optimizer 구현 | **유지** |

`similar_images_remover.py` 에서 결함 두 개를 새로 찾았다 — 계획에도 없던 것이다.

1. **`--help` 가 크래시했다.** `help='Black mask (in %) to apply'` 의 맨 `%` 를 argparse 가
   %-포매팅하려다 `ValueError: unsupported format character ')'`. **도움말조차 읽을 수
   없었다.**
2. **import 이 불가했다.** `from imaging import ...` 라 파일을 직접 실행할 때만 해석되고
   `-m` 이나 import 에서는 `ModuleNotFoundError`.

둘 다 고쳐 직접 실행·`-m`·import 세 경로가 모두 동작한다. 편집한 파일이므로 CLAUDE.md 의
"Reduce them in files you touch" 에 따라 이 파일의 기존 ruff 3건(W291/W293)과 제 편집이
만든 I001 도 함께 정리했다 — **ruff 9 → 6**.

`muon.py` 는 유지한다. 완성된 optimizer 이고, **참조 0건은 삭제 근거가 아니다**(§2.7 이 그
반례). CLAUDE.md Project Structure 에 "muon.py has no importer" 로 이미 적어 두었으므로
다음 사람이 같은 조사를 반복하지 않는다.

### 7.7 최종 상태

| 항목 | 결과 |
|---|---|
| Corner board / onnx | **0.481 / 0.460** 무변화 |
| Front board / onnx | **0.704 / 0.697** 무변화 |
| pytest | **116 passed, 2 skipped** |
| ruff | **6** (착수 시 9, D-020 착수 시 14) |
| 잔여 ruff | `confusion_matrix.py` W291 · `custom_coco_eval.py` W293·E722 · `custom.py` E711 · `dair.py`·`symmetric.py` F811 — 전부 이번 작업이 손대지 않은 파일 |
