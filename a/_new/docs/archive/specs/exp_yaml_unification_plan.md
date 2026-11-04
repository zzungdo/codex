# exp.yaml 통합 설계 계획

**상태**: **구현 완료** (2026-08-04, 브랜치 `feat/eval-consolidation`). 결정 기록은 **D-021**.
6단계 전부 수행했고 §7 인수 기준을 통과했다 — board **0.481** / float **0.460** /
GT-in-ROI **21,742** / Δ **+0.0211** 무변화, pytest **111 passed**, ruff **9**,
1-epoch 학습 스모크 exit 0, 복사-재실행 파생 확인.
구현 중 확정·이탈한 사항은 **§10** 에 모았다.
**작성**: 2026-08-04, 브랜치 `feat/eval-consolidation` HEAD `96e141b`
**선행**: D-020 (평가 통폐합) · `docs/specs/eval_consolidation_plan.md`

목표 셋 — (1) `exp.yaml` 이 `mycfg.yaml` 을 대체한다, (2) `eval:` 의 `_shared` 와
variant 밑 `pred` 래퍼를 없앤다, (3) `exp.yaml` 구성을 통폐합한다.

설정 파라미터 전수는 **§8**, 제거 예정 키는 **§8.11**, `exp_params` 에 `eval` 을
통합하는 방안 조사는 **§9** 에 있다.

---

## 0. 최우선 제약

> **① 수치가 바뀌지 않는다.** board `0.481` · float `0.460` · GT-in-ROI `21,742`.
> 허용 오차 per-class ±0.003, mAP ±0.002, GT 개수 정확히 일치.
> **② 명령어 인터페이스가 바뀌지 않는다.** 아래 표의 명령은 한 글자도 달라지지 않는다.
>
> 두 제약이 다른 모든 목표(스키마 미관·중복 제거·사용성)보다 앞선다. 수정 비용은
> 고려하지 않지만 이 둘은 타협하지 않는다.

### 0.1 유지되는 명령어 (전수)

| 명령 | 유지 근거 |
|---|---|
| `tools/train_custom_trainer.py -w <cfg> -d <gpu>` | 입력 파일의 **내용**만 바뀐다. 인자·기본값·`--cfg_options` 문법 불변 |
| `tools/customEval.py -w <exp.yaml> [--eval <v>...]` + 14개 플래그 | `--eval` 이 가리키는 variant **이름**(`onnx`/`board`)을 보존한다 |
| `tools/validate_custom.py -w <cfg>` | 손대지 않는다 |
| `bash tools/dist_train_custom.sh <N> -w <cfg>` | 손대지 않는다 |
| `pytest tests/` · `EVAL_BASELINE_FULL=1` | 게이트 그대로 |

**바뀌는 것은 딱 하나** — 학습이 `save_dir` 에 남기는 파일명이 `mycfg.yaml` → `exp.yaml`.
이것은 산출물이고 읽는 코드가 0건이므로(§1.1) 명령어 인터페이스가 아니다.

### 0.2 회귀 확인 명령

```bash
W=configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml
python3 tools/customEval.py -w $W --eval board   # → 0.481
python3 tools/customEval.py -w $W --eval onnx    # → 0.460
docker compose run --rm --no-deps -e EVAL_BASELINE_FULL=1 dair-train \
    bash -lc 'uv run pytest tests/ -q -rs'       # → 83 passed, 2 skipped
```

---

## 1. 현황 — 코드가 말하는 사실

### 1.1 `mycfg.yaml` 은 산출물이고 읽는 코드가 없다

쓰는 곳 4개 — 전부 트레이너 생성자, 학습 시작 시점:

```
projects/src/models/yolo/detect/train/custom.py:53     save_my_cfg(custom_cfg, self.save_dir / 'mycfg.yaml')
projects/src/models/yolo/detect/train/dair.py:52       (동일)
projects/src/models/yolo/detect/train/symmetric.py:82  (동일)
projects/src/models/yolo/kd/train/draft.py:79          (동일)
```

**읽는 코드는 0건이다.** 전수 검색 결과 나머지 참조는 전부 MANIFEST 의 `as_run` 필드와
문서·주석이다. 파일명을 바꾸는 데 코드상 걸림돌이 없다.

기록 순서도 확인했다 (`tools/train_custom_trainer.py:60-68`):

```
get_my_cfg(-w)  →  apply_cfg_options(CLI)  →  derive_run_dir(경로 파생)  →  Trainer(...)  →  save_my_cfg
```

즉 덤프는 **CLI 덮어쓰기와 경로 파생이 모두 적용된 최종 해석값**이다. 이미 as-run 기록의
자격을 갖췄다.

### 1.2 두 스키마는 이미 같다

`configs/archive/exp_yaml_2026H1/{672x384,tlds_test}/mycfg.yaml` 이 **입력 config 로
쓰인 전례**다. `get_my_cfg` 가 받는 것과 `save_my_cfg` 가 쓰는 것이 같은 dict 이므로
포맷이 하나다. 이름만 둘인 상태다.

### 1.3 `exp.yaml` 최상위 키 — 10파일 실측

| 키 | 출현 | 읽는 곳 |
|---|---|---|
| `_meta_` | 10/10 | 없음 (사람용). `train_custom_trainer.py:64` 가 출력만 |
| `model_name` | 10/10 | `_MODEL_REGISTRY.get(...)` — 미등록 키는 조용히 다른 아키텍처 |
| `overrides` | 10/10 | ultralytics trainer args |
| `exp_params` | 10/10 | 트레이너·validator |
| `weight_path` | 10/10 | `cfg.weight_path` |
| `dataset_name` | 7/10 | 데이터셋 빌더 |
| `dair_params` | 3/10 | dair·consistency |
| `save_dir` | 1/10 (`Front_Truck/v01`) | **`symmetric` 만** (`symmetric.py:81`) |
| `cls_weights` | 1/10 | Side_Truck |
| `eval` | 1/10 | `tools/customEval.py` (D-020 신설) |

### 1.4 ROI 가 한 파일에 세 번 나온다 — 정본이 없다

`configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml` 에서 `[428, 8, 1072, 1072]` 는:

| 위치 | 읽는 곳 |
|---|---|
| `exp_params.data_args.eval_roi_crop` | **없음** (§1.5) |
| `eval._shared.gt.roi` | `Evaluator.eval_` → `filter_gt_roi` |
| `eval.onnx.pred.inf_func.args.crop` | `crop_inference_v8` |

**세 값이 갈리면 수치가 조용히 달라진다.** §0 제약에 직접 걸리는 위험이고, 이 계획의
가장 강한 통폐합 근거다.

### 1.5 소비처 0건인 키 두 개

| 키 | 코드 소비처 | 원래 용도 |
|---|---|---|
| `exp_params.data_args.eval_roi_crop` | **0건** | 저장소 밖 평가 도구(`ncn_metric`)가 읽었다 |
| `exp_params.data_args.val_resize` | **0건** | 같음. `stretch` = 레터박스 없는 plain resize |

`data_args` 는 데이터셋 빌더로 넘어가지만(`projects/src/data/dataset/__init__.py:28`
→ `dycrop=data_args`) 이 두 키는 거기서도 읽히지 않는다. Corner v01 MANIFEST 가 이미
"소비처가 0건이다. 평가 도구가 repo 밖에 있다"로 기록해 두었다.

**D-020 이 평가를 저장소 안으로 들여왔으므로 이 두 키에 이제 소비처가 생길 수 있다.**
그것이 이 계획 (3)의 핵심이다.

### 1.6 데이터셋 yaml 과 `eval:` 의 중복

`configs/datasets/cvd_corner.yaml`:

```yaml
path: /dataset/CVD/Corner
train: Train/images
val: Validation/images
names: {0: bus, 1: car, 2: motorcycle, 3: person, 4: truck}
```

`eval._shared` 가 같은 정보를 다시 적는다 — `classes: [bus, car, motorcycle, person, truck]`
와 `gt.path: /dataset/CVD/Corner/Validation`. `overrides.data` 가 이미 이 파일을 가리키고
있으므로 **양쪽이 갈릴 수 있는 중복**이다.

### 1.7 round-trip 함정 — 덤프를 복사해 재실행하면 옛 경로로 간다

`derive_run_dir` 이 `overrides.project`/`name` 을 채운 **뒤에** 덤프가 저장된다. 그래서
덤프에는 `project: BSD_F606/Corner_Truck` · `name: v01` 이 들어 있다. 이 파일을
`configs/experiments/<P>/<C>/v02/exp.yaml` 로 복사해 실행하면 `derive_run_dir` 이
"이미 값이 있으므로 no-op"(`cfg.py:164`)이 되어 **v01 로 다시 쓴다.**

실측 확인:

```python
increment_path(Path("runs/detect/BSD_F606/Corner_Truck/v01"), exist_ok=False)
# → runs/detect/BSD_F606/Corner_Truck/v012
```

즉 조용히 틀린 곳도 아니고 `v012` 라는 이상한 디렉토리가 생긴다. `symmetric` 은 최상위
`save_dir` 로 같은 함정을 갖는다.

**(1)을 하려면 이 문제를 반드시 함께 풀어야 한다** — 그러지 않으면 "덤프를 복사해서 다음
버전을 만든다"는 가장 자연스러운 사용 흐름이 함정이 된다.

### 1.8 `mycfg.yaml` 의 신뢰도 이력

과거 릴리스에서 `mycfg.yaml` 은 여러 번 as-run 기록으로서 실패했다 —
`Front_Truck/v01` 은 **YAML 파싱조차 안 되고**(`eval_roi_crop` 이 리스트 안으로 들여쓰기)
`args.yaml` 과 `save_dir`·`eval_roi_crop` 이 어긋나 MANIFEST 가
`mycfg-unreliable` 로 등재했다. `Side_Truck/v01` 은 `weight_path` 가 실제 학습값이 아니다.

**원인은 사람이 손으로 조립한 파일을 as-run 이라 불렀기 때문이다.** 덤프를 정본
스키마(`exp.yaml`)로 만들면서 **기계가 쓴 것과 사람이 쓴 것을 구분**해야 재발하지 않는다.

---

## 2. 설계

### 2.1 원칙 — 옮기지 않고 파생한다

릴리스 `exp.yaml` 은 동결이다. 학습 키를 재배치하면 그 동결을 깬다. 그래서 통폐합을
**값을 옮겨서** 하지 않고 **기존 키에서 파생해서** 한다.

> `exp_params.data_args.eval_roi_crop` 을 정본으로 삼고 `eval.gt.roi` 와 `crop` 을
> 거기서 파생한다. 동결된 파일은 한 글자도 바뀌지 않는데 중복이 사라진다.

이 원칙이 (2)와 (3)을 동시에 가능하게 한다 — `eval:` 블록에서 뺄 수 있는 키가 늘어나면
`_shared` 에 남는 것이 없어지고, `_shared` 는 자연히 사라진다.

### 2.2 (1) `mycfg.yaml` → `exp.yaml`, 그리고 기록/입력 분리

**`_` 로 시작하는 최상위 키는 기록이며 절대 입력으로 읽지 않는다.** 기존 `_meta_` 관례를
규칙으로 승격한다.

```yaml
# runs/detect/BSD_F606/Corner_Truck/v01/exp.yaml   ← 학습이 남긴 파일

# ── 사람이 쓴 것 (입력 파일에서 그대로) ────────────────────────
_meta_: {...}
model_name: custom
dataset_name: dycrop
weight_path: configs/releases/.../pretrained_weight.pt
exp_params: {...}
overrides: {imgsz: [320,320], epochs: 50, ...}   # project/name 없음
eval: {...}

# ── 기계가 쓴 것 (기록. 다시 읽지 않는다) ──────────────────────
_resolved_:
  save_dir: runs/detect/BSD_F606/Corner_Truck/v01
  derived_from: configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml
  overrides: {project: BSD_F606/Corner_Truck, name: v01}   # derive_run_dir 산출
  cfg_options: {}                                          # CLI 덮어쓰기
_run_:
  timestamp: 2026-08-04 09:58:17
  git_sha: 96e141b
  command: tools/train_custom_trainer.py -w ... -d 0
```

이 분리가 주는 것:

| 얻는 것 | 어떻게 |
|---|---|
| **복사-재실행이 안전하다** | 파생 키가 `_resolved_` 안에 있으므로 `configs/experiments/<P>/<C>/v02/` 로 복사하면 `derive_run_dir` 이 v02 를 새로 파생한다 (§1.7 해소) |
| **as-run 기록이 더 정확해진다** | CLI 덮어쓰기와 파생 결과가 분리 기록돼 "무엇을 썼고 무엇이 계산됐는지" 구분된다 (§1.8 재발 방지) |
| **파일명이 하나다** | `-w` 가 받는 것과 학습이 남기는 것이 같은 이름·같은 스키마 |

**재실행 안내** — `runs/` 아래의 `exp.yaml` 을 `-w` 로 직접 주면 `derive_run_dir` 이
no-op(경로가 `configs/` 밖)이고 `project`/`name` 도 없으므로 ultralytics 기본값
`runs/detect/train` 으로 떨어진다. 로더가 이 상황을 감지해 경고한다:

```
[cfg] 경고: runs/ 아래의 실행 기록을 입력으로 받았다.
      다음 버전을 만들려면 configs/experiments/<PRODUCT>/<Camera>/<Version>/exp.yaml 로 복사하라.
      현재 출력 위치: runs/detect/train  (파생 아님)
```

`_meta_` 와 마찬가지로 `_resolved_`/`_run_` 은 로더가 **무시**한다. 실수로 입력에 남아
있어도 동작이 바뀌지 않는다.

#### 2.2.1 이 파일은 사람이 읽는 용도다 (6-2 결정)

저장소 안에서 읽는 코드가 0건이고 저장소 밖에도 없다는 것이 확인됐다. **유일한 독자가
사람이다.** 그러면 설계 기준이 하나 늘어난다 — 기계가 파싱만 되면 되는 파일이 아니라
**나중에 사람이 열어보고 이해할 수 있어야 한다.**

**문제 — 덤프는 입력 파일의 주석을 전부 잃는다.** `save_my_cfg` 는
`YAML.save` → `yaml.dump` 이고 dump 는 주석을 복원하지 않는다. Corner v01 의 exp.yaml 은
`_meta_.deviations` 3건과 키별 주석(예: `workers: 4  # 학습 당시 8`)으로 근거를 설명하는데,
덤프에는 `_meta_` 는 값이라 남지만 **키별 주석은 사라진다.**

**대응 — 이미 있는 `header` 인자를 쓴다.** `save_my_cfg(cfg, dst_path, header="")` 가
`YAML.save(header=...)` 로 그대로 넘기고, `YAML.save` 는 그 문자열을 파일 맨 앞에
그대로 쓴다. 지금은 트레이너 4곳이 아무것도 넘기지 않아 빈 헤더다.

```yaml
# 이 파일은 학습이 자동 생성한 실행 기록이다. 손으로 고치지 않는다.
#
# 입력 config : configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml
# 출력 위치   : runs/detect/BSD_F606/Corner_Truck/v01
# 실행        : tools/train_custom_trainer.py -w <입력 config> -d 0
# 시각        : 2026-08-04 09:58:17 (KST)   커밋: 96e141b
#
# 다음 버전을 만들려면 이 파일을 그대로 쓰지 말고 아래로 복사한 뒤 고친다:
#   configs/experiments/BSD_F606/Corner_Truck/<새버전>/exp.yaml
# `_` 로 시작하는 최상위 키(`_meta_` `_resolved_` `_run_`)는 기록이며 로더가 무시한다.
#
# 원본 config 의 주석은 YAML 덤프 과정에서 사라진다 — 근거는 원본을 보라.

_meta_: {...}
model_name: custom
...
```

**키 순서도 사람 기준으로 유지된다** — `YAML.save` 가 `sort_keys=False` 로 dump 하므로
입력 파일의 키 순서가 그대로 남는다. 알파벳순으로 재배열되지 않는다. `allow_unicode=True`
라서 한글 `_meta_` 도 escape 되지 않는다. 둘 다 이미 그렇게 되어 있어 바꿀 것이 없다.

| 사람이 읽는 용도라서 하는 것 | 하지 않는 것 |
|---|---|
| 생성 헤더로 출처·재실행 방법·주석 소실 사실을 알린다 | 주석 보존을 위해 ruamel.yaml 같은 의존성을 새로 들이지 않는다 |
| `_resolved_`/`_run_` 으로 기계값을 한군데 모아 사람이 건너뛰기 쉽게 한다 | 원본 주석을 재생성하려 시도하지 않는다 (근거는 원본이 정본) |
| 키 순서·유니코드를 그대로 둔다 (이미 그렇다) | 포맷을 예쁘게 만들기 위해 dump 옵션을 더 손대지 않는다 |

### 2.3 (2) `eval:` 평탄화 — `_shared` 와 `pred` 제거

3단계 → **2단계**. 공통 키는 `eval:` 직하로 올리고, variant 본문이 곧 예측 지정이 된다.

```yaml
eval:
  # 공통 (예전 _shared 의 내용이 그대로 한 단계 올라온다)
  gt: {...}
  classes: [...]
  mode: [ap, f1_score]
  conf: 0.01

  # variant — 이름이 곧 출력 디렉토리, `pred:` 래퍼 없음
  onnx:  {rep: onnx, model_path: ..., inf_func: ...}
  board: {rep: ipu_test, path: ..., cat_map: {...}}
```

**공통 키와 variant 를 어떻게 구분하는가** — 두 겹으로 판정한다.

1. **예약어 목록** — `gt` `classes` `mode` `conf` `cm_conf` `iou_thr` `cut`
   `merge_options` `target` `save_pred`. 이 이름은 공통 키다.
2. **구조 판정** — 예약어가 아닌 키는 variant 이며 **`rep` 를 반드시 가져야 한다.**
   없으면 `ValueError` 로 거부한다.

두 겹인 이유는 오타를 잡기 위해서다. `conff: 0.05` 라고 쓰면 예약어가 아니므로 variant 로
읽히는데 `rep` 가 없어 즉시 실패한다 — 조용히 무시되지 않는다. 반대로 variant 이름이
예약어와 겹치면(`--eval gt`) 이름 검증에서 거부한다.

`--eval <variant>` 가 가리키는 이름이 그대로 유지되므로 **명령어 인터페이스는 불변**이다
(§0.1).

**`Evaluator` 내부 형태** — 파일 형태와 내부 형태를 둘로 두지 않는다. `Evaluator.__init__`
이 평탄한 형태를 정본으로 받고, `DEFAULT_CFG` 와 `val/dair.py` 의 호출도 함께 맞춘다.
`evaluation.json` 의 `config.resolved` 는 최종 해석값이므로 형태가 바뀌어도 재현 근거로서
역할이 같다.

### 2.4 (3) 통폐합 — 파생 규칙

**정본을 하나로 정하고 나머지를 파생시킨다.** 명시하면 항상 명시가 이긴다.

| 값 | 정본 | 파생되는 곳 | 근거 |
|---|---|---|---|
| **ROI** | `exp_params.data_args.eval_roi_crop` | `eval.gt.roi` · variant 의 `crop` | §1.4 삼중 중복 제거. 정본은 이미 `(x,y,w,h)` 로 `roi` 와 같은 표기 |
| **클래스** | `overrides.data` → 데이터셋 yaml `names` | `eval.classes` | §1.6. `names` 는 `{0: bus, ...}` 이므로 키 정렬로 리스트가 된다 |
| **검증 이미지 경로** | `overrides.data` → `path` + `val` | `eval.gt.path` · `rep: onnx` variant 의 `path` | §1.6. 규칙은 `dirname(path/val)` — Corner 는 `/dataset/CVD/Corner/Validation` |
| **전처리 규약** | `exp_params.data_args.val_resize` + `input_scale` | variant 의 `pre_proc`·`inf_func` | §1.5. `stretch`+byte → `yolov8_bgr255_preproc`, `letterbox` → `letterbox_inference_v8` |
| **입력 크기** | ONNX 입력 shape | 없음 | 이미 자동 (`open_onnx_ession` 후 `shape[2:]`) |

파생 후 Corner v01 의 `eval:` 블록:

```yaml
eval:
  onnx:
    rep: onnx
    model_path: configs/releases/BSD_F606/Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx
  board:
    rep: ipu_test
    path: configs/releases/BSD_F606/Corner_Truck/v01/analysis/iputest/output.json
    cat_map: {BUS: 0, CAR: 1, MOTOR: 2, PERSON: 3, TRUCK: 4}
```

**51줄 → 8줄.** 남은 것은 파생할 수 없는 것뿐이다 — 모델 파일 위치, 보드 덤프 위치,
보드 카테고리 매핑.

**파생은 반드시 보이게 한다.** `--dry-run` 이 값과 함께 출처를 찍는다:

```
eval.board (해석 결과)
  gt.path      /dataset/CVD/Corner/Validation   ← overrides.data(configs/datasets/cvd_corner.yaml).path+val
  gt.roi       [428, 8, 1072, 1072]             ← exp_params.data_args.eval_roi_crop
  gt.roi_mode  center                           ← 기본값
  classes      [bus, car, motorcycle, person, truck]  ← 데이터셋 yaml names
  conf         0.01                              ← 기본값
  pred.cat_map {BUS: 0, ...}                     ← 명시
  save_dir     runs/detect/BSD_F606/Corner_Truck/v01/eval/board  ← 경로 파생
```

이것이 사용성의 핵심이다 — 파생은 편하지만 보이지 않으면 위험하다. `evaluation.json` 의
`params` 도 이미 최종값을 전부 기록하므로 사후 감사도 된다.

### 2.5 파생하지 않는 것

| 값 | 왜 파생하지 않는가 |
|---|---|
| `iou_thr` (NMS) | `overrides.iou` 는 학습·검증용 기본 0.7 이고 평가는 0.45 다. 의미가 달라 잇는 순간 수치가 바뀐다 |
| `conf` | `overrides.conf` 와 목적이 다르다 (D-020). 기본 0.01 고정 |
| `cat_map` | 보드 문자열은 모델·펌웨어가 정한다. 저장소 안에 정본이 없다 |
| `model_path` | 어느 ONNX 를 평가할지는 사람의 선택이다 (float/quantized/pruned) |

---

## 3. 스키마 전 / 후

### 3.1 학습 산출 디렉토리

```
변경 전                                    변경 후
runs/detect/<P>/<C>/<V>/                   runs/detect/<P>/<C>/<V>/
├── mycfg.yaml      ← 입력 사본             ├── exp.yaml     ← 같은 스키마, 재실행 가능
├── args.yaml       ← ultralytics           ├── args.yaml    (그대로)
├── training_log.log                        ├── training_log.log
├── weights/                                ├── weights/
└── eval/<variant>/                         └── eval/<variant>/
```

### 3.2 `eval:` 블록

```yaml
# ── 변경 전 (3단계, 51줄) ───────────────────────────────────────
eval:
  _shared:
    gt: {path: ..., rep: cxywhn, sep: ' ', scale: [1,1], roi: [428,8,1072,1072], roi_mode: center}
    classes: [bus, car, motorcycle, person, truck]
    mode: [ap, f1_score]
    conf: 0.01
    iou_thr: 0.45
    cut: 0.99
  onnx:
    pred:
      rep: onnx
      path: /dataset/CVD/Corner/Validation
      model_path: .../bsd_f606a_corner_320x320.onnx
      pre_proc:  {from_: ..., import_: yolov8_bgr255_preproc}
      post_proc: {from_: ..., import_: basic_postproc_v8}
      inf_func:  {from_: ..., import_: crop_inference_v8, args: {crop: [428,8,1072,1072]}}
  board:
    pred:
      rep: ipu_test
      path: .../analysis/iputest/output.json
      sep: ','
      scale: [1.0, 1.0]
      cat_map: {BUS: 0, CAR: 1, MOTOR: 2, PERSON: 3, TRUCK: 4}

# ── 변경 후 (2단계, 8줄) ────────────────────────────────────────
eval:
  onnx:
    rep: onnx
    model_path: configs/releases/BSD_F606/Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx
  board:
    rep: ipu_test
    path: configs/releases/BSD_F606/Corner_Truck/v01/analysis/iputest/output.json
    cat_map: {BUS: 0, CAR: 1, MOTOR: 2, PERSON: 3, TRUCK: 4}
```

명시적으로 덮어쓰고 싶으면 그냥 쓰면 된다 — 파생보다 항상 우선한다:

```yaml
eval:
  conf: 0.05                    # 기본 0.01 을 덮어쓴다
  gt: {roi: [0, 0, 1920, 1080]} # eval_roi_crop 파생을 덮어쓴다
  onnx: {rep: onnx, model_path: ...}
```

---

## 4. 구현 단계

각 단계 끝에 §0.2 를 돌린다. 수치가 어긋나면 그 단계는 실패다.

### ☑ 1. 데이터셋 yaml 20개의 `val` 레이아웃 조사 — 완료 (2026-08-04)

읽기 전용이라 계획 작성과 함께 수행했다. **파생 규칙이 검출 데이터셋 16개 전부에서
성립한다.**

| 결과 | 개수 | 내용 |
|---|---|---|
| `dirname(path/val)` 성립 | **16** | 검출 데이터셋 전부. `val` 이 `Validation/images`(대문자) 또는 `validation/images`(소문자) 형태이고 부모가 곧 split 디렉토리다 |
| 성립하지 않음 | 4 | `legacy/ncnvox_{cmd,kws,kws_full}.yaml`(오디오 classify — 검출 평가 대상이 아니다), `legacy/unlabeled.yaml`(`val` 키 자체가 없다) |
| `names` 가 연속 `0..n-1` 또는 리스트 | **20/20** | 클래스 파생에 구멍이 없다 |

대소문자가 데이터셋마다 다르다 — `cvd_corner` 는 `Validation`, `cvd_side` 는 `validation`.
파생은 yaml 의 문자열을 그대로 이어붙이므로 문제가 되지 않는다. 추측하지 않고 실측했다.

**결론** — `gt.path` 와 `classes` 파생을 3단계에서 무조건 구현할 수 있다. 오디오
데이터셋은 검출 평가 경로를 타지 않으므로 범위 밖이고, `names` 불연속 방어 코드는
그래도 넣는다(장래 데이터셋 대비).

### ☑ 2. `eval:` 평탄화 — medium

- **What**: (2). `_shared` 와 `pred` 래퍼 제거.
- **Where**: `tools/customEval.py`(`resolve_variant`), `projects/metrics/evaluator.py`
  (`__init__` 정규화·`DEFAULT_CFG`), `projects/src/models/yolo/detect/val/dair.py`
  (`Evaluator(None, ...)` 호출), Corner v01 `exp.yaml` 의 `eval:` 블록
- **How**: 예약어 목록 + `rep` 구조 판정(§2.3). variant 이름이 예약어면 `ValueError`.
  `rep` 없는 비예약 키도 `ValueError` — 오타를 잡는다.
- **Test**: 단위 — 예약어 충돌·`rep` 누락·평탄 파싱. **게이트 — 0.481 / 0.460.**

### ☑ 3. 파생 규칙 구현 + 출처 표시 — medium

- **What**: (3). ROI·클래스·경로·전처리 파생.
- **Where**: `tools/customEval.py`(파생 계층 신설), `projects/custom_utils/cfg.py`
  (데이터셋 yaml 로딩 헬퍼)
- **How**: 파생값마다 `(값, 출처 문자열)` 을 함께 만들어 `--dry-run` 이 §2.4 표 형태로
  찍는다. 명시값은 출처 `명시` 로 표시한다. `names` 가 0..n-1 연속이 아니면 `ValueError`.
  `pre_proc` 의 입력 규약 축은 **`weights/input_scale.json` sidecar 를 먼저 보고 없으면
  `byte`** 로 둔다 (6-4 결정, D-013 의 sidecar 기구를 재사용).
- **Test**: 단위 — 파생 5종 각각, 명시 우선, `names` 불연속 거부.
  **게이트 — `eval:` 을 8줄로 줄인 뒤 0.481 / 0.460 재현.** 이 단계가 §0 위험이 가장 크다.

### ☑ 4. `mycfg.yaml` → `exp.yaml` + 기록/입력 분리 — medium

- **What**: (1).
- **Where**: 트레이너 4곳(`custom.py:53` `dair.py:52` `symmetric.py:82` `draft.py:79`),
  `projects/custom_utils/cfg.py`(`save_my_cfg` 확장·`_` 키 무시 규칙),
  `tools/train_custom_trainer.py`(재실행 경고)
- **How**: `derive_run_dir`·`apply_cfg_options` 가 넣은 값을 `_resolved_` 로 분리해
  덤프한다(§2.2). 로더는 `_` 시작 최상위 키를 무시한다. `runs/` 아래 config 를 `-w` 로
  받으면 경고한다. **`save_my_cfg` 의 기존 `header` 인자에 생성 헤더를 넘긴다**(§2.2.1) —
  이 파일의 독자는 사람뿐이므로(6-2) 출처·재실행 방법·주석 소실 사실을 파일 안에서 알린다.
  `sort_keys=False`·`allow_unicode=True` 는 이미 그러하므로 손대지 않는다.
- **Test**: 복사-재실행 회귀 — `runs/.../exp.yaml` 을
  `configs/experiments/<P>/<C>/v_test/exp.yaml` 로 복사해 `--dry-run` 했을 때 출력 경로가
  **`v_test` 로 파생**되는지 (§1.7 함정 해소 확인). `symmetric` 의 `save_dir` 경로도 같이.

### ☑ 5. 기존 config 마이그레이션 — small

- **What**: `eval:` 블록을 가진 config 를 새 형태로.
- **Where**: 현재 1개 (`Corner_Truck/v01`).
- **How**: 학습 키는 **건드리지 않는다** — 동결 유지(§2.1). `eval:` 만 교체한다.
- **Test**: 게이트.

### ☑ 6. 문서 + D-021 — small

- **Where**: `docs/decisions.md`(D-021), `CLAUDE.md`, `README.md`,
  `docs/operations_guide.md`, `docs/diagrams.md`(`mycfg.yaml` 표기 3곳),
  `docs/research.md`, `docs/specs/eval_consolidation_plan.md`(§3.2 스키마 갱신)
- **How**: MANIFEST 의 기존 `as_run: weight/mycfg.yaml` 은 **과거 NAS 번들을 가리키므로
  고치지 않는다.** 신규 릴리스만 `as_run: weight/exp.yaml`. `_` 접두 키 규칙을 CLAUDE.md
  Conventions 에 적는다.
- **Test**: 문서의 명령을 복사해 실행.

---

## 5. 위험과 대응

| 위험 | 단계 | 대응 |
|---|---|---|
| **파생이 §0 수치를 바꾼다** | 3 | 단계마다 게이트. 파생값이 현재 명시값과 **바이트 단위로 같은지** 먼저 단정문으로 확인한 뒤 명시를 지운다 |
| `gt.path` 파생 규칙이 일부 데이터셋에서 틀린다 | 1·3 | 1단계에서 전수 조사. 성립하지 않으면 그 데이터셋은 명시 유지 |
| 파생이 보이지 않아 잘못된 값으로 오래 돈다 | 3 | `--dry-run` 출처 표시 + `evaluation.json.params` 기록. 두 곳에서 확인 가능 |
| 예약어/variant 판정이 오타를 삼킨다 | 2 | `rep` 필수 + 예약어 충돌 거부. 조용한 무시 경로를 만들지 않는다 |
| `_resolved_` 를 사람이 입력에 복사한다 | 4 | 로더가 `_` 키를 무시하므로 무해. 경고는 남긴다 |
| 덤프 파일명 변경이 외부 도구를 깨뜨린다 | 4 | 읽는 코드 0건 확인(§1.1). 외부에서 `mycfg.yaml` 을 읽는 스크립트가 있는지 사람 확인 필요 (§6-2) |
| `symmetric` 의 최상위 `save_dir` 이 `_resolved_` 로 옮겨져 학습이 깨진다 | 4 | `symmetric.py:81` 이 `custom_cfg.save_dir` 를 읽는다. 파생값을 `_resolved_` 에 **기록**하되 런타임 주입은 그대로 둔다 — 파일과 메모리를 구분한다 |
| 학습 1-step 스모크 없이 넘어간다 | 2·4 | 트레이너를 건드리는 4단계는 `epochs=1 fraction=0.004` 실제 실행 필수 |

---

## 6. 결정이 필요한 것

### 6.0 결정 결과 (2026-08-04)

| # | 항목 | 결정 |
|---|---|---|
| 6-1 | `_resolved_` 배치 | **블록으로 둔다** — 파일 하나로 완결되고 복사해도 무해하다 |
| 6-2 | 저장소 밖에서 `mycfg.yaml` 을 읽는 도구 | **없다.** 이 파일은 **사람이 읽는 용도**다 → §2.2.1 |
| 6-3 | `eval_roi_crop` 개명 | **개명하지 않는다.** 동결 릴리스를 고쳐야 하므로. 문서에서 "화각 정본"임을 명시한다 |
| 6-4 | `pre_proc` 파생의 입력 규약 축 | **`input_scale.json` sidecar 우선**, 없으면 `byte` 기본. variant 의 명시가 항상 이긴다 |
| 6-5 | `cls_weights` | **"미구현"으로 문서화하고 남긴다.** 동결 릴리스에 있어 제거는 동결을 깬다 |
| 6-6 | `exp_params.eval` 통합 | **하지 않는다 (안 A).** `eval:` 을 최상위에 유지한다 — 동결 블록 대조 능력을 지킨다 (§9.4) |
| 6-7 | 제거 예정 5키의 코드 삭제 | **이번 범위 밖.** 문서에서만 내린다 (§8.11). 코드 제거는 D-010 기각이므로 별도 D-번호 |

아래 6-1~6-7 항목은 그 결정에 이른 근거로 남긴다.


### 6-1. `_resolved_` 대신 별도 파일?

`runs/.../exp.yaml` 을 **사람이 쓴 것만** 담게 하고 파생값은 별도
`runs/.../resolved.yaml` 로 뺄 수도 있다.

| 안 | 장점 | 단점 |
|---|---|---|
| **`_resolved_` 블록** (권장) | 파일 하나로 완결. 복사해도 무해 | 파일이 길어진다 |
| 별도 파일 | 입력 파일이 깨끗하다 | 두 파일이 갈릴 수 있고, 하나만 복사하는 실수가 생긴다 |

### 6-2. 외부에서 `mycfg.yaml` 을 읽는 곳이 있는가

저장소 안에는 0건이다. **NAS 번들 관리·릴리스 스크립트·다른 사람의 로컬 도구**가 이 이름을
읽는지는 확인할 수 없다. 사람 확인이 필요하다. 확인 전이라면 4단계에서 `exp.yaml` 을 쓰고
`mycfg.yaml` 도 함께 남기는 과도기를 둘 수 있다 (비용은 고려하지 않아도 된다는 지침에 따라
과도기 없이 가는 것을 기본으로 둔다).

### 6-3. `eval_roi_crop` 이 정본이 되면 이름이 맞는가

정본이 되는 순간 이 키는 "평가용 ROI"를 넘어 **화각의 단일 정의**가 된다. 그런데 이름에
`eval` 이 들어 있어 학습·export 와 무관해 보인다. 이름을 바꾸면 동결된 릴리스 파일을
고쳐야 하므로 **바꾸지 않는 것을 권장**하고, 대신 문서에서 "이 키가 화각의 정본"임을
명시한다.

### 6-4. `val_resize` → `pre_proc` 파생의 두 번째 축

`stretch`/`letterbox` 만으로는 부족하다 — 입력 규약이 `[0,255]` BGR(우리 모델군)인지
`[0,1]` RGB(upstream export)인지가 갈린다. D-013 이 만든
`weights/input_scale.json` sidecar 가 그 정본이다.

| 안 | 내용 |
|---|---|
| sidecar 우선 (권장) | `input_scale.json` 이 있으면 그것을 읽고, 없으면 `byte` 기본 |
| variant 키 | `input_scale: byte\|unit` 을 variant 에 명시 |

ONNX 는 sidecar 가 가중치 옆에 있고 `network/` 에는 없으므로 **둘 다 지원**하고 명시가
이기게 한다.

### 6-5. `cls_weights` 를 어떻게 할 것인가 — §8.10

소비처 0건이고 동결 릴리스 `Side_Truck/v01` 에만 있다. 상세와 권장안은 §8.10.

### 6-6. `exp_params.eval` 통합 — 조사 완료, 결정 대기

§9 에 세 선택지와 근거를 정리했다. **A(최상위 유지)를 권장**한다 — 기술적으로는 통합이
가능하지만(학습 경로 무변화 확인), 동결 릴리스를 NAS 원본과 **블록 단위로 대조**하는 능력을
잃는 것이 §0 제약을 지키는 실무 수단을 없애는 일이다.

### 6-7. 제거 예정 5키의 코드까지 지울 것인가

`qat_params` `teacher_params` `nmqt_params` `burn_up_epochs` `burn_up_step` 을 문서에서
내리는 것(§8.11)과 **코드를 지우는 것은 다른 결정**이다. D-010 이 QAT 코드·config 보존을
명시했으므로 코드 제거는 D-010 기각에 해당하고, 이 계획의 범위가 아니다. 필요해지면 별도
D-번호로 결정한다. 이 계획에서는 **문서에서만 내린다.**

---

## 7. 인수 기준

| 항목 | 조건 |
|---|---|
| board mAP@0.50 | **0.481** (±0.002) |
| float mAP@0.50 | **0.460** (±0.002) |
| per-class AP | ±0.003 |
| GT-in-ROI | **21,742** 정확히 일치 |
| `num_dropped_by_cat_map` | 0 |
| pytest | `83 passed` 이상 / ruff 9건 이하 |
| 명령어 | §0.1 표의 명령이 전부 그대로 동작 |
| 복사-재실행 | `runs/.../exp.yaml` → `configs/experiments/.../v_test/` 복사 후 출력이 `v_test` 로 파생 |
| 학습 스모크 | `epochs=1 fraction=0.004` 완주, exit 0 |
| `eval:` 블록 | Corner v01 이 51줄 → 8줄 |
| `--dry-run` | 파생값마다 출처가 함께 출력 |
| 덤프 헤더 | `runs/.../exp.yaml` 첫 줄부터 생성 헤더가 있고, 입력 config 경로·재실행 방법·주석 소실 사실이 적혀 있다 |
| 키 순서 | 덤프의 키 순서가 입력 파일과 같다 (`sort_keys=False`) |

---

## 8. `exp.yaml` 설정 파라미터 전수

**코드에서 추출했다** — 실제로 읽히는 키만 적고, 읽히지 않는 키는 그렇다고 표시한다.
`읽는 곳` 이 **없음**인 키는 값을 바꿔도 아무 일도 일어나지 않는다.

표기 — `기본값` 은 코드의 fallback 이다. `필수` 는 없으면 실패하는 키.

### 8.1 최상위 키

| 키 | 타입 | 기본값 | 읽는 곳 | 비고 |
|---|---|---|---|---|
| `_base_` | path | — | `cfg.py:59 load_exp_yaml` | 현재 파일 기준 상대경로. deep-merge 후 **자식이 이긴다**. 순환 참조는 `ValueError` |
| `_meta_` | dict | — | **없음** (`train_custom_trainer.py:64` 가 출력만) | 사람용 출처·근거 기록. 자유 형식 |
| `model_name` | str | `custom` | `train_custom_trainer.py:67` | **`.get(key, (CustomYOLO, CustomDetectionTrainer))` 조회** — 미등록 키는 실패하지 않고 다른 아키텍처를 학습한다. 등록 키: `custom` `consi`(≠`consistency`) `dair` `dms_pose` `qat` `fx_qat` `symmetric`. 전체 표는 `docs/diagrams.md` |
| `dataset_name` | str | `dycrop` | `get_custom_dataset` | `dycrop` \| `default`(upstream) \| `unlabeled`. `custom`·`dair`·`consi` 에 필요 |
| `weight_path` | path | 필수 | `train_custom_trainer.py:65` | 시작 가중치 |
| `save_dir` | path | — | **`symmetric` 만** (`symmetric.py:82`) | `CustomDetectionTrainer` 계열은 **무시한다.** 그 계열은 `overrides.project`+`name`. `derive_run_dir` 이 해당하는 쪽만 채운다 |
| `overrides` | dict | `{}` | ultralytics trainer args (§8.2) | |
| `exp_params` | dict | `{}` | 트레이너·validator·loss (§8.3) | 프로젝트 전용 |
| `dair_params` | dict | — | `dair`·`consi` (§8.5) | 없으면 DAIR 증강 비활성 |
| `cls_weights` | dict | — | **없음** | `Side_Truck/v01` 에만 있고 **소비처 0건**. §1.5 의 `eval_roi_crop`·`val_resize` 와 같은 부류 |
| `eval` | dict | — | `tools/customEval.py` (§8.8) | D-020 신설. 학습은 읽지 않는다 |

### 8.2 `overrides` — ultralytics trainer args

우리가 정의하는 키가 아니라 **ultralytics 8.4.6 의 args 를 그대로 덮어쓰는 통로**다.
전체 목록은 `ultralytics/cfg/default.yaml` 이고, 이 저장소의 exp.yaml 10개가 실제로 쓰는
것은 **47종**이다.

| 빈도 | 키 |
|---|---|
| 10/10 | `imgsz` `epochs` `batch` `data` `workers` `bgr` |
| 8~9/10 | `lr0` `patience` `lrf` `optimizer` |
| 5~7/10 | `cls` `cos_lr` `degrees` `amp` `box` `dfl` `close_mosaic` `mosaic` `scale` `shear` `perspective` `hsv_h` `hsv_s` `hsv_v` `warmup_epochs` `weight_decay` |
| 2~4/10 | `copy_paste` `fliplr` `mixup` `translate` `multi_scale` `warmup_bias_lr` `warmup_momentum` |
| 1/10 | `auto_augment` `cache` `erasing` `flipud` `freeze` `kobj` `momentum` `plots` `pose` `save` `save_period` `seed` `simplify` `workspace` |

프로젝트에서 특별히 주의할 것:

| 키 | 주의 |
|---|---|
| `imgsz` | 리스트 `[320, 320]` 또는 스칼라. **`dycrop` 은 세로를 `exp_params.data_args.imgsz_y` 로 따로 받는다** |
| `bgr` | val 은 항상 `1.0` 고정, train 만 config 값 (D-004) |
| `workers` | 듀얼 GPU 호스트에서 **`<= 4`** (RAM 125 GB) |
| `amp` | Front 4K GA 튜닝은 **`false`** — fp16 gradient overflow → EMA NaN → checkpoint 저장 스킵 (D-006) |
| `data` | `configs/datasets/<d>.yaml` 경로. §2.4 의 `classes`·`gt.path` 파생이 이 파일을 읽는다 |
| `project` · `name` | **적지 않는다.** `derive_run_dir` 이 config 경로에서 파생한다 (명시하면 명시가 이기고 §1.7 함정이 생긴다) |
| `iou` · `conf` | 학습·검증용이다. 평가의 `eval.iou_thr`(0.45)·`eval.conf`(0.01)와 **의미가 다르니 잇지 않는다** (§2.5) |

### 8.3 `exp_params` — 프로젝트 전용

| 키 | 타입 | 기본값 | 읽는 곳 | 역할 |
|---|---|---|---|---|
| `rect_mode` | bool | — | `dair.py:70` | 직사각 배치 |
| `interp` | int | — | `dair.py:71` | cv2 보간 상수 (`1`=LINEAR, `3`=AREA) |
| `lr_scheduler` | bool | — | `dair.py:91` | 커스텀 스케줄러 사용 |
| `num_plots` | int | `3` | `dair.py:121` · `custom.py` | 배치 플롯 장수. `<=0` 이면 3 |
| `fitness_mode` | str | — | `dair.py:124` | 체크포인트 선정 기준. 예 `f1_50` |
| `bn_freeze` | bool/dict | `False` | `custom.py:182` | BatchNorm 동결 |
| `normalize` | dict | `None` | `custom.py:56-60` | `{scale, mean, std}` **3키 모두 필수**. 켜면 `mean·scale`/`std·scale` 로 정규화 — D-012·D-013 의 `[0,255]` 규약을 벗어나는 선택지 |
| `data_args` | dict | `DEFAULT_DYCROP_ARGS` | 데이터셋 빌더 (§8.4) | |
| `tal_topk` | int | `10` | `consistency.py:31` | TAL assigner top-k |
| `expand_tl` | str | `'none'` | `tal.py:298` | traffic light 확장 |
| `replace_tl_score` | list | `['none', 0]` | `tal.py:299` | traffic light 점수 치환 |
| `mask_occlusion` | float | `0.1` | `dms_pose.py:24` | DMS 마스크 가림 확률 |
| `sunglasses_occlusion` | float | `0.1` | `dms_pose.py:25` | DMS 선글라스 가림 확률 |
| `paper_texture_path` | path | `assets/textures/paper.jpg` | `dms_pose.py:26` | 가림 텍스처 |
| `landmark_weights` | bool | `True` | `loss.py:1187` | keypoint 부위별 가중 사용 |
| `kpt_w_jaw` | float | `0.5` | `loss.py:1189` | 턱 |
| `kpt_w_brow` | float | `1.0` | `loss.py:1190` | 눈썹 |
| `kpt_w_nose` | float | `1.0` | `loss.py:1191` | 코 |
| `kpt_w_eyes` | float | `1.0` | `loss.py:1192` | 눈 |
| `kpt_w_mouth` | float | `0.5` | `loss.py:1193` | 입 |
| `kpt_w_pupils` | float | `1.0` | `loss.py:1194` | 동공 |
| `shape_loss` | bool | `True` | `loss.py:1200` | 형상 손실 사용 |
| `shape_w_dir` | float | `0.06` | `loss.py:1202` | 방향 항 |
| `shape_w_len` | float | `0.045` | `loss.py:1203` | 길이 항 |
| `shape_w_pair` | float | `0.03` | `loss.py:1204` | 쌍 항 |

### 8.4 `exp_params.data_args` — DynamicCrop 과 증강

`dataset_name: dycrop` 일 때 `DynamicCropYOLODataset` → `DynamicCropWrapper` /
`v8_transforms_dycrop` 로 그대로 전달된다.

| 키 | 타입 | 기본값 | 읽는 곳 | 역할 |
|---|---|---|---|---|
| `crop_width_ratio` | float\|str | `1280/1920` | `data/dynamic_crop.py:50` | crop 가로 비율. `"1280/1920"` 분수 문자열 허용 (`_parse_ratio`) |
| `crop_height_ratio` | float\|str | `1024/1080` | `data/dynamic_crop.py:51` | crop 세로 비율 |
| `xy_margin` | tuple | `((0,0),(0,0))` | `data/dynamic_crop.py:53` | random crop 샘플링 마진 비율 `((x_low,y_low),(x_right,y_bottom))` |
| `imgsz_y` | int | `480` | `data/dataset/dynamic_crop.py:38` | **crop 후 세로 크기.** `overrides.imgsz` 가 가로. export·QAT 도 이 키를 읽는다 (`nova_qat_pipeline.py:83`, `run_finetune_pipeline_v2.py:767`) |
| `aug_type` | str | `'front'` | `augment.py:478` | 증강 프리셋. `front` \| `rear` \| `side` \| 그 외(=`Identity`, 증강 없음) |
| `coco_seg_overlay` | dict | `None` | `augment.py:479` | COCO 세그 오버레이. `front`·`rear` 프리셋에서 사용 |
| `complexity_paste` | dict | `None` | `augment.py:480` | 복잡도 붙이기. `front` 만 |
| `dark_paste` | dict | `None` | `augment.py:481` | 야간 붙이기. `front` 만 |
| `val_resize` | str | — | **없음** | `stretch` 등. **소비처 0건** — §2.4 에서 `pre_proc` 파생의 정본으로 승격 예정 |
| `eval_roi_crop` | list | — | **없음** | `(x, y, w, h)`. **소비처 0건** — §2.4 에서 **화각 정본**으로 승격 예정 |

> **`eval_roi_crop` 은 `(x, y, w, h)` 다** — `(x1, y1, x2, y2)` 가 아니다.
> README 입력 사양표의 Crop 좌표는 `(x1,y1,x2,y2)` 라 같은 영역을 다르게 적는다.
> Corner 의 `(428, 8, 1500, 1080)` = `eval_roi_crop: [428, 8, 1072, 1072]`.

> **`dynamic_crop.py` 는 두 파일이다** — `projects/src/data/dataset/dynamic_crop.py` 가
> `DynamicCropYOLODataset`(데이터셋, `imgsz_y` 를 읽는다), `projects/src/data/dynamic_crop.py`
> 가 `DynamicCropWrapper`(변환, crop 비율·마진을 읽는다). 위 표의 경로는 `projects/src/`
> 기준이다.

### 8.5 `dair_params` — DAIR / Consistency

| 키 | 역할 |
|---|---|
| `dair_method_mode` | 증강 일관성 결합 방식. `org` \| `or` \| `aug` \| `and` \| `prev` |
| `_lambda` | 일관성 손실 가중 |
| `gamma` | 일관성 손실 감쇠 |
| `num_choice` | `Select` 가 고를 증강 개수 (ColorJitter / RandomAdjustSharpness / GaussianBlur 중) |
| `assigner` | assigner 선택 |

### 8.6 · 8.7 제거 예정 키 — §8.11

`qat_params` · `teacher_params` · `nmqt_params` · `burn_up_epochs` · `burn_up_step` 은
**제거 예정**이다. 위 표에서 내리고 문서 하단 §8.11 에 주석 처리해 두었다.

### 8.8 `eval` — 평가 (§2.3 평탄화 후 형태)

**공통 키 (예약어)** — `eval:` 직하에 쓴다.

| 키 | 타입 | 기본값 | 역할 |
|---|---|---|---|
| `gt.path` | dir/json | 파생 | GT. 디렉토리면 `images/`+`labels/`. 데이터셋 yaml 에서 파생(§2.4) |
| `gt.rep` | str | `cxywhn` | `coco` \| `cxywhn` \| `xyxy` \| `xywh` |
| `gt.sep` | str | `' '` | 라벨 txt 구분자 |
| `gt.scale` | list | `[1.0, 1.0]` | 좌표 스케일 `[y, x]` |
| `gt.roi` | list | 파생 | `[cx, cy, cw, ch]`. **없으면 필터하지 않는다** (전체 GT 평가) |
| `gt.roi_mode` | str | `center` | GT 중심점 포함 판정. **다른 값은 `NotImplementedError`** |
| `classes` | list | 파생 | 순서가 class id. 데이터셋 yaml `names` 에서 파생 |
| `mode` | list | `[ap]` | `ap` \| `precision` \| `recall` \| `f1_score` \| `cm`. **미지원 값은 `ValueError`** |
| `conf` | float | **`0.01`** | 예측 생성 임계. AP·P·R·F1 전부에 영향 |
| `cm_conf` | float | `0.25` | ConfusionMatrix `default` 열. `conf` 와 **합치지 않는다** (D-020) |
| `iou_thr` | float | `0.45` | NMS IoU |
| `cut` | float | `0.99` | 3단계 임계값 구간을 정하는 maxF1 비율 |
| `merge_options` | dict | `{}` | `{merged_ids, merged_classes}` 클래스 병합 |
| `target` | dict | — | `{ap: {클래스: 값}, f1: {...}}`. 있으면 PASS/Non-Pass 판정 + 미달 시 종료 코드 1 |
| `save_pred` | bool | `false` | `prediction.json` 저장 (`--save-pred`) |

**variant 키** — 예약어가 아닌 이름이 variant 이고 **`rep` 를 반드시 가진다**(§2.3).
이름이 곧 출력 디렉토리(`eval/<variant>/`)이고 `--eval <variant>` 로 지정한다.

| 키 | 타입 | 기본값 | 역할 |
|---|---|---|---|
| `rep` | str | **필수** | `onnx`(추론) \| `ipu_test`·`sim`(보드 덤프) \| `coco` \| `cxywhn`·`xyxy`·`xywh`(txt) |
| `path` | dir/file | 파생 | `rep: onnx` 면 추론 이미지 소스, 그 외는 예측 파일 |
| `model_path` | path | — | `rep: onnx` 필수. ONNX 경로. 입력 크기는 여기서 자동 |
| `cat_map` | dict | — | 예측 카테고리 → 클래스 인덱스. 숫자·문자열 키 모두, 문자열은 대소문자 무시. **미매핑은 drop 되고 개수가 `evaluation.json` 에 남는다** |
| `sep` | str | `,` | 예측 txt 구분자 |
| `scale` | list | `[1.0, 1.0]` | 예측 좌표 스케일 `[y, x]` |
| `pre_proc` | dict | 파생 | `{from_, import_, args}`. `yolov8_bgr255_preproc`(우리 모델군) \| `yolov8_std_preproc`(upstream) |
| `post_proc` | dict | `basic_postproc_v8` | 2-출력 split-head. 단일 출력은 `yolov8_std_postproc` |
| `inf_func` | dict | 파생 | `crop_inference_v8`(화각 crop) \| `letterbox_inference_v8` \| `basic_inference_v8`(전체 화각) |
| `crop` | list | 파생 | `crop_inference_v8` 의 crop 영역. `eval_roi_crop` 에서 파생(§2.4) |

**`gt.roi` 와 `crop` 은 다른 기구이고 둘 다 필요하다** — `crop` 은 *입력*을 잘라 보드
화각을 재현하고 `gt.roi` 는 *GT* 를 ROI 로 제한한다. 하나만 켜면 수치가 어긋난다.

### 8.9 소비처 0건 키 (값을 바꿔도 아무 일도 없다)

| 키 | 어디에 있는가 | 계획상 처리 |
|---|---|---|
| `exp_params.data_args.eval_roi_crop` | 3개 exp.yaml | **화각 정본으로 승격** (§2.4) |
| `exp_params.data_args.val_resize` | 2개 exp.yaml | **`pre_proc` 파생 정본으로 승격** (§2.4, §6-4) |
| `cls_weights` | `Side_Truck/v01` | 미결 — 구현하거나 문서에 "미구현"으로 표기 (§8.10) |
| `_meta_` | 10/10 | 의도된 것 (사람용 기록) |

### 8.10 이 절에서 새로 드러난 미결

**6-5. `cls_weights` 를 어떻게 할 것인가.** `Side_Truck/v01/exp.yaml` 에만 있고 읽는 코드가
없다. 클래스 불균형 가중을 의도한 키로 보이며 `configs/datasets/cvd_side.yaml` 주석이
"`mycfg.yaml` 의 `cls_weights`" 를 언급한다. 세 갈래다 — (a) 구현한다, (b) 문서에
"미구현"으로 명시하고 남긴다, (c) 제거한다. `Side_Truck/v01` 은 동결 릴리스이므로 (c) 는
동결을 깬다. **(b) 를 권장**하고, 구현이 필요하면 별도 D-번호로 뗀다.

### 8.11 제거 예정 키 — 주석 보존

아래 5개는 **제거 예정**이다. 신규 config 에 쓰지 않는다. 코드는 남아 있으므로 값을 넣으면
아직 동작하지만, 문서상 지원 대상이 아니다.

```yaml
# ── 제거 예정 (신규 config 에 쓰지 않는다) ─────────────────────────────
#
# qat_params:            # QAT — D-010 으로 보류. qat·fx_qat 트레이너가 읽는다
#   target: novatek_pct_8a12w   # quantization 타깃
#   qat_warmup_epochs: 8        # QAT 시작 전 float warmup
#   quantize_head: false        # head 양자화 여부
#   percentile: 99.99           # activation 관측 percentile
#   fx_overrides: {}            # FX 그래프 모드 전용 오버라이드
#   npu_strategy_path: ...      # NPU 전략 파일 경로
#
# nmqt_params: {}        # NMQT QAT 변형. nmqt 툴킷이 미선언 의존성이라 실행 자체가 불가
#
# teacher_params:        # Knowledge Distillation — draft.py 만 읽는다
#   model: ...                  # teacher 가중치
#   ema_keep_rate: 0.999        # teacher EMA 유지율
#   teacher_update_epochs: 1    # teacher 갱신 주기
#   unlabeled_data: ...         # 비라벨 데이터셋
#
# burn_up_epochs: 10     # KD — teacher 갱신 시작 epoch
# burn_up_step: 1000     # KD — teacher 갱신 시작 step
```

**제거 근거와 영향 범위** (실측):

| 키 | 읽는 코드 | 이 키를 쓰는 config | 판단 |
|---|---|---|---|
| `qat_params` | `fx_qat.py` · `base_qat.py` · `run_finetune_pipeline_v2.py` | `configs/experiments/legacy/{qat,fx_qat,front_finetune,d22_front_finetune}/` 5개 | **D-010 으로 QAT 보류.** 코드·config 는 보존 지시가 있으므로(D-010) 삭제하지 않고 문서에서만 내린다 |
| `nmqt_params` | `nmqt_qat.py` · `nova_qat_pipeline.py` | **0개** | `nmqt` 툴킷이 `pyproject.toml` 에 없어 import 부터 실패한다. 실행 불가 상태 |
| `teacher_params` | `kd/train/draft.py` | `configs/experiments/legacy/kd_large/` 1개 + archive 3개 | KD 는 실험 잔존물이다 (`kd_draft` 는 README 에서도 미사용 표기) |
| `burn_up_epochs` · `burn_up_step` | `kd/train/draft.py` | 같음 | `teacher_params` 와 한 묶음 |

**주의 — 활성 트리에 아직 남아 있다.** `qat_params` 는 `configs/experiments/legacy/` 5개
파일에, `teacher_params`·`burn_up_*` 는 `configs/experiments/legacy/kd_large/kd_test.yaml`
에 있다. `configs/experiments/legacy/` 는 archive 가 아니라 **활성 트리이고 `tools/` 기본값이
아직 여기를 가리킨다**(CLAUDE.md). 따라서:

- **코드는 지우지 않는다** — D-010 이 QAT 코드·config 보존을 명시했고, KD 도 같은 성격이다.
- 문서에서만 내리고, 위 주석 블록이 "있었다"는 기록이 된다.
- 실제 코드 제거가 필요해지면 **별도 D-번호**로 결정한다. D-010 을 기각하는 형태가 되므로
  이 계획의 범위가 아니다.

**정본 릴리스 10개 중 이 5키를 쓰는 것은 0개다** — 제거 예정 표기가 현재 출하 경로에
영향을 주지 않는다.

---

## 9. `exp_params` 에 `eval` 을 통합할 수 있는가 — 조사

### 9.1 기술적으로 가능한가 — 가능하다

| 확인 항목 | 결과 |
|---|---|
| `exp_params` 를 통째로 펼치거나 순회하는 코드 | **0건.** 전부 키 단위 `.get()` / `[]` 접근 (§8.3 의 24개 지점 전수) |
| `exp_params` 가 ultralytics args 로 흘러가는가 | **아니다.** `overrides.update(self.custom_cfg.overrides)` 로 **`overrides` 만** 넘어간다 (5개 트레이너 전부 동일) |
| `custom_cfg` 를 순회하는 코드 | **0건** (`custom.py:34` 는 주석, `nova_qat_pipeline.py:546` 은 `custom_cfg['overrides']` 만) |
| 미지 키가 에러를 내는 지점 | 없음 — `exp_params` 에 `eval` 을 넣어도 학습은 무반응 |

즉 **`exp_params.eval` 로 옮겨도 학습 경로는 한 줄도 바뀌지 않는다.** 바뀌는 것은 읽는 쪽
3곳뿐이다.

```
tools/customEval.py:199   cfg.get('eval')            → cfg.get('exp_params', {}).get('eval')
tools/customEval.py:206   raw_eval 순회               (동일)
tests/test_eval_baseline.py:50  cfg['eval']          → cfg['exp_params']['eval']
```

**명령어 인터페이스는 불변이다** — `-w`·`--eval <variant>` 는 그대로다 (§0.1).

### 9.2 통합의 이점

| 이점 | 내용 |
|---|---|
| **파생이 한 블록 안에서 닫힌다** | §2.4 의 화각 정본 `exp_params.data_args.eval_roi_crop` 과 소비처 `eval.gt.roi` 가 같은 `exp_params` 안에 놓인다. 최상위 키를 넘나들지 않으므로 "ROI 는 `exp_params` 에서 정한다"가 한 문장으로 성립한다 |
| **성격이 맞는다** | `exp_params` 는 "ultralytics 가 모르는 우리 전용 설정"이다. `eval` 이 정확히 그것이다 |
| **이미 평가 키가 거기 있다** | `data_args.eval_roi_crop` 과 `val_resize` 는 이름부터 평가용인데 `exp_params` 에 있다 (§1.5). `eval` 을 최상위에 두면 평가 설정이 **두 최상위 키에 흩어진** 상태가 유지된다 |
| **최상위가 얕아진다** | 15종 → 14종. 최상위는 스캔 대상이므로 적을수록 읽기 쉽다 |

### 9.3 통합의 대가

| 대가 | 내용 |
|---|---|
| **동결 릴리스의 검증이 어려워진다** | ⚠ **가장 무거운 항목.** D-020 은 `eval:` 을 최상위에 **추가**했기 때문에 "학습 키를 한 자도 건드리지 않았다"가 성립했고, 그것이 동결 예외의 근거였다. `exp_params` 안으로 들어가면 `exp_params` 블록의 diff 가 비지 않아 **NAS 원본과 블록 단위로 대조하는 검증이 깨진다** |
| **경로가 한 단계 깊어진다** | `exp_params.eval.gt.roi` (4단계) 대 `eval.gt.roi` (3단계). §2.3 에서 `_shared`·`pred` 를 없애 한 단계 줄인 것을 부분적으로 되돌린다 |
| `--cfg_options` 가 길어진다 | `--cfg_options exp_params.eval.conf=0.05` 대 `eval.conf=0.05`. 자주 쓰는 인자라 체감된다 |
| 소비 시점이 다르다 | `exp_params` 는 학습 루프가 매 epoch 읽고, `eval` 은 별도 도구가 학습 후에 읽는다. 같은 블록에 두면 같은 종류로 보인다 |

### 9.4 선택지 세 개

| 안 | 내용 | 평가 |
|---|---|---|
| **A. 최상위 유지** (권장) | `eval:` 을 최상위에 그대로. 파생으로만 `data_args` 와 잇는다 | 동결 검증이 보존되고 경로가 얕다. §9.2 의 "성격" 이점은 문서로 설명 가능하다 |
| B. `exp_params.eval` 로 통합 | 옮긴다 | 개념 정합성은 최상이나 동결 검증이 깨지고 경로가 깊어진다 |
| C. 반대 방향 — 평가 키를 `eval` 로 모은다 | 신규 config 는 `eval.gt.roi` 를 직접 쓰고 `data_args.eval_roi_crop` 은 동결 파일 호환용으로만 읽는다 | 개념 정합성을 얻으면서 동결도 지킨다. 다만 파생 정본이 **두 곳**이 되어 §2.4 의 "정본 하나" 원칙이 흔들린다 |

**권장은 A** — 이유는 셋이다.

1. §0 제약이 "수치 동일성"이고, 동결 릴리스와 NAS 원본을 **블록 단위로 대조**하는 능력이
   그 제약을 지키는 실무 수단이다. B 는 그 수단을 없앤다.
2. §9.2 의 이점 중 실질은 "파생이 한 블록에서 닫힌다" 하나이고, 그것은 `--dry-run` 이
   출처를 함께 찍으면(§2.4) 문서·도구로 해결된다. 스키마를 바꿀 필요가 없다.
3. B 의 이득은 **읽기 편의**인데 대가는 **경로 길이**여서 서로 상쇄된다.

**A 를 택하면 남는 숙제** — 평가 설정이 `exp_params.data_args`(정본)와 `eval`(소비)로
나뉘어 있다는 사실을 문서에 명시해야 한다. §8.4 의 `eval_roi_crop` 행과 §2.4 의 파생 표가
그 역할을 한다.

### 9.5 어느 안을 택하든 바뀌지 않는 것

| 항목 | 이유 |
|---|---|
| 명령어 | `-w` · `--eval` · 14개 플래그 전부 그대로 (§0.1) |
| 수치 | `eval` 의 **위치**만 바뀌고 **값**은 그대로다. 게이트로 확인 |
| variant 이름 | `onnx` / `board` 가 출력 디렉토리와 `--eval` 인자를 결정한다 |
| `runs/.../eval/<variant>/` | 산출물 경로는 config 경로에서 파생되므로 무관 |

---

## 10. 구현 기록 (2026-08-04)

결정 기록은 **D-021**. 각 단계 끝에 §0.2 게이트를 돌렸고 전부 무변화였다.

### 10.1 단계별 결과

| 단계 | 결과 | 게이트 |
|---|---|---|
| 1. 데이터셋 `val` 레이아웃 조사 | 검출 데이터셋 **16/16** 성립 | 코드 변경 없음 |
| 2. `eval:` 평탄화 | 3단계 → 2단계. `_shared`·`pred` 제거 | 0.481 / 0.460, 94 passed |
| 3. 파생 + 출처 표시 | ROI 삼중 중복 제거 | 0.481 / 0.460, 105 passed |
| 4. `exp.yaml` 대체 | 헤더·`_resolved_` 분리·복사 재실행 | 0.481 / 0.460, 111 passed, **학습 exit 0** |
| 5. config 마이그레이션 | Corner v01 `eval:` 51 → 23줄 | 게이트 포함 |
| 6. 문서 + D-021 | `CLAUDE.md` · `README.md` · `operations_guide` · `diagrams` · `research` | — |

### 10.2 의도한 계획 이탈 2건

**① 내부 cfg 형태는 평탄화하지 않았다** (계획 §2.3 이 하라고 했다).

`eval:` 최상위로 예측 키를 펼치면 `path`·`rep`·`sep`·`scale` 이 `gt` 블록의 **동명 키와
나란히** 놓인다. `sep: ','` 를 GT 의도로 쓰면 조용히 예측 구분자가 되고, 그것은 좌표가
틀어지는 오류다 — §0 이 막으려는 부류다. 그래서 **사람이 쓰는 파일 형태만 평탄**하게 두고
variant 본문은 `resolve_eval_variant` 가 `pred` 로 정규화한다. 정규화 형태는 사람이 손으로
쓰지 않으므로 "두 형태"의 문제가 실제로 발생하지 않고, 덕분에 `Evaluator`·`DEFAULT_CFG`·
`val/dair.py` 를 손대지 않아 §0 위험도 줄었다.

**② `conf`·`iou_thr`·`cut` 을 릴리스 config 에 계속 명시했다.**

세 값 모두 코드 기본값과 같아 지울 수 있었다. 지우지 않은 이유는 하나다 — **나중에 기본값이
바뀌면 동결 릴리스의 기록 수치가 조용히 움직인다.** §0 이 줄 수보다 앞선다. 그래서 Corner
v01 의 `eval:` 은 계획이 예상한 8줄이 아니라 **51 → 23줄**이 됐다. 실제 목표였던 ROI 삼중
중복 제거와 데이터셋 중복 제거는 그대로 달성했다.

### 10.3 파생 안전 절차 — 실제로 걸러낸 것

계획 §5 대로 **명시를 지우기 전에** 파생값이 현재 명시값과 바이트 단위로 같은지 대조했다.
그 대조가 실제로 하나를 걸러냈다 — `board.pred.path`(보드 덤프 위치)가 DIFF 로 나왔고,
확인해 보니 **설계상 파생 대상이 아닌 키**(§2.5)를 프로브가 잘못 지운 것이었다. 규칙을
바꾸지 않고 프로브를 고쳤다. 나머지 파생값 7종은 전부 SAME 이었다.

이 절차가 없었다면 `board.pred.path` 를 지우고 나서야 "예측 파일이 없다"로 실패했을 것이다.

### 10.4 §1.7 함정 해소 확인

실제 파일로 검증했다.

```
runs/detect/BSD_F606/Corner_Truck/v01_smoke/exp.yaml   ← 학습이 남긴 기록
  overrides.project : (없음)      ← _resolved_.derived 로 분리됨
  overrides.name    : (없음)
  overrides.epochs  : 1           ← --cfg_options 값은 사람이 쓴 블록에 반영

→ configs/experiments/BSD_F606/Corner_Truck/v01_smoke2/exp.yaml 로 복사 후 해석
  project/name : BSD_F606/Corner_Truck  v01_smoke2      ← v01_smoke 가 아니다
  로드 직후 기록 키 : 없음 (버려짐)
```

`symmetric` 의 최상위 `save_dir` 도 같은 함정이 있어 단위 테스트로 함께 잠갔다.

### 10.5 새로 만든 인터페이스

`projects/custom_utils/cfg.py` — CLI 와 테스트가 **한 구현**을 쓴다 (해석 규칙이 두 곳으로
갈리면 테스트가 CLI 와 다른 것을 검증하게 된다).

| 함수 | 역할 |
|---|---|
| `split_eval_block(raw_eval)` | 예약어 = 공통, 나머지 = variant(`rep` 필수) |
| `resolve_eval_variant(raw_eval, variant)` | 공통 + variant → `Evaluator` 입력 |
| `eval_variant_names(raw_eval)` | `--eval` 기본값 |
| `derive_eval_settings(cfg, variant_cfg)` | 파생 + `(값, 출처)` 반환 |
| `derive_eval_dir(cfg_path, cfg, variant)` | 산출물 경로 (`derive_run_dir` 과 같은 3중 gate) |
| `record_run_context(cfg, cfg_path, command)` | 트레이너가 쓸 출처를 CLI 가 기록 |
| `save_run_config(cfg, save_dir, ...)` | `save_dir/exp.yaml` + 헤더 + 기록/입력 분리 |
| `load_dataset_yaml` · `derive_classes` · `derive_gt_path` · `read_input_scale` | 파생 재료 |

### 10.6 남은 것

| 항목 | 상태 |
|---|---|
| `cls_weights` | 소비처 0건. 동결 릴리스에만 있어 "미구현" 문서화만 (6-5) |
| 제거 예정 5키의 코드 | 문서에서만 내렸다. 코드 제거는 D-010 기각이라 별도 D-번호 (6-7) |
| `runs/detect/.../v01_smoke` | 스모크 산출물이 디스크에 남았다 — `runs/` 삭제가 에이전트에게 거부돼 있어 사람이 지운다. `.gitignore` 대상이라 저장소 영향 없음 |
