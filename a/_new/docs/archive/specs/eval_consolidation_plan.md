# 평가 기능 통폐합 — 설계 계획

**상태**: **구현 완료** (2026-08-04, 브랜치 `feat/eval-consolidation`). 결정 기록은 **D-020**.
7단계 전부 수행했고 §7.5 회귀 게이트를 통과했다 — board **0.481** / float **0.460** /
GT-in-ROI **21,742**, 모두 Δ +0.000. 구현 중 확정·변경된 사항은 §10 에 모았다.
**작성**: 2026-08-04, HEAD `10f2f91`
**정본 결정**: 이 저장소의 `projects/metrics/` 를 평가 정본으로 한다.
`~/Downloads/ncn_metric/` 오버레이는 흡수한다.

참고 코드: `~/Downloads/ncn_metric/` (드라이버 660줄 — `eval.py` 302 ·
`nvt_eval_funcs.py` 218 · `board_json_merge.py` 38 · `run.sh` 28 · config 3개)

---

## 0. 최우선 제약 — 수치 동일성

> **같은 모델과 같은 데이터를 넣으면 같은 결과가 나와야 한다.**
> 이 제약이 다른 모든 목표(코드 정리·인터페이스 통일·기능 추가)보다 앞선다.
> 어떤 단계든 §7의 기준값을 허용 오차 안에서 재현하지 못하면 **그 단계는 실패로 간주하고
> 되돌린다.** 리팩터링의 편의나 코드 미관을 위해 수치를 바꾸지 않는다.

### 0.1 구속되는 기준값

`CVD/Corner/Validation` (이미지 5,000 / GT 30,567), ROI `[428, 8, 1072, 1072]`,
`conf=0.01` · `iou=0.45`, mAP@0.50.

| 대상 | 기준값 | per-class |
|---|---|---|
| board (`ipu_test`) | **0.481** | bus 0.416 · car 0.414 · motorcycle 0.549 · person 0.523 · truck 0.504 |
| float (`onnx`) | **0.460** | bus 0.284 · car 0.393 · motorcycle 0.549 · person 0.571 · truck 0.504 |
| GT (ROI 내부) | **21,742** 박스 | — |

허용 오차 — per-class **±0.003**, mAP **±0.002**, GT 개수 **정확히 일치**.
근거와 재현 기록은 §7.

### 0.2 각 단계에서 지켜야 할 것

1. **단계를 시작하기 전에** 현행 기준값을 다시 측정해 기록한다 (환경 변화 배제).
2. **단계를 마친 직후** 같은 명령으로 재측정해 §0.1과 대조한다.
3. 수치가 어긋나면 **원인을 찾을 때까지 다음 단계로 넘어가지 않는다.** 허용 오차 안의
   변화도 원인을 설명할 수 있어야 한다.
4. GT 개수는 오차를 허용하지 않는다 — 21,742가 아니면 ROI 판정이 바뀐 것이다.
5. 리팩터링 단계(1·6)는 **수치가 한 자리도 바뀌지 않아야** 한다. 기능 추가 단계(2·3·4)는
   새 경로가 기준값을 재현해야 하고 기존 경로는 무변화여야 한다.

### 0.3 회귀 확인 명령

구현 후 이 두 명령이 §0.1을 재현해야 한다 (§4.4에 상세).

```bash
W=configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml
python3 tools/customEval.py -w $W --eval board   # → mAP 0.481
python3 tools/customEval.py -w $W --eval onnx    # → mAP 0.460
```

---

## 1. 배경

검출 정확도를 계산하는 코드 경로가 **5개**로 갈라져 있고, 보드(NPU) 평가에 필요한 기능은
저장소 밖에만 있다. `ncn_metric` 드라이버는 우리 `projects/metrics/` 를 호출하도록
설계됐지만 **현재 우리 CLI로는 실행되지 않는다**.

### 1.1 정확도 계산 경로 5개 (실측)

| # | 위치 | 코어 |
|---|---|---|
| 1 | `projects/metrics/evaluator.py:176` | `customCOCOeval` (pycocotools 확장) ← **정본 후보** |
| 2 | `tools/eval_test.py:92` | raw `COCOeval` — 우리 확장을 우회 |
| 3 | `projects/src/models/yolo/detect/val/dair.py:76` | inline `from pycocotools.coco import COCO` |
| 4 | `projects/src/utils/metric.py:41` | 자체 `ap_per_class` (upstream `compute_ap` 기반) |
| 5 | `tools/eval_det_txt.py:23` | upstream `ap_per_class` 직접 |

### 1.2 중복 (바이트 단위 동일)

`cxywhn2xywh` · `xyxy2xywh` · `xywh2xywh` · `to_xywh` 네 함수가
`projects/metrics/utils.py` 와 `projects/custom_utils/drawing.py` 에 **완전히 동일하게** 존재.

### 1.3 결손 — 참고 코드에는 있고 우리에게 없는 것

| 항목 | 근거 |
|---|---|
| `cat_map` | 저장소 전체 검색 **0건**. config 3개가 모두 사용 |
| `cfg['target']` 소비 | `evaluator.py` 에서 **읽히지 않는다** (`target_mode` 와 이름만 겹침) |
| PASS/Non-Pass 판정 표 | `"AP per class for All Dataset"` 문자열이 저장소에 **없다** |
| 기계 판독 결과 출력 | `save_result()` 가 `# TODO : Not implemented` 스텁 — `fp_list.txt`/`fn_list.txt` 만 씀 |
| 이어붙인 보드 JSON | `parse_ipu_result:579` 가 `json.load` 단발 → `Extra data` 실패 |
| crop / letterbox 화각 통제 | `basic_inference_v8` 은 plain resize 고정 |
| ONNX CPU 폴백 | `utils.py:230` 이 `['CUDAExecutionProvider']` 단독 |
| 이미지 읽기 실패 처리 | `img is None` 검사 없음 → `img.shape` 에서 죽는다 |

### 1.4 왜 드라이버가 stdout을 파싱하는가

`eval_()` 의 `save_result(results)` 호출이 **`if 'cm' in target_mode:` 블록 안에**
있다(`evaluator.py:254-265`). config 의 `mode` 에 `cm` 이 없으면 결과가 디스크에 전혀
남지 않는다. 그래서 `ncn_metric/eval.py:116 parse_map()` 이 로그 표를 정규식으로 긁는다.

**이 계획의 핵심 목표 하나가 이 우회를 없애는 것이다** — 결과를 JSON으로 내보내면
드라이버든 CI든 stdout을 파싱할 필요가 없다.

### 1.5 인터페이스 불일치

드라이버(`eval.py:197`)가 넘기는 인자와 우리가 받는 인자가 다르다.

| 드라이버가 넘김 | `tools/customEval.py` (18줄) |
|---|---|
| `<config>` positional | ✓ `config_file` |
| `--data` `--save-dir` `--pred` `--no-merge-motor-person` | ✗ **전부 없음** → `unrecognized arguments` |

---

## 2. 기능 리스트

`현황` 열: ✓ 있음 · △ 부분 · ✗ 없음

### 2.1 입력 처리

| ID | 기능 | 현황 | 목표 |
|---|---|---|---|
| F-01 | GT 로드 — COCO json | ✓ | 유지 |
| F-02 | GT 로드 — YOLO txt (`cxywhn`/`xyxy`/`xywh`) | ✓ | 유지 |
| F-03 | 예측 로드 — COCO json | ✓ | 유지 |
| F-04 | 예측 로드 — 보드 덤프 (`ipu_test`) | △ 단일 배열만 | **이어붙인 배열 지원** |
| F-05 | 예측 로드 — 시뮬레이터 (`sim`) | △ | `ipu_test` 와 동일 경로로 통합 |
| F-06 | 예측 생성 — ONNX 추론 (`onnx`) | ✓ | CPU 폴백·읽기실패 처리 추가 |
| F-07 | 클래스 매핑 `cat_map` (숫자/문자열 키, 미매핑 drop) | ✗ | **신설** |
| F-08 | 보드 점수 스케일 보정 (`>1` → `/100`) | ✓ `score_` | 유지 |
| F-09 | 좌표 스케일 `scale: [y, x]` | ✓ | 유지 |

### 2.2 추론 변형 (화각·전처리 통제)

| ID | 기능 | 현황 | 목표 |
|---|---|---|---|
| F-10 | plain resize + BGR `[0,255]` | ✓ | 유지 (기본) |
| F-11 | plain resize + RGB `/255` (표준 ultralytics) | ✗ | **신설** |
| F-12 | **crop FOV 재현** — 보드 crop 화각 일치 | ✗ | **신설 (최우선)** |
| F-13 | letterbox 전처리 + 역매핑 | ✗ | **신설** |
| F-14 | 2-출력 split-head 디코드 (`dist2bbox`) | ✓ `basic_postproc_v8` | 유지 |
| F-15 | 단일 출력 `[1,4+nc,N]` 디코드 | ✗ | **신설** |
| F-34 | **GT ROI 필터** — ROI 밖 GT를 평가에서 제외 | ✗ | **신설 (필수)** |

> **F-12 와 F-34 는 다른 기구다.** F-12는 *입력*을 crop해 추론하고, F-34는 *GT*를 ROI로
> 제한한다. 둘을 어떻게 조합하느냐로 수치가 갈린다 —
> F-12만 쓰면 ROI 밖 객체가 FN이 되어 mAP가 낮아지고(참고 README의 crop-FOV float 0.378),
> F-12+F-34를 함께 쓰면 ROI 밖이 평가에서 빠진다(우리 기록 float **0.460**).
> §7의 재현 검증은 **F-12+F-34 조합**이 기록값을 정확히 재현함을 확인했다.
> config 키는 `gt.roi: [cx, cy, cw, ch]` 로 두고, 지정 시 GT 중심이 ROI 내부인 것만 남긴다.

### 2.3 지표 산출

| ID | 기능 | 현황 | 목표 |
|---|---|---|---|
| F-16 | mAP / per-class AP (pycocotools) | △ **계산되지만 출력 안 됨** | **`ap` mode 신설** |
| F-17 | precision / recall (IoU@50, @75, @50-95) | ✓ | 유지 |
| F-18 | **3단계 confidence 임계값** (sensitive/optimal/insensitive) | ✓ `f1_score` — 이미 구현 | 유지 + 출력 연결 |
| F-35 | 임계값 산출 독립 실행 (모델·예측파일 양쪽 입력) | ✗ | **신설 검토** (§3.5) |
| F-19 | Confusion matrix + FP/FN 목록 | ✓ `cm` | 유지 |
| F-20 | **`target` 대비 PASS/Non-Pass 판정** | ✗ | **신설** |
| F-21 | 클래스 병합 (`merge_options`) | ✓ | 유지 |

### 2.4 출력

| ID | 기능 | 현황 | 목표 |
|---|---|---|---|
| F-22 | **`evaluation.json` 기계 판독 결과** | ✗ 스텁 | **신설** |
| F-23 | `fp_list.txt` / `fn_list.txt` | ✓ | 유지 |
| F-24 | 콘솔 표 (AP/P/R per class) | △ P/R만 | **AP 표 + PASS 열 추가** |
| F-25 | confusion matrix 이미지 | ✓ | 유지 |
| F-26 | mc_curve 플롯 | ✓ | 유지 |
| F-27 | **`mode` 와 무관하게 결과 저장** | ✗ `cm` 필요 | **수정** |

### 2.5 실행·설정

| ID | 기능 | 현황 | 목표 |
|---|---|---|---|
| F-28 | YAML 설정 파일 단일 구동 | ✓ | **학습 `exp.yaml` 의 `eval:` 블록으로 통합** (§3.0) |
| F-29 | CLI 인자로 설정 덮어쓰기 | ✗ | **신설** |
| F-30 | 경로 토큰 치환 (`{datasets}` `{weights}`) | ✗ | **범위 제외** — 학습 config 통합으로 불필요 (§3.0) |
| F-31 | `from_`/`import_` 동적 함수 주입 | ✓ `from_import` | 유지 |
| F-32 | `--dry-run` 해석 결과 확인 | ✗ | **신설** |
| F-33 | 여러 config 연속 실행 + 비교표 | ✗ | **신설** |

---

## 3. 인터페이스 명세

### 3.0 설계 방향 — 학습 config 통합 (2026-08-04 결정)

**평가 설정을 별도 트리(`configs/eval/`)에 두지 않고 학습 `exp.yaml` 안의 `eval:` 블록으로
통합한다. 평가 산출물도 학습 산출물 디렉토리 안에 저장한다.**

```
configs/{experiments,releases}/<P>/<C>/<V>/exp.yaml
  ├─ model_name · dataset_name · weight_path · exp_params · overrides   (학습, 기존)
  └─ eval:  {onnx: {...}, board: {...}}                                 (평가, 신설)
        ↓  derive_run_dir 로 파생
runs/<task>/<P>/<C>/<V>/
  ├─ weights/ · mycfg.yaml · training_log.log     (학습, 기존)
  └─ eval/<variant>/                              (평가, 신설)
        ├─ evaluation.json · prediction.json
        ├─ fp_list.txt · fn_list.txt
        └─ confusion_matrix*.png · mc_curve.png
```

**이 방향의 이점**

| 이점 | 내용 |
|---|---|
| 한 모델 = 한 config = 한 디렉토리 | `derive_run_dir` 이 이미 보장하는 불변식을 평가까지 확장한다 |
| 출처 추적 | 어느 학습이 어떤 평가 수치를 냈는지 디렉토리 구조가 말해준다 |
| 릴리스 동결과 정합 | `exp.yaml` 이 동결되면 평가 설정도 함께 동결된다 — 재현의 근거가 한 파일에 모인다 |
| 트리 추가 없음 | `configs/` 에 다섯 번째 트리를 만들지 않는다 |
| 학습 CLI 와 같은 관례 | `-w <exp.yaml>` 로 통일된다 |

**대가** — `exp.yaml` 이 커진다. 평가 variant 가 여러 개면(`onnx`/`board`/`crop`) 블록이
길어지므로, `eval:` 안에서 공통 부분을 `_shared:` 로 빼는 것을 허용한다(§3.2).

### 3.1 CLI — `tools/customEval.py`

학습 CLI(`tools/train_custom_trainer.py`)와 같은 `-w` 관례를 쓴다.

```
python3 tools/customEval.py -w <exp.yaml> [--eval <variant> ...] [옵션]
```

| 인자 | 타입 | 기본 | 덮어쓰는 설정 키 |
|---|---|---|---|
| `-w` / `--cfg_path` | path | (필수) | — |
| `--eval` | 복수 | `eval:` 의 전체 variant | 실행할 variant 선택 |
| `--data` / `--dataset` | dir | config | `eval.<v>.gt.path`, `pred.path`(rep=onnx) |
| `--pred` | file | config | `eval.<v>.pred.path` |
| `--rep` | `onnx\|ipu_test\|sim\|coco\|cxywhn\|xyxy\|xywh` | config | `eval.<v>.pred.rep` |
| `--onnx-model` | path | config | `eval.<v>.pred.model_path` |
| `--save-dir` | dir | **`derive_run_dir` 결과 + `/eval/<v>`** | 출력 위치 |
| `--mode` | 복수 | config | `eval.<v>.mode` |
| `--conf` | float | **0.01** | `eval.<v>.conf` |
| `--iou-thr` | float | 0.45 | `eval.<v>.iou_thr` |
| `--cut` | float | 0.99 | `eval.<v>.cut` |
| `--cm-conf` | float | 0.25 | `eval.<v>.cm_conf` (§3.2 conf 설명) |
| `--cfg_options KEY=VAL` | 복수 | — | 임의 키 (`DictAction`) |
| `--dry-run` | flag | off | 해석 결과만 출력 |

**우선순위**: CLI 인자 > `--cfg_options` > config 파일 > 기본값.
`projects/custom_utils/cfg.py:174 apply_cfg_options` 의 기존 패턴을 따른다.

**variant 를 여러 개 실행하면** 순차 처리하고 마지막에 비교표를 출력한다 (F-33).

**토큰 치환 불필요** — 학습 config 는 이미 저장소 상대경로(`configs/releases/...`)와
`/dataset/...` 컨테이너 경로를 쓴다. `{datasets}`/`{weights}` 토큰(F-30)은 참고 코드가
외부 저장소를 가리키기 위해 쓴 것이므로 **채택하지 않는다.** `--data` 로 덮어쓰면 충분하다.

### 3.2 설정 파일 스키마 — `exp.yaml` 의 `eval:` 블록

**위치**: `configs/{experiments,releases}/<PRODUCT>/<Camera>/<Version>/exp.yaml`
학습 키(`model_name` `dataset_name` `weight_path` `exp_params` `overrides`)는 그대로 두고
`eval:` 블록만 추가한다.

```yaml
# ── 학습 (기존, 변경 없음) ────────────────────────────────────────
_meta_: {product: BSD_F606, camera: Corner_Truck, version: v01, ...}
model_name: custom
dataset_name: dycrop
weight_path: configs/releases/BSD_F606/Corner_Truck/v01/weight/pretrained_weight.pt
exp_params: {...}
overrides: {imgsz: 320, epochs: 50, data: configs/datasets/cvd_corner.yaml, ...}

# ── 평가 (신설) ───────────────────────────────────────────────────
eval:
  _shared:                                     # 선택. 모든 variant 에 deep merge (variant 승)
    gt:
      path: /dataset/CVD/Corner/Validation     # 필수. dir(images/labels) 또는 coco json
      rep: cxywhn                              # 필수. coco|cxywhn|xyxy|xywh
      sep: ' '                                 # 선택, 기본 ' '
      scale: [1.0, 1.0]                        # 선택, 기본 [1,1]. [y, x]
      roi: [428, 8, 1072, 1072]                # 선택(F-34). [cx, cy, cw, ch] source 픽셀
      roi_mode: center                          # 선택, 기본 center. 다른 값은 NotImplementedError
    classes: [bus, car, motorcycle, person, truck]   # 필수. 순서가 class id
    mode: [ap, f1_score, cm]                   # 선택, 기본 [ap]
    conf: 0.01                                 # 선택, 기본 0.01. 예측 생성 임계
    cm_conf: 0.25                              # 선택, 기본 0.25. ConfusionMatrix default 열
    iou_thr: 0.45                              # 선택, 기본 0.45. NMS IoU
    cut: 0.99                                  # 선택, 기본 0.99. 3단계 임계값 F1 비율
    merge_options: {}                           # 선택. 클래스 병합
    target:                                     # 선택(F-20). 있으면 PASS/Non-Pass 판정
      ap: {bus: 0.5, car: 0.5, motorcycle: 0.5, person: 0.5, truck: 0.5}
      f1: {bus: 0.5, car: 0.5, motorcycle: 0.5, person: 0.5, truck: 0.5}

  onnx:                                        # variant 이름 = 출력 하위 디렉토리 이름
    pred:
      rep: onnx
      path: /dataset/CVD/Corner/Validation     # 추론 이미지 소스
      model_path: configs/releases/BSD_F606/Corner_Truck/v01/network/bsd_f606a_corner_320x320.onnx
      cat_map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # 선택(F-07). 숫자 키
      pre_proc:  {from_: projects.metrics.inference_funcs, import_: yolov8_bgr255_preproc}
      post_proc: {from_: projects.metrics.utils,           import_: basic_postproc_v8}
      inf_func:  {from_: projects.metrics.inference_funcs, import_: crop_inference_v8,
                  args: {crop: [428, 8, 1072, 1072]}}      # F-12

  board:
    pred:
      rep: ipu_test
      path: configs/releases/BSD_F606/Corner_Truck/v01/analysis/iputest/output.json
      sep: ','
      scale: [1.0, 1.0]
      cat_map: {BUS: 0, CAR: 1, MOTOR: 2, PERSON: 3, TRUCK: 4}   # 문자열 키 필수
```

**`save_dir` 은 쓰지 않는다.** `derive_run_dir` 이 계산한 학습 출력 경로에 `eval/<variant>/`
를 붙인다 — `runs/detect/BSD_F606/Corner_Truck/v01/eval/{onnx,board}/`. 학습 config 에
`project`/`name`/`save_dir` 을 넣지 않는 기존 규칙(CLAUDE.md)과 같은 이유다.

**`_shared` 병합** — `deep_merge(deepcopy(eval._shared), eval[variant])` 로 해석한다
(variant 승). `cfg.py:13 deep_merge` 와 `load_pipeline_yaml` 의 기존 패턴을 그대로 쓴다.

**`eval:` 이 없는 config** — 평가를 실행하면 `ValueError`. 학습에는 영향 없다(무시).

**`gt.roi` 판정 기준 — `center` 로 확정 (2026-08-04)**

GT 박스의 **중심점**이 ROI 사각형 안에 있으면 남기고, 밖이면 평가에서 제외한다.
경계는 포함(`cx <= mx <= cx+cw`)이다.

```python
mx, my = bx + bw / 2, by + bh / 2
keep = (cx <= mx <= cx + cw) and (cy <= my <= cy + ch)
```

**근거** — 이 기준으로 필터했을 때 GT가 **정확히 21,742개**가 되어 기록된 nGT와 오차 0으로
일치했고, per-class AP도 재현됐다(§7.2, §7.3). 다른 기준은 다른 수치를 낸다.

| 기준 | 채택 | 이유 |
|---|---|---|
| **center** — 중심점 포함 | ✅ | 기준값을 오차 0으로 재현. 경계 객체를 한 번만 세어 중복이 없다 |
| IoA 임계 (`area∩ROI / area > t`) | ✗ | `t` 를 정해야 하고 값에 따라 수치가 흔들린다. 재현 근거가 없다 |
| 완전 포함 | ✗ | 경계에 걸친 객체를 전부 버려 GT가 21,742보다 적어진다 |
| 클리핑 (박스를 ROI로 잘라 유지) | ✗ | 박스 크기가 바뀌어 IoU 매칭 결과가 달라진다 |

`roi_mode` 는 스키마에 두지만 **`center` 외의 값은 `NotImplementedError`** 로 거부한다.
다른 기준이 필요해지면 그때 D-번호로 결정하고 추가한다 — 조용히 기본값이 바뀌어 수치가
달라지는 것을 막기 위함이다.

**`conf` — 0.01 기본, config 최우선 (2026-08-04 조사·확정)**

```
우선순위:  CLI --conf  >  config 의 eval.<v>.conf  >  기본값 0.01
```

**조사 결과 — `conf` 는 지금 두 가지 다른 일에 쓰이고 있다**

| 쓰임 | 현재 값의 출처 | 영향 |
|---|---|---|
| ① 예측 생성 임계 (ONNX 추론) | **`utils.py:247` 에 `confidence_thres=0.01` 하드코딩** | AP·P·R·F1 전부 |
| ② ConfusionMatrix 의 `default` 열 | `cfg.get('conf', 0.25)` (`evaluator.py:102`) | confusion matrix 만 |

②의 실체가 명확하다 — `confusion_matrix.py:155` 가 3단계 임계값
(`sensitive`/`optimal`/`insensitive`) 뒤에 `[self.conf] * nc` 를 **네 번째**로 붙이고,
이것이 `eval_conf_mat` 의 `characters=[..., 'default']` 와 대응한다. 즉 "기본 운영 임계로
봤을 때의 혼동행렬"이다.

**따라서 `cfg['conf']` 는 AP에 전혀 영향을 주지 않는다.** 기본값 0.25를 0.01로 바꾸는 것의
영향 범위는 **confusion matrix 의 `default` 열 하나**다.

**진짜 문제는 ①의 하드코딩이다.** 예측 생성 임계를 config 로 바꿀 수 없어 "설정 파일 최우선"
원칙이 실제로 깨져 있다. §0.1 기준값이 재현된 것도 이 하드코딩 0.01 덕분이며, 의도된
설정이 아니라 우연이다.

**결정 — 두 값을 분리한다**

| 키 | 기본 | 역할 |
|---|---|---|
| `conf` | **0.01** | 예측 생성 임계. `utils.py:247` 하드코딩을 제거하고 이 값을 전달 |
| `cm_conf` | **0.25** | ConfusionMatrix `default` 열. 기존 `conf` 의 역할을 이름만 바꿔 보존 |

하나로 합치면 안 된다 — `conf=0.01` 을 혼동행렬에 넘기면 거의 모든 예측이 통과해
`default` 열이 무의미해지고, `conf=0.25` 로 추론하면 AP가 떨어져 §0.1 을 재현하지 못한다.

**왜 예측 생성 임계가 0.01 이어야 하나** — mAP와 F1 곡선은 낮은 confidence까지 예측이
있어야 정확하다. 0.25로 자르면 recall 상단이 잘리고, 3단계 임계값 탐색도 0.25 미만 구간을
볼 수 없어 `sensitive` 가 0.25에 고정된다. 참고 코드 `calc_class_thresholds.py` 도 같은
이유로 `--conf` 기본을 0.001로 두고 docstring 에 "높은 conf 로 필터된 파일이면 그 floor
미만 threshold 는 탐색 불가"라고 적고 있다.

**회귀 위험은 없다** — 활성 평가 config 가 **0개**다. `gt:`/`pred:` 키를 가진 yaml 19개는
전부 `configs/archive/coco_yaml/`(CLAUDE.md 참조 금지 트리)에 있고 모두 `conf` 미명시다.
게다가 `tools/eval.py` 의 기본 config 경로 `configs/experiments/legacy/eval_test.yaml` 은
**존재하지 않아** 인자 없이 실행하면 즉시 실패한다.

**하위 호환** — 기존 config 는 `cat_map`·`target`·`roi`·`cut`·`cm_conf` 가 없어도 동작해야
한다. `gt.roi` 가 없으면 필터를 적용하지 않는다(전체 GT 평가).
`mode` 기본값에 `ap` 를 추가하되 기존 `[precision, recall, f1_score]` 도 유효.

**`mode` 유효값** — 현재 `f1_score`·`precision`·`recall`·`cm` 외에는
`evaluator.py:250` 이 조용히 `continue` 한다. `ap` 를 신설하고, **미지원 값은
`ValueError` 로 거부**한다(오타가 조용히 무시되는 것을 막는다).

### 3.3 Python API

```python
from projects.metrics.evaluator import Evaluator

ev = Evaluator(cfg=exp_cfg["eval"]["onnx"])   # dict. exp.yaml 의 eval 블록 한 variant
results = ev.eval_()                                # → dict (3.4 스키마)
```

**신설 함수** — `projects/metrics/inference_funcs.py`

| 함수 | 시그니처 계약 |
|---|---|
| `crop_inference_v8(path, model_path, classes, im_filename2id, pre_proc, post_proc, **kw)` | `kw['crop']=[cx,cy,cw,ch]`. 반환: COCO detection dict 리스트 (`image_id` `category_id` `bbox`(xywh, source 좌표) `score`) |
| `letterbox_inference_v8(...)` | 동일 반환. 패딩·비율 역매핑 후 source 좌표 |
| `yolov8_bgr255_preproc(img, imgsz, **kw)` | `(1,3,H,W) float32`, BGR `[0,255]` |
| `yolov8_std_preproc(img, imgsz, **kw)` | `(1,3,H,W) float32`, RGB `/255` |
| `yolov8_std_postproc(outputs, imgsz, orgsz, **kw)` | `[[xywh(int32), score(float), label(int)], ...]` |

**`inf_func` 계약** — `post_proc` 는 `[[xywh, score, raw_label], ...]` 를 돌려주고,
`inf_func` 가 `cat_map` 을 적용해 COCO dict로 만든다. 이 분담을 지켜야 `from_`/`import_`
주입이 서로 호환된다.

**신설 유틸** — `projects/metrics/utils.py`

| 함수 | 역할 |
|---|---|
| `merge_board_json(in_path) -> list` | 이어붙인 JSON 배열을 flat 리스트로 (멱등, F-04) |
| `apply_cat_map(cat, cat_map) -> int \| None` | 매핑, 미매핑은 `None` (F-07) |
| `filter_gt_roi(anno, roi) -> COCO` | GT 중심점이 ROI 내부인 것만 남긴다 (F-34) |

### 3.4 3단계 confidence 임계값 (F-18)

**이미 구현돼 있다** — `Evaluator.f1_score` (`evaluator.py:286`). 참고 코드
`~/Downloads/outputs/calc_class_thresholds.py` 의 `curve_thresholds()` 와 **같은
알고리즘**이고, 그 파일 주석도 "evaluator.f1_score 와 동일 방식"이라고 밝히고 있다.
이번 작업에서는 **알고리즘을 바꾸지 않고 출력에 연결**한다.

**정의**

| 이름 | 정의 | 용도 |
|---|---|---|
| `optimal` | F1 곡선이 최대인 confidence (`argmax`) | 기본 운영 임계 |
| `sensitive` | `F1 >= max(F1) * cut` 구간의 **최저** confidence | recall 우선 (놓치지 않기) |
| `insensitive` | 같은 구간의 **최고** confidence | precision 우선 (오검출 줄이기) |

**파라미터**

| 항목 | 값 | 위치 |
|---|---|---|
| conf grid 해상도 | 1001 (`np.linspace(0, 1, 1001)`) | `evaluator.py:294` |
| `cut` | **0.99** | `evaluator.py:308` `cut_val`. config `cut` 로 노출 |
| 보간 | `np.interp(-x, -score, recall, left=0)` / `precision, left=1` | `evaluator.py:303-304` |
| F1 | `2*p*r / (p+r)` | `evaluator.py:306` |

**구현 시 확인해야 할 차이 한 가지** — 구간 판정 부등호가 다르다.

```python
# 우리 (evaluator.py:311)
threshold_range = np.where(f1_score > f1_criteria)[0]     # 초과
# 참고 코드 (calc_class_thresholds.py:349)
rng = np.where(f1 >= crit)[0]                             # 이상
```

경계에서 grid 한 칸(0.001)만큼 `sensitive`/`insensitive` 가 달라질 수 있다.
**§0.1 기준값 재현으로 어느 쪽이 기록값을 만들었는지 판정한다** — 회수한
`thresholds_detail.json` 이 정답표다(아래).

**검증 자산** — `configs/releases/BSD_F606/Corner_Truck/v01/analysis/thresholds_detail.json`
이 float(onnx) 실행의 3단계 임계값 정답이고, `metrics_corner_onnx.txt` 의
`sens`/`opt`/`insens` 열과 정확히 일치한다.

| class | sensitive | optimal | insensitive | maxF1 | P | R | nGT |
|---|---|---|---|---|---|---|---|
| bus | 0.031 | 0.056 | 0.118 | 0.429 | 0.739 | 0.302 | 1,568 |
| car | 0.207 | 0.253 | 0.337 | 0.481 | 0.588 | 0.406 | 8,579 |
| motorcycle | 0.103 | 0.174 | 0.229 | 0.618 | 0.776 | 0.513 | 2,264 |
| person | 0.128 | 0.149 | 0.226 | 0.621 | 0.742 | 0.534 | 7,485 |

`TP`/`FP`/`FN` 카운트도 함께 들어 있다 (bus 474/167/1094 등). `evaluation.json` 의
`metrics.thresholds` 는 이 구조를 그대로 따른다(§3.5).

### 3.5 임계값 산출 독립 실행 (F-35) — 신설 검토

참고 코드 `calc_class_thresholds.py` (554줄)는 `Evaluator` 와 별개로 임계값만 뽑는 도구다.
**흡수 여부는 선택**이며, 흡수하면 얻는 것이 셋이다.

| 기능 | 값 |
|---|---|
| `--pred-file` 모드 | **모델 없이** 예측 json만으로 임계값 산출. board 덤프에도 쓸 수 있다 |
| `.pt` / `.onnx` 양쪽 입력 | `.onnx` 는 출력 1개(decoded) / 2개(raw cls+dfl) 자동 판별, `_make_anchors` 로 `dist2bbox` 직접 수행 |
| F1-Confidence 곡선 png | 클래스별 곡선 + 3점 마커(`<` sensitive / `o` optimal / `>` insensitive) |

**주의할 설계 차이** — 이 도구는 `Evaluator` 를 쓰지 않고 IoU 매칭(`match()`)·
누적 PR·F1 곡선을 **자체 구현**한다. 즉 §1.1의 정확도 계산 경로가 **여섯 번째로 늘어난다.**
흡수한다면 매칭 로직을 pycocotools 경로로 바꿔 경로 수를 늘리지 않아야 한다.

**권장** — 이번 범위에서는 **정의와 파라미터만 참고**(§3.4)하고 도구 자체는 흡수하지 않는다.
`--pred-file` 모드는 `tools/customEval.py --mode f1_score` 로 대체 가능하고,
곡선 png는 기존 `plot_mc_curve` (`evaluator.py:598`)를 쓰면 된다. 흡수는 별도 작업으로 뗀다.

### 3.6 출력 산출물

평가 산출물은 **학습 산출물 디렉토리 안에** 저장한다 (§3.0).
**`mode` 와 무관하게 `evaluation.json` 은 항상 쓴다** (F-27).

```
runs/<task>/<PRODUCT>/<Camera>/<Version>/        ← derive_run_dir 결과 (학습과 공유)
├── weights/ · mycfg.yaml · training_log.log       학습 산출물 (기존)
└── eval/
    ├── onnx/                                      variant 이름
    │   ├── evaluation.json      ← 신설 (F-22). 항상
    │   ├── prediction.json      COCO 예측. --save-pred 시에만
    │   ├── fp_list.txt · fn_list.txt              mode 에 cm 포함 시
    │   ├── confusion_matrix*.png                  mode 에 cm 포함 시
    │   ├── mc_curve.png                           mode 에 f1_score 포함 시
    │   └── eval.log
    └── board/
        └── (동일 구조)
```

`--save-dir` 로 덮어쓸 수 있으나 기본값은 위 경로다. 릴리스 트리의 산출물 규약
(`{analysis,network,weight}/`)과 달리 `runs/` 는 전체가 `.gitignore` 대상이므로 용량 제약이 없다.

**`evaluation.json` 스키마**

```json
{
  "config":  {"path": "...", "resolved": { }},
  "meta":    {"timestamp": "...", "gt_rep": "cxywhn", "pred_rep": "onnx",
              "model_path": "...", "num_images": 1234, "num_gt": 5678,
              "num_pred": 4321, "num_dropped_by_cat_map": 12},
  "metrics": {
    "ap":        {"all": 0.3976, "bus": 0.378, "car": 0.299, "...": 0.0},
    "precision": {"all": {"IoU@50-95": 0.0, "IoU@50": 0.0, "IoU@75": 0.0}},
    "recall":    {"all": {"IoU@50-95": 0.0}},
    "f1_50":     {"bus": 0.429, "car": 0.481},
    "thresholds": {
      "bus": {"sensitive": 0.031, "optimal": 0.056, "insensitive": 0.118,
              "maxF1": 0.429, "P": 0.739, "R": 0.302, "ngt": 1568,
              "TP": 474, "FP": 167, "FN": 1094}
    }
  },
  "params": {"conf": 0.01, "iou_thr": 0.45, "cut": 0.99,
             "roi": [428, 8, 1072, 1072], "roi_mode": "center"},
  "target":  {"ap": {"bus": {"value": 0.378, "target": 0.5, "pass": false}}},
  "verdict": "NON-PASS"
}
```

`num_dropped_by_cat_map` 을 반드시 남긴다 — 문자열 키 config에 숫자 `cat_map` 을 쓰면
전량 drop되어 mAP 0이 되는 함정이 있고, 이 숫자가 그것을 즉시 드러낸다.

---

## 4. 모듈 배치

### 4.1 변경 후

```
projects/metrics/
├── evaluator.py          Evaluator — config 해석·GT/예측 로드·지표·저장
├── custom_coco_eval.py   customCOCOeval (accumulate override)
├── confusion_matrix.py   ConfusionMatrix
├── inference_funcs.py    ← 신설. ONNX 추론 변형 (F-10~15)
├── report.py             ← 신설. 콘솔 표 + PASS 판정 + evaluation.json (F-20,22,24)
└── utils.py              전처리/후처리/좌표 — 좌표 4함수는 custom_utils 재수출

tools/customEval.py       ← 평가 단일 엔트리 (3.1)
configs/{experiments,releases}/<P>/<C>/<V>/exp.yaml   ← eval: 블록 (3.2)
```

### 4.2 정리 대상

| 대상 | 처분 | 근거 |
|---|---|---|
| `tools/eval_test.py` (137) | `Evaluator` 위임 또는 폐기 | raw `COCOeval` 우회. 문서 참조 0건 |
| `val/dair.py:76` inline COCO | `metrics/` 호출로 교체 | 3번째 경로 |
| `tools/eval_det_txt.py` (344) | **유지** | txt 직접 비교 고유 용도. upstream `ap_per_class` 사용 사실을 파일 상단에 명시 |
| `projects/src/utils/metric.py` | **유지** | 학습 중 validator 전용. pycocotools로 옮기면 학습이 느려진다 |
| `metrics/utils.py` 좌표 4함수 | `custom_utils/drawing.py` 재수출 | 바이트 동일 중복 |

**최종 경로 2개** — 오프라인/보드 평가는 `projects/metrics/` (pycocotools),
학습 중 검증은 `projects/src/utils/metric.py` (자체 `ap_per_class`).

### 4.3 변경 전 / 변경 후 구조 대조

변경되는 파일만 표시한다. `→` 왼쪽이 현재, 오른쪽이 목표.

```
projects/metrics/
  evaluator.py         636줄  → 수정   cat_map·gt.roi·save_result 위치·ap mode
  utils.py             276줄  → 수정   좌표 4함수 재수출, merge_board_json·apply_cat_map 추가
  custom_coco_eval.py  144줄  → 무변경
  confusion_matrix.py  275줄  → 무변경
  (없음)                      → 신설   inference_funcs.py   추론 변형 5종 (F-10~15)
  (없음)                      → 신설   report.py            AP표·PASS판정·evaluation.json

projects/custom_utils/
  drawing.py            63줄  → 무변경 (좌표 4함수의 정본이 된다)

projects/src/
  utils/metric.py      194줄  → 무변경 (학습 중 validator 전용)
  models/yolo/detect/val/dair.py → 수정   inline COCO import 제거 (:76)

tools/
  customEval.py         18줄  → 수정   CLI 인자 14개로 확장 (§3.1)
  eval_test.py         137줄  → 폐기 또는 Evaluator 위임
  eval_det_txt.py      344줄  → 무변경 (헤더 주석만 추가)
  validate_custom.py    49줄  → 무변경
  run_sim_batch.py     185줄  → 무변경 (F-34 도입 후 사용법만 문서화)

configs/releases/BSD_F606/Corner_Truck/v01/
  exp.yaml              → 수정   eval: 블록 추가 (variant onnx / board)
  (다른 릴리스도 동일 방식으로 필요 시 추가)
```

**순증** — 파일 2개(`inference_funcs.py` · `report.py`). 새 config 트리 없음.
**순감** — 파일 0~1개(`eval_test.py` 폐기 시).

### 4.4 실행 방법 — 변경 전 / 변경 후

#### board(NPU) 평가

**변경 전 — 실행할 수 없다.** 세 곳에서 막힌다.

```bash
python3 tools/customEval.py <config>     # positional 인자 하나만 받는다
```

| 막히는 지점 | 증상 |
|---|---|
| `--pred` 미지원 | 보드 JSON 경로를 넘길 방법이 없다 (config `pred.path` 에 하드코딩해야 함) |
| `cat_map` 없음 | board category `MOTOR` → `classes.index('motor')` → **`ValueError`** |
| `gt.roi` 없음 | GT 전체(30,567)로 평가 → mAP **0.356** (기록 0.481, −0.125) |

**변경 후**

```bash
python3 tools/customEval.py \
    -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml --eval board
# → mAP@0.50 0.481
#   runs/detect/BSD_F606/Corner_Truck/v01/eval/board/evaluation.json

# 데이터셋·예측만 바꿔 재사용
python3 tools/customEval.py -w <exp.yaml> --eval board \
    --data /dataset/CVD/Corner/Validation \
    --pred configs/releases/BSD_F606/Corner_Truck/v01/analysis/iputest/output.json
```

#### float(ONNX) 평가

**변경 전 — 화각을 통제할 수 없다.**

```bash
# 이 경로는 plain resize 전체 화각 고정. crop 지정 수단이 없다.
python3 tools/customEval.py <config>          # inf_func = basic_inference_v8
# 또는
python3 tools/infer_onnx_batch.py --onnx <m> --img-dir <d> --conf 0.25 --iou 0.45
python3 tools/eval_det_txt.py --gt <gt> --pred <out> --out report.md   # 다른 mAP 코어
```

**변경 후**

```bash
python3 tools/customEval.py \
    -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml --eval onnx
# → mAP@0.50 0.460
#   runs/detect/BSD_F606/Corner_Truck/v01/eval/onnx/evaluation.json
#   eval.onnx.pred.inf_func = crop_inference_v8 (crop: [428,8,1072,1072])
#   eval._shared.gt.roi     = [428,8,1072,1072]
```

#### 두 조건 비교

**변경 전** — 없음. 사람이 두 번 돌리고 손으로 표를 만든다.

**변경 후**

```bash
python3 tools/customEval.py \
    -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml --eval onnx board
# --eval 생략 시 eval: 의 전체 variant 를 실행한다
```

```
============================================================
 mAP@0.50 비교
============================================================
  onnx   (float, ROI crop)            : 0.4600
  board  (board ipu_test)             : 0.4810
------------------------------------------------------------
  Δ(board − float, 동일 ROI) = +0.0210  → 양자화 손실 지표
============================================================
```

#### 인자 해석만 확인

```bash
python3 tools/customEval.py -w <exp.yaml> --eval board --dry-run
# _shared 병합 결과·최종 config·산출물 경로를 출력하고 종료
```

#### 학습 중 검증 — 변경 없음

```bash
python3 tools/validate_custom.py -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml
```

이 경로는 `projects/src/utils/metric.py` 를 쓰고 이번 작업에서 손대지 않는다.
**pycocotools 경로와 수치가 다를 수 있다** — 목적이 다르므로 의도된 차이이고,
§4.2에 그 사실을 문서화한다.

---

## 5. 구현 단계

각 단계는 독립 검증 가능하고 끝난 시점에 저장소가 동작 상태여야 한다.

### ☑ 1. 좌표 변환 중복 제거 — small

- **What**: 4함수를 한 곳에만.
- **Where**: `projects/metrics/utils.py`, `projects/custom_utils/drawing.py`
- **How**: `custom_utils/drawing.py` 를 정본으로(upstream 의존 없는 순수 유틸).
  `metrics/utils.py` 는 import로 재수출해 기존 경로를 유지.
- **Test**: `assert metrics.utils.to_xywh is drawing.to_xywh`. 기존 평가 1회 실행해 수치 무변화.

### ☑ 2. `cat_map` + 보드 JSON 정규화 — medium

- **What**: F-04, F-07.
- **Where**: `projects/metrics/utils.py` (`merge_board_json`, `apply_cat_map`),
  `evaluator.py` (`parse_ipu_result`, `pred2cocoJson`), `basic_inference_v8`
- **How**: `cat_map` 있으면 우선 적용·미매핑 drop, 없으면 기존 `get_class_id` 유지.
  드롭 개수를 집계해 결과에 넣는다. `parse_ipu_result` 는 `json.load` 대신
  `merge_board_json` 을 쓴다(`JSONDecoder.raw_decode` 루프, 멱등).
- **Test**: 픽스처 4종 — 이어붙인 JSON / 단일 배열 / 문자열 cat_map / 숫자 cat_map.
  드롭 개수와 매핑 결과 검증. `cat_map` 없는 기존 config 회귀.

### ☑ 3. `inference_funcs.py` 신설 — medium

- **What**: F-10~15.
- **Where**: `projects/metrics/inference_funcs.py` (신설)
- **How**: 참고 코드의 함수 5개를 이식하고 두 가지 보강 —
  provider `['CUDAExecutionProvider','CPUExecutionProvider']`, `if img is None: continue`.
  **`crop_inference_v8` 이 이 단계의 핵심 가치**(§7 인수 기준 참조).
- **Test**: 합성 이미지로 crop 오프셋이 source 좌표로 복원되는지, letterbox 패딩 역매핑이
  정확한지 단위 테스트.

### ☑ 4. `report.py` 신설 — medium

- **What**: F-20, F-22, F-24, F-27.
- **Where**: `projects/metrics/report.py` (신설), `evaluator.py` (`save_result` 교체)
- **How**: `save_result` 의 `# TODO : Not implemented` 스텁을 실구현으로 바꾸고
  **`cm` 블록 밖으로 꺼낸다**(`evaluator.py:265` → `eval_()` 말미). AP per-class 표에
  `target` 대비 PASS/Non-Pass 열을 붙인다. `evaluation.json` 을 3.4 스키마로 쓴다.
- **Test**: `mode: [ap]` 만으로 실행해 `evaluation.json` 이 생기는지. `target` 미지정 시
  `verdict` 가 `null` 인지. 스키마 키 존재 검증.

### ☑ 5. CLI 단일화 + 설정 규약 — medium

- **What**: F-29~33, 3.1·3.2.
- **Where**: `tools/customEval.py`, `configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml`
  (`eval:` 블록 추가), `evaluator.py`(오버라이드 수용)
- **How**: 3.1 인자를 구현하고 토큰 치환·config 해소·`--dry-run`·복수 config 비교표를
  넣는다. 우선순위는 `apply_cfg_options` 패턴. 참고 코드의 config 3개를
  참고 코드의 config 3개를 `exp.yaml` 의 `eval:` variant 로 이식한다(토큰은 채택하지 않음).
- **Test**: `--dry-run` 인자 해석 확인. `ncn_metric/eval.py --ncn-root <이 저장소> --dry-run`
  이 뱉는 명령이 실제로 실행되는지.

### ☑ 6. 경로 통합 (5 → 2) — large

- **What**: §4.2 정리.
- **Where**: `tools/eval_test.py`, `val/dair.py`, `tools/eval_det_txt.py`
- **How**: `eval_test.py` 를 `Evaluator` 위임으로 교체(불가하면 폐기 — 문서 참조 0건).
  `val/dair.py:76` inline COCO 제거. `eval_det_txt.py` 는 유지하되 다른 코어를 쓴다는
  사실을 명시.
- **Test**: **같은 데이터셋으로 통합 전/후 per-class AP 표를 비교**해 허용 오차 내 일치.
  이 단계가 수치 회귀 위험이 가장 크다.

### ☑ 7. 문서화 + 결정 기록 — small

- **What**: 통합 결과와 화각 통제 사실 기록.
- **Where**: `docs/decisions.md` (D-020), `README.md`, `CLAUDE.md`,
  `docs/operations_guide.md`, `docs/research.md`
- **How**: D-020에 정본 결정과 근거. **화각 통제 사실을 함께 남긴다**(§7).
  미문서화된 sim 평가 3종(`prepare_sim_eval_imgs` → `run_sim_batch` → `eval_det_txt`)
  워크플로도 적는다.
- **Test**: 문서의 명령을 그대로 복사해 실행.

---

## 6. 위험과 대응

| 위험 | 단계 | 대응 |
|---|---|---|
| **수치 회귀** | 6 | 통합 전 per-class AP를 baseline으로 고정. §7 인수 기준을 게이트로 |
| 회귀 데이터 부재 — 데이터셋·가중치·보드 JSON이 다른 호스트(`~/Developments/data/`) | 2·3·6 | **조율 필요.** 최소 픽스처(이미지 10장 + 보드 JSON 1개 + onnx 1개)를 확보해 단위 테스트에 넣는다 |
| 검출 경로 테스트 0개 | 전부 | 각 단계에 단위 테스트를 함께. 지금 평가 경로 5개 중 테스트가 있는 것은 **없다** |
| `cat_map` 하위 호환 파괴 | 2 | `cat_map` 없으면 기존 `get_class_id` 경로 유지 |
| `mode` 미지원 값 거부가 기존 config를 깨뜨림 | 4 | 먼저 경고만 내는 단계를 거치고, 기존 config 전수 확인 후 `ValueError` 로 승격 |
| 순환 import (`metrics` ↔ `custom_utils`) | 1 | 의존은 `metrics` → `custom_utils` 단방향만 |
| ONNX provider 변경으로 GPU 미사용 | 3 | CUDA 우선 순서 유지 + 실제 사용 provider를 로그에 남긴다 |
| `save_result` 를 `cm` 밖으로 꺼내며 기존 동작 변화 | 4 | `fp_list`/`fn_list` 는 여전히 `cm` 일 때만. `evaluation.json` 만 항상 |

**조율이 필요한 항목** — 회귀 픽스처 확보(다른 호스트), `ncn_metric` 을
`adassw/pipeline` 등 다른 사용자가 함께 쓰는지 확인(정본을 이 저장소로 정했으므로
그쪽 드라이버가 우리 CLI를 가리키도록 바꿔야 할 수 있다).

---

## 7. 인수 기준 (실측 검증 완료)

### 7.1 기준선 자산 — 회수 완료

참고 코드 README의 수치(`0.498 / 0.398 / 0.378`)는 **640×512 Front v1 모델**의 것이라
이 작업 범위에서 **배제한다**. 대신 이 저장소가 직접 측정해 NAS_2에 남긴 산출물을
기준선으로 쓴다. 2026-08-04에 로컬로 회수했다 (`.gitignore:42` 로 미추적).

| 파일 | 크기 | 위치 |
|---|---|---|
| `bsd_f606a_corner_320x320.onnx` | 12 MB | `configs/releases/BSD_F606/Corner_Truck/v01/network/` |
| `gen_config.txt` | 5.1 KB | 같은 위치 |
| `output.json` (board 덤프) | 14 MB | `.../v01/analysis/iputest/` |
| `metrics_corner_board.txt` · `metrics_corner_onnx.txt` | 각 1.2 KB | `.../v01/analysis/` |
| `thresholds_detail.json` | 1.1 KB | 같은 위치 |

데이터셋은 로컬에 있다 — `~/Workspace/data/CVD/Corner/Validation`, 이미지 5,000 / 라벨 5,000.

**입출력 실측**

| 항목 | 값 |
|---|---|
| 이미지 해상도 | 1920 × 1072 (W × H) |
| ONNX 입력 | `images [1,3,320,320]` |
| ONNX 출력 | `output0 [1,5,2100]` (cls) + `433 [1,4,2100]` (ltrb dist) — `basic_postproc_v8` 규약과 일치 |
| board detection | 91,018개 / 4,955장 (단일 flat 배열) |
| board category | `BUS` `CAR` `MOTOR` `PERSON` `TRUCK` (5종, `T.L.`/`NO-OBJECT` 없음) |
| board bbox | source 좌표 xyxy. x ≤ 1500, y ≤ 1080 |
| board score | 0.0 ~ 0.961 (이미 0~1, `score_` 재스케일 불필요) |
| GT 클래스 | 0~4만 존재 (class 5 traffic_light 없음). 총 30,567 박스 |

### 7.2 ROI 가설 — 오차 0으로 확정

기록 파일 헤더가 `ROI=[428.0, 8.0, 1072.0, 1072.0]` 이고, 기록된 nGT 합이 21,742인데
전체 GT는 30,567이다. **GT 중심이 ROI 내부인 것만 남기면 정확히 21,742개**가 되어
가설이 오차 0으로 확정됐다. board bbox 범위(x ≤ 1500 = 428+1072, y ≤ 1080 = 8+1072)도
같은 ROI를 가리킨다.

### 7.3 board 재현 — 완전 일치 ✅

우리 `Evaluator.get_coco_gt` + `parse_ipu_result` 로 입력을 만들고 pycocotools로
per-class AP@0.50을 계산한 결과.

| class | 기록 AP | 전체 GT | **ROI GT** | Δ(ROI−기록) | 기록 nGT |
|---|---|---|---|---|---|
| bus | 0.416 | 0.299 | **0.416** | −0.000 | 1,568 |
| car | 0.414 | 0.269 | **0.414** | +0.000 | 8,579 |
| motorcycle | 0.549 | 0.413 | **0.547** | −0.002 | 2,264 |
| person | 0.523 | 0.407 | **0.524** | +0.001 | 7,485 |
| truck | 0.504 | 0.391 | **0.504** | +0.000 | 1,846 |
| **mAP@0.50** | **0.481** | 0.356 | **0.481** | **+0.000** | 21,742 |

**ROI를 적용하지 않으면 0.356으로 0.125를 과소평가한다.** 이것이 F-12(화각 통제)가
필요한 정량적 근거다. 현재 `tools/run_sim_batch.py` 로 NPU 결과를 비교하면 이 0.125를
양자화 손실로 오인한다.

### 7.4 float(onnx) 재현 — 완전 일치 ✅

ROI crop 추론(F-12) + GT ROI 필터(F-34) 조합으로 5,000장 전량 추론.
전처리는 BGR `[0,255]` plain resize, 디코드는 우리 `basic_postproc_v8`,
`conf=0.01` · `iou=0.45`.

| class | 기록 AP | 재현 AP | Δ |
|---|---|---|---|
| bus | 0.284 | **0.284** | +0.000 |
| car | 0.393 | **0.393** | +0.000 |
| motorcycle | 0.549 | **0.548** | −0.001 |
| person | 0.571 | **0.571** | −0.000 |
| truck | 0.504 | **0.504** | −0.000 |
| **mAP@0.50** | **0.460** | **0.460** | **+0.000** |

### 7.5 회귀 게이트

| 대상 | 기준값 (mAP@0.50) | 조건 |
|---|---|---|
| board (`ipu_test`) | **0.481** | GT ROI 필터(F-34) |
| float (`onnx`) | **0.460** | ROI crop 추론(F-12) + GT ROI 필터(F-34) |
| Δ (board − float) | **+0.021** | 동일 ROI |
| float, ROI 미적용 | 0.356 (board 기준) | **잘못된 비교의 예** |

허용 오차 — per-class **±0.003**, mAP **±0.002**. 관측된 최대 편차는 board
motorcycle 0.002, onnx motorcycle 0.001이다.

Δ가 **양수**라는 사실에 주의한다 — 같은 ROI에서 board가 float보다 **높다**. INT8
양자화가 정확도를 떨어뜨리지 않았다는 뜻이고, 참고 README의 crop-FOV Δ(+0.02)와
방향·크기가 일치한다. **화각을 통제하면 양자화 손실이 없다**는 결론이 서로 독립적인
두 측정에서 확인된다.

ROI를 적용하지 않으면 board가 0.481 → 0.356으로 떨어져 **0.125를 과소평가**한다.
현재 `tools/run_sim_batch.py` 로 NPU 결과를 비교하면 이 0.125를 양자화 손실로 오인한다.

### 7.6 단계별 게이트

| 단계 | 통과 조건 |
|---|---|
| 1 | 기존 평가 수치 무변화 |
| 2 | `cat_map` 4종 픽스처 통과 + 기존 config 회귀. **`MOTOR` 매핑 필수**(§8-1) |
| 3 | crop/letterbox 좌표 복원 단위 테스트 + **§7.5 float 0.460 재현** |
| 4 | `mode: [ap]` 만으로 `evaluation.json` 생성 + AP 표 출력 |
| 5 | `--dry-run` 해석 정확 + **§7.5 board 0.481 재현** |
| 6 | per-class AP 통합 전/후 일치 |
| 7 | 문서 명령 복사 실행 성공 |

전 단계 공통 — `pytest` 기준선 유지(착수 시 `56 passed, 2 skipped`), ruff 14건 이하 유지
(이 숫자가 늘면 회귀).

---

## 8. 해소된 항목과 미결

### 8.1 해소 (2026-08-04 검증)

| 항목 | 결과 |
|---|---|
| 회귀 기준선 확보 | ✅ NAS_2에서 6개 파일 회수. 데이터셋은 로컬에 있음 (§7.1) |
| 기준선 재현 가능성 | ✅ board 0.481 · float 0.460 **양쪽 모두 Δ +0.000** (§7.3, §7.4) |
| ROI 방법론 | ✅ GT 중심 기준 ROI 필터 → nGT 21,742 **정확히 일치** (§7.2) |
| 640×512 모델 | ✅ **Front v1 것으로 확인, 범위에서 배제.** README 수치는 정성적 결론의 참조로만 인용 |
| `cat_map` 필요성 | ✅ **실증됨.** board category가 `MOTOR` 인데 `classes` 에는 `motorcycle` 이라 현행 `get_class_id` 의 `classes.index('motor')` 가 `ValueError` 를 낸다. 검증에서는 `classes` 를 `[bus,car,motor,person,truck]` 로 바꿔 우회했다 — 이 우회는 클래스 이름을 왜곡하므로 정식 해법이 아니다 |
| 이어붙인 JSON 필요성 | ⚠️ **회수한 board JSON은 단일 배열이다** (`][` 0건, `json.load` 성공). F-04는 이 자산으로 검증되지 않는다 — 합성 픽스처가 필요하다 |
| `gt.roi` 판정 기준 | ✅ **`center` 로 확정** (§3.2). `roi_mode` 에 다른 값이 오면 `NotImplementedError` |
| `conf` 기본값 | ✅ **0.01 로 확정, config 값이 최우선** (§3.2). 기존 `Evaluator` 기본값 0.25는 폐기 |
| 3단계 임계값 정의 | ✅ **`Evaluator.f1_score` 에 이미 구현돼 있음.** 참고 코드 `calc_class_thresholds.py` 와 동일 알고리즘(`resolution=1001`, `cut=0.99`, 동일 보간). 정답표는 회수한 `thresholds_detail.json` (§3.4) |
| `sens`/`opt`/`insens` 열의 의미 | ✅ **해독 완료** — recall 우선 / F1 최대 / precision 우선 (§3.4) |

### 8.2 미결 — 상세

`gt.roi` 판정 기준은 **`center` 로 확정**됐다(§3.2). 아래 5건이 남는다.

---

#### 미결 1 — 해소됨 (`ncn_metric` 단독 사용 확인)

`adas_pipeline` 등 다른 워크플로가 `ncn_metric` 을 함께 쓰지 않는다는 것이 확인됐다
(2026-08-04). 따라서 §3.1 의 CLI 인자 이름·기본값을 외부 호환 없이 **이 저장소 관례대로
자유롭게 설계**할 수 있고, 그래서 학습 CLI 와 같은 `-w` 관례를 채택했다.

`~/Downloads/ncn_metric/eval.py` 드라이버는 더 이상 맞춰야 할 대상이 아니다 — 참고
소스로만 쓰고, 통합이 끝나면 그 역할(토큰 렌더·보드 JSON 병합·mAP 파싱)은 전부
`tools/customEval.py` 안으로 들어온다.

#### 미결 2 — 해소됨 (평가 config 통합)

`configs/eval/` 신설 안은 폐기됐다. **평가 설정을 학습 `exp.yaml` 의 `eval:` 블록으로
통합하고, 평가 산출물도 학습 산출물 디렉토리(`runs/.../eval/<variant>/`)에 저장한다** (§3.0).

`configs/` 트리는 네 개(`datasets` `experiments` `releases` `archive`)로 유지된다.
CLAUDE.md Project Structure 절은 손댈 필요가 없고, 대신 `exp.yaml` 스키마에 `eval:` 키가
추가된 사실을 §4.2(config 스키마) 설명에 반영한다.

**부수 효과** — `{datasets}`/`{weights}` 토큰 치환(F-30)이 **불필요해졌다.** 학습 config 는
이미 저장소 상대경로와 `/dataset/...` 컨테이너 경로를 쓰므로, 외부 저장소를 가리키기 위한
토큰이 필요 없다. F-30 은 범위에서 제외한다.

#### 미결 3. `prediction.json` 저장 여부 — 용량 대 재현성

**무엇이 문제인가** — 평가에 쓴 예측을 COCO 형식으로 남기면 **재평가·디버깅이 쉽다.**
GT나 지표 코드를 고친 뒤 추론을 다시 돌리지 않고 비교할 수 있다. 대신 커진다.

**실측 규모** (Corner 5,000장)

| 산출물 | 크기 |
|---|---|
| board 원본 덤프 | 14 MB (91,018 detection) |
| ONNX 예측 (conf=0.01) | 추정 20~40 MB — 임계가 낮아 detection이 많다 |

**갈리는 선택**

| 선택 | 결과 |
|---|---|
| 항상 저장 | 재현성 최상. `runs/eval/` 가 평가 횟수에 비례해 커진다 |
| `--save-pred` 플래그로 선택 | 기본은 안 저장. 필요할 때만. **권장** |
| 저장 안 함 | 매번 재추론. ONNX 5,000장이 수 분 걸린다 |

**부수 확인 필요** — `runs/` 는 `.gitignore` 전체 제외이므로 git에는 영향이 없다.
디스크만 문제다. 현재 로컬 여유 1.2 TB.

---

#### 미결 4. 구획별 지표 재현 범위 — F-24 범위 확정

**해소된 부분** — `sens`/`opt`/`insens` 열의 정의는 해독됐고(§3.4), 우리
`Evaluator.f1_score` 가 이미 같은 알고리즘으로 산출한다. `TP`/`FP`/`FN` 카운트도
회수한 `thresholds_detail.json` 에 정답이 있다.

**남은 것** — `metrics_corner_board.txt` 의 구획별 재집계다.

```
[영역별 edge(좌+우)] AP / P / R / F1     ← ROI 를 좌·우·중앙으로 나눈 재집계
[영역별 center]      AP / P / R / F1
```

**모르는 것** — 분할 경계 좌표가 기록돼 있지 않다. ROI `[428, 8, 1072, 1072]` 를
가로로 3등분한 것으로 보이나(edge = 좌·우, center = 중앙) 비율이 불명이다.

**갈리는 선택**

| 범위 | 작업량 | 얻는 것 |
|---|---|---|
| 구획별 제외 | — | §0.1 기준값 검증에 충분. **권장** |
| 3등분 가정으로 구현 | medium | 기존 산출물과 형식 일치. 다만 경계가 틀리면 수치가 안 맞아 검증 불가 |
| 경계를 config 로 노출 | medium | `gt.regions: {edge: [...], center: [...]}`. 유연하나 기존 값 재현은 여전히 불가 |

**권장** — 이번 범위에서 **제외**한다. 기존 산출물의 구획 경계를 아는 사람이 확인되면
별도 작업으로 뗀다. 화각 위치별 성능 분석은 유용하지만 §0의 수치 동일성 제약과 무관하다.

---

## 9. 검증 스크립트

§7의 재현에 쓴 스크립트다. 세션 scratchpad에 있고, **구현 착수 시 `tests/` 로 옮겨 회귀
테스트로 만드는 것을 권한다** — 현재 평가 경로에 테스트가 하나도 없다.

| 스크립트 | 검증 대상 | 실행 시간 |
|---|---|---|
| `verify_board.py` | board JSON → §7.3 표 + ROI 가설 (GT 21,742) | 약 10초 |
| `verify_onnx.py` | ONNX 5,000장 추론 → §7.4 표 | 약 3분 (GPU) |

실행 방법 (컨테이너에서, scratchpad 를 `/probe` 로 마운트):

```bash
docker compose run --rm --no-deps -v <scratchpad>:/probe dair-train \
    bash -lc 'cd /workspace && uv run python3 /probe/verify_board.py'
```

두 스크립트가 쓰는 자산은 §7.1에 전부 회수돼 있다. 데이터셋만 `/dataset` 마운트가 필요하다.

**테스트로 옮길 때 다듬을 것**

1. ONNX 5,000장은 CI에 넣기엔 무겁다. `--limit` 을 받아 앞 N장만 돌리는 빠른 모드를 두고,
   전량 검증은 릴리스 전 수동 실행으로 남긴다.
2. `verify_board.py` 는 `classes` 를 `[bus,car,motor,person,truck]` 로 바꿔 `MOTOR` 를
   우회했다. `cat_map`(F-07) 구현 후에는 정식 경로로 고친다.
3. 기준값을 스크립트 상단 상수에서 §0.1 을 참조하는 형태로 바꿔 한 곳만 고치면 되게 한다.

---

## 10. 구현 기록 (2026-08-04)

결정 기록은 **D-020**. 이 절은 계획과 구현이 어긋난 지점, 계획이 남긴 미결의 처리,
그리고 구현하면서 새로 알게 된 사실을 모은 것이다.

### 10.1 미결 처리

| 미결 | 처리 |
|---|---|
| §3.4 부등호 (`>` 대 `>=`) | **해소 — 부등호는 원인이 아니다.** board 예측으로 두 부등호를 직접 비교한 결과 산출 구간이 완전히 동일했고 `f1 == crit` 인 grid 점이 **0개**였다. `crit = maxF1 * 0.99` 라는 부동소수 값에 1001점 격자가 정확히 걸리지 않기 때문이다. 기존 `>` 를 유지했다 (수치 무변화) |
| §8.2 미결 3 — `prediction.json` | **권장안 채택 — `--save-pred` 플래그, 기본 미저장.** board 5,000장 실측 **8.4 MB** (계획의 20~40 MB 추정보다 작다. `round(x, 3)` 로 좌표를 줄인 덕이다) |
| §8.2 미결 4 — 구획별(edge/center) 지표 | **권장안 채택 — 범위에서 제외.** 분할 경계 좌표가 기록돼 있지 않아 재현 검증이 불가능하다. 경계를 아는 사람이 확인되면 별도 작업으로 뗀다 |
| §3.5 F-35 — `calc_class_thresholds.py` 흡수 | **권장안 채택 — 흡수하지 않는다.** 자체 IoU 매칭을 들고 오면 정확도 경로가 여섯 번째로 늘어난다. 정의와 파라미터(`resolution=1001`, `cut=0.99`, 보간)만 참고했고 그 정의는 이미 `Evaluator.f1_score` 에 있다 |

### 10.2 임계값의 남은 편차 — 원인 규명

회수한 `thresholds_detail.json` 대비 3단계 임계값이 최대 **0.01** (truck optimal 0.231 대
0.222) 어긋난다. 부등호가 원인이 아님이 확정됐으므로 원인은 하나다 —
**기록값은 pycocotools 를 쓰지 않는 독립 구현으로 산출됐다.** 참고
`calc_class_thresholds.py` 는 자체 `match()` 와 누적 PR 을 구현한다. board 예측은 양쪽에서
바이트 동일한 입력인데도 P/R 곡선이 미세하게 다르고, AP 는 3자리에서 일치하지만 F1 곡선
꼭대기가 평탄해 `argmax` 가 다른 격자점을 고른다.

board 실행에서 확인된 대조 (기록 → 재현):

| class | P | R | maxF1 | optimal |
|---|---|---|---|---|
| bus | 0.648 → **0.648** | 0.432 → **0.432** | 0.519 → **0.519** | 0.211 → **0.211** |
| car | 0.631 → **0.631** | 0.424 → **0.424** | 0.507 → **0.507** | 0.225 → **0.225** |
| person | 0.692 → **0.692** | 0.487 → **0.487** | 0.572 → **0.572** | 0.209 → **0.209** |
| truck | 0.800 → 0.801 | 0.517 → **0.517** | 0.628 → **0.628** | 0.222 → 0.231 |

P·R·F1 은 사실상 일치하고 `optimal` 만 truck 에서 벗어난다 — 평탄한 꼭대기의 증거다.
**§0.1 의 구속 기준값(AP)은 전부 재현됐으므로 인수 기준은 충족이다.** 임계값을 기록값과
자리까지 맞추려면 매칭 구현을 맞춰야 하고, 그것은 경로를 늘리는 일이므로 하지 않았다.

### 10.3 계획의 사실 정정

| 계획의 기술 | 실제 |
|---|---|
| §1.1 #3 `val/dair.py:76` 이 정확도 계산 경로 | **아니다.** `:76` 은 GT annotation 을 읽기 위한 `COCO` 의 지연 import 이고, 정확도는 `eval_json`(`:190`)이 `Evaluator` 에 위임한다. 이미 정본 경로를 쓰고 있었다. 조치는 지연 import 를 모듈 최상단으로 올린 것뿐이다. **따라서 실제 경로는 처음부터 4개였다** |
| §3.6 `mc_curve.png` | 실제 파일명은 `F1_Confidence.png` (`evaluator.py` 의 기존 동작). 이름을 바꾸지 않았다 |
| §3.6 `metrics.f1_50` 과 `metrics.thresholds` | 둘 다 나간다. `thresholds` 는 `flatten_thresholds` 가 `{class: {sensitive, optimal, insensitive, maxF1, P, R, F1, ngt, TP, FP, FN}}` 로 접은 형태이고 `thresholds_detail.json` 과 같은 스키마다 |
| §2.3 F-16 "계산되지만 출력 안 됨" | 정확히는 `precision` mode 의 `IoU@50` 열이 AP@0.50 과 같은 값이었다. 다만 AP 라는 이름으로 읽히지 않아 실질적으로 감춰져 있었다. `ap` mode 를 신설해 이름을 붙였다 |
| §5-5 "`--dry-run` … `ncn_metric/eval.py` 가 뱉는 명령이 실행되는지" | 이 검증은 하지 않았다. 미결 1 에서 `ncn_metric` 이 더 이상 맞출 대상이 아님이 확정됐으므로 대상이 사라졌다 |

### 10.4 계획에 없던 구현 판단

| 항목 | 판단과 이유 |
|---|---|
| `DetectionResult(list)` | 추론 결과에 `dropped` 개수를 달아 나르기 위한 list 서브클래스. pycocotools `loadRes` 가 `type(anns) == list` 를 **정확히** 검사하므로(서브클래스 거부) 넘길 때 `list(...)` 로 되돌린다 |
| `Evaluator.close()` | loguru `logger` 가 전역이라 variant 를 연속 평가하면 sink 가 쌓여 두 번째 variant 의 로그가 첫 번째의 `eval.log` 에도 들어간다. 핸들러 id 를 들고 있다가 떼어낸다. 실행 후 양쪽 `eval.log` 의 교차 오염 0건 확인 |
| `pred2cocoJson` 의 `.pop` → `.get` | `Evaluator(None)` 은 모듈 전역 `DEFAULT_CFG` 를 **복사 없이** 참조한다. 기존 `.pop` 은 그 전역 dict 를 영구히 변형했다. `.get` 으로 바꿔 잠재 버그를 없앴고, 덤으로 `evaluation.json` 의 `config.resolved` 에 주입된 함수가 기록된다 |
| `apply_cat_map` 의 대소문자 무시 | 보드는 `MOTOR`, config 는 `motor` 로 쓰는 일이 흔하다. 숫자 키와 문자열 키를 모두 받고 `np.int64` 도 흡수한다 |
| `--mode` 의 argparse `choices` | CLI 단계에서 먼저 걸러 오타가 실행 전에 잡힌다. `report.validate_modes` 의 `ValueError` 는 config 로 들어온 값에 대해 여전히 동작한다 |
| 종료 코드 1 | `target` 미달(NON-PASS)이면 종료 코드 1 을 낸다 — CI 가 로그를 파싱하지 않아도 되게 |
| `iter_images` 정렬 | `os.listdir` 순서는 파일시스템에 따라 다르다. 정렬해 실행 간 재현성을 확보했다 (AP 에는 영향 없지만 로그·디버깅에 영향) |
| GT 에 없는 이미지 건너뛰기 | `images/` 에 있으나 GT json 에 없는 파일이 있으면 `im_filename2id` 조회가 `KeyError` 로 죽었다. 건너뛴다 |
| `filter_gt_roi` 가 `catToImgs` 도 갱신 | §7 검증 스크립트는 `anns`/`imgToAnns` 만 걸렀다. COCOeval 은 `imgToAnns` 만 타므로 AP 에는 영향이 없지만, 객체를 자체 정합하게 두는 편이 낫다. 갱신 후에도 board/onnx 수치 무변화 확인 |

### 10.5 검증 결과

| 항목 | 결과 |
|---|---|
| board (`ipu_test`) mAP@0.50 | **0.481** (기준 0.481, Δ +0.000). per-class 최대 편차 motorcycle −0.002 |
| float (`onnx`) mAP@0.50 | **0.460** (기준 0.460, Δ +0.000). per-class 최대 편차 motorcycle −0.001 |
| Δ (board − float) | **+0.0211** (계획 예상 +0.021) |
| GT (ROI 내부) | **21,742** / 전체 30,567 — 정확히 일치 |
| `num_dropped_by_cat_map` | **0** (보드 5종 전부 매핑) |
| ROI 미적용 시 | 0.356 — 회귀 테스트가 이 하락을 명시적으로 검사한다 |
| pytest | **82 passed, 3 skipped** (기준선 56/2). 단위 24 + 기준값 재현 3, ONNX 전량 1건은 `EVAL_BASELINE_FULL=1` 전용 |
| ruff | **9건** — 착수 시 14건에서 감소. 편집한 파일의 항목만 정리했다 (§10.7) |
| 종료 코드 | NON-PASS 1 / PASS 0 확인 |

### 10.6 §9 검증 스크립트의 처리

scratchpad 의 `verify_board.py` · `verify_onnx.py` 는 `tests/test_eval_baseline.py` 로
옮겼고, 계획이 지시한 세 가지를 반영했다.

1. ONNX 전량 재현은 `EVAL_BASELINE_FULL=1` 뒤로 숨겼다 (5,000장, GPU 1~2분).
2. `verify_board.py` 의 `classes: [bus, car, motor, ...]` 우회를 없앴다 — `cat_map` 으로
   정식 경로를 탄다.
3. 기준값을 스크립트 상단 상수에서 읽는 대신 **`exp.yaml` 의 `eval:` 블록을 그대로 읽어**
   실행한다. config 가 바뀌면 테스트가 바뀐 값을 본다.

자산이 없으면 `pytest.skip` 이므로 **초록색이 게이트 통과를 뜻하지 않는다** — skip 여부를
확인해야 한다. CLAUDE.md 의 Testing 절에 이 경고를 적었다.

### 10.7 ruff 기준선 14 → 9

CLAUDE.md 는 `ignore` 목록의 잔여 항목에 대해 **"Reduce them in files you touch; do not
sweep the tree"** 를 규약으로 둔다. 착수 시 14건 중 **5건이 이 작업이 편집한 파일에**
있었으므로 그 5건만 정리했다.

| 파일 | 항목 | 처리 |
|---|---|---|
| `metrics/evaluator.py:669` | `E722` bare `except` (라벨 파일 열기) | `except OSError` — GT 가 없는 이미지는 정상이므로 빈 목록으로 넘기는 의도는 그대로다 |
| `metrics/evaluator.py:680` | `E722` bare `except` (`float(l)` 변환) | `except ValueError` — 값이 `line.split(sep)` 에서 오므로 발생 가능한 예외는 이것뿐이다 |
| `metrics/evaluator.py:86,89` | `W291`/`W293` (클래스 docstring) | docstring 을 다시 썼다 — 기존 내용은 `TODO: save result as txt` 였고 이번에 구현됐다 |
| `metrics/utils.py:140` | `W293` (`postprocess` docstring) | 공백 제거 |

**남긴 9건** — 이번 작업이 손대지 않은 파일에 있다. bare `except` 1건
(`custom_coco_eval.py:127`), `F811` 2건(`dair.py:13` · `symmetric.py:31`, 둘 다 `RANK`
중복 import), `E711` 1건(`custom.py:100`), `W291`/`W293` 5건
(`similar_images_remover.py` ×3 · `confusion_matrix.py` ×1 · `custom_coco_eval.py` ×1).

**예외를 좁히는 변경은 정확도 경로에 닿는다** — `read_txt` 가 GT 라벨 txt 를 읽는 함수다.
따라서 기준값 게이트를 다시 돌려 확인했다: `EVAL_BASELINE_FULL=1` 로
**83 passed, 2 skipped**, board 0.481 / float 0.460 무변화.
