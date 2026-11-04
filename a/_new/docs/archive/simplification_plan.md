# 간소화 현황·결정 기록

2026-09-28에 간소화 계획·검토·정적 인벤토리를 통합했다.
2026-09-28에 완료 기록으로 archive에 이관했다. 절 번호는 코드·테스트의 근거 참조용으로 보존한다.
이 문서의 상태·수치는 각 기록일 기준이며 현재 실행 방법은 아래 링크를 따른다.
실행 방법은 [사용법](../usage.md), 배포·검증 상태는 [NPU 포팅·배포·검증](../npu.md)를 따른다.

## 1. 현재 상태

2026-09-09의 지원 범위 결정과 1단계 삭제, 2026-09-10의 구조 통합은 완료됐다.
2026-09-23에는 보조 도구 22개를 추가 정리했다.
남은 검증 과제는 §10에 모았다. 아래 날짜별 수치·커밋은 당시 기록이다.

### 1.1 유지 범위

| 항목 | 현재 지원 |
|---|---|
| 모델 | custom, symmetric, dair, dms_pose |
| 학습 | 단일 GPU, DDP, resume |
| 평가 | 학습 중 Ultralytics 평가 + 릴리스용 customEval |
| 추론·분석 | PT/ONNX 시각화, ROI·정답 필터, 예측 공간 분석 |
| 포팅 | Novatek, Sigmastar, DMS ONNX export |
| 환경 | Ultralytics 8.4.6 고정, pyproject.toml + uv.lock, checkout + editable |
| 보존 | 릴리스 설정·가중치·MANIFEST, 과거 설정과 설계 결정 기록 |

### 1.2 모델 선택 계약

`projects/src/engine/model.py`의 `MODEL_REGISTRY`가 모델·trainer·validator·predictor 조합의
정본이다. 지원하지 않는 이름과 제거된 이름은 `ValueError`로 실패한다.
과거의 알 수 없는 이름 → custom 폴백은 제거했다.
Ultralytics의 `model_name` 속성과 충돌하지 않도록 내부 선택 키는 `registry_key`를 쓴다.

## 2. tools 정리 — 2026-09-23

유지하는 실행 파일은 8개다.

| 파일 | 목적 |
|---|---|
| `train_custom_trainer.py` | 공통 학습 진입점 |
| `dist_train_custom.sh` | 분산 학습; 사용자 지정 유지 대상 |
| `validate.py` | 계열별 PT 검증 |
| `customEval.py` | ONNX·보드 결과 평가 |
| `infer.py` | 추론·시각화 |
| `export_onnx.py` | 보드별 ONNX 출력 규약 |
| `analyze_crop.py` | ROI crop·정답 필터 |
| `analyze_heatmap.py` | 클래스 임계·1:1 매칭·10×10 공간 분석 |

추가 삭제한 보조 도구 22개:

| 분류 | 삭제 파일 |
|---|---|
| 옛 튜닝 런처 | `train_tuner_iter.py`, `train_tuner_tlds.py`, `dist_train_tuner_iter.sh`, `dist_train_tuner_tlds.sh` |
| 데이터 통계·조합 | `get_label_stats.py`, `data_analysis.py`, `sample_val_balanced.py`, `merge_datasets.py` |
| 단순 시각화 | `drawing_labels.py`, `vis_class5.py` |
| 보조 crop 생성 | `prepare_train_bbox_crops.py`, `prepare_coco_seg_crops.py` |
| 중복 추론·평가 | `infer_onnx_batch.py`, `eval_test.py` |
| SDK 배치 보조 | `prepare_sim_eval_imgs.py`, `run_sim_batch.py`, `eval_det_txt.py` |
| 일회성 검증·변환 | `verify_base.py`, `verify_gate.sh`, `make_smoke_subset.py`, `convert_legacy_checkpoint.py` |
| 예측 JSON 단독 출력 | `predict_to_json.py`; 사용자 지정 정리 대상 |

이 삭제는 §3.1(d)의 **지원 중단 결정**이다.
평가의 예측 저장은 `customEval.py --save-pred`, SDK 변환·시뮬레이션은 각 보드 wrapper를 쓴다.
예전 도구 이름은 삭제 이력이며 현재 실행 안내가 아니다.

테스트용 릴리스 경로는 `tests/conftest.py`로 옮겼다.
학습 smoke는 공통 학습 CLI와 임시 출력 경로를 사용하고, 공통 decode는
`projects/metrics/postprocess.py`에서 읽는다.
`runs/verify/base/`는 보관된 회귀 자산으로 유지하며 자산 부재는 `NOT_RUN`이다.
마지막 소비자가 없어진 dev 의존성 `xlsxwriter`, `imutils`도 제거했다.

당시 검증: CUDA 비활성화에서 **234 passed, 9 skipped, 4 xfailed**.
skip은 GPU 학습·resume 4건, CUDA forward 3건, 전체 ONNX 평가 opt-in 2건이었다.
Python CLI 7개의 help, 분산 shell 구문, offline lock 검사, 변경 파일 lint는 통과했다.
전체 저장소 Ruff 98건은 기존 오류로 남았다. 이 수치는 현재 테스트 개수의 고정 기준이 아니다.

## 3. 삭제 근거와 초기 조사

### 3.1 삭제 기준

| 구분 | 판단 |
|---|---|
| (a) 도달 불가 | 현재 소비자·진입 경로가 없음 |
| (b) 실행 전제 부재 | 필요한 의존성·설정·데이터 또는 코드가 없어 기존 경로가 성립하지 않음 |
| (c) 대체·폐기 | 같은 업무가 다른 유지 경로로 통합됐거나 폐기됨 |
| (d) 지원 중단 | 제품·학습·운영 범위에서 제외하기로 명시적으로 결정함 |

CLI에 importer가 없다는 사실, 로컬 데이터가 없다는 사실만으로 삭제를 확정하지 않는다.
정적 중복은 후보 목록이며 동작 동등성의 증거가 아니다.
YAML 내용이 같아도 상속 경로·파일명·run 디렉터리 유도가 다르면 합치지 않는다.

### 3.2 2026-09-08 정적 인벤토리

| 항목 | 당시 관측 |
|---|---|
| Python | 인벤토리 157개, 후속 검토 시 tracked 158개; AST parse 실패 0 |
| 규모 | projects 91파일·19,415행, tools 55파일·10,621행, tests 9파일·1,971행 |
| 중복 후보 | AST 함수 본문 27그룹, byte-identical YAML 8그룹 |
| YAML | 활성 95개, archive 205개 |
| 의존성 | pyproject와 requirements 각각 20개, 집합 동일 |
| 별도 확인 | 당시 ignore되던 `sigmastar/converter/` 등 자체 작성 소스 |

함수 본문 비교는 signature·decorator·상속·전역 상태를 제외했다.
YAML 비교는 상속과 실행 경로 의미를 해석하지 않았다.
SDK vendor 전체 소스 조사도 아니었다. 상세 스냅샷은 Git 이력에 남으며 현재 파일 수로 사용하지 않는다.

당시 검토에서 확인한 export 반환값 누락과 Novatek 생산자/소비자 출력 경로 불일치는 먼저 수정됐다.
이후 각각 공통 exporter와 manifest로 통합됐다(§6-D/G).
검토 시점의 컨테이너 미실행·데이터 부재 추정은 후속 실데이터 검증으로 갱신됐다.
호스트 데이터는 `/mnt/HDD_1/Workspace/data`, 컨테이너 마운트는 `/dataset`이었다.

## 4. 삭제 실행 이력

### 4.1 muon

`24940ce`: 미사용 optimizer 1파일·373행 삭제. 당시 pytest 123, Ruff 100,
보드 결과 재평가 지표는 유지됐다.

### 4.2 KD

`d8a739e`: 런처·모델·trainer 등 12파일·1,387행 삭제. 당시 pytest 126,
보드 결과 재평가 지표는 유지됐다.

### 4.3 QAT

`1759f09`: QAT/FX-QAT/NMQT 계열, 전용 설정·호환 코드·테스트 등 24파일·4,190행 삭제.
D-026이 과거 D-010의 보류·보존 결정을 대체했다.

**삭제 근거는 §3.1(d) 지원 중단이다.**
당시 registry와 YAML에서 QAT에 도달할 수 있었으므로 D-026의 “도달 불가” 표현을
삭제 근거로 확대하지 않는다. [설계 결정 기록](decisions.md)은 append-only 이력으로 보존한다.

전용 테스트 15개가 함께 삭제돼 당시 pytest는 117로 줄었다.
이는 회귀 실패와 구분해 기준을 갱신했으며, 학습 loss 6항목은 소수 4자리까지 동일했다.

### 4.4 옛 스크립트

`fca1023`: 일회성·중복 스크립트와 소비자를 잃은 생성기 등 9파일·873행 삭제.
옛 PT 로딩에 필요한 별칭은 잠시 유지한 뒤 PT 변환을 마치고 제거했다(§10.2).
이때 유지했던 시각화·검증 도구 중 일부도 2026-09-23에 추가 삭제했다(§2).

### 4.5 CVD-H210 및 지원 업무

| 대상 | 커밋 | 당시 범위·결과 |
|---|---|---|
| 옛 CVD-H210 dump | `47b0a07` | 2파일 삭제; 과거 실험 설정은 이후 archive로 이동 |
| 다단계 finetune | `d3320f3` | 6파일·1,255행 삭제; pytest 119 |
| ncnvox | `5ec431e` | 31파일·3,131행 삭제; 전용 테스트 제외 후 pytest 88 |
| consi | `16217ac` | 3파일·609행 + loss 125행 삭제; pytest 91 |

ncnvox·consi·다단계 finetune은 지원 범위 결정에 따른 삭제다.
릴리스 MANIFEST의 `consistency:`는 as-run 정합 기록 필드로 모델 이름 `consi`와 다르다.
`configs/archive/`의 과거 설정은 실행 대상으로 안내하지 않으며 이력으로 보존한다.

## 5. 복구와 참조 관리

### 5.1 복구 단위

삭제 묶음의 커밋은 복구의 출발점이다. **항상 한 번의 충돌 없는 revert로 복구되는 것은 아니다.**
muon·KD·QAT revert 시험에서는 공통 문서·registry·테스트 충돌이 있었고,
당시 `fca1023`, `47b0a07`은 충돌 없이 적용됐다.
복구 시 현재 registry·의존성·설정·테스트 계약에 맞추고 관련 경계 검증을 다시 수행한다.
옛 pickle 별칭을 자동으로 되살리지 않는다.

### 5.2 참조 검사

`tests/test_removal.py`가 삭제 파일과 import·실행 안내의 잔여 참조를 검사한다.
`docs/archive/`, `configs/archive/`, 이 문서는 역사 기록이므로 제외한다.
릴리스 exp.yaml·MANIFEST의 당시 실행 기록도 개별 예외로 보존한다.
활성 사용법에는 삭제된 도구를 실행하는 명령을 남기지 않는다.

## 6. 구조 통합 결과

### A. registry와 학습 런처

`MODEL_REGISTRY`로 4개 모델의 조합을 통합했다. 전용 symmetric 런처는 제거했고
`train_custom_trainer.py`와 `dist_train_custom.sh`를 사용한다.
`registry_key` 속성 충돌 수정은 `181b97b`에 기록됐다.
CLI override를 적용한 실험 설정이 checkpoint에서 주입된 data/imgsz보다 우선한다.
DDP·resume 경로는 유지한다.

### B. 다단계 finetune

v1/v2 파이프라인과 `load_pipeline_yaml`을 지원 중단 결정에 따라 제거했다.
단순히 현재 호출이 없다는 사실을 근거로 삼지 않았다.

### C. 전처리·후처리

`projects/metrics/preprocess.py`의 `InputProfile`·`Geometry`·`profile_for_weight`와
`postprocess.py`의 공통 decode/NMS를 사용한다.
릴리스 메타데이터와 `input_scale.json`을 읽고 명시적 CLI override를 우선한다.
Detection 기본은 BGR·[0,255]·stretch·INTER_AREA다.

과거 추론 CLI의 RGB+letterbox 오류를 수정해 2026-09-10에 회귀 결과를 다시 기록했다.
Corner smoke 10장 custom 검출은 13→17, dair는 1→3으로 바뀌었다.
고정 tensor forward digest는 유지됐다. PC 공통 NMS와 보드 NMS의 차이는
[NPU 포팅·배포·검증](../npu.md)에 명시한다.

### D. Novatek 산출물 경로

`novatek/convert/_common.sh`가 `runs/npu/novatek/<model>/`로 출력을 분리한다.
`run_manifest.json`의 model_dir·run_dir·pattern·imgsz·test_img_dir·debug_dir·staged_at·npu_sdk가
생산자와 소비자의 경로 기준이다. SDK worker는 torch 없이 실행된다.
배치 보조 CLI는 §2에서 삭제했으며 compiler·simulator·visualizer wrapper는 유지한다.
6클래스 시각화와 manifest 경로는 `tests/test_npu_manifest.py`로 보호한다.

### E. 설정

모델 구조 YAML은 `configs/models/`, 실험은 `configs/experiments/`에 모았다.
옛 CVD-H210 실험은 archive로 이동했다. 릴리스 설정은 당시 기록으로 보존한다.
파일명·상속·run 경로 유도 의미를 유지하며, 이동 전후 4개 모델의 구조·파라미터 수를 대조했다.

### F. 평가 진입점

계열별 PT 검증 런처를 `tools/validate.py`로 통합했다.
custom/symmetric는 Model.val, dair는 DynamicCrop 데이터 로더를 사용하는 경로다.
학습 중 평가 코어와 릴리스용 `tools/customEval.py`는 용도가 달라 유지한다.
중복 보조 평가 CLI는 §2에서 제거했다.

### G. ONNX export

`tools/export_onnx.py --target novatek|sigmastar|dms`로 통합했다.
커스텀 SymmetricDetect와 순정 Detect를 자동 판별한다.

| target | 출력 |
|---|---|
| novatek | sigmoid 클래스 점수 + DFL 적용 거리, 2출력 |
| sigmastar | 정규화 xyxy·max_score·클래스 점수, 단일 출력 |
| dms | stage별 pose 3출력 |

순정 Detect의 기존 decoded 단일 출력 문제를 수정하고 ONNX Runtime으로 계약을 검증한다.
Detection head에 DMS target을 적용하면 명확히 실패한다.
DMS 실제 PT에 대한 검증 자산은 §10의 미결 항목이다.

### H. 의존성·SDK 소스

`pyproject.toml`과 `uv.lock`을 의존성 정본으로 하고 중복 requirements·setup shim을 제거했다.
마지막 소비자가 사라진 audio extra는 2026-09-22, 추가 dev 의존성은 §2에서 정리했다.
Sigmastar Python 3.7 SDK 의존성은 학습 환경과 분리한다.

Sigmastar 자체 작성 converter 소스는 추적하고 SDK·데이터·산출물은 제외한다.
Novatek 모델 레시피는 텍스트 설정을 추적하고 바이너리·이미지는 제외한다.
2026-09-10 Sigmastar 정리 검증에서는 같은 slug의 .sim 7개와 .img 해시가 이전 실행과 같았다.
slug가 달라지면 바이너리 해시도 달라질 수 있다.
결과 JPEG는 동일 코드 재실행에서도 달라져 모델 동등성 근거로 사용하지 않았다.

## 7. 유지하는 디렉터리 구조

| 경로 | 역할 |
|---|---|
| `tools/` | 사용자 CLI 8개 |
| `projects/src/` | Ultralytics 구조를 따르는 모델·학습 코드 |
| `projects/metrics/` | 평가·전후처리·공간 분석 |
| `configs/{models,datasets,experiments,releases,archive}/` | 모델·데이터·실험·릴리스·과거 설정 |
| `docker/{train,novatek,sigmastar}/` | 실행 환경 |
| `novatek/convert/`, `sigmastar/converter/` | SDK wrapper·변환 코드 |
| `runs/` | 비추적 실행·검증 산출물 |

`projects/src/`를 별도 training/inference/deployment 패키지로 다시 나누지 않는다.
현재 checkpoint의 pickle 클래스 경로와 코드 탐색 구조를 유지한다.
SDK 디렉터리를 옮기는 추상 계층이나 모든 작업을 감싸는 단일 CLI도 추가하지 않는다.

## 8. 검증 기록과 운영

### 8.0 회귀 기준 자산

2026-09-09에 학습 1 epoch, 고정 tensor forward, 실제 이미지 추론,
PT→ONNX, Novatek 시뮬레이션과 저장된 보드 결과 평가를 확보했다.
대표 모델은 custom=Corner_Truck/v01, symmetric=Front_Truck/v01, dair=Side_Truck/v01이다.
smoke 데이터는 CVD/Corner train 64장·val 33장, seed 고정이다.

`runs/verify/base/`에 manifest·gate·train/infer/export JSON·고정 입력/출력 NPZ를 보관했다.
2026-09-10의 전처리 수정 후 이미지 추론 결과는 갱신했고 이전 RGB 결과는 별도로 보존했다.
이 경로는 Git 비추적이며 파일이 없는 환경에서는 검증이 실행되지 않는다.
당시 생성 도구는 §2에서 삭제돼 현재 재생성 명령으로 안내하지 않는다.

기록상 Corner 신규 ONNX는 출하 ONNX와 출력 max diff 0.0이었다.
Front 출하 ONNX는 best.pt 신규 export와 출력 차이 0.049 / 6.15가 있어 출처 확인이 남았다.
순정 head의 당시 잘못된 단일 출력은 §6-G에서 수정했으므로 옛 shape를 현재 계약으로 사용하지 않는다.
회귀 기준은 “이전과 동일함”을 확인하며, 알려진 입력에 대한 독립 정답 검증을 대신하지 않는다.

### 8.1 당시 전체 검증 기준

| 항목 | 기록·해석 |
|---|---|
| 2026-09-08 pytest | 116 passed, 2 skipped |
| Ruff | 전체 실행 100건 확인; 과거 6건 기록은 재현되지 않음 |
| Corner 보드 결과 mAP@0.50 | 0.481; ROI GT 30,567→21,742 |
| Front 보드 결과 mAP@0.50 | 0.704; ROI GT 61,588→50,371 |
| 2026-09-23 pytest·lint | §2의 234 passed / 9 skipped / 4 xfailed, 전체 Ruff 98건 |

삭제한 기능의 전용 테스트가 줄어드는 것과 살아 있는 기능의 회귀 실패를 구분한다.
숫자만 유지하는 게이트로 학습·export·변환 정상 여부를 판단하지 않는다.
Ruff의 대규모 공백 정리는 기능 변경과 분리하며 기존 오류를 숨기기 위해 규칙을 끄지 않는다.

### 8.2 경계 검증

| 경계 | 검증 내용 |
|---|---|
| 설정 | _base_ 병합·상대경로·순환 오류, registry 선택, CLI 우선순위 |
| 데이터 | crop·bbox 재투영, 채널·범위·resize, 빈 라벨 |
| 모델 | 릴리스 PT 로딩·클래스 수, 고정 tensor 출력 |
| 학습 | 지원 계열 smoke, DDP·resume 상태 |
| 추론·export | 원본 좌표 복원, target별 ONNX 규약과 텐서 수치 |
| NPU | manifest 경로, 다중 클래스, 실제 SDK 시뮬레이션 |
| 평가 | 동일 GT·ROI·클래스·예측 형식의 결과 |
| 공간 분석 | 임계 경계, 순서 보존 1:1 매칭, 격자·크기·클래스 집계 |

합성 입력 검증과 실제 자산 검증을 나눈다.
학습 smoke·checkpoint·ONNX·SDK·보드 검증이 빠졌으면 그 범위를 명시한다.

### 8.3 삭제 보호

`tests/test_removal.py`는 제거된 모델 이름의 폴백 금지, 남은 trainer·YOLO subclass,
삭제 파일·참조, 릴리스 PT 로딩과 고정 forward를 검사한다.
정리 대상이 아직 남은 검사는 strict xfail로 기록하고 제거 후 활성화한다.
외부 자산이 없는 검사는 `NOT_RUN:` 사유로 skip한다. skip은 통과가 아니다.

### 8.4 현재 실행

학습 컨테이너 `/workspace`에서:

```bash
uv run pytest tests/ -q -rs
uv run ruff check .
uv run python3 tools/customEval.py \
  -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml --eval board
uv run python3 tools/customEval.py \
  -w configs/releases/BSD_F606/Front_Truck/v01/exp.yaml --eval board
```

전체 ONNX 기준 평가를 포함할 때는 `EVAL_BASELINE_FULL=1 uv run pytest tests/ -q -rs`를 쓴다.
데이터·가중치·GPU·SDK가 필요한 검사는 실행 환경을 준비한다.
결과에는 커밋·lock 해시·장치·seed·입력 자산과 pass/fail/not_run 사유를 기록한다.

## 9. 남겨 둔 제약

- 릴리스 PT의 pickle 경로가 현재 모듈 구조에 의존하므로 폴더 이름만 바꾸지 않는다.
- source·실행 디렉터리와 SDK 사본을 혼동하면 이전 산출물을 검증할 수 있다. manifest·해시로 구분한다.
- 평가 수치가 같아도 학습·resume·export가 정상이라는 뜻은 아니다. 해당 경계를 직접 검사한다.
- 디렉터리 이동·지원 삭제·지표 변경·Ultralytics 업데이트·대규모 포맷 변경은 별도로 다룬다.
- 실물 보드 검증을 SDK 시뮬레이션으로 대체하지 않는다.

## 10. 확정 결정과 미결 항목

### 10.1 제품 결정 — 2026-09-09 확정

| 항목 | 결정 |
|---|---|
| 지원 업무 | detection·DMS pose 유지, ncnvox 제거 |
| 지원 보드 | Novatek·Sigmastar 유지 |
| 학습 | custom·symmetric·dair·dms_pose, DDP·resume 유지; consi·다단계 finetune 제거 |
| 호환성 | 기존 PT를 한 번 변환한 뒤 별칭 제거; CLI·YAML 형식 변경 허용 |

### 10.2 옛 PT 변환 — 2026-09-10 완료

릴리스 PT 7개를 pickle 수준에서 조사했고 `projects.dair(.model)` 옛 경로를 쓴 것은
Rear/v04 best.pt 하나였다. 현재 클래스 경로로 재저장하고 `legacy_conversion` 출처를 남겼다.
model/ema state_dict는 비트 동일, 고정 448×256 입력 forward 차이는 0.0이었다.
optimizer·train_args 등 나머지 키도 보존했다.

원본 `best.legacy.pt`와 읽기 전용 NAS 원본은 유지했다.
원본 해시 접두는 `8426d8d9`, 변환본은 `43e3dfad`이며 릴리스 MANIFEST에 기록했다.
옛 별칭을 차단한 별도 프로세스에서 원본 로딩 실패·변환본 성공을 확인한 뒤 `projects/compat/`를 제거했다.
보호 테스트는 `test_rear_v04_loads_without_any_alias`다.

일회성 변환 CLI도 2026-09-23에 제거됐다.
현재는 **현재 모듈 경로로 저장된 PT만 지원**하며 미변환 NAS 원본은 명확히 실패한다.
삭제된 CLI를 정상 운영 절차로 다시 안내하지 않는다.

### 10.3 미결 검증 자산·품질

| 항목 | 현재 상태 |
|---|---|
| Front 출하 ONNX 출처 | 기존 `network/260706_BSD-F606A_Front_v01.onnx`와 best.pt 신규 export의 불일치 출처 미확인. 최근 포팅은 best.pt에서 새로 export |
| DMS pose PT | 실제 PT 기준 회귀 검증 자산 확보 필요; 지원 유지 결정과 별개 |
| 전체 Ruff 오류 | 기존 98건의 후속 정리 필요(2026-09-23 기록) |
| Front/Novatek 양자화 | 2026-09-23 검증 최소 DFL cosine 0.9405로 0.95 미달 |
| 실물 보드 | 색공간·클래스·crop·펌웨어 조합·전체 정확도 확인 필요 |

NPU 검증의 상세 결과와 미확인 항목은 [배포 가이드](../npu.md)에서 관리한다.
