# DAIR-YOLO 코드베이스 상황 설명

 전체 구조와 영향 범위를 파악하도록 쓴 설명서다.

모든 수치는 실제 코드에서 직접 센 값이고 근거 파일·줄을 함께 적었다.

조사 시점 HEAD `9c8d09e` (= 태그 `v0.1.0`).

이 저장소는  **학습 파이프라인**이다. 

CLI script → Trainer → Dataset → 파일시스템 순으로 설명한다.

---

## 0. 표기 통일

같은 개념을 여러 이름으로 부르지 않기 위해 아래로 고정한다. 기술 용어는 영어를 그대로 쓴다.


| 개념                      | 이 문서의 표기                          | 쓰지 않는 표현   |
| ----------------------- | --------------------------------- | ---------- |
| 모델 마지막 출력부              | **head**                          | 헤드, 머리     |
| 예측 위치에 정답을 짝지음          | **assigner**                      | 배정기, 할당기   |
| 학습이 줄이려는 값              | **loss**                          | 손실, 로스     |
| 학습 데이터 인위 변형            | **augmentation**                  | 증강, 어그멘테이션 |
| 이름 → 클래스 대응표            | **registry**                      | 등록표, 레지스트리 |
| 오류 없이 다른 것이 대신 쓰임       | **fallback**                      | 폴백, 대체 동작  |
| 중간 계산값을 꺼내는 장치          | **hook**                          | 후크, 훅      |
| ONNX로 뽑기                | **export**                        | 내보내기       |
| ONNX → NPU 바이너리         | **convert** (컴파일 단계는 **compile**) | 변환, 빌드     |
| 정수 연산으로 줄이기             | **quantization** (학습 병행은 **QAT**) | 양자화        |
| 본체 옆에 두는 메타데이터 파일       | **sidecar**                       | 표식 파일      |
| 물체 찾기 태스크               | **detection**                     | 검출, 객체 탐지  |
| 얼굴 점 찾기 태스크             | **pose**                          | 자세 추정, 포즈  |
| DFL이 분포를 거리로 바꿀 때 곱하는 값 | **projection**                    | 투영 벡터      |
| 조건 미충족 시 함수가 그냥 반환      | **no-op**                         | 아무 일도 안 함  |
| 반드시 만족해야 하는 선행 조건       | **gate**                          | 통과 조건, 관문  |
| 한 번의 학습 실행              | **run**                           | 실행, 시행     |
| 파이프라인의 한 칸              | **stage**                         | 단계         |


`model_name`, `dataset_name`, `exp_params`, `overrides` 처럼 코드에 있는 키 이름은
항상 원문 그대로 쓴다.

---

## 1. 한눈에

**Ultralytics YOLO 8.4.6을 상속으로 확장한 학습 저장소**다. 최종 타깃은 대시캠에 들어가는
**Novatek CNN25 NPU** 이고 세 가지 태스크를 다룬다.


| 태스크                  | 내용                                   |
| -------------------- | ------------------------------------ |
| detection            | 차량·보행자·신호등. 전방 4K / 후방 QHD / 측방      |
| pose                 | 운전자 감시(DMS) 얼굴 70 keypoint           |
| audio classification | wakeword (KWS 2종) + command (CMD 8종) |


직접 만든 자산 세 개가 나머지 복잡도의 원인이다.

1. **DAIR** — 원본과 augmentation 결과의 예측이 서로 달라지면 loss에 벌점을 준다.
2. **Symmetric DFL** — 박스 거리를 음수까지 표현한다.
3. **NPU quantization·convert 경로** — 컴파일러가 정한 정밀도를 다시 학습에 반영한다.

### 규모


| 영역                       | 파일      | 줄          |
| ------------------------ | ------- | ---------- |
| `projects/src/` (확장 본체)  | 75      | 14,712     |
| `tools/` (CLI script)    | 55      | 10,083     |
| `projects/metrics/`      | 4       | 1,331      |
| `projects/custom_utils/` | 6       | 1,247      |
| `tests/`                 | 7       | 1,179      |
| `novatek/`               | 2       | 569        |
| `quant_tools/`           | 2       | 96         |
| **Python 전체**            | **171** | **32,068** |


YAML 294개, 셸 스크립트 11개. `tools/` 중 46개가 argparse CLI다.

큰 파일 순: `utils/loss.py` 1377 · `data/augment.py` 1289 ·
`tools/nova_qat_pipeline.py` 1065 · `tools/ga_tuner.py` 874 ·
`tools/run_finetune_pipeline_v2.py` 799 · `train/symmetric.py` 684.

---

## 2. 확장 전략 — fork가 아니라 상속

`projects/src/` 는 Ultralytics의 디렉토리 배치를 그대로 따른다
(`models/yolo/detect/{train,val,predict}`, `nn/{modules,tasks}`, `data/`, `engine/`,
`utils/`). 각 파일은 upstream 클래스를 상속해 **필요한 메서드만** override한다.

❗**대가는 버전 고정이다.** `ultralytics==8.4.6` 로 못 박혀 있다 (`pyproject.toml:18`).
upstream 내부 API에 직접 의존하므로 상위 버전은 `RandomPerspective` 같은 내부 변경만으로
깨진다.

❗**규칙은 복붙이 아니라 위임이다.** D-002가 근거다 — `dair.py` 가 upstream
`BaseTrainer._do_train`(약 167줄)을 복사해 재구현하고 있었고, `super()` 위임으로 바꿔
약 365줄을 줄였다.

❗**위임할 수 없는 한 곳이 있다.** upstream `DetectionModel.__init__` 은 upstream
`parse_model` 을 부르는데 그것이 우리 `SymmetricDetect`/`DMSPose` 를 모른다. 그래서
커스텀 모델은 이렇게 한다.

```python
super(DetectionModel, self).__init__()   # 한 단계 위로 건너뛴다
# 이후 DetectionModel.__init__ 본문을 복제하고 parse_model 만 우리 것으로 교체
```

우리 `parse_model` 은 `projects/src/nn/tasks/parse_util.py` 에 있다. D-017은
`symmetric.py` 만 이 패턴을 어겨 `KeyError: 'SymmetricDetect'` 로 죽었던 회귀를 기록한다.

---

## 3. 아키텍처 레이어와 패턴

### 3.1 레이어 구성


| 레이어            | 이 저장소의 실체                                      | 파일                                                     |
| -------------- | ---------------------------------------------- | ------------------------------------------------------ |
| User Interface | argparse CLI script                            | `tools/train_custom_trainer.py` 등 46개                  |
| Config         | config loader                                  | `projects/custom_utils/cfg.py`                         |
| Domain Logic   | Trainer / Validator / Predictor                | `projects/src/models/yolo/detect/{train,val,predict}/` |
| Domain Model   | Model / head / loss / assigner                 | `projects/src/nn/`, `projects/src/utils/`              |
| Data           | Dataset (파일 읽기·변환)                             | `projects/src/data/dataset/`                           |
| Persistence    | **File System** (`runs/`, `configs/releases/`) | —                                                      |
| 외부 시스템         | Novatek SDK (셸 wrapper)                        | `novatek/convert/*.sh`                                 |


### 3.2 쓰이는 디자인 패턴


| 패턴                    | 어디에                                                                                                     | 형태                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Template Method**   | 전 계열                                                                                                    | upstream이 흐름(`_do_train`, `__call__`)을 갖고 우리는 `build_dataset`/`get_model`/`preprocess_batch`/`get_validator` 같은 확장점만 override |
| **Registry**          | `_MODEL_REGISTRY` (`train_custom_trainer.py:30`), `TARGET_REGISTRY` (`quant_utils.py`)                  | 이름 → 클래스·팩토리 dict                                                                                                             |
| **Factory**           | `get_custom_model()` (`nn/tasks/__init__.py:9`), `get_custom_dataset()` (`data/dataset/__init__.py:10`) | 이름으로 클래스 선택                                                                                                                   |
| **Strategy**          | `dair_method_mode` (`or`/`and`/`none`), `assigner` (`tal`/`simota`), quantization target                | 설정값이 알고리즘 부품을 교체                                                                                                              |
| **Adapter / Wrapper** | `C2fTrace`, `ConcatTrace` (`nn/modules/`), `DynamicCropWrapper` (`data/dynamic_crop.py`)                | 원본을 감싸 특정 환경(FX trace, 추론)에 맞춤                                                                                                |
| **Observer (hook)**   | `SymmetricDetectionLoss._register_hook`, `DistillationLoss.register_hook`                               | 중간 feature를 forward hook으로 수집                                                                                                 |
| **Composite**         | `Select(Compose)` (`augment.py:28`)                                                                     | augmentation 목록에서 `num_choice` 개 선택                                                                                           |
| **Null Object**       | `_NoOpEMA` (`base_qat.py:16`)                                                                           | QAT 중 EMA를 무력화하는 빈 구현                                                                                                         |
| **Memento**           | `PipelineContext` save/load (`nova_qat_pipeline.py:25`)                                                 | stage 중단 지점 재개                                                                                                                |


**의도적으로 안 쓰는 것** — DI 컨테이너 없음, ORM 없음, 커스텀 예외 계층 없음(§3.3),
플러그인 등록 데코레이터 없음.

### 3.3 예외 처리 규격

**커스텀 예외 계층이 없다.** 표준 예외만 쓰고, CLAUDE.md가 이를 명시적 결정으로 못 박고
있다 (decision record 없이 base exception 클래스를 만들지 말 것).

실측 발생 건수 (`projects/`, `tools/`, `novatek/`, `quant_tools/`):


| 예외                                          | 건수    | 언제                                                                                                       |
| ------------------------------------------- | ----- | -------------------------------------------------------------------------------------------------------- |
| `ValueError`                                | 75    | 잘못된 config·인자 값. `load_exp_yaml` 순환 참조, `load_pipeline_yaml` 의 `model_name` 누락, `get_qconfig` 미등록 target |
| `FileNotFoundError`                         | 25    | weight·dataset·SDK 산출물 없음                                                                                |
| `RuntimeError`                              | 20    | 선택 의존성 미설치, 변환 중간 산출물 검증 실패                                                                              |
| `SystemExit`                                | 17    | CLI 조기 종료                                                                                                |
| `NotImplementedError`                       | 7     | 미지원 variant                                                                                              |
| `KeyError` · `TypeError` · `AssertionError` | **0** | 직접 raise 하지 않는다                                                                                          |


**bare `except:` 3곳** (`metrics/evaluator.py` ×2, `metrics/custom_coco_eval.py`)은
알려진 예외이고 따라야 할 패턴이 아니다.

**NPU wrapper의 실패는 절대 삼키지 않는다** — 중간 stage를 조용히 건너뛰는 것이
D-016이 반복 지적하는 실패 양식이다.

로그는 두 갈래를 쓴다 — `projects/src/` 안에서는 ultralytics `LOGGER`,
`tools/` CLI에서는 `print()`.

### 3.4 반환 규격 (웹의 Response Format 대응)

HTTP 응답 스키마는 없지만 **계층 간 반환 계약**이 있고, 이것이 깨지면 조용히 오작동한다.


| 경계                             | 반환 계약                                             |
| ------------------------------ | ------------------------------------------------- |
| Dataset `__getitem__`          | `label` dict — §4.3                               |
| `collate_fn`                   | `batch` dict — §4.4                               |
| `Trainer.preprocess_batch`     | 같은 `batch` dict (`img` 는 `[0,255]` float32)       |
| head `forward` (학습)            | `dict`: `{boxes, scores, feats}` (+ pose는 `kpts`) |
| head `forward` (export)        | **tuple, 순서가 모드마다 다르다** — §8.2, §8.3              |
| `loss.__call__`                | `(total_loss, loss_items)` — upstream 계약          |
| `dair_loss()`                  | **3-tuple** `(원본 loss, aug loss, 차이 벌점)` — §9.1   |
| Validator                      | ultralytics metrics dict                          |
| stage 함수 (`nova_qat_pipeline`) | `PipelineContext` (누적 갱신)                         |


---

## 4. Data Flow — 스키마와 전체 경로

### 4.1 전체 경로 한 장

```
[디스크] dataset                 $DATA_ROOT/<Dataset>/<Camera>/{Train,Validation}/{images,labels}
   │                             컨테이너에서는 /dataset/... 로 보인다
   │
   ├─ configs/datasets/*.yaml    path + names(클래스 목록)만 정의
   │
[CLI] tools/train_custom_trainer.py  -w <exp.yaml> -d <gpu>
   │  ① get_my_cfg()            exp.yaml → IterableSimpleNamespace (_base_ 상속 해석)
   │  ② apply_cfg_options()      --cfg_options 를 cfg.overrides 안으로 병합
   │  ③ derive_run_dir()         출력 위치를 config 경로에서 계산
   │  ④ _MODEL_REGISTRY          model_name → (Model 클래스, Trainer 클래스)
   ▼
[Service] Trainer
   │  overrides.update(cfg.overrides)     설정 파일이 kwargs 를 다시 덮는다
   │  save_run_config() → save_dir/exp.yaml 실행 기록 (D-021, 옛 mycfg.yaml)
   ▼
[Repository] Dataset  get_custom_dataset(dataset_name)
   │  __getitem__ → label dict (§4.3)
   │  augmentation 적용 (BGR 유지)
   │  collate_fn → batch dict (§4.4)
   ▼
[Service] Trainer.preprocess_batch
   │  .float() 만 — 255로 나누지 않는다 (D-005/D-012/D-013)
   │  exp_params.normalize 있으면 mean/std 적용
   ▼
[Domain] Model → head → loss(+assigner)
   ▼
[Persistence] runs/<task>/<PRODUCT>/<Camera>/<Version>/
              weights/{best.pt,last.pt,input_scale.json} · exp.yaml · training_log.log
   ▼
export → *.onnx → runs/npu/novatek/<dir>/nvtai_tool/output/<PATTERN>/nvt_model.bin
   ▼
승격: configs/releases/<PRODUCT>/<Camera>/<Version>/{exp.yaml, MANIFEST.yaml}
```

### 4.2 Config 스키마 (`exp.yaml`)

release exp.yaml 10개 실측 기준 최상위 키.


| 키              | 필수    | 내용                                                                                                    |
| -------------- | ----- | ----------------------------------------------------------------------------------------------------- |
| `_meta_`       | 10/10 | 출처 기록 — `product` `camera` `version` `release` `source` `verified_against` `deviations`               |
| `model_name`   | 10/10 | §6.1 registry 키. **학습될 아키텍처를 결정**                                                                     |
| `weight_path`  | 10/10 | 시작 가중치 (`.pt` 또는 아키텍처 `.yaml`)                                                                        |
| `exp_params`   | 10/10 | 우리가 추가한 파라미터 — `rect_mode` `fitness_mode` `interp` `lr_scheduler` `num_plots` `data_args` `normalize` |
| `overrides`    | 10/10 | **ultralytics 표준 인자** — `imgsz` `epochs` `batch` `optimizer` `data` `workers` `bgr` `lrf` `degrees` … |
| `dataset_name` | 7/10  | §6.1의 `custom` 계열에서 필수. Dataset 클래스 선택                                                                |
| `dair_params`  | 3/10  | `dair_method_mode` `num_choice` 등 DAIR 전용                                                             |
| `cls_weights`  | 1/10  | 클래스별 loss 가중치                                                                                         |


`**exp_params` 와 `overrides` 의 구분이 중요하다.** `overrides` 는 그대로 ultralytics에
넘어가고, `exp_params` 는 우리 코드만 읽는다. 새 파라미터는 `exp_params` 에 넣어야
upstream이 unknown key로 거부하지 않는다.

### 4.3 샘플 단위 스키마 — `label` dict

`DynamicCropYOLODataset.get_image_and_label` (`data/dataset/dynamic_crop.py:81`) 이 만든다.


| 키               | 타입                           | 만들어지는 곳                 |
| --------------- | ---------------------------- | ----------------------- |
| `img`           | `np.ndarray` (H,W,3) **BGR** | `load_image()`          |
| `ori_shape`     | `(h, w)`                     | `load_image()`          |
| `resized_shape` | `(h, w)`                     | `load_image()`          |
| `ratio_pad`     | `(h비율, w비율)`                 | `get_image_and_label()` |
| `cls`           | `np.ndarray` (n,1)           | 라벨 파일                   |
| `im_file`       | `str`                        | 라벨 캐시                   |
| `instances`     | `Instances` 객체               | `update_labels_info()`  |


`update_labels_info` (`:94`)가 하는 변환이 핵심이다 — `bboxes`/`segments`/`keypoints`/
`bbox_format`/`normalized` 를 **꺼내서 `Instances` 하나로 합친다.** segment는
`resample_segments` 로 1000점(OBB는 100점)으로 리샘플되고, 없으면
`(0, 1000, 2)` 빈 배열이 들어간다. 이후 augmentation은 `Instances` 를 통해서만 좌표를
건드리므로 박스·세그·keypoint가 함께 변환된다.

### 4.4 배치 단위 스키마 — `batch` dict

`collate_fn` (`:61`)이 만든다. 키별로 합치는 방식이 다르다.


| 키                                                   | 합치는 방식                         | 용도                                |
| --------------------------------------------------- | ------------------------------ | --------------------------------- |
| `img`                                               | `torch.stack` → (B,3,H,W)      | 입력. `**[0,255]` float32**         |
| `cls` `bboxes` `keypoints` `masks` `segments` `obb` | `torch.cat` (배치 축 없이 이어붙임)     | 정답. 개수가 샘플마다 달라 stack 불가          |
| `batch_idx`                                         | `cat` 후 **각 원소에 이미지 인덱스를 더한다** | 이어붙인 정답이 어느 이미지 것인지 표시            |
| `im_file` `ori_shape` `ratio_pad` `imgsz`           | tuple 유지                       | 후처리·시각화                           |
| `aug_img`                                           | `torch.stack`                  | **DAIR 전용** — augmentation 결과 이미지 |
| `aug_cls` `aug_bboxes`                              | `cat`                          | **DAIR 전용** — augmentation 후 정답   |


`aug_*` 세 키의 존재가 DAIR 구조의 뿌리다. 일반 YOLO 배치에는 없고, 이것이 있어야
loss가 원본·augmentation 예측을 동시에 받을 수 있다 (§9.1).

`batch_idx` 보정 루프(`:75-77`)는 upstream `build_targets()` 계약을 맞추기 위한 것으로,
여기를 잘못 고치면 정답이 다른 이미지에 붙는다.

### 4.5 산출물과 영속화


| 산출물               | 위치                                      | git 추적                 |
| ----------------- | --------------------------------------- | ---------------------- |
| 학습 결과             | `runs/<task>/<P>/<C>/<V>/`              | ✗ 전체 무시                |
| 실행 설정 기록          | `runs/.../exp.yaml`                     | ✗                      |
| 입력 범위 sidecar     | `runs/**/weights/input_scale.json`      | ✗                      |
| release 정본        | `configs/releases/<P>/<C>/<V>/exp.yaml` | ✓                      |
| release 기록        | `.../MANIFEST.yaml`                     | ✓                      |
| 가중치·ONNX·NPU 바이너리 | `.../{analysis,network,weight}/`        | ✗ (`.gitignore:42-47`) |
| GA 탐색 결과          | `runs/.../tune_results.csv`             | ✗                      |


`configs/releases/` 실측 42 MB이고 **8개 카메라 중 3개만** 실제 산출물을 갖고 있다
(`Front_Truck` 18M · `Side_Truck` 12M · `Corner_Truck` 12M). 나머지는 텍스트뿐이라
`MANIFEST.nas_source` (NAS_2)에서 받아야 한다.

### 4.6 데이터베이스 — MongoDB 경로가 있으나 동작하지 않는다

**유일한 DB 경로다.** `CustomTuner` (`projects/src/engine/custom_tuner.py:18`)가
upstream `Tuner` 를 상속하고, GA 하이퍼파라미터 탐색 결과를 MongoDB에 저장하는 코드를
호출한다.

```python
if self.mongodb:                                        # custom_tuner.py:64,116
    self._save_to_mongodb(fitness, mutated_hyp, metrics, i + 1)
    self._sync_mongodb_to_csv()
    total_mongo_iterations = self.collection.count_documents({})
```

`_save_to_mongodb` / `_sync_mongodb_to_csv` / `_init_mongodb` 는 **우리 코드가 아니라
upstream `ultralytics/engine/tuner.py` (217, 239, 182행)에 있다.** 우리는 호출만 한다.

문서 스키마 (upstream `tuner.py:227-235`):


| 필드                | 타입                                                 |
| ----------------- | -------------------------------------------------- |
| `fitness`         | float — `fitness` 내림차순 index가 걸린다 (`tuner.py:200`) |
| `hyperparameters` | dict — tensor는 `.item()` 으로 변환                     |
| `metrics`         | dict — mAP·precision·recall·loss 전체                |
| `timestamp`       | datetime                                           |
| `iteration`       | int                                                |


**그런데 `pymongo` 가 설치돼 있지 않고 `pyproject.toml` 에도 없다.** 컨테이너에서
`import pymongo` 는 `ModuleNotFoundError` 다. 기본값이 `self.mongodb = None`
(`tuner.py:133`)이고 `mongodb_uri` 인자를 줄 때만 연결을 시도하므로, **현재는 도달할 수
없는 경로**다. 분산 GA 탐색을 실제로 쓰려면 `pymongo` 를 의존성에 추가해야 한다.

`_save_to_mongodb` 는 실패를 `LOGGER.warning` 으로만 남기고 넘어간다 — 저장이 안 돼도
탐색은 계속된다.

### 4.7 트랜잭션 · 캐시 · 이벤트 발행


| 항목                        | 현황                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **DB 트랜잭션**               | 없음. MongoDB 경로도 `insert_one` 단건이고 트랜잭션·세션을 쓰지 않는다                                                                                         |
| **파일 원자성**                | 보장하지 않는다. 체크포인트 저장은 `torch.save` 직접 호출이고, 중간에 죽으면 깨진 파일이 남는다. `base_qat.py` 의 `_save_qat_model` 만 try/except로 감싸 학습이 죽지 않게 한다 (`ca1cb8d`) |
| **캐시 (Redis 등)**          | 없음. 캐시는 두 종류의 로컬 캐시뿐 — ultralytics 라벨 캐시(`*.cache`)와 `_get_blend_mask` 의 kernel 캐시(`ec2e2eb`)                                             |
| **이벤트 발행 (Kafka/PubSub)** | 없음                                                                                                                                        |
| **비동기 큐 (Celery 등)**      | 없음. 병렬성은 `torch.distributed` DDP와 `DataLoader` worker뿐                                                                                    |
| **stage 재개**              | DB가 아니라 파일로 한다 — `PipelineContext` save/load, `pipeline.yaml` 의 stage status                                                              |


즉 **영속화는 사실상 파일시스템 단일 경로**이고, 동시 실행 시 서로를 덮어쓰지 않는 근거는
`derive_run_dir` 이 run 디렉토리를 config 경로로 유일하게 정한다는 것뿐이다 (§5.3).

---

## 5. Config 시스템 — `projects/custom_utils/cfg.py` (336줄)

공개 함수 네 개가 전부다.

### 5.1 `load_exp_yaml` — `_base_` 상속

`_base_` 값을 **그 파일 기준 상대경로**로 보고 재귀 로드해 deep merge한다(자식 승).
`_base_` 키는 소비되어 결과에 남지 않는다. `_seen` 집합으로 순환 참조를 잡아
`ValueError`. `_base_` 없는 파일은 그대로 읽히므로 기존 config가 그대로 동작한다.

### 5.2 `load_pipeline_yaml` — 여러 stage를 한 파일에

`shared` + `stage_N`(`^stage_\d+$`) 구조를 요구한다. 각 stage는 `shared` 에 자기 블록을
덮어 만들어지고, **병합 후 `model_name` 이 없으면 `ValueError`**. 허용되지 않은 최상위
키도 거부한다.

이 loader를 쓰는 것은 `**tools/run_finetune_pipeline_v2.py` 하나뿐이다**
(`v2.py:41,677`). §13.4의 `nova_qat_pipeline.py` 는 자체 `load_pipeline_config` →
`PipelineContext` 를 쓰는 **별개 경로**다.

### 5.3 `derive_run_dir` — 출력 위치를 config 경로에서 계산

```
configs/{experiments,releases}/<PRODUCT>/<Camera>/<Version>/exp.yaml
  → runs/<task>/<PRODUCT>/<Camera>/<Version>/
```

**gate 3개** 중 하나라도 어긋나면 **no-op** 이 되어 upstream 기본 경로
(`runs/detect/train`, `train2`, …)에 쌓인다.

1. 파일 이름이 정확히 `exp.yaml`
2. 경로가 `configs/experiments` 또는 `configs/releases` 밑 (`cfg.py:112`)
3. 그 밑으로 경로 조각이 4개 이상 (`<PRODUCT>/<Camera>/<Version>/<파일>`)

gate 1이 필요한 이유가 코드 주석에 있다 — legacy 트리는 더 깊게 중첩되므로 조각 수만
세면 `legacy/finetune/fp_bg/org.yaml` 을 PRODUCT=`legacy`, Camera=`finetune` 으로 잘못
읽는다. gate 3이 `configs/experiments/legacy/` 를 자연히 걸러낸다.

이 함수가 흡수하는 함정이 하나 더 있다 — **Trainer 두 계열이 서로 다른 키를 읽는다.**
`model_name` 이 `symmetric` 이면 최상위 `save_dir` 를, 나머지는
`overrides.project` + `overrides.name` 을 채운다. 이미 값이 있으면 건드리지 않는다.

### 5.4 `apply_cfg_options` — CLI가 config 파일을 이기게 한다

Trainer 생성 시 `overrides.update(self.custom_cfg.overrides)` (`custom.py:33`) 로
**config 파일 값이 kwargs 위에 다시 덮인다.** 그래서 `--cfg_options` 를 kwargs로만 넘기면
파일이 정의한 키는 전부 파일 값으로 되돌아간다.

`apply_cfg_options` 가 CLI 옵션을 `cfg.overrides` **안으로** 먼저 넣어야 비로소 CLI가
이긴다. 덕분에 `runs/.../exp.yaml` 이 실제 실행 설정의 정확한 기록이 된다 (D-021 이 이름을
`mycfg.yaml` 에서 바꿨다). 기계가 파생한 값은 `_resolved_` 로 분리되므로 이 파일을 다음 버전
디렉토리로 복사해 그대로 재실행할 수 있다.

`DictAction` 은 `키=값1,값2` / `키=[값1,값2]` / `키=[(값1,값2),(값3,값4)]` 중첩 괄호까지
파싱하고 int/float/bool을 자동 변환한다. 이 parser만 약 100줄이다.

---

## 6. 모델 선택

### 6.1 registry가 세 곳에 흩어져 있다

**(가) `tools/train_custom_trainer.py:30-37` — CLI script가 쓰는 registry**


| `model_name` | Model             | Trainer                       | config 사용 수 |
| ------------ | ----------------- | ----------------------------- | ----------- |
| `custom`     | `CustomYOLO`      | `CustomDetectionTrainer`      | 23          |
| `consi`      | `ConsistencyYOLO` | `ConsistencyDetectionTrainer` | 24          |
| `dair`       | `DairYOLO`        | `DairDetectionTrainer`        | 13          |
| `dms_pose`   | `DMSPoseYOLO`     | `DMSPoseTrainer`              | 5           |
| `qat`        | `QATYOLO`         | `QATDetectionTrainer`         | 6           |
| `fx_qat`     | `FXQATYOLO`       | `FXQATDetectionTrainer`       | 5           |
| `symmetric`  | `SymmetricYOLO`   | `SymmetricTrainer`            | 3           |


조회가 `.get(키, (CustomYOLO, CustomDetectionTrainer))` 다. **이름을 틀리면 에러 없이
다른 모델이 학습된다.** 이 저장소에서 가장 눈치채기 어려운 실패다.

**(나) `projects/src/engine/model.py` — `task_map` registry**

`Model` 상속 클래스 8개(`ConsistencyYOLO` `CustomYOLO` `SymmetricYOLO` `DairYOLO`
`DraftKDYOLO` `QATYOLO` `DMSPoseYOLO` `FXQATYOLO`)가 각자 `task_map` 으로
model·trainer·validator·predictor 네 개를 묶는다. `QATYOLO`·`FXQATYOLO` 는 `CustomYOLO`
를 상속하고 model만 `CustomDetectionModel` 을 공유한다.

**(다) `projects/src/nn/tasks/__init__.py:9` `get_custom_model`** — 세 번째 registry.
`custom`/`qat`/`fx_qat` → `CustomDetectionModel`, `KD` → `DraftKDModel`, 나머지는 각자.
**미등록 이름에는 `None` 을 반환**하고, `custom.py:100-101` 이 `None` 이면 upstream
`DetectionModel` 로 fallback한다.

> 같은 `model_name` 이 세 곳에서 해석되고 **세 곳의 키 집합이 일치하지 않는다.**
> `KD` 는 (다)에만 있다. 세 곳 모두 미등록 이름을 조용히 fallback시킨다.

### 6.2 Trainer 상속 — 14개

```
upstream DetectionTrainer
├── CustomDetectionTrainer          custom.py:30       ← 대부분의 부모
│   ├── ConsistencyDetectionTrainer consistency.py:36
│   ├── DMSPoseTrainer              dms_pose.py:8
│   └── BaseQATTrainer              base_qat.py:36     (D-010 보류)
│       ├── QATDetectionTrainer     qat.py:23
│       ├── FXQATDetectionTrainer   fx_qat.py:34
│       └── NMQTQATTrainer          nmqt_qat.py:16
├── DairDetectionTrainer            dair.py:36         ← Custom 을 거치지 않는다
├── SymmetricTrainer                symmetric.py:61    ← Custom 을 거치지 않는다
└── DraftKDTrainer                  kd/train/draft.py:58

upstream BaseTrainer
└── KnowledgeDistllationDetectionTrainer  detect/train/kd_draft.py:16
                                          (클래스명 오타가 코드에 그대로 있다)

upstream ClassificationTrainer
└── OnTheFlyClassifyTrainer         ncnvox/_base_trainer.py:24  (D-015)
    ├── WakewordClassifyTrainer     19줄
    └── CommandClassifyTrainer      19줄
```

`**dair` 와 `symmetric` 이 `CustomDetectionTrainer` 를 거치지 않는다**는 점이 실무에서
중요하다. `custom.py` 의 normalize·multi_scale·BN freeze 로직이 그 둘에 전파되지 않고
각자 따로 구현한다 (`dair.py:127`).

`OnTheFlyClassifyTrainer` 는 D-015가 KWS/CMD trainer의 중복 약 85줄을 뽑아낸 결과다.
자식은 `_loader_builder = staticmethod(build_*_loaders)` 한 줄만 지정하면 부모가 알아서
호출한다 — 그래서 두 trainer가 19줄이다.

---

## 7. 모듈 지도 — 무엇이 어디에 있고 무엇을 재사용할 수 있나

### 7.1 패키지별 역할


| 패키지                          | 역할                                                           | 손대면 파급되는 곳           |
| ---------------------------- | ------------------------------------------------------------ | -------------------- |
| `projects/custom_utils/`     | config 로딩, 좌표 변환, 시각화. **upstream 의존 없는 순수 유틸**              | 전 계열 (§16.1)         |
| `projects/src/nn/modules/`   | head·block·conv 커스텀 (`SymmetricDetect` `DMSPose` `C2fTrace`) | loss, export, NPU 변환 |
| `projects/src/nn/tasks/`     | Model 클래스 + `parse_model`                                    | Trainer 전체           |
| `projects/src/utils/`        | loss, assigner(`tal`), metric, FX 호환                         | Trainer, Validator   |
| `projects/src/data/`         | Dataset + augmentation                                       | Trainer, GA 탐색       |
| `projects/src/models/yolo/`  | Trainer / Validator / Predictor                              | CLI script           |
| `projects/src/engine/`       | `Model` wrapper, `CustomTuner`, `NMQTExporter`               | CLI script           |
| `projects/src/quantization/` | qconfig, QAT 준비·변환                                           | QAT trainer, NPU 전략  |
| `projects/src/ncnvox/`       | 오디오 전용 (dataset·trainer·mel·모델)                              | 오디오 CLI만             |
| `projects/metrics/`          | 평가 지표, COCO eval, confusion matrix                           | Validator, 분석 script |
| `projects/model_overrides/`  | SimOTA assigner                                              | loss (선택적)           |
| `projects/optimizers/`       | Muon, Cosine warmup restart                                  | Trainer optimizer    |


### 7.2 바로 재사용할 수 있는 helper

**새 코드를 쓰기 전에 여기부터 본다.**


| 함수/클래스                                                   | 위치                                                | 하는 일                               |
| -------------------------------------------------------- | ------------------------------------------------- | ---------------------------------- |
| `deep_merge(base, override)`                             | `custom_utils/cfg.py:13`                          | dict 재귀 병합 (자식 승)                  |
| `load_exp_yaml(path)`                                    | `cfg.py:28`                                       | `_base_` 상속 포함 config 로드           |
| `get_my_cfg(cfg)`                                        | `cfg.py:207`                                      | 경로·dict·namespace 무엇이든 config로 정규화 |
| `save_my_cfg(cfg, dst)`                                  | `cfg.py:217`                                      | rank 0에서만 YAML 저장                  |
| `DictAction`                                             | `cfg.py:223`                                      | argparse `키=값` (중첩 리스트·튜플 지원)      |
| `derive_run_dir(cfg, path)`                              | `cfg.py:115`                                      | 출력 경로 계산                           |
| `cxywhn2xywh` `xyxy2xywh` `xywh2xywh` `to_xywh`          | `custom_utils/drawing.py:30-58`                   | 박스 좌표 변환                           |
| `draw_bboxes` `draw_bbox`                                | `drawing.py:4-9`                                  | 박스 그리기                             |
| `custom_plot_images`                                     | `custom_utils/plotting.py:13`                     | 배치 시각화 (라벨 포함)                     |
| `name2clsnum` `clsnum2name`                              | `custom_utils/data_convert.py:14-41`              | 클래스 이름 ↔ 번호                        |
| `read_txt` `write_txt` `clip_bbox` `normalize`           | `data_convert.py:46-99`                           | 라벨 파일 입출력                          |
| `rewrite_label` `collect_layout`                         | `data_convert.py:159-203`                         | 데이터셋 레이아웃 재구성                      |
| `SimilarImagesRemover`                                   | `custom_utils/similar_images_remover.py:13`       | 중복 프레임 제거                          |
| `compare_frames_change_detection`                        | `custom_utils/imaging.py:33`                      | 프레임 변화량 비교                         |
| `ap_per_class` `CustomDetMetrics`                        | `src/utils/metric.py`                             | AP 계산 — **학습 중 검증 전용** (D-020)      |
| `basic_preproc` `basic_postproc_v8` `basic_inference_v8` | `metrics/utils.py`                                | ONNX 추론 전·후처리                      |
| `merge_board_json` `apply_cat_map` `filter_gt_roi`       | `metrics/utils.py`                                | 보드 JSON 정규화·클래스 매핑·GT ROI 필터 (D-020) |
| `crop_inference_v8` `letterbox_inference_v8`             | `metrics/inference_funcs.py`                      | 화각(FOV) 통제 추론 (D-020)                |
| `yolov8_bgr255_preproc` `yolov8_std_preproc`             | `metrics/inference_funcs.py`                      | 전처리 규약 두 종 ([0,255] BGR / RGB ÷255)  |
| `flatten_thresholds` `judge_targets` `build_evaluation`  | `metrics/report.py`                               | 3단계 임계값 평면화·PASS 판정·`evaluation.json` |
| `Evaluator`                                              | `metrics/evaluator.py:64`                         | 통합 평가 — **오프라인·보드 정본** (D-020)     |
| `is_fx_tracing(x)`                                       | `src/utils/trace.py:4`                            | FX trace 중인지 판정                    |
| `replace_modules_for_fx_tracing` `prepare_model_for_fx`  | `src/utils/fx_compat.py:15-57`                    | FX 추적 가능하게 모듈 교체                   |
| `get_qconfig(target)` `register_target(name, fn)`        | `src/quantization/quant_utils.py:524-551`         | quantization 설정 조회·등록              |
| `compute_similarity(a, b)`                               | `quant_utils.py:561`                              | 변환 전후 cosine 비교                    |
| `Select(Compose)`                                        | `src/data/augment.py:28`                          | augmentation 중 n개 선택               |
| `_ROIBlurBase` `_BboxPasteBase`                          | `augment.py:47`, `augment_complexity_paste.py:17` | ROI 처리·붙이기 공통 부모                   |
| `get_landmark_weights(num_kpts, part_w)`                 | `src/utils/loss.py:1061`                          | keypoint 부위별 가중치                   |


**새 augmentation을 만들 때는 `_ROIBlurBase` 또는 `_BboxPasteBase` 를 상속한다** —
`0d0107e`, `76bf5b7` 가 중복을 뽑아낸 결과이고, 여기에 붙이면 blend mask 캐시와
IoA 필터를 그대로 쓴다.

---

## 8. head — `projects/src/nn/modules/head.py` (230줄)

### 8.1 `SymmetriceDFL` — 좌우 대칭 DFL

upstream DFL의 projection은 `arange(0, reg_max)` 로 0 이상만 표현한다. 이것은 빼기를 넣어
**음수 거리까지 표현**한다.

```python
offset = c1 // 4          # reg_max=16 → offset=4 → 표현 범위 -4 ~ +12
x = torch.arange(c1, dtype=torch.float) - offset
```

`NegativeDFL`(`block.py:29`)은 `offset = reg_max // 2` 로 다른 분할(-8 ~ +8)을 쓴다.

**문제는 loss가 이 offset을 head 종류로부터 되짚어 알아낸다는 점이다** (`loss.py:102-107`):

```python
if   isinstance(m.dfl, NegativeDFL):    self.offset = m.reg_max // 2
elif isinstance(m.dfl, SymmetriceDFL):  self.offset = m.reg_max // 4
else:                                   self.offset = 0
```

D-003이 기록한 사고가 정확히 여기다 — `NegativeDFL` 검사만 있던 시절 Symmetric 조합에서
offset이 0이 되고 박스가 뒤집혀 **mAP=0** 이 됐다. head와 loss 사이의 문서화되지 않은
계약이며, **새 DFL 변형을 추가하면 `loss.py` 의 이 분기도 반드시 함께 고쳐야 한다.**

`forward` 가 tuple unpacking 대신 `x.shape[0]`/`x.shape[2]` 인덱스 접근을 쓰는 이유는
FX trace 시 `Proxy.__iter__` 오류를 피하기 위한 것으로, 코드 주석에 적혀 있다.

### 8.2 `SymmetricDetect` — export 분기 3갈래

`forward` 가 세 갈래로 갈린다.

1. `is_fx_tracing(x)` → `_forward_fx`: 반복 대신 `range(self.nl)` 정수 인덱스만 쓴다.
환 `(dbox, scores.sigmoid())`.
2. export + ONNX → `forward_head` 후 `_inference`. `skip_postprocess` 면 후처리를
너뛰고 dbox/scores를 그대로 낸다.
3. 그 외 → upstream `forward`.

클래스 속성 플래그 6개(`export` `format` `skip_postprocess` `skip_dfl` `xyxy` `legacy`)가
export 형태를 제어한다. `__init__` 의 절반은 upstream 본문을 주석으로 보존한 것이다 —
어디를 바꿨는지 대조하려는 의도로 보이나, §18.5의 주석 코드 부채다.

### 8.3 `DMSPose` — P5 한 스케일만 쓴다

70 keypoint pose head인데 **가장 작은 feature map(P5) 하나만 실행한다.**

```
cv4 ModuleList 는 3개 스케일을 전부 갖고 있다   ← 체크포인트 호환 목적
그러나 forward 는 pose_head[-1](x[-1]) 만 실행  ← P5 만 학습되고 gradient를 받는다
```

학습·검증 경로에서는 전체 anchor 수에 맞춰 **P3/P4 구간을 0으로 채워** 반환하므로
(`_inference` 227-230행) 뒤쪽 코드는 표준 pose 출력과 같은 shape을 본다.

export 시 **출력 순서가 stage마다 다르다.**


| stage   | `nc` | 출력 순서                                      |
| ------- | ---- | ------------------------------------------ |
| Stage-1 | 1    | `(cls_p5, dfl_p5, kpts_p5_visibility포함)`   |
| Stage-2 | 2 이상 | `(kpts_p5_visibility제외, dfl_all, cls_all)` |


**박스는 raw DFL 거리이고 `dist2bbox` 디코딩을 하지 않는다. keypoint xy도 raw다.**
legacy ONNX 구조와 맞추기 위한 것으로 docstring에 명시돼 있다(커밋 `d1eca1d`).
Stage-1만 visibility에 sigmoid를 적용하고, Stage-2는 visibility 채널을 잘라낸다.

---

## 9. loss — `projects/src/utils/loss.py` (1377줄)

클래스 15개가 네 갈래로 나뉜다.

### 9.1 DAIR — augmentation 일관성 (저장소 이름의 근거)

`v8DairDetectionLoss.__call__(pred_orgs, pred_augs, batch)` 가 **원본 예측과
augmentation 예측을 동시에** 받는다. `pred_augs` 가 없으면 upstream loss로 fallback한다.
이 입력이 가능한 이유가 §4.4의 `aug_img`/`aug_cls`/`aug_bboxes` 키다.

```python
def dair_loss(self, loss, loss_norm, loss_aug, loss_norm_aug):
    return (self.gamma * (loss.sum() / loss_norm),                  # 원본 loss
            (1.0 - self.gamma) * (loss_aug.sum() / loss_norm_aug),  # augmentation loss
            self._lambda * (loss - loss_aug).pow(2).mean())         # 둘의 차이 벌점
```

기본값은 `gamma=1.0`, `_lambda=0.72`. `**gamma=1.0` 이면 두 번째 항이 0이 되어
"원본 loss + 차이 벌점" 만 남는다** — augmentation 자체의 loss는 학습에 들어가지 않는다.

주석으로 남은 이전 버전은 `(loss.pow(0.5) - loss_aug.pow(0.5)).pow(2)` 였다.

`dair_method_mode` 가 assigner를 고른다. **config에서 실제로 쓰이는 값**:


| 값      | 사용 수  | 고르는 assigner                   |
| ------ | ----- | ------------------------------ |
| `or`   | 77    | `DairTaskAlignedAssigner`      |
| `and`  | 12    | `DairTaskAlignedAssigner`      |
| `none` | 7     | upstream `TaskAlignedAssigner` |
| `prev` | **0** | `DairTaskAlignedAssigner`      |


`prev` 는 어떤 config도 쓰지 않는다 — 크래시 수정 커밋(`c6829af`)만 남은 죽은 경로다.
코드가 `'org'`/`'aug'` 도 언급하지만 실제 분기 조건은 `in ['or','and','prev']` 이므로
그 둘은 upstream assigner로 떨어진다. `assigner` 키로 `tal` / `simota` 를 고른다.

### 9.2 Symmetric

`bbox2dist_symmetric` · `SymmetricDFLoss` · `SymmetricBboxLoss` ·
`SymmetricDetectionLoss` 네 개. `SymmetricDetectionLoss` 는 DAIR와 같은
`dair_loss_`/`assigner_preprocess` 인터페이스를 갖고, hook을 등록·해제해
(`_register_hook`/`_remove_hook`) 중간 feature를 꺼낸다. 커밋 `1b7b772` 가
`super().__init__()` 호출로 상위 버전 호환을 고친 지점이다.

### 9.3 Knowledge Distillation

`CWDLoss`(channel-wise) · `MGDLoss`(masked generative) · `FeatureLoss` ·
`DistillationLoss` · `GramLoss`. `DistillationLoss` 가 `_find_layers` 로 teacher·student
레이어를 짝지어 hook한다. `E2ELossConsistency` / `SymmetricE2ELossConsistency` 는
KL/JS divergence(`compute_kl_loss`, `compute_js_loss`)를 쓰는 별개 계열이다.

### 9.4 DMSPoseLoss

`get_landmark_weights(num_kpts=70, part_w=...)` 가 **얼굴 부위별 keypoint 가중치**를
YAML로 설정 가능하게 한다(커밋 `30939ef`). `relational_group_loss_coarse` 는 keypoint
그룹 중심 사이의 관계를 loss에 넣는다.

`DMSPoseLoss.loss` (1207-1334행, 약 128줄)는 **D-009로 50줄 기준 면제를 받은 유일한
함수**다. 쪼개면 helper 인자로 넘길 상태가 절약분보다 많다는 것이 근거로 기록돼 있다.

---

## 10. assigner — `projects/src/utils/tal.py` (634줄)

두 클래스 모두 upstream `TaskAlignedAssigner` 를 상속한다.

`DairTaskAlignedAssigner.forward` 는 **원본·augmentation 예측을 둘 다** 받아
`get_merged_pos_mask` → `merge_forward` 로 양쪽 positive mask를 합친다
(`or` 는 합집합, `and` 는 교집합에 해당).

도메인이 드러나는 기능이 두 개 있다.

- `expand_traffic_right_bboxes_ratio(gt_bboxes, last_class_ind, expand_ratio=1.5)` 와
`..._static(..., val)` — **마지막 클래스(신호등)의 GT 박스만 키운다.** 작은 물체의
후보를 늘리려는 조치로, `Dashcam/Front_traffic_light` 가 신호등을 포함한 6-class
데이터셋인 것과 대응한다.
- `select_candidates_in_gts(..., offset=0)` 와 `SymmetricTaskAlignedAssigner` 쪽의
`(..., radius=4)` — §8.1의 DFL offset과 후보 반경을 assigner까지 전파한다.

`SymmetricTaskAlignedAssigner` 는 진입점이 네 개
(`forward`/`_forward`/`dair_forward`/`merge_forward`)이고 `tal_topk2` 로 두 번째 topk를
따로 줄 수 있다. `replace_over_threshold` 계열 static method가 임계 초과값을 처리한다.

---

## 11. data 와 augmentation

### 11.1 입력 범위는 `[0,255]` float32 (D-005/D-012/D-013)

```python
batch["img"] = batch["img"].float()   # / 255  ← 주석 처리돼 있다 (custom.py:75)
```

**255로 나누지 않는다.** 전체가
`cv2.imread(BGR) → augmentation(BGR) → Format(bgr=파라미터) → .float() → model` 로
고정돼 있다. `[0,1]` 은 명시적 옵션 뒤에서만 쓴다.

D-013이 D-012를 정정한 이력이 중요하다 — "detection trainer 다수가 `[0,1]` 로 학습한다"는
D-012의 전제가 코드 정독 결과 틀렸고, 실제 손댈 곳은 KWS `wakeword_dataset.py` 하나였다.
이미 학습된 가중치의 입력 범위는 `runs/**/weights/input_scale.json` sidecar로 남기고
`tools/infer.py` 가 자동 인식한다.

`exp_params.normalize` 가 있으면 mean/std를 `scale` 배해서 적용한다 (`custom.py:56-61`) —
`[0,255]` 범위에 맞춘 mean/std라는 뜻이다.

### 11.2 Dataset


| 클래스                      | 파일                                        | 역할                               |
| ------------------------ | ----------------------------------------- | -------------------------------- |
| `DynamicCropYOLODataset` | `data/dataset/dynamic_crop.py` (178)      | 학습 중 동적 crop. §4.3·§4.4 스키마를 만든다 |
| `UnlabeledYOLODataset`   | `data/dataset/unlabeled_dataset.py` (330) | KD pseudo label                  |


`get_custom_dataset(dataset_name='dycrop')` 가 factory다. `CustomDetectionTrainer`
계열에서 `dataset_name` 이 필수인 이유다.

> **이름이 같은 파일 두 개 주의** — `dynamic_crop.py` 가 두 곳에 있고 내용이 다르다.
> `data/dataset/dynamic_crop.py` → `DynamicCropYOLODataset` (학습),
> `data/dynamic_crop.py` (252줄) → `DynamicCropWrapper` (추론, predictor 3개가 쓴다).
> 중복이 아니라 다른 것이다.

### 11.3 augmentation — 실질 자산

`data/augment.py` (1289줄) + 별도 모듈 3개. 파이프라인 builder
`v8_transforms_custom` / `v8_transforms_dycrop` 와 카메라별 preset
`build_{front,rear,side}_augmentations` / `build_strong_aug`.


| augmentation                                    | 무엇을 모사하는가               |
| ----------------------------------------------- | ----------------------- |
| `SunFlareAug` / `TensorSunFlareAug`             | 역광·렌즈 플레어               |
| `SpeakerOverlayAug` / `TensorSpeakerOverlayAug` | 대시캠 스피커 그릴 가림           |
| `MaskLikeOcclusion`                             | 마스크 착용 (DMS)            |
| `SunglassesCircleOcclusion`                     | 선글라스 (DMS)              |
| `ROIMotionBlur` / `ROIPaste`                    | ROI motion blur·붙이기     |
| `CopyPasteTL`                                   | 신호등 copy-paste          |
| `MosaicShape`                                   | 종횡비를 지키는 mosaic         |
| `CocoSegOverlayAug`                             | COCO segment crop 겹치기   |
| `ComplexityBboxPasteAug`                        | Laplacian 분산이 큰 영역에 붙이기 |
| `DarkRegionBboxPasteAug`                        | 어두운 구조 영역에 붙이기          |


`Select(Compose)` 는 후보 중 `num_choice` 개를 뽑는 wrapper로,
`CustomDetectionTrainer` 생성자가 torchvision 변환 3종을 이것으로 감싼다.

---

## 12. audio — 별개 계열

`projects/src/ncnvox/` 파일 11개. wakeword(KWS 2종)와 command(CMD 8종) 두 축이고,
detection 쪽과 코드를 거의 공유하지 않는다.

### 12.1 mel 구현이 두 개인 이유

`mel.py` (211줄)에 `compute_log_mel_librosa` 와 `compute_log_mel_numpy` 가 **둘 다** 있다.
librosa 쪽이 기준이고 numpy 쪽은 **온디바이스 C 코드와 결과가 같은지 검증하는 용도**다.
`_periodic_hann_window`, `_librosa_stft_window`, `_mel_weight_librosa`,
`_build_mel_weight_htk` 가 window와 filterbank를 양쪽에서 맞춘다.
`docs/ncnvox_c_code_analysis.md` (699줄)가 C 원본 분석이고
`tests/test_ncnvox_mel.py` 가 동등성을 지킨다.

`log_mel_to_model_input` 출력에 `* 255` 가 들어간 것이 D-013의 KWS 범위 정합이다.

### 12.2 Dataset과 Trainer

`AudioClassificationDataset` → `WakewordDataset` / `CommandDataset`.
`build_wakeword_loaders` / `build_command_loaders` 가 on-the-fly loader를 만들고,
`OnTheFlyClassifyTrainer` 가 `_loader_builder` 로 호출한다.
augmentation은 `wakeword_augment.py` (253줄) — `NoisePool` · `speed_perturb` ·
`time_shift` · `noise_mixer`.

커밋 `3a54420` 이 on-the-fly loader를 upstream `InfiniteDataLoader` 로 교체했다 —
upstream DataLoader 계약을 지켜야 하는 제약이 있다.

### 12.3 femto 최종 채택 (D-016)


| 모델                 | 구조          | 파라미터 | NPU 실측                             |
| ------------------ | ----------- | ---- | ---------------------------------- |
| `yolov8-cmd-femto` | Conv만       | 207K | **288 KB, 0.24 ms, 76 layer** ← 채택 |
| `cmd_crnn`         | Conv + LSTM | 890K | 크기 3.2배, latency 4.8배, 284 layer   |


D-016의 결론은 두 가지다. ① femto 채택, CRNN은 보관만. ② **PyTorch `nn.LSTM` 을
CNN25 NPU에서 쓸 수 있음을 확인** — 컴파일러의 `RewriteLSTMONNX8` 이 24 timestep ×
약 12 연산 = 약 290 layer로 펼치고, 모든 layer cosine ≥ 0.9974, 최종 Softmax 1.0.

LSTM export 제약 4건 — standalone wrap 모델은 `YOLO.export()` 불가(직접
`torch.onnx.export`), `dynamic_axes=None` 정적 shape 강제, export wrapper에
`torch.softmax` 추가 필수, 컴파일러가 출력 이름을 `Softmax_<원본>_Y` 로 바꿈.

한계도 남아 있다 — TTS positive + SC2 negative가 mel 공간에서 선형 분리되어 **두 모델
모두 top1=1.0으로 포화**, 정확도만으로는 우열을 못 가린다.

`cmd_crnn.py` 와 `cmd_yolov8_crnn.py` 가 `InputRescale`/`ReshapeToSequence`/
`LSTMLastStep` 을 각각 따로 정의한다.

---

## 13. quantization — 보류 상태 (D-010)

**코드와 config를 그대로 남긴다. 지우지 않는다.**

### 13.1 세 갈래 QAT


| 갈래       | Trainer                 | 방식                         |
| -------- | ----------------------- | -------------------------- |
| eager    | `QATDetectionTrainer`   | `prepare_qat`              |
| FX graph | `FXQATDetectionTrainer` | `prepare_qat_fx`           |
| NMQT     | `NMQTQATTrainer`        | 외부 `nmqt` 툴킷 (선언되지 않은 의존성) |


`BaseQATTrainer` (224줄)가 공통 2-phase lifecycle을 갖는다 — float warmup →
QAT 전환 시 `_save_ptq_snapshot`, `_check_enable_fake_quant` 콜백,
`enable/disable_fake_quant`·`enable/freeze_observers` 추상 메서드. `_NoOpEMA` 가
QAT 중 EMA를 무력화한다.

FX 갈래의 실패 이력이 길다 — `GraphModule` 이 pickle 불가라 state_dict 저장(`d94967d`),
학습 후 재로드 오류 처리(`47c05a8`), `prepare_qat_fx` 에 `model.model` 이 아니라
`DetectionModel` 을 넘겨야 함(`4607b0d`), trace 후 head 플래그 되돌리기(`fd37910`),
저장 실패가 학습을 죽이지 않게 감싸기(`ca1cb8d`).

### 13.2 FX trace 호환

`utils/fx_compat.py` (80줄)의 `replace_modules_for_fx_tracing` 이
`C2f`→`C2fTrace`, `Concat`→`ConcatTrace` 로 바꿔 끼운다. 세 수정 모두 원인이 같다 —
trace 중 텐서가 `Proxy` 가 되어 반복·길이 조회가 안 된다.

- `C2fTrace` 는 `torch.chunk` unpacking 대신 슬라이스 인덱싱(`329b5e8`), `torch.split` 로
ONNX Split 생성(`9d90a72`)
- `ConcatTrace` 는 `len(x)` 제거(`882e8b7`)

### 13.3 Novatek 전용 qconfig — `quant_utils.py` (599줄)

`POTFakeQuantize` 가 **power-of-two scale**을 강제한다 (CNN25 격자 정합, `b96b4c8`).
`PercentileObserver` 가 percentile clipping을 한다.

`TARGET_REGISTRY` 에 **13개**가 등록돼 있다.

```
generic · qnnpack · fbgemm · aarch64 · percentile · aarch64_percentile
generic_int16 · percentile_int16
novatek_8a12w · novatek_pct_8a12w · novatek_pot_8a12w
novatek_pot_pct_8a12w · novatek_cnn25_clip
```

`8a12w` 는 activation 8bit / weight 12bit 혼합 정밀도다. 미등록 이름을 주면 `get_qconfig`
가 `**ValueError` 를 던진다** — §6.1의 model registry와 달리 여기는 fallback하지 않는다.
`register_target(이름, 함수)` 로 확장할 수 있다.

**가장 특이한 설계는 되먹임이다.** `tools/generate_npu_strategy.py` 가 NPU 컴파일러 출력
(`quan_info_bitinfo`)에서 layer별 bit-width를 뽑아 YAML로 만들고, `fx_qconfig.py` 의
`load_npu_strategy` / `build_qconfig_mapping_from_strategy` 가 그것을 읽어 PyTorch QAT에
적용한다. 즉 **컴파일러가 정한 정밀도를 학습에 다시 넣는다.**

### 13.4 파이프라인은 두 개이고 서로 다르다

혼동이 잦은 지점이라 분명히 구분한다.


| script                                                    | stage                                 | config loader                                 | 용도                 |
| --------------------------------------------------------- | ------------------------------------- | --------------------------------------------- | ------------------ |
| `tools/run_finetune_pipeline.py` (653줄) · `_v2.py` (799줄) | **5개** — `stage_0`~`stage_3`+`export` | v2만 §5.2의 `load_pipeline_yaml`                | stage별 finetune 학습 |
| `tools/nova_qat_pipeline.py` (1065줄)                      | **9개** — `_STAGES` 1~9                | 자체 `load_pipeline_config` → `PipelineContext` | Novatek QAT 전체 변환  |


`nova_qat_pipeline.py` 의 9 stage:

```
1 float ONNX export → 2 ONNX rename → 3 compile 해서 bit-width 정보 얻기
→ 4 quant model convert → 5 calibrate + PTQ export → 6 NMQT QAT finetune
→ 7 QAT ONNX export → 8 QAT compile → 9 simulate
```

`PipelineContext` 가 save/load로 중단 지점 재개를 지원하고, `_STAGE_PREREQS`(1~9)가
각 stage의 선행 산출물을 검사한다. D-016이 반복 지적하는 실패가 **중간 stage를 조용히
건너뛰는 것**이라 각 stage가 산출물 존재를 확인한다 (예: `dac7a0d` — stage 5에서
`calib_deploy.onnx` 확인).

---

## 14. NPU convert — `novatek/convert/`

셸 wrapper 3개 + 시각화 2개. **wrapper만 쓴다. SDK 명령을 직접 쓰지 않는다.**

```
run_pipeline.sh <model_dir> [test_img] [conf]
  ├── run_compiler.sh   compiler.nvtai --use-onnx --chip 530
  │                     --config-dir ./nvtai_tool --pattern-name <대문자>
  ├── run_simulator.sh  simulator.nvtai (emu530)
  └── visualize_sim.py  (DMS는 visualize_dms_sim.py)
```

- model_dir 이름이 **대문자로 바뀌어** `--pattern-name` 이 된다
(`front_260401` → `FRONT_260401`, `run_compiler.sh:36-37`).
- simulator가 `nvtai_tool/input/data/<대문자>/test_img_list.txt` 를 읽는다. 없으면
**compile은 성공하고 simulate만 실패**한다.
- `LD_LIBRARY_PATH` 를 덮어쓰지 말고 **뒤에 이어붙인다.** npu-convert 컨테이너가
`libprotobuf.so.17` 을 `/opt/pb36/lib` 에 두고 compose가 주입하므로, 덮어쓰면 그
경로가 사라진다.
- 결과물은 `runs/npu/novatek/<dir>/nvtai_tool/output/<대문자>/`.

`tools/fuse_mean_std_norm.py` 가 `/255 + mean/std` 를 첫 Conv에 융합한다 — 전처리를
NPU에 떠넘기지 않기 위한 것이다.

---

## 15. 실행 환경


| 서비스           | base image                                      | 핵심                                                                                                                                 |
| ------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `dair-train`  | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` | uv 설치, **CPython 3.12를 이미지에 굽는다** (`uv.lock` 의 onnxruntime-gpu가 cp311/cp312 전용). `shm_size: 16gb` — 기본 64MB면 DataLoader가 bus error |
| `npu-convert` | `ubuntu:20.04`                                  | **protobuf 3.6.1 소스 빌드** — SDK 컴파일러가 `libprotobuf.so.17` 을 요구하고 시스템 protobuf와 충돌을 피해야 한다                                           |


`UV_PYTHON_INSTALL_DIR=/opt/uv-python` 으로 Python 설치 위치를 이미지 안에 고정한다.
기본값(`$HOME/.local/share/uv`)이면 venv의 `bin/python` 링크 대상이 컨테이너와 함께
사라져 venv가 깨진다. `/opt/venv` 는 named volume(`dair-venv`)이고, uv가 마운트 지점을
지울 수 없어 `uv sync` 로도 복구되지 않는다.

`**.env` 파일이 없다.** compose 기본값으로 `dair-train` 은 돌아가지만
`NOVATEK_SDK_DIR` 이 `/tmp/novatek-sdk-unset` 이 되어 `**npu-convert` 는 실패한다.**

---

## 16. 영향 범위 분석

### 16.1 `projects/src` 내부 의존 (import 실측 집계)

많이 import되는 모듈이 곧 파급이 넓은 모듈이다.


| 모듈                                      | 피참조 수 | 여기를 고치면                      |
| --------------------------------------- | ----- | ---------------------------- |
| `src.data.dataset`                      | 10    | 전 Trainer의 데이터 로딩            |
| `src.data.augment`                      | 9     | 전 Trainer + GA 탐색            |
| `src.utils.loss`                        | 7     | 전 Trainer의 학습 신호             |
| `src.models.yolo.detect.val`            | 6     | 전 Trainer의 검증                |
| `custom_utils.cfg`                      | 6     | **전 CLI script + 전 Trainer** |
| `src.nn.tasks`                          | 5     | 모델 생성 전체                     |
| `src.data.dynamic_crop`                 | 4     | predictor 3개 (추론)            |
| `src.utils.trace`                       | 3     | head 분기 + FX QAT             |
| `src.nn.modules.head`                   | 3     | 모델 구조 + export + NPU         |
| `src.models.yolo.detect.train.base_qat` | 3     | QAT 3갈래                      |


### 16.2 변경 유형별 파급 경로


| 무엇을 바꾸면                                               | 반드시 함께 확인                                                                                       |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **head 출력 shape/순서** (`nn/modules/head.py`)           | loss(`utils/loss.py`) → Validator → `tools/export_*.py` → NPU `nvtai_tool` 설정 → §14 simulate 비교 |
| **DFL offset**                                        | `loss.py:102-107` 분기 (§8.1), `tal.py` 의 `offset`/`radius` 전파                                    |
| `**batch` dict 키** (`collate_fn`)                     | 전 Trainer `preprocess_batch`, 전 loss `__call__`, Validator, `visualize_dair_augmentations.py`   |
| `**label` dict / `Instances`** (`update_labels_info`) | 전 augmentation (좌표 변환이 `Instances` 경유)                                                          |
| **augmentation 추가**                                   | `_ROIBlurBase`/`_BboxPasteBase` 상속 확인, `v8_transforms_*` 등록, 카메라별 preset                        |
| `**model_name` registry**                             | **세 곳 모두** (§6.1 가·나·다)                                                                         |
| **config 스키마**                                        | `exp_params` 인지 `overrides` 인지 (§4.2) — 후자는 upstream이 검증한다                                      |
| `**derive_run_dir` 규칙**                               | `configs/experiments/**`, `configs/releases/**` 전체 + legacy 트리                                  |
| **quantization target 추가**                            | `TARGET_REGISTRY` + `_PERCENTILE_TARGETS` + `_SKIP_INT8_CONVERT_TARGETS`                        |
| **ultralytics 버전**                                    | 전 계열. §2의 상속 지점 전부                                                                              |


### 16.3 외부 시스템 의존


| 대상                         | 결합 방식                             | 끊어지면                                       |
| -------------------------- | --------------------------------- | ------------------------------------------ |
| Novatek CNN25 SDK          | 셸 wrapper + `NOVATEK_SDK_DIR` 마운트 | `npu-convert` 전체 정지 (현재 `.env` 없어 이미 그 상태) |
| NAS_3 (`192.168.103.183`)  | NFS rw, dataset 원본                | 신규 데이터 반입 불가 (로컬 사본은 유지)                   |
| NAS_2 (`192.168.103.182`)  | WebDAV **ro**, release 가중치 원본 겸 백업 | `MANIFEST.nas_source` 로 산출물 회수 불가          |
| GitLab (`192.168.101.218`) | git remote                        | push 불가                                    |
| MongoDB                    | **선택, 현재 미동작** (§4.6)             | 영향 없음                                      |
| `nmqt` 툴킷                  | 미선언 의존성                           | NMQT QAT 경로 정지 (D-010 보류 중이라 무해)           |


### 16.4 동시 실행 시 충돌 지점


| 자원                                               | 충돌 가능성                                                                                                                        |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `runs/<P>/<C>/<V>/`                              | `derive_run_dir` 이 config 경로로 유일하게 정하므로 서로 다른 config는 안 겹친다. **같은 config를 두 번 돌리면** `increment_path` 로 `v03`, `v032`, … 가 생긴다 |
| GPU                                              | `-d` 로 수동 지정. 자동 조정 없음                                                                                                        |
| `DataLoader` worker                              | `workers <= 4` (RAM 125 GB 제약). GA 탐색이 이 상한을 강제한다(`f606145`)                                                                  |
| `/opt/venv` named volume                         | 컨테이너 여러 개가 공유. 한쪽에서 `uv sync` 하면 다른 쪽에 영향                                                                                     |
| `runs/npu/novatek/<model_dir>/nvtai_tool/output/` | 같은 `PATTERN` 이름을 두 번 컴파일하면 덮어쓴다                                                                                               |


---

## 17. 테스트 현황 — 가장 약한 지점

파일 7개, 1,179줄. **테스트 함수 정의 53개 → pytest 수집 58개**
(`test_ncnvox_mel.py:67` 의 `@pytest.mark.parametrize` 하나가 6개로 늘어난다).
기준선 `56 passed, 2 skipped`.


| 파일                                  | 개수  | 대상                         |
| ----------------------------------- | --- | -------------------------- |
| `test_npu_strategy.py`              | 18  | NPU layer별 quantization 전략 |
| `test_ncnvox_cmd_crnn.py`           | 8   | CMD CRNN                   |
| `test_landmark_weights.py`          | 7   | DMS keypoint 가중치           |
| `test_ncnvox_mel.py`                | 7   | mel 동등성 (librosa ↔ numpy)  |
| `test_audio_classification_base.py` | 5   | audio dataset base         |
| `test_command_dataset.py`           | 4   | CMD dataset                |
| `test_ncnvox_wakeword_dataset.py`   | 4   | KWS dataset                |


### detection 경로 테스트가 0개다

Trainer 14개, loss 15개, augmentation 20여 종, `derive_run_dir` 의 gate 3개, §4.3·§4.4의
스키마 계약 — **어느 것도 테스트가 없다.** 커버되는 것은 audio(21) · NPU 전략(18) ·
DMS keypoint 가중치(7)뿐이다.

더 구체적으로, **detection head를 다룬 유일한 테스트가 삭제됐다.**

```
6a0ae00  test: add output shape tests for SymmetricDetect export paths
2504a44  chore: remove temporary test scripts    ← tests/test_output_shapes.py 삭제
```

§8.2의 `SymmetricDetect` 는 export 분기가 3갈래 × 플래그 6개이고, §8.3의 `DMSPose` 는
stage마다 출력 순서가 바뀐다. **shape 테스트가 가장 필요한 곳에서 사라진 상태다.**

---

## 18. 위험 목록

### 18.1 죽은 코드

- `**projects/model_overrides/simota.py**` (`SimOTAAssigner`) — 참조 0건. 실제로 쓰이는
것은 `simota_gpt.py` 의 같은 이름 클래스이고 `simota_dair.py:5` 가 그쪽을 import한다.
보관된 `configs/archive/.../simota_test.yaml` 의 `assigner: simota` 도 이 파일을 거치지
않는다.
- `**dair_method_mode: prev**` — config 사용 0건.
- **MongoDB 경로** — `pymongo` 미설치로 도달 불가 (§4.6).

### 18.2 눈치채기 어려운 실패 지점


| 위치                                        | 무엇이 조용한가                                                                |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| `train_custom_trainer.py:67`              | 미등록 `model_name` → `CustomYOLO` fallback. 다른 모델이 학습되고 에러가 없다            |
| `nn/tasks/__init__.py` `get_custom_model` | 미등록 이름에 `None` → `custom.py:101` 이 upstream `DetectionModel` 로 fallback |
| `cfg.py` `derive_run_dir`                 | gate 3개 중 하나만 어긋나도 no-op → `runs/detect/train2`                         |
| `loss.py:102-107`                         | 새 DFL 변형에서 이 분기를 놓치면 offset=0 → 박스 뒤집힘 → mAP=0 (D-003)                  |
| `collate_fn` `batch_idx` 보정               | 잘못 고치면 정답이 다른 이미지에 붙는다 (§4.4)                                           |
| NPU convert 중간 stage                      | 건너뛰어도 진행된다 (D-016)                                                      |
| `_save_to_mongodb`                        | 실패를 warning으로만 남긴다 (upstream 코드)                                        |


### 18.3 헷갈리기 쉬운 구조

- `dynamic_crop.py` 두 개 (학습 dataset vs 추론 wrapper).
- `InputRescale`/`ReshapeToSequence`/`LSTMLastStep` 이 `cmd_crnn.py` 와
`cmd_yolov8_crnn.py` 에 각각 따로 정의.
- `model_name` 이 세 곳에서 해석되고 키 집합 불일치 (§6.1).
- finetune 파이프라인(5 stage)과 QAT 파이프라인(9 stage)이 다른 script·다른 loader (§13.4).
- `KnowledgeDistllationDetectionTrainer` — 클래스 이름에 오타(`Distllation`)가 있다.
- `configs/datasets/` 20개 중 **8개만** 이 호스트에서 경로가 해석된다.

### 18.4 선언되지 않은 의존성


| 패키지                                   | 필요한 곳                                                                                                      | 현재 영향                      |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------- |
| `nmqt`                                | `quant_tools/rename_onnx.py`, `tools/nova_qat_pipeline.py`, `nmqt_qat.py`                                  | QAT 보류(D-010)라 무해          |
| `pymongo`                             | `CustomTuner` 분산 GA 탐색 (§4.6)                                                                              | 분산 모드 사용 불가                |
| `librosa`/`soundfile` (`audio` extra) | `mel.py` 등은 명시적 `RuntimeError`, `tools/{prepare_ncnvox_data,split_voice_utterances}.py` 는 즉시 `ImportError` | `uv sync --extra dev` 로 해결 |


### 18.5 코드 위생

ruff **9건**이 기준선이다 — bare `except` 1건(`custom_coco_eval.py:127`), `F811` 2건
(`dair.py:13`, `symmetric.py:31` — 둘 다 `RANK` 중복 import), `E711` 1건
(`custom.py:100`, `builder == None`), `W291`/`W293` 5건. 전부 CLAUDE.md에 "고칠 수 없어
남긴 항목"으로 기록돼 있다. **이 숫자가 늘면 회귀다.**

2026-08-04 이전에는 14건이었다. D-020 이 편집한 파일의 항목만 정리했다 —
`evaluator.py` 의 bare `except` 2건을 `OSError`/`ValueError` 로 좁히고
`evaluator.py`·`utils.py` 의 공백 3건을 없앴다. CLAUDE.md 의 "Reduce them in files you
touch; do not sweep the tree" 규약을 따른 것이고, 남은 9건은 이번 작업이 손대지 않은
파일에 있다.

주석 처리된 코드가 남은 곳 — `head.py` `SymmetricDetect.__init__` 의 절반,
`**loss.py:112**` (261자, 저장소 최장 줄 — 주석 처리된 assigner 분기),
`loss.py:180-183`(이전 `dair_loss` 버전들), `custom.py:34-38`. CLAUDE.md 스스로
"죽은 코드는 주석 처리하지 말고 지운다"고 적고 있어 규칙과 실태가 어긋난다.

> CLAUDE.md는 이 최장 줄을 `loss.py:101` 로 적고 있으나 실제 위치는 **112행**이다
> (길이 261자는 일치). 문서가 코드 줄 번호를 직접 인용하면 코드가 조금만 움직여도
> 이렇게 어긋난다 — 이 문서의 줄 번호도 같은 이유로 언젠가 틀리게 된다.

---

## 19. 실무 체크리스트

1. `**model_name` 은 §6.1 표와 대조한다.** 오타가 조용히 다른 모델을 학습한다.
2. **config 파일 이름은 `exp.yaml`, 위치는
configs/{experiments,releases}/****/****/****/`.** 아니면 출력 위치가
산되지 않는다.
3. **새 파라미터는 `exp_params` 에 넣는다.** `overrides` 는 upstream이 검증한다 (§4.2).
4. **255로 나누지 않는다.** 입력은 `[0,255]` float32다.
5. **DFL offset을 바꾸면 `loss.py:102-107` 도 같이 바꾼다.**
6. `**batch`/`label` 키를 바꾸면 §16.2의 파급 경로를 전부 확인한다.**
7. **새 helper를 쓰기 전에 §7.2 목록을 본다.**
8. **NPU convert는 wrapper만 쓴다.** 중간 stage를 건너뛰어도 조용히 통과한다.
9. `**workers <= 4**`, 전방 4K GA 탐색은 `amp: false` (D-006).
10. **detection 경로 변경은 테스트가 잡아주지 않는다.** 1 step 실제 학습으로 확인한다.
11. **QAT 코드는 지우지 않는다** (D-010).
12. **파이프라인을 말할 때 어느 쪽인지 밝힌다** — finetune 5 stage인지 QAT 9 stage인지.

