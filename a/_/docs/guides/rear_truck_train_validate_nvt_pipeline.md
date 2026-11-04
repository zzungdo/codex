# Rear_Truck 모델 학습 → 검증 → NVT 평가 표준 파이프라인 (SOP)

BSD_F606 Rear_Truck에서 새 모델을 하나 만들 때, 실험 정의부터 board(NPU) 실측·기존 모델과의
비교·문서화까지 빠뜨리지 않고 진행하기 위한 재사용 가능한 절차서다.

**이 문서는 특정 실험(v01/v02/v01_dair/v01_symmetric)의 회고가 아니다.** 그 실험들의 상세
결과와 원인 분석은 [`docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`](../debug/20260806_rear_truck_v01_npu_quant_gap.md)에,
요약은 [`configs/experiments/BSD_F606/Rear_Truck/README.md`](../../configs/experiments/BSD_F606/Rear_Truck/README.md)에
있다 — 이 문서는 그 절차를 일반화한 것으로, 아래에서는 그 실험들을 "이 절차가 검증된 근거"로만
인용한다.

`<experiment>` 는 새 실험 이름(예: `v03`, `v01_dair_matched_symmetric`)으로 치환한다.

---

## 평가 이미지 수: 7,894장과 6,152장 (처음 보는 사람을 위한 설명)

이 파이프라인에는 서로 다른 이미지 수를 쓰는 평가가 두 번 나온다(4절과 6절) — 실수로 섞어서
비교하기 쉬우므로 먼저 관계를 명확히 한다.

```text
Original validation
7,894 images
    │
    ├─ Internal Float validation → 7,894장 전체 (4절)
    │
    └─ Physical Board 처리 불가 1,742장 제외
                ↓
       Common evaluation set
             6,152장
                ↓
       Float ONNX ↔ Board NPU
       quantization 비교 (6절 · 11절 · 12절)
```

- **원래 validation**: 7,894장 (project 표준 val-split).
- **Board 처리 불가**: 1,742장 — physical NVT Board에서 이 이미지들은 정상적인 inference
  결과를 얻지 못했다. **원인은 확인되지 않았다 — 추측하지 않는다** (파일 포맷/해상도/전처리/
  board 제약 등으로 단정하지 않는다).
- **공통 평가 집합**: `7,894 - 1,742 = 6,152`장. **임의로 sampling한 subset이 아니다** —
  "Board에서 결과를 얻을 수 있는 이미지"와 "Float ONNX 평가가 가능한 이미지"의 교집합이며,
  Float 쪽에서도 동일하게 이 6,152장만 사용해야 Board와 apples-to-apples 비교가 된다.

**따라서: 4절(7,894장 Internal Float)과 11절(6,152장 Board) 결과를 직접 비교해서 12절의
quantization loss를 계산하지 않는다.** quantization loss는 반드시 **6절(6,152장 Float
ONNX)과 11절(동일 6,152장 Board) 사이에서만** 계산한다.

---

## 0. 실험 시작 전 정의

학습을 시작하기 전에 아래 항목을 **먼저 기록**한다 (`exp.yaml`의 `_meta_` 블록에, 아래 1절 참조).

| 항목 | 예시 |
|---|---|
| 실험 이름 | `v03` |
| 목적 / 가설 | 무엇을 확인하려는 실험인지 한 문장 |
| baseline 모델 | 비교 기준이 될 기존 모델 (`v01`, `v02` 등) |
| baseline 대비 변경점 | **하나만 바꿨는지, 여러 개를 바꿨는지 명시** |
| dataset / train·val split | dataset yaml 경로 |
| architecture | `weight_path`의 모델 yaml |
| Trainer / Validator | `model_name` 및 실제 클래스 (2절 참조) |
| imgsz | `overrides.imgsz` |
| batch | `overrides.batch` |
| optimizer / lr0 / lrf / cos_lr | |
| DAIR 여부 및 `dair_params` | `mode`/`_lambda`/`gamma`/`num_choice` |
| `TensorSpeakerOverlayAug` 상태 | `projects/src/models/yolo/detect/train/dair.py`의 `preprocess_batch()`에서 활성/비활성 |
| seed | |
| pretrained 여부 | |
| epochs / patience | |
| 기타 주요 augmentation | |

**"무엇을 바꿨는지"와 "무엇을 고정했는지"를 실험 시작 전에 명시하는 것이 가장 중요하다.**
v01_dair와 v01_symmetric을 비교했을 때, DAIR 유무 외에 optimizer/lr/batch/cos_lr도 함께 달라져
있어 "symmetric이 좋은 이유가 DAIR 제거 때문인지 다른 하이퍼파라미터 때문인지"를 사후에 분리할
수 없었다 — 이 정보가 없으면 원인 분석 단계에서 결론을 내릴 수 없다.

---

## 1. config 생성 및 사전 검증

### 위치 규칙

```
configs/experiments/BSD_F606/Rear_Truck/<experiment>/exp.yaml
```

파일명은 정확히 `exp.yaml`이어야 한다 — `derive_run_dir`(`projects/custom_utils/cfg.py`)가
`configs/{experiments,releases}/<P>/<C>/<V>/exp.yaml` 형태만 인식해 출력 디렉토리를
`runs/detect/<P>/<C>/<V>/`로 파생시킨다. 파일명이 다르거나 경로 조각이 부족하면 조용히
`runs/detect/train2` 같은 기본 경로로 떨어진다 — 에러 없이 실패하는 지점이다.

기존 실험에서 복사해서 시작한다:

```bash
mkdir -p configs/experiments/BSD_F606/Rear_Truck/<experiment>
cp configs/experiments/BSD_F606/Rear_Truck/v02/exp.yaml \
   configs/experiments/BSD_F606/Rear_Truck/<experiment>/exp.yaml
# _meta_ / model_name / dair_params / overrides 등을 0절 정의에 맞게 수정
```

`model_name`에 따라 `overrides.project`+`name` 방식(대부분)과 top-level `save_dir` 방식
(`symmetric`)이 갈린다 — [`docs/diagrams.md`](../diagrams.md)의 registry 표로 확인한다.

### config 작성 후 체크리스트

- [ ] `overrides.data` (또는 `dataset_name`)가 가리키는 dataset yaml이 이 PC에 실제로 존재
- [ ] `weight_path`(architecture yaml)가 존재
- [ ] `model_name`이 의도한 Trainer/Validator와 일치 (`.get(key, (CustomYOLO, CustomDetectionTrainer))` —
      미등록 키는 에러 없이 다른 아키텍처를 학습한다)
- [ ] `imgsz` / `batch` 확인 (dual-GPU 호스트는 `workers <= 4`)
- [ ] optimizer / lr0 / lrf / cos_lr이 0절에서 정의한 값과 일치
- [ ] DAIR 여부(`dair_params` 존재 여부)와 `TensorSpeakerOverlayAug` 코드 상태가 의도한 조합인지
- [ ] `runs/detect/BSD_F606/Rear_Truck/<experiment>/`가 아직 존재하지 않음 (기존 결과 덮어쓰기 방지)
- [ ] `project`/`name`/`save_dir`를 이 config에 직접 넣지 않았음 (경로가 파생되도록)

---

## 2. 학습 실행

```bash
python3 tools/train_custom_trainer.py \
    -w configs/experiments/BSD_F606/Rear_Truck/<experiment>/exp.yaml -d <gpu>
```

- `-w`는 `--cfg_path`의 별칭.
- `-d`는 GPU 인덱스. DDP가 필요하면 `tools/` 하위 dist 스크립트 사용(현재 Rear_Truck 실험은
  단일 GPU로 진행).
- 출력: `runs/detect/BSD_F606/Rear_Truck/<experiment>/` — `results.csv`, `weights/best.pt`,
  `weights/last.pt`.
- 학습 30초 이상 걸리는 작업은 예상 소요 시간을 먼저 알린다.

### 학습 중 확인할 것

- epoch time — 비정상적으로 느려지면 아래 리소스 간섭 항목부터 의심
- GPU utilization
- loss 이상 여부 (NaN 등)
- EarlyStopping(patience) 상태
- **동일 PC에서 NVT compile 등 heavy 작업이 동시에 돌고 있는지** — Novatek 컴파일러는
  CPU-heavy하고 컨테이너에 cgroup 제한이 없어 32코어를 다 나눠 쓴다. 학습과 동시에 무거운
  compile을 돌리면 epoch time이 크게 늘어날 수 있다. 병행이 필요하면 compile 쪽에
  `nice -n 19`를 걸어 학습에 우선순위를 준다.
- 백그라운드 실행 스크립트를 쓸 때 `cmd1 && cmd2 || cmd3 && cmd4` 형태의 셸 연산자 우선순위를
  주의한다 — 왼쪽부터 같은 우선순위로 묶여 `((cmd1 && cmd2) || cmd3) && cmd4`로 평가되므로,
  의도와 달리 `cmd4`(예: 재시작 로직)가 무조건 실행돼 중복 학습이 생길 수 있다.

---

## 3. 학습 종료 확인

학습이 끝났다고 바로 4절로 넘어가지 않는다.

- [ ] EarlyStopping(patience) 또는 `epochs` 완주로 정상 종료했는지 (`results.csv` 마지막 행,
      로그의 종료 사유)
- [ ] `weights/best.pt` 존재
- [ ] `weights/last.pt` 존재
- [ ] `best.pt`가 `torch.load`로 로드 가능한지 (파일 손상 여부)
- [ ] `results.csv`의 best epoch가 patience 곡선상 합리적인지
- [ ] `args.yaml`(as-run 기록)이 의도한 config와 일치하는지
- [ ] 동일 run 디렉토리에 의도치 않은 중복 학습(예: `<experiment>2`)이 없는지 —
      `diff`로 `args.yaml`의 `name`/`save_dir`만 다르고 나머지가 동일하면 중복 의심

문제가 발견되면 **기존 파일을 삭제하거나 덮어쓰지 않고** 별도로 검증부터 한다 (프로세스 종료 →
3중 확인 → 필요시에만 격리된 디렉토리 삭제).

---

## 4. Internal Float validation

학습 로그의 마지막 epoch 지표를 그대로 가져다 쓰지 않는다. `best.pt`를 **실제 Trainer/Validator
로 다시 validation**해서 baseline으로 기록한다. 이 단계는 **원래 validation 전체 7,894장**을
쓴다 — 6절의 6,152장(위 "평가 이미지 수" 절 참고)과는 다른 집합이므로, 이 단계의 mAP를
Board mAP와 직접 비교하지 않는다.

**주의: `tools/validate_custom.py`와 `tools/eval.py`는 이 목적에 그대로 쓸 수 없다.**
- `tools/validate_custom.py`는 `model_name`과 무관하게 항상 `CustomDetectionValidator`를 쓴다 —
  DAIR 모델이 필요로 하는 독립적 x/y-gain box scaling(REAR의 비정방형 1920x1080→320x320
  resize에 필요)이 빠져 있다.
- `tools/eval.py`는 (a) `overrides`에 `model` 키가 없으면 `BaseTrainer.__init__`이 `args.model
  is None`으로 죽고, (b) 존재하지 않는 `trainer.testset`을 참조한다 (이 repo의 Trainer들은
  `self.data`라는 단일 dict만 가진다).

대신 아래 패턴으로 매 실험마다 1회성 스크립트를 작성해 컨테이너 안에서 실행한다
(`model_name`에 따라 Trainer/Validator만 교체):

```python
from pathlib import Path
from projects.custom_utils.cfg import get_my_cfg
from projects.src.models.yolo.detect.train import DairDetectionTrainer  # 또는 SymmetricTrainer 등

cfg = get_my_cfg("configs/experiments/BSD_F606/Rear_Truck/<experiment>/exp.yaml")
kwargs = {
    "task": "detect", "mode": "val",
    "model": "runs/detect/BSD_F606/Rear_Truck/<experiment>/weights/best.pt",
}
trainer = DairDetectionTrainer(overrides=kwargs, custom_cfg=cfg)
trainer.save_dir = Path("runs/detect/_verify/<experiment>_float_baseline")
trainer.save_dir.mkdir(parents=True, exist_ok=True)
trainer.args.plots = False
trainer.setup_model()
trainer.test_loader = trainer.get_dataloader(
    trainer.data.get("val") or trainer.data.get("test"),
    batch_size=trainer.batch_size * 2, rank=-1, mode="val",
)
validator = trainer.get_validator()
validator.save_dir = trainer.save_dir
results = validator(model=trainer.model)
```

`SymmetricTrainer`는 `super().__init__()`을 호출하지 않고 `custom_cfg.save_dir`를 직접
읽으므로, 이 클래스를 쓸 때는 construction 전에 `cfg.save_dir = "..."`를 명시적으로 설정해야
한다 (`IterableSimpleNamespace`는 없는 키를 조회하면 `None`이 아니라 `AttributeError`를 낸다).

기록할 항목: image count, Precision, Recall, mAP50, mAP50-95, 사용한 Validator 클래스,
dataset config 경로, imgsz, batch.

**주의: 모델마다 Trainer/Validator가 다르면(특히 Symmetric Validator vs DAIR Validator —
box scaling 방식 자체가 다르다) Internal Float 수치를 서로 apples-to-apples로 직접 비교하지
않는다.** 모델 간 공정 비교는 6절의 동일 ONNX 평가에서만 한다.

---

## 5. ONNX export

```bash
python3 tools/export_custom.py \
    -w runs/detect/BSD_F606/Rear_Truck/<experiment>/weights/best.pt \
    -i 320 -y 320 \
    -s runs/detect/BSD_F606/Rear_Truck/<experiment>/weights/best.onnx
```

Rear_Truck 320 모델 기준 조건 (`tools/export_custom.py` 기본값 확인됨):
- `opset=13` (기본값)
- `simplify=False` (`-f`를 주지 않으면 기본 False)
- `nms=False` (하드코딩, 옵션 없음)
- `skip_postprocess=True` (기본값 — `-p`로 끌 필요 없음)

확인할 것:
- [ ] export 성공 (에러 없이 `.onnx` 생성)
- [ ] ONNX input shape이 `(1, 3, 320, 320)`
- [ ] ONNX output이 cls `(1, 5, 2100)` + DFL-decoded box `(1, 4, 2100)` 형태인지 (metadata 확인)
- [ ] 기존 `best.onnx`를 덮어쓰지 않았는지 (다른 experiment 디렉토리에 쓰는지 확인)

---

## 6. 동일 6,152장 Float ONNX 평가

**이 단계가 모델 간 apples-to-apples Float baseline이다.** 4절의 Internal Float validation은
Trainer/Validator 차이 때문에 모델 간 비교에 그대로 쓸 수 없지만, 이 단계는 모든 모델을
동일 GT·동일 전처리·동일 NMS·동일 Evaluator로 평가하므로 직접 비교 가능하다.

공통 workspace: `runs/_eval_scratch/` — 디렉토리 구조와 규칙은
[`runs/_eval_scratch/README.md`](../../runs/_eval_scratch/README.md)를 따른다 (이 문서에서
중복 서술하지 않는다).

공통 조건:
- GT: `ipu_gt/`, 6,152장(= 7,894장 원본 val에서 board 처리 불가 1,742장을 제외한 공통
  집합 — 위 "평가 이미지 수" 절 참고), `cxywhn`
- 전처리: `basic_preproc`
- 후처리: `basic_postproc_v8_dual_nms` (`min_overlap_thres=0.6`, `union_overlap_thres=0.3`) —
  보드의 온디바이스 dual-threshold NMS와 동일하게 맞춰 NMS 차이를 배제

새 모델마다 `runs/_eval_scratch/configs/onnx/onnx_<experiment>_dualnms_cfg.yaml`을 **기존
config를 복사해서** 새로 만든다 (기존 config를 덮어쓰지 않는다). `model_path`/`save_dir`만
바꾼다.

```bash
docker exec ultralytics_projects-dair-train-1 bash -lc \
  'cd /workspace && uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_<experiment>_dualnms_cfg.yaml'
```

기록: Float ONNX mAP50, mAP50-95 — 이 값이 12절 quantization loss 계산의 **분모/기준값**이다.

---

## 7. NVT conversion

**현재 권장 baseline은 3-tensor low-cost 구성이다** (v02/v01_dair/v01_symmetric에서 사용한
동일 조건). `gen_config.txt`의 `[precision/high_precison/spec_demarc_tensors]`에 아래 3개만
지정한다:

```
Softmax_/model.22/dfl/Softmax_output_0_Y
Sigmoid_output0_Y
Conv_/model.22/dfl/conv/Conv_output_0_Y
```

model.2/model.4의 T1~T6까지 보호하는 9-tensor 구성은 **새 모델의 기본 baseline compile에
사용하지 않는다** — v01에서 시도했던 workaround로, latency 약 12.25ms · DRAM 약 78.5MB ·
model size 약 9.39MB까지 비용이 커졌다. 3-tensor 결과가 목표(13절)를 만족하지 못할 때만
선택적으로 고려한다.

### 디렉토리

```bash
mkdir -p novatek/convert/rear_truck_<experiment>_320
```

기존 production/v01/v02 등 변환 디렉토리를 복사하거나 덮어쓰지 않고, 새 디렉토리를 만든다.
`nvtai_tool/config/<PATTERN>/cnn25/gen_config.txt`·`sim_config.txt`는 기존 3-tensor 구성
(`rear_truck_v02_320` 등)에서 `[path/model_dir]`만 새 `<PATTERN>`으로 바꿔 복사한다.
`<PATTERN>`은 디렉토리명을 대문자로 바꾼 것(`--pattern-name` 규칙)과 일치해야 한다 — 예:
`rear_truck_v03_320` → `REAR_TRUCK_V03_320`.

`nvtai_tool/input/model/customer/<PATTERN>/deploy_float.onnx`에 5절에서 export한 ONNX를
복사하고, **소스 ONNX와 md5가 일치하는지 확인한다** (잘못된 모델을 compile에 넣는 실수를
여기서 잡는다 — 실제로 board 결과가 다른 모델과 동일하게 나왔을 때 이 지점부터 역추적해서
원인이 아님을 확인한 사례가 있다, 9절 참조).

`nvtai_tool/input/data/<PATTERN>/`는 공유 calibration/test 이미지(`../REAR_TRUCK_V01_320/jpg`
등)를 symlink로 연결한다 — `run_simulator.sh`가 `<PATTERN>` 이름으로 이 경로를 강제로
참조하므로, 실제 이미지를 매 실험마다 복사할 필요는 없다.

### compile

```bash
novatek/convert/run_compiler.sh rear_truck_<experiment>_320
```

기록: compile 성공 여부, latency, DRAM bandwidth, model size, board binary 위치
(`<work_dir>/nvtai_tool/output/<PATTERN>/sdk/nvt_model.bin`).

현재 3-tensor baseline 참고값(v02 실측, **강제 기준이 아니라 regression 확인용**):
latency 약 6.368ms, DRAM 약 36.54MB, model size 약 4.71MB. 새 모델이 architecture를 바꾸지
않았는데 이 값에서 크게 벗어나면 compile 설정을 먼저 의심한다.

---

## 8. NVT simulator

```bash
# compile + simulate + visualize를 한 번에 (신규 실험 기본 경로)
novatek/convert/run_pipeline.sh rear_truck_<experiment>_320 [test_img] [conf_thresh]

# 또는 단계별로
novatek/convert/run_simulator.sh rear_truck_<experiment>_320 [test_img]
python3 novatek/convert/visualize_sim.py REAR_TRUCK_<EXPERIMENT>_320 --img <test_img>
```

`run_compiler.sh` + `run_simulator.sh`만으로는 비교 이미지가 자동 생성되지 않는다 —
`visualize_sim.py`를 별도로 호출해야 `sim_comparison_*.jpg`가 만들어진다. **기존 simulator
dump(`debug/golden_tensor/`, `debug/sim_layer/`)가 이미 있으면 compile/simulate를 다시 돌릴
필요 없이 `visualize_sim.py`만 재실행해서 시각화만 다시 생성할 수 있다.**

기록할 output cosine: `Softmax(dfl)`, `Sigmoid [OUT]`, `dfl_conv [OUT]`.

**주의: single-image cosine은 진단용 신호이지 최종 성능 지표가 아니다.** 높은 cosine이
긍정적 신호일 수는 있지만, 모델 최종 판정(13절)은 반드시 6,152장 실측 board mAP로 한다.

---

## 9. Board binary 전달 및 실제 보드 추론

최종 binary: `novatek/convert/rear_truck_<experiment>_320/nvtai_tool/output/<PATTERN>/sdk/nvt_model.bin`

보드 작업 체크리스트:
- [ ] 올바른 experiment의 binary인지 파일 경로로 재확인
- [ ] 파일 크기 확인
- [ ] 가능하면 md5 확인 (다른 experiment의 `nvt_model.bin`과 서로 달라야 함)
- [ ] 보드에서 기존 inference 프로세스 종료 여부
- [ ] 새 binary가 실제로 적용됐는지 (재플래싱 여부)
- [ ] inference 프로세스 재시작 여부

### stale-model 검증 (필수, 덤프를 받은 직후 바로 수행)

**과거 v01_symmetric과 v01_dair 두 사례 모두, 보드가 이전 모델 binary를 그대로 실행 중인
상태로 덤프해서 이전 모델의 6,152장 detection과 100% 동일한 결과가 나온 적이 있다.** 파일
자체의 md5/크기는 달라도(재전송·재압축 등으로) 내용은 동일할 수 있으므로, **덤프를 받으면
평가하기 전에 반드시** 직전에 확보한 다른 모델들의 merged dump와 detection 단위로 비교한다:

```python
import json

def key(d):
    return (d["name"], tuple(d["bbox"]), round(d["score"], 6), d["category"])

new_keys = set(key(d) for d in json.load(open("merged/<experiment>/...json")))
for other in ["v01", "v02", "v01_symmetric", "v01_dair", "production"]:
    other_keys = set(key(d) for d in json.load(open(f"merged/{other}/...json")))
    overlap = len(new_keys & other_keys) / max(1, len(new_keys | other_keys))
    print(other, "jaccard overlap:", overlap)
```

서로 다른 모델인데 filename/bbox/score/category가 6,152장 전체에서 완전히 동일하다면(overlap
≈ 1.0), **그 결과를 확정하지 않는다.** compile 단계(`best.pt`/`best.onnx`/`deploy_float.onnx`
의 md5가 각 experiment 소스와 일치하는지)부터 역순으로 확인해 compile 문제가 아님을 배제한
뒤, board flashing/model 교체 문제로 보고 재덤프를 요청한다.

---

## 10. Board dump 보존

```
runs/_eval_scratch/board_dumps/<experiment>/<experiment>_output.json
```

기존 dump를 덮어쓰지 않는다. 재덤프인 경우 `<experiment>_output_redump.json`처럼 별도
이름으로 보존한다. **stale/invalid로 판정된 덤프도 삭제하지 않고** 그대로 두어(또는
`_invalid` 접미사로) 이후 조사에서 "왜 이 값을 안 썼는지" 추적 가능하게 남긴다.

---

## 11. merge + Board 평가

```bash
python3 runs/_eval_scratch/tools/merge_board_dump.py \
  runs/_eval_scratch/board_dumps/<experiment>/<experiment>_output.json \
  runs/_eval_scratch/merged/<experiment>/ipu_pred_merged_<experiment>.json
```

`runs/_eval_scratch/configs/board/board_<experiment>_3tensor_cfg.yaml`을 기존 board config
(`board_v02_3tensor_cfg.yaml` 등)에서 복사해 `pred.path`만 바꿔 새로 만든다.

```bash
docker exec ultralytics_projects-dair-train-1 bash -lc \
  'cd /workspace && uv run python3 tools/customEval.py runs/_eval_scratch/configs/board/board_<experiment>_3tensor_cfg.yaml'
```

동일 6,152 GT, 동일 dual-NMS 전제(단, board 예측은 이미 온디바이스 NMS를 거쳤으므로
`pred.rep: ipu_test`로 추가 후처리 없이 그대로 읽는다), 동일 Evaluator.

기록: Board mAP50, Board mAP50-95.

---

## 12. Quantization loss 계산

**반드시 6절 Float ONNX 결과(6,152장)와 11절 Board 결과(동일 6,152장)끼리 비교한다.** 4절
Internal Float validation 결과(7,894장)를 Board와 직접 비교하지 않는다 — Trainer/Validator
경로가 다를 뿐 아니라 이미지 수 자체가 다르다 (위 "평가 이미지 수" 절 참고).

```
absolute loss = Board mAP50-95 - Float ONNX mAP50-95
relative loss (%) = (Board mAP50-95 / Float ONNX mAP50-95 - 1) × 100
```

음수가 손실을 의미한다. 예: Float ONNX 0.561 → Board 0.521이면
absolute loss = -0.040, relative loss = -7.1%.

---

## 13. 모델 판정 기준

- **acceptance target (2026-08-19 개정)**: Float → NPU mAP50-95 relative loss `≤ 1%`
- ~~1차 acceptance target: ≤ 10%~~ / ~~stretch target: ≤ 5%~~ — 상위 target으로 대체됨. 이전
  target 기준의 과거 결론(예: 아래 "현재 best reference" 단락, `20260806_...md`, Rear_Truck
  `README.md`의 "목표 달성" 서술)은 **모두 구 기준(10%) 기준이었다는 점을 감안해서 읽는다** —
  자세한 배경/영향은 `docs/decisions.md`의 해당 D-번호 참조.

quantization loss만으로 판정하지 않는다. 함께 본다:
- Float ONNX mAP50-95 (절대 성능)
- Board NPU mAP50-95 (절대 성능)
- quantization relative loss
- latency / DRAM / model size

Float 자체가 낮아서 loss 비율만 작아 보이는 모델은 좋은 후보로 보지 않는다 — 절대 성능과
loss 비율을 함께 봐야 한다.

**새 target(≤1%) 기준으로는 지금까지 측정된 5개 모델 중 이를 만족하는 것이 하나도 없다** —
가장 좋았던 `v01_symmetric`도 -7.1%로 새 target에 7배 이상 못 미친다. `docs/specs/qat_mixed_precision.md`의
자체 기준(">2% 드랍이면 mixed-precision/full QAT 필요")보다도 훨씬 엄격한 target이므로, ≤1%를
현재의 post-training quantization(calibration 튜닝, tensor 보호 확장 등) 접근만으로 달성할
가능성은 낮고 **QAT가 사실상 필수 경로**로 봐야 한다(D-010 "QAT 보류" 결정의 재검토 필요 — 위
개정 근거 참조).

`v01_symmetric`(relative loss -7.1%)은 **구(舊) 10% target 기준**으로는 5개 모델 중 유일하게
목표를 달성했던 결과였지만, 새 target(≤1%) 기준으로는 미달이다. 새 실험에서 결과가 나오면 이
문서가 아니라 `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`의 "최종 결론"과
`configs/experiments/BSD_F606/Rear_Truck/README.md`를 갱신한다 (이 SOP 문서 자체는 특정 모델
이름을 reference로 박아두지 않는다).

---

## 14. 기존 모델과 비교

| Model | Float ONNX mAP50-95 | Board NPU mAP50-95 | Absolute Loss | Relative Loss | Latency | DRAM | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<experiment>` | | | | | | | |

**한 번에 여러 조건이 다른 모델끼리 비교해서 특정 원인 하나로 단정하지 않는다.** 비교 목적에
따라 matched control을 명확히 정한다:

- augmentation ablation (예: OverlayAug on/off) → 그 항목만 다르고 나머지가 동일한 쌍으로 비교
- 학습 방식 비교 (예: DAIR vs symmetric) → dataset/split/주요 하이퍼파라미터가 일치하는 쌍으로
  비교 (하이퍼파라미터까지 다르면 "어느 쪽 효과인지" 분리 불가 — 0절 참조)
- 하이퍼파라미터 영향만 보고 싶다면 → 학습 방식은 고정하고 하이퍼파라미터만 다른 쌍으로 비교

---

## 15. 문서화

새 실험 완료 후 최소한 갱신:

- **요약**: `configs/experiments/BSD_F606/Rear_Truck/README.md`
- **상세 조사/quantization 결과**: `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`
- **`_eval_scratch`**: 새 config/board dump/merged 결과는 `runs/_eval_scratch/README.md`의
  디렉토리 규칙(`configs/onnx/`, `configs/board/`, `board_dumps/<model>/`, `merged/<model>/`)을
  따른다.

실험이 계속 늘어나 `docs/debug/20260806_...md` 한 파일이 과도하게 비대해지면, 그 문서 안에
새 실험을 계속 append하는 대신 실험별로 별도 결과 문서(예:
`docs/debug/rear_truck_experiments/<experiment>.md`)로 분리하고 `20260806_...md`는 원인
분석과 최종 결론만 남기는 구조가 나을 수 있다 — 실험 수가 5~6개를 넘어가는 시점에 재검토를
권장한다. 이 SOP 문서에서 그 판단을 미리 강제하지는 않는다.

---

## 최종 체크리스트

- [ ] 실험 목적/변경점 정의 (0절)
- [ ] `exp.yaml` 생성 및 사전 검증 (1절)
- [ ] 학습 실행, 리소스 간섭 확인 (2절)
- [ ] 학습 정상 종료 확인, 중복 실행 없음 확인 (3절)
- [ ] Internal Float validation (4절)
- [ ] ONNX export, shape 확인 (5절)
- [ ] 6,152장 Float ONNX 평가 (6절)
- [ ] 3-tensor NVT compile, resource 기록 (7절)
- [ ] simulator 실행, cosine 확인 (8절)
- [ ] visualization 확인 (8절)
- [ ] board binary 확인 (경로/크기/md5) (9절)
- [ ] board inference 수행
- [ ] **stale model/dump 여부 검증 (9절, 필수 — 다른 모델 dump와 detection 단위 비교)**
- [ ] dump merge (11절)
- [ ] 6,152장 Board 평가 (11절)
- [ ] quantization loss 계산 (12절)
- [ ] 기존 baseline과 비교, matched control 확인 (13, 14절)
- [ ] README 갱신 (15절)
- [ ] 상세 결과 문서화 (15절)
