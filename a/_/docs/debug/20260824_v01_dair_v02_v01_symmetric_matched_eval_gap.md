# v01_dair / v02 / v01_symmetric_matched 평가 방식 차이 검증 — evaluator 효과만으로는 손실이 해소되지 않는 모델들

- **작성**: 2026-08-24 (KST)
- **통합 안내**: 이 문서는 원래 `docs/debug/20260824_v01_dair_eval_gap.md`,
  `docs/debug/20260824_v02_eval_gap.md`, `docs/debug/20260824_v01_symmetric_matched_eval_gap.md`
  세 개의 별도 문서로 작성됐던 것을 하나로 통합한 것이다 — 세 모델 모두 "evaluator를
  ultralytics로 바꿔도 손실이 사실상 사라지지 않는다"는 공통점으로 묶인다(`v01_symmetric`만
  예외). 원본 세 파일의 내용은 그대로 보존했고, 네 모델을 함께 비교하던 표(구
  v01_symmetric_matched 문서 §8/§8-1)만 기준 문서인
  `docs/debug/20260824_v01_symmetric_eval_gap.md` 하단(§10/§10-1)으로 옮겼다. 원본 세 파일은
  이 통합 문서로 대체됐다.
- **기준/선행 문서**: `docs/debug/20260824_v01_symmetric_eval_gap.md` — 세 모델 모두 이 문서의
  실험 방법·스크립트·조건을 그대로 재사용했다. **v01_symmetric 쪽 실험/결과는 어느 것도
  재실행하거나 수정하지 않았다.**
- **관련 정본**: `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md` (v01_dair -20.7%,
  v02 -18.8%, v01_symmetric_matched -28.15%의 기존 quantization loss가 처음 기록된 문서,
  전부 미수정)

## 목적 (세 모델 공통)

`docs/debug/20260824_v01_symmetric_eval_gap.md`에서 v01_symmetric에 대해 확인된 다음
현상이 다른 모델들에서도 재현되는지 확인한다:

1. pycocotools(customEval.py) 기준으로는 loss가 크게 나온다
2. ultralytics `ap_per_class` 기준으로는 loss가 작게(또는 사라지게) 나온다
3. NMS 종류보다 evaluator 선택이 loss 크기에 훨씬 큰 영향을 준다

세 모델을 순서대로 검증하면서 가설이 어떻게 다듬어졌는지 그 과정을 그대로 보존한다:
v01_dair·v02는 "evaluator 효과는 재현되지만 baseline 손실이 커서 손실이 남는다"는 하나의
패턴으로 수렴했고, v01_symmetric_matched는 **그 패턴과도 다른 세 번째(전체로는 네 번째)
독립 패턴**으로 나타났다 — 이름·트레이너가 v01_symmetric과 같다는 이유로 같은 패턴을 보일
것이라 가정하지 않고 실측으로 확인한 결과다.

## 공통 조건 (세 모델 전부 동일하게 적용)

- `confidence_thres = 0.001` (golden 쪽, `tools/debug/inf_conf001.py` 재사용 — 원본
  `basic_inference_v8`의 `confidence_thres=0.01` 하드코딩을 우회하는 override)
- AP용 IoU threshold: `0.50:0.95`, 10개 (`tools/debug/ultralytics_style_ap.py`가
  `torch.linspace(0.5, 0.95, 10)`로 고정, pycocotools 쪽은 기본 `Params.iouThrs` 동일)
- GT: `runs/_eval_scratch/ipu_gt` (6,152장, 세 모델 전부 각 모델의 기존 config로 동일 GT임을
  확인 — 가정하지 않음)
- dual-threshold NMS: `min_overlap_thres=0.6`, `union_overlap_thres=0.3` (각 모델의 기존
  board 재현 config와 동일값)
- 표준 단일-IoU NMS: `iou_thres=0.45` (`basic_postproc_v8` 기본값)
- board는 이미 온디바이스 NMS를 거친 최종 detection이므로 **추가 NMS 없이 그대로 평가**
  (`merge_board_dump.py`가 병합·카테고리 리매핑만 하고 필터링을 하지 않음은 기존 문서에서
  코드로 확인됨)
- 재사용한 기존 스크립트: `tools/debug/ultralytics_style_ap.py`, `tools/debug/inf_conf001.py`,
  `tools/customEval.py` — 전부 수정 없이 그대로 사용
- 기존 dual-NMS(conf=0.01) golden 평가·board 평가 결과가 있으면 재추론하지 않고 재사용

---

## 1. v01_dair

### 1.1 사용 파일

| 항목 | 경로 |
|---|---|
| Float ONNX | `runs/detect/BSD_F606/Rear_Truck/v01_dair/weights/best.onnx` |
| board 원본 덤프 | `runs/_eval_scratch/board_dumps/v01_dair/v01_dair_output.json` |
| board 병합 예측 | `runs/_eval_scratch/merged/v01_dair/ipu_pred_merged_v01_dair_3tensor.json` |
| 기존 dual-NMS golden 평가(conf=0.01) | `runs/_eval_scratch/configs/onnx/onnx_v01_dair_dualnms_cfg.yaml` → mAP50=0.734, mAP50-95=0.531 |
| 기존 board 평가(pycocotools) | `runs/_eval_scratch/configs/board/board_v01_dair_3tensor_cfg.yaml` → mAP50=0.669, mAP50-95=0.421 |

### 1.2 신규 추가 파일 (config만, 코드 없음)

- `runs/_eval_scratch/configs/onnx/onnx_v01_dair_conf001_dualnms_cfg.yaml`
- `runs/_eval_scratch/configs/onnx/onnx_v01_dair_stdnms_conf001_cfg.yaml`

### 1.3 실행 명령 (재현용)

```bash
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v01_dair_conf001_dualnms_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v01_dair_stdnms_conf001_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/board/board_v01_dair_3tensor_cfg.yaml

uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco \
  --pred_path runs/_eval_scratch/results/onnx_v01_dair_conf001_dualnms/onnx_v01_dair_conf001_dualnms_cfg/20260824_063047/coco_prediction.json
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco \
  --pred_path runs/_eval_scratch/results/onnx_v01_dair_stdnms_conf001/onnx_v01_dair_stdnms_conf001_cfg/20260824_063347/coco_prediction.json
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format ipu_test \
  --pred_path runs/_eval_scratch/merged/v01_dair/ipu_pred_merged_v01_dair_3tensor.json
```

### 1.4 결과 — 2×2 (mAP50-95)

| NMS \ Evaluator | pycocotools | ultralytics `ap_per_class` |
|---|---|---|
| **dual-threshold** | golden 0.741/0.533, board 0.669/0.421, abs=-0.112, **rel=-21.01%** | golden 0.7574/0.5521, board 0.7117/0.4620, abs=-0.0901, **rel=-16.32%** |
| **표준 단일-IoU** | golden 0.731/0.524, board 0.669/0.421, abs=-0.103, **rel=-19.66%** | golden 0.7465/0.5411, board 0.7117/0.4620, abs=-0.0791, **rel=-14.62%** |

(형식: golden mAP50/mAP50-95, board mAP50/mAP50-95, abs/rel은 mAP50-95 기준)

### 1.5 결과 — 2×2 (mAP50, 참고)

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold | 0.741→0.669 (**-9.72%**) | 0.7574→0.7117 (**-6.03%**) |
| 표준 단일-IoU | 0.731→0.669 (**-8.48%**) | 0.7465→0.7117 (**-4.66%**) |

### 1.6 축별 효과 분리 (mAP50-95 relative loss 기준)

- NMS 축: pycocotools 안에서 -21.01%→-19.66%(**1.35%p**), ultralytics 안에서
  -16.32%→-14.62%(**1.70%p**)
- Evaluator 축: dual-threshold 안에서 -21.01%→-16.32%(**4.69%p**), 표준 NMS 안에서
  -19.66%→-14.62%(**5.04%p**)

### 1.7 v01_dair 결론

1. evaluator에 따라 loss가 크게 달라진다 — 4.7~5.0%p 차이로, NMS 축 효과(1.35~1.70%p)보다
   뚜렷이 크다.
2. evaluator 축 효과가 NMS 축 효과를 압도하는 패턴(pycocotools > ultralytics, 4~6%p 대
   1~2%p)이 v01_symmetric과 동일한 크기로 재현된다.
3. **그러나 evaluator를 바꿔도 여전히 두 자릿수(%) 손실이 남는다**(-21.0%→-16.3%) — 이는
   v01_symmetric(-7.3%→-1.96%, 손실이 거의 사라짐)과 정반대다. "두 evaluator가 비슷해서
   symmetric만 차이가 크다"는 전제는 성립하지 않았다 — evaluator 간 격차(%p)는 오히려 두
   모델에서 비슷하고, 다른 것은 **evaluator 선택과 무관한 baseline 손실 자체의 크기**다.

---

## 2. v02

### 2.1 ONNX 구조 확인 (평가 전 필수 확인 사항)

v02의 `best.onnx`를 v01_symmetric의 `260821_best.onnx`와 직접 비교했다(`onnx.load` +
input/output shape, node 목록, opset 확인) — 추측이 아니라 실제 구조 비교:

| 항목 | v02 `best.onnx` | v01_symmetric `260821_best.onnx` |
|---|---|---|
| input | `images [1,3,320,320]` | 동일 |
| output | `output0 [1,5,2100]`(cls, sigmoid) + `433 [1,4,2100]`(box, raw distance) | **완전히 동일** |
| opset / producer | 13 / pytorch 2.5.1 | 동일 |
| node 개수 | 235 | 동일 |
| 마지막 15개 op 시퀀스 | `... Softmax, Conv, Constant, Reshape, Sigmoid` | **완전히 동일** |

**판단: `skip_postprocess=True`로 export됐다.** 그래프 안에 decode/NMS 모두 없음 — 이는
`docs/debug/20260806_rear_truck_v01_npu_quant_gap.md` §3의 v01/v02 export 방법
(`tools/export_custom.py -i 320 -y 320`, 기본값 `skip_postprocess=True`)과 일치. 기존
raw-output 전제 스크립트를 그대로 사용할 수 있다고 판단했다.

### 2.2 사용 파일

| 항목 | 경로 |
|---|---|
| Float ONNX | `runs/detect/BSD_F606/Rear_Truck/v02/weights/best.onnx` |
| board 원본 덤프 | `runs/_eval_scratch/board_dumps/v02/v02_output.json` |
| board 병합 예측 | `runs/_eval_scratch/merged/v02/ipu_pred_merged_v02_3tensor.json` |
| GT | `runs/_eval_scratch/ipu_gt` (6,152장 — `onnx_v02_dualnms_cfg.yaml`/`board_v02_3tensor_cfg.yaml`에서 직접 확인) |
| 기존 dual-NMS golden 평가(conf=0.01) | `runs/_eval_scratch/configs/onnx/onnx_v02_dualnms_cfg.yaml` → mAP50=0.748, mAP50-95=0.536 |
| 기존 board 평가(pycocotools) | `runs/_eval_scratch/configs/board/board_v02_3tensor_cfg.yaml` → mAP50=0.678, mAP50-95=0.435 |

board 덤프는 이미 온디바이스 NMS를 거친 최종 detection(형식이 v01_symmetric/v01_dair와
동일 — `{bbox, score, name, category, time}`)이므로 추가 NMS 없이 그대로 평가했다.

### 2.3 신규 추가 파일 (config만, 코드 없음)

- `runs/_eval_scratch/configs/onnx/onnx_v02_conf001_dualnms_cfg.yaml`
- `runs/_eval_scratch/configs/onnx/onnx_v02_stdnms_conf001_cfg.yaml`

### 2.4 실행 명령 (재현용)

```bash
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v02_conf001_dualnms_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v02_stdnms_conf001_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/board/board_v02_3tensor_cfg.yaml

uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco \
  --pred_path runs/_eval_scratch/results/onnx_v02_conf001_dualnms/onnx_v02_conf001_dualnms_cfg/20260824_064654/coco_prediction.json
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco \
  --pred_path runs/_eval_scratch/results/onnx_v02_stdnms_conf001/onnx_v02_stdnms_conf001_cfg/20260824_064849/coco_prediction.json
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format ipu_test \
  --pred_path runs/_eval_scratch/merged/v02/ipu_pred_merged_v02_3tensor.json
```

### 2.5 결과 — 2×2 (mAP50-95)

| NMS \ Evaluator | pycocotools | ultralytics `ap_per_class` |
|---|---|---|
| **dual-threshold** | golden 0.754/0.538, board 0.678/0.435, abs=-0.103, **rel=-19.14%** | golden 0.7707/0.5581, board 0.7183/0.4755, abs=-0.0826, **rel=-14.80%** |
| **표준 단일-IoU** | golden 0.750/0.531, board 0.678/0.435, abs=-0.096, **rel=-18.08%** | golden 0.7661/0.5493, board 0.7183/0.4755, abs=-0.0738, **rel=-13.44%** |

### 2.6 결과 — 2×2 (mAP50, 참고)

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold | 0.754→0.678 (**-10.08%**) | 0.7707→0.7183 (**-6.80%**) |
| 표준 단일-IoU | 0.750→0.678 (**-9.60%**) | 0.7661→0.7183 (**-6.24%**) |

### 2.7 축별 효과 분리 (mAP50-95 relative loss 기준)

- NMS 축: pycocotools 안에서 -19.14%→-18.08%(**1.06%p**), ultralytics 안에서
  -14.80%→-13.44%(**1.36%p**)
- Evaluator 축: dual-threshold 안에서 -19.14%→-14.80%(**4.34%p**), 표준 NMS 안에서
  -18.08%→-13.44%(**4.64%p**)

### 2.8 v02 결론

1. evaluator에 따라 loss가 크게 달라진다 — 4.3~4.6%p 차이로 NMS 축 효과(1.06~1.36%p)보다
   뚜렷이 크다. NMS 축보다 evaluator 축 효과가 크다는 패턴이 (v01_symmetric, v01_dair에 이어)
   세 번째 모델에서도 재현됐다.
2. **v02의 baseline loss는 symmetric이 아니라 dair 쪽에 훨씬 가깝다.** 같은 DAIR 트레이너
   계열(v01_dair, v02)은 evaluator를 바꿔도 큰 손실이 남고(-14.80%/-13.44%), symmetric
   트레이너(v01_symmetric)만 evaluator 효과로 손실이 사실상 사라진다 — evaluator 의존성
   자체는 트레이너와 무관하게 같은 크기로 나타나지만, 그 효과를 빼고 남는 baseline 손실의
   크기는 트레이너 계열(DAIR vs symmetric)에 달려 있다는 가설이 세 번째 데이터 포인트로
   뒷받침됐다.

---

## 3. v01_symmetric_matched

### 3.1 ONNX 구조 확인 (평가 전 필수 확인 사항)

`v01_symmetric_matched`의 `best.onnx`를 `v01_symmetric`의 `260821_best.onnx`와 직접
비교했다:

| 항목 | v01_symmetric_matched `best.onnx` | v01_symmetric `260821_best.onnx` |
|---|---|---|
| input | `images [1,3,320,320]` | 동일 |
| output | `output0 [1,5,2100]`(cls, sigmoid) + `433 [1,4,2100]`(box, raw distance) | **완전히 동일** |
| node 개수 | 235 | 동일 |
| 마지막 15개 op 시퀀스 | `... Softmax, Conv, Constant, Reshape, Sigmoid` | **완전히 동일** |

**판단: `skip_postprocess=True`.** decode(dist2bbox)/NMS 모두 그래프 밖 — 기존 raw-output
스크립트를 그대로 사용할 수 있다. (참고: 같은 weights 디렉토리에 `best_decoded.onnx`(단일
병합 출력 `[1,9,2100]`, 다른 세션 산출물)도 존재하나 이번 검증에는 사용하지 않았다 — 기존
`onnx_v01_symmetric_matched_dualnms_cfg.yaml`이 이미 `best.onnx`를 참조하고 있어 그대로
따랐다.)

### 3.2 사용 파일

| 항목 | 경로 |
|---|---|
| Float ONNX | `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_matched/weights/best.onnx` |
| board 원본 덤프 | `runs/_eval_scratch/board_dumps/v01_symmetric_matched/v01_symmetric_matched_output.json` |
| board 병합 예측 | `runs/_eval_scratch/merged/v01_symmetric_matched/ipu_pred_merged_v01_symmetric_matched_3tensor.json` |
| 기존 dual-NMS golden 평가(conf=0.01) | `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_dualnms_cfg.yaml` → mAP50=0.754, mAP50-95=0.547 |
| 기존 board 평가(pycocotools) | `runs/_eval_scratch/configs/board/board_v01_symmetric_matched_3tensor_cfg.yaml` → mAP50=0.672, mAP50-95=0.393 |

board 덤프는 이미 온디바이스 NMS를 거친 최종 detection이므로 추가 NMS 없이 그대로 평가했다.

### 3.3 신규 추가 파일 (config만, 코드 없음)

- `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_conf001_dualnms_cfg.yaml`
- `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_stdnms_conf001_cfg.yaml`

### 3.4 실행 명령 (재현용)

```bash
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_conf001_dualnms_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_matched_stdnms_conf001_cfg.yaml
uv run python3 tools/customEval.py runs/_eval_scratch/configs/board/board_v01_symmetric_matched_3tensor_cfg.yaml

uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco --pred_path <위 두 실행의 coco_prediction.json>
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format ipu_test \
  --pred_path runs/_eval_scratch/merged/v01_symmetric_matched/ipu_pred_merged_v01_symmetric_matched_3tensor.json
```

### 3.5 결과 — 2×2 (mAP50-95)

| NMS \ Evaluator | pycocotools | ultralytics `ap_per_class` |
|---|---|---|
| **dual-threshold** | golden 0.758/0.549, board 0.672/0.393, abs=-0.156, **rel=-28.42%** | golden 0.7753/0.5699, board 0.6973/0.4160, abs=-0.1539, **rel=-27.00%** |
| **표준 단일-IoU** | golden 0.752/0.541, board 0.672/0.393, abs=-0.148, **rel=-27.36%** | golden 0.7687/0.5593, board 0.6973/0.4160, abs=-0.1433, **rel=-25.62%** |

### 3.6 결과 — 2×2 (mAP50, 참고)

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold | 0.758→0.672 (**-11.35%**) | 0.7753→0.6973 (**-10.06%**) |
| 표준 단일-IoU | 0.752→0.672 (**-10.64%**) | 0.7687→0.6973 (**-9.29%**) |

### 3.7 축별 효과 분리 (mAP50-95 relative loss 기준)

- NMS 축: pycocotools -28.42%→-27.36%(**1.06%p**), ultralytics -27.00%→-25.62%(**1.38%p**)
- Evaluator 축: dual -28.42%→-27.00%(**1.42%p**), 표준 -27.36%→-25.62%(**1.74%p**)

mAP50 기준으로도 evaluator 축 효과는 ~1.3%p로 작다(3.7~4.1%p였던 v01_symmetric/v01_dair/v02
보다 뚜렷이 작음).

### 3.8 v01_symmetric_matched 결론

1. **이름/트레이너가 같은 v01_symmetric형이 아니다.** evaluator를 ultralytics로 바꿔도
   -25~28%대의 큰 손실이 그대로 남아, symmetric처럼 손실이 사라지거나 역전되는 패턴과
   정반대다.
2. **dair/v02형도 아니다.** dair/v02는 evaluator 효과가 4.3~5.0%p(mAP50-95 기준)로 뚜렷했지만,
   v01_symmetric_matched는 evaluator 효과가 1.4~1.7%p로 가장 작다 — 오히려 세 모델의 "NMS 축
   효과" 크기(0.7~1.7%p)에 더 가깝다.
3. **네 번째 독립 패턴이다**: NMS 축과 evaluator 축 두 효과가 모두 작고 비슷한 크기이며, 4개
   셀 전부 -25% 이상의 가장 큰 손실로 수렴한다. "이름이 비슷하면 같은 계열일 것"이라는
   가정이 틀렸음을 실측으로 확인한 사례다.
4. **기존 가설("evaluator 효과는 항상 4~6%p로 재현되고, baseline 손실 크기만 트레이너
   계열을 따른다")은 이 네 번째 데이터 포인트로 약화된다.** evaluator 의존성의 크기 자체가
   모델마다 다르게 나타날 수 있다는 뜻이며, "왜 v01_symmetric_matched만 evaluator 효과가
   작은가"는 새로운 질문으로 남는다.
5. **NMS 축 효과는 네 모델 모두에서 일관되게 작다(0.7~1.7%p)** — 이 부분만은 네 모델 전부에서
   유지되는 유일한 공통점이다.

---

## 4. 세 모델(및 네 모델 전체) 종합 결론

- **evaluator 축 효과가 NMS 축 효과보다 크다는 패턴은 v01_dair·v02에서 재현됐다**
  (4.3~5.0%p vs 1.06~1.70%p), 그러나 **v01_symmetric_matched에서는 재현되지 않았다**
  (1.4~1.7%p로 오히려 NMS 축과 비슷한 크기).
- **v01_dair와 v02는 사실상 같은 패턴이다** — 둘 다 DAIR 트레이너 계열이고, evaluator를
  바꿔도 두 자릿수(%) 손실이 남는다. baseline 손실의 절대 크기가 트레이너 계열(DAIR vs
  symmetric)을 따라간다는 가설이 이 둘로 뒷받침된다.
- **v01_symmetric_matched는 이름·트레이너가 v01_symmetric과 같은데도 정반대의 패턴을
  보인다** — 이름/트레이너 유사성만으로 evaluator 의존성 패턴을 예측할 수 없다는 것을
  실측으로 확인한 사례다. 네 모델 전체를 한 번에 비교하는 표는
  `docs/debug/20260824_v01_symmetric_eval_gap.md` §10/§10-1에 있다(원래
  v01_symmetric_matched 문서 §8/§8-1이었으나 그 문서로 옮김).
- **네 모델 전부에서 공통되는 유일한 것은 NMS 축 효과가 작다는 것(0.7~1.7%p)뿐이다.**
  evaluator 축 효과의 크기는 모델마다 다르며, 그 이유는 규명되지 않았다.

## 5. 아직 확인되지 않은 사항 (세 모델 공통)

- 세 모델 모두, pycocotools `evaluateImg`와 ultralytics `match_predictions`의 그리디 매칭
  알고리즘 자체를 라인 단위로 대조하지 않았다 — evaluator 차이의 진짜 알고리즘적 원인은
  네 모델 전부에서 여전히 미확정이다.
- "evaluator/NMS와 무관한 baseline quantization 손실이 왜 DAIR 계열(v01_dair, v02)에서 크고
  symmetric 계열(v01_symmetric)에서 작은가"는 이 문서들의 범위(evaluator/NMS 검증)를
  벗어나며, `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`의 DAIR vs symmetric
  트레이너 비교 조사와 같은 축의 질문이다.
- **왜 v01_symmetric_matched만 evaluator 효과가 작은지**는 전혀 조사되지 않았다 — 후보로
  생각해볼 수 있는 요소(detection 개수/score 분포, 클래스별 손실 패턴, 애초의 낮은 절대
  성능 자체)는 검토하지 않았다.
- 이번 세 검증 모두에서 quantization 설정 변경/재컴파일이나 추가 ablation을 수행하지
  않았고, 어느 evaluator가 "정답"인지도 판단하지 않았다.
- 네 모델만으로 "evaluator 의존성 크기가 무엇에 좌우되는가"에 대한 일반 결론을 내리기에는
  표본이 작다 — 추가 모델/조건에서의 재현 여부가 필요하다.
