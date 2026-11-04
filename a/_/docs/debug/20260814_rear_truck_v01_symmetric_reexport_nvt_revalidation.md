# Rear_Truck v01_symmetric ONNX 재-export + NVT 재검증 (2026-08-14)

기존 문서 [`20260806_rear_truck_v01_npu_quant_gap.md`](20260806_rear_truck_v01_npu_quant_gap.md)에 기록된
Rear_Truck v01_symmetric 결과(Float mAP50-95=0.561, NVT/Board mAP50-95=0.521, quant loss -7.1%)를,
기존 결론을 전제하지 않고 처음부터 새 격리 디렉토리에서 재-export → 재검증한 기록. 기존
`best.pt`/`best.onnx`/NVT 산출물/board dump는 어느 것도 수정하지 않았다.

## 0. 대상 체크포인트

`runs/detect/BSD_F606/Rear_Truck/v01_symmetric/weights/best.pt` (sha256
`6ecedc40d07e889830108245c2282c6965c45f68df3e7655c850883313b6d298`)를
`runs/debug/rear_truck_v01_symmetric_reexport/source_best.pt`로 읽기 전용 복사 후 확인:

- `checkpoint_keys`: date/version/license/docs/epoch/best_fitness/**model**/ema/updates/optimizer/scaler/
  train_args/train_metrics/train_results/git
- `model_present=True`, `ema_present=False` — `model` 키만 존재
- `model_info.model_class = SymmetricDetectionModel`, `nc=5`, `num_state_dict_keys=355`
- `train_args_model_field = projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml`

체크포인트 클래스가 `SymmetricDetectionModel`이므로 "관습적으로" 맞는 export 스크립트는
`tools/export_symmetric.py`다. 기존 문서는 `tools/export_custom.py -i 320 -y 320`을 사용했다고
기록되어 있어 — 이 불일치가 재검증 대상.

## 1. 재-export

```
uv run python3 tools/export_symmetric.py -i 320 -y 320 --skip_postprocess \
    --weights runs/debug/rear_truck_v01_symmetric_reexport/source_best.pt
```

결과: `runs/debug/rear_truck_v01_symmetric_reexport/exported_from_best.onnx`
(opset 13, input `images[1,3,320,320]`, output `output0[1,5,2100]` + `433[1,4,2100]`).
`onnx.checker.check_model()` PASS, NaN/Inf 없음.

## 2. 기존 `best.onnx`와의 비교 (Level 1-3)

`tools/debug/reference_compare_graph_and_initializers.py`, `reference_compare_raw_output.py`로
기존 `runs/detect/BSD_F606/Rear_Truck/v01_symmetric/weights/best.onnx`와 직접 비교:

| Level | 결과 |
|---|---|
| Graph 구조 (op_type/attr/IO count) | **node_diffs_count = 0** (235 node 전체 동일) |
| Initializer | **127/127 exact_identical** (different=0) |
| Raw output (CPU, 10개 고정 입력: zeros/ones/half/random/impulse/실제 이미지 5장) | **전 텐서 bit-exact** (`output0`, `433` 모두 `max_abs_diff=0.0`) |

파일 SHA256은 다르다 (`exported_from_best.onnx`=`ed4d30a7...` vs `best.onnx`=`605d11fd...`) — 이는
export 실행마다 달라지는 파일 메타데이터(노드 이름의 autogen suffix 등) 때문이며, 실제 파라미터/그래프
구조/추론 결과는 완전히 동일하다.

**결론: `tools/export_symmetric.py --skip_postprocess`와 `tools/export_custom.py`(기존 문서가
사용한 방법)는 이 체크포인트에 대해 동일한 ONNX를 생성한다.** 두 wrapper(`SymmetricYOLO`/`CustomYOLO`)의
`.export()`가 결국 동일한 `Model.export()` 호출로 귀결되기 때문 — 클래스 불일치는 이 case에서
실질적 영향이 없었다. 기존 Float ONNX baseline은 신뢰할 수 있다.

## 3. Float ONNX mAP 재측정

기존 config `runs/_eval_scratch/configs/onnx/onnx_v01_symmetric_dualnms_cfg.yaml`을 복제해
`model_path`/`save_dir`만 새 경로로 교체
(`runs/debug/rear_truck_v01_symmetric_reexport/onnx_v01_symmetric_reexport_dualnms_cfg.yaml`),
GT(6,152장)/전처리/dual-threshold NMS(`min_overlap_thres=0.6, union_overlap_thres=0.3`)는 완전히 동일하게 유지.

```
uv run python3 tools/customEval.py runs/debug/rear_truck_v01_symmetric_reexport/onnx_v01_symmetric_reexport_dualnms_cfg.yaml
```

| Metric | 기존 baseline (20260806) | 재측정 |
|---|---|---|
| mAP50 | 0.774 | **0.774** |
| mAP50-95 | 0.561 | **0.561** |
| mAP75 | (미기록) | 0.606 |

Level 1-3의 bit-exact 결과와 정확히 일치하는 결과 — 재측정값이 기존 baseline과 완전히 동일하다.

## 4. NVT 재변환 (격리 디렉토리)

`novatek/convert/rear_truck_v01_symmetric_320_reexport_20260814/`를 기존
`rear_truck_v01_symmetric_320/`과 동일한 레이아웃으로 새로 생성:

- `nvtai_tool/input/model/customer/REAR_TRUCK_V01_SYMMETRIC_320_REEXPORT_20260814/deploy_float.onnx`
  = 위 2번에서 재-export한 onnx
- calibration 이미지(940장, `REAR_TRUCK_V01_320/video_calib_jpg`)와 test 이미지
  (`REAR_TRUCK_V01_SYMMETRIC_320/jpg`, 500장)는 기존 `rear_truck_v01_320/nvtai_tool/input/data/`의
  실제 파일을 심볼릭 링크로 참조 — 원본 파일은 전혀 건드리지 않음(단, 시뮬레이터가 자체적으로
  `test_img_list.txt`를 덮어쓰므로, 공유 디렉토리를 통째로 심볼릭 링크하지 않고 각 파일 단위로
  개별 심볼릭 링크 + `test_img_list.txt`만 로컬 복사해 원본 오염을 방지했다)
- `gen_config.txt`/`sim_config.txt`는 `path/model_dir`(과 sim_config의 `path/out_dir`/데이터 경로)만
  새 pattern 이름으로 바꾸고 나머지 모든 압축/precision/preprocessing 옵션은 byte-identical로 복제
  (`diff` 확인 완료 — `spec_demarc_tensors`, `calib_mode=4`, `percentile=0.999`, `cal_refine_iter=5`,
  `cle_en=1` 등 전부 동일).

### 4-1. Compiler 실행 결과

```
./run_compiler.sh rear_truck_v01_symmetric_320_reexport_20260814
```

| 항목 | 기존 baseline | 재변환 |
|---|---|---|
| Latency | 6.368ms | **6.36765ms** |
| DRAM | 36.54MB (36,540,852 bytes) | **36,540,852 bytes** |
| Model size | 4.71MB | **4,711,336 bytes (4.71MB)** |

### 4-2. Simulator 실행 결과 (동일 테스트 이미지, `KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC1_PGL_DVR_NC22015_0126.jpg`)

| Tensor | 기존 baseline cosine | 재변환 cosine |
|---|---|---|
| Softmax(dfl) | 0.9850 | **0.984977** |
| Sigmoid[OUT] | 0.9824 | **0.982404** |
| dfl_conv[OUT] | 0.9933 | **0.993315** |

Compile 통계, cosine 유사도 모두 기존 문서 값과 사실상 동일(소수 4째 자리까지 일치) — NVT 변환
파이프라인도 기존 baseline을 그대로 재현한다.

## 5. Board(NPU 실기) 검증 — 미완료, 물리 하드웨어 필요

기존 문서의 Board mAP(0.751/0.521, quant loss -7.1%)는 `사용자가 실제 보드에서 직접 추론`한 덤프
(`runs/_eval_scratch/board_dumps/v01_symmetric/v01_symmetric_output.json`, 18MB)를
`merge_board_dump.py`로 병합 후 `tools/customEval.py`로 평가한 결과다. 이 세션(Claude Code)은
물리 Novatek NPU 보드에 접근할 수 없어 이 단계는 직접 재현이 불가능하다.

Step 0-4(export → Float ONNX mAP → NVT compile → NVT simulator cosine)까지 전 단계가 기존 baseline과
사실상 bit-exact로 일치했으므로, Board mAP도 동일하게 재현될 것으로 **강하게 추정**되지만 실측 없이는
확정할 수 없다. 재현하려면:

1. 새로 컴파일된 NVT 모델(`novatek/convert/rear_truck_v01_symmetric_320_reexport_20260814/nvtai_tool/output/REAR_TRUCK_V01_SYMMETRIC_320_REEXPORT_20260814/sdk/nvt_model.bin`)을 보드에 탑재
2. 기존과 동일한 6,152장 검증셋에 대해 보드 추론 실행, JSON 덤프
3. 덤프를 `runs/_eval_scratch/board_dumps/v01_symmetric_reexport/`에 저장 (기존 `v01_symmetric/` 디렉토리는 건드리지 않음)
4. `merge_board_dump.py` → `runs/_eval_scratch/merged/v01_symmetric_reexport/`
5. 기존 `board_v01_symmetric_3tensor_cfg.yaml`을 복제해 `pred.path`/`save_dir`만 교체 후 `customEval.py` 실행

## 6. 종합 비교표

| | 기존 baseline (20260806) | 재현 (20260814) | 판정 |
|---|---|---|---|
| Export 스크립트 | `tools/export_custom.py` | `tools/export_symmetric.py --skip_postprocess` | 스크립트는 다르지만 **동일 ONNX 산출** (Level 1-3 bit-exact) |
| Float ONNX mAP50 / mAP50-95 | 0.774 / 0.561 | 0.774 / 0.561 | **완전 재현** |
| NVT Latency / DRAM / Model size | 6.368ms / 36.54MB / 4.71MB | 6.368ms / 36.54MB / 4.71MB | **완전 재현** |
| NVT cosine (Softmax/Sigmoid/dfl_conv) | 0.9850 / 0.9824 / 0.9933 | 0.984977 / 0.982404 / 0.993315 | **완전 재현** |
| Board mAP50 / mAP50-95 | 0.751 / 0.521 | (미측정 — 물리 보드 필요) | **보류** |
| Float→NVT quant loss | -7.1% | (미측정) | **보류** |

## 7. 판단 질문에 대한 답

1. **`tools/export_custom.py`로 export한 기존 Float ONNX baseline이 여전히 유효한가?**
   → **YES.** 체크포인트 클래스는 `SymmetricDetectionModel`이지만, "class-correct"한
   `tools/export_symmetric.py`로 재-export해도 initializer/raw output이 bit-exact로 동일하다.
   두 wrapper의 `.export()`가 동일한 `Model.export()`로 귀결되기 때문.

2. **재-export한 ONNX가 기존 NVT 변환 결과(compile stats, cosine)를 재현하는가?**
   → **YES.** Latency/DRAM/Model size, 3개 핵심 tensor의 cosine 유사도 전부 기존 값과
   소수 4자리까지 일치.

3. **Float→NVT quant loss(-7.1%)가 재현되는가?**
   → **미확인 (물리 보드 필요).** Export부터 NVT compile/simulator까지 모든 단계가 bit-exact/거의
   bit-exact로 재현되었으므로 재현될 것으로 추정되나, 실제 board mAP 실측 없이 확정할 수 없다.
