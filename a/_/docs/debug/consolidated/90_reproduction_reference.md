# 재현 부록 — 공통 환경 · 평가 방법론 · 재현 명령 · 산출물 식별 · pattern-name 이슈

> 이 문서는 평상시 매번 읽을 필요는 없다 — 01~06 주제 문서가 필요할 때 이 문서의 특정 절을
> 링크한다. 여기 있는 정의는 **여러 주제 문서에서 반복 사용되는 공통 사실**만 모은 것이고,
> 각 주제 고유의 수치·표·실패사례는 해당 topic 문서에 있다(이 문서로 옮기지 않았다).
> 원문 출처: `20260810_npu_pattern_name_board.md`(PN, §5), 그 외 절은 01~06 각 topic 문서의
> "공통 조건"으로 반복 등장한 내용을 원문 근거와 함께 종합.

## 1. 공통 실행 환경

- **컨테이너**: 학습/평가/export는 `dair-train`(compose 서비스명, 이미지 내 `uv run
  python3 ...`로 실행). NPU(Novatek) 변환은 `npu-convert`(`dair-yolo/npu:latest`,
  `--chip 530`) 컨테이너에서 `run_compiler.sh <model_dir>` → `run_simulator.sh <model_dir>
  <test_image>` 순서. SigmaStar(IPU) 변환은 `dair-yolo/sigmastar:latest`를
  `/home/ncn/Workspace/jdh/sgs`에 bind-mount해 실행([06](06_sigmastar_conversion.md) §4
  참조, python3.6/3.7 심볼릭 링크 함정 주의).
- **공통 export 스크립트**: `tools/export_custom.py`(`CustomYOLO` 계열, `model_name=custom`
  또는 `dair` 체크포인트용, `--skip_postprocess` 기본값 True) / `tools/export_symmetric.py`
  (`SymmetricYOLO` 계열, `model_name=symmetric` 체크포인트용) — **어느 스크립트를 쓸지는
  체크포인트의 `model_class`(torch.load로 확인 가능: `SymmetricDetectionModel`/
  `CustomDetectionModel`/`DairDetectionModel`)로 결정하지, 파일명/폴더명으로 추측하지
  않는다**([01](01_rear_quantization_and_release.md) §7, [04](04_corner_reproduction.md)
  §1에서 실측 확인). 두 wrapper 모두 결국 표준 `Model.export()`로 귀결되므로, 같은
  `--skip_postprocess` 옵션이면 동일 ONNX를 만든다(bit-exact 확인 사례:
  [01](01_rear_quantization_and_release.md) §7). 표준 stock `ultralytics.nn.modules.
  head.Detect`(skip_postprocess 속성 자체가 없는 체크포인트, 예: Side_Truck)는 이 두
  스크립트 모두 no-op이라 **monkeypatch export**가 필요하다
  ([05](05_side_reproduction.md) §2).
- **공통 평가 스크립트**: `tools/customEval.py <cfg.yaml>` → `projects.metrics.evaluator.
  Evaluator`(pycocotools 기반). ultralytics 자체 엔진 비교용 진단 스크립트:
  `tools/debug/ultralytics_style_ap.py`(coco 형식 또는 board raw 병합 json을 읽어
  `ap_per_class`/`match_predictions`로 재계산), `tools/debug/inf_conf001.py`
  (`basic_inference_v8`의 `confidence_thres=0.01` 하드코딩을 `0.001`로 바꾼 override —
  config로 override 불가능해서 필요, 원본 미수정).
- **board dump 병합**: `runs/_eval_scratch/tools/merge_board_dump.py` — 이미지당 JSON
  array가 concatenated된 board 원본 덤프를 병합 + 카테고리명 리매핑(`MOTOR`→`motorcycle`
  등)만 수행, **추가 필터링/NMS/confidence threshold는 전혀 적용하지 않는다**(코드 확인) —
  즉 board raw dump는 이미 온디바이스 NMS를 거친 최종 detection 형식(`{bbox,score,name,
  category,time}`)이며, 이 문서 어디에서도 board dump에 추가 NMS를 적용한 적이 없다.
- **경로 정리(2026-08-12)**: Rear_Truck 관련 평가 config/결과 경로가 `runs/_eval_scratch/`
  아래로 재정리됐다 — 각 topic 문서에서 "재정리 후 경로"라고 표시된 것이 현재 유효 경로다.

## 2. 평가 방법론 — NMS 종류와 evaluator 종류

이 프로젝트 전체에서 반복 등장하는 두 개의 독립 축이다. **절대 하나로 뭉뚱그리지 말 것** —
아래 §2.3의 4-combo 표기(`dual+pycoco`, `dual+ultra`, `std+pycoco`, `std+ultra`)가 이 두
축의 조합을 가리키는 표준 표기다.

### 2.1 NMS 축

| | dual-threshold NMS(board 재현용) | 표준 단일-IoU NMS |
|---|---|---|
| 구현 | `projects/metrics/utils.py`의 `basic_postproc_v8_dual_nms`(`cnn_nt98530_post_process.cpp`의 `nms()/_rect_overap()` 포팅) | `basic_postproc_v8`(`cv2.dnn.NMSBoxes`) 또는 ultralytics `non_max_suppression` |
| 파라미터 | `min_overlap_thres=0.6`, `union_overlap_thres=0.3`(`min_overlap=inter/min(area_a,area_b)`, `union_overlap=inter/union`) | `iou_thres`(보통 0.45 또는 0.7 — 문서마다 다름, 각 topic 문서에서 명시한 값을 따를 것) |
| class 처리 | **class-aware**(클래스별로 개별 NMS) | **class-agnostic**(전체 detection 한 번에) |
| 용도 | Float ONNX를 board와 동일 조건으로 비교(apples-to-apples) | project 표준 val-split 평가, 또는 Rear 공식 acceptance 지표(§2.3) |
| 실측 효과 크기 | Rear_Truck 전 모델에서 NMS 축을 바꿔도 손실%에 미치는 영향은 **항상 작다(0.7~1.7%p)** — [01](01_rear_quantization_and_release.md) §6 | — |

### 2.2 Evaluator 축

| | pycocotools(`customEval.py`) | ultralytics(`ap_per_class`) |
|---|---|---|
| 구현 | 실제 `pycocotools.coco.COCO` + `customCOCOeval(COCOeval)`(`accumulate()`만 오버라이드, `Params`는 스톡 그대로) | `ultralytics.utils.metrics.ap_per_class` + `BaseValidator.match_predictions`(자체 그리디 매칭, pycocotools 미사용) |
| IoU threshold | `np.linspace(0.5,0.95,10)`(스톡 `Params.iouThrs`) | `torch.linspace(0.5,0.95,10)` — **둘 다 동일**(코드 확인, 원인 아님) |
| AP 적분 | 표준 101-point COCO 보간(스톡) | `compute_ap`, 동일하게 101-point 보간(주석 "101-point interp (COCO)") — **둘 다 동일**(원인 아님) |
| areaRng | 기본 4종 중 index 0("all") 고정 사용 | 없음(면적 구분 없이 전부 평가) — 차이점이나 실질 영향 미확인 |
| maxDets | 기본 `[1,10,100]` 중 index -1(=100) | 제한 없음 — **Rear_Truck 실측(6,152장 전수)으로 board 이미지당 detection 최대 60건(p99=42) 확인, 100 제한에 걸리는 사례 0건 → 원인에서 배제** |
| 매칭(TP/FP) 알고리즘 | `COCOeval.evaluateImg`(스톡, 그리디) | `match_predictions`(자체 그리디) — **둘 다 그리디이나 서로 다른 구현. 이 알고리즘 차이가 evaluator 간 절대 손실% 격차의 남은 유력 원인으로 추정되나, 라인 단위 직접 대조는 어느 topic 문서에서도 수행되지 않았다(미확정으로 남음)** |
| 실측 효과 크기 | — | Rear_Truck에서 NMS 축(0.7~1.7%p)보다 **훨씬 크다**(모델별로 1.4~6.0%p, 모델마다 다름 — "항상 일정한 크기"라는 가정은 성립하지 않음, [01](01_rear_quantization_and_release.md) §6) |

### 2.3 공식 acceptance 조합 (Rear_Truck에서 확정, 2026-09-02)

Rear_Truck 조사에서 **mAP50 + 표준 단일-IoU NMS + ultralytics evaluator**의 줄임 표기로
**"std+ultra"**를 쓴다(다른 조합은 `dual+pycoco`/`dual+ultra`/`std+pycoco`). 이 조합이
공식 acceptance 판정 기준으로 확정됐다([01](01_rear_quantization_and_release.md) §0 참조) —
**다른 topic(Front/Corner/Side)에는 이 acceptance 기준이 그대로 적용된다고 확인된 바
없다** — Front/Corner/Side 문서는 각자 다른 evaluator 조합(pycocotools 단독, ultralytics
단독, 또는 둘 다 병기)을 쓰므로, 그 문서의 표를 볼 때는 그 문서가 명시한 조합을 확인할 것.

### 2.4 모델(제품)별 GT 데이터셋 규모 — 혼동 방지용 일람

같은 제품이라도 문서마다 다른 부분집합/서로 다른 목적의 데이터셋을 가리킨다. 아래는
각 topic 문서에서 실제로 쓰인 규모다(장수만 나열, 상세 조건은 각 문서 참조):

| 제품 | 데이터셋 | 장수 | 용도 |
|---|---|---|---|
| Rear_Truck | 원본 val split | 7,894장 | project 표준 val(7,894장 전체, `DairDetectionValidator` 내부 검증) |
| Rear_Truck | Float+Board 공통 집합 | **6,152장** | apples-to-apples quantization loss 비교(§전체, [01](01_rear_quantization_and_release.md) §1) — 1,742장은 board 미처리(원인 불명, 추정 금지) |
| Front_Truck | Validation(NAS Release 스펙) | 6,076장 | Release 성능표, [02](02_front_reference_and_reproduction.md) §4 |
| Front_Truck | Validation(재검증, this-repo) | 6,076장/53,262 인스턴스 | [03](03_front_training_experiments.md) 학습 실험 재검증(§3) — Front_Truck 표기가 "6,075장"으로도 등장(±1장, 원문 그대로 병기, [02](02_front_reference_and_reproduction.md) §7) |
| Corner_Truck | Validation(전체, 미마운트로 평가 미실행) | 5,000장 | [04](04_corner_reproduction.md) §1 Level 6 |
| Side_Truck | CVD/Side validation | 14,267장/58,885 인스턴스 | repo 기록 PyTorch baseline과 비교, [05](05_side_reproduction.md) §5 |
| Side_Truck | BSD/Side/val(직접) | 11,001장/10,999 라벨 | 사용자 제공 "Original" 표와 비교, [05](05_side_reproduction.md) §7~8 |
| Side_Truck | BSD/Side/val_iputest | 5,943장/18,539 인스턴스 | 사용자 제공 "Board" 표와 비교, [05](05_side_reproduction.md) §6 |
| Side_Truck | IPU_TEST 리스트 서브셋 | 4,247장 | Original 표 재현 시도, [05](05_side_reproduction.md) §9~10 |
| Side_Truck | 실제 물리 보드 테스트 공통 이미지 | 4,248장 | [05](05_side_reproduction.md) §11 |

## 3. 재현 명령 템플릿 (공통 패턴)

각 topic 문서에 실제 실행된 전체 명령이 있다 — 여기서는 반복되는 패턴만 발췌.

```bash
# PT -> ONNX export (체크포인트의 model_class에 맞는 스크립트 선택, §1 참조)
docker compose exec dair-train bash -lc '
uv run python3 tools/export_custom.py -w <best.pt> -i <W> -y <H> --skip_postprocess
# 또는: tools/export_symmetric.py -w <best.pt> -i <W> -y <H> --skip_postprocess
'

# NVT compile + simulator (Novatek, 격리 디렉토리 사용 — 원본 NAS 산출물 절대 덮어쓰지 않음)
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_compiler.sh <model_dir_name>'
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_simulator.sh <model_dir_name> <test_image.jpg>'

# Float ONNX 평가 (pycocotools 엔진)
docker compose exec dair-train bash -lc '
uv run python3 tools/customEval.py <eval_cfg.yaml>
'

# 동일 예측을 ultralytics 엔진으로 재계산 (재추론 없음)
docker compose exec dair-train bash -lc '
uv run python3 tools/debug/ultralytics_style_ap.py --pred_format coco --pred_path <coco_prediction.json>
'
```

## 4. 산출물 식별 규칙

- **`nvt_model.bin`**: Novatek NPU 컴파일 산출물, 경로 패턴
  `novatek/convert/<model_dir>/nvtai_tool/output/<PATTERN_NAME>/sdk/nvt_model.bin`. 같은
  ONNX·같은 calibration·같은 config로 재컴파일해도 **byte 단위로 완전히 결정론적이지
  않다**(수십~수천 bytes 차이가 정상 범위로 관측됨 — [05](05_side_reproduction.md) §4/§8,
  섹션 크기/버퍼 오프셋 같은 내부 메타데이터가 실행마다 달라질 수 있음, 컴파일러 버전 자체는
  `gentool/profile.txt`의 `VERSION` 필드로 확인 가능).
- **`gen_config.txt`/`sim_config.txt`**: NVT 컴파일 설정 파일. `path/model_dir`,
  `path/ref_img_dir` 등 **path 관련 값만 격리 디렉토리에 맞게 바꾸고, 그 외 precision/
  preprocessing 값은 원본과 byte-identical로 유지**하는 것이 이 프로젝트의 공통 관례
  (모든 topic 문서에서 반복 확인). 일부 원본은 `${width}`, `${height}` 같은 미치환
  템플릿 상태로 존재하기도 한다([05](05_side_reproduction.md) §3, `gen-config-unsubstituted`
  이슈).
- **`model_dir`/pattern-name**: `novatek/convert/run_compiler.sh`가 `model_dir` 폴더명을
  **자동으로 대문자화**해서 `--pattern-name`으로 컴파일러에 넘긴다(`run_compiler.sh:33-34`).
  이 값의 대소문자/길이가 시뮬레이터 정확도(cosine/avg_diff)에는 전혀 영향을 주지 않지만,
  **실제 물리 보드의 인식 여부에는 영향을 준다** — 상세는 §5.
- **`layer_diff.txt` / `debug/golden_tensor/` / `debug/sim_layer/`**: Novatek 시뮬레이터가
  생성하는 레이어별 float(golden) vs NVT dequantized 비교 리포트(cosine+avg_diff, 171개
  레이어 전체) — `golden_tensor/<Tensor>.bin`은 순수 float 참조값, `sim_layer_<이미지>/
  float/<Tensor>.bin`은 dequantized 값, `sim_layer_<이미지>/fixed/`는 raw fixed-point(정수)
  값이다([01](01_rear_quantization_and_release.md) §3.3 RC§17-1에서 코드 근거로 확인).
  SigmaStar SDK의 대응물은 `DumpDebug/auto_dump_debug.sh` + `compareAccuracy`
  (MSE/COS/RMSE + 양자화 dtype까지 표시, [06](06_sigmastar_conversion.md) §6).
- **Board dump 파일**: 이미지당 JSON array가 줄바꿈 없이 이어붙은 형식(표준 JSON 파서로 한
  번에 안 읽힘 — `json.JSONDecoder().raw_decode()`를 반복 호출해 파싱). 이미 온디바이스
  NMS를 거친 최종 detection(`{bbox,score,name,category,time}`)이며, 추가 NMS를 적용하지
  않는다(§1). **stale-model 검증(중요)**: 서로 다른 모델의 board dump가 md5/detection
  내용이 우연히 100% 동일하면(잘못된 모델이 실제로 플래싱되지 않은 경우) 그 덤프는 무효로
  간주하고 재덤프를 요청한다 — jaccard overlap 0에 가까운지로 확인
  ([01](01_rear_quantization_and_release.md) 여러 곳에서 반복된 관례).

## 5. NPU pattern-name 대소문자/길이가 보드 인식에 영향 (2026-08-10)

> 원문: `20260810_npu_pattern_name_board.md`(전체). BSD_F606 Front_Truck v01 재현 컴파일
> (`v01_repro`, 이 repo에서 재학습한 것 — [03](03_front_training_experiments.md)의
> `v01_repro3`와는 별개의 재현 스트림, 혼동 금지)에서 발견됐으나, 원인 메커니즘 자체는
> 제품에 국한되지 않는 **NPU 컴파일/배포 공통 이슈**로 취급한다.

**배경**: `configs/releases/BSD_F606/Front_Truck/v01`을 재학습(`v01_repro`)한 뒤 이 repo에서
**detection 모델로는 첫 NPU 컴파일**을 시도(`novatek/convert/run_compiler.sh` 경유).
시뮬레이터의 `[OUT]` 태그 레이어(`Sigmoid__model.22_Sigmoid_output_0_Y`) cosine이
0.976~0.993으로 정상 범위였으나, **실제 NT98530 보드에 올려서 돌리면 바운딩박스가 전혀 안
그려졌다**(검출 자체가 안 됨).

**조사 과정**:
1. 가이드 문서(`/mnt/NAS_2/03_SDK/novatek/NPU/250205_NT98530_Converting_Guide_김훈민.pptx`,
   DRM 암호화라 직접 못 읽어 사용자가 슬라이드 내용을 전달)를 단계별로 대조.
2. **작업 구조 차이**: 가이드는 SDK 압축 해제본 자체의 `novaic/test-tutorial/nvtai_tool/`
   아래(`config/<project>/cnn25/`, `input/model/customer/<project>/`)에서 작업하는데, 이
   repo의 wrapper(`run_compiler.sh`)는 별도로 `novatek/convert/<model_dir>/nvtai_tool/`을
   만들어 쓴다 — SDK 밖, repo 안.
3. **`deploy_float.onnx` 명명 규칙이 반대**: 가이드는 `onnx2novaonnx_converter.py`를
   **수동으로 먼저** 실행해서(`--input <raw>.onnx --output deploy_float.onnx`) 나온
   **변환 후 결과물**을 `deploy_float.onnx`라 부른다. 이 repo의 wrapper는 raw export를
   그대로 `deploy_float.onnx`라 부르고, 컴파일러가 자동 변환해 `deploy.onnx`를 만든다 —
   **이름 규칙이 정반대.**
4. **자동 변환 시 추가 파라미터**: wrapper 자동 변환 `onnx2novaonnx_converter.py -v 4 -i
   deploy_float.onnx -o deploy.onnx -b 0 -is "{}" -os ""` vs 가이드 수동 변환
   `onnx2novaonnx_converter.py --input <raw>.onnx --output deploy_float.onnx`(추가 인자
   없음). raw export를 미리 수동 변환해 두면 컴파일러가 "This model is already a nova onnx
   model, skip the conversion process..."를 출력하고 자동 변환(및 `-b/-is/-os` 인자)을
   타지 않는다.
5. **`--use-heteroge 1` 플래그**: 가이드의 컴파일 명령에는 이 repo wrapper가 안 쓰는
   `--use-heteroge 1 --dump-onnx-en 1`이 붙어 있었다.
6. **1차 재현**: raw onnx를 수동으로 먼저 변환(4번 방식)하고 `--use-heteroge 1`까지 추가해
   재컴파일. 시뮬레이터 cosine은 이전과 **완전히 동일**(0.989518/avg_diff 0.000158, 소수점
   까지 일치) — 즉 수동/자동 변환도, `--use-heteroge`도 **시뮬레이터 수치에는 영향이
   없었다.** 그런데 **보드에서는 여전히 인식 안 됨**(`BSD_F606_FRONT_TRUCK_V01_REPRO_REAL`,
   wrapper가 폴더명을 자동 대문자화한 pattern-name).
7. **2차 재현 — 이름만 바꿈**: 같은 경로, 같은 수동 변환 결과물을 그대로 쓰되 pattern-name만
   가이드 예시와 동일한 규칙(소문자, 짧게: `bsd_f606_front_truck_v01`)으로 맞춰서 컴파일러를
   **직접** 호출(wrapper 우회, 자동 대문자화 회피). **보드에서 인식됨** — 바운딩박스가
   정상적으로 그려졌다.

**결론**: **NPU pattern-name의 대소문자·길이가 보드 인식 여부에 실제로 영향을 준다.** 호스트
시뮬레이터(`simulator.nvtai`)의 정확도 수치(cosine/avg_diff)는 pattern-name과 무관하게
완전히 동일하게 나오므로, **이 문제는 시뮬레이터로는 절대 검출할 수 없다.**
- 근본 메커니즘은 **미확인**이다(보드 쪽 모델 로더가 pattern-name을 고정 길이 필드로 다루거나
  대소문자를 구분해서 매칭할 가능성으로 추정하나, 벤더 문서로 확인되지 않음 — 이 관찰과
  메커니즘 자체를 혼동하지 말 것).
- `run_compiler.sh`가 `model_dir` 폴더명을 자동 대문자화해 `--pattern-name`으로 넘기는
  동작(`run_compiler.sh:33-34`)은 시뮬레이터 검증만 할 땐 무해하지만, **보드 배포용 최종
  컴파일에는 위험 요소**다.

**실전 권고**: (1) `model_dir` 이름을 **소문자, 짧게** 짓는다(가이드 예시
`00001_inception_v2`, `dummy_aiisp_4K` 컨벤션). (2) raw ONNX export를
`onnx2novaonnx_converter.py --input <raw>.onnx --output deploy_float.onnx`로 **수동으로
먼저 변환**해 두고 컴파일러를 돌린다(컴파일 로그에 "already a nova onnx model, skip the
conversion" 문구 확인 가능). (3) `run_compiler.sh`를 쓰면 자동 대문자화가 걸리므로, 보드
배포용은 `compiler.nvtai`를 직접 호출해 `--pattern-name`을 원하는 대소문자로 명시하는 편이
안전하다.

**Front_Truck `v01_repro` 보드 실측 quantization loss 수치**는 이 문서가 아니라
[02_front_reference_and_reproduction.md](02_front_reference_and_reproduction.md) §7에
있다(mAP50-95 -29.5%) — 이 절은 pattern-name 원인 분석만 다룬다.

## 6. 확인되지 않은 참조 (범위 밖)

아래는 원문에서 언급되지만 이번 통합 작업의 입력 범위(18개 문서) 밖에 있어 확인할 수 없는
참조다 — 삭제하지 않고 "현재 입력 범위에서 확인 불가"로 표시한다:

- `docs/specs/qat_mixed_precision.md`, `docs/guides/rear_truck_train_validate_nvt_pipeline.md`,
  `docs/decisions.md`(D-006, D-010, D-019 등), `docs/glossary.yaml` — 전부 이 통합 작업의
  입력 파일 목록에 없다.
- `configs/experiments/BSD_F606/Rear_Truck/README.md`, `configs/releases/BSD_F606/
  Front_Truck/v01/MANIFEST.yaml`, `20260729_config_canonicalization.md`(config
  canonicalization 문서, 이번 통합 범위에서 제외된 6개 파일 중 하나) — 원문에서 인용되지만
  이 통합본의 대상이 아니다.
- `standard/validation.md`(Front_Truck) — 원문에서 참조되나 이 통합 작업의 입력 파일이
  아니다.
