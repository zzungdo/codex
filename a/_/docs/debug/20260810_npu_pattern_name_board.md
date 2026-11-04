# NPU pattern-name 대소문자/길이가 보드 인식에 영향 — BSD_F606 Front_Truck v01 재현 컴파일

2026-08-10 (KST)

## 배경

`configs/releases/BSD_F606/Front_Truck/v01`을 재학습(`v01_repro`)한 뒤, 이 repo에서
**detection 모델로는 첫 NPU 컴파일**을 시도했다 (`novatek/convert/run_compiler.sh` 경유).
시뮬레이터(`run_simulator.sh`)의 `[OUT]` 태그 레이어(`Sigmoid__model.22_Sigmoid_output_0_Y`)
cosine이 0.976~0.993으로 나와 정상 범위로 보였으나, **실제 NT98530 보드에 올려서 돌리면
바운딩박스가 전혀 안 그려졌다** (검출 자체가 안 됨).

## 조사 과정

1. **가이드 문서 확인** — `/mnt/NAS_2/03_SDK/novatek/NPU/250205_NT98530_Converting_Guide_김훈민.pptx`.
   DRM 암호화(`NX5200803SECURE FILE`)라 직접 읽지 못해, 사용자가 슬라이드 내용을 직접
   전달받아 단계별로 따라갔다.
2. **작업 구조 차이 발견** — 가이드는 SDK 압축 해제본 자체의
   `novaic/test-tutorial/nvtai_tool/` 아래에서 작업한다 (`config/<project>/cnn25/`,
   `input/model/customer/<project>/`). 이 repo의 wrapper(`run_compiler.sh`)는 별도로
   `novatek/convert/<model_dir>/nvtai_tool/`을 만들어 쓴다 — SDK 밖, repo 안.
3. **`deploy_float.onnx` 명명 규칙이 반대** — 가이드는 `onnx2novaonnx_converter.py`를
   **수동으로 먼저** 실행해서(`--input <raw>.onnx --output deploy_float.onnx`) 나온
   **변환 후 결과물**을 `deploy_float.onnx`라고 부른다. 이 repo의 wrapper는 raw export를
   그대로 `deploy_float.onnx`라고 부르고, 컴파일러가 알아서 자동 변환해 `deploy.onnx`를
   만든다 — **이름 규칙이 정반대**였다.
4. **자동 변환 시 추가 파라미터 확인** — 컴파일 로그를 대조하면:
   - wrapper 자동 변환: `onnx2novaonnx_converter.py -v 4 -i deploy_float.onnx -o deploy.onnx -b 0 -is "{}" -os ""`
   - 가이드 수동 변환: `onnx2novaonnx_converter.py --input <raw>.onnx --output deploy_float.onnx` (추가 인자 없음)
   raw export를 미리 수동 변환해 두면 컴파일러가 `"This model is already a nova onnx model,
   skip the conversion process..."`를 출력하고 자동 변환(및 `-b/-is/-os` 인자)을 타지 않는다.
5. **`--use-heteroge 1` 플래그** — 가이드의 컴파일 명령에는 이 repo wrapper가 안 쓰는
   `--use-heteroge 1 --dump-onnx-en 1`이 붙어 있었다.
6. **1차 재현** — repo 안에서 raw onnx를 수동으로 먼저 변환(4번 방식)하고
   `--use-heteroge 1`까지 추가해서 재컴파일. 시뮬레이터 cosine은 이전과 **완전히 동일**
   (0.989518 / avg_diff 0.000158, 소수점까지 일치) — 즉 수동/자동 변환도, `--use-heteroge`도
   **시뮬레이터 수치에는 영향이 없었다**. 그런데 **보드에서는 여전히 인식 안 됨**
   (`BSD_F606_FRONT_TRUCK_V01_REPRO_REAL`, wrapper가 폴더명을 자동 대문자화한 pattern-name).
7. **2차 재현 — 이름만 바꿈** — 같은 repo 경로, 같은 수동 변환 결과물을 그대로 쓰되
   pattern-name만 가이드 예시와 동일한 규칙(소문자, 짧게: `bsd_f606_front_truck_v01`)으로
   맞춰서 컴파일러를 **직접** 호출(wrapper 우회, 자동 대문자화 회피). **보드에서 인식됨** —
   바운딩박스가 정상적으로 그려졌다.

## 결론

**NPU pattern-name의 대소문자·길이가 보드 인식 여부에 실제로 영향을 준다.** 호스트
시뮬레이터(`simulator.nvtai`)의 정확도 수치(cosine/avg_diff)는 pattern-name과 무관하게
완전히 동일하게 나오므로, **이 문제는 시뮬레이터로는 절대 검출할 수 없다.**

- 근본 메커니즘은 미확인이다 (보드 쪽 모델 로더가 pattern-name을 고정 길이 필드로
  다루거나 대소문자를 구분해서 매칭할 가능성으로 추정하나, 벤더 문서 확인 안 됨).
- `novatek/convert/run_compiler.sh`가 `model_dir` 폴더명을 자동 대문자화해서
  `--pattern-name`으로 넘기는 동작(`run_compiler.sh:33-34`)은 시뮬레이터 검증만 할 땐
  무해하지만, **보드 배포용 최종 컴파일에는 위험 요소**다.

## 실전 권고

보드에 실제로 올릴 `.bin`을 만들 때는:
1. `model_dir` 이름을 **소문자, 짧게** 짓는다 (가이드 예시 `00001_inception_v2`,
   `dummy_aiisp_4K`와 같은 컨벤션).
2. raw ONNX export를 `onnx2novaonnx_converter.py --input <raw>.onnx --output
   deploy_float.onnx`로 **수동으로 먼저 변환**해 두고 컴파일러를 돌린다 (컴파일 로그에
   "already a nova onnx model, skip the conversion" 문구가 나오는지로 확인 가능).
3. `run_compiler.sh`를 쓰면 자동 대문자화가 걸리므로, 보드 배포용은 `compiler.nvtai`를
   직접 호출해 `--pattern-name`을 원하는 대소문자로 명시하는 편이 안전하다.

## 보드 실측 mAP — 양자화 손실 정량화

pattern-name을 고치고 보드에서 정상 인식된 뒤(`bsd_f606_front_truck_v01`, repo 안),
사용자가 board 추론 결과(`output.json`, val set 6,076장 전체 — 이미 디코딩된
bbox+score+category)를 `novatek/convert/bsd_f606_front_truck_v01/analysis/`에 올려서
GT와 직접 비교했다.

**전처리 조건 확인**: board의 bbox는 **crop 로컬 좌표**(1280x1024, x/y 0부터 시작)로 나온다
— GT(원본 1920x1080)와 맞추려면 **x에 +320**(ROI x-offset)을 더해야 한다. 여러 박스로
수 픽셀 이내 일치 확인.

**golden 비교 기준 재검토**: 처음엔 MANIFEST의 기존 `pytorch_val_baseline`(640x480, 학습
해상도)과 비교했으나, 보드는 **640x512**(배포 해상도)로 추론했으므로 조건이 다르다
(이미 알려진 `train-deploy-size-mismatch` 이슈와 같은 축). 공정한 비교를 위해
`best.onnx`(raw export, float, onnxruntime)로 **동일한 640x512 전처리**(crop
320,0,1280,1024 → resize 640x512, BGR, [0,255])로 val set 전체를 새로 추론해서
golden@640x512 를 만들었다.

**첫 시도 실패 → 버그 발견**: `output0`(1,10,6720)의 채널 0~3을 golden_tensor 비교에
썼던 raw dfl distance로 착각해서 `dist2bbox`(anchor 기반) 디코딩을 적용했더니 전부
mAP=0. 확인해보니 **`output0`은 이미 그래프 안에서 dist2bbox+stride 를 거친 xywh
(local pixel) 좌표였다** — golden_tensor 덤프(`Conv__model.22_dfl_conv_Conv_output_0_Y`)는
그래프 **중간** 텐서라 raw distance가 맞지만, 그래프 **최종 출력**은 이미 디코딩된
box다. 채널 0~3을 그냥 `[cx,cy,w,h]`로 직접 해석하도록 고치니 GT와 수 픽셀 오차로
일치했다. (board 출력도 이미 디코딩된 형태였던 이유가 이걸로 설명된다 — 그래프 자체가
box decode까지 포함한 채 컴파일된다.)

**GT 필터링**: 보드가 못 본 ROI 밖 객체를 "놓친 것"으로 잘못 셀 수 있어, GT box 중심이
ROI(320,0,1600,1024) 안에 있는 것만 사용 (전체 GT 61,587개 → 필터 후 50,371개, MANIFEST의
53,262와 비슷한 규모).

### 결과 (val 6,075장, 640x512 동일 조건)

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| golden (float, `best.onnx`) | 0.806 | 0.697 | 0.782 | 0.567 |
| 보드 (NPU 양자화) | 0.737 | 0.600 | 0.655 | 0.400 |
| 손실 | −0.069 | −0.097 | **−0.127 (−16.2%)** | **−0.167 (−29.5%)** |

클래스별로는 **person·traffic_light 손실이 가장 크다**(mAP50 −0.19/−0.18) — 작은/미세한
형태의 객체가 양자화에 더 취약한 전형적 패턴. car/truck/bus(큰 물체)는 상대적으로 손실이
적다(−0.08 안팎).

**미해결**: 이 실측은 `gen_config.txt`가 원본 v01 그대로(`calib_mode=4`, `in_hp` 전부
0)인 상태에서 나온 수치다. 가이드가 기본값으로 제시했던 `calib_mode=1` + `in_hp` 전부 1 +
`spec_demarc_tensors` 지정(§ 위 조사 과정 참조, 이번 컴파일에는 미적용)을 적용하면 손실이
줄어들지 아직 확인 안 됨 — 다음 사이클에서 시도 예정.

**분석 스크립트**: `novatek/convert/bsd_f606_front_truck_v01/analysis/eval_board_vs_gt.py`,
`eval_golden_640x512.py` (둘 다 `novatek/convert/` 하위라 gitignore 대상, repo에는
안 올라감 — 재현하려면 이 파일 경로 참조).

## 참조

- `configs/releases/BSD_F606/Front_Truck/v01/MANIFEST.yaml` — 원본 v01의
  `sim-config-foreign-calib` 이슈(이번 재현에선 NAS_1 실측 calibration으로 대체)
- `novatek/convert/run_compiler.sh`, `run_simulator.sh`
- SDK 내 `novaic/test-tutorial/nvtai_tool/` (가이드가 참조하는 실제 작업 구조,
  `~/Workspace/jdh/novatek_sdk/novatek/tool/release/release/ai_tool/novatek/novaic/`)
