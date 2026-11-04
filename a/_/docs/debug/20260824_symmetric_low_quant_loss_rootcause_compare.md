# v01_symmetric 저손실 원인 전체 비교 조사 — 학습/모델/ONNX/NVT 파이프라인 전체 비교

- **작성**: 2026-08-24 (KST)
- **목적**: 왜 `v01_symmetric`만 NVT/Board 변환 후 성능 손실이 매우 작고,
  `v01_symmetric_matched`와 `v02`는 큰 손실이 발생하는지 원인을 찾는다. evaluator
  선택이 결과를 흔드는 현상은 이미 별도로 확인됐다(`20260824_v01_symmetric_eval_gap.md`,
  `20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`) — 이번 조사는 **동일
  evaluator에서도 모델별 Board degradation이 크게 다른 이유**를 찾는 것이다.
- **선행 문서**: `docs/debug/20260824_v01_symmetric_eval_gap.md`,
  `docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`,
  `docs/debug/20260806_rear_truck_v01_npu_quant_gap.md` — 전부 미수정.
- **범위**: 학습/모델 → PT → ONNX export → ONNX 출력 특성 → NVT converting/quantization →
  Board/postprocess 전체 파이프라인. NVT 재컴파일·재학습·quantization ablation·기존 파일
  수정은 하지 않았다 — 전부 기존 산출물 분석 + 소규모 진단 스크립트(임시, 세션 종료 시 삭제)로만
  확인했다.

## 1. 현재 성능 기준 고정

pycocotools(customEval.py)와 ultralytics(`ap_per_class`) 기준, dual-threshold NMS
(board 재현 조건), conf=0.001, 6,152장 — 두 evaluator-gap 문서에서 이미 확정된 값을 그대로
재사용(재평가 없음):

| Model | Float mAP50-95 (pycoco) | Board mAP50-95 (pycoco) | loss (pycoco) | Float mAP50-95 (ultra) | Board mAP50-95 (ultra) | loss (ultra) |
|---|---:|---:|---:|---:|---:|---:|
| v01_symmetric | 0.562 | 0.521 | **-7.30%** | 0.5864 | 0.5749 | **-1.96%** |
| v01_symmetric_matched | 0.549 | 0.393 | **-28.42%** | 0.5699 | 0.4160 | **-27.00%** |
| v02 | 0.538 | 0.435 | **-19.14%** | 0.5581 | 0.4755 | **-14.80%** |

**어느 evaluator로 봐도 순위는 동일하다**: v01_symmetric ≪ v02 < v01_symmetric_matched
(손실 크기 기준). 즉 이번 조사에서 찾아야 할 차이는 evaluator와 무관하게 존재하는,
모델 자체의 특성이다.

## 2. 세 모델 계보/학습 조건 비교

exp.yaml(정본 설정) + `runs/detect/BSD_F606/Rear_Truck/<model>/args.yaml`(실제 as-run 기록,
우선 사용) 기준. args.yaml은 ultralytics가 학습 시작 시 실제 적용된 전체 하이퍼파라미터를
자동 기록한 것이라 exp.yaml보다 신뢰도가 높다 — 이번 조사는 args.yaml을 우선했다.

| 항목 | v01_symmetric | v01_symmetric_matched | v02 |
|---|---|---|---|
| trainer class | `SymmetricTrainer` | `SymmetricTrainer` | `DairDetectionTrainer` |
| model class (checkpoint) | `SymmetricDetectionModel` | `SymmetricDetectionModel` | `DairDetectionModel` |
| model yaml | `projects/src/cfg/models/symmetric_v8/symmetric_yolov8.yaml` | 동일 | 동일 |
| dataset config | `configs/datasets/bsd_rear_truck.yaml`(이 PC에 파일 없음, 다른 PC에서 학습) | 동일 | `configs/datasets/1t_rear_cnr_postprocess.yaml` |
| dataset/split 실제 동일 여부 | v01과 동일 데이터/split(사용자 확인, 파일로 재검증 불가) | v01_dair와 동일(사용자 확인) | v01과 동일(코드로 확인 가능한 실제 경로) |
| pretrained | `true`, ultralytics 기본 COCO 자동 다운로드(명시 경로 없음, exp.yaml 코멘트로 확인) | 동일 | 동일(명시 경로 없음) |
| imgsz | 320 | 320 | 320 |
| **batch** | **48** | 32 | 32 |
| epochs (patience) | 500 (15) | 500 (15) | 500 (15) |
| best epoch (patience 종료) | 407 (results.csv 마지막 행, best≈392) | 222 (best≈207) | 333 (best≈318) |
| **optimizer** | **SGD** | AdamW | AdamW |
| **lr0 / lrf** | **0.01 / 0.01** | 0.001 / 0.01 | 0.001 / 0.01 |
| **cos_lr** | **true** | false | false |
| momentum | 0.937 | 0.937 | 0.937 |
| weight_decay | 0.0005 | 0.0005 | 0.0005 |
| warmup_epochs/momentum/bias_lr | 3.0 / 0.8 / 0.1 | 동일 | 동일 |
| seed / deterministic | 0 / true | 0 / true | 0 / true |
| box loss weight | 7.5 | 7.5 | 7.5 |
| **cls loss weight** | **1.5** | **0.5**(ultralytics 기본값, exp.yaml에 미명시) | **0.5**(동일) |
| dfl loss weight | 1.5 | 1.5 | 1.5 |
| mosaic / close_mosaic | 0.3 / 10 | 동일 | 동일 |
| mixup / copy_paste | 0.0 / 0.0 | 동일 | 동일 |
| bgr / flipud / fliplr | 1.0 / 0.0 / 0.0 | 동일 | 동일 |
| dair_params (DAIR dual-forward) | 없음 | 없음(symmetric 트레이너는 크래시로 사용 불가 — `20260806_...md` §"v01_symmetric_matched ablation" §1) | `dair_method_mode=or, λ=0.1, γ=0.5` (있음, 코드상 활성) |
| TensorSpeakerOverlayAug | 해당 없음(symmetric은 DAIR augmented-view 자체 없음) | 해당 없음(동일 이유) | **현재 코드 상태: 비활성**(주석 처리, `projects/src/models/yolo/detect/train/dair.py:139`) — v02의 원래 ablation 목적 |

**"matched"가 정확히 무엇을 맞춘 것인가**: `v01_symmetric_matched`의 config 헤더 코멘트와
args.yaml을 직접 대조한 결과, **`v01_dair`의 optimizer(AdamW)/lr0(0.001)/lrf(0.01)/
batch(32)/cos_lr(false)를 그대로 복사**했다 — `v01_symmetric`(SGD/lr0=0.01/batch=48/
cos_lr=true) 쪽 하이퍼파라미터가 아니다. 즉 "matched"는 **v01_dair와 맞춘 것이지
v01_symmetric과 맞춘 것이 아니다** — 이름이 `v01_symmetric_matched`인 것은 트레이너
(symmetric)가 같다는 뜻일 뿐, 하이퍼파라미터 계보는 v01_dair 쪽이다.

**가장 중요한 새 발견 (이 조사에서 처음 확인)**: `cls`(classification loss weight)가
`v01_symmetric`만 **1.5**이고 `v01_symmetric_matched`/`v02` 둘 다 ultralytics 기본값인
**0.5**다 — 3배 차이. `v01_symmetric`의 exp.yaml
(`configs/experiments/BSD_F606/Rear_Truck/v01_symmetric/exp.yaml:69`)에 `cls: 1.5`가
명시적으로 오버라이드돼 있고, 나머지 두 config에는 `cls` 키 자체가 없어 기본값(0.5)을
그대로 쓴다 — args.yaml로 실측 확인. 이 항목은 batch/optimizer/lr0/cos_lr처럼 이미 알려진
차이가 아니라, **이번 조사에서 새로 찾은 차이**다.

## 3. PT 구조 비교

세 `best.pt`를 직접 로드해 확인(container 내 `torch.load`):

| 항목 | v01_symmetric | v01_symmetric_matched | v02 |
|---|---|---|---|
| 파라미터 수 | 3,011,823 | 3,011,823 | 3,011,823 |
| state_dict key 수 | 355 | 355 | 355 |
| key 이름/순서 | 동일(`model.0.conv.weight` ~ `model.22.dfl.conv.weight`) | 동일 | 동일 |
| 모델 클래스(wrapper) | `SymmetricDetectionModel` | `SymmetricDetectionModel` | `DairDetectionModel` |

**architecture 자체는 세 모델이 완전히 동일하다** — layer 구조/shape/파라미터 수 전부 일치.
차이는 학습된 weight 값 자체에 있다. (지시사항대로 "weight가 다르다"는 사실만으로 원인을
결론짓지 않는다 — 아래 §3-1에서 그 weight가 quantization sensitivity와 연결될 수 있는
**구체적** 통계를 확인했다.)

### 3-1. BatchNorm 통계 비교 — 가장 강한 단서

전체 float 파라미터(learnable weight + buffer)에서 절대값이 큰 순으로 정렬한 결과, 상위권은
전부 **BatchNorm의 `running_var`/`running_mean` 버퍼**였다(실제 conv/head weight가 아니라
학습 중 관측된 활성화 통계). 57개 BN 레이어 전체를 다음 기준으로 스캔:

| Model | BN 레이어 수 | running_var max > 500인 레이어 수 | running_var에 **Inf 포함**된 레이어 수 |
|---|---:|---:|---:|
| v01_symmetric | 57 | **1** | **0** |
| v01_symmetric_matched | 57 | **7** | 0 |
| v02 | 57 | 6 | **1** |

`model.0.bn`(전체 네트워크의 첫 BatchNorm, stem conv 직후 — 이후 모든 레이어가 이 출력에
의존)의 `running_var` 값(16채널 전체를 실측, 요약 아님 — 채널 수가 적어 전량 확인 가능):

| Model | running_var 최대값(유한) | Inf 채널 | 비고 |
|---|---:|---|---|
| v01_symmetric | 14,432 | 없음 | 나머지 15개 채널은 2,000~8,616 범위, 완만한 분포 |
| v01_symmetric_matched | **58,880** | 없음 | v01_symmetric의 약 4배, 나머지 채널도 965~13,320으로 넓게 퍼짐 |
| v02 | 46,176 | **채널 4 = Inf**, `running_mean[4]=-666.5` | **실제 무한대 값이 체크포인트에 저장돼 있음** |

model.22(Detect head)의 box 회귀 브랜치(`cv2.*`) BN 레이어에서도 같은 패턴이 반복된다 —
예: `model.22.cv2.2.1.bn.running_var` max가 v01_symmetric은 수십 단위인 반면
v01_symmetric_matched=1,602, v02=1,077로 한 자릿수 이상 크다(상위-8 절대값 랭킹에서 실측
확인, `model.22.cv2.*` 계열이 matched/v02 양쪽에서 반복적으로 상위권에 나타남 — box
회귀 브랜치에 집중된 패턴으로 보인다).

**해석 (correlation, 아직 causation 아님)**: `running_var`가 크다는 것은 학습 중 그 채널의
활성화 분포가 넓게 퍼져 있었다는 뜻이다. NVT 컴파일러의 calibration은 실제 이미지로 활성화
range를 다시 관측하므로 이 통계 자체가 곧 calibration range는 아니지만, **BN
running_var가 이렇게까지 벌어진다는 것은 그 지점의 활성화 분포 자체가 학습 내내
불안정했다는 신호**이며, 특히 v02의 `model.0.bn` 채널 4에 **실제 Inf가 저장돼 있다는 것은
학습 중 특정 시점에 그 채널의 배치 분산이 발산했다는 직접 증거**다. PyTorch/ONNX
float 추론에서는 `x / sqrt(inf) = 0`으로 안전하게 처리돼 겉보기에 정상 동작하지만(§5에서
float 성능 자체는 세 모델이 큰 차이가 없음을 확인), **NVT의 calibration/quantization
파이프라인이 이 case를 동일하게 안전 처리하는지는 확인되지 않았다** — 이 문서 범위 밖(§16
참조).

## 4. ONNX/export 비교

| 항목 | v01_symmetric (`260821_best.onnx`) | v01_symmetric_matched (`best.onnx`) | v02 (`best.onnx`) |
|---|---|---|---|
| source PT | `best.pt` (재-export, 원본 체크포인트와 bit-exact 동일함이 `20260814_rear_truck_v01_symmetric_reexport_nvt_revalidation.md`에서 이미 검증됨) | `best.pt` | `best.pt` |
| export script | `tools/export_custom.py`(v01_symmetric은 `tools/export_symmetric.py --skip_postprocess`로도 재검증, bit-exact 동일) | `tools/export_custom.py -i 320 -y 320` | `tools/export_custom.py -i 320 -y 320` |
| opset | 13 | 13 | 13 |
| input | `images [1,3,320,320]` | 동일 | 동일 |
| output | `output0 [1,5,2100]`(cls, sigmoid) + `433 [1,4,2100]`(box, raw distance) | **완전히 동일** | **완전히 동일** |
| node 수 | 235 | 235 | 235 |
| 마지막 15개 op | `... Softmax, Conv, Constant, Reshape, Sigmoid` | 동일 | 동일 |
| `skip_postprocess` | **True** (decode/NMS 그래프 밖) | **True** | **True** |
| dynamic axes | 없음(정적 shape) | 없음 | 없음 |
| simplify | False(기본값) | False | False |

**세 ONNX는 그래프 구조가 완전히 동일하다** — export 단계는 원인 후보에서 배제
(Category C). 이는 이미 evaluator-gap 문서들에서 개별 확인된 내용을 이번 조사에서
`onnx.load` + shape/node 비교로 재확인한 것이다.

## 5. PT ↔ ONNX numerical parity

고정 입력(zeros, 255-uniform, seed=0 random uniform[0,255]) 3종으로 PT 모델(`ckpt["model"]`
직접 로드, eval)과 ONNX Runtime(CUDA) 출력을 비교(6,152장 재추론 없음, 3개 고정 입력만):

| Model | max_abs_diff (3개 입력 평균) | mean_abs_diff (3개 입력 평균) |
|---|---:|---:|
| v01_symmetric | ~3.18e+02 | ~4.79e+01 |
| v01_symmetric_matched | ~3.27e+02 | ~4.91e+01 |
| v02 | ~3.19e+02 | ~4.93e+01 |

**세 모델이 거의 동일한 크기의 diff를 보인다.** 이 절대값 자체는 비교 스크립트가 PT의 원본
출력 텐서 레이아웃과 ONNX의 `[cls, box]` 분리 출력을 단순 채널 결합(concat)해서 비교한
결과라 채널 순서/스케일이 완전히 정렬되지 않았을 가능성이 있어 **절대값 자체는 신뢰하지
않는다** — 그러나 **세 모델이 통계적으로 구분되지 않을 만큼 비슷한 값**이라는 점은 확인
됐고, 이는 "PT→ONNX export 단계에서 특정 모델만 오차가 커진다"는 가설을 반박한다.
**PT→ONNX export는 원인 후보에서 배제**(Category C).

## 6. Float ONNX 출력 분포 비교

`ipu_gt/images`에서 정렬 순 앞 5장(전체 6,152장 재추론 없음)에 대해 각 모델의 raw
`output0`(cls, sigmoid 통과 후) / `433`(box, DFL 이전 raw distance) 텐서 통계:

### classification tensor (sigmoid, 5장 합산)

| Model | min | max | mean | std | p95 | p99 | p99.9 | frac>0.5 | frac<0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v01_symmetric | 0.0000 | 0.9493 | 0.0064 | 0.0635 | 0.00018 | 0.184 | 0.886 | 0.64% | 98.31% |
| v01_symmetric_matched | 0.0000 | 0.9498 | 0.0060 | 0.0625 | 0.00027 | 0.099 | 0.899 | 0.61% | 98.31% |
| v02 | 0.0000 | 0.9421 | 0.0058 | 0.0613 | 0.0001 | 0.105 | 0.889 | 0.59% | 98.54% |

**cls 분포는 세 모델이 사실상 구분 불가능할 만큼 비슷하다.** "v01_symmetric의 cls 출력이
INT8에 유리한 분포를 갖는가"라는 질문에는 **이 5장 표본 기준으로는 아니다**로 답한다 —
`cls: 1.5` 오버라이드(§2)가 존재함에도 sigmoid 출력 분포 자체에는 뚜렷한 차이가 나타나지
않았다(표본이 5장뿐이라는 한계는 있음, §16).

### box(raw distance) tensor (DFL 디코드 이전, 5장 합산)

| Model | min | max | mean | std | p95 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v01_symmetric | -0.587 | 9.669 | 1.741 | 1.328 | 4.257 | 6.442 | 8.403 |
| v01_symmetric_matched | -0.956 | 10.401 | 1.766 | 1.375 | 4.573 | 6.429 | 8.628 |
| v02 | -0.505 | **10.573** | **1.964** | **1.528** | **4.704** | **7.121** | **9.368** |

**box 텐서에서는 v02가 세 모델 중 가장 넓은 dynamic range**(std/p99/p99.9/max 전부 최대)를
보인다 — §3-1의 BN running_var 발견(v02의 box 회귀 브랜치 BN이 가장 불안정)과 방향이
일치하는 corroborating evidence다. v01_symmetric_matched는 v01_symmetric과 v02의 중간
정도다. 다만 5장 표본이라 통계적으로 확정적이지는 않다.

## 7. NVT converting 비교

세 모델의 실제 NVT 변환 디렉토리(`novatek/convert/rear_truck_{v01_symmetric,
v01_symmetric_matched,v02}_320/`)를 직접 비교(재컴파일 없음, 기존 파일만 읽음):

| 항목 | v01_symmetric | v01_symmetric_matched | v02 |
|---|---|---|---|
| gen_config.txt | `[path/model_dir]`만 다르고 **나머지 전부 byte-identical**(`diff` 확인) | 동일 | 동일 |
| calibration 이미지 디렉토리 | `rear_truck_v01_320/nvtai_tool/input/data`(symlink) | **동일 symlink 대상**(`readlink -f`로 확인) | **동일 symlink 대상** |
| calibration 이미지 수 | 940장(`ref_data/num=940`, gen_config 공통값) | 동일 | 동일 |
| calib_mode / percentile | 4 / 0.999(공통) | 동일 | 동일 |
| spec_demarc_tensors | H1/H2/H3 3개(Softmax dfl / Sigmoid / Conv dfl conv), 공통값 | 동일 | 동일 |
| cle_en / cal_refine_iter | 1 / 5(공통) | 동일 | 동일 |
| chip / compiler | 530, 동일 SDK | 동일 | 동일 |
| 컴파일된 `nvt_model.bin` 크기 | 4,711,336 bytes | 4,711,336 bytes | 4,711,336 bytes |
| Simulator cosine (Softmax(dfl)/Sigmoid/dfl_conv) | 0.9850 / 0.9824 / 0.9933 | 0.913 / 0.947 / 0.949 | 0.906 / 0.924 / 0.966 |

**NVT compile 설정 자체는 세 모델이 완전히 동일하다**(모델 경로 지정 외 config 차이 없음,
동일 calibration 데이터, 동일 tensor 보호, 동일 컴파일러) — **NVT converting 단계는 원인
후보에서 배제**(Category C). 실제로 갈리는 것은 오직 **동일 조건에서 나온 simulator
cosine 결과** — v01_symmetric이 세 지표 모두 가장 높고(0.98대), v02가 가장 낮다(0.90대).
이 cosine 자체는 §3-1/§6에서 확인한 활성화 분포 차이의 **결과**로 보는 것이 합리적이다
(같은 컴파일 조건에서 나온 값이므로 compile 설정 차이가 원인일 수 없다).

## 8. Quantization tensor/config 비교

§7에서 확인했듯 `gen_config.txt`의 `spec_demarc_tensors`(고정밀도 보호 대상)는 세 모델
모두 동일한 3개 텐서(H1/H2/H3)로 고정돼 있고, 그 외 tensor별 개별 scale/zero-point/range를
모델별로 저장한 별도 산출물(예: `tensors.json` 류)은 확인되지 않았다 — Novatek 컴파일러의
현재 출력물(`sdk/`, `gentool/`, `debug/`)에는 layer별 quantization scale을 명시적으로
덤프하는 파일이 없다(`debug/sim_layer/`, `debug/golden_tensor/`는 activation 값 자체이지
quantization scale 파라미터가 아님). 따라서 이 절의 "suspicious top-N tensor" 표는 **§7의
simulator cosine**(사실상 유일하게 존재하는, 모델별로 다른 quantization-post 수치)으로
대체한다:

| tensor([OUT] 태그) | v01_symmetric | v01_symmetric_matched | v02 | v01_symmetric 대비 최대 격차 |
|---|---:|---:|---:|---:|
| Softmax(dfl) | 0.9850 | 0.913 | 0.906 | **-0.079 (v02)** |
| Sigmoid | 0.9824 | 0.947 | 0.924 | **-0.058 (v02)** |
| dfl_conv | 0.9933 | 0.949 | 0.966 | **-0.044 (matched)** |

세 지표 모두에서 v01_symmetric이 가장 높다는 것은 일관되지만, matched/v02 사이의 순위는
텐서마다 다르다(Softmax/Sigmoid는 v02가 더 낮고, dfl_conv는 matched가 더 낮다) — 이는 두
모델의 "손실 원인"이 완전히 동일한 메커니즘이 아닐 수 있음을 시사한다(§13 참조).

## 9. Board/postprocess 비교

세 모델의 board 평가 config(`board_v01_symmetric_3tensor_cfg.yaml`,
`board_v01_symmetric_matched_3tensor_cfg.yaml`, `board_v02_3tensor_cfg.yaml`)를 직접
대조 — `pred.path`(모델별 병합 예측 파일)만 다르고 `gt`/`rep`/`mode`/`classes` 전부
동일하다(이미 `20260824_v01_symmetric_eval_gap.md` §4-2, `20260824_...matched_eval_gap.md`
에서 확인된 내용의 재확인). board raw dump는 세 모델 모두 이미 온디바이스 NMS를 거친 최종
detection 형식(`{bbox, score, name, category, time}`)이고, 이 조사를 포함한 모든 세션에서
**board dump에 추가 NMS를 적용한 적이 없다**. **Board postprocess 설정 자체도 세 모델이
동일하다** — 원인 후보에서 배제(Category C).

## 10. 확인된 동일점

- 아키텍처(레이어 구조, 파라미터 수 3,011,823, state_dict key 355개) — 완전 동일
- ONNX export 방식/그래프 구조(`skip_postprocess=True`, 2-tensor 출력, opset 13, node 235개)
  — 완전 동일
- PT→ONNX numerical parity — 세 모델 통계적으로 구분 불가
- NVT compile 설정(gen_config.txt, calibration 940장 동일 symlink, spec_demarc_tensors,
  calib_mode/percentile/cle_en 등) — model_dir 외 완전 동일
- 컴파일된 `nvt_model.bin` 크기 — 동일(4,711,336 bytes)
- Board postprocess/평가 config 구조 — pred.path 외 완전 동일
- Float cls(sigmoid) 출력 분포 — 5장 표본 기준 통계적으로 구분 불가
- 학습 데이터/split(사용자 확인 기준) — 세 모델 모두 동일 계열

## 11. 확인된 차이점

| 항목 | v01_symmetric | v01_symmetric_matched | v02 | 근거 |
|---|---|---|---|---|
| trainer | Symmetric | Symmetric | DAIR(dual-forward) | args.yaml, model class |
| optimizer/lr0/lrf/batch/cos_lr | SGD/0.01/0.01/48/true | AdamW/0.001/0.01/32/false | AdamW/0.001/0.01/32/false | args.yaml |
| **cls loss weight** | **1.5** | **0.5** | **0.5** | exp.yaml, args.yaml (§2) |
| DAIR dual-forward + TensorSpeakerOverlayAug | 없음 | 없음 | 있음(단, 현재 코드상 augmentation 자체는 비활성) | exp.yaml, dair.py |
| BN `running_var` 이상치(model.0 등) | 최대 14,432, Inf 없음 | 최대 58,880, Inf 없음 | 최대 46,176 + **Inf 1개 채널** | torch.load 실측 (§3-1) |
| box raw distance 텐서 dynamic range | 가장 좁음(std 1.328) | 중간(std 1.375) | 가장 넓음(std 1.528) | ONNX 5장 추론 실측 (§6) |
| NVT simulator cosine(3종) | 0.983~0.993 (최고) | 0.913~0.949 | 0.906~0.966 (Softmax/Sigmoid 최저) | 기존 simulator 로그 (§7,8) |
| Board quantization loss(pycoco, mAP50-95) | -7.30% | -28.42% | -19.14% | §1 |

## 12. 원인 후보 우선순위화

### A. 가능성 높음

- **BatchNorm 활성화 통계의 이상치/불안정성**(§3-1) — v02는 실제 Inf, matched는 가장 큰
  유한 이상치를 학습 종료 시점 체크포인트에 갖고 있다. v01_symmetric만 전체 57개 BN 레이어
  중 이상치가 1개뿐이다. 활성화 분포가 넓고 불안정할수록 INT8 calibration이 그 분포를
  대표하는 range를 잡기 어려워진다는 것은 quantization의 일반적 메커니즘과 직접 연결된다.
  §6의 box tensor dynamic range 실측(v02 최대, symmetric 최소)도 같은 방향으로 일치한다.
- **`cls` loss weight 차이(1.5 vs 0.5)**(§2) — v01_symmetric만의 고유한 명시적 설정이며,
  classification 학습 강도가 3배 다르다는 것은 학습된 weight/activation의 성격 자체를
  바꿀 수 있는 직접적인 하이퍼파라미터다. 다만 §6에서 cls(sigmoid) 출력 분포 자체는 5장
  표본 기준 뚜렷한 차이가 없었다 — "왜 아직 효과가 안 보이는지"는 §15/§16에서 후속 검증
  필요.

### B. 가능성 중간

- **optimizer/lr0/lrf/cos_lr/batch 차이**(SGD+cos_lr+batch48 vs AdamW+batch32) — 이미
  `20260806_...md`의 `v01_symmetric_matched ablation`에서 "quantization loss 차이의
  가장 큰 설명 변수"로 지목된 축이지만, 그 문서는 evaluator를 conf=0.01/기존 방식으로만
  봤다 — 이번 조사에서 pycoco/ultra 양쪽에서 순위가 동일함을 재확인했을 뿐, optimizer 자체가
  "왜" BN 통계나 box dynamic range를 이렇게 다르게 만드는지의 메커니즘은 규명하지 않았다.
  Correlation은 강하지만(§2), causation 경로(옵티마이저 선택 → BN 통계 불안정성)는
  검증되지 않았다.
- **DAIR dual-forward 학습 방식(v02 고유)** — v02만 DAIR 트레이너를 쓰지만
  `v01_symmetric_matched`(DAIR 미사용)도 큰 손실을 보이므로, DAIR 자체가 필요조건은
  아니다(`20260806_...md`가 이미 이 결론에 도달). 다만 v02의 Inf(§3-1)가 DAIR 특유의
  dual-forward loss 계산과 관련이 있을 가능성은 배제되지 않았다.

### C. 가능성 낮음 / 배제

- **ONNX export 방식/그래프 구조**(§4) — 세 모델 완전히 동일.
- **PT→ONNX export 정합성**(§5) — 세 모델 통계적으로 구분 불가.
- **NVT compile 설정/calibration 데이터**(§7) — model_dir 외 완전히 동일.
- **Board postprocess/평가 설정**(§9) — 완전히 동일.
- **아키텍처 자체**(§3) — 완전히 동일.
- **cls(sigmoid) 출력 분포**(§6) — 5장 표본 기준 구분 불가(단, 표본 확대 시 달라질
  가능성은 남아있음, §16).

### Top 5 원인 후보

| 순위 | 후보 | 근거 | v01_symmetric | matched/v02 | 다음 검증 |
|---|---|---|---|---|---|
| 1 | BatchNorm 활성화 통계 불안정성(특히 v02의 Inf) | §3-1: 57개 BN 중 이상치 레이어 수(1 vs 6~7), v02는 실제 Inf 1채널 | 이상치 1개, Inf 없음 | matched 7개(최대 58,880), v02 6개+Inf 1개 | NVT calibration 로그에서 해당 레이어(특히 model.0, model.22.cv2.*)의 실제 activation range/clipping 여부 확인 |
| 2 | box 회귀 브랜치 활성화 dynamic range | §6: box tensor std/p99/max가 v02>matched>symmetric 순 | std 1.328 | std 1.375(matched)/1.528(v02) | 6,152장 전체에서 box tensor 분포 재확인(현재 5장 표본의 한계 보완) |
| 3 | `cls` loss weight (1.5 vs 0.5) | §2: config로 확인된 v01_symmetric만의 고유 설정 | 1.5 | 0.5 | cls=1.5로 matched/v02를 재학습해 BN 통계/quantization loss가 바뀌는지 확인(재학습 필요 — 이번 세션 범위 밖) |
| 4 | optimizer/lr0/lrf/cos_lr/batch (SGD+cos_lr vs AdamW) | `20260806_...md`가 이미 지목한 축, 이번 조사에서 evaluator 무관하게 재확인 | SGD/cos_lr=true/batch48 | AdamW/cos_lr=false/batch32 | strict ablation(하이퍼파라미터만 다르고 나머지 전부 동일한 재학습) — 이미 "다음 확인 사항"으로 기록됨, 재학습 필요 |
| 5 | DAIR dual-forward(v02만) | v02 고유, matched는 없어도 손실이 크므로 필요조건 아님 | 없음 | v02만 있음 | matched가 이미 이 후보를 상당 부분 반증 — 추가 검증 우선순위 낮음 |

## 13. 현재 가장 유력한 원인

**단일 원인으로 확정할 수 없다.** 이번 조사로 새로 확인된 가장 강력한 단서는 **§3-1의
BatchNorm 활성화 통계 불안정성**이다 — 특히 v02 체크포인트에 실제 Inf 값이 저장돼 있다는
것은 추측이 아니라 `torch.load`로 직접 확인한 사실이며, "학습 중 활성화가 불안정했다"는
정성적 결론을 뒷받침하는 가장 구체적인 증거다. 이는 §6의 box tensor dynamic range 실측과도
방향이 일치한다.

다만 이 BN 통계 차이 자체가 **왜** 생겼는지는 이번 조사에서 규명하지 못했다 — optimizer
(SGD vs AdamW), lr schedule(cos_lr), 또는 `cls` loss weight 차이가 학습 과정에서 이런 BN
불안정성을 유발했을 가능성이 있지만, 이는 **후보 4번 순위 항목(§12)과의 인과관계를
가정하는 것이지 확인된 것이 아니다.** "차이 발견 → 연결 가능성 → 증거 수준" 원칙(§11 지시
사항)에 따라, 현재는:

- **차이 발견**: BN 통계 불안정성(v02 Inf 포함), box tensor dynamic range, cls loss weight,
  optimizer 계열 — 전부 실제로 존재하는 차이(확인됨)
- **연결 가능성**: BN 통계 불안정성 → INT8 calibration 어려움 → quantization loss, 이 경로는
  일반적인 quantization 지식과 부합하고 §6/§7/§8의 실측과도 방향이 일치(가능성 높음)
- **증거 수준**: 상관관계는 강하지만, "옵티마이저/lr/cls weight가 BN 불안정성을
  유발했다"는 인과관계, 그리고 "BN 불안정성이 실제 NVT calibration에서 clipping/오차로
  이어졌다"는 인과관계 둘 다 **직접 검증되지 않았다.**

## 14. 아직 증명되지 않은 것

- optimizer/lr schedule/batch/cls weight 중 **어느 것이(또는 몇 개가 조합돼) BN 활성화
  불안정성의 실제 원인인지** — strict ablation(하나씩만 바꾼 재학습) 없이는 분리 불가.
- BN 통계 불안정성이 **NVT 컴파일러의 calibration/quantization 단계에서 실제로 어떻게
  처리되는지** — Novatek 컴파일러가 Inf/극단값을 어떻게 clipping/처리하는지의 내부 동작은
  이번 조사에서 확인하지 않았다(벤더 코드 접근 불가).
- `cls` loss weight 차이가 실제로 quantization sensitivity에 영향을 주는지 — §6에서
  sigmoid 출력 분포 자체는 5장 표본 기준 차이가 없었으므로, 이 후보의 실제 영향력은
  아직 뒷받침되지 않았다(표본 확대나 재학습 실험이 필요).
- box tensor dynamic range 차이(§6)가 6,152장 전체에서도 유지되는지 — 5장 표본의 한계.
- Softmax/Sigmoid는 v02가 더 낮고 dfl_conv는 matched가 더 낮다는 §8의 비일관성 — 두 모델의
  손실 메커니즘이 완전히 같지 않을 가능성을 시사하지만 원인은 규명되지 않았다.

## 15. 다음 최소 검증 실험 (제안, 실행하지 않음)

우선순위 순:

1. **BN 통계 불안정성의 재현성 확인**: 6,152장 전체(또는 더 큰 표본)로 box/cls tensor
   분포를 재확인해 §6의 5장 표본 결과가 유지되는지 검증 — 재학습·재컴파일 없이 가능한
   가장 저비용 다음 단계.
2. **NVT calibration 로그 재분석**: 이미 존재하는 `debug/sim_layer/`,
   `debug/golden_tensor/`(3개 모델 전부 이미 compile 완료됨, §7) 덤프에서 `model.0`,
   `model.22.cv2.*` 레이어에 해당하는 activation range를 직접 추출해, 실측 BN
   running_var가 큰 레이어의 실제 calibration range/clipping 흔적이 있는지 확인 —
   재컴파일 불필요, 기존 산출물만 재분석.
3. (재학습 필요, 이번 세션 범위 밖) `cls: 1.5`만 단독으로 바꾼 v01_symmetric_matched
   재학습 — 다른 하이퍼파라미터는 그대로 두고 `cls` weight만 조정해 BN 통계/quantization
   loss가 바뀌는지 확인하는 strict ablation.

## 16. 이번 조사의 방법론적 한계

- Float output distribution(§6)은 6,152장이 아니라 5장 표본이다 — 지시사항의 "6,152장
  재추론 금지, 최소한만" 원칙을 따른 결과이며, 표본 확대가 필요한 결론(특히 cls 분포
  "차이 없음")은 잠정적이다.
- PT↔ONNX parity(§5) 비교 스크립트는 절대값 자체의 신뢰도가 낮다(채널 정렬 이슈 가능성) —
  세 모델 간 상대 비교(구분 불가)만 결론으로 채택했다.
- 이번 조사는 "차이 발견"과 "그 차이가 quantization loss와 실제로 인과관계가 있는지"를
  구분하려 노력했으나, 재학습/재컴파일 없이는 인과관계를 최종 확정할 수 없다는 근본적
  한계가 있다(지시사항에 따라 이번 세션에서는 수행하지 않음).

---

## 17. 후속 검증 (2026-08-24, layer-by-layer Float→NVT divergence 분석)

**목적**: §15에서 제안한 "다음 최소 검증 실험" 1순위(box/cls tensor 분포 재확인)를 수행하는
대신, 이미 존재하는 NVT 산출물 중 **정확히 이 목적을 위해 벤더 툴체인이 이미 생성해둔
파일**(`simtool/layer_diff.txt`)을 발견해 활용했다 — 재추론·재컴파일 없이 세 모델의
전체 171개 레이어에 대한 Float(golden) vs NVT-simulator 오차를 층별로 직접 비교했다.
**§3-1/§12의 기존 BN 가설은 이번 분석으로 상당 부분 약화되고, 새로운 후보(DFL Softmax
연산에서의 오차 증폭)가 발견됐다** — 아래에서 기존 가설과 이번 결과를 구분해서 기록한다.

### 17-1. dump 의미 확인 (코드 근거)

`novatek/convert/visualize_sim.py:81-94`를 직접 읽어 확인:

- `debug/golden_tensor/<TensorName>.bin`: **Float(ONNX/PT) 참조값**(`dtype=np.float32`로
  명시적으로 로드) — 양자화 이전의 순수 float 계산 결과.
- `debug/sim_layer_<이미지명>/float/<TensorName>.bin`: **NVT 시뮬레이터의 dequantized
  (양자화 후 float로 역변환한) 값** — 동일 텐서에 대해 quantized 연산을 시뮬레이션한 뒤
  float 스케일로 되돌린 것.
- `debug/sim_layer_<이미지명>/fixed/`: 시뮬레이터의 raw fixed-point(정수, 양자화 도메인)
  값 — 이번 분석에서는 사용하지 않음(§17 범위는 float-domain 비교로 충분).
- `simtool/layer_diff.txt`(및 이미지별 `layer_diff_1_<이미지명>.jpg.txt`): 위 두 dump를
  **레이어(ONNX 노드 출력)마다 cosine similarity + avg_diff로 미리 비교해둔 벤더 툴체인
  산출물**. 3개 모델 전부 **같은 테스트 이미지**(`KR_20240318_..._0126.jpg`, 파일명으로
  확인)에 대한 결과이고, 171개 레이어 전부에서 **레이어 이름/순서가 세 모델 간 완전히
  동일**(positional 정렬 가능, 코드로 확인) — 즉 tensor pair가 직접 비교 가능함을 확인했다.
- **교차검증**: `golden_tensor`/`sim_layer/float`의 `Sigmoid_output0_Y.bin` raw 값을 직접
  읽어 cosine을 재계산한 결과(symmetric 0.9824, matched 0.9473, v02 0.9236)가
  `layer_diff.txt`에 기록된 값과 **완전히 일치** — `layer_diff.txt`가 실제로 이 두 dump를
  비교한 결과임을 확인했다(맹신하지 않고 재계산으로 검증).
- **주의**: golden(10,500개 = 5class×2100anchor)과 sim(10,528개) 사이에 shape 불일치가
  있었다 — sim 쪽에 HW 정렬용 padding이 있는 것으로 추정, 앞 10,500개로 truncate해 비교(세
  모델 모두 동일한 padding 크기라 상대 비교에는 영향 없음).

### 17-2. 전체 layer 매핑 및 오차 계산

171개 레이어 전부 이름 기준 1:1 매핑 성공(공통 tensor 171개, 세 모델 전부). 각 레이어에
대해 `layer_diff.txt`가 제공하는 cosine/avg_diff를 그대로 사용했고, `Sigmoid_output0_Y`에
한해 raw bin을 직접 읽어 MAE/RMSE/max_abs_err/min·max/포화 비율까지 추가 계산해 대표
샘플로 검증했다(전체 171개에 대해 raw 비교를 반복하지 않음 — 지시사항의 "요약 통계만"
원칙에 따름):

| Model | golden min/max | sim min/max | MAE | RMSE | max_abs_err | cosine(재계산) | frac==0(sim) |
|---|---|---|---:|---:|---:|---:|---:|
| symmetric | 0.0000 / 0.9105 | 0.0002 / 0.8682 | 0.00133 | 0.01137 | 0.322 | 0.9824 | 0.0000 |
| matched | 0.0000 / 0.9143 | 0.0002 / 0.9277 | 0.00202 | 0.01934 | 0.501 | 0.9473 | 0.0000 |
| v02 | 0.0000 / 0.8739 | 0.0002 / 0.9160 | 0.00214 | 0.02074 | 0.580 | 0.9236 | 0.0000 |

**포화(0/1로 값이 몰리는) 흔적은 없다** — `frac==0`이 세 모델 다 0, min/max 범위도
세 모델이 비슷한 폭(0~0.87~0.93)이라 "특정 값에 값이 눌려 찍히는" 전형적인 clipping
패턴은 보이지 않는다. MAE/RMSE/max_abs_err는 symmetric이 가장 작고 matched/v02가
2~2.5배 크다 — **분포가 넓게 퍼진 quantization noise**로 보이며, **"clipping
의심"으로 확정하지 않는다**(지시사항대로 scale/zero-point 정보가 없어 확정 불가).

### 17-3. 최초 divergence 탐색 — 결과

171개 레이어를 순서대로(model.0 → model.22) 훑은 결과:

- **backbone 전반(idx 2~90, model.0~model.9/SPPF)**: 세 모델의 cosine이 대체로
  0.98~1.00 범위에서 서로 뒤섞여 있고, **뚜렷한 방향성이 없다** — 오히려 idx 83~91
  (model.9 SPPF~model.10 업샘플) 구간에서는 **symmetric이 0.98~0.99로 떨어지고
  matched/v02가 0.999로 더 높은 구간**도 존재한다(예: idx91 `Resize__model.10`:
  sym=0.9825, matched=0.9994, v02=0.9994). `model.0`(idx 2,3) 자체는 **세 모델이
  사실상 동일**(cos 0.9996~1.0000, 차이 <0.0004) — 기존에 지목했던 "model.0부터
  갈린다"는 지점은 **이번 분석에서 재현되지 않았다.**
- **neck 후반부(idx 93~120, model.12~model.15)**: 국소적으로 뒤섞인 fluctuation이
  계속된다 — symmetric이 더 나쁠 때도(idx 105~120 구간 다수), matched/v02가 더 나쁠 때도
  (idx 112~113, 119~120) 있다. 아직 "symmetric만 확실히 좋다"는 패턴은 아니다.
- **Detect head(idx 122~192, model.22 cv2.*/cv3.* 3개 scale + model.16~21 neck-head
  연결부)**: 이 구간부터 matched/v02가 symmetric보다 나쁜 사례가 **더 자주, 더 크게**
  나타나기 시작한다(§17-4 Top 20 참조) — 다만 **cv2(box)뿐 아니라 cv3(classification)
  브랜치에서도 동일하게(오히려 더 크게) divergence가 나타난다**(예: idx182
  `Mul__model.22_cv3.2_cv3.2.0_act`: sym=0.9637, matched=0.8801(!), v02=0.9343 — Top 1
  divergence가 cv3 브랜치임).
- **가장 크고 일관된 단일 cliff — DFL Softmax (idx 196)**: idx192
  (`Concat__model.22_Concat_1_output_0_Y`, box+cls 최종 결합)까지는 세 모델 모두 비교적
  높은 cosine(sym 0.9965, matched 0.9793, v02 0.9849 — 격차 0.017~0.024, 완만함)을
  유지하다가, 바로 다음 연산인 **`Softmax__model.22_dfl_Softmax_output_0_Y`(idx 196)에서
  matched/v02 둘 다 급격히 무너진다**: sym 0.9850(거의 그대로) vs matched **0.9126**(-0.067),
  v02 **0.9058**(-0.079). **직전 레이어 대비 낙폭이 이 지점에서 가장 크고, symmetric은
  거의 영향받지 않는데 matched/v02는 동시에 크게 영향받는 유일한 지점**이다. 뒤이은
  `Sigmoid_output0_Y`(idx199)와 `Conv__model.22_dfl_conv`(idx211, 최종 box 출력)에서도
  격차가 유지/확대된다.

### 17-4. 차이가 가장 큰 Top 10 (sym_cos − min(matched_cos, v02_cos) 기준)

| order | tensor | symmetric cos | matched cos | v02 cos | 판단 |
|---:|---|---:|---:|---:|---|
| 182 | `Mul__model.22_cv3.2_cv3.2.0_act_Mul` (cls 브랜치, scale2) | 0.9637 | 0.8801 | 0.9343 | Detect head cls 브랜치, 최대 격차 |
| 196 | `Softmax__model.22_dfl_Softmax` | 0.9850 | 0.9126 | 0.9058 | **직전 대비 급락, 두 모델 동시 붕괴 — 가장 구조적으로 의미있는 지점** |
| 199 | `Sigmoid_output0_Y`(최종 cls) | 0.9824 | 0.9473 | 0.9236 | Softmax 붕괴 직후 cls 최종 출력에도 전파 |
| 186 | `Mul__model.22_cv3.2_cv3.2.1_act_Mul` (cls, scale2) | 0.9860 | 0.9273 | 0.9680 | cv3 브랜치 |
| 162 | `Mul__model.22_cv3.1_cv3.1.1_act_Mul` (cls, scale1) | 0.9639 | 0.9111 | 0.9622 | cv3 브랜치, matched만 크게 나쁨 |
| 171 | `Mul__model.21_m.0_cv1_act_Mul`(neck-head 연결부) | 0.9635 | 0.9115 | 0.9318 | neck 후반 |
| 126 | `Mul__model.22_cv3.0_cv3.0.0_act_Mul` (cls, scale0) | 0.9710 | 0.9384 | 0.9230 | cv3 브랜치, v02가 더 나쁨 |
| 52 | `Mul__model.6_m.0_cv1_act_Mul`(backbone 중반) | 0.9961 | 0.9723 | 0.9506 | backbone, v02가 더 나쁨 |
| 36 | `Mul__model.4_m.1_cv1_act_Mul`(backbone 초반) | 0.9955 | 0.9780 | 0.9510 | backbone, v02가 더 나쁨 |
| 211 | `Conv__model.22_dfl_conv`(최종 box 출력) | 0.9933 | 0.9490 | 0.9664 | Softmax 붕괴가 최종 box 출력까지 전파 |

**cv2(box)만이 아니라 cv3(classification) 브랜치에서도 큰 격차가 나타난다** — Top 10 중
4개(idx 126,162,182,186)가 cv3, 2개(idx211 및 §17-3에서 언급한 idx179/187 등)가 cv2 계열,
나머지는 backbone/neck 산발 지점이다. "box branch(cv2)만의 문제"라는 원래 가설(B)은
**부분적으로만 맞다** — 실제로는 Detect head의 두 브랜치(box+cls) 모두, 그리고 backbone
중반 일부 지점에서도 산발적으로 나타난다.

### 17-5. 기존 가설 재평가

**A. model.0/초기 activation 가설 — 약화(반증에 가까움)**: `model.0`(idx2,3) 자체는 세
모델 간 cosine 차이가 0.0004 이내로, 사실상 구분 불가능하다. §3-1에서 확인한
`model.0.bn.running_var`의 극단적 차이(symmetric 14,432 vs matched 58,880 vs v02
Inf)는 **실제 학습된 통계로는 존재하지만, 이 한 장의 calibration 이미지에 대한
NVT simulator 결과에서는 그 지점에서 눈에 띄는 quantization 오차 증폭으로 이어지지
않았다.** 즉 BN 통계 이상치가 실재한다는 사실과, 그것이 §4에서 예상한 대로 "가장
먼저" 오차를 만든다는 것은 별개였다 — 후자는 이번 분석으로 반증됐다.

**B. box branch(model.22.cv2.*) 가설 — 부분 강화, 범위 수정 필요**: Detect head에서
오차가 크게 벌어지는 것 자체는 맞지만(§17-3, §17-4), **cv2(box)만이 아니라 cv3(cls)
브랜치도 동등하거나 더 크게 벌어진다.** "box regression만의 문제"로 좁히면 안 되고
**"Detect head 전체(box+cls 양쪽)"**로 가설을 넓혀야 한다.

**C. 특정 중간 레이어 — 신규 후보로 채택**: 원래 가설에 없던 새로운, 매우 구체적인
지점을 찾았다 — **`Softmax__model.22_dfl_Softmax_output_0_Y`(idx196)**. 이 연산 직전
(idx192)까지는 완만한 격차였다가, Softmax를 통과하는 순간 matched/v02만 급격히
악화된다. Softmax는 입력 logit의 작은 차이도 출력 확률분포에서 크게 증폭시킬 수 있는
비선형 연산이라(특히 분포가 뾰족/확신적일수록 민감도가 커짐), **"softmax 입력으로 들어가는
DFL box-distance logit의 분포 뾰족함(confidence)이 모델마다 다르고, 그 차이가 quantization
noise를 비선형적으로 증폭시켰을 가능성"**이 새로운 유력 후보다. 다만 이는 correlation
수준이며, "왜 matched/v02의 pre-softmax logit 분포가 더 뾰족한지(혹은 다른지)"는
검증되지 않았다.

**D. 전반적 학습 recipe 차이(분산 누적) — 강하게 지지됨**: backbone/neck 전반(idx
2~120)에서 세 모델의 cosine이 뒤섞여 오르내리는 패턴(어느 한쪽이 항상 좋지 않음)은
**단일 원인 레이어가 아니라 전반에 걸친 미세한 quantization-noise 차이가 누적**되는
모습과 일치한다. 여기에 idx196(Softmax dfl)이라는 **하나의 크고 명확한 증폭 지점**이
추가로 존재하는, **D(전반적 누적) + C(특정 증폭 지점)의 혼합**이 이번 분석이 실제로
지지하는 결론이다 — A(model.0)는 반증됐고, B(box branch 단독)는 범위를 넓혀야 한다.

### 17-6. 모델 간 핵심 비교 — 답변

1. **Float→NVT divergence가 처음 의미 있게 발생하는 레이어는?** 단일 "최초 지점"은
   없다 — backbone/neck 전반에서 산발적·양방향으로 작은 격차가 누적되다가,
   **idx196(Softmax dfl)에서 가장 크고 명확한 단일 증폭이 발생**한다.
2. **세 모델에서 같은 위치인가?** idx196의 증폭은 matched/v02 **둘 다 동시에** 발생한다
   (symmetric만 무사함) — 이 지점만큼은 공통적이다. 그러나 그 이전 backbone/neck의
   산발적 격차는 matched와 v02가 서로 다른 지점에서 서로 다른 크기로 나타난다(§17-4,
   idx52/36은 v02가 더 나쁘고 idx162/186은 matched가 더 나쁨).
3. **symmetric만 오차가 작은 구간은?** Detect head 전체(idx122~211)와, backbone 중반
   일부 지점(idx36,52 등)에서 symmetric이 일관되게 낮은 오차를 유지한다. backbone
   초반(model.0~model.9)에서는 symmetric이 특별히 더 낫지 않다(오히려 SPPF/업샘플
   구간은 matched/v02가 더 나음).
4. **matched와 v02가 같은 방식으로 무너지는가?** **부분적으로 같고 부분적으로 다르다.**
   idx196(Softmax dfl)에서는 거의 동일한 크기로 함께 무너진다(0.9126 vs 0.9058). 하지만
   idx211(dfl_conv, 최종 box)에서는 matched(0.9490)가 v02(0.9664)보다 더 나쁘고, backbone
   중반(idx36,52)에서는 반대로 v02가 matched보다 더 나쁘다 — **완전히 동일한 붕괴 패턴은
   아니다.**
5. **BN 가설: 강화/약화/판단불가?** → **약화.** model.0 자체에서는 오차 확대가 관찰되지
   않았다(§17-5 A). BN 통계 이상치(특히 v02의 Inf)가 실재한다는 사실 자체는 바뀌지
   않지만, "그것이 quantization 오차의 시작점"이라는 인과 경로는 이번 분석으로 뒷받침되지
   않았다.
6. **box-head 가설: 강화/약화/판단불가?** → **부분 강화(범위 수정)** — Detect head에서
   오차가 커지는 것은 맞으나 cv2(box) 단독이 아니라 cv3(cls)도 동등하게 관여한다. "box
   branch만의 문제"는 약화되고, "Detect head 전체(+ 그 직후 Softmax 연산)"로 수정된
   가설이 강화됐다.
7. **학습 recipe 차이 → quantization robustness 가설과 일치하는가?** → **일치한다,
   단 메커니즘은 두 갈래.** (a) backbone/neck 전반의 미세하고 산발적인 오차 차이는 학습
   recipe(optimizer/lr/cls weight 등, §12)에 따른 전반적 quantization 강건성 차이와
   부합한다(D). (b) 그와 별개로 idx196의 급격한 단일 증폭은 recipe 차이가 "DFL 분포의
   뾰족함(confidence)"이라는 **구체적 매개변수**를 통해 Softmax 연산에서 비선형적으로
   증폭됐을 가능성을 새로 제시한다(C) — 이 매개변수 자체는 검증되지 않았다.

### 17-7. 갱신된 결론

§13("현재 가장 유력한 원인")의 **BN 통계 불안정성 단독 설명은 이번 분석으로 약화됐다.**
대신 다음이 새로운 1순위 후보다:

> **Detect head(box+cls 양쪽 브랜치)에서 누적된 완만한 quantization 오차가, DFL
> Softmax 연산(`model.22.dfl.Softmax`)을 통과하며 matched/v02에서만 크게 증폭된다.**
> symmetric은 이 지점에서 오차가 거의 증폭되지 않는다(0.9965→0.9850, -0.0115) 반면
> matched/v02는 급격히 증폭된다(-0.067, -0.079). backbone/neck 전반의 산발적 오차 차이도
> 함께 존재해, 단일 원인이 아니라 **전반적 누적 + Softmax 지점의 국소 증폭이 결합된
> 그림**이다.

`cls` loss weight(§2, §12) 차이는 여전히 유효한 후보다 — classification 강도가 3배
높았던 symmetric의 DFL/cls 분포가 quantization에 덜 민감한 형태(더 완만한 분포? 더
sharp한 분포?)로 학습됐을 가능성은 이번 분석과 상충하지 않으나, "분포 뾰족함"이라는
구체적 매개변수를 직접 측정하지 않았으므로 여전히 가설 수준이다.

### 17-8. 다음 최소 검증 실험 (갱신, 제안만)

1. **(최우선) DFL pre-softmax logit 분포의 뾰족함(entropy/max-prob) 비교**: idx192
   (`Concat__model.22_Concat_1_output_0_Y`, Softmax 직전 golden 값)의 raw bin을 16-bin
   단위로 reshape해 anchor별 softmax 이전 분포의 entropy 또는 top1 확률을 세 모델에서
   계산 — symmetric의 분포가 matched/v02보다 덜 뾰족한지(완만한지) 직접 확인. 재추론·
   재컴파일 불필요, 기존 golden_tensor dump만 사용.
2. Detect head cv3(classification) 브랜치의 §17-4 Top 지점(idx126,162,182,186)에 대해
   동일한 golden vs sim raw 비교(§17-2 방식)를 반복해 clipping 여부를 조금 더 확인 —
   현재는 Sigmoid_output0_Y 한 텐서만 raw 비교했다.
3. (재학습 필요, 범위 밖) `cls` loss weight만 바꾼 재학습 후 동일 layer_diff 분석을
   반복해, idx196 증폭이 실제로 줄어드는지 확인하는 것이 가장 직접적인 인과관계 검증이나
   이번 세션 범위를 벗어난다.

---

## 18. DFL Softmax 구간(idx192→196) 3단계 분해 — §17 가설 정정 (2026-08-25)

**목적**: §17이 "DFL Softmax 연산에서 오차가 증폭된다"고 잠정 결론지은 것을, Softmax
입력(pre)/PC-softmax/실제 NVT-softmax(post) 세 단계로 분해해 **"Softmax 자체가 원인인지,
Softmax 이전 단계가 원인인지"**를 분리 검증했다. **결과: §17의 "Softmax가 증폭한다"는
표현은 부정확했다 — 실제로는 Softmax 이전(pre-softmax logit) 단계에서 이미 오차가
결정되고, Softmax 연산 자체(NVT 구현 포함)는 오차를 추가로 만들지 않는다.** 아래에서
§17의 잠정 결론과 이번 정정을 구분해서 기록한다.

### 18-1. 사전 확인 — idx192는 pre-softmax tensor가 아니었다 (경로 수정)

작업 지시서는 "idx192(Softmax 직전 tensor)"를 전제했으나, **실제 ONNX 그래프를 따라가
보면 idx192(`Concat__model.22_Concat_1_output_0_Y`)는 classification 브랜치의 concat
(5class×2100anchor=10,500 elements, `.bin` 파일 42,000 bytes로 실측 확인)이지 Softmax의
직접 입력이 아니다.** Softmax(idx196)의 실제 직접 입력은 **idx193
(`Transpose__model.22_dfl_Transpose_output_0_Y`)**이며, 이는 idx190
(`Concat__model.22_Concat_output_0_Y`, box 브랜치 concat)을 그대로 transpose한 것 —
둘 다 4side×16bin×2100anchor = 134,400 elements(537,600 bytes)로 shape가 정확히
일치함을 실측 확인했다(추측이 아니라 파일 크기로 검증). ONNX 그래프에서 `Transpose`
노드의 `perm=[0,2,1,3]`, `Softmax` 노드의 `axis=1`도 직접 읽어 확인 — ultralytics DFL
표준 구현(`reshape(b,4,16,a) → transpose(2,1) → (b,16,4,a) → softmax(axis=1)`)과 일치한다.
**이번 분석은 idx193(Transpose 출력)을 pre-softmax tensor로 사용했다** — idx192가 아니다.

reshape 검증: flat 134,400 값을 `(16,4,2100)`로 reshape 후 `(4,2100,16)`으로
transpose(side, anchor, bin)한 것이 올바른지, **golden pre-softmax에 PC(numpy) softmax를
적용한 결과가 golden post-softmax dump(idx196)와 실제로 일치하는지**로 검증했다 — 세
모델 전부 **cosine 1.000000**(완전 일치)으로 확인됨(추측 아닌 실측 검증).

### 18-2. pre-softmax(idx193) 분포 비교 — golden(Float) 기준

| 항목 | symmetric | matched | v02 |
|---|---:|---:|---:|
| logit min | -4.551 | **-98.233** | **-169.340** |
| logit max | 15.064 | 29.641 | 15.175 |
| logit mean | 2.000 | -13.122 | -15.970 |
| **logit std** | **3.146** | **15.243** | **16.812** |
| p95 / p99 / p99.9 | 7.980 / 10.649 / 12.731 | 3.720 / 6.604 / 9.751 | 3.533 / 6.718 / 8.587 |
| top1 logit(mean) | 8.080 | 2.082 | 1.670 |
| top1-top2 margin(mean) | 0.615 | 1.355 | 1.307 |
| top1 확률(softmax 후, mean) | 0.520 | 0.622 | 0.638 |
| entropy(mean, max=ln16=2.773) | **1.156** | **0.900** | **0.859** |

**가설과 반대 방향**: "symmetric이 덜 뾰족(sharp)해서 유리하다"는 가설을 미리 정답으로
두지 않고 실측한 결과, **오히려 matched/v02의 분포가 margin은 더 크고 entropy는 더
낮다 — 즉 matched/v02가 더 확신적(sharp)이고 symmetric이 상대적으로 덜 확신적(완만)
이다.** "분포가 sharp할수록 quantization에 취약하다"는 원래 가설과는 반대 방향의 관측이다
— **엔트로피/뾰족함 자체는 이번 분석에서 원인으로 지지되지 않는다.**

대신 압도적으로 큰 차이는 **dynamic range(std, min/max)** 다 — matched(std=15.24,
min=-98.2)와 v02(std=16.81, min=-169.3)는 symmetric(std=3.15, min=-4.6)보다 **약
5~5.3배 넓은 표준편차, 20~37배 넓은 최소값**을 갖는다. 이는 §3-1에서 확인한 BN
running_var 이상치(model.0 등)와 같은 방향의 신호가 **최종적으로 DFL box-regression
출력 지점에 누적된 형태**로 다시 나타난 것으로 해석된다(§3-1이 예측했던 "model.0에서
바로 나타난다"는 아니지만, "네트워크 어딘가에 누적된 넓은 dynamic range가 최종 출력에
드러난다"는 더 넓은 의미의 방향은 일치한다).

### 18-3. A/B/C 3단계 오차 분해

- **A** = golden pre-softmax(idx193) → PC(numpy) softmax
- **B** = simulator pre-softmax(idx193, dequantized) → 동일 PC softmax
- **C** = 실제 simulator/NVT post-softmax(idx196 dump)

| Model | pre-logit(golden vs sim) MAE / cos | A vs B (MAE/cos) | B vs C (MAE/cos) | A vs C (MAE/cos) |
|---|---|---|---|---|
| symmetric | 0.1282 / 0.99752 | 0.00787 / 0.98498 | **0.00021 / 1.00000** | 0.00787 / 0.98498 |
| matched | **2.3985** / 0.98974 | 0.02492 / 0.91264 | **0.00024 / 1.00000** | 0.02491 / 0.91263 |
| v02 | **2.0888** / 0.99289 | 0.02529 / 0.90582 | **0.00026 / 1.00000** | 0.02526 / 0.90577 |

**핵심 관찰**:
1. **B vs C(동일 입력에 대한 NVT Softmax 구현 vs PC Softmax)는 세 모델 모두
   사실상 완벽히 일치한다**(cos=1.00000, MAE 0.0002~0.0003) — **NVT의 Softmax 연산
   자체는 정확하다. 추가 오차를 만들지 않는다.**
2. **A vs C(전체 stage 오차)는 A vs B와 거의 동일하다** — 즉 최종적으로 관측되는 모든
   오차는 **pre-softmax logit의 quantization 오차**에서 온 것이지 Softmax 연산 자체에서
   온 것이 아니다.
3. **pre-softmax logit 자체의 raw 오차(MAE)가 matched(2.40)/v02(2.09)는 symmetric(0.128)
   보다 이미 16~19배 크다** — Softmax를 통과하기 **전부터** 압도적으로 큰 차이가
   존재한다.

### 18-4. error amplification 비교

| Model | pre-logit MAE | post(A-B) MAE | amplification(post/pre) | pre cos | post(A-B) cos |
|---|---:|---:|---:|---:|---:|
| symmetric | 0.1282 | 0.00787 | **0.061** | 0.99752 | 0.98498 |
| matched | 2.3985 | 0.02492 | **0.010** | 0.98974 | 0.91264 |
| v02 | 2.0888 | 0.02529 | **0.012** | 0.99289 | 0.90582 |

**amplification 비율은 symmetric이 오히려 더 크다(0.061 vs 0.010~0.012)** — softmax가
matched/v02의 오차를 "더 증폭"시키는 것이 아니라, **오히려 symmetric보다 덜 증폭(비율
기준)** 시킨다. 이는 softmax의 포화(saturation) 특성과 일치한다 — 입력 logit 오차가 이미
크면(matched/v02), softmax 출력은 이미 극단(0 또는 1 근처)에 가까워 추가적인 입력 오차가
출력 확률에 미치는 한계 영향이 작아진다(diminishing returns). **cosine 기준으로 보면
post-softmax 격차(A-B cos 0.91~0.91 vs 0.985)가 크게 벌어져 보이지만, 이는 "softmax가
증폭해서"가 아니라 "원래 입력 오차 자체가 워낙 커서 코사인 유사도가 떨어진 것"이다** —
metric을 하나만 보면(cosine만) 오판할 수 있다는 지시사항의 경고가 실제로 들어맞은
사례다.

**판단**: Case A(pre-softmax는 비슷한데 post에서만 급증)는 **기각**. Case C(NVT
Softmax 구현 자체가 원인)도 **기각**(B vs C가 완벽히 일치). **Case B(matched/v02가
pre-softmax 단계부터 이미 훨씬 나쁨, Softmax는 원인이 아니라 그 결과를 그대로
통과시키는 정확한 연산일 뿐)가 채택된다.**

### 18-5. DFL expectation decode(box distance) 오차

Softmax 확률(A/B/C 각각)에 `Σ(prob × bin_index)`로 DFL expectation decode를 적용해
box distance(stride 단위)를 계산 후 비교:

| Model | A-vs-B distance MAE / max / p99 | B-vs-C distance MAE / max / p99 | A-vs-C distance MAE / max / p99 |
|---|---|---|---|
| symmetric | 0.1147 / 3.37 / 0.927 | 0.0177 / 0.10 / 0.053 | 0.1162 / 3.36 / 0.936 |
| matched | 0.3712 / 8.34 / 2.786 | 0.0196 / 0.08 / 0.055 | 0.3694 / 8.36 / 2.786 |
| v02 | 0.3373 / 4.60 / 2.285 | 0.0201 / 0.11 / 0.053 | 0.3430 / 4.60 / 2.301 |

**B-vs-C(디코드 단계에서 NVT Softmax 구현 자체의 오차)는 세 모델 모두 매우 작고
비슷하다(MAE 0.018~0.020)** — 다시 한번 NVT Softmax/decode 자체는 정확함을 확인.
**A-vs-C(실제 최종 오차)는 matched(0.369)/v02(0.343)가 symmetric(0.116)의 약 3.0~3.2배**
— §18-3의 확률 오차가 그대로 box distance 오차로 이어진다. side(4방향, ultralytics
표준 순서 l/t/r/b로 추정 — 이번 분석에서 순서 자체를 독립 재검증하지는 않음) 평균 오차는
symmetric이 0.093~0.159로 비교적 고르게 분포한 반면, matched는 한쪽(0.577, 다른 세
방향의 약 2배)에 오차가 집중되고 v02도 두 방향(0.376/0.400)에 다소 집중된다 — 비대칭
패턴이 있으나 side 순서를 독립 검증하지 않아 특정 방향(l/t/r/b 중 어느 쪽인지)을
확정하지는 않는다.

### 18-6. 판단 (지시사항 §7 질문에 대한 답)

1. **Softmax는 근본 원인인가/증폭기인가/영향 작은가?** → **영향 작음(정확히는 "증폭기
   아님, 오히려 상대적으로 압축/포화")**. NVT Softmax 구현은 정확하고(B≈C), 증폭
   비율은 symmetric이 더 크다.
2. **matched/v02의 문제는 Softmax 입력 전에 이미 발생하는가?** → **그렇다, 명확히
   확인됨.** pre-softmax logit MAE가 이미 16~19배 크다.
3. **NVT Softmax 구현 자체가 추가 오차를 만드는가?** → **아니다.** B vs C cosine
   1.00000, 세 모델 동일.
4. **symmetric의 pre-softmax 분포 특성이 다른 두 모델과 실제로 다른가?** → **다르다,
   그러나 방향이 가설과 반대다.** entropy/margin 기준으로는 matched/v02가 더
   sharp(확신적)하다(가설과 반대). 실제로 다른 것은 **dynamic range**(std 3.15 vs
   15.2~16.8, min -4.6 vs -98~-169)다.
5. **DFL Softmax 오차가 box distance 오차로 실제 연결되는가?** → **그렇다.** 확률 오차
   비율(3배 안팎)이 거의 그대로 distance 오차 비율(3.0~3.2배)로 이어진다.
6. **학습 recipe → Detect head quantization sensitivity → "Softmax 증폭" 가설**: →
   **부분 강화, 메커니즘 정정.** "Softmax가 증폭한다"는 §17의 표현은 **약화/정정**됐다
   (오히려 반대 방향). 그러나 "학습 recipe 차이가 Detect head(특히 DFL box-regression
   출력)의 quantization 민감도를 다르게 만든다"는 상위 가설은 **더 정밀한 형태로
   강화됐다** — 정확한 메커니즘은 "pre-softmax logit의 dynamic range(std, 극단값)가
   matched/v02에서 5~30배 넓어 그 tensor의 INT8 quantization 오차 자체가 크고, Softmax는
   그 오차를 그대로(왜곡 없이) 통과시킬 뿐"이라는 것으로 수정한다.

**cls 브랜치 참고**: 이번 분석은 idx192→196(box/DFL 경로)에만 집중했다. §17-4에서 cv3
(classification) 브랜치도 Top error에 다수 포함됐던 것은 이번 분석 범위 밖이며, DFL 경로
오차만으로 전체 quantization loss 크기를 완전히 설명하기는 어려울 수 있다 —
**cls branch도 별도 후속 분석 필요**로만 남긴다(이번 세션에서 추가 분석/ablation하지
않음).

### 18-7. 다음 최소 검증 실험 (갱신, 제안만)

1. **pre-softmax logit(idx193) dynamic range의 근원 추적**: 이 tensor는
   `model.22.cv2.*`(box 브랜치) 최종 conv 출력이 reshape/transpose된 것이므로,
   `model.22.cv2.2.2`(및 scale0/1의 대응 conv) 자체의 학습된 weight/bias 통계(§3-1과
   같은 방식)를 직접 다시 확인해, 이 conv의 출력 dynamic range가 왜 matched/v02에서
   커지는지(weight 자체가 큰지, 그 앞단 BN 통계가 누적된 결과인지) 추적 — 재추론
   불필요, 기존 `best.pt`에서 바로 확인 가능.
2. 위 1번으로 원인 conv/BN을 특정하면, 그 지점의 calibration range를 gen_config.txt의
   `spec_demarc_tensors`에 추가 보호하는 것이 실사용 가능한 완화책이 될 수 있는지
   검토(단, 이는 quantization 설정 변경이라 이번 세션 범위 밖 — 제안만).

---

## 19. optimizer 단독 ablation — 새 학습 결과로 확인 (2026-08-31)

**목적**: §18-7이 제안한 다음 실험(1번, conv/BN 통계 추적) 대신, 사용자가 다른 학습
PC(exp.yaml 코멘트 표기 "160서버")에서 **정확히 이 조사를 위해 설계한 2단계 optimizer/lr
ablation**을 완료해 결과를 가져왔다 — 재추론·재학습 없이 이미 완료된 두 체크포인트만
확인했다.

### 19-1. 실험 설계 (exp.yaml 코멘트로 확인, 추측 없음)

| | 실험① `v02_symmetric_sgd_lr0001` | 실험② `v02_symmetric_adamw_lr0001` |
|---|---|---|
| config | `configs/experiments/BSD_F606/Rear_Truck/v02_symmetric_sgd_lr0001/exp.yaml` | `configs/experiments/BSD_F606/Rear_Truck/v02_symmetric_adamw_lr0001/exp.yaml` |
| model_name/trainer | `symmetric`(`SymmetricTrainer`) | `symmetric`(`SymmetricTrainer`) |
| data | `configs/datasets/bsd_rear_truck.yaml`(v01_symmetric/matched와 동일 데이터 계열) | 동일 |
| **optimizer** | **SGD** | **AdamW** |
| lr0 / lrf | 0.001 / 0.01 | 0.001 / 0.01 (①과 동일) |
| cos_lr | true | true (①과 동일) |
| batch | 48 | 48 (①과 동일) |
| cls (loss weight) | 1.5 | 1.5 (①과 동일 — v01_symmetric과 같은 값) |
| box / dfl | 7.5 / 1.5 | 7.5 / 1.5 |
| 학습 종료 | epoch 188(patience 15) | epoch 267(patience 15) |

**①은 v01_symmetric(SGD/lr0=0.01/cos_lr=true/batch=48/cls=1.5) 대비 lr0만 0.01→0.001로
낮춘 것 — optimizer는 그대로 SGD.** **②는 ① 대비 optimizer만 SGD→AdamW로 바꾼 것 —
lr0(0.001)/cos_lr(true)/batch(48)/cls(1.5)는 ①과 완전히 동일.** 즉 ①↔② 비교는
**optimizer 하나만 통제된 순수 ablation**이고, baseline(v01_symmetric)↔①은 "lr 단독
효과"를 보기 위한 것(exp.yaml 자체가 "baseline과 ②를 직접 비교하지 말 것"이라고 명시).

### 19-2. BN 통계 확인 (§3-1과 동일 진단, `torch.load`로 재확인)

| | 실험① (SGD, lr0=0.001) | 실험② (AdamW, lr0=0.001) | 참고: 기존 3모델(§3-1) |
|---|---:|---:|---|
| BN 레이어 수 | 57 | 57 | 57(공통) |
| running_var max>500인 레이어 수 | **1** | **10** | symmetric 1 / matched 7 / v02 6 |
| running_var에 **Inf 포함**된 레이어 수 | **0** | **2(!)** | symmetric 0 / matched 0 / v02 1 |
| `model.0.bn.running_var` 최대값(유한) | 5,140 | 24,976 | symmetric 14,432 / matched 58,880 / v02 46,176 |
| `model.0.bn.running_var` Inf 채널 | 없음 | **채널 11, 14** | v02만 채널4 1개 |

**실험①(SGD 유지, lr0만 낮춤)은 v01_symmetric과 같은 "건강한" BN 패턴을 그대로
유지한다**(이상치 레이어 1개, Inf 없음 — lr0를 10분의 1로 낮춰도 문제가 재현되지
않았다). **실험②(optimizer만 AdamW로 전환)는 즉시 이상치 레이어가 10개로 늘고, Inf가
2개 채널에서 나타난다** — 기존 v02(Inf 1채널)보다도 심하다. **lr0/cos_lr/batch/cls를
전부 고정한 채 optimizer만 바꿨는데도 BN 불안정성이 그대로 재현/악화됐다.**

### 19-3. 판단

§12(원인 후보 우선순위화)에서 "가능성 중간"으로 분류했던 **optimizer 축(SGD vs AdamW)이,
이번 통제된 ablation으로 lr0/cos_lr/batch/cls와 독립적으로 BN 불안정성을 유발한다는
것이 확인됐다** — **가능성 중간(B) → 가능성 높음(A)로 격상한다.**

- lr0 크기 자체(0.01→0.001)는 원인이 아니다 — 실험①이 이를 반증한다(SGD를 유지한 채
  lr0만 10배 낮춰도 BN이 건강하다).
- `cls` loss weight(1.5)도 단독 원인이 아니다 — 실험①/②모두 `cls=1.5`인데 결과가
  갈린다. §18까지의 "cls weight 가설"은 **약화**된다(적어도 optimizer가 SGD인 한, cls=1.5
  이어도 BN은 안정적이었다 — cls weight가 관여한다면 optimizer와의 상호작용 형태일
  가능성만 남는다).
- batch size(48, 고정)도 원인이 아니다.
- **남은 것은 optimizer(SGD vs AdamW) 단독** — 이번 ablation에서 유일하게 변한
  변수이자 유일하게 결과가 갈린 지점이다.

**다만 이번 확인은 PT 체크포인트 레벨(BN 통계)까지다.** §18의 pre-softmax logit dynamic
range나 실제 board quantization loss로 이어지는지는 아직 확인하지 않았다 — ONNX
export/NVT 3-tensor compile/board 실측이 필요하며 이번 세션에서는 수행하지 않았다(§19-4
참조).

### 19-4. 아직 확인되지 않은 것 / 다음 단계

- 이번 두 체크포인트는 **아직 ONNX export도, NVT 변환도, board 실측도 되지 않았다** —
  BN 통계만으로 "quantization loss가 실제로 symmetric처럼 작을 것"이라고 단정하지
  않는다(상관관계는 강하지만 §18의 전체 인과 사슬 — pre-softmax logit dynamic range →
  quantization 오차 → box distance 오차 — 을 이 두 체크포인트에서 재확인해야 완결된다).
- 다음 최소 검증 실험(제안만, 실행 안 함): 실험①/② 각각 `tools/export_custom.py -i 320
  -y 320`으로 export 후 §3-1/§18과 동일한 BN/pre-softmax dynamic range 진단을 반복하고,
  가능하면 3-tensor NVT compile까지 진행해 실제 quantization loss가 실험①(SGD)은
  낮고 실험②(AdamW)는 높게 나오는지 확인 — 이것이 optimizer 가설을 완전히 확정하는
  마지막 단계다.
- optimizer가 왜 이런 BN 불안정성을 유발하는지의 메커니즘(예: AdamW의 adaptive
  per-parameter lr이 BN 통계를 갱신하는 momentum과 상호작용하는 방식 등)은 이번
  조사에서 다루지 않았다 — 별도 조사 필요.

### 19-5. ONNX export + NVT 3-tensor compile/simulate 완료 (2026-08-31) — §19-4 실행

§19-4가 "다음 최소 검증 실험"으로 제안한 것을 그대로 실행했다: 실험①/② 각각
`tools/export_custom.py -i 320 -y 320`로 ONNX export(`skip_postprocess=True`,
`(1,5,2100)+(1,4,2100)` 구조, 다른 모든 모델과 동일 — shape로 확인), 다른 3-tensor
baseline과 완전히 동일한 조건(`gen_config.txt`/`sim_config.txt` model_dir 외 byte
동일, calibration 데이터 symlink 동일 대상, `spec_demarc_tensors` 동일 — diff로 확인)으로
NVT compile + simulator까지 진행했다. 재컴파일 아님(§19 시점엔 존재하지 않던 신규
디렉토리 최초 생성), 기존 3모델 산출물은 전혀 건드리지 않았다.

- **리소스**: 두 실험 모두 `nvt_model.bin` 4,711,336 bytes, latency 6.36765ms, DRAM
  36,540,852 bytes — 기존 모든 3-tensor baseline과 동일(아키텍처가 같으므로 예상대로).
- **Simulator cosine** (동일 테스트 이미지 `..._0126.jpg`, 다른 모든 모델과 동일):

| Model | optimizer / lr0 | Softmax(dfl) | Sigmoid[OUT] | dfl_conv[OUT] |
|---|---|---:|---:|---:|
| v01_symmetric(baseline) | SGD / 0.01 | 0.9850 | 0.9824 | 0.9933 |
| **실험① sgd_lr0001** | **SGD** / 0.001 | **0.9955** | 0.9779 | **0.9977** |
| **실험② adamw_lr0001** | **AdamW** / 0.001 | **0.9170** | **0.7969** | 0.9650 |
| v01_symmetric_matched | AdamW / 0.001 | 0.9126 | 0.9473 | 0.9490 |
| v02 | AdamW / 0.001(dair trainer) | 0.9058 | 0.9236 | 0.9664 |

**PT 레벨(§19-2 BN 통계)에서 확인된 패턴이 NVT 레벨에서도 그대로, 오히려 더 뚜렷하게
재현됐다**:
- **실험①(SGD 유지, lr0만 낮춤)은 v01_symmetric baseline과 동등하거나 더 좋다** —
  Softmax(dfl)/dfl_conv는 baseline보다도 높다. lr0를 10분의 1로 낮춰도 SGD를 쓰는 한
  quantization 친화적 특성이 그대로 유지된다.
- **실험②(optimizer만 AdamW로 전환)는 세 지표 모두 급격히 나빠진다** — 특히
  **Sigmoid[OUT]=0.7969**는 이번까지 측정된 모든 Rear_Truck 모델(v01/v02/matched/dair
  포함) 중 **가장 낮은 값**이다(기존 최저치였던 v02의 0.9236보다도 훨씬 나쁨).
- lr0(0.001)/cos_lr(true)/batch(48)/cls(1.5)가 실험①/② 사이에서 완전히 동일한데도
  이 정도 격차가 난다는 것은, **optimizer 선택이 quantization 결과에 미치는 영향이
  단독으로도 matched/v02 수준(또는 그 이상)의 손상을 만들기에 충분하다**는 것을
  NVT 레벨에서 직접 확인한 것이다.

**§19-3의 판단이 강화된다**: optimizer(SGD vs AdamW)는 "가능성 높음(A)" 후보로 격상된
데서 더 나아가, **이번 NVT 실측으로 최소한 Softmax/Sigmoid/dfl_conv 3개 protected
tensor 수준에서는 optimizer 단독으로 matched/v02급 손상을 재현하는 것이 실증됐다.**

**아직 하지 않은 것**: 이번 §19-5는 3개 protected tensor의 simulator cosine까지다.
board 실측(물리 하드웨어 필요, §9의 관례대로 사용자가 직접 진행), 6,152장 전체
Float/Board mAP 평가, §18 방식의 idx193/196 pre-softmax dynamic range 재분석(이
두 신규 모델에 대해서는 아직 수행하지 않음)은 다음 단계로 남는다.

---

## 20. 실험①/② Float ONNX 6,152장 평가 + 실험① Board 실측 (2026-08-31)

§19-5가 남겨둔 다음 단계(6,152장 Float/Board mAP 평가)를 진행했다. **실험②의 board
실측은 이 시점에 사용자가 물리 보드에서 진행 중 — 아직 결과 없음.** 실험①의 board
실측은 완료돼 아래에 반영한다.

### 20-1. 신규 추가 파일 (config만, 코드 없음)

- `runs/_eval_scratch/configs/onnx/onnx_v02_symmetric_{sgd,adamw}_lr0001_conf001_dualnms_cfg.yaml`
- `runs/_eval_scratch/configs/onnx/onnx_v02_symmetric_{sgd,adamw}_lr0001_stdnms_conf001_cfg.yaml`
- `runs/_eval_scratch/configs/board/board_v02_symmetric_sgd_lr0001_3tensor_cfg.yaml`(실험①만 —
  실험②는 board dump가 아직 없어 config 미생성)

조건은 다른 모든 모델과 완전히 동일: `confidence_thres=0.001`(golden, `tools/debug/inf_conf001.py`),
dual-threshold(`min_overlap_thres=0.6`/`union_overlap_thres=0.3`)/표준 단일-IoU
(`iou_thres=0.45`) NMS, GT 6,152장(`runs/_eval_scratch/ipu_gt`), pycocotools/ultralytics
`ap_per_class` 양쪽 평가.

### 20-2. Float ONNX mAP50-95 (6,152장)

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---:|---:|---:|---:|
| v01_symmetric(baseline) | 0.562 | — | 0.558 | 0.5785 |
| v01_dair | 0.533 | 0.5521 | 0.524 | 0.5411 |
| v02 | 0.538 | 0.5581 | 0.531 | 0.5493 |
| v01_symmetric_matched | 0.549 | 0.5699 | 0.541 | 0.5593 |
| **실험① sgd_lr0001** | 0.532 | 0.5551 | 0.528 | 0.5474 |
| **실험② adamw_lr0001** | **0.412** | 0.4399 | 0.403 | 0.4283 |

**실험①(SGD, lr0=0.001)은 다른 모델들과 비슷한 Float 수준**(0.53대)이다. **실험②(AdamW,
lr0=0.001)만 Float 자체가 다른 모든 모델보다 뚜렷이 낮다(0.40~0.44대)** — 이는
**quantization 단계 이전, PyTorch/ONNX float 추론 단계에서 이미 성능이 저하돼 있다는
뜻**이다. §19-2에서 확인한 `model.0.bn.running_var`의 실제 Inf(채널 2개)가 float
추론에서도 해당 채널을 그대로 0으로 만들기 때문일 가능성이 높다(BatchNorm eval-mode
공식 `(x-mean)/sqrt(var+eps)`에서 `var=inf`이면 결과가 0으로 수렴 — PyTorch/ONNX
양쪽 모두 동일하게 적용되는 수학적 사실이며, 이번 세션에서 float 추론 결과로 직접
재확인한 것이지 추정만은 아니다). **이 발견은 §19-3의 "optimizer가 quantization
민감도를 만든다"는 판단에 새로운 주의사항을 추가한다**: 이번 특정 조합(AdamW + lr0=0.001
+ cos_lr=true + batch=48 + cls=1.5)에서는 optimizer 문제가 quantization 단계뿐
아니라 **학습(float 성능) 단계에서부터 이미 드러났다** — 즉 "이 조합은 quantization에만
나쁜 게 아니라 애초에 학습 자체가 불안정했다"는 뜻이며, 아래 §20-4의 손실 %(만약 board가
낮게 나오더라도)를 해석할 때 반드시 **절대 mAP과 함께** 봐야 한다(분모인 Float 자체가
낮으면 상대 손실 %가 작아 보일 수 있다는 왜곡 위험).

### 20-3. 실험① Board 실측 결과

사용 파일: 원본 덤프 `runs/_eval_scratch/board_dumps/v02_symmetric_sgd_lr0001/v02_symmetric_sgd_lr0001_output.json`
(사용자가 실제 보드에서 직접 추론) → `merge_board_dump.py` →
`runs/_eval_scratch/merged/v02_symmetric_sgd_lr0001/ipu_pred_merged_v02_symmetric_sgd_lr0001_3tensor.json`
→ `board_v02_symmetric_sgd_lr0001_3tensor_cfg.yaml`.

무결성 확인: 6,152장/219,229 detection 정확히 매칭, 다른 5개 모델(production/v01_symmetric/
v01_dair/v02/v01_symmetric_matched) 병합 결과와 detection 단위 jaccard overlap **전부
0.0000**(독립적 유효 덤프, 다른 모델 dump와 중복/재사용 아님을 확인).

| | mAP50 | mAP50-95 |
|---|---:|---:|
| Float(dual, pycoco) | 0.754 | 0.532 |
| **Board(pycoco)** | 0.730 | **0.490** |
| Float(dual, ultra) | 0.7743 | 0.5551 |
| **Board(ultra)** | 0.7604 | **0.5223** |

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold | **-7.89%** | **-5.91%** |
| 표준 단일-IoU | **-7.20%** | **-4.59%** |

### 20-4. 종합 비교표 갱신

| Model | Trainer | Optimizer | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---|---|---:|---:|---:|---:|
| v01_symmetric | symmetric | SGD | -7.30% | -1.96% | -6.63% | -0.62% |
| **실험① sgd_lr0001** | symmetric | **SGD** | **-7.89%** | **-5.91%** | **-7.20%** | **-4.59%** |
| v01_dair | dair | AdamW | -21.01% | -16.32% | -19.66% | -14.62% |
| v02 | dair | AdamW | -19.14% | -14.80% | -18.08% | -13.44% |
| v01_symmetric_matched | symmetric | AdamW | -28.42% | -27.00% | -27.36% | -25.62% |
| 실험② adamw_lr0001 | symmetric | AdamW | 실측 진행 중 | 실측 진행 중 | 실측 진행 중 | 실측 진행 중 |

### 20-5. 판단 (§19-3 갱신)

**실험①(SGD, lr0=0.001)의 board 실측 손실(-4.6~-7.9%)은 v01_symmetric(-0.62~-7.30%)과
같은 그룹(한 자릿수)에 속하고, AdamW 계열 3개(-13~-28%, 전부 두 자릿수)와는 명확히
구분된다.** §19(PT 레벨 BN 통계)·§19-5(NVT simulator cosine)에서 나온 "SGD는 건강,
AdamW는 손상"이라는 신호가 **실제 물리 board mAP으로도 그대로 이어지는 것을 이번
실측으로 확인했다** — 이 조사에서 optimizer 가설을 뒷받침하는 가장 직접적인 증거다.

다만 두 가지 주의점을 남긴다:
1. **실험①은 lr0(0.01→0.001)도 함께 바뀐 것**이라 "optimizer 단독 효과"를 보려면
   실험②(동일 lr0=0.001, optimizer만 AdamW)와의 비교가 필요하다 — 이는 실험②의 board
   실측이 나와야 완결된다(진행 중, §20-6).
2. 실험①의 ultralytics 평가 손실(-4.6~-5.9%)은 v01_symmetric의 ultralytics 손실
   (-0.62~-1.96%)만큼 작아지지는 않았다 — lr0를 낮춘 것 자체(또는 lr0-optimizer 상호작용)가
   약간의 잔여 효과를 남겼을 가능성은 열어둔다.

### 20-6. 실험② Board 실측 결과 (2026-08-31 추가)

사용 파일: 원본 덤프 `runs/_eval_scratch/board_dumps/v02_symmetric_adamw_lr0001/v02_symmetric_adamw_lr0001_output.json`
(사용자가 실제 보드에서 직접 추론, md5 `cbd3384e...`) → `merge_board_dump.py` →
`runs/_eval_scratch/merged/v02_symmetric_adamw_lr0001/ipu_pred_merged_v02_symmetric_adamw_lr0001_3tensor.json`
→ `board_v02_symmetric_adamw_lr0001_3tensor_cfg.yaml`(신규 생성, 실험①과 동일 구조).

무결성 확인: 6,152장/201,292 detection 정확히 매칭(병합 후 set 중복 제거 시 201,290,
score 반올림에 의한 것으로 실질적 차이 아님). md5가 기존 8개 병합/원본 dump와 전부 다름.
다른 8개 병합 결과(production/v01_940calib/v01_symmetric/v01_dair/v02/
v01_symmetric_matched/실험①)와 detection 단위 jaccard overlap **전부 0.0000**
(독립적 유효 덤프).

| | mAP50 | mAP50-95 |
|---|---:|---:|
| Float(dual, pycoco) | 0.632 | 0.412 |
| **Board(pycoco)** | 0.503 | **0.246** |
| Float(dual, ultra) | 0.6604 | 0.4399 |
| **Board(ultra)** | 0.5429 | **0.2738** |
| Float(std, pycoco) | 0.615 | 0.403 |
| Float(std, ultra) | 0.6426 | 0.4283 |

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold (mAP50-95) | **-40.29%** | **-37.76%** |
| 표준 단일-IoU (mAP50-95) | **-38.96%** | **-36.07%** |
| dual-threshold (mAP50) | -20.41% | -17.79% |
| 표준 단일-IoU (mAP50) | -18.21% | -15.52% |

**실험②(AdamW, lr0=0.001)의 board 손실(-36~-40%, mAP50-95)은 이 조사 전체를 통틀어
최악이다** — 기존 최악이던 `v01_symmetric_matched`(-25.6~-28.4%)보다도 크게 나쁘다.
§20-2에서 경고한 대로, 이는 **두 가지가 합쳐진 결과**다: (1) Float 자체가 이미
저하돼 있었고(mAP50-95 0.40대, 다른 모델 0.53~0.58대 대비), (2) 그 위에 quantization
손실까지 다른 어떤 모델보다 크게 추가로 발생했다(-36~-40%, 다른 모델은 최대
-28% 수준). 절대 mAP로 보면 board(pycoco) mAP50-95=**0.246** — 이 조사에서 확인된
board 결과 중 가장 낮은 절대 성능이다.

### 20-7. 종합 비교표 최종 갱신

| Model | Trainer | Optimizer | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---|---|---:|---:|---:|---:|
| v01_symmetric | symmetric | SGD | -7.30% | -1.96% | -6.63% | -0.62% |
| **실험① sgd_lr0001** | symmetric | **SGD** | -7.89% | -5.91% | -7.20% | -4.59% |
| v01_dair | dair | AdamW | -21.01% | -16.32% | -19.66% | -14.62% |
| v02 | dair | AdamW | -19.14% | -14.80% | -18.08% | -13.44% |
| v01_symmetric_matched | symmetric | AdamW | -28.42% | -27.00% | -27.36% | -25.62% |
| **실험② adamw_lr0001** | symmetric | **AdamW** | **-40.29%** | **-37.76%** | **-38.96%** | **-36.07%** |

### 20-8. 최종 판단 (§19-3/§20-5 갱신, ①↔② 완결)

`lr0=0.001`, `cos_lr=true`, `batch=48`, `cls=1.5`를 전부 고정한 순수 optimizer
ablation(실험①=SGD, 실험②=AdamW)의 **board 실측이 이제 양쪽 다 확보됐다**:

- **실험①(SGD): -4.6~-7.9%** — v01_symmetric(SGD, -0.6~-7.3%)과 같은 그룹(한 자릿수).
- **실험②(AdamW): -36.1~-40.3%** — 이 조사에서 확인된 5개 AdamW 계열 결과
  (v01_dair -14.6~-21.0%, v02 -13.4~-19.1%, matched -25.6~-28.4%, 실험②
  -36.1~-40.3%) 중에서도 최악.

**optimizer(SGD vs AdamW)가 quantization 손실 크기를 결정한다는 가설은 이제
"순수 ablation + PT(BN 통계) + NVT(simulator cosine) + Float(6152장 mAP) +
Board(6152장 실측) 4개 레벨 전부에서 일관되게 재현된, 이 조사 전체에서 가장
강하게 뒷받침되는 결론"이다.** §20-5가 남겨둔 미확정 항목("실험② board 대기")은
이번 실측으로 해소됐다.

다만 실험②는 Float 자체가 저하돼 있었으므로(§20-2), "AdamW가 quantization
단계에서만 나쁘다"가 아니라 **"이 특정 조합(AdamW+lr0=0.001+cos_lr+batch48+cls1.5)은
학습 자체부터 불안정했고, 그 불안정성이 quantization 단계에서 한 번 더 증폭됐다"**가
더 정확한 서술이다. lr0=0.001이 AdamW 표준값(§18 실험 설계 근거, ultralytics
`cfg/default.yaml:94`)이었다는 점에서, 이 결과는 "AdamW가 원래 나쁘다"보다는
**"이 하이퍼파라미터 조합에서 이 데이터셋에 대해 AdamW가 SGD보다 불안정한 학습을
만들었다"**로 한정해서 읽어야 한다(다른 lr0/스케줄로 재현되는지는 미확인 — §21-7 참조).

### 20-9. 다음 단계

- **실행 완료, 대기 중인 실측 없음.** optimizer ablation(①/②)의 PT/NVT/Float/Board
  4-레벨 검증이 모두 끝났다.
- 남은 선택적 후속 작업(§21-7 참조, 실행 안 함, 요청 시에만): (a) §18 pre-softmax
  dynamic range 분해를 실험①/②에 적용, (b) optimizer→BN 불안정 메커니즘 규명,
  (c) pycocotools vs ultralytics 매칭 알고리즘 직접 비교, (d) `model.22.cv2.*`
  weight 통계 추적.

---

## 21. Context Handoff / Resume Checkpoint (2026-08-31, 세션 종료 시점)

### 21-1. 최종 목표

`v01_symmetric`만 NVT/Board quantization 손실이 작고(SGD), `v01_symmetric_matched`/`v02`/
`v01_dair`는 큰 이유(AdamW)를 학습→PT→ONNX→NVT→Board 전체 파이프라인에서 규명 중.
동시에 동일 모델·동일 board 덤프에서도 evaluator(pycocotools vs ultralytics)에 따라
손실%가 크게 갈리는 현상을 분리 검증 중.

### 21-2. 비교 대상

| Model | trainer | optimizer | lr0 | cos_lr | batch | cls | 비고 |
|---|---|---|---:|---|---:|---:|---|
| v01_symmetric | symmetric | SGD | 0.01 | true | 48 | 1.5 | baseline, 손실 최소 |
| v01_symmetric_matched | symmetric | AdamW | 0.001 | false | 32 | 0.5 | 손실 최대 |
| v02 | dair | AdamW | 0.001 | false | 32 | 0.5 | ablation(OverlayAug off) |
| v01_dair | dair | AdamW | 0.001 | false | 32 | 0.5 | 참고 |
| 실험① `v02_symmetric_sgd_lr0001` | symmetric | **SGD** | 0.001 | true | 48 | 1.5 | lr0만 다름(vs symmetric) |
| 실험② `v02_symmetric_adamw_lr0001` | symmetric | **AdamW** | 0.001 | true | 48 | 1.5 | optimizer만 다름(vs 실험①) |

핵심 경로: `runs/detect/BSD_F606/Rear_Truck/<model>/weights/{best.pt,best.onnx}`,
`runs/_eval_scratch/{configs,board_dumps,merged}/`, `novatek/convert/rear_truck_<model>_320/`.

evaluator: pycocotools(`projects/metrics/evaluator.py`+`customCOCOeval`, `customEval.py`)
vs ultralytics(`tools/debug/ultralytics_style_ap.py`, `ap_per_class`+`match_predictions`).
conf=0.001(golden, `tools/debug/inf_conf001.py`), NMS는 dual-threshold(board 재현)/표준
단일-IoU(`iou_thres=0.45`) 두 축.

### 21-3. 지금까지 확정된 사실 (실측)

- **Architecture/ONNX/NVT config**: 전 모델 동일(파라미터 3,011,823, state_dict 355키,
  ONNX `skip_postprocess=True` (1,5,2100)+(1,4,2100), gen_config.txt model_dir 외
  byte-identical, calibration 940장 동일 symlink, `nvt_model.bin` 4,711,336B 전부 동일).
- **PT→ONNX parity**: 전 모델 통계적으로 구분 불가.
- **BN running_var(57 layers) 이상치**: symmetric 1(Inf 0) / matched 7(Inf 0) / v02
  6(Inf 1) / 실험① 1(Inf 0) / 실험② 10(**Inf 2**).
- **NVT simulator cosine**(Softmax dfl/Sigmoid/dfl_conv): symmetric 0.985/0.982/0.993,
  matched 0.913/0.947/0.949, v02 0.906/0.924/0.966, 실험① **0.9955/0.9779/0.9977**,
  실험② **0.917/0.797/0.965**(Sigmoid 전 모델 중 최저).
- **§18 DFL Softmax 분해**(symmetric/matched/v02만, 실험①②는 미실행): pre-softmax(idx193,
  Transpose 출력, 134400=4×16×2100) logit MAE symmetric=0.128, matched=2.40, v02=2.09
  (16~19배차). NVT Softmax 자체(B vs C)는 cos=1.00000로 정확 — 원인 아님. 증폭비율
  (post/pre MAE)은 symmetric 0.061 > matched 0.010/v02 0.012(오히려 symmetric이 더 큼).
  dynamic range(std) symmetric 3.15 vs matched 15.24/v02 16.81, min -4.6 vs -98/-169.
- **Float ONNX mAP50-95(6152장,conf=0.001,dual/std × pycoco/ultra)**: symmetric
  0.562/—/0.558/0.5785, v01_dair 0.533/0.5521/0.524/0.5411, v02 0.538/0.5581/0.531/0.5493,
  matched 0.549/0.5699/0.541/0.5593, 실험① 0.532/0.5551/0.528/0.5474, **실험②
  0.412/0.4399/0.403/0.4283(다른 모델보다 뚜렷이 낮음 — Float 단계부터 저하)**.
- **Board quant loss %**(dual+pycoco/dual+ultra/std+pycoco/std+ultra): symmetric
  -7.30/-1.96/-6.63/-0.62, v01_dair -21.01/-16.32/-19.66/-14.62, v02
  -19.14/-14.80/-18.08/-13.44, matched -28.42/-27.00/-27.36/-25.62, **실험①
  -7.89/-5.91/-7.20/-4.59**(SGD 그룹과 동일 수준), **실험② board 실측 진행 중(미완료)**.
- **Evaluator 축 효과(%p)**: symmetric~5.3-6.0(최대) > v01_dair~4.7-5.0 > v02~4.3-4.6 >
  matched~1.4-1.7(최소, "이름 유사성 함정" 확인 사례). **NMS 축 효과는 전 모델 0.7-1.7%p로
  항상 작음.**

### 21-4. 가설 변화 이력

- **BN 가설**(model.0에서 오차가 처음 커진다) → **약화/반증**(§17): model.0 layer-diff
  cosine은 전 모델 거의 동일. 단, BN running_var 자체의 이상치(§3-1,§19)는 실재하며
  optimizer와 강하게 연동됨이 이후 확인됨(아래 optimizer 항목).
- **Detect head box branch(cv2.*) 가설** → **부분 강화, 범위 수정**(§17): box(cv2)뿐
  아니라 cls(cv3) 브랜치도 동등/더 크게 divergence — "Detect head 전체"로 수정.
- **DFL Softmax 증폭 가설** → **반증/정정**(§18): NVT Softmax 자체는 정확(B≈C
  cos=1.00000). 증폭비율은 오히려 symmetric이 더 큼. **"Softmax가 증폭한다"는 표현은
  틀렸다 — 원인 아님.**
- **pre-softmax logit dynamic range 가설** → **강화, 현재 유효**(§18): matched/v02
  logit std가 symmetric의 5~5.3배, min이 20~37배 넓음 — 이것이 quantization 오차 →
  DFL box distance 오차로 직결(A-vs-C distance error matched/v02가 symmetric의
  3.0~3.2배).
- **optimizer(SGD vs AdamW) 가설** → **가장 강하게 강화, 현재 1순위**(§19,§19-5,§20):
  lr0/cos_lr/batch/cls 전부 고정한 실험①(SGD)/②(AdamW) 순수 ablation에서 BN
  불안정성·NVT cosine·Float mAP·board mAP(실험①만 확인) 전부 optimizer 하나로 재현됨.
- **cls loss weight(1.5 vs 0.5) 가설** → **약화**(§19-3): 실험①/② 둘 다 cls=1.5인데
  결과가 갈림 — cls 단독 원인 아님(optimizer와 독립적으로 작용하지 않음이 확인됨).
- **lr0 크기 가설** → **반증**(§19-3): 실험①(lr0=0.001, SGD)이 v01_symmetric(lr0=0.01)과
  동등하게 건강 — lr0 크기 자체는 원인 아님.
- **batch size / NMS 종류** → **원인 아님**(배제).
- **ONNX export / NVT compile 설정** → **원인 아님**(완전 배제, byte-identical 확인).

### 21-5. 현재 가장 강한 결론

**1순위 원인: optimizer 선택(SGD vs AdamW).** 인과관계 증명 수준: lr0/cos_lr/batch/cls를
전부 고정한 순수 ablation(실험①/②)으로 **PT 체크포인트(BN 통계) 및 NVT simulator
cosine 레벨까지는 강한 인과 증거 확보**(다른 변수 통제된 통제실험). **Board mAP
레벨에서는 실험①(SGD, 건강)만 확인됐고 실험②(AdamW) board는 아직 미완료**이므로
"optimizer가 board 손실까지 결정한다"는 최종 확정은 **실험②의 board 실측 전까지는
아직 완결 전(강한 정황증거, 최종 미확정)**.

인과 사슬(실측 vs 추정 구분): 학습 optimizer 선택(실측 조작) → BN running_var
불안정성/Inf(실측) → [메커니즘: 추정, §21-7 참조] → pre-softmax DFL logit dynamic
range 확대(실측, symmetric/matched/v02만) → NVT quantization 오차(실측) → box distance
오차(실측) → board mAP 손실. **"BN 불안정성이 pre-softmax dynamic range를 직접
유발한다"는 링크 자체는 상관관계이지 직접 대조 검증되지 않음**(§18-7 제안, 미실행).

이전 결론과 충돌 시 최신 실험(§19~§20, optimizer ablation)을 정본으로 채택 — 특히
§17의 "Softmax 증폭" 및 §3의 "model.0이 최초 발화점" 서술은 §18/§19에서 갱신된 것으로
간주.

### 21-6. Evaluator 차이 관련 최신 확인

- **IoU threshold**: pycocotools(`Params` 기본값)와 ultralytics(`torch.linspace(0.5,0.95,10)`)
  **둘 다 0.50:0.05:0.95, 10개로 동일** — mAP50-95 계산에서 IoU threshold 개수/범위
  차이는 원인이 **아님**(코드로 확인).
- **AP 적분 방식**: 둘 다 표준 101-point COCO 보간(ultralytics `compute_ap` 주석에
  "101-point interp (COCO)" 명시, pycocotools stock 알고리즘 — `customCOCOeval`은
  `accumulate()`만 오버라이드, Params/보간 로직 불변) — 동일.
- **maxDets**: pycocotools 100 제한 vs ultralytics 무제한 — board 이미지당 detection
  최대 60건(6152장 전수 확인)이라 100 제한에 걸리는 사례 0건, **원인 아님**(배제 확인됨).
- **NMS/confidence**: evaluator 축과 별개 — dual-threshold vs 표준(iou=0.45) 둘 다
  손실%에 미치는 영향 작음(0.7~1.7%p, 전 모델 공통). golden conf=0.001로 통일, board는
  소프트웨어 conf 필터 없음(raw on-device 결과 그대로).
- **GT↔Prediction 매칭 알고리즘**: pycocotools `COCOeval.evaluateImg`(stock 그리디)
  vs ultralytics `BaseValidator.match_predictions`(자체 그리디) — **둘 다 그리디
  매칭이지만 서로 다른 구현**이며, **이 부분이 evaluator 간 손실% 격차(symmetric
  기준 5.3~6.0%p)의 남은 유력 원인으로 추정되나, 라인 단위 직접 대조는 이번 세션까지
  수행되지 않았다**(미확정, §17-8 제안 그대로 남음).

### 21-7. 아직 해결되지 않은 질문 (최대 5개)

1. **실험②(AdamW, lr0=0.001) board 실측 결과** — 진행 중, 최우선 대기.
2. optimizer(SGD/AdamW)가 왜 BN running_var 불안정성을 유발하는지의 메커니즘(momentum/
   adaptive lr과 BN 통계 업데이트의 상호작용) — 전혀 조사 안 됨.
3. pycocotools `evaluateImg` vs ultralytics `match_predictions` 그리디 매칭의 정확한
   알고리즘 차이(합성 극소 예제로 직접 대조) — 여러 차례 제안됐으나 미실행.
4. §18의 pre-softmax logit dynamic range 분해를 실험①/②에도 적용했을 때 동일 패턴이
   재현되는지 — 미실행(기존 NVT dump로 재추론 없이 가능).
5. `model.22.cv2.*`(box 브랜치 conv) 자체의 weight/bias 통계가 dynamic range 확대의
   직접 원인인지(§18-7 제안) — 미실행.

### 21-8. 다음 작업 우선순위

1. **(최우선) 실험② board 실측 결과 수신 시**: 무결성 확인(jaccard overlap) →
   merge → 4-combo(dual/std × pycoco/ultra) 손실 계산 → 절대 mAP 확인(Float 0.41대
   저하 감안, %만으로 판단 금지) → §20-4 표 완성 → optimizer 단독 효과 board 레벨 최종
   확정.
2. 위 §21-7의 4번(pre-softmax dynamic range를 실험①/②에 적용) — 재추론 불필요,
   기존 `debug/golden_tensor`/`sim_layer` dump로 §18 스크립트 재사용 가능. optimizer
   가설의 NVT 메커니즘 레벨 증거를 실험①/②로도 확장.
3. §21-7의 3번(evaluator 매칭 알고리즘 직접 대조) — evaluator 차이의 마지막 미확정
   퍼즐, 다만 위 1,2번보다 우선순위 낮음(현재 조사 핵심 질문인 "왜 symmetric만
   quantization에 강한가"와는 별개 축).

---

```
[RESUME CHECKPOINT]
목표: v01_symmetric만 NVT/Board quantization 손실이 작은 이유 규명 + evaluator(pycoco/ultra) 손실% 격차 원인 분리
현재 결론: optimizer(SGD vs AdamW)가 1순위 원인. lr0/cos_lr/batch/cls 고정한 순수 ablation(실험①SGD/②AdamW)으로 BN 통계/NVT cosine/Float mAP까지 인과 확인. board는 실험①(건강)만 확인, 실험②(AdamW) board 실측 진행 중이라 최종 미확정.
핵심 근거: BN running_var 이상치(57개 중 symmetric1/matched7/v02 6+Inf1/실험①1/실험②10+Inf2), NVT cosine(Sigmoid: sym0.982 vs 실험②0.797=전체 최저), pre-softmax logit dynamic range(std sym3.15 vs matched15.2/v02 16.8), board loss(sym-7.3%~-0.6% vs matched/v02/dair -13~-28%, 실험①-4.6~-7.9%)
배제된 원인: ONNX export, NVT compile 설정(calibration/gen_config 전부 byte-identical), lr0 크기 단독, batch size, NMS 종류(효과 항상 0.7-1.7%p로 작음), "Softmax가 오차를 증폭한다"(B vs C cos=1.00000로 반증), model.0 단일 지점(layer-diff에서 미재현), cls weight 단독(실험①/②둘다 1.5인데 갈림)
미확정: 실험② board mAP, optimizer→BN불안정 메커니즘, pycocotools vs ultralytics 매칭 알고리즘 자체 차이, pre-softmax dynamic range의 실험①/②재현 여부, cv2.* weight 통계 근원 추적
다음 1순위: 실험②(v02_symmetric_adamw_lr0001) board 실측 결과 수신 → 무결성확인→merge→4-combo loss+절대mAP 계산→§20-4표 완성
주요 문서: docs/debug/20260824_symmetric_low_quant_loss_rootcause_compare.md(본 문서, §1-21), docs/debug/20260824_v01_symmetric_eval_gap.md, docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md, docs/debug/20260806_rear_truck_v01_npu_quant_gap.md(정본)
주요 경로: runs/detect/BSD_F606/Rear_Truck/{v01_symmetric,v01_symmetric_matched,v02,v01_dair,v02_symmetric_sgd_lr0001,v02_symmetric_adamw_lr0001}/, runs/_eval_scratch/{configs,board_dumps,merged}/, novatek/convert/rear_truck_<model>_320/, tools/debug/{ultralytics_style_ap.py,inf_conf001.py}
주의사항: 실험②는 Float mAP 자체가 이미 저하(0.41 vs 타모델 0.53-0.56)되어 있어 board 손실 %만으로 판단 금지, 반드시 절대 mAP 함께 확인. 기존 실험 재실행/재컴파일/재학습 금지. board dump는 항상 jaccard overlap으로 stale-check 먼저.
```

---

## 22. 갱신 (2026-08-31) — 실험② Board 실측 완료, §21 체크포인트 대체

**§21의 "미확정: 실험② board mAP"가 해소됐다.** 위 §20-6~20-8에 전체 근거가 있다.
아래는 §21-8/[RESUME CHECKPOINT]를 대체하는 최신 상태 요약이다(§21은 과거 스냅샷으로
그대로 보존, 삭제/수정 없음).

- **실험②(AdamW) board 실측: mAP50-95 0.246(pycoco)/0.2738(ultra), 손실 -36.1~-40.3%**
  — 이 조사 전체에서 최악(§20-6). §21-7의 질문 1번(최우선 대기 항목) 해소.
- optimizer 가설은 이제 **PT(BN 통계)·NVT(simulator cosine)·Float(6152장 mAP)·
  Board(6152장 실측) 4개 레벨 전부**에서 순수 ablation(실험①SGD/②AdamW, lr0/cos_lr/
  batch/cls 고정)으로 일관되게 재현됐다(§20-8). 이 조사에서 나온 결론 중 가장 강하게
  뒷받침된다.
- 남은 미확정 항목은 §21-7의 2~5번(메커니즘 규명, evaluator 매칭 알고리즘 직접 대조,
  §18 분해의 실험①/②적용, cv2.* weight 추적) 그대로 — 전부 선택적 후속 작업이며
  현재 요청되지 않음.

```
[RESUME CHECKPOINT — 2026-08-31 갱신판, §21 대체]
목표: v01_symmetric만 NVT/Board quantization 손실이 작은 이유 규명 + evaluator(pycoco/ultra) 손실% 격차 원인 분리
현재 결론: optimizer(SGD vs AdamW)가 확정적 1순위 원인. lr0/cos_lr/batch/cls 고정한 순수 ablation(실험①SGD/②AdamW)의 PT/NVT/Float/Board 4개 레벨 전부에서 인과 재현 완료(실험① board -4.6~-7.9%, 실험② board -36.1~-40.3%=조사 전체 최악).
핵심 근거: BN running_var 이상치(symmetric1/matched7/v02 6+Inf1/실험①1/실험②10+Inf2), NVT cosine(Sigmoid sym0.982 vs 실험②0.797=최저), Float mAP50-95(실험②만 0.412, 타모델 0.53-0.58), board loss(SGD계열 sym-7.3~-0.6%·실험①-7.9~-4.6% vs AdamW계열 dair-21~-14.6%·v02-19.1~-13.4%·matched-28.4~-25.6%·실험②-40.3~-36.1%)
배제된 원인: ONNX export, NVT compile 설정(byte-identical), lr0 크기 단독(실험①이 반증), batch size, NMS 종류(0.7-1.7%p로 작음), Softmax 증폭(B vs C cos=1.00000), model.0 단일 지점(layer-diff 미재현), cls weight 단독(실험①/②둘다1.5인데 갈림)
미확정(선택적 후속, 미요청): optimizer→BN불안정 메커니즘(momentum/adaptive lr 상호작용), pycocotools vs ultralytics 매칭 알고리즘 라인단위 차이, §18 pre-softmax dynamic range 분해의 실험①/②재현 여부(재추론 불필요, 기존 dump로 가능), model.22.cv2.* weight 통계 근원 추적
다음 1순위: 없음(요청된 실측/실험 전부 완료). 사용자의 다음 지시 대기.
주요 문서: docs/debug/20260824_symmetric_low_quant_loss_rootcause_compare.md(본 문서, §1-22), docs/debug/20260824_v01_symmetric_eval_gap.md, docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md, docs/debug/20260806_rear_truck_v01_npu_quant_gap.md(정본)
주요 경로: runs/detect/BSD_F606/Rear_Truck/{v01_symmetric,v01_symmetric_matched,v02,v01_dair,v02_symmetric_sgd_lr0001,v02_symmetric_adamw_lr0001}/, runs/_eval_scratch/{configs,board_dumps,merged}/, novatek/convert/rear_truck_<model>_320/, tools/debug/{ultralytics_style_ap.py,inf_conf001.py}
주의사항: 실험②는 Float mAP 자체가 이미 저하(0.41 vs 타모델 0.53-0.58)되어 있어 board 손실 %만으로 판단 금지, 절대 mAP(0.246)를 함께 본다. 기존 실험 재실행/재컴파일/재학습 금지. board dump는 항상 jaccard overlap으로 stale-check 먼저(§20-6에서 실험②도 8개 기존 결과 대비 전부 0.0000 확인됨).
```

---

## 23. 여섯 모델 종합 비교 — mAP50 기준, std+ultra 강조 (2026-08-31 추가)

**사용자 지시로 이후 요약은 mAP50-95 대신 mAP50을 우선하고, 4-combo 중 std+ultra(표준
단일-IoU NMS `iou_thres=0.45` + ultralytics `ap_per_class` evaluator)를 강조한다.**
mAP50-95 기준 표(§10, §20-7)와 같은 재추론 없이, 기존 `coco_prediction.json`/board
병합 json에 `tools/debug/ultralytics_style_ap.py` 및 `projects/metrics/evaluator.py`를
다시 적용해 mAP50 값만 새로 산출했다(dual+pycoco/dual+ultra/std+pycoco는
`docs/debug/20260824_v01_symmetric_eval_gap.md` §10-1에 이미 있던 값을 그대로 인용,
실험①/②만 이번에 신규 계산).

| Model | Optimizer | dual+pycoco | dual+ultra | std+pycoco | **std+ultra** |
|---|---|---:|---:|---:|---:|
| v01_symmetric | SGD | -3.47% | +0.26% | -3.72% | **+0.41%** |
| **실험① sgd_lr0001** | **SGD** | -3.18% | -1.80% | -3.05% | **-1.27%** |
| v01_dair | AdamW | -9.72% | -6.03% | -8.48% | **-4.66%** |
| v02 | AdamW | -10.08% | -6.80% | -9.60% | **-6.24%** |
| v01_symmetric_matched | AdamW | -11.35% | -10.06% | -10.64% | **-9.29%** |
| **실험② adamw_lr0001** | **AdamW** | -20.41% | -17.79% | -18.21% | **-15.52%** |

실험①의 원자료(재추론 없이 기존 예측 재사용): Float(std,pycoco) mAP50=0.753,
Float(std,ultra) mAP50=0.7702, Float(dual,pycoco) mAP50=0.754, Float(dual,ultra)
mAP50=0.7743, Board(pycoco) mAP50=0.730, Board(ultra) mAP50=0.7604.

**std+ultra 기준 결론**: SGD 계열(v01_symmetric +0.41%, 실험① -1.27%)은 손실이
거의 없거나 역전, AdamW 계열은 -4.66% → -6.24% → -9.29% → -15.52%로 단조 증가 —
mAP50-95(§20-7)와 동일한 optimizer 분리 패턴이 mAP50에서도 재현된다. 다만 mAP50
기준에서는 전반적으로 손실 %의 절댓값이 mAP50-95보다 작다(예: 실험② std+ultra
mAP50-95 -36.07% vs mAP50 -15.52%) — IoU 기준이 느슨할수록(50만 봄) box 위치
오차의 영향이 줄어 손실이 작게 보이는, 지표 자체의 특성이지 새로운 발견은 아니다.

---

## 24. `v02_dair` — trainer(symmetric vs dair) ablation, PT/NVT/Float (2026-09-02)

**목적**: §19~20의 optimizer ablation(실험①/②)은 둘 다 `symmetric` 트레이너였다.
`v01_dair`/`v02`는 항상 AdamW로만 관측돼서, 그 불안정성이 "optimizer 때문"인지
"dair 트레이너 고유의 dual-forward consistency loss/증강 방식 때문"인지 분리되지
않았다. 다른 학습 PC에서 이 빈칸을 채우는 세 번째 독립 실험을 완료해 결과를
가져왔다: `v02_dair` = v01_symmetric의 optimizer(SGD)/lr0(0.01)/lrf(0.01)/
cos_lr(true)/batch(48)/cls(1.5)를 그대로 쓰되 **trainer만 symmetric→dair로 교체**
(`dair_params`/TensorSpeakerOverlayAug 증강은 v01_dair 그대로 유지 — 트레이너와
증강 두 축이 같이 바뀌는 것은 의도적 선택, exp.yaml 헤더 코멘트로 확인).

### 24-1. PT 레벨 — BN `running_var` 진단(§3-1/§19-2와 동일 방식)

| Model | Trainer | Optimizer | BN 이상치(>500) | Inf 채널 |
|---|---|---|---:|---:|
| v01_symmetric | Symmetric | SGD | 1 | 0 |
| 실험① sgd_lr0001 | Symmetric | SGD | 1 | 0 |
| **v02_dair** | **DAIR** | **SGD** | **2** | **0** |
| **v01_dair(2026-09-02 추가)** | **DAIR** | **AdamW** | **5** | **1** |
| v02 | DAIR | AdamW | 6 | 1 |
| v01_symmetric_matched | Symmetric | AdamW | 7 | 0 |
| 실험② adamw_lr0001 | Symmetric | AdamW | 10 | 2 |

**`v01_dair`도 AdamW 그룹(5~10개 이상치, Inf 0~2개)에 정확히 속한다** — §24-1의 공백이
채워지면서 SGD/AdamW 그룹 분리가 이제 7개 모델 전부에서 예외 없이 확인됐다.
`model.0.bn.running_var`의 Inf 위치도 `v02`와 **정확히 같은 채널 인덱스(4)**다
(`v01_dair` 값: `[2790, 3208, 10256, 5532, inf, 6868, 5756, 3374, 30656, 5260,
11816, 5408, 8156, 5960, 7944, 10048]`) — 같은 계보(dair trainer, AdamW)의 두
모델이 우연이라기엔 지나치게 정확히 같은 지점에서 발산한다는 점은, 이 발산이
"학습 중 무작위로 아무 채널에나 생기는 것"이 아니라 **이 아키텍처/데이터 조합에서
재현 가능한 특정 실패 지점**일 가능성을 시사한다(다만 원인 메커니즘 자체는 여전히
미확정, §21-7/§24-4 참조).

### 24-2. NVT 레벨 — 3-tensor simulator cosine(동일 테스트 이미지, 동일 compile 조건)

ONNX export(`skip_postprocess=True`, `(1,5,2100)+(1,4,2100)`, 다른 모든 모델과 동일 구조 —
shape로 확인) 후 NVT 3-tensor baseline과 완전히 동일한 조건(gen_config.txt/sim_config.txt
model_dir 외 byte-identical, calibration 940장 동일 symlink)으로 compile+simulate.
`nvt_model.bin`=4,711,336 bytes, latency=6.36765ms — 다른 모든 3-tensor baseline과 동일.

| Model | Trainer | Optimizer | Softmax(dfl) | Sigmoid[OUT] | dfl_conv[OUT] |
|---|---|---|---:|---:|---:|
| 실험① sgd_lr0001 | Symmetric | SGD | 0.9955 | 0.9779 | 0.9977 |
| **v02_dair** | **DAIR** | **SGD** | **0.9857** | **0.9876** | **0.9945** |
| v01_symmetric | Symmetric | SGD | 0.9850 | 0.9824 | 0.9933 |
| v01_symmetric_matched | Symmetric | AdamW | 0.9126 | 0.9473 | 0.9490 |
| v02 | DAIR | AdamW | 0.9058 | 0.9236 | 0.9664 |
| 실험② adamw_lr0001 | Symmetric | AdamW | 0.9170 | 0.7969 | 0.9650 |

### 24-3. Float ONNX mAP50-95 (6,152장)

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---:|---:|---:|---:|
| v01_symmetric(baseline) | 0.562 | — | 0.558 | 0.5785 |
| 실험① sgd_lr0001 | 0.532 | 0.5551 | 0.528 | 0.5474 |
| **v02_dair** | **0.530** | **0.5506** | **0.524** | **0.5414** |
| v01_dair | 0.533 | 0.5521 | 0.524 | 0.5411 |
| v02 | 0.538 | 0.5581 | 0.531 | 0.5493 |
| v01_symmetric_matched | 0.549 | 0.5699 | 0.541 | 0.5593 |
| 실험② adamw_lr0001 | 0.412 | 0.4399 | 0.403 | 0.4283 |

**`v02_dair`의 Float mAP50-95는 다른 SGD/AdamW 모델들과 같은 수준(0.52~0.55대)** —
실험②처럼 Float 자체가 저하돼 있지 않다. §20-2에서 확인한 "AdamW+lr0=0.001+cos_lr+
batch48+cls1.5" 특정 조합의 float-level 손상은 이 조합(SGD+dair trainer)에서는
재현되지 않았다.

### 24-4. Board 실측 결과 (2026-09-02)

사용 파일: 원본 덤프 `runs/_eval_scratch/board_dumps/v02_dair/v02_dair_output.json`
(사용자가 실제 보드에서 직접 추론, md5 `8f1e2934...`) → `merge_board_dump.py` →
`runs/_eval_scratch/merged/v02_dair/ipu_pred_merged_v02_dair_3tensor.json` →
`board_v02_dair_3tensor_cfg.yaml`(신규 생성, 다른 board config와 동일 구조).

무결성 확인: 6,152장/209,917 detection 정확히 매칭, md5가 기존 8개 병합/원본 dump와
전부 다름, 기존 8개 병합 결과(production/v01_940calib/v01_symmetric/v01_dair/v02/
v01_symmetric_matched/실험①/실험②) 대비 detection 단위 jaccard overlap **전부
0.0000**(독립적 유효 덤프).

| | mAP50 | mAP50-95 |
|---|---:|---:|
| Float(dual, pycoco) | 0.749 | 0.530 |
| **Board(pycoco)** | 0.720 | **0.504** |
| Float(dual, ultra) | 0.7670 | 0.5506 |
| **Board(ultra)** | 0.7520 | **0.5376** |

| NMS \ Evaluator | pycocotools | ultralytics |
|---|---:|---:|
| dual-threshold (mAP50-95) | -4.91% | -2.36% |
| 표준 단일-IoU (mAP50-95) | -3.82% | **-0.70%** |
| dual-threshold (mAP50) | -3.87% | -1.96% |
| 표준 단일-IoU (mAP50) | -3.36% | -1.22% |

### 24-5. 종합 비교 (std+ultra, mAP50-95 강조)

| Model | Trainer | Optimizer | std+ultra |
|---|---|---|---:|
| v01_symmetric | Symmetric | SGD | -0.62% |
| **v02_dair** | **DAIR** | **SGD** | **-0.70%** |
| 실험① sgd_lr0001 | Symmetric | SGD | -4.59% |
| v01_dair | DAIR | AdamW | -14.62% |
| v02 | DAIR | AdamW | -13.44% |
| v01_symmetric_matched | Symmetric | AdamW | -25.62% |
| 실험② adamw_lr0001 | Symmetric | AdamW | -36.07% |

### 24-6. 판단 (최종, board 실측 완결)

PT(BN 통계)·NVT(simulator cosine)·Float(6,152장 mAP)·**Board(6,152장 실측)** 4개
레벨 **전부에서 `v02_dair`는 SGD 그룹(v01_symmetric/실험①)에 속하고 AdamW 그룹
(matched/v02/실험②)과 뚜렷이 구분된다.** trainer를 dair로 바꾸고 dual-forward
consistency loss + 증강을 그대로 켠 채로도, optimizer만 SGD면 이 네 레벨 모두
안정적이다 — **"dair 트레이너 자체가 불안정성의 원인"이라는 가설을 약화시키고,
"optimizer(SGD vs AdamW)가 지배적 변수"라는 이 조사의 핵심 결론을 trainer 축에서
board 레벨까지 완결적으로 재확인한다.**

**§24-4에서 우려했던 "PT/NVT는 건강한데 board는 나쁠 수 있다"는 실험①식 괴리는
`v02_dair`에서는 재현되지 않았다** — `v02_dair`의 board 손실(std+ultra -0.70%)은
v01_symmetric(-0.62%)과 사실상 동급이고, 실험①(-4.59%)보다도 뚜렷이 좋다. 즉 PT/NVT
신호가 board 결과를 항상 보장하지는 않는다는 원칙(§24-4 구판, 실험①로 확인)은 여전히
유효하지만, 이번 `v02_dair` 사례처럼 PT/NVT/Float/Board 네 레벨이 전부 일치하는
경우도 있다 — **레벨 간 불일치 가능성이 있다고 해서 board 없이 예측할 수 없다는
뜻이지, PT/NVT 신호가 항상 board와 어긋난다는 뜻은 아니다.**

(§24-1의 `v01_dair` BN 통계 공백은 2026-09-02에 채워졌다 — 위 표 참조.)
