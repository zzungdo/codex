# Corner_Truck v01 NVT Calibration / Compile / Simulator Validation

## 목적

Corner_Truck v01의 Float ONNX를 기준으로 calibration data와 NAS `gen_config.txt`를 사용해 NVT
compile 및 simulator를 수행한 결과를 기록하고, 이후 실제 Board 검증을 위한 baseline을 남긴다.

## 입력 모델

### Float ONNX

```
runs/debug/corner_truck_reference_compare/exported_from_best.onnx
```

이 ONNX는 이전 독립 조사에서 NAS reference ONNX와 사실상 동일(bit-exact 동급)하다고 검증된
파일이다 — 재-export하지 않고 그대로 재사용했다.

확인된 구조:

```
input:
images [1,3,320,320]

outputs:
output0 [1,5,2100]
433     [1,4,2100]

opset 13
node 235
```

### gen_config

NAS 원본을 복사해서 사용:

```
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Corner_Truck/v01/network/gen_config.txt
```

수정한 항목은 새 격리 디렉토리에 맞춘 path 관련 값(`path/model_dir`, `path/ref_img_dir`,
`path/ref_list_path`)뿐이며, 그 외 모든 precision/preprocessing 값은 NAS 원본 그대로다.

Corner 원본 config의 주요 설정:

```
ref_data/num                                      = 300
precision/high_precison/spec_demarc_tensors        = (주석 처리, 미사용)
precision/cal_refine_iter                          = 0
precision/cle_en                                   = 0
precision/in_hp/bnscale_en                         = 0
precision/in_hp/eltwise_en                         = 0
precision/in_hp/conv_op                            = 0
precision/percentile                               = (설정 없음)
```

Corner NAS gen_config에서는 위 precision 보조 옵션들이 비활성화되어 있다.

sim_config는 NAS에 원본이 없어 320x320 입력에 맞춰 신규 작성했다
(`src/width=320, height=320, channel=3`).

## Calibration dataset

- Source: `/mnt/NAS_1/jpg` (원본, 읽기 전용 — 삭제/이동 없음)
- 방식: `rsync -a --include="*.jpg" --exclude="*"`로 로컬 격리 디렉토리에 복사
- 결과: **300장**, jpg 확장자만 (원본에 `.db` 파일 1개가 섞여 있었으나 제외), **약 128MB**
- gen_config의 `ref_data/num=300`과 실제 이미지 300장이 정확히 일치했다

## NVT compile

### 실행 조건

- Container: `npu-convert` (`dair-yolo/npu:latest`)
- 명령: `./run_compiler.sh corner_truck_v01_320_quant_compare_20260814` (`--chip 530`)
- Input ONNX sha256: `bfe92adbac0c4ff952db7a74bb28a6ecf7f229443335fd2e3b36d85b338fb6ac`
- calibration image count: 300

### 결과

| 항목 | 값 |
|---|---|
| Latency | 6.24694 ms |
| DRAM | 35,045,388 bytes (약 33.4 MB) |
| Model size | 4,703,564 bytes (약 4.70 MB) |

### nvt_model.bin

```
novatek/convert/corner_truck_v01_320_quant_compare_20260814/nvtai_tool/output/CORNER_TRUCK_V01_320_QUANT_COMPARE_20260814/sdk/nvt_model.bin
```

SHA256: `80778a2cbb66f9f372e8e6618b3a33b14ce0448912ca68fdc37160556b2b8769`

NAS 기존 `nvt_model.bin`과 신규 생성본은 크기가 4 bytes 다르고(NAS 4,703,568 bytes vs 신규
4,703,564 bytes) SHA256도 다르다. 차이는 매우 작지만 정확한 원인은 binary-level 추가 분석 없이
확정하지 않는다.

## Simulator

Container: `npu-convert`, 명령: `./run_simulator.sh corner_truck_v01_320_quant_compare_20260814
<test_image>`. 테스트 이미지: `KR_20240621_TRUCK1T_CORNER_140CM_RIGHT_205FISH_CROP_NC22015_0992.jpg`
(Corner 실제 검증셋 샘플).

### 주요 output tensor

| Tensor | Cosine |
|---|---|
| Softmax(dfl) | 0.987600 |
| Sigmoid[OUT] | 0.996472 |
| dfl_conv[OUT] | 0.994926 |

### 전체 tensor 통계

CSV: `runs/debug/corner_truck_v01_320_quant_compare_20260814_tensor_cosine.csv` (171개 tensor 전체)

```
전체 tensor: 171
mean cosine: 0.9837
min cosine:  0.9604

cosine < 0.999: 163
cosine < 0.995: 147
cosine < 0.990: 132
```

### Lowest cosine Top 10

```
0.9604  Mul__model.19_act_Mul_output_0_Y
0.9621  Mul__model.21_m.0_cv2_act_Mul_output_0_Y
0.9624  Mul__model.18_m.0_cv2_act_Mul_output_0_Y
0.9640  Conv__model.21_m.0_cv2_conv_Conv_output_0_Y
0.9648  Conv__model.19_conv_Conv_output_0_Y
0.9651  Mul__model.22_cv3.1_cv3.1.0_act_Mul_output_0_Y
0.9690  Conv__model.22_cv3.1_cv3.1.1_conv_Conv_output_0_Y
0.9691  Split__model.8_Split_output_0_Y
0.9703  Mul__model.18_m.0_cv1_act_Mul_output_0_Y
0.9711  Conv__model.21_m.0_cv1_conv_Conv_output_0_Y
```

### 해석

Corner_Truck의 주요 detection output 관련 tensor는 약 0.988~0.996 cosine을 보였다. 전체 tensor
평균 cosine은 약 0.984이며 일부 neck/head tensor는 약 0.960까지 낮아진다. 따라서 simulator 상
quantization으로 인한 numerical difference는 존재한다.

단, cosine 값만으로 실제 Board mAP 손실을 직접 판단할 수는 없다. Board 성능은 실제 Board
validation으로 확인해야 한다.

## 기존 Corner Board 결과

### 과거 baseline

**과거 Corner_Truck 기존 Board 결과** (이번에 새로 생성한 `nvt_model.bin`을 보드에서 측정한 결과가
아니다):

```
Class         ONNX     NVT BIN    Delta(NVT-ONNX)
bus           0.284    0.416      +0.132
car           0.393    0.414      +0.021
motorcycle    0.549    0.549       0.000
person        0.571    0.523      -0.048
truck         0.504    0.504       0.000
all           0.460    0.481      +0.021
```

Corner는 전체(all) mAP 기준으로는 quantization 손실이 없지만(오히려 +0.021), class별로는 편차가
존재한다(bus +0.132, person -0.048).

## 현재 상태

이번에 새로 생성한 `nvt_model.bin`에 대해서는 아직 Board 측정을 수행하지 않았다.

## 다음 단계 — 신규 NVT BIN Board 검증

이번에 새로 생성한 Corner_Truck `nvt_model.bin`을 실제 NT98529/NT98530 Board에 배포하여, 기존
Float ONNX와 동일한 validation dataset / preprocessing / threshold / postprocess 조건으로 Board
inference 결과를 생성한다.

그 후:

```
Float ONNX mAP
vs
새 NVT BIN Board mAP
```

를 class별/all 기준으로 비교한다.

향후 비교가 공정하려면 다음 조건이 동일해야 한다:

```
Validation dataset
image preprocessing
crop / resize
BGR/RGB
confidence threshold
NMS IoU
class mapping
postprocess
evaluator
```

가능하면 과거 Corner ONNX/NVT Board 성능을 만들 때 사용했던 동일 evaluation pipeline을 사용한다.

## 재현 명령

```bash
# 1. 격리 디렉토리 (이미 생성됨)
novatek/convert/corner_truck_v01_320_quant_compare_20260814/

# 2. compile
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_compiler.sh corner_truck_v01_320_quant_compare_20260814'

# 3. simulator
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_simulator.sh corner_truck_v01_320_quant_compare_20260814 \
   KR_20240621_TRUCK1T_CORNER_140CM_RIGHT_205FISH_CROP_NC22015_0992.jpg'
```

## 최종 상태 요약

```
Corner_Truck v01

Float ONNX:
정상

Calibration:
300 images

NVT Compile:
PASS

Latency:
6.24694 ms

DRAM:
33.4 MB

Model size:
4.70 MB

Simulator:
PASS

Softmax(dfl):
0.987600

Sigmoid:
0.996472

dfl_conv:
0.994926

New nvt_model.bin:
생성 완료

New NVT BIN Board validation:
NOT RUN

기존 Board baseline:
ONNX 0.460
NVT BIN 0.481
Delta +0.021
```
