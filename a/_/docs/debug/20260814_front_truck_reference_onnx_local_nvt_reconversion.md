# Front_Truck Reference ONNX → Local NVT Reconversion

## 목적

동일한 NAS 배포 ONNX(`260706_BSD-F606A_Front_v01.onnx`)를 현재 로컬 NVT converting pipeline
(`npu-convert` 컨테이너, `run_compiler.sh`, `--chip 530`)으로 다시 변환했을 때, 과거 Release NVT
모델과 동일한 수준의 Board 결과가 나오는지 확인한다. 이를 통해 현재 사용 중인 ONNX → NVT
converting 파이프라인 자체에 문제가 있는지 검증한다. `.pt → ONNX export`, source checkpoint 재탐색,
ONNX 재-export는 이번 작업 범위가 아니다.

기존 NAS Release BIN은 Board 테스트 대상이 아니라 **비교 기준(reference artifact)**이다. 이번에
새로 만든 BIN이 Board 테스트 모델이다.

## Source ONNX

```
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx
```

NAS 원본은 읽기 전용, 로컬 격리 디렉토리로 복사 후 사용. 복사본 SHA256이 원본과 동일함을 확인:

```
SHA256: a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0  (원본/복사본 동일)
input: images [1,3,512,640]
outputs: output0 [1,6,6720], 433 [1,4,6720]
opset: 13
```

## NAS Release BIN

```
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.bin
```

```
Size:   4,758,800 bytes
SHA256: fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b
```

기존 Board Release 결과 (`analysis/metrics_front_board.txt`, `[front_board]` 섹션):

```
class              AP      P      R     F1
bus             0.696  0.828  0.676  0.744
car              0.78  0.839  0.782   0.81
motorcycle      0.725  0.793  0.701  0.744
person          0.742  0.806  0.689  0.743
truck           0.687  0.816  0.682  0.743
traffic_light   0.592  0.797  0.559  0.657
mAP/mF1         0.704                 0.74
```

이 값은 `NAS Release BIN → Board` 결과다.

## Release gen_config

NAS 원본 `network/gen_config.txt`, `network/sim_config.txt`를 로컬 격리 디렉토리로 복사, path 관련
항목만 신규 경로로 수정하고 그 외 precision/preprocessing/quantization 값은 전혀 변경하지 않았다
(`diff`로 path 항목 외 변경 없음 확인).

주요 설정 (Release 원본 그대로):

```
input resize:  width=640, height=512
chip:          530
ref_data/num:  500
calib_mode:    4
cle_en:        0
cal_refine_iter: 0
spec_demarc_tensors: 주석 처리(미사용)
in_hp bnscale/eltwise/conv_op: 0/0/0
sim_config src/bintype: 1 (BLOB_TYPE_UINT8)
```

## Calibration set

Release gen_config가 가리키는 원래 경로:

```
/home/ncn/Developments/data/weights/_pipeline_inputs/nt98530_d224k_front/calib/jpg
```

는 원본 컴파일 머신의 로컬 경로로, 현재 host/NAS/repo 어디에도 존재하지 않는다. Front_Truck NAS
폴더, NAS_1/NAS_3 관련 위치, 로컬 repo, 컨테이너 `/dataset` 마운트를 확인했으나 이 정확한
500장 세트(또는 `d224k_front`라는 이름의 흔적)를 찾지 못했다.

```
EXACT RELEASE CALIBRATION SET: NOT FOUND
```

**SUBSTITUTE CALIBRATION SET** 사용:

```
/dataset/Dashcam/Front_traffic_light/Validation/images  (컨테이너 마운트)
```

Front_Truck v01의 클래스 목록(bus/car/motorcycle/person/truck/traffic_light)과 정확히 일치하는
Front 카메라 도메인 데이터셋이다. 전체 6,076장 중 파일명 정렬 후 앞 500장을 `rsync`/`cp`로 로컬
격리 디렉토리에 복사(`gen_config`의 `ref_data/num=500`과 개수 일치). NAS/컨테이너 원본은 이동/삭제
하지 않았다.

**제약사항**: ONNX와 gen_config는 Release와 동일하지만 calibration dataset이 Release와 동일하지
않으므로, 아래에서 확인되는 신규 BIN과 Release BIN의 차이가 converting code 때문인지 calibration
data 차이 때문인지 완전히 분리할 수 없다.

## NVT compile

### 실행 조건

- Container: `npu-convert` (`dair-yolo/npu:latest`), `--chip 530` — Rear/Corner와 동일 pipeline
- 격리 디렉토리: `novatek/convert/front_truck_v01_reference_onnx_local_reconvert_20260814/`
- 명령: `./run_compiler.sh front_truck_v01_reference_onnx_local_reconvert_20260814`
- Source ONNX SHA256: `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0`
- calibration image count: 500 (`Ref Data Num = 500` in `profile.txt`로 확인)

### 결과

| 항목 | 값 |
|---|---|
| Latency | 19.628 ms |
| DRAM (Total Bw) | 103,244,064 bytes |
| Model size | 4,758,792 bytes |

Warning/Error 없이 compile 완료 (`=== Done ===`, exit code 0).

### 신규 nvt_model.bin

```
novatek/convert/front_truck_v01_reference_onnx_local_reconvert_20260814/nvtai_tool/output/FRONT_TRUCK_V01_REFERENCE_ONNX_LOCAL_RECONVERT_20260814/sdk/nvt_model.bin

Size:   4,758,792 bytes
MD5:    a98945c548d2666574c25ab3d91d266b
SHA256: d5727e652931270f0771f87ba2372f978736c0c9e3f66f31ac33efe3f4e26028
```

## Release BIN vs Local BIN

| 항목 | NAS Release BIN | Local Reconverted BIN | 차이 |
|---|---|---|---|
| Size | 4,758,800 bytes | 4,758,792 bytes | 8 bytes |
| MD5 | `619b83d9bb519be69f1b9f3b37ee0a4c` | `a98945c548d2666574c25ab3d91d266b` | 다름 |
| SHA256 | `fa795cb4...ca00b` | `d5727e65...26028` | 다름 |
| `cmp` byte diff count | — | — | 3,136 bytes 차이 (공통 4,758,792 bytes 구간 기준) |
| 첫 diff 위치 (`cmp`) | — | — | byte 26 |

SHA256이 다르다고 해서 conversion 실패로 판단하지 않는다. NVT binary는 compiler
metadata/calibration 통계 등을 포함할 수 있고, 특히 이번에는 calibration dataset 자체가
Release와 다르므로 이 차이는 참고값이다. Size는 8 bytes(약 0.0002%) 차이로 사실상 동일 규모다.

## Simulator

신규 Local BIN에 대해서만 sanity check 실행 (기존 NAS Release BIN은 compile 중간 산출물이 없어
동일 simulator pipeline으로 직접 재실행 불가능하므로 무리하게 재구성하지 않음).

```
./run_simulator.sh front_truck_v01_reference_onnx_local_reconvert_20260814 <test_image>
```

- Simulator 정상 종료 (exit code 0, `=== Simulator Done ===`)
- 171개 tensor diff 로그 생성, NaN/Inf 없음
- 주요 output tensor cosine:

| Tensor | Cosine |
|---|---|
| Softmax(dfl) | 0.981978 |
| Sigmoid[OUT] | 0.976482 |
| dfl_conv[OUT] | 0.992463 |

과도한 layer-level 분석은 하지 않음 — 이번 목적은 Board 투입 전 sanity check다.

## BOARD TEST MODEL

```
========================================
LOCAL RECONVERTED BOARD TEST MODEL
========================================

Source ONNX:
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx

Local nvt_model.bin:
/home/ncn/Workspace/jdh/ultralytics_projects/novatek/convert/front_truck_v01_reference_onnx_local_reconvert_20260814/nvtai_tool/output/FRONT_TRUCK_V01_REFERENCE_ONNX_LOCAL_RECONVERT_20260814/sdk/nvt_model.bin

Source ONNX SHA256:
a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0

Local NVT BIN SHA256:
d5727e652931270f0771f87ba2372f978736c0c9e3f66f31ac33efe3f4e26028

NAS Release BIN (reference, NOT for board test):
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.bin

NAS Release BIN SHA256:
fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b

Status:
READY FOR BOARD TEST
========================================
```

## Board 측정 후 판정 기준

Board 실행은 이번 작업에서 수행하지 않는다. 이후 위 Local Reconverted BIN을 Board에 올려 기존
Release와 동일한 validation dataset / preprocessing / ROI crop / resize / RGB-BGR / confidence /
NMS IoU / class mapping / postprocess / evaluator 조건으로 mAP를 측정한 뒤 다음과 같이 판정한다.

**Case A — 거의 동일** (예: Release 0.704, Local reconvert 0.70~0.71 근처, class별로도 유사):
→ 현재 ONNX → NVT converting pipeline은 과거 Release 결과를 재현할 수 있으며, converting 코드
자체에 큰 문제가 있을 가능성은 낮다.

**Case B — 크게 다름** (예: Release 0.704, Local reconvert 0.63 등):
→ 동일 source ONNX에서 새 NVT 결과가 다르므로 현재 pipeline과 Release 당시 pipeline 사이에 의미
있는 차이가 있다. 이 경우 calibration dataset 차이(위 제약사항)를 먼저 분리해서 고려해야 하며,
그 다음 gen_config/compiler version/conversion wrapper/precision options/preprocessing 순으로
조사한다.
