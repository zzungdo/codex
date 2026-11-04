# Front_Truck Reference ONNX → NVT Board Preparation

> 이 문서는 NAS 기존 BIN 탐색 기록이며, 실제 local reconversion 실험은
> [`20260814_front_truck_reference_onnx_local_nvt_reconversion.md`](20260814_front_truck_reference_onnx_local_nvt_reconversion.md)에서
> 수행한다.

## 목적

NAS에 저장된 실제 Front_Truck v01 배포 ONNX를 그대로 NVT 변환했을 때 기존 Board 성능을 재현할 수
있는지 확인하기 위해, Board 테스트용 NVT 모델(`nvt_model.bin`)을 준비한다. `.pt → ONNX export`는
이번 작업과 무관하며, ONNX 구조/PT source 재조사는 하지 않는다.

## Source ONNX

```
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx
```

읽기 전용으로만 취급, 수정/삭제 없음.

- SHA256: `a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0`
- Size: 12,071,501 bytes
- mtime: 2026-07-07 10:27:08
- input: `images [1,3,512,640]`, output: `output0 [1,6,6720]` + `433 [1,4,6720]`, opset 13
  (기존 조사에서 확인된 구조, 재검증하지 않음)

## Existing NVT artifact search

`Front_Truck/v01/` 하위 전체 확인 결과, `network/` 디렉토리에 ONNX와 함께 다음 파일이 존재:

```
network/260706_BSD-F606A_Front_v01.onnx
network/260706_BSD-F606A_Front_v01.bin     <- 기존 NVT bin 후보
network/gen_config.txt
network/sim_config.txt
```

### FOUND — 판단 근거

| 근거 | 확인 내용 |
|---|---|
| 동일 폴더 | `.onnx`/`.bin` 모두 `Front_Truck/v01/network/`에 함께 존재 |
| 동일 파일명 basename | 둘 다 `260706_BSD-F606A_Front_v01.*` — 확장자만 다름 |
| 생성 시각 | onnx 2026-07-07 10:27:08, bin 2026-07-07 10:27:06 — **2초 차이**, 같은 파이프라인 실행분 |
| gen_config의 input size | `preproc/resize/width=640, height=512` — ONNX의 `images[1,3,512,640]`와 **정확히 일치** |
| gen_config의 model_dir | `.../bsd_f606a_front/v2/bsd_f606a_front_640x512/530/...` — Front, 640x512, chip 530과 일치 |

→ **Existing Front NVT model: FOUND**

```
Existing nvt_model.bin (NAS):
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.bin

Size: 4,758,800 bytes
SHA256: fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b
MD5:    619b83d9bb519be69f1b9f3b37ee0a4c
```

이 근거들로 이 `.bin`이 해당 ONNX에서 만들어졌다고 판단하기에 충분하다고 결론짓는다. **신규
compile을 진행하지 않는다.**

## gen_config / calibration

`gen_config.txt`/`sim_config.txt` 모두 NAS `network/` 원본을 그대로 읽었을 뿐 복사/수정하지 않음
(신규 compile을 하지 않으므로 격리 디렉토리도 만들지 않았다).

- calibration 경로: `path/ref_img_dir = /home/ncn/Developments/data/weights/_pipeline_inputs/nt98530_d224k_front/calib/jpg`,
  `ref_data/num = 500` — 이 경로는 이 host에 **존재하지 않는다** (원본 컴파일 머신의 로컬 경로).
  다만 기존 BIN을 그대로 사용하기로 했으므로 calibration 재확보는 필요하지 않다.

## NVT compile

**실행하지 않음.** NAS에서 provenance가 충분히 확인된 기존 `nvt_model.bin`을 그대로 사용하기 때문.

## Simulator

**실행하지 않음.** 위와 동일한 이유.

## Board test model

```
========================================
BOARD TEST MODEL
========================================

nvt_model.bin:
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.bin

Source ONNX:
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx

Source ONNX SHA256:
a4e15dea408937774dd30646cf55e62371018a1fa69574b961f7a07685eeb1d0

NVT BIN SHA256:
fa795cb4efe782a55250c793b0dc12c6db0d894ba9dea7c9b972705bb4fca00b

Input:
1x3x512x640

Status:
READY FOR BOARD TEST
========================================
```

NAS 파일은 그대로 두었으며 복사/수정하지 않았다.

## 다음 단계

준비된 `nvt_model.bin`(NAS 원본 경로)을 실제 Board에 적용하고, 과거 Front_Truck Board 평가와
동일한 validation dataset / preprocessing / postprocess / threshold / evaluator로 결과를 생성한다.

그 후:

```
기존 측정 Board 결과 (analysis/metrics_front_board.txt, mAP 0.704)
vs
이번 Board 결과
```

를 class별 및 all 기준으로 비교한다. Board 실행 자체는 이번 작업에서 수행하지 않았다.
