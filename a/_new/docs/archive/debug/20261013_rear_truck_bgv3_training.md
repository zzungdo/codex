# Rear_Truck 배경(background_rear_v3) 학습 — 인수인계 (2026-10-13 KST)

적용: `BSD_F606/Rear_Truck` 릴리스 `v01_symmetric` 의 오검출 개선 실험 · 기준: 2026-10-13 10:00 · 요약 갱신: 2026-10-16

## 한눈에 보기 (2026-10-16)

### 결론

- **후보는 E1 `v01_symmetric_ft_bgv3` 다.** Board 0.763 으로 가장 높고(v01 0.751), FP 케이스 영상 검출 프레임이 −14% 다. 학습은 9시간이다.
- **배경 v3 는 오검출을 줄이고 mAP 는 깎지 않는다.**
  - FP 영상에서는 E1·E2b 만 검출이 줄었다(−14%, −15%). 배경이 없는 E0 는 v01 과 비슷했다.
  - 6,152장 평가셋은 객체 없는 이미지가 0장이라 배경 효과가 mAP 에 드러나지 않는다(E1 = E0).
- **파인튜닝 이득(+0.007~0.012)은 배경이 아니라 lr 을 한 번 더 낮춰 다시 수렴시킨 효과다.** 배경 없는 E0 도 똑같이 올랐다.
- **스크래치 + 배경이 낮았던 원인은 조기종료였다.** 같은 레시피를 patience 100 으로 끝까지 돌린 E2b 는 v01 보다 높다(E2 0.760 → E2b 0.785, ONNX A).
- **low_cam(70cm·140° 카메라)은 쓰지 않는다.** E3 는 E1 보다 낮다(0.780 vs 0.787).
- **보드 손실은 두 부분으로 나뉜다** (E1 기준, A 0.787 → Board 0.763).
  - 보드 NMS 규칙(차량 묶음, IoM 0.6): −0.015. truck 은 −0.05 다.
    - 그 대부분은 **앞 트럭에 가려 일부만 보이는 다른 트럭**이 IoM 0.6 규칙에 지워진 것이다. 라벨은 정상이다(사용자 확인).
    - 6,152장 중 9% 가 이런 정차 클립 하나다. 이 144쌍을 빼고 재면 A→B 는 −0.005(truck −0.003)다.
  - 양자화: −0.009 다.
- **변환 설정을 바꿔도 보드 점수가 오르지 않았다.**
  - C1(percentile 0.9999)은 0.761 이다.
  - C3(16비트 보호 추가)는 SDK 가 설정을 무시했다.

### 모델별 결과 (6,152장, customEval mAP50)

| 모델 | 출발 · 데이터 | 학습 | val mAP50 | ONNX A | ONNX B | Board | B→Board | FP 영상 검출 프레임 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v01 (릴리스) | 스크래치 · 기존 | 407 ep | 0.799 | 0.780 | 0.764 | 0.751 | −0.013 | 2,860 |
| **E1** `ft_bgv3` | v01 · 기존 + 배경 | 89 ep | 0.798 | **0.787** | **0.772** | **0.763** | −0.009 | 2,466 |
| E0 `ft_nobg` (대조군) | v01 · 기존 | 92 ep | 0.803 | 0.787 | 0.771 | 0.763 | −0.008 | 2,780 |
| E2 `scratch_bgv3` | 스크래치 · 기존 + 배경 | 201 ep (조기종료) | 0.775 | 0.760 | 0.744 | — | — | — |
| E2b `scratch_bgv3_pat100` | 스크래치 · 기존 + 배경 | 500 ep | 0.795 | 0.785 | 0.769 | 0.758 | −0.011 | **2,438** |
| E3 `ft_bgv3_lowcam` | v01 · 기존 + 배경 + low_cam | 58 ep | 0.796 | 0.780 | 0.767 | — | — | 2,542 |
| E4 `ft_bgv3_ft2` | E1 · 기존 + 배경, lr 1e-4 | 1 ep 에서 취소 | — | — | — | — | — | — |
| C1 (E1 변환 변형) | percentile 0.9999 | — | — | 0.787 | 0.772 | 0.761 | −0.011 | — |

- **지표 정의**
  - ONNX A: PC 기본 후처리다.
  - ONNX B: 보드 NMS 규칙을 흉내 낸 후처리다. 보드 순위와 맞는다.
  - Board: 보드 덤프를 거르지 않고 평가한 값이다(Front 방식). v01 덤프만 보드에서 0.01 로 잘려 있다.
- **FP 영상**: `260911_CVD-F410_REAR_FP_case` 영상 3개(4,175프레임)를 conf 0.25 로 집계했다.
- **ultralytics 코어(0.01 컷)**: Board mAP50 은 v01 0.800 / E1 0.807 / E0 0.807 / E2b 0.806 이다.
  - 릴리스 기록의 "v01 Float 0.796 / Board 0.800(손실 +0.45%)"는 컷 차이에서 생긴 착시다.

### 배포용 파일 (E1)

| 항목 | 위치 |
|---|---|
| `nvt_model.bin` | `runs/npu/novatek/rear_truck_v01_symmetric_ft_bgv3_320/nvtai_tool/output/REAR_TRUCK_V01_SYMMETRIC_FT_BGV3_320/sdk/nvt_model.bin` (md5 `25e5ea3c23b77abbeacdc7b9e3591681`) |
| ONNX | `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/weights/best_novatek.onnx` (md5 `ceb76d59…`) |
| 보드 덤프 · 평가 | `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/{board,eval}/` |
| 클래스별 임계값 | `runs/analysis/rear_truck_thresholds/` — optimal bus 0.291 / car 0.500 / motorcycle 0.250 / person 0.273 / truck 0.251 |

- 보드 점수는 고정된 값(0.274 / 0.343 / 0.409 / 0.500 …)으로만 나온다. 임계값은 인접한 두 값 사이 어디에 두어도 결과가 같다.
- v01 릴리스 임계값과 실제로 다른 것은 bus(0.413 → 0.291)뿐이다.

### 기타 확인한 사실

- 6,152장 iputest = 학습 val 7,894장에서 DPI 144 JPEG 1,742장을 뺀 집합이다. 집합이 정확히 일치한다. 보드 디코더에서 재현하지는 않았다.
- NAS `Rear_Truck/val` = 로컬 `1T_REAR/pure/val` 이다(md5 일치). NAS `Rear_Truck/train` = 로컬 학습 데이터 + low_cam 7,088장이다.
- 학습·평가·변환은 모두 BGR [0,255] 다(`bgr: 1.0`, D-004·D-005).

아래는 날짜순 상세 기록이다.

## 기준 모델과 평가 방식

- 릴리스 모델: `v01_symmetric`
  - old 저장소 `/home/ncn/Workspace/jdh/ultralytics_projects/runs/detect/BSD_F606/Rear_Truck/v01_symmetric/`
  - ONNX `260821_best.onnx` = 이 저장소에서 새로 export 한 `best_novatek.onnx` (가중치 127/127 동일)
  - 같은 폴더의 `best.onnx`(09-07)는 출력이 1개인 다른 export 라 2-출력 후처리로 평가할 수 없다.
- 평가는 Front 와 같은 customEval 방식이다.
  - 평가셋: 보드 평가셋 6,152장 (GT 34,267, ROI 없음, 320x320 stretch, BGR [0,255])
  - 설정·명령: `runs/detect/BSD_F606/Rear_Truck/_eval_cfg/README.md`
  - old 저장소의 `std+ultra` 수치와는 mAP 계산 방식이 달라 직접 비교하지 않는다(D-020).

| 모델 | ONNX A | ONNX B | Board (conf 0.01) |
|---|---:|---:|---:|
| v01_symmetric (릴리스) | 0.780 | 0.764 | 0.751 |
| v02_symmetric_bg | 0.762 | 0.746 | 0.734 |
| 260914_v02_symmetric_new_bg | 0.757 | 0.740 | 0.728 (보드 덤프 2026-10-08 수신, 이때 처음 측정) |

ONNX B 는 Front 보드 후처리 근사라 Rear 에서는 참고값이다.

## 배경 데이터셋 만든 경위 (2026-10-12)

1. NAS `01_DATASET/00_WORKING_DB/01_주행영상/01_TRUCK1T/03_REAR/999_output_auto_label` 를 HDD 로 스냅샷
   - 위치: `/mnt/HDD_1/BSD_F606_REAR/auto_label_snapshot_20261012/`
   - 구성: `background_images` 6,422장, `no_bg` 3,578장
2. v01 ONNX(`260821_best.onnx`)로 추론. YOLO 라벨, 시각화, `detected/no_detection` 분할을 만들었다.
   - conf 0.25: `/mnt/HDD_1/BSD_F606_REAR/infer_rear_v01_symmetric_conf025_20261012/`
   - conf 0.1: `/mnt/HDD_1/BSD_F606_REAR/infer_rear_v01_symmetric_conf010_20261012/` (`background_images` 의 시각화 폴더 이름은 사용자가 `vis_conf010` 으로 바꿨다)
3. NAS 정리 (원본 위치에서 이동, 기록은 HDD `/mnt/HDD_1/BSD_F606_REAR/*.tsv|txt`)
   1. `background_images` 일부를 `bg_2` 로 이동
   2. 두 폴더를 conf 0.1 기준 `yes_detect/no_detect` 로 분류
   3. 사용자 검수 후 `final/images` 로 이동
   4. 시각화를 `final/vis` 로 복사
4. 중복 제거 (기록: `/mnt/HDD_1/BSD_F606_REAR/final_dup_check_20261012/`)
   - MD5 완전 중복 6쌍: 1분 영상 마지막 프레임(`_00001800`)과 다음 영상 첫 프레임(`_00000000`)
   - 정차 장면: 세션 내, 고정 영역 마스크, 96x54, 상관 ≥ 0.995. 예시 시트로 기준을 확인했다.
   - 105장을 `final/dup/` 으로 옮겼다.
5. 육안 검수에서 객체가 있는 75장을 `final/no_bg/{vis,images}` 로 옮겼다.
6. 최종 4,724장은 모두 사용자가 확인한 객체 없는 배경이다. 그중 v01 오검출이 있던 이미지가 다수다(제외 전 4,799장 기준 conf 0.25 오검출 이미지 26.5%).
7. 학습용 고정 사본: `/home/ncn/Workspace/data/1T_REAR/background_rear_v3/train/images` (4,724장, 라벨 없음, `README.md`)

스크립트: `/mnt/HDD_1/BSD_F606_REAR/_scripts/` (`rear_infer.py`, `rear_vis.py`, `split_det.py`, `bg_fp_infer.py`)

## 사용자 결정

- 평가용 배경을 떼지 않는다. 4,724장 전부 학습에 쓴다.
- 학습에 쓰지 않은 기존 배경(`background_rear` 3,772 + `background_rear_v2` 2,210)은 **평가에 쓰지 않는다**. 평가는 6,152장 val 로만 한다.
  - 이미 만든 추론 파일 `eval/bg_fp_conf001.json` (v01, ft_bgv3)은 집계하지 않았다.
- 대조군 E0 은 한때 빼기로 했다가, GPU0 이 비는 동안 돌리기로 다시 정했다.

## 실험

- config 는 브랜치 `feat/rear-bgv3-train`(worktree `.claude/worktrees/rear-bgv3-train`)에 커밋했다(`6623979`, `b78f99f`, `6552e9f`, push 안 함).
- 학습은 기존 체크아웃에 복사한 같은 config 로 돌렸다.

| 실험 | config | 출발점 | 데이터 | 설정 | GPU · 컨테이너 |
|---|---|---|---|---|---|
| E1 | `v01_symmetric_ft_bgv3` | v01 `best.pt` | 기존 95,586 + 배경 4,724 | lr0 1e-3, 100 epoch, patience 30, warmup 1 | GPU0 `rear_v01_symmetric_ft_bgv3` |
| E0 | `v01_symmetric_ft_nobg` | v01 `best.pt` | 기존 95,586 | E1 과 같음 (data 만 다름) | GPU0 `rear_v01_symmetric_ft_nobg` (E1 종료 후 대기 스크립트로 자동 시작) |
| E2 | `v01_symmetric_scratch_bgv3` | 무작위 초기화 | 기존 + 배경 4,724 | v01 레시피 그대로 (lr0 0.01, 500 epoch, patience 15) | GPU1 `rear_v01_symmetric_scratch_bgv3` |

- 데이터셋 yaml
  - `configs/datasets/1t_rear_cnr_postprocess_bg_v3.yaml`: 배경 포함
  - `configs/datasets/1t_rear_cnr_postprocess.yaml`: 배경 없음
  - v01 이 쓴 `bsd_rear_truck.yaml` 은 저장소에 없다. 이미지 수(95,586)가 같은 구성으로 대신했다.
- 무작위 초기화: 모델 yaml 로 시작하면 `pretrained: true` 여도 아무 가중치도 불러오지 않는다. v01_symmetric 도 같은 코드라 같은 조건이다.
- 확인 실행: `--cfg_options` 로는 overrides 를 바꿀 수 없다. 별도 config 를 만들어 돌렸다.
- 진행 확인: `watch -n 60 python3 runs/detect/BSD_F606/Rear_Truck/rear_progress.py`, 로그 `runs/detect/BSD_F606/Rear_Truck/<run>.train.log`
- 대기 스크립트: `runs/detect/BSD_F606/Rear_Truck/queue_ft_nobg.sh` (로그 `queue_ft_nobg.log`, 이미 실행 완료)

### 상태 (2026-10-13 10:03)

| 실험 | 상태 | 학습 중 val mAP50 (pure/val 7,894, v01 0.799) |
|---|---|---|
| E1 | 완료 10-13 01:05, 89 epoch 조기종료, best ep59 | 0.798 (P 0.868, R 0.694) |
| E0 | 완료 10-13 09:54, 92 epoch | 최고 0.803 (ep57). ep59 기준 0.802 |
| E2 | 진행 중 175/500 | 0.773. 최대 예상 10-14 16:17 |

E2 는 2026-10-13 12:16 에 201 epoch 에서 조기종료됐다(patience 15, best ep186, 20.0시간). best.pt 의 val mAP50 은 0.775 (P 0.860, R 0.664)다. v01_symmetric 은 같은 레시피로 ep390 까지 가서 0.799 였다.

### E1 평가 (6,152장, 2026-10-13)

| | ONNX A | ONNX B | optimal 기준 TP / FP / FN (A) |
|---|---:|---:|---|
| v01_symmetric | 0.780 | 0.764 | 23,671 / 4,558 / 10,596 |
| E1 ft_bgv3 | 0.787 | 0.772 | 23,836 / 4,358 / 10,431 |

- 모든 클래스가 올랐고 truck 이 가장 많이 올랐다(+0.017 / +0.019). 오검출 −4.4%, 놓침 −165.
- 이 개선에는 배경 효과와 추가 학습 효과가 섞여 있다. E0 과 비교해야 분리된다.
- 결과: `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/eval/onnx_{A_conf0001,B_boardpp}/`, ONNX `weights/best_novatek.onnx`

### E0 평가 (6,152장, 2026-10-13 10:15)

| | ONNX A | ONNX B | optimal 기준 TP / FP / FN (A) |
|---|---:|---:|---|
| v01_symmetric | 0.780 | 0.764 | 23,671 / 4,558 / 10,596 |
| E1 ft_bgv3 | 0.787 | 0.772 | 23,836 / 4,358 / 10,431 |
| E0 ft_nobg | 0.787 | 0.771 | 23,653 / 4,117 / 10,614 |

- E1 과 E0 의 mAP 가 같다(A 0.787 / 0.787, B 0.772 / 0.771). v01 대비 +0.007 은 추가 학습 효과이고, 이 평가셋에서는 배경 효과가 보이지 않는다.
- TP/FP/FN 은 모델마다 다른 클래스별 optimal threshold 에서 센 값이라 FP 수를 모델 간에 직접 비교할 수 없다. E0 는 FP 가 적은 대신 TP 도 적다.
- 결과: `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_nobg/eval/onnx_{A_conf0001,B_boardpp}/`, ONNX `weights/best_novatek.onnx`

### 손실 분해 — Front 방식 (2026-10-13 17:40)

Front 는 임계값 0.001 빌드의 보드 덤프를 거르지 않고 평가한다(`Front_Truck/v02_symmetric_scratch/eval/board_stage4`, 148,062개, 0.713). Rear E1·E0·C1 덤프도 0.001 빌드라 같은 방식으로 맞췄다.

| 모델 | A | B | Board (거르지 않음) | A→B | B→Board | A→Board |
|---|---:|---:|---:|---:|---:|---:|
| v01 (덤프가 0.01 로 잘려 있음) | 0.780 | 0.764 | 0.751 | −2.1% | −1.6% | −3.7% |
| E1 | 0.787 | 0.772 | 0.763 | −1.9% | −1.1% | −3.0% |
| C1 | 0.787 | 0.772 | 0.761 | −1.9% | −1.4% | −3.3% |
| E0 | 0.787 | 0.771 | 0.763 | −2.1% | −1.0% | −3.0% |
| Front v02 (참고) | 0.738 | 0.723 | 0.713 | −2.0% | −1.4% | −3.4% |

- A→Board −3% 중 약 −2% 는 후처리 차이(A→B)다. 양자화 몫(B→Board)은 −1.0~1.6% 다.
- 따라서 16비트 보호(C3)로 줄일 수 있는 몫은 최대 0.008 정도다. 보드 후처리(이중 임계 NMS) 쪽이 더 큰 몫이다.
- 미검증: ONNX B 는 Front 보드 소스를 따른 근사다(Front 에서 2장 기준 약 75% 일치). Rear 보드 후처리가 같은지는 모른다.
- C1 거르지 않은 평가: `v01_symmetric_ft_bgv3/eval/board_c1p9999_conf001/`

### truck A→B 손실의 원인 — 겹쳐 보이는 트럭 (2026-10-16)

> **정정 (2026-10-16, 사용자 확인):** 처음에는 아래 겹침을 "같은 트럭의 전체 + 캐빈 중복 라벨(GT 오류)"로 해석했다. 사용자가 해당 클립과 다른 클립 사례를 확인한 결과 **모두 정상 라벨**이다. 작은 박스는 앞 트럭 뒤에 일부만 보이는 **다른 트럭**이다. 따라서 아래 "중복 제거 GT" 수치는 GT 수정 효과가 아니다. **보드 IoM 규칙이 가려진 트럭을 지워서 생기는 손실의 크기**로 읽는다.

- **6,152장 GT 에는 트럭 박스 안에 다른 트럭 박스가 대부분 들어가는 경우(IoM ≥ 0.6)가 144쌍 있다.**
  - 그중 131쌍이 한 클립 `KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC2_PGL_NC22015` 에서 나왔다. 정차한 트럭 뒤로 다른 트럭이 일부 보이는 주차장 장면이고, 이 클립이 6,152장 중 548장(9%)이다.
  - 다른 클립과 학습셋 사례(약 400쌍)도 대부분 줄지어 선 트럭이 서로 가린 장면이다.
  - 보드 NMS 는 작은 박스 기준 겹침 ≥ 0.6 이면 지우므로, 모델이 둘 다 맞혀도 뒤쪽 트럭은 FN 이 된다.
- car-truck 겹침(44쌍)은 대부분 실제로 다른 차량이 겹쳐 보이는 정상 라벨이다.
- 안쪽(작은) truck GT 144개를 뺀 GT 로 E1 을 다시 채점했다.
  - GT: `runs/detect/BSD_F606/Rear_Truck/_eval_cfg/gt_variants/ipu_gt_no_inner_truck.json` (34,267 → 34,123)
  - 결과: `v01_symmetric_ft_bgv3/eval/gtfix_*`

| GT | truck A | truck B | truck Board | truck A→B | 전체 A | 전체 B | 전체 Board |
|---|---:|---:|---:|---:|---:|---:|---:|
| 원래 | 0.778 | 0.728 | 0.716 | −0.050 | 0.787 | 0.772 | 0.763 |
| 중복 제거 | 0.761 | 0.758 | 0.746 | **−0.003** | 0.783 | 0.778 | **0.769** |

- **truck 의 A→B −0.050 은 거의 전부 이 144쌍(가려진 트럭)에서 나온다.** 그 밖의 truck 에서 보드 규칙 손실은 −0.003 이다.
- 모델은 가려진 트럭도 잘 찾는다(A 0.778). 보드의 고정 규칙(IoM 0.6)이 이를 지운다. 규칙은 바꿀 수 없으므로 Rear 보드의 고정 비용이다.
- 학습셋에도 같은 겹침이 있다. pure/train 220쌍, CnR 171쌍, vignetting 11쌍, new_labeling 12쌍이다(truck 박스 약 43,700개 중 약 1%). 정상 라벨이라 손댈 대상이 아니다.
- 평가셋의 9% 가 이 한 정차 클립이라, 이 장면의 손실이 truck 점수를 크게 흔든다. 평가셋 구성 문제로 기록해 둔다.
- 예외로 실제 중복 라벨이 하나 확인됐다. `KR_20250424_TRUCK1T_REAR_250CM_180_LDC2_OUTDOOR_NC22015_041_10744` 에서 한 픽업트럭에 truck 박스가 2개 있다(2번·3번 줄, 위쪽 경계만 약간 다름).
  - 같은 클래스·IoU ≥ 0.7 인 거의 같은 박스 쌍은 pure/val 16(6,152장 안 13), pure/train 49, CnR 40, vignetting 5, new_labeling 5, low_cam 2 다.
  - 대부분 person 이다. 실제로 겹쳐 선 사람도 섞여 있을 수 있어 전부 오류라고 단정하지 않는다. 수가 적어 지표 영향은 작다.
  - val 16건의 목록과 이미지: `runs/analysis/rear_truck_label_check/dup_val.{csv,jpg}`. 16건 중 15건이 중복으로 보인다(#5 는 애매).
  - **중복을 지운 GT 로 E1 재평가 (2026-10-16, 테스트)**
    - 6,152장 안 12건(#5 제외)에서 각 쌍의 두 번째 박스를 지운 사본 `_eval_cfg/gt_variants/ipu_gt_dedup12.json` 을 썼다(34,267 → 34,255). 원본 GT 는 md5 가 그대로다.
    - 결과는 `v01_symmetric_ft_bgv3/eval/gtdedup_{onnx_A,onnx_B,board}/` 에 있다.
    - mAP50 변화: A 0.7866 → 0.7868, B 0.7718 → 0.7719, Board 0.7631 → 0.7635
    - 클래스별 변화는 최대 +0.0014 다(Board person). 중복 라벨의 지표 영향은 무시할 수준이다.

### E1 보드 클래스별 임계값 (2026-10-15)

Front 와 같은 `calc_class_thresholds.py --pred-file`(IoU 0.5, cut 0.99)로 보드 덤프에서 구했다. 상세는 `runs/analysis/rear_truck_thresholds/README.md` 다.

| 클래스 | sensitive | optimal | insensitive |
|---|---:|---:|---:|
| bus | 0.189 | 0.291 | 0.500 |
| car | 0.344 | 0.500 | 0.500 |
| motorcycle | 0.165 | 0.250 | 0.308 |
| person | 0.165 | 0.273 | 0.343 |
| truck | 0.125 | 0.251 | 0.409 |

- 보드 점수는 클래스당 50여 개의 고정값으로만 나온다(0.274 / 0.343 / 0.500 / 0.657 등). 임계값은 인접 값 사이 어디든 결과가 같다.
- 원인은 cls logit 의 8비트 양자화로 추정한다(미확인).

### 보드식 NMS 규칙 변형 (2026-10-14, E1 ONNX, customEval, conf 0.001)

- 모듈: `projects/metrics/board_postproc_variants.py` (새 파일; `board_postproc.py` 는 그대로)
- 결과: `v01_symmetric_ft_bgv3/eval/nms_<변형>/`

| 후처리 | 차량 묶음 | IoU | IoM | mAP50 | bus | car | moto | person | truck | optimal TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A (PC 기본) | — | 0.45 | — | 0.787 | 0.875 | 0.784 | 0.785 | 0.712 | 0.778 | 23,836 / 4,358 / 10,431 |
| B = b_ref (보드 근사) | O | 0.3 | 0.6 | 0.772 | 0.856 | 0.767 | 0.794 | 0.715 | 0.728 | 23,859 / 4,321 / 10,408 |
| b_nogroup | X | 0.3 | 0.6 | 0.785 | 0.876 | 0.774 | 0.794 | 0.715 | 0.764 | 23,951 / 4,377 / 10,316 |
| b_noiom | O | 0.3 | — | 0.785 | 0.863 | 0.777 | 0.795 | 0.718 | 0.771 | 24,068 / 4,417 / 10,199 |
| b_iou045 | O | 0.45 | 0.6 | 0.774 | 0.856 | 0.770 | 0.794 | 0.722 | 0.728 | 23,914 / 4,366 / 10,353 |
| **b_nogroup_noiom** | X | 0.3 | — | **0.795** | 0.878 | 0.780 | 0.795 | 0.718 | **0.802** | 24,107 / 4,441 / 10,160 |
| (실제 Board, 참고) | | | | 0.763 | 0.853 | 0.759 | 0.787 | 0.701 | 0.716 | 23,776 / 4,863 / 10,491 |

- b_ref 가 B 와 검출 수·mAP 까지 같다. 변형 코드가 맞다는 검증이다.
- truck 손실의 원인은 **차량 묶음 NMS 와 IoM 0.6 조건 둘 다**다. 하나씩 풀면 각각 +0.013 이고, 둘 다 풀면 +0.023 이다. 둘 다 풀면 A(0.787)보다도 높다.
- IoU 0.3 → 0.45 는 효과가 거의 없다(+0.002).
- 보드 예상: Board 0.763 + 0.023 ≈ 0.786 이다. B→Board 차이가 그대로 유지된다고 가정한 추정치다.
- 주의
  - 묶음을 풀면 같은 차량에 car·truck 박스가 겹쳐 남는 경우가 생길 수 있다.
  - IoM 을 풀면 큰 박스 안의 작은 박스가 남는다.
  - optimal 기준 FP 가 120 늘었다(+2.8%). 경고 로직에서 중복 박스가 문제가 되는지는 보드·제품 쪽 판단이 필요하다.
- 보드 규칙은 바꿀 수 없다(사용자, 2026-10-14). 이 표는 고정된 보드 규칙이 truck 을 얼마나 깎는지 보여주는 분석이다.
- **보드 소스 확인** (old 저장소 `temp/nn_post_process.{cpp,h}`, 2026-08-10 사본; 실제 보드 빌드와 같은지는 미확인)
  - `post_proc_baseline`: anchor 마다 top-1 클래스를 고른다. `check_vehicle()` 이 `CLASS_MERGE_EN (1)` 일 때 BUS·TRUCK·CAR 를 CAR 묶음 하나로 넣는다. 결과 라벨은 원래 클래스를 유지한다(`_class`).
  - 묶음별로 점수순 정렬 후 상위 `ADAS_BBOX_BUF_MAX`(= `DETECTION_LABEL_MAX_NUM`, 값은 이 사본에 없음)개만 남긴다.
  - `nms_proc_baseline`: `inter/min(면적) >= NMS_MIN_OVERLAP_THRESH (0.6)` 또는 `IoU >= NMS_UNION_OVERLAP_THRESH (0.3)` 이면 제거한다.
  - B 근사의 규칙(차량 묶음, IoM 0.6 / IoU 0.3, top-1)과 일치한다. B 의 상위 256 은 소스 값을 확인하지 못했다.
  - 이 사본은 점수 하한을 `threshold_value = 0.01` 로 고정한다. 그런데 E1·E0·C1 보드 덤프의 최소 점수는 0.001 대다. 측정에 쓴 빌드는 이 사본과 하한이 다르다.
  - Rear 모델이 이 함수를 쓰는지(dispatch, `cnn_det_nt98530_api.cpp`)는 확인하지 못했다. 같은 파일에 `post_proc_dfl` 도 있다.

### old 릴리스 기록 v01 "Float 0.796 / Board 0.800" 재현 (2026-10-14)

- 스크립트: `runs/detect/BSD_F606/Rear_Truck/_eval_cfg/ultra_ap.py` (ultralytics `ap_per_class` + `match_predictions` 방식 매칭)

| 채점 코어 | v01 ONNX conf 0.001 | v01 ONNX conf 0.01 | v01 Board (보드 0.01 컷) |
|---|---:|---:|---:|
| ultralytics ap_per_class (old "std+ultra") | 0.7966 | 0.8107 | 0.8000 |
| customEval | 0.7799 | 0.7755 | 0.7510 |

- old 기록(Float P 0.864 / R 0.721 / 0.796 / mAP50-95 0.579, Board 0.800)이 클래스별까지 재현된다.
- ultralytics 코어에서는 conf 를 0.001 → 0.01 로 올리면 mAP 가 오른다.
  - 매칭이 점수 순이 아니라 IoU 순이라, 낮은 점수 예측이 GT 를 가져가 높은 점수 예측을 FP 로 만든다.
- 그래서 old 기록의 "v01 양자화 손실 +0.45%(보드가 더 높음)"는 컷이 달라서 생긴 착시다.
  - 비교 대상이 conf 0.001 의 ONNX 와 0.01 로 잘린 보드였다.
  - 같은 0.01 컷이면 0.811 → 0.800, −0.011(−1.4%)이다.
- D-020: 두 코어의 수치를 섞어 비교하지 않는다.

ultralytics 코어로 본 전체 모델 (같은 0.01 컷, mAP50 / mAP50-95):

| 모델 | ONNX | Board | 손실 mAP50 | 손실 mAP50-95 |
|---|---|---|---:|---:|
| v01 | 0.811 / 0.601 | 0.800 / 0.575 | −1.3% | −4.4% |
| E1 ft_bgv3 | 0.818 / 0.612 | 0.807 / 0.574 | −1.4% | −6.3% |
| C1 (E1 변환 변형) | 0.818 / 0.612 | 0.804 / 0.591 | −1.7% | −3.5% |
| E0 ft_nobg | 0.817 / 0.611 | 0.807 / 0.578 | −1.2% | −5.3% |
| E2 scratch_bgv3 | 0.790 / 0.575 | — | | |
| E3 ft_bgv3_lowcam | 0.812 / 0.605 | — | | |

- old 방식(ONNX conf 0.001)이면 ONNX 는 v01 0.797, E1 0.804, E0 0.804, E2 0.776, E3 0.798 이다.
- 새 보드 덤프(0.001 빌드)를 거르지 않으면 E1 0.793, C1 0.790, E0 0.793 이다.
- C1 은 mAP50 은 E1 보다 낮지만 mAP50-95 는 0.591 로 높다. 클래스별로 bus 0.697 → 0.720, truck 0.664 → 0.696 이다.
  - box 위치 정밀도가 좋아졌다. 시뮬 dfl cosine 이 0.9947 → 0.9979 로 오른 것과 맞는다.

### 보드 덤프 ↔ ONNX 검출 일치도 (2026-10-14, Front 2026-10-12 방식)

- 방법: 보드 덤프의 score ≥ 0.3 검출마다, 같은 이미지·클래스에서 IoU ≥ 0.7 인 ONNX 검출(score ≥ 0.01)이 있는 비율과 |Δscore| 중앙값을 본다.
- 스크립트: `runs/detect/BSD_F606/Rear_Truck/_eval_cfg/board_onnx_match.py` (numpy 가 필요해 dair-train 컨테이너에서 실행)

| 보드 덤프 \ ONNX | v01 | E1 | E0 | E2 | E3 |
|---|---|---|---|---|---|
| v01 | **0.966 / 0.036** | 0.952 / 0.045 | 0.954 / 0.046 | 0.891 / 0.042 | 0.960 / 0.040 |
| E1 | 0.954 / 0.035 | **0.964 / 0.035** | 0.959 / 0.038 | 0.885 / 0.046 | 0.956 / 0.036 |
| C1 | 0.957 / 0.045 | **0.967 / 0.049** | 0.959 / 0.052 | 0.888 / 0.045 | 0.959 / 0.047 |
| E0 | 0.954 / 0.035 | 0.957 / 0.035 | **0.963 / 0.033** | 0.880 / 0.049 | 0.954 / 0.034 |

- 모든 덤프가 자기 ONNX 와 가장 잘 맞는다. 덤프와 bin 의 짝이 맞다.
  - v01 덤프는 `260821_best.onnx` 계열이다. v01 old 레시피 `deploy_float.onnx` 는 md5 가 달랐지만 같은 가중치 계열로 본다.
- 파인튜닝 모델끼리는(v01·E0·E1·E3) 0.95 대로 붙어 있어 짝 구분 폭이 0.004~0.012 로 작다. 독립 학습한 E2 는 0.88~0.89 다.
- 자기 짝 일치율 0.963~0.967 은 Front 대조군(0.979)보다 조금 낮다.
- C1 은 일치율은 같지만 |Δscore| 가 0.049 로 E1(0.035)보다 크다. percentile 0.9999 가 범위를 넓힌 만큼 점수 분해능이 거칠어진 것과 맞는다. Board mAP 가 오르지 않은 것과도 맞는다.

### FP 케이스 영상 3개 (2026-10-13 10:30)

- 영상: old 저장소 `runs/video_infer/_input/260911_CVD-F410_REAR_FP_case/` (3개, 4,175프레임, GT 없음)
- 조건: old `tools/debug/video_infer_compare.py` 와 같은 전처리·후처리 (320 stretch, conf 0.25, NMS 0.45). v01 은 `260821_best.onnx`
- 결과: `runs/video_infer/rear_infer_260911_CVD-F410_FP_case_bgv3/` (`*_3models.mp4` 는 왼쪽부터 v01 | E1 | E0, 그 밖에 `*_per_frame.csv`, `summary.json`)

| 모델 | 검출 있는 프레임 | car | truck | motorcycle | person | bus |
|---|---:|---:|---:|---:|---:|---:|
| v01 | 2,860 | 18,212 | 396 | 325 | 452 | 26 |
| E1 ft_bgv3 | 2,466 | 16,871 | 219 | 229 | 336 | 20 |
| E0 ft_nobg | 2,780 | 17,813 | 434 | 222 | 432 | 38 |

- E1 만 검출이 줄었다(검출 있는 프레임 −14%, truck −45%, person −26%). E0 는 v01 과 비슷하다. 6,152장 평가에서는 보이지 않던 배경 효과가 여기서는 보인다.
- 이 숫자는 오검출 수가 아니라 전체 검출 수다. 줄어든 검출이 오검출이었는지는 영상을 보고 확인해야 한다(미검증).

### E1·E0 NPU 변환 (2026-10-13 11:00–11:57)

- 레시피: `novatek/convert/rear_truck_v01_symmetric_{ft_bgv3,ft_nobg}_320/`. 설정은 old 저장소 v01 레시피(`rear_truck_v01_symmetric_320`)와 같고 경로·PATTERN 만 다르다.
  - 3-tensor 보호, percentile 0.999, refine 5, 1920x1080 YUV 입력 → 320 resize
  - 보정 940장(주간 520 / 야간 420)과 시험 500장은 old `rear_truck_v01_320` 에서 복사했다. md5 는 원본과 같다.
- 실행: `run_pipeline.sh <dir> KR_20240318_TRUCK5T_REAR_350CM_B_180FISH_LDC1_PGL_DVR_NC22015_0126.jpg 0.25 --names bus car motorcycle person truck` (`nice -n 19`), 모델당 약 28분. 로그 `runs/npu/novatek/<dir>.pipeline.log`
- 컴파일 지표는 두 모델 모두 v01 과 같다: latency 6.368 ms, DRAM 36.54 MB, bin 4,711,336 B

| 모델 | deploy_float.onnx md5 | nvt_model.bin md5 | Softmax(dfl) / Sigmoid / dfl_conv cosine | Golden → Sim 검출 (conf 0.25) |
|---|---|---|---|---|
| v01 (old 기록) | — | `6b49a6d091458a96fdadb50e7dd87af7` | 0.9850 / 0.9824 / 0.9933 | — |
| E1 ft_bgv3 | `ceb76d591d51d5d7326c84a5f11b388d` | `25e5ea3c23b77abbeacdc7b9e3591681` | 0.9895 / 0.9855 / 0.9947 | 16 → 16 |
| E0 ft_nobg | `eccce6f09e7ba7bada747edc3b74f911` | `ce38781e90f4d92d885676ace646daee` | 0.9829 / 0.9825 / 0.9920 | 17 → 19 |

- bin: `runs/npu/novatek/<dir>/nvtai_tool/output/<PATTERN>/sdk/nvt_model.bin`
- cosine 과 검출 수는 시험 이미지 1장으로 본 진단값이다. 보드 측정은 사용자가 직접 한다.

### E1 보드 측정 (덤프 2026-10-13 13:34 수신)

- 덤프: `runs/detect/BSD_F606/Rear_Truck/v01_symmetric_ft_bgv3/board/v01_symmetric_ft_bgv3_output.json` (md5 `8c1ffcebb11233f85e1a25237f4d9462`, 원래 이름 `261013_REAR_bg3_output.json`). 6,152장, 검출 180,049개, 최소 score 0.0012
- stale-model 검사: old `board_dumps/` 덤프 14개와 detection-key jaccard 가 모두 0.0001 이하다. 이전 bin 결과가 아니다.
- **보드 덤프의 score 하한이 덤프마다 다르다.** `customEval --conf` 는 보드 덤프를 거르지 않는다.
  - v01 덤프는 보드에서 0.01 로 잘려 있다(92,743개, 최소 0.0103).
  - E1 덤프(최소 0.0012)와 v02_new_bg 덤프(최소 0.0015)는 0.01 미만이 남아 있다.
  - 그래서 v01 과 맞추려면 score ≥ 0.01 로 거른 사본(`..._output_score001.json`, 88,106개)으로 평가해야 한다.
  - v02_new_bg 의 0.728 은 거르지 않고 잰 값이라 v01 0.751 과 조건이 다르다.

| 모델 | ONNX A | Board (score ≥ 0.01) | 상대 손실 | optimal TP / FP / FN (Board) |
|---|---:|---:|---:|---|
| v01_symmetric | 0.780 | 0.751 | −3.7% | 23,584 / 5,016 / 10,683 |
| E1 ft_bgv3 | 0.787 | **0.759** | −3.6% | 23,776 / 4,863 / 10,491 |

- 결과: `v01_symmetric_ft_bgv3/eval/board_score001/`. 거르지 않은 덤프로 잰 값(0.763)은 `eval/board_conf001/` 에 있고 참고용이다.
- 보드에서 v01 대비 +0.008 이다. 양자화 손실은 v01 과 같은 수준이라 float 에서 얻은 개선이 보드까지 유지됐다. 손실 ≤1% 기준은 v01 과 마찬가지로 미달이다.

### E0 보드 측정 (덤프 2026-10-13 14:48 수신)

- 덤프: `v01_symmetric_ft_nobg/board/v01_symmetric_ft_nobg_output.json` (md5 `5064a56442c35a4e4ec19f9b2e6cd398`, 원래 이름 `261013_BSD-F606_REAR_FT_nobg_output.json`). 검출 180,633개, score 0 인 검출 1개
- stale-model 검사: 기존 덤프 14개, E1 덤프와 jaccard 가 모두 0.0001 이하다.
- score ≥ 0.01 로 거른 사본(89,122개)으로 평가했다.

| 모델 | ONNX A | Board (score ≥ 0.01) | 상대 손실 | optimal TP / FP / FN (Board) |
|---|---:|---:|---:|---|
| v01_symmetric | 0.780 | 0.751 | −3.7% | 23,584 / 5,016 / 10,683 |
| E1 ft_bgv3 | 0.787 | 0.759 | −3.6% | 23,776 / 4,863 / 10,491 |
| E0 ft_nobg | 0.787 | 0.759 | −3.6% | 23,300 / 3,992 / 10,967 |

- 보드에서도 E1 과 E0 의 mAP 가 같다. 6,152장 평가셋에서는 배경 효과가 보드에서도 보이지 않는다. v01 대비 +0.008 은 추가 학습 효과다.
- TP/FP/FN 은 모델마다 다른 optimal threshold 에서 센 값이라 FP 수를 모델 간에 비교할 수 없다.
- 배경 효과가 보인 곳은 지금까지 FP 케이스 영상 3개뿐이다(float ONNX, conf 0.25).

### 평가셋 구성 확인 (2026-10-13)

- 학습 val `/dataset/1T_REAR/pure/val`: 7,894장, GT 42,642. iputest GT: 6,152장, GT 34,267. 6,152장은 모두 val 안에 있다.
- 두 세트 모두 객체 없는 이미지가 0장이다. 배경 학습의 효과(무객체 장면 오검출 감소)를 mAP 로 잴 수 없는 구성이다.
- **DPI 144 = 보드 처리 불가 1,742장.**
  - val 7,894장은 모두 1920x1080 RGB JPEG 이다.
  - 그중 JFIF density 가 144 dpi 인 이미지가 정확히 1,742장이고, 모두 iputest 에서 빠진 이미지다.
  - 나머지 6,152장은 dpi 정보가 없고, 모두 iputest 에 들어 있다.
  - old SOP 에 "원인 미확인"으로 남아 있던 보드 처리 불가의 원인이 이 헤더일 가능성이 높다. 집합이 정확히 일치한다는 것만 확인했고, 보드 디코더에서 재현하지는 않았다.
  - 스크립트: 세션 scratchpad `dpi.py` (PIL `Image.info["dpi"]`)

### E2 평가 (6,152장, 2026-10-13 16:10)

- ONNX: `v01_symmetric_scratch_bgv3/weights/best_novatek.onnx` (md5 `af431f3e34db0339d2c10fdda03ce0c0`)

| 모델 (스크래치 계열) | 배경 | ONNX A | ONNX B | optimal TP / FP / FN (A) |
|---|---|---:|---:|---|
| v01_symmetric | 없음 | 0.780 | 0.764 | 23,671 / 4,558 / 10,596 |
| E2 scratch_bgv3 | v3 4,724장 | 0.760 | 0.744 | 22,812 / 4,458 / 11,455 |
| v02_symmetric_bg | 구 배경 3,772장, 600 epoch | 0.762 | 0.746 | 22,944 / 4,258 / 11,323 |
| 260914_v02_symmetric_new_bg | v2 2,210장 | 0.757 | 0.740 | 22,709 / 4,128 / 11,558 |

- 스크래치에 배경을 넣은 세 실험이 모두 v01 보다 0.02 안팎 낮다. 가장 크게 떨어진 클래스는 truck(−0.041)이다.
- **원인은 배경이 아니라 조기종료다** (2026-10-13 학습 곡선 비교로 정정).
  - 같은 epoch 의 val mAP50 이 네 스크래치 실험 모두 거의 같다. ep200 기준 v01 0.774 / E2 0.773 / new_bg 0.772 / v02_bg 0.772 다.
  - 배경 실험들은 patience 15 로 ep201~318 에서 멈췄다. 이때 lr 은 아직 4.6e-3~6.6e-3 이었다.
  - v01 은 ep407(lr 9.4e-4)까지 가서 ep390 에 0.799 가 됐다. 마지막 상승은 lr 이 1e-3 근처로 내려간 구간에서 나왔다.
  - 따라서 "스크래치 + 배경은 손해"라고 결론 낼 수 없다. 공정하게 비교하려면 patience 를 크게 잡거나 꺼서 500 epoch 를 다 돌려야 한다.
- 파인튜닝(E1·E0)의 +0.007 도 같은 설명이 맞는다. v01 은 lr 9.4e-4 에서 멈췄고, 파인튜닝이 lr 1e-3 → 1e-5 cosine 으로 남은 수렴 구간을 채웠다.

### E1 양자화 손실 줄이기 — 변환 설정 변형 (2026-10-13 16:00–17:05)

모두 E1 ONNX(`ceb76d59…`)를 그대로 쓰고 변환 설정만 바꿨다. 레시피는 `novatek/convert/rear_truck_v01_symmetric_ft_bgv3_<변형>_320/` 이다.

| 변형 | 변경 | Softmax / Sigmoid / dfl_conv cosine (0126.jpg) | bin md5 |
|---|---|---|---|
| 기준 E1 | — | 0.9895 / 0.9855 / 0.9947 | `25e5ea3c…` |
| C1 `c1p9999` | percentile 0.999 → 0.9999 | 0.9949 / 0.9869 / 0.9979 | `1f9edf94df84dd162f18f8750e7355c5` |
| C2 `c2traincal` | 보정 940장을 1T_REAR pure/train 846장(val 의 1T/5T×주야 비율에 맞춤) + 배경 v3 94장으로 교체 | 0.9895 / 0.9881 / 0.9943 | `a4a69fd6083fbb4e47a6ee960ae02281` |
| C3 `c3hp12` | 보호 텐서 3 → 12 (production 의 model.18/19/21/22 9개 추가) | E1 과 같음 | `25e5ea3c…` (**E1 과 동일**) |

- C1·C2 는 latency·DRAM·크기가 E1 과 같다.
- **C1 보드 결과 (덤프 17:24 수신)**
  - 덤프: `v01_symmetric_ft_bgv3/board/v01_symmetric_ft_bgv3_c1p9999_output.json` (md5 `10726f74eac1e2da355b118b40e510a6`, 원래 이름 `261013_BSD-F606_REAR_FT_bg_v3_c1p9999_output.json`)
  - stale 검사를 통과했다. E1 덤프와 jaccard 0.0001 이다.
  - score ≥ 0.01 로 거르면 88,062개다. 평가 결과는 `eval/board_c1p9999_score001/` 에 있다.
  - **Board 0.758 로 E1(0.759)과 차이가 없다.**
    - truck 은 0.709 → 0.717 로 올랐고, car(0.757 → 0.753)·motorcycle(0.783 → 0.779)은 내렸다.
    - optimal 기준 TP 23,508 / FP 4,565 / FN 10,759 다.
  - percentile 조정만으로는 양자화 손실이 줄지 않았다.
- **C3 는 추가한 9개가 모두 무시됐다.**
  - 경고 없이 `process_bit_info.txt` 가 E1 과 바이트 단위로 같았다.
  - production(`rear_dair_pp_320`)도 같은 9개를 적었지만 실제로는 8비트였다.
  - 과거에 보호가 적용된 v01 9-tensor 변환에는 지정 텐서마다 `RewriteHpConv*_XH/_XL` 분할이 생겼다. C3 에는 생기지 않았다. 원인은 미확인이다.
  - 매뉴얼(NT98539A 판 `Novaic_Convertor_User_Guide_en.pdf`; NT98530 판은 사내 DRM 암호화로 열리지 않음)에 따르면 CNN25 에서는 `in_hp/conv_op=3` 일 때 conv 입력을 분할해 16비트로 계산한다.
- E1 시뮬 20장 평균 cosine 기준 최약 레이어는 model.21(P5 C2f)과 model.22 cv3(cls 분기)였다(0.92~0.95). box 쪽은 0.99 다.
- 1장 cosine 은 이미지마다 cls 0.83~0.99 로 흔들려 판단 근거로 약하다.
- 시뮬레이터 채점 도구: `runs/npu/novatek/_screen/{sim_screen.sh,score_screen.py,score_eval.sh}`
  - 래퍼를 이미지마다 반복 호출하고, 결과를 GT 부분집합으로 customEval 한다. 1장에 약 14초 걸린다.
  - 사용자 판단으로 대량 채점은 하지 않았다.

### low_cam 확인과 E3·E4 학습 (2026-10-13 17:01 시작)

- NAS `Rear_Truck/train`(102,673) = 로컬 학습 95,585 + **low_cam 7,088** 이다.
  - low_cam: 2023-05~09, 1T 70cm·140° 카메라, 라벨 37,368박스
  - 사용자는 높이·화각이 달라 뺐다고 한다.
  - 박스 높이 중앙값 0.168(pure/train 0.096), 세로 위치 중앙값 0.62(0.21)다. 시점이 정면 수평이고 어안 왜곡이 없어 배포 카메라와 다른 도메인이다.
- 학습 두 개를 시작했다. 진행 상황은 `rear_progress.py` 로 본다.

| 실험 | config | 출발 | 데이터 | 설정 | GPU · 컨테이너 |
|---|---|---|---|---|---|
| E3 | `v01_symmetric_ft_bgv3_lowcam` | v01 best.pt | E1 + low_cam (107,397장) | E1 과 같음 | GPU0 `rear_v01_symmetric_ft_bgv3_lowcam` |
| E4 | `v01_symmetric_ft_bgv3_ft2` | E1 best.pt | E1 과 같음 (100,309장) | lr0 1e-4, 50 epoch, warmup 0, patience 30 | GPU1 `rear_v01_symmetric_ft_bgv3_ft2` |

- 데이터셋 yaml `configs/datasets/1t_rear_cnr_postprocess_bg_v3_lowcam.yaml` 을 새로 만들었다.
- **E3 결과**
  - 10-13 23:18 에 58 epoch 에서 조기종료됐다. fitness(F1) best 가 ep28 이다. val mAP50 은 0.796 이다.
  - 6,152장 평가에서 ONNX A 0.780 / B 0.767 이다. E1(0.787 / 0.772)보다 낮고 v01(0.780 / 0.764)과 같은 수준이다.
  - truck 0.778 → 0.762, bus 0.875 → 0.869 로 내렸다.
  - **low_cam 은 넣지 않는다.** 사용자 판단으로 보드 측정도 하지 않는다.
  - ONNX: `v01_symmetric_ft_bgv3_lowcam/weights/best_novatek.onnx`
- **E4 는 1 epoch(val 0.798)에서 취소했다** (17:12). E1 이 이미 lr 4.5e-5 까지 수렴해 기대 효과가 작다고 판단했다. run dir `v01_symmetric_ft_bgv3_ft2/` 는 남아 있다.
- 대신 GPU1 에서 **E2b `v01_symmetric_scratch_bgv3_pat100`** 를 시작했다 (17:15). E2 와 다른 점은 patience 15 → 100 하나다.
  - 조기종료를 끈 버전(patience 0)을 잠깐 띄웠다가, 사용자 의견으로 100 으로 바꿔 다시 시작했다. patience 0 run 은 2분 만에 지웠다.
  - 500 epoch 를 다 돌면 약 40시간 걸린다.
- **E2b 결과**
  - 10-15 09:15 에 500 epoch 를 다 돌고 끝났다(40.0시간). best 는 ep448 이고 val mAP50 0.795 다(v01 0.799 @ep390).
  - 같은 epoch 비교로 v01 과 곡선이 겹친다. ep300 0.785 / 0.784, ep400 0.795 / 0.797 이다.
  - 6,152장 평가에서 ONNX A 0.785 / B 0.769 다. v01(0.780 / 0.764)보다 높고 E1(0.787 / 0.772)보다 조금 낮다. person 은 0.723 으로 모든 모델 중 가장 높다.
  - **스크래치에 배경을 넣어도 손해가 없다.** E2(0.760)가 낮았던 것은 조기종료 때문이었다.
  - ONNX: `v01_symmetric_scratch_bgv3_pat100/weights/best_novatek.onnx` (md5 `172e2e2b5949d32d460b1a584f8d11c2`)
  - FP 케이스 영상 3개 (conf 0.25, 결과 `runs/video_infer/rear_infer_260911_CVD-F410_FP_case_bgv3_pat100/`, 왼쪽부터 v01 | E1 | E2b)

    | 모델 | 검출 프레임 | car | truck | motorcycle | person | bus |
    |---|---:|---:|---:|---:|---:|---:|
    | v01 | 2,860 | 18,212 | 396 | 325 | 452 | 26 |
    | E1 | 2,466 | 16,871 | 219 | 229 | 336 | 20 |
    | E2b | 2,438 | 16,853 | 212 | 98 | 518 | 31 |

  - E2b 도 v01 대비 검출 프레임이 줄었다(−15%, E1 −14%).
  - 클래스 구성은 E1 과 다르다. motorcycle 은 크게 줄고(98) person 은 v01 보다 늘었다(518).
  - 늘어난 person 이 오검출인지는 영상으로 확인해야 한다(미확인).
- **E2b NPU 변환** (2026-10-15 13:37–13:48)
  - 레시피 `novatek/convert/rear_truck_v01_symmetric_scratch_bgv3_pat100_320/` 는 E1 과 같다(보정 940장, percentile 0.999, 3-tensor).
  - latency 6.368 ms / DRAM 36.54 MB / bin 4,711,336 B 로 E1 과 같다.
  - cosine(0126.jpg) Softmax 0.9887 / Sigmoid **0.9706** / dfl_conv 0.9954 다. E1 은 0.9895 / 0.9855 / 0.9947 이다.
  - Golden → Sim 검출은 14 → 12 다. 0.39 대 bus 두 개 중 하나가 빠졌고 하나는 0.281 로 남았다.
  - 같은 21장 시뮬 평균으로 보면 반대다.
    - Sigmoid: E1 0.960(최저 0.832), E2b **0.973**(최저 0.898)
    - Softmax: 0.983 / 0.984, dfl_conv: 0.992 / 0.993
    - E2b 가 E1 보다 낮은 이미지는 7/21 장이다. 0126.jpg 한 장의 낮은 값은 대표성이 없다.
  - bin: `runs/npu/novatek/rear_truck_v01_symmetric_scratch_bgv3_pat100_320/nvtai_tool/output/REAR_TRUCK_V01_SYMMETRIC_SCRATCH_BGV3_PAT100_320/sdk/nvt_model.bin` (md5 `0cf09e38af2e20d106ce62eae59ce3c8`)
  - 보드 측정은 사용자가 한다. 덤프는 `v01_symmetric_scratch_bgv3_pat100/board/` 에 둔다.
- **E2b 보드 결과 (덤프 2026-10-15 17:08 수신)**
  - 덤프: `v01_symmetric_scratch_bgv3_pat100_output.json` (md5 `f59c29d8cf56113edd5cec7378bb9b03`, 원래 이름 `261015_BSD-F606_REAR_scratch_bgv3_output.json`). 170,599개, 최소 0.0012
  - stale 검사를 통과했다. 기존 덤프와 E1·C1·E0 덤프 모두와 jaccard 0 이다.
  - 짝 확인에서 자기 ONNX 와 일치율 0.964 / |Δscore| 0.034 로 가장 잘 맞는다. 파인튜닝 계열과는 0.89 대다.
  - customEval(덤프 그대로, Front 방식): ONNX B 0.769 → Board **0.758** (−0.011)
    - 클래스별: bus 0.859→0.843, car 0.762→0.756, moto 0.787→0.783, person 0.723→0.711, truck 0.715→0.697
    - optimal 기준 TP / FP / FN = 23,594 / 4,410 / 10,673 이다. E1 은 23,776 / 4,863 / 10,491 이다.
  - ultralytics 코어(0.01 컷): ONNX 0.818 / 0.610 → Board 0.806 / 0.578 (mAP50 / 50-95)
  - 보드 임계값(optimal): bus 0.348, car 0.365, moto 0.248, person 0.248, truck 0.201 (`runs/analysis/rear_truck_thresholds/E2b_*`)
  - E1 대비 Board −0.005 다. FP 는 적고 FN 은 많다(정밀도 쪽으로 치우침).

### E1 대 E2b 블라인드 판정 (2026-10-16)

- 대상: 주행 영상 8개(`998_input_auto_label/infer_video`)와 FP 케이스 영상 3개를 15프레임 간격으로 본 것이다.
- 검출 기준: 모델별 보드 optimal 임계값
- 엇갈린 객체 1,602건: E2b 만 검출 1,091 / E1 만 검출 370 / 클래스 불일치 141
- 종류별 16건씩 48건을 모델을 가린 채 사용자가 판정했다.
- 페이지·데이터: https://claude.ai/artifact/K9nQiGBhH89HRZJayY1QUz, `runs/analysis/rear_e1_vs_e2b_review/` (`cases_key.json` 이 모델 매핑, `judgments_dump/` 가 판정)

| 종류 | 표본 | 맞음 | 틀림 (없음 / 다른 클래스) | 애매 | 전체 건수로 환산 |
|---|---:|---:|---:|---:|---|
| E2b 만 검출 | 16 | **14 (88%)** | 2 (1 / 1) | 0 | 맞게 더 잡음 약 955 / 잘못 더 잡음 약 136 |
| E1 만 검출 | 16 | 9 (56%) | 6 (4 / 2) | 1 | 맞게 더 잡음 약 208 / 잘못 더 잡음 약 139 |
| 클래스 불일치 | 16 | E2b 맞음 10, E1 맞음 5 | — | 1 | |

- 클래스 불일치 중 car/truck 11건이 있다. 정답은 truck 10, car 1 이었다.
  - E1 이 맞은 5건은 모두 E1=truck 이었고, E2b 가 맞은 6건 중 5건은 E2b=truck 이었다.
  - 두 모델 모두 트럭(주로 1T 픽업·탑차)을 car 로 부르는 경우가 있고, 어느 쪽이 낫다고 보기 어렵다(5 대 6).
  - 나머지 클래스(motorcycle·bus) 4건은 모두 E2b 가 맞았다.
- **배포 임계값 기준으로는 E2b 가 낫다.**
  - E2b 는 E1 이 놓치는 실제 객체를 많이 더 잡는다(약 955). 그 대가로 늘어나는 오검출(약 136)은 E1 이 혼자 내는 오검출(약 139)과 비슷하다.
  - 단 표본이 작다. 14/16 대 9/16 의 Fisher 정확 검정 p = 0.11 이다.
  - E2b 의 car 임계값(0.365)이 E1(0.500)보다 낮은 것도 차이에 기여한다.
- 6,152장 보드 mAP(E1 0.763 / E2b 0.758)와는 순위가 다르다. 사용자는 val GT 를 완전히 신뢰하지 않는다고 했다.
- **2차 판정** (같은 날, 1차 장면 ±60프레임 제외, 시작 프레임 7)
  - 엇갈림: E2b 만 664 / E1 만 203 / 클래스 62
  - 표본: E2b 만 20, E1 만 20, 클래스 8
  - 페이지 https://claude.ai/artifact/9iZVXmp1KggyxwanPTPmnv, `runs/analysis/rear_e1_vs_e2b_review_r2/`
  - 결과: E1 만 검출 16/20 맞음(80%), E2b 만 검출 14/20 맞음(70%, 애매 4), 클래스 E2b 5 / E1 2 / 둘 다 아님 1
- **1·2차 합계**

| 종류 | 표본 | 맞음 | 틀림 (없음 / 다른 클래스) | 애매 | 전체 건수로 환산 (1·2차 합 E1 573 / E2b 1,755) |
|---|---:|---:|---:|---:|---|
| E2b 만 검출 | 36 | 28 (78%) | 4 (3 / 1) | 4 | 맞게 더 잡음 약 1,365 / 잘못 더 잡음 약 195 |
| E1 만 검출 | 36 | 25 (69%) | 9 (6 / 3) | 2 | 맞게 더 잡음 약 398 / 잘못 더 잡음 약 143 |
| 클래스 불일치 | 24 | E2b 15, E1 7 | 둘 다 아님 1 | 1 | car/motorcycle 5:0, car/truck 8:6 으로 E2b |

- 한 모델만 낸 검출의 정확도는 두 모델이 비슷하다(78% 대 69%, Fisher p = 0.59). 1차의 88% 대 56% 차이는 2차에서 재현되지 않았다.
- 차이는 **양**이다. E2b 는 배포 임계값에서 E1 이 놓치는 실제 객체를 약 3배 더 잡는다. 늘어나는 오검출은 약 50건 더 많은 수준이다(15프레임 간격 표본 기준).
- 클래스별로 E1 만 낸 motorcycle 은 9건 중 4건만 맞았다(다른 클래스 3, 없음 1). E1 만 낸 person 은 17건 중 4건이 오검출이었다. E2b 만 낸 person 은 14건 중 13건이 맞았다.

### 주행 영상 8개 보드 결과 (2026-10-16)

- 입력: NAS `998_input_auto_label/infer_video/{v01_release, 261015_BSD-F606A_Rear_v02_test (= E1, 사용자 확인), 261015_BSD-F606_REAR_scratch_bgv3 (= E2b)}/<영상>.json`
  - 보드가 영상을 직접 처리한 덤프다(frame 0~1799, score 하한 0.0012).
- 프레임 대응: 영상을 순서대로 읽은 순번이 덤프 frame 과 맞는다(오프셋 0 에서 v01 일치율 0.96~1.00).
  - cv2 의 `CAP_PROP_POS_FRAMES` 로 건너뛰면 AVI 에서 다른 프레임이 읽혀 일치율이 0.2 로 떨어진다. 비교할 때는 순서대로 읽어야 한다.
- 표시 기준: 모델별 보드 optimal 임계값(v01 은 NAS 릴리스 값)
- 영상: `runs/video_infer/rear_board_998_infer_video_v01_E1_E2b/*_board3.mp4` (v01 | E1 | E2b)

| 모델 (보드) | car | truck | motorcycle | person | bus | 합계 | v01 대비 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v01 release | 30,097 | 3,767 | 11,310 | 31,349 | 629 | 77,152 | |
| E1 ft_bgv3 | 29,414 | 3,650 | 10,127 | 29,636 | 690 | 73,517 | −4.7% |
| E2b scratch_bgv3_pat100 | 30,034 | 3,438 | 10,204 | 30,318 | 673 | 74,667 | −3.2% |

### 오인식 케이스 영상 5개 보드 결과 (2026-10-16)

- 입력: NAS `998_input_auto_label/infer_video/오인식케이스/{v01_release, v02_tf_bgv3 (= E1), v02_scratch_bgv3 (= E2b)}/` (영상 160500·160559·160659·161523·161635)
  - 처음 넣은 JSON 은 골목주행 결과의 복사본(md5 동일)이라 다시 받았다.
- 짝 확인(160659, 순서대로 읽은 600프레임 중 10프레임 간격): v01_release↔v01 0.994, v02_scratch↔E2b 0.986
  - v02_tf 는 v01 0.990 / E1 0.987 로 둘을 가르지 못한다(가까운 모델). 검출 수가 v01 덤프와 달라 다른 bin 이다. 사용자 확인으로 E1 로 둔다.
- 영상: `runs/video_infer/rear_board_fpcase_v01_E1_E2b/*_board3.mp4` (모델별 보드 임계값)

| 모델 (보드) | 검출 프레임 | car | truck | motorcycle | person | bus |
|---|---:|---:|---:|---:|---:|---:|
| v01 release | 7,689 | 30,646 | 1,212 | 361 | 1,691 | 1,210 |
| E1 | 7,404 | 29,589 | 1,180 | **135** | 1,002 | 1,367 |
| E2b | 7,410 | 28,703 | 1,137 | 385 | **882** | 1,342 |

- person 은 두 모델 모두 v01 대비 크게 줄었다(E1 −41%, E2b −48%).
- motorcycle 은 E1 만 크게 줄었다(−63%). E2b 는 v01 과 비슷하다. 160659 영상에서는 E2b 가 v01 보다 많다(208 대 112).

### 3차 판정 — 보드 결과 (2026-10-16)

- 골목주행 8 + 오인식케이스 5 영상에서 E1·E2b **보드 결과**(모델별 임계값)가 엇갈린 객체를 뽑았다.
  - 프레임은 15 간격, 시작 11 이다. 1·2차 ±60프레임은 뺐다.
- 엇갈림 수(1·2차 장면 ±60프레임 제외 후): 골목주행 E2b 만 191 / E1 만 141 / 클래스 27, 오인식케이스 E1 만 318 / E2b 만 258 / 클래스 12
  - 정정: 골목주행 수가 적은 것은 제외 범위 때문이다. 제외 없이 세면 E1 vs E2b 보드 엇갈림은 1,100 으로, ONNX(1,302)와 비슷하다.
- 페이지 https://claude.ai/artifact/XVfj4zNNQqaWU2BYRgiKGo, `runs/analysis/rear_e1_vs_e2b_review_r3/` (`mine_board_cases.py`, `population.json`)

| 영상 묶음 | 종류 | 표본 | 맞음 | 틀림 | 전체 건수로 환산 (맞음 / 틀림) |
|---|---|---:|---:|---:|---|
| 골목주행 | E1 만 | 9 | 6 (67%) | 3 | 94 / 47 |
| 골목주행 | E2b 만 | 11 | 8 (73%) | 3 | 139 / 52 |
| 오인식케이스 | E1 만 | 11 | 9 (82%) | 2 | 260 / 58 |
| 오인식케이스 | E2b 만 | 9 | 8 (89%) | 1 | 229 / 29 |
| 둘 다 | 클래스 불일치 | 8 | E1 4, E2b 4 | | |

- **보드 결과로는 두 모델이 사실상 같다.** 정확도 차이가 6~7%p 이고 표본이 작다.
- 오인식케이스의 엇갈림은 대부분 car 였고 거의 다 맞았다. 문제 클래스(person·motorcycle) 표본은 4건뿐이라 판단 근거가 되지 않는다.
- 1~3차를 종합하면 ONNX(배포 임계값)에서는 E2b 가 실제 객체를 더 잡는 경향이 있었다. 보드 결과 판정에서는 그 경향이 보이지 않는다.
- 모델 쌍별 보드 엇갈림(13개 영상, 15프레임 간격, 제외 없음)

| 묶음 | v01 vs E1 | v01 vs E2b | E1 vs E2b |
|---|---|---|---|
| 골목주행 | 668 (v01 만 430 / E1 만 179 / 클래스 59) | 1,191 | 1,100 |
| 오인식케이스 | 374 (v01 만 243 / E1 만 126 / 클래스 5) | 655 | 588 |

  - E1 은 v01 과 가장 가깝다. 차이는 주로 v01 만 낸 박스다.
  - 오인식케이스에서 v01 만 낸 박스는 person 58 / motorcycle 13 이고, E1 만 낸 박스는 person 7 / motorcycle 1 이다.

### 1T 검증 이미지만으로 본 비교 (2026-10-16)

- 6,152장 중 파일명에 TRUCK1T 가 있는 3,894장(GT 25,108)이다. 학습 val 의 1T 3,894장이 모두 들어 있다(DPI 144 제외분은 전부 5T).
- GT 사본: `_eval_cfg/gt_variants/ipu_gt_truck1t.json`
- 예측은 기존 6,152장 결과에서 1T 만 걸렀다(`gt_variants/preds_truck1t/`). 결과는 `gt_variants/eval_truck1t/` 다.
  - 1T GT 로 ONNX 를 직접 돌리면 이미지 폴더의 5T 이미지에서 KeyError 가 난다.

| 모델 | ONNX A | ONNX B | Board | Board bus / car / moto / person / truck | Board optimal TP / FP / FN |
|---|---:|---:|---:|---|---|
| v01 | 0.755 | 0.744 | 0.736 | 0.842 / 0.712 / 0.771 / 0.673 / 0.680 | 16,420 / 3,721 / 8,688 |
| E1 | 0.764 | 0.754 | **0.748** | 0.856 / 0.718 / 0.786 / 0.680 / 0.699 | 16,538 / 3,551 / 8,570 |
| E2b | 0.762 | 0.749 | 0.740 | 0.850 / 0.716 / 0.776 / 0.692 / 0.667 | 16,243 / 3,036 / 8,865 |

- 4차 판정: 1T 이미지에서 보드 결과가 엇갈린 객체(E1 만 2,723 / E2b 만 2,250 / 클래스 228)에서 48건을 골랐다(43개 클립).
  - 페이지 https://claude.ai/artifact/7fV6gzDLcmTagqKd8ZKJgL, `runs/analysis/rear_e1_vs_e2b_review_r4/`
  - `cases_key.json` 에 참고용 GT 최대 IoU 를 적었다.
  - 판정 기준을 사용자와 정했다: 박스가 그 물체의 박스로 맞으면 맞음, 다른 물체에 더 맞으면 다른 클래스, 크게 벗어나면 애매함.
- **4차 결과**
  - E1 만 검출: 20건 중 맞음 14(70%) / 틀림 3 / 애매 3. 환산하면 맞음 약 1,906 / 틀림 약 408 이다.
  - E2b 만 검출: 20건 중 맞음 15(75%) / 틀림 4 / 애매 1. 환산하면 맞음 약 1,688 / 틀림 약 450 이다.
  - 클래스 불일치: E1 4 / E2b 4
  - **두 모델이 같다.** Fisher 검정 p = 1.0 이다.
- **판정과 GT 대조** (한쪽만 검출, 애매 제외 36건)
  - 판정에서 맞음인 29건 중 **14건은 GT 에 박스가 없다**(GT 누락).
  - 판정에서 틀림인 7건은 GT 에도 없거나 다른 클래스였다.
  - 두 모델이 엇갈리는 객체에서 GT 누락이 많다. 이 GT 로 재는 FP 수와 mAP 는 검출을 더 많이 내는 모델을 불리하게 만든다. 사용자가 val GT 를 신뢰하지 않은 것과 맞는 결과다.

### 릴리스 v02 준비 (2026-10-16)

- 사용자 결정: **E1 을 v02 로 릴리스한다.** bin 은 E1 기본 변환이다. 로컬만 정리하고 커밋은 나중에 한다.
- `configs/releases/BSD_F606/Rear_Truck/v02/`
  - `exp.yaml`: 실험 config 그대로다. 머리말과 `_meta_` 의 status·version 만 바꿨고 학습 키는 같다. 동결한다.
  - `MANIFEST.yaml`: 출처, 해시, 지표, 선정 근거, issues
  - `weight/`: best.pt(`ba7b1cf0…`), last.pt, 학습 곡선, training_log
  - `network/`: `261016_BSD-F606A_Rear_v02.bin`(`25e5ea3c…`), `.onnx`(`ceb76d59…`), gen/sim_config, profile
  - `analysis/`: `v02_thresholds.json`, `ipu_test/v02_output.json`(6,152장 보드 덤프), evaluation.json 4종
- git 이 추적하는 파일은 `exp.yaml`, `MANIFEST.yaml`, `weight/exp.yaml` 이다(.gitignore 규칙).
- NAS `Rear_Truck/v02` 는 아직 올리지 않았다.

### 크기별 Recall (ROI 200,380–1720,1080, 2026-10-16)

- old `tools/debug/roi_recall.py` 를 그대로 썼다. GT 면적의 30% 이상이 ROI 와 겹치면 영역 안이다. 각 모델의 optimal 임계값으로 거른 뒤 IoU 0.5 로 매칭했다. GT 는 6,152장 34,267개이고, 그중 5,305개가 영역 안이다.
- 보드 덤프는 배열을 이어 붙인 원본이다. 합치고 `MOTOR` 를 `motorcycle` 로 바꾼 사본을 따로 만들어 넣었다. 원본은 건드리지 않았다.
- 사용자 보고서 표의 "v01" 값(all 0.886 / 0.632)은 v02_dair Float 결과와 정확히 같다. v01_symmetric 의 값이 아니다.

| all (in / out) | v01_symmetric | v02 |
|---|---|---|
| Float | 0.893 / 0.646 | 0.893 / 0.651 |
| Board | 0.905 / 0.649 | 0.907 / 0.655 |

### 보고서 "인식 예시 (온보드)" (2026-10-16)

- 골목주행 8 + 오인식케이스 5 영상에서 v01 release 와 v02 의 보드 결과(모델별 임계값)를 비교했다. 한 모델만 낸 박스가 10프레임 이상 이어지는 트랙 152개를 뽑고, 눈으로 보고 6장면을 골랐다.
- 고른 장면: 화분 → motorcycle, 마네킹·문 → person, 기둥 → person, 원형 입간판 → motorcycle 오인식 제거; 차량 앞 주차 오토바이를 car 한 박스에서 motorcycle·car 로 분리; 원거리 bus 검출
- 오인식케이스에서 2장면을 더했다: 주차장 벽 점 무늬 → person, 주차장 기둥 → car(큰 박스) 오인식 제거. 이 묶음에서 v01 만 낸 person 은 대부분 실제 원거리 보행자여서 v02 가 놓친 쪽이다.
- 사용자 지정 구간(161523 40초~끝, 161635 처음~15초, `mine_segment.py`, 최소 길이 4)에서 3장면을 더했다: 라바콘 → motorcycle 0.50, 라바콘 → person·motorcycle 0.26, 주차장 경사로 벽 → truck 0.41 오인식 제거. 같은 구간에서 v02 가 놓친 실제 객체(경사로 위 트럭, 원거리 차량 3건)도 있다. v02 만 낸 트랙은 없었다.
- 후보 중 v02 가 놓친 실제 객체(인도 위 주차 오토바이, 버스정류장에 앉은 사람, 원거리 오토바이)도 있다. 보고서에 개선만 싣더라도 이 점은 남겨 둔다.
- `runs/analysis/rear_v02_onboard_examples/` (`mine_tracks.py`, `render_candidates.py`, `render_final.py`, `cand/`, `final/`)

### 260911 FP 케이스 영상 보드 결과 (2026-10-20)

- 입력: NAS `/mnt/NAS_1/new_infer/{<영상>.avi, v01/, v02/}`. 사용자가 올린 보드 덤프이고 폴더 이름대로 v01 / v02 로 본다. 영상은 old `_input/260911_CVD-F410_REAR_FP_case/` 와 md5 가 같다.
- 보드 검출 수(모델별 optimal 임계값)

| 영상 | 모델 | 검출 프레임 | car | truck | motorcycle | person | bus |
|---|---|---:|---:|---:|---:|---:|---:|
| 105025 | v01 | 855 | 2,679 | 200 | 154 | 95 | 0 |
| | v02 | 629 | 2,623 | 18 | 86 | 70 | 0 |
| 132126 | v01 | 91 | 26 | 3 | 6 | 58 | 12 |
| | v02 | 68 | 21 | 18 | 0 | 27 | 3 |
| 132154 | v01 | 1,744 | 10,987 | 137 | 270 | 330 | 10 |
| | v02 | 1,702 | 10,578 | 92 | 206 | 269 | 13 |

- 개선 장면 6개: `runs/analysis/rear_v02_onboard_examples/new_infer/final/` (`new_infer.py`). 경사로 벽·주차장 벽·기둥 → truck, 먼 벽 → motorcycle, 화단 → bus 오인식 제거
- v02 가 놓친 실제 객체: 132154 주차된 트럭 2건(f93, f119). v02 만 낸 오인식: 132126 터널 입구 truck 0.41(f231, 12프레임) — 132126 의 truck 증가(3 → 18)가 이것이다.

### 배경 데이터 NAS 업로드 (2026-10-20)

- `background_rear_v3/train/images` 4,724장(2.4GB)을 NAS_3 `02_datasets/BSD/Rear_Truck/background/images/` 에 복사하고 README.md 를 두었다.
- 확인: 파일 이름과 크기가 모두 같고, 200장 표본의 md5 가 같다. NAS 가 그룹 변경(chgrp)을 막아 소유권은 NAS 기본값이다.
- `_unused_split/` 은 쓰지 않은 분할 기록이라 올리지 않았다.

### NAS 업로드 준비 (2026-10-20)

- `/mnt/NAS_2` 는 읽기 전용 마운트(`/etc/fstab` davfs ro)라 쓰지 못했다. 업로드용 폴더를 `runs/nas_upload/BSD_F606/Rear_Truck/v02/` 에 만들었다.
  - network/·weight/ 는 NAS `Rear_Truck/v01` 과 같은 파일 구성이다. ONNX 는 `weight/weights/261016_best.onnx`, 설정 파일은 `exp.yaml`(v01 의 `mycfg.yaml`).
  - analysis/ 는 사용자 요청으로 NAS `Front_Truck/v02/analysis` 와 같은 구성으로 만들었다(`runs/analysis/rear_truck_v02/`, 출처 `SOURCES.md`).
- 영역별 지표는 ROI 가 없어 원본 폭 1920 을 3등분했다. ONNX edge 0.799 / center 0.744, Board edge 0.768 / center 0.730.

## 남은 일 (2026-10-16)

- 릴리스 v02: NAS 업로드, 커밋, master 병합(사람이 push)

0. 선택: 평가셋 구성 — 한 정차 클립(`KR_20240318_…_LDC2_PGL`)이 9%(548장)를 차지한다. 비중 조정 여부를 결정한다.

1. ~~E0·E1·E2·E2b·E3 평가, E1·E0·C1·E2b 변환·보드 측정~~ (완료)
2. 후보 결정: E1(Board 0.763) 대 E2b(0.758, FP 영상 person 증가 확인 필요)
3. 선택: "배경은 나중에 넣어야 하는가" 확인 — E2b 에서 E1 과 같은 파인튜닝을 돌려 E1 과 비교한다(약 9시간, 미착수).
4. 선택: 보정 이미지를 바꾼 C2 bin(`a4a69fd6…`)의 보드 측정. 기대 효과는 낮다.

## 미검증

- 오검출 개선은 FP 케이스 영상 3개의 검출 수로만 봤다. 줄어든 검출이 모두 오검출이었는지는 영상을 일부만 확인했다. 배경 이미지 자체로 측정하지 않은 것은 사용자 결정이다.
- 보드 후처리 규칙은 old 저장소 `temp/nn_post_process.cpp`(2026-08-10 사본)로 확인했다. 실제 보드 빌드와 같은지, 특히 점수 하한과 Rear 분기는 미확인이다.
- C3 의 16비트 보호가 무시된 원인은 미확인이다.
- 보드 점수 이산화의 원인(cls logit 8비트 양자화)은 추정이다.
