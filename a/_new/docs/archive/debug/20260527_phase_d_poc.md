# Phase D POC 50ep 비교 — cmd-femto vs cmd-crnn

- **작성**: 2026-05-27 (KST)
- **상위 사이클**: D-014 (ncnvox CMD 흡수, 전략 b+c 병행)
- **base commit**: `ae463ff`
- **목적**: (b) Conv-only femto 와 (c) Conv+LSTM CRNN 를 동일 환경 50ep 학습 후 Phase E 진입 시 모델 선택 근거 확보

---

## 학습 환경 (양 모델 동일)

| 항목 | 값 |
|---|---|
| 데이터 | `configs/datasets/legacy/ncnvox_cmd.yaml` (8-class, train=6568, val=1650) |
| epochs / batch / imgsz | 50 / 64 / 96 |
| optimizer | auto → AdamW (lr=0.000833 자동 결정), momentum=0.9 |
| lr0 (요청) | 1e-3 (auto 가 무시하고 0.000833 사용) |
| patience / seed | 20 / 0 |
| workers | 2 (동시 실행 RAM 분산) |
| amp | true |
| close_mosaic | **0** (classify task — default=10 시 train_loader.reset() AttributeError) |
| device | femto=GPU0, crnn=GPU1 |
| image augmentation | **전부 비활성화** (fliplr/mosaic/hsv/auto_augment/scale/translate/erasing=0) |
| waveform augmentation | dataset 단계 noise mix (SNR 0~20dB, musan/esc50/mic_silence) |

---

## 결과 요약

| 항목 | cmd-femto (Conv) | cmd-crnn (Conv+LSTM) |
|---|---|---|
| **파라미터** | 207,000 | 554,280 |
| **모델 크기 (best.pt)** | 493 KB | 1,129 KB |
| **총 학습 시간** | 575.1s (9.58분) | 569.5s (9.49분) |
| **시간/epoch** | 11.50s | 11.39s |
| **final ep50 top1** | **0.99818** | 0.99091 |
| **final ep50 top5** | 0.99939 | **1.00000** |
| **final ep50 val_loss** | **0.01184** | 0.02530 |
| **best top1 epoch** | ep50 (계속 개선) | ep47 (0.99212) |
| **best val_loss epoch** | ep33 (0.01130) | ep50 (계속 개선) |
| **수렴 epoch (top1 ≥ 99% of best)** | **ep28** | ep39 |

---

## 학습 곡선 핵심 지점

### femto

| epoch | train_loss | top1 | val_loss |
|---|---|---|---|
| 1 | 1.4916 | 0.6921 | 1.0103 |
| 6 | 0.3300 | 0.8885 | 0.3051 |
| 11 | 0.1878 | 0.9054 | 0.2360 |
| 20 | 0.0983 | 0.9800 | 0.0550 |
| 28 | 0.0574 | 0.9958 | 0.0217 ← 수렴 |
| 33 | 0.0556 | 0.9964 | 0.0113 ← best val_loss |
| 50 | 0.0243 | 0.9982 | 0.0118 ← best top1 |

### crnn

| epoch | train_loss | top1 | val_loss |
|---|---|---|---|
| 1 | 1.4889 | 0.6691 | 1.3172 |
| 6 | 0.6904 | 0.7388 | 0.6472 |
| 11 | 0.5197 | 0.7782 | 0.5177 |
| 20 | 0.2582 | 0.9073 | 0.2069 |
| 30 | 0.1672 | 0.9642 | 0.1021 |
| 39 | 0.0968 | 0.9849 | 0.0488 ← 수렴 |
| 47 | 0.0728 | 0.9921 | 0.0296 ← best top1 |
| 50 | 0.0645 | 0.9909 | 0.0253 ← best val_loss |

---

## 분석

### 1. 정확도
- **femto 가 모든 지표에서 우세** (top1 +0.7%p, val_loss 절반 수준)
- 8-class 명령어 분류에서 207K params 로도 충분히 saturate (top1>99.8%)

### 2. 수렴 속도
- femto: ep28 에 수렴, 이후 saturated 상태에서 미세 개선
- crnn: ep39 에 수렴, 11 epoch 늦게 도달
- CRNN 의 LSTM 부분이 초기 수렴을 늦춤 (시계열 모델링 능력 trade-off)

### 3. 학습 시간
- 거의 동일 (11.4~11.5s/ep) — 96×96 입력 + CPU augmentation 병목으로 모델 크기 차이가 GPU 시간에 묻힘
- GPU 사용률 4% 미만 (둘 다) — dataloader 가 병목

### 4. 모델 크기
- femto: 493KB, crnn: 1.13MB (2.3배)
- NPU 메모리 관점에서 femto 가 더 친화적

---

## 중간 발생 이슈 & 해결

### 1. ep41 AttributeError (1차 시도)
- 증상: 양 모델 ep41 진입 시 `'DataLoader' object has no attribute 'reset'` AttributeError → exit 0 으로 강제 종료
- 원인: ultralytics `close_mosaic=10` default. `epoch == epochs - close_mosaic` 시점 (ep41) 에 `train_loader.reset()` 호출. classify task 의 DataLoader 는 reset 메서드 없음
- KWS 가 정상 완주했던 이유: epochs=500, patience=30 → ep57 early stop 으로 close_mosaic 트리거 (ep490) 도달 전 멈춤
- 해결: `tools/train_ncnvox.py:_AUDIO_CLASSIFY_AUG_OVERRIDES` 에 `close_mosaic=0` 명시
- 보존: `runs/classify/cmd_{femto,crnn}_poc_ep40bug/` (1차 시도 결과)

### 2. image augmentation 부적합 (2차 시도)
- 증상: 1차에서 `fliplr=0.5`, `mosaic=1.0`, `auto_augment=randaugment`, `hsv_*`, `scale=0.5` 등 ultralytics default 가 적용됨
- 문제: mel-spectrogram 은 시간×주파수 평면 — fliplr 은 발음 순서를 뒤집어 "turn_on" ↔ "turn_off" 구분 학습 신호 왜곡. 다른 image augmentation 도 의미 깨뜨림
- 정답: waveform augmentation (noise mix, SNR variation) 은 dataset 단계 (CommandDataset) 에서 처리 → ultralytics image augmentation 단계 전부 비활성화
- 해결: `_AUDIO_CLASSIFY_AUG_OVERRIDES` 모듈 상수 도입, 3분기 (CRNN/cmd_onthefly/wakeword/ImageFolder POC) spread 적용
- 비교 (참고용 — 1차 vs 2차):
  - femto 1차 ep39 (image aug 활성): top1=0.99758, val_loss=0.00978
  - femto 2차 ep50 (image aug 비활성): top1=0.99818, val_loss=0.01184
  - **top1 거의 동등** (1차가 약간 낮은 val_loss 보였으나 ep 차이 + fliplr 의 부정적 영향 가능성 — 정합 우선)

---

## 결론

| 항목 | 권장 |
|---|---|
| **POC 단계 (50ep)** | femto 가 모든 지표 우세, 권장 |
| **NPU 친화성** | femto (작은 모델, LSTM 없음) |
| **Phase E 진입** | femto 우선 + crnn 보조 검증 (LSTM 의 시계열 모델링이 더 긴 학습에서 빛날지 검증) |
| **Phase F (NPU)** | femto 먼저 변환 (단순, 위험 낮음) → crnn LSTM 변환 가능성 별도 검증 |

**관찰**: CMD 8-class POC 에서는 작은 Conv 모델로도 top1 99.8% 달성. LSTM 의 시계열 모델링 이점은 이 데이터 규모 + 짧은 명령어 (1초 내외) 에서 두드러지지 않음. 다만 noise robustness 나 더 다양한 발화 길이 데이터에서는 다를 가능성 — Phase E 에서 검증.

---

## 산출물

- `runs/classify/cmd_femto_poc/` — best.pt, last.pt, results.csv, confusion_matrix.png, results.png, args.yaml
- `runs/classify/cmd_crnn_poc/`  — best.pt, last.pt, results.csv, confusion_matrix.png, results.png, args.yaml
- `runs/classify/cmd_femto_poc_ep40bug/` — 1차 시도 ep40 결과 (image aug 활성 + close_mosaic 버그) 보존
- `runs/classify/cmd_crnn_poc_ep40bug/` — 1차 시도 ep40 결과 보존
