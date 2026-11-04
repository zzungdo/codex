# Phase E — 정식 학습 (b) cmd-femto + (c) cmd-crnn 결과

- **작성**: 2026-05-27 (KST)
- **base commit**: `3a54420` (`fix: D-014 후속 — on-the-fly DataLoader → InfiniteDataLoader 교체`)
- **상위 사이클**: D-014 (ncnvox CMD 흡수) — Phase E
- **목적**: 50ep POC 의 saturate 직전 상태를 200ep 정식 학습으로 plateau 도달 확정

---

## 결과 요약

| 모델 | epochs (early stop) | best ep | best top1 | best val_loss | 학습 시간 | best.pt 크기 | params |
|---|---|---|---|---|---|---|---|
| **cmd-femto** | 103 / 200 | **73** | **1.00000** | 0.00200 | 17.9분 (0.299h) | 0.5 MB | 207K (Conv only) |
| **cmd-crnn**  | 86 / 200  | **56** | **1.00000** | **0.00121** | 14.7분 (0.245h) | 1.1 MB | 554K (Conv + LSTM) |

둘 다 top1=1.0 도달 → 단순 정확도로는 우열 판정 불가. 데이터셋 난이도 한계 (TTS positive + SC v2 negative 가 mel domain 에서 거의 linearly separable) 가 원인.

---

## POC (50ep) vs Phase E (200ep early stop) 비교

| 지표 | femto POC ep50 | femto Phase E best | crnn POC ep50 | crnn Phase E best |
|---|---|---|---|---|
| top1 | 0.99818 | **1.00000** (+0.00182) | 0.99091 | **1.00000** (+0.00909) |
| val_loss | 0.01184 | 0.00200 (5.9× ↓) | 0.02530 | 0.00121 (20.9× ↓) |
| best ep | 50 | 73 | 47 | 56 |

→ 양 모델 모두 POC 대비 향상. **crnn 이 val_loss 측면에서 20× 개선**, femto 도 5.9× 개선. 단순 top1 saturate 외에 confidence 측면에서 추가 학습 효과 있었음.

---

## 학습 hyperparameter (Phase E)

```yaml
epochs: 200
patience: 30          # early stop 발동: femto ep103, crnn ep86
batch: 64
imgsz: 96
workers: 2
optimizer: auto       # SGD (batch≤256 → SGD 자동 선택)
lr0: 0.001            # (관찰: 실제 results.csv LR 은 약 30배 큰 값. 아래 별도 항목 참조)
lrf: 0.01
cos_lr: true          # one_cycle cosine annealing (Phase D 의 linear 대비 변경)
warmup_epochs: 3.0
warmup_bias_lr: 0.1
amp: true
close_mosaic: 10      # InfiniteDataLoader fix 적용 (Phase D 는 0 으로 우회했음)
seed: 0
```

---

## 학습 곡선 핵심 지점

### femto (Conv only, 207K)

| ep | train_loss | top1 | val_loss | lr/pg0 |
|---|---|---|---|---|
| 1 | 1.4159 | 0.07333 ⚠️ | 3.33706 | 0.009902 |
| 5 | 0.3724 | 0.82364 | 0.43084 | 0.029970 |
| 10 | 0.1909 | 0.88788 | 0.30932 | 0.029850 |
| 20 | 0.1046 | 0.98061 | 0.05054 | 0.029340 |
| 50 | 0.0428 | 0.99394 | 0.02269 | 0.025810 |
| **73 (best)** | **0.0331** | **1.00000** | **0.00200** | 0.021470 |
| 103 (end) | 0.0218 | 0.99818 | 0.00394 | 0.014680 |

⚠️ ep1 anomaly: 작은 femto 모델이 LR peak (~0.03) 의 첫 step 에서 일시적 학습 충격. ep2 부터 정상 회복 (top1=0.685).

### crnn (Conv + LSTM, 554K)

| ep | train_loss | top1 | val_loss | lr/pg0 |
|---|---|---|---|---|
| 1 | 1.5363 | 0.66909 | 1.37023 | 0.009902 |
| 5 | 0.6814 | 0.76303 | 0.54866 | 0.029970 |
| 10 | 0.3829 | 0.85091 | 0.30825 | 0.029850 |
| 20 | 0.1319 | 0.95030 | 0.11768 | 0.029340 |
| 50 | 0.0368 | 0.99515 | 0.01486 | 0.025810 |
| **56 (best)** | **0.0290** | **1.00000** | **0.00121** | 0.024790 |
| 86 (end) | 0.0176 | 0.99879 | 0.00215 | 0.018620 |

LSTM 의 시계열 모델링이 ep1 부터 정상 수렴 (femto 의 ep1 anomaly 없음). 초기 epoch 의 robustness 가 더 좋음.

---

## 중간 발견 — LR scaling 관찰

`args.yaml` 에 `lr0=0.001` 명시되어 있으나 `results.csv` 의 실제 LR 은 약 30배 큰 값 (peak ~0.03) 으로 적용됨.

| 시점 | POC (cos_lr=False) | Phase E (cos_lr=True) | 비율 |
|---|---|---|---|
| ep1 lr | 0.000274 | 0.009902 | 36× |
| ep5 lr (peak 근처) | 0.000797 | 0.029970 | 38× |
| ep50 lr | 2.48e-5 | 0.025810 | 1041× |

**Phase E 의 cos_lr=True 도입 + ultralytics classify task 의 default scaling 조합**이 트리거. POC 의 cos_lr=False (linear) 에서는 lr0=0.001 이 정상 적용됐음.

- 학습 결과 측면: 더 큰 LR 가 **plateau 도달 가속**에 기여 — POC 50ep saturate 직전 → Phase E ep~56~73 에서 top1=1.0 도달
- 부작용: ep1 anomaly (femto top1=0.07) — 작은 모델이 큰 첫 step 에 일시 충격
- **별도 조사 사이클로 분리** (Phase F 후): ultralytics 의 LR scaling 소스 위치 + lr0 의도 적용 가능 여부 확인

---

## 데이터셋 난이도 분석 (top1=1.0 saturate 원인)

| 요인 | 영향 |
|---|---|
| Positive 7 commands 전부 TTS 합성 | 동일 합성 엔진 → 발화 다양성 거의 0 |
| Commands 가 multi-syllable 영어 구문 | "take_video", "turn_up_volume" 등 음운 패턴 명확 |
| Background 가 SC v2 1-syllable 영어 단어 | "yes/no/up" 등 짧고 단음절 — commands 와 mel 패턴 충돌 거의 없음 |
| bg_ratio 2.0, neg_cap 239/단어 | calibration 안정, 5,488 negative 충분 |

→ 8-class 가 mel domain 에서 거의 linearly separable. femto (207K) 조차 saturate.

**진짜 우열 판정을 위한 다음 옵션**:
1. **NPU sim latency 비교** (Phase F — 다음 사이클) — LSTM 의 quantization/추론 비용
2. **Real 마이크 녹음 평가셋** — TTS ≠ 실 발화 도메인 갭 측정
3. **SNR 낮추기 / hard negative 추가** — robustness 차이 드러내기

---

## Phase F 진입 준비

| 모델 | best.pt 경로 | NPU 변환 우선순위 | 비고 |
|---|---|---|---|
| cmd-femto | `runs/classify/cmd_femto_full/weights/best.pt` | 1순위 | Conv-only → NPU 친화적, 0.5MB |
| cmd-crnn | `runs/classify/cmd_crnn_full/weights/best.pt` | 2순위 (검증용) | LSTM 정적 형상 export 호환성 검증 필요 |

Phase F 핵심 검증: **CRNN 의 LSTM 이 ultralytics export → onnx2novaonnx → NPU 컴파일 chain 에서 정상 동작하는가**. 호환 안 되면 femto 단독 production.

---

## 완료 기준 (Phase E plan 대비)

| 기준 | 결과 |
|---|---|
| femto best.pt 존재 + top1 ≥ 0.998 | ✅ top1=1.00000 (POC 0.99818 회귀 없음) |
| crnn best.pt 존재 + top1 ≥ 0.991 | ✅ top1=1.00000 (POC 0.99091 대비 +0.00909) |
| 200ep (또는 early stop) 정상 종료 | ✅ femto ep103, crnn ep86 (양쪽 EarlyStopping 정상 발동) |
| 결과 비교 doc 작성 | ✅ 본 문서 |
| @verifier ✅ | (대기) |
