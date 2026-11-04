# v01_symmetric 저손실 원인 — 최종 결론 정리 (2026-09-02 갱신)

- **작성**: 2026-08-31 (KST), **갱신**: 2026-09-02 (KST, §24 trainer-axis ablation 결과 반영)
- **원본 조사 문서(시간순 실험 기록, 전체 §1~§24 그대로 보존)**:
  `docs/debug/20260824_symmetric_low_quant_loss_rootcause_compare.md`
- **이 문서의 목적**: 원본 문서는 조사 순서대로 누적돼 초기 가설(반증됨)과 최종 결론이
  섞여 있다. 이 문서는 **시간순이 아니라 최신 결론 중심**으로 재구성한 것이며, 새
  실험/재학습/재컴파일/재평가는 하지 않았다 — 전부 원본 문서(및 그 문서가 인용하는
  `docs/debug/20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`)에 이미
  기록된 실측값만 재구성했다.
- **2026-09-02 갱신 요약**: 원본 문서에 §24(trainer-axis ablation, `v02_dair`)가
  추가되면서 이 요약 문서 전체를 그 내용에 맞춰 갱신했다. 핵심 추가: optimizer
  ablation(실험①/②, symmetric 트레이너 내부)에 이어 **trainer(symmetric vs dair)
  축의 두 번째 독립 ablation(`v02_dair` = dair 트레이너 + v01_symmetric의 SGD
  하이퍼파라미터)**이 PT/NVT/Float/Board **4개 레벨 전부에서 optimizer 가설을
  재확인**했다 — board 결과가 v01_symmetric과 사실상 동급(-0.70% vs -0.62%,
  std+ultra mAP50-95). 또한 `v01_dair`의 BN 통계 공백이 채워졌고(5개 이상치+Inf1,
  AdamW 그룹 패턴), §4.2의 "실험①이 baseline과 동등하거나 더 좋다"는 서술을
  Sigmoid cosine 기준으로는 부정확했음을 바로잡았다(3개 중 2개만 더 좋음).

---

## 1. Executive Summary

**왜 `v01_symmetric`의 NVT/Board quantization loss가 작았는가?**
학습에 쓴 optimizer가 **SGD**였기 때문이다. `v01_symmetric_matched`/`v02`/`v01_dair`는
전부 **AdamW**로 학습됐고, 이 차이가 BatchNorm 통계 불안정성 → DFL pre-softmax logit의
넓은 dynamic range → NVT quantization 오차 확대 → Board mAP 저하로 이어지는 사슬의
시작점이다.

**현재 가장 강하게 확인된 원인**: **optimizer(SGD vs AdamW)**. 이 결론은 **서로 다른
축을 통제한 두 개의 독립 ablation**으로 확인됐다:
1. **optimizer-axis** (`lr0=0.001`/`cos_lr=true`/`batch=48`/`cls=1.5`를 고정하고
   optimizer만 바꿈, 실험① SGD / 실험② AdamW, 둘 다 symmetric 트레이너)
2. **trainer-axis** (v01_symmetric의 optimizer=SGD/lr0=0.01/cos_lr=true/batch=48/
   cls=1.5를 그대로 두고 trainer만 symmetric→dair로 바꿈, `v02_dair`, dair 고유의
   dual-forward consistency loss/증강은 그대로 유지)

두 ablation 모두에서 optimizer가 SGD인 쪽은 BN 통계·NVT simulator cosine·Float
mAP·Board mAP **4개 레벨 전부**에서 건강했고, AdamW인 쪽만 불안정했다 —
"symmetric 트레이너 안에서만 그런 것"이라는 대안 설명은 `v02_dair`(dair 트레이너 +
SGD)가 4개 레벨 전부 건강하게 나오면서 배제됐다.

**어느 수준까지 인과관계가 검증됐는가?**
- **실측(관찰 사실)**: optimizer만 다른 체크포인트 쌍(실험①/②)과 trainer만 다른
  체크포인트(`v02_dair`) 양쪽에서 BN 이상치 개수/Inf 여부, NVT simulator cosine,
  Float 6,152장 mAP, Board 6,152장 실측 mAP가 전부 일관되게 갈렸다 — 이 네 층위의
  **상관관계는 두 개의 독립 실험으로 직접 실측 확인**됐다.
- **아직 추정인 부분**: "AdamW가 왜 BN 통계를 불안정하게 만드는가"(momentum/adaptive
  update의 내부 메커니즘)는 조사하지 않았다. 즉 **optimizer→결과의 상관관계는 강한
  실측 증거가 있지만, 그 내부 메커니즘은 미확정**이다.

**아직 모르는 것은 무엇인가?** optimizer가 BN 불안정성을 만드는 메커니즘, 그 BN
불안정성이 pre-softmax dynamic range를 직접 유발하는 경로, pycocotools/ultralytics
evaluator의 매칭 알고리즘 자체 차이(원인과는 별개 축, §8 참조).

**중요한 해석 범위 제한**: 이 결과를 **"AdamW는 quantization에 일반적으로 나쁘다"로
확대 해석하지 않는다.** 정확한 서술은: **이번 데이터셋/학습 조건(BSD_F606
Rear_Truck, symmetric 트레이너, 이 하이퍼파라미터 조합)에서 AdamW가 SGD보다 불안정한
학습 및 quantization robustness 저하를 만들었다**는 것이다. 실험②는 Float mAP
자체도 이미 저하돼 있었으므로(§3.3), 이는 "학습 자체의 불안정성"과 "그 위에 더해진
quantization 손실"이 결합된 결과다.

---

## 2. 최종 비교 대상 / 실험 설계

| Model | trainer | optimizer | lr0 | cos_lr | batch | cls | 목적/역할 |
|---|---|---|---:|---|---:|---:|---|
| `v01_symmetric` | Symmetric | **SGD** | 0.01 | true | 48 | 1.5 | 원 baseline — quantization loss가 가장 작은 모델 |
| `v01_dair` | DAIR(dual-forward) | AdamW | 0.001 | false | 32 | 0.5 | 손실이 큰 비교군 1 (원래 trainer 계열) |
| `v02` | DAIR(dual-forward) | AdamW | 0.001 | false | 32 | 0.5 | 손실이 큰 비교군 2 (v01_dair와 유사 계보) |
| `v01_symmetric_matched` | Symmetric | AdamW | 0.001 | false | 32 | 0.5 | "matched"라는 이름과 달리 실제로는 v01_dair의 하이퍼파라미터를 그대로 복사한 것 — v01_symmetric과 맞춘 것이 아님 |
| **실험① `v02_symmetric_sgd_lr0001`** | Symmetric | **SGD** | **0.001** | **true** | **48** | **1.5** | v01_symmetric 대비 lr0만 0.01→0.001로 낮춘 것(optimizer는 SGD 유지) |
| **실험② `v02_symmetric_adamw_lr0001`** | Symmetric | **AdamW** | **0.001** | **true** | **48** | **1.5** | 실험① 대비 **optimizer만** SGD→AdamW로 바꾼 것 — 나머지 전부 동일 |
| **`v02_dair`** | **DAIR(dual-forward)** | **SGD** | **0.01** | **true** | **48** | **1.5** | v01_symmetric의 optimizer/lr0/lrf/cos_lr/batch/cls를 그대로 두고 **trainer만** symmetric→dair로 바꾼 것(dair_params/증강은 v01_dair 그대로 유지) |

**실험①↔②가 optimizer-axis 핵심 증거다**: `lr0=0.001 / cos_lr=true / batch=48 /
cls=1.5`를 모두 고정한 상태에서 **optimizer 하나만 SGD→AdamW로 바꾼 순수
ablation**이며, "optimizer가 원인" 결론의 1차 근거다. baseline(`v01_symmetric`)
↔실험①은 "lr0 크기 단독 효과"를 보기 위한 것으로, 이 비교에서 lr0 크기 자체는
원인이 아님이 확인됐다(§6).

**`v01_symmetric`↔`v02_dair`가 trainer-axis 증거다**: optimizer/lr0/lrf/cos_lr/
batch/cls를 전부 동일하게 맞추고 **trainer만** 바꿨다(dair 고유의 dual-forward
consistency loss + TensorSpeakerOverlayAug 증강은 켠 채로 유지 — 순수 단일 변수
ablation은 아니고 "trainer+증강" 두 축이 같이 바뀌지만, 목적 자체가 "실제 쓰는
dair가 SGD로도 안정적인가"이므로 의도된 설계다). 이 비교로 "symmetric 트레이너
안에서만 optimizer 효과가 있다"는 대안 설명이 배제됐다(§4, §6).

---

## 3. 최종 성능 결과

### 3.1 Quantization loss (mAP50-95, Board vs Float, 4-combo)

pycocotools/ultralytics × dual-threshold NMS/표준 단일-IoU NMS 2×2, GT 6,152장, 전
모델 conf=0.001 golden 기준 동일 조건(evaluator/NMS 축 자체의 검증은 §8 참조).

| Model | Trainer | Optimizer | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---|---|---:|---:|---:|---:|
| **v01_symmetric** | Symmetric | **SGD** | -7.30% | -1.96% | -6.63% | -0.62% |
| **`v02_dair`** | **DAIR** | **SGD** | -4.91% | -2.36% | -3.82% | **-0.70%** |
| **실험① sgd_lr0001** | Symmetric | **SGD** | -7.89% | -5.91% | -7.20% | -4.59% |
| v01_dair | DAIR | AdamW | -21.01% | -16.32% | -19.66% | -14.62% |
| v02 | DAIR | AdamW | -19.14% | -14.80% | -18.08% | -13.44% |
| v01_symmetric_matched | Symmetric | AdamW | -28.42% | -27.00% | -27.36% | -25.62% |
| **실험② adamw_lr0001** | Symmetric | **AdamW** | **-40.29%** | **-37.76%** | **-38.96%** | **-36.07%** |

**SGD 계열(한 자릿수 손실, -0.7~-7.9%) vs AdamW 계열(두 자릿수 손실, -13.4~-40.3%)로
어느 evaluator/NMS 조합을 봐도 명확히 갈린다.** `v02_dair`(trainer=DAIR인데도
SGD)는 std+ultra 기준 **v01_symmetric과 사실상 동급(-0.70% vs -0.62%)**이고
실험①(-4.59%)보다도 좋다 — trainer가 달라도 optimizer=SGD면 이 그룹에 속한다.
AdamW 계열 내에서도 손실 크기가 `v02(-13.4~-19.1%) < v01_dair(-14.6~-21.0%) <
matched(-25.6~-28.4%) < 실험②(-36.1~-40.3%)` 순으로 벌어진다 — 실험②가 이 조사
전체에서 최악.

### 3.2 참고: mAP50 기준 (std+ultra 강조)

같은 예측 결과를 mAP50 기준으로 다시 본 값(재추론 없음, 기존 예측 재계산). IoU 기준이
느슨해질수록 손실 %의 절댓값은 작아지지만, **SGD/AdamW 그룹 분리와 단조 순위는
mAP50-95와 동일하게 유지된다**.

| Model | Optimizer | **std+ultra (mAP50)** |
|---|---|---:|
| v01_symmetric | SGD | **+0.41%** |
| `v02_dair` | SGD | **-1.22%** |
| 실험① sgd_lr0001 | SGD | **-1.27%** |
| v01_dair | AdamW | **-4.66%** |
| v02 | AdamW | **-6.24%** |
| v01_symmetric_matched | AdamW | **-9.29%** |
| 실험② adamw_lr0001 | AdamW | **-15.52%** |

### 3.3 Float mAP50-95 자체 비교 (quantization 이전 단계)

| Model | Float mAP50-95 (dual+pycoco) |
|---|---:|
| v01_symmetric | 0.562 |
| v01_dair | 0.533 |
| v02 | 0.538 |
| v01_symmetric_matched | 0.549 |
| 실험① sgd_lr0001 | 0.532 |
| `v02_dair` | 0.530 |
| **실험② adamw_lr0001** | **0.412** |

**실험②만 Float 자체가 다른 모든 모델보다 뚜렷이 낮다.** 나머지 6개 모델은 전부
0.53~0.56대로 통계적으로 비슷한 반면, 실험②는 0.412 — quantization 단계 이전에
이미 학습 결과 자체가 저하돼 있었다는 뜻이다. 따라서 **실험②의 board 손실 %를
"quantization 단계에서만 나빠졌다"고 서술하지 않는다** — 학습(float) 단계의
저하와 quantization 단계의 추가 손실이 합쳐진 결과다(§4.3, §4.4).

---

## 4. optimizer 원인을 지지하는 증거 (인과 단계별)

### 4.1 학습/PT — BatchNorm 통계

`torch.load`로 57개 BN 레이어 전체의 `running_var`를 직접 스캔한 결과.

| Model | Trainer | Optimizer | 이상치 레이어 수(running_var max>500) | Inf 포함 레이어 수 |
|---|---|---|---:|---:|
| v01_symmetric | Symmetric | SGD | 1 | 0 |
| **실험① (SGD, lr0=0.001)** | Symmetric | **SGD** | **1** | **0** |
| **`v02_dair`** | **DAIR** | **SGD** | **2** | **0** |
| v01_dair | DAIR | AdamW | 5 | 1 |
| v02 | DAIR | AdamW | 6 | 1 |
| v01_symmetric_matched | Symmetric | AdamW | 7 | 0 |
| **실험② (AdamW, lr0=0.001)** | Symmetric | **AdamW** | **10** | **2** |

`lr0/cos_lr/batch/cls`를 전부 고정한 실험①↔②에서 **optimizer 하나만 바꿨는데도
이상치 레이어 수(1→10)와 Inf 채널(0→2)이 그대로 재현/악화**됐다 — 이것이 이 조사에서
가장 핵심적인 단일 증거다. **trainer만 symmetric→dair로 바꾼 `v02_dair`는 SGD를
유지한 채로 이상치 2개/Inf 0개로 SGD 그룹에 남았다** — trainer 축으로도 같은
패턴이 재현됨. `v01_dair`(이상치 5개+Inf 1개)도 AdamW 그룹에 정확히 속한다.
흥미롭게도 `v01_dair`와 `v02`의 `model.0.bn.running_var` Inf는 **정확히 같은
채널 인덱스(4)**에서 발생했다 — 우연이 아니라 이 아키텍처/데이터 조합에서
재현 가능한 특정 실패 지점일 가능성을 시사한다(메커니즘 자체는 미확정, §7).

### 4.2 NVT simulator cosine (Softmax(dfl) / Sigmoid / dfl_conv, 3개 protected tensor)

동일 테스트 이미지, 동일 NVT compile 조건(calibration/gen_config 전부 byte-identical,
§6에서 배제 확인)에서 측정.

| Model | Trainer | Optimizer | Softmax(dfl) | Sigmoid | dfl_conv |
|---|---|---|---:|---:|---:|
| **실험① (SGD)** | Symmetric | **SGD** | **0.9955** | 0.9779 | **0.9977** |
| **`v02_dair`** | **DAIR** | **SGD** | **0.9857** | **0.9876** | **0.9945** |
| v01_symmetric | Symmetric | SGD | 0.9850 | 0.9824 | 0.9933 |
| v01_symmetric_matched | Symmetric | AdamW | 0.9126 | 0.9473 | 0.9490 |
| v02 | DAIR | AdamW | 0.9058 | 0.9236 | 0.9664 |
| **실험② (AdamW)** | Symmetric | **AdamW** | **0.9170** | **0.7969** | 0.9650 |

**실험②의 Sigmoid cosine(0.7969)은 이 조사에서 측정된 모든 Rear_Truck 모델 중 최저치**
(기존 최저였던 v02의 0.9236보다도 훨씬 나쁨). **실험①은 baseline 대비 3개 중 2개
(Softmax/dfl_conv)만 더 좋고 Sigmoid는 오히려 근소하게 나쁘다**(0.9779 vs 0.9824,
-0.0045)— "동등하거나 더 좋다"는 이전 서술은 부정확했다(2026-09-02 정정). 이
차이는 AdamW 그룹과의 격차(0.10~0.19)에 비하면 미미해서 "SGD 그룹 안에서는
거의 동급"이라는 결론 자체는 유지된다. `v02_dair`는 3개 지표 전부 0.98~0.99대로
SGD 그룹에 뚜렷이 속한다. (v01_dair는 이 3-tensor simulator cosine 비교에
포함되지 않았다 — 원본 조사가 이 분석을 symmetric/matched/v02/실험①/②/`v02_dair`에
대해서만 수행함.)

**중요한 방법론적 주의사항**: 이 cosine은 **딱 1장짜리 테스트 이미지**로 측정한
값이다. 실험①의 board 실측(§4.4)이 baseline보다 나빴던 것에서 보듯, 1장 기준
cosine이 baseline과 비슷/우수하다고 해서 6,152장 전체의 board mAP까지 그렇다는
보장은 없다 — 심하게 나쁜 값(예: 실험②의 0.7969)은 신뢰할 수 있는 경고 신호지만,
이미 건강한 범위(0.98대) 안에서의 미세한 우열은 board 실측 없이 판단할 수 없다
(2026-09-02 확인, `v02_dair` 사례에서는 우연히 board까지 일치했다 — §4.4 참조).

### 4.3 Float (6,152장)

§3.3 참조 — 실험②만 Float mAP50-95 자체가 0.412로 저하. 나머지 6개 모델(`v02_dair`
포함)은 0.53~0.56대로 통계적으로 비슷하다.

### 4.4 Board (6,152장 실측)

§3.1 참조 — 실험①(SGD) board 손실 -4.6~-7.9%(baseline과 같은 그룹), **`v02_dair`
(trainer=DAIR, optimizer=SGD) board 손실 -0.70~-4.91%(baseline과 사실상 동급, 실험①
보다도 좋음)**, 실험②(AdamW) board 손실 -36.1~-40.3%(조사 전체 최악).

**`v02_dair`는 PT/NVT/Float/Board 4개 레벨 전부가 일관되게 SGD 그룹에 속한 사례다** —
실험①이 보여준 "PT/NVT는 건강한데 board는 baseline보다 나쁠 수 있다"는 괴리
(§4.2 방법론적 주의사항)가 `v02_dair`에서는 재현되지 않았다. 이는 그 괴리가
"PT/NVT 신호는 항상 board와 어긋난다"는 뜻이 아니라, **레벨 간 불일치 가능성이
있으니 board 없이 최종 판단할 수 없다는 뜻**임을 보여준다 — 어떤 경우엔 일치하고
(v02_dair), 어떤 경우엔 어긋난다(실험①).

### 인과 사슬 — 실측 vs 추정 구분

```
optimizer 변경(SGD→AdamW)          [실측: 유일하게 통제된 변수]
    ↓
BN running_var 이상치/Inf 증가      [실측: §4.1, torch.load 직접 확인]
    ↓
pre-softmax DFL logit dynamic range 확대   [실측이지만 실험①/②에는 미적용 —
                                             symmetric/matched/v02 3모델에서만
                                             §18 분해로 확인(std 3.15 vs 15.2~16.8)]
    ↓
NVT simulator cosine 저하           [실측: §4.2]
    ↓
Board mAP 저하                     [실측: §4.4]
```

**"BN 이상치 → pre-softmax dynamic range 확대"라는 화살표 자체는 실험①/②에서
직접 재현 검증되지 않았다** — 이 두 모델에는 §18 방식의 pre-softmax 분해가 아직
적용되지 않았다(§9의 선택적 후속 항목). 나머지 화살표(optimizer→BN, BN 존재와
무관하게 optimizer→NVT cosine→Board)는 실험①/②로 직접 재현됐다.

---

## 5. 확인된 중간 현상 (원인이 아니라 "관측된 현상")

원인 자체와, 원인으로 인해 관측되는 중간 현상을 구분한다.

| 현상 | 확인된 사실 | 해석 |
|---|---|---|
| BN `running_var` 이상치/Inf | symmetric 1개, matched 7개, v02 6개+Inf1, v01_dair 5개+Inf1, 실험① 1개, 실험② 10개+Inf2, `v02_dair` 2개+Inf0(전부 `torch.load` 직접 확인) | optimizer와 강하게 연동(trainer와는 무관 — `v02_dair`가 반증)되지만, 이것 자체가 root cause임이 직접 증명되지는 않음(§6 참조) |
| pre-softmax DFL logit dynamic range | symmetric std=3.15, matched std=15.24, v02 std=16.81(golden 실측, §18) | BN 통계 이상치가 최종 DFL 출력 지점에 누적된 형태로 보이나, "BN→dynamic range" 인과 자체는 직접 검증 안 됨 |
| Detect head(box+cls 양쪽) Float→NVT 오차 증가 | 171-layer `layer_diff.txt` 비교, idx122~211 구간에서 matched/v02가 symmetric보다 일관되게 나쁨(§17) | cv2(box)뿐 아니라 cv3(cls)에서도 동등하게 나타남 — "box branch만의 문제"는 아님 |
| NVT simulator cosine 저하 | §4.2 표 | pre-softmax dynamic range 차이의 결과로 보임(같은 compile 조건에서 나온 값이므로 compile 설정이 원인일 수 없음) |
| Board mAP 저하 | §3.1/§4.4 | 위 모든 중간 현상이 최종적으로 드러나는 지점 |

---

## 6. 배제되거나 약화된 가설

| 후보 | 현재 판단 | 근거 |
|---|---|---|
| ONNX export 방식/그래프 구조 | **배제** | 세 모델(symmetric/matched/v02) 그래프 구조 완전 동일(node 235개, shape, opset 13 전부 일치) |
| NVT compile 설정/calibration 데이터 | **배제** | `gen_config.txt` model_dir 외 byte-identical, calibration 940장 동일 symlink, `spec_demarc_tensors` 동일 |
| lr0 크기 단독 | **배제** | 실험①(SGD, lr0=0.001)이 v01_symmetric(SGD, lr0=0.01)과 같은 "건강한" 결과를 유지 — lr0를 10분의 1로 낮춰도 문제 재현 안 됨 |
| batch size | **배제** | 실험①/②가 batch=48로 동일한데도 결과가 갈림 |
| `cls` loss weight(1.5 vs 0.5) 단독 | **약화** | 실험①/② 둘 다 `cls=1.5`인데 결과가 완전히 갈림 — cls weight 단독으로는 설명 안 됨(optimizer와의 상호작용 가능성만 열어둠) |
| NMS 종류(dual-threshold vs 표준) | **주요 원인 아님** | 손실 %에 미치는 영향이 전 모델 공통으로 작음(0.7~1.7%p) |
| `model.0`이 최초 divergence 지점 | **반증** | 171-layer 분석에서 `model.0` 자체는 세 모델 간 cosine 차이 0.0004 이내로 사실상 동일(§17-5) |
| DFL Softmax 연산 자체가 오차를 증폭 | **반증** | A/B/C 3단계 분해 결과 NVT Softmax 구현은 cos=1.00000로 정확, 오차는 전부 Softmax **이전** 단계(pre-softmax logit)에서 이미 결정됨(§18) |
| box branch(cv2)만의 문제 | **수정** | Detect head 전체(box branch cv2 + cls branch cv3 양쪽)에서 동등하게 오차 확대 관찰(§17-4) |
| BN 통계 이상치 자체를 독립 root cause로 보는 해석 | **수정** | BN 이상치는 실재하고 optimizer와 강하게 연동되지만, "BN 이상치 → NVT 오차"라는 인과 경로 자체가 직접 증명된 것은 아님(model.0에서 오차가 시작되지 않는다는 §17 결과와 상충하지 않게 하려면, BN 이상치를 "누적된 결과의 한 표지"로 보는 것이 더 정확) |
| `dair` 트레이너 자체(dual-forward consistency loss/증강)가 불안정성의 원인 | **약화** | `v02_dair`(dair 트레이너 + dair_params/증강 그대로 켠 채로 optimizer만 SGD)가 PT/NVT/Float/Board 4개 레벨 전부에서 SGD 그룹에 속함 — trainer가 아니라 optimizer가 지배적 변수임을 재확인(§24) |

과거 유력했던 가설(BN 이상치 단독, Softmax 증폭, model.0 최초 발화, box branch 단독,
dair 트레이너 자체)을 현재 결론처럼 서술하지 않는다 — 전부 위 표의 판단으로 대체됐다.

---

## 7. 최종 Root Cause 판단

### 확정/가장 강한 증거

- **optimizer(SGD vs AdamW)** — **두 개의 독립 ablation**으로 PT(BN 통계)·
  NVT(simulator cosine)·Float(6,152장 mAP)·Board(6,152장 실측) **4개 레벨 전부에서
  재현**됐다:
  1. optimizer-axis(`lr0/cos_lr/batch/cls` 고정, 실험① SGD ↔ 실험② AdamW, 둘 다
     symmetric 트레이너)
  2. trainer-axis(optimizer/lr0/lrf/cos_lr/batch/cls 고정, v01_symmetric(SGD) ↔
     `v02_dair`(trainer만 dair로 교체, SGD 유지) — trainer가 dair로 바뀌고
     dual-forward consistency loss/증강이 그대로 켜져 있어도 optimizer=SGD면
     4개 레벨 전부 건강했다.

  이 조사에서 나온 결론 중 가장 강하게 뒷받침된다 — 서로 다른 축을 통제한 두
  실험이 같은 결론(optimizer가 지배적 변수)으로 수렴했기 때문이다.

### 강하게 연관된 중간 현상

- BN 활성화 통계 불안정성(이상치 레이어 수, Inf 채널)
- pre-softmax DFL logit dynamic range 확대(symmetric/matched/v02 3모델에서 확인,
  실험①/②는 미검증)
- NVT simulator cosine 저하(특히 Sigmoid)
- Detect head(box+cls 양쪽) 전반의 Float→NVT 오차 확대

### 아직 미확정인 메커니즘

- AdamW가 왜 이 데이터셋/조건에서 BN 통계를 불안정하게 만드는지(momentum/adaptive
  per-parameter update와 BN 통계 갱신의 상호작용) — 전혀 조사되지 않음
- BN 통계 불안정성이 pre-softmax dynamic range 확대를 **직접** 유발하는지, 아니면
  둘 다 별개로 optimizer의 영향을 받는 병렬 현상인지
- optimizer 내부의 어떤 구체적 요소(momentum 계수, adaptive lr, 둘의 조합)가
  핵심 변수인지
- pycocotools vs ultralytics 매칭 알고리즘 자체의 라인 단위 차이(§8, root cause와는
  별개 축)

---

## 8. Evaluator 이슈는 별도 축 (root cause와 혼동 금지)

pycocotools(`customCOCOeval`)와 ultralytics(`ap_per_class`+`match_predictions`) 간
손실 % 차이는 **위 root cause 분석과 독립적인 축**이다.

- **IoU threshold 범위/개수는 두 evaluator가 완전히 동일**(0.50:0.05:0.95, 10개) —
  코드로 확인, mAP50-95 계산 방식의 차이가 원인이 아님.
- **AP 적분 방식도 동일**(둘 다 표준 101-point COCO 보간).
- evaluator를 바꾸면 손실 %의 **절댓값은 달라진다**(예: v01_symmetric std+pycoco
  -6.63% vs std+ultra -0.62%, §3.1) — 그러나 **모델 간 손실 순위, 그리고 "SGD 계열
  vs AdamW 계열"이라는 optimizer 결론은 어느 evaluator를 쓰든 동일하게 유지된다**
  (§3.1, §3.2 두 표 모두 같은 순서).
- **maxDets**(pycocotools 100 제한): board 이미지당 detection 최대 60건이라 실질적
  영향 없음(배제 확인).
- **NMS 종류(dual-threshold vs 표준)의 효과는 상대적으로 작다**(0.7~1.7%p, 전
  모델 공통).
- 남은 미확정: GT↔Prediction 그리디 매칭 알고리즘(`COCOeval.evaluateImg` vs
  `BaseValidator.match_predictions`) 자체의 구현 차이가 evaluator 간 절대 손실 %
  격차의 남은 유력 원인으로 추정되나, 라인 단위 직접 대조는 수행되지 않았다.

---

## 9. 현재 상태 / 남은 선택적 후속

**필수 실험 완료. 대기 중인 실측 없음.**

아래는 "다음에 반드시 해야 할 작업"이 아니라, **원인을 더 깊게 설명하고 싶을 경우에만
수행할 선택적 분석**이다:

- optimizer → BN 불안정 메커니즘 분석(momentum/adaptive update 관점)
- 실험①/②/`v02_dair`에 §18 방식의 pre-softmax dynamic range 분해 적용(재추론
  불필요, 기존 NVT dump로 가능)
- `model.22.cv2.*`(box 브랜치 conv) weight/bias 통계 근원 추적
- pycocotools vs ultralytics 매칭 알고리즘 직접 비교(합성 예제)
- `v01_dair`/`v02`의 `model.0.bn` Inf가 정확히 같은 채널(인덱스 4)에서 발생하는
  이유 추적(§4.1) — 우연인지 재현 가능한 실패 지점인지 미확정

---

## Appendix: 조사 흐름 요약

1. 3모델(symmetric/matched/v02) 파이프라인 전체(학습설정→PT→ONNX→NVT→Board) 비교 →
   ONNX export/NVT compile/calibration 전부 동일 확인, BN 이상치가 가장 유력한
   단서로 부상.
2. 171-layer `layer_diff.txt` 분석 → "model.0 최초 발화" 가설 반증, DFL Softmax
   구간(idx196)에서 급격한 cosine 붕괴 발견 → "Softmax가 증폭한다"는 잠정 결론.
3. Softmax를 pre/PC-softmax/실제-NVT 3단계로 분해 → Softmax 자체는 정확(B≈C),
   오차는 이미 pre-softmax logit 단계에서 결정됨을 확인 → "Softmax 증폭" 가설 정정.
4. 사용자가 lr0/cos_lr/batch/cls를 고정하고 optimizer만 바꾼 실험①(SGD)/②(AdamW)를
   별도로 학습 → PT 레벨 BN 통계에서 즉시 재현 확인.
5. 실험①/②를 ONNX export + NVT compile/simulate → NVT cosine 레벨에서도 재현
   (실험② Sigmoid cosine 0.797=전체 최저).
6. 실험①/② Float 6,152장 평가 → 실험②만 Float 자체 저하 발견(0.412), 해석 시
   절대 mAP 병기 필요성 확인.
7. 실험① Board 실측(-4.6~-7.9%, SGD 그룹과 동일) → 실험② Board 실측
   (-36.1~-40.3%, 전체 최악) → optimizer 가설이 optimizer-axis에서
   PT/NVT/Float/Board 4개 레벨 전부에서 완결.
8. 사용자가 trainer-axis 두 번째 독립 ablation(`v02_dair` = dair 트레이너 +
   v01_symmetric의 SGD 하이퍼파라미터 그대로)을 별도 학습 → PT(이상치 2개/Inf0)/
   NVT(Softmax 0.9857/Sigmoid 0.9876/dfl_conv 0.9945)/Float(0.53대, 저하 없음)
   4개 레벨 중 3개가 즉시 SGD 그룹 재현. 동시에 `v01_dair`의 BN 통계 공백도
   채워져(이상치 5개+Inf1) AdamW 그룹 패턴이 7개 모델 전부에서 예외 없이 확인됨.
9. `v02_dair` Board 실측(-0.70~-4.91%, v01_symmetric과 사실상 동급, 실험①보다
   좋음) → trainer-axis에서도 4개 레벨 전부 완결, optimizer가 root cause라는
   결론이 서로 다른 축을 통제한 두 독립 실험으로 수렴.
