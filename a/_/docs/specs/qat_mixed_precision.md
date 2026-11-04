# QAT Mixed-Precision 분석 — Novatek CNN25 NPU

> ⚠️ **2026-05-13 QAT 미사용 결정 (D-010)** — 본 문서의 분석은 보존되며 QAT 부활 시 그대로 참조 가능. 결정 사유는 `docs/decisions.md` D-010 참조.

- **프로젝트명**: DAIR-YOLO
- **작성일**: 2026-03-13 (KST)
- **소스 데이터**: `quan_info_bitinfo.json` (221 nodes)

---

## 1. CNN25 실제 Bit Profile

### 전체 요약

| 카테고리 | 노드 수 |
|---------|---------|
| 순수 8-bit | 31 |
| 순수 16-bit | 84 |
| Mixed 8/16-bit | 106 |
| **합계** | **221** |

### Conv 레이어 (64개) — 모든 weight는 12-bit

| Input → Weight → Output | 수 | 위치 |
|------------------------|---|------|
| 8b → **12b** → 16b | 50 | Backbone, neck, 대부분 head conv |
| 16b → **12b** → 16b | 7 | C2f bottleneck cv1 + DFL conv |
| 8b → **12b** → 8b | 7 | SPPF cv1, final head conv (cv2.X.2, cv3.X.2) |

> Weight는 **12-bit 고정소수점** (Novatek 포맷) — 표준 INT8/INT16 아님.

### Non-Conv 레이어

| 타입 | 출력 bit | 수 | 비고 |
|------|---------|---|------|
| Sigmoid | 16b | 57 | 전체 |
| Mul (SiLU dominant) | **8b** | 46 | SiLU 후 압축 |
| Mul (SiLU C2f residual) | **16b** | 10 | C2f cv1/bottleneck cv2 경로 |
| Concat | 8b | 13 | FPN neck |
| Concat | 16b | 2 | DFL branch |
| Reshape/Transpose/Softmax | 16b | 4 | DFL head |
| Upsample, MaxPool, Add, Split | 8b | ~13 | Neck/SPPF/residual |

### 핵심 아키텍처 패턴

**SiLU dominant (46 Muls → 8b)**:
```
Conv (out=16b) → Sigmoid (out=16b) → Mul (out=8b)
```

**C2f residual exception (10 Muls → 16b)**:
```
Conv (out=16b) → Sigmoid (out=16b) → Mul (out=16b) → Split / residual Add
```

**DFL head (전체 16b)**:
```
cat → view → transpose → softmax → dfl_conv → view (out=16b)
```

---

## 2. PyTorch Native FakeQuantize 시뮬레이션 가능 여부

**결론: 직접적으로 불가능.**

PyTorch `prepare_qat`는 **모듈 경계**에 FakeQuantize 삽입. NPU mixed precision은 **컴파일러 후처리 정책** (`gen_config.txt`).

### 가능한 것

| 항목 | 비고 |
|------|------|
| Per-module QConfig | `module.qconfig = X` before `prepare_qat` |
| DFL head 제외 | `quantize_head=False` |
| 12-bit weight 시뮬레이션 | `novatek_pct_8a12w` (quant_min=-2048, quant_max=2047) |
| Uniform 8-bit activation | SiLU→8b dominant path 근사 |

### 불가능한 것

| NPU 동작 | 이유 |
|----------|------|
| SiLU Mul → 8b | fused Conv 내부 Mul — hook 포인트 없음 |
| C2f residual Mul → 16b | 동일 Conv 내부 — 선택적 16b 불가 |
| Per-Conv 출력 precision (8b vs 16b) | 개별 Conv별 QConfig 수동 할당 필요 |
| 동일 레이어 타입 다른 precision | PyTorch는 모든 Mul을 동일 취급 |

---

## 3. 현재 파이프라인 시뮬레이션 범위

**타겟: `novatek_pct_8a12w`** = Uniform INT8 activation (PercentileObserver) + 12-bit per-channel weight

| NPU 경로 | QAT 시뮬레이션 | 정확도 |
|----------|--------------|--------|
| 12b weight (64 Conv) | 12b weight | **정확** |
| 8b SiLU Mul (46 nodes) | 8b 모듈 경계 (위치 차이) | 근사 |
| 16b C2f residual Mul (10 nodes) | 8b (구분 없음) | **불일치** |
| 16b DFL | Unquantized | **정확** |
| 8b→8b head conv (7 nodes) | 8b | 근사 |
| 16b C2f bottleneck Conv input (7 nodes) | 8b (uniform) | **불일치** |

---

## 4. 정밀 시뮬레이션 필요 조건

`torch.fx` graph surgery 또는 per-module qconfig 수동 할당 필요:

1. 7개 `8b→8b` Conv 식별 (SPPF cv1, cv2.X.2, cv3.X.2) → INT8 output QConfig
2. 7개 `16b→16b` Conv 식별 (C2f bottleneck cv1, DFL conv) → INT16 input+output QConfig
3. 각 SiLU Mul 뒤 FakeQuantize 삽입 → `torch.fx` 필요
4. 10개 C2f residual Mul에 16b FakeQuantize → `torch.fx` 필요
5. DFL 경로 unquantized → `quantize_head=False` (이미 적용)

---

## 5. 실용적 권장사항

`novatek_pct_8a12w` uniform 근사가 올바른 시작점. NPU 컴파일 후 mAP drop으로 판단:

| mAP drop (float → NPU) | 권장 조치 |
|------------------------|----------|
| < 0.5% | `novatek_pct_8a12w` 충분 |
| 0.5% – 2% | 7개 head conv + 7개 C2f bottleneck conv per-module qconfig 검토 |
| > 2% | `torch.fx` 기반 full mixed-precision FakeQuantize 삽입 |

NPU 컴파일러는 QAT observer 통계와 독립적으로 scale 재보정 (ONNX export 시 FakeQuantize 제거됨). QAT의 주 이점은 **weight의 quantization noise 내성** → uniform 근사가 충분히 방어적.
