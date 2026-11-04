# QAT 코드 리뷰 결과

- **프로젝트명**: DAIR-YOLO
- **작성일**: 2026-03-13 (KST)
- **범위**: QAT quantization pipeline
- **수행**: code-reviewer agent

---

## 요약

| 심각도 | 발견 | 수정 |
|--------|------|------|
| Critical | 1 | 1 |
| High | 2 | 2 |
| Medium | 3 | 3 |
| Low | 3 | 2 |

---

## 이슈 상세

### [CRITICAL-1] INT16 qconfig `dtype=torch.qint8` — `tq.convert()` 시 overflow

**위치**: `quant_utils.py` — `_generic_int16_qconfig`, `_percentile_int16_qconfig`
**상태**: 수정 완료

`torch.qint8`는 `[-128, 127]` 저장. `quant_min=-32768, quant_max=32767` 사용 시 `tq.convert()` 때 실제 정수 텐서에서 overflow 발생. FakeQuantize forward는 정상 (clamping 사용) 이지만 변환된 모델은 손상됨.

**수정**: `dtype=torch.qint8` → `dtype=torch.qint32` (`[-2^31, 2^31-1]`, INT16 범위 안전 포함)

---

### [HIGH-1] INT16 시뮬레이션 qconfig 부재

**위치**: `quant_utils.py` lines 65-119
**상태**: 수정 완료

PyTorch에 native `torch.qint16` 없음. `dtype=torch.qint8` + `quant_min/max` override로 시뮬레이션 필요했으나 해당 target 미존재.

**수정**: `_INT16_QUANT_MIN/MAX` 상수, `generic_int16`/`percentile_int16` factory 및 TARGET_REGISTRY 등록

---

### [HIGH-2] `PercentileObserver.calculate_qparams` — private API로 `quant_min/max` 무시

**위치**: `quant_utils.py` lines 55-58
**상태**: 수정 완료

`_compute_quantization_params`가 dtype 기본 범위 `[-128, 127]` 사용. INT16 target에서도 INT8 noise 생성.

**수정**: `self.min_val/max_val` 복사 후 `super().calculate_qparams()` 호출 (public path)

---

### [MEDIUM-1] `get_qconfig` dispatch else-branch 잘못된 factory 라우팅

**위치**: `quant_utils.py` lines 131, 153-154
**상태**: 수정 완료

if/else 체인에서 새 percentile target이 잘못된 factory로 라우팅. `_PERCENTILE_FACTORIES` dict로 교체.

---

### [MEDIUM-2] `_convert_and_save_int8` guard flag 순서

**위치**: `train/qat.py` line 276
**상태**: 수정 완료

`self._int8_converted = True`가 `try` 전에 설정 — save 실패 시 재시도 영구 차단. `torch.save` 성공 후로 이동.

---

### [MEDIUM-3] 출력 파일명 `best_int8.pt` hardcoded

**위치**: `train/qat.py` line 284
**상태**: 수정 완료

INT16 target에서도 `best_int8.pt` 사용. `f"best_{self.qat_target}.pt"` 로 변경.

---

### [LOW-1] `base_qat.yaml` `model_name: custom` — 오해 소지

**위치**: `base_qat.yaml` line 5
**상태**: 수정 완료

`qat`로 변경 + 설명 주석 추가.

---

### [LOW-2] INT16 QAT noise ~256x 세밀 — weight adaptation 효과 미미할 수 있음

**상태**: 인지 (코드 수정 불필요)

INT16 step size ≈ 1.8e-4 (범위 [-6,6] 기준) → float32 rounding/SGD noise와 유사. NPU PTQ 후 mAP < 0.1% 하락이면 QAT 불필요.

---

### [LOW-3] `quant_min/quant_max` observer override — PyTorch 버전 민감

**상태**: 인지 (코드 수정 불필요)

PyTorch 1.10+ 필요. 향후 버전에서 propagation 변경 가능성 있음.

---

## 잔여 리스크

- `PercentileObserver`가 `self.histogram` (HistogramObserver 내부 attribute) 접근 — PyTorch 내부 변경 시 fallback
- NPU 컴파일러가 QAT observer 통계와 독립적으로 재보정 가능성 — Novatek 툴체인 확인 필요
- `_aarch64_qconfig` == `_generic_qconfig` — 향후 qnnpack 등 분화 가능
