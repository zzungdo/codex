# 개발 계획 — DAIR-YOLO

- **최초 작성**: 2026-04-29 (KST)
- **기술 스택**: Python, PyTorch 2.5+cu121, ultralytics, Novatek NPU SDK

> Phase/Sprint 단위 로드맵. Phase 전환 전 `docs/checklist.md` 해당 Phase 전체 ✅ 확인 필수.

## Phase 개요

| Phase | 목표 | 기간 | 관련 결정 | 상태 |
|-------|------|------|-----------|------|
| Phase 1 | DAIR/QAT/DMS 베이스라인 학습 | 2026-03 ~ 04 | D-002 ~ D-005 | ✅ 완료 |
| Phase 2 | GA 튜닝 + 200ep full training | 2026-04-29 ~ 05-07 | D-006~D-008 | ✅ 완료 |
| Phase 3 | NPU 변환·시뮬·온보드 검증 | TBD | - | 대기 |

---

## Phase 1 — 베이스라인 학습

### 목표
- [x] Rear 베이스라인 (`runs/detect/rear/weights/best.pt`, mAP50=0.763)
- [x] Front 베이스라인 (`runs/detect/front/weights/best.pt`, mAP50=0.688, mAP50-95=0.487)
- [x] DAIR 트레이너·밸리데이터 super() 위임 리팩토링 (D-002)
- [x] DMS Stage-1/2 학습 + ONNX export
- [x] SymmetriceDFL offset 버그 수정 (D-003)

### Sprint 1
| 작업 | 담당 | 완료 기준 |
|------|------|---------|
| Rear/Front 베이스라인 학습 | - | mAP50 ≥ 0.7 (Rear), mAP50-95 ≥ 0.45 (Front) |
| DAIR 리팩토링 | - | `_do_train`/`__call__` 제거 + pytest 통과 |
| DMS Stage-1/2 학습 | - | mAP50-95(P) ≥ 0.99 (Stage-1) |

---

## Phase 2 — GA 하이퍼파라미터 튜닝

### 목표
- [x] Rear GA 튜닝 — 20 generations, best fitness=0.52995
- [x] Front GA 튜닝 — 5 generations, best fitness=0.45432 (4K 대용량으로 generation 적음)
- [x] full200 비교 — GA params는 25ep horizon overfit (D-007), baseline200이 우월
- [x] Front production: baseline200 ep199, mAP50-95=0.49588 (D-008)

### Sprint 1
| 작업 | 담당 | 완료 기준 |
|------|------|---------|
| Rear GA 진행 모니터링 | - | Gen ≥ 5 또는 fitness 수렴 |
| Front GA AMP=False 재실행 | - | Gen 0 정상 완료 (NaN 미발생) |

---

## Phase 3 — NPU 변환 및 검증

### 목표
- [ ] best PT → ultralytics ONNX (fp32) export
- [ ] `onnx2novaonnx_converter.py` → nova ONNX 변환
- [ ] `run_compiler.sh` → NPU 바이너리 컴파일
- [ ] `run_simulator.sh` → cosine similarity 검증
- [ ] `visualize_sim.py` → Golden vs Sim 비교

### Sprint 1
| 작업 | 담당 | 완료 기준 |
|------|------|---------|
| Rear 416x256 NPU 변환 | - | cls cos / dfl cos ≥ 0.95 |
| Front 480x256 NPU 변환 | - | cls cos / dfl cos ≥ 0.95 |
| DMS Stage-1/2 NPU 변환 | - | sim_comparison_*.jpg 생성 + 시각 검증 |

---

## 설계 원칙

- TDD: 테스트 먼저 작성 → 실패 확인 → 구현 → 통과 확인
- Phase 전환 기준: `docs/checklist.md` 해당 Phase 전체 ✅
- 설계 결정은 `docs/decisions.md`에 D-번호로 기록
- 외부 SDK·컴파일러 작업 전: 공식 워크플로우 전체 단계 먼저 나열·확인 (D-001 wrapper 스크립트만)
