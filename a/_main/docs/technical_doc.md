# 기술 문서 (Technical Document)

- **프로젝트명**: DAIR-YOLO
- **최초 작성**: 2026-04-28 (KST)

> 소단위 작업이 완료될 때마다 verifier가 해당 기능 설명을 여기에 누적한다.

---

## 아키텍처 개요

> 전체 구조는 `docs/plan.md` 참조.

---

## 구현 상세

<!-- verifier가 소단위 완료 시마다 아래 형식으로 항목을 추가한다 -->

<!--
### [기능명] — YYYY-MM-DD HH:MM (KST)

**구현 내용**
- 무엇을 구현했는지

**주요 로직**
- 핵심 처리 흐름, 알고리즘, 설계 결정

**관련 파일**
- `path/to/file.py` — 역할 설명

**인터페이스**
- 입력: ...
- 출력: ...
-->

---

### Phase 1·2 학습 결과 — 2026-05-07 (KST)

**구현 내용**
- Rear/Front DAIR 베이스라인 150ep + GA 튜닝 + 200ep full training 비교 실험
- 최종 production 모델: Front baseline200 (ep199 best.pt)

**주요 로직**
- 모델: SymmetricDetect head 적용 YOLOv8n (3.0M params, 8.2 GFLOPs)
- 학습: from-scratch (weight_path=`.yaml`) — DAIR Trainer + DynamicCrop dataset
- DAIR augmentation: SunFlareAug, SpeakerOverlayAug, FlipOverlay, ROIMotionBlur (Front Pre-Mosaic + Tensor Select 양쪽)
- GA 튜닝: 23 continuous + 5 categorical 파라미터, LHS init, SBX crossover, 25ep trial-budget
- horizon mismatch (D-007): 25ep optimal hyperparams ≠ 200ep optimal, baseline params가 long-run에서 우월

**관련 파일**
- `projects/exp_yaml/baseline/{rear,front}_dair_150ep.yaml` — 베이스라인 config
- `projects/exp_yaml/baseline/{rear,front}_dair_ga.yaml` — GA 탐색 base
- `projects/exp_yaml/baseline/{rear,front}_dair_full200.yaml` — GA best 200ep config
- `projects/exp_yaml/baseline/front_baseline200.yaml` — 베이스라인 + 200ep production config
- `tools/ga_tuner.py` — GA 튜너 (LHS, SBX, polynomial mutation, elitism)
- `runs/detect/{rear,front}/weights/best.pt` — 베이스라인 150ep
- `runs/detect/runs/detect/{rear_full200,front_full200,front_baseline200}/` — 200ep 결과들

**최종 메트릭** (Front, 검증셋 17,507)
| 모델 | mAP50 | mAP50-95 | 비고 |
|---|---|---|---|
| Baseline 150ep | 0.6876 | 0.4869 | 기준 |
| GA full200 | 0.6587 | 0.4419 | -0.045 ❌ horizon mismatch |
| **Baseline 200ep (production)** | **0.69679** | **0.49588** | **+0.0090 ✅** |

**최종 메트릭** (Rear, 검증셋 5,709)
| 모델 | mAP50 | mAP50-95 | 비고 |
|---|---|---|---|
| Baseline 150ep | 0.7627 | **0.5433** ⭐ | mAP50-95 우세 |
| GA full200 (best ep188) | **0.7784** ⭐ | 0.5391 | mAP50 우세 |
| Baseline 200ep | (미수행) | (미수행) | 추후 검토 |

**인터페이스**
- 입력: BGR uint8 [0,255] → augmentation → float32 [0,255] (정규화 안 함, D-005)
- 출력: 5 classes (car, bus, truck, motorcycle, person)
- 추론 해상도: Front 480x256, Rear 416x256

### 알 수 없음 — 

**구현 내용**
-

**주요 로직**
-

**관련 파일**
- 없음

**인터페이스**
- 입력: -
- 출력: -

---
### DMS pose 부위별 가중치 yaml 설정 가능화 — Phase 2 코드 스멜 해소 — 2026-05-08 10:45 KST

**구현 내용**
shape loss 분기의 _select_target_keypoints 중복 호출 + clone() + 별도 stride 변환을 제거하고 첫 호출 결과(selected_keypoints)를 직접 재사용

**주요 로직**
L1331에서 1회 호출 → L1332에서 [..,:2] /= stride(in-place) → keypoint_loss 블록은 [masks] 슬라이스 사용 → shape_loss 블록은 selected_keypoints 전체 직접 참조

**관련 파일**
- `projects/src/utils/loss.py` — DMSPoseLoss.calculate_keypoints_loss 내 이중 호출 제거
- `docs/decisions.md` — D-009 append — DMSPoseLoss.loss 50줄 임계 의식적 예외 기록

**인터페이스**
- 입력: calculate_keypoints_loss(masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts)
- 출력: (kpts_loss: scalar tensor, kpts_obj_loss: scalar tensor)

---
### DMS pose 부위별 가중치 yaml 설정 가능화 — Phase 2 마무리 — 2026-05-08 11:20 KST

**구현 내용**
_select_target_keypoints 이중 호출 제거 — 첫 호출 결과(anchor space 변환 완료)를 shape_loss 블록에서 직접 재사용

**주요 로직**
calculate_keypoints_loss에서 _select_target_keypoints 1회 호출 → xy /= stride (anchor space) → keypoint_loss 블록 + shape_loss 블록 모두 동일 selected_keypoints 참조. visibility 채널은 수정되지 않으므로 공유 안전

**관련 파일**
- `projects/src/utils/loss.py` — DMSPoseLoss.calculate_keypoints_loss — 이중 호출 제거, selected_keypoints 재사용
- `docs/decisions.md` — D-009 append — DMSPoseLoss.loss() 50줄 임계 의식적 예외 기록

**인터페이스**
- 입력: masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
- 출력: (kpts_loss, kpts_obj_loss) — 변경 없음

---
### QAT 당분간 미사용 처리 (soft disable) 문서 업데이트 — 2026-05-13 14:30 KST

**구현 내용**
QAT/FX-QAT 워크플로우를 코드 보존 상태로 5개 문서에만 deprecated 표기 (soft disable)

**주요 로직**
1) decisions.md D-010 append → 2) CLAUDE.md 3곳 취소선+D-010 → 3) README.md 4곳 취소선+D-010 → 4) glossary.yaml status:deprecated+D-010 → 5) qat_mixed_precision.md 경고 블록 삽입

**관련 파일**
- `docs/decisions.md` — D-010 결정 기록 (append-only)
- `CLAUDE.md` — 명령어·variants·알려진 제약 3곳 deprecated 표기
- `README.md` — 소개문구·기능목록·NPU pipeline·명령어표 4곳 deprecated 표기
- `docs/glossary.yaml` — QAT 용어 status:deprecated + definition 갱신
- `docs/qat_mixed_precision.md` — 문서 상단 deprecated 경고 블록 추가

**인터페이스**
- 입력: 없음 (문서 전용 변경)
- 출력: 없음 (문서 전용 변경)

---
### docs/ 디렉토리 하네스 정합화 — 2026-05-14 15:10 KST

**구현 내용**
ACHIEVEMENT.md + CHANGELOG.md 두 파일을 changelog.md 1개로 병합, qat/harness 문서 하위 디렉토리 이동, README 및 lessons.md 참조 갱신

**주요 로직**
CHANGELOG 날짜 역순 구조를 척추로 유지하고 ACHIEVEMENT 고유 내용(잔여 리스크 섹션, 누적 파일 목록 테이블, 항목별 목적/성과 수치)을 각 해당 날짜 항목에 흡수. git mv로 docs/qat_mixed_precision.md → docs/specs/, docs/harness_fix_report.md → docs/debug/ 이동

**관련 파일**
- `docs/changelog.md` — 신규 병합 파일 (ACHIEVEMENT + CHANGELOG 통합)
- `docs/ACHIEVEMENT.md` — 원본 — 검증 후 삭제 예정 (현재 보존 중)
- `docs/CHANGELOG.md` — 원본 — 검증 후 삭제 예정 (현재 보존 중)
- `docs/specs/qat_mixed_precision.md` — git mv 이동됨 (구 docs/qat_mixed_precision.md)
- `docs/debug/20260429_harness_check.md` — git mv 이동됨 (구 docs/harness_fix_report.md)
- `README.md` — 문서 테이블 링크 갱신 (changelog, specs/qat 경로)
- `.claude/memory/lessons.md` — L27 harness 경로 참조 갱신

**인터페이스**
- 입력: docs/ACHIEVEMENT.md (125줄), docs/CHANGELOG.md (179줄)
- 출력: docs/changelog.md (238줄) — 두 원본 고유 정보 무손실 통합

---