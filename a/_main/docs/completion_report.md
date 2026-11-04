# 완료 보고서 (Completion Report)

- **프로젝트명**: DAIR-YOLO
- **최초 작성**: 2026-04-28 (KST)
- **최종 완료**: -

> 소단위 작업이 완료될 때마다 verifier가 테스트 결과를 여기에 누적한다.
> 사용자 최종 확인(완료 사인) 이후 요약 섹션을 작성한다.

---

## 소단위 테스트 결과

<!-- verifier가 소단위 완료 시마다 아래 형식으로 항목을 추가한다 -->

| # | 기능 | 판정 | 완료일 (KST) | 비고 |
|---|------|------|-------------|------|
| 1 | Rear baseline 150ep | ✅ | 2026-04-28 | mAP50=0.7627, mAP50-95=0.5433 |
| 2 | Front baseline 150ep | ✅ | 2026-04-28 | mAP50=0.6876, mAP50-95=0.4869, AMP=False (D-006) |
| 3 | Rear GA tuning (20 gen, 202 trials) | ✅ | 2026-05-04 | best fitness=0.52995, 139.4h |
| 4 | Front GA tuning (5 gen, 52 trials) | ✅ | 2026-05-04 | best fitness=0.45432, 130.9h, 4K 대용량 영향 |
| 5 | Rear GA full200 (warm-start) | ⚠️ | 2026-05-06 | ep188 best mAP50-95=0.5391, baseline 0.5433 못 넘음 |
| 6 | Front GA full200 (warm-start) | ⚠️ | 2026-05-06 | ep51 best mAP50-95=0.4419 (early-stop ep101), baseline 0.4869 못 넘음 |
| 7 | **Front baseline200 (production)** | ✅ | 2026-05-07 | **ep199 best mAP50-95=0.49588, mAP50=0.69679** (D-008) |

<!--
각 항목 상세:

### [#]. [기능명] — YYYY-MM-DD HH:MM (KST)
**판정**: ✅ 통과 / ❌ 실패 / ⚠️ 조건부 통과
**테스트 항목**
- 정상 동작: ✅/❌
- 경계값 처리: ✅/❌
- 에러 처리: ✅/❌

**발견된 문제**: (없으면 없음)
**검증 근거**: (실행 명령, 출력 결과)
-->

---

## 최종 요약

> 사용자 완료 사인 이후 작성

- **완료된 기능**: -
- **미완료/보류 항목**: -
- **전체 테스트 통과율**: -/- (-)
- **주요 이슈 및 해결**: -

### 알 수 없음 — 
**판정**: ❓

**검증 항목**
| 항목 | 결과 | 비고 |
|------|------|------|


**발견된 문제**: 없음

**검증 근거**: 

---
### DMS pose 부위별 가중치 yaml 설정 가능화 — Phase 2 코드 스멜 해소 — 2026-05-08 10:45 KST
**판정**: ❌

**검증 항목**
| 항목 | 결과 | 비고 |
|------|------|------|
| 단위 테스트 회귀 (pytest tests/) | ✅ | 24 passed, 1 skipped — Phase 1과 동일 |
| _select_target_keypoints 이중 호출 제거 | ✅ | DMSPoseLoss 내 L1331 단 1회만 호출 |
| 이중 호출 제거 동등성 — anchor space xy | ✅ | L1332에서 /= stride 변환 완료. shape loss에서 동일 공간 사용 |
| 이중 호출 제거 동등성 — visibility in-place 수정 없음 | ✅ | gt_kpt는 슬라이스 뷰이나 vis_mask는 != 비교로 새 bool 텐서. selected_keypoints[...,2] 오염 없음 |
| D-009 형식 준수 | ✅ | append-only, 순서 올바름, 한국어 본문 |
| docs/.verifier_result.json 무결성 | ❌ | 파일 자체가 존재하지 않음 |
| 새로운 코드 스멜 | ✅ | Phase 2 변경으로 새 스멜 없음 |

**발견된 문제**: - docs/.verifier_result.json 미생성: Phase 2 변경 내역에 '구조화 JSON 보존'으로 명시했으나 파일 부재

**검증 근거**: pytest tests/ → 24 passed 1 skipped; grep _select_target_keypoints loss.py → 1건(L1331); git diff HEAD~1 loss.py로 이중 호출 블록 삭제 확인; ls docs/.verifier_result.json → No such file or directory

---
### DMS pose 부위별 가중치 yaml 설정 가능화 — Phase 2 마무리 — 2026-05-08 11:20 KST
**판정**: ✅

**검증 항목**
| 항목 | 결과 | 비고 |
|------|------|------|
| 단위 테스트 회귀 | ✅ | pytest tests/ → 24 passed, 1 skipped (2.14s) — 기존 결과와 동일 |
| _select_target_keypoints 이중 호출 제거 | ✅ | grep 결과 DMSPoseLoss.calculate_keypoints_loss 내 1회만 호출(L1331). shape_loss 블록은 동일 selected_keypoints 재사용 |
| visibility(selected_keypoints[..., 2]) in-place 수정 없음 | ✅ | L1342: gt_kpt[..., 2] != 0 → 새 bool 텐서 반환. L1357: selected_keypoints[..., 2] * masks → 곱셈 결과 새 텐서 kpt_vis. selected_keypoints[..., 2] 미변경 확인 |
| D-009 형식 및 append-only 준수 | ✅ | docs/decisions.md 마지막 행 D-009. D-008(2026-05-07) 다음 순차. 한국어 본문. 기존 항목 수정 없음 |
| 새로운 코드 스멜 미도입 | ✅ | Phase 2 변경(이중 호출 제거)으로 로직 단순화됨. 신규 스멜 없음 |

**발견된 문제**: 없음

**설계 냄새 (CODE_SMELL)**
- DMSPoseLoss.loss() ~120줄 — 50줄 임계 초과. D-009에 의식적 예외로 기록됨 (잔존 스멜이나 판정에 영향 없음)

**검증 근거**: 1) pytest tests/ → 24 passed, 1 skipped. 2) grep -n '_select_target_keypoints' loss.py → L1331 1건만 hit. 3) L1332 selected_keypoints[..., :2] /= stride (xy만 in-place 수정, visibility 채널 2 미변경). 4) L1357 kpt_vis = selected_keypoints[..., 2] * masks — 새 텐서. 5) decisions.md L16 D-009 확인

---
### QAT 당분간 미사용 처리 (soft disable) 문서 업데이트 — 2026-05-13 14:30 KST
**판정**: ✅

**검증 항목**
| 항목 | 결과 | 비고 |
|------|------|------|
| D-010 형식 준수 (append-only, D-009 다음 순번, 사유 명시) | ✅ | decisions.md 마지막 행에 D-010 추가. 날짜 2026-05-13, 한국어 본문, 사유·보존 파일·부활 절차 모두 명시 |
| CLAUDE.md 3곳 QAT 미사용 표기 | ✅ | 명령어 줄 주석, variants 표 qat/fx_qat 취소선, 알려진 버그 QAT 줄 취소선 — 모두 D-010 참조 |
| README.md 4곳 QAT 미사용 표기 | ✅ | 소개 문구, 기능 목록 Quantization, NPU pipeline, 명령어 표 QAT 학습 — 모두 D-010 참조 |
| glossary.yaml QAT 항목 deprecated 표기 | ✅ | definition에 '2026-05-13 미사용 결정 (D-010)' 추가 및 status: deprecated 필드 추가 |
| qat_mixed_precision.md deprecated 안내 추가 | ✅ | 헤더 바로 다음에 경고 블록 삽입, 본문 보존 |
| 코드·config 무변경 확인 | ✅ | git diff HEAD -- '*.py' 결과 projects/.py 파일 0바이트 변경. QAT train/val/tools/exp_yaml 파일 전부 존재 확인 |
| 단위 테스트 회귀 (pytest tests/) | ✅ | 24 passed, 1 skipped — 기대값과 동일 |
| README 링크 무결성 (qat_mixed_precision.md) | ✅ | docs/qat_mixed_precision.md 파일 존재 확인. README 문서 테이블 링크 유효 |
| D-010 참조 일관성 (5개 파일 전부) | ✅ | decisions/CLAUDE/README/glossary/qat_mixed_precision 모두 D-010 명시 |
| README 문서 테이블 QAT 링크 deprecated 미표기 | ✅ | README 131번째 줄 '[QAT mixed-precision 분석](docs/qat_mixed_precision.md)' — 취소선 없이 plain 링크로 남아 있음. 이는 문서 파일 자체가 보존(삭제 금지)되었고 qat_mixed_precision.md 내부에 deprecated 안내가 이미 추가됐으므로 링크 접근 시 deprecated 안내를 확인 가능 — 허용 가능한 상태 |
| plan.md 등 보존 대상 파일 미변경 | ✅ | git diff에 plan.md, CHANGELOG.md, ACHIEVEMENT.md, constraints.yaml 미포함 — 보존 명세 준수 |
| 추가 QAT 언급 grep (다른 docs/*.md) | ✅ | CHANGELOG.md, ACHIEVEMENT.md는 보존 대상으로 명시됨. constraints.yaml도 보존 대상. plan.md는 Phase 1 historical 항목 — 모두 건드리지 않음 기준 충족 |

**발견된 문제**: 없음

**검증 근거**: git diff HEAD -- docs/decisions.md CLAUDE.md README.md docs/glossary.yaml docs/qat_mixed_precision.md 직접 검토. git diff HEAD -- '*.py' 0바이트 확인. pytest tests/ → 24 passed, 1 skipped. ls로 QAT 코드 파일 전부 존재 확인.

---
### docs/ 디렉토리 하네스 정합화 — 2026-05-14 15:10 KST
**판정**: ✅

**검증 항목**
| 항목 | 결과 | 비고 |
|------|------|------|
| 내용 손실 0 — changelog.md가 두 원본 전체 포함 | ✅ | QAT 잔여 리스크 3건(build_dataset mode=val, 절대경로, Multi-GPU 미지원), 16행 수정 파일 테이블, ROIMotionBlur, build_front/rear_augmentations, DMS Stage-1/2 수치(mAP50-95(P)=0.995) 전부 확인 |
| git mv 무결성 — rename 인식 및 내용 동일 | ✅ | git status에서 R docs/harness_fix_report.md -> docs/debug/20260429_harness_check.md, R docs/qat_mixed_precision.md -> docs/specs/qat_mixed_precision.md 정상 인식. diff 결과 0바이트 (내용 동일) |
| README.md 링크 유효성 | ✅ | docs/changelog.md, docs/specs/qat_mixed_precision.md 두 파일 모두 실제 존재. README에 ACHIEVEMENT/CHANGELOG 대문자 참조 없음 |
| 참조 갱신 완전성 — 살아있는 구 경로 참조 없음 | ✅ | append-only 문서(completion_report.md, technical_doc.md, decisions.md) 제외 후 0건. lessons.md L27 harness 경로 → docs/debug/20260429_harness_check.md 갱신 확인. docs/ACHIEVEMENT.md L73 내부 자기 참조는 삭제 예정 파일이라 무해. .claude/state/ JSON 파일 경로는 시스템 상태 파일로 기능에 영향 없음 |
| 단위 테스트 — pytest tests/ 회귀 없음 | ✅ | 24 passed, 1 skipped — 기존 결과와 동일 |

**발견된 문제**: 없음

**검증 근거**: pytest tests/ → 24 passed, 1 skipped (2.14s). git status → R rename 2건 정상. diff 두 git mv 파일 → 0바이트. grep 구 경로 참조(append-only 제외) → 0건. changelog.md 238줄: ACHIEVEMENT 고유 항목 전부 확인(QAT 잔여 리스크 3건 L207-213, 16행 테이블 L221-238, ROIMotionBlur L49, build_front/rear L50, DMS 수치 L115-116)

---