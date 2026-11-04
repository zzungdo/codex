# 병합 감사 문서 (Merge Audit)

> 평상시 AI가 읽을 필요 없는 문서다 — 통합 작업의 근거·검증 기록만 담는다. 작업 자체는
> 문서 정리에 한정됐다: 문서에 적힌 과거 지시·재학습·변환·보드 테스트·커밋·배포를 이번
> 통합 작업에서 실행하지 않았다.

## 1. 입력 목록 확정 경위

`docs/debug/` 루트에는 24개 마크다운이 있다(사용자 사전 예상 18개와 불일치). 확인 결과:

- **6개는 범위 밖**: `20260313_qat_review.md`, `20260526_ncnvox_kws_data_inventory.md`,
  `20260527_phase_d_poc.md`, `20260527_phase_e_full.md`, `20260527_phase_f_npu.md`,
  `20260729_config_canonicalization.md`. 앞 5개는 QAT 코드 리뷰·ncnvox KWS(키워드 스포팅)
  데이터/학습 Phase D·E·F로, 트럭 검지(Rear/Front/Corner/Side/SigmaStar)와 무관한 별개
  프로젝트다. `20260729_config_canonicalization.md`는 BSD_F606 전체 config 정본화에 관한
  범용 문서로 특정 트럭 주제에 속하지 않는다. 이 판단은 헤더/본문 내용을 직접 읽어 확인한
  것이며, 최종적으로 **사용자가 turn 중간에 "20260806이후부터를 하면 돼"라고 명시적으로
  확정**해 이 6개를 범위에서 제외했다.
- **날짜 20260806 이후 18개**가 정확히 트럭 검지 디버그 문서이며 사용자의 원래 예상(18개)과
  일치한다 — 이것이 최종 입력 목록이다(아래 §2 대응표).

작업 환경: git worktree(`docs-debug-consolidation`, 브랜치
`worktree-docs-debug-consolidation`)에서 진행. 원본 18개 파일은 worktree에 untracked
상태로 복사해 읽기 전용으로 취급했고(worktree는 `master` 기준으로 생성돼 이 파일들이
포함되지 않았기 때문 — 실제 작업 디렉토리인 `docs/debug/`에서는 아직 git 커밋되지 않은
상태로 존재), 산출물은 `docs/debug/consolidated/`에 새로 작성했다.

## 2. 원문 절 → 통합 문서/절 대응표

| 원문 파일 | 원문 절(주요) | 목적지 |
|---|---|---|
| `20260806_rear_truck_v01_npu_quant_gap.md` | 전체(질문1/2, calib 튜닝, 2026-08-10~11 추가조사, spec_demarc subset 탐색, v02 convergence/weight 분석, v02 Float Validation/ONNX Baseline, 외부학습비교 v01_dair/v01_symmetric, 최종결론 §1~§7, v01_symmetric_matched ablation) | [01](01_rear_quantization_and_release.md) §1(데이터셋), §3.1(연대기 최초 발견), §3.2(matched ablation), §8(실패시도 표) |
| `20260810_npu_pattern_name_board.md` | 배경/조사과정/결론/실전권고 | [90](90_reproduction_reference.md) §5(pattern-name 이슈 전체) |
| `20260810_npu_pattern_name_board.md` | "보드 실측 mAP — 양자화 손실 정량화" 절 | [02](02_front_reference_and_reproduction.md) §7(Front v01_repro 별도 재현 스트림) |
| `20260814_corner_truck_reference_onnx_vs_nas_best_pt.md` | 전체(요약표, 비교대상, 6단계 판정, 3-way 검증) | [04](04_corner_reproduction.md) §1 |
| `20260814_corner_truck_v01_nvt_calibration_simulator_comparison.md` | 전체(calibration/compile/simulator, 기존 board 결과, 다음단계) | [04](04_corner_reproduction.md) §2~§4 |
| `20260814_front_truck_reference_onnx_local_nvt_reconversion.md` | 전체(calibration NOT FOUND, 로컬 재변환, board 판정기준) | [02](02_front_reference_and_reproduction.md) §3 |
| `20260814_front_truck_reference_onnx_nvt_board_prepare.md` | 전체(기존 NVT artifact FOUND, 신규 compile 미실행) | [02](02_front_reference_and_reproduction.md) §3 |
| `20260814_front_truck_reference_onnx_source_checkpoint_search.md` | 전체(Search Scope A/B, NOT FOUND, 탐색 안 한 범위) | [02](02_front_reference_and_reproduction.md) §2 |
| `20260814_front_truck_reference_onnx_vs_best_pt.md` | 전체(5-level 판정, 최초 divergence) | [02](02_front_reference_and_reproduction.md) §1 |
| `20260814_rear_truck_v01_symmetric_reexport_nvt_revalidation.md` | 전체(재-export bit-exact, board 미완료) | [01](01_rear_quantization_and_release.md) §7 |
| `20260814_side_full_pipeline_reproduction.md` | 전체(§1~§11, 최종결론, First Divergence 표) | [05](05_side_reproduction.md) 전체(§0~§11 그대로 대응) |
| `20260819_front_golden_reference_for_rear_1pct.md` | §4~§18, §20, §24~§25(Front 자체 spec/성능/재현) | [02](02_front_reference_and_reproduction.md) §4~§6 |
| `20260819_front_golden_reference_for_rear_1pct.md` | §19,§21~§23(가설), REAR Experiment A(spec_demarc ablation, 중단) | [01](01_rear_quantization_and_release.md) §8(실패시도표), §9(Front golden reference가 Rear에 전이된 시사점) |
| `20260824_symmetric_low_quant_loss_rootcause_compare.md` | §1~§16(3모델 파이프라인 비교, 원인후보 우선순위화) | [01](01_rear_quantization_and_release.md) §3.3 |
| `20260824_symmetric_low_quant_loss_rootcause_compare.md` | §17~§18(layer-by-layer, DFL Softmax 분해) | [01](01_rear_quantization_and_release.md) §3.3 (같은 절 하단) |
| `20260824_symmetric_low_quant_loss_rootcause_compare.md` | §19~§24(optimizer ablation, trainer-axis v02_dair) | [01](01_rear_quantization_and_release.md) §3.4 |
| `20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md` | 전체(v01_dair/v02/matched 2×2 evaluator 검증) | [01](01_rear_quantization_and_release.md) §6 |
| `20260824_v01_symmetric_eval_gap.md` | §1~§9(2×2 isolation, 결론) | [01](01_rear_quantization_and_release.md) §6 |
| `20260824_v01_symmetric_eval_gap.md` | §10~§10-1(네 모델 종합비교) | [01](01_rear_quantization_and_release.md) §6(같은 절 하단) |
| `20260831_symmetric_quant_loss_rootcause_summary.md` | 전체(Executive Summary~Appendix) | [01](01_rear_quantization_and_release.md) §3.4(최종판단)·§4(중간현상/배제표)·§5(요약 수치 근거) — 최종 결론의 1차 근거 문서 |
| `20260902_rear_truck_release_candidate_selection.md` | 전체(§1~§6, 결정기준, 진행상황 트래커) | [01](01_rear_quantization_and_release.md) §0(최신 상태)·§5(릴리즈 후보 선정) |
| `20260907_rear_truck_v01_symmetric_sigmastar_conversion.md` | 전체(§0~§7) | [06](06_sigmastar_conversion.md) 전체(§0~§8 그대로 대응) |
| `20260909_front_truck_v01_experiment_comparison.md` | 전체(§1~§5, 2026-09-14 갱신분 포함) | [03](03_front_training_experiments.md) 전체(§0~§5 그대로 대응) |

**중복 병합 처리**: 여러 원문이 같은 사실(예: Rear 6모델 quant loss 비교표)을 반복
인용하는 경우, 가장 완전한 형태(최신·최다 조건 포함) 1곳에 기록하고 나머지는 그 절을
링크했다 — 예: 6모델 종합표는 `20260831_..._rootcause_summary.md` §3.1의 값을
[01](01_rear_quantization_and_release.md) §3.4에 정본으로 채택했고, 동일 표가 등장하는
`20260824_..._rootcause_compare.md` §20-7/§23/§24-5는 그 앞선 절(§3.3/§3.4)에서 "이 표는
이후 §3.4/§4에서 최종본으로 갱신됨"이라는 흐름으로 서술해 중복 나열을 피했다.

## 3. 고유 정보 보존 점검 (절 대응만으로 주장하지 않음 — 개별 재확인)

아래는 절 대응표와 별개로, **실제로 값이 옮겨졌는지** 원문과 통합본을 다시 대조한 표본
점검이다(전수 검사는 아니며, 통합 과정에서 위험도가 높다고 판단한 항목 위주).

| 항목 | 원문 값 | 통합본 위치 | 확인 |
|---|---|---|---|
| Rear 공식 acceptance(mAP50,std+ultra) | v01_symmetric +0.41%, v02_dair -1.22% | [01](01_rear_quantization_and_release.md) §0 | 일치 |
| Rear 참고 mAP50-95(std+ultra) | v01_symmetric -0.62%, v02_dair -0.70% | [01](01_rear_quantization_and_release.md) §0 | 일치 |
| Rear 후반 표의 다른 값 | v01_symmetric +0.45%, v02_dair -1.23%(릴리즈 스펙 표, 독립 재계산) | [01](01_rear_quantization_and_release.md) §0, §5 | 일치, 두 값 다 병기하고 임의 통일하지 않음 |
| 전체/공통 GT 장수 | 7,894장 / 6,152장(1,742장 board 미처리, 원인불명) | [01](01_rear_quantization_and_release.md) §1 | 일치 |
| v01_symmetric_matched 실험 confound | TensorSpeakerOverlayAug가 symmetric 트레이너 크래시로 구조적으로 함께 빠짐(loss.py:428) | [01](01_rear_quantization_and_release.md) §3.2 | 일치 |
| optimizer ablation BN 이상치 표(7개 모델) | symmetric1/matched7/v02 6+Inf1/v01_dair5+Inf1/실험①1/실험②10+Inf2/v02_dair 2 | [01](01_rear_quantization_and_release.md) §3.4 | 일치 |
| Softmax 증폭 가설 반증(B vs C cos=1.00000) | §18-3 | [01](01_rear_quantization_and_release.md) §3.3 | 일치 |
| Front source checkpoint 탐색 범위/미탐색 범위 | Scope A=B(3개 후보), 62개 `.pt` 미탐색 | [02](02_front_reference_and_reproduction.md) §2 | 일치 |
| Front vs best.pt 5-level 판정 + 최초 divergence(node 0) | VB 전체 | [02](02_front_reference_and_reproduction.md) §1 | 일치 |
| Front calibration 500장 NOT FOUND + 대체 세트 | LR | [02](02_front_reference_and_reproduction.md) §3 | 일치 |
| Front golden reference mAP50-95 +10.31%(근사) | FG §10/§18 | [02](02_front_reference_and_reproduction.md) §5 | 일치 |
| Front v01_repro 보드 실측 -29.5%(mAP50-95) | PN 하단 표 | [02](02_front_reference_and_reproduction.md) §7 | 일치, §5의 +10.31%와 혼동 금지 문구 포함 확인 |
| Corner 6-level 판정 + PT-ONNX 3-way bit-exact | CV 전체 | [04](04_corner_reproduction.md) §1 | 일치 |
| Corner NVT bin 4bytes 차이(4,703,568 vs 4,703,564) | CN | [04](04_corner_reproduction.md) §2 | 일치 |
| Side monkeypatch export 코드 스니펫 | §2 | [05](05_side_reproduction.md) §2 | 일치(코드 블록 그대로 보존) |
| Side 물리보드 99.53% + person 불일치율 0.23% | §11 | [05](05_side_reproduction.md) §11 | 일치 |
| Side NVT bin 비결정성(29 bytes / 2,273 bytes) | §4,§8,§11 | [05](05_side_reproduction.md) §4 | 일치 |
| SigmaStar 그래프 서저리 코드(ReduceMax/Unsqueeze/Concat/Transpose) | §1 | [06](06_sigmastar_conversion.md) §2 | 일치, opset13 Unsqueeze 함정 설명 포함 확인 |
| SigmaStar 최종 산출물 경로(.img) | §6 | [06](06_sigmastar_conversion.md) §7 | 일치 |
| Front 학습실험 640→512 재검증 함정(-p/-s 파일명 충돌) | §3-2 | [03](03_front_training_experiments.md) §3.2 | 일치 |
| Front standard NAS 실제배포 vs 우리export 가중치 다름(126/127) | §3-3 | [03](03_front_training_experiments.md) §3.3 | 일치 |
| Front v02_symmetric_bg epoch422 조기종료(정점219) | §1,§3-6 | [03](03_front_training_experiments.md) §1, §3.6 | 일치 |
| pattern-name 이슈(대소문자/길이, 시뮬레이터로 검출불가) | PN 전체 | [90](90_reproduction_reference.md) §5 | 일치 |
| NMS/evaluator 정의(dual-threshold vs 표준, pycoco vs ultra) | EG1 §4 | [90](90_reproduction_reference.md) §2 | 일치 |

**재현 명령/경로/해시**: 각 topic 문서에 원문의 재현 명령 블록을 축약 없이(경로 표기까지)
그대로 포함시켰다(01/02/04/05/06 각 문서에 최소 1개 이상의 `bash` 코드 블록 보존, 05는
전체 재현 명령 블록을 그대로 포함). SHA256/MD5 해시는 요약이 아니라 원문 값 그대로
인용했다(예: Front reference ONNX sha256 `a4e15dea...`, Corner best.pt sha256
`56d02250...`, Side best.pt sha256 `0c148dca...` 등) — 축약 표기 없이 전체 해시 문자열을
유지했다.

**실패/미확정 사항 보존 점검**: model.2/4 subset 탐색 미실행, REAR Experiment A(A1만
compile), DAIR strict on/off ablation 미실행(confound 잔존), ROI recall 경계 편향(미해결),
Side의 두 사용자 표 산출 조건 미확인, SigmaStar 500장 전체 재실행 미완료 — 전부 각 topic
문서의 "실패/미완료" 절에 조건→결과→배운점 형태로 남겼다(§8 실패시도표, 각 문서의 "다음
단계"/"알려진 한계" 절).

## 4. 주요 정정·상태 갱신·미해결 충돌

| 항목 | 내용 | 근거 |
|---|---|---|
| **상태 갱신**: "DAIR이 근본 원인" 가설 | 2026-08-12 강한 후보 → 2026-08-20 v01_symmetric_matched 실측으로 반증(오히려 DAIR 쓰는 쪽이 유리) → 2026-08-31~09-02 optimizer가 확정 원인 | [01](01_rear_quantization_and_release.md) §3.1~§3.4 |
| **상태 갱신**: "Softmax가 오차를 증폭한다" | 2026-08-24(§17) 잠정 결론 → 2026-08-25(§18) 3단계 분해로 반증/정정(오차는 pre-softmax 단계에서 이미 결정) | [01](01_rear_quantization_and_release.md) §3.3 |
| **상태 갱신**: Rear acceptance target | 2026-08-12 시점 "10% 이내" → 2026-08-19 "≤1%"로 상향(D-019) → 2026-09-02 지표를 mAP50+std+ultra로 확정 | [01](01_rear_quantization_and_release.md) §0, RG 상단 개정공지 |
| **정정**: 실험① NVT cosine "동등하거나 더 좋다" | 2026-08-31 서술(3개 중 3개 우위로 오기) → 2026-09-02 정정(Sigmoid는 근소 열세, 3개 중 2개만 우위) | [01](01_rear_quantization_and_release.md) §3.4 |
| **정정**: Side 불일치 66건 원인 | 최초 "threshold 경계선 근처"로 기록 → 재분석으로 42건(64%)은 "완전 미스"(NVT 모델이 아예 미탐지)로 정정 | [05](05_side_reproduction.md) §11 |
| **미해결 충돌(그대로 병기)**: Rear acceptance 수치 | +0.41%/-1.22%(RS/RC 인용치) vs +0.45%/-1.23%(릴리즈 스펙 표 독립 재계산) — 계산 조건 동일성이 원문에 명시되지 않아 통합자가 임의로 통일하지 않음 | [01](01_rear_quantization_and_release.md) §0 |
| **미해결 충돌**: Corner NVT bin 4bytes 차이 원인 | binary-level 추가 분석 없이 확정하지 않음(원문 자체가 미확정으로 남김) | [04](04_corner_reproduction.md) §2 |
| **미해결**: Side "person 불일치"의 정확한 원인 | Float ONNX 단계 표 비교에서만 불일치가 있고 실제 물리 보드에서는 재현 안 됨 — 두 사용자 표의 정확한 산출 evaluator/dataset는 끝내 미확인 | [05](05_side_reproduction.md) §6~§10 |
| **미해결**: Rear ROI recall 경계 편향 | 좌측 근접 대형차가 y=380 경계에 반복적으로 잘리는 패턴 발견, 결론으로 격상하지 않음 | [01](01_rear_quantization_and_release.md) §5 |

## 5. 원본 무변경 확인 및 링크 유효성 검사

- **원본 무변경**: 이 통합 작업에서 읽은 18개 입력 파일 전부를 원래 호스트 경로
  (`/home/ncn/Workspace/jdh/ultralytics_projects/docs/debug/`)의 파일과 md5sum으로
  재대조한 결과 **전부 동일**(불일치 0건) — 통합 작업 중 원본을 수정/삭제/이동하지
  않았음을 확인했다.
- **로컬 링크 유효성**: `consolidated/` 내부의 마크다운 상호 링크(`[텍스트](파일.md)`
  형태)를 전수 검사한 결과, 이 문서(99) 작성 전 시점에 README.md의 `99_merge_audit.md`
  링크 2건만 "파일 없음"으로 잡혔다 — 이 문서를 작성함으로써 해소됐다. 그 외 01~06/90
  사이의 상호 링크는 전부 유효하다.
- **범위 밖 참조**: 각 topic 문서 본문에서 백틱으로 인용한 원문 파일명(`20260806_..._
  quant_gap.md` 등 18개)은 마크다운 링크가 아니라 **출처 인용**이며, 실제로는 이
  `consolidated/`의 상위 디렉터리(`docs/debug/`)에 그 파일들이 존재한다(§1에서 확인).
  `20260729_config_canonicalization.md`([90](90_reproduction_reference.md) §6에서 언급)는
  입력 범위 밖 문서로 별도 표시했다. `docs/specs/qat_mixed_precision.md`,
  `docs/guides/rear_truck_train_validate_nvt_pipeline.md`, `docs/decisions.md`,
  `docs/glossary.yaml`, `configs/experiments/BSD_F606/Rear_Truck/README.md`,
  `configs/releases/BSD_F606/Front_Truck/v01/MANIFEST.yaml`, `standard/validation.md` 등
  코드/설정 경로 참조는 이 통합 작업의 입력 파일이 아니므로 "현재 입력 범위에서 확인 불가"로
  [90](90_reproduction_reference.md) §6에 명시했다.

## 6. 파일 수 · 바이트 수 · 문자 수

측정 도구: `wc -c`(UTF-8 바이트), `wc -m`(문자 수, 로케일 UTF-8 기준). **토큰 수는 이
환경에 사용 가능한 토크나이저(tiktoken/transformers 등)가 없어 측정하지 않았다** — 추정치를
실측처럼 기재하지 않는다.

| 구분 | 파일 수 | 바이트(UTF-8) | 문자 수 |
|---|---:|---:|---:|
| **입력(원본 18개)** | 18 | 519,364 | 371,051 |
| **출력 전체(consolidated/, README+01~06+90, 99 제외 — 이 표 작성 시점)** | 8 | 183,285 | 128,594 |
| **출력 중 일상 열람분(README+01~06, 90/99 제외)** | 7 | 163,578 | 114,975 |
| **90(재현 부록, 필요시만 열람)** | 1 | 19,707 | (바이트만 측정) |

- **99(이 문서) 자체는 위 "출력 전체" 집계에 포함되지 않았다** — 이 문서를 쓰는 도중에는
  자기 자신의 최종 크기를 측정할 수 없기 때문이다(작성 완료 직후 갱신 측정을 §7에 별도
  기록).
- **압축률 표현에 대한 주의**: 위 "출력 전체"(8개 파일, 90 포함)와 "일상 열람분"(7개 파일,
  README+01~06)을 구분한 이유는, 감사 문서(99)와 재현 부록(90)을 포함한 수치를 "일상적으로
  읽는 분량"으로 착각하지 않게 하기 위함이다. **원본(519,364 bytes) 대비 일상 열람분
  (163,578 bytes)의 축소율은 약 68.5%**이나, 이는 §7에서 99를 더한 최종 "출력 전체" 기준이
  아니라 90/99를 제외한 수치이므로 "전체 압축률"이라고 표현하지 않는다.

## 7. 최종 크기 재측정 (99 작성 완료 후)

이 절은 99 작성 완료 직후, 실제 최종 상태를 다시 측정해 채워 넣는다(§6은 99 작성 전
스냅샷이므로 두 수치가 정확히 같지 않을 수 있다 — 아래가 최종 정본).

| 구분 | 파일 수 | 바이트(UTF-8) | 문자 수 |
|---|---:|---:|---:|
| **입력(원본 18개, 원본 위치 기준, 변경 없음)** | 18 | 519,364 | 371,051 |
| **출력 전체(consolidated/, 99 포함)** | 9 | 201,101 | 141,152 |
| **출력 중 일상 열람분(README+01~06, 90/99 제외)** | 7 | 163,578 | 114,975 |
| **90+99(재현 부록+감사, 필요시만 열람)** | 2 | 37,523 | 26,177 |

- **로컬 링크 재검사(99 포함, 최종)**: consolidated/ 내 전체 마크다운 링크 전수 재검사
  결과 BROKEN 0건 — README→99 링크를 포함해 01~06/90/99 상호 링크 전부 유효.
- **일상 열람분 대 원본 비율**: 163,578 / 519,364 ≈ **31.5%**(즉 원본의 약 68.5% 분량
  축소). 이는 90/99를 제외한 "일상 열람" 기준이며, 90/99를 포함한 출력 전체(201,101
  bytes)는 원본의 약 38.7%다. **"고유 정보 손실 없음을 보장한다"는 의미가 아니라, §3의
  표본 점검에서 확인된 항목들이 보존됐다는 것과, §3에서 점검하지 않은 세부 수치(예: 원문의
  모든 개별 class별 표, 모든 재현 명령의 모든 인자)까지 전수 대조하지는 않았다는 한계를
  함께 명시한다.**
- **토큰 수**: 위와 동일하게 이 환경에 사용 가능한 토크나이저가 없어 측정하지 않았다.

## 8. 검증 한계 — 명시적으로 확인하지 못한 것

- §3의 "고유 정보 보존 점검"은 **표본 점검**이다 — 18개 입력 파일의 모든 표/모든 수치를
  한 줄씩 원문과 diff하는 전수 대조는 수행하지 않았다. 특히 다음은 통합본 작성자(나 자신)의
  주의로 옮겼다고 판단했으나 재대조 스크립트로 재확인하지는 않았다: (1) 각 topic 문서
  본문의 표 안 개별 숫자 하나하나, (2) 원문의 모든 코드 블록/재현 명령의 인자 하나하나.
- 원문에 있던 **경로/해시 표기**는 옮기며 줄바꿈·공백을 정리했을 수 있으나, 문자열 자체
  (경로, sha256/md5 값)를 축약하거나 임의로 바꾸지는 않았다 — 다만 이것도 표본 점검(§3)
  수준의 확인이며, 전수 문자 단위 대조는 아니다.
- pattern-name 이슈([90](90_reproduction_reference.md) §5)의 "보드 로더가 pattern-name을
  고정 길이 필드로 다루거나 대소문자를 구분해서 매칭할 가능성" 같은 **원문 자체의 추정
  서술**은 추정임을 명시한 문구를 그대로 유지했다 — 통합자가 이를 확정 사실로 격상하지
  않았음을 재확인.
- 이 통합 작업 자체는 문서 재구성에 한정됐고, 문서에 기록된 재학습·재변환·보드 테스트·
  커밋·배포는 실행하지 않았다(사용자 지시 §4 준수).

**결론적으로, "정보 손실 0"을 무조건 보증하지는 않는다** — §3~§4의 표본 점검과 §5의 원본/
링크 검증은 완료했으나, 전수 대조는 수행하지 않았으므로 이 한계를 명시적으로 남긴다.
