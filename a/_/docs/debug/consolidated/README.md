# BSD_F606 디버그 문서 통합본 — 안내

`docs/debug/` 원본 18개 마크다운(2026-08-06~2026-09-14)을 주제별로 재구성한 문서 세트다.
원본은 수정하지 않았다 — 이 폴더가 새 열람 지점이다. 감사/검증 상세는
[99_merge_audit.md](99_merge_audit.md)(평상시 읽을 필요 없음).

## 지금 가장 중요한 결론 3가지

1. **Rear_Truck 릴리즈 후보는 `v01_symmetric`으로 확정됐다(2026-09-03, 사용자 지시)** —
   공식 acceptance 지표(mAP50 + 표준 단일-IoU NMS + ultralytics evaluator = "std+ultra")
   기준 손실 **+0.41%(개선)**로 ≤1% target을 통과한다. 경쟁 후보 `v02_dair`는 같은 기준
   -1.22%로 근소 미달. **단, 이는 "후보 확정"이지 "릴리즈 완료"가 아니다** — README/MANIFEST
   반영 등 후속 작업이 문서 작성 시점(2026-09-02) 기준 미실행이었다. → [01](01_rear_quantization_and_release.md) §0, §5
2. **Rear_Truck 양자화 손실의 근본 원인은 optimizer 선택(SGD vs AdamW)이다** —
   서로 다른 두 축(optimizer 단독 ablation, trainer 단독 ablation)의 순수 실험이 같은
   결론으로 수렴했다. "DAIR이 원인"이라는 과거 가설과 "AdamW가 일반적으로 나쁘다"는 확대
   해석 둘 다 명시적으로 반증/기각됐다. → [01](01_rear_quantization_and_release.md) §3
3. **NAS Release 산출물(.pt/.onnx)의 실제 재현성은 제품마다 다르다** — Corner_Truck/
   Side_Truck은 올바른 export 방법을 쓰면 float32 잡음 수준까지 재현되지만(사실상 동일
   모델), **Front_Truck은 그래프는 같아도 파라미터가 다른, 서로 다른 모델**이고 source
   checkpoint는 조사 범위 내에서 찾지 못했다(NOT FOUND). → [02](02_front_reference_and_reproduction.md) §0~§2, [04](04_corner_reproduction.md) §0, [05](05_side_reproduction.md) §0

## 질문별로 어디를 읽을지

| 질문 | 읽을 문서 |
|---|---|
| Rear_Truck 양자화 손실이 왜 작거나 큰지, 원인이 뭔지 | [01](01_rear_quantization_and_release.md) §1~§4 |
| Rear_Truck 릴리즈 후보를 왜 그렇게 정했는지, 남은 리스크는 | [01](01_rear_quantization_and_release.md) §5 |
| pycocotools vs ultralytics로 손실 %가 왜 이렇게 다르게 나오는지 | [01](01_rear_quantization_and_release.md) §6, [90](90_reproduction_reference.md) §2 |
| Front_Truck NAS 배포 산출물이 이 repo의 `best.pt`와 같은 모델인지 | [02](02_front_reference_and_reproduction.md) §1~§3 |
| Front_Truck의 학습 실험(해상도/from-scratch/DAIR/배경 데이터)을 비교하고 싶을 때 | [03](03_front_training_experiments.md) |
| Corner_Truck reference ONNX 재현이 되는지 | [04](04_corner_reproduction.md) |
| Side_Truck 전체 파이프라인(PT→ONNX→NVT→보드)이 재현되는지, "person 불일치" 미스터리 | [05](05_side_reproduction.md) |
| SigmaStar(IPU) 디바이스로 변환하는 방법과 겪은 실패 | [06](06_sigmastar_conversion.md) |
| NVT 컴파일 설정, evaluator/NMS 정의, board pattern-name 문제, 재현 명령 템플릿 | [90](90_reproduction_reference.md) |

## 판단 성격 구분 — 이 문서 세트 전체에서 지키는 표기 원칙

- **실측(measured)** / **사용자 결정(user decision)** / **가설(hypothesis)** /
  **반증·대체(refuted/superseded)** / **미확인(unconfirmed)**을 각 topic 문서 본문에서
  구분해 표기했다. "확정된 결론"으로 적힌 것과 "당시 시점 가설이었다가 이후 반증됨"으로
  적힌 것을 섞어 읽지 말 것 — 특히 [01](01_rear_quantization_and_release.md)은 조사가
  여러 차례 자기 수정을 거쳤으므로 §3의 연대기 순서를 따라 읽는 것을 권장한다.
- **작업 상태**는 완료/보류/미완료로 별도 표시했다. 예: Rear 릴리즈는 "후보 확정"이지
  "완료"가 아니고, SigmaStar는 "변환 성공"이지 "디바이스 실측 완료"가 아니며(15장 샘플
  기준), Front의 NPU 변환은 "1장 기준 compile/simulate/visualize 완료"이지 "board 실측
  완료"가 아니다.
- **원문 자체의 과도한 인과 해석·일반화**는 관측값과 분리해 표시했다(예: "AdamW가 나쁘다"가
  아니라 "이 데이터셋/조합에서 AdamW가 SGD보다 불안정했다").

## 이 통합본이 다루지 않는 것

- 이 통합본 자체가 문서에 적힌 과거 지시·재학습·변환·보드 테스트·커밋·배포를 실행한 결과는
  아니다 — 원문에 기록된 실측/실험 결과를 재구성했을 뿐이다.
- `docs/debug/` 폴더에는 이 통합 범위(2026-08-06 이후 18개) 밖에 6개 문서
  (QAT 코드 리뷰, ncnvox KWS 데이터 인벤토리, Phase D/E/F, config canonicalization)가 더
  있으나, 트럭 검지(Rear/Front/Corner/Side/SigmaStar) 주제와 무관한 별개 프로젝트(키워드
  스포팅/오디오, 범용 설정 정본화)로 확인돼 이번 통합에서 제외했다(사용자 확인,
  [99](99_merge_audit.md) §1 참조).

## 남아있는 미해결/미확정 사항 (요약, 상세는 각 topic 문서)

- Side_Truck의 "person 불일치"를 만든 두 사용자 제공 표(Board표/Original표)의 정확한
  evaluator/dataset/threshold 조건 — 실제 물리 보드 실측(99.53% 일치)으로 파이프라인
  자체는 문제없음이 확인됐으나, 그 두 표 자체의 산출 조건은 끝내 확인되지 않았다.
- Rear_Truck의 "ROI 인식 영역" recall 탐색은 경계 편향(좌측 근접 대형차 배제 가능성)이
  발견돼 결론으로 격상하지 않았다.
- Rear_Truck의 optimizer→BN 불안정성 메커니즘 자체(momentum/adaptive lr과의 상호작용)는
  전혀 조사되지 않았다.
- 두 릴리즈 후보의 acceptance 수치가 원문 내에서 소수점 차이로 두 번(+0.41%/-1.22% vs
  +0.45%/-1.23%) 등장한다 — 임의로 통일하지 않고 병기했다([01](01_rear_quantization_and_release.md) §0).
