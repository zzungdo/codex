# Rear_Truck — 양자화 손실 원인 규명과 릴리즈 후보 선정

> 원문: `20260806_rear_truck_v01_npu_quant_gap.md`(RG), `20260814_rear_truck_v01_symmetric_reexport_nvt_revalidation.md`(RX),
> `20260824_symmetric_low_quant_loss_rootcause_compare.md`(RC), `20260824_v01_symmetric_eval_gap.md`(EG1),
> `20260824_v01_dair_v02_v01_symmetric_matched_eval_gap.md`(EG2), `20260831_symmetric_quant_loss_rootcause_summary.md`(RS),
> `20260902_rear_truck_release_candidate_selection.md`(RSel), `20260819_front_golden_reference_for_rear_1pct.md`(FG, Rear 실험 A·과거 가설 부분만).
> 공통 재현 환경·evaluator 정의·재현 명령 템플릿은 [90_reproduction_reference.md](90_reproduction_reference.md) 참조.
> SigmaStar(IPU) 변환은 [06_sigmastar_conversion.md](06_sigmastar_conversion.md)로 분리.

## 0. 현재 상태 요약 (가장 최신 판단부터)

- **공식 acceptance 지표 (2026-09-02 확정, RSel §1 사용자 지시)**: **mAP50 + 표준 단일-IoU NMS
  + ultralytics evaluator(`ap_per_class`)** — 이하 "std+ultra". pycocotools/dual-threshold NMS
  기준 값은 참고용으로만 남기고 acceptance 판정에는 쓰지 않는다. acceptance target은 **≤1%**
  (`docs/guides/rear_truck_train_validate_nvt_pipeline.md` §13, `docs/decisions.md` D-019,
  2026-08-19).
  - `v01_symmetric`: std+ultra mAP50 손실 **+0.41%**(개선, RC §23) — **통과**.
  - `v02_dair`: std+ultra mAP50 손실 **-1.22%**(RC §24-4) — 1% 허용치를 0.22%p 초과, **미달**.
  - 참고 mAP50-95(std+ultra): `v01_symmetric` **-0.62%**, `v02_dair` **-0.70%**(RC §21-3/§24-5).
  - **정정 불가한 수치 차이**: 릴리즈 스펙 문서(RSel §3-7, 클래스별 P/R/mAP 표에서 독립
    재계산)에는 같은 두 모델의 mAP50 all-row 손실이 **`v01_symmetric` +0.45%, `v02_dair`
    -1.23%**로 기록돼 있다. §1의 +0.41%/-1.22%(RS/RC 인용치)와 소수점 단위로 다르다 — 두
    수치 모두 원문에 그대로 있고, 계산 스크립트/반올림 조건이 동일하다는 근거가 없어
    **임의로 하나로 통일하지 않는다**. 두 값 다 "SGD 계열은 통과, v02_dair는 근소 미달"이라는
    결론은 동일하게 가리킨다.
- **릴리즈 후보 확정 (2026-09-03, 사용자 지시, RSel §4/§5)**: `v01_symmetric`을 릴리즈 후보로
  확정하고 릴리즈 절차 진행을 지시받음. **이것은 "후보 확정"이지 "릴리즈 완료"가 아니다** —
  RSel §6 기준 README/정본 문서 최신화, `configs/releases/`로 복사 + MANIFEST 작성,
  `ruff check && pytest` + 커밋이 **아직 미실행**으로 남아 있었다(RSel 작성 시점, 2026-09-02).
- **Root cause (최종, RS §7)**: **optimizer 선택(SGD vs AdamW)**이 학습→PT→NVT→Board 전
  파이프라인에서 재현되는 가장 강한 원인. `v01_symmetric_matched`가 제기했던 "DAIR을 빼면
  개선된다"는 과거 가설은 **반증**됐고(§3), "Softmax가 오차를 증폭한다"는 과거 가설도
  **반증/정정**됐다(§4). 상세 인과 사슬은 §3~§5.

---

## 1. 데이터셋 규모와 공통 평가 집합

- Rear_Truck 원본 validation split: **7,894장**.
- 이 중 **1,742장은 물리 NVT Board에서 처리되지 않아 board 추론 결과 자체가 없다** — 원인은
  **확인되지 않았고, 추정하지 않는다**(RG, 2026-08-12 추가 주석).
- 나머지 `7,894 − 1,742 = 6,152장`이 **Float ONNX와 Board 양쪽에서 결과를 얻을 수 있는 공통
  집합**이며, 임의 sampling이 아니다. 이 문서의 quantization loss 표는 전부 이 6,152장
  기준이다(별도 표기 없는 한).
- **Float 기준 표에 따라 7,894장(project 표준 val, `DairDetectionValidator` 내부 검증)과
  6,152장(dual-NMS/ONNX apples-to-apples)을 혼동하지 말 것** — RG는 이 둘을 항상 별개
  섹션으로 유지했다(§ "v02 Float Validation Baseline" vs "v02 Float ONNX Baseline").

## 2. 공식 acceptance 축과 원인 축의 분리

이 조사에는 서로 **독립적인 두 개의 축**이 있고, 절대 섞어서 해석하면 안 된다(RC §21-6, RS §8):

1. **Evaluator/NMS 축** — 같은 모델·같은 board 예측이라도 평가 방식(pycocotools dual-threshold
   NMS vs ultralytics 표준 단일-IoU NMS)에 따라 손실 %가 크게 달라지는 현상. 원인 규명이
   아니라 **측정 방법론** 문제다. → §6.
2. **Optimizer/원인 축** — 동일 evaluator 안에서도 모델(학습 조건)에 따라 board degradation
   크기 자체가 다른 현상. → §3~§5.

## 3. Root cause 조사 연대기 (가설 변화 이력)

### 3.1 최초 손실 발견과 augmentation 가설 (2026-08-06~08-12, RG)

**질문 1 — 양자화 손실 실측** (같은 6,152장 GT, dual-threshold NMS로 순수 양자화 손실만 비교):

| 모델 | Float ONNX mAP50-95 | Board mAP50-95 | 절대 손실 | 상대 손실 |
|---|---:|---:|---:|---:|
| Production(`best_REAR_320x320_DAIR_postprocess.onnx`, `ultralytics_projects-main`) | 0.521 | 0.435 | -0.086 | -16.5% |
| v01 재현(이 repo, 3-tensor baseline) | 0.543 | 0.363 | -0.180 | **-33.1%**(production의 약 2배) |

기준: `docs/specs/qat_mixed_precision.md`의 ">2% 드랍이면 QAT 필요" — 둘 다 초과.

**질문 2 — 통합 파이프라인(v01) vs 이전 비통합 환경(production)**: Float 기준 v01(+4.2%,
0.543 vs 0.521)이 production보다 **더 좋음**(학습 파이프라인 자체는 문제 없음). Board 기준은
v01이 훨씬 나쁨(-16.6%p 격차) → 문제는 학습이 아니라 **NPU 변환 설정**(calibration 724장·
spec_demarc_tensors가 production 모델용으로 튜닝된 것을 그대로 재사용)으로 추정. 근거:
시뮬레이터 cosine(0126.jpg 기준) — production은 backbone(model.0~model.9)이 거의 무손실인데
v01은 **model.2, model.4의 첫 C2f 블록부터** 눈에 띄게 나쁨(`Conv__model.2_m.0_cv2` 0.996→0.966
등). `[OUT]` 텐서: DFL Softmax 0.9630→0.9185, Sigmoid 0.9815→0.9423, dfl_conv 0.9888→0.9719.

**부록 — calib_mode 시도(둘 다 실패)**: PERCENTILE(0.999, 기존·최선) Sigmoid 0.9423/dfl_conv
0.9719/Softmax 0.9185 > MAX(0.7638/0.9189/0.8251) > KLD(0.7127/0.9178/0.8287). **calib_mode를
바꾸는 방향은 틀렸다** — PERCENTILE(0.999) 유지가 최종 결정.

**2026-08-10~11 추가 조사(전부 시도 → 배운 점/배제 형태로 압축)**:

| 시도 | 조건 | 결과 | 배운 점 |
|---|---|---|---|
| calibration 재구성 | 500장(production 재사용)→724장(연속 프레임, 저다양성)→940장(주간26+야간21, 영상당20장 균등, NAS_1) | cls/dfl cosine 0.9427/0.9691→0.9423/0.9719→0.9486/0.9672 (개선폭 미미) | calibration 다양성은 주 원인이 아님 |
| percentile 0.995 | — | cls 0.9178/dfl 0.9630 (악화) | 0.999 유지 |
| Front production config 이식(CLE off, `cal_refine_iter=0`, `bnscale_en=0`) | — | cls 0.9323/dfl 0.9533 (악화) | 타 카메라 config 이식은 역효과 |
| 538칩 mixed-precision sensitivity 분석(`--mp-method 0/1`) | 원본 538 config 미보유, 530 config 이식 | `sorted_sensitivity.json` 생성 안 됨, 정보 획득 실패 | 폐기(`run_compiler_538.sh`만 잔존) |
| **실측 기반 spec_demarc_tensors(model.2/4, 9텐서)** | 기존 3텐서(H1-3)+실측 최악 6개(T1-6, model.2/4의 C2f 블록) | cls 0.9869/dfl 0.9809 (9텐서 historical 대비 대폭 개선) | **돌파구** — 6,152장 재확인: v01(9텐서historical) 0.543/0.368/-32.2% → **v01(model.2/4 실측9텐서) 0.543/0.435/-19.9%, production과 동일**. threshold 0.45로 올려도 격차 유지(확인 완료, 계산 방식 때문 아님). |

**그러나 자원 비용 2배(RG §4)**: production(9텐서, 헤드 근처 model.18/19/21/22) 6.25ms/35.0MB/
4.70MB vs v01 3텐서 baseline 6.37ms/34.9MB/4.71MB vs **v01 model.2/4 9텐서 12.25ms/78.5MB/
9.39MB**. 원인: 보호 텐서가 backbone 초반부라 16bit 전파 범위가 넓어짐(production 보호
텐서는 헤드 근처라 전파가 짧게 끝남). Rear_Truck 실사용 목표 10~15fps, NPU를 Front/Side/DMS와
공유 — **이 2배 비용은 가볍게 넘길 사안이 아니다**. **결론: board mAP는 production 수준으로
맞췄지만 근본 해결이 아니라 하드웨어 자원을 2배 써서 보완한 것.**

**코드 레벨 원인 조사(production 학습 환경 `ultralytics_projects-main` vs 현재 repo, RG §5)**:
`loss.py`/`augment.py`/`symmetric_yolov8.yaml`/`metric.py`/`cfg.py`/`dair.py`(val/model) 차이
없음(whitespace/주석뿐). `tal.py`: `bg_idx` 버그(이 repo로 이전 시 `self.bg_idx = num_classes`
줄이 삭제됐던 것, 세션 중 수정 완료, commit `a713f73`). **`train/dair.py`: `TensorSpeakerOverlayAug`
— production은 주석 처리(비활성), v01은 활성 상태로 학습됨. 유일한 실질적 학습 코드 차이.**

**v02 ablation**: `TensorSpeakerOverlayAug`를 주석 처리(production과 동일)하고 나머지는 v01과
동일 재학습(epoch333에서 patience 정상 종료). **결과(RG "v02 Float ONNX Baseline" §5-1)**:
Float 0.536, Board mAP50-95 0.435, **손실 -18.8%**(기존 v01 -33.1%에서 격차의 대부분을
좁힘, production -16.5%에 근접, model.2/4 9텐서 보호 없이 저비용 3텐서 baseline 그대로).
**TensorSpeakerOverlayAug 가설: 지지(강함), 그러나 완전 해결은 아님**(-2.3%p 잔여 격차).

**v01_dair / v01_symmetric 외부 학습 비교(다른 PC에서 완료된 실험, RG "Rear_Truck 외부 학습
비교" §1~§6)**: 두 모델 다 이 repo 기존 v01과 동일 데이터/split(사용자 확인, 파일로 재검증
불가 — `bsd_rear_truck.yaml`이 이 PC에 없음). 실제 구성(코드/args.yaml 실측, 추측 아님):

| | v01_dair | v01_symmetric |
|---|---|---|
| Trainer/Validator | `DairDetectionTrainer`/`DairDetectionValidator`(독립 x/y gain) | `SymmetricTrainer`(super().__init__ 미호출)/`SymmetricDetectionValidator`(균일 gain, x/y 독립 아님 — REAR는 1920x1080→320x320 비정방 리사이즈라 잠재 편향 요인이나 ONNX+dual-NMS 평가는 영향 안 받음) |
| optimizer/lr0/batch | AdamW/0.001/32 | SGD/0.01(from-scratch 기본)/48, cos_lr=true |
| dair_params | mode=or, λ=0.1, γ=0.5 | 없음 |
| 학습 종료 epoch | 206(500중) | 407(500중) |

Internal Float(각 모델 내부 파이프라인, 7,894장): v01_dair mAP50-95 **0.5414**, v01_symmetric
**0.5681**. ONNX+dual-NMS(6,152장): production 0.521/기존v01 0.543/v02 0.536/**v01_dair
0.531**/**v01_symmetric 0.561**(지금까지 측정된 모든 Rear_Truck float 결과 중 최고).

**v01_dair board(RG "비교 표 — 최종" §4, 최초 생략 판단 → 철회 → 실측 완료)**: 처음엔
"기존 v01과 사실상 같은 recipe 계열"이라 board 측정을 의도적으로 생략(TBD 아니라 확정된
결정으로 문서화)했으나, 이후 사용자가 실측 덤프를 추가 확보해 진행. 1차 덤프는 v02와
6,152장 detection이 픽셀 단위로 완전히 동일(stale-model, 무효 처리) → 재덤프(jaccard≈0
확인) → **Board mAP50 0.669/mAP50-95 0.421, 절대손실 -0.110, 상대손실 -20.7%**. 생략 판단의
전제("동일 recipe면 board 손실도 비슷할 것")가 실측과 어긋남(기존 v01 -33.1%보다 12.4%p
더 좋고 오히려 v02 -18.8%에 근접) — **판단이 틀렸다기보다 가정이 부정확했다**는 것으로 기록.
같은 recipe 계열이라도 run-to-run 변동성이 수십 %p 갈릴 수 있다는 새 관찰.

**v01_symmetric board(RG §5-1, 재덤프)**: 1차 덤프 무효(v02와 픽셀 단위 완전 동일) → 재덤프 →
**Float 0.561, Board mAP50 0.751/mAP50-95 0.521, 절대손실 -0.040, 상대손실 -7.1%** — 당시
측정된 모든 모델 중 손실 최소, 당시 target(10% 이내)을 최초 달성.

**cosine 예측 실패 사례(RG, 중요한 방법론 경고)**: 3-tensor NVT 시뮬레이터 cosine 서열은
`v01_symmetric(0.985/0.982/0.993) > v01_dair(0.926/0.952/0.974) > v02(0.906/0.924/0.966)`
였으나, 실제 board 손실 순위는 `v01_symmetric > v02(-18.8%) > v01_dair(-20.7%)`로
**v01_dair/v02의 순서가 뒤바뀌었다** — "single-image cosine은 진단 신호일 뿐 최종 성능
지표가 아니다"라는 원칙(SOP `docs/guides/rear_truck_train_validate_nvt_pipeline.md` §8)의
실측 재확인.

**2026-08-12 시점 최종 결론 요약(당시 target=10% 이내 기준, 이후 §3.2에서 갱신)**: 순위
`v01_symmetric(-7.1%) > production(-16.5%) > v02(-18.8%) > v01_dair(-20.7%) >`(이후 추가된)
`v01_symmetric_matched(-28.15%) > 기존v01(-33.1%)`. 당시 "강한 후보/단서"였던 **"DAIR
dual-forward 학습 방식 자체가 TensorSpeakerOverlayAug보다 더 근본적인 원인일 가능성"**은
**2026-08-20 v01_symmetric_matched 실측으로 반대 방향임이 확정**됐다(§3.2).

> **⚠ 문서 자체의 target 개정 공지(RG 상단, 2026-08-19 삽입)**: 위 "10% 이내 달성"·"목표
> 달성" 서술은 2026-08-12 당시 target 기준이다. acceptance target이 이후 **≤1%**로 상향됐고
> (D-019), 새 기준으로는 RG에서 측정된 5개 모델 전부(최선인 v01_symmetric -7.1%포함, 이
> -7.1%는 dual+pycoco 기준 — std+ultra 기준으로는 §0의 +0.41%) 미달이다. RG 본문 수치는
> 조사 당시 기록 그대로 보존됐다 — 현재 유효 target과의 비교는 항상 RS/RSel(§0)을 기준으로
> 판단할 것.

### 3.2 `v01_symmetric_matched` strict ablation — DAIR 가설 반증 (2026-08-19~20, RG)

**설계**(RG "v01_symmetric_matched ablation" §1, 학습/export/NVT는 2026-08-13에 이미 완료,
문서화만 2026-08-19~21에 진행): `v01_symmetric_matched`는 `v01_dair`의 optimizer=AdamW/
lr0=0.001/lrf=0.01/batch=32/cos_lr=false를 그대로 복사하고 **Trainer만 symmetric으로 교체**
(DAIR dual-forward 없음). **잔여 confound**: symmetric 트레이너에 `dair_params`(및 그에
딸린 `TensorSpeakerOverlayAug`)를 넣으면 `RuntimeError: expand`로 즉시 크래시(`projects/src/
utils/loss.py:428`, `fraction=0.01 epochs=1`로 재현) — 의도적으로 뺀 게 아니라 **뺄 수밖에
없었다**. 즉 "하이퍼파라미터는 동일, augmentation 축은 여전히 v01_dair에만 있음" 상태.

**결과(2026-08-20 board 실측 확정)**:

| | v01_dair(DAIR 사용) | v01_symmetric_matched(DAIR 미사용, 나머지 동일) |
|---|---:|---:|
| Float mAP50-95 | 0.531 | 0.547 |
| Softmax(dfl)/Sigmoid/dfl_conv cosine | 0.926/0.952/0.974 | 0.913/0.947/0.949 |
| Board mAP50-95 | **0.421** | 0.393 |
| Quant loss | **-20.7%** | **-28.15%** |

**DAIR을 빼는 것 자체는 quantization을 개선하지 않았다 — 오히려 7.45%p 악화시켰다**(§5의
과거 가설과 반대 방향). 반면 `matched → symmetric`(하이퍼파라미터만 SGD/lr0=0.01/batch48/
cos_lr=true로 변경, DAIR 미사용 유지) 구간은 -28.15%→-7.1%, **+21.05%p 개선**(DAIR 축의
거의 3배 크기). **확정된 결론(당시)**: `v01_symmetric`이 `v01_dair`보다 quantization을 훨씬
잘 버틴 주된 이유는 DAIR을 안 써서가 아니라 **optimizer/lr0/batch/cos_lr 차이 때문**이다.
DAIR 유무 축 자체는 중립이 아니라 **DAIR을 쓰는 쪽이 오히려 유리한 방향**이라는, 애초
가설과 반대되는 사실이 확정됐다(단, `TensorSpeakerOverlayAug` confound 잔존 — strict
on/off ablation은 symmetric 트레이너의 DAIR 크래시 버그 수정이 선행돼야 하며 **미실행**).

acceptance target 관련: 이 실험은 D-019(≤1% 상향) **이전**에 설계된 것이라 새 target 달성
여부와 무관 — `v01_symmetric_matched`(-28.15%)도 새 target에는 당연히 크게 못 미친다. 가치는
target 달성이 아니라 원인 분리에 있다.

### 3.3 전체 파이프라인 rootcause 비교와 layer-by-layer 분석 (2026-08-24, RC §1~§16)

`v01_symmetric`/`v01_symmetric_matched`/`v02` 세 모델을 학습 조건→PT→ONNX→NVT→Board 전
파이프라인에서 비교(evaluator 효과와는 별개 축 — pycoco/ultra 어느 쪽으로 봐도 순위는
동일: `symmetric ≪ v02 < matched`).

**새로 발견된 차이(RC §2)**: `cls`(classification loss weight)가 **v01_symmetric만 1.5**이고
matched/v02는 ultralytics 기본값 0.5(3배 차이) — 이전에 알려지지 않았던 차이. 단 §3.4에서
효과가 반증됨. 또한 "matched"라는 이름과 달리 실제로는 **v01_dair의 하이퍼파라미터를 복사한
것이지 v01_symmetric과 맞춘 것이 아니다.**

**배제된 것(byte-identical/구조 동일 확인, Category C)**: 아키텍처(파라미터 3,011,823/
state_dict 355키 전부 동일), ONNX export(235 node, skip_postprocess=True, opset13, 세 모델
완전 동일), PT↔ONNX numerical parity(통계적 구분 불가), NVT compile 설정(gen_config.txt
model_dir 외 byte-identical, calibration 940장 동일 symlink, spec_demarc H1-3 동일),
Board/postprocess 설정(pred.path 외 완전 동일).

**BatchNorm 통계(RC §3-1, 당시 가장 강한 단서)**: 57개 BN 레이어 중 `running_var>500`
이상치 레이어 수 symmetric **1**/matched **7**/v02 **6**, **Inf 포함 레이어**: v02만 **1개**
(`model.0.bn.running_var[4]` = Inf, `running_mean[4]=-666.5`). `model.0.bn.running_var`
최대(유한): symmetric 14,432/matched 58,880(약4배)/v02 46,176.

**Float 출력 분포(5장 표본, RC §6)**: cls(sigmoid) 분포는 세 모델 통계적 구분 불가(cls=1.5
오버라이드에도 뚜렷한 차이 없음). box(raw distance, DFL 이전) 텐서는 **v02가 가장 넓은
dynamic range**(std 1.528, symmetric 최소 1.328, matched 중간 1.375) — BN 발견과 방향 일치.

**171-layer `layer_diff.txt` 분석(RC §17, 2026-08-24)**: `model.0`(idx2,3) 자체는 세 모델
cosine 차이 0.0004 이내 — **"model.0부터 갈린다"는 기존 가설은 재현되지 않음(약화/반증에
가까움)**. **가장 크고 일관된 단일 cliff: `Softmax__model.22_dfl_Softmax`(idx196)** —
직전(idx192)까지는 완만(0.017~0.024 격차)하다가 이 지점에서 matched/v02만 급락(symmetric
0.9850 거의 그대로 vs matched **0.9126(-0.067)**, v02 **0.9058(-0.079)**). Detect head는
**cv2(box)뿐 아니라 cv3(classification)도 동등하거나 더 크게** divergence — "box branch
단독" 가설은 **부분적으로만 맞음, Detect head 전체(box+cls)로 범위 수정 필요**.

**DFL Softmax 3단계 분해(RC §18, 2026-08-25, §17 가설 정정)**: Softmax 직접 입력은 idx193
(`Transpose__model.22_dfl_Transpose`, idx192 아님 — 경로 수정). golden pre-softmax(idx193)
분포: logit std symmetric **3.146** vs matched **15.243**(5배) vs v02 **16.812**(5.3배),
min symmetric -4.6 vs matched -98.2(20배) vs v02 -169.3(37배). entropy는 matched/v02가
**더 낮음**(더 확신적/sharp) — "덜 뾰족해서 유리하다"는 가설과 **반대 방향**(엔트로피/뾰족함
자체는 원인으로 지지되지 않음). A(golden pre→PC softmax)/B(sim pre→PC softmax)/C(실제 sim
post-softmax) 분해: **B vs C는 세 모델 모두 cos=1.00000로 완벽 일치 — NVT Softmax 연산
자체는 정확하고 추가 오차를 만들지 않는다(확정, 반증).** 오차 증폭 비율(post/pre MAE)은
오히려 symmetric이 더 큼(0.061 vs matched 0.010/v02 0.012, softmax 포화 특성) — **"Softmax가
증폭한다"는 §17의 표현은 부정확했다, 오차는 이미 pre-softmax logit 단계에서 결정된다.**
DFL expectation decode(box distance) 오차: A-vs-C가 matched(0.369)/v02(0.343)는
symmetric(0.116)의 약 **3.0~3.2배** — 확률 오차 비율이 그대로 distance 오차로 이어짐.
**갱신된 결론**: "학습 recipe 차이가 Detect head quantization sensitivity를 다르게 만든다"는
상위 가설은 유지되나, 메커니즘은 "pre-softmax logit의 **dynamic range**(std/극단값)가
matched/v02에서 5~30배 넓어 그 텐서의 INT8 quantization 오차 자체가 크고, Softmax는 그
오차를 그대로(왜곡 없이) 통과시킬 뿐"으로 수정.

### 3.4 Optimizer 순수 ablation — 두 개의 독립 축으로 확정 (2026-08-31~09-02, RC §19~§24, RS)

**optimizer-axis(실험①/②, 둘 다 symmetric 트레이너, "160서버"에서 학습)**: `lr0=0.001`,
`cos_lr=true`, `batch=48`, `cls=1.5` 전부 고정, optimizer만 실험①=**SGD**/실험②=**AdamW**로
바꾼 순수 ablation. baseline(v01_symmetric, lr0=0.01)↔실험①은 "lr0 크기 단독 효과" 검증용.

PT(BN, torch.load 재확인): 이상치 레이어/Inf — 실험① **1/0**(v01_symmetric과 동일 건강 패턴,
**lr0를 10분의 1로 낮춰도 문제 재현 안 됨 → lr0 크기 자체는 원인 아님, 반증**), 실험②
**10/2(!)**(기존 최악이던 v02의 Inf 1개보다도 심함) — **optimizer 하나만 바꿨는데 BN
불안정성이 그대로 재현/악화**. `cls=1.5`는 실험①/②모두 동일한데 결과가 갈림 → **cls
loss weight 가설 약화**(단독 원인 아님). batch(48 고정)도 원인 아님.

NVT 3-tensor simulator cosine: v01_symmetric(baseline) 0.9850/0.9824/0.9933, **실험①
0.9955/0.9779/0.9977**(baseline과 동등하거나 더 좋음 — 단 Sigmoid는 3개 중 유일하게 근소
열세 0.9779<0.9824, "동등하거나 더 좋다"는 2026-08-31 서술은 2026-09-02 정정됨), **실험②
0.9170/0.7969/0.9650**(Sigmoid=0.7969, 전 모델 중 최저치, 기존 최저 v02의 0.9236보다도 나쁨).

Float ONNX(6,152장, 4-combo): 실험① 0.532/0.5551/0.528/0.5474(다른 모델과 비슷한 수준),
**실험②만 0.412/0.4399/0.403/0.4283 — 다른 모든 모델(0.53~0.58대)보다 뚜렷이 낮음**
(quantization 이전, float 단계부터 이미 저하). 원인 추정(실측): `model.0.bn.running_var`의
실제 Inf(채널 2개)가 float 추론에서도 그 채널을 0으로 만들기 때문(BN eval-mode 공식
`(x-mean)/sqrt(var+eps)`, `var=inf`→결과 0 — PyTorch/ONNX 공통 수학적 사실, float 추론
결과로 직접 재확인).

**Board 실측(6,152장, jaccard overlap으로 stale-check 완료)**:

| Model | Trainer | Optimizer | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---|---|---:|---:|---:|---:|
| v01_symmetric | Symmetric | SGD | -7.30% | -1.96% | -6.63% | -0.62% |
| 실험① sgd_lr0001 | Symmetric | **SGD** | -7.89% | -5.91% | -7.20% | -4.59% |
| v01_dair | DAIR | AdamW | -21.01% | -16.32% | -19.66% | -14.62% |
| v02 | DAIR | AdamW | -19.14% | -14.80% | -18.08% | -13.44% |
| v01_symmetric_matched | Symmetric | AdamW | -28.42% | -27.00% | -27.36% | -25.62% |
| **실험② adamw_lr0001** | Symmetric | **AdamW** | **-40.29%** | **-37.76%** | **-38.96%** | **-36.07%** |

실험②(mAP50-95 board(pycoco) **0.246**, 이 조사 전체 최저 절대 성능, Float 자체 저하 때문에
"quantization 단계에서만 나쁘다"고 서술하지 않음 — 학습 자체의 불안정성 + quantization 추가
손실의 결합). **결론(RC §20-8)**: 이 특정 조합(AdamW+lr0=0.001+cos_lr+batch48+cls1.5)은 학습
자체부터 불안정했고, 그 불안정성이 quantization 단계에서 한 번 더 증폭됐다 — "AdamW가
일반적으로 나쁘다"로 확대하지 않는다(lr0=0.001은 ultralytics AdamW 표준값이었다는 점에서
"이 데이터셋/조합에서" AdamW가 SGD보다 불안정했다로 한정).

**trainer-axis(`v02_dair`, 2026-09-02, 다른 학습 PC)**: v01_symmetric의 optimizer=SGD/
lr0=0.01/lrf=0.01/cos_lr=true/batch=48/cls=1.5를 그대로 두고 **trainer만 symmetric→dair로
교체**(`dair_params`/`TensorSpeakerOverlayAug`는 v01_dair 그대로 유지 — 의도적 설계, "실제
쓰는 dair가 SGD로도 안정적인가"가 목적). PT BN: 이상치 **2**/Inf **0**(SGD 그룹). NVT cosine
0.9857/0.9876/0.9945(SGD 그룹). Float mAP50-95 0.530(저하 없음). **Board: std+ultra mAP50-95
-0.70%, std+ultra mAP50 -1.22%** — v01_symmetric(-0.62%/+0.41%)과 사실상 동급, 실험①보다도
좋음. **`v01_dair`의 BN 통계 공백도 이때 채워짐**(이상치 5개+Inf1, AdamW 그룹). `v01_dair`와
`v02`의 `model.0.bn.running_var` Inf가 **정확히 같은 채널 인덱스(4)**에서 발생 — 우연이라기엔
재현 가능한 특정 실패 지점일 가능성(메커니즘은 미확정).

**최종 판단(RC §24-6, RS §7)**: PT(BN)·NVT(cosine)·Float(mAP)·Board(실측) **4개 레벨 전부**에서
`v02_dair`는 SGD 그룹(symmetric/실험①)에 속하고 AdamW 그룹(matched/v02/실험②)과 뚜렷이
구분된다 — **"dair 트레이너 자체가 원인"이라는 가설은 약화되고, optimizer(SGD vs AdamW)가
지배적 변수라는 결론이 서로 다른 두 축(optimizer-axis, trainer-axis)에서 수렴 확정**됐다.
"실험① 식 PT/NVT는 건강한데 board는 나쁠 수 있다"는 괴리는 `v02_dair`에서는 재현되지 않음
— **PT/NVT 신호가 항상 board와 어긋난다는 뜻이 아니라, board 없이 최종 예측은 불가하다는
뜻**으로 해석(레벨 간 불일치 "가능성"이 있다는 것이지 "항상"은 아님).

### 3.5 아직 미확정인 메커니즘 (RS §7)

- AdamW가 **왜** 이 데이터셋/조건에서 BN running_var를 불안정하게 만드는지(momentum/adaptive
  per-parameter update와 BN 통계 갱신의 상호작용) — 전혀 조사되지 않음.
- "BN 이상치 → pre-softmax dynamic range 확대"가 직접 인과인지, 둘 다 optimizer의 병렬
  결과인지 — 상관관계만 확인, 직접 대조 미실행(실험①/②에 §3.3 방식 pre-softmax 분해 미적용).
- optimizer 내부의 어떤 구체 요소(momentum 계수/adaptive lr/조합)가 핵심 변수인지.
- `model.22.cv2.*`(box 브랜치) weight/bias 통계의 dynamic range 확대 근원 추적(재추론 불필요,
  미실행).
- `v01_dair`/`v02`의 `model.0.bn` Inf가 왜 정확히 같은 채널(인덱스 4)에서 발생하는지.

## 4. 확인된 중간 현상 vs 배제된 가설 (원인과 혼동 금지, RS §5/§6)

| 항목 | 성격 | 현재 판단 |
|---|---|---|
| BN `running_var` 이상치/Inf | 중간 현상(실측) | optimizer와 강하게 연동(trainer와 무관, `v02_dair`가 반증). "BN 이상치 → NVT 오차"라는 인과 경로 자체는 직접 증명되지 않음 — "누적된 결과의 한 표지"로 해석 |
| pre-softmax DFL logit dynamic range | 중간 현상(실측, symmetric/matched/v02 3모델만) | matched/v02가 symmetric의 5~30배 — quantization 오차·board mAP 저하와 직결 확인(§3.3) |
| Detect head(box+cls 양쪽) Float→NVT 오차 확대 | 중간 현상(실측, 171-layer 분석) | cv2뿐 아니라 cv3도 동등 관여 — "box branch만의 문제"는 수정됨 |
| ONNX export 방식/그래프 구조 | — | **배제**(완전 동일 확인) |
| NVT compile 설정/calibration 데이터 | — | **배제**(byte-identical) |
| lr0 크기 단독 | — | **배제**(실험①이 반증) |
| batch size | — | **배제** |
| `cls` loss weight(1.5 vs 0.5) 단독 | — | **약화**(실험①/②둘다1.5인데 결과 갈림) |
| NMS 종류(dual-threshold vs 표준) | — | **주요 원인 아님**(0.7~1.7%p로 항상 작음, §6) |
| `model.0`이 최초 divergence 지점 | 과거 가설 | **반증**(171-layer 분석, cos 차이 0.0004 이내) |
| DFL Softmax 연산 자체가 오차를 증폭 | 과거 가설 | **반증/정정**(B vs C cos=1.00000, 오차는 pre-softmax 단계에서 결정) |
| box branch(cv2)만의 문제 | 과거 가설 | **수정**(Detect head 전체로 범위 확대) |
| DAIR dual-forward 학습 방식 자체가 근본 원인 | 과거 가설(RG §5, 2026-08-12) | **반증**(v01_symmetric_matched, 2026-08-20 — DAIR을 빼는 것 자체는 오히려 악화) |
| `dair` 트레이너 자체(dual-forward+증강)가 원인 | 과거 가설 | **약화**(`v02_dair`가 SGD 유지 시 4레벨 전부 건강, §3.4) |

## 5. 릴리즈 후보 선정 (`v01_symmetric` vs `v02_dair`, 2026-09-02~03, RSel)

SGD 그룹으로 확인된 두 후보 중 하나를 선정. **아키텍처는 완전히 동일**(둘 다
`symmetric_yolov8.yaml`) — 유일한 실질 차이는 trainer 기능: `v02_dair`는 매 배치마다
원본+augmented view를 함께 forward하고 일치를 강제하는 consistency loss(`dair_params`:
mode=or, λ=0.1, γ=0.5)를 추가하며, 증강 풀에 `TensorSpeakerOverlayAug`(대시보드/스피커 그릴
오버레이를 프레임 하단 1/4에 alpha blend, 실제 후방 카메라의 물리적 가림 시뮬레이션)와
`TensorSunFlareAug`(역광 시뮬레이션) 등이 포함된다 — **제품 관련 강건성 훈련**이지 단순
학습 트릭이 아니다. `v01_symmetric`은 이 훈련을 받지 않았다.

**미해결 리스크로 시작 → 해소**: 세션 시작 시 로컬 uncommitted 변경으로 `dair.py`의
`TensorSpeakerOverlayAug` 줄이 주석 처리(비활성)돼 있었다(HEAD 커밋은 활성) — 이 조사에서
수정한 적 없는 기존 상태. `v02_dair`는 다른 PC에서 학습돼 이 uncommitted 상태와 직접 인과는
없으나, "실제로 활성 상태로 학습됐는가"가 불확실했다. **해소(§3-1)**: `v02_dair` 학습
시각화(`train_batch{0,1,2}_aug.jpg`, 48타일)에서 프레임 하단 1/4에 도로 장면에 있을 수 없는
격자/그릴 패턴이 반복 확인 — augmentation이 실제로 활성 상태였음을 정성적으로 확인.

네 가지 독립 근거가 전부 같은 방향(`v01_symmetric` 우위)을 가리켰다(RSel §4):

1. **공식 acceptance 지표(mAP50, std+ultra)**: `v01_symmetric` 통과(+0.41%/+0.45%, §0),
   `v02_dair` 미달(-1.22%/-1.23%, 1% 초과).
2. **절대 성능(클래스별)**: `v02_dair`가 전 클래스에서 board mAP가 낮음 — motorcycle/person/
   truck mAP50-95 -0.050~-0.052 격차. Precision/Recall 분해: **Recall은 두 모델 사실상
   동급**이나 **Precision은 v02_dair가 car/motorcycle/person에서 -6.4~-8.4pt 낮음** →
   v02_dair는 검출을 놓치는 게 아니라 **오탐(false positive)이 더 많다.**
3. **조건별(NIGHT 근사) 강건성**: GT 6,152장 중 파일명 `NIGHT` 태그 1,833장/비-NIGHT
   4,319장(주의: 정확한 "프레임 하단 가림" 조건이 아니라 "야간/저조도" 조건 — DAIR 증강
   풀에 야간 직접 타겟 항목 없음, 근사 검증임을 명시). NIGHT mAP50-95: symmetric 0.5993 >
   dair 0.5653. non-NIGHT: symmetric 0.5635 > dair 0.5268. **양쪽 다 symmetric 우위** —
   DAIR의 강건성 훈련은 실재하지만(§3-1) 이 근사 검증에서는 v02_dair의 전반적으로 낮은
   절대 성능을 상쇄하지 못한다(야간 격차 -0.034가 비-야간 격차 -0.037보다 약간 작다는
   미약한 신호는 있으나 역전에는 못 미침).
4. **클래스별 F1-최적 confidence threshold**(별도 도구로 독립 재계산, IoU=0.5, 6,152장 전체):
   bus 제외 4개 클래스(car/motorcycle/person/truck) 전부 `v01_symmetric` F1 우위
   (+0.018~+0.035). bus만 사실상 동률(0.867 vs 0.865).

**릴리즈 스펙 데이터(2026-09-03 완료, RSel §3-7)** — Network info(YOLOv8n Symmetric DFL,
BGR, 5class object detection, NT98530/chip530, input `images[1,3,320,320]`, output
`output0[1,5,2100]`(cls,sigmoid)+`433[1,4,2100]`(box,raw DFL distance)):

Float ONNX 클래스별(std NMS, all row):

| Class | v01_symmetric P/R/mAP50/mAP50-95 | v02_dair P/R/mAP50/mAP50-95 |
|---|---|---|
| bus | 0.893/0.844/0.888/0.720 | 0.926/0.822/0.873/0.696 |
| car | 0.731/0.746/0.785/0.583 | 0.728/0.727/0.763/0.555 |
| motorcycle | 0.885/0.678/0.793/0.503 | 0.875/0.631/0.746/0.458 |
| person | 0.853/0.603/0.719/0.436 | 0.821/0.575/0.676/0.396 |
| truck | 0.960/0.735/0.797/0.651 | 0.945/0.693/0.749/0.602 |
| **all** | **0.864/0.721/0.796/0.579** | **0.859/0.690/0.761/0.541** |

Board INT8 클래스별(all row + Float 대비 손실%, mAP50/mAP50-95):

| Class | v01_symmetric P/R/mAP50/mAP50-95 | 손실 | v02_dair P/R/mAP50/mAP50-95 | 손실 |
|---|---|---:|---|---:|
| bus | 0.925/0.814/0.887/0.701 | -0.11%/-2.64% | 0.924/0.813/0.864/0.692 | -1.03%/-0.57% |
| car | 0.726/0.736/0.775/0.561 | -1.27%/-3.77% | 0.662/0.735/0.742/0.535 | -2.75%/-3.60% |
| motorcycle | 0.895/0.668/0.805/0.504 | +1.51%/+0.20% | 0.822/0.674/0.750/0.454 | +0.54%/-0.87% |
| person | 0.846/0.605/0.721/0.436 | +0.28%/±0.00% | 0.762/0.614/0.667/0.386 | -1.33%/-2.53% |
| truck | 0.958/0.678/0.812/0.673 | +1.88%/+3.38% | 0.938/0.641/0.737/0.621 | -1.60%/+3.16% |
| **all** | **0.870/0.700/0.800/0.575** | **+0.45%/-0.62%** | **0.822/0.695/0.752/0.538** | **-1.23%/-0.70%** |

(이 all-row 값이 §0에서 언급한 "+0.45%/-1.23%" 출처 — RC/RS의 +0.41%/-1.22%와는 독립적으로
재계산된 값이며, 원문에 정밀도 차이 설명이 없어 그대로 병기한다.)

**크기별 recall(탐색적, 2026-09-03 완료, RSel §3-8)**: 처음 COCO 절대 기준(small<32²px²)을
썼으나 bus/truck은 원래 크게 찍혀 small 구간이 거의 0(bus small n=0, truck small n=2)이라는
근본 한계 발견 → **클래스별 상대 기준(자기 분포의 33/66 percentile, tertile)**으로 교체
(절대 크기 비교가 아니라 클래스 내 근/원거리 난이도 비교가 목적). 전체 크기별 recall:

| Size | v01_symmetric Float/Board | v02_dair Float/Board |
|---|---:|---:|
| small | 0.406/0.416 | 0.381/0.345 |
| medium | 0.731/0.725 | 0.718/0.698 |
| large | 0.909/0.917 | 0.908/0.913 |

`v02_dair`가 small/medium(특히 truck: Float 0.397→0.323, Board 0.370→0.276)에서 뚜렷이
열세 — 격차가 주로 **원거리(작게 찍히는) 쪽**에서 벌어짐.

**"중요 인식 영역"(ROI) recall(2026-09-03/04, RSel §3-9) — 미완결, 결론 아님**: 사용자가
육안으로 가정한 ROI `[200,380]-[1720,1080]`(공식 스펙 아님 — README 입력 사양 표에
Rear_Truck 전용 crop/ROI 정의 없음, 확인됨). GT box 면적 30% 이상 겹치면 "영역 내". in-ROI는
두 모델 거의 동급, **out-ROI(멀고 작은 개체)에서 v01_symmetric이 뚜렷이 우위** — §3-8과 같은
결론을 재확인. **그러나 이 ROI 자체에 경계 편향이 발견됐다**: GT 이미지 13장 육안 확인 결과
좌측의 크고 가까운 차량(버스/트럭)이 상단 경계(y=380)에 반복적으로 잘리는 패턴(13장 중
5장) — 이 어안 후방 카메라는 좌측 인접 차선의 근접 대형차일수록 화면 위쪽까지 찍히는 경향이
있어, 현재 ROI가 BSD에서 가장 중요한 "바로 옆 대형차량" 사례를 체계적으로 배제할 가능성이
있다. **경계값 조정 여부, Front_Truck 방식(edge/center 좌우 3등분)으로 바꿀지는 사용자 확인
대기 중 — 이 탐색은 참고용이며 §4 결정을 좌우하는 확정 근거로 격상하지 않는다.**

**최종 결정(RSel §4/§5)**: "**2026-09-02 확보된 근거로는 `v01_symmetric`이 `v02_dair`보다
명확히 우세한 릴리즈 후보다**"라는 판단 이후, **2026-09-03 사용자가 `v01_symmetric`을
릴리즈 후보로 확정하고 릴리즈 절차 진행을 지시**했다. §3-7~3-9(릴리즈 스펙/크기별/ROI
탐색)도 전부 같은 방향을 재확인 — 결론을 바꾸지 않았다. §3-4(pre-softmax 분해)·§3-5(재현성
재학습)는 선택 사항으로 보류 유지.

**남은 작업(RSel §6, 문서 작성 시점 2026-09-02 기준 미실행)**: (1) README(`configs/
experiments/BSD_F606/Rear_Truck/README.md`)·정본 문서(RG) 최신화 — optimizer 결론과
v01_symmetric 최종 선정 근거 반영. (2) `configs/releases/.../v01_symmetric/`로 복사 +
`weight/network/analysis/` 산출물 복사 + MANIFEST.yaml 작성(Front_Truck v01 릴리즈의
`analysis/` 폴더 구조를 템플릿으로 참고 — `thr_tool_*.json`/`F1conf_*.png`는 §3-6/3-7
산출물로 이미 대응 가능, `metrics_*.json`의 "regions"(edge/center)는 §3-9 미완결이라
보류, `errorcases/`는 사용자 지시로 이번 범위에서 제외). (3) `ruff check && pytest` →
커밋. **SigmaStar(IPU) 변환은 이와 별도로 이미 진행됨 — [06_sigmastar_conversion.md](06_sigmastar_conversion.md) 참조.**

## 6. Evaluator/NMS 축 — 손실 %가 왜 크게 갈리는가 (EG1, EG2)

같은 모델·같은 board 예측이라도 **NMS 종류**(dual-threshold vs 표준 단일-IoU)와
**evaluator**(pycocotools `customEval.py` vs ultralytics `ap_per_class`)를 바꾸면 손실 %가
크게 달라진다. 정의와 코드 근거는 [90_reproduction_reference.md](90_reproduction_reference.md)
§2 참조. 이 절은 Rear_Truck에서 실측된 결과만 정리한다.

**v01_symmetric 2×2 isolation(EG1 §5/§6, conf=0.001 통일)**:

| NMS \ Evaluator | pycocotools | ultralytics `ap_per_class` |
|---|---:|---:|
| dual-threshold | golden 0.562, board 0.521, **-7.30%** | golden 0.5864, board 0.5749, **-1.96%** |
| 표준 단일-IoU | golden 0.558, board 0.521, **-6.63%** | golden 0.5785, board 0.5749, **-0.62%** |

NMS 축 효과는 항상 작다(1.5%p 이내: -7.30→-6.63, -1.96→-0.62). **Evaluator 축 효과가 훨씬
크다(5~6%p)** — 동일한 board 예측·동일 GT인데도 evaluator만 바꾸면 board mAP50-95 자체가
0.521→0.5749로 바뀐다(+10.3%). **"왜" 두 evaluator가 이만큼 다른지의 알고리즘적 원인은
완전히 규명되지 못했다** — IoU threshold(둘 다 0.50:0.05:0.95, 10개, 동일 확인), AP 적분
방식(둘 다 101-point COCO 보간, 동일 확인), `maxDets`(board 최대 60건/이미지라 pycocotools
100 제한 무관, 배제 확인)는 전부 원인에서 배제됐고, **남은 유력 후보는 GT↔Prediction
그리디 매칭 알고리즘 자체의 구현 차이**(`pycocotools COCOeval.evaluateImg` vs `ultralytics
BaseValidator.match_predictions`, 둘 다 그리디이나 서로 다른 구현)로 추정되나 **라인 단위
직접 대조는 수행되지 않았다(미확정)**.

**네 모델(v01_symmetric/v01_dair/v02/v01_symmetric_matched) 종합(EG1 §10/§10-1, EG2)**:

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra | NMS축 효과 | Evaluator축 효과 |
|---|---:|---:|---:|---:|---:|---:|
| v01_symmetric | -7.30% | -1.96% | -6.63% | -0.62% | ~0.7~1.3%p | ~5.3~6.0%p |
| v01_dair | -21.01% | -16.32% | -19.66% | -14.62% | ~1.35~1.70%p | ~4.7~5.0%p |
| v02 | -19.14% | -14.80% | -18.08% | -13.44% | ~1.06~1.36%p | ~4.3~4.6%p |
| **v01_symmetric_matched** | **-28.42%** | **-27.00%** | **-27.36%** | **-25.62%** | **~1.06~1.38%p** | **~1.4~1.7%p** |

mAP50 기준(참고, EG1 §10-1):

| Model | dual+pycoco | dual+ultra | std+pycoco | std+ultra |
|---|---:|---:|---:|---:|
| v01_symmetric | -3.47% | **+0.26%** | -3.72% | **+0.41%** |
| v01_dair | -9.72% | -6.03% | -8.48% | -4.66% |
| v02 | -10.08% | -6.80% | -9.60% | -6.24% |
| v01_symmetric_matched | -11.35% | -10.06% | -10.64% | -9.29% |

**핵심 관찰(EG2 §4)**: evaluator 축 효과가 NMS 축 효과보다 크다는 패턴은 v01_dair·v02에서
재현됐으나(4.3~5.0%p vs 1.06~1.70%p), **v01_symmetric_matched에서는 재현되지 않았다**
(1.4~1.7%p, 오히려 NMS 축과 비슷한 크기) — **"이름/트레이너가 같으면 evaluator 의존성
패턴도 같을 것"이라는 가정은 v01_symmetric_matched(이름·트레이너는 v01_symmetric과 동일)가
정반대 패턴을 보이며 틀렸음이 확인됐다.** v01_dair와 v02는 evaluator를 바꿔도 두 자릿수(%)
손실이 그대로 남는 사실상 같은 패턴(DAIR 트레이너 계열). **네 모델 전부에서 공통되는
유일한 것은 NMS 축 효과가 항상 작다는 것(0.7~1.7%p)뿐**이며, evaluator 축 효과의 크기
자체는 모델마다 다르고 그 이유는 규명되지 않았다.

## 7. 재-export 재현성 검증 (2026-08-14, RX)

기존 v01_symmetric 결과(Float 0.561/NVT 0.521/-7.1%, dual+pycoco 기준)를 전제하지 않고
격리 디렉토리에서 처음부터 재-export→재검증. 체크포인트 클래스는 `SymmetricDetectionModel`
이라 "관습적으로 맞는" export 스크립트는 `tools/export_symmetric.py`이나, 기존 문서는
`tools/export_custom.py`를 썼다는 불일치가 재검증 대상이었다.

- **Graph/Initializer/Raw output**: node_diffs_count=0(235 node 전체 동일), initializer
  127/127 exact_identical, 10개 고정 입력(zeros/ones/half/random/impulse/실제 이미지 5장)
  전 텐서 **bit-exact**(max_abs_diff=0.0). 파일 SHA256만 다름(export마다 달라지는 메타데이터
  때문) — **`export_symmetric.py --skip_postprocess`와 `export_custom.py`는 동일 ONNX를
  생성한다**(두 wrapper의 `.export()`가 동일한 `Model.export()`로 귀결).
- **Float ONNX mAP 재측정**: mAP50 0.774=0.774, mAP50-95 0.561=0.561(bit-exact와 일치).
- **NVT 재변환**(격리 디렉토리, calibration/test 이미지는 개별 symlink+`test_img_list.txt`만
  로컬 복사로 원본 오염 방지): Latency 6.368ms=6.36765ms, DRAM 36.54MB=36,540,852bytes,
  size 4.71MB=4,711,336bytes, cosine(Softmax/Sigmoid/dfl_conv) 0.9850/0.9824/0.9933 =
  0.984977/0.982404/0.993315(소수 4째 자리까지 일치).
- **Board(NPU 실기) 검증: 미완료** — 물리 하드웨어가 필요해 이 세션(Claude Code)은 직접
  재현 불가능. Step 0~4가 전부 bit-exact/거의 bit-exact로 재현됐으므로 board mAP도 동일하게
  재현될 것으로 **강하게 추정하나, 실측 없이는 확정할 수 없다.**

## 8. 실패/중단된 시도 요약 (조건 → 결과 → 배운 점/배제 범위)

| 시도 | 조건 | 결과 | 배운 점/배제 |
|---|---|---|---|
| calib_mode 변경 | MAX(0), KLD(2) vs PERCENTILE(4,0.999) | 둘 다 대폭 악화 | PERCENTILE(0.999) 유지 확정 |
| calibration 다양성 개선 | 500→724(연속프레임)→940장(다양 샘플링) | cls/dfl cosine 개선폭 미미(0.9427→0.9423→0.9486 / 0.9691→0.9719→0.9672) | calibration 다양성은 이번 gap의 주 원인이 아님 |
| Front production config 이식(CLE off 등) | rear_truck_v01_320_frontstyle | 악화(cls 0.9323/dfl 0.9533) | 타 카메라 config 일괄 이식은 역효과 — one-variable-at-a-time 원칙의 근거 |
| 538칩 mixed-precision sensitivity 분석 | 원본 538 config 미보유, 530 이식 | `sorted_sensitivity.json` 생성 안 됨 | 정보 획득 실패, `run_compiler_538.sh`만 잔존 |
| model.2/4 spec_demarc_tensors 최소 subset 탐색 | round1(backward elimination 6종)/round2(coarse A/B) 설계 완료 | v02 학습과 CPU 자원 경쟁으로 컴파일 시간 19분→2시간20분+ 급증, drop_T1 프로세스 kill 후 **전면 중단**. round2 coarse A/B조차 실행 안 됨 | v01_symmetric이 저비용 3텐서로 -7.1% 달성해 우선순위 하락 — 폐기 아닌 선택적 후속 연구로 남음(미실행) |
| REAR Experiment A(spec_demarc 3-tensor ablation, FG 제안) | drop_H1/H2/H3 각 1개씩 제거 | **A1(drop_H1) compile만 완료(nvt_model.bin 생성), simulator/board 전부 미실행** — 2026-08-20 사용자 지시로 중단 | H1/H2/H3 각각의 기여도 판정 불가(실험 미완료로 결론 없음) |
| DAIR 유무 strict on/off ablation(augmentation까지 완전 통제) | symmetric 트레이너의 DAIR 모드 크래시 버그 수정이 선행 조건 | **미실행**(`v01_symmetric_matched`가 근사치이나 confound 잔존) | v02_dair(trainer-axis)로 우회 확인했으나, "TensorSpeakerOverlayAug 유무" 자체를 독립적으로 격리한 실험은 여전히 없음 |
| seed 반복 실험(run-to-run 변동성 확인) | 미실행 | — | v01(-33.1%)과 v01_dair(-20.7%)가 "같은 recipe 계열"인데도 12.4%p 차이 — 우선순위 상승했으나 미실행 |
| pycocotools vs ultralytics 매칭 알고리즘 직접 대조(합성 예제) | 미실행 | — | evaluator 축 격차의 마지막 미확정 퍼즐로 남음 |

## 9. Front golden reference — Rear에 전이 가능한 시사점 (FG, 2026-08-19~20)

Front_Truck v01 Release를 golden reference 삼아 "Front는 왜 양자화에 강한가"를 역추적해
Rear에 적용 가능한 요소를 찾으려는 시도. **Front 자체의 상세 성능·재현 데이터는
[02_front_reference_and_reproduction.md](02_front_reference_and_reproduction.md) §Front
golden reference 절 참조** — 이 절은 Rear 조사에 전이된 부분만 정리한다.

- **핵심 관찰**: Front는 spec_demarc_tensors **0개**(전체 주석 처리) + CLE/cal_refine_iter/
  bnscale 전부 비활성인데도 mAP50-95 기준 손실이 없다(오히려 **+10.31%** 개선, 실험0 근사
  재현치). Rear의 3-tensor 보호로도 -7.1%(당시)였던 것과 정반대.
- **당시(2026-08-19) 가설**: "tensor 보호가 많다고 강건성이 보장되지 않는다", "원인은
  config가 아니라 학습된 weight/activation 분포 자체에 있을 가능성이 높다"(backbone-early
  cosine이 Front 0.984~0.998 vs Rear ~0.96대로 근사 비교됨). **단, final output cosine만
  보면 Front(0.976~0.982)가 Rear(0.982~0.985)보다 낫지 않아 "cosine 높으면 mAP 유지"라는
  단순 관계는 아니라는 점도 이미 지적됨.**
- **이 가설과 §3의 관계**: FG의 "weight/activation 분포 자체" 가설은 방향은 맞았으나(§3.3의
  BN/dynamic range 발견과 상통), **구체적 원인이 "config 이식 대상"이 아니라 "학습
  optimizer 선택"이라는 것은 FG 작성 시점(2026-08-19~20)에는 규명되지 않았다** — FG가
  제안한 one-variable-at-a-time 실험(Experiment A~E)은 결과적으로 대부분 실행되지 않았고,
  실제 원인 규명은 §3.4의 optimizer ablation(2026-08-31~09-02, FG와 무관한 별도 실험
  경로)으로 이뤄졌다. FG의 가설은 **틀리지 않았으나 이후 조사로 대체(superseded)**된
  과거 단계로 취급한다.
- **미실행으로 남은 FG의 제안**: Experiment A(spec_demarc 1개 변경, §8에 기록된 대로 A1만
  compile), B(cal_refine_iter만 변경), C(head tensor precision만 변경), D(calibration
  preprocessing만 변경), E(QAT) — B/C/D/E는 착수조차 되지 않았다.
