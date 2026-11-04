# BSD_F606 Rear_Truck 릴리즈 후보 선정 계획 — `v01_symmetric` vs `v02_dair`

- **작성**: 2026-09-02 (KST)
- **목적**: quantization loss 조사(`docs/debug/20260824_symmetric_low_quant_loss_rootcause_compare.md`,
  요약: `docs/debug/20260831_symmetric_quant_loss_rootcause_summary.md`)에서 SGD 그룹으로
  확인된 두 후보 `v01_symmetric`과 `v02_dair` 중 **Rear_Truck 릴리즈 후보를 하나 선정**한다.
- **전제 (2026-09-02 정정)**: acceptance target(≤1%, `docs/guides/
  rear_truck_train_validate_nvt_pipeline.md` §13)의 기준 지표를 **mAP50 + 표준 단일-IoU
  NMS + ultralytics evaluator(std+ultra)**로 확정했다(사용자 지시, pycocotools·dual-NMS
  값은 이후 사용하지 않는다). 이 기준으로: **`v01_symmetric`은 +0.41%로 통과(손실 없음,
  오히려 개선), `v02_dair`는 -1.22%로 1% 허용치를 0.22%p 초과해 미달**이다 — "둘 다
  미달"이었던 이전 판단(dual+pycoco 기준)은 폐기한다. 이 문서의 목적은 이제 "target을
  통과하는 `v01_symmetric`을 그대로 채택할지, 아니면 `v02_dair`의 다른 이점(DAIR
  강건성)이 이 target 미달을 감수할 만큼 큰지"를 판단하는 것으로 좁혀진다.
- **범위**: 이 문서는 **선정 계획**이다. 재학습은 하지 않는다 — 기존 산출물 재분석과,
  필요시 재추론(재학습 아님) 수준의 저비용 검증만 수행한다.

---

## 1. 지금까지 확인된 사실 (quantization loss, 재정리)

`20260831_...summary.md` §3.1/§24 참조, 재계산 없이 인용.

| | `v01_symmetric` | `v02_dair` |
|---|---|---|
| Trainer | Symmetric (`SymmetricTrainer`) | DAIR (`DairDetectionTrainer`, dual-forward) |
| Architecture yaml | `symmetric_yolov8.yaml` | 동일(`symmetric_yolov8.yaml`) — **아키텍처 자체는 같다** |
| optimizer/lr0/lrf/cos_lr/batch/cls | SGD/0.01/0.01/true/48/1.5 | 동일(의도적으로 맞춤) |
| dair_params(dual-forward consistency + 증강) | 없음(symmetric은 구조적으로 미지원) | 있음(mode=or, λ=0.1, γ=0.5) |
| **공식 acceptance 지표: Quant loss(mAP50, std+ultra)** | **+0.41% — 통과** | **-1.22% — 미달(1% 초과)** |
| 참고: Quant loss(mAP50-95, std+ultra) | -0.62% | -0.70% |
| BN 이상치/Inf(PT) | 1개/0 | 2개/0 |
| NVT cosine(Softmax/Sigmoid/dfl_conv) | 0.9850/0.9824/0.9933 | 0.9857/0.9876/0.9945 |

**(2026-09-02 정정) 공식 acceptance 지표(mAP50, std+ultra)로는 `v01_symmetric`만
통과한다.** `v02_dair`는 -1.22%로 1% 허용치를 0.22%p 초과해 미달이다. **이 문서 전체에서
pycocotools/dual-NMS 기준 값은 쓰지 않는다**(사용자 지시) — 결정에는 mAP50+표준
단일-IoU NMS+ultralytics evaluator(std+ultra)만 쓴다. 이 지표 하나만으로도
`v01_symmetric` 쪽으로 무게가 실린다.

---

## 2. 두 후보의 본질적 차이 — trainer가 실제로 다른 것

두 모델은 **아키텍처(symmetric DFL 포함)가 완전히 같다.** 유일한 실질 차이는 trainer가
갖는 기능이다:

- **`v01_symmetric` (SymmetricTrainer)**: DAIR 없음. 표준 학습.
- **`v02_dair` (DairDetectionTrainer)**: 매 배치마다 원본 이미지와, 아래 증강 풀에서 뽑은
  augmented view를 **함께 forward**하고, 두 view의 예측/feature가 일치하도록 강제하는
  **consistency loss**(`dair_params`: mode=or, λ=0.1, γ=0.5)를 추가한다(`docs/glossary.yaml`
  "DAIR" 정의). 증강 풀(`projects/src/models/yolo/detect/train/dair.py:134-139`, `Select`):
  - `ColorJitter`, `RandomAdjustSharpness`, `GaussianBlur`, `TensorSunFlareAug`(역광 시뮬레이션)
  - `TensorSpeakerOverlayAug`(`projects/src/data/dynamic_crop.py:169` `SpeakerOverlayAug` —
    **대시보드/스피커 그릴 오버레이 이미지를 프레임 하단 1/4에 alpha blend** — 실제 트럭
    후방 카메라 영상에서 흔히 보이는 물리적 가림 요소를 시뮬레이션)

**즉 DAIR trainer의 실질적 목적은 "이 프레임 하단부 가림(대시보드/장착물)이나 역광 조건에서도
예측이 흔들리지 않게 만드는 것"이라는 제품 관련 강건성 훈련이지, 단순한 학습 트릭이 아니다.**
`v01_symmetric`은 이 강건성 훈련을 받지 않았다 — quantization loss가 비슷하더라도, **실제
가림/역광 조건에서의 견고성은 두 모델이 다를 가능성이 있다.**

### 2-1. 확인이 필요한 리스크 — `TensorSpeakerOverlayAug` 코드 상태 (미해결, 사용자 확인 필요)

`git diff HEAD -- projects/src/models/yolo/detect/train/dair.py` 확인 결과, 이 저장소의
**현재 커밋되지 않은 로컬 변경**으로 `TensorSpeakerOverlayAug(alpha=(0.3, 0.5))` 줄이
방금 주석 처리돼 비활성 상태다(HEAD 커밋 상태는 활성). 이 조사 세션에서 이 파일을 수정한
적은 없다 — 세션 시작 전부터 있던 uncommitted 변경이다.

- `v02_dair`는 **다른 학습 PC**에서 학습됐으므로 이 로컬 저장소의 uncommitted 상태와
  직접적 인과관계는 없다 — `v02_dair` exp.yaml 헤더는 "TensorSpeakerOverlayAug 포함
  Select 증강 풀도 v01_dair와 동일하게 활성"이라고 명시하지만, 이는 **그 학습 PC의 당시
  코드 상태를 근거로 한 사용자 확인**이지 이 저장소의 코드로 직접 검증된 것은 아니다.
- **미해결 질문**: 이 uncommitted 변경은 왜, 언제 생겼는가? `v02_dair`가 실제로
  `TensorSpeakerOverlayAug`를 켠 채 학습됐다는 것을 재확인할 방법이 있는가(예: 그 PC의
  학습 로그/시각화 이미지에 overlay 흔적이 있는지)? 이 답이 없으면, "DAIR이 제품
  강건성을 실제로 훈련시켰다"는 §2의 핵심 근거 자체가 불확실해진다.
- **(2026-09-02 해소, §3-1 참조)**: `v02_dair`의 학습 시각화(`train_batch{0,1,2}_aug.jpg`,
  48개 타일)를 직접 확인한 결과, 여러 타일에서 프레임 하단 1/4에 도로/횡단보도 장면에
  자연스럽게 있을 수 없는 동일한 격자/그릴 패턴이 반복적으로 나타난다 — `SpeakerOverlayAug`가
  정확히 이 방식(대시보드/스피커 그릴 이미지를 하단 1/4에 alpha blend)으로 동작하므로,
  **`TensorSpeakerOverlayAug`가 실제로 활성 상태였다는 정성적 확인**이 됐다. 이 저장소의
  현재 uncommitted 비활성화 변경은 `v02_dair` 학습(다른 PC, 이전 시점)과는 무관한 것으로
  판단한다.

---

## 3. 검증 계획 (제안, 우선순위 순)

### 3-1. (최우선, 저비용) `v02_dair` 학습 시각화 자료로 augmentation 활성 여부 확인 — **완료**

`runs/detect/BSD_F606/Rear_Truck/v02_dair/{train_batch*.jpg,train_batch*_aug.jpg}`가 이미
존재한다(학습 시 자동 저장, `dair.py:plot_training_samples`가 `batch["aug_img"]`를 저장하는
바로 그 파일 — DAIR 증강 풀을 거친 실제 뷰임을 코드로 확인). 3개 batch(48타일) 전부 확인
→ **§2-1에서 활성 확인 완료.**

### 3-2. 클래스별 성능 비교 (재추론 없음, 기존 예측 재계산) — **완료, 중요한 반전 발견**

두 모델의 board(ultralytics evaluator) 클래스별 mAP를 직접 대조(pycocotools 값은 쓰지
않음, 사용자 지시):

| Class | mAP50 sym/dair | diff | mAP50-95 sym/dair | diff |
|---|---:|---:|---:|---:|
| bus | 0.887/0.864 | -0.023 | 0.701/0.692 | -0.009 |
| car | 0.775/0.742 | -0.033 | 0.561/0.535 | -0.026 |
| motorcycle | 0.805/0.750 | -0.055 | 0.504/0.454 | **-0.050** |
| person | 0.721/0.667 | -0.054 | 0.436/0.386 | **-0.050** |
| truck | 0.812/0.737 | -0.075 | 0.673/0.621 | **-0.052** |

**`v02_dair`가 전 클래스에서 `v01_symmetric`보다 절대 mAP가 낮다** — mAP50 기준으로도
motorcycle/person/truck은 0.05~0.075 차이로 격차가 더 크게 보인다. Precision/Recall
분해:

| Class | Precision sym/dair | diff | Recall sym/dair | diff |
|---|---:|---:|---:|---:|
| car | 0.726/0.662 | -0.064 | 0.736/0.735 | -0.001 |
| motorcycle | 0.895/0.822 | -0.073 | 0.668/0.674 | +0.006 |
| person | 0.846/0.762 | **-0.084** | 0.605/0.614 | +0.009 |
| truck | 0.958/0.938 | -0.020 | 0.678/0.641 | -0.037 |

**Recall은 두 모델이 사실상 동급(일부 클래스는 dair가 근소 우위)이지만, Precision은
`v02_dair`가 car/motorcycle/person에서 뚜렷이 낮다(-6.4~-8.4pt)** — 즉 `v02_dair`는
검출을 놓치는 게 아니라 **오탐(false positive)이 더 많다.**

**§1의 공식 acceptance 지표(mAP50, std+ultra) 판정과 방향이 일치한다**: `v02_dair`의
Float 자체가 이미 `v01_symmetric`보다 낮고, 그 격차가 board까지 그대로 이어져 절대
성능만 보면 `v01_symmetric`이 거의 전 클래스에서 우위다. **§4 결정 기준에 이 절대 성능
차이를 반드시 반영해야 한다.**

### 3-3. 가림/역광 조건 하위셋에서의 성능 비교 — **완료 (NIGHT 태그 기준)**

GT 6,152장 파일명에 `NIGHT` 태그가 있는 이미지가 1,833장, 없는(주간/기타) 이미지가
4,319장이다(파일명 직접 확인, 새 라벨링 불필요). **주의**: 이건 정확히 "프레임 하단
가림"(SpeakerOverlayAug가 시뮬레이션하는 것)은 아니고 "야간/저조도" 조건이다 — DAIR의
증강 풀에는 야간을 직접 겨냥한 항목이 없다(TensorSunFlareAug는 역광/태양광 시뮬레이션).
즉 **이 결과는 "DAIR이 목표로 한 정확한 조건"의 검증이 아니라 "일반적인 열악한 조건에서의
상대적 강건성"에 대한 근사 검증**이다.

재추론 없이 기존 병합 board 예측 + `ultralytics.utils.metrics.ap_per_class`로 NIGHT/
non-NIGHT 서브셋을 나눠 재계산(스크립트: 임시, 세션 종료 시 삭제):

| | NIGHT(n=1,833) mAP50 / mAP50-95 | non-NIGHT(n=4,319) mAP50 / mAP50-95 |
|---|---|---|
| `v01_symmetric` | 0.8197 / 0.5993 | 0.7924 / 0.5635 |
| `v02_dair` | 0.7781 / 0.5653 | 0.7431 / 0.5268 |

**`v01_symmetric`이 NIGHT/non-NIGHT 양쪽 모두에서 `v02_dair`보다 절대 mAP가 높다** —
DAIR의 강건성 훈련이 실제로 있었다는 것(§3-1)과, 그 강건성이 **이 NIGHT 서브셋에서는
`v02_dair`의 전반적으로 낮은 절대 성능(§3-2)을 상쇄하지 못한다는 것**은 별개의 사실이다.
NIGHT 조건에서 두 모델의 격차(mAP50-95: 0.599 vs 0.565, -0.034)가 non-NIGHT 격차(0.564
vs 0.527, -0.037)보다 **오히려 약간 작다** — "야간에는 DAIR이 상대적으로 덜 불리해진다"는
정도의 미약한 신호는 있지만, 격차 자체를 뒤집을 정도는 아니다.

**결론**: 이 근사 검증(정확한 occlusion 조건이 아님)에서는 DAIR의 강건성 이점이 §3-2의
절대 성능 격차를 상쇄한다는 증거를 찾지 못했다. `v01_symmetric`이 야간 조건에서도 여전히
우위다.

### 3-4. §18 방식 pre-softmax dynamic range 분해 (선택, 이미 알려진 후속 항목)

`20260831_...summary.md` §9에 이미 선택적 후속으로 남아있던 항목 — 두 모델에 적용하면
quantization 관점에서 더 세밀한 차이가 있는지 확인 가능(재추론 불필요, 기존 NVT dump 사용).
결정에 필수는 아니지만 참고 자료로 유용.

### 3-5. 재현성/안정성 (보류, 재학습 필요 — 이번 계획 범위 밖)

seed를 바꿔 재학습해 결과가 안정적인지 보는 것은 가장 확실하지만 재학습이 필요해 이번
저비용 검증 범위에서 제외한다. 3-1~3-4로 결정이 안 나면 다음 단계로 고려.

### 3-6. 클래스별 F1-최적 confidence threshold — **완료 (2026-09-02)**

별도 체크아웃(`ultralytics_projects-main/jdh_tools/calc_class_thresholds.py`, 이 문서
작성자가 아닌 기존 프로젝트 도구)를 `tools/debug/calc_class_thresholds.py`로 복사해
`--pred-file` 모드(재추론 없음)로 실행 — 기존 board 병합 예측
(`ipu_pred_merged_{v01_symmetric,v02_dair}_3tensor.json`, category만 소문자로 변환한
임시 사본 사용, 원본 미수정) + GT(`ipu_gt_coco_format.json`), IoU=0.5(고정), 6,152장 전체.

| Class | Optimal conf(sym/dair) | F1(sym/dair) | P(sym/dair) | R(sym/dair) | n_gt |
|---|---:|---:|---:|---:|---:|
| bus | 0.413 / 0.225 | 0.867 / 0.865 | 0.938 / 0.924 | 0.806 / 0.813 | 3,049 |
| car | 0.500 / 0.449 | **0.741 / 0.723** | 0.797 / 0.813 | 0.691 / 0.651 | 7,293 |
| motorcycle | 0.259 / 0.225 | **0.775 / 0.741** | 0.850 / 0.822 | 0.712 / 0.674 | 6,880 |
| person | 0.244 / 0.192 | **0.706 / 0.681** | 0.775 / 0.762 | 0.648 / 0.615 | 13,675 |
| truck | 0.243 / 0.225 | **0.797 / 0.762** | 0.943 / 0.938 | 0.690 / 0.641 | 3,370 |

**F1(각 모델의 최적 conf threshold에서)이 bus를 제외한 4개 클래스 전부 `v01_symmetric`
우위**(car +0.018, motorcycle +0.034, person +0.025, truck +0.035) — §3-2(절대 mAP)와
같은 방향으로 독립적인 지표(threshold-sweep F1)에서도 재확인됐다. bus만 사실상
동률(0.867 vs 0.865). 부수적으로 `v01_symmetric`의 optimal conf threshold가 대체로
더 높다(예: truck 0.243 vs 0.225, car 0.500 vs 0.449) — confidence score가 더 잘
보정(calibrated)돼 있어 더 높은 threshold에서도 F1이 유지된다는 해석과 부합한다.

저장: `runs/_eval_scratch/thresholds/{v01_symmetric,v02_dair}_thresholds.json` +
F1-Confidence 곡선(`_f1_curve.png`).

### 3-7. 릴리즈 스펙 데이터 — Network info + Float/Board 클래스별 성능(all row 포함) — **완료 (2026-09-03)**

릴리즈 문서화에 필요한 항목을 정리(재추론 없음, 기존 산출물 재계산만).

**Network info** (두 모델 ONNX 직접 로드해 확인, 완전히 동일):

| 항목 | 값 |
|---|---|
| Model type | YOLOv8n (Symmetric DFL), `symmetric_yolov8.yaml` |
| Color space | BGR |
| Task type | Object Detection (5class: bus/car/motorcycle/person/truck) |
| Target NPU | Novatek CNN25, NT98530 (chip 530) |
| Input | `images` [1,3,320,320] float32 |
| Output 1 | `output0` [1,5,2100] — cls, sigmoid |
| Output 2 | `433` [1,4,2100] — box, raw DFL distance(디코드 전) |

**Float ONNX 클래스별(std NMS, all row 포함)**:

| Class | v01_symmetric P/R/mAP50/mAP50-95 | v02_dair P/R/mAP50/mAP50-95 |
|---|---|---|
| bus | 0.893/0.844/0.888/0.720 | 0.926/0.822/0.873/0.696 |
| car | 0.731/0.746/0.785/0.583 | 0.728/0.727/0.763/0.555 |
| motorcycle | 0.885/0.678/0.793/0.503 | 0.875/0.631/0.746/0.458 |
| person | 0.853/0.603/0.719/0.436 | 0.821/0.575/0.676/0.396 |
| truck | 0.960/0.735/0.797/0.651 | 0.945/0.693/0.749/0.602 |
| **all** | **0.864/0.721/0.796/0.579** | **0.859/0.690/0.761/0.541** |

**Board INT8 클래스별(all row + Float 대비 손실%)**:

| Class | v01_symmetric P/R/mAP50/mAP50-95 | 손실(mAP50/mAP50-95) | v02_dair P/R/mAP50/mAP50-95 | 손실(mAP50/mAP50-95) |
|---|---|---:|---|---:|
| bus | 0.925/0.814/0.887/0.701 | -0.11%/-2.64% | 0.924/0.813/0.864/0.692 | -1.03%/-0.57% |
| car | 0.726/0.736/0.775/0.561 | -1.27%/-3.77% | 0.662/0.735/0.742/0.535 | -2.75%/-3.60% |
| motorcycle | 0.895/0.668/0.805/0.504 | +1.51%/+0.20% | 0.822/0.674/0.750/0.454 | +0.54%/-0.87% |
| person | 0.846/0.605/0.721/0.436 | +0.28%/±0.00% | 0.762/0.614/0.667/0.386 | -1.33%/-2.53% |
| truck | 0.958/0.678/0.812/0.673 | +1.88%/+3.38% | 0.938/0.641/0.737/0.621 | -1.60%/+3.16% |
| **all** | **0.870/0.700/0.800/0.575** | **+0.45%/-0.62%** | **0.822/0.695/0.752/0.538** | **-1.23%/-0.70%** |

`v01_symmetric`의 mAP50 기준 all 손실이 +0.45%(개선)인 반면 `v02_dair`는 -1.23%(손실) —
§1의 acceptance 판정과 일관됨.

### 3-8. 크기별 recall — 탐색적 분석 (2026-09-03, 완료)

**방법 변경 이력**: 처음 COCO 절대 기준(small<32²=1024px², medium<96²=9216px², large≥9216px²,
원본 1920x1080 좌표)으로 계산했으나, **bus/truck은 원래 항상 크게 찍혀 small 구간이
거의 0**이 되는 근본 한계 발견(예: bus small n=0, truck small n=2) — 사용자 지적으로
**클래스별 상대 기준(자기 분포의 33/66 percentile, tertile)**으로 교체. 정당성: 절대
기준은 클래스마다 표본 수가 극단적으로 불균형해(car small n=65) recall 추정이
불안정한 반면, tertile은 클래스별 표본을 균등 분배(car small n=2,407)해 모든
칸에서 비슷한 통계적 신뢰도를 확보한다. 단, "작다"의 절대 픽셀 의미가 클래스마다
다르다는 한계는 남는다(클래스 간 절대 크기 비교가 아니라 클래스 내 근/원거리
난이도 비교가 목적이므로 허용 가능).

**클래스별 tertile 경계(px², 두 모델 GT 공통)**: bus small<21,002/medium<78,961,
car small<4,546/medium<16,112, motorcycle small<5,941/medium<20,545,
person small<3,116/medium<11,070, truck small<15,913/medium<71,775.

**전체(all class) 크기별 recall**:

| Size | v01_symmetric Float/Board | v02_dair Float/Board |
|---|---:|---:|
| small | 0.406/0.416 | 0.381/0.345 |
| medium | 0.731/0.725 | 0.718/0.698 |
| large | 0.909/0.917 | 0.908/0.913 |

**결론**: `v02_dair`가 small/medium 구간(특히 truck: Float 0.397→0.323, Board
0.370→0.276)에서 `v01_symmetric`보다 뚜렷이 낮다 — §3-2/§3-6과 같은 방향, 격차가
주로 근/원거리 중 **원거리(작게 찍히는) 쪽**에서 벌어짐을 추가로 확인.
상세 클래스×크기 표는 세션 로그 참조(재계산 스크립트: `tools/debug/size_recall.py`,
`--`인자로 threshold/pred 경로 지정, 클래스 상대 tertile이 기본 동작).

### 3-9. "중요 인식 영역"(ROI) recall 탐색 — **미완결, 결론 아님** (2026-09-03/04)

사용자가 육안으로 가정한 ROI `[200,380]-[1720,1080]`(공식 스펙 아님 — README 입력
사양 표에 Rear_Truck 전용 crop/ROI 정의 없음, 확인됨)에 대해, GT box가 **면적의
30% 이상 겹치면 "영역 내"**로 포함하는 규칙(`tools/debug/roi_recall.py`)으로
in-ROI/out-ROI recall을 계산:

| | v01_symmetric Float in/out | v01_symmetric Board in/out | v02_dair Float in/out | v02_dair Board in/out |
|---|---:|---:|---:|---:|
| all | 0.893/0.646 | 0.905/0.649 | 0.886/0.632 | 0.896/0.610 |

in-ROI(가깝고 큰 개체 위주)는 두 모델이 거의 동급이고, **out-ROI(멀고 작은 개체)에서
`v01_symmetric`이 뚜렷이 우위** — §3-8과 같은 결론을 다른 방식으로 재확인.

**중요 — 이 ROI 자체는 검증 중 문제가 발견됐다**: GT 이미지 13장에 ROI를 그려 육안
확인한 결과, **좌측에 있는 크고 가까운 차량(버스/트럭)이 상단 경계(y=380)에 반복적으로
잘리는 패턴**(13장 중 5장)을 확인 — 이 어안 후방 카메라는 좌측 인접 차선의 근접
대형차일수록 화면 위쪽까지 찍히는 경향이 있어, 현재 ROI가 BSD에서 가장 중요한
"바로 옆 대형차량" 사례를 체계적으로 배제할 가능성이 있다. **경계값 조정 여부,
그리고 Front_Truck 리포트 방식(`edge`/`center` 좌우 3등분)으로 바꿀지, 지금의
y축 기준을 유지할지는 사용자 확인 대기 중 — 아직 결론 내리지 않았다.** 이 항목은
참고용 탐색 결과이지 §4 결정을 좌우하는 확정 근거가 아니다.

**제외한 것**: 오탐/누락 케이스 갤러리(`errorcases/`+montage)는 사용자 지시로
이번 범위에서 제외.

---

## 4. 결정 기준 (2026-09-02 §3-3/§3-6 및 acceptance 지표 확정 반영 갱신)

네 가지 독립적인 근거가 전부 같은 방향을 가리킨다:

1. **공식 acceptance 지표(mAP50, std+ultra)** — `v01_symmetric` 통과(+0.41%), `v02_dair`
   미달(-1.22%, 1% 초과)(§1).
2. **절대 성능(클래스별 board mAP)** — `v01_symmetric`이 거의 전 클래스에서 우위, 특히
   `v02_dair`는 person precision -0.084로 오탐이 뚜렷이 많다(§3-2).
3. **조건별(NIGHT 근사) 강건성** — DAIR의 강건성 훈련은 실재하는 것으로 확인됐지만(§3-1),
   NIGHT/non-NIGHT 어느 조건에서도 `v01_symmetric`의 절대 우위를 상쇄하지 못했다(§3-3).
4. **클래스별 F1-최적 threshold** — 별도 도구(`calc_class_thresholds.py`)로 독립 재계산해도
   bus를 제외한 4개 클래스에서 `v01_symmetric`의 F1이 더 높다(§3-6).

**결론: 현재 확보된 근거로는 `v01_symmetric`이 `v02_dair`보다 명확히 우세한 릴리즈
후보다.** `v02_dair`를 재고할 남은 근거는 §3-3이 정확히 겨냥하지 못한 조건(실제 프레임
하단 물리적 가림 — dashboard/mount)뿐이며, 이는 그런 조건의 이미지 서브셋이 실제로
존재해야 검증 가능하다(현재 미확인, §3-3 참조 — GT에 이 조건을 구분할 태그가 없다).

**(2026-09-03 확정) 사용자가 `v01_symmetric`을 릴리즈 후보로 확정하고 릴리즈 절차
진행을 지시했다.** §3-7~3-9(릴리즈 스펙 데이터, 크기별/ROI 탐색)도 전부 같은 방향을
재확인 — 결론을 바꾸지 않았다. §3-4(pre-softmax 분해)·§3-5(재현성 재학습)는 보류
유지(선택 사항).

이 기준 자체도 확정이 아니라 제안이다 — 사용자가 다른 우선순위(예: "가림 강건성보다
person 클래스 recall이 더 중요하다")를 가지면 그에 맞게 조정한다.

---

## 5. 진행 상황 트래커

| 항목 | 상태 |
|---|---|
| §1 quantization loss 재정리 | 완료 (기존 문서 인용) |
| §2 trainer 기능 차이 확인 | 완료 (코드 근거) |
| §2-1 `TensorSpeakerOverlayAug` uncommitted 변경 발견 | 완료, **해소됨**(§3-1) |
| §3-1 학습 시각화로 augmentation 활성 확인 | **완료** — 활성 확인됨 |
| §3-2 클래스별 비교 | **완료** — `v01_symmetric`이 거의 전 클래스 절대 mAP 우위, `v02_dair`는 precision 저하 확인 |
| §3-3 NIGHT 근사 서브셋 비교 | **완료** — `v01_symmetric`이 NIGHT/non-NIGHT 양쪽 다 우위 |
| §3-4 pre-softmax 분해 | 보류 (선택, 결론 바뀔 가능성 낮음) |
| §3-5 재현성 재학습 | 보류 |
| §3-6 클래스별 F1-최적 threshold | **완료** — bus 제외 4개 클래스 F1 `v01_symmetric` 우위 |
| §3-7 릴리즈 스펙 데이터(network info+Float/Board all row) | **완료** |
| §3-8 크기별 recall(클래스 상대 tertile) | **완료** — `v02_dair`가 small/medium에서 열세 재확인 |
| §3-9 중요인식영역(ROI) recall | **미완결** — ROI 경계 자체에 편향 의심 발견, 결론 아님 |
| 최종 결정 | **`v01_symmetric`으로 확정(2026-09-03, 사용자 지시)** — 릴리즈 절차 진행 중 |

---

## 6. 다음 단계

**모델 선정은 끝났다 — `v01_symmetric`으로 확정, 릴리즈 절차(operations_guide.md
§5-6) 진행 중.** 남은 작업:

1. README(`configs/experiments/BSD_F606/Rear_Truck/README.md`)·정본 문서
   (`docs/debug/20260806_rear_truck_v01_npu_quant_gap.md`) 최신화 — optimizer 결론과
   `v01_symmetric` 최종 선정 근거 반영. **아직 미실행.**
   - 실제 학습 데이터: `configs/datasets/1t_rear_cnr_postprocess.yaml`
     (`v01`(dair)과 동일, `configs/datasets/bsd_rear_truck.yaml`은 이 PC에 파일 없음)
   - NVT 설정 경로: `novatek/convert/rear_truck_v01_symmetric_320/nvtai_tool/config/
     REAR_TRUCK_V01_SYMMETRIC_320/cnn25/{gen_config.txt,sim_config.txt}`
2. `cp -r configs/experiments/.../v01_symmetric/ configs/releases/.../v01_symmetric/`
   + `weight/network/analysis/` 산출물 복사 + MANIFEST.yaml 작성. **아직 미실행.**
   - Front_Truck v01 릴리즈의 `analysis/` 폴더 구조(`metrics_*.json/txt`,
     `thr_tool_*.json`, `F1conf_*.png`, `iputest/`)를 참고 템플릿으로 확인함 —
     `thr_tool_*.json`/`F1conf_*.png`는 §3-6/3-7 산출물로 이미 대응 가능.
     `metrics_*.json`의 "regions"(edge/center) 항목은 §3-9가 미완결이라 보류 상태.
     `errorcases/`는 사용자 지시로 이번 범위에서 제외.
3. `ruff check && pytest` → 커밋(머지/푸시는 별도 확인 후).

§3-9(ROI 편향 의심)는 결정에 영향 없는 참고 탐색이므로 위 순서를 막지 않는다 —
필요하면 나중에 별도로 재검토.
