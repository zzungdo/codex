# 변경 이력 (Changelog)

- **프로젝트명**: DAIR-YOLO
- **최초 작성**: 2026-03-13 (KST)
- **최종 통합**: 2026-05-14 (KST) — `docs/ACHIEVEMENT.md` 병합 (마일스톤 성과·잔여 리스크·누적 수정 파일 목록 흡수)

> 역시간순. 각 항목: 날짜, 목적/범위, 변경·수정·추가 내용.
> 마일스톤 단위 성과는 각 항목의 **목적** 줄과 결과 수치를 참조.

---

## 2026-05-26 — ncnvox KWS NPU 변환 chain 완료 (Phase 2.2–2.6)

**범위**: `tools/eval_kws_baseline.py` (신규), `novatek/convert/kws_femto/` (산출물 — .gitignore)

**진행**:
- **Phase 2.2 정식 학습**: `yolov8-kws-femto` 500ep 학습 → EarlyStopping at ep27, val top1=1.000, 0.030h (1.8분), best.pt 477KB (197,250 params)
- **Phase 2.3 ONNX export**: `best.pt → best.onnx` (opset 12, 0.8MB). 308 val 클립 PT vs ONNX cosine=1.000000, agreement=308/308, max abs diff=5.8e-5
- **Phase 2.4 NPU compile**: `run_compiler.sh kws_femto` → 76 layers, **nvt_model 280KB** (weight 212KB), 추론 0.241ms. compiler 가 자동으로 `deploy_float.onnx → deploy.onnx` 변환
- **Phase 2.5 NPU sim**: `run_simulator.sh kws_femto cal_0000.jpg` → **Softmax 출력 cosine=1.000000** (avg_diff=0.000215). 중간 layer 양자화 손실 0.97~1.0 → 최종 출력 perfect
- **Phase 2.6 baseline 등가성**: TF tc_resnet8 (119K params) vs 우리 (199K) val top1 = 1.0000 vs 1.0000, agreement 308/308 → ±2% PASS (diff=0)

**gen_config 정합성 (사용자 점검 요청 반영)**:
- `log_mel_to_model_input` 의 `np.stack([x,x,x],-1)` 로 R=G=B 보장 → RGB/BGR/swap 무관
- `normalize/scale=1/255` 로 cal_img uint8→[0,1] 복원만 정확하면 됨 (검증: max diff 2%)
- `meansub/en=0`, `spec_demarc_tensors=Softmax_output0_Y`

**파라미터 비교** (NT98530 capacity 기준): YOLOv8n 3.16M = 100% → 우리 KWS 199K = 6.3%, TF tc_resnet8 119K = 3.8%. 둘 다 capacity 여유 큼.

---

## 2026-05-26 — Detect 학습 결과 경로 이중 prefix fix (Phase 2.0)

**범위**: `tools/train_custom_trainer.py`

**문제**: ultralytics `get_save_dir` 가 상대 `project` 인자를 `RUNS_DIR/<task>/<project>` 로 prepend → `runs/detect/runs/detect/<name>` 이중 prefix 가 만들어짐. front_baseline200, rear_full200 등 production 가중치가 모두 이중 경로에 저장됨.

**수정**:
- `kwargs['project']` 가 상대 경로일 때 `Path.resolve()` 로 절대 경로 변환 → ultralytics prepend 우회
- 향후 detect 학습은 `runs/detect/<name>` 1단 출력
- 기존 production 가중치 경로 (`runs/detect/runs/detect/front_baseline200`) 는 그대로 보존 — 새 학습부터 적용

> 같은 버그가 `tools/train_ncnvox.py` 에도 있어 함께 fix (ncnvox migration commit 에 포함).

---

## 2026-05-26 — ncnvox → Ultralytics 마이그레이션 Phase 0+1+2.1 (KWS on-the-fly)

**목적**: ncnvox(TF/Keras KWS) 코드를 DAIR-YOLO 본 프로젝트로 흡수해 학습·NPU 변환 인프라 통합 (베이스라인 tag `ncnvox-migration-baseline-20260526` HEAD `d48b17e`).
**범위**: `projects/src/ncnvox/`, `projects/cfg/models/`, `configs/datasets/legacy/ncnvox_kws*.yaml`, `tools/{prepare_ncnvox_data,train_ncnvox}.py`, `tests/test_ncnvox_*.py`

### Phase 0 — mel pipeline (96x96x3 log-mel)
- `audio_config.py`, `mel.py`: librosa fast + NumPy reference 두 경로, cosine ≥ 0.999 검증
- `tests/test_ncnvox_mel.py`: 12 케이스 (입출력 shape/dtype, librosa vs reference, edge cases)
- `tools/prepare_ncnvox_data.py`: WAV → PNG 변환 (kws/cmd 모드)

### Phase 1 — POC KWS 학습 파이프라인
- `yolov8-kws-femto.yaml`: femto scale 모델 (~150K param 목표, TC-ResNet8 수준)
- `ncnvox_kws.yaml`: POC 디렉토리 기반 classify yaml
- `tools/train_ncnvox.py`: 학습 진입점 (--mode dispatch + yaml 인식)

### Phase 2.1 — on-the-fly KWS dataloader (TF 베이스라인 + 실시간 augmentation)

**정책 (D-011)**: positive 504(hey_vueroid 전체) + SC2 negative ~989(23단어 × 43 cap, val_frac=0.2 seed=42) + NoisePool(musan + esc50 + _mic_silence_, max_clips_per_dir=500)

**추가**:
- `wakeword_augment.py`: NoisePool, speed_perturb, time_shift, noise_mixer (TF `common/augment.py` 포팅)
- `wakeword_dataset.py`: WakewordDataset(index 기반) + build_wakeword_loaders + `_wakeword_collate` (tuple→dict)
  - train 분기: 시스템 entropy RNG → epoch-variant augmentation (verifier code_smell fix 후)
  - val 분기: deterministic center crop, augmentation 없음
- `wakeword_trainer.py`: WakewordClassifyTrainer (ClassificationTrainer 서브클래스). `get_dataset` / `build_dataset` / `get_dataloader` 오버라이드. `channels=3` 명시
- `ncnvox_kws_full.yaml`: `mode: wakeword_onthefly`. YAML 1.1 yes/no boolean 회피 위해 neg_words 따옴표 처리. ultralytics validator 통과용 train/val placeholder 포함
- `tests/test_ncnvox_wakeword_dataset.py`: 4건 단위 테스트 (shape/dtype, split 재현성, val deterministic, class balance)
- `docs/debug/20260526_ncnvox_kws_data_inventory.md`: 113,114 WAV 인벤토리 + TF 정책 분석

**검증**:
- pytest 40 passed / 1 skipped
- 1ep CPU dry-run val_top1=0.67208 (308 val 샘플 기준 final_eval)
- POC 회귀 dry-run 통과 (`ncnvox_kws.yaml` 디렉토리 기반 classify)
- verifier ✅ 2회 (RNG epoch-불변 버그 fix 후 재검증)

**dry-run 디버깅 fix (메인 발견·수정 5건)**: get_dataset 오버라이드, `channels=3`, yaml yes/no 따옴표, `collate_fn` tuple→dict, train/val placeholder.

**다음 단계 (Phase 2.2~6)**: 정식 500ep · patience=30 학습 → ONNX export → NPU 변환 → baseline ±2% 동등성 검증.

---

## 2026-04-27 — DAIR Refactoring: Super-Delegation Pattern

**목적**: BaseTrainer/BaseValidator 복붙 재구현 → `super()` 위임 전환 (ultralytics 업데이트 호환성 확보)
**범위**: `train/dair.py`, `val/dair.py`, `loss.py`

**문제**: `_do_train` (~157줄), `__call__` (~142줄)이 BaseTrainer/BaseValidator 복붙 재구현. ultralytics 업데이트마다 API 불일치 발생.

**수정**:
- `train/dair.py`: `_do_train()`, `build_dair_params()` 제거 (~167줄). BaseTrainer의 OOM/NaN recovery, BN freeze 자동 지원
- `val/dair.py`: `__call__()`, `update_metrics()`, `gt_to_json()` 제거 (~198줄). dict API 마이그레이션, `scale_preds()` 추가 (DynamicCrop 독립 x/y gain)
- `loss.py`: `v8DairDetectionLoss` 1D tensor (12 elements) 반환. Validator loss 누적 버그 수정 (`self.loss += ...`)

**순감소**: ~365줄 삭제, ~40줄 추가

---

## 2026-04-27 — GA Tuner: Workers Memory Fix & DAIR Search Space

**범위**: `tools/ga_tuner.py`

**수정**:
- workers=8 → 48 DataLoader workers → 73GB+ RAM → swap 100%. `workers = min(config, 4)` cap 추가. RAM 64GB → 35GB

**추가**:
- DAIR 하이퍼파라미터 탐색: 22→27 파라미터. `model_name`, `dair_lambda`, `dair_gamma`, `dair_method_mode`, `dair_num_choice` 추가

---

## 2026-04-27 — Custom Augmentations: SunFlareAug & DAIR Batch Integration

**목적**: 대시캠 detection 도메인 특화 augmentation
**범위**: `augment.py`, `dynamic_crop.py`, `train/dair.py`

**추가**:
- `SunFlareAug`: 태양 역광 시뮬레이션 (상단 35%, radial glow, 5-12 light rays, warm haze)
- `TensorSunFlareAug` / `TensorSpeakerOverlayAug`: GPU batch augmentation (DAIR pipeline)
- `SpeakerOverlayAug`: 대시보드 스피커 오버레이 + geometric transforms (scale 0.8-1.2, translate ±10%, rotate ±5°, shear ±3°)
- `ROIMotionBlur`: ROI 영역 zoom/linear motion blur
- `build_front_augmentations()` / `build_rear_augmentations()` 에 `RandomChoice` 통합

---

## 2026-04-26 — GA-Based Hyperparameter Tuner

**목적**: 수동 튜닝 대체 자동 하이퍼파라미터 탐색
**범위**: `tools/ga_tuner.py`, `tools/run_ga_tuning.sh`

**추가**:
- GA 엔진: LHS 초기화, SBX crossover, polynomial mutation, elite 보존
- 27개 탐색 파라미터: learning rates, augmentation, loss weights, optimizer, DAIR method
- 서브프로세스 기반 trial (`start_new_session=True`) + `_kill_process_tree()` 정리 (zombie/orphan 방지)
- Resume 지원: population, RNG state, generation counter, elapsed time 영속
- Configurable: trial timeout, max hours, patience, population size
- 듀얼 GPU 메모리 안전 운영 (workers=4 cap)

**후속 수정**:
- Zombie 프로세스: `os.killpg(SIGKILL)` fallback
- 반복 실패 시 무한루프 방지: N회 연속 실패 → break
- Resume RNG/counter 복원, timeout stdout 캡처

---

## 2026-04-25 — DMS Stage-2 Export & Unified Inference

**목적**: 모든 모델 타입/포맷 단일 추론 스크립트
**범위**: `tools/export_dms.py`, `tools/infer.py`

**변경**:
- Stage-2 export: xy-only keypoints (140 values = 70x2, no visibility) 레거시 포맷 매칭
- ONNX: raw DFL distances (decoded xywh 아님)

**추가**:
- `tools/infer.py`: PT/ONNX, detection/DMS 모드, stage 자동 감지, box/keypoint decoding, NMS, 시각화 (bounding boxes, class labels, 70-point facial landmarks)

---

## 2026-04-22 — DMS Pose: Scale-Split TAL & OKS Sigma Fix

**범위**: `loss.py`, `val/dms_pose.py`

**문제**: `mAP50(P)` = 0 (mAP50(B) = 0.995). 원인 2가지:
1. OKS sigma=1/nkpt≈0.014 → exp(-564)≈0 → vanishing gradient
2. 단일 TAL이 face GT를 P3/P4에 할당 → keypoint=0

**수정**:
- `DMSPoseLoss`: OKS sigma=0.1, scale-group TAL (face→P5, non-face→P3/P4), 그룹별 독립 normalizer
- `DMSPoseValidator`: `self.sigma = np.ones(nkpt) * 0.1`, P3/P4 detection에 P5 keypoint 복사

---

## 2026-04-21 — DMS Pose: Augmentation Overhaul & Real-Data Training

**범위**: `augment.py`, `train/dms_pose.py`, data/exp YAML

**수정**:
- `self.args.augment`는 TTA 플래그 (기본 False) — augmentation 게이트로 잘못 사용 → `if mode == "train":` 으로 변경
- `MaskLikeOcclusion`: `cv2.dilate` 기반 마진 확장, normalized GaussianBlur, 단일 마스크 합성
- `SunglassesCircleOcclusion`: `roundpent` 완전 구현, `roundrect` 회전 지원, per-lens max-blend

**추가**:
- `dms_stage1.yaml`, `dms_stage2.yaml` (data), `dms_stage1.yaml`, `dms_stage2.yaml` (exp config)

**학습 결과**:
- Stage-1: 150ep, mAP50-95(P)=0.995 (19K face images)
- Stage-2: 150ep, 5-class detection + keypoints (42K images)

---

## 2026-04-21 — DMS Pose: 70-Keypoint Integration

**목적**: 모델 아키텍처 → ONNX 변환까지 70-keypoint DMS facial pose 전체 파이프라인
**범위**: head.py, loss.py, tasks, trainer, validator, engine, export, augment

**추가**:
- `DMSPose(Pose)`: P5-only 70-keypoint head (jaw, brows, nose, eyes, mouth, pupils). P3/P4는 detection only
- `DMSPoseLoss(v8PoseLoss)`: `_assign_p5_faces()` P5 전용 TAL, 그룹별 독립 normalizer
- 70-keypoint 인덱스 상수, `get_landmark_weights()`, `relational_group_loss_coarse`
- `DMSPoseDetectionModel`, `DMSPoseTrainer`, `DMSPoseValidator`, `DMSPoseYOLO`
- `tools/export_dms.py`, `tools/create_dms_dummy_dataset.py`
- `MaskLikeOcclusion`, `SunglassesCircleOcclusion` augmentations
- ONNX export: 레거시 NPU 포맷 매칭 (raw DFL boxes, raw keypoints)

---

## 2026-03-18 — Finetune Pipeline: 3-Stage (Consistency → BN Freeze → QAT)

**목적**: 3-Stage finetune 파이프라인 (Consistency → BN Freeze → QAT)
**범위**: `tools/run_finetune_pipeline.py`

**변경**:
- Stage 2 = BN freeze (`CustomDetectionTrainer`), Stage 3 = QAT (`QATDetectionTrainer`)
- `--stage {1,2,3,export,all}` 지원, GPU cleanup between stages
- Consistency trainer: `dair_params=None` 시 표준 `v8DetectionLoss` 경로 사용

---

## 2026-03-17 — BN Freeze Support

**범위**: `train/custom.py`, `tests/test_bn_freeze.py`

**추가**:
- `_apply_bn_freeze()`: `nn.BatchNorm2d` + `nn.SyncBatchNorm` → eval()
- `_model_train()`: `exp_params.bn_freeze=True` 시 BN freeze 적용
- 7개 pytest 테스트

---

## 2026-03-16 — ultralytics 8.4.6 호환 수정

**범위**: augment.py, loss.py, tal.py, train/consistency.py, val/consistency.py, pipeline

**수정**:
- `Detect.forward_head()` dict 반환 대응 (`isinstance(feats, dict)`)
- `select_highest_overlaps()` align_metric 인자 추가
- `ROIMotionBlur`/`ROIPaste`: `np.random.randint` 1-element array 수정
- `Bboxes` attribute `.data` → `.bboxes`
- `v8_transforms_dycrop` MixUp 해상도 불일치 수정
- `SymmetricDetectionLoss`: default gamma/lambda, hook stubs, dair_params None 처리

---

## 2026-03-13 — QAT Pipeline: INT16 + Novatek CNN25 NPU Targets

**목적**: Quantization-Aware Training 파이프라인 구축 및 버그 수정
**범위**: `quant_utils.py`, `train/qat.py`, `base_qat.yaml`

**추가**:
- `novatek_8a12w`, `novatek_pct_8a12w`: INT8 activation + 12-bit per-channel weight
- `generic_int16`, `percentile_int16`: INT16 시뮬레이션
- `PercentileObserver`: histogram 기반 percentile calibration

**수정**:
- AMP 비호환 근본 원인 수정: `HistogramObserver` fp16 → `MovingAverageMinMaxObserver` 교체
- `PercentileObserver.calculate_qparams`: private API → public path
- INT16 dtype overflow: `torch.qint8` → `torch.qint32`
- `_convert_and_save_int8` guard flag 순서, 파일명 동적화
- `get_qconfig` dispatch: if/else → `_PERCENTILE_FACTORIES` dict

**코드 리뷰**: 17건 발견 → 16건 수정 완료 (High 5, Medium 7, Low 4). 상세는 `docs/debug/20260313_qat_review.md`, mixed-precision 분석은 `docs/specs/qat_mixed_precision.md`

> ⚠️ QAT 워크플로우는 2026-05-13 미사용 결정 (D-010) — 코드·config 보존, 부활 시 즉시 사용 가능.

---

## 2026-03-13 — Fix tune_custom_trainer.py and custom_tuner.py

**범위**: `tools/tune_custom_trainer.py`, `custom_tuner.py`

**변경**:
- 불필요 import 제거, `--use_ray` 플래그, `cfg.overrides` 적용
- `custom_cfg_path` hardcoded fallback → `ValueError` guard
- hardcoded device `-d "0,1"` → config 기반 device 전파

---

## QAT Pipeline 잔여 리스크 (2026-03-13 기준)

> QAT는 D-010으로 미사용 처리됐으나, 부활 시 아래 리스크가 그대로 유효하다.

- `build_dataset` 가 `mode='val'` hardcode — 사용자 결정으로 유지
- `base_qat.yaml` 절대 경로 — 머신별 수동 편집 필요
- Multi-GPU QAT 미지원 (`RuntimeError` 가드)

---

## 누적 수정 파일 목록

> 위 마일스톤 전반에 걸쳐 수정된 핵심 파일.

| 파일 | 변경 내용 |
|------|----------|
| `train/qat.py` | AMP disable, validate() override, on_train_end callback, multi-GPU guard |
| `val/qat.py` | eval() guard, exception logging |
| `dynamic_crop.py` | safe ratio parser, randint clamp, SpeakerOverlayAug 경로/None-check |
| `quant_utils.py` | Observer 교체, INT16/Novatek qconfig 추가 |
| `quant_model.py` | fusion failure logging |
| `cfg.py` | `_return_dict` + cycle detection |
| `export_custom.py` | assert → raise FileNotFoundError |
| `train_qat.py` | multi-GPU guard, checkpoint comment |
| `train/dair.py` | super() 위임 전환 |
| `val/dair.py` | dict API 마이그레이션 |
| `loss.py` | 1D tensor 반환, SymmetriceDFL offset 수정 |
| `tal.py` | SymmetricTAL/DairTAL align_metric 인자 대응 |
| `train/consistency.py` | loss_names 3-element, dair_params None 처리 |
| `val/consistency.py` | loss 누적 패턴 수정 |
| `augment.py` | SunFlareAug, MaskLikeOcclusion, SunglassesCircleOcclusion 등 |
| `ga_tuner.py` | workers cap, DAIR search space, zombie 방지 |
