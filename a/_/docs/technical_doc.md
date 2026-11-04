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
- `configs/experiments/legacy/baseline/{rear,front}_dair_150ep.yaml` — 베이스라인 config
- `configs/experiments/legacy/baseline/{rear,front}_dair_ga.yaml` — GA 탐색 base
- `configs/experiments/legacy/baseline/{rear,front}_dair_full200.yaml` — GA best 200ep config
- `configs/experiments/legacy/baseline/front_baseline200.yaml` — 베이스라인 + 200ep production config
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
### Phase 0 — ncnvox mel pipeline — 2026-05-26 14:10 KST

**구현 내용**
ncnvox WAV → 96×96×3 PNG mel 파이프라인 (librosa fast path + NumPy reference + prepare CLI)

**주요 로직**
WAV → crop_or_pad(32000) → STFT(n_fft=512, hop=160, win=400 periodic-Hann 앞배치) → HTK mel(64bins,80-8kHz) → log(floor=1e-6) → min-max정규화 → bilinear resize(96×96) → 3ch 복제

**관련 파일**
- `projects/src/ncnvox/audio_config.py` — 보드/학습 공용 오디오 파라미터 단일 소스
- `projects/src/ncnvox/mel.py` — librosa fast path + NumPy reference 두 경로, 공통 후처리
- `tests/test_ncnvox_mel.py` — 12개 단위 테스트: cosine≥0.999 일치, shape/dtype, 헬퍼
- `tools/prepare_ncnvox_data.py` — kws/cmd 모드 WAV→PNG Ultralytics classify 폴더 생성 CLI

**인터페이스**
- 입력: WAV float32 [-1,1] 또는 int16 PCM, 임의 길이 (32000샘플로 crop/pad)
- 출력: log-mel [n_frames,64] float32 / 모델입력 [96,96,3] float32 [0,1]

---
### Phase 1 — KWS femto POC — 2026-05-26 13:15 KST

**구현 내용**
ncnvox KWS 2-class(background/hey_vueroid) 미니 분류기를 YOLOv8-cls femto scale(199K params)로 학습, ONNX export까지 완전한 파이프라인 구축

**주요 로직**
log-mel 96x96 이미지 -> yolov8-kws-femto(femto scale [0.33,0.16,256]) -> Classify head -> train/val -> ONNX export (best.onnx 0.8MB, ep39 stop)

**관련 파일**
- `projects/cfg/models/yolov8-kws-femto.yaml` — femto scale YOLOv8-cls 모델 정의 (nc=2, 96x96 입력)
- `configs/datasets/legacy/ncnvox_kws.yaml` — 2-class KWS 데이터셋 경로 명세
- `tools/train_ncnvox.py` — YOLO classify 학습 wrapper (yaml->폴더 경로 resolve 포함)
- `data/ncnvox_kws/` — Phase 0 prepare_ncnvox_data.py 생성 mel 이미지 (train 1203 / val 301)
- `runs/classify/runs/classify/kws_femto_poc-2/weights/best.pt` — 학습 완료 가중치 (ep38, 468KB)
- `runs/classify/runs/classify/kws_femto_poc-2/weights/best.onnx` — export 완료 ONNX (0.8MB, input [1,3,96,96] output [1,2])

**인터페이스**
- 입력: log-mel spectrogram 96x96x3 float32 [0,1] (grayscale 3채널 복제)
- 출력: (1,2) float32 softmax 확률 — index 0: background, index 1: hey_vueroid

---
### Phase 2.0 — train_ncnvox.py 경로 이중 prefix 수정 — 2026-05-26 17:10 KST

**구현 내용**
tools/train_ncnvox.py의 --project 인자를 Path.resolve()로 절대경로화해 ultralytics의 이중 prefix 문제 수정

**주요 로직**
args.project (상대경로 'runs/classify') → Path(args.project).resolve() → 절대경로 전달 → ultralytics가 settings.datasets_dir 기준으로 재해석하는 문제 차단

**관련 파일**
- `tools/train_ncnvox.py` — line 69: project_root = str(Path(args.project).resolve()) 적용. model.train(project=project_root)로 전달.

**인터페이스**
- 입력: --project runs/classify (상대경로 허용)
- 출력: 실제 저장 경로: /workspace/runs/classify/<name> (이중 prefix 없음)

---
### Phase 2.0 별건 — train_custom_trainer.py 경로 이중 prefix fix — 2026-05-26 15:40 KST

**구현 내용**
train_custom_trainer.py에서 ultralytics get_save_dir의 RUNS_DIR/<task>/<project> prepend로 인한 이중 prefix(runs/detect/runs/detect/<name>) 문제를 상대 project 경로를 절대 경로로 변환하여 회피

**주요 로직**
kwargs에 project 키가 있고 값이 상대 경로일 때 Path.resolve()로 절대 경로 변환. kwargs.update(cfg_options) 이후, ModelClass 인스턴스화 이전에 수행되어 cfg_options로 덮어쓴 값도 처리됨

**관련 파일**
- `tools/train_custom_trainer.py` — detect 학습 진입점 — line 71-76에 이중 prefix fix 추가

**인터페이스**
- 입력: kwargs['project']: str — exp_yaml overrides.project 또는 --cfg_options project=... 값
- 출력: kwargs['project']: str — 절대 경로로 변환된 값 (이미 절대 경로면 무변경)

---
### Phase 2.1 on-the-fly KWS dataloader (WakewordDataset + WakewordClassifyTrainer) — 2026-05-27 09:54 KST

**구현 내용**
TF ncnvox KWS 학습 파이프라인을 PyTorch ultralytics 기반 on-the-fly wav→mel 데이터로더로 포팅

**주요 로직**
ncnvox_kws_full.yaml(mode=wakeword_onthefly) → train_ncnvox.py 분기 → WakewordClassifyTrainer.train() → _build_train_pipeline → get_dataloader(mode=train/val) → build_wakeword_loaders() [positive 504 + SC2 neg 989, split seed=42, NoisePool 1015클립] → WakewordDataset [train: speed_perturb+time_shift+noise_mixer, val: deterministic] → _wakeword_collate [tuple→{img,cls} dict] → ClassificationTrainer 학습 루프

**관련 파일**
- `projects/src/ncnvox/wakeword_augment.py` — NoisePool + speed_perturb + time_shift + noise_mixer (numpy 기반, rng 인자)
- `projects/src/ncnvox/wakeword_dataset.py` — WakewordDataset(Dataset) + build_wakeword_loaders + _wakeword_collate
- `projects/src/ncnvox/wakeword_trainer.py` — WakewordClassifyTrainer: get_dataset/build_dataset/get_dataloader 오버라이드
- `configs/datasets/legacy/ncnvox_kws_full.yaml` — wakeword_onthefly 모드 데이터셋 yaml (경로, neg_words, noise_dirs, val_frac 등)
- `tools/train_ncnvox.py` — wakeword_onthefly 분기 추가 (_is_wakeword_onthefly 감지)
- `tools/train_custom_trainer.py` — Phase 2.0: detect 학습 이중 prefix fix
- `tests/test_ncnvox_wakeword_dataset.py` — 단위 테스트 4건 (shape, split, augment, class balance)

**인터페이스**
- 입력: WAV 파일 경로 목록 + 레이블. ncnvox_kws_full.yaml 경로
- 출력: __getitem__: (tensor[3,96,96] float32, int label∈{0,1}). DataLoader batch: {img: Tensor[B,3,96,96], cls: Tensor[B]}

---
### ncnvox WakewordDataset RNG 버그 fix — epoch-variant augmentation — 2026-05-27 10:30 KST

**구현 내용**
WakewordDataset.__getitem__ train 분기에서 epoch-불변 시드(seed+idx)를 제거하고 np.random.default_rng() 시스템 entropy 시드 사용으로 변경

**주요 로직**
train 분기: rng=np.random.default_rng() → crop_or_pad_to_clip(random_offset=True) → speed_perturb → time_shift → noise_mixer. val 분기: rng 없음, crop_or_pad_to_clip(random_offset=False) 만 호출

**관련 파일**
- `projects/src/ncnvox/wakeword_dataset.py` — WakewordDataset.__getitem__ RNG 시드 fix 적용 위치

**인터페이스**
- 입력: wav_paths: list[Path], labels: list[int], noise_pool: Optional[NoisePool], augment: bool, seed: int(split 재현성용)
- 출력: (tensor[3, 96, 96] float32, int label∈{0,1})

---
### ncnvox Phase 2.1 최종 통합 검증 (3차 — docstring fix + changelog) — 2026-05-27 10:50 KST

**구현 내용**
ncnvox TF/Keras KWS를 Ultralytics로 마이그레이션 — Phase 0(mel pipeline) + Phase 1(POC 학습) + Phase 2.1(on-the-fly dataloader + augmentation) 완료. docstring fix + changelog 추가로 최종 정합.

**주요 로직**
WAV→16kHz 리샘플 → crop_or_pad(2s) → [train: speed_perturb/time_shift/noise_mix via system entropy RNG | val: center crop only] → log-mel 96x96 → [3,96,96] tensor. split은 seed=42 고정 재현, augmentation은 매 epoch 다름.

**관련 파일**
- `projects/src/ncnvox/wakeword_dataset.py` — WakewordDataset + build_wakeword_loaders. seed: 설명 docstring fix 포함
- `projects/src/ncnvox/wakeword_augment.py` — NoisePool, speed_perturb, time_shift, noise_mixer
- `projects/src/ncnvox/wakeword_trainer.py` — WakewordClassifyTrainer (ClassificationTrainer 서브클래스)
- `docs/changelog.md` — Phase 2.0 + 2.1 마일스톤 2건 추가 (역시간순 맨 위)
- `tests/test_ncnvox_wakeword_dataset.py` — 4건 단위 테스트 (shape/dtype, split 재현성, val deterministic, class balance)

**인터페이스**
- 입력: ncnvox_kws_full.yaml (mode: wakeword_onthefly) + WAV 파일 디렉토리
- 출력: DataLoader → (tensor[3,96,96] float32 [0,1], int label∈{0,1})

---
### WakewordDataset.__init__ 미사용 파라미터 제거 (target_sr, clip_duration_s) — 2026-05-26 17:10 KST

**구현 내용**
WakewordDataset.__init__ 에서 self에 저장되지 않고 __getitem__ 에서도 참조되지 않던 dead 파라미터 2개(target_sr, clip_duration_s) 및 docstring 설명 삭제

**주요 로직**
시그니처: (wav_paths, labels, noise_pool, augment, seed=42) — 5개 파라미터만 유지. 동작 로직(__getitem__, build_wakeword_loaders, augmentation, mel 변환) 변경 없음

**관련 파일**
- `projects/src/ncnvox/wakeword_dataset.py` — WakewordDataset 정의 — 시그니처 청소 대상

**인터페이스**
- 입력: wav_paths: list[Path], labels: list[int], noise_pool: Optional[NoisePool], augment: bool, seed: int = 42
- 출력: __getitem__(idx) -> (tensor[3, 96, 96] float32, int label∈{0,1})

---
### docs 갱신 재검증 (Phase 2.2-2.6 + D-012 + eval_kws_baseline.py) — 2026-05-26 23:10 KST

**구현 내용**
docs 3개 파일 갱신 (decisions.md D-012 추가, changelog.md Phase 2.2-2.6, lessons.md NPU gen_config 교훈) + eval_kws_baseline.py 신규 추가

**주요 로직**
코드 변경 0 — 결정 기록·문서 갱신·평가 도구 신규만 포함. 회귀 가능성 없음. D-012 는 다음 학습 사이클에 코드 적용 예정.

**관련 파일**
- `docs/decisions.md` — D-012 입력 파이프라인 통일 결정 append
- `docs/changelog.md` — Phase 2.2-2.6 KWS NPU chain 완료 항목 역시간순 삽입
- `.claude/memory/lessons.md` — KWS NPU gen_config 교훈 2건 추가
- `tools/eval_kws_baseline.py` — TF ONNX vs ultralytics PT 등가성 비교 도구 (Phase 2.6)

**인터페이스**
- 입력: eval_kws_baseline.py: --baseline-onnx / --our-weights / --data-yaml / --threshold
- 출력: top1 / agreement / diff / PASS|FAIL 등가성 판정 stdout 출력

---
### D-013 KWS input pipeline alignment + sidecar — 2026-05-26 18:30 KST

**구현 내용**
KWS wakeword_dataset [0,1]→[0,255] 정합화 + 기존 학습 모델 input_scale.json sidecar 일괄 생성 + infer.py sidecar 자동 로딩

**주요 로직**
1) wakeword_dataset.py __getitem__: log_mel_to_model_input([0,1]) → *255 → [0,255] 텐서 반환. 2) mark_input_scale.py: 9개 best.pt 경로별 scale/trainer 하드코딩 테이블 → --apply 시 input_scale.json sidecar 생성. 3) infer.py: _load_input_scale() helper가 best.pt 옆 sidecar 읽어 'unit'이면 /255.0, 없으면 'byte' fallback. PTInferencer.predict()와 ONNXInferencer.preprocess() 양쪽에 분기 추가.

**관련 파일**
- `projects/src/ncnvox/wakeword_dataset.py` — KWS 데이터셋 [0,255] 출력 정합화 (line 174 추가)
- `tools/mark_input_scale.py` — 9개 best.pt에 input_scale.json sidecar 일괄 생성 스크립트 (신설)
- `tools/infer.py` — _load_input_scale helper + PTInferencer/ONNXInferencer sidecar 로딩 + /255 분기 추가

**인터페이스**
- 입력: best.pt 경로 (PTInferencer/ONNXInferencer 생성자)
- 출력: _load_input_scale → 'byte'|'unit'. PTInferencer.predict/ONNXInferencer.preprocess: input_scale에 따라 [0,255] 또는 [0,1] 텐서 생성

---
### Phase A — ncnvox CMD 데이터 인벤토리 + cfg yaml 초안 — 2026-05-27 10:30 KST

**구현 내용**
ncnvox CMD 8-class 분류기 위한 데이터 디렉토리 신설 및 yaml 초안 작성

**주요 로직**
data/ncnvox_cmd/ (tts 7class + SC v2 neg + noise) 신설, data/ncnvox_kws/ 경로 갱신 (temp/ → data/), data/legacy/ncnvox_kws_png/ 기존 PNG 보존, configs/datasets/legacy/ncnvox_cmd.yaml(mode=cmd_onthefly, nc=8), projects/cfg/models/yolov8-cmd-femto.yaml(KWS backbone 답습 nc=8) 신설

**관련 파일**
- `configs/datasets/legacy/ncnvox_cmd.yaml` — CMD 8-class 데이터 yaml — tts_dir/negative_dir/neg_words/noise_dirs 경로 정의
- `configs/datasets/legacy/ncnvox_kws_full.yaml` — KWS yaml — path를 temp/ncnvox/training/data → data/ncnvox_kws 갱신
- `projects/cfg/models/yolov8-cmd-femto.yaml` — CMD femto 모델 cfg — KWS backbone 답습, nc=8, 207K params

**인터페이스**
- 입력: data/ncnvox_cmd/{tts,cmd,noise,_mic_silence_}/ wav 파일 트리
- 출력: Phase B CommandClassifyTrainer 가 ncnvox_cmd.yaml + yolov8-cmd-femto.yaml 로 학습 가능한 상태

---
### Phase B1 — AudioClassificationDataset base 추출 + WakewordDataset 리팩토링 — 2026-05-27 11:20 KST

**구현 내용**
AudioClassificationDataset base를 신설하고 WakewordDataset가 이를 상속하는 빈 subclass로 리팩토링. backward-compat alias 4개 추가.

**주요 로직**
wav→mel→[3,96,96] float32 [0,255] 변환 로직을 AudioClassificationDataset._wav_to_tensor에 단일화. WakewordDataset는 라벨 의미(binary KWS)와 build_wakeword_loaders만 보유. module-level helpers(load_wav_16k 등)는 base 모듈에 정의, wakeword_dataset에 _prefix alias로 재노출.

**관련 파일**
- `projects/src/ncnvox/audio_classification_dataset.py` — 신설 — WAV→tensor 변환 base, module-level helpers 4개, audio_classify_collate
- `projects/src/ncnvox/wakeword_dataset.py` — 수정 — base 상속으로 단순화(180줄), backward-compat alias 4개, build_wakeword_loaders 그대로 유지
- `tests/test_audio_classification_base.py` — 신설 — 5개 단위 테스트 (concrete 인스턴스화, shape/dtype/range, 상속, collate dict, alias identity)

**인터페이스**
- 입력: wav_paths: list[Path], labels: list[int], noise_pool: Optional[NoisePool], augment: bool
- 출력: __getitem__ → (tensor[3,96,96] float32 [0,255], int label)

---
### Phase B2 — CommandDataset + CommandClassifyTrainer 신설 — 2026-05-27 15:30 KST

**구현 내용**
ncnvox CMD 8-class on-the-fly 데이터셋 + 트레이너 신설 (Phase B2)

**주요 로직**
CommandDataset은 AudioClassificationDataset 빈 subclass. build_command_loaders: tts/<cmd>/ 7클래스 → label 0~6, SC v2 neg_words bg_ratio 배수 cap sampling → label 7. CommandClassifyTrainer: ClassificationTrainer 상속, build_dataset/get_dataloader 오버라이드로 CommandDataset 주입. train_ncnvox.py: _data_yaml_mode 헬퍼로 wakeword/cmd 분기 통합.

**관련 파일**
- `projects/src/ncnvox/command_dataset.py` — CommandDataset 클래스 + build_command_loaders 빌더
- `projects/src/ncnvox/command_trainer.py` — CommandClassifyTrainer — ClassificationTrainer 서브클래스
- `tools/train_ncnvox.py` — _data_yaml_mode 헬퍼 + cmd_onthefly 분기 추가
- `tests/test_command_dataset.py` — 4개 단위 테스트 (상속, 라벨, collate, smoke)

**인터페이스**
- 입력: ncnvox_cmd.yaml (path, nc=8, names[7]=background, neg_words, bg_ratio, val_frac, tts_dir)
- 출력: (train_loader, val_loader) — batch {'img': [B,3,96,96] float32 [0,255], 'cls': [B] long 0~7}

---
### /simplify 사이클 1 — OnTheFlyClassifyTrainer base 추출 (D-015) — 2026-05-27 18:00 KST

**구현 내용**
WakewordClassifyTrainer/CommandClassifyTrainer의 5개 공통 메서드를 OnTheFlyClassifyTrainer base로 추출 — subclass는 _loader_builder 클래스 속성 1줄만 지정

**주요 로직**
_loader_builder=None(base) → staticmethod(build_*_loaders)(subclass). _ensure_loaders가 None 체크 후 NotImplementedError. _get_data_yaml이 절대/상대경로 처리 + 캐시. get_dataset/build_dataset/get_dataloader는 캐시된 loader 반환

**관련 파일**
- `projects/src/ncnvox/_base_trainer.py` — OnTheFlyClassifyTrainer base 신설 (~126줄)
- `projects/src/ncnvox/wakeword_trainer.py` — WakewordClassifyTrainer — base 상속 + _loader_builder 지정 (~20줄)
- `projects/src/ncnvox/command_trainer.py` — CommandClassifyTrainer — base 상속 + _loader_builder 지정 (~20줄)
- `docs/decisions.md` — D-013, D-014, D-015 항목 추가

**인터페이스**
- 입력: subclass에서 _loader_builder = staticmethod(build_*_loaders) 클래스 속성 지정. build_*_loaders(data_yaml_dict, batch_size, workers, seed) → (train_loader, val_loader)
- 출력: ultralytics trainer 호환 get_dataset/build_dataset/get_dataloader 제공

---
### /simplify 사이클 2 — yaml 1회 파싱 + command_dataset 단계 주석/rglob 정리 — 2026-05-27 09:30 KST

**구현 내용**
train_ncnvox yaml 파싱 2회→1회 리팩토링 + command_dataset 단계 주석/rglob pre-check 제거

**주요 로직**
main() 내 _data_yaml_mode(args.data) 1회 호출로 mode 결정, if mode in (...)으로 분기. _is_wakeword_onthefly/_is_cmd_onthefly는 _data_yaml_mode 위임으로 backward-compat 유지. command_dataset에서 split_wav_dir 후 빈 리스트로 empty 감지 (rglob pre-scan 제거)

**관련 파일**
- `tools/train_ncnvox.py` — yaml 1회 파싱, mode 기반 trainer 선택
- `projects/src/ncnvox/command_dataset.py` — 단계 주석 제거, rglob pre-check 제거

**인터페이스**
- 입력: data yaml 경로 (ncnvox_kws_full.yaml / ncnvox_cmd.yaml)
- 출력: trainer 선택 분기 또는 DataLoader tuple

---
### Phase C — CRNN PyTorch port + ultralytics 등록 — 2026-05-27 15:30 KST

**구현 내용**
TF CRNN(~554K params) PyTorch 1:1 port + ultralytics ClassificationModel 상속으로 standalone wrap 등록

**주요 로직**
nn.Module.__init__ 직접 호출로 parse_model 우회 → _build_crnn_sequential(nc)로 모델 생성 → _predict_once 오버라이드로 from-routing 제거 → train_ncnvox.py에서 model_arch=crnn 감지 시 trainer.model에 직접 주입 후 trainer.train() 호출

**관련 파일**
- `projects/src/ncnvox/cmd_crnn.py` — CRNNClassificationModel + 구성 블록 (InputRescale/ConvBNReLU/ReshapeToSequence/LSTMLastStep)
- `projects/cfg/models/yolov8-cmd-crnn.yaml` — 메타 yaml (nc=8, channels=3, model_arch=crnn) — 모델 정의는 코드에 있음
- `tests/test_ncnvox_cmd_crnn.py` — 단위 테스트 8개 (빌딩블록 + 전체 모델 forward/backward/param/nc)
- `tools/train_ncnvox.py` — _model_yaml_arch 함수 추가 + crnn 분기 (L101-136)

**인터페이스**
- 입력: CRNNClassificationModel(cfg: str|dict, ch=3, nc=None, verbose=True) — cfg는 yaml 경로 또는 dict
- 출력: forward(x: Tensor[B,3,96,96] float32 [0,255]) → Tensor[B,nc] float32 logits

---
### phase_d_poc_50ep — 2026-05-27 15:45 KST

**구현 내용**
ncnvox CMD 8-class (femto/crnn) 50ep POC 학습 완료 + augmentation override 버그 2건 수정

**주요 로직**
1) ep41 AttributeError 수정: close_mosaic=0을 _AUDIO_CLASSIFY_AUG_OVERRIDES에 추가. 2) mel-spectrogram image augmentation 비활성화: fliplr/mosaic/hsv/auto_augment/scale/translate/erasing 전부 0. 3) 해당 상수를 train_ncnvox.py 3분기(CRNN/wakeword+cmd_onthefly/ImageFolder) 모두에 spread 적용. 4) 2차 학습 실행: femto(GPU0), crnn(GPU1) 50ep 완주. femto best top1=0.99818, crnn best top1=0.99091.

**관련 파일**
- `tools/train_ncnvox.py` — _AUDIO_CLASSIFY_AUG_OVERRIDES 모듈 상수 도입, 3분기 spread 적용
- `runs/classify/cmd_femto_poc/results.csv` — femto 50ep 학습 결과
- `runs/classify/cmd_crnn_poc/results.csv` — crnn 50ep 학습 결과
- `docs/debug/20260527_phase_d_poc.md` — Phase D POC 비교 보고서 (핵심지점 수치 일부 부정확)

**인터페이스**
- 입력: configs/datasets/legacy/ncnvox_cmd.yaml (8-class CMD, train=6568, val=1650), epochs=50, batch=64, imgsz=96
- 출력: runs/classify/cmd_femto_poc/weights/best.pt (493KB, top1=0.99818), runs/classify/cmd_crnn_poc/weights/best.pt (1.13MB, top1=0.99091)

---
### phase_d_poc_50ep — 2026-05-27 21:10 KST

**구현 내용**
/simplify 후 on-the-fly 분기 dead args 제거 + 모듈 상수 주석 갱신 + doc 수치 정합화

**주요 로직**
on-the-fly 분기(wakeword/cmd_onthefly, CRNN)는 OnTheFlyClassifyTrainer.build_dataset/get_dataloader가 ClassificationTrainer 동명 메서드를 완전 shadow → ultralytics augmentation 경로 미도달 → _AUDIO_CLASSIFY_AUG_OVERRIDES 불필요. 해당 분기에서 close_mosaic=0 만 남김(ep41 AttributeError 회피 목적). ImageFolder POC else 블록은 ultralytics 경로 사용 → **_AUDIO_CLASSIFY_AUG_OVERRIDES 유지.

**관련 파일**
- `tools/train_ncnvox.py` — on-the-fly 분기 dead args 제거 + _AUDIO_CLASSIFY_AUG_OVERRIDES 주석 갱신
- `docs/debug/20260527_phase_d_poc.md` — Phase D POC 비교 보고서 — CSV 정합 수치로 수정

**인터페이스**
- 입력: train_ncnvox.py CLI (--model, --data, --epochs, --imgsz, --batch, --device 등)
- 출력: runs/classify/cmd_{femto,crnn}_poc/ (results.csv, weights/best.pt, args.yaml)

---
### infinite_dataloader_fix — 2026-05-27 23:40 KST

**구현 내용**
OnTheFlyClassifyTrainer의 DataLoader → InfiniteDataLoader 교체로 ultralytics BaseTrainer.reset() 계약 충족

**주요 로직**
command_dataset.py, wakeword_dataset.py의 DataLoader import를 ultralytics.data.build.InfiniteDataLoader로 교체. train_ncnvox.py에서 3분기 close_mosaic=0 명시 제거 + _AUDIO_CLASSIFY_AUG_OVERRIDES에서 close_mosaic 키 제거. InfiniteDataLoader는 DataLoader 서브클래스이므로 args 호환 100%

**관련 파일**
- `projects/src/ncnvox/command_dataset.py` — InfiniteDataLoader import, train/val loader 생성
- `projects/src/ncnvox/wakeword_dataset.py` — InfiniteDataLoader import, train/val loader 생성
- `tools/train_ncnvox.py` — close_mosaic=0 명시 제거, _AUDIO_CLASSIFY_AUG_OVERRIDES 상수 정리

**인터페이스**
- 입력: data_yaml_dict, batch_size, workers, seed
- 출력: (InfiniteDataLoader, InfiniteDataLoader) — train_loader, val_loader

---
### Phase E — CMD 정식 학습 (femto + crnn, 200ep) — 2026-05-27 07:30 KST

**구현 내용**
tools/train_ncnvox.py에 --cos_lr CLI flag 추가 후 femto(GPU0)/crnn(GPU1) 200ep 정식 학습 완료. 양 모델 EarlyStopping 정상 발동 (femto ep103, crnn ep86).

**주요 로직**
argparse --cos_lr(action=store_true) → 3분기(crnn/onthefly/POC) 모두 cos_lr=args.cos_lr 전달. cos_lr=True + ultralytics classify LR scaling 조합으로 peak LR~0.03(lr0=0.001의 약 30배). femto ep73/crnn ep56에서 top1=1.0 도달.

**관련 파일**
- `tools/train_ncnvox.py` — --cos_lr argparse flag 추가 + 3분기 cos_lr 전달
- `docs/debug/20260527_phase_e_full.md` — Phase E 결과 비교 문서 (신규)
- `runs/classify/cmd_femto_full/` — femto 200ep 학습 산출물 (best.pt, results.csv, args.yaml, plots)
- `runs/classify/cmd_crnn_full/` — crnn 200ep 학습 산출물 (best.pt, results.csv, args.yaml, plots)

**인터페이스**
- 입력: python3 tools/train_ncnvox.py --model <yaml> --data <yaml> --epochs 200 --cos_lr ...
- 출력: runs/classify/<name>/weights/best.pt + results.csv + args.yaml + plots

---
### Phase F — CMD NPU 변환/시뮬레이션 (femto + crnn) — 2026-05-27 08:30 KST

**구현 내용**
cmd-femto + cmd-crnn 두 모델 Novatek CNN25 NPU 변환/시뮬레이션 완료 — 양 모델 Softmax cosine=1.0

**주요 로직**
1) femto: YOLO.export(onnx) → cmd_femto/deploy_float.onnx 배치 → run_compiler.sh → run_simulator.sh. 2) crnn: tools/export_ncnvox_crnn.py (_CRNNExportWrapper Softmax 추가, 정적 입력, opset=12) → cmd_crnn/deploy_float.onnx 배치 → run_compiler.sh(LSTM RewriteLSTMONNX8 unroll) → run_simulator.sh. 3) tools/prepare_cmd_npu_cal.py로 val set mel JPG 50개 cal_img/ 추출

**관련 파일**
- `tools/export_ncnvox_crnn.py` — CRNN standalone 모델 ONNX export (Softmax wrap + 정적 형상)
- `tools/prepare_cmd_npu_cal.py` — CMD val set mel JPG 50개 추출 (NPU calibration)
- `docs/debug/20260527_phase_f_npu.md` — NPU 변환/시뮬레이션 결과 비교 문서
- `novatek/convert/cmd_femto/` — femto NPU 변환 디렉토리 (gen_config, sim_config, cal_img, nvtai_tool)
- `novatek/convert/cmd_crnn/` — crnn NPU 변환 디렉토리 (동일 구조, LSTM unroll 산출물)
- `runs/classify/cmd_femto_full/weights/best.onnx` — femto ultralytics export ONNX (829KB, opset=12)
- `runs/classify/cmd_crnn_full/weights/best.onnx` — crnn torch.onnx.export ONNX (2171KB, Softmax_output0_Y 출력명)

**인터페이스**
- 입력: best.pt (femto/crnn), CMD val set mel JPG (cal_img), gen_config.txt
- 출력: nvt_model.bin (femto 288KB 0.24ms, crnn 911KB 1.16ms), layer_diff.txt (cosine=1.000000)

---
### Phase G — D-014 사이클 마무리 (D-016 + glossary + lessons) — 2026-05-27 14:20 KST

**구현 내용**
D-014 사이클 마무리 — production 확정(cmd-femto), D-016 결정 기록, glossary 6개 용어, lessons LSTM NPU export 패턴 추가

**주요 로직**
docs/decisions.md D-016 append → docs/glossary.yaml 6개 term 추가 → .claude/memory/lessons.md LSTM export 패턴 1행 추가

**관련 파일**
- `docs/decisions.md` — D-016 신규 행 추가 (append-only, D-015 다음). cmd-femto production 확정 + LSTM NPU 호환 결론
- `docs/glossary.yaml` — CMD/cmd-femto/cmd-crnn/AudioClassificationDataset/OnTheFlyClassifyTrainer/RewriteLSTMONNX8 6개 용어 추가
- `.claude/memory/lessons.md` — 2026-05-27 PyTorch nn.LSTM NPU export 패턴 1행 추가 (InfiniteDataLoader 행 바로 다음)
- `tasks/todo.md` — Phase G plan 문서

**인터페이스**
- 입력: Phase F NPU 변환 결과 수치 (docs/debug/20260527_phase_f_npu.md) + Phase E 학습 결과 (docs/debug/20260527_phase_e_full.md)
- 출력: D-016 결정 기록, glossary 14개 항목(기존 8 + 신규 6), lessons 35줄

---
### Post-D-014 정리 — README classify 섹션 + CMD sidecar 정합화 — 2026-05-27 18:30 KST

**구현 내용**
README에 ncnvox audio classify 헤더 및 전용 섹션 추가, tools/mark_input_scale.py TARGETS에 CMD 4개 추가 후 sidecar 실제 생성

**주요 로직**
README line 3 blockquote 끝에 ncnvox 언급 추가 → line 134~180에 ## Audio Classify 섹션 신규 작성 (모델 표/명령어/데이터셋 정책). mark_input_scale.py TARGETS에 cmd_femto_full/cmd_crnn_full/cmd_femto_poc/cmd_crnn_poc 4행 추가(byte/classification) → --apply 실행으로 sidecar 4개 생성

**관련 파일**
- `README.md` — line 3 헤더 + line 134~180 Audio Classify 섹션 신규
- `tools/mark_input_scale.py` — TARGETS 리스트에 CMD 4개 추가 (기존 9→13)
- `runs/classify/cmd_femto_full/weights/input_scale.json` — CMD femto full sidecar 신규
- `runs/classify/cmd_crnn_full/weights/input_scale.json` — CMD crnn full sidecar 신규
- `runs/classify/cmd_femto_poc/weights/input_scale.json` — CMD femto poc sidecar 신규
- `runs/classify/cmd_crnn_poc/weights/input_scale.json` — CMD crnn poc sidecar 신규

**인터페이스**
- 입력: mark_input_scale.py --dry-run / --apply
- 출력: input_scale.json sidecar: {input_scale, trainer, trained_at, note}

---
### ncnvox classify 모델 4개 NPU 재변환 (postproc/en=0 수정) — 2026-06-03 10:15 KST

**구현 내용**
classify NPU 4개 bin의 출력 포맷을 postproc/en=0으로 수정 후 재컴파일·재시뮬레이션

**주요 로직**
gen_config [postproc/en]=1 + spec_demarc_tensors 조합이 AddPostProc_post_proc 노드를 그래프 끝에 자동 부착 → 출력 (1,5,2,1,1) INT32 40B. 두 항목 제거 후 재변환하면 그래프 끝이 Softmax 노드로 유지 → 출력 (1,1,nc,1,1) INT16

**관련 파일**
- `novatek/convert/cmd_crnn/gen_config.txt` — [postproc/en]=0, spec_demarc_tensors 주석 처리
- `novatek/convert/cmd_yolov8crnn/gen_config.txt` — [postproc/en]=0, spec_demarc_tensors 주석 처리
- `novatek/convert/cmd_femto/gen_config.txt` — [postproc/en]=0, spec_demarc_tensors 주석 처리
- `novatek/convert/kws_femto/gen_config.txt` — [postproc/en]=0, spec_demarc_tensors 주석 처리
- `novatek/convert/cmd_crnn/nvtai_tool/config/CMD_CRNN/cnn25/gen_config.txt` — mirror — 동일 수정
- `novatek/convert/cmd_yolov8crnn/nvtai_tool/config/CMD_YOLOV8CRNN/cnn25/gen_config.txt` — mirror — 동일 수정
- `novatek/convert/cmd_femto/nvtai_tool/config/CMD_FEMTO/cnn25/gen_config.txt` — mirror — 동일 수정
- `novatek/convert/kws_femto/nvtai_tool/config/KWS_FEMTO/cnn25/gen_config.txt` — mirror — 동일 수정

**인터페이스**
- 입력: classify ONNX (96×96 입력, 각 모델 nc=8 또는 nc=2)
- 출력: NPU bin — 출력 텐서 (1,1,nc,1,1) BLOB_TYPE_INT16, cmd 16B / kws 4B


### web_demo_removal_and_harness_cleanup — 2026-06-04 16:00 KST

**구현 내용**
web_demo 디렉토리 제거 및 6개 문서에서 web_demo 관련 내용 삭제, lessons.md 하네스 패턴 1행 추가

**주요 로직**
rm -rf web_demo → 문서 6개 grep 정리 → checklist.md 임시 시각/✅ 제거 → lessons.md 음성 데모 3행 제거 + 하네스 패턴 신규 1행 추가

**관련 파일**
- `/workspace/CLAUDE.md` — web_demo 명령어 블록 제거
- `/workspace/docs/checklist.md` — Phase2 row1 + Phase3 row1 시각/✅ 제거
- `/workspace/docs/completion_report.md` — web_demo verifier 보고서 섹션 제거 (700줄)
- `/workspace/docs/technical_doc.md` — web_demo 기술 섹션 제거 (621줄)
- `/workspace/.claude/memory/lessons.md` — web_demo 음성 3행 제거 + 하네스 패턴 1행 추가 (39줄)

**인터페이스**
- 입력: 없음 (문서/디렉토리 정리 작업)
- 출력: 없음 (문서/디렉토리 정리 작업)

---
### augmentation 모듈 7개 한국어 docstring 보강 — 2026-06-04 17:30 KST

**구현 내용**
augmentation 모듈 7개에 한국어 docstring을 추가하되 코드 동작은 일절 변경하지 않음

**주요 로직**
모듈 헤더("""...""") → 클래스 선언 직후 docstring → 공개 빌더 함수 직후 docstring 순서로 삽입. __init__/__call__/forward 등 자명 메서드는 의도적으로 제외. wakeword_augment.py는 이미 완비되어 변경 0.

**관련 파일**
- `projects/src/data/augment.py` — 모듈 헤더 + 12개 클래스 + 6개 빌더함수 docstring 추가
- `projects/src/data/augment_dark_paste.py` — 모듈 헤더 + DarkRegionBboxPasteAug docstring 추가
- `projects/src/data/augment_complexity_paste.py` — 모듈 헤더 + _BboxPasteBase / ComplexityBboxPasteAug docstring 추가
- `projects/src/data/augment_coco_seg.py` — 모듈 헤더 + CocoSegOverlayAug docstring 추가
- `projects/src/data/dynamic_crop.py` — 모듈 헤더 + DynamicCropWrapper / FlipOverlayAug / SpeakerOverlayAug docstring 추가
- `projects/src/data/dataset/dynamic_crop.py` — 모듈 헤더 + DynamicCropYOLODataset docstring 추가
- `projects/src/ncnvox/wakeword_augment.py` — 변경 없음 (기존에 한국어 docstring 완비)

**인터페이스**
- 입력: 없음 (docstring 추가 전용, 런타임 인터페이스 미변경)
- 출력: 없음 (코드 동작 무변경)

---
---

### 누적 도메인 제약 — ncnvox / Novatek NPU / ultralytics — 2026-07-27 (KST)

**구현 내용**

Claude Code 하네스(`.claude/memory/lessons.md`)에 누적돼 있던 기술 사실을 이관한다.
하네스 제거로 해당 파일이 사라지므로, 코드로 파악할 수 없는 도메인 제약만 여기에 보존한다.
행동 교정·절차 항목은 이관 대상이 아니다.

**NPU capacity 판단**

- 모델 사이즈 적정성은 스펙시트 추측이 아니라 **같은 칩에서 실증된 다른 모델**(파라미터·입력 해상도·FPS)을 기준으로 판단한다. NT98530 에서 YOLOv8n(~3M, 320×320)이 실시간 동작하므로 ncnvox CRNN 554K 를 "sweet spot" 이라 단정할 근거가 없었다.
- **이미 동작 중인 모델의 구성 layer 는 정의상 호환**이다. SiLU·Softmax·C2f 등에 "정밀도 손실 가능" 류의 경고를 붙이지 않는다. 호환성 경고는 그 모델에 없던 신규 op 를 도입할 때만 유효하다.

**classify 모델 NPU 변환 (KWS / CMD)**

- SDK 는 두 위치의 config 를 별도 운영한다: `<dir>/sim_config.txt` 와 `nvtai_tool/config/<P>/cnn25/sim_config.txt`. **후자가 컴파일러·시뮬레이터의 실제 입력**이다.
- 컴파일러가 `deploy_float.onnx`(raw export)를 입력으로 받아 **onnx2novaonnx 를 자동 호출**해 `deploy.onnx` 를 만든다 — classify 경로에서는 사전 변환이 불필요하다. raw `best.onnx` 를 `nvtai_tool/input/model/customer/<P>/deploy_float.onnx` 로 복사하면 된다.
- classify 는 `spec_demarc_tensors=Softmax_output0_Y` 16-bit 지정만으로 cosine=1.0 이 나온다.
- **`[postproc/en]=1` 은 시뮬레이터 accuracy 측정용이 아니라 runtime bin 에도 후처리 노드를 박는다** — 양산 변환 시 반드시 0. ncnvox 동봉 gen_config 가 명시적으로 0 인 이유가 이것이다.
- 보드 출력 shape 불일치 디버깅의 첫 단계는 `gentool/siminfo.txt` 의 마지막 `PROCESS_INFO` 에서 `tensor_name`/`origin_name`/`dim` 을 직접 읽어 컴파일러가 그래프 끝에 무엇을 붙였는지 확인하는 것이다(예: 자동 부착된 `AddPostProc_post_proc`).

**mel 입력 gen_config 정합성**

- `log_mel_to_model_input` 이 `np.stack([x,x,x], axis=-1)` 로 3채널에 동일 mel 을 repeat 하므로 R=G=B 가 보장된다 → **RGB/BGR/swap 선택이 결과에 무관**하다.
- `normalize/scale=1/255` 로 cal_img uint8 → `[0,1]` 복원만 정확하면 정합이 맞는다.
- calibration JPG quality=95 의 lossy 영향은 최대 2% (uint8 quantization 1/256 수준)로 분포 추정에 충분하다.

**PyTorch LSTM NPU export (D-016 확정)**

1. standalone wrap 모델은 ultralytics `YOLO.export()` 미지원 → `torch.onnx.export` 직접 호출 필수
2. `dynamic_axes=None` 정적 형상 강제 필수 — LSTM 의 dynamic shape 는 NPU 변환 불가
3. `spec_demarc_tensors=Softmax_output0_Y` 매칭을 위해 export wrapper 에 `torch.softmax(logits, dim=1)` 추가 (`tools/export_ncnvox_crnn.py`)
4. 컴파일러가 output name 을 `Softmax_<원본>_Y` 로 자동 rename 하지만 매칭은 그대로 동작 — gen_config 텐서명 변경 불필요
5. LSTM 1 layer → CNN-SCALEUP 24 timestep × 12 op ≈ 290 NPU layer 로 unroll (`RewriteLSTMONNX8`), cosine=1.0 검증됨

**ultralytics trainer subclass 의 DataLoader 계약**

- `BaseTrainer` 는 train_loader 가 자체 `InfiniteDataLoader` 계약을 따른다고 가정하고 `reset()` 등 framework-specific 메서드를 호출한다. 표준 `torch.utils.data.DataLoader` 를 반환하면 계약 위반이다(ep41 `AttributeError: 'DataLoader' object has no attribute 'reset'`).
- `close_mosaic=0` 은 트리거 회피일 뿐 근본 fix 가 아니다 — `from ultralytics.data.build import InfiniteDataLoader` 로 교체한다. `DataLoader.__init__` 을 호출하므로 args 호환이 100% 이고 worker reuse 이점도 따라온다.

**데이터셋 경로 작업**

- `configs/datasets/legacy/` 은 `FRONT/`·`REAR/`·`dms` 등 카테고리별 subdir 관행이 있다. 조사는 항상 재귀(`find -name '*.yaml'`, `grep -r`) — glob `*.yaml` 는 하위를 빼먹는다(top-level 16개로 오산, 실제 38개).
- 경로 정규화는 문자열 정합으로 끝내지 말고 **실데이터에 역매핑해 대조 검증**한다(train/val dir 존재 + images == labels 개수). 실디스크는 `Dashcam/<카메라>/{Train,Validation}` 구조이며 `DMS_stage_1` ↔ `Dashcam/DMS_stage1`, `DashCam` ↔ `Dashcam` 대소문자 등 어긋남이 있었다.

**런타임 / 컨테이너**

- 인터프리터·의존성 가용성은 base 로그인 셸이 아니라 **실제 런타임(uv `.venv`)** 에서 확인한다. `/usr/bin/python3` 에는 torch 가 없다. 실행 wrapper 는 `uv run` 을 박아 venv 를 캡슐화한다(`tools/dist_train_custom.sh`).
- 마운트·데이터·GPU 파이프라인 빠른 실증에는 호스트에 이미 있는 `ultralytics/ultralytics:latest` 로 실데이터 1-step(`fraction=0.003`, `model=yolov8n.yaml`)이 유효하다. 단 stock 이미지는 8.4.104 라 우리 fork/커스텀 트레이너는 미검증이며, 실행 후 데이터폴더의 root 소유 `labels.cache` 는 docker 로 정리해야 한다.

**관련 파일**

- `tools/export_ncnvox_crnn.py`, `projects/src/ncnvox/mel.py`, `projects/src/ncnvox/_base_trainer.py`
- `novatek/convert/` (compiler / simulator wrapper), `tools/dist_train_custom.sh`

**컨테이너화 (python 버전 / uv 설치 위치)**

- **이 repo 컨테이너화는 python 3.11+ 필수.** `uv.lock` 에 핀된 `onnxruntime-gpu==1.24.3` 이 cp311/cp312 휠만 제공하므로 ubuntu22.04 기본 python 3.10 에서는 `uv sync` 가 실패한다.
- base 이미지를 교체할 필요는 없다 — `UV_PYTHON=3.12` 로 uv 관리형 CPython 3.12 를 쓰면 torch 2.5.1+cu121 · ultralytics 8.4.6 이 정상 동작한다.
- uv 를 `/root/.local/bin` 에 설치하면 `/root` 가 0700 이라 non-root user(1000)가 실행하지 못한다. `UV_INSTALL_DIR=/usr/local/bin` 으로 시스템 경로에 설치한다.
- 검증 완료: py3.12 / torch 2.5.1+cu121 / `cuda.is_available()` True / 2 GPU / ultralytics 8.4.6 에서 커스텀 트레이너(로스 컬럼 `_m`/`_o`) Dashcam 1-step exit 0.
