# ncnvox 원본 — 디렉토리·데이터·보드 통합 reference

> 위치: `/workspace/temp/ncnvox/` (원본), `/workspace/data/ncnvox_cmd/` (우리 학습용 subset)
> 백업: `ncn@192.168.103.163:/mnt/HDD_1/DATA/Datasets/99_RAW/ncnvox_training_data/` (16 GB)
> 작성: 2026-05-29 (KST)

ncnvox 는 **Novatek NT98530 NPU 보드용 음성인식 production stack** 입니다. 모델 학습 코드뿐 아니라 데이터 수집·NPU 변환·보드 runtime·테스트·문서까지 다 들어있습니다. 본 문서는 **모델 자체 외 부분** (데이터 파이프라인, NPU 워크플로우, 보드 통합, 테스트, 문서) 위주로 정리합니다.

---

## 1. 디렉토리 구조 (top-level)

```
/workspace/temp/ncnvox/
├── README.md                  전체 진입점
├── CLAUDE.md                  Claude Code용 작업 가이드 (우리 repo 와 별도)
├── build.sh                   빌드 entry
├── requirements.txt           Python 의존성
│
├── training/                  ★ 학습 + ONNX export + NPU 변환 (Python/TF)
├── libncnvox/                 ★ 보드용 runtime 라이브러리 (C 소스)
├── lib/nt98530/               ★ NT98530 칩셋 SDK (bin / include / lib)
├── 3rdparty/                  외부 종속 (aarch64 / nt98530 / tflite)
├── test/                      통합 테스트 + 데이터 수집 도구
└── docs/                      문서 (10+ md/pdf — 본 문서 §7 참고)
```

분량 정리:
- `training/data/` 16 GB (실제 raw wav 풀)
- `lib/nt98530/` ~수십 MB (보드 binary + lib)
- `3rdparty/` ~수백 MB (aarch64 / tflite / Novatek SDK 일부)
- `libncnvox/` ~수백 KB (C 소스)

---

## 2. training/ — 데이터·학습·NPU 변환

### 2.1 디렉토리

```
training/
├── data/                      raw 데이터 풀 (16 GB)
│   ├── tts/<command>/         7 명령어 × 392 wav (TTS 합성)
│   ├── cmd/.../sc_v2/         Speech Commands v2 (background용, 105K wav)
│   ├── custom_words/hey_vueroid/  504 wav (KWS TTS)
│   ├── noise/musan/, esc50/   4,016 wav (SNR 노이즈 믹싱용)
│   ├── _mic_silence_/         15 wav (실보드 마이크 무음)
│   ├── noise_car/             0 wav (예정 — 차량 노이즈)
│   ├── openslr/               0 wav (예정 — 다국어/억양)
│   └── DATASETS.md            ★ 데이터 출처/구성 + 외부 데이터셋 추천 가이드
│
├── models/                    모델 정의
│   ├── crnn.py                CMD 8-class (Conv + LSTM 128, ~554K)
│   └── tc_resnet8.py          KWS 2-class (pure Conv, ~66K)
│
├── train/                     학습 entry point
│   ├── train_crnn.py          CMD 학습 (100ep default)
│   └── train_tc_resnet8.py    KWS 학습
│
├── datasets/pipeline.py       데이터 파이프라인 (tf.data, augmentation 조립)
│
├── common/                    공용 모듈
│   ├── audio_config.py        ★ sample_rate, mel, frame, image 사양 — SSOT
│   ├── mel_preprocess.py      waveform → log-mel → 96×96×3 image
│   └── augment.py             ★ NoisePool + speed_perturber + time_shifter + noise_mixer
│
└── exports/                   학습 산출물
    └── novaic_bundle/         NPU 변환 완료된 모델들
        ├── crnn_best/         (CMD)  deploy.onnx + nvt_model.bin + gen_config + label
        └── tc_resnet8_best/   (KWS)  동일 구조
```

### 2.2 데이터 파이프라인 (`datasets/pipeline.py`)

**클래스 분배 정책** (`build_command_datasets`):
1. TTS 7 명령어 → label 0~6 각각
2. `bg_ratio=2.0` → background 총 클립 = positive 총 × 2 로 cap
3. Speech Commands v2 23개 단어 (yes/no/up/down/...) 에서 cap/단어 만큼 → label 7 (background)
4. `val_frac=0.2` 로 train/val split (deterministic seed)

**Augmentation chain** (`_make_augment_fn`):
```
waveform
  ↓ speed_perturber  (rate ∈ [0.9, 1.1], prob=0.5)
  ↓ time_shifter     (±10%, prob=0.5)
  ↓ noise_mixer      (SNR ∈ [5, 20] dB, prob=0.8)  ← 우리는 [0, 20] (차이!)
  ↓ waveform_to_log_mel  →  log_mel_to_model_input  →  96×96×3 [0,1]
```

→ 우리 PyTorch port (`projects/src/ncnvox/wakeword_augment.py`) 와 비교 시 **SNR 하한이 우리 0 vs ncnvox 5** 가 가장 큰 차이.

### 2.3 Audio config (`common/audio_config.py`)

```
TARGET_SAMPLE_RATE_HZ = 16000
CLIP_DURATION_S       = 2.0            (clip = 32,000 samples)
SLIDING_HOP_MS        = 100            (실시간 추론 stride)
FRAME_LENGTH_SAMPLES  = 400            (~25 ms window)
FRAME_STEP_SAMPLES    = 160            (~10 ms hop)
FFT_LENGTH            = 512
NUM_MEL_BINS          = 64
LOWER_EDGE_HZ         = 80
UPPER_EDGE_HZ         = 8000
MODEL_IMAGE_SIZE      = (96, 96)       (3채널 복제 → 96×96×3)
```

→ 보드 inference 사양과 SSOT. 변경 시 학습/배포 모두 같이 갈리면 안 됨.

### 2.4 NoisePool

```
NoisePool.from_dirs(
    noise_dirs       = [data/noise/musan, data/noise/esc50],
    mic_silence_dir  = data/_mic_silence_,
    max_clips_per_dir= 500   (RAM 절약)
)
```

실제 우리 학습 시 log:
```
[NoisePool] musan: 500 클립
[NoisePool] esc50: 500 클립
[NoisePool] _mic_silence_: 15 클립
[NoisePool] 총 1015 클립
```

---

## 3. NPU 변환 워크플로우 (Novatek toolchain)

### 3.1 전체 파이프라인

```
1. Keras (.h5)  학습 산출물
   ↓
2. ONNX export (TF→ONNX)
   ↓
3. onnx2novaonnx  (Novatek 표준 ONNX → nova ONNX, 자동)
   ↓
4. nvt compile    deploy_float.onnx → nvt_model.bin
   ↓
5. simulator      nvt_model.bin → 시뮬레이션 결과
   ↓
6. 보드 시험      run_board_wav_eval_models.sh (24 wav)
```

### 3.2 표준 변수 (`scripts/run_npu_convert.sh`)

```
PREPROC_PROFILE   = norescale       uint8 → (x-127.5)/127.5 = [-1,1] (QAT 정합)
PRECISION_PROFILE = hp_cle_pc       per-channel weight quant + CLE
CALIB_MODE        = 1               min/max (4=percentile, 0=KL)
QUAN_EN           = 1               int8 weight 압축
HP_CONV_OP        = 0               size 안전 (1=host KL, 사이즈↑)
CALIB_LIMIT       = 100             mel calib 이미지 수
```

### 3.3 실험 자동화

- `scripts/run_npu_convert.sh` 호출 → CSV (`docs/npu_experiments.csv`) 자동 append
- 같은 호출이 `docs/NPU_BOARD_EXPERIMENT_LOG.md` 의 markdown stub 도 append
- 보드 시험 후 사람이 stub 안에 acc/top1-bias/confusion 채움
- 4주간 38개 entry 누적 (2026-05-13~05-19)

### 3.4 실험 통계 요약

| 모델 | 보드 평가 결과 | 상태 |
|---|---|---|
| `mobilenet a=1.0` (id 1, 9) | 6/24 (25%), _unknown_/_silence_ collapse | 실패 |
| `mobilenet a=0.5` (id 6) | 2/24 (8.3%), all_pred_down | 실패 |
| `dscnn_l_qat` (id 14~20) | fit-safe 만 (보드 검증 미진행) | 보류 |
| `crnn_dryrun` (id 33) | LSTM 동작 확인, NPU latency 2.1ms | OK (toy) |
| **`crnn_qat`** (id 34) | **TODO** (변환 OK, 보드 평가 미실행) | 미완료 |
| `tcresnet8_qat` (id 36) | 10/24 (41.7%), _unknown_ soft collapse | mel mismatch |
| `tcresnet8_qat_yuv420` (id 37) | **KERNEL_PANIC** (DLA_RESET) | 실패 |
| **`tcresnet8_plain_qat`** (id 38) | GPU val 96%, **보드 full eval 필요** | 미완료 |

→ **본격 production 모델의 실 발화 평가는 미완료 상태로 작업 멈춤** (2026-05-19).

---

## 4. 보드 통합 (NT98530 NPU)

### 4.1 보드용 binary 위치 (`lib/nt98530/`)

```
lib/nt98530/
├── bin/                       보드용 실행 파일 (aarch64)
│   ├── ncnvox_test            기본 inference 테스트
│   ├── ncnvox_mic_test        마이크 입력 → inference 실시간 데모
│   ├── ww_model.bin           NPU compiled — KWS 모델
│   ├── ww_label.txt           negative / hey_vueroid
│   ├── cmd_model.bin          NPU compiled — CMD 모델
│   └── cmd_label.txt          7 commands + background
├── include/ncnvox.h           public API
└── lib/libncnvox.so           shared library
```

### 4.2 ncnvox runtime 라이브러리 (`libncnvox/`)

```
libncnvox/
├── include/                   public C 헤더
├── src/                       C 소스 — mel 전처리, NPU inference, 후처리
├── lib/                       빌드 산출물 (.a / .so)
├── reference/                 참고 자료
├── docs/DESIGN.md             설계 문서
├── Makefile
└── README.md
```

보드 위에서 stream 처리 흐름:
```
mic PCM int16  →  /32768.0  →  mel  →  96×96×3 mel image
                                          ↓
                                        NPU inference
                                          ↓
                                  KWS (2-class) + CMD (8-class)
                                          ↓
                                  threshold 후 명령 trigger
```

### 4.3 펌웨어 (이 디렉토리에는 **없음**)

- OS kernel image (`uImage` / `zImage`) — 없음
- Device tree (`.dtb`) — 없음
- Bootloader (U-Boot 등) — 없음
- Root filesystem — 없음

→ Novatek 의 별도 BSP/SDK 패키지에서 빌드/flash 한 보드 위에 이 stack 의 lib + bin + model 을 복사해서 동작.

---

## 5. test/ — 통합 테스트 + 데이터 수집

```
test/
├── ncnvox_test/              기본 unit test (lib API)
├── ncnvox_mic_test/          마이크 입력 → 실시간 추론 (보드 데모)
├── nt98530_audio_rt/         NT98530 realtime audio 파이프라인
├── voice_reg_wav_rec/        ★ wav 녹음 도구 (사용자 발화 수집)
└── googledataset/            Speech Commands v2 처리/검증
```

각 디렉토리 안에는 `Makefile`, `main.c`, 빌드 스크립트 가 있음. **`voice_reg_wav_rec/` 가 실 발화 데이터 수집 시 활용 가능** — 외국인 300명 데이터 수집 계획의 도구로 재사용 검토.

---

## 6. 외부 의존 (`3rdparty/`)

```
3rdparty/
├── aarch64/                  보드 ARM 용 미리 빌드된 라이브러리
│   └── include/lvgl/         LVGL GUI (UI 표시용)
├── nt98530/                  Novatek SDK source 일부 (vendor)
│   └── novaic/ai2/           AI2 toolchain (NPU 컴파일러)
└── tflite/                   TensorFlow Lite (비교용 또는 fallback)
```

→ 보드 빌드 시 vendor SDK 의존, x86 학습 시 TF/Keras 의존.

---

## 7. `docs/` — 문서 인덱스

| 파일 | 내용 |
|---|---|
| `README.md` | 문서 디렉토리 진입점 |
| **`AI_Speech_Recognition_Primer.md`** | 입문서 (개념/용어/도메인 overview) |
| **`AI_Speech_Recognition_Comprehensive_Guide_NT98530_260417_153114.md`** (+.pdf) | 종합 가이드 (학습 → 변환 → 보드 전 단계) |
| **`KWS_PARAMS_AND_TROUBLESHOOTING.md`** | 학습/export/변환/보드런타임/진단 카탈로그 |
| **`CNN25_정밀도_조정_경험_For_Customer_한글정리.md`** | Novatek 컴파일러 정밀도 조정 노하우 (한국어) |
| `cnn25网络精度调整经验_for_cus.docx` | 위 한국어 정리의 원본 (중국어) |
| **`NOVAIC_ONNX_TO_BIN_PIPELINE.md`** | ONNX → 보드 bin 변환 전체 흐름 |
| **`NPU_BOARD_EXPERIMENT_LOG.md`** | 38개 실험 시간순 로그 (CSV 와 페어) |
| `npu_experiments.csv` | 38개 실험 정량 CSV (board_acc, bin_bytes, compile_status) |
| `NT98530_AUDIO_RT_OPTIONS.md` | 실시간 audio 옵션 (지연/버퍼) |
| `NT98530_NPU_SAMPLES_VS_SPEECH.md` | Novatek vendor_ai 샘플 vs ncnvox 코드 매핑 |
| `GPU_TRAINING_SETUP.md` | 학습 환경 셋업 |
| `TRAINING_WINDOW_AND_TUNING.md` | 학습 윈도우 + tuning |
| `VOICE_REG_WAV_REC_MINI_SPEC.md` | 사용자 발화 수집 도구 spec |
| `NC_Speech_Recognition(ncnvox)_Intro.pdf` | 프로젝트 소개 (외부 공유용) |
| `gen_ppt.py` | (보고용 ppt 생성 스크립트?) |

### 가장 가치 있는 reference 페어
- `npu_experiments.csv` ↔ `NPU_BOARD_EXPERIMENT_LOG.md` — 4주 실험의 정량 + 정성 기록
- `CNN25_정밀도_조정_경험_*` — Novatek 컴파일러 정밀도 hyperparameter 의 의미·튜닝 가이드 (한국어)
- `KWS_PARAMS_AND_TROUBLESHOOTING.md` — 학습 후 단계 별 진단

---

## 8. 빌드 / 실행

### 8.1 학습 (x86 GPU)

```bash
cd /workspace/temp/ncnvox/training
pip install -r requirements.txt

# CMD 학습 (100ep default)
python train/train_crnn.py --data-dir data --epochs 100

# KWS 학습
python train/train_tc_resnet8.py --data-dir data --epochs 100
```

### 8.2 NPU 변환

```bash
cd /workspace/temp/ncnvox
bash scripts/run_npu_convert.sh <model_path>   (scripts/ 는 우리 환경에 없으면 따로 받아야)
```

산출물: `nvt_model.bin` (보드 배포용) + `deploy.onnx` + `gen_config.txt`.

### 8.3 보드 배포

1. 보드 펌웨어 (Novatek BSP) 가 부팅된 상태에서 시작
2. SSH/serial 로 접속
3. `lib/nt98530/lib/libncnvox.so` 를 `/usr/lib/` 복사
4. `lib/nt98530/bin/*` 을 보드의 작업 디렉토리에 복사
5. `./ncnvox_test` 또는 `./ncnvox_mic_test` 실행

---

## 9. 이 stack 의 현 상태 (2026-05-29 기준)

### 완료
- ✅ 학습 파이프라인 (CRNN / TC-ResNet8 둘 다)
- ✅ NPU 변환 (deploy.onnx + nvt_model.bin 산출물 존재)
- ✅ 보드 runtime lib + binary (`libncnvox.so`, `ncnvox_test`, `ncnvox_mic_test`)
- ✅ 24 wav 작은 평가셋 + 변환 자동화
- ✅ NPU 정밀도 조정 노하우 문서 (`CNN25_정밀도_조정_경험_*`)

### 미완료
- ❌ **production 모델의 실 발화 본격 평가** — 다양한 화자/환경, 큰 평가셋
- ❌ `crnn_qat` 보드 평가 (TODO 로 남음)
- ❌ `tcresnet8_plain_qat` 보드 full eval (val_acc=96% 까지만, 보드 검증 미진행)
- ❌ `noise_car`, `openslr` 데이터 (디렉토리 비어있음, 추후 수집 예정)
- ❌ 사용자 발화 수집 도구 (`voice_reg_wav_rec`) 실 운영

### 흥미로운 발견
- ncnvox 원본팀이 이미 **CRNN (LSTM) vs TC-ResNet8 (LSTM-free)** 두 가지 architecture 를 분리 설계
  - KWS = TC-ResNet8 (단순, NPU 안전)
  - CMD = CRNN (LSTM, NPU 변환 risk 있음)
- 즉 LSTM 못 쓸 경우의 fallback 을 architecture 단에서 미리 분리

---

## 10. 우리 (DAIR-YOLO) 와의 관계

| 항목 | ncnvox 원본 | 우리 (D-014 이후) |
|---|---|---|
| Frame | TensorFlow / Keras | PyTorch (ultralytics 베이스) |
| CMD 모델 | crnn.py (TF) | `cmd_crnn.py` (PyTorch 1:1 port) + `cmd_yolov8_crnn.py` (yolov8n + LSTM hybrid) |
| KWS 모델 | tc_resnet8.py (TF) | `kws_femto` (yolov8n-cls scale femto — port 차이 있음) |
| 데이터 | `temp/ncnvox/training/data/` (16 GB) | `data/ncnvox_cmd/` (subset 복사) |
| Augmentation | SNR 5~20 dB | SNR **0~20 dB** (차이!) |
| Best (sliding 91%) | **cmd-yolov8crnn** (ours) — 91% / 35 | 미작성 |
| Best (ncnvox 자체) | crnn_best (sliding 82%) | — |

**우리 yolov8crnn 이 ncnvox 원본 보다 sliding 평가에서 우수 (91 > 82)**. backbone (yolov8 C2f) 의 capacity 효과로 추정.

---

## 11. 참고

- 원본 GitHub: (사내 — 위치 별도 확인)
- 본 stack 의 ncnvox 자체 진입점: `temp/ncnvox/README.md`
- 학습 quickstart: `temp/ncnvox/training/QUICKSTART.md`
- Claude Code 작업 가이드 (ncnvox 자체): `temp/ncnvox/CLAUDE.md` (우리 repo 의 CLAUDE.md 와 별도)
