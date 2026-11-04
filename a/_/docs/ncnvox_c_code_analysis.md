# ncnvox — C 코드 상세 분석

> 위치: `/workspace/temp/ncnvox/{libncnvox, lib/nt98530, test, 3rdparty/nt98530}`
> 자매 문서: `docs/ncnvox_overview.md` (전체 overview)
> 작성: 2026-05-29 (KST)

ncnvox 의 C 코드는 크게 **4개 그룹**으로 나뉜다:
1. **`libncnvox/`** — 보드 위 runtime 라이브러리 (multi-backend 추상화)
2. **`test/`** — 통합 테스트·녹음·실시간 데모 application
3. **`lib/nt98530/`** — 미리 빌드된 NT98530 보드 binary
4. **`3rdparty/nt98530/`** — Novatek vendor SDK

본 문서는 위 4개 그룹의 구조, public API, 데이터 흐름, 의존성을 상세 분석한다.

---

## 1. `libncnvox/` — runtime 라이브러리

### 1.1 디렉토리

```
libncnvox/
├── include/ncnvox.h              public C API  (opaque handle)
├── src/
│   ├── ncnvox.c                  main impl (init/deinit/set/get/infer/stream)
│   ├── ncnvox_log.{h,c}          logging
│   ├── ncnvox_labels.{h,c}       label 파싱
│   ├── ncnvox_mel.{h,c}          mel 전처리 (PCM int16 → mel image)
│   ├── ncnvox_mel_tables.h       mel filterbank weight (precomputed)
│   ├── ncnvox_backend.h          ★ backend vtable (opaque, internal)
│   ├── ncnvox_backend_tflite.c   TFLite 백엔드
│   ├── ncnvox_backend_npu_novatek.c    Novatek NPU 백엔드
│   ├── ncnvox_backend_npu_sigmastar.c  Sigmastar NPU 백엔드
│   ├── ncnvox_backend_npu_allwinner.c  Allwinner NPU 백엔드
│   └── ncnvox_backend_none.c     스텁 (API 검증용)
├── lib/                          빌드 산출물 (.a / .so)
├── reference/98530/              참고: 칩셋별 known bug 노트
│   └── kflow_ai_input_addr_bug.md
├── docs/DESIGN.md                ★ 설계 문서 (반드시 참고)
├── Makefile                      ★ BACKEND 변수로 백엔드 선택 + CROSS_COMPILE
└── README.md
```

### 1.2 Public API (`include/ncnvox.h`)

#### 핵심 함수

```c
/* 초기화 / 해제 */
ncnvox_ctx_t    *ncnvox_init        (const char *model_path);
void             ncnvox_deinit      (ncnvox_ctx_t *ctx);

/* 설정 / 조회 (set/get 모두 키-값 형식) */
int              ncnvox_set         (ncnvox_ctx_t *ctx, ncnvox_key_t key, const void *value);
int              ncnvox_get         (ncnvox_ctx_t *ctx, ncnvox_key_t key, void *out);

/* Clip API — 2초 (32,000 sample) PCM 1회 추론 */
int              ncnvox_infer_clip  (ncnvox_ctx_t *ctx,
                                     const int16_t *pcm16, int samples,
                                     ncnvox_result_t *result);

/* Stream API — 실시간 sliding 추론 */
ncnvox_stream_t *ncnvox_stream_create     (ncnvox_ctx_t *ctx,
                                           const ncnvox_stream_cfg_t *cfg);
void             ncnvox_stream_destroy    (ncnvox_stream_t *st);
int              ncnvox_stream_push_infer (ncnvox_stream_t *st,
                                           const int16_t *pcm16, size_t n,
                                           ncnvox_result_t *results, int max, int *nresults);
void             ncnvox_stream_flush      (ncnvox_stream_t *st);
```

#### opaque handle 패턴

```c
typedef struct ncnvox_ctx    ncnvox_ctx_t;     /* incomplete type → forward decl */
typedef struct ncnvox_stream ncnvox_stream_t;
```

→ 사용자가 내부 struct 멤버에 접근 불가. ABI 안정.

#### `ncnvox_set/get` 키 (enum `ncnvox_key_t`)

| 키 | 의미 | 값 타입 |
|---|---|---|
| `NCNVOX_KEY_BACKEND_HINT` | "tflite" / "npu" / "auto" 강제 | `const char *` |
| `NCNVOX_KEY_TFLITE_THREADS` | TFLite 추론 스레드 1~8 | `const int *` |
| `NCNVOX_KEY_GATE_PEAK_LE` | PCM peak ≤ 이 값 → `_silence_` 강제 (0~32768) | `const int *` |
| `NCNVOX_KEY_GATE_RMS_LE` | PCM RMS ≤ 이 값 → `_silence_` 강제 | `const float *` |
| `NCNVOX_KEY_BACKEND` | (get) 현재 활성 백엔드 이름 | `char[32]` |
| `NCNVOX_KEY_NUM_LABELS` | (get) 클래스 수 | `int *` |
| `NCNVOX_KEY_LABELS` | (get) 레이블 문자열 배열 (ctx 수명 동일, free 불필요) | `const char *const **` |
| `NCNVOX_KEY_INFER_OK_COUNT` | (get) 누적 추론 성공 수 | `int *` |

→ **key-value 패턴**으로 ABI breakage 없이 항목 추가 가능. 일반적인 C API 확장 패턴.

#### 에러 코드

```c
NCNVOX_OK              = 0
NCNVOX_ERR_NOT_INIT
NCNVOX_ERR_INVALID_ARG
NCNVOX_ERR_BACKEND
NCNVOX_ERR_MEL
NCNVOX_ERR_LABEL_RANGE
NCNVOX_ERR_UNSUPPORTED
```

#### Model path 포맷

```
"path/to/model.bin@path/to/labels.txt"
            ↑         ↑
       모델 파일    label 파일 (한 줄당 한 클래스)
```

→ 한 문자열로 모델 + label 같이 전달. `'@'` separator.

### 1.3 내부 구조 — Backend vtable

`ncnvox_backend.h`:

```c
typedef enum {
    NCNVOX_MEL_FMT_FLOAT_HWC    = 0,   /* float [0,1] HWC (H×W×3 floats)   — TFLite */
    NCNVOX_MEL_FMT_UINT8_PLANAR = 1,   /* uint8 [0,255] CHW (3×H×W bytes)  — Novatek NPU */
} ncnvox_mel_fmt_t;

typedef struct {
    int tflite_threads;
    int num_labels;
} ncnvox_backend_cfg_t;

typedef struct {
    const char       *name;          /* "tflite" / "novatek" / ... */
    ncnvox_mel_fmt_t  mel_fmt;       /* backend 가 요구하는 mel 포맷 */
    int  (*init)    (const char *model_path, const ncnvox_backend_cfg_t *cfg, void **state);
    void (*shutdown)(void *state);
    int  (*infer)   (void *state, const void *mel_buf,
                     int *out_label, float *out_score);
} ncnvox_backend_ops_t;
```

각 backend 파일이 정적 vtable 인스턴스 export:

| 파일 | export 심볼 |
|---|---|
| `ncnvox_backend_tflite.c` | `g_ncnvox_tflite_ops` |
| `ncnvox_backend_npu_novatek.c` | `g_ncnvox_npu_novatek_ops` |
| `ncnvox_backend_npu_sigmastar.c` | `g_ncnvox_npu_sigmastar_ops` |
| `ncnvox_backend_npu_allwinner.c` | `g_ncnvox_npu_allwinner_ops` |
| `ncnvox_backend_none.c` | `g_ncnvox_none_ops` |

→ `ncnvox.c` 가 backend hint 따라 적절한 vtable 선택. 컴파일 시 일부 백엔드만 link 가능 (Makefile `BACKEND` 변수).

### 1.4 Mel 전처리 (`ncnvox_mel.{h,c}`)

```c
int ncnvox_mel_run(const int16_t *pcm16,        /* 32,000 samples (16 kHz × 2 s) */
                   float *pad_buf,              /* 호출자 제공 work buf */
                   float *logmel_buf,           /* 호출자 제공 work buf */
                   void *out,                   /* 출력 buf (fmt 따라 byte 수 결정) */
                   ncnvox_mel_fmt_t fmt);
```

**핵심 설계**: work buffer 가 **호출자 소유** → 라이브러리 내부에 global state 없음 → **재진입 안전 (re-entrant)**. 멀티스레드/멀티 ctx 안전.

mel filterbank weight 는 `ncnvox_mel_tables.h` 에 **precomputed const 배열** 로 저장 → 학습 환경 (Python `audio_config.py`) 과 SSOT 정합. Makefile 의 `make gen-speech-mel-tables` 로 생성.

### 1.5 빌드 시스템 (`Makefile`)

```
make BACKEND=novatek                          # NT98530 NPU (libvendor_ai2) [기본]
make BACKEND=sigmastar                        # Sigmastar NPU (mi_ipu)
make BACKEND=allwinner                        # Allwinner NPU (aw_nna)
make BACKEND=tflite                           # TFLite only (x86/aarch64 개발)
make BACKEND=auto                             # NPU 우선, fallback TFLite
make BACKEND=none                             # 스텁 (API 검증)

make CROSS_COMPILE=aarch64-ca53-linux-gnu-    # 보드용 cross-build
make install PREFIX=/usr/local                # 설치

# 자동 참조
NVT_SDK     = ../3rdparty/nt98530         (novatek 백엔드)
TFLITE_ROOT = ../3rdparty/tflite/install/(x86_64|aarch64)
```

→ 빌드 시 **백엔드 1개만 link** 가능. 커스텀 빌드로 사이즈 최소화.

### 1.6 known bug reference

`reference/98530/kflow_ai_input_addr_bug.md`:
NT98530 `kflow_ai` API 의 입력 주소 처리 버그 노트. 보드 펌웨어 / vendor SDK 버전별 회피 방법.

---

## 2. `test/` — 통합 테스트 + 데모 application

### 2.1 디렉토리 개요

```
test/
├── ncnvox_test/                   기본 정적 wav 추론 도구
│   ├── main.c
│   ├── wav_io.{h,c}               WAV 로더 (16-bit PCM mono)
│   └── Makefile
│
├── ncnvox_mic_test/               실시간 마이크 추론 (보드 데모)
│   ├── main.c                     state machine (WAKE → COMMAND → BLINK)
│   ├── platform/mic_platform.h    플랫폼 추상화
│   ├── ncnvox_mic_build.sh
│   └── Makefile
│
├── nt98530_audio_rt/              ★ realtime audio pipeline (가장 풍부)
│   ├── speech_config.h            audio_config.py 정합 (SSOT)
│   ├── pcm_ring.{h,c}             SPSC ring buffer (mono int16)
│   ├── speech_sliding.{h,c}       sliding window pull
│   ├── speech_mel_cpu.c           mel 전처리 (CPU 구현)
│   ├── speech_mel.h, *_tables.h   mel filterbank weight (precomputed)
│   ├── speech_infer.{h,c}         backend 추상화 (mel → label)
│   ├── speech_npu_backend.h
│   ├── speech_npu_vendor_ai.c     Novatek vendor_ai 호출 구현
│   ├── speech_npu_none.c          NPU stub
│   ├── speech_tflite_backend.h
│   ├── speech_tflite_run.c        TFLite 실행
│   ├── speech_tflite_none.c       TFLite stub
│   ├── speech_wav_eval.{h,c}      24 wav 평가 도구
│   ├── speech_wav_eval_main.c
│   ├── speech_commands_labels.h   speech_commands v2 24 클래스
│   ├── speech_mono_time.h         시간 측정 유틸
│   ├── main.c                     wav 파일 + 슬라이딩 (호스트 검증용)
│   ├── board/                     보드용 부속 코드
│   │   ├── audio_capture.c        보드 audio capture
│   │   ├── audio_output_only.c    capture 없이 출력만
│   │   └── board_wav_eval_main.c  보드용 wav_eval entry
│   ├── 2026-05-07_nvt_model_bin/  실험 산출물 (snapshot)
│   ├── wav_tests/                 평가 wav 풀
│   ├── run_wav_eval_bins.sh       호스트용 평가 실행
│   ├── run_board_wav_eval_models.sh  ★ 보드용 평가 실행 (NPU_BOARD_EXPERIMENT_LOG.md 기록 시 사용)
│   ├── deploy_crnn_board.sh       CRNN 보드 배포
│   ├── nt98530_board_build.sh     보드 빌드
│   └── Makefile
│
├── voice_reg_wav_rec/             ★ 보드 위 사용자 발화 녹음 (LVGL GUI + 터치)
│   ├── voice_reg_wav_rec.c
│   ├── novatek_build.sh
│   ├── Makefile
│   └── README.md
│
└── googledataset/                 Speech Commands v2 검증/처리
    ├── main.c
    └── Makefile
```

### 2.2 `ncnvox_test` — 정적 wav 추론

사용법:
```
ncnvox_test api                                            API/에러 검증
ncnvox_test info     <model@labels>                        label 출력
ncnvox_test wav      <model@labels> <wav>...               단순 추론
ncnvox_test tone     <model@labels>                        440 Hz tone 추론
ncnvox_test wakeword <ww@labels> [--thr N] <wav>...        KWS 전용
ncnvox_test command  <cmd@labels> <wav>...                 CMD 전용
ncnvox_test command  <cmd@labels> --wav-dir <dir>          디렉토리 재귀
ncnvox_test cascade  <ww@labels> <cmd@labels> [--thr N] <wav>...   2-stage
```

- `--thr N`: KWS 판정 임계값 (default 0.75, 0.0~1.0)
- 환경변수 `NCNVOX_BACKEND`: tflite / npu / none

cascade 동작:
1. Stage 1 (KWS): TC-ResNet8 등 가벼운 모델 항상 실행
2. Stage 2 (CMD): KWS 임계 통과 시에만 CRNN 등 정확도 높은 모델 실행

### 2.3 `ncnvox_mic_test` — 실시간 마이크 데모

```
ncnvox_mic_test wakeword <ww.bin@labels.txt>                 [--ww-thr N]
ncnvox_mic_test cascade  <ww.bin@labels.txt> <cmd.bin@labels.txt>
                                                             [--cmd-thr N]
ncnvox_mic_test compare  <model_a@labels_a> <model_b@labels_b>
```

환경변수:
```
NCNVOX_WW_THR          KWS 임계값 (default 0.65)
NCNVOX_CMD_THR         CMD 임계값 (default 0.80)
NCNVOX_CMD_TIMEOUT_MS  KWS 통과 후 CMD 대기 시간 (default 5000)
NCNVOX_COOLDOWN_MS     같은 명령 쿨다운 (default 1200)
NCNVOX_BYPASS_WAKE=1   STATE_WAKE 건너뛰고 바로 COMMAND
NCNVOX_ACAP_PULL_PATH  out0 → HD_AUDIOCAP_0_OUT_0 (NT98530)
```

**State machine** (cascade 모드):

```
STATE_WAKE      ← KWS 항상 실행
   ↓ KWS score ≥ NCNVOX_WW_THR
STATE_COMMAND   ← CMD 활성, 최대 NCNVOX_CMD_TIMEOUT_MS 동안 대기
   ↓ CMD score ≥ NCNVOX_CMD_THR  ─→ 명령 실행
   ↓ timeout
STATE_BLINK     ← 결과 표시 + cooldown
   ↓ NCNVOX_COOLDOWN_MS 경과
STATE_WAKE
```

### 2.4 `nt98530_audio_rt` — 가장 풍부한 RT 코드 묶음

#### 2.4.1 `speech_config.h` — SSOT (사양 동기화)

```c
#define SPEECH_TARGET_SAMPLE_RATE_HZ 16000
#define SPEECH_CLIP_DURATION_S       2.0f
#define SPEECH_CLIP_SAMPLES          32000   /* 16000 × 2 s */

#define SPEECH_SLIDING_HOP_MS        100
#define SPEECH_HOP_SAMPLES           1600    /* 16000 × 0.1 s */
```

주석: `training/common/audio_config.py 와 동일한 숫자 — 정합 검사: make verify-audio-board-training`

→ 학습-보드 사이 차이 나면 빌드 검증 단에서 실패. 정합 유지 강제.

#### 2.4.2 `pcm_ring` — SPSC ring buffer

```c
typedef struct {
    int16_t *buf;
    size_t cap;     /* 샘플 개수 */
    size_t w;       /* write index */
    size_t r;       /* read index */
    size_t count;
} pcm_ring_t;

int    pcm_ring_init    (pcm_ring_t *ring, int16_t *storage, size_t cap_samples);
size_t pcm_ring_write   (pcm_ring_t *ring, const int16_t *src, size_t n);
int    pcm_ring_peek    (const pcm_ring_t *ring, int16_t *dst, size_t n);
int    pcm_ring_consume (pcm_ring_t *ring, size_t n);
```

설계:
- **SPSC (Single Producer Single Consumer)** 가정 → lock-free 가능
- 호출자가 storage 제공 (re-entrant)
- `peek` (consume X) + `consume` (read pointer 만 전진) 분리 → window 처리 시 같은 데이터 재사용 가능

#### 2.4.3 `speech_sliding` — window pull

```c
typedef struct {
    unsigned window_samples;    /* = 32000 (2 s) */
    unsigned hop_samples;       /* = 1600  (0.1 s) */
} speech_sliding_cfg_t;

typedef struct {
    pcm_ring_t            ring;
    speech_sliding_cfg_t  cfg;
} speech_stream_t;

int    speech_stream_push        (speech_stream_t *st,
                                  const int16_t *pcm, size_t samples);
size_t speech_stream_count       (const speech_stream_t *st);
int    speech_stream_pull_window (speech_stream_t *st, int16_t *out_window);
```

흐름:
```
mic → pcm_ring (write)
                ↓
   speech_stream_pull_window()
                ↓  (peek 32000 sample → consume 1600 sample → 다음 window 까지 ring 에 30400 sample 남김)
   추론 (mel → backend → label/score)
                ↓
   100 ms 후 다음 window 추론
```

#### 2.4.4 Backend 추상화 (`speech_infer`, `speech_*_backend.h`)

```c
/* speech_npu_backend.h — vendor 별 분기 */
speech_npu_vendor_ai.c    Novatek vendor_ai (실제 NPU 호출)
speech_npu_none.c         스텁

/* speech_tflite_backend.h — TFLite 분기 */
speech_tflite_run.c       TFLite 실행
speech_tflite_none.c      스텁
```

`libncnvox/` 가 추후 만들어졌고, `nt98530_audio_rt` 가 **선행 prototype** 인 듯. 추후 libncnvox 로 통합 예정 또는 둘이 병존.

#### 2.4.5 `speech_wav_eval` — 24 wav 평가 도구

`run_board_wav_eval_models.sh` 가 호출하는 실 보드 평가 entry. `npu_experiments.csv` 의 `board_acc` 컬럼 채우는 데 사용된 도구.

#### 2.4.6 `board/` — 보드 전용 코드

```c
audio_capture.c           보드 마이크 capture (NT98530 HDAL)
audio_output_only.c       capture 없이 출력만 (테스트용)
board_wav_eval_main.c     보드용 wav_eval entry
```

### 2.5 `voice_reg_wav_rec` — 사용자 발화 녹음 도구

#### 2.5.1 핵심 기능

```c
#define SAMPLE_RATE_HZ        16000
#define REC_SECONDS           2
#define REC_SAMPLES           (16000 × 2)
#define EDGE_SILENCE_MS       100              /* 양끝 silence trim */
#define VOICE_DIR             "/mnt/mmc/voice" /* SD 카드 저장 위치 */

#define COUNT_MIN 1
#define COUNT_MAX 10000

#define AUDIN_BLOCKSIZE_BYTES 1024
#define AUDIN_BITS            16
```

#### 2.5.2 보드 의존

```c
#include "hdal.h"                 /* HD Abstraction Layer — NT98530 audio */
#include "nt98530-common.h"
#include "nt98530_hal.h"

#include "lvgl/lvgl.h"            /* GUI */
#include "lvgl/src/drivers/evdev/lv_evdev.h"  /* 터치 input */
```

#### 2.5.3 상태 머신

```c
typedef enum {
    VR_STATE_IDLE,
    VR_STATE_CONFIRM_OVERWRITE,
    VR_STATE_RECORDING
} vr_state_t;
```

#### 2.5.4 화면/터치 설정

- F606 보드 가정: fb 1280×480 (가로) / 패널 480×1280 (세로 회전)
- `VOICE_REG_DISPLAY_ROT` 환경변수 (기본 270)
- `VOICE_REG_TOUCH=/dev/input/eventN` (없으면 터치 비활성)
- LVGL 이 rotation 자동 변환

→ **300명 외국인 발화 수집 시 이 도구를 그대로 활용 가능**. SD 카드에 wav 저장 → 추후 학습 데이터로.

### 2.6 `googledataset` — Speech Commands 검증

```
test/googledataset/main.c
```

SC v2 dataset 검증/통계 출력. 학습 측 `data/cmd/speech_commands_v2/` 의 무결성 확인용으로 추정.

---

## 3. `lib/nt98530/` — 미리 빌드된 보드용 binary

```
lib/nt98530/
├── lib/libncnvox.so       (aarch64, BACKEND=novatek 으로 빌드된 산출물)
├── include/ncnvox.h       (= libncnvox/include/ncnvox.h 와 동일)
└── bin/
    ├── ncnvox_test        보드용 정적 추론 도구 binary
    ├── ncnvox_mic_test    보드용 마이크 데모 binary
    ├── ww_model.bin       NPU compiled — KWS (TC-ResNet8 → nvt_model.bin)
    ├── ww_label.txt       negative / hey_vueroid
    ├── cmd_model.bin      NPU compiled — CMD (CRNN → nvt_model.bin)
    └── cmd_label.txt      7 commands + background
```

이 디렉토리는 **소스 없이 binary 만 배포** 가능한 형태. 즉:
1. 보드에 펌웨어 (Novatek BSP) flash
2. `lib/libncnvox.so` 를 `/usr/lib/` 에 복사
3. `bin/*` 을 보드 작업 디렉토리에 복사
4. `./ncnvox_test cascade ww_model.bin@ww_label.txt cmd_model.bin@cmd_label.txt`

→ **production 배포 형태**.

---

## 4. `3rdparty/nt98530/` — Novatek vendor SDK

```
3rdparty/nt98530/
├── novaic/                  Novatek AI/AV 통합 SDK
│   ├── ai/                  1세대 AI (vendor_ai)
│   ├── ai2/                 ★ 2세대 AI (vendor_ai2) — 현재 사용
│   ├── dsp/                 DSP 처리
│   ├── vos/                 vendor OS abstraction
│   └── cv/                  computer vision (이미지 처리 — 음성 무관)
├── contrib/                 contributed 도구
├── samples/                 Novatek 제공 샘플 코드
├── util/                    유틸리티
├── vendor/                  vendor 전용 모듈
├── include/                 BSP 헤더
└── lib/                     prebuilt 라이브러리
```

### 4.1 핵심: `novaic/ai2/`

NT98530 의 **AI 2세대 API**. `libncnvox/src/ncnvox_backend_npu_novatek.c` 가 이 API 호출.

핵심 함수 (samples 에서 발견되는 패턴):
```c
vendor_ai_init()
vendor_ai_net_set(net_handle, CFG_MODEL, ...)
vendor_ai_net_set(net_handle, CFG_INTLBUF, ...)   // 내부 버퍼 (NT98530_NPU_SAMPLES_VS_SPEECH.md 참고)
vendor_ai_net_open(net_handle)
vendor_ai_net_set(net_handle, CFG_WORKBUF, ...)
vendor_ai_net_param_in(net_handle, layer, idx, ...)
vendor_ai_net_param_out(net_handle, layer, ...)
vendor_ai_net_run(net_handle, ...)
vendor_ai_net_close(net_handle)
```

호출 순서가 미세하게 잘못되면 **SRAM carryover 버퍼 문제** 발생 → `NPU_BOARD_EXPERIMENT_LOG.md` 의 id=37 `KERNEL_PANIC` 같은 결과.

### 4.2 vendor SDK 의 contribution

- `novaic/ai2/` 의 토큰 (`CFG_MODEL`, `CFG_INTLBUF` 등) 은 **NDA 자료**. 외부 공개 X
- 이 stack 은 `libncnvox/src/ncnvox_backend_npu_novatek.c` 에서 적절히 wrap 해서 노출 안 함

---

## 5. 빌드 / 의존성 그래프

```
                       ┌──────────────────────────────┐
                       │  test/ application 들        │
                       │  (ncnvox_test, mic_test,     │
                       │   nt98530_audio_rt, ...)     │
                       └──────────┬───────────────────┘
                                  │  link libncnvox.so / .a
                       ┌──────────▼───────────────────┐
                       │  libncnvox  (0.2.0)          │
                       │  Multi-backend C API         │
                       └──┬───────────────────────────┘
                          │ link backend (택 1)
                ┌─────────┼──────────┐
                ▼         ▼          ▼
       ┌────────────┐  ┌──────┐  ┌─────────────┐
       │ Novatek    │  │TFLite│  │None (stub)  │
       │ vendor_ai2 │  │      │  │             │
       └─────┬──────┘  └──┬───┘  └─────────────┘
             │            │
             ▼            ▼
       3rdparty/        3rdparty/
       nt98530/         tflite/
       novaic/ai2       install/
       (NDA 자료)        (open)
```

### 빌드 명령 패턴

```bash
# x86 개발 (TFLite 만)
make BACKEND=tflite

# 보드 cross-build (Novatek NPU)
make BACKEND=novatek \
     CROSS_COMPILE=aarch64-ca53-linux-gnu- \
     NVT_SDK=/path/to/nt98530_sdk

# 보드용 자동 빌드 스크립트 (각 test 디렉토리에 있음)
test/ncnvox_mic_test/ncnvox_mic_build.sh
test/voice_reg_wav_rec/novatek_build.sh
test/nt98530_audio_rt/nt98530_board_build.sh
```

---

## 6. Realtime 흐름 — 보드 위에서 무슨 일이 일어나는가

### 6.1 ncnvox_mic_test cascade 모드 흐름 (의사 코드)

```c
/* 초기화 */
ncnvox_ctx_t *ww  = ncnvox_init("ww_model.bin@ww_label.txt");
ncnvox_ctx_t *cmd = ncnvox_init("cmd_model.bin@cmd_label.txt");
ncnvox_set(ww,  NCNVOX_KEY_GATE_RMS_LE, &rms_thr);     /* 무음 게이트 */

ncnvox_stream_t *ww_st  = ncnvox_stream_create(ww,  &cfg);
ncnvox_stream_t *cmd_st = ncnvox_stream_create(cmd, &cfg);

/* mic 시작 (NT98530 HDAL — vendor_audin 모듈) */
mic_start(16000, 1, 16);

/* main loop */
int16_t pcm_block[AUDIN_BLOCKSIZE_BYTES / 2];     /* = 512 samples */
while (running) {
    int n = mic_read(pcm_block, 512);              /* ~ 32 ms 분량 */

    if (state == STATE_WAKE) {
        /* KWS 실행 — 매 100 ms 마다 sliding 1회 */
        ncnvox_result_t results[8];
        int nr = 0;
        ncnvox_stream_push_infer(ww_st, pcm_block, n, results, 8, &nr);
        for (int i = 0; i < nr; i++) {
            if (strcmp(results[i].label, "hey_vueroid") == 0
                && results[i].score >= ww_thr) {
                state = STATE_COMMAND;
                state_enter_ms = now_ms();
                break;
            }
        }
    }
    else if (state == STATE_COMMAND) {
        /* CMD 실행 */
        ncnvox_result_t results[8];
        int nr = 0;
        ncnvox_stream_push_infer(cmd_st, pcm_block, n, results, 8, &nr);
        for (int i = 0; i < nr; i++) {
            if (results[i].score >= cmd_thr) {
                execute_command(results[i].label);
                state = STATE_BLINK;
                cooldown_until_ms = now_ms() + cooldown_ms;
                break;
            }
        }
        if (now_ms() - state_enter_ms > cmd_timeout_ms) {
            state = STATE_WAKE;
        }
    }
    else if (state == STATE_BLINK) {
        if (now_ms() >= cooldown_until_ms) {
            state = STATE_WAKE;
        }
    }
}
```

### 6.2 Latency 추정

| 구간 | 시간 |
|---|---|
| Mic capture buffer (한 chunk 32ms) | 32 ms |
| PCM ring write + window pull (peek+consume) | <1 ms |
| Mel 변환 (CPU) | 5~10 ms |
| NPU inference (CRNN unroll 24-step) | 2.1 ms (D-016) |
| Label/score 분석 | <1 ms |
| **per-window 총** | **~40~50 ms** |

→ 100 ms hop 와 충분히 맞음. **CPU 가 mel 처리 도중 NPU 가 idle 일 수도** → 추가 최적화 여지.

---

## 7. 한국어 주석·문서가 가치 있는 위치

| 파일 | 가치 |
|---|---|
| `libncnvox/docs/DESIGN.md` | ★★★ 전체 설계 청사진 (한국어) |
| `libncnvox/include/ncnvox.h` | ★★ 키별 의미 한국어 주석 |
| `test/nt98530_audio_rt/speech_config.h` | ★ SSOT 정합 정책 명시 |
| `test/voice_reg_wav_rec/voice_reg_wav_rec.c` | ★ F606 디스플레이/터치 노하우 |
| `libncnvox/reference/98530/kflow_ai_input_addr_bug.md` | ★★ 보드 known bug 회피 |

---

## 8. 알아둘 만한 사항

### 8.1 backend abstraction 의 깔끔함

- `ncnvox_backend.h` 의 vtable + key-value 설정 패턴은 일반적 C API design 의 모범적 예
- 새 칩셋 (Realtek 등) 지원 시 backend 파일 1개만 추가하면 됨
- libncnvox 자체는 backend 와 완전 분리 (Makefile 로 link 선택)

### 8.2 nt98530_audio_rt 와 libncnvox 의 관계

- `nt98530_audio_rt/` 가 시간상 먼저 개발된 prototype
- 이후 `libncnvox/` 가 그 경험을 흡수해 multi-backend 추상화 라이브러리로 정리
- 현재는 둘 다 존재 — `nt98530_audio_rt/main.c` 가 호스트용 검증 도구로 남음

### 8.3 mel 처리가 CPU side 인 이유

NPU 가 input 으로 **mel image 96×96×3** 을 받는 구조 (Conv 1st op). PCM → mel 변환은 NPU 가 못 함 → CPU 가 처리. 그래서 `speech_mel_cpu.c` / `ncnvox_mel.c` 가 별도 존재.

### 8.4 두 backend 분리: TFLite vs NPU

- **TFLite**: float [0,1] HWC 입력 → CPU fallback / 개발 환경
- **NPU**: uint8 [0,255] CHW 입력 → 보드 production

mel 변환 함수가 `ncnvox_mel_fmt_t` 인자로 두 포맷 모두 지원. 호출자는 단순.

### 8.5 데이터 수집 도구 재활용 가능성

`test/voice_reg_wav_rec/` 가 NT98530 보드 + LVGL + HDAL audio capture 로 사용자 발화 녹음 도구. **외국인 300명 데이터 수집 계획 시 이 도구를 (큰 변경 없이) 활용 가능**.

---

## 9. 다음 단계 reference

- 새 backend 추가: `libncnvox/src/ncnvox_backend_<vendor>.c` 추가 + `Makefile` 케이스 추가
- 새 모델 추가: `libncnvox/src/ncnvox_labels.c` + label 파일만 변경
- 보드 평가 자동화: `test/nt98530_audio_rt/run_board_wav_eval_models.sh` 활용
- 보드 위 실시간 데모: `lib/nt98530/bin/ncnvox_mic_test` 그대로 실행
