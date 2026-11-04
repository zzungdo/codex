# ncnvox KWS 데이터 인벤토리 (Phase 2.1)

- **프로젝트**: DAIR-YOLO (ncnvox 마이그레이션)
- **작성일**: 2026-05-26 (KST)
- **목적**: 정식 KWS 학습 데이터 출처·규모·정책 확정. todo.md "4,536 + 9,072" 추정치 출처 검증.

---

## 원본 데이터 인벤토리

루트: `temp/ncnvox/training/data/`

| 카테고리 | 경로 | 개수 | sr | 길이 | 용도 |
|---|---|---:|---|---|---|
| positive | `custom_words/hey_vueroid/` | 504 | 16kHz | 2.0s 균일 | KWS target |
| negative pool (SC2) | `cmd/cmd/speech_commands_v2/{23 단어}/` | ~78,000 (단어당 ~3,500) | 16kHz | 1.0s 균일 | KWS negative |
| noise pool | `noise/noise/{musan/{speech,noise,music}, esc50/ESC-50-master}/` | 4,016 | **16k/44.1k 혼재** | **5s ~ 600s 가변** | NoisePool augmentation |
| mic silence | `_mic_silence_/` | 15 | 16kHz | 2.0s | NoisePool 보강 (실보드 무음) |
| TTS commands (Phase 3 용) | `tts/{8 카테고리}/` | 각 392 (총 3,136) | — | — | CMD 8-class, Phase 2 범위 외 |
| `custom_words/myvoice/` | — | 0 | — | — | 비어있음 (미사용) |
| `noise_car/` | — | 0 | — | — | 비어있음 |
| `openslr/` | — | 0 | — | — | 비어있음 |
| **총 WAV** | | **113,114** | | | |

## SC2 단어별 분포 (NEG_WORDS 23개)

| 단어 | 개수 | 단어 | 개수 |
|---|---:|---|---:|
| yes | 4,044 | nine | 3,934 |
| no | 3,941 | one | 3,890 |
| up | 3,723 | seven | 3,998 |
| down | 3,917 | six | 3,860 |
| left | 3,801 | three | 3,727 |
| right | 3,778 | tree | 1,759 |
| stop | 3,872 | two | 3,880 |
| go | 3,880 | zero | 4,052 |
| bed | 2,014 | eight | 3,787 |
| bird | 2,064 | five | 4,052 |
| cat | 2,031 | four | 3,728 |
| dog | 2,128 | | |
| **합계** | | | **~78,000** |

---

## ncnvox TF KWS 학습 정책 (`datasets/pipeline.py:build_wakeword_datasets`)

| 항목 | 값 |
|---|---|
| positive | `hey_vueroid` 504 전체 → label=1 |
| neg cap | `wakeword × 2 = 1,008` (`max_neg_per_word = 1008 // 23 = 43` 단어당) |
| split | `val_frac=0.2`, `seed=42` (재현 가능) |
| augmentation (train) | speed perturb + time shift + NoisePool noise mix |
| augmentation (val) | 없음 (`random_offset=False`, `.cache()`) |
| 입력 | wav → resample (16kHz) → crop_or_pad → on-the-fly mel |

실제 셈:
- positive 504 → train 403, val 101
- neg cap 1,008 (단어당 43, 23 단어): 실제 ~989 (단어당 43, `tree` 1,759 도 캡 43 적용)
  - 단어당 split: 43 → train 34, val 9
  - neg total: train ~782, val ~207
- **종합 ~1,512 클립**: train ~1,185 / val ~308 (POC 1,203 / 301 과 거의 일치 ✅)

---

## todo.md "4,536 + 9,072" 출처 결론

ncnvox 코드 어디에도 해당 수치 없음. **출처 미상 추정치**로 판단 — 폐기. TF 베이스라인 실제 규모는 1,512 클립.

---

## Phase 2 정식 학습 정책 결정 (D-011 참조)

1. **데이터셋 규모**: TF 베이스라인 그대로 — positive 504 + SC2 neg ~989 (단어당 43 균등 캡)
2. **Augmentation**: **on-the-fly** — ultralytics classify dataloader 커스터마이징, 학습 중 wav→mel + speed perturb + time shift + NoisePool noise mix
3. **NoisePool**: musan + esc50 + _mic_silence_, `max_clips_per_dir=500` (TF 기본값)
4. **val_frac=0.2 / seed=42** (TF 재현 가능 split 유지)

---

## 미사용 데이터 (의도)

- `custom_words/myvoice/`, `noise_car/`, `openslr/` — 비어있음
- `tts/{8 카테고리}/` — Phase 3 CMD 8-class 용
- SC2 단어별 ~3,500 중 ~3,457 (~99%) — neg cap 43 적용으로 미사용 (필요 시 cap 늘려 확장 가능)

---

## 후속

- **D-011** (`docs/decisions.md`) — KWS 정식 학습 정책 기록
- **todo.md Phase 2 #1 재서술** — PNG 변환 가정 폐기, on-the-fly dataloader 구현 항목 신설
- **Phase 2 후속 sub-task**: ultralytics classify dataloader 커스터마이징 (`projects/src/ncnvox/wakeword_dataset.py`) — Phase 2.1.3
