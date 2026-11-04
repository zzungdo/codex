# Front_Truck Reference ONNX Source Checkpoint Search

작성일: 2026-08-14. 독립 신규 조사 문서 — `20260814_front_truck_reference_onnx_vs_best_pt.md`에 이어지지만
그 문서에 append하지 않고 별도로 작성한다. 그 문서의 결론(Graph SAME / Initializer·Raw output·Detection
DIFFERENT, 첫 Conv부터 다름)을 출발점(기존 확정 사실)으로만 삼고, 재검증하지 않는다.

## 목적

`260706_BSD-F606A_Front_v01.onnx`를 실제로 만든 source `.pt` checkpoint를 찾는다. 판정 기준은
파일명·폴더명·mtime·mAP 유사도가 아니라 **ONNX initializer의 parameter 값 자체**다.

## Reference ONNX

```text
/mnt/NAS_2/02_Trained_weights/05_Yolov8_nano/Novatec_NT98530/BSD_F606/Front_Truck/v01/network/260706_BSD-F606A_Front_v01.onnx
```

절대 읽기 전용으로만 취급했다 (분석용 사본은 이전 조사에서 만든
`runs/debug/front_truck_reference_compare/reference_NAS.onnx`를 재사용 — md5 원본과 동일 확인됨,
이번에도 수정하지 않음).

## 기존 확정 사실 (이전 문서에서 가져옴, 재검증하지 않음)

- 현재 repo `weights/best.pt`와 NAS `weight/weights/best.pt`는 SHA256 `f55b8b91...`로 완전히 동일한 파일.
- 이 `best.pt`를 `--skip_postprocess`로 export한 ONNX는 reference ONNX와:
  - Graph structure: SAME
  - Initializer: DIFFERENT (127개 중 126개, `model.0.conv`부터 이미 다름)
  - Raw output / Detection: DIFFERENT
- 따라서 `weight/weights/best.pt`는 reference ONNX의 직접 source가 아니다 — 이번 조사의 출발점.

## 검색 전략

```text
.pt 후보 수집 → SHA256 중복 제거 → checkpoint load/호환성 확인 → model.0 fused fingerprint
→ 대부분 즉시 탈락 → 추가 2개 layer fingerprint → 소수 shortlist → ONNX export
→ 127 initializer 비교 → raw output 비교
```

범위는 `Front_Truck/v01` → `Front_Truck` 전체로 점진적으로 넓히도록 계획했다. **조사 도중 사용자가
"Front_Truck이 아니면 관계 없을 것"이라고 확인해줘서, `Front_Truck` 밖의 다른 제품 라인(`Front`,
`DMS`, `Rear`, `Side`, `Corner_Truck`, `Rear_Truck`, `Side_Truck`)으로는 확장하지 않고 검색을
종료했다** — 아래 "탐색하지 않은 범위"에 명시.

## Reference fingerprint

`tools/debug/extract_onnx_fingerprint.py`로 추출, `reference_fingerprint.json`에 저장.

| initializer | shape | sha256(앞 12자) | min | max | mean | std |
|---|---|---|---|---|---|---|
| `model.0.conv.weight` | [16,3,3,3] | (json 참고) | — | — | — | — |
| `model.0.conv.bias` | [16] | — | — | — | — | — |
| `model.9.cv2.conv.weight` | [256,512,1,1] | `b2857d43b737` | -0.391 | 0.283 | -0.0004 | 0.0279 |
| `model.9.cv2.conv.bias` | [256] | `103e17bacffc` | -4.818 | 8.292 | -0.304 | 1.767 |
| `model.22.cv3.0.1.conv.weight` | [64,64,3,3] | `ac8459b30171` | -6.169 | 2.912 | -0.0164 | 0.202 |
| `model.22.cv3.0.1.conv.bias` | [64] | `5c48ac888141` | -12.658 | 3.544 | -0.804 | 2.843 |

(전체 값은 `reference_fingerprint.json` 참고 — `model.0` 두 개는 파일에 있고 표에는 생략된 수치만 축약.)

## Search Scope A — Front_Truck/v01

### 후보 수

`find .../Front_Truck/v01 -iname "*.pt"` — **3개** (다른 확장자·백업·old·temp·archive 디렉터리 없음,
전체 디렉터리 트리를 `find -maxdepth 5`로 확인):

```text
weight/pretrained_weight.pt
weight/weights/last.pt
weight/weights/best.pt
```

### SHA256 중복

3개 전부 서로 다른 SHA256 — 중복 없음, 3개 모두 unique candidate.

`candidate_list.csv`:

| candidate_id | path | size | mtime | sha256 |
|---|---|---|---|---|
| C001 | `weight/pretrained_weight.pt` | 12,339,826 | 2026-07-08 13:43:22 | `83e51540...` |
| C002 | `weight/weights/last.pt` | 6,233,891 | 2026-07-07 10:43:59 | `84a35ccc...` |
| C003 | `weight/weights/best.pt` | 6,233,891 | 2026-07-07 10:43:59 | `f55b8b91...`(=repo `best.pt`) |

reference onnx mtime = `2026-07-07 10:27:08` — 셋 다 이 시각보다 **늦다**(C002/C003은 16분 늦음,
C001은 다음날). mtime은 참고 정보일 뿐 판정 근거로 쓰지 않았다(아래 "파일 시간" 절 참고).

### Checkpoint 구조 확인 (`load_checkpoint_info.py`)

| candidate | load_status | model 존재 | ema 존재 | model_class | nc | train_args.model |
|---|---|---|---|---|---|---|
| C001 | OK_DICT | **False** | **True** | SymmetricDetectionModel | 6 | `/results/FrontDAIRv8/train3/weights/best.pt` |
| C002 | OK_DICT | True | False | SymmetricDetectionModel | 6 | `results/BSD-F606A/Front_v01/pretrained_weight.pt` |
| C003 | OK_DICT | True | False | SymmetricDetectionModel | 6 | `results/BSD-F606A/Front_v01/pretrained_weight.pt` |

C001은 `model`이 없고 `ema`만 있다 — 이 경우 `ema`를 그 candidate의 유일한 state로 취급했다(임의
가정 아님, 존재하는 state가 그것뿐).

### First Conv 결과 (`candidate_first_conv_fingerprint.py`, `fuse_conv_and_bn` — 이 repo가 실제
쓰는 `ultralytics.utils.torch_utils.fuse_conv_and_bn` 그대로 재사용)

`candidate_first_conv_compare.csv`:

| candidate | state | model.0 weight max_abs_diff | model.0 bias max_abs_diff | model.9.cv2 bias max_abs_diff | model.22.cv3.0.1 bias max_abs_diff | 판정 |
|---|---|---|---|---|---|---|
| C001 (pretrained_weight) | ema | 0.0514 | 0.9193 | 4.7935 | 2.8615 | **CLEAR_MISMATCH** |
| C002 (last) | model | 0.0278 | 0.8261 | 4.3090 | 2.9051 | **CLEAR_MISMATCH** |
| C003 (best, = repo `best.pt`) | model | 0.0297 | 0.9420 | 4.4922 | 3.1263 | **CLEAR_MISMATCH** |

세 candidate 모두 `array_equal=False`이고 차이 크기(bias 기준 0.8~4.8)가 부동소수점 오차 범위를
훨씬 넘는다 — **EXACT도 NEAR도 아닌 명백한 불일치.** 셋의 차이 패턴(어느 것이나 `model.9.cv2`에서
가장 크게 벌어짐)이 서로 비슷한 것도, "이 셋 중 하나가 정답에 가깝다"가 아니라 "셋 다 reference와
무관하게 비슷한 거리만큼 떨어져 있다"는 것을 시사한다.

### Shortlist

**없음.** 3개 candidate 전부 1차 필터(model.0 fingerprint)에서 CLEAR_MISMATCH로 탈락 — 추가
layer 비교(5단계)·ONNX export(6단계)로 넘어갈 후보가 없었다.

## Search Scope B — Front_Truck 전체

```text
find .../BSD_F606/Front_Truck -maxdepth 2 -type d
  -> v01 (다른 버전 디렉터리 없음)
```

`Front_Truck`에는 `v01` 하나뿐이라 **Scope B는 Scope A와 동일한 3개 candidate로 귀결**된다 —
새로 발견된 후보 없음.

## 탐색 범위를 여기서 멈춘 이유

`Front_Truck` 상위(`BSD_F606` 전체: `Front`, `DMS`, `Rear`, `Side`, `Corner_Truck`, `Rear_Truck`,
`Side_Truck` 등, 총 62개 `.pt`)까지 파일 목록과 SHA256만 확인했으나, **조사 도중 사용자가 "Front_Truck이
아니면 아마 관계 없을 것"이라고 확인해줘서 그 이상은 진행하지 않았다.** 이 판단은 사양표상으로도
근거가 있다 — `Front`/`Rear`/`Side`/`Corner_Truck`/`DMS`는 클래스 구성이 다르다(traffic_light
클래스가 있는 6클래스 모델은 Front_Truck 계열뿐, 나머지는 5클래스이거나 아예 다른 task인 DMS
pose다). 다만 이건 이번 조사에서 실측 검증한 사실이 아니라 사용자 확인 + 기존 문서의 사양표 근거를
따른 것이므로, "확인됨"이 아니라 "탐색하지 않음"으로 기록한다.

**부수적으로 발견한 사실 하나**: `Front_Truck/v01/weight/pretrained_weight.pt`(C001)의 SHA256
(`83e51540...`)가 `BSD_F606/Front/v08/weight/weights/best.pt`와 **완전히 동일(byte-identical)**
하다 — Front_Truck 학습이 `Front/v08`의 best 가중치를 초기값으로 썼다는 뜻이고, C001의
`train_args.model`(`FrontDAIRv8/train3/...`)과도 계보가 맞는다. 하지만 이 파일은 이미 C001로
first-conv fingerprint 비교를 마쳤고 결과가 CLEAR_MISMATCH였으므로, 사용자 지시에 따라 이 계보
(`Front/v01`~`v08`)를 더 따라가지는 않았다.

## Shortlist 상세 비교 / Candidate ONNX full initializer 비교 / Raw output 확인

**해당 없음.** Scope A(=B) 3개 candidate 전부 1차 필터에서 탈락해 shortlist가 비어 있어, ONNX
export(6단계)·127 initializer 전수 비교(7단계)·raw output 확인(8단계)을 수행할 대상이 없었다.

## 최종 결론

```text
Source checkpoint: NOT FOUND
```

`Front_Truck/v01`(=`Front_Truck` 전체) 안에서 접근 가능한 3개 `.pt`(`best.pt`/`last.pt`/
`pretrained_weight.pt`, `ema` state 포함 총 3개 state) 중 어느 것도 reference ONNX의 첫 Conv
레이어조차 재현하지 못한다. 억지로 가장 가까운 것을 source라고 부르지 않는다.

**Top nearest candidate (참고용, source 아님)** — model.0 weight/bias 절대 차이 기준 정렬:

| rank | candidate | model.0 weight max_abs_diff | model.0 bias max_abs_diff |
|---|---|---|---|
| 1 | C002 (`last.pt`) | 0.0278 | 0.8261 |
| 2 | C003 (`best.pt`, 현재 repo 것) | 0.0297 | 0.9420 |
| 3 | C001 (`pretrained_weight.pt`, ema) | 0.0514 | 0.9193 |

세 후보의 차이가 서로 크지 않다(0.028~0.051 / 0.83~0.94) — 이는 "그중 하나가 정답에 근접"이
아니라 "셋 다 reference와 거리가 있는, 서로 무관한 수준의 차이"로 해석하는 게 맞다.

**탐색 완료 범위**: `BSD_F606/Front_Truck`(=`v01`) 전체 — 이 안의 모든 `.pt` 파일(3개, 중복 없음)과
그 안의 `model`/`ema` state를 빠짐없이 확인했다.

**아직 탐색하지 않은 범위**: `BSD_F606` 안의 `Front_Truck` 외 다른 제품 라인(`Front` v01–v08,
`DMS` v01–v04/DMS2_v01, `Rear` v01–v04, `Side` v01–v04, `Corner_Truck` v01, `Rear_Truck` v01/260320,
`Side_Truck` v01/241106 — 총 62개 `.pt`, 파일 목록·SHA256만 `scopeC_sha256_list.txt`에 기록해뒀고
fingerprint 비교는 하지 않음), 그리고 `05_Yolov8_nano` 상위(다른 SoC/제품군)까지는 전혀 열어보지
않았다.

## 생성 파일

신규 스크립트(`tools/debug/`, 전부 읽기 전용):

- `extract_onnx_fingerprint.py` — onnx initializer fingerprint 추출
- `load_checkpoint_info.py` — candidate `.pt` 안전 로드 및 구조 기록
- `candidate_first_conv_fingerprint.py` — `fuse_conv_and_bn` 재사용한 fused layer fingerprint 비교

신규 산출물(`runs/debug/front_truck_source_checkpoint_search/`, 기존
`front_truck_reference_compare/`·`onnx_compare/`와 분리):

```text
reference_fingerprint.json
candidate_list.csv
candidate_first_conv_compare.csv
search_log.txt
scopeC_sha256_list.txt                (Scope C 파일 목록·SHA256만, fingerprint 비교 없음)
candidates/
  C001_pretrained_weight.pt / _info.json / _fingerprint.json
  C002_last.pt / _info.json / _fingerprint.json
  C003_best.pt / _info.json / _fingerprint.json
```

NAS 원본(`.pt`, `.onnx`, reference onnx 포함)은 전부 읽기만 했고 수정·삭제하지 않았다 — 위 `candidates/`는
전부 로컬 사본이며, 복사 후 SHA256가 원본과 동일함을 확인했다.

## 재현 명령

```bash
docker compose exec dair-train bash -lc '
# 0단계: reference fingerprint
uv run python3 tools/debug/extract_onnx_fingerprint.py \
  --onnx runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --names model.0.conv.weight model.0.conv.bias model.9.cv2.conv.weight model.9.cv2.conv.bias \
          model.22.cv3.0.1.conv.weight model.22.cv3.0.1.conv.bias \
  --out_json runs/debug/front_truck_source_checkpoint_search/reference_fingerprint.json

# 3단계: candidate 구조 확인 (candidates/*.pt 는 NAS 원본을 미리 복사해둔 것)
uv run python3 tools/debug/load_checkpoint_info.py \
  --pt runs/debug/front_truck_source_checkpoint_search/candidates/C003_best.pt \
  --out_json runs/debug/front_truck_source_checkpoint_search/candidates/C003_best_info.json

# 4단계: first-conv 등 fused fingerprint 비교
uv run python3 tools/debug/candidate_first_conv_fingerprint.py \
  --pt runs/debug/front_truck_source_checkpoint_search/candidates/C003_best.pt \
  --reference_onnx runs/debug/front_truck_reference_compare/reference_NAS.onnx \
  --candidate_id C003_best \
  --out_json runs/debug/front_truck_source_checkpoint_search/candidates/C003_best_fingerprint.json
'
```
