# BSD_F606 Rear_Truck `v01_symmetric` — SigmaStar(IPU) 변환 전 과정 기록

- **작성**: 2026-09-07 (KST)
- **대상 모델**: `runs/detect/BSD_F606/Rear_Truck/v01_symmetric/weights/best.onnx`
  (아키텍처 `symmetric_yolov8.yaml`, 5class: bus/car/motorcycle/person/truck, 입력
  `images` `[1,3,320,320]` BGR, mean=0/std=1 — `docs/debug/20260902_rear_truck_release_candidate_selection.md`
  §3-7에서 확정한 릴리즈 후보와 동일 가중치)
- **목적**: 이 모델을 SigmaStar IPU 디바이스 바이너리(`.img`)로 변환할 수 있는지 확인하고,
  실제로 변환·검증까지 완료한다.
- **결론(요약)**: **성공.** 최종 산출물은
  `/home/ncn/Workspace/jdh/sgs/data/output/rear_truck_v01_symmetric_xyxy/img/rear_truck_v01_symmetric_xyxy_concat_postprocess_fixed.img`.
  중간에 두 번의 실패(그래프 구조 문제, 박스 좌표 스케일 문제)가 있었고 둘 다 SDK 버전
  문제가 아니라 **우리가 만든 onnx 그래프 자체의 결함**이었다.

---

## 0. 사전 조사 — 변환 가능성 판단

### 0-1. 이미 존재하던 SigmaStar 관련 자산

| 위치 | 내용 |
|---|---|
| `/home/ncn/Workspace/jdh/sgs` | **검증된 ONNX→SigmaStar 변환 파이프라인** (README, `converter/`, `sdk/SGS_IPU_SDK_vQ_0.2.1`, `reference/`, `data/output/{CVD_xyxy,model_skip_postprocess,241031_opt_yolox_...}`). `CVD_xyxy`가 이미 `.img`까지 컴파일된 동일 계열(YOLOv8, xyxy export) 레퍼런스로 존재. |
| `ultralytics_projects/sigmastar/convert/rear_truck_v01_symmetric_320/` | **이 모델에 대한 선행 변환 시도** (`build_xyxy_onnx.py`, `add_maxscore_transpose.py`) — float/fixed sim까지는 성공했으나 최종 컴파일(`.img`) 단계에서 크래시 (`logs/onchip_native.log`, `logs/onchip_retry.log`: `sgs_concat: graph_transformations/infect_datatype.c:1444: SGS_InfectDatatypeTransform: Assertion \`0' failed. Aborted (core dumped)`) |
| `sigmastar_sdk/SGS_IPU_SDK_vQ_0.1.7`, `sigmastar_sdk_v0.2.1/SGS_IPU_SDK_vQ_0.2.1` | 이 모델(`rear_truck_v01_symmetric_320`)에 대한 **더 이전의(자동화 스크립트 없이 SDK 툴 직접 실행) 변환 흔적** (`log/output/unknown_rear_truck_v01_symmetric_320_fixed.sim_KR_20240318_..._0126.jpg.txt`) |

### 0-2. `onnx2sgsimg.py` 모델 타입 자동 감지 조건 (`sgs/converter/onnx2sgsimg.py`)

```
출력 1개 & 3D [1,N,C]  => YOLOv8 등 '디코딩 내장' 모델 (후처리는 NMS만)
그 외(보통 raw 3-head, 4D)  => YOLOX 식 (후처리에서 BoxDecoder+NMS)
```

대상 모델의 원본 onnx 출력은 `output0 [1,5,2100]`(cls-sigmoid) + `433 [1,4,2100]`(box raw
DFL distance, dist2bbox 전) — **두 형식 중 어디에도 맞지 않는다.** CVD_xyxy와 같은
"decode 내장 단일 출력 `[1,N,C]`" 형태로 그래프를 변환해야 이 파이프라인을 그대로 쓸 수
있다는 결론 → 진행 방침: `ultralytics_projects/sigmastar`의 선행 시도(그래프 구조는
얼추 맞지만 컴파일 실패)를 참고해 **`sgs` 파이프라인 위에서 처음부터 다시 진행**.

---

## 1. 1차 실패 — onnx 그래프 구조 결함 (max_score 채널 누락 + Transpose 누락)

`ultralytics_projects/sigmastar`가 만든 `weights/best.onnx`(xyxy 변환본)를 CVD_xyxy와
비교(`onnx.checker`+노드 덤프, opset 13 vs CVD_xyxy opset 12):

| | CVD_xyxy(레퍼런스, 정상 컴파일됨) | `best.onnx`(1차 변환본) |
|---|---|---|
| 출력 shape | `[1,2100,10]` | `[1,9,2100]` |
| 채널 구성 | box(4) + **max_score(1)** + cls(5) = 10 | box(4) + cls(5) = **9 (max_score 없음)** |
| 마지막 연산 | `Concat(axis=1)` → **`Transpose(perm=[0,2,1])`** | `Concat(axis=1)`에서 그대로 종료 (Transpose 없음) |
| 축 순서 | `[1,N,C]` (postprocess 모듈이 기대하는 형태) | `[1,C,N]` — **틀림** |

`postprocess_method/yolov8_xyxy_320x320.py`는 입력을 `[1,N,C]`, `C = 4(xyxy)+1(max_score)+
num_classes`로 가정하고 `num_classes = C-5`를 계산한다. `best.onnx`는 이 가정과 채널
수·축 순서가 둘 다 어긋나 있었고, `onnx2sgsimg.py`의 모델 타입 분기(`is_decoded_single`)
자체는 "출력 1개 & 3D"라 통과했지만 이후 postprocess 그래프 빌더가 잘못된 `model_cfg`
(`bbox_head_shape`, `shape` 등)로 그래프를 구성 → SDK 컴파일러(`sgs_concat`)의 데이터타입
추론(`infect_datatype.c`)이 이 기형 그래프를 처리하지 못하고 assert.

### SDK 버전(v0.1.7 vs v0.2.1) 확인 — 크래시와 무관함을 확인

크래시 당시 `ultralytics_projects/sigmastar/tool`은
`sigmastar_sdk_v0.2.1/SGS_IPU_SDK_vQ_0.2.1`을 bind-mount(`docker inspect` 확인),
`sgs/sdk/SGS_IPU_SDK_vQ_0.2.1`도 **동일 버전** — 두 SDK 사본의 `sgs_compiler` MD5
체크섬과 `ReleaseNote.txt`가 **완전히 동일**함을 직접 확인했다. `ReleaseNote.txt`의
"4.Modify infect datatype condition of elementwise op." 항목은 v0.1.7 시점에 이미
반영된 누적 변경이라 v0.2.1에도 그대로 있다 — **SDK 버전 차이가 원인이 아니었다.**
`sigmastar_sdk/SGS_IPU_SDK_vQ_0.1.7`의 흔적은 이번 세션과 무관한, 훨씬 이전의 별도
시도였다.

### 수정 (그래프 서저리, opset 13 대응)

`onnx.helper`로 아래 노드를 추가/치환 (스크립트는 세션 스크래치 파일, repo에 저장 안 함):

1. `ReduceMax(cls_sigmoid_tensor, axes=[1], keepdims=0)` → `maxscore_reduced` `[1,2100]`
2. `Unsqueeze(maxscore_reduced, axes=[1])` — **opset13은 axes가 attribute가 아니라
   두 번째 input(int64 tensor)**이어야 함(`onnx.checker`가 `input size 1 not in range
   [2,2]`로 잡아줌) → `maxscore_unsq` `[1,1,2100]`
3. 기존(틀렸던) 마지막 `Concat` 노드를 제거하고, `Concat([box(4), maxscore_unsq(1),
   cls(5)], axis=1)` → `[1,10,2100]`
4. `Transpose(perm=[0,2,1])` → 최종 `output0` `[1,2100,10]`

이 시점까지는 `onnxruntime`으로 shape만 확인(`[1,2100,10]`)하고 넘어갔는데, **박스 값
자체의 스케일은 검증하지 않아 2차 실패로 이어졌다** (§2).

---

## 2. 2차 실패 — 박스 좌표가 정규화되지 않음 (pixel-scale vs [0,1])

수정된 onnx를 `sgs` 파이프라인(`converter/sync_to_sdk.sh` → `onnx2sgsimg.py -p1
yolov8_xyxy_320x320`)에 넣고 캘리브레이션 이미지 10장(처음엔 도메인이 안 맞는 fisheye
이미지, 이후 `1T_REAR/pure/val/images`의 실제 Rear_Truck 이미지로 교체)으로 Float→
Fixed→Offline까지 **크래시 없이** 완주했다. 그런데 결과 이미지에 박스가 하나도 안 그려짐.

원인: `onnx2sgsimg.py`가 만든 `detection_..._img_*.txt` 로그에서 박스 좌표가
`[75158.125, 5239.53, 21115.49, 15509.94]`처럼 **수만 단위**로 찍힘. `postprocess_method/
yolov8_xyxy_320x320.py`의 주석("박스(xyxy)는 그래프 안(DFL)에서 이미 디코딩되어
**정규화[0,1] 상태**")과 달리, 우리 그래프의 box decode(`dist2bbox`)는 앵커 위치 ±
distance에 **stride(8/16/32) 상수를 곱해 320×320 입력 기준 절대 픽셀 좌표**를 만들고
있었다(`Constant_14` 값 확인: `[8,8,8,...,16,...,32,...]`, 표준 ultralytics
`decode_bboxes(...)*strides` 패턴). postprocess 모듈이 "이미 [0,1]"이라고 가정하고
원본 이미지 크기를 다시 곱하니 좌표가 ~320배 부풀려져 화면 밖으로 나가버린 것.

### 수정

그래프 서저리에 정규화 스텝 추가:

```
box_normalized = Div(box_pixel_scale, Constant(320.0))
```

이후 `Concat(box_normalized, maxscore, cls)` → `Transpose` 순서는 동일. 수정 후
`onnxruntime` 샘플 출력에서 box 값이 `[-0.035, 1.118]` 범위(정상적인 [0,1] 근방)로
나오는 것을 확인하고 재변환 → **Rear_Truck 실제 이미지에서 트럭/버스/사람 박스가 정확한
위치에 정상적으로 그려짐**을 확인.

---

## 3. 환경 — SigmaStar 변환 실행 방법

### 3-1. `sgs` 저장소 구조 (`/home/ncn/Workspace/jdh/sgs`, 상세는 `README.md`)

```
sdk/SGS_IPU_SDK_vQ_0.2.1/     # 압축 해제된 SDK (converter/에서 sync됨)
converter/                     # 커스텀 소스 (onnx2sgsimg.py, 전/후처리 모듈, sync_to_sdk.sh)
data/input/models/             # 입력 onnx
data/input/images/             # 캘리브레이션+테스트 이미지 (-p/-d 공용)
data/output/<model_name>/      # {ini,onnx,sim,img,result}
```

### 3-2. Docker 실행 — `sgs/docker/`(micromamba) 대신 기존 이미지 재사용

`sgs/docker/Dockerfile`은 처음부터 이미지를 빌드해야 해서(tensorflow/torch 등 다운로드,
수 분) 시간이 걸린다. 이미 존재하는 **`dair-yolo/sigmastar:latest`**(Ubuntu18.04 +
i386 멀티아치 + python3.7 + `onnx==1.13.0`/`onnxruntime==1.7.0`/`tensorflow==1.14.0`,
`ultralytics_projects/docker/Dockerfile.sigmastar`로 빌드된 이미지)를 `sgs/`에
바인드마운트해서 재사용했다 — **SDK가 v0.2.1로 동일**하고 이미 검증된 환경이라 그대로
호환된다.

```bash
docker run -d --name sgs-convert-run \
  -v /home/ncn/Workspace/jdh/sgs:/workspace -w /workspace \
  dair-yolo/sigmastar:latest sleep infinity
```

**주의 — `python3` 심볼릭 링크 함정**: 이 이미지의 `/usr/bin/python3`는 시스템 기본
**3.6**(패키지 없음)을 가리키고, 우리가 쓰는 패키지들은 **3.7**에 설치돼 있다.
`onnx2sgsimg.py`가 내부적으로 다른 스크립트(`ConvertTool.py`, `calibrator.py`,
`compiler.py`, `simulator.py`, `bin/concat_net`)를 서브프로세스로 띄울 때 `python3`를
그대로 호출해서, 그냥 실행하면 `ImportError: No module named tensorflow/cv2`가 남 (SDK
문서의 micromamba 방식은 이 문제를 env activate로 우회함). 해결: 심볼릭 링크로 PATH
앞단에서 `python3`/`python`이 3.7을 가리키게 shim:

```bash
docker exec sgs-convert-run bash -lc \
  "mkdir -p /tmp/pybin && ln -sf /usr/bin/python3.7 /tmp/pybin/python3 && ln -sf /usr/bin/python3.7 /tmp/pybin/python"
# 이후 모든 실행에서 PATH=/tmp/pybin:$PATH 를 앞에 둔다
```

### 3-3. 변환 실행

```bash
docker exec sgs-convert-run bash -lc "bash converter/sync_to_sdk.sh"   # 커스텀 소스 → SDK 반영

docker exec sgs-convert-run bash -lc '
export PATH=/tmp/pybin:$PATH
export SGS_IPU_DIR=/workspace/sdk/SGS_IPU_SDK_vQ_0.2.1
export LD_LIBRARY_PATH=${SGS_IPU_DIR}/libs/x86_64:${SGS_IPU_DIR}/libs/x86_32:${LD_LIBRARY_PATH}
export PYTHONPATH=${SGS_IPU_DIR}/Scripts:${SGS_IPU_DIR}/Scripts/postprocess:${PYTHONPATH}
cd $SGS_IPU_DIR
python3 onnx2sgsimg.py \
  -m /workspace/data/input/models/rear_truck_v01_symmetric_xyxy.onnx \
  -p /workspace/data/input/calib_940 \
  -d /workspace/data/input/images \
  -b /workspace/data/output \
  -p1 yolov8_xyxy_320x320 -p2 yolox_nano_320x320 -c 0.45
'
```

파이프라인 단계: `Convert(ConvertTool) → postprocess 빌드 → concat_net → calibrator
(양자화) → compiler(.img) → simulator(Float/Fixed/Offline)`. 각 단계 로그는
`Mod : {Float,Fixed,Offline} is {Starting,Done}!`으로 진행 확인 가능.

---

## 4. 캘리브레이션/테스트 이미지 — 세 번의 교체

| 시도 | 캘리브레이션(-p) | 테스트(-d) | 결과 |
|---|---|---|---|
| 1차 | `ultralytics_projects/sigmastar`에 있던 임의 fisheye 10장(도메인 불일치, Front류) | 동일 | 박스는 그려지나 좌표 오류(§2)로 화면 밖 |
| 2차 | `/home/ncn/Workspace/data/1T_REAR/pure/val/images`(실제 학습 val셋)에서 라벨 있는 10장 | 동일 | §2 수정 후 정상 검출 확인 (트럭4+사람3, conf 0.78~0.89) |
| **3차(최종)** | **Novatek 변환 시 쓴 캘리브레이션셋 전부(940장)**, `ultralytics_projects/novatek/convert/rear_truck_v01_symmetric_320/nvtai_tool/input/data/REAR_TRUCK_V01_320/video_calib_jpg/` (Novatek `gen_config.txt`의 `path/ref_img_dir`) | Novatek 테스트셋 500장(`.../REAR_TRUCK_V01_SYMMETRIC_320/jpg/`, `sim_config.txt`의 `path/test_img_dir`) 중 **15장 샘플**(`awk 'NR%33==1'`로 균등 추출) | 낮/밤 모두 정상 검출 (버스 다수, 세단, 오토바이+라이더 등) |

**3차를 택한 이유(사용자 지시)**: Novatek 변환 때 실제로 쓰인 정본 이미지셋을 그대로
재사용하면 SigmaStar 쪽 양자화 캘리브레이션 품질을 Novatek과 동일 조건으로 비교/신뢰할
수 있다. 다만 **테스트(-d) 500장 전체는 안 씀** — `onnx2sgsimg.py`의 Float/Fixed/Offline
3단계가 각각 -d 이미지 전부를 실제 시뮬레이터로 순회하기 때문에(캘리브레이션은 1회성
비용, 시뮬레이션은 3배) 500장 전체를 돌리면 수 시간이 걸릴 것으로 추정돼 15장 샘플로
축소했다(사용자 확인 후 결정). **940장 캘리브레이션 자체는 내부적으로 멀티프로세스(10
worker) 병렬 처리라 실측 약 10분**으로 예상보다 훨씬 빨랐다(전체 파이프라인 약 12분).

**남은 일**: 500장 테스트셋 전체로 더 넓은 정확도(예: mAP) 검증은 아직 안 함(시간
문제로 보류, 필요시 백그라운드로 재실행 가능).

---

## 5. 레이어별 양자화 정확도 — `DumpDebug` (Novatek `layer_diff.txt`의 SigmaStar 대응물)

Novatek 변환 산출물에 `simtool/layer_diff.txt`, `debug/golden_tensor/*.bin` 같은
레이어별 float-vs-quantized 비교 리포트가 있는 것을 참고해, SigmaStar SDK에도 동일
기능이 있는지 확인 → **있음**: `sgs/sdk/SGS_IPU_SDK_vQ_0.2.1/DumpDebug/`
(`auto_dump_debug.sh` + 사전 컴파일된 `compareAccuracy` 바이너리).

### 실행

```bash
# label 인자는 필수지만 Detection accuracy 지표 없이 그냥 레이어 덤프만 필요하므로
# SDK의 Fake_Label 포맷(공백 6줄)과 동일한 더미 파일을 직접 만들어 사용
printf ' \n \n \n \n \n \n' > /workspace/DumpDebug_fake_label.txt

cd /workspace/sdk/SGS_IPU_SDK_vQ_0.2.1/DumpDebug
bash auto_dump_debug.sh \
  /workspace/sdk/SGS_IPU_SDK_vQ_0.2.1 \
  <..._concat_postprocess.sim>        `# float` \
  <..._concat_postprocess_fixed.sim>  `# fixed(INT8)` \
  <이미지 1장.jpg> \
  /workspace/DumpDebug_fake_label.txt \
  Detection \
  yolox_nano_320x320
```

내부적으로 `simulator.py --dump_rawdata`로 Float/Fixed 각각의 레이어별 raw tensor를
덤프하고, 이름으로 레이어를 매칭한 뒤 각 쌍에 `compareAccuracy`(MSE/COS/RMSE, 양자화
dtype도 같이 표시)를 실행한다. 이미지 1장 기준 수 초 만에 완료, 결과는
`DumpDebug/Detection_<fixed_sim>_DumpDebug_out/doAnalyzeRmse/doAnalyzeRmse_run.log`.

### 결과 (이미지 1장, 192개 레이어)

- 백본→넥→헤드 거의 전 레이어에서 **cosine similarity 0.996~0.9999** — INT8 양자화
  손실이 매우 작다(940장 캘리브레이션이 잘 작동했다는 방증).
- 디코딩된 box+score가 최종 형태로 합쳐지는 지점(`concat10_pre_transpose`, `output0`
  Transpose)까지도 cosine 0.999대 유지.
- **마지막 2개 레이어만 급격히 낮음** — `SGS_unpack6`(class index, argmax 결과):
  `COS 0.335`, `MSE 4.09`; `detectionBoxes`(NMS 이후 최종 출력): `COS 0.395`. **이건
  실제 품질 저하가 아니라 비교 방법의 한계**로 판단: class index는 정수 argmax라 스코어가
  근소하게만 달라져도 인덱스가 통째로 바뀌어 연속값 지표(cosine/MSE)가 왜곡되고,
  `detectionBoxes`는 NMS 이후 값이라 float/fixed에서 살아남는 박스의 개수·순서가 다르면
  "슬롯 i" 단위 비교 자체가 어긋난다. 그 직전 단계(디코딩된 box+score, NMS 진입 전)까지는
  cosine이 정상이므로 네트워크 자체의 양자화 정밀도는 양호하다고 판단.

Novatek `layer_diff.txt`(cosine + avg_diff, top-5 diff) 대비 SigmaStar `compareAccuracy`
쪽은 **레이어별 양자화 dtype(SGS_U8/SGS_S16)까지 같이 보여줘서 정보량이 더 많다.**

---

## 6. 최종 산출물

```
/home/ncn/Workspace/jdh/sgs/
├── data/input/
│   ├── models/rear_truck_v01_symmetric_xyxy.onnx        # 그래프 서저리 최종본(§1+§2 수정 반영)
│   ├── calib_940/                                        # Novatek 캘리브레이션 940장
│   ├── images/                                           # Novatek 테스트 15장 샘플
│   └── images_prev10_backup/, images_frontref_backup/    # 이전 시도 이미지 백업
└── data/output/rear_truck_v01_symmetric_xyxy/
    ├── onnx/rear_truck_v01_symmetric_xyxy_refine.onnx
    ├── ini/*_cfg.ini, *_concat_cfg.ini                   # BGR, mean=0, std=1 확인됨
    ├── sim/*_concat_postprocess.sim                      # float
    ├── sim/*_concat_postprocess_fixed.sim                # INT8 (940장 캘리브레이션 반영)
    ├── img/rear_truck_v01_symmetric_xyxy_concat_postprocess_fixed.img   # ★ 디바이스 바이너리
    └── result/{Float,Fixed,Offline}/                     # 검출 박스 그려진 결과 이미지
```

`sgs/sdk/SGS_IPU_SDK_vQ_0.2.1/DumpDebug/Detection_..._fixed.sim_DumpDebug_out/` —
레이어별 정확도 비교 리포트(§5).

---

## 7. 다음 단계 (미완료)

1. **Novatek 테스트셋 500장 전체**로 Float/Fixed/Offline 재실행 — 15장 샘플이 아닌 전체
   기준 정확도(가능하면 mAP) 확인. 3단계 시뮬레이션 특성상 수 시간 소요 예상, 백그라운드
   실행 필요.
2. 그래프 서저리 스크립트(§1/§2)는 현재 세션 스크래치 디렉터리에만 있고 **repo에
   커밋되지 않았다** — 재현 가능하도록 `sgs/converter/` 하위에 정식으로 옮겨 저장할지
   결정 필요(현재는 결과물인 `data/input/models/rear_truck_v01_symmetric_xyxy.onnx`만
   남아있음).
3. `DumpDebug` 레이어별 비교를 15장(혹은 500장) 전체로 확장해 이미지별 편차가 있는지
   확인(현재는 1장만 확인).
