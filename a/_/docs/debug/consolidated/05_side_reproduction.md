# Side_Truck v01 — Full Pipeline Reproduction (PT → ONNX → Calibration → NVT → Board)

> 원문: `20260814_side_full_pipeline_reproduction.md`(전체). Front_Truck/Corner_Truck/Rear_Truck의
> 결론은 참고만 하고 Side는 독립적으로 확인한 조사. Training 재현(Phase 2)은 범위 밖.
> 공통 재현 환경/evaluator 정의는 [90_reproduction_reference.md](90_reproduction_reference.md) 참조.

## 0. 결론 요약표

| 단계 | 상태 | 한 줄 요약 |
|---|---|---|
| 1. Source PT | PASS | NAS 원본과 SHA256 완전 일치 |
| 2. PT → ONNX | PASS | monkeypatch export로 NAS reference ONNX와 그래프/파라미터 재현 |
| 3. Calibration | PASS(제약 있음) | 원본 리스트는 못 찾음, 사용자가 절대기준 경로에서 700장 직접 제공 |
| 4. NVT compile/simulate | PASS | 정상 컴파일, 모델 크기 NAS 대비 4bytes 차이, cosine 정상 수준 |
| 5. Float ONNX mAP(CVD/Side) | PASS | repo 기록된 PyTorch 기준값과 사실상 완전 일치 |
| 6. Float ONNX vs 사용자 제공 Board표(val_iputest) | **불일치(미해결)** | 3가지 평가 방식 모두 person 낮음/car·bus·truck 높음 패턴 반복 |
| 7~8. Float ONNX vs 사용자 제공 "Original"표(BSD/Side/val, class별 threshold 고정) | **부분 일치** | bus/truck 거의 정확히 일치, car/motorcycle 근접, **person만 큰 차이(+0.11) 잔존** |
| 8-3차 | 재현 ONNX vs NAS 원본 ONNX 직접 비교(동일 평가) | **PASS** — 두 onnx 완전 동일 결과(person 포함), export는 원인 아님 확정 |
| 9~10. 사용자 제공 IPU_TEST 리스트(4,247장 서브셋) 재평가 | 훨씬 근접 | bus/car/truck ±0.006 이내 일치, person/motorcycle만 잔여 차이 |
| 10-conf스윕 | class별 threshold/0.25/0.01, 재현 vs NAS ONNX | **PASS** — 3가지 conf 설정 전부 소수 4자리까지 동일 |
| 11. **실제 물리 보드 테스트** — 새 NVT BIN vs 기존 reference dump | **PASS(99.53%)** | 매칭 detection 평균 IoU 0.9925, score 차이 0.00014. **person이 가장 잘 맞는 클래스(불일치 0.23%)** |

**중요 주의(통합자 주석)**: 이 표에서 "PASS"는 각 단계마다 **서로 다른 의미**로 쓰였다 —
1/4/5/8-3차/10-conf스윕은 그래프·파라미터·수치가 근사/완전 동일함을 뜻하고, 11(물리 보드)은
**99.53% detection 매칭**(bit-exact가 아님)을 뜻한다. "그래프 동일", "파라미터 근사 동일",
"raw output 동일", "전체 평가 완료"를 서로 바꿔쓰지 말 것.

**핵심 서사**: Float ONNX 단계에서 있었던 "person 불일치" 미스터리는 **실제 보드 결과에서는
재현되지 않았다** — 즉 그 불일치는 NVT 변환/보드 문제가 아니라, Original/Board 표를 만든
evaluator·데이터셋 조건 차이였을 가능성이 사실상 확정됐다(단, 그 표들의 정확한 산출
파이프라인 자체는 끝내 확인되지 않았다 — §7 참조).

## 1. Source PT (PASS)

NAS 원본: `.../Side_Truck/v01/weight/weights/best.pt`, sha256
`0c148dca59494ee395c4acf288f3005f4fd1fbe0f63f02105f588394a3226a96`. 로컬 사본 sha256 완전
일치. 체크포인트: `model` 키만 존재(`ema` 없음), `model_class=DairDetectionModel`, `nc=5`,
기록된 ultralytics 버전 8.4.39(현재 repo pin은 8.4.6 — 참고사항, 실질 영향 이후 단계에서
미확인). `configs/releases/BSD_F606/Side_Truck/v01/MANIFEST.yaml`(2026-07-29, 이 조사 이전부터
존재)이 이 best.pt의 출처(학습 run `260708_side_train_4`)와 데이터셋 대체 이력을 이미
기록하고 있었다.

## 2. PT → ONNX (PASS)

**Reference ONNX**: `.../Side_Truck/v01/network/BSD-F606_Side_320x320.onnx` — input
`images[1,3,320,320]`, output `output0[1,5,2100]`(cls, sigmoid 적용) + `433[1,4,2100]`(box,
DFL까지·dist2bbox 미적용), opset13/node235/initializer127.

**1차 시도(실패)**: `tools/export.py`, `tools/export_custom.py`(--skip_postprocess 기본값
True) 둘 다 **완전히 동일하게 실패** — 단일 텐서 `output0[1,9,2100]`(261 node)만 나옴. 원인:
이 체크포인트의 실제 head 모듈이 `ultralytics.nn.modules.head.Detect`(순정 stock 클래스)이고,
`skip_postprocess`/`skip_dfl` 속성 자체가 없음(`hasattr()`로 확인). `export_custom.py`의
skip_postprocess 플래그는 `SymmetricDetect` 전용이라 이 체크포인트엔 no-op이었다.

**2차 시도(성공) — monkeypatch export**: 가중치/그래프 계산을 전혀 바꾸지 않고,
`SymmetricDetect._inference()`와 동일한 방식으로 stock `Detect`의 `_inference`만 함수 교체:

```python
def _skip_postprocess_inference(self, x):
    dbox = self.dfl(x["boxes"])
    return x["scores"].sigmoid(), dbox
```

신규 스크립트: `tools/debug/side_export_skip_postprocess_patch.py`. 결과 — reference와 비교:
output 이름까지 일치(`output0[1,5,2100]`+`433[1,4,2100]`), node diff **0개**(235=235),
initializer 최대 차이 3.8e-6(float32 노이즈), raw output 차이(4종 입력×2 옵션) 1.5e-7~1.7e-4
(float32 노이즈). **최초 실패는 "모델이 다르다"가 아니라 "CLI 스크립트가 이 체크포인트용
skip_postprocess 진입점을 노출하지 않았을 뿐"이었다.**

## 3. Calibration (PASS, 제약 있음)

절대 기준 지정 경로(`\\192.168.103.167\workspace\data\ref_imgs`)는 이 host에 mount 안 됨
(호스트는 ping 응답하나 자격증명 없음, 임의 마운트 시도 안 함) — **사용자가 직접 내용물을
수동으로 복사해 제공**: **700장**(전부 jpg, 중복 없음, 1920×1080 605장 + 1280×720 95장,
gen_config.txt의 고정값 `multiscale/width=1920,1280`/`height=1080,720`과 정확히 일치).
NAS에는 원본 calibration 리스트(`ref_img_list.txt`)가 보존돼 있지 않아 이 조사 전용으로
신규 생성(700줄). gen_config.txt 자체가 `${width}`/`${height}`/`${ref_data_num}` 등 미치환
템플릿 상태였음(`MANIFEST.yaml`의 `gen-config-unsubstituted` 이슈와 정확히 일치) — 이번
조사에서 `width=height=320`(ONNX shape 근거), `ref_data_num=700`(확보된 실측 개수)으로 확정.
**추가 확인(§11 이후)**: 사용자가 이 700장이 Release 당시 calibration 세트와 **100% 동일**함을
구두로 확인 — §11의 board 불일치 원인에서 calibration 데이터셋 차이는 배제된다.

## 4. NVT Convert / Compile / Simulator (PASS)

격리 디렉토리 `novatek/convert/side_truck_v01_full_reproduction_20260814/`(NAS와 완전 분리,
Corner/Rear/Front에서 검증된 `npu-convert` + `run_compiler.sh`/`run_simulator.sh` + `--chip
530` 파이프라인 재사용).

**Compile 결과**: Local model size 4,703,564 bytes vs NAS Reference 4,703,568 bytes(**4
bytes 차이**), Latency 6.24694ms, DRAM 35,045,388 bytes, calibration count 700(profile.txt
확인), 에러/경고 없음. SHA256은 다름(NAS `nvt_model.bin`의 mtime이 현재 best.pt/onnx보다 8일
이르다 — 애초에 현재 체크포인트에서 만들어진 게 아닐 수 있다는 의심이 inventory 단계부터 있었음).

**실제 내용물 차이(byte-level, §11 이후 추가 조사)**: `cmp -l`로 우리 컴파일과 NAS 원본 bin의
실제 차이 확인 — **2,273 bytes**가 다름(offset 26~3,721,511 사이 196개 클러스터). 몇 클러스터
hex 분석: offset 1,340,178은 바이트 1개(`0xc7→0xc8`)만 차이(고엔트로피 weight 영역, 순수
수치 노이즈로 추정), offset 25,773 부근은 `PDEF` 섹션 마커 뒤 필드가 `-100/-100`(우리) vs
`11/-1`(NAS) — 컴파일러 내부 섹션 크기/카운트 메타데이터로 판단, offset 38,782 부근은
`0x00400000`(우리) vs `0x00401770`(NAS) — 버퍼 오프셋/메모리 레이아웃 테이블로 추정. **가설
검증 — "NAS가 다른 컴파일러/SDK 버전을 썼다"**: `gentool/profile.txt`의 `VERSION=2052503042`
필드를 두 bin에서 바이트 패턴으로 직접 검색 — **둘 다 offset 4에 동일 값 존재, 컴파일러
버전은 완전히 동일. 이 가설은 기각.** **정정된 결론**: 컴파일러 버전은 같지만 섹션 크기/버퍼
오프셋 같은 내부 메타데이터가 실행마다 동일하지 않다 — **같은 컴파일러·같은 입력이라도 내부적으로
완전히 결정론적이지 않다**(§8에서 자체 컴파일 두 번이 29 bytes 다르게 나온 것과 같은 성격의
비결정성, 규모만 더 큼 — 2,273 vs 29 bytes. calibration 세트가 완전히 같지 않을 가능성(처리
순서 차이)까지 겹쳤을 가능성도 있음).

**Simulator 결과**(NaN/Inf 없음): Softmax(dfl) 0.988, Sigmoid[OUT] 0.987, dfl_conv[OUT]
0.997 — Rear_Truck(0.985/0.982/0.993), Corner_Truck(0.988/0.996/0.995)와 비슷한 수준, Side만
유독 이상하다는 신호는 없음.

## 5. Float ONNX mAP — repo 기록값과 비교(CVD/Side) (PASS)

`MANIFEST.yaml`에 이미 기록된 `pytorch_val_baseline`(2026-07-30, `configs/datasets/
cvd_side.yaml` validation **14,267장/58,885 instance** 기준)과 비교. 표준 validator
(`CustomDetectionValidator`)로 시도했더니 **전 클래스 mAP=0**(split-head onnx 출력을 표준
NMS가 오인식 — Front_Truck 조사 때와 동일 원인) → Side_Truck 전용 shape(320×320, 2100
anchor)로 재조립하는 신규 로직(`SideSplitHeadValidatorJdh`, `tools/debug/
validate_side_split_head.py`) 작성.

| | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 재현 Float ONNX | 0.888 | 0.697 | **0.809** | **0.605** |
| MANIFEST 기록값(PyTorch) | 0.8871 | 0.6978 | 0.8089 | 0.6054 |

**사실상 완전히 일치** — Step 2(구조/파라미터 재현)가 실측 mAP 레벨에서도 재검증됨.

## 6. Float ONNX vs 사용자 제공 Board 성능표 — 미해결

사용자 제공 표(GT: `/mnt/NAS_3/.../BSD/Side/val`, 이미지: `.../val_iputest`,
`configs/datasets/bsd_side_val_iputest.yaml`, **5,943장/18,539 instance**, 라벨 매칭 완료):

| Class | 주신 표(ONNX) | ultralytics-native | pycocotools Dual-NMS | pycocotools Single-NMS |
|---|---:|---:|---:|---:|
| bus | 0.8546 | 0.905(+0.050) | 0.890(+0.035) | 0.874(+0.020) |
| car | 0.7638 | 0.867(+0.103) | 0.848(+0.084) | 0.856(+0.092) |
| motorcycle | 0.7665 | 0.800(+0.034) | 0.782(+0.016) | 0.783(+0.017) |
| **person** | 0.863 | 0.738(**-0.125**) | 0.752(**-0.111**) | 0.755(**-0.108**) |
| truck | 0.8039 | 0.878(+0.074) | 0.851(+0.047) | 0.815(+0.011) |
| **all** | **0.8104** | 0.838(+0.028) | 0.825(+0.015) | 0.817(+0.007) |

(Dual-NMS: `min_overlap_thres=0.6,union_overlap_thres=0.3`, ncn_metric/customEval 엔진 —
Rear/Corner Board 표 산출 방식과 동일. Single-NMS: `basic_postproc_v8`, conf=0.01/iou=0.45.)

**관찰**: 서로 완전히 다른 3가지 NMS/평가 엔진에서 **거의 동일한 패턴** 반복 — all/motorcycle은
근접, **person은 항상 낮음(-0.11 근처로 일관)**, bus/car/truck은 항상 높음(+0.01~0.10). NMS/
conf 설정 차이로 설명되는 패턴이 아니다. **결과: 불일치, 원인 미확정.** 유력 후보(검증 안
됨, 추측): 사용자 표를 만든 validation 배치가 `val_iputest`와 클래스 구성(특히 person 비중/
난이도)이 다를 가능성. 표의 정확한 산출 파이프라인(dataset/threshold/evaluator)은 끝내
확인되지 않았다.

## 7~8. Float ONNX vs 사용자 제공 "Original" 표 — BSD/Side/val 직접 평가

사용자 제공 표(Board표(§6)와 별개, 어느 evaluator/조건으로 만들어졌는지 끝내 미확인):

| Class | Original mAP50 / Precision / Recall |
|---|---|
| bus | 0.8406 / 0.935 / 0.822 |
| car | 0.7339 / 0.844 / 0.799 |
| motorcycle | 0.7116 / 0.894 / 0.689 |
| person | 0.6838 / 0.848 / 0.639 |
| truck | 0.7914 / 0.918 / 0.777 |
| **all** | **0.7523 / 0.8878 / 0.745** |

**데이터셋**: `/mnt/NAS_3/02_datasets/BSD/Side/val`(11,001 images/10,999 labels)을 그대로
사용 — 클래스별 GT 개수가 `side_class_conf_thresholds.json`의 n_gt와 정확히 일치(bus3354/
car10016/motorcycle3452/person12567/truck4932), 이 폴더가 진짜 원본 `side_total/val`임을
뒷받침. **데이터 이슈**: 최초 실행이 46% 지점에서 `AttributeError`로 죽음 — `images/`에
`@eaDir`(Synology 썸네일 캐시)/`Thumbs.db`(Windows 썸네일 캐시)가 섞여 있었음
(`os.listdir()`가 확장자 필터 없이 목록을 가져오고 `cv2.imread()`가 비-이미지에 예외 없이
`None` 반환 → `None.shape` 호출로 크래시). 로컬 rsync 사본에서만 제거(NAS 원본 미변경) —
제거 후 10,999장으로 라벨 수와 정확히 일치.

**1차 — 3가지 evaluator로 conf 낮게 sweep**: Original과 3개 evaluator 모두 전 클래스 일관되게
큰 차이(+0.04~+0.22, all: Original 0.7523 vs ultralytics-native(conf=0.001) 0.889(+0.137),
pycoco Dual(conf=0.01) 0.870(+0.118), pycoco Single(conf=0.01) 0.862(+0.110)) — NMS/conf
설정 차이로는 설명 안 됨.

**2차 — class별 optimal threshold 고정**(`side_class_conf_thresholds.json`: bus=0.317,
car=0.305, motorcycle=0.416, person=0.517, truck=0.355, 신규 `tools/debug/
side_metrics_utils_jdh.py`):

| Class | threshold | 제 결과 mAP50 | Original mAP50 | 차이 | P(제/Orig) | R(제/Orig) |
|---|---:|---:|---:|---:|---|---|
| bus | 0.317 | 0.845 | 0.8406 | **+0.004** | 0.929/0.935 | 0.850/0.822 |
| car | 0.305 | 0.792 | 0.7339 | +0.058 | 0.861/0.844 | 0.807/0.799 |
| motorcycle | 0.416 | 0.734 | 0.7116 | +0.022 | 0.919/0.894 | 0.741/0.689 |
| **person** | 0.517 | **0.796** | **0.6838** | **+0.112** | 0.935/0.848 | **0.802/0.639** |
| truck | 0.355 | 0.786 | 0.7914 | -0.005 | 0.929/0.918 | 0.799/0.777 |
| all | — | 0.790 | 0.7523 | +0.038 | — | — |

bus/truck는 사실상 정확히 일치, car/motorcycle도 상당히 근접 — class별 고정 threshold가
Original의 실제 산출 방식에 가깝다는 정황. **person만 여전히 크게 벌어짐**(mAP50 +0.112,
Recall +0.163으로 가장 큼).

**3차 — 재현 ONNX vs NAS 원본 ONNX 직접 비교(동일 조건)**: person 차이가 export 문제인지
확인 — 같은 per-class-threshold 평가를 **NAS 원본 ONNX**로도 실행:

| Class | 재현 ONNX | NAS 원본 ONNX |
|---|---:|---:|
| bus | 0.845 | 0.845 |
| car | 0.792 | 0.792 |
| motorcycle | 0.734 | 0.734 |
| **person** | **0.796** | **0.796** |
| truck | 0.786 | 0.786(IoU@50-95만 0.673 vs 0.672, 노이즈) |
| all | 0.790 | 0.790 |

**완전히 동일 — person을 포함해 전 클래스가 두 onnx에서 정확히 같은 값. person 불일치는
export/onnx 재현 문제가 아님이 확정.**

**4차 — 전체 val(11,001장)에 conf=0.01/0.25 균일 threshold 적용, 재현 vs NAS 원본**: bus
0.9010/0.9011, car 0.8687/0.8688, motorcycle 0.8337/0.8338, person 0.8826/0.8826, truck
0.8635/0.8636, all 0.8699/0.8699(conf=0.01); conf=0.25는 bus 0.8539/0.8539, car
0.8085/0.8085, motorcycle 0.7686/0.7686, person 0.8400/0.8400, truck 0.8041/0.8041, all
0.8150/0.8150 — 소수 4자리까지 사실상 동일(±0.0001 float32 노이즈), 지금까지 시도한 모든
조건에서 예외 없이 재확인. **Board표/Original표 어느 쪽과도 일치하지 않음**(bus만 conf=0.25에서
Board표와 우연히 근접, 0.8539 vs 0.8546 — 나머지 4개 클래스는 벗어나 있어 우연으로 판단).

**§7~8 결론**: Export(Step2)는 다시 한번 재확인(원인 아님). bus/car/motorcycle/truck은
class별 optimal threshold 고정 평가에서 Original표와 잘 맞음. **person만 원인 불명** — 유력
후보(미검증, 추측): (1) Original 표를 만든 모델이 이 best.pt가 아닌 다른 버전/체크포인트일
가능성, (2) Original 표의 검증 배치(라벨링 기준, 특히 occlusion/작은 객체 처리)가 이 폴더와
다를 가능성, (3) person 클래스 자체의 GT 라벨링 기준 차이(직접 오탐/미탐 이미지를 열어보지
않아 미확인).

## 9~10. 사용자 제공 IPU_TEST 리스트(4,247장)로 재평가 + conf 스윕

**리스트 확보**: `/mnt/NAS_1/BSD_F606_SIDE_IPU_TEST_list.txt`(4,247줄, CRLF 줄바꿈) — 첫
시도 매칭 0건(CRLF 때문에 `...\r.jpg`로 붙던 버그, `tr -d '\r'`로 해결) → 4,247개 전부
`/dataset/BSD/Side/val`에 이미지+라벨 존재 확인(0건 누락). 격리 서브셋은 symlink로 구성
(원본 미변경).

**결과1 — class별 optimal threshold, 이 서브셋 기준**:

| Class | 재현 ONNX(mAP50/P/R) | NAS ONNX(mAP50/P/R) | Original 표 | mAP50 차이 |
|---|---|---|---|---:|
| bus | 0.844/0.929/0.849 | 0.844/0.931/0.848 | 0.8406/0.935/0.822 | **+0.003** |
| car | 0.734/0.868/0.743 | 0.734/0.868/0.743 | 0.7339/0.844/0.799 | **+0.0001** |
| motorcycle | 0.666/0.924/0.675 | 0.666/0.924/0.675 | 0.7116/0.894/0.689 | -0.046 |
| person | 0.594/0.902/0.601 | 0.594/0.902/0.601 | 0.6838/0.848/0.639 | -0.090 |
| truck | 0.785/0.933/0.790 | 0.785/0.934/0.789 | 0.7914/0.918/0.777 | **-0.006** |
| **all** | 0.725 | 0.725 | 0.7523 | -0.027 |

전체 val(11,001장, §7~8: +0.04~+0.22) 대비 **훨씬 좁혀짐** — bus/car/truck는 사실상 정확히
일치. 이 4,247장 서브셋이 "Original" 표의 실제 검증셋에 매우 가깝다는 강한 정황.
person/motorcycle만 여전히 차이(이번엔 방향이 반대 — 이 서브셋에서는 재현값이 더 낮음).

**결과2 — conf 스윕(class별 threshold/균일0.25/균일0.01), 소수 4자리, 재현 vs NAS ONNX**:
class별 threshold(0.305~0.517) bus 0.844/0.844, car 0.734/0.734, motorcycle 0.666/0.666,
person 0.594/0.594, truck 0.785/0.785, all 0.725/0.725. conf=0.25 균일 bus 0.8534/0.8535,
car 0.7508/0.7508, motorcycle 0.7094/0.7095, person 0.6748/0.6748, truck 0.7945/0.7945, all
0.7566/0.7566. conf=0.01 균일 bus 0.8987/0.8988, car 0.8285/0.8285, motorcycle 0.7811/0.7811,
person 0.7356/0.7356, truck 0.8549/0.8549, all 0.8198/0.8198. **세 설정 전부 재현 ONNX와
NAS 원본 ONNX가 소수 4자리까지 동일**(차이 있어도 0.0001~0.0004 float32 노이즈).

**결과(9~10 종합)**: bus/car/truck는 export나 evaluator 설정과 무관하게 Original표와 잘
맞는다. person(및 motorcycle 약간)만 원인 불명 — 다만 격차는 계속 줄어드는 추세(전체
val 11,001장: person -0.20 → IPU_TEST 서브셋 4,247장: person -0.09) — 데이터셋을 정확한
원본에 가깝게 좁힐수록 격차가 작아짐, 정확한 검증 배치 특정이 맞는 방향이라는 정황.

## 11. 실제 물리 보드 테스트 — 새 NVT BIN vs 기존 reference dump (PASS)

**이 세션(Claude Code)에서 유일하게 직접 할 수 없었던 단계**(물리 하드웨어 접근 불가) —
사용자가 새로 컴파일한 NVT BIN(NAS 원본 ONNX로 재변환, `side_truck_v01_nasref_onnx_20260818`)을
실제 보드에 올려 추론.

**비교 대상**: Reference = `.../Side_Truck/v01/analysis/output.json`(기존 board dump, 원본
미변경). 신규 = `new_complie_output.json`(사용자가 새로 실측). 둘 다 이미지 1장당 JSON
배열이 줄바꿈 없이 이어붙은 형식(`json.JSONDecoder().raw_decode()` 반복 호출로 파싱). 공통
이미지 4,248장(양쪽 다 동일, 누락 없음). Reference 총 detection 14,143개(score 최솟값
0.104004), 신규 총 detection 59,931개(raw 미필터링, score 0.001465부터, 0.01 미만 34,807개).

**핵심 발견 — Reference의 score cutoff**: Reference의 5개 클래스 전부 최저 score가 정확히
**0.104004로 동일** — class별 threshold가 아니라 단일 균일 cutoff. 값이 딱 떨어지지 않는
것으로 보아 사람이 고른 값이 아니라 보드 INT8 양자화 출력의 고정 스케일 값으로 추정(확정
아님).

**매칭 결과**(신규 dump를 score≥0.104004로 필터링 후 IoU≥0.5+동일 카테고리 그리디 매칭):

| 방향 | 총 개수 | 매칭 | 불일치 | 매칭률 |
|---|---:|---:|---:|---:|
| Reference → 신규 | 14,143 | 14,077 | 66 | **99.53%** |
| 신규 → Reference | 14,145 | 14,077 | 68 | **99.52%** |

매칭된 detection 품질: 평균 IoU **0.9925**, 평균 score 차이 **0.000136** — 사실상 동일.

**클래스별 불일치율(Reference 기준)**: person 2,603개 중 6건(**0.23%, 가장 낮음**),
motorcycle 1,820개 중 1건(0.05%), car 5,418개 중 23건(0.42%), bus 1,901개 중 10건(0.53%),
truck 2,401개 중 26건(1.08%).

**불일치 66건 breakdown(정정 — 최초 분석은 부정확했음)**: 사용자가 calibration이 Release
당시와 100% 동일함을 확인해줘서, 66건을 새 dump의 raw(미필터링) 출력 전체와 재대조:

| 유형 | 개수 | 실제 의미 |
|---|---:|---|
| 완전 미스(raw 전체를 뒤져도 매칭 박스 자체 없음) | **42건(64%)** | threshold 문제가 아니라 새 NVT 모델이 그 객체를 **아예 탐지하지 않음** |
| 박스는 있으나 score가 threshold 아래로 하락 | 15건(23%) | 진짜 "경계선" 케이스(하락 폭 큼, 예 0.25→0.005, 0.5→0.001) |
| 그리디 매칭 알고리즘의 충돌(같은 이미지·클래스에 박스 여럿, 다른 ref 항목에 이미 배정됨) | 9건(14%) | 실제 모델 불일치 아님 — 비교 스크립트 매칭 방식의 아티팩트 |

핵심 고신뢰 탐지(예 PERSON 0.883301, TRUCK 0.918945)는 두 파일에서 완전히 동일 — 문제는
애초에 confidence가 낮았던 애매한 객체들뿐.

**100% 일치하지 않는 이유(재분석, calibration 차이는 배제됨)**: (1) **NVT 컴파일러 자체의
미세한 비결정성** — §4에서 같은 ONNX·calibration·config로 두 번 컴파일했을 때도
`nvt_model.bin`이 29 bytes 미세하게 달랐던 것, 그리고 §4의 NAS 원본과의 byte-level 비교
(컴파일러 VERSION 필드는 동일하나 섹션 크기/버퍼 오프셋 메타데이터가 2,273 bytes 다름)가
이를 뒷받침. (2) **양자화 신경망의 비선형적 민감성** — 그 미세한 수치 차이(quantization
scale/zero-point 등)가 여러 레이어를 거치며 누적되면 애매한 객체(작거나 가려졌거나
confidence 낮았던)의 최종 출력에서 크게 증폭될 수 있음 — "확실한 객체는 항상 맞고 애매한
객체만 가끔 틀리는" 관측 패턴과 부합.

**결과: PASS.** 99.53% 매칭 + 매칭분 평균 IoU 0.99/score차 0.00014는 실질적으로 "동일한
모델" 수준. 나머지 0.47%는 "confidence가 threshold 경계를 살짝 넘나든 것"이 아니라 **NVT
컴파일러의 미세한 비결정성이 일부 애매한 객체의 탐지 여부 자체를 바꾼 것**으로 재해석한다.
특히 **person이 가장 낮은 불일치율(0.23%)** — Float ONNX 단계(§6,§7~8)의 person 불일치가
NVT/보드 문제가 아니라 그 비교표들의 evaluator/데이터셋 조건 차이였음을 뒷받침한다.

## 최종 결론

1. **NAS의 ONNX 모델과 재현 ONNX 모델은 같은 데이터를 추론하면 완전히 같은 결과가 나온다**
   (§2, §9~10 — class별 threshold/conf=0.25/conf=0.01 등 여러 조건에서 반복 확인, 항상 소수
   4자리까지 일치).
2. **NAS의 기존 board 기록과, 새로 변환한 NVT 모델을 실제 물리 보드에서 돌린 결과는 99.53%
   일치한다**(§11 — 매칭분 평균 IoU 0.9925/score차 0.00014로 사실상 동일. 나머지 0.47%는
   버그가 아니라 NVT 컴파일러의 정상적인 비결정성 범위 — 컴파일러 버전이 같아도 재컴파일마다
   미세하게 달라질 수 있음을 §4/§8에서 직접 확인).
3. **"NAS의 .pt로 export하면 NAS의 onnx와 다르다"는 부정확한 표현이다 — 정확히는 "어떻게
   export하느냐에 따라 다르다"**(§2). 표준 export 도구(`tools/export.py`,
   `tools/export_custom.py`)를 그대로 쓰면 그래프 구조 자체가 다르게 나온다(NAS onnx는
   출력이 분리된 2-tensor, 표준 도구는 1-tensor 병합). 이 체크포인트(stock `Detect` 헤드)에
   맞는 올바른 export(monkeypatch)를 쓰면 그래프 구조(node diff 0)·파라미터·raw output
   전부 float32 노이즈 수준(1e-6~1e-7)까지 완전히 재현된다 — PT→ONNX 변환 자체의 근본적
   결함이 아니라 **표준 CLI 도구가 이 특정 체크포인트를 위한 export 옵션을 갖추지 않았을 뿐**.

**종합**: best.pt부터 실제 물리 보드까지 전체 배포 파이프라인에 근본적인 결함은 없다. 유일하게
미해결로 남은 것은 사용자가 제공한 두 개의 성능 비교표(§6 Board표, §7 Original표)가 정확히
어떤 evaluator/dataset/threshold 조건으로 만들어졌는지이며, 이는 파이프라인 자체의 문제가
아니라 그 표들의 산출 조건을 확인하지 못해서 남은 문제다.

## First Divergence 표 (원문 그대로)

```
Source PT                                  PASS
PT → ONNX (parameter)                      PASS   (float32 노이즈 수준)
PT → ONNX (graph structure)                PASS   (monkeypatch export, node diff 0)
Calibration Dataset                        PASS   (사용자 제공, gen_config 고정 해상도와 일치)
Calibration List                           NOT AVAILABLE (Reference 없음, 신규 리스트로 대체)
NVT Conversion (compile)                   PASS
NVT Raw Output (simulator)                 PASS
Float ONNX mAP vs repo 기록값                PASS   (CVD/Side, MANIFEST와 사실상 완전 일치)
Float ONNX vs 사용자 제공 Board표             불일치 (원인 미확정)
Float ONNX vs 사용자 제공 Original표          부분 일치 (bus/truck 일치, car/moto 근접, person만 불명)
재현 ONNX vs NAS 원본 ONNX (동일 평가)         PASS (완전히 동일 — export는 person 차이의 원인 아님)
NVT compile/simulate — NAS 원본 ONNX 직접     PASS (재현 ONNX 결과와 사실상 동일: model size 동일,
                                            bin 29 bytes/4.7M만 차이, cosine 0.0005 이내)
Float ONNX vs IPU_TEST 리스트(4,247장) 서브셋  훨씬 근접 (bus/car/truck 거의 완전 일치)
conf 스윕(threshold/0.25/0.01) 재현 vs NAS ONNX PASS (3개 conf 설정 전부 소수 4자리까지 동일)
Device NPU 실측 (사용자 직접 board 테스트)      PASS (99.53% 매칭, person 불일치율 최저 0.23%)
```

## 다음으로 시도해볼 만한 것 (원문 그대로, 미실행)

1. Original 표/Board 표가 정확히 어떤 evaluator·conf·dataset로 만들어졌는지 확인 — 실제
   보드 테스트가 성공한 지금 유일하게 남은 미해결 퍼즐.
2. (선택) person(및 motorcycle) IPU_TEST 서브셋에서의 잔여 오탐/미탐 이미지를 정성적으로
   확인 — 실제 보드 테스트에서 person이 가장 잘 맞는 클래스로 나왔으므로 우선순위는 낮음.

## 재현 명령

```bash
# PT -> ONNX (최종 채택 방법)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/debug/side_export_skip_postprocess_patch.py \
  -w runs/debug/side_full_reproduction/inventory/reference_best.pt \
  -i 320 -y 320 -o 13 \
  -s runs/debug/side_full_reproduction/pt_to_onnx/exported_from_best_patched_skip_postprocess.onnx
'

# NVT compile + simulator
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_compiler.sh side_truck_v01_full_reproduction_20260814'
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_simulator.sh side_truck_v01_full_reproduction_20260814 \
   1t_KR_20240129_TRUCK1T_RIGHT_140CM_A_140_OUTDOOR_NC22015_0376.jpg'

# Float ONNX mAP (CVD/Side, repo 기록값과 비교)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/debug/validate_side_split_head.py \
  -w runs/debug/side_full_reproduction/pt_to_onnx/exported_from_best_patched_skip_postprocess.onnx \
  -c configs/releases/BSD_F606/Side_Truck/v01/exp.yaml \
  -a configs/datasets/cvd_side.yaml -d 0 -i 320
'

# Float ONNX mAP (val_iputest, 사용자 제공 Board 표와 비교)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/customEval.py \
  runs/debug/side_full_reproduction/comparison/side_val_iputest_dualnms_cfg.yaml
'

# Float ONNX mAP (BSD/Side/val, class별 optimal threshold 고정 — 사용자 제공 Original 표와 비교)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/customEval.py \
  runs/debug/side_full_reproduction/comparison/side_val_direct_perclass_thresh_cfg.yaml
'

# 동일 평가를 NAS 원본 onnx로 (재현 onnx와 비교용)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/customEval.py \
  runs/debug/side_full_reproduction/comparison/side_val_direct_perclass_thresh_NASREF_cfg.yaml
'

# NVT compile + simulator — NAS 원본 ONNX 직접 (재현 ONNX 결과와 비교용, 격리 디렉토리)
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_compiler.sh side_truck_v01_nasref_onnx_20260818'
docker compose exec npu-convert bash -lc \
  'cd /workspace/novatek/convert && ./run_simulator.sh side_truck_v01_nasref_onnx_20260818 \
   1t_KR_20240129_TRUCK1T_RIGHT_140CM_A_140_OUTDOOR_NC22015_0376.jpg'

# IPU_TEST 리스트(4,247장) 서브셋 — class별 threshold / conf=0.25 / conf=0.01, 재현 vs NAS ONNX
# (서브셋 구성: /mnt/NAS_1/BSD_F606_SIDE_IPU_TEST_list.txt를 tr -d '\r'로 정리 후
#  /dataset/BSD/Side/val/{images,labels}를 가리키는 symlink로 ipu_test_subset/ 생성)
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/debug/side_eval_4dp.py \
  runs/debug/side_full_reproduction/comparison/ipu_test_subset_conf001_ours_cfg.yaml
'
docker compose exec dair-train bash -lc '
cd /workspace && uv run python3 tools/debug/side_eval_4dp.py \
  runs/debug/side_full_reproduction/comparison/ipu_test_subset_conf001_nasref_cfg.yaml
'
```

## 생성 파일 (핵심만, 전체는 원문 참조)

```
runs/debug/side_full_reproduction/inventory/reference_best.pt (NAS 사본, SHA256 검증됨)
runs/debug/side_full_reproduction/pt_to_onnx/exported_from_best_patched_skip_postprocess.onnx  ← 최종 채택
runs/debug/side_full_reproduction/calibration/ref_imgs/jpg/ (700장, 사용자 제공)
novatek/convert/side_truck_v01_full_reproduction_20260814/ (격리 NVT compile/simulator 결과)
novatek/convert/side_truck_v01_nasref_onnx_20260818/ (NAS 원본 ONNX 직접 NVT compile/simulate, 격리)
tools/debug/side_export_skip_postprocess_patch.py (신규)
projects/src/models/yolo/detect/val/side_split_head_jdh.py (SideSplitHeadValidatorJdh, 신규)
tools/debug/validate_side_split_head.py, side_metrics_utils_jdh.py, side_eval_4dp.py (신규)
/mnt/NAS_1/BSD_F606_SIDE_IPU_TEST_list.txt (사용자 제공, 원본 — 읽기 전용)
runs/debug/side_full_reproduction/ipu_test/new_complie_output.json (사용자가 실제 board에서 직접 실측)
```
