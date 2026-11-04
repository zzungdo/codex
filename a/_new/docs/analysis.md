# 분석 도구 규격

적용: ROI·정답 필터와 예측 위치·크기 분석 · 기준: 2026-09-28

명령어와 옵션은 [사용법의 분석 절](usage.md#분석)을 따른다.
학습 컨테이너 `/workspace`에서 실행하며 도구별 데이터 형식은 아래와 같다.

## ROI·정답 필터

도구: `tools/analyze_crop.py`

### 입력

- 원본 릴리스 디렉터리의 `images/`와 `labels/`; 라벨은 YOLO 정규화 형식이다.
- ROI는 원본 좌표의 `--crop X Y W H`, 잔존율은 `--min-keep`으로 지정한다.
- `--lens CX CY RX RY`도 원본 이미지 좌표다.
- 클래스 순서는 `--classes`이며 기본은 bus, car, motorcycle, person, truck이다.

### 처리 규칙

class → edge → crop → min_keep → clip → size 순서로 판정한다.
가장자리는 crop 전 원본 좌표의 박스 중심으로 판정한다.
`--edge-margin`은 퍼센트 단위다.

`--clip-ellipse`는 crop과 타원의 교집합을 가시영역으로 사용하고 중심점 기반 edge 탈락을 끈다.
`--mask-outside`를 함께 쓰면 타원 밖 픽셀을 검게 처리한다.
최소 크기는 clip 후 박스에 적용하며 `--size-metric`의 기본은 sqrt_area,
짧은 변 기준은 min_side다. 클래스별 크기는 공통 `--min-size`보다 우선한다.

### 출력

| 산출물 | 내용 |
|---|---|
| `images/`, `labels/` | ROI 이미지와 재투영·필터링한 라벨 |
| `spec.json` | 생성 조건 |
| `report.json` | 판정 집계 |
| `--verify-against` | 지정 릴리스와 라벨 바이트 비교; 일반 생성과 구분 |

### 제약

새 출력 디렉터리를 사용한다. `--drop-empty`는 정답이 남지 않은 이미지도 제외한다.
렌즈 클리핑은 격자 샘플링 근사이며 `--ellipse-grid`가 정밀도·처리량을 조절한다.
`--link-images`는 crop이 필요 없는 경우에만 사용한다.

## 예측 위치·크기 분석

도구: `tools/analyze_heatmap.py`

저장된 COCO JSON을 클래스별 고정 임계값과 IoU 0.5로 분석한다.
표준 라이브러리만 사용하므로 별도 Python 3 환경에서도 실행할 수 있다.
출력은 `<output-dir>/analysis_<tag>.json`이며 `--output-dir` 기본값은 현재 디렉터리다.
콘솔에는 전체 지표, 구역별 recall/fprate, 최외곽 비중, 크기별 recall, 클래스별 recall/precision을 출력한다.

### 입력

- GT: `images` (`id`, `width`, `height`), `categories` (`id`, `name`), `annotations`
  (`image_id`, `category_id`, `bbox`)를 가진 COCO 객체.
- 예측: `image_id`, `category_id`, `bbox`, `score`를 가진 배열 또는 이 배열을 `annotations`에 담은 객체.
- 양쪽 `bbox`는 원본 이미지의 절대 픽셀 `[x, y, width, height]`다. 상자를 이미지 경계로 자르지 않는다.
- 양쪽 ID와 타입이 같아야 한다. 예를 들어 문자열 `"001"`과 숫자 `1`을 자동 변환하지 않는다.
  예측 생성 도구가 파일명을 ID로 쓰는 경우에도 GT와 동일한 ID를 사용해야 한다.
- 클래스 ID를 고정하지 않고 GT `categories`의 이름으로 기준값을 찾는다.
  예측에 등장하는 모든 클래스의 기준값이 필요하며 누락·잘못된 ID는 오류로 알린다.
  `bus`, `truck` 등 다른 클래스가 있으면 기준값을 추가한다. 전체/구역/크기 집계에는 모두 포함하고,
  `per_class`는 명세의 `car`, `motorcycle`, `person` 세 항목을 항상 출력한다.
- 이미지 크기는 양수, 상자의 너비·높이는 0 이상, 좌표·점수·기준값은 유한한 숫자여야 한다.

### 처리 규칙

`score >= 해당 클래스 기준값`인 예측만 남긴다. GT `annotations` 순서대로 같은 이미지·클래스의
미사용 예측 중 IoU가 가장 큰 것을 선택한다. 동점은 필터 통과 후 입력 순서가 앞선 예측을 선택한다.
IoU가 0.5 이상일 때만 양쪽을 사용 완료 처리하고, 실패한 GT는 FN, 최종 미사용 예측은 FP다.
합집합 면적이 0인 IoU는 0이다. `iscrowd`/`ignore` 특례, score 정렬, NMS는 적용하지 않는다.

이미지마다 가로·세로를 10등분하며 상자 중심의 상대 위치에 floor를 적용하고 행/열을 0–9로 자른다.
행 또는 열이 0/9이면 border(36칸), 나머지는 center(64칸)다.
TP/FN의 위치는 GT 중심, FP의 위치는 예측 중심을 쓴다.

모든 지표 묶음은 `n`, `precision`, `recall`, `f1`, `fprate`를 가진다.
`n = TP + FN`, `precision = TP/(TP+FP)`, `recall = TP/n`, `f1 = 2PR/(P+R)`,
`fprate = 1-precision`이다. P/R/F1의 분모가 0이면 0이며, 이 경우 precision이 0이면 fprate는 1이다.
각 지표는 반올림 전 비율에서 계산한 뒤 소수 셋째 자리까지 저장한다(half-up 반올림).

### 출력

| JSON 키 | 내용 |
|---|---|
| `n_images`, `thresholds` | GT 이미지 수(상자가 없는 이미지도 포함), 사용한 기준값 |
| `total` | 전체 GT·전체 FP의 지표 |
| `zone.center`, `zone.border` | GT와 FP를 각 상자 구역으로 선택한 지표 |
| `zone_share` | `area_border: 36`, GT/FN/FP 각각의 최외곽 비중(%, 소수 첫째 자리). 전체 수가 0이면 비중 0 |
| `size_px` | 짧은 변 기준 `[0,40)`, `[40,60)`, `[60,100)`, `[100,200)`, `[200,∞)`. GT와 FP 각각 자기 크기로 배정 |
| `size_rel` | GT 짧은 변 / **이미지 width** × 100. `[0,2)`, `[2,3)`, `[3,5)`, `[5,10)`, `[10,100)`. **모든 구간에 전체 FP를 재사용** |
| `per_class` | car/motorcycle/person 각각 해당 클래스 GT·FP의 지표 |
| `grid` | `cnt`, `hit`, `fp`, `tpp` 10×10 정수 배열. 인덱스는 `[행][열]` |

### 제약

`size_px` 키는 `<40`, `40-60`, `60-100`, `100-200`, `200+`이며,
`size_rel` 키는 `<2%`, `2-3%`, `3-5%`, `5-10%`, `10%+`다.
`10%+`도 지정된 상한에 따라 **100% 미만**만 포함한다. 100% 이상 GT는 다른 집계에는 포함된다.
`size_rel`의 precision은 모든 구간에서 전체 FP를 사용하므로 주로 recall 비교에 쓴다.

`cnt`는 GT 수, `hit`는 매칭된 GT 수, `fp`는 미사용 예측 수다.
`tpp`는 매칭된 **예측 중심**에 누적하므로 `hit`와 셀별 값은 달라도 전체 합은 같다.

학습 컨테이너 `/workspace`에서 규칙을 검증한다.

```bash
uv run pytest tests/test_spatial_analysis.py -q
```
