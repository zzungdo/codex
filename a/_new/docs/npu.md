# NPU 포팅·배포·검증

적용: Novatek NT98530·Sigmastar SAV837 detection, DMS ONNX export · 기준: 2026-09-29

PT → ONNX → 양자화·컴파일 → SDK 시뮬레이션 → 보드 배포 → 평가 순서로 진행한다.
[환경 설정](setup.md), [명령어·옵션](usage.md#포팅),
[Sigmastar 변환 구현](../sigmastar/README.md)을 함께 참조한다.
최근 변환 산출물은 2026-09-23 기준이며 **실물 보드 검증은 수행하지 않았다.**

## 대상 모델과 준비물

필수 입력은 릴리스의 `exp.yaml`, `weight/weights/best.pt`, 클래스 순서,
학습 화각을 적용한 보정 이미지, 별도 검증 이미지와 GT다.
원본 PT·ONNX·NPU 바이너리·설정·이미지 목록의 해시를 함께 보관한다.

| BSD_F606 모델 | 입력 W×H | ROI (x, y, w, h) | 클래스 순서 |
|---|---|---|---|
| Corner_Truck/v01 | 320×320 | 428, 8, 1072, 1072 | bus, car, motorcycle, person, truck |
| Front_Truck/v01 | 640×512 | 320, 0, 1280, 1024 | 위 5개 + traffic_light |

PC 입력 규약은 ROI를 이미지 경계로 자른 뒤 INTER_AREA resize, BGR float32 [0,255]다.
`/255` 정규화를 하지 않는다. 이미지 파일명은 GT와 일치시킨다.
DMS pose는 아래 별도 ONNX 규약을 사용하며 detection 보드 후처리를 적용하지 않는다.

최근 생성한 패키지는 로컬 `runs/porting/bsd_f606_v01_20260923/`와 같은 이름의
`.tar.gz`에 있다. 이 경로는 Git에 포함되지 않는다.

| 보드 | 패키지 내부 Corner 파일 | 패키지 내부 Front 파일 |
|---|---|---|
| Novatek | `Corner_Truck/novatek_nt98530/corner_truck_v01.bin` | `Front_Truck/novatek_nt98530/front_truck_v01.bin` |
| Sigmastar | `Corner_Truck/sigmastar_sav837/corner_truck_v01.img` | `Front_Truck/sigmastar_sav837/front_truck_v01.img` |

패키지의 `preparation.json`은 원본·이미지·ROI, `manifest.json`은 해시·검증 요약,
`SHA256SUMS`는 전체 파일 해시, 각 보드의 `verification.json`은 상세 비교 결과다.

## 공통 규약

흐름은 PT → 칩별 ONNX → 양자화·컴파일 → SDK 시뮬레이션 → 보드 실행 → 동일 조건 평가다.
명령은 [사용법](usage.md)의 포팅 절을 따른다.

### 출력·필터 차이

두 ONNX 모두 입력은 `[1,3,H,W]`다. `A`는 앵커 수, `nc`는 클래스 수다.

| 항목 | Novatek | Sigmastar |
|---|---|---|
| ONNX target | `novatek` | `sigmastar` |
| ONNX 출력 | `[1,nc,A]` sigmoid 점수 + `[1,4,A]` DFL 적용 거리 | `[1,A,5+nc]`: 정규화 xyxy, max_score, 클래스 점수 |
| 기본 opset | 13 | 12 |
| 변환 산출물 | `.bin` | `.img` |
| 모델 내 NMS | 없음 | score 0.45−1e−5, IoU 0.40−1e−5, 최대 100개 |
| 테스터 후처리 | Front 행(index 0), sensitivity 1(MID)의 클래스별 임계 | `iputest` 경로에서 score > 0.45 |
| 보드 NMS | min overlap 0.6 / union overlap 0.3, 클래스당 최대 100개 | 모델 내 NMS 결과 사용 |
| 보드 입력 | YUV420/NV12 → SDK BGR | 소스의 FORMAT은 RGB; 실제 버퍼 채널 순서 확인 필요 |

`SymmetricDetect`와 순정 `Detect`는 exporter가 구분한다. 서로 다른 head로 가중치를
이식하지 않는다. Novatek의 4채널 출력은 16-bin DFL 분포가 아닌 **DFL 적용 후 거리**다.
`433` 같은 출력 이름, 노드 수, SDK 변환 후 출력 순서는 산출물마다 확인한다.
의미·shape·dtype·opset·decode 포함 여부를 확인하고, 이름이나 순서만으로 텐서를 판별하지 않는다.

### 화각과 좌표계

| 입력 준비 | 보드 설정 | 결과 좌표 | 원본 GT 평가 전 처리 |
|---|---|---|---|
| 원본 이미지 | crop 지원 Novatek 테스터에 ROI 설정 | 원본 전체프레임 | 추가 변환 없음 |
| ROI로 자른 이미지 | crop 키 없음 | crop 내부 좌표 | ROI 오프셋을 한 번 더함 |
| ROI crop 후 모델 크기로 resize한 이미지 | crop 키 없음 | 준비 이미지 좌표 | resize 역변환 후 ROI 오프셋을 한 번 더함 |

crop 내부 `[100,100,200,200]`에 Corner ROI 오프셋을 더하면 원본
`[528,108,628,208]`이다. 이미지와 GT를 모두 crop 좌표계로 준비하면 복원이 필요 없다.

평가기의 `gt.roi`는 **GT 필터**다. `parse_ipu_result`는 xyxy→xywh와 scale만 적용하며
crop 오프셋을 자동으로 더하지 않는다. 사전 crop 이미지에 보드 crop을 다시 적용하지 않는다.

## ONNX export

학습 컨테이너 `/workspace`에서 실행한다.
원본 릴리스 PT를 작업 디렉터리로 복사한 후 [사용법의 export 명령](usage.md#포팅)을 적용한다.
`-i`는 너비, `-y`는 높이이며 `-y` 생략 시 정사각형이다.
export 후 실제 입력·출력 shape와 ONNX Runtime 결과를 확인한다.
숫자 텐서명과 노드 수를 다른 모델에 그대로 적용하지 않는다.

### DMS 출력 규약

`--target dms`는 stage에 따라 3출력을 만든다.
`K`는 keypoint 수, `A5 = H5×W5`는 P5 앵커 수다.

| stage | output0 | output1 | output2 |
|---|---|---|---|
| Stage-1 (nc=1) | `[1,nc,A5]` cls | `[1,4,A5]` DFL 거리 | `[1,K×3,A5]` keypoint xy·vis |
| Stage-2 (nc>1) | `[1,K×2,A5]` keypoint xy | `[1,4,A]` DFL 거리 | `[1,nc,A]` cls |

Stage-2는 vis 채널을 export 그래프에서 제거한다.
metadata에 task·stage·imgsz·kpt_shape·nc·stride·names를 기록한다.
구현은 [onnx_export.py](../projects/src/engine/onnx_export.py)를 기준으로 한다.
실제 DMS PT를 사용한 회귀 검증 자산은 아직 확보가 필요하다.

## Novatek NT98530

### Novatek 입력 준비

모델마다 `novatek/convert/<model_dir>/`를 사용하며 `<PATTERN>`은 모델 디렉터리명의 대문자다.
새 모델은 기존 레시피의 **텍스트 설정**을 기준으로 준비하고 모델·이미지·목록을 새 입력으로 채운다.

```text
novatek/convert/<model_dir>/
  gen_config.txt
  sim_config.txt
  model/best.onnx
  cal_img/
  nvtai_tool/
    config/<PATTERN>/cnn25/{gen_config,sim_config}.txt
    input/model/customer/<PATTERN>/deploy_float.onnx
    input/data/<PATTERN>/jpg/
    input/data/<PATTERN>/ref_img_list.txt
    input/data/<PATTERN>/test_img_list.txt
```

`best.onnx`를 raw ONNX인 `deploy_float.onnx`로 복사한다.
이 복사는 Nova ONNX 변환을 대신하지 않는다. compiler wrapper가
ONNX → Nova ONNX → 보정·양자화 → NPU 컴파일을 순서대로 실행한다.
SDK 명령을 직접 호출해 중간 단계를 생략하지 않는다.

보정 이미지는 학습 화각·입력 해상도·분포에 맞추고 검증 이미지를 별도로 준비한다.
`ref_img_list.txt`와 `ref_data/num`이 실제 보정 목록에 맞아야 한다.
`sim_config.txt`의 시험 이미지 디렉터리·목록은 시뮬레이션 입력을 가리킨다.

호스트 저장소 루트, Corner 설정 동기화 예:

```bash
MODEL_DIR=novatek/convert/bsd_f606_corner_v01_20260923
PATTERN=BSD_F606_CORNER_V01_20260923
cp "$MODEL_DIR/gen_config.txt" "$MODEL_DIR/nvtai_tool/config/$PATTERN/cnn25/gen_config.txt"
cp "$MODEL_DIR/sim_config.txt" "$MODEL_DIR/nvtai_tool/config/$PATTERN/cnn25/sim_config.txt"
```

wrapper는 `nvtai_tool/config/<PATTERN>/cnn25/`를 실행 디렉터리로 복사한다.
상단 설정만 수정하면 적용되지 않는다.
`./input/...` 경로는 wrapper가 소스·실행 위치에 맞는 절대경로로 바꾼다.
직접 작성한 `../` 상대경로는 이 변환 대상이 아니므로 아래 레시피 형식을 따른다.

### Novatek gen_config

아래 값은 [Corner v01 레시피](../novatek/convert/bsd_f606_corner_v01_20260923/gen_config.txt) 기준이다.
[Front v01 레시피](../novatek/convert/bsd_f606_front_v01_20260923/gen_config.txt)는 640×512·6클래스다.
SDK 옵션의 허용 조합·하드웨어 상한은 해당 SDK의 공식 매뉴얼로 확인한다.
입력 enum과 출력 enum은 서로 같지 않다.

| 키 | 기준값·목적 |
|---|---|
| `path/model_dir` | `./input/model/customer/<PATTERN>` |
| `path/ref_img_dir`, `path/ref_list_path` | `./input/data/<PATTERN>/jpg`, 보정 이미지 목록 |
| `path/out_dir` | `./output`; wrapper가 실행 디렉터리 경로로 변환 |
| `ref_data/num` | 300; 실제 보정 이미지 수 |
| `multiscale/en` | 0; 단일 입력 해상도 |
| `preproc/in/fmt`, `in/channel` | 2, 2; NV12/YUV420 입력과 Y·UV plane |
| `preproc/in/type`, `in/frac_bit_num` | 1, 0; 입력 타입·소수부 설정은 fmt에 맞춤 |
| `preproc/in/width`, `in/height` | Corner 320×320, Front 640×512 |
| `preproc/in/batch`, `in/time` | 1, 1 |
| `preproc/resize/mode`, `resize/width`, `resize/height` | 0, 모델 W/H; 레시피의 RESIZE_WH 설정과 준비 이미지 크기 정합 |
| `preproc/meansub/en` | 0; mean subtraction 미사용 |
| `preproc/meansub/mode`, `meansub/fmt`, `meansub/fmt_type` | 0, 0, 1; meansub 비활성 기준의 부가 설정 |
| `preproc/normalize/en`, `normalize/scale` | 0, 1; BGR [0,255], /255 정규화 없음 |
| `preproc/out_fmt` | 3; BGR. 입력의 fmt=3(FEAT)과 혼동 금지 |
| `postproc/en` | 0; 보드용 출력에 SDK 후처리 노드를 추가하지 않음 |
| `precision/in_hp/conv_en`, `in_hp/conv_op` | 0, 3; 선택 텐서 정밀도 설정 |
| `precision/in_hp/bnscale_en` | 1 |
| `precision/in_hp/{deconv,fc,eltwise,roipool}_en`, `in_hp/eltwise_mode` | 0 |
| `precision/high_precison/spec_demarc_tensors` | 실제 ONNX의 DFL·P3 box·P3 cls 출력 이름; 16-bit 유지 |
| `precision/out_hp/eltwise_mode` | 0; 자동 |
| `precision/calib_mode`, `percentile`, `cal_refine_iter` | 4(PERCENTILE), 0.9999, 16 |
| `precision/ref_relu_outval_en`, `cle_en` | 1, 1 |
| `precision/balance_weight/out_range_ratio` | 1000.0 |
| `precision/bias_bitdepth_mode`, `round_mode` | 2, 1 |
| `compression/method/quan_en`, `method/vlc_en` | 1, 0 |
| `performance/mode`, `shrink_en`, `cnn_engine_mode`, `job_schedule_en` | 1, 1, 0, 1 |

`high_precison`은 SDK 키의 실제 철자다.
텐서명은 모델별로 확인하며, 기존 모델의 이름을 복사하는 것만으로 정밀도가 보장되지는 않는다.
RGB [0,1]로 학습한 모델에는 별도 RGB 출력·정규화 설정이 필요하다.
DMS에도 detection 후처리나 출력 텐서 설정을 그대로 적용하지 않는다.

### Novatek 변환·배치

호스트 저장소 루트에서, 입력 준비가 끝난 Corner 레시피를 실행하는 예:

```bash
docker compose up -d npu-convert
docker compose exec npu-convert bash novatek/convert/run_pipeline.sh \
  bsd_f606_corner_v01_20260923 test_0000.jpg 0.45 \
  --names bus car motorcycle person truck
```

Front는 레시피를 `bsd_f606_front_v01_20260923`으로 바꾸고 클래스 끝에
`traffic_light`를 추가한다. 레시피에 대응하는 ONNX·보정·시험 이미지가 필요하다.

산출물은 `runs/npu/novatek/<model_dir>/nvtai_tool/output/<PATTERN>/`에 생성된다.
`sdk/*.bin`, `gentool/`, `simtool/`, `debug/`와 상위 `run_manifest.json`을 보관한다.
경로는 manifest를 기준으로 읽는다.

완성된 패키지에서 SD로 복사하는 예(호스트, SD 마운트 경로 수정):

```bash
SD_ROOT=/media/sdcard
PORT_DIR=runs/porting/bsd_f606_v01_20260923
mkdir -p "$SD_ROOT/model" "$SD_ROOT/images" "$SD_ROOT/ipu-result"
cp "$PORT_DIR/Corner_Truck/novatek_nt98530/corner_truck_v01.bin" \
  "$SD_ROOT/model/yolov8n_320x320_Side.bin"
```

| 모델 | SD `model/` 파일명 | ini `model` | `sclwid / sclhei` |
|---|---|---|---|
| Corner | `yolov8n_320x320_Side.bin` | `yolov8n_320x320_Side` | 320 / 320 |
| Front | `yolov8n_640x512_Front.bin` | `yolov8n_640x512_Front` | 640 / 512 |

`model`은 등록된 후처리 선택자다. 파일 탐색이 부분 문자열 매칭이므로 같은 선택자에
여러 파일이 걸리지 않게 한다. pose 선택자를 detection에 사용하지 않는다.
Front는 **6클래스와 raw ID 5 = traffic_light**에 맞는 후처리·클래스 테이블이 필요하다.

### Novatek 입력·설정

`fmf/NT98530` 소스는 crop 4키를 파싱해 스케일러의 crop+resize와 출력 오프셋 복원에 쓴다.
`fmf/ncn_cv`의 해당 테스터는 crop을 지원하지 않는다.
NAS 배포 `vcap_iputest`의 대응 소스 커밋은 미확인이다. 실제 바이너리의 crop 지원을 먼저 확인한다.

SD 최상단 `iputest.ini`, Corner 원본 이미지 + crop 지원 테스터 예:

```ini
modeldir = /mnt/mmc
imgwid   = 1920
imghei   = 1088
maxwid   = 3840
maxhei   = 2160
sclwid   = 320
sclhei   = 320
sclopt   = 0
inputfmt = YUV420
model    = yolov8n_320x320_Side
output   = output.json
outdir   = /mnt/mmc/ipu-result
scldir   =
cropx    = 428
cropy    = 8
cropw    = 1072
croph    = 1072
```

`imgwid/imghei`는 SDK의 실제 입력 크기 결정을 대신하지 않는다.
`inputfmt`는 gen_config의 `preproc/in/fmt`와 맞춘다(YUV420 = 2).
사전 crop 절차에서는 crop 4키를 빼고, crop 이미지와 [화각·좌표 규칙](#화각과-좌표계)의 좌표 복원 규칙을 사용한다.
Front 원본을 넣는 경우 ROI와 모델·해상도를 [대상 모델](#대상-모델과-준비물)과 [배포 파일명](#novatek-변환배치)의 값으로 함께 바꾼다.

### 임계와 실행

검출 기본 경로는 `ncn_adas_new/.../soc/funcs/nn_post_process.cpp`의
`stPostProcessInfoClass::post_proc_baseline`이다.
`vcap_iputest`는 sensitivity 1(MID), input_channel 0(Front 행)을 사용한다.
낮은 임계로 평가하려면 이 경로의 임계를 바꾸고 ADAS 라이브러리·테스터를 재빌드·설치한다.
다른 후처리 함수나 LOW 열만 바꿔서는 적용되지 않는다.

라이브러리 빌드(워크스페이스의 `ncn_adas_new/`):

```bash
./release_library.sh nt98530 NC-NOVA ../fmf RELEASE
```

생성 라이브러리가 포함된 fmf 펌웨어·테스터를 보드에 설치하고 버전·해시를 기록한다.
라이브러리 빌드만으로 보드가 갱신되지는 않는다. SD에 테스터·ini·모델·입력 이미지를 배치한 뒤
보드에서 실행한다.

```bash
cd /mnt/mmc
./vcap_iputest
```

출력은 `/mnt/mmc/output.json`과 `/mnt/mmc/ipu-result/`다.
모델·테스터·라이브러리·펌웨어 해시와 ini, 입력 목록, 실행 로그를 함께 회수한다.

## Sigmastar SAV837

### Sigmastar 변환·배치

변환 코드 정본은 `sigmastar/converter/`다. SDK 사본에 반영한 뒤 실행한다.
`prepared_bgr`는 resize·정규화를 하지 않으므로 이미지를 **각 모델의 입력 해상도**로
미리 crop·resize한다. 보정 300장과 검증 10장은 별도 디렉터리로 준비한다.

호스트 저장소 루트, Corner 예:

```bash
bash docker/sigmastar/run.sh 'bash converter/sync_to_sdk.sh'
bash docker/sigmastar/run.sh 'bash converter/run.sh \
  data/input/models/bsd_f606_corner_v01_20260923.onnx yolov8_xyxy_320x320 0.45 \
  data/input/bsd_f606_corner_v01_20260923/calibration \
  data/input/bsd_f606_corner_v01_20260923/simulation prepared_bgr'
```

Front는 모델·이미지 경로의 `corner`를 `front`로 바꾼다. 후처리 모듈 이름에 320x320이
들어 있지만 클래스·앵커 수는 ONNX shape에서 읽는다. Front 입력은 640×512다.
기본 전처리 모듈의 320×320 resize와 `prepared_bgr`를 혼동하지 않는다.

출력은 `sigmastar/data/output/<model_name>/`이다.
`img/*_concat_postprocess_fixed.img`가 배포 파일이고,
`sim/`과 `result/{Float,Fixed,Offline}/`는 시뮬레이션 산출물이다.

SD 배치(호스트):

```bash
SD_ROOT=/media/sdcard
PORT_DIR=runs/porting/bsd_f606_v01_20260923
mkdir -p "$SD_ROOT/model" "$SD_ROOT/img" "$SD_ROOT/ipu-result"
cp "$PORT_DIR/Corner_Truck/sigmastar_sav837/corner_truck_v01.img" \
  "$SD_ROOT/model/onnx_yolox.img"
```

보드 소스는 `ncn_adas/CVD-H210` + `fmf/ncn_cv` 조합을 기준으로 한다.
`ncn_adas_new`의 Novatek 설치 경로를 그대로 적용하지 않는다.
라이브러리 변경 시 `ncn_adas/`에서 `./sav83x_release.sh <PRODUCT> ../fmf`를 실행한 뒤
제품별 fmf 펌웨어를 빌드·설치한다. 이 스크립트는 실패를 종료 코드로 전파하지 않는 사례가 있어
`Build failed` 로그와 실제 갱신된 라이브러리 해시를 확인한다.

### Sigmastar 입력·설정

테스터는 crop을 지원하지 않는다. ROI가 적용된 이미지를 SD `img/`에 넣고,
원본 GT 평가 시 [화각·좌표 규칙](#화각과-좌표계)에 따라 좌표를 복원한다. SD 최상단 설정:

```ini
modeldir = /mnt/mmc
maxwid   = 3840
maxhei   = 2160
sclwid   = 320
sclhei   = 320
sclopt   = 0
output   = output.json
outdir   = /mnt/mmc/ipu-result
scldir   =
firmdir  =
```

Front는 `sclwid=640`, `sclhei=512`다.
실제 추론 크기는 모델 텐서에서 읽으며 스케일러 크기도 일치시킨다.
`model`, `inputfmt`, `imgwid`, `imghei`는 파싱하지 않으며 `firmdir`는 파싱만 한다.
모델 경로는 `<modeldir>/model/onnx_yolox.img`로 고정된다.

### 임계·클래스·실행

임계는 두 곳에 있다.

| 위치 | 현재 규칙 | 변경 방법 |
|---|---|---|
| 모델 NMS | score 0.45−1e−5, IoU 0.40−1e−5, 최대 100개 | `yolov8_xyxy_320x320.py` 수정 → `sync_to_sdk.sh` → 재변환 |
| 보드 `iputest` | score > `NMS_M2SSD_THRESHOLD`(0.45) | 보드 소스 변경 → 라이브러리·테스터 재빌드·설치 |

**`run.sh` 세 번째 인자와 `onnx2sgsimg.py -c`는 시각화 임계만 바꾼다.**
모델 임계만 낮추면 보드의 0.45 필터는 남는다.
`iputest`가 전달하는 sensitivity 50은 `savnn_test_image`에서 -1로 바뀌므로
클래스별 임계 테이블도 적용되지 않는다.

모델 출력 클래스는 0부터 시작한다. 시뮬레이터의 COCO 덤프와 보드 후처리가 각각 +1을 적용하지만
서로 다른 경로이므로 중복 가산이 아니다.

| 모델 raw ID | 보드 ID | 클래스 |
|---|---|---|
| 0 / 1 / 2 / 3 / 4 | 1 / 2 / 3 / 4 / 5 | BUS / CAR / MOTOR / PERSON / TRUCK |
| 5 (Front만) | 6 | traffic_light — 기존 WHEEL 매핑 수정 필요 |

`DL_YOLOV8_CV`의 BG 슬롯과 `gClassNames[]`를 맞춘다.
`DL_YOLOV8_NANO`는 같은 매핑이 아니며 truck ID 5가 배열 범위를 넘는다.
Novatek도 raw ID 5의 WHEEL/traffic_light 의미를 확인해야 한다.
5클래스 모델에 WHEEL 검출이 생기는 것은 아니므로 코너뷰의 WHEEL 사용 경로를 별도 확인한다.

보드에서 실행:

```bash
cd /mnt/blackbox/bin
./iputest
```

제품 스크립트가 SD `iputest.ini`를 보고 자동 실행할 수 있으므로 중복 기동을 피한다.
출력은 SD `output.json`과 `ipu-result/`다.
소스의 `FORMAT "RGB"`와 PC 모델의 BGR 입력이 실제로 일치하는지는 알려진 색 패턴으로
보드에서 확인해야 한다.

## 평가와 검증 기준

### 결과 회수·평가

호스트 저장소 루트에서 SD 결과를 회수한다.

```bash
SD_ROOT=/media/sdcard
mkdir -p runs/board/Corner_Truck
cp "$SD_ROOT/output.json" runs/board/Corner_Truck/output.json
cp -r "$SD_ROOT/ipu-result" runs/board/Corner_Truck/
```

학습 컨테이너 `/workspace`에서 평가한다. 좌표 복원이 필요한 결과는 먼저 변환한다.

```bash
uv run python3 tools/customEval.py \
  -w configs/releases/BSD_F606/Corner_Truck/v01/exp.yaml \
  --eval board --pred runs/board/Corner_Truck/output.json \
  --save-dir runs/board/Corner_Truck/eval
```

Front는 위 경로의 `Corner_Truck`를 `Front_Truck`로 바꾸고 클래스 매핑을 확인한다.
`eval.board.rep: ipu_test`, `scale: [1.0,1.0]`과
`cat_map: {BUS: 0, CAR: 1, MOTOR: 2, PERSON: 3, TRUCK: 4}`가 Corner 기준이다.
Front에는 보드가 출력한 traffic_light 문자열에 대한 ID 5 매핑도 필요하다.

보드 JSON은 프레임별 배열이 이어진 `...][...` 형식일 수 있다.
평가기의 `merge_board_json`이 처리하므로 단일 `json.load`로 읽지 않는다.
`bbox`는 xyxy, `name`은 입력 파일명, `category`는 클래스 문자열, `time`은 추론 ms다.

`customEval.py --conf`와 `--iou-thr`는 ONNX 예측 생성에 적용된다.
보드 JSON을 같은 임계·NMS로 다시 처리하는 옵션이 아니다.
보드가 버린 저점수 검출은 복원할 수 없으며, conf 0.1 결과도 0.1 미만의 PR 구간은 포함하지 않는다.
두 경로의 차이는 전처리·필터·NMS까지 포함한 **전체 파이프라인 정확도 차이**로 기록한다.

### 단계별 증거

| 단계 | 보관·확인 항목 |
|---|---|
| PT → ONNX | 원본 해시, 실제 입력 shape/dtype/채널/범위, 출력 의미, 텐서별 오차 |
| 변환 | SDK 버전, 옵션·보정 목록, 로그, 바이너리·펌웨어 해시 |
| 시뮬레이션 | 모드별 덤프, 매칭 IoU·score 오차·누락·추가 검출 |
| 보드 | 색 패턴, 알려진 위치·클래스, crop·좌표 복원, 입력 목록·실행 로그 |
| 전체 평가 | 동일 GT·이미지·평가기·필터 조건의 지표 |
| 성능 | 워밍업·반복 수·측정 구간, 평균·분위수, 런타임·펌웨어 버전 |

검출 수가 같다는 사실만으로 bbox·score가 같다고 판단하지 않는다.
Novatek cosine 기준은 0.95 이상이지만 전체 검출 매칭과 정확도 평가도 별도로 필요하다.
Sigmastar Float는 `*_concat_postprocess.sim`, Fixed는 `*_concat_postprocess_fixed.sim`,
Offline은 `.img`를 사용한다. `*_cmodel_float.sim`은 Float 입력으로 사용하지 않는다.
SDK `log/output/`은 실행마다 초기화되므로 모드별 덤프를 즉시 따로 보관한다.
Novatek visualizer는 SDK의 `sim_layer/float/` 역양자화 cls·DFL과 manifest의
`test_img_dir`를 읽는다. Golden 최대값으로 scale을 추정하지 않으며 6클래스를 유지한다.
거리·앵커·stride로 decode한 뒤 NMS를 적용해 Golden과 Sim 박스를 비교한다.

### 최근 검증 기록 — 2026-09-23

두 모델 각각 학습 이미지 300장으로 보정하고 별도 Validation 10장으로 검증했다(seed 20260923).
PyTorch ↔ ONNX 및 ONNX profile 간 수치 비교는 통과했다.

| 모델 | Novatek 최소 cls / DFL cosine | Golden → Sim 검출 / IoU≥0.5 매칭 | Sigmastar Float / Fixed / Offline |
|---|---|---|---|
| Corner | 0.9843 / 0.9846 | 21 → 20 / 20 | 21 / 21 / 21 |
| Front | 0.9553 / 0.9405 | 54 → 50 / 49 | 54 / 53 / 53 |

Sigmastar 두 모델은 Fixed/Offline의 bbox·score까지 일치했다.
**Front/Novatek은 DFL cosine 0.95 미달로 양자화 품질 검토가 필요하다.**
Corner도 Golden 검출 1건이 누락됐다. 이 결과를 전체 검출 100% 일치나 보드 배포 승인으로 해석하지 않는다.
컴파일러 추정 시간 Corner 7.53743ms, Front 23.6874ms는 보드 실측 시간이 아니다.

### 변환 환경 재현 — 2026-09-29

truck_dashcam `BSD_F606/Front_Truck` v02 변환(2026-09-09, `best_split.onnx`, 보정 Validation 500장)을
`ubuntu20p04:202407013` base 이미지와 SDK v02.05로 다시 실행해 `deploy.onnx`와 `nvt_model.bin`이
바이트 일치했다. SDK 시뮬레이션 1장만 확인했고 보드 실행은 하지 않았다.
NAS `BSD_F606/Front/`는 car_dashcam 제품으로 이 기록과 다르다.
조건·해시는 [조사 기록](archive/debug/20260929_front_truck_v01_onnx_mismatch.md)을 따른다.

### 과거 기록의 적용 범위

| 기록 | 확인 범위·제한 |
|---|---|
| 2026-09-08 저장된 보드 결과 재평가 | Corner mAP@0.50 0.481, ROI GT 21,742; Front 0.704, ROI GT 50,371. 새 보드 실행 결과가 아님 |
| 과거 Corner 2,500장 float 0.286 / board 0.262, 평균 9.3ms | 이미지 목록·설정·바이너리 해시·측정 조건 미확인. 위 전체 검증셋 결과와 직접 비교 불가 |
| Sigmastar CVD_xyxy mAP@0.45 73.46% | Cartucho 평가 코어의 과거 기록. customEval 지표와 직접 비교 불가 |
| 과거 `run.sh ... 0.10` 변환 2건 | 81/75 검출, 최소 score 0.463965/0.509292. 0.45 미만 없음; 시각화 인자로 모델 임계가 바뀌지 않음을 확인 |
| Novatek v02.05 ↔ v02.06, Corner 1모델·4장 | bin 3,947,956B, 버전 스탬프 3B 차이, cosine 소수 6자리 일치. 다른 모델·보드 성능까지 동일하다는 증거는 아님 |
| 과거 Corner 보드 JSON 점검 | 검출 91,018건, 검출이 있는 파일명 4,955개. 처리 이미지 수나 crop 좌표계의 근거로 사용하지 않음 |

## 미확인 항목과 문제 해결

| 항목 | 필요한 확인·조치 |
|---|---|
| NAS Novatek 테스터의 crop 지원 | 바이너리 해시와 대응 소스 커밋 확보 |
| Front/Novatek 양자화 | 최소 DFL cosine 미달 이미지의 덤프·박스·score 및 전체 평가 |
| Sigmastar RGB/BGR | 알려진 색 패턴으로 실제 입력 버퍼·전처리 확인 |
| Front 클래스 매핑 | Novatek raw 5, Sigmastar board 6을 traffic_light로 정합; 보드 결과 확인 |
| Sigmastar 툴체인/펌웨어 조합 | vQ_0.2.1 변환 + vQ_0.1.7 런타임의 호환 근거와 실물 검증 |
| 출처 불명 `ipu_firmware.bin` | md5 접두 `6b1c8ebe`; SDK 0.1.7/0.2.1 동봉본과 불일치 |
| 과거 ONNX·보드 결과의 출처 | 원본 PT, 이미지 목록, 좌표계, 임계·NMS, 바이너리 해시 확보 |

| 증상 | 확인할 곳 |
|---|---|
| Novatek 라이브러리 로딩 오류 | SDK의 `repo/proprietary/`까지 완전 추출했는지, 실행 비트가 있는지 |
| Novatek GLIBCXX·`libprotobuf.so.17` 오류 | base가 `ubuntu20p04:202407013`인지 확인. 이 이미지의 `/lib/libprotobuf.so.17`을 쓴다 |
| Novatek `libpython3.8.so.1.0` 없음, `No module named 'onnx'` | 순정 Ubuntu base 이미지다. [환경 설정](setup.md#npu-환경-실행)대로 Novatek 이미지를 로드한 뒤 재빌드 |
| Novatek 설정 수정 무효 | 상단 gen_config·sim_config와 nvtai_tool/config 사본 동기화 |
| Novatek 출력 shape 변경 | raw 2출력 ONNX인지, postproc/en=0인지 확인 |
| Novatek 클래스·채널 불일치 | 입력 fmt/channel, 출력 out_fmt, 클래스 수와 실제 텐서명 대조 |
| Novatek 모델 선택 실패 | SD 파일명·ini 선택자·등록된 후처리 이름 |
| Sigmastar 임계 변경 무효 | 시각화 인자, 모델 NMS, 보드 필터를 구분; 수정 후 sync·재변환·재빌드 |
| Sigmastar 라벨 오정합 | `DL_YOLOV8_CV`, BG 슬롯, `gClassNames[]`, Front traffic_light |
| Sigmastar DL 매크로 변경 미반영 | CFLAGS 변경에 따른 configure·의존성 캐시 정리 후 재빌드 |
| Sigmastar prebuilt 누락 | fmf `ncn_cv`의 `3rdparty/sav83x/util/`와 제품 빌드 설정 |
| 평가 급락 | crop·resize·좌표 복원·색공간·필터 조건을 먼저 대조 |

## 버전·소스 근거

2026-09-08 검토에서 사용한 외부 소스 기준이다. 현재 브랜치의 최신 상태나 설치된 바이너리와의
일치를 보증하지 않는다.

| 저장소 / 브랜치 | 검토 커밋 |
|---|---|
| ultralytics_projects | `0f64df558bd91be5769771e13b5a7477010cdcdf` |
| fmf / NT98530 | `898a371df294ebdcb968de9723a20f09381b4361` |
| fmf / ncn_cv | `e373849f06828542fae428a150a8e51990f2ff57` |
| ncn_adas / CVD-H210 | `33081df24e153ed8c544627ae9c0ab0923519d56` |
| ncn_adas_new / master | `cf7139c2eff1597c5378657003892f034d64c8a5` |

| 환경 | 경로·버전 |
|---|---|
| Novatek 변환 SDK | `.env`의 `NOVATEK_SDK_DIR` → 컨테이너 `/opt/npu-sdk`. 이 호스트 추출본 v02.05.2503042(profile VERSION 2052503042). 2026-09-23 기록의 v02.06.2507310은 이 호스트에서 추출본·산출물을 찾지 못했다(2026-09-29) |
| Novatek 변환 이미지 | `ubuntu20p04:202407013`(NAS `03_SDK/novatek/NPU/`) 위에 빌드. 같은 폴더의 `nvt_converter:250818`은 이 이미지에 SDK v02.05를 넣은 사내 이미지 |
| Sigmastar 변환 SDK | `sigmastar/sdk/SGS_IPU_SDK_vQ_0.2.1/`, Python 3.7 환경 |
| Sigmastar 런타임 | `/mnt/blackbox/adas/bin/ipu_firmware_Q_0_1_7.bin`, 과거 확인 md5 접두 `e574bdc7` |
| Novatek 배포 자산 | NAS `/mnt/NAS_2/03_SDK/novatek/NPU/IPU-TEST/` 아래 `260128_sd_backup/`, `260128_firmware/`, `fmf_NCNOVA_patch_260127_*.tar.gz` |
| Sigmastar SDK 문서 | NAS `/mnt/NAS_2/03_SDK/sigmastar/Docs/`, `인수인계/Docs/` |

Novatek 옵션 참고: `Novaic_Convertor_User_Guide_en.pdf` v1.09의 Table 3-2~3-7,
Table 3-12(입출력 허용 조합) 및 해당 SDK의 `Novaic_CNN25_SupportSpec_HW`.
이전 포팅 문서의 classify·LSTM 사례는 [과거 NPU 검증 기록](archive/debug/20260527_phase_f_npu.md)과
[변경 이력](archive/changelog.md)에 보존돼 있으며 현재 지원 모델에 일반화하지 않는다.

보드 구현 확인 지점:

- `fmf/binsrc/adas_ncn/test/vcap_iputest/vcap_iputest.c`: Novatek ini·crop·감도·JSON.
- `ncn_adas_new/3rdparty/adas/ncn/detector/soc/nt98530/cnn_det_nt98530_api.cpp`: 후처리 dispatch.
- 같은 저장소 `detector/soc/funcs/nn_post_process.cpp`, `detector/include/detector_classinfo.h`: 임계·NMS·클래스.
- `fmf/binsrc/adas_ncn/test/iputest/iputest.c`: Sigmastar ini·감도·클래스 문자열.
- `ncn_adas/3rdparty/adas/ncn/detector/soc/sav83x/cnn_det_sav83x_api.cpp`와 include 헤더: 입력·경로·classID +1.
- 같은 저장소 `framework/include/adas_common.h`, `framework/adas_corner/core/corner_view_warning.cpp`: 클래스·임계·WHEEL 사용.
- [평가 유틸리티](../projects/metrics/utils.py): `merge_board_json`, `filter_gt_roi`, `apply_cat_map`.
- [설계 결정 기록](archive/decisions.md): D-020 평가·화각 정합, D-021 exp.yaml 통합.
