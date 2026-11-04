# 배포 가이드 (Deployment Guide)

- **프로젝트명**: DAIR-YOLO
- **기술 스택**: Python, PyTorch 2.5, ultralytics, DAIR 커스텀 모델, Novatek NPU
- **최초 작성**: 2026-04-28 (KST)

> 개발 중 환경 세팅 정보를 여기에 누적한다. 최종 완료 후 정리한다.

---

## 환경 요구사항

| 항목 | 버전/사양 |
|------|---------|
| OS | Linux 5.15 (Ubuntu) |
| GPU | 2x RTX 4090 24GB |
| CUDA | 12.2 (cu121) |
| Python | 3.x |
| PyTorch | 2.5.1+cu121 |
| RAM | 125GB (workers<=4 필수) |

---

## 환경 변수

```env
# 필수
# (현재 별도 환경변수 없음, config yaml로 관리)
```

---

## 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt
pip install -e .

# 2. 학습 실행 (예: Rear 모델)
python tools/train_custom_trainer.py --cfg projects/exp_yaml/baseline/rear_dair.yaml

# 3. 평가
python tools/validate_custom.py --weights runs/detect/rear/weights/best.pt

# 4. ONNX 변환
python tools/export_custom.py --weights runs/detect/rear/weights/best.pt
```

---

## Novatek NPU 변환

```bash
# novatek/convert/ 내 wrapper 스크립트 사용 (인라인 금지)
cd novatek/convert/rear_260401
bash run_compiler.sh
```

---

## 개발 중 환경 세팅 메모

> 개발하면서 발견한 환경 관련 주의사항을 여기에 기록한다.

- 2026-04-28: 초기 세팅
- workers > 4 시 RAM 73GB 초과 → workers=4 제한
- Docker 내 root git 작업 후 호스트 push 시 `.git/logs/` Permission denied → `sudo chown -R $(whoami) .git/logs/`
