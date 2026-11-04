# 하네스 수정 레포트

- **프로젝트**: DAIR-YOLO
- **점검일**: 2026-04-29 (KST)
- **판정**: ⚠️ 경미 (핵심 hook은 동작하나 행동 교정·외부 툴체인 가드 미흡)

---

## 발견된 문제 요약

| # | 문제 | 심각도 | 유형 |
|---|------|--------|------|
| 1 | `tasks/lessons.md` 자동 갱신 메커니즘 부재 (UserPromptSubmit hook 0개) | ❌ High | 템플릿 구조 |
| 2 | 외부 SDK·컴파일러 흐름 사전 숙지 가드 누락 (NPU 변환에서 onnx2novaonnx 단계 건너뜀 사고) | ⚠️ Medium | 템플릿 구조 + 일회성 |
| 3 | `runs/` 자동증가 디렉토리명 vs 학습 모델 의미 불일치 (`train`=Rear, `train-2`=Front) | ⚠️ Low | 일회성 |
| 4 | `PostToolUse hook ran` 로그 — Write/Bash 매처에 의한 정상 동작 (의도됨) | ℹ️ Info | 정상 |

---

## Upstream 수정 대상 (claude_skills 레포지토리)

> 아래 두 항목은 템플릿 구조에서 비롯된 것으로, claude_skills 레포의 `project-init` plugin에서 수정해야 한다.

### [문제 #1] UserPromptSubmit hook 신설 — 사용자 교정 자동 감지

**근본 원인**: CLAUDE.md `사용자 교정 발생 시 tasks/lessons.md 업데이트` 자연어 규칙은 LLM 우선순위에서 누락된다. Claude Code에서 자동화는 hook으로 강제해야 한다.

**영향 파일 (claude_skills 레포 기준)**

신설:
- `plugins/project-init/hooks/detect_user_correction.py`

수정:
- `plugins/project-init/hooks/hooks.json`

**신설 스크립트** (`detect_user_correction.py`)
```python
#!/usr/bin/env python3
"""UserPromptSubmit hook — 한국어 교정 신호 감지 시 메인 Claude에 alert.
직접 lessons.md를 Edit하지 않는다 (4컬럼 패턴 추출은 LLM 판단 필요).
"""
import json, re, sys
from pathlib import Path

STRONG = [r"아니야", r"아니지", r"틀렸", r"잘못(됐|했|된)", r"그게\s*아니",
          r"하지\s*마", r"하면\s*안\s*돼", r"다시\s*해"]
MEDIUM = [r"말고", r"대신", r"~?로\s*바꿔", r"왜\s+\S+했", r"아까\s+왜",
          r"방금\s+왜", r"빠뜨렸", r"안\s*날라", r"\bno,\b", r"\bwrong\b",
          r"\bdon'?t\b", r"\binstead\b"]

def detect(text):
    for p in STRONG:
        if re.search(p, text, re.IGNORECASE): return "STRONG"
    for p in MEDIUM:
        if re.search(p, text, re.IGNORECASE): return "MEDIUM"
    return None

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    prompt = data.get("prompt", "")
    if not prompt: sys.exit(0)
    level = detect(prompt)
    if not level: sys.exit(0)
    cwd = Path.cwd()
    lessons = None
    for p in [cwd] + list(cwd.parents):
        cand = p / "tasks" / "lessons.md"
        if cand.exists():
            lessons = cand
            break
    if lessons is None: sys.exit(0)
    div = "━" * 57
    print("\n".join([
        "", div,
        f"[USER CORRECTION DETECTED — {level}]", div, "",
        "이번 사용자 메시지에 교정 신호가 감지되었다.",
        "다음 응답 시작 전 반드시:",
        f"  1. {lessons} '행동 교정 패턴' 표에 항목 추가",
        "     컬럼: 날짜(KST) | 상황 | 잘못한 접근 | 올바른 접근",
        "  2. 동일 실수 재발 방지 규칙 1줄 도출",
        "  3. 그 후 사용자 요청에 응답",
        div, "",
    ]))

if __name__ == "__main__":
    main()
```

**hooks.json 수정** — `hooks` 객체에 `UserPromptSubmit` 키 추가

현재:
```json
{
  "description": "...",
  "hooks": {
    "PostToolUse": [
      { "matcher": "Write", "hooks": [...] },
      { "matcher": "Bash", "hooks": [...] }
    ]
  }
}
```

수정 후:
```json
{
  "description": "project-init plugin hooks — verifier 결과 감지 + 사용자 교정 감지",
  "hooks": {
    "PostToolUse": [
      { "matcher": "Write", "hooks": [...] },
      { "matcher": "Bash", "hooks": [...] }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/detect_user_correction.py",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

**수정 이유**: hook이 사용자 입력에서 한국어 교정 패턴을 감지하면 alert를 stdout으로 출력 → Claude Code가 이를 다음 turn 컨텍스트에 주입 → 메인 Claude가 lessons.md 갱신을 우선 수행. 직접 Edit이 아닌 alert 방식인 이유는 4컬럼 패턴 추출은 LLM 판단이 필요해서 자동 Edit 시 잡음 항목 누적·신뢰도 하락 위험.

---

### [문제 #2] 템플릿 CLAUDE.md `## 개발 워크플로우` — 외부 SDK 가드 항목 추가

**영향 파일 (claude_skills 레포 기준)**
- `plugins/project-init/skills/project-init/assets/templates/CLAUDE.md`

**현재** (`## 개발 워크플로우` 섹션, "버그 보고" 줄과 "사용자 교정" 줄 사이):
```markdown
- 버그 보고 받으면 묻지 않고 바로 수정 — CI 실패도 능동적으로 처리
- 사용자 교정 발생 시 `tasks/lessons.md` 업데이트 (세션 시작 시 복습)
```

**수정 후**:
```markdown
- 버그 보고 받으면 묻지 않고 바로 수정 — CI 실패도 능동적으로 처리
- 외부 SDK·컴파일러·변환 툴체인 작업 전: 공식 워크플로우 전체 단계를 먼저 나열·확인
  - 단계 누락 사고 방지 — 중간 변환·서명·검증 단계가 묵음 생략되기 쉬움
  - wrapper 스크립트가 있으면 그것만 사용 (인라인 명령 금지)
  - 모르는 단계는 추측하지 말고 사용자에게 확인
- 사용자 교정 발생 시 `tasks/lessons.md` 업데이트 (세션 시작 시 복습)
```

**수정 이유**: 이번 세션 NPU 변환 사고처럼 외부 SDK는 사전·사후 변환·서명 단계가 묵음 생략되기 쉽다. 자연어 가드 1줄로 사고 패턴 자체를 명시화.

---

## 이 프로젝트 내 즉시 조치 사항

> Upstream 반영을 기다리지 않고 지금 적용해야 할 것들

- [ ] `/workspace/tasks/lessons.md` "행동 교정 패턴" 표에 이번 세션 두 사례 수동 기록
  - **NPU 변환 흐름 미숙지**: 잘못 → ultralytics ONNX를 deploy_float.onnx로 단순 rename 후 컴파일러 실행 / 올바름 → 작업 전 wrapper 스크립트와 변환 단계(onnx2novaonnx) 먼저 확인
  - **기존 데이터 보호**: 잘못 → 새 디렉토리 작업 시 정리 명령을 사용자 승인 없이 실행하려 함 / 올바름 → `rm -rf` 등 파괴 명령 직전에 영향 범위 명시 + 사용자 승인
- [ ] `/workspace/tasks/lessons.md` "하네스 관련 패턴" 표에 이번 점검 결과 기록
- [ ] `/workspace/CLAUDE.md` `## 개발 워크플로우` 에 외부 SDK 사전 확인 항목 추가 (Upstream과 동일 내용 + Novatek 특화 한 줄)
- [ ] `/workspace/CLAUDE.md` `## 알려진 버그 / 제약` 의 Novatek NPU 항목 보강:
  - 추가 문구: `ultralytics export ONNX → onnx2novaonnx로 nova ONNX 변환 → compiler.nvtai 입력. raw export ONNX를 그대로 rename해서 컴파일러에 넣지 말 것.`
- [ ] (선택) `/workspace/CLAUDE.md` `## 명령어` 의 학습 명령에 `name=<Rear|Front|...>` 명시 권장 주석 추가 (자동증가 디렉토리 의미 불일치 방지)

---

## 검토되었으나 채택하지 않은 옵션

- **PreToolUse hook으로 `compiler.nvtai` 명령 차단·확인**: 강력하지만 wrapper 스크립트가 이미 존재 + Novatek 특화 → 일반화된 plugin에 넣기엔 과함. CLAUDE.md 자연어 가드로 충분.
- **detect_user_correction.py가 lessons.md를 직접 Append**: 잡음 항목 누적·신뢰도 하락 위험. alert 방식이 안전.
