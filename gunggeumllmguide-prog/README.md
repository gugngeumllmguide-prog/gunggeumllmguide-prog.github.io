# 창경궁 AI 가이드 — 프로토타입 실행 가이드

## 📁 파일 구성

```
프로젝트 폴더/
├── app.py          ← FastAPI 서버 (이 파일)
├── index.html      ← 브라우저 프론트엔드
├── character.png   ← 캐릭터 이미지 (직접 추가)
└── kb_cache.json   ← 자동 생성됨 (RAG 데이터 캐시)
```

---

## ⚙️ 설치

```bash
pip install fastapi uvicorn python-multipart
# 기존 패키지 (이미 있음): requests, google-genai
```

---

## 🚀 실행

```bash
# 프로젝트 폴더에서 실행
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 → **http://localhost:8000** 열기

---

## 📱 핸드폰에서 접속 (같은 와이파이)

1. PC의 IP 주소 확인
   - Windows: `ipconfig` → IPv4 주소 (예: 192.168.0.10)
2. `index.html` 상단의 SERVER 주소 변경
   ```javascript
   const SERVER = "http://192.168.0.10:8000";  // PC IP로 변경
   ```
3. 핸드폰 브라우저에서 `http://192.168.0.10:8000` 접속

---

## 🏷️ ArUco 마커 출력

https://chev.me/arucogen/ 에서 생성
- Dictionary: **4x4** 선택
- Marker ID: **24, 25, 26, 27, 28**
- A4 출력 후 각 장소에 부착

---

## 🗺️ 마커 ID ↔ 장소 매핑

| ID | 장소 |
|----|------|
| 24 | 문정전 |
| 25 | 환경전 |
| 26 | 홍화문 |
| 27 | 통명전 |
| 28 | 춘당지 |

---

## 🎮 시연 팁

- 📌 버튼 (우하단): 카메라 없이 장소 직접 선택 가능 → 발표 시연용
- 🎤 버튼: 음성 입력 (Chrome 권장)
- 텍스트 입력: 키보드로도 질문 가능
- 배지: RAG 응답(파란색) vs LLM 응답(초록색) 구분 표시

---

## ⚠️ 주의사항

- **Chrome** 브라우저 사용 권장 (STT/TTS 지원)
- 카메라 권한 허용 필요
- 서버(app.py)가 실행 중이어야 답변 가능
- API 키는 `app.py` 상단에서 본인 키로 교체
