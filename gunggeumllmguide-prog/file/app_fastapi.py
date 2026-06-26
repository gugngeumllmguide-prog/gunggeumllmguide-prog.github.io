# app_fastapi.py — 사용자 조사용 프로토타입
# 실행: uvicorn app:app --host 0.0.0.0 --port 8000

import requests, time, sys, io, os, json, csv
import xml.etree.ElementTree as ET
from datetime import datetime
from google import genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

os.environ["PYTHONIOENCODING"] = "utf-8"
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ── Gemini ──
client   = genai.Client(api_key="AIzaSyC2Pj9_XiSDvkwYsgtzwCyjh5XBNGCF7Ag")
model_id = "gemini-2.5-flash"

# ── 히스토리 (세션별) ──
sessions = {}   # session_id → { questions: [], responses: [] }

# ── 로그 파일 ──
LOG_FILE = "../user_study_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "session_id", "location",
                         "question", "answer", "type"])



# ── 대화 히스토리 ──
user_question_history = []
response_history      = []

# ── RAG 지식베이스 캐시 ──
CACHE_FILE = "../kb_cache.json"
_kb_cache  = None


# ════════════════════════════════════════
# 1. 데이터 수집 (기존 코드 그대로)
# ════════════════════════════════════════
def fetch_changgyeonggung_data():
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache

    # 캐시 파일이 있으면 로드
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _kb_cache = json.load(f)
        print(f"[KB] 캐시에서 {len(_kb_cache)}개 항목 로드")
        return _kb_cache

    knowledge_base = {}

    # 국가유산청 API
    list_url   = "http://www.khs.go.kr/cha/SearchKindOpenapiList.do"
    detail_url = "http://www.khs.go.kr/cha/SearchKindOpenapiDt.do"
    list_params = {"ccbaMnm1": "창덕궁", "ccbaCtcd": "11", "pageUnit": 30}
    try:
        list_res = requests.get(list_url, params=list_params, timeout=10)
        if list_res.status_code == 200:
            list_root = ET.fromstring(list_res.content)
            for item in list_root.findall('.//item'):
                name = (item.findtext('ccbaMnm1') or "").strip()
                detail_params = {
                    "ccbaKdcd": item.findtext('ccbaKdcd'),
                    "ccbaAsno": item.findtext('ccbaAsno'),
                    "ccbaCtcd": item.findtext('ccbaCtcd'),
                }
                detail_res = requests.get(detail_url, params=detail_params, timeout=10)
                if detail_res.status_code == 200:
                    detail_root = ET.fromstring(detail_res.content)
                    content = detail_root.findtext('.//content')
                    if name and content:
                        knowledge_base[name] = content.strip()
    except Exception as e:
        print(f"[KB] 국가유산청 API 오류: {e}")

    # 궁궐 상세 API
    gung_url = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do?gung_number=2"
    try:
        gung_res = requests.get(gung_url, timeout=10)
        if gung_res.status_code == 200:
            gung_root = ET.fromstring(gung_res.content)
            for item in gung_root.findall('.//list'):
                name = (item.findtext('contents_kor') or "").strip()
                desc = (item.findtext('explanation_kor') or "").strip()
                if name and desc:
                    knowledge_base[name] = desc
    except Exception as e:
        print(f"[KB] 궁궐 API 오류: {e}")

    # 캐시 저장
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    _kb_cache = knowledge_base
    print(f"[KB] {len(knowledge_base)}개 항목 수집 완료")
    return knowledge_base


# ════════════════════════════════════════
# 2. LLM 함수 (기존 코드 그대로)
# ════════════════════════════════════════
def call_gemini_with_retry(prompt, retries=5, delay=2):
    for i in range(retries):
        try:
            res = client.models.generate_content(model=model_id, contents=prompt)
            return res.text.strip()
        except Exception as e:
            print(f"[Gemini] 재시도 {i+1}: {e}")
            time.sleep(delay)
    return "응답에 실패했습니다."


def classify_user_intent(query, kb_topics):
    prev_q = user_question_history[-1] if user_question_history else "없음"
    prev_r = response_history[-1]       if response_history       else "없음"

    prompt = f"""
이전 질문: {prev_q}
이전 답변: {prev_r}
현재 질문: {query}
정보 목록: {list(kb_topics)[:15]}

분류 규칙:
1. 현재 질문이 목록의 장소나 창덕궁 역사에 대한 것이면 'RAG_apply'.
2. 질문에 주어가 없어도 이전 대화가 창덕궁 관련이면서 현재 질문의 내용이 역사와 관련되면 'RAG_apply'.
3. 단순 인사나 창덕궁 관람시간, 창덕궁 경치 등 역사와 무관한 잡담이면 'RAG_except'.
결과를 'RAG_apply' 또는 'RAG_except'로만 출력.
"""
    result = call_gemini_with_retry(prompt)
    return "RAG_apply" if "RAG_apply" in result else "RAG_except"


def generate_rag_response(query, kb):
    context_list = []
    prev_q       = user_question_history[-2] if len(user_question_history) > 1 else ""
    search_query = query + prev_q

    for key, value in kb.items():
        clean_key = key.replace("창덕궁 ", "")
        if clean_key in search_query or key in search_query:
            context_list.append(f"[{key}]: {value}")

    if not context_list and ("안" in query or "어디" in query):
        for key, value in kb.items():
            if any(k in prev_q for k in [key, key.replace("창덕궁 ", "")]):
                context_list.append(f"[{key}]: {value}")

    context         = "\n".join(context_list[:3]) if context_list else "창덕궁 내 주요 전각에 대한 정보입니다."
    history_context = "\n".join(response_history[-2:])

    prompt = f"""
정보: {context}
이전 대화: {history_context}
현재 질문: {query}

당신은 창덕궁 가이드입니다. 제공된 정보를 엄격히 따르되, 이전 대화의 맥락을 이어가며 친절하고 간결하게 핵심만 답변하세요.
"""
    result = call_gemini_with_retry(prompt)
    response_history.append(result)
    return result


def generate_general_response(query):
    history_context = "\n".join(response_history[-2:])
    prompt = f"이전 대화: {history_context}\n질문: {query}\n\n창덕궁 가이드로서 친절하고 간결하며 핵심 내용을 포함하여 대화하세요. 뜬금없는 인사는 하지마세요."
    result = call_gemini_with_retry(prompt)
    response_history.append(result)
    return result


# ════════════════════════════════════════
# 3. FastAPI 앱
# ════════════════════════════════════════
app = FastAPI(title="창덕궁 LLM 가이드 API")

# 브라우저 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# index.html을 루트에서 서빙
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "../0522/index.html"))


# ── 요청/응답 모델 ──
class AskRequest(BaseModel):
    question: str
    location: str = "알 수 없음"   # ArUco 마커로 인식된 장소명

class AskResponse(BaseModel):
    answer: str
    type: str   # "RAG" or "LLM"

class WelcomeRequest(BaseModel):
    location: str   # 마커 감지 시 자동 호출

class WelcomeResponse(BaseModel):
    message: str    # TTS로 읽어줄 환영 메시지


# ── 엔드포인트 1: 마커 감지 → 환영 메시지 ──
@app.post("/welcome", response_model=WelcomeResponse)
def welcome(req: WelcomeRequest):
    kb  = fetch_changgyeonggung_data()
    loc = req.location

    intro_q = f"{loc}은 어떤 역할을 하던 곳이야?"
    user_question_history.append(intro_q)
    ans = generate_rag_response(intro_q, kb)

    message = f"안녕하세요. 현재 위치는 {loc}입니다. {ans}"
    return WelcomeResponse(message=message)


# ── 엔드포인트 2: 사용자 질문 → RAG/LLM 분기 응답 ──
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    kb = fetch_changgyeonggung_data()

    intent = classify_user_intent(req.question, kb.keys())
    user_question_history.append(req.question)

    if intent == "RAG_apply":
        answer = generate_rag_response(req.question, kb)
        type_  = "RAG"
    else:
        answer = generate_general_response(req.question)
        type_  = "LLM"

    return AskResponse(answer=answer, type=type_)


# ── 엔드포인트 3: 대화 히스토리 초기화 (새 마커 스캔 시) ──
@app.post("/reset")
def reset():
    user_question_history.clear()
    response_history.clear()
    return {"status": "ok"}


# ── 서버 시작 시 KB 미리 로드 ──
@app.on_event("startup")
def startup():
    print("[서버] 지식베이스 로딩 중...")
    fetch_changgyeonggung_data()
    print("[서버] 준비 완료 — http://localhost:8000")
