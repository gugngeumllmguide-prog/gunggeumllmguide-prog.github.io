import requests, time, os, json, csv
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
import cv2
import numpy as np
import uuid
import base64
from datetime import datetime


os.environ["PYTHONIOENCODING"] = "utf-8"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model_id = "gemini-3.5-flash"


user_question_history = []
response_history = []

CACHE_FILE = "kb_cache.json"
LOCAL_KB_FILE = "local_kb.json"
_kb_cache = None
_local_kb = None

IMAGE_DIR = "/tmp/log_images"
AUDIO_BASE_DIR = "/tmp/AIR_data"

LOG_FILE = os.path.join(AUDIO_BASE_DIR, "user_study_log.csv")

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)





BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_element_kb():
    global _local_kb
    if _local_kb is not None:
        return _local_kb
    if not os.path.exists(LOCAL_KB_FILE):
        _local_kb = []
        return _local_kb
    with open(LOCAL_KB_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        _local_kb = raw
    elif isinstance(raw, dict):
        _local_kb = [
            {
                "name": k,
                "category": "",
                "locations": "",
                "side": "",
                "keywords": [],
                "description": v
            }
            for k, v in raw.items()
        ]
    else:
        _local_kb = []
    return _local_kb


def search_element_kb(query="", location="", element="", top_k=3):
    kb = load_local_element_kb()
    results = []
    search_text = " ".join(filter(None, [query, location, element]))
    for item in kb:
        name = str(item.get("name", ""))
        category = str(item.get("category", ""))
        locations = str(item.get("locations", ""))
        side = str(item.get("side", ""))
        description = str(item.get("description", ""))
        keywords = item.get("keywords", [])
        keywords_text = " ".join(map(str, keywords)) if isinstance(keywords, list) else str(keywords)
        haystack = " ".join([name, category, locations, side, keywords_text, description])
        score = 0
        if location:
            if location == locations:
                score += 10
            elif location in locations or locations in location:
                score += 7
            elif location in haystack:
                score += 4
        if element:
            if element == name:
                score += 10
            if element == category:
                score += 8
            if element in keywords_text:
                score += 7
            if element in name:
                score += 5
            if element in category:
                score += 5
            if element in description:
                score += 2
        for token in search_text.split():
            if len(token) < 2:
                continue
            if token in name:
                score += 3
            if token in category:
                score += 2
            if token in keywords_text:
                score += 2
            if token in description:
                score += 1
        if score > 0:
            results.append((score, item))
    results.sort(key=lambda x: -x[0])
    context = []
    for score, item in results[:top_k]:
        context.append(
            f"[요소:{item.get('name','')} / "
            f"분류:{item.get('category','')} / "
            f"위치:{item.get('locations','')} / "
            f"방향:{item.get('side','')}] "
            f"{item.get('description','')}"
        )
    return context


def fetch_gung_data():
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _kb_cache = json.load(f)
        print(f"[KB] 캐시에서 {len(_kb_cache)}개 항목 로드")
        return _kb_cache

    knowledge_base = {}

    list_url = "http://www.khs.go.kr/cha/SearchKindOpenapiList.do"
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

    element_url = "http://www.khs.go.kr/cha/SearchKindOpenapiList.do"
    keywords_list = ["단청", "기둥", "지붕", "창호", "난간", "현판"]
    for kw in keywords_list:
        try:
            res = requests.get(element_url, params={"ccbaMnm1": kw, "ccbaCtcd": "11", "pageUnit": 10}, timeout=10)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('.//item'):
                    name = (item.findtext('ccbaMnm1') or "").strip()
                    if name and name not in knowledge_base:
                        knowledge_base[name] = name
        except Exception as e:
            print(f"[KB] {kw} 오류: {e}")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    _kb_cache = knowledge_base
    print(f"[KB] {len(knowledge_base)}개 항목 수집 완료")
    return knowledge_base


def search_gung_kb(query="", location="", top_k=3):
    kb = fetch_gung_data()
    results = []
    search_text = " ".join(filter(None, [query, location]))
    for key, value in kb.items():
        score = 0
        clean_key = key.replace("창덕궁 ", "").strip()
        if location:
            if location == key or location == clean_key:
                score += 10
            elif location in key or clean_key in location:
                score += 6
        for token in search_text.split():
            if len(token) < 2:
                continue
            if token in key or token in clean_key:
                score += 3
            if token in value:
                score += 1
        if score > 0:
            results.append((score, key, value))
    results.sort(key=lambda x: -x[0])
    return [f"[{key}]: {value}" for score, key, value in results[:top_k]]


def build_context(query, kb, location="", element=""):
    if element:
        context_list = search_element_kb(query=query, location=location, element=element)
    else:
        context_list = search_gung_kb(query=query, location=location)
    return "\n".join(context_list[:5]) if context_list else "관련 정보를 찾지 못했습니다."


def call_gemini_with_retry(prompt, retries=3, delay=2):
    for i in range(retries):
        try:
            res = client.models.generate_content(model=model_id, contents=prompt)
            return res.text.strip().replace("*", "").replace("#", "")
        except Exception as e:
            print(f"[Gemini] 재시도 {i+1}: {e}")
            time.sleep(delay)
    return "응답에 실패했습니다."


def classify_user_intent(query, kb_topics):
    prev_q = user_question_history[-1] if user_question_history else "없음"
    prev_r = response_history[-1] if response_history else "없음"
    local_kb = load_local_element_kb()
    element_names = [item.get("name", "") for item in local_kb[:10]]
    all_topics = list(kb_topics)[:15] + element_names
    prompt = f"""
이전 질문: {prev_q}
이전 답변: {prev_r}
현재 질문: {query}
정보 목록: {all_topics}
5줄 내로 짧게 설명.
분류 규칙:
1. 현재 질문이 목록의 장소, 창덕궁 역사, 건축 요소(문양, 기둥, 단청 등)에 대한 것이면 'RAG_apply'.
2. 질문에 주어가 없어도 이전 대화가 창덕궁 관련이면서 현재 질문의 내용이 역사·건축과 관련되면 'RAG_apply'.
3. 단순 인사나 창덕궁 관람시간, 창덕궁 경치 등 역사·건축과 무관한 잡담이면 'RAG_except'.
결과를 'RAG_apply' 또는 'RAG_except'로만 출력.

"""
    result = call_gemini_with_retry(prompt)
    result = result.strip("*")
    return "RAG_apply" if "RAG_apply" in result else "RAG_except"


def generate_rag_response(query, kb, location="", element=""):
    context = build_context(query, kb, location=location, element=element)
    history_context = "\n".join(response_history[-2:])
    prompt = f"""
정보: {context}
이전 대화: {history_context}
현재 질문: {query}

당신은 창덕궁 가이드입니다. 제공된 정보를 엄격히 따르되, 이전 대화의 맥락을 이어가며 친절하고 간결하게 핵심만 답변하세요. 5줄 내로 짧게 설명.
"""
    result = call_gemini_with_retry(prompt)
    result = result.strip("*")
    response_history.append(result)
    return result


def generate_rag_response_with_element(location, element, kb):
    context_list = search_element_kb(query="", location=location, element=element)
    context = "\n".join(context_list[:3]) if context_list else (
        f"{location}의 {element}에 대한 직접 자료를 찾지 못했습니다."
    )
    prompt = f"""
정보: {context}
위치: {location}
건축 요소: {element}

당신은 창덕궁 가이드입니다.
현재 방문자가 {location}에서 {element}을 보고 있습니다.
제공된 정보를 바탕으로 {element}에 대해 친절하고 간결하게 설명하세요.
"""
    result = call_gemini_with_retry(prompt)
    result= result.strip("*")
    response_history.append(result)
    return result


def generate_general_response(query):
    history_context = "\n".join(response_history[-2:])
    prompt = f"이전 대화: {history_context}\n질문: {query}\n창덕궁 가이드로서 친절하고 간결하며 핵심 내용을 포함하여 대화하세요. 뜬금없는 인사는 하지마세요."
    result = call_gemini_with_retry(prompt)
    result = result.strip("*")
    response_history.append(result)
    return result


def log(label, text):
    print(f"\n{'='*50}")
    print(f"[{label}] {text}")
    print(f"{'='*50}\n")

def log_question(session_id, location, question, image_path="", session_dir=""):
    log_file = os.path.join(session_dir, "log.csv") if session_dir else LOG_FILE
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "location",
                "user_question", "ai_answer", "type", "image_path", "row_id"
            ])
    row_id = str(uuid.uuid4())[:8]
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id, location, question, "", "", image_path, row_id
        ])
    return row_id, log_file

def log_answer(row_id, answer, type_, log_file=None):
    if log_file is None:
        log_file = LOG_FILE
    rows = []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 8 and row[7] == row_id:
                row[4] = answer
                row[5] = type_
            rows.append(row)
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def log_interaction(session_id, location, question, answer, type_, image_path="", session_dir=""):
    log_file = os.path.join(session_dir, "log.csv") if session_dir else LOG_FILE
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "session_id", "location",
                "user_question", "ai_answer", "type", "image_path", "row_id"
            ])
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id, location, question, answer, type_, image_path, ""
        ])



app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)
@app.after_request
def add_ngrok_header(response):
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

@app.route("/")
def root():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/welcome", methods=["POST"])
def welcome():
    kb = fetch_gung_data()
    loc = request.json.get("location", "알 수 없음")
    session_id = request.json.get("session_id", "unknown")
    session_dir = request.json.get("session_dir", "")
    intro_q = f"{loc}은 어떤 역할을 하던 곳이야?"
    user_question_history.append(intro_q)
    ans = generate_rag_response(intro_q, kb, location=loc)
    message = f"안녕하세요. 현재 위치는 {loc}입니다. {ans}"
    log_interaction(session_id, loc, intro_q, message, "RAG_welcome", session_dir=session_dir)
    log("환영 답변", message)
    return jsonify({"message": message})

@app.route("/element", methods=["POST"])
def element_route():
    kb = fetch_gung_data()
    data = request.json
    location = data.get("location", "알 수 없음")
    session_id = data.get("session_id", "unknown")
    session_dir = data.get("session_dir", "")
    el = data.get("element", "")
    if not el:
        return jsonify({"answer": "건축 요소를 지정해 주세요.", "type": "error"})
    answer = generate_rag_response_with_element(location, el, kb)
    log_interaction(session_id, location, f"[건축요소] {el}", answer, "RAG_element", session_dir=session_dir)
    return jsonify({"answer": answer, "type": "RAG_element"})


@app.route("/ask", methods=["POST"])
def ask():
    kb = fetch_gung_data()
    data = request.json
    question = data.get("question", "").strip("*")
    location = data.get("location", "알 수 없음").strip("*")
    element = data.get("element", "").strip("*")
    session_id = data.get("session_id", "unknown")
    session_dir = data.get("session_dir", "")
    image_b64 = data.get("image", "")

    image_path = ""
    image_description = ""

    if image_b64:
        try:
            img_data = base64.b64decode(image_b64.split(",")[-1])
            save_dir = session_dir if session_dir else IMAGE_DIR
            fname = os.path.join(save_dir, f"img_{datetime.now().strftime('%H%M%S')}.jpg")
            with open(fname, "wb") as f:
                f.write(img_data)
            image_path = fname
            print(f"[이미지] 저장: {fname}")

            import PIL.Image
            import io
            img_pil = PIL.Image.open(io.BytesIO(img_data))
            image_description_result = client.models.generate_content(
                model=model_id,
                contents=[img_pil, "이 사진에서 보이는 건축 요소나 사물을 한 단어 또는 짧은 명사구로만 답하세요. 예: 단청, 현판, 기둥, 연못, 전구 등"]
            )
            image_description = image_description_result.text.strip().replace("*", "").replace("#", "")
            print(f"[이미지 분석] {image_description}")
        except Exception as e:
            print(f"[이미지 저장/분석 오류] {e}")

    if image_description and not element:
        element = image_description

    combined_question = question
    if image_description:
        combined_question = f"{question} (사진 속 요소: {image_description})" if question else f"사진 속 {image_description}에 대해 설명해주세요."

    row_id, log_file = log_question(session_id, location, combined_question, image_path, session_dir)
    intent = classify_user_intent(combined_question, kb.keys())
    user_question_history.append(combined_question)

    if intent == "RAG_apply":
        answer = generate_rag_response(combined_question, kb, location=location, element=element)
        type_ = "RAG"
    else:
        answer = generate_general_response(combined_question)
        type_ = "LLM"

    log_answer(row_id, answer, type_, log_file)
    log(f"AI 답변 [{type_}]", answer)
    return jsonify({"answer": answer, "type": type_})



@app.route("/detect_marker", methods=["POST"])
def detect_marker():
    data = request.json
    image_b64 = data.get("image", "")
    try:
        img_data = base64.b64decode(image_b64.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(img)

        if ids is not None and len(ids) > 0:
            marker_id = int(ids[0][0])
            return jsonify({"marker_id": marker_id})
        else:
            return jsonify({"marker_id": None})
    except Exception as e:
        print(f"[마커 인식 오류] {e}")
        return jsonify({"marker_id": None})


@app.route("/init_session", methods=["POST"])
def init_session():
    data = request.json
    session_id = data.get("session_id", "unknown")
    now = datetime.now()
    date_str = now.strftime("%m%d")
    time_str = now.strftime("%H%M")
    session_dir = os.path.join(AUDIO_BASE_DIR, "record", date_str, f"start{time_str}_{session_id[-4:]}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"[세션] 폴더 생성: {session_dir}")
    return jsonify({"session_dir": session_dir})



import subprocess
@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.json
    audio_b64 = data.get("audio", "")
    mime_type = data.get("mime_type", "audio/webm")
    session_dir = data.get("session_dir", AUDIO_BASE_DIR)
    temp_fname = ""
    temp_mp3 = ""
    try:
        audio_data = base64.b64decode(audio_b64.split(",")[-1])
        ts = datetime.now().strftime("%H%M%S")
        ext = "ogg" if "ogg" in mime_type else "mp4" if "mp4" in mime_type else "webm"
        temp_fname = f"temp_audio_{ts}.{ext}"
        temp_mp3 = f"temp_audio_{ts}.mp3"
        save_fname = os.path.join(session_dir, f"audio_{ts}.mp3")  # 항상 mp3

        with open(temp_fname, "wb") as f:
            f.write(audio_data)

        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_fname, "-ar", "16000", "-ac", "1", temp_mp3],
            check=True, capture_output=True
        )

        uploaded = client.files.upload(file=temp_mp3)
        for _ in range(20):
            file_info = client.files.get(name=uploaded.name)
            state = file_info.state
            state_str = state.name if hasattr(state, "name") else str(state)
            print(f"[STT] 파일 상태: {state_str}")
            if "ACTIVE" in state_str:
                break
            if "FAILED" in state_str:
                raise Exception(f"파일 처리 실패: {state_str}")
            time.sleep(1)

        result = client.models.generate_content(
            model=model_id,
            contents=[uploaded, "이 음성을 한국어로 정확히 텍스트로 변환해주세요. 변환된 텍스트만 출력하세요."]
        )
        os.rename(temp_mp3, save_fname)  # mp3를 최종 저장
        if os.path.exists(temp_fname):
            os.remove(temp_fname)
        text = result.text.strip().replace("*", "").replace("#", "")
        print(f"[STT] 저장: {save_fname}")
        return jsonify({"text": text, "audio_path": save_fname})
    except Exception as e:
        print(f"[STT 오류] {e}")
        for f in [temp_fname, temp_mp3]:
            if f and os.path.exists(f):
                os.remove(f)
        return jsonify({"text": "", "audio_path": ""})


@app.route("/reset", methods=["POST"])
def reset():
    user_question_history.clear()
    response_history.clear()
    print("\n[세션 초기화]\n")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[서버] 지식베이스 로딩 중...")
    fetch_gung_data()
    load_local_element_kb()
    # print("[서버] 준비 완료 — https://carefully-deceptive-unbutton.ngrok-free.dev")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)