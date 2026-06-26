import requests
import xml.etree.ElementTree as ET
from google import genai
from google.genai import errors

client = genai.Client(api_key="AIzaSyC2Pj9_XiSDvkwYsgtzwCyjh5XBNGCF7Ag")
model = "gemini-2.5-flash"

knowledge_base = {}
def fetch_changgyeonggung_data():
    knowledge_base = {}
    url = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do?gung_number=3"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall('.//list'):
                name = item.findtext('contents_kor')
                desc = item.findtext('explanation_kor')
                if name and desc:
                    knowledge_base[name.strip()] = desc.strip()
    except Exception:
        pass
    return knowledge_base

def classify_user_intent(query):
    prompt = f"질문: {query}\n유형 분류 기준: 창경궁 정보 질문이면 'RAG_apply', 관람 시간 등 일반 질문이면 'RAG_except'. 결과만 출력하시오."
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return

def call_gemini_safely(prompt):
    try:
        response = client.models.generate_content(
            model= model,
            contents=prompt
        )
        return response.text
    except errors.ClientError as e:
        if "429" in str(e):
            return "API 사용량 초과입니다."
        return f"에러: {e}"

def generate_rag_response(query, location, kb):
    context = kb.get(location, "해당 장소는 창경궁의 주요 건물입니다.")
    prompt = f"정보: {context}\n질문: {query}\n위 정보를 엄격하게 유지하면서 정확한 정보만으로 간결하게 답하세요. 답을 끝낸 후 추가 질문이 있는지 물어보세요."

    print(f"\n[유형: 문화재 관련 질의 (RAG 적용)]")
    print(f"참조 지식: {context[:]}...")
    result = call_gemini_safely(prompt)
    print(f"응답: {result}")
    return result

def generate_general_response(query):
    prompt = f"질문: {query}\n친절한 가이드로서 대답하되, 간결하게 설명하세요. 답을 끝낸 후 추가 질문이 있는지 물어보세요."

    print(f"\n[유형: 현장 맥락/자유 질의 (일반 LLM)]")
    result = call_gemini_safely(prompt)
    print(f"응답: {result}")
    return result

def main():
    kb = fetch_changgyeonggung_data()
    current_location = "창경궁"

    while True:
        user_input = input("\n질문을 입력하세요 (종료: q): ")
        if user_input.lower() == 'q':
            break

        intent = classify_user_intent(user_input)

        if "RAG_apply" in intent:
            generate_rag_response(user_input, current_location, kb)
        if "RAG_except" in intent:
            generate_general_response(user_input)

if __name__ == "__main__":
    main()