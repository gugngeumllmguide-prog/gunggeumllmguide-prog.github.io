import requests
import xml.etree.ElementTree as ET
from google import genai
from google.genai import errors
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

client = genai.Client(api_key="AIzaSyC2Pj9_XiSDvkwYsgtzwCyjh5XBNGCF7Ag")


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
    heritage_keywords = ["뭐", "무엇", "어떤", "역사", "기능", "용도", "의미", "언제", "왕", "공간", "건물", "누가", "지었", "이곳", "전각", "설명"]
    is_heritage_query = any(keyword in query for keyword in heritage_keywords)
    if is_heritage_query:
        return "RAG_REQUIRED"
    return "GENERAL_LLM"


def call_gemini_safely(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except errors.ClientError as e:
        if "429" in str(e):
            return "현재 API 사용량이 초과되었습니다. 잠시 후(약 1분 뒤) 다시 시도해 주세요."
        return f"API 오류가 발생했습니다: {e}"


def generate_rag_response(query, location, kb):
    context = kb.get(location, "창경궁의 주요 전각입니다.")
    prompt = f"당신은 창경궁 가이드입니다. 아래 정보를 바탕으로 질문에 답하세요.\n정보: {context}\n질문: {query}"

    print(f"\n[유형: 문화재 관련 질의 (RAG 적용)]")
    print(f"참조 지식: {context[:]}...")
    result = call_gemini_safely(prompt)
    print(f"응답: {result}")
    return result


def generate_general_response(query):
    prompt = f"당신은 친절한 가이드입니다. 다음 질문에 친근하게 대화하세요.\n질문: {query}"

    print(f"\n[유형: 현장 맥락/자유 질의 (일반 LLM)]")
    result = call_gemini_safely(prompt)
    print(f"응답: {result}")
    return result

def speech_output(llm_response):

    llm_response = llm_response
    config = XttsConfig()
    config.load_json("/path/to/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
    model.cuda()

    outputs = model.synthesize(
        llm_response,
        config,
        speaker_wav="broadcast_00034017.wav",
        gpt_cond_len=3,
        language="ko",
    )
    return outputs


def main():
    kb = fetch_changgyeonggung_data()
    current_location = "창경궁 문정전"

    while True:
        user_input = input("\n질문을 입력하세요 (종료: q): ")
        if user_input.lower() == 'q':
            break

        intent = classify_user_intent(user_input)

        if intent == "RAG_REQUIRED":
            llm_response = generate_rag_response(user_input, current_location, kb)
            speech_output(llm_response)
        else:
            llm_response = generate_general_response(user_input)
            speech_output(llm_response)


if __name__ == "__main__":
    main()