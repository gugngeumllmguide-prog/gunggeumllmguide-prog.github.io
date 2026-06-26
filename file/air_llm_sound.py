import requests
import xml.etree.ElementTree as ET
from google import genai
import time, sys, io, os
import scipy.io.wavfile as wavfile
import sounddevice as sd
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from torch.serialization import add_safe_globals
from pydub import AudioSegment
import coqpit.coqpit

add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

os.environ["COQUI_TOS_AGREED"] = "1"
AudioSegment.converter = r"C:\Users\user\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\user\ffmpeg\bin\ffprobe.exe"

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
client = genai.Client(api_key="AIzaSyC2Pj9_XiSDvkwYsgtzwCyjh5XBNGCF7Ag")
model_id = "gemini-2.5-flash"
CACHE_FILE = "../kb_cache.json"

user_question_history = []
response_history = []
now_position = ""




class LLM_call(now_position):
    def __init__(self):
        self.now_position = now_position
        self.original_deserialize = coqpit.coqpit._deserialize
        self.rag_data = None


    def fetch_changgyeonggung_data(self):
        knowledge_base = {}
        list_url = "http://www.khs.go.kr/cha/SearchKindOpenapiList.do"
        detail_url = "http://www.khs.go.kr/cha/SearchKindOpenapiDt.do"
        list_params = {"ccbaMnm1": "창경궁", "ccbaCtcd": "11", "pageUnit": 30}

        try:
            list_res = requests.get(list_url, params=list_params)
            if list_res.status_code == 200:
                list_root = ET.fromstring(list_res.content)
                for item in list_root.findall('.//item'):
                    name = item.findtext('ccbaMnm1').strip()
                    detail_params = {
                        "ccbaKdcd": item.findtext('ccbaKdcd'),
                        "ccbaAsno": item.findtext('ccbaAsno'),
                        "ccbaCtcd": item.findtext('ccbaCtcd')
                    }
                    detail_res = requests.get(detail_url, params=detail_params)
                    if detail_res.status_code == 200:
                        detail_root = ET.fromstring(detail_res.content)
                        content = detail_root.findtext('.//content')
                        if name and content:
                            knowledge_base[name] = content.strip()
        except Exception:
            pass

        gung_url = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do?gung_number=3"
        try:
            gung_res = requests.get(gung_url)
            if gung_res.status_code == 200:
                gung_root = ET.fromstring(gung_res.content)
                for item in gung_root.findall('.//list'):
                    name = item.findtext('contents_kor').strip()
                    desc = item.findtext('explanation_kor').strip()
                    if name and desc:
                        knowledge_base[name] = desc
        except Exception:
            pass

        return knowledge_base

    def classify_user_intent(self, query, kb_topics):
        prev_q = user_question_history[-1] if user_question_history else "없음"
        prev_r = response_history[-1] if response_history else "없음"

        prompt = f"""
        이전 질문: {prev_q}
        이전 답변: {prev_r}
        현재 질문: {query}
        정보 목록: {list(kb_topics)[:15]}
    
        분류 규칙:
        1. 현재 질문이 목록의 장소나 창경궁 역사에 대한 것이면 'RAG_apply'.
        2. 질문에 주어가 없어도 이전 대화가 창경궁 관련이면서 현재 질문의 내용이 역사와 관련되면 'RAG_apply'.
        3. 단순 인사나 창경궁 관람시간, 창경궁 경치 등 역사와 무관한 잡담이면 'RAG_except'.
        결과를 'RAG_apply' 또는 'RAG_except'로만 출력.
        """
        result = self.call_gemini_with_retry(prompt)
        return "RAG_apply" if "RAG_apply" in result else "RAG_except"


    def call_gemini_with_retry(self, prompt, retries=5, delay=2):
        for i in range(retries):
            try:
                res = client.models.generate_content(model=model_id, contents=prompt)
                return res.text.strip()
            except Exception:
                time.sleep(delay)
        return "응답에 실패했습니다."





    def generate_rag_response(self, query, kb):
        context_list = []
        prev_q = user_question_history[-2] if len(user_question_history) > 1 else ""
        search_query = query + prev_q

        for key, value in kb.items():
            clean_key = key.replace("창경궁 ", "")
            if clean_key in search_query or key in search_query:
                context_list.append(f"[{key}]: {value}")

        if not context_list and "안" in query or "어디" in query:
            for key, value in kb.items():
                if any(k in prev_q for k in [key, key.replace("창경궁 ", "")]):
                    context_list.append(f"[{key}]: {value}")

        context = "\n".join(context_list[:3]) if context_list else "창경궁 내 주요 전각에 대한 정보입니다."
        history_context = "\n".join(response_history[-2:])

        prompt = f"""
        정보: {context}
        이전 대화: {history_context}
        현재 질문: {query}
    
        당신은 창경궁 가이드입니다. 제공된 정보를 엄격히 따르되, 이전 대화의 맥락을 이어가며 친절하고 간결하게 핵심만 답변하세요.
        """
        result = self.call_gemini_with_retry(prompt)
        print(f"\n[RAG 응답]\n응답: {result}")
        response_history.append(result)
        return result


    def generate_general_response(self, query):
        history_context = "\n".join(response_history[-2:])
        prompt = f"이전 대화: {history_context}\n질문: {query}\n\n창경궁 가이드로서 친절하고 간결하며 핵심 내용을 포함하여 대화하세요. 뜬금없는 인사는 하지마세요."
        result = self.call_gemini_with_retry(prompt)
        print(f"\n[일반 응답]\n응답: {result}")
        response_history.append(result)
        return result



    def patched_deserialize(self,value, field_type):
        try:

            return self.original_deserialize(value, field_type)
        except TypeError:
            return value

    coqpit.coqpit._deserialize = patched_deserialize




    def LLM_main(self):
        kb = self.fetch_changgyeonggung_data()
        speech_synth = Speech_synth()
        print("시스템 준비 완료")

        loc = self.now_position
        q = f"{loc}은 어떤 역할을 하던 곳이야?"

        user_question_history.append(q)

        ans = self.generate_rag_response(q, kb)
        speech_synth.speech_output(f"안녕하세요. 현재 위치는 {loc}입니다. {ans}")

        while True:
            u_input = input("\n질문(종료:q): ")
            if u_input.lower() == 'q': break
            intent = self.classify_user_intent(u_input, kb.keys())
            user_question_history.append(u_input)
            ans = self.generate_rag_response(u_input, kb) if intent == "RAG_apply" else self.generate_general_response(u_input)
            speech_synth.speech_output(ans)


class Speech_synth():
    def __init__(self):
        self.tts = TTS()

    def init_tts(self):

        if self.tts is None:
            from TTS.api import TTS
            print("XTTS 모델 로딩 중")
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            self.tts.to(device)
        return self.tts

    def speech_output(self, text):
        tts = self.init_tts()
        output_dir = "../output_wav"
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        file_path = os.path.join(output_dir, "output.wav")
        tts.tts_to_file(text=text, file_path=file_path, language="ko", speaker_wav="117_117_003_0019.wav", speed=1.0)

        sr, data = wavfile.read(file_path)
        sd.play(data, sr)
        sd.wait()

if __name__ == "__main__":
    now_position = "문정전"
    LLM_call = LLM_call(now_position)
    LLM_call.LLM_main()