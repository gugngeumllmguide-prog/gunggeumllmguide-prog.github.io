import requests
import xml.etree.ElementTree as ET


changyunggung_list_url = "https://www.heritage.go.kr/heri/gungDetail/gogungListOpenApi.do?gung_number=3"

try:

    response = requests.get(changyunggung_list_url)


    if response.status_code == 200:

        root = ET.fromstring(response.content)


        for item in root.findall('.//list'):
            gung_name = item.findtext('gung_name')
            serial_number = item.findtext('serial_number')
            contents_kor = item.findtext('contents_kor')
            explanation_kor = item.findtext('explanation_kor')

            print(f"순번 : {serial_number}")
            print(f"문화재 명칭 : {contents_kor}")
            print(f"설명 : {explanation_kor}")
            print("-" * 20)

    else:
        print(f"데이터 로드 불가 [ {response.status_code} ] ")

except Exception as e:
    print(f"[ error ] {e}")