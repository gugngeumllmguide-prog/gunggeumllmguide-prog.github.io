import requests
import xml.etree.ElementTree as ET

list_url = "http://www.khs.go.kr/cha/SearchKindOpenapiList.do"
detail_url = "http://www.khs.go.kr/cha/SearchKindOpenapiDt.do"

list_params = {
    "ccbaMnm1": "창경궁",
    "ccbaCtcd": "11",
    "pageUnit": 10
}

try:
    list_response = requests.get(list_url, params=list_params)

    if list_response.status_code == 200:
        list_root = ET.fromstring(list_response.content)
        items = list_root.findall('.//item')

        for item in items:
            kdcd = item.findtext('ccbaKdcd')
            asno = item.findtext('ccbaAsno')
            ctcd = item.findtext('ccbaCtcd')
            name = item.findtext('ccbaMnm1')

            detail_params = {
                "ccbaKdcd": kdcd,
                "ccbaAsno": asno,
                "ccbaCtcd": ctcd
            }
            detail_res = requests.get(detail_url, params=detail_params)

            if detail_res.status_code == 200:
                detail_root = ET.fromstring(detail_res.content)

                asdt = detail_root.findtext('.//ccbaAsdt')
                content = detail_root.findtext('.//content')

                print(f"명칭: {name}")
                print(f"지정일: {asdt if asdt else '정보 없음'}")
                print(f"내용: {content if content else '정보 없음'}")
                print("-" * 50)

    else:
        print(f"에러 발생: {list_response.status_code}")

except Exception as e:
    print(f"오류: {e}")
