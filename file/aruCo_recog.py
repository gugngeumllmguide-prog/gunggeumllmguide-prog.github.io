import cv2
import cv2.aruco as aruco
import numpy as np


#  설정 및 이미지 로드

IMAGE_PATH = "ArUco Test.png"

overlay_image = cv2.imread(IMAGE_PATH)

if overlay_image is None:
    print(f"❌ '{IMAGE_PATH}' 이미지를 찾을 수 없습니다!")
    print("코드와 같은 폴더에 이미지를 넣고 파일명을 정확히 맞춰주세요. (확장자 .jpg, .png 등 확인)")
    exit()


# ArUco 마커 설정

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)


#  웹캠 켜기

cap = cv2.VideoCapture(0)

print("창덕궁에 오신걸 환영합니다! 카메라에 마커를 보여주세요. (종료: 'q' 키)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 마커 탐지를 위해 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # 화면에 마커가 하나라도 보인다면?
    if ids is not None:
        # 인식된 모든 마커에 대해 사진을 덮어씌웁니다.
        for i in range(len(ids)):
            # 1. 카메라 속 마커의 4개 모서리 좌표 가져오기
            c = corners[i][0]

            # 2. 덮어씌울 사진의 가로, 세로 크기 확인
            h, w, _ = overlay_image.shape

            # 3. 덮어씌울 사진의 원래 4개 모서리 좌표
            pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]])

            # 4. 카메라 속 마커의 4개 모서리 (사진이 들어갈 목표 위치)
            pts_dst = c

            # 5. 수학의 마법 (Homography): 원본 사진을 마커의 기울기에 맞게 찌그러뜨리는 계산법
            matrix, status = cv2.findHomography(pts_src, pts_dst)

            # 6. 계산된 비율대로 사진을 변형 (Warp)
            warped_image = cv2.warpPerspective(overlay_image, matrix, (frame.shape[1], frame.shape[0]))

            # 7. 마커 영역만 도려내기 위한 '마스크(가림막)' 만들기
            mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(pts_dst), 255)

            # 8. 현재 웹캠 화면에서 마커가 있는 부분만 까맣게 파내기
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

            # 9. 변형된 사진에서 마커 크기만큼만 남기고 배경 지우기
            warped_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)

            # 10. 파낸 웹캠 화면(8) + 찌그러진 사진(9) 합체!
            frame = cv2.add(frame_bg, warped_fg)

    # 결과 보여주기
    cv2.imshow("ArUco EX", frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()