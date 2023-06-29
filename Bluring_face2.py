import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("C:/Users/Salomé/Desktop/STAGEM1/JPJK_1686575055_ID_Mountain_cam_4.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('C:/Users/Salomé/Desktop/STAGEM1/JPJK_1686575055_ID_Mountain_cam_4.mp4', fourcc, 25, (1920, 1080))

pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    width = 1920
    height = 1080
    img = cv2.resize(img, (width, height))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            # For blur box
            bbox_x = int(bboxC.xmin * iw)
            bbox_y = int(bboxC.ymin * ih)
            old_width = int(bboxC.width * iw)
            old_height = int(bboxC.height * ih)

            # For rectange
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            img[bbox_y:bbox_y + old_height, bbox_x:bbox_x + old_width] = cv2.medianBlur(
                img[bbox_y:bbox_y + old_height, bbox_x:bbox_x + old_width], 75)

            # cv2.rectangle(img, bbox, (255,0,255), 2)

    out.write(img)
    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break