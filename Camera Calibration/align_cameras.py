import cv2


# Initialize the camera captures
cap1 = cv2.VideoCapture('rtsp://admin:kamera_fyzio_01@192.168.1.111:554/cam/realmonitor?channel=1&subtype=0')
cap2 = cv2.VideoCapture('rtsp://admin:kamera_fyzio_02@192.168.1.112:554/cam/realmonitor?channel=1&subtype=0')
cap3 = cv2.VideoCapture('rtsp://admin:kamera_fyzio_03@192.168.1.113:554/cam/realmonitor?channel=1&subtype=0')
cap4 = cv2.VideoCapture('rtsp://admin:kamera_fyzio_04@192.168.1.114:554/cam/realmonitor?channel=1&subtype=0')

# Loop to continuously switch views between cameras every second
while True:
    # Read a frame from each camera
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] / 2), int(frame1.shape[0] / 2)))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] / 2), int(frame2.shape[0] / 2)))
    frame3 = cv2.resize(frame3, (int(frame3.shape[1] / 2), int(frame3.shape[0] / 2)))
    frame4 = cv2.resize(frame4, (int(frame4.shape[1] / 2), int(frame4.shape[0] / 2)))

    frame_height = frame1.shape[0]
    frame_width = frame1.shape[1]
    # Check if all frames were successfully read
    if not (ret1 and ret2):
        break

    # Add a cross to each frame at the center
    center_x = int(frame_width / 2)
    center_y = int(frame_height / 2)

    cross_size = 50
    cross_thickness = 2
    cross_color = (0, 255, 0)
    cv2.line(frame1, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, cross_thickness)
    cv2.line(frame1, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, cross_thickness)
    cv2.line(frame2, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, cross_thickness)
    cv2.line(frame2, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, cross_thickness)
    cv2.line(frame3, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, cross_thickness)
    cv2.line(frame3, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, cross_thickness)
    cv2.line(frame4, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, cross_thickness)
    cv2.line(frame4, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, cross_thickness)

    # Display the frames in a grid
    grid = cv2.vconcat([cv2.hconcat([frame1, frame2]), cv2.hconcat([frame3, frame4])])
    cv2.imshow('Camera grid', grid)
    cv2.waitKey(1)