import cv2
import pandas as pd


def get_blur_radius(distance: int):
    """
    Get the blur radius for a given distance between nose and ear.
    :param distance:
    :return:
    """
    blur_radius = int(distance * 5)
    while blur_radius > 100:
        blur_radius /= 2
    while blur_radius < 15:
        blur_radius *= 2
    blur_radius = int(blur_radius)
    return blur_radius


def anonymize(video: str, csv: str):
    """
    Blur the face of the person doing the exercise.
    Ouputs the new video to .avi format, with the same name and anon suffix.
    :param video: Filepath of the input video
    :param csv: Filepath of the input csv (with landmarks coordinates)
    :return:
    """

    # Input data
    cap = cv2.VideoCapture(video)
    data = pd.read_csv(csv, header=None)
    # Output
    width_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video[:-4] + '_anon.avi', fourcc, fps, (width_original, height_original))

    nose = data.iloc[:, 0:2]
    left_ear = data.iloc[:, 24:26]
    nose_x, nose_y = nose.iloc[0, 0], nose.iloc[0, 1]
    ear_x, ear_y = left_ear.iloc[0, 0], left_ear.iloc[0, 0]
    distance = abs((nose_y - ear_y) / (nose_x - ear_y))
    blur_radius = get_blur_radius(distance)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or (frame_count == data.shape[0]-1):
            break

        frame_count += 1
        frame_height, frame_width, _ = frame.shape
        nose_x, nose_y = int(nose.iloc[frame_count, 0] * frame_width), int(nose.iloc[frame_count, 1] * frame_height)

        face_roi = frame[max(0, nose_y - blur_radius):min(nose_y + blur_radius, frame_height),
                   max(0, nose_x - blur_radius):min(nose_x + blur_radius, frame_width)]

        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 0)

        frame[max(0, nose_y - blur_radius):min(nose_y + blur_radius, frame_height),
        max(0, nose_x - blur_radius):min(nose_x + blur_radius, frame_width)] = blurred_roi

        # cv2.imwrite(f"ano/ano_{frame_count}.jpg", frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    out.release()


'''
Usage Example : 
anonymize("/Users/quentinveyrat/Desktop/NicolasVideos/BRGM_1686573510_ID_Nicolas_cam_3.mp4",
          "/Users/quentinveyrat/Desktop/NicolasCSV/BRGM_1686573510_ID_Nicolas_cam_3.csv")
'''
