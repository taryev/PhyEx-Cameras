import os
import cv2
import csv
import glob
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Function used to draw landmarks on images
    Code came from Mediapipe's code examples
    [https://github.com/googlesamples/mediapipe/blob/cd8753722b4c1052a12e019ededdbdebbbc1a313/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb]
    :param rgb_image: Mediapipe image for drawing landmarks
    :param detection_result: PoseLandmarkerResult object
    :return:
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def write_to_csv(video_path: str, output_path: str):
    """
    Function to get landmarks from a video file in a csv file
    :param video_path: Path of the video to be analyzed
    :param output_path: Paj:th of the csv file
    :return:
    """
    mp_model_path = 'pose_landmarker_full.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_model_path),
        running_mode=VisionRunningMode.VIDEO)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Read the video feed
        video_feed = cv2.VideoCapture(video_path)
        csv_file = open(output_path, mode='w')
        csv_writer = csv.writer(csv_file)
        timestamp = 0

        while True:
            ret, image = video_feed.read()
            if not ret:
                break
            # Convert current frame to Mediapipe's Image object
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # timestamp = int(video_feed.get(cv2.CAP_PROP_POS_MSEC))

            mp_image = mp.Image(
                data=np.array(image),
                image_format=mp.ImageFormat.SRGB
            )
            # Process current image
            detection_result = landmarker.detect_for_video(mp_image, timestamp)
            timestamp = timestamp+1  # Dirty workaround
            pose_landmarks = detection_result.pose_landmarks  # poseLandmarkerResult object
            keypoints = []
            if pose_landmarks is not None:
                # Save to CSV
                for landmarks in pose_landmarks:  # landmarks list (33 landmarks)
                    for landmark in landmarks:  # for each landmark object
                        keypoints.append((round(landmark.x, 4), round(landmark.y, 4), round(landmark.z, 4)))
                    csv_writer.writerow([kp for kps in keypoints for kp in kps])
                # Display the image with landmarks
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                cv2.imshow('Pose Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        video_feed.release()
        cv2.destroyAllWindows()


def get_filename(file_path: str) -> str:
    """
    Get only the filename without path and extension
    :param file_path: File path
    :return: Filename without extension
    """
    file_name: str = os.path.basename(file_path).rsplit('.', 1)[0]
    return file_name


def export(videos_folder: str, csv_path: str):
    """
    Export CSVs of videos in a folder
    :param videos_folder: Path to folder where videos are stored
    :param csv_path: Folder path for storing generated CSVs
    :return:
    """
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    videos = glob.glob(f'{videos_folder}/*.mp4')
    videos.sort()

    for video in videos:
        write_to_csv(video, f'{csv_path}/{get_filename(video)}.csv')
