import cv2
import os


def extract_frames():
    """
    Extract frames from videos captured by the 4 cameras
    """
    # Create VideoCapture objects to read from the video files
    cap1 = cv2.VideoCapture('calibration_videos/video1.mp4')
    cap2 = cv2.VideoCapture('calibration_videos/video2.mp4')
    cap3 = cv2.VideoCapture('calibration_videos/video3.mp4')
    cap4 = cv2.VideoCapture('calibration_videos/video4.mp4')

    frame_num = 100

    # Initialize the frame counters for each file
    frame_count1 = 0
    frame_count2 = 0
    frame_count3 = 0
    frame_count4 = 0

    # Loop through the frames of the video files
    while True:
        # Read a frame from each video file
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        # Break the loop if any of the frames couldn't be read
        if not ret1 or not ret2 or not ret3 or not ret4:
            break
        # Increment the frame counters for each file
        frame_count1 += 1
        frame_count2 += 1
        frame_count3 += 1
        frame_count4 += 1

        # If the directory does not exist, create it
        if not os.path.exists("calibration_images"):
            os.makedirs("calibration_images")

        # Check if the counter for each file is a multiple of frame_num
        if frame_count1 % frame_num == 0:
            # Construct the filename for the saved frame
            filename = f"calibration_images/cam_1_frame_{frame_count1 // frame_num}.jpg"
            # Save the frame to disk
            cv2.imwrite(filename, frame1)

        if frame_count2 % frame_num == 0:
            filename = f"calibration_images/cam_2_frame_{frame_count2 // frame_num}.jpg"
            cv2.imwrite(filename, frame2)

        if frame_count3 % frame_num == 0:
            filename = f"calibration_images/cam_3_frame_{frame_count3 // frame_num}.jpg"
            cv2.imwrite(filename, frame3)

        if frame_count4 % frame_num == 0:
            filename = f"calibration_images/cam_4_frame_{frame_count4 // frame_num}.jpg"
            cv2.imwrite(filename, frame4)

    # Release the VideoCapture objects
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
