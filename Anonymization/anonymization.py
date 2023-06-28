import cv2
import os
import numpy as np


# load all npy files in folder

video_input_folder = r'C:\Users\adolfjin\Videos\Originalni'

npy_input_folder = 'Data/npy'
list_of_all_videos = [f for f in os.listdir(npy_input_folder) if f.endswith('.npy')]
blur_kernel_size = (31, 31)
rectangle_size = (80, 80)
width, height = rectangle_size
for filename in list_of_all_videos:

    # Load a video
    video = cv2.VideoCapture(video_input_folder + '\\' + filename[:-4]+'.mp4')
    # get width and height
    width_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get fps of video
    fps = video.get(cv2.CAP_PROP_FPS)

    # create a new video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Data/Output/'+filename[:-4]+'_anony.avi',fourcc, fps, (width_original, height_original))

    # Video frame count
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # read pose data from npy file
    pose_data = np.load(npy_input_folder + '\\' + filename)

    i = 0
    # Read a frame from the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Convert the frame to RGB format
        # iterate i
        i += 1

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run the pose estimation on the frame
        # plot a circle to the frame to location given by the lanmark saved in the npy file
        try:
            x = int(pose_data[i,0,0])
            y = int(pose_data[i,0,1])
            top_left = (x - width // 2, y - height // 2)
            bottom_right = (x + width // 2, y + height // 2)
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
            frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi
          #  cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        except:

            #
            try:
                bottom_right = bottom_right_last
                roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)
                frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi
            except:
                #frame = np.full((height_original, width_original, 3), 255, dtype=np.uint8)
                frame = last_frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
        # create a copy of frame
        bottom_right_last = bottom_right
        last_frame = frame.copy()
    # Release the video and close the CSV file
    video.release()
    cv2.destroyAllWindows()
    out.release()


print('Ok')






