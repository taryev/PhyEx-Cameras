# PhyEx Cameras
Scripts and snippets used to operate the cameras on the PhyEx Project

## Camera calibration
The aim of camera calibration is to obtain for each of the cameras, the _intrinsic_ and _extrinsic_ parameters in order to obtain images with a maximum-reduced distortion to proceed to the 3D tracking.

**Step 0: Camera setup**  
A prerequisite is the correct alignment of the cameras.  
To do this, place an object (for example a camera tripod) in the center of the scene and run the `align_cameras.py` script to visualize the cameras feeds. Change camera orientation if needed, all green crosses must point to the same point (the top of the tripod).   
*⚠️ Do not move the cameras after this step.*

**Step 1: Video recording**   
Record simultaneously 4 cameras feeds using PhyEx Recording.  
It is important to rename each video to `video{i}.mp4` with `{i}` the camera number and then move the files to the `calibration_videos` folder.

**Step 2: Frame extraction**  
In order to extract frames from the videos to use them for calibration purposes, you have to use the `extract_frames()` function from the `extraction.py` script.  
This function will output in the `calibration_images` folder images to be used for calibration.

**Step 3: Camera calibration**  
Once the frames have been extracted, we can proceed to the calibration and obtain the camera matrices by running the `get_calibration_files()` function from the `calibration.py` script.
The script will output the calibration files for each camera in the `calibration_files` folder.   

To get cameras parameters from the file, you can use this snippet :
```py
import numpy as np
file_name = "calibration_files/calibration_cam_X.npz"
with np.load(file_name) as data:
  mtx = data["mtx"] # Camera Matrix
  dist = data["dist"] # Distortion coefficients
  rvecs = data["rvecs"] # Rotation vectors
  tvecs = data["tvecs"] # Translation vectors
```

**Step 4: Undistort**

The function `undistort()` of the script `undistort.py` allows to apply the undistortion method of OpenCV with the camera parameters computed before.

## Motion tracking
*Todo*  

**Prerequisites:**
- Download Mediapipe's model bundle in the working directory from [here](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)  

