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
or
```py
import numpy as np
file_name = "calibration_files/calibration_cam_X.npz"
with np.load(file_name) as data:
    mtx, dist, _, _ = [data[i] for i in ('mtx','dist','rvecs','tvecs')]
```

**Step 4: Undistort**

The function `undistort()` of the script `undistort.py` allows to apply the undistortion method of OpenCV with the camera parameters computed before.

## Motion tracking

**Prerequisites:**  
Download Mediapipe's model bundle in the working directory from [here](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)  

**Step 1: Coordinates extraction**  
Get Mediapipe's landmark coordinates in a CSV file using `video_to_csv.py` and `write_to_csv(video_path, output_path)` function.

**Step 2: 2D plotting**
You can obtain the Matplotlib plot for each frame of the video using the `export_plot(csv_file)` function in `2D_plotting.py`.

**Step 2-Bis: let's make a GIF**  
Using [ffmpeg](https://ffmpeg.org/), we can make a GIF of the plots to recreate the movement.  
Here's an example command that you can use for a 24 fps GIF:  
`ffmpeg -r 24 -i plot_row_%d.png -vf "fps=24,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output.gif`  
Make sure to run this command in the folder where plots are saved.
![YPST GIF](https://cdn.discordapp.com/attachments/907742958135148634/1112729038574862406/output.gif)

## Calculate angles

The script allows to calculate angles while using the id of each interests points you need to describe the function.

For example if you want to calculate the knee angle which is described by the angle between the hip (24), the knee (26) and the ankle (26), 
you just have to use the function with the name of your file and these three numbers : read_angles_csvs('name_of_csv', 24, 26, 28)

You repeat it for the second CSV with the interests points you need, and at the end you will get a plot with two curves from the two CSV.

![Angles](https://i.ibb.co/WpnWjcY/Angles.png)

We are still working on the interface for calculate angles : 'Interface_Calculate_Angles.py',

To display the curve of an angle selected from any CSV, you first need to select the folder you want to work in and then the file you want to study.

Then select the three points of interest that form the angle and press the button to draw the curve.

## Face anonymization

The `anonymization_student.py` script allow the blurring of the subject face.