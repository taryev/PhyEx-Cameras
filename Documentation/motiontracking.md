# Motion tracking

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