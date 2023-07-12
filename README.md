# PhyEx Cameras
Scripts and snippets used to operate the cameras on the PhyEx Project.

🔗 Useful links : 
- [Mediapipe documentation](https://developers.google.com/mediapipe/api/solutions/python/mp)
- [Mediapipe/Openpose landmarks](Documentation/landmarks.md)

## Installation
Clone this git repository, then open the PhyEx_Cameras folder in PyCharm.
Before running anything, install all dependencies using pip or the `requirements.txt` file.

## Usage
Each section explains how the code works and gives usage examples.
### Graphic User Interface
🎯 Propose a graphical solution to simplify usage of functionalities  
📖 Documentation : [PhyEx Cameras GUI](Documentation/gui.md)

### Camera calibration
🎯 Get for each of the cameras, intrinsic and extrinsic parameters in order to obtain images with a maximum-reduced distortion.  
📖 Documentation : [Camera calibration](Documentation/calibration.md)

### Motion tracking & coordinates extraction (MediaPipe)
🎯 "Convert" videos to csv files by generating files containing Mediapipe's landmarks coordinates for each video frame.  
📖 Documentation : [Motion tracking](Documentation/motiontracking.md)

### Features extraction
🎯 Extract features from coordinates  
🔌 Currently supported features : angles, distance, parallelism, alignment  
📖 Documentation : [Features extraction](Documentation/features.md)

### Face anonymization
🎯 Blur faces on the videos before uploading to project's server  
📖 Documentation : [Face anon](Documentation/anonymization.md)

## Contributing
Since I am not longer an intern at CTU, the best way to contribute to this project if you don't have permission to commit is to fork this repo.