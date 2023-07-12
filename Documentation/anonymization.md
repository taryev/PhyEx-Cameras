# Face anonymization

To blur faces two solutions are available, depending on the motion tracking software used.  
In `anonymization_students.py`, the function `anonymize(video, data_file)` outputs an anonymized video.  
To determine the area to blur, distance between nose and ear is calculated.  
Note that in some exercices or camera views (TBD), blurring is not accurate.

The script `anonymization_students_openpose.py` have the same feature but using OpenPose data instead of Mediapipe.