# Features extraction
This section details how we managed to extract features from coordinates files.
## Feature list
We created a database of features in collaboration with physiotherapists. These features are detailed in a Excel document.   
Currently, four types of feature are supported :
- Angles : angle mesurement between 3 points
- Distance : distance between 2 points
- Parallelism : parallelism between 2 segments
- Alignment : alignment between n points

🏗 Future:  JSON format may be more appropriate.

## Data handler `data_handler.py`
To enable easy access to data, we created a data handler, it permits to have a similar data structure when using Mediapipe or Openpose coordinates.

### Usages examples
#### Creating a new data instance

```py 
import data_handler
mp = MediapipeData("SMSQ_1686575947_ID_Mountain_cam_4.csv")
op = OpenposeData("BRK4_1686654805_ID_Guitar_cam_4.npy")
```
These constructors return Pandas DataFrames. Learn how to manipulate those objects with [Pandas API Reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

MediapipeData objects allow only csv files with specific format.   
OpenposeData objects allow only npy files with specific format.   
🏗 Future: xlsx support for Mediapipe data.
#### Accessing data
Landmarks name and indexes can be found [here](landmarks.md).  
⚠️ For now, Mediapipe and Openpose landmarks don't have the same name.  
🏗 Future: unify landmark names

```py
# Using landmark name
# dataframe.data['LANDMARK_NAME'][FRAME]
nosex = mp.data['nose_x'][0]
rightbigtoex = op.data['rbigtoe_x'][73]
neckz_1 = mp.data['neck_z'][45]

# Using landmark index
# dataframe.get_by_index(LANDMARK_INDEX).iloc[FRAME,COORDINATE]
# COORDINATE = 0 for x, 1 for y and 2 for z
neckz_2 = mp.get_by_index(33, True).iloc[45,2]
```

## Angles
### Usage examples
```py
import data_handler as dh
import features

mp_1lbr = dh.MediapipeData("1LBR_1686643789_ID_Telescope_cam_4.csv")
mp_rhip = mp_1lbr.data[['right_hip_x', 'right_hip_y']]
mp_rknee = mp_1lbr.data[['right_knee_x', 'right_knee_y']]
mp_rankle = mp_1lbr.data[['right_ankle_x', 'right_ankle_y']]

angle = features.get_angle(mp_rhip, mp_rknee, mp_rankle)
```

## Distance
### Usage examples

```py
from matplotlib import pyplot as plt
import data_handler as dh
import features

op_clmb_col = dh.OpenposeData("/Users/quentinveyrat/Downloads/OpenPoseData/CLMB_1686655244_ID_Coline_cam_3.npy")
mp_clmb_col = dh.MediapipeData("/Users/quentinveyrat/Downloads/CLMB_1686655244_ID_Guitar_cam_3.csv")

op_ear_r = op_clmb_col.data[['rear_x', 'rear_y']]
op_ear_l = op_clmb_col.data[['lear_x', 'lear_y']]
op_shoulder_r = op_clmb_col.data[['rshoulder_x', 'rshoulder_y']]
op_shoulder_l = op_clmb_col.data[['lshoulder_x', 'lshoulder_y']]

op_elbow_l = op_clmb_col.data[['lelbow_x', 'lelbow_y']]
op_elbow_r = op_clmb_col.data[['relbow_x', 'relbow_y']]
op_wrist_r = op_clmb_col.data[['rwrist_x', 'rwrist_y']]
op_wrist_l = op_clmb_col.data[['lwrist_x', 'lwrist_y']]

mp_elbow = mp_clmb_col.data[['right_elbow_x', 'right_elbow_y']]
mp_wrist = mp_clmb_col.data[['right_wrist_x', 'right_wrist_y']]

op_dist_r = features.get_distance(op_elbow_r, op_wrist_r)
op_dist_l = features.get_distance(op_elbow_l, op_wrist_l)

op_es_r = features.get_distance(op_shoulder_r, op_ear_r)
op_es_l = features.get_distance(op_shoulder_l, op_ear_l)

diff = [abs(a - b) for a, b in zip(op_es_r, op_es_l)]
dmax = max(diff)
diff_per = diff / dmax * 100

plt.figure()
plt.plot(op_es_r, label="right ear-shoulder")
plt.plot(op_es_l, label="left ear-shoulder")
plt.plot(diff_per, label="difference%")
plt.legend()
plt.show()
```
## Parallelism
### Usage examples


## Alignment
### Usage examples

```py
import data_handler as dh
import features

op_brgm_emilio = dh.OpenposeData("BRGM_1685003497_ID_9 Emilio_cam_4.npy")
ankle = op_brgm_emilio.data[['rankle_x', 'rankle_y']]
shoulder = op_brgm_emilio.data[['rshoulder_x', 'rshoulder_y']]
hip = op_brgm_emilio.data[['rhip_x', 'rhip_y']]
knee = op_brgm_emilio.data[['rknee_x', 'rknee_y']]

align_op = features.get_point_alignment((ankle, shoulder, hip, knee))

mp_brgm_emilio = dh.MediapipeData("BRGM_1685003497_ID_Rainbow_cam_4.csv")
ankle = mp_brgm_emilio.data[['right_ankle_x', 'right_ankle_y']]
shoulder = mp_brgm_emilio.data[['right_shoulder_x', 'right_shoulder_y']]
hip = mp_brgm_emilio.data[['right_hip_x', 'right_hip_y']]
knee = mp_brgm_emilio.data[['right_knee_x', 'right_knee_y']]

align_mp = features.get_point_alignment((ankle, shoulder, hip, knee))

```