import pandas as pd
import numpy as np
import data_handler as datah
from matplotlib import pyplot as plt


def get_distance(point1, point2):
    p1_x, p1_y, p2_x, p2_y = point1.iloc[:,:1], point1.iloc[:,1:2], point2.iloc[:,:1], point2.iloc[:,1:2]
    min_len = min(len(pt.index) for pt in (point1,point2))
    distances = []
    for i in range(0,min_len):
        dist = np.sqrt((p2_x.iloc[i][0]-p1_x.iloc[i][0])**2+(p2_y.iloc[i][0]-p1_y.iloc[i][0])**2)
        distances.append(dist)
    return distances


op_clmb_col = datah.OpenposeData("/Users/quentinveyrat/Downloads/OpenPoseData/CLMB_1686655244_ID_Coline_cam_3.npy")
mp_clmb_col = datah.MediapipeData("/Users/quentinveyrat/Downloads/CLMB_1686655244_ID_Guitar_cam_3.csv")

op_elbow = op_clmb_col.data[['relbow_x', 'relbow_y']]
op_wrist = op_clmb_col.data[['rwrist_x', 'rwrist_y']]

mp_elbow = mp_clmb_col.data[['right_elbow_x', 'right_elbow_y']]
mp_wrist = mp_clmb_col.data[['right_wrist_x', 'right_wrist_y']]

op_dist = get_distance(op_elbow, op_wrist)
mp_dist = get_distance(mp_elbow, mp_wrist)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(op_dist)
plt.title("Openpose / Clmb")
plt.subplot(1, 2, 2)
plt.plot(mp_dist)
plt.title("Mediapipe / Clmb")
plt.show()