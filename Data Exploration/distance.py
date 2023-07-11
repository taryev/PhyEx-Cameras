import numpy as np


def get_distance(point1, point2):
    p1_x, p1_y, p2_x, p2_y = point1.iloc[:,:1], point1.iloc[:,1:2], point2.iloc[:,:1], point2.iloc[:,1:2]
    min_len = min(len(pt.index) for pt in (point1,point2))
    distances = []
    for i in range(0,min_len):
        dist = np.sqrt((p2_x.iloc[i][0]-p1_x.iloc[i][0])**2+(p2_y.iloc[i][0]-p1_y.iloc[i][0])**2)
        distances.append(dist)
    return distances


"""
Usage example

import data_handler as datah
from matplotlib import pyplot as plt

op_clmb_col = datah.OpenposeData("/Users/quentinveyrat/Downloads/OpenPoseData/CLMB_1686655244_ID_Coline_cam_3.npy")
mp_clmb_col = datah.MediapipeData("/Users/quentinveyrat/Downloads/CLMB_1686655244_ID_Guitar_cam_3.csv")

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

op_dist_r = get_distance(op_elbow_r, op_wrist_r)
op_dist_l = get_distance(op_elbow_l, op_wrist_l)

op_es_r = get_distance(op_shoulder_r, op_ear_r)
op_es_l = get_distance(op_shoulder_l, op_ear_l)

diff = [abs(a - b) for a, b in zip(op_es_r, op_es_l)]
dmax = max(diff)
diff_per = diff/dmax*100

# import dtw
# print(dtw.dtw_distance(op_dist_l,op_dist_r))
plt.figure()
plt.plot(op_es_r, label="right ear-shoulder")
plt.plot(op_es_l, label="left ear-shoulder")
#plt.plot(diff, label="difference")
plt.plot(diff_per, label="difference%")
plt.legend()
plt.show()
"""