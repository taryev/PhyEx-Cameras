import numpy as np

def get_point_alignment(keypoints):
    '''
    Returns a correlation score corresponding to the alignment of points.
    :param keypoints:
    :return:
    '''
    cors = []
    min_len = min(len(pt.index) for pt in keypoints)
    for i in range(min_len):
        x = []
        y = []
        for keypoint in keypoints:
            point = keypoint.iloc[i, 0:2]
            x.append(point.iloc[0])
            y.append(point.iloc[1])
        cor = np.corrcoef(x, y)
        cors.append(cor[0, 1])

    return cors
"""
Usage examples

import data_handler as datah
from matplotlib import pyplot as plt

op_brgm_emilio = datah.OpenposeData("/Users/quentinveyrat/Downloads/OpenPoseData/BRGM_1685003497_ID_9 Emilio_cam_4.npy")
ankle = op_brgm_emilio.data[['rankle_x', 'rankle_y']]
shoulder = op_brgm_emilio.data[['rshoulder_x', 'rshoulder_y']]
hip = op_brgm_emilio.data[['rhip_x', 'rhip_y']]
knee = op_brgm_emilio.data[['rknee_x', 'rknee_y']]

align = get_point_alignment((ankle,shoulder,hip,knee))

plt.title("OP BRGM EMILIO")
plt.plot(align)
plt.show()

mp_brgm_emilio = datah.MediapipeData("/Users/quentinveyrat/Downloads/BRGM_1685003497_ID_Rainbow_cam_4.csv")
ankle = mp_brgm_emilio.data[['right_ankle_x', 'right_ankle_y']]
shoulder = mp_brgm_emilio.data[['right_shoulder_x', 'right_shoulder_y']]
hip = mp_brgm_emilio.data[['right_hip_x','right_hip_y']]
knee = mp_brgm_emilio.data[['right_knee_x', 'right_knee_y']]

align = get_point_alignment((ankle,shoulder,hip,knee))

plt.title("MP BRGM EMILIO")
plt.plot(align)
plt.show()
"""

def get_distance(point1, point2):
    '''
    Returns the distance between two points
    :param point1:
    :param point2:
    :return:
    '''
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

def get_angle(point1, point2, point3) -> float:
    '''
    Calculates angles between three points
    :param point1: First point
    :param point2: Mid point
    :param point3: Last point
    :return: Angle in degrees
    '''
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = point1.iloc[:, :1], point1.iloc[:, 1:2], point2.iloc[:, :1], point2.iloc[:, 1:2], point3.iloc[:, :1], point3.iloc[:, 1:2]
    min_len = min(len(pt.index) for pt in (point1, point2, point3))
    angles = []
    for i in range(0, min_len):
        AB = (p2_y.iloc[i][0] - p1_y.iloc[i][0]) / (p2_x.iloc[i][0] - p1_x.iloc[i][0])
        BC = (p3_y.iloc[i][0] - p2_y.iloc[i][0]) / (p3_x.iloc[i][0] - p2_x.iloc[i][0])
        rad = np.arctan2((AB - BC), (1 + AB * BC))
        angle = 180 - np.abs(rad * 180.0 / np.pi)
        angles.append(angle)
    return angles
"""
Usage example

import data_handler as dh
from matplotlib import pyplot as plt
mp_1lbr = dh.MediapipeData("/Users/quentinveyrat/Downloads/1LBR_1686643789_ID_Telescope_cam_4.csv")

mp_rhip = mp_1lbr.data[['right_hip_x', 'right_hip_y']]
mp_rknee = mp_1lbr.data[['right_knee_x', 'right_knee_y']]
mp_rankle = mp_1lbr.data[['right_ankle_x', 'right_ankle_y']]

ang = calculate_angle(mp_rhip, mp_rknee, mp_rankle)
plt.figure()
plt.plot(ang)
plt.show()
"""