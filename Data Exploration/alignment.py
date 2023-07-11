import numpy as np


def get_point_alignment(keypoints):
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