import numpy as np
import data_handler as dh
from matplotlib import pyplot as plt


def calculate_angle(point1, point2, point3) -> float:
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
mp_1lbr = dh.MediapipeData("/Users/quentinveyrat/Downloads/1LBR_1686643789_ID_Telescope_cam_4.csv")

mp_rhip = mp_1lbr.data[['right_hip_x', 'right_hip_y']]
mp_rknee = mp_1lbr.data[['right_knee_x', 'right_knee_y']]
mp_rankle = mp_1lbr.data[['right_ankle_x', 'right_ankle_y']]

ang = calculate_angle(mp_rhip, mp_rknee, mp_rankle)
plt.figure()
plt.plot(ang)
plt.show()
"""
