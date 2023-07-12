import numpy as np


def get_point_alignment(keypoints):
    """
    Returns a correlation score corresponding to the alignment of points.
    :param keypoints:
    :return:
    """
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


def get_distance(point1, point2):
    """
    Returns the distance between two points
    :param point1:
    :param point2:
    :return:
    """
    p1_x, p1_y, p2_x, p2_y = point1.iloc[:, :1], point1.iloc[:, 1:2], point2.iloc[:, :1], point2.iloc[:, 1:2]
    min_len = min(len(pt.index) for pt in (point1, point2))
    distances = []
    for i in range(0, min_len):
        dist = np.sqrt((p2_x.iloc[i][0] - p1_x.iloc[i][0]) ** 2 + (p2_y.iloc[i][0] - p1_y.iloc[i][0]) ** 2)
        distances.append(dist)
    return distances


def get_angle(point1, point2, point3) -> float:
    """
    Calculates angles between three points
    :param point1: First point
    :param point2: Mid point
    :param point3: Last point
    :return: Angle in degrees
    """
    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = point1.iloc[:, :1], point1.iloc[:, 1:2], point2.iloc[:, :1], point2.iloc[:,
                                                                                                      1:2], point3.iloc[
                                                                                                            :,
                                                                                                            :1], point3.iloc[
                                                                                                                 :, 1:2]
    min_len = min(len(pt.index) for pt in (point1, point2, point3))
    angles = []
    for i in range(0, min_len):
        AB = (p2_y.iloc[i][0] - p1_y.iloc[i][0]) / (p2_x.iloc[i][0] - p1_x.iloc[i][0])
        BC = (p3_y.iloc[i][0] - p2_y.iloc[i][0]) / (p3_x.iloc[i][0] - p2_x.iloc[i][0])
        rad = np.arctan2((AB - BC), (1 + AB * BC))
        angle = 180 - np.abs(rad * 180.0 / np.pi)
        angles.append(angle)
    return angles
