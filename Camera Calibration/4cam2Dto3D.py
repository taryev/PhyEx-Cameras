import pandas as pd
import numpy as np
import cv2
import csv
import glob

def calculate_3D(x1, x2, x3, x4, y1, y2, y3, y4):
    def load_calib(file):
        with np.load(file) as X:
            mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
            return mtx, dist, rvecs[0], tvecs[0]

    def get_projection_matrix(mtx, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvecs)
        RT = np.column_stack((R, tvecs))
        P = np.matmul(mtx, RT)
        return P

    # Load calibration parameters and calculate projection matrices for all cameras
    P_mats = []
    for i in range(1, 5):  # adjust this range according to your number of cameras
        mtx, dist, rvecs, tvecs = load_calib(f'calibration_files/calibration_cam_{i}.npz')
        P = get_projection_matrix(mtx, rvecs, tvecs)
        P_mats.append(P)

    # Assume these points are from different cameras for the same 3D point
    points_img = []
    points_img.append(np.array([[[x1, y1]]], dtype=np.float32))  # from camera 1
    points_img.append(np.array([[[x2, y2]]], dtype=np.float32))  # from camera 2
    points_img.append(np.array([[[x3, y3]]], dtype=np.float32))  # from camera 3
    points_img.append(np.array([[[x4, y4]]], dtype=np.float32))  # from camera 4

    points_undistorted = []
    for i in range(4):  # adjust this range according to your number of cameras
        mtx, dist, _, _ = load_calib(f'calibration_files/calibration_cam_{i+1}.npz')
        undistorted = cv2.undistortPoints(points_img[i], mtx, dist, None, mtx)
        points_undistorted.append(undistorted)

    # Triangulate the 3D point
    point_3D = cv2.triangulatePoints(P_mats[0], P_mats[1], points_undistorted[0].T, points_undistorted[1].T)

    # Normalize the 3D point
    point_3D /= point_3D[3]
    return (point_3D)

'''
x1 = 400
x2 = 400
x3 = 400
x4 = 400
y1 = 400
y2 = 400
y3 = 400
y4 = 400

calculated_point_3D =  calculate_3D(x1, x2, x3, x4, y1, y2, y3, y4)

print("3D Point: ", calculated_point_3D)
'''


def start(ex_name: str):
    paths = glob.glob(f"../Motion Tracking/csv_export/{ex_name}_168448*")
    csv_file = open("csv_3D.csv", mode='w')
    csv_writer = csv.writer(csv_file)
    data = []
    for i, path in enumerate(paths):
        data.append(pd.read_csv(path, header=None))
    print(data)

    min_len = min(len(file.index) for file in data)
    print("Min len = ", min_len)
    _, num_cols = data[0].shape

    for i in range(min_len):
        csv_row = []
        for j in range(0, num_cols, 3):
            x1 = data[0].iloc[i, j]
            y1 = -data[0].iloc[i, j + 1]
            x2 = data[1].iloc[i, j]
            y2 = -data[1].iloc[i, j + 1]
            x3 = data[2].iloc[i, j]
            y3 = -data[2].iloc[i, j + 1]
            x4 = data[3].iloc[i, j]
            y4 = -data[3].iloc[i, j + 1]

            calculated_point_3D =  calculate_3D(x1, x2, x3, x4, y1, y2, y3, y4)
            csv_row.append(calculated_point_3D)
        csv_writer.writerow([kp[0] for kps in csv_row for kp in kps])



# start("1LBR")
