import numpy as np
import cv2
import pandas as pd



common_name_of_the_videos = "1LBR_1684830276_ID_Emilio"

file_names = [common_name_of_the_videos+'_cam_1.csv',
              common_name_of_the_videos +'_cam_2.csv',
              common_name_of_the_videos+'_cam_3.csv',
              common_name_of_the_videos+'_cam_4.csv']

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
        mtx, dist, rvecs, tvecs = load_calib(f'calib_cam_{i}.npz')
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
        mtx, dist, _, _ = load_calib(f'calib_cam_{i+1}.npz')
        undistorted = cv2.undistortPoints(points_img[i], mtx, dist, None, mtx)
        points_undistorted.append(undistorted)
    # Triangulate the 3D point
    # Pairwise triangulation
    points_3D_12 = cv2.triangulatePoints(P_mats[0], P_mats[1], points_undistorted[0].T, points_undistorted[1].T)
    points_3D_13 = cv2.triangulatePoints(P_mats[0], P_mats[2], points_undistorted[0].T, points_undistorted[2].T)
    points_3D_14 = cv2.triangulatePoints(P_mats[0], P_mats[3], points_undistorted[0].T, points_undistorted[3].T)
    points_3D_23 = cv2.triangulatePoints(P_mats[1], P_mats[2], points_undistorted[1].T, points_undistorted[2].T)
    points_3D_24 = cv2.triangulatePoints(P_mats[1], P_mats[3], points_undistorted[1].T, points_undistorted[3].T)
    points_3D_34 = cv2.triangulatePoints(P_mats[2], P_mats[3], points_undistorted[2].T, points_undistorted[3].T)

    # Normalizing the 3D points (dividing by the fourth element to get Euclidean coordinates)
    points_3D_12 /= points_3D_12[3]
    points_3D_13 /= points_3D_13[3]
    points_3D_14 /= points_3D_14[3]
    points_3D_23 /= points_3D_23[3]
    points_3D_24 /= points_3D_24[3]
    points_3D_34 /= points_3D_34[3]
    point_3D = (points_3D_12 + points_3D_13 + points_3D_14 + points_3D_23 + points_3D_24 + points_3D_34) / 6
    return (point_3D)



# Create an empty list to store the first items
first_items = []
# Loop through file names

all_3D_points = []

df = pd.read_csv("../../Motion_data/"+file_names[0])

min_len = float('inf')
min_file = ''

for file in file_names:
    df = pd.read_csv("../../Motion_data/" + file)
    current_len = len(df)
    if current_len < min_len:
        min_len = current_len
        min_file = file

all_3D_points = []
all_body_model_points =   []
for ii in range(0, 33):

    all_3D_points = []
    for i in range(0, min_len):
        first_items = []

        for file in file_names:
            # Read CSV
            df = pd.read_csv("../../Motion_data/"+file)
            # Get first item of the first and second column and append them to the list
            first_items.append(df.iloc[i, 0+3*ii])  # First item in first column
            first_items.append(df.iloc[i, 1+3*ii])  # First item in second column
        calculated_point_3D= calculate_3D(*first_items)
        all_3D_points.append(calculated_point_3D)
        # I want to save with each iterattion 4 values of the 3D point to a row and next point as a next row
        all_3D_points_np = np.array(all_3D_points).reshape(-1, 4)
        # I need to take only the 3 first columns
        all_3D_points_np = all_3D_points_np[:, 0:3]
    #print on the same line in the console and delete the previous line
        print(str(i) + " / " + str(min_len), end="\r")
    if ii == 0:
        all_body_model_points = all_3D_points_np
    else:
        all_body_model_points = np.concatenate((all_body_model_points, all_3D_points_np), axis =1)
    print("Done for keypoint: ", ii)
print("done")

dfvysl = pd.DataFrame(all_body_model_points)
coordinates = []
# Generate the coordinates
for i in range(33):
    coordinates.append(f'x_{i}')
    coordinates.append(f'y_{i}')
    coordinates.append(f'z_{i}')
dfvysl.columns = coordinates
dfvysl.to_csv("3D_points_from_calibration_files4cam.csv", index=False)





