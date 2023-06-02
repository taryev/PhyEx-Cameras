import numpy as np
import cv2
import glob
import re

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Array to store object points and image points for each camera
camera_dict = {}

# get list of calibration images
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract camera index from filename
    camera_index = int(re.search('cam_(.*?)_frame', fname).group(1))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points, image points
    if ret == True:
        if camera_index not in camera_dict:
            camera_dict[camera_index] = {'objpoints': [], 'imgpoints': []}

        camera_dict[camera_index]['objpoints'].append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        camera_dict[camera_index]['imgpoints'].append(corners2)
    print(f'Camera {camera_index} and {fname}  processed')
# Calibrate the cameras
for camera_index in camera_dict.keys():
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(camera_dict[camera_index]['objpoints'],
                                                       camera_dict[camera_index]['imgpoints'],
                                                       gray.shape[::-1], None, None)
    # Save the camera calibration result for each camera
    np.savez(f'calib_cam_{camera_index}.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
