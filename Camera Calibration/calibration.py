import numpy as np
import cv2
import glob
import os


def get_calibration_files():
    """
    Get the camera calibration files to .npz format.
    """
    # Setting up calibration
    pattern_size = (8, 6)  # Number of inners corners of the chessboard used for calibration
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    obj_points = []
    img_points = []

    for i in range (1, 5):
        print(f"Starting analysis of camera {i} frames")
        images = glob.glob(f'calibration_images/cam_{i}_frame_*.jpg')

        # Loop over all images
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert it to grayscale according to CV2's documentation

            # Find calibration pattern points
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points and image points
            if ret:
                obj_points.append(pattern_points)
                img_points.append(corners)

                # Draw the corners on the image
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                # cv2.imwrite(f'chessboard_detection/{fname}', img)  # Uncomment to save the frame with chessboard drawn

            # Display the calibration image
            cv2.imshow('Calibration', img)
            cv2.waitKey(500)

        # When all frames are analyzed, compute the camera parameters
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        print(f"Finished analysis of camera {i} frames")
        # Save the camera parameters to a file
        if not os.path.exists("../calibration_files"):
            os.makedirs("../calibration_files")
        np.savez(f'calibration_files/calibration_cam_{i}.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"Calibration data saved to calibration_files/calibration_cam_{i}.npz")

