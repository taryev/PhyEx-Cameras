import numpy as np
import cv2
import glob


def undistort():
    """
    Allow the undistortion of images.
    """

    for i in range(1, 5):
        file_name = f"calibration_files/calibration_cam_{i}.npz"
        with np.load(file_name) as data:
            mtx = data["mtx"]
            dist = data["dist"]
            fname = glob.glob(f"calibration_images/cam_{i}_frame_*.jpg")
            for file in fname:
                img = cv2.imread(file)
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                # cv2.show('Undistorted no-crop', dst) # Uncomment to see what an uncropped image looks like

                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                cv2.show('Undistorted', dst)
