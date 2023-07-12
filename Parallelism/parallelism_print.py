import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_parallel(csv: str, point_a: int, point_b: int, point_c: int, point_d: int):
    cap = cv2.VideoCapture("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_Bicycle_cam_4.mp4")  # Replace "path/to/video" with the actual path of your video file
    fps = cap.get(cv2.CAP_PROP_FPS)

    data = pd.read_csv(csv, header=None)
    slope_diff_percentages = []
    num_rows, _ = data.shape

    plt.ion()  # Turn on interactive mode for dynamic plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    line, = ax.plot([], [], label="Parallelisme")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Parallelisme')
    ax.set_title('Percentage Difference Between the Two Line Slopes - MEDIA PIPE')
    ax.legend()

    for j in range(num_rows):
        ret, frame = cap.read()
        if not ret:
            break

        if j in data.index:
            x_a = data.iloc[j, 3 * point_a]
            y_a = data.iloc[j, 3 * point_a + 1]
            x_b = data.iloc[j, 3 * point_b]
            y_b = data.iloc[j, 3 * point_b + 1]
            x_c = data.iloc[j, 3 * point_c]
            y_c = data.iloc[j, 3 * point_c + 1]
            x_d = data.iloc[j, 3 * point_d]
            y_d = data.iloc[j, 3 * point_d + 1]

            if (x_b - x_a) != 0 and (x_d - x_c) != 0:
                AB_slope = (y_b - y_a) / (x_b - x_a)
                CD_slope = (y_d - y_c) / (x_d - x_c)
                slope_diff_percentage = abs(AB_slope - CD_slope) / ((AB_slope + CD_slope) / 2) * 100
                slope_diff_percentages.append(slope_diff_percentage)

            # Update the plot with real-time percentage difference
            parallelism_value = abs(1 - slope_diff_percentage / 100)
            line.set_data(range(j + 1), parallelism_value)
            line.set_color('green' if parallelism_value < 0.05 else 'red')
            ax.relim()
            ax.autoscale_view()

            # Display the video frame with OpenCV
            cv2.putText(frame, f"Percentage Difference: {slope_diff_percentage:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if parallelism_value < 0.05 else (0, 0, 255), 2)
            cv2.imshow("Video", frame)
            cv2.waitKey(int(1000 / fps))

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode


calculate_parallel("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_Bicycle_cam_4.csv", 11, 15, 12, 16)
