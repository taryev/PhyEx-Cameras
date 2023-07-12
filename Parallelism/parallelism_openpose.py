import numpy as np
import matplotlib.pyplot as plt

def calculate_parallel(npy_file: str, point_a: int, point_b: int, point_c: int, point_d: int):
    # Load the npy file
    data = np.load(npy_file)

    slope_diff_percentages = []

    num_frames, num_keypoints, _ = data.shape

    for j in range(num_frames):
        # We get the coordinates for each point
        x_a, y_a = data[j, point_a][:2]
        x_b, y_b = data[j, point_b][:2]
        x_c, y_c = data[j, point_c][:2]
        x_d, y_d = data[j, point_d][:2]

        # We calculate the slopes of AB and CD
        if (x_b - x_a) != 0 and (x_d - x_c) != 0:  # Avoid dividing by zero
            AB_slope = (y_b - y_a) / (x_b - x_a)
            CD_slope = (y_d - y_c) / (x_d - x_c)

            # Calculate the percentage difference
            slope_diff_percentage = abs(AB_slope - CD_slope) / ((AB_slope + CD_slope) / 2) * 100
            slope_diff_percentages.append(slope_diff_percentage)

    # Only keep parallelism_values below a certain threshold
    parallelism_values = [abs(1 - diff / 100) for diff in slope_diff_percentages if abs(1 - diff / 100) < 20]

    fig = plt.figure(figsize=(10, 8))

    # Plot Parallel
    plt.plot(range(len(parallelism_values)), parallelism_values, label='Parallelisme')
    plt.xlabel('Index')
    plt.ylabel('Parallelisme')
    plt.title('Percentage Difference Between the Two Line Slopes - OPENPOSE')
    plt.legend()

    plt.show()

calculate_parallel("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_1 salome_cam_4.npy", 2,4,5,7)
