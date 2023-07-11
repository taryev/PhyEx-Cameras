import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_parallel(csv: str, point_a: int, point_b: int, point_c: int, point_d: int):

    data = pd.read_csv(csv, header=None)

    slope_diff_percentages = []

    num_rows, _ = data.shape

    for j in range(num_rows):
        if (j in data.index):

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

    parallelism_values = [abs(1 - diff / 100) for diff in slope_diff_percentages]


    fig = plt.figure(figsize=(10, 8))


    plt.plot(range(len(parallelism_values)), parallelism_values, label="Parallelisme")
    plt.xlabel('Index')
    plt.ylabel('Parallelisme')
    plt.title('Percentage Difference Between the Two Line Slopes - MEDIA PIPE')
    plt.legend()

    plt.show()


calculate_parallel("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_Bicycle_cam_4.csv", 11, 15, 12, 16)
