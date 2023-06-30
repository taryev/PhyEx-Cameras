import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def calculate_distances(csv: str, point_a: int, point_b: int, point_c: int, point_d: int):
    # Read the CSV
    data = pd.read_csv(csv, header=None)

    AB = []
    CD = []

    num_rows, _ = data.shape

    for j in range(num_rows):

        if (j in data.index):
            # We get the coordinates for each point
            x_a = data.iloc[j, 3 * point_a]
            y_a = data.iloc[j, 3 * point_a + 1]

            x_b = data.iloc[j, 3 * point_b]
            y_b = data.iloc[j, 3 * point_b + 1]

            x_c = data.iloc[j, 3 * point_c]
            y_c = data.iloc[j, 3 * point_c + 1]

            x_d = data.iloc[j, 3 * point_d]
            y_d = data.iloc[j, 3 * point_d + 1]

            # We calculate the distance for each csv
            AB.append(np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2))
            CD.append(np.sqrt((x_d - x_c) ** 2 + (y_d - y_c) ** 2))

    # Calculate stats
    stats = {
        'AB': {
            'mean': np.mean(AB),
            'variance': np.var(AB),
            'std_dev': np.std(AB),
            'max': np.max(AB),
            'min': np.min(AB),
            'range': np.max(AB) - np.min(AB)
        },
        'CD': {
            'mean': np.mean(CD),
            'variance': np.var(CD),
            'std_dev': np.std(CD),
            'max': np.max(CD),
            'min': np.min(CD),
            'range': np.max(CD) - np.min(CD)
        },
    }


    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(6, 1, height_ratios=[4, 1, 0.5, 4, 1, 1])

    # Plot AB and CD
    ax1 = plt.subplot(gs[0:4])
    ax1.plot(range(len(AB)), AB, label="AB")
    ax1.plot(range(len(CD)), CD, label="CD", color='orange')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distances AB and CD')
    ax1.legend()

    # Set Y limits
    ax1.set_ylim([0.15, max(max(AB), max(CD)) * 1.1])

    # Table
    ax2 = plt.subplot(gs[5])
    table_data = [
        ['', 'AB', 'CD'],
        ['Mean', stats['AB']['mean'], stats['CD']['mean']],
        ['Variance', stats['AB']['variance'], stats['CD']['variance']],
        ['Standard Deviation', stats['AB']['std_dev'], stats['CD']['std_dev']],
        ['Max', stats['AB']['max'], stats['CD']['max']],
        ['Min', stats['AB']['min'], stats['CD']['min']],
        ['Range', stats['AB']['range'], stats['CD']['range']]
    ]
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_column_width([0, 1, 2])
    table.scale(1, 1.5)
    ax2.axis('off')


    plt.subplots_adjust(hspace=0.5)

    plt.show()


calculate_distances("C:/Users/Salomé/Downloads/CLMB_1684487736_ID_Physio_cam_3.csv", 11, 15, 12, 16)
