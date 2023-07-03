import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def calculate_distances(csv: str, point_a: int, point_b: int, point_c: int, point_d: int):
    # Read the CSV
    data = pd.read_csv(csv, header=None)

    AB = []
    CD = []
    ecart = []

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

            AB_val = np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2)
            CD_val = np.sqrt((x_d - x_c) ** 2 + (y_d - y_c) ** 2)

            # We add the calculated distances to the AB and CD lists
            AB.append(AB_val)
            CD.append(CD_val)

            # We calculate the pourcentage of difference
            ecart_val = abs(AB_val - CD_val) / ((AB_val + CD_val) / 2) * 100
            ecart.append(ecart_val)

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
    gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 2, 2, 2, 2])

    # Plot AB and CD
    ax1 = plt.subplot(gs[0:3])  # adjust the range here
    ax1.plot(range(len(AB)), AB, label="AB")
    ax1.plot(range(len(CD)), CD, label="CD", color='orange')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distances AB and CD')
    ax1.legend()
    ax1.set_xlim([0, len(AB)])  # X limits from 0 to the length of AB
    ax1.set_ylim([0, max(max(AB), max(CD)) * 1.1])  # Y limits from 0 to 110% of the max value

    # Plot the % of difference between AB and CD
    ax2 = plt.subplot(gs[4:6])  # adjust the range here
    ecart_above_threshold = [val if val > 5 else np.nan for val in ecart]
    ecart_below_threshold = [val if val <= 5 else np.nan for val in ecart]
    ax2.plot(range(len(ecart)), ecart_below_threshold, label="Ecart <= 5%", color='blue')
    ax2.plot(range(len(ecart)), ecart_above_threshold, label="Ecart > 5%", color='red')
    ax2.set_ylim([0, max(ecart) * 1.1])  # Y limits from 0 to 110% of the max value
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Pourcentage de différence')
    ax2.legend()

    # Table
    ax3 = plt.subplot(gs[6])  # adjust the range here
    table_data = [
        ['', 'AB', 'CD'],
        ['Mean', round(stats['AB']['mean'], 4), round(stats['CD']['mean'], 4)],
        ['Variance', round(stats['AB']['variance'], 4), round(stats['CD']['variance'], 4)],
        ['Standard Deviation', round(stats['AB']['std_dev'], 4), round(stats['CD']['std_dev'], 4)],
        ['Max', round(stats['AB']['max'], 4), round(stats['CD']['max'], 4)],
        ['Min', round(stats['AB']['min'], 4), round(stats['CD']['min'], 4)],
        ['Range', round(stats['AB']['range'], 4), round(stats['CD']['range'], 4)]
    ]
    table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_column_width([0, 1, 2])
    table.scale(1, 1.5)
    ax3.axis('off')

    table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_column_width([0, 1, 2])
    table.scale(1, 1.5)
    ax3.axis('off')

    plt.subplots_adjust(hspace=1.2)

    plt.show()

calculate_distances("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_Bicycle_cam_4.csv", 11, 15, 12, 16)
