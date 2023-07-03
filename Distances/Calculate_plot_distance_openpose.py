import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

def calculate_distances(npy: str, point_a: int, point_b: int, point_c: int, point_d: int):

    data = np.load(npy)
    num_rows = data.shape[0]

    AB = []
    CD = []
    ecart = []

    for j in range(num_rows):
        # We get the coordinates for each point
        x_a, y_a, _ = data[j][point_a]
        x_b, y_b, _ = data[j][point_b]
        x_c, y_c, _ = data[j][point_c]
        x_d, y_d, _ = data[j][point_d]

        x_a /= 1280  # replace with the actual width of your image
        y_a /= 720  # replace with the actual height of your image

        x_b /= 1280  # replace with the actual width of your image
        y_b /= 720  # replace with the actual height of your image

        x_c /= 1280  # replace with the actual width of your image
        y_c /= 720  # replace with the actual height of your image

        x_d /= 1280  # replace with the actual width of your image
        y_d /= 720  # replace with the actual height of your image



        AB_val = np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2)
        CD_val = np.sqrt((x_d - x_c) ** 2 + (y_d - y_c) ** 2)


        AB.append(AB_val)
        CD.append(CD_val)


        ecart_val = abs(AB_val - CD_val) / ((AB_val + CD_val) / 2) * 100
        ecart.append(ecart_val)


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
    ax1 = plt.subplot(gs[0:3])
    ax1.plot(range(len(AB)), AB, label="AB")
    ax1.plot(range(len(CD)), CD, label="CD", color='orange')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distances AB and CD')
    ax1.legend()
    ax1.set_xlim([0, len(AB)])  # X limits from 0 to the length of AB
    ax1.set_ylim([0, max(max(AB), max(CD)) * 1.1])  # Y limits from 0 to 110% of the max value

    # Plot the % of difference between AB and CD
    ax2 = plt.subplot(gs[4:6])
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

calculate_distances("C:/Users/Salomé/Downloads/CLMB_1684851174_ID_1 salome_cam_4.npy", 2, 4, 5, 7)


