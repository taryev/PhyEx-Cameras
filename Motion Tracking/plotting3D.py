import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def basic_3D(csv_file: str):
    """
    Export Matplotlib 3D-plot for a CSV file
    :param csv_file:
    :return:
    """

    data = pd.read_csv(csv_file, header=None)

    x = []
    y = []
    z = []
    num_rows, num_cols = data.shape
    for i in range(0,1):
        for j in range(0, num_cols, 4):
            x_val = data.iloc[i, j]
            y_val = data.iloc[i, j + 1]
            z_val = data.iloc[i, j + 2]
            x.append(x_val)
            y.append(y_val)
            z.append(z_val)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=-90)
    #ax.mouse_init(rotate_btn=None)
    ax.scatter(x, y, z)

    # Plot the connections between points
    plt.plot([x[8], x[6], x[5], x[4], x[0], x[1], x[2], x[3], x[7]],
             [y[8], y[6], y[5], y[4], y[0], y[1], y[2], y[3], y[7]],
             [z[8], z[6], z[5], z[4], z[0], z[1], z[2], z[3], z[7]],
             color="cyan")  # Face
    plt.plot(x[9:11],
             y[9:11],
             z[9:11],
             color="pink")  # Mouth
    plt.plot([x[12], x[11], x[23], x[24], x[12]],
             [y[12], y[11], y[23], y[24], y[12]],
             [z[12], z[11], z[23], z[24], z[12]],
             color='purple')  # Trunk
    plt.plot(x[12:17:2],
             y[12:17:2],
             z[12:17:2],
             color='purple')  # Left arm
    plt.plot(x[11:16:2],
             y[11:16:2],
             z[11:16:2],
             color='purple')  # Right arm
    plt.plot(x[24:29:2],
             y[24:29:2],
             z[24:29:2],
             color='purple')  # Left leg
    plt.plot(x[23:28:2],
             y[23:28:2],
             z[23:28:2],
             color='purple')  # Right leg
    plt.plot([x[28], x[30], x[32], x[28]],
             [y[28], y[30], y[32], y[28]],
             [z[28], z[30], z[32], z[28]],
             color='purple')  # Left foot
    plt.plot([x[27], x[29], x[31], x[27]],
             [y[27], y[29], y[31], y[27]],
             [z[27], z[29], z[31], z[27]],
             color='purple')  # Right foot
    plt.plot([x[16], x[18], x[20], x[22], x[16]],
             [y[16], y[18], y[20], y[22], y[16]],
             [z[16], z[18], z[20], z[22], z[16]],
             color='purple')  # Left foot
    plt.plot([x[15], x[17], x[19], x[21], x[15]],
             [y[15], y[17], y[19], y[21], y[15]],
             [z[15], z[17], z[19], z[21], z[15]],
             color='purple')  # Left foot

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
