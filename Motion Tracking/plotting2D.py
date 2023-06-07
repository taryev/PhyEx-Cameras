import os
import pandas as pd
import matplotlib.pyplot as plt


def export_plot(csv_file: str, stationary_axis: bool = False):
    """
    Export Matplotlib plot for a CSV file
    :param csv_file: CSV file URL
    :return:
    """

    data = pd.read_csv(csv_file, header=None)
    if not os.path.exists("2D_plots"):
        os.makedirs("2D_plots")

    # Extract the coordinates
    x = []
    y = []
    num_rows, num_cols = data.shape
    for i in range(num_rows):
        for j in range(0, num_cols, 3):
            x_val = data.iloc[i, j]
            y_val = -data.iloc[i, j + 1]
            x.append(x_val)
            y.append(y_val)

        # Plot the coordinates in 2D
        plt.scatter(x, y, color="gray", marker='.')

        # Plot the connections between points
        plt.plot([x[8], x[6], x[5], x[4], x[0], x[1], x[2], x[3], x[7]], [y[8], y[6], y[5], y[4], y[0], y[1], y[2], y[3], y[7]], color="cyan") # Face
        plt.plot(x[9:11], y[9:11], color="pink") # Mouth
        plt.plot([x[12],x[11], x[23], x[24], x[12]], [y[12],y[11],y[23],y[24], y[12]], color='purple') # Trunk
        plt.plot(x[12:17:2], y[12:17:2], color='purple') # Left arm
        plt.plot(x[11:16:2], y[11:16:2], color='purple') # Right arm
        plt.plot(x[24:29:2], y[24:29:2], color='purple') # Left leg
        plt.plot(x[23:28:2], y[23:28:2], color='purple') # Right leg
        plt.plot([x[28], x[30], x[32], x[28]], [y[28], y[30], y[32], y[28]], color='purple') # Left foot
        plt.plot([x[27], x[29], x[31], x[27]], [y[27], y[29], y[31], y[27]], color='purple') # Right foot
        plt.plot([x[16], x[18], x[20], x[22], x[16]], [y[16], y[18], y[20], y[22], y[16]], color='purple') # Left foot
        plt.plot([x[15], x[17], x[19], x[21], x[15]], [y[15], y[17], y[19], y[21], y[15]], color='purple') # Left foot

        # Add annotations for point numbers
        # for i, (x_val, y_val) in enumerate(zip(x, y)):
        #     plt.annotate(str(i), (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center')

        # Set labels for the axes
        plt.xlabel('X')
        plt.ylabel('Y')

        # Limit axis range
        if (stationary_axis):
            plt.xlim(0.25, 0.70)
            plt.ylim(-1, 0.25)

        # Save plot to file
        plt.savefig(f"2D_plots/_plot_row_{i}.png", format="png")
        plt.close()
        x = []
        y = []
