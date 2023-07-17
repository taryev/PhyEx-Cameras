
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import os
# Set the folder path where the files are located

folder_path = 'C:\\Users\\MihailS\\Documents\\Project\\Test\\Openpose\\TestSample'

# Get a list of all XLSX and CSV files in the folder
files = glob.glob(folder_path + '/*.npy')

# Process each file
for file in files:
    data1 = np.load(file)


    def read_angle_for_1_csv(joint1: int, joint2: int, joint3: int, joint4: int, joint5: int, joint6: int,
                             angle_to_analyse: int):
        file_name = os.path.basename(file)
        angles1 = []
        angles2 = [] #add as many vectors as you want, add 3 joints per each in the function

        num_rows1 = data1.shape[0]

        for j in range(num_rows1):
            try:
                x_a1 = data1[j, joint1,0] #x,y,z dimensions since the data1 shape is [n,25,3]
                y_a1 = data1[j, joint1,1]

                x_b1 = data1[j, joint2,0]
                y_b1 = data1[j, joint2,1]

                x_c1 = data1[j, joint3,0]
                y_c1 = data1[j, joint3,1]

                x_a2 = data1[j, joint4,0]
                y_a2 = data1[j, joint4,1]

                x_b2 = data1[j, joint5,0]
                y_b2 = data1[j, joint5,1]

                x_c2 = data1[j, joint6,0]
                y_c2 = data1[j, joint6,1]

                if np.all(x_b1 != x_a1):
                    AB1 = (y_b1 - y_a1) / (x_b1 - x_a1)
                else:
                    AB1 = 0.0

                if np.all(x_b1 != x_c1):
                    BC1 = (y_c1 - y_b1) / (x_c1 - x_b1)
                else:
                    BC1 = 0.0

                if np.all(x_b2 != x_a2):
                    AB2 = (y_b2 - y_a2) / (x_b2 - x_a2)
                else:
                    AB2 = 0.0

                if np.all(x_b2 != x_c2):
                    BC2 = (y_c2 - y_b2) / (x_c2 - x_b2)
                else:
                    BC2 = 0.0

                radians1 = np.arctan2((AB1 - BC1), (1 + AB1 * BC1))
                angle1 = 180 - np.abs(radians1 * 180.0 / np.pi)
                radians2 = np.arctan2((AB2 - BC2), (1 + AB2 * BC2))
                angle2 = 180 - np.abs(radians2 * 180.0 / np.pi)

                angles1.append(angle1)
                angles2.append(angle2)
            except (ZeroDivisionError, RuntimeWarning):
                angles1.append(np.nan)
                angles2.append(np.nan)

        angles1 = np.asarray(angles1)
        angles2 = np.asarray(angles2)

        inf_interval = 0.8 * angle_to_analyse
        sup_interval = 1.2 * angle_to_analyse

        valid_angles1 = np.logical_and(np.logical_not(np.isnan(angles1)),
                                       np.logical_and(angles1 >= inf_interval, angles1 <= sup_interval))
        valid_angles2 = np.logical_and(np.logical_not(np.isnan(angles2)),
                                       np.logical_and(angles2 >= inf_interval, angles2 <= sup_interval))

        count = np.count_nonzero(valid_angles1) + np.count_nonzero(valid_angles2)
        print('angle is respected', (count / num_rows1) * 100)

        N1 = 150  # window length
        w1 = 4  # polyorders
    #Savintsky-Golay filter
        angles_filt1 = signal.savgol_filter(angles1, N1, w1)
        angles_filt2 = signal.savgol_filter(angles2, N1, w1)

        plt.figure()
        plt.subplot(121)
        plt.plot(angles1, color='purple')
        plt.plot(angles_filt1, color='orange')
        plt.title("Knee")

        plt.subplot(122)
        plt.plot(angles2, color='blue')
        plt.plot(angles_filt2, color='red')
        plt.title("Elbow")

        plt.suptitle(file_name)
        plt.tight_layout()
        plt.show()

        return angles1, angles_filt1, angles2, angles_filt2, file_name, num_rows1


    #save the angles and filters of each file
    a1, af1, a2, af2, file_name, num_rows1 = read_angle_for_1_csv(9, 10, 11, 2, 3, 4, 120)
    #cost function
    def dtw_distance(angles1, angles2):
        m = len(angles1)
        n = len(angles2)

        # Create a cost matrix
        cost_matrix = np.zeros((m, n))

      # Initialise the first row and column of the cost matrix

        cost_matrix[0, 0] = abs(angles1[0] - angles2[0])
        for i in range(1, m):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + abs(angles1[i] - angles2[0])
        for j in range(1, n):
            cost_matrix[0, j] = cost_matrix[0, j-1] + abs(angles1[0] - angles2[j])

        # Fill in the rest of the cost matrix
        for i in range(1, m):
            for j in range(1, n):
                cost_matrix[i, j] = abs(angles1[i] - angles2[j]) + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1],
                                                                       cost_matrix[i - 1, j - 1])

            # Calculate the distance
        cost = cost_matrix[m - 1, n - 1]

        return cost

    cost1 = dtw_distance(a1, af1)
    cost2 = dtw_distance(a2, af2)


    print(file_name, "Distance DTW Knee:", cost1)
    print(file_name, "Distance DTW Elbow:", cost2)
    #enter the CSV file, the number of the joints which form the ABC triangle, and the angle you want to check
    # save the data in the folder
    output_dir = os.path.dirname(file)
    base_name = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_dir, f'{base_name}_angle_results.txt')

    # Output the results to the individualized text file
    angle_to_analyse=120
    inf = 0.8 * angle_to_analyse
    sup = 1.2 * angle_to_analyse
    count = 0
    with open(output_file, 'w') as f:
        for angle in a1:
            if (angle >= inf) and (angle <= sup):
                count += 1
                f.write(f'{angle}\n')
                f.write("Respected knee angle\n")
        f.write(f'angle is respected {(count / num_rows1) * 100}\n')
        count=0
        for angle in a2:
            if (angle >= inf) and (angle <= sup):
                count += 1
                f.write(f'{angle}\n')
                f.write("Respected elbow angle\n")
        f.write(f'angle is respected {(count / num_rows1) * 100}\n')
        f.write(f'Distance DTW knee: {cost1}\n')
        f.write(f'Distance DTW elbow: {cost2}\n')
    #saving the figure
    plt.figure()

    plt.subplot(121)
    plt.plot(a1, color='purple')
    plt.plot(af1, color='orange')
    plt.title("Knee")

    plt.subplot(122)
    plt.plot(a2, color='blue')
    plt.plot(af2, color='red')
    plt.title("Elbow")

    plt.tight_layout()
    plt.suptitle(file_name)

    plt.savefig(os.path.join(output_dir, f'{base_name}_plot.png'))

#    def data_type(file:str, joint1:int):
#        file_name= os.path.basename(file)
#        shape=data1.shape
#        return file_name,shape
#    name,shape=data_type(file,3)
#    print(name,' ',shape)
    #enter the CSV file and the number of the joints which form the ABC triangle