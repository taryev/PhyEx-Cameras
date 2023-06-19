import cv2
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_angles_csvs(csv1 : str,  joint1 : int, joint2 : int, joint3 : int, csv2 : str,joint4 : int, joint5 : int, joint6 : int):
    
    #Read the CSV
    data1 = pd.read_csv(csv1, header=None)
    data2 = pd.read_csv(csv2, header=None)

    angles1=[]
    angles2=[]

    num_rows1,_=data1.shape
    num_rows2,_=data2.shape
    

    for j in range (max(num_rows1 , num_rows2)):
        
        if (j in data1.index and j in data2.index):
            #We get the coordinates for each csv
            x_a = data1.iloc[j,3 * joint1] 
            y_a = data1.iloc[j,3 * joint1 + 1]

            x_a_2 = data2.iloc[j,3 * joint4] 
            y_a_2 = data2.iloc[j,3 * joint4 + 1]

            x_b = data1.iloc[j,3 * joint2] 
            y_b = data1.iloc[j,3 * joint2 + 1]

            x_b_2 = data2.iloc[j,3 * joint5] 
            y_b_2 = data2.iloc[j,3 * joint5 + 1]

            x_c = data1.iloc[j,3 * joint3] 
            y_c = data1.iloc[j,3 * joint3 + 1]

            x_c_2 = data2.iloc[j,3 * joint6] 
            y_c_2 = data2.iloc[j,3 * joint6 + 1]

            #We calculate the distance for each csv

            AB = (y_b - y_a)/(x_b - x_a)
            BC = (y_c - y_b)/(x_c - x_b)

            AB_2 = (y_b_2 - y_a_2)/(x_b_2 - x_a_2)
            BC_2 = (y_c_2 - y_b_2)/(x_c_2 - x_b_2)

            #We calculate and convert the angle for each csv

            radians1 = np.arctan2((AB - BC), (1 + AB * BC))
            radians2 = np.arctan2((AB_2 - BC_2), (1 + AB_2 * BC_2))

            angle1 = 180 - np.abs(radians1*180.0/np.pi)
            angle2 = 180 - np.abs(radians2*180.0/np.pi)
    
            angles1.append(angle1)
            angles2.append(angle2)
        
        elif j in data1.index:

            x_a = data1.iloc[j,3 * joint1] 
            y_a = data1.iloc[j,3 * joint1 + 1]

            x_b = data1.iloc[j,3 * joint2] 
            y_b = data1.iloc[j,3 * joint2 + 1]

            x_c = data1.iloc[j,3 * joint3] 
            y_c = data1.iloc[j,3 * joint3 + 1]

            AB = (y_b - y_a)/(x_b - x_a)
            BC = (y_c - y_b)/(x_c - x_b)

            radians1 = np.arctan2((AB - BC), (1 + AB * BC))
            angle1 = 180 - np.abs(radians1*180.0/np.pi)
            angles1.append(angle1)

        elif j in data2.index :

            x_a_2 = data2.iloc[j,3 * joint4] 
            y_a_2 = data2.iloc[j,3 * joint4 + 1]

            x_b_2 = data2.iloc[j,3 * joint5] 
            y_b_2 = data2.iloc[j,3 * joint5 + 1]

            x_c_2 = data2.iloc[j,3 * joint6] 
            y_c_2 = data2.iloc[j,3 * joint6 + 1]

            AB_2 = (y_b_2 - y_a_2)/(x_b_2 - x_a_2)
            BC_2 = (y_c_2 - y_b_2)/(x_c_2 - x_b_2)

            radians2 = np.arctan2((AB_2 - BC_2), (1 + AB_2 * BC_2))
            angle2 = 180 - np.abs(radians2*180.0/np.pi)
            angles2.append(angle2)

    ax1 = plt.subplot(211)
    ax1.plot(angles1, color = 'purple')
    ax1.set_ylabel('Angles')
    ax1.set_title('Angles from the first CSV')

    ax2 = plt.subplot(212)
    ax2.plot(angles2, color = 'orange')
    ax2.set_ylabel('Angles')
    ax2.set_title('Angles from the second CSV')

    #Automatically adjust the spacing between subplots
    plt.tight_layout()
    plt.show()


    #plt.yticks(range(0, 150, 10))
  

#enter the CSV file and the number of the joints which form the ABC triangle 
read_angles_csvs('1LBR_Victor.csv', 24, 26, 28, '1LBR_Physio.csv', 24, 26, 28)