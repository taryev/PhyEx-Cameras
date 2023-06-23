import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from PIL import ImageTk, Image


Window=tk.Tk()

Label=tk.Label(Window, text= 'Calculate angles')
Label.pack()

### To create buttonFrame

buttonFrame1 = tk.Frame(Window)
buttonFrame1.pack( side="top", padx=20, pady=20, anchor='nw')

buttonFrame2 = tk.Frame(Window)
buttonFrame2.pack(side="top", padx=20, pady=20, anchor='nw')

buttonFrame3 = tk.Frame(Window)
buttonFrame3.pack(side="left", padx=20, pady=20, anchor='ne')

buttonFrame4 = tk.Frame(Window)
buttonFrame4.pack( side="right",padx=20, pady=20, anchor='nw')


### To chose the file

Label1=tk.Label(buttonFrame1, text= 'Choose the first file to compare')
Label1.pack()

def on_file_selected1(event):

    global file_path1

    selected_folder = folder_combobox1.get()
    if selected_folder:
        folder_path = os.path.join(base_folder_path, selected_folder)
        files = os.listdir(folder_path)
        file_combobox1['values'] = files
    
    selected_file = file_combobox1.get()
    if selected_folder and selected_file:
        file_path1 = os.path.join(base_folder_path, selected_folder, selected_file)

def on_file_selected2(event):

    global file_path2

    selected_folder = folder_combobox2.get()
    if selected_folder:
        folder_path = os.path.join(base_folder_path, selected_folder)
        files = os.listdir(folder_path)
        file_combobox2['values'] = files
    
    selected_file = file_combobox2.get()
    if selected_folder and selected_file:
        file_path2 = os.path.join(base_folder_path, selected_folder, selected_file)
### Copy your path for all CSV files here

base_folder_path = "C:\\Users\\33770\\Documents\\Stage_2A\\PhyEx-Cameras\\CSV"

folder_combobox1 = ttk.Combobox(buttonFrame1, state='readonly', width=20)
folder_combobox1.bind('<<ComboboxSelected>>', on_file_selected1)
folder_combobox1.pack()

file_combobox1 = ttk.Combobox(buttonFrame1, state='readonly', width=40)
file_combobox1.bind('<<ComboboxSelected>>', on_file_selected1)
file_combobox1.pack()

folders1 = os.listdir(base_folder_path)
folder_combobox1['values'] = folders1


Label4=tk.Label(buttonFrame1, text= 'Choose the second file to which the first is compared')
Label4.pack()

folder_combobox2 = ttk.Combobox(buttonFrame1, state='readonly', width=20)
folder_combobox2.bind('<<ComboboxSelected>>', on_file_selected2)
folder_combobox2.pack()

file_combobox2 = ttk.Combobox(buttonFrame1, state='readonly', width=40)
file_combobox2.bind('<<ComboboxSelected>>', on_file_selected2)
file_combobox2.pack()


folders2 = os.listdir(base_folder_path)
folder_combobox2['values'] = folders2

### To chose the joint

Label2=tk.Label(buttonFrame2, text= 'Chose the joint by these three points of interest')
Label2.pack()

selected_joint1 = None
selected_joint2 = None
selected_joint3 = None

def select_joint1(joint1):
    global selected_joint1
    selected_joint1 = joint1

def select_joint2(joint2):
    global selected_joint2
    selected_joint2 = joint2

def select_joint3(joint3):
    #print("Option sélectionnée :", joint)
    global selected_joint3
    selected_joint3 = joint3

list_of_joints=list(range(33))

menu1 = tk.StringVar(Window)
menu1.set("Please select the first point of interest")

menu2 = tk.StringVar(Window)
menu2.set("Please select the second point of interest")

menu3 = tk.StringVar(Window)
menu3.set("Please select the third point of interest")

drop_down_menu_1 = tk.OptionMenu(buttonFrame2, menu1, *list_of_joints, command=select_joint1)
drop_down_menu_1.config(width=36)
drop_down_menu_1.pack()

drop_down_menu_2 = tk.OptionMenu(buttonFrame2, menu2, *list_of_joints, command=select_joint2)
drop_down_menu_2.config(width=36)
drop_down_menu_2.pack()

drop_down_menu_3 = tk.OptionMenu(buttonFrame2, menu3, *list_of_joints, command=select_joint3)
drop_down_menu_3.config(width=36)
drop_down_menu_3.pack()

### To plot the curve

Label3=tk.Label(buttonFrame3, text= 'Plot the curve')
Label3.pack()


def read_angles_csvs(csv1 : str,  joint1 : int, joint2 : int, joint3 : int, csv2 : str,joint4 : int, joint5 : int, joint6 : int):
    
    #Read the CSV
    data1 = pd.read_csv(csv1, header=None)
    data2 = pd.read_csv(csv2, header=None)

    global angles1
    angles1 = []

    global angles2
    angles2 = []


    num_rows1, _ = data1.shape
    num_rows2, _ = data2.shape
    

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

        elif j in data2.index:

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

    fig = plt.figure()

    ax1 = plt.subplot(211)
    ax1.plot(angles1, color='purple')
    ax1.set_ylabel('Angles')
    ax1.set_title('Angles from the first CSV')

    ax2 = plt.subplot(212)
    ax2.plot(angles2, color='orange')
    ax2.set_ylabel('Angles')
    ax2.set_title('Angles from the second CSV')

    fig.subplots_adjust(hspace=0.5)

    canvas = FigureCanvasTkAgg(fig, master=buttonFrame3)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


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
            cost_matrix[i, j] = abs(angles1[i] - angles2[j]) + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])
    
    return cost_matrix[m-1,n-1]

def plot():
    read_angles_csvs(file_path1, selected_joint1, selected_joint2, selected_joint3, file_path2, selected_joint1, selected_joint2, selected_joint3 )

bouton = tk.Button(buttonFrame3, text="Click here", command=plot)
bouton.pack()

Label5=tk.Label(buttonFrame4, text= 'Show the DTW distance')
result_label = tk.Label(buttonFrame4, text="")

def print_result():
    result = dtw_distance(angles1, angles2)
    result_label.configure(text=result)

print_button = tk.Button(buttonFrame4, text="Click here", command=print_result)

Label5.pack()
print_button.pack()
result_label.pack()

def close_window():
    Window.destroy()
    Window.quit()

Window.protocol("WM_DELETE_WINDOW", close_window)

Window.mainloop()
