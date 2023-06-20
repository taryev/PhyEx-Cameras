import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

Window=tk.Tk()
Label=tk.Label(Window, text= 'Calculate angles')
Label.pack()

### To chose the file

buttonFrame1 = tk.Frame(Window)
buttonFrame1.pack(side="top", padx=20, pady=20, anchor="nw")

buttonFrame2 = tk.Frame(Window)
buttonFrame2.pack(side="top", padx=20, pady=20, anchor="nw")

def on_file_selected(event):

    global file_path

    selected_folder = folder_combobox.get()
    if selected_folder:
        folder_path = os.path.join(base_folder_path, selected_folder)
        files = os.listdir(folder_path)
        file_combobox['values'] = files
    
    selected_file = file_combobox.get()
    if selected_folder and selected_file:
        file_path = os.path.join(base_folder_path, selected_folder, selected_file)

base_folder_path = "C:\\Users\\33770\\Documents\\Stage_2A\\PhyEx-Cameras\\CSV"

folder_combobox = ttk.Combobox(buttonFrame1, state='readonly', width=20)
folder_combobox.bind('<<ComboboxSelected>>', on_file_selected)
folder_combobox.pack()

file_combobox = ttk.Combobox(buttonFrame1, state='readonly', width=40)
file_combobox.bind('<<ComboboxSelected>>', on_file_selected)
file_combobox.pack()

folders = os.listdir(base_folder_path)
folder_combobox['values'] = folders

### To chose the joint

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
drop_down_menu_1.pack()

drop_down_menu_2 = tk.OptionMenu(buttonFrame2, menu2, *list_of_joints, command=select_joint2)
drop_down_menu_2.pack()

drop_down_menu_3 = tk.OptionMenu(buttonFrame2, menu3, *list_of_joints, command=select_joint3)
drop_down_menu_3.pack()

### To plot the curve

def read_angles_csvs(csv1 : str,  joint1 : int, joint2 : int, joint3 : int):
    
    #Read the CSV
    data1 = pd.read_csv(csv1, header=None)

    angles1=[]

    num_rows1,_=data1.shape
    

    for j in range (num_rows1):
        
        #We get the coordinates for each csv
        x_a = data1.iloc[j,3 * joint1] 
        y_a = data1.iloc[j,3 * joint1 + 1]

        x_b = data1.iloc[j,3 * joint2] 
        y_b = data1.iloc[j,3 * joint2 + 1]

        x_c = data1.iloc[j,3 * joint3] 
        y_c = data1.iloc[j,3 * joint3 + 1]

        #We calculate the distance for each csv

        AB = (y_b - y_a)/(x_b - x_a)
        BC = (y_c - y_b)/(x_c - x_b)

        radians1 = np.arctan2((AB - BC), (1 + AB * BC))
        angle1 = 180 - np.abs(radians1*180.0/np.pi)
        angles1.append(angle1)
                
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(angles1, color = 'purple')
    ax.set_ylabel('Angles')
    ax.set_title('Angles from the CSV')

    canvas = FigureCanvasTkAgg(fig, master=Window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def plot():
    read_angles_csvs(file_path, selected_joint1, selected_joint2, selected_joint3 )

bouton = tk.Button(Window, text="Plot the curve of the selected angle", command=plot)
bouton.pack()

def close_window():
    Window.destroy()
    Window.quit()

Window.protocol("WM_DELETE_WINDOW", close_window)

Window.mainloop()

