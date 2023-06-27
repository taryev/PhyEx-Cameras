import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Variables globales
file_path1 = ""
file_path2 = ""

# Fonction appelée lors du clic sur le bouton "Sélectionner un fichier"
def select_file1():
    global file_path1
    file_path1 = filedialog.askopenfilename()
    if file_path1:
        file_button1.configure(text=file_path1)

def select_file2():
    global file_path2
    file_path2 = filedialog.askopenfilename()
    if file_path2:
        file_button2.configure(text=file_path2)

def select_angle1(event):
    global selected_angle1
    selected_angle1 = combobox1.get()

def select_angle2(event):
    global selected_angle2
    selected_angle2 = combobox2.get()

def get_plot():

    def read_angles_csvs(csv1 : str,  joint1 : int, joint2 : int, joint3 : int, csv2 : str,joint4 : int, joint5 : int, joint6 : int):
        
        #Read the CSV

        if csv1[-3:]=='csv':
            data1 = pd.read_csv(csv1, header=None)
        elif csv1[-3:]=='lsx':
            data1 = pd.read_excel(csv1, header=None)
        
        if csv2[-3:]=='csv':
            data2 = pd.read_csv(csv2, header=None)
        elif csv2[-3:]=='lsx':
            data2 = pd.read_excel(csv2, header=None)
        

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
        fig.subplots_adjust(hspace=0.5)

        ax1 = plt.subplot(211)
        ax1.plot(angles1, color='purple')
        ax1.set_ylabel('Angles')
        ax1.set_title('Angles from the first CSV')

        ax2 = plt.subplot(212)
        ax2.plot(angles2, color='orange')
        ax2.set_ylabel('Angles')
        ax2.set_title('Angles from the second CSV')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        espace.pack()
        canvas.get_tk_widget().pack()

    def get_points_of_interest(excel):

        selected_joint1 = []
        selected_joint2 = []
        selected_joint3 = []
        selected_joint4 = []
        selected_joint5 = []
        selected_joint6 = []

        data=pd.read_excel(excel)
        
        angles=['right_knee_angle', 'left_knee_angle', 'right_elbow_angle', 'left_elbow_angle', 'right_shoulder_angle', 'left_shoulder_angle', 'right_body', 'left_body']
        for i,angle in enumerate(angles):
            if selected_angle1==angle:
                selected_joint1=data.iloc[i,1]
                selected_joint2=data.iloc[i,2]
                selected_joint3=data.iloc[i,3]

            if selected_angle2==angle:
                selected_joint4=data.iloc[i,1]
                selected_joint5=data.iloc[i,2]
                selected_joint6=data.iloc[i,3]

        return selected_joint1, selected_joint2, selected_joint3, selected_joint4, selected_joint5, selected_joint6
            
    selected_joint1, selected_joint2, selected_joint3, selected_joint4, selected_joint5, selected_joint6=get_points_of_interest('Joints.xlsx')
    
    read_angles_csvs(file_path1, int(selected_joint1), int(selected_joint2), int(selected_joint3), file_path2,int(selected_joint4), int(selected_joint5), int(selected_joint6))

def get_DTW():
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
    
    distance=dtw_distance(angles1, angles2)
    dtw_label.configure(text = distance)
    

# Create the main window
window = ctk.CTk()
window.title("Interface Physical Rehabilitation")
window.geometry("800x650")


# Create the widgets
file_button1 = ctk.CTkButton(window, text="Select the first file", command=select_file1)
file_button2 = ctk.CTkButton(window, text="Select the second file", command=select_file2)


#Espace
espace = ctk.CTkLabel(window, text=" ")


# Create the combobox to select the joints
angles=['Select the angle ','right_knee_angle', 'left_knee_angle', 'right_elbow_angle', 'left_elbow_angle', 'right_shoulder_angle', 'left_shoulder_angle', 'right_body', 'left_body']

combobox1 = ctk.CTkComboBox(window, values=angles, button_color= 'orange',command=select_angle1)
combobox2 = ctk.CTkComboBox(window, values=angles, button_color= 'orange',command=select_angle2)



#Button to plot the curve
get_plot_button = ctk.CTkButton(window, text="Plot the curve", command=get_plot)
get_DTW_button = ctk.CTkButton(window, text="Plot the DTW distance", command=get_DTW)
dtw_label = ctk.CTkLabel(window, text=" ")


# Placement des widgets dans la fenêtre

file_button1.pack(padx=5,pady=10)

file_button2.pack(padx=5,pady=10)


combobox1.pack(padx=5,pady=10)

combobox2.pack(padx=5,pady=10)


# Bouton pour récupérer les valeurs avant le mainloop
get_plot_button.pack(padx=5,pady=10)

get_DTW_button.pack(padx=5,pady=10)
dtw_label.pack()

def close_window():
    window.destroy()
    window.quit()

window.protocol("WM_DELETE_WINDOW", close_window)

# Exécution de la boucle principale
window.mainloop()




