from PIL import Image, ImageTk
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose


ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

window = ctk.CTk()
window.title("Interface Physical Rehabilitation")
window.geometry("1000x650")

# # Configure grid layout (4x4)
# window.grid_columnconfigure(0, weight=0)
# window.grid_columnconfigure(1, weight=0)


# Global variables
file_path1 = ""
file_path_video = ""
canvas = ""

# Create tabview

window.tabview = ctk.CTkTabview(window, width= 300,  height=600)
window.tabview.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20)
window.tabview.grid_rowconfigure(4, weight=1)
window.tabview.add("Plot curves")
window.tabview.add("Watch videos")
window.tabview.add("All scores")


# # Configure grid of individual tabs

# window.tabview.tab("Plot curves").grid_columnconfigure(0, weight=1)  
# window.tabview.tab("All scores").grid_columnconfigure(0, weight=0)
# #window.tabview.tab("Watch videos").grid_columnconfigure(0, weight=0)


# Create frames

frame_1_left = ctk.CTkFrame(window.tabview.tab("Plot curves"), width=300, height = 550)
frame_1_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_1_left.grid_propagate(False)
frame_1_left.grid_rowconfigure((0,1,2,3,4,5), weight=1)


frame_1 = ctk.CTkFrame(window.tabview.tab("Plot curves"), width=850, height = 550)
frame_1.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_1.grid_propagate(False)
frame_1.grid_rowconfigure(4, weight=0)

frame_2_left = ctk.CTkFrame(window.tabview.tab("Watch videos"), width=300, height = 550)
frame_2_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_2_left.grid_propagate(False)
frame_2_left.grid_rowconfigure(4, weight=0)

frame_2 = ctk.CTkFrame(window.tabview.tab("Watch videos"), width=850, height = 550)
frame_2.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_2.grid_propagate(False)
frame_2.grid_rowconfigure(4, weight=0)

frame_3_left = ctk.CTkFrame(window.tabview.tab("All scores"), width=300, height = 550)
frame_3_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_3_left.grid_propagate(False)
frame_3_left.grid_rowconfigure(4, weight=0)

frame_3 = ctk.CTkFrame(window.tabview.tab("All scores"), width=850, height = 550)
frame_3.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_3.grid_propagate(False)
frame_3.grid_rowconfigure(4, weight=0)

######################################################################## PLOT CURVES ##################################################################################

# Called fonction when you use one of the button on the interface

def select_file1():
    global file_path1
    file_path1 = filedialog.askopenfilename()
    informations = file_path1.split("/")
    name = informations[len(informations) - 1]
    if file_path1:
        window.file_button1.configure(text=name)

def select_angle1(event):
    global selected_angle1
    selected_angle1 = window.combobox1.get()

def select_angle2(event):
    global selected_angle2
    selected_angle2 = window.combobox2.get()

def get_plot():

    def read_angles_csvs(csv1 : str,  joint1 : int, joint2 : int, joint3 : int, csv2 : str,joint4 : int, joint5 : int, joint6 : int):
        
        global canvas
        
        if canvas:
            canvas.get_tk_widget().grid_remove()
        
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

        fig = plt.figure(figsize=(12.7, 8.3))
        fig.subplots_adjust(hspace=0.5)

        ax1 = plt.subplot(211)
        ax1.plot(angles1, color='purple')
        ax1.set_ylabel('Angles')
        ax1.set_title('Angles from the CSV you chose')

        ax2 = plt.subplot(212)
        ax2.plot(angles2, color='orange')
        ax2.set_ylabel('Angles')
        ax2.set_title('Angles from the physio')

        canvas = FigureCanvasTkAgg(fig, master=frame_1)
        canvas.draw()
        canvas.get_tk_widget().grid()

    def get_points_of_interest(excel):

        selected_joint1 = []
        selected_joint2 = []
        selected_joint3 = []
        selected_joint4 = []
        selected_joint5 = []
        selected_joint6 = []

        data=pd.read_excel(excel, header=None)
        
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
            
    selected_joint1, selected_joint2, selected_joint3, selected_joint4, selected_joint5, selected_joint6=get_points_of_interest('C:/Users/33770/Documents/Stage_2A/PhyEx-Cameras/Angles/Joints_List.xlsx')
    
    def get_matching_physio_file(file_path):
        informations = file_path.split("/")
        name = informations[len(informations) - 1]
        informations2 = name.split("_")
        exercise_name = informations2[0]
        number = informations2[5]
        number = number[:1]

        path_physio = 'C:/Users/33770/Documents/Stage_2A/PhyEx-Cameras/CSV/CSV_Physio'

        for file in os.listdir(path_physio):
            informations_file2 = file.split("_")
            exercise_name2 = informations_file2[0]
            number2 = informations_file2[5]
            number2 = number2[:1]
            if exercise_name2 == exercise_name and number2 == number:
                return os.path.join(path_physio, file)

        return ""
    
    file_path2_test=get_matching_physio_file(file_path1)

    read_angles_csvs(file_path1, int(selected_joint1), int(selected_joint2), int(selected_joint3), file_path2_test, int(selected_joint4), int(selected_joint5), int(selected_joint6))

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
        
        return cost_matrix[m-1, n-1]
    
    distance = dtw_distance(angles1, angles2)
    window.dtw_label.configure(text=distance)

    if distance > 10000:
        window.mark.configure(text = 'Bad angle', text_color= 'red')
    else: window.mark.configure(text = 'Good angle', text_color= 'green')
   
# Create the button to select the file

window.file_button1 = ctk.CTkButton(frame_1_left, text="Select the file you want to analyse", command=select_file1)
window.file_button1.grid(row=0, column=0, padx=20, pady=(10, 10))


# Create the combobox to select the joints

angles = ['Select the angle ','right_knee_angle', 'left_knee_angle', 'right_elbow_angle', 'left_elbow_angle', 'right_shoulder_angle', 'left_shoulder_angle', 'right_body', 'left_body']

window.combobox1 = ctk.CTkComboBox(frame_1_left, values=angles, button_color= 'orange',command=select_angle1)
window.combobox1.grid(row=1, column=0,padx=10, pady=(10, 10), sticky="ew")

window.combobox2 = ctk.CTkComboBox(frame_1_left, values=angles, button_color= 'orange',command=select_angle2)
window.combobox2.grid(row=2, column=0, padx=10, pady=(10, 10), sticky="ew")

# Create buttons to plot the curve and get the DTW distance

window.get_plot_button = ctk.CTkButton(frame_1_left, text="Plot the curve", command=get_plot)
window.get_plot_button.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="ew")

window.get_DTW_button = ctk.CTkButton(frame_1_left, text="Plot the DTW distance", command=get_DTW)
window.get_DTW_button.grid(row=4, column=0, padx=10, pady=(10, 10), sticky="ew")

window.dtw_label = ctk.CTkLabel(frame_1_left, text=" ")
window.dtw_label.grid(row=5, column=0, padx=10, pady=(10, 10), sticky="ew")

window.mark = ctk.CTkLabel(frame_1_left, text=" ")
window.mark.grid(row=6, column=0, padx=10, pady=(10, 10), sticky="ew")

######################################################################## SHOW VIDEOS ##################################################################################

def select_video():
    global file_path_video
    file_path_video = filedialog.askopenfilename()
    informations = file_path_video.split("/")
    name = informations[len(informations) - 1]    
    if file_path_video:
        window.video_button1.configure(text=name)

def get_video():

    video = cv2.VideoCapture(file_path_video)

    def show_video_with_mediapipe(cap, new_width, new_height):

        if not hasattr(show_video_with_mediapipe, 'canvas1_created'):
            
            show_video_with_mediapipe.canvas1 = tk.Canvas(frame_2, width=1300, height=800)
            show_video_with_mediapipe.canvas1.grid(row=4, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
            show_video_with_mediapipe.canvas1_created = True
    
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read() 

            
                resized_frame = cv2.resize(frame, (new_width, new_height))

                results = pose.process(resized_frame)
                
                mp_drawing.draw_landmarks(resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw
                
                photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
                show_video_with_mediapipe.canvas1.create_image(0, 0, image=photo, anchor=tk.NW)
                show_video_with_mediapipe.canvas1.photo = photo
                
                window.update()
                
                if cv2.waitKey(10) & 0xFF == ord('q'): 
                    break

            cap.release()
            cv2.destroyAllWindows()

    show_video_with_mediapipe(video, 1300, 800)
    
# Create the button to select the file

window.video_button1 = ctk.CTkButton(frame_2_left, text="Select the video you want to show", command=select_video)
window.video_button1.grid(row=1, column=0, padx=20, pady=(10,10))

# Create the button to show the video

window.show_video = ctk.CTkButton(frame_2_left, text="Show the video", command=get_video)
window.show_video.grid(row=2, column=0, padx=20, pady=(10, 10))

##########################################################################################################################################################
window.label_2 = ctk.CTkLabel(frame_3_left, text="Working on it")
window.label_2.grid(row=0, column=0, padx=20, pady=20)

########################################################################################################################################################################


# close the window when we click on the cross

def close_window():
    window.destroy()
    window.quit()
window.protocol("WM_DELETE_WINDOW", close_window)

# Execution of the main loop
window.mainloop()