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
import data_handler as dh
import features as feat


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
selected_feature = ""
canvas = ""
list_of_data=[]


# Create tabview

window.tabview = ctk.CTkTabview(window, width= 300,  height=600)
window.tabview.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20)
window.tabview.grid_rowconfigure(4, weight=1)
window.tabview.add("Plot curves")
window.tabview.add("Watch videos")
window.tabview.add("All scores")

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

def select_feature(event):
    global selected_feature
    global list_of_data
    selected_feature = window.combobox1.get()

    if len(list_of_data) != 0 : 
        list_of_data=[]

    
    for feature in ['Angle', 'Distance', 'Alignment', 'Parallelism']:
        if selected_feature == feature:

            file=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)
            num_rows_f, _ = file.shape
            
            for i in range (1, num_rows_f):
                list_of_data.append(file.iloc[i,0])

            window.combobox2.configure(values=list_of_data)
    print(list_of_data)    

def select_data(event):
    global selected_data
    selected_data = window.combobox2.get()

def get_plot():

    global canvas
                   
    if canvas:
        canvas.get_tk_widget().grid_remove()

    if selected_feature=='Angle':

        file_angle=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)
              
        def read_angle_for_1_csv(csv1 : str,  joint1 : int, joint2 : int, joint3 : int):
            
            global canvas
                   
            if canvas:
                canvas.get_tk_widget().grid_remove()

            if csv1[-3:]=='csv':
                    data1 = pd.read_csv(csv1, header=None)
            elif csv1[-3:]=='lsx':
                    data1 = pd.read_excel(csv1, header=None)

            angles1 = []

            num_rows1, _ = data1.shape

            for j in range (0, num_rows1):
                
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


            # inf_interval = 0.8*angle_to_analyse
            # sup_interval = 1.2*angle_to_analyse
            # count=0
            
            # for angle in angles1:
            #     if (angle>=inf_interval) and (angle<=sup_interval) :
            #         count+=1

            # print('angle is respected : ', (count/num_rows1)*100)

            fig = plt.figure(figsize=(12.7, 8.3))
            plt.plot(angles1, color='purple')

            canvas = FigureCanvasTkAgg(fig, master=frame_1)
            canvas.draw()
            canvas.get_tk_widget().grid()

        def get_points_of_interest(file):

            selected_joint1 = []
            selected_joint2 = []
            selected_joint3 = []
        
            for i,angle in enumerate(list_of_data):
                if selected_data==angle:
                    selected_joint1=file.iloc[i+1,1]
                    selected_joint2=file.iloc[i+1,2]
                    selected_joint3=file.iloc[i+1,3]

            return selected_joint1, selected_joint2, selected_joint3
      
        selected_joint1, selected_joint2, selected_joint3=get_points_of_interest(file_angle) 
        
        read_angle_for_1_csv(file_path1, int(selected_joint1), int(selected_joint2), int(selected_joint3))

    # if selected_feature=='Distance':
    #     file_distance=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

    if selected_feature=='Alignment':

        file_alignment=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        def get_points_of_interest(file):

            if canvas:
                canvas.get_tk_widget().grid_remove()

            list=[]
                
            for i,angle in enumerate(list_of_data):      
                if selected_data==angle:
                    for j in range(1,7):
                        if pd.notna(file.iloc[i+1, j]):
                            list.append(file.iloc[i+1,j])
               
            return list

        list_of_points_to_test=get_points_of_interest(file_alignment)

        data = dh.MediapipeData(file_path1)

        list_with_name = []

        for point in list_of_points_to_test:
            list_with_name.append(data.get_by_index(point))

        alignement = feat.get_point_alignment(list_with_name)

        fig = plt.figure(figsize=(12.7, 8.3))
        plt.plot(alignement, color='orange')

        canvas = FigureCanvasTkAgg(fig, master=frame_1)
        canvas.draw()
        canvas.get_tk_widget().grid()

        # plt.figure()
        # plt.plot(alignement)
        # plt.show()


    if selected_feature=='Parallelism':  
      
        file_parallelism=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)


   
# Create the button to select the file

window.file_button1 = ctk.CTkButton(frame_1_left, text="Select the file you want to analyse", command=select_file1)
window.file_button1.grid(row=0, column=0, padx=20, pady=(10, 10))


# Create the combobox to select the points

features=['Select the features ','Angle', 'Distance', 'Alignment', 'Parallelism']
 
window.combobox1 = ctk.CTkComboBox(frame_1_left, values=features, button_color= 'orange',command=select_feature)
window.combobox1.grid(row=1, column=0,padx=10, pady=(10, 10), sticky="ew")

window.combobox2 = ctk.CTkComboBox(frame_1_left, values=[], button_color= 'orange',command=select_data)
window.combobox2.grid(row=2, column=0, padx=10, pady=(10, 10), sticky="ew")

## Create buttons to plot the curve 

window.get_plot_button = ctk.CTkButton(frame_1_left, text="Plot the curve", command=get_plot)
window.get_plot_button.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="ew")

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
