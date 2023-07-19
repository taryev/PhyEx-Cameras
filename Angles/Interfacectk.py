from PIL import Image, ImageTk
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import mediapipe as mp
import data_handler as dh
import matplotlib.gridspec as gridspec
import features as feat

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

window = ctk.CTk()
window.title("Interface Physical Rehabilitation")

# Size of the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
    
desired_height=int(screen_height*0.9)

# Configuration the size of the interface to match the screen
window.geometry(f"{screen_width}x{desired_height}+{0}+{0}")

# Global variables

file_path1 = ""
name=""
file_path_video = ""
selected_feature = ""
canvas = ""
canvas_score = ""
list_of_data=[]

# Create tabview

tabview_width = 0.97*screen_width
tabview_height = 0.92*desired_height

window.tabview = ctk.CTkTabview(window, width= tabview_width,  height=tabview_height)
window.tabview.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=20)
window.tabview.grid_rowconfigure(4, weight=1)
window.tabview.add("Watch videos")
window.tabview.add("Plot curves")
window.tabview.add("All scores")

# Create frames

frame_height = 0.92 * tabview_height
frame_left_width = 0.25 * tabview_width
frame_right_width = 0.67 * tabview_width

frame_1_left = ctk.CTkFrame(window.tabview.tab("Plot curves"), width=frame_left_width, height = frame_height)
frame_1_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_1_left.grid_propagate(False)
frame_1_left.grid_rowconfigure((0,1,2,3,4,5), weight=1)

frame_1 = ctk.CTkFrame(window.tabview.tab("Plot curves"), width=frame_right_width, height = frame_height)
frame_1.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_1.grid_propagate(False)
frame_1.grid_rowconfigure(4, weight=0)

frame_2_left = ctk.CTkFrame(window.tabview.tab("Watch videos"), width=frame_left_width, height = frame_height)
frame_2_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_2_left.grid_propagate(False)
frame_2_left.grid_rowconfigure(4, weight=0)

frame_2 = ctk.CTkFrame(window.tabview.tab("Watch videos"), width=frame_right_width, height = frame_height)
frame_2.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_2.grid_propagate(False)
frame_2.grid_rowconfigure(4, weight=0)

frame_3_left = ctk.CTkFrame(window.tabview.tab("All scores"), width=frame_left_width, height = frame_height)
frame_3_left.grid(row=0, column=0, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_3_left.grid_propagate(False)
frame_3_left.grid_rowconfigure(4, weight=0)

frame_3 = ctk.CTkFrame(window.tabview.tab("All scores"), width=frame_right_width, height = frame_height)
frame_3.grid(row=0, column=1, rowspan=4, sticky="nsew",padx=20, pady=20)
frame_3.grid_propagate(False)
frame_3.grid_rowconfigure(4, weight=0)

######################################################################## PLOT CURVES ##################################################################################

# Called fonction when you use one of the button on the interface

def select_file1():
    global file_path1
    global name
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

    
    for feature in ['Angle', 'Distance', 'Alignment', 'Parallelism','Same_coordinate']:
        if selected_feature == feature:

            file=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)
            num_rows_f, _ = file.shape
            
            for i in range (1, num_rows_f):
                list_of_data.append(file.iloc[i,0])

            window.combobox2.configure(values=list_of_data)
        
def select_data(event):
    global selected_data
    selected_data = window.combobox2.get()

def get_plot():

    global canvas
                   
    if canvas:
        canvas.get_tk_widget().grid_remove()

    if selected_feature=='Angle':

        file_angle=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        print(file_angle)
              
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

            fig = plt.figure(figsize=(0.015*frame_right_width, 0.015*frame_height))
            plt.plot(angles1, color='purple')
            plt.title("Angle")

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
      
        selected_joint_angle_1, selected_joint_angle_2, selected_joint_angle_3=get_points_of_interest(file_angle) 
        
        read_angle_for_1_csv(file_path1, int(selected_joint_angle_1), int(selected_joint_angle_2), int(selected_joint_angle_3))

    if selected_feature=='Distance':
        file_distance=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        def get_points_of_interest_distance(file):

            selected_joint1 = []
            selected_joint2 = []
            selected_joint3 = []
            selected_joint4 = []
        
            for i,angle in enumerate(list_of_data):
                if selected_data==angle:
                    selected_joint1=file.iloc[i+1,1]
                    selected_joint2=file.iloc[i+1,2]
                    selected_joint3=file.iloc[i+1,3]
                    selected_joint4=file.iloc[i+1,4]

            return selected_joint1, selected_joint2, selected_joint3, selected_joint4
      
        selected_joint_distance_1, selected_joint_distance_2, selected_joint_distance_3, selected_joint_distance_4=get_points_of_interest_distance(file_distance)

        def calculate_distances(csv: str, point_a: int, point_b: int, point_c: int, point_d: int):
            
            global canvas
                   
            if canvas:
                canvas.get_tk_widget().grid_remove()
            
            # Read the CSV or xlsx file
            if csv[-3:]=='csv':
                    data = pd.read_csv(csv, header=None)
            elif csv[-3:]=='lsx':
                    data = pd.read_excel(csv, header=None)

            AB = []
            CD = []
            ecart = []

            num_rows, _ = data.shape

            for j in range(num_rows):

                if (j in data.index):
                    # We get the coordinates for each point
                    x_a = data.iloc[j, 3 * point_a]
                    y_a = data.iloc[j, 3 * point_a + 1]

                    x_b = data.iloc[j, 3 * point_b]
                    y_b = data.iloc[j, 3 * point_b + 1]

                    x_c = data.iloc[j, 3 * point_c]
                    y_c = data.iloc[j, 3 * point_c + 1]

                    x_d = data.iloc[j, 3 * point_d]
                    y_d = data.iloc[j, 3 * point_d + 1]

                    AB_val = np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2)
                    CD_val = np.sqrt((x_d - x_c) ** 2 + (y_d - y_c) ** 2)

                    # We add the calculated distances to the AB and CD lists
                    AB.append(AB_val)
                    CD.append(CD_val)

                    # We calculate the pourcentage of difference
                    ecart_val = abs(AB_val - CD_val) / ((AB_val + CD_val) / 2) * 100
                    ecart.append(ecart_val)

            

            fig = plt.figure(figsize=(12.7, 8.3))
            gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 2, 2, 2, 2])

            # Plot AB and CD
            ax1 = plt.subplot(gs[0:3])  # adjust the range here
            ax1.plot(range(len(AB)), AB, label="AB")
            ax1.plot(range(len(CD)), CD, label="CD", color='orange')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Distance')
            ax1.set_title('Distances AB and CD')
            ax1.legend()
            ax1.set_xlim([0, len(AB)])  # X limits from 0 to the length of AB
            ax1.set_ylim([0, max(max(AB), max(CD)) * 1.1])  # Y limits from 0 to 110% of the max value

            # Plot the % of difference between AB and CD
            ax2 = plt.subplot(gs[4:6])  # adjust the range here
            ecart_above_threshold = [val if val > 10 else np.nan for val in ecart]
            ecart_below_threshold = [val if val <= 10 else np.nan for val in ecart]
            ax2.plot(range(len(ecart)), ecart_below_threshold, label="Ecart <= 10%", color='blue')
            ax2.plot(range(len(ecart)), ecart_above_threshold, label="Ecart > 10%", color='red')
            ax2.set_ylim([0, max(ecart) * 1.1])  # Y limits from 0 to 110% of the max value
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Pourcentage de différence')
            ax2.legend()

            
            plt.subplots_adjust(hspace=1.2)

            canvas = FigureCanvasTkAgg(fig, master=frame_1)
            canvas.draw()
            canvas.get_tk_widget().grid()

        calculate_distances(file_path1, int(selected_joint_distance_1), int(selected_joint_distance_2), int(selected_joint_distance_3), int(selected_joint_distance_4))

    if selected_feature=='Alignment':

        file_alignment=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        def get_points_of_interest_alignment(file):

            if canvas:
                canvas.get_tk_widget().grid_remove()

            list=[]
                
            for i,angle in enumerate(list_of_data):      
                if selected_data==angle:
                    for j in range(1,7):
                        if pd.notna(file.iloc[i+1, j]):
                            list.append(file.iloc[i+1,j])
               
            return list

        list_of_points_to_test=get_points_of_interest_alignment(file_alignment)

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

    if selected_feature=='Parallelism':  
      
        file_parallelism=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        def calculate_parallel(file: str, point_a: int, point_b: int, point_c: int, point_d: int):

            global canvas
                   
            if canvas:
                canvas.get_tk_widget().grid_remove()

            if file[-3:]=='csv':
                data = pd.read_csv(file, header=None)
            elif file[-3:]=='lsx':
                data = pd.read_excel(file, header=None)

            slope_diff_percentages = []

            num_rows, _ = data.shape

            for j in range(num_rows):
                if (j in data.index):

                    x_a = data.iloc[j, 3 * point_a]
                    y_a = data.iloc[j, 3 * point_a + 1]

                    x_b = data.iloc[j, 3 * point_b]
                    y_b = data.iloc[j, 3 * point_b + 1]

                    x_c = data.iloc[j, 3 * point_c]
                    y_c = data.iloc[j, 3 * point_c + 1]

                    x_d = data.iloc[j, 3 * point_d]
                    y_d = data.iloc[j, 3 * point_d + 1]


                    if (x_b - x_a) != 0 and (x_d - x_c) != 0:
                        AB_slope = (y_b - y_a) / (x_b - x_a)
                        CD_slope = (y_d - y_c) / (x_d - x_c)


                        slope_diff_percentage = abs(AB_slope - CD_slope) / ((AB_slope + CD_slope) / 2) * 100

                        slope_diff_percentages.append(slope_diff_percentage)

            parallelism_values = [abs(1 - diff / 100) for diff in slope_diff_percentages]


            fig = plt.figure(figsize=(12.7, 8.3))
            plt.plot(range(len(parallelism_values)), parallelism_values, label="Parallelisme", color='purple')

        

            plt.xlabel('Index')
            plt.ylabel('Parallelisme')
            plt.title('Percentage Difference Between the Two Line Slopes - MEDIA PIPE')
            plt.legend()

            canvas = FigureCanvasTkAgg(fig, master=frame_1)
            canvas.draw()
            canvas.get_tk_widget().grid()

        def get_points_of_interest_parallelism(file):

            selected_joint1 = []
            selected_joint2 = []
            selected_joint3 = []
            selected_joint4 = []
        
            for i,angle in enumerate(list_of_data):
                if selected_data==angle:
                    selected_joint1=file.iloc[i+1,1]
                    selected_joint2=file.iloc[i+1,2]
                    selected_joint3=file.iloc[i+1,3]
                    selected_joint4=file.iloc[i+1,4]

            return selected_joint1, selected_joint2, selected_joint3, selected_joint4
      
        selected_joint_parallelisme_1, selected_joint_parallelisme_2, selected_joint_parallelisme_3, selected_joint_parallelisme_4=get_points_of_interest_parallelism(file_parallelism)    

        calculate_parallel(file_path1, int(selected_joint_parallelisme_1), int(selected_joint_parallelisme_2), int(selected_joint_parallelisme_3),  int(selected_joint_parallelisme_4))
    
    if selected_feature=='Same_coordinate':  
      
        file_scoord=pd.read_excel('Features.xlsx',header=None, sheet_name=selected_feature)

        def get_points_of_interest_scoord(file):

            selected_joint1 = []
            selected_joint2 = []
            selected_joint3 = []
            selected_joint4 = []
        
            for i,angle in enumerate(list_of_data):
                if selected_data==angle:
                    selected_joint1=file.iloc[i+1,1]
                    selected_joint2=file.iloc[i+1,2]
                    selected_joint3=file.iloc[i+1,3]
                    selected_joint4=file.iloc[i+1,4]

            return selected_joint1, selected_joint2, selected_joint3, selected_joint4
        
        selected_point_scoord1, selected_point_scoord2, selected_point_scoord3, selected_point_scoord4 = get_points_of_interest_scoord(file_scoord)

        def same_coordinate(file : str, point1 : int, point2 : int, point3 : int, point4 : int, test ):

            global canvas
                   
            if canvas:
                canvas.get_tk_widget().grid_remove()

            if file[-3:]=='csv':
                data = pd.read_csv(file, header=None)
            elif file[-3:]=='lsx':
                data = pd.read_excel(file, header=None)

            num_rows, _ = data.shape

            AB_x=[]
            AB_y=[]

            CD_x=[]
            CD_y=[]

            for j in range (0, num_rows):

                # Stock coordinate from each point
                        
                x_a = data.iloc[j,3 * point1] 
                y_a = data.iloc[j,3 * point1 + 1]

                x_b = data.iloc[j,3 * point2] 
                y_b = data.iloc[j,3 * point2 + 1]

                x_c = data.iloc[j,3 * point3] 
                y_c = data.iloc[j,3 * point3 + 1]

                x_d = data.iloc[j,3 * point4] 
                y_d = data.iloc[j,3 * point4 + 1]

                if test == 'same_x':
                    AB_x.append(abs((x_b - x_a)/(x_b + x_a)))
                    CD_x.append(abs((x_d - x_c)/(x_d + x_c)))
                
                if test == 'same_y':
                    AB_y.append(abs((y_b - y_a)/(y_b + y_a)))
                    CD_y.append(abs((y_d - y_c)/(y_d + y_c)))
                    
                if test == "same_xy" :

                    AB_x.append(abs((x_b - x_a)/(x_b + x_a)))
                    CD_x.append(abs((x_d - x_c)/(x_d + x_c)))

                    AB_y.append(abs((y_b - y_a)/(y_b + y_a)))
                    CD_y.append(abs((y_d - y_c)/(y_d + y_c)))
                    
            
            fig = plt.figure(figsize=(12.7, 8.3))

            if len(AB_x)!=0:
                plt.plot(AB_x, color= 'blue', label = 'left x')
                plt.plot(CD_x, color = 'red', label = 'right x')  
            if len(AB_y)!=0:   
                plt.plot(AB_y, color= 'green', label = 'left y')
                plt.plot(CD_y, color = 'orange', label = 'right y') 
                
            canvas = FigureCanvasTkAgg(fig, master=frame_1)
            canvas.draw()
            canvas.get_tk_widget().grid()
        
        split_path=name.split("_")
        print(split_path)
        exercise_name = split_path[0]
        print(exercise_name)
        
        file_exercice=pd.read_excel('Features.xlsx',header=None, sheet_name='Exercices')

        num_row_ex, _ = file_exercice.shape

        print(file_exercice.iloc[5,2])

        for i in range (0, num_row_ex):
            if exercise_name==file_exercice.iloc[i,0]:
                if file_exercice.iloc[i,1]=='Same_coordinate':
                    if file_exercice.iloc[i,2] == selected_data:
                        name_test= file_exercice.iloc[i,4]

        same_coordinate(file_path1, int(selected_point_scoord1), int(selected_point_scoord2), int(selected_point_scoord3), int(selected_point_scoord4), name_test)

# Create the button to select the file

window.file_button1 = ctk.CTkButton(frame_1_left, text="Select the file you want to analyse", command=select_file1, width= 260, anchor='center')
window.file_button1.grid(row=0, column=0, padx=20, pady=(10, 10))


# Create the combobox to select the points

features=['Select the features ','Angle', 'Distance', 'Alignment', 'Parallelism','Same_coordinate']
    
window.combobox1 = ctk.CTkComboBox(frame_1_left, values=features, button_color= 'orange',command=select_feature)
window.combobox1.grid(row=1, column=0,padx=10, pady=(10, 10), sticky="ew")

window.combobox2 = ctk.CTkComboBox(frame_1_left, values=[''], button_color= 'orange',command=select_data)
window.combobox2.grid(row=2, column=0, padx=10, pady=(10, 10), sticky="ew")

## Create buttons to plot the curve and get the DTW distance

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

#############################################################################SCORES#############################################################################
window.label_2 = ctk.CTkLabel(frame_3_left, text="Working on it")
window.label_2.grid(row=0, column=0, padx=20, pady=20)

def select_exercise():
    global path_ex
    global selected_exercise
    selected_exercise = filedialog.askopenfilename()
    informations = selected_exercise.split("/")
    path_ex = informations[len(informations) - 1]
    if selected_exercise:
        window.score_button.configure(text=path_ex)

def give_score():

    global canvas_score
                   
    if canvas_score:
        canvas_score.get_tk_widget().grid_remove()

    list_of_data_score=[]

    split_path=path_ex.split("_")
    exercise_name = split_path[0]
    number = split_path[5]
    number = number[:1]
    
    file_exercice=pd.read_excel('Features.xlsx',header=None, sheet_name='Exercices')

    num_row_ex, _ = file_exercice.shape

    fig_score, ax =plt.subplots(1,1,figsize = (12.7, 8.3))
    column_labels=["Name of exercise", "Feature", "Data", "Mark"]
    data=[[None] * (len(column_labels)) for _ in range(10)]

    c=0

    for i in range (0, num_row_ex):
        if exercise_name==file_exercice.iloc[i,0]:
            if int(number) == file_exercice.iloc[i,3]:

                #Stock corresponding values from the excel

                good_feature = file_exercice.iloc[i,1]
                good_data = file_exercice.iloc[i,2]
                required_value= file_exercice.iloc[i,4]
                tolerance=file_exercice.iloc[i,5]

                #Find the right list of data

                if len(list_of_data_score) != 0 : 
                    list_of_data_score=[]

                for feature in ['Angle', 'Distance', 'Alignment', 'Parallelism']:
                    if good_feature == feature:

                        corresponding_file=pd.read_excel('Features.xlsx',header=None, sheet_name=good_feature)
                        num_rows_f, _ = corresponding_file.shape
                    
                        for i in range (1, num_rows_f):
                            list_of_data_score.append(corresponding_file.iloc[i,0])

                if good_feature=='Angle':
                    
                    def get_points_of_interest(file):

                        selected_joint1 = []
                        selected_joint2 = []
                        selected_joint3 = []
                    
                        for i,angle in enumerate(list_of_data_score):
                            if good_data==angle:
                                selected_joint1=file.iloc[i+1,1]
                                selected_joint2=file.iloc[i+1,2]
                                selected_joint3=file.iloc[i+1,3]

                        return selected_joint1, selected_joint2, selected_joint3
                
                    angle1, angle2, angle3 = get_points_of_interest(corresponding_file)
                    
                    def score_angle(csv1 : str,  joint1 : int, joint2 : int, joint3 : int, angle_to_analyse : int):
    
                        if csv1[-3:]=='csv':
                                data1 = pd.read_csv(csv1, header=None)
                        elif csv1[-3:]=='lsx':
                                data1 = pd.read_excel(csv1, header=None)

                        angles = []

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

                            radians = np.arctan2((AB - BC), (1 + AB * BC))
                            angle = 180 - np.abs(radians*180.0/np.pi)
                            
                            angles.append(angle)


                        inf_interval = ((100-tolerance)/100)*angle_to_analyse
                        sup_interval = ((100+tolerance)/100)*angle_to_analyse
                        count=0
                        
                        for angle in angles:
                            if (angle>=inf_interval) and (angle<=sup_interval) :
                                count+=1

                        return (count/num_rows1)*100
                    
                    mark_angle=score_angle(selected_exercise,int(angle1), int(angle2), int(angle3), required_value)

                    data[c][0]=path_ex
                    data[c][1]=good_feature
                    data[c][2]=good_data
                    data[c][3]=mark_angle

                    c=c+1

    df=pd.DataFrame(data,columns=column_labels)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values,
            colLabels=df.columns,
            colColours =["yellow"] * 4,
            loc="center")

    canvas_score = FigureCanvasTkAgg(fig_score, master=frame_3)
    canvas_score.draw()
    canvas_score.get_tk_widget().grid()

window.score_button = ctk.CTkButton(frame_3_left, text="Select the file you want to analyse",command=select_exercise, anchor='center')
window.score_button.grid(row=1, column=0,padx=10, pady=(10, 10))

window.show_video = ctk.CTkButton(frame_3_left, text="Show the score", command=give_score)
window.show_video.grid(row=2, column=0, padx=20, pady=(10, 10))

########################################################################################################################################################################


# close the window when we click on the cross

def close_window():
    window.destroy()
    window.quit()
window.protocol("WM_DELETE_WINDOW", close_window)

# Execution of the main loop
window.mainloop()