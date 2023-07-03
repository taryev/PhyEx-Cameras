import pandas as pd
import numpy as np


class MediapipeData:
    columns_names = [
            "nose_x",
            "nose_y",
            "nose_z",
            "left_eye_inner_x",
            "left_eye_inner_y",
            "left_eye_inner_z",
            "left_eye_x",
            "left_eye_y",
            "left_eye_z",
            "left_eye_outer_x",
            "left_eye_outer_y",
            "left_eye_outer_z",
            "right_eye_inner_x",
            "right_eye_inner_y",
            "right_eye_inner_z",
            "right_eye_x",
            "right_eye_y",
            "right_eye_z",
            "right_eye_outer_x",
            "right_eye_outer_y",
            "right_eye_outer_z",
            "left_ear_x",
            "left_ear_y",
            "left_ear_z",
            "right_ear_x",
            "right_ear_y",
            "right_ear_z",
            "mouth_left_x",
            "mouth_left_y",
            "mouth_left_z",
            "mouth_right_x",
            "mouth_right_y",
            "mouth_right_z",
            "left_shoulder_x",
            "left_shoulder_y",
            "left_shoulder_z",
            "right_shoulder_x",
            "right_shoulder_y",
            "right_shoulder_z",
            "left_elbow_x",
            "left_elbow_y",
            "left_elbow_z",
            "right_elbow_x",
            "right_elbow_y",
            "right_elbow_z",
            "left_wrist_x",
            "left_wrist_y",
            "left_wrist_z",
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "left_pinky_x",
            "left_pinky_y",
            "left_pinky_z",
            "right_pinky_x",
            "right_pinky_y",
            "right_pinky_z",
            "left_index_x",
            "left_index_y",
            "left_index_z",
            "right_index_x",
            "right_index_y",
            "right_index_z",
            "left_thumb_x",
            "left_thumb_y",
            "left_thumb_z",
            "right_thumb_x",
            "right_thumb_y",
            "right_thumb_z",
            "left_hip_x",
            "left_hip_y",
            "left_hip_z",
            "right_hip_x",
            "right_hip_y",
            "right_hip_z",
            "left_knee_x",
            "left_knee_y",
            "left_knee_z",
            "right_knee_x",
            "right_knee_y",
            "right_knee_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
            "left_heel_x",
            "left_heel_y",
            "left_heel_z",
            "right_heel_x",
            "right_heel_y",
            "right_heel_z",
            "left_foot_index_x",
            "left_foot_index_y",
            "left_foot_index_z",
            "right_foot_index_x",
            "right_foot_index_y",
            "right_foot_index_z",
            "neck_x",
            "neck_y",
            "neck_z",
            "midhip_x",
            "midhip_y",
            "midhip_z"
        ]

    def __init__(self, path: str):
        self.path = path
        self.data = pd.read_csv(path, header=None)
        for col in range(0,6):
            self.data[f'{col}'] = np.nan
        self.data.columns = self.columns_names
        self.data['neck_x'] = (self.data['left_shoulder_x']+self.data['right_shoulder_x'])/2
        self.data['neck_y'] = (self.data['left_shoulder_y']+self.data['right_shoulder_y'])/2
        self.data['neck_z'] = (self.data['left_shoulder_z']+self.data['right_shoulder_z'])/2
        self.data['midhip_x'] = (self.data['left_hip_x']+self.data['left_hip_x'])/2
        self.data['midhip_y'] = (self.data['left_hip_y']+self.data['left_hip_y'])/2
        self.data['midhip_z'] = (self.data['left_hip_z']+self.data['left_hip_z'])/2



class OpenposeData:
    columns_names = [
        "nose_x",
        "nose_y",
        "neck_x",
        "neck_y",
        "rshoulder_x",
        "rshoulder_y",
        "relbow_x",
        "relbow_y",
        "rwrist_x",
        "rwrist_y",
        "lshoulder_x",
        "lshoulder_y",
        "lelbow_x",
        "lelbow_y",
        "lwrist_x",
        "lwrist_y",
        "midhip_x",
        "midhip_y",
        "rhip_x",
        "rhip_y",
        "rknee_x",
        "rknee_y",
        "rankle_x",
        "rankle_y",
        "lhip_x",
        "lhip_y",
        "lknee_x",
        "lknee_y",
        "lankle_x",
        "lankle_y",
        "reye_x",
        "reye_y",
        "leye_x",
        "leye_y",
        "rear_x",
        "rear_y",
        "lear_x",
        "lear_y",
        "lbigtoe_x",
        "lbigtoe_y",
        "lsmalltoe_x",
        "lsmalltoe_y",
        "lheel_x",
        "lheel_y",
        "rbigtoe_x",
        "rbigtoe_y",
        "rsmalltoe_x",
        "rsmalltoe_y",
        "rheel_x",
        "rheel_y",
    ]

    def __init__(self, path: str):
        self.path = path
        self.data_npy = np.load(path)
        dataframe = pd.DataFrame(columns = self.columns_names)
        frame_count = self.data_npy.shape[0]
        for i in range(0, frame_count):
            frame = self.data_npy[i][:,0:2]
            frame = np.transpose(frame)
            frame = np.ravel(frame, order='F')
            tmp_df = pd.DataFrame([frame], columns=self.columns_names)
            dataframe = pd.concat([dataframe, tmp_df], ignore_index=True)
        self.data = dataframe


'''
# Usages examples
mp = MediapipeData("/Users/quentinveyrat/Desktop/NicolasCSV/SMSQ_1686575947_ID_Nicolas_cam_4.csv")
op = OpenposeData("/Users/quentinveyrat/Downloads/OpenPoseData/BRK4_1686654805_ID_Coline_cam_4.npy")

# dataframe.data['LANDMARK_NAME']['FRAME']

nosex = mp.data['nose_x'][0]
neckz = mp.data['neck_z'][45]

rightbigtoex = op.data['rbigtoe_x'][73]

'''