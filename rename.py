import re
import os
import glob


def rename(files: list[str], old_name: str, new_name: str, extention: str):
    '''
    :param files: Files paths to be renamed
    :param extention: Extension of the file (npy, xlsx, csv...)
    :return:
    '''
    regex = fr"^(.*?)_(\d+)_ID_{old_name}_cam_(.*?)(\d+)\.{extention}$"
    for file in files:
        match = re.match(regex, file)
        if match:
            new_fname = re.sub(regex, r"\1_\2_ID_" + new_name + r"_cam_\4."+ extention, file)
            os.rename(file, new_fname)
            print(f"Renamed {file} to {new_fname}")
        else:
            print(f"No match for file {file}")

files = glob.glob("C:/Users/Salomé/Desktop/OpenPoseData/*1 salome*.npy")

#print(f"Found {len(files)} files to check")

rename(files,"1 salome", "Bicycle", "npy")
'''
Usage example :
We get all mp4 files in /Volumes/USB/NicolasVideos/
Then we replace the name by Mountain : Video `BRGM_1686573510_ID_Nicolas_cam_4.mp4`
will be renamed `BRGM_1686573510_ID_Mountain_cam_4.mp4` by using these lines :

files = glob.glob("/Volumes/USB/NicolasVideos/*.mp4")
rename(files, "Mountain")
'''