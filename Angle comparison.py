
import pandas as pd
import math
import numpy as np

chemin_fichier2 = r"C:\Users\Salomé\Desktop\STAGE M1\OLDL_1684482294_ID_julie_cam_4.csv"
chemin_fichier1 =r"C:\Users\Salomé\Desktop\STAGE M1\OLDL_1684851718_ID_1_salome_cam_4.csv"
def angle(csv_file):



    # Colonnes à extraire du premier fichier CSV
    colonnes_fichier1 = ['AH', 'AI', 'BR', 'BS', 'CD', 'CE']

    # Colonnes à extraire du deuxième fichier CSV
    colonnes_fichier2 = ['AK', 'AL', 'BU', 'BV', 'CG', 'CH']

    # Fonction pour extraire les colonnes spécifiées d'un fichier CSV
    def extraire_colonnes_csv(file, colonnes):
        dataframe = pd.read_csv(file, header=1, names=colonnes)
        colonnes_extraites = dataframe[colonnes]
        return colonnes_extraites

    # Fonction pour calculer l'angle entre 3 points


        # Calcul des longueurs des côtés du triangle
        cote_a = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        cote_b = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        cote_c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        # Calcul de l'angle en radians
        angle_rad = np.arctan2((cote_a - cote_c ) , (1+cote_a*cote_c))

        # Conversion de l'angle en degrés
        angle_deg = math.degrees(angle_rad)

        return 180-angle_deg

    # Extraction des colonnes du premier fichier CSV
    donnees_fichier1 = extraire_colonnes_csv(chemin_fichier1, colonnes_fichier1)

    # Calcul des angles pour chaque ligne du premier fichier CSV
    angles_fichier1 = []
    for index, ligne in donnees_fichier1.iterrows():
        x1, y1, x2, y2, x3, y3 = ligne
        angle = calculer_angle(x1, y1, x2, y2, x3, y3)
        angles_fichier1.append(angle)

    # Affichage des données extraites
    print("Angles calculés pour chaque ligne du premier fichier CSV :")


    for angle in angles_fichier1:
        print(angle)
        print()  # Ajouter un saut de ligne après chaque angle


#angle( r"C:\Users\Salomé\Desktop\STAGE M1\OLDL_1684851718_ID_1_salome_cam_4.csv")



import pandas as pd

def extraire_colonnes_csv(file, colonnes):
    try:
        dataframe = pd.read_csv(file)
        colonnes_extraites = dataframe.loc[:, colonnes]
        print(colonnes_extraites)
    except Exception as e:
        print("Une erreur s'est produite lors de l'extraction des colonnes :", e)

extraire_colonnes_csv(r"C:\Users\Salomé\Desktop\STAGE M1\OLDL_1684851718_ID_1_salome_cam_4.csv", ['AH', 'AI', 'BR', 'BS', 'CD', 'CE'])

#code doesnt work, colomn extraction not ok.




