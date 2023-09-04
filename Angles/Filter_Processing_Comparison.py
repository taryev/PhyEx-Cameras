import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy import signal
import pandas as pd
import glob
import os
import math


def anglefilt(csv, npy, j11, j12, j13, j21, j22, j23,txt, ang, tol,exe,cam):
    data1=pd.read_csv(csv)
    angles1 = []
    num_rows1 = data1.shape[0]
    output_dir = os.path.dirname(csv)
    base_name = os.path.splitext(csv)[0]
    for j in range(num_rows1):
        try:
            x_a1 = data1.iloc[j, 3*j11]
            y_a1 = data1.iloc[j, 3*j11+1]
            x_b1 = data1.iloc[j, 3*j12]
            y_b1 = data1.iloc[j, 3*j12+1]
            x_c1 = data1.iloc[j, 3*j13]
            y_c1 = data1.iloc[j, 3*j13+1]

            if np.all(x_b1 != x_a1):
                AB = (y_b1 - y_a1) / (x_b1 - x_a1)
            else:
                AB = 0.0
            if np.all(x_b1 != x_c1):
                BC = (y_c1 - y_b1) / (x_c1 - x_b1)
            else:
                BC = 0.0

            radians1 = math.atan2((AB - BC), (1 + AB * BC))
            angle1 = np.round(180 - np.abs(radians1 * 180.0 / np.pi), 2)
            angles1 = np.append(angles1, angle1)
        except (ZeroDivisionError, RuntimeWarning):
            angles1 = np.append(angles1, np.nan)


    data2=np.load(npy)
    angles2 = []
    num_rows2 = data2.shape[0]

    for j in range(num_rows2):
        try:
            x_a2 = data2[j, j21, 0]
            y_a2 = data2[j, j21, 1]
            x_b2 = data2[j, j22, 0]
            y_b2 = data2[j, j22, 1]
            x_c2 = data2[j, j23, 0]
            y_c2 = data2[j, j23, 1]

            if np.all(x_b2 != x_a2):
                AB = (y_b2 - y_a2) / (x_b2 - x_a2)
            else:
                AB = 0.0
            if np.all(x_b2 != x_c2):
                BC = (y_c2 - y_b2) / (x_c2 - x_b2)
            else:
                BC = 0.0

            radians2 = math.atan2((AB - BC), (1 + AB * BC))
            angle2 = np.round(180 - np.abs(radians2 * 180.0 / np.pi), 2)
            angles2 = np.append(angles2, angle2)
        except (ZeroDivisionError, RuntimeWarning):
            angles2 = np.append(angles2, np.nan)


    angles_filt1=[]
    angles_filt2=[]
    if num_rows1 != num_rows2:
        if num_rows1 < num_rows2:
            l = num_rows1
            m = num_rows2
            while (l <= m):
                angles1 = np.append(angles1, np.nan)
                angles_filt1 = np.append(angles_filt1, np.nan)
                l += 1
        if num_rows1 > num_rows2:
            l = num_rows1
            m = num_rows2
            while (m <= l):
                angles2 = np.append(angles2, np.nan)
                angles_filt2 = np.append(angles_filt2, np.nan)
                m += 1
    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    angles_filt1 = signal.savgol_filter(angles1, N1, w1)
    angles_filt2 = signal.savgol_filter(angles2, N1, w1)
    diff_filt = []
    x = 0
    a = len(angles_filt1) - 1
    output_dir = os.path.dirname(csv)
    diff_filt = []

    output_filename = os.path.join(output_dir, f'angle_{txt}_{exe}_{cam}.txt')

    try:
        with open(output_filename, 'w') as f:
            while x < a:
                d2 = angles_filt1[x] - angles_filt2[x]
                diff_filt.append(d2)

                f.write(f'Angle = {ang}, Tolerance = {tol}\n')
                try:
                    if angles_filt1[x] <= (ang + tol) and angles_filt1[x] >= (ang - tol):
                        f.write(f'CSV Angle is respected: {angles_filt1[x]}\n')
                    else:
                        f.write(f'CSV Angle is not respected: {angles_filt1[x]}\n')
                    if angles_filt2[x] <= (ang + tol) and angles_filt2[x] >= (ang - tol):
                        f.write(f'NPY Angle is respected: {angles_filt2[x]}\n')
                    else:
                        f.write(f'NPY Angle is not respected: {angles_filt2[x]}\n')
                except (ZeroDivisionError, RuntimeWarning):
                    f.write(f'CSV Angle calculation error for: {angles_filt1[x]}\n')
                    f.write(f'NPY Angle calculation error for: {angles_filt2[x]}\n')

                x += 1
    except OSError as e:
        print(f"Error writing to file: {e}")
    plt.figure()
    up = ang + tol
    down = ang - tol
    n = 0
    green1 = np.empty_like(angles_filt1)
    red1 = np.empty_like(angles_filt1)
    green2 = np.empty_like(angles_filt2)
    red2 = np.empty_like(angles_filt2)
    for n in range(a):
        if angles_filt1[n] <= up and angles_filt1[n] >= down:
            green1[n] = angles_filt1[n]
            red1[n] = np.nan
        else:
            green1[n] = np.nan
            red1[n] = np.nan
        if angles_filt2[n] <= up and angles_filt2[n] >= down:
            green2[n] = angles_filt2[n]
            red2[n] = np.nan
        else:
            green2[n] = np.nan
            red2[n] = np.nan

    plt.subplot(131)
    plt.plot(angles1, color='blue', label='CSV signal')
    plt.plot(angles_filt1, color='orange', label='CSV filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(132)
    plt.plot(angles2, color='blue', label='NPY signal')
    plt.plot(angles_filt2, color='orange', label='NPY filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    #plt.subplot(233)
    #plt.plot(green1, color='green', label='Interval Respected')
    #plt.plot(red1, color='red', label='Interval Not Respected')
    #plt.title('CSV Angle Adequacy')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    #plt.subplot(234)
    #plt.plot(green2, color='green', label='Interval Respected')
    #plt.plot(red2, color='red', label='Interval Not Respected')
    #plt.title('NPY Angle Adequacy')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(133)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'{txt}_{exe}')
    plt.tight_layout()
    figure_filename = os.path.join(output_dir, f'angle_{txt}_{exe}_{cam}.png')
    plt.savefig(figure_filename)
    plt.show()
    print(data2.shape)
def alignfilt3(file1, file2, p11,p12,p13,p21,p22,p23,txt,tol,exe,cam):
    tol=float(tol)
    data1=pd.read_csv(file1)
    data2=np.load(file2)
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]
    distancesAB1=[], distancesBC1=[],distancesAC1=[]
    distancesAB2=[], distancesBC2=[],distancesAC2=[]
    alignment_npy=[]
    alignment_csv=[]
    for j in range(num_rows1):
        try:
            x_a1 = data1[j, p11, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a1 = data1[j, p11, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b1 = data1[j, p12, 0]
            y_b1 = data1[j, p12, 1]
            x_c1 = data1[j, p13, 0]
            y_c1 = data1[j, p13, 1]
            if np.all(x_c1 != x_a1):
                AC1 = (y_c1 - y_a1) / (x_c1 - x_a1)  # calculating the segments
            else:
                AC1 = 0.0
            if np.all(x_a1 != x_b1):
                AB1 = (y_a1 - y_b1) / (x_a1 - x_b1)
            else:
                AB1 = 0.0
            if np.all(x_c1 != x_b1):
                BC1 = (y_c1 - y_b1) / (x_c1 - y_b1)
            else:
                BC1 = 0.0
            distancesAC1 = np.append(distancesAC1, AC1)
            distancesAB1 = np.append(distancesAB1, AB1)
            distancesBC1 = np.append(distancesBC1, BC1)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAC1 = np.append(distancesAC1, np.nan)
            distancesAB1 = np.append(distancesAB1, np.nan)
            distancesBC1 = np.append(distancesBC1, np.nan)

    for j in range(num_rows2):
        try:
            x_a2 = data2[j, p21, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a2 = data2[j, p21, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b2 = data2[j, p22, 0]
            y_b2 = data2[j, p22, 1]
            x_c2 = data2[j, p23, 0]
            y_c2 = data2[j, p23, 1]
            if np.all(x_c2 != x_a2):
                AC2 = (y_c2 - y_a2) / (x_c2 - x_a2)  # calculating the segments
            else:
                AC2 = 0.0
            if np.all(x_a2 != x_b2):
                AB2 = (y_a2 - y_b2) / (x_a2 - x_b2)
            else:
                AB2 = 0.0
            if np.all(x_c2!= x_b2):
                BC2=(y_c2-y_b2)/(x_c2-y_b2)
            else:
                BC2=0.0
            distancesAC2 = np.append(distancesAC2, AC2)
            distancesAB2 = np.append(distancesAB2, AB2)
            distancesBC2 = np.append(distancesBC2, BC2)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAC2 = np.append(distancesAC2, np.nan)
            distancesAB2 = np.append(distancesAB2, np.nan)
            distancesBC2 = np.append(distancesBC2, np.nan)
    k=0

    while k < num_rows1:
        lat3csv=(distancesAB1[k]+distancesAC1[k]+distancesBC1[k])/3
        index3csv=math.sqrt(3)*lat3csv*lat3csv/4
        scsv=(distancesAB1[k]+distancesAC1[k]+distancesBC1[k])/2
        areacsv=math.sqrt(scsv*(scsv-distancesAB1)*(scsv-distancesBC1)*(scsv-distancesAC1))
        alignment_csv=np.append(alignment_csv,1-areacsv/index3csv)

        lat3npy=(distancesAB2[k]+distancesBC2[k]+distancesAC2[k])/3
        index3npy=math.sqrt(3)*lat3npy*lat3npy/4
        snpy=(distancesAB2[k]+distancesAC2[k]+distancesBC2[k])/2
        areanpy=math.sqrt(snpy*(snpy-distancesAB2)*(snpy-distancesBC2)*(snpy-distancesAC2))
        alignment_npy=np.append(alignment_npy,1-areanpy/index3npy)

    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    afiltcsv=signal.savgol_filter(alignment_csv,N1,w1)
    afiltnpy=signal.savgol_filter(alignment_npy,N1,w1)

    x = 0
    a = len(afiltcsv)
    diff_filt=[]
    while x < a:
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        d= afiltcsv[x]-afiltnpy[x]
        diff_filt = np.append(diff_filt, d)

        # Writing to the output file for each angle
        output_filename = f'alignment3_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'{txt}, Tolerance = {tol}\n')
        f = open(output_filename, 'a')
        with open(output_filename, 'a') as f:
            try:
                if afiltcsv[x] >= (1- tol/100):
                    f.write(f'CSV Alignment is respected: {afiltcsv[x]}\n')
                else:
                    f.write(f'CSV Alignment is not respected: {afiltcsv[x]}\n')
                if afiltnpy[x] >= (1- tol/100):
                    f.write(f'NPY Alignment is respected: {afiltnpy[x]}\n')
                else:
                    f.write(f'NPY Alignment is not respected: {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference: {diff_filt[x]}\n')
            except (ZeroDivisionError, RuntimeWarning):
                f.write(f'CSV Alignment calculation error for:{x} {afiltcsv[x]}\n')
                f.write(f'NPY Alignment calculation error for:{x} {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference calculation error for {x} {diff_filt[x]}\n')
        x += 1

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()
    thresh=1-tol/100
    n = 0
    red1 = []
    green1 = []
    red2 = []
    green2 = []
    while n < a:
        if afiltcsv[n] <= thresh:
            green1 = np.append(green1, afiltcsv[n])
            red1 = np.append(red1, np.nan)
        else:
            green1 = np.append(green1, np.nan)
            red1 = np.append(red1, np.nan)
        if afiltnpy[n] <= thresh:
            green2 = np.append(green2, afiltnpy[n])
            red2 = np.append(red2, np.nan)
        else:
            green2 = np.append(green2, np.nan)
            red2 = np.append(red2, np.nan)

    plt.subplot(151)
    plt.plot(alignment_csv, color='blue', label='CSV signal')
    plt.plot(afiltcsv, color='orange', label='CSV filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(152)
    plt.plot(alignment_npy, color='blue', label='NPY signal')
    plt.plot(afiltnpy, color='orange', label='NPY filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(153)
    plt.plot(green1, color='green', label='Tolerance Respected')
    plt.plot(red1, color='red', label='Tolerance Not Respected')
    plt.title('CSV Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(154)
    plt.plot(green2, color='green', label='Tolerance Respected')
    plt.plot(red2, color='red', label='Tolerance Not Respected')
    plt.title('NPY Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(155)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'align3_{txt}_{base_name}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'align3_{txt}_{base_name}_{exe}_{cam}.png'))

def alignfilt4(file1, file2, p11,p12,p13,p14,p21,p22,p23,p24,txt,tol,exe,cam):
    tol=float(tol)
    data1=pd.read_csv(file1)
    data2=np.load(file2)
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]
    distancesAB1=[], distancesBC1=[],distancesAC1=[],distancesCD1=[],distancesAD1=[]
    distancesAB2=[], distancesBC2=[],distancesAC2=[],distancesCD2=[],distancesAD2=[]
    alignment_npy=[]
    alignment_csv=[]
    for j in range(num_rows1):
        try:
            x_a1 = data1[j, p11, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a1 = data1[j, p11, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b1 = data1[j, p12, 0]
            y_b1 = data1[j, p12, 1]
            x_c1 = data1[j, p13, 0]
            y_c1 = data1[j, p13, 1]
            x_d1 = data1[j, p14, 0]
            y_d1 = data1[j, p14, 1]
            if np.all(x_d1 != x_a1):
                AD1 = (y_d1 - y_a1) / (x_d1 - x_a1)  # calculating the segments
            else:
                AD1 = 0.0
            if np.all(x_a1 != x_b1):
                AB1 = (y_a1 - y_b1) / (x_a1 - x_b1)
            else:
                AB1 = 0.0
            if np.all(x_c1 != x_b1):
                BC1 = (y_c1 - y_b1) / (x_c1 - y_b1)
            else:
                BC1 = 0.0
            if np.all(x_c1 !=x_d1):
                CD1 = (y_d1-y_c1)/(x_d1-x_c1)
            else:
                CD1 = 0.0
            if np.all(x_c1 != x_a1):
                AC1 = (y_c1 - y_a1) / (x_c1 - x_a1)  # calculating the segments
            else:
                AC1 = 0.0
            distancesAD1 = np.append(distancesAD1, AD1)
            distancesAB1 = np.append(distancesAB1, AB1)
            distancesBC1 = np.append(distancesBC1, BC1)
            distancesCD1 = np.append(distancesCD1, CD1)
            distancesAC1 = np.append(distancesAC1, AC1)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAD1 = np.append(distancesAD1, np.nan)
            distancesAB1 = np.append(distancesAB1, np.nan)
            distancesBC1 = np.append(distancesBC1, np.nan)
            distancesCD1 = np.append(distancesCD1, np.nan)
            distancesAC1 = np.append(distancesAC1, np.nan)

    for j in range(num_rows2):
        try:
            x_a2 = data2[j, p21, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a2 = data2[j, p21, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b2 = data2[j, p22, 0]
            y_b2 = data2[j, p22, 1]
            x_c2 = data2[j, p23, 0]
            y_c2 = data2[j, p23, 1]
            x_d2 = data2[j, p24, 0]
            y_d2 = data2[j, p24, 1]
            if np.all(x_d2 != x_a2):
                AD2 = (y_d2 - y_a2) / (x_d2 - x_a2)  # calculating the segments
            else:
                AD2 = 0.0
            if np.all(x_a2 != x_b2):
                AB2 = (y_a2 - y_b2) / (x_a2 - x_b2)
            else:
                AB2 = 0.0
            if np.all(x_c2!= x_b2):
                BC2=(y_c2-y_b2)/(x_c2-y_b2)
            else:
                BC2=0.0
            if np.all(x_c2 !=x_d2):
                CD2 = (y_d2-y_c2)/(x_d2-x_c2)
            else:
                CD2 = 0.0
            if np.all(x_c2 != x_a2):
                AC2 = (y_a2 - y_c2) / (x_a2 - x_c2)
            else:
                AC2 = 0.0

            distancesAD2 = np.append(distancesAD2, AD2)
            distancesAB2 = np.append(distancesAB2, AB2)
            distancesBC2 = np.append(distancesBC2, BC2)
            distancesCD2 = np.append(distancesCD2, CD2)
            distancesAC2 = np.append(distancesAC2, AC2)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAD2 = np.append(distancesAD2, np.nan)
            distancesAB2 = np.append(distancesAB2, np.nan)
            distancesBC2 = np.append(distancesBC2, np.nan)
            distancesCD2 = np.append(distancesCD2, np.nan)
            distancesAC2 = np.append(distancesAC2, np.nan)

    k = 0
    while k < num_rows1:
        lat4csv=(distancesAB1[k]+distancesAD1[k]+distancesBC1[k]+distancesCD1[k])/4
        index4csv=lat4csv*lat4csv
        scsv1=(distancesAB1[k]+distancesAC1[k]+distancesBC1[k])/2
        areacsv1=math.sqrt(scsv1*(scsv1-distancesAC1)*(scsv1-distancesBC1)*(scsv1-distancesAB1))
        scsv2 = (distancesCD1[k]+ distancesAD1[k]+ distancesAC1[k]) / 2
        areacsv2 = math.sqrt(scsv2 * (scsv2 - distancesCD1) * (scsv2 - distancesAC1) * (scsv2 - distancesCD2))
        areacsv=areacsv1+areacsv2
        alignment_csv=1-areacsv/index4csv

        lat4npy=(distancesAB2[k]+distancesBC2[k]+distancesAD2[k]+distancesCD2[k])/4
        index4npy=lat4npy*lat4npy
        snpy1=(distancesAB2[k]+distancesAC2[k]+distancesBC2[k])/2
        areanpy1=math.sqrt(snpy1*(snpy1-distancesAC2)*(snpy1-distancesBC2)*(snpy1-distancesAB2))
        snpy2 = (distancesBC2[k]+ distancesAC2[k]+ distancesCD2[k]) / 2
        areanpy2 = math.sqrt(snpy2 * (snpy2 - distancesAC2) * (snpy2 - distancesAD2) * (snpy2 - distancesCD2))
        areanpy=areanpy2+areanpy1
        alignment_npy=1-areanpy/index4npy
        k+=1
    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    afiltcsv=signal.savgol_filter(alignment_csv,N1,w1)
    afiltnpy=signal.savgol_filter(alignment_npy,N1,w1)

    x = 0
    a = len(afiltcsv)
    diff_filt = []
    while x < a:
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        d= afiltcsv[x]-afiltnpy[x]
        diff_filt = np.append(diff_filt, d)

        # Writing to the output file for each angle
        output_filename = f'alignment4_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'{txt}, Tolerance = {tol}\n')
        with open(output_filename, 'a') as f:
            try:
                if afiltcsv[x] >= (1 - tol / 100):
                    f.write(f'CSV Alignment is respected: {afiltcsv[x]}\n')
                else:
                    f.write(f'CSV Alignment is not respected: {afiltcsv[x]}\n')
                if afiltnpy[x] >= (1 - tol / 100):
                    f.write(f'NPY Alignment is respected: {afiltnpy[x]}\n')
                else:
                    f.write(f'NPY Alignment is not respected: {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference: {diff_filt[x]}\n')
            except (ZeroDivisionError, RuntimeWarning):
                f.write(f'CSV Alignment calculation error for:{x} {afiltcsv[x]}\n')
                f.write(f'NPY Alignment calculation error for:{x} {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference calculation error for {x} {diff_filt[x]}\n')
        x += 1

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()
    thresh=1-tol/100
    n = 0
    red1 = []
    green1 = []
    red2 = []
    green2 = []
    while n < a:
        if afiltcsv[n] <= thresh:
            green1 = np.append(green1, afiltcsv[n])
            red1 = np.append(red1, np.nan)
        else:
            green1 = np.append(green1, np.nan)
            red1 = np.append(red1, np.nan)
        if afiltnpy[n] <= thresh:
            green2 = np.append(green2, afiltnpy[n])
            red2 = np.append(red2, np.nan)
        else:
            green2 = np.append(green2, np.nan)
            red2 = np.append(red2, np.nan)

    plt.subplot(151)
    plt.plot(alignment_csv, color='blue', label='CSV signal')
    plt.plot(afiltcsv, color='orange', label='CSV filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(152)
    plt.plot(alignment_npy, color='blue', label='NPY signal')
    plt.plot(afiltnpy, color='orange', label='NPY filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(153)
    plt.plot(green1, color='green', label='Tolerance Respected')
    plt.plot(red1, color='red', label='Tolerance Not Respected')
    plt.title('CSV Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(154)
    plt.plot(green2, color='green', label='Tolerance Respected')
    plt.plot(red2, color='red', label='Tolerance Not Respected')
    plt.title('NPY Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(155)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'align3_{txt}_{base_name}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'align4_{txt}_{base_name}_{exe}_{cam}.png'))

def alignfilt5(file1, file2, p11, p12, p13, p14,p15, p21, p22, p23, p24,p25, txt, tol, exe, cam):
    tol = float(tol)
    data1 = pd.read_csv(file1)
    data2 = np.load(file2)
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]
    distancesAB1=[], distancesBC1=[],distancesAC1=[],distancesCD1=[],distancesAD1=[],distancesDE1=[],distancesAE1=[]
    distancesAB2=[], distancesBC2=[],distancesAC2=[],distancesCD2=[],distancesAD2=[],distancesDE2=[],distancesAE2=[]
    alignment_npy=[]
    alignment_csv=[]
    for j in range(num_rows1):
        try:
            x_a1 = data1[j, p11, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a1 = data1[j, p11, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b1 = data1[j, p12, 0]
            y_b1 = data1[j, p12, 1]
            x_c1 = data1[j, p13, 0]
            y_c1 = data1[j, p13, 1]
            x_d1 = data1[j, p14, 0]
            y_d1 = data1[j, p14, 1]
            x_e1 = data1[j, p15, 0]
            y_e1 = data1[j, p15, 1]
            if np.all(x_e1 != x_a1):
                AE1 = (y_e1 - y_a1) / (x_e1 - x_a1)  # calculating the segments
            else:
                AE1 = 0.0
            if np.all(x_a1 != x_b1):
                AB1 = (y_a1 - y_b1) / (x_a1 - x_b1)
            else:
                AB1 = 0.0
            if np.all(x_c1 != x_b1):
                BC1 = (y_c1 - y_b1) / (x_c1 - y_b1)
            else:
                BC1 = 0.0
            if np.all(x_c1 != x_d1):
                CD1 = (y_d1 - y_c1) / (x_d1 - x_c1)
            else:
                CD1 = 0.0
            if np.all(x_c1 != x_a1):
                AC1 = (y_c1 - y_a1) / (x_c1 - x_a1)  # calculating the segments
            else:
                AC1 = 0.0
            if np.all(x_d1 != x_e1):
                DE1=(y_d1-y_e1)/(x_d1-x_e1)
            else:
                DE1=0.0
            if np.all(x_d1 != x_a1):
                AD1 = (y_d1 - y_a1) / (x_d1 - x_a1)  # calculating the segments
            else:
                AD1 = 0.0


            distancesAE1 = np.append(distancesAE1, AE1)
            distancesAB1 = np.append(distancesAB1, AB1)
            distancesBC1 = np.append(distancesBC1, BC1)
            distancesCD1 = np.append(distancesCD1, CD1)
            distancesAC1 = np.append(distancesAC1, AC1)
            distancesDE1 = np.append(distancesDE1, DE1)
            distancesAD1 = np.append(distancesAD1, AD1)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAE1 = np.append(distancesAE1, np.nan)
            distancesAB1 = np.append(distancesAB1, np.nan)
            distancesBC1 = np.append(distancesBC1, np.nan)
            distancesCD1 = np.append(distancesCD1, np.nan)
            distancesAC1 = np.append(distancesAC1, np.nan)
            distancesDE1 = np.append(distancesDE1, np.nan)
            distancesAD1 = np.append(distancesAD1, np.nan)

    for j in range(num_rows2):
        try:
            x_a2 = data2[j, p21, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a2 = data2[j, p21, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b2 = data2[j, p22, 0]
            y_b2 = data2[j, p22, 1]
            x_c2 = data2[j, p23, 0]
            y_c2 = data2[j, p23, 1]
            x_d2 = data2[j, p24, 0]
            y_d2 = data2[j, p24, 1]
            x_e2 = data1[j, p25, 0]
            y_e2 = data1[j, p25, 1]
            if np.all(x_d2 != x_a2):
                AE2 = (y_e2 - y_a2) / (x_e2 - x_a2)  # calculating the segments
            else:
                AE2 = 0.0
            if np.all(x_a2 != x_b2):
                AB2 = (y_a2 - y_b2) / (x_a2 - x_b2)
            else:
                AB2 = 0.0
            if np.all(x_c2 != x_b2):
                BC2 = (y_c2 - y_b2) / (x_c2 - y_b2)
            else:
                BC2 = 0.0
            if np.all(x_c2 != x_d2):
                CD2 = (y_d2 - y_c2) / (x_d2 - x_c2)
            else:
                CD2 = 0.0
            if np.all(x_c2 != x_a2):
                AC2 = (y_a2 - y_c2) / (x_a2 - x_c2)
            else:
                AC2 = 0.0
            if np.all(x_d2 != x_e2):
                DE2=(y_d2-y_e2)/(x_d2-x_e2)
            else:
                DE2=0.0
            if np.all(x_d2 != x_a2):
                AD2 = (y_d2 - y_a2) / (x_d2 - x_a2)  # calculating the segments
            else:
                AD2 = 0.0

            distancesAE2 = np.append(distancesAE2, AE2)
            distancesAB2 = np.append(distancesAB2, AB2)
            distancesBC2 = np.append(distancesBC2, BC2)
            distancesCD2 = np.append(distancesCD2, CD2)
            distancesAC2 = np.append(distancesAC2, AC2)
            distancesDE2 = np.append(distancesDE2, DE2)
            distancesAD2 = np.append(distancesAD2, AD2)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAE2 = np.append(distancesAE2, np.nan)
            distancesAB2 = np.append(distancesAB2, np.nan)
            distancesBC2 = np.append(distancesBC2, np.nan)
            distancesCD2 = np.append(distancesCD2, np.nan)
            distancesAC2 = np.append(distancesAC2, np.nan)
            distancesDE2 = np.append(distancesDE2, np.nan)
            distancesAD2 = np.append(distancesAD2, np.nan)
    k=0
    while k < num_rows1:
        lat5csv = (distancesAB1[k]+ distancesAE1[k]+ distancesBC1[k]+distancesCD1[k]+distancesDE1[k]) / 5
        index5csv = lat5csv * lat5csv*math.sqrt(5*(5+2*math.sqrt(5)))/4
        scsv1 = (distancesAB1[k]+ distancesAC1[k]+ distancesBC1) / 2
        scsv2 = (distancesCD1[k]+ distancesAD1[k]+ distancesAC1) / 2
        scsv3 = (distancesAE1[k]+ distancesDE1[k]+ distancesAD1) / 2
        areacsv1 = math.sqrt(scsv1 * (scsv1 - distancesAB1) * (scsv1 - distancesBC1) * (scsv1 - distancesAC1))
        areacsv2 = math.sqrt(scsv2 * (scsv2 - distancesCD1) * (scsv2 - distancesAD1) * (scsv2 - distancesAC1))
        areacsv3 = math.sqrt(scsv3 * (scsv3 - distancesAD1) * (scsv3 - distancesDE1) * (scsv3 - distancesAE1))
        areacsv = areacsv1 + areacsv2+areacsv3
        alignment_csv = 1 - areacsv / index5csv

        lat5npy = (distancesAB2[k]+ distancesBC2[k]+ distancesAE2[k]+ distancesCD2[k]+distancesDE2[k]) / 5
        index5npy = lat5npy * lat5npy*math.sqrt(5*(5+2*math.sqrt(5)))/4
        snpy1 = (distancesAB2[k]+ distancesAC2[k]+ distancesBC2[k]) / 2
        snpy2 = (distancesBC2[k]+ distancesAC2[k]+ distancesCD2[k]) / 2
        snpy3 = (distancesAE2[k]+ distancesDE2[k]+ distancesAD2[k]) / 2
        areanpy1 = math.sqrt(snpy1 * (snpy1 - distancesAB2) * (snpy1 - distancesBC2) * (snpy1 - distancesAC2))
        areanpy2 = math.sqrt(snpy2 * (snpy2 - distancesCD2) * (snpy2 - distancesAD2) * (snpy2 - distancesAC2))
        areanpy3 = math.sqrt(snpy3 * (snpy3 - distancesAE2) * (snpy3 - distancesDE2) * (snpy3 - distancesAD2))
        areanpy = areanpy1 + areanpy2 + areanpy3
        alignment_npy = 1 - areanpy / index5npy
        k+=1
    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    afiltcsv = signal.savgol_filter(alignment_csv, N1, w1)
    afiltnpy = signal.savgol_filter(alignment_npy, N1, w1)

    x = 0
    a = len(afiltcsv)
    diff_filt = []
    while x < a:
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        d = afiltcsv[x] - afiltnpy[x]
        diff_filt = np.append(diff_filt, d)

        # Writing to the output file for each angle
        output_filename = f'alignment5_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'{txt}, Tolerance = {tol}\n')
        with open(output_filename, 'a') as f:
            try:
                if afiltcsv[x] >= (1 - tol / 100):
                    f.write(f'CSV Alignment is respected: {afiltcsv[x]}\n')
                else:
                    f.write(f'CSV Alignment is not respected: {afiltcsv[x]}\n')
                if afiltnpy[x] >= (1 - tol / 100):
                    f.write(f'NPY Alignment is respected: {afiltnpy[x]}\n')
                else:
                    f.write(f'NPY Alignment is not respected: {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference: {diff_filt[x]}\n')
            except (ZeroDivisionError, RuntimeWarning):
                f.write(f'CSV Alignment calculation error for:{x} {afiltcsv[x]}\n')
                f.write(f'NPY Alignment calculation error for:{x} {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference calculation error for {x} {diff_filt[x]}\n')
        x += 1

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()
    thresh = 1 - tol / 100
    n = 0
    red1 = []
    green1 = []
    red2 = []
    green2 = []
    while n < a:
        if afiltcsv[n] <= thresh:
            green1 = np.append(green1, afiltcsv[n])
            red1 = np.append(red1, np.nan)
        else:
            green1 = np.append(green1, np.nan)
            red1 = np.append(red1, np.nan)
        if afiltnpy[n] <= thresh:
            green2 = np.append(green2, afiltnpy[n])
            red2 = np.append(red2, np.nan)
        else:
            green2 = np.append(green2, np.nan)
            red2 = np.append(red2, np.nan)

    plt.subplot(151)
    plt.plot(alignment_csv, color='blue', label='CSV signal')
    plt.plot(afiltcsv, color='orange', label='CSV filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(152)
    plt.plot(alignment_npy, color='blue', label='NPY signal')
    plt.plot(afiltnpy, color='orange', label='NPY filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(153)
    plt.plot(green1, color='green', label='Tolerance Respected')
    plt.plot(red1, color='red', label='Tolerance Not Respected')
    plt.title('CSV Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(154)
    plt.plot(green2, color='green', label='Tolerance Respected')
    plt.plot(red2, color='red', label='Tolerance Not Respected')
    plt.title('NPY Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(155)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'align3_{txt}_{base_name}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'align5_{txt}_{base_name}_{exe}_{cam}.png'))

def alignfilt6(file1, file2, p11, p12, p13, p14,p15,p16, p21, p22, p23, p24,p25,p26, txt, tol, exe, cam):
    tol = float(tol)
    data1 = pd.read_csv(file1)
    data2 = np.load(file2)
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]
    distancesAB1=[], distancesBC1=[],distancesAC1=[],distancesCD1=[],distancesAD1=[],distancesDE1=[],distancesAE1=[]
    distancesAB2=[], distancesBC2=[],distancesAC2=[],distancesCD2=[],distancesAD2=[],distancesDE2=[],distancesAE2=[]
    alignment_npy=[]
    alignment_csv=[]
    distancesAF1=[], distancesEF1=[],distancesAF2=[],distancesEF2=[]
    for j in range(num_rows1):
        try:
            x_a1 = data1[j, p11, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a1 = data1[j, p11, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b1 = data1[j, p12, 0]
            y_b1 = data1[j, p12, 1]
            x_c1 = data1[j, p13, 0]
            y_c1 = data1[j, p13, 1]
            x_d1 = data1[j, p14, 0]
            y_d1 = data1[j, p14, 1]
            x_e1 = data1[j, p15, 0]
            y_e1 = data1[j, p15, 1]
            x_f1 = data1[j, p16, 0]
            y_f1 = data1[j, p16, 1]

            if np.all(x_e1 != x_a1):
                AE1 = (y_e1 - y_a1) / (x_e1 - x_a1)  # calculating the segments
            else:
                AE1 = 0.0
            if np.all(x_a1 != x_b1):
                AB1 = (y_a1 - y_b1) / (x_a1 - x_b1)
            else:
                AB1 = 0.0
            if np.all(x_c1 != x_b1):
                BC1 = (y_c1 - y_b1) / (x_c1 - y_b1)
            else:
                BC1 = 0.0
            if np.all(x_c1 != x_d1):
                CD1 = (y_d1 - y_c1) / (x_d1 - x_c1)
            else:
                CD1 = 0.0
            if np.all(x_c1 != x_a1):
                AC1 = (y_c1 - y_a1) / (x_c1 - x_a1)  # calculating the segments
            else:
                AC1 = 0.0
            if np.all(x_d1 != x_e1):
                DE1=(y_d1-y_e1)/(x_d1-x_e1)
            else:
                DE1=0.0
            if np.all(x_d1 != x_a1):
                AD1 = (y_d1 - y_a1) / (x_d1 - x_a1)  # calculating the segments
            else:
                AD1 = 0.0
            if np.all(x_f1 != x_a1):
                AF1=(y_f1-y_a1)/(x_f1-x_a1)
            else:
                AF1=0.0
            if np.all(x_e1 != x_f1):
                EF1 = (y_e1 - y_f1) / (x_e1 - x_f1)  # calculating the segments
            else:
                EF1 = 0.0

            distancesAE1 = np.append(distancesAE1, AE1)
            distancesAB1 = np.append(distancesAB1, AB1)
            distancesBC1 = np.append(distancesBC1, BC1)
            distancesCD1 = np.append(distancesCD1, CD1)
            distancesAC1 = np.append(distancesAC1, AC1)
            distancesDE1 = np.append(distancesDE1, DE1)
            distancesAD1 = np.append(distancesAD1, AD1)
            distancesEF1 = np.append(distancesEF1, EF1)
            distancesAF1 = np.append(distancesAF1, AF1)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAE1 = np.append(distancesAE1, np.nan)
            distancesAB1 = np.append(distancesAB1, np.nan)
            distancesBC1 = np.append(distancesBC1, np.nan)
            distancesCD1 = np.append(distancesCD1, np.nan)
            distancesAC1 = np.append(distancesAC1, np.nan)
            distancesDE1 = np.append(distancesDE1, np.nan)
            distancesAD1 = np.append(distancesAD1, np.nan)
            distancesEF1 = np.append(distancesEF1, np.nan)
            distancesAF1 = np.append(distancesAF1, np.nan)

    for j in range(num_rows2):
        try:
            x_a2 = data2[j, p21, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a2 = data2[j, p21, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b2 = data2[j, p22, 0]
            y_b2 = data2[j, p22, 1]
            x_c2 = data2[j, p23, 0]
            y_c2 = data2[j, p23, 1]
            x_d2 = data2[j, p24, 0]
            y_d2 = data2[j, p24, 1]
            x_e2 = data1[j, p25, 0]
            y_e2 = data1[j, p25, 1]
            x_f2 = data1[j, p26, 0]
            y_f2 = data1[j, p26, 1]

            if np.all(x_d2 != x_a2):
                AE2 = (y_e2 - y_a2) / (x_e2 - x_a2)  # calculating the segments
            else:
                AE2 = 0.0
            if np.all(x_a2 != x_b2):
                AB2 = (y_a2 - y_b2) / (x_a2 - x_b2)
            else:
                AB2 = 0.0
            if np.all(x_c2 != x_b2):
                BC2 = (y_c2 - y_b2) / (x_c2 - y_b2)
            else:
                BC2 = 0.0
            if np.all(x_c2 != x_d2):
                CD2 = (y_d2 - y_c2) / (x_d2 - x_c2)
            else:
                CD2 = 0.0
            if np.all(x_c2 != x_a2):
                AC2 = (y_a2 - y_c2) / (x_a2 - x_c2)
            else:
                AC2 = 0.0
            if np.all(x_d2 != x_e2):
                DE2=(y_d2-y_e2)/(x_d2-x_e2)
            else:
                DE2=0.0
            if np.all(x_d2 != x_a2):
                AD2 = (y_d2 - y_a2) / (x_d2 - x_a2)  # calculating the segments
            else:
                AD2 = 0.0
            if np.all(x_f2 != x_e2):
                EF2=(y_f2-y_e2)/(x_f2-x_e2)
            else:
                EF2=0.0
            if np.all(x_f2 != x_a2):
                AF2 = (y_f2 - y_a2) / (x_f2 - x_a2)  # calculating the segments
            else:
                AF2 = 0.0


            distancesAE2 = np.append(distancesAE2, AE2)
            distancesAB2 = np.append(distancesAB2, AB2)
            distancesBC2 = np.append(distancesBC2, BC2)
            distancesCD2 = np.append(distancesCD2, CD2)
            distancesAC2 = np.append(distancesAC2, AC2)
            distancesDE2 = np.append(distancesDE2, DE2)
            distancesAD2 = np.append(distancesAD2, AD2)
            distancesEF2 = np.append(distancesEF2, EF2)
            distancesAF2 = np.append(distancesAF2, AF2)
        except (ZeroDivisionError, RuntimeWarning):
            distancesAE2 = np.append(distancesAE2, np.nan)
            distancesAB2 = np.append(distancesAB2, np.nan)
            distancesBC2 = np.append(distancesBC2, np.nan)
            distancesCD2 = np.append(distancesCD2, np.nan)
            distancesAC2 = np.append(distancesAC2, np.nan)
            distancesDE2 = np.append(distancesDE2, np.nan)
            distancesAD2 = np.append(distancesAD2, np.nan)
            distancesEF2 = np.append(distancesEF2, np.nan)
            distancesAF2 = np.append(distancesAF2, np.nan)
    k=0
    while k < num_rows1:
        lat6csv = (distancesAB1[k]+distancesEF1[k]+distancesBC1[k]+distancesCD1[k]+distancesDE1[k]+distancesAF1[k]) / 6
        index6csv = lat6csv * lat6csv*3*math.sqrt(3)/2
        scsv1 = (distancesAB1[k]+distancesAC1[k]+distancesBC1[k]) / 2
        scsv2 = (distancesCD1[k]+distancesAD1[k]+distancesAC1[k]) / 2
        scsv3 = (distancesAE1[k]+distancesDE1[k]+distancesAD1[k]) / 2
        scsv4 = (distancesAE1[k]+distancesEF1[k]+distancesAF1[k]) / 2
        areacsv1 = math.sqrt(scsv1 * (scsv1 - distancesAB1) * (scsv1 - distancesBC1) * (scsv1 - distancesAC1))
        areacsv2 = math.sqrt(scsv2 * (scsv2 - distancesCD1) * (scsv2 - distancesAD1) * (scsv2 - distancesAC1))
        areacsv3 = math.sqrt(scsv3 * (scsv3 - distancesAD1) * (scsv3 - distancesDE1) * (scsv3 - distancesAE1))
        areacsv4 = math.sqrt(scsv4 * (scsv4 - distancesAE1) * (scsv4 - distancesAF1) * (scsv4 - distancesEF1))

        areacsv = areacsv1 + areacsv2+areacsv3+areacsv4
        alignment_csv = 1 - areacsv / index6csv

        lat6npy = (distancesAB2[k]+distancesBC2[k]+distancesEF2[k]+distancesCD2[k]+distancesDE2[k]+distancesAF2[k]) / 6
        index6npy = lat6npy * lat6npy*math.sqrt(3)/2
        snpy1 = (distancesAB2[k]+distancesAC2[k]+distancesBC2[k]) / 2
        snpy2 = (distancesBC2[k]+distancesAC2[k]+distancesCD2[k]) / 2
        snpy3 = (distancesAE2[k]+distancesDE2[k]+distancesAD2[k]) / 2
        snpy4 = (distancesAE2[k]+distancesEF2[k]+distancesAF2[k]) / 2
        areanpy1 = math.sqrt(snpy1 * (snpy1 - distancesAB2) * (snpy1 - distancesBC2) * (snpy1 - distancesAC2))
        areanpy2 = math.sqrt(snpy2 * (snpy2 - distancesCD2) * (snpy2 - distancesAD2) * (snpy2 - distancesAC2))
        areanpy3 = math.sqrt(snpy3 * (snpy3 - distancesAE2) * (snpy3 - distancesDE2) * (snpy3 - distancesAD2))
        areanpy4 = math.sqrt(snpy4 * (snpy4 - distancesAE2) * (snpy4 - distancesEF2) * (snpy4 - distancesAF2))

        areanpy = areanpy1 + areanpy2 + areanpy3 + areanpy4
        alignment_npy = 1 - areanpy / index6npy

    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    afiltcsv = signal.savgol_filter(alignment_csv, N1, w1)
    afiltnpy = signal.savgol_filter(alignment_npy, N1, w1)

    x = 0
    a = len(afiltcsv)
    diff_filt = []
    while x < a:
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        d = afiltcsv[x] - afiltnpy[x]
        diff_filt = np.append(diff_filt, d)

        # Writing to the output file for each angle
        output_filename = f'alignment6_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'{txt}, Tolerance = {tol}\n')
        with open(output_filename, 'a') as f:
            try:
                if afiltcsv[x] >= (1 - tol / 100):
                    f.write(f'CSV Alignment is respected: {afiltcsv[x]}\n')
                else:
                    f.write(f'CSV Alignment is not respected: {afiltcsv[x]}\n')
                if afiltnpy[x] >= (1 - tol / 100):
                    f.write(f'NPY Alignment is respected: {afiltnpy[x]}\n')
                else:
                    f.write(f'NPY Alignment is not respected: {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference: {diff_filt[x]}\n')
            except (ZeroDivisionError, RuntimeWarning):
                f.write(f'CSV Alignment calculation error for:{x} {afiltcsv[x]}\n')
                f.write(f'NPY Alignment calculation error for:{x} {afiltnpy[x]}\n')
                f.write(f'CSV-NPY difference calculation error for {x} {diff_filt[x]}\n')
        x += 1

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()
    thresh = 1 - tol / 100
    n = 0
    red1 = []
    green1 = []
    red2 = []
    green2 = []
    while n < a:
        if afiltcsv[n] <= thresh:
            green1 = np.append(green1, afiltcsv[n])
            red1 = np.append(red1, np.nan)
        else:
            green1 = np.append(green1, np.nan)
            red1 = np.append(red1, np.nan)
        if afiltnpy[n] <= thresh:
            green2 = np.append(green2, afiltnpy[n])
            red2 = np.append(red2, np.nan)
        else:
            green2 = np.append(green2, np.nan)
            red2 = np.append(red2, np.nan)

    plt.subplot(151)
    plt.plot(alignment_csv, color='blue', label='CSV signal')
    plt.plot(afiltcsv, color='orange', label='CSV filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(152)
    plt.plot(alignment_npy, color='blue', label='NPY signal')
    plt.plot(afiltnpy, color='orange', label='NPY filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(153)
    plt.plot(green1, color='green', label='Tolerance Respected')
    plt.plot(red1, color='red', label='Tolerance Not Respected')
    plt.title('CSV Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(154)
    plt.plot(green2, color='green', label='Tolerance Respected')
    plt.plot(red2, color='red', label='Tolerance Not Respected')
    plt.title('NPY Alignment Adequacy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(155)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'align3_{txt}_{base_name}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'align6_{txt}_{base_name}_{exe}_{cam}.png'))
# distance function
def distfilt(csv, npy,d11, d12, d13, d14, d21, d22, d23, d24,txt,exe,cam):
    data1=pd.read_csv(csv)
    data2=np.load(npy)
    distances1AC = []
    distances1BD = []
    distances2AC = []
    distances2BD = []
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]
    for j in range(num_rows1):
        try:
            x_a1 = data1[j, d11, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a1 = data1[j, d11, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b1 = data1[j, d12, 0]
            y_b1 = data1[j, d12, 1]
            x_c1 = data1[j, d13, 0]
            y_c1 = data1[j, d13, 1]
            x_d1 = data1[j, d14, 0]
            y_d1 = data1[j, d14, 1]
            if np.all(x_c1 != x_a1):
                AC = (y_c1 - y_a1) / (x_c1 - x_a1)  # calculating the segments
            else:
                AC = 0.0
            if np.all(x_d1 != x_b1):
                BD = (y_d1 - y_b1) / (x_d1 - x_b1)
            else:
                BD = 0.0
            distances1AC = np.append(distances1AC, AC)
            distances1BD = np.append(distances1BD, BD)
        except (ZeroDivisionError, RuntimeWarning):
            distances1AC = np.append(distances1AC, np.nan)
            distances1BD = np.append(distances1BD, np.nan)
    for j in range(num_rows2):
        try:
            x_a2 = data2[j, d21, 0]  # x,y,z dimensions since the data1 shape is [n,25,3]
            y_a2 = data2[j, d21, 1]
            # you need {x,y}_{a,b,c}n for every angle you need to calculate
            x_b2 = data2[j, d22, 0]
            y_b2 = data2[j, d22, 1]
            x_c2 = data2[j, d23, 0]
            y_c2 = data2[j, d23, 1]
            x_d2 = data2[j, d24, 0]
            y_d2 = data2[j, d24, 1]
            if np.all(x_c2 != x_a2):
                AC = (y_c2 - y_a2) / (x_c2 - x_a2)  # calculating the segments
            else:
                AC = 0.0
            if np.all(x_d2 != x_b2):
                BD = (y_d2 - y_b2) / (x_d2 - x_b2)
            else:
                BD = 0.0
            distances2AC = np.append(distances2AC, AC)
            distances2BD = np.append(distances2BD, BD)
        except (ZeroDivisionError, RuntimeWarning):
            distances2AC = np.append(distances2AC, np.nan)
            distances2BD = np.append(distances2BD, np.nan)

    max_length = max(len(distances1AC), len(distances2AC))
    distances1AC.extend([np.nan] * (max_length - len(distances1AC)))
    distances1BD.extend([np.nan] * (max_length - len(distances1BD)))
    distances2AC.extend([np.nan] * (max_length - len(distances2AC)))
    distances2BD.extend([np.nan] * (max_length - len(distances2BD)))

    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    dfilt1AC = signal.savgol_filter(distances1AC, N1, w1)
    dfilt2AC = signal.savgol_filter(distances2AC, N1, w1)
    dfilt1BD = signal.savgol_filter(distances1BD, N1, w1)
    dfilt2BD = signal.savgol_filter(distances2BD, N1, w1)

    x = 0
    a = len(dfilt1AC)
    diff_filt1 = []
    diff_filt2 = []
    while x < a:
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        d1 = dfilt1AC[x]-dfilt2AC[x]
        d2 = dfilt1BD[x]-dfilt2BD[x]

        diff_filt1 = np.append(diff_filt1, d1)
        diff_filt2 = np.append(diff_filt2, d2)

        # Writing to the output file for each angle
        output_filename = f'distance_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'Distance\n')
        with open(output_filename, 'a') as f:
            f.write(f'{x} AC (CSV): {dfilt1AC[x]}\n')
            f.write(f'{x} AC (NPY): {dfilt1AC[x]}\n')
            f.write(f'{x} BD (CSV): {dfilt1AC[x]}\n')
            f.write(f'{x} BD (CSV): {dfilt1AC[x]}\n')
            f.write(f'{x} Filter difference AC: {diff_filt1[x]}\n')
            f.write(f'{x} Filter Difference BD: {diff_filt2[x]}\n')

        x += 1

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()

    plt.subplot(151)
    plt.plot(distances1AC, color='blue', label='First distance')
    plt.plot(distances1BD, color='purple', label='Second distance')
    plt.plot(dfilt1AC, color='green', label='First dist filtered signal')
    plt.plot(dfilt1BD, color='orange', label='Second dist filtered signal')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(152)
    plt.plot(dfilt1AC, color='green', label='First dist filtered signal')
    plt.plot(dfilt1BD, color='orange', label='Second dist filtered signal')
    plt.title('Filtered CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(153)
    plt.plot(distances2AC, color='blue', label='First distance')
    plt.plot(distances2BD, color='purple', label='Second distance')
    plt.plot(dfilt2AC, color='green', label='First dist filtered signal')
    plt.plot(dfilt2BD, color='orange', label='Second dist filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(154)
    plt.plot(dfilt2AC, color='green', label='First dist filtered signal')
    plt.plot(dfilt2BD, color='orange', label='Second dist filtered signal')
    plt.title('Filtered NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)


    plt.subplot(155)
    plt.plot(diff_filt1, color='blue', label="CSV-NPY difference AC")
    plt.plot(diff_filt2, color='red', label="CSV-NPY difference BD")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'{txt}_{filename1}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'dist_{txt}_{base_name}_{exe}_{cam}_distance.png'))

# parallelism
def parafilt(csv, npy,mp11, mp12, mp13, mp14, mp21, mp22, mp23, mp24, txt, exe, cam):
    data1 = pd.read_csv(csv, header=None)
    data2 = np.load(npy)
    slope_diff_percentages_csv = []
    slope_diff_percentages_npy = []
    num_rows1, _ = data1.shape[0]
    num_rows2, _ = data2.shape[0]

    for j in range(num_rows1):
        if (j in data1.index):

            x_a = data1[j, mp11, 0]
            y_a = data1[j, mp11, 1]
            x_b = data1[j, mp12, 0]
            y_b = data1[j, mp12, 1]
            x_c = data1[j, mp13, 0]
            y_c = data1[j, mp13, 1]
            x_d = data1[j, mp14, 0]
            y_d = data1[j, mp14, 1]

            if (x_b - x_a) != 0 and (x_d - x_c) != 0:
                AB_slope = (y_b - y_a) / (x_b - x_a)
                CD_slope = (y_d - y_c) / (x_d - x_c)

                slope_diff_percentage = abs(AB_slope - CD_slope) / ((AB_slope + CD_slope) / 2) * 100
                slope_diff_percentages_csv.append(slope_diff_percentage)
    parallelism_values_csv = [abs(1 - diff / 100) for diff in slope_diff_percentages_csv]
    for j in range(num_rows2):
        if (j in data2.index):

            x_a = data2[j, mp21, 0]
            y_a = data2[j, mp21, 1]
            x_b = data2[j, mp22, 0]
            y_b = data2[j, mp22, 1]
            x_c = data2[j, mp23, 0]
            y_c = data2[j, mp23, 1]
            x_d = data2[j, mp24, 0]
            y_d = data2[j, mp24, 1]

            if (x_b - x_a) != 0 and (x_d - x_c) != 0:
                AB_slope = (y_b - y_a) / (x_b - x_a)
                CD_slope = (y_d - y_c) / (x_d - x_c)

                slope_diff_percentage = abs(AB_slope - CD_slope) / ((AB_slope + CD_slope) / 2) * 100
                slope_diff_percentages_npy.append(slope_diff_percentage)
    parallelism_values_npy = [abs(1 - diff / 100) for diff in slope_diff_percentages_npy]
    for j in range(num_rows1):
        if np.length(num_rows1) != np.length(num_rows2):
            if np.length(num_rows1) < np.length(num_rows2):
                l = np.length(num_rows1)
                m = np.length(num_rows2)
                while (l <= m):
                    parallelism_values_csv = np.append(parallelism_values_csv, np.nan)

                    l += 1
            if np.length(num_rows1) > np.length(num_rows2):
                l = np.length(num_rows1)
                m = np.length(num_rows2)
                while (m <= l):
                    paralelism_values_npy = np.append(parallelism_values_npy, np.nan)
                    m += 1

    N1 = 150  # window length, determined through manual trial
    w1 = 4  # polyorders value, determined through manual trial
    # Savitky-Golay filter
    parafiltcsv = signal.savgol_filter(parallelism_values_csv, N1, w1)
    parafiltnpy = signal.savgol_filter(parallelism_values_npy, N1, w1)

    a = np.length(num_rows1)
    x = 0
    diff_filt = []
    while x < a:
        d1 = parafiltcsv[x] - parafiltnpy[x]
        diff_filt = np.append(diff_filt, d1)
        output_dir = os.path.dirname(filename1)
        base_name = os.path.splitext(filename1)[0]

        # Writing to the output file for each angle
        output_filename = f'parallelism_{base_name}_{txt}_{exe}_{cam}.txt'
        f.write(f'Distance\n')
        with open(output_filename, 'a') as f:
            f.write(f'{x} Value (CSV): {parafiltcsv[x]}\n')
            f.write(f'{x} Value (NPY): {parafiltnpy[x]}\n')
            f.write(f'{x} Difference: {diff_filt[x]}\n')
        x += 1
    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    plt.figure()

    plt.subplot(131)
    plt.plot(parallelism_values_csv, color='blue', label='CSV parallelism')
    plt.plot(parafiltcsv, color='orange', label='CSV filtered parallelism')
    plt.title('CSV')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(132)
    plt.plot(parallelism_values_npy, color='green', label='First dist filtered signal')
    plt.plot(parafiltnpy, color='orange', label='Second dist filtered signal')
    plt.title('NPY')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.subplot(133)
    plt.plot(diff_filt, color='blue', label="CSV-NPY difference")
    plt.title('CSV NPY comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=1)

    plt.suptitle(f'{txt}_{filename1}')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f'para_{txt}_{base_name}_{exe}_{cam}_plot.png'))
# same coordinate
def samefilt(csv, npy, sd11, sd12, sd13, sd14, sd21, sd22, sd23, sd24, alignment_type, txt, exe, cam):
    data1=pd.read_csv(csv)
    data2=np.load(npy)
    a1=[]
    cxa1=[],cxb1=[],cxc1=[],cxd1=[],cxa2=[],cxb2=[],cxc2=[],cxd2=[]
    cya1 = [], cyb1 = [], cyc1 = [], cyd1 = [], cya2 = [], cyb2 = [], cyc2 = [], cyd2 = []
    num_rows1, _ = data1.shape[0]
    for j in range(num_rows1):
        if (j in data1.index):
            x_a1 = data1[j, sd11, 0]
            y_a1 = data1[j, sd11, 1]
            x_b1 = data1[j, sd12, 0]
            y_b1 = data1[j, sd12, 1]
            x_c1 = data1[j, sd13, 0]
            y_c1 = data1[j, sd13, 1]
            x_d1 = data1[j, sd14, 0]
            y_d1 = data1[j, sd14, 1]

            cxa1 = np.append(cxa1, x_a1)
            cya1 = np.append(cya1, y_a1)
            cxb1 = np.append(cxb1, x_b1)
            cyb1 = np.append(cyb1, y_b1)
            cxc1 = np.append(cxc1, x_c1)
            cyc1 = np.append(cyc1, y_c1)
            cxd1 = np.append(cxd1, x_d1)
            cyd1 = np.append(cyd1, y_d1)

    num_rows2, _ = data2.shape[0]
    for j in range(num_rows2):
        if (j in data2.index):
            x_a2 = data2[j, sd21, 0]
            y_a2 = data2[j, sd21, 1]
            x_b2 = data2[j, sd22, 0]
            y_b2 = data2[j, sd22, 1]
            x_c2 = data2[j, sd23, 0]
            y_c2 = data2[j, sd23, 1]
            x_d2 = data2[j, sd24, 0]
            y_d2 = data2[j, sd24, 1]

            cxa2 = np.append(cxa2, x_a2)
            cya2 = np.append(cya2, y_a2)
            cxb2 = np.append(cxb2, x_b2)
            cyb2 = np.append(cyb2, y_b2)
            cxc2 = np.append(cxc2, x_c2)
            cyc2 = np.append(cyc2, y_c2)
            cxd2 = np.append(cxd2, x_d2)
            cyd2 = np.append(cyd2, y_d2)

    output_dir = os.path.dirname(filename1)
    base_name = os.path.splitext(filename1)[0]
    output_file = os.path.join(output_dir, f'{txt}_{base_name}_{exe}_{cam}.txt')
    with open(output_file, 'w') as f:
        if alignment_type == "same_x":
            f.write(f"{txt}_{exe}_{cam}_{alignment_type}\n")
            for j in range(num_rows1):
                if cxa1[j] == cxb1[j] and cxa1[j] == cxc1[j] and cxa1[j] == cxd1[j]:
                    f.write(f"{j} It does align (CSV)\n")
                    if cxa2[j] == cxb2[j] and cxa2[j] == cxc2[j] and cxa2[j] == cxd2[j]:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
                else:
                    f.write(f'{j} It does not align (CSV)\n')
                    if cxa2[j] == cxb2[j] and cxa2[j] == cxc2[j] and cxa2[j] == cxd2[j]:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
        elif alignment_type == "same_y":
            f.write(f"{txt}_{exe}_{cam}_{alignment_type}\n")
            for j in range(num_rows1):
                if cya1 == cyb1 and cya1 == cyc1 and cya1 == cyd1:
                    f.write(f"{j} It does align (CSV)\n")
                    if cya2 == cyb2 and cya2 == cyc2 and cya2 == cyd2:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
                else:
                    f.write(f'{j} It does not align (CSV)\n')
                    if cya2 == cyb2 and cya2 == cyc2 and cya2 == cyd2:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
        else:
            a1 = np.append(a1, f"{txt}_{exe}_{cam}_{alignment_type}")
            for j in range(num_rows1):
                if cya1 == cyb1 and cya1 == cyc1 and cya1 == cyd1 and cxa1[j] == cxb1[j] and cxa1[j] == cxc2[j] and cxa1[j] == cxd1[j]:
                    f.write(f"{j} It does align (CSV)\n")
                    if cya2 == cyb2 and cya2 == cyc2 and cya2 == cyd2 and cxa1[j] == cxb2[j] and cxa1[j] == cxc2[j] and cxa1[j] == cxd2[j]:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
                else:
                    f.write(f'{j} It does not align (CSV)\n ')
                    if cya2 == cyb2 and cya2 == cyc2 and cya2 == cyd2 and cxa2[j] == cxb2[j] and cxa2[j] == cxc2[j] and cxa2[j] == cxd2[j]:
                        f.write(f'{j} It does align (NPY)\n')
                    else:
                        f.write(f'{j} It does not align (NPY)\n')
            output_filename = f'parallelism_{base_name}_{txt}_{exe}_{cam}.txt'


folder_path = 'C:\\Users\\MihailS\\Documents\\Project\\Test\\Openpose\\TestSample'
files1 = glob.glob(folder_path + '/*.csv')
files2 = glob.glob(folder_path + '/*.npy')
# finding the files with the correct camera
for file1,file2 in zip(files1,files2):
    filename1 = os.path.basename(file1)
    if filename1[0:3] == "1LBR":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 100, 10, filename1[0:3],
                      filename1[-5])
            alignfilt3(file1, file2, 11, 23, 26, 2, 9, 13, "shoulder_hip_oknee", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "BBRW":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 15, 13, 11, 7, 6, 5, "right_elbow", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 90, 20, filename1[0:3], filename1[-5])

    elif filename1[0:3] == "BRGM":
        if filename1[-5] == "3":
            alignfilt4(file1, file2, 27, 25, 23, 11, 11, 10, 9, 2, "right_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 28, 26, 24, 12, 14, 13, 12, 5, "left_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
    elif filename1[0:3] == "BRK4":
        if filename1[-5] == "4":
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "BRKZ":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 180, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 180, 10, filename1[0:3],
                      filename1[-5])
    elif filename1[0:3] == "BULG":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 7, 11, 23, 25, 17, 2, 9, 10, "right_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            alignfilt5(file1, file2, 8, 12, 24, 26, 28, 18, 5, 12, 13, 14, "left_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "CHOP":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, 90, 10, "left_knee", filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 0, 12, 34, 0, 5, 8, "nose_shoulder_midhip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "CLMB":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 7, 11, 23, 17, 2, 9, "right_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 11, 13, 15, 2, 3, 4, "right_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 15, 12, 16, 2, 4, 5, 7, "shoulder_wrist", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "FRLG":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 29, 26, 32, 24, 13, 19, "right_feet_oknee_otoes", 15, filename1[0:3],
                       filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 24, 26, 28, 12, 13, 19, "left_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 7, 11, 23, 17, 2, 9, "right_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "GLUT":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 32, 28, 26, 20, 21, 13, "left_twk", 90, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 24, 26, 28, 12, 13, 19, "left_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "HMST":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 180, 10, filename1[0:3],
                      filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 90, 20, filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 23, 25, 27, 9, 10, 11, "right_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "HMSZ":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 180, 10, filename1[0:3],
                      filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 90, 20, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 31, 27, 25, 23, 24, 10, "right_twk", 90, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 7, 11, 23, 17, 2, 9, "right_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "HSSE":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 32, 28, 26, 20, 21, 13, "left_twk", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 31, 27, 25, 23, 24, 10, "right_twk", 90, 10, filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 28, 26, 24, 12, 14, 13, 12, 5, "left_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 27, 25, 23, 11, 11, 10, 9, 2, "right_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "IPST":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 170, 10, filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 7, 11, 23, 25, 17, 2, 9, 10, "right_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 7, 11, 23, 25, 17, 2, 9, 10, "right_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
    elif filename1[0:3] == "JPJK":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 180, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 180, 10, filename1[0:3],
                      filename1[-5])
            anglefilt(file1, file2, 15, 13, 11, 7, 6, 5, "right_elbow", 180, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 16, 14, 12, 4, 3, 2, "left_elbow", 180, 10, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 33, 11, 5, 1, 2, "shoulder_neck_shoulder", 15, filename1[0:3],
                       filename1[-5])
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt6(file1, file2, 28, 26, 24, 12, 8, 16, 14, 13, 12, 5, 18, 7,
                       "ankle_knee_hip_shoulder_ear_hand", 15, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "LGST":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 7, 11, 23, 17, 2, 9, "right_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "LPLT":
        if filename1[-5] == "4":
            alignfilt6(file1, file2, 15, 13, 11, 12, 14, 16, 4, 3, 2, 5, 6, 7,
                       "right_hand_elbow_shoulder_left_shoulder_elbow_hand", 15, filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt4(file1, file2, 28, 26, 24, 12, 14, 13, 12, 5, "left_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
            alignfilt6(file1, file2, 28, 26, 24, 12, 8, 16, 14, 13, 12, 5, 18, 7,
                       "ankle_knee_hip_shoulder_ear_hand", 15, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "LTDB":
        if filename1[-5] == "4":
            alignfilt5(file1, file2, 8, 12, 24, 26, 28, 18, 5, 12, 13, 14, "left_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
            alignfilt5(file1, file2, 7, 11, 23, 25, 27, 17, 2, 9, 10, 11, "right_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
            parafilt(file1, file2, 16, 14, 12, 24, 7, 6, 1, 8, "left_wrist_elbow_shoulder_hip", filename1[0:3],
                     filename1[-5])
            parafilt(file1, file2, 15, 13, 11, 23, 14, 13, 11, 10, "right_wrist_elbow_shoulder_hip",
                     filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 13, 23, 14, 24, 3, 9, 6, 12, "elbow_hip", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "LTLG":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 180, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 180, 10, filename1[0:3],
                      filename1[-5])
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            parafilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 120, 20, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 24, 26, 28, 12, 13, 19, "left_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "LTPL":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            parafilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "MOZR":
        if filename1[-5] == "4":
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
            parafilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 120, 20, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 32, 28, 26, 20, 21, 13, "left_twk", 90, 10, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "OLDL":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            alignfilt5(file1, file2, 7, 11, 23, 25, 27, 17, 2, 9, 10, 11, "right_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 14, 12, 24, 6, 5, 12, "left_shoulder", 90, 10, filename1[0:3],
                      filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 24, 26, 28, 12, 13, 19, "left_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 0, 33, 34, 0, 5, 8, "nose_neck_midhip", 15, filename1[0:3], filename1[-5])
            parafilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "PLCI":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 15, 13, 11, 7, 6, 5, "right_elbow", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 13, 11, 23, 3, 2, 9, "right_shoulder", 100, 20, filename1[0:3],
                      filename1[-5])
            alignfilt4(file1, file2, 7, 11, 23, 25, 17, 2, 9, 10, "right_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 13, 11, 14, 12, 3, 2, 6, 5, "elbow_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "PLLU":
        if filename1[-5] == "4":
            distfilt(file1, file2, 0, 13, 0, 14, 0, 3, 0, 6, "nose_elbow", filename1[0:3], filename1[-5])
            distfilt(file1, file2, np.nan, np.nan, np.nan, np.nan, 10, 8, 13, 8, "midhip_knee", filename1[0:3],
                     filename1[-5])
            parafilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "PLM1":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 7, 11, 23, 17, 2, 9, "right_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt5(file1, file2, 7, 11, 23, 25, 27, 17, 2, 9, 10, 11, "right_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 15, 13, 11, 7, 6, 5, "right_elbow", 90, 20, filename1[0:3], filename1[-5])
            samefilt(file1, file2, 27, 25, 28, 26, 11, 10, 14, 13, "toe_knee", "same_y", filename1[0:3],
                     filename1[-5])
        elif filename1[-5] == "3":
            parafilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 0, 13, 0, 14, 0, 3, 0, 6, "nose_elbow", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "PLTO":
        if filename1[-5] == "4":
            alignfilt4(file1, file2, 28, 26, 24, 12, 14, 13, 12, 5, "left_ankle_knee_hip_shoulder", 15,
                       filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 14, 16, 5, 6, 7, "left_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
            samefilt(file1, file2, 16, 11, 15, 12, 7, 2, 4, 5, "hand_oshoulder", "same_xy", filename1[0:3],
                     filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 160, 40, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "PUSH":
        if filename1[-5] == "4":
            alignfilt4(file1, file2, 8, 12, 24, 26, 18, 5, 12, 13, "left_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 15, 12, 16, 2, 4, 5, 7, "shoulder_wrist", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            samefilt(file1, file2, 12, 14, 11, 13, 5, 6, 2, 3, "hand_oshoulder", "same_y", filename1[0:3],
                     filename1[-5])
    elif filename1[0:3] == "RFSN":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 160, 40, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 16, 26, 30, 7, 13, 21, "left_hand_knee_heel", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 12, 14, 16, 5, 6, 7, "left_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 14, 16, 5, 6, 7, "left_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "SKIS":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 75, 25, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "SKJP":
        if filename1[-5] == "4":
            parafilt(file1, file2, 11, 23, 26, 28, 2, 9, 13, 14, "shoulder_hip_oknee_oankle", filename1[0:3],
                     filename1[-5])
            distfilt(file1, file2, 27, 25, 28, 26, 11, 10, 14, 13, "ankle_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 27, 24, 28, 9, 11, 12, 14, "hip_ankle", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 75, 25, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "SMSQ":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 25, 23, 24, 26, 10, 9, 12, 13, "knee_hip_hip_knee", 15, filename1[0:3],
                       filename1[-5])
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 90, 20, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "SQOC":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            parafilt(file1, file2, 25, 31, 26, 32, 10, 23, 13, 20, "knee_toe", filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 100, 20, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 20, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "SQUA":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 24, 18, 5, 12, "left_ear_shoulder_hip", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 100, 20, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 15, filename1[0:3], filename1[-5])
            parafilt(file1, file2, 25, 31, 26, 32, 10, 23, 13, 20, "knee_toe", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "STRZ":
        if filename1[-5] == "4":
            anglefilt(file1, file2, 24, 26, 28, 9, 10, 11, "left_knee", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 23, 25, 27, 12, 13, 14, "right_knee", 90, 20, filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 7, 11, 23, 25, 17, 2, 9, 10, "right_ear_shoulder_hip_knee", 15,
                       filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
            parafilt(file1, file2, 11, 23, 26, 28, 2, 9, 13, 14, "shoulder_hip_oknee_oankle", filename1[0:3],
                     filename1[-5])
    elif filename1[0:3] == "TRID":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 24, 26, 28, 12, 13, 19, "left_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 23, 25, 27, 9, 10, 11, "right_ankle_knee_hip", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 15, 13, 11, 7, 6, 5, "right_elbow", 90, 40, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 120, 25, filename1[0:3], filename1[-5])
            samefilt(file1, file2, 12, 14, 11, 13, 5, 6, 2, 3, "hand_oshoulder", "same_y", filename1[0:3],
                     filename1[-5])
        elif filename1[-5] == "3":
            samefilt(file1, file2, 12, 14, 11, 13, 5, 6, 2, 3, "hand_oshoulder", "same_x", filename1[0:3],
                     filename1[-5])
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
    elif filename1[0:3] == "TRST":
        if filename1[-5] == "4":
            alignfilt3(file1, file2, 12, 14, 26, 5, 6, 13, "left_shoulder_elbow_knee", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 11, 23, 25, 5, 12, 13, "right_shoulder_hip_knee", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 0, 33, 12, 0, 1, 5, "left_neck", 45, 15, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 0, 33, 11, 0, 1, 2, "right_neck", 45, 15, filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt3(file1, file2, 8, 12, 14, 18, 5, 6, "left_ear_elbow_shoulder", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:7] == "YMWS":
        if filename1[-5] == "4":
            distfilt(file1, file2, 13, 23, 14, 24, 3, 9, 6, 12, "elbow_hip", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt5(file1, file2, 8, 12, 24, 26, 28, 18, 5, 12, 13, 14, "left_ear_shoulder_hip_knee_ankle",
                       15, filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 16, 14, 12, 7, 6, 5, "left_hand_elbow_hip", 15, filename1[0:3],
                       filename1[-5])
    elif filename1[0:3] == "YPST":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 15, 7, 16, 8, 4, 17, 7, 18, "hand_ear", filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 14, 16, 5, 6, 7, "left_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 11, 13, 15, 2, 3, 4, "right_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 150, 15, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 150, 15, filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            alignfilt6(file1, file2, 16, 14, 12, 24, 26, 28, 7, 6, 5, 12, 13, 14,
                       "left_hand_elbow_shoulder_knee_hip_ankle", 15, filename1[0:3], filename1[-5])
    elif filename1[0:3] == "ZRCS":
        if filename1[-5] == "4":
            distfilt(file1, file2, 7, 11, 8, 12, 17, 2, 18, 5, "ear_shoulder", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 15, 25, 16, 26, 4, 10, 7, 13, "hand_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 23, 25, 24, 26, 9, 10, 12, 13, "hip_knee", filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 14, 26, 5, 6, 13, "left_shoulder_elbow_knee", 15, filename1[0:3],
                       filename1[-5])
            alignfilt3(file1, file2, 11, 13, 25, 2, 3, 10, "right_shoulder_elbow_knee", 15, filename1[0:3],
                       filename1[-5])
            parafilt(file1, file2, 28, 26, 27, 25, 14, 13, 11, 10, "ankle_knee", filename1[0:3], filename1[-5])
            distfilt(file1, file2, 11, 23, 12, 24, 2, 9, 5, 12, "shoulder_hip", filename1[0:3], filename1[-5])
        elif filename1[-5] == "3":
            anglefilt(file1, file2, 12, 24, 26, 5, 12, 13, "left_hip", 90, 10, filename1[0:3], filename1[-5])
            anglefilt(file1, file2, 11, 23, 25, 2, 9, 10, "right_hip", 90, 10, filename1[0:3], filename1[-5])
            alignfilt4(file1, file2, 8, 12, 14, 24, 18, 5, 6, 12, "left_ear_elbow_shoulder_hip", 15,
                       filename1[0:3], filename1[-5])
            alignfilt3(file1, file2, 12, 14, 16, 5, 6, 7, "left_shoulder_elbow_wrist", 15, filename1[0:3],
                       filename1[-5])
    else:
        # Handle files that do not match any condition here
        # You can print a message or perform any other desired action
        print(f"Unmatched file: {filename1}")
#anglefilt('1LBR_1686643789_ID_Telescope_cam_4.csv', '1LBR_1686643789_ID_Telescope_cam_4.npy', 23, 25, 27, 12, 13, 14, "right_knee", 100, 10, "1LBR","4")
#alignfilt3('1LBR_1686643789_ID_Telescope_cam_4.csv', '1LBR_1686643789_ID_Telescope_cam_4.npy', 11, 23, 26, 2, 9, 13, "shoulder_hip_oknee", 15,"1LBR","4")






    
