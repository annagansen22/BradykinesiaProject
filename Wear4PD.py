# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 03:57:41 2020

@author: Z250201
"""
import simplejson as json
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from scipy import fftpack
import scipy.io as sio
import math
from scipy.signal import argrelextrema
from scipy import signal

def readJSONFile(filename):
    """Reads JSON file and returns the data as a Python dictionary.
    
    Parameters:
        filename(string): name of the JSON file
    
    Returns: 
        dict: data of JSON file
    """
    f = open(filename,) 
    data = json.load(f) 
    
    return data

def reject_outliers(data, m=3, remove = 0):
    npdata = np.array(data)
    out = np.count_nonzero((abs(npdata - np.mean(npdata)) < m * np.std(npdata)) == 0)
    if (remove > 0):
        new_data = npdata[abs(npdata - np.mean(npdata)) < m * np.std(npdata)][remove:-remove]
    else:
        new_data = npdata[abs(npdata - np.mean(npdata)) < m * np.std(npdata)]
    return new_data, out

def segmentIntegrals(integrals):
    # for local maxima
    maxi = argrelextrema(integrals, np.greater, order = 50)

    # for local minima
    mini = argrelextrema(integrals, np.less, order = 50)
    
    return maxi, mini

def findHesitationsFreezing(CSA_T, threshold, main_freq, dt):
    T = 1/float(main_freq)
    steps = int((T/dt)/3)
    bool_arr = CSA_T < threshold
    counter = 0
    i = 0
    for e in bool_arr:
        if e:
            i = i + 1
        else:
            if (i > steps):
                counter = counter + 1
            i = 0
    return counter

def getAmplitudes(x, y):
    integral = F(x,y)
    (maxi,), (mini,) = segmentIntegrals(integral)

    absAmp = np.array([])
    durations = np.array([])
    while ((mini.size != 0 and maxi.size != 0)):
        if (min(mini) < min(maxi)):
            absAmp = np.append(absAmp, integral[mini[0]] - integral[maxi[0]])
            durations = np.append(durations, x[maxi[0]] - x[mini[0]])
            mini = mini[1:]
        else:
            absAmp = np.append(absAmp, integral[maxi[0]] - integral[mini[0]])
            durations = np.append(durations, x[mini[0]] - x[maxi[0]])
            maxi = maxi[1:]
    return integral, absAmp, durations
    
def createHeaderDict(header):
    gyro_scale = header['header'][0][0][6][0][0][1][0][0][0][0][0]
    gyro_sensorID = header['header'][0][0][6][0][0][1][0][0][1][0][0]
    gyro_FS = header['header'][0][0][6][0][0][1][0][0][2][0][0]
    gyro_dataPayload = header['header'][0][0][6][0][0][1][0][0][3][0][0]
    gyro_calib = header['header'][0][0][6][0][0][1][0][0][4][0]
    gyro_nbSamples = header['header'][0][0][6][0][0][1][0][0][5][0][0]

    deviceID = header['header'][0][0][0][0][0]
    deviceType = header['header'][0][0][1][0][0]
    bodyLocation = header['header'][0][0][2][0]
    firmwareVersion = header['header'][0][0][3][0]
    startDate = header['header'][0][0][4]
    baseFrequency = header['header'][0][0][5][0][0]
    stopDate = header['header'][0][0][7]
    measureID = header['header'][0][0][8][0][0]
    
    # Create dict
    header_df = {"gyro_scale" : gyro_scale, "gyro_sensorID" : gyro_sensorID, "gyro_FS" : gyro_FS, "gyro_dataPayload" : gyro_dataPayload, "gyro_calib" : gyro_calib, "gyro_nbSamples" : gyro_nbSamples, "deviceID" : deviceID, "deviceType" : deviceType, "bodyLocation" : bodyLocation, "firmwareVersion" : firmwareVersion, "startDate" : startDate, "baseFrequency" : baseFrequency, "stopDate" : stopDate, "measureID" : measureID}
    
    return header_df

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def F(x, y):
    integral = np.array([])
    old = 0.0
    for velocity in y:
        integral = np.append(integral, old + np.diff(x)[0]*velocity)
        old = old + np.diff(x)[0]*velocity
    return integral

def getSensorData(data, start, end):
    gyro_data = data['sensorData'][0][1][3]
    gyro_FS = data['sensorData'][0][1][2][0][0]
    gyro_timestamps = np.concatenate(data['sensorData'][0][1][4]).ravel()
    
    start_index = np.where(gyro_timestamps == find_nearest(gyro_timestamps, start))[0][0]
    end_index = np.where(gyro_timestamps == find_nearest(gyro_timestamps, end))[0][0]
    
    # return gyro data along the y-axis and z-axis in the timeslot of the pronation supination movements
    return gyro_data[start_index:end_index,1:3], gyro_FS, gyro_timestamps[start_index:end_index]

def createSubjectFromData(filename, timeslot_choice):
    """Given a dictionary of the data, an object of the class Subject is initiated and returned.
    
    Parameters:
        filename(dict): filename of the JSON file
    
    Returns: 
        Subject object: initiated object of the Subject class containing the data
    """
    
    # Read wearable sensor data .mat files
    data_LW = sio.loadmat('SensorData/' + filename + '_LW_sensorData.mat')
    data_RW = sio.loadmat('SensorData/' + filename + '_RW_sensorData.mat')
    header_LW = createHeaderDict(sio.loadmat('SensorData/' + filename + '_LW_header.mat'))
    header_RW = createHeaderDict(sio.loadmat('SensorData/' + filename + '_RW_header.mat'))
    
    timeslots_10_seconds = {"CA01_LW_start" : 358.0, "CA01_LW_end" : 368.0, "CA01_RW_start" : 318.7, "CA01_RW_end" : 328.7, 
                 "CA02_LW_start" : 235.6, "CA02_LW_end" : 245.6, "CA02_RW_start" : 206.5, "CA02_RW_end" : 216.5,
                "CA03_LW_start" : 552.0, "CA03_LW_end" : 562.0, "CA03_RW_start" : 510.0, "CA03_RW_end" : 520.0,
                "CA11_LW_start" : 244.3, "CA11_LW_end" : 254.3, "CA11_RW_start" : 216.7, "CA11_RW_end" : 226.7,
                "CA13_LW_start" : 357.0, "CA13_LW_end" : 367.0, "CA13_RW_start" : 326.0, "CA13_RW_end" : 336.0,
                "CA15_LW_start" : 356.4, "CA15_LW_end" : 366.4, "CA15_RW_start" : 330.6, "CA15_RW_end" : 340.6,
                "CA16_LW_start" : 281.5, "CA16_LW_end" : 291.5, "CA16_RW_start" : 254.9, "CA16_RW_end" : 264.9,
                "CA25_LW_start" : 324.6, "CA25_LW_end" : 334.6, "CA25_RW_start" : 301.7, "CA25_RW_end" : 311.7,
                "CA29_LW_start" : 252.0, "CA29_LW_end" : 262.0, "CA29_RW_start" : 266.5, "CA29_RW_end" : 276.5,
                "CA37_LW_start" : 505.4, "CA37_LW_end" : 515.4, "CA37_RW_start" : 459.0, "CA37_RW_end" : 469.0,
                "CA39_LW_start" : 435.3, "CA39_LW_end" : 445.3, "CA39_RW_start" : 381.2, "CA39_RW_end" : 391.2,
                "CA40_LW_start" : 363.0, "CA40_LW_end" : 373.0, "CA40_RW_start" : 327.1, "CA40_RW_end" : 337.1,
                "CA41_LW_start" : 296.4, "CA41_LW_end" : 306.4, "CA41_RW_start" : 258.5, "CA41_RW_end" : 268.5,
                "CA44_LW_start" : 247.5, "CA44_LW_end" : 257.5, "CA44_RW_start" : 216.0, "CA44_RW_end" : 226.0,
                "CA46_LW_start" : 346.0, "CA46_LW_end" : 356.0, "CA46_RW_start" : 314.0, "CA46_RW_end" : 324.0,
                "CA52_LW_start" : 413.6, "CA52_LW_end" : 423.6, "CA52_RW_start" : 419.0, "CA52_RW_end" : 429.0,
                "CA55_LW_start" : 391.6, "CA55_LW_end" : 401.6, "CA55_RW_start" : 362.5, "CA55_RW_end" : 372.5,
                "CA56_LW_start" : 293.4, "CA56_LW_end" : 303.4, "CA56_RW_start" : 290.0, "CA56_RW_end" : 300.0,
                "CA59_LW_start" : 294.3, "CA59_LW_end" : 304.3, "CA59_RW_start" : 298.5, "CA59_RW_end" : 308.5,
                "HC01_LW_start" : 650.5, "HC01_LW_end" : 660.5, "HC01_RW_start" : 621.0, "HC01_RW_end" : 631.0,
                "HC08_LW_start" : 294.4, "HC08_LW_end" : 304.4, "HC08_RW_start" : 278.5, "HC08_RW_end" : 288.5,
                "HC09_LW_start" : 214.7, "HC09_LW_end" : 224.7, "HC09_RW_start" : 194.2, "HC09_RW_end" : 204.2,
                "HC11_LW_start" : 300.0, "HC11_LW_end" : 310.0, "HC11_RW_start" : 289.5, "HC11_RW_end" : 299.5,
                "HC12_LW_start" : 358.0, "HC12_LW_end" : 368.0, "HC12_RW_start" : 332.0, "HC12_RW_end" : 342.0,
                "HC13_LW_start" : 356.3, "HC13_LW_end" : 366.3, "HC13_RW_start" : 318.0, "HC13_RW_end" : 328.0,
                "HC14_LW_start" : 293.0, "HC14_LW_end" : 303.0, "HC14_RW_start" : 269.9, "HC14_RW_end" : 279.9,
                "HC17_LW_start" : 448.8, "HC17_LW_end" : 458.8, "HC17_RW_start" : 422.7, "HC17_RW_end" : 432.7,
                "HC20_LW_start" : 439.0, "HC20_LW_end" : 449.0, "HC20_RW_start" : 383.8, "HC20_RW_end" : 393.8,
                "HC22_LW_start" : 550.0, "HC22_LW_end" : 560.0, "HC22_RW_start" : 422.0, "HC22_RW_end" : 432.0,
                "HC23_LW_start" : 325.7, "HC23_LW_end" : 335.7, "HC23_RW_start" : 303.0, "HC23_RW_end" : 313.0,
                "HC25_LW_start" : 482.3, "HC25_LW_end" : 492.3, "HC25_RW_start" : 463.7, "HC25_RW_end" : 473.7,
                "HC27_LW_start" : 379.7, "HC27_LW_end" : 389.7, "HC27_RW_start" : 366.0, "HC27_RW_end" : 376.0,
#                 "HC28_LW_start" : 0.0, "HC28_LW_end" : 10.0, "HC28_RW_start" : 358.6, "HC28_RW_end" : 368.6,
                "HC30_LW_start" : 347.5, "HC30_LW_end" : 357.5, "HC30_RW_start" : 295.0, "HC30_RW_end" : 305.0,
                "HC33_LW_start" : 360.7, "HC33_LW_end" : 370.7, "HC33_RW_start" : 344.8, "HC33_RW_end" : 354.8,
                "HC35_LW_start" : 398.3, "HC35_LW_end" : 408.3, "HC35_RW_start" : 392.9, "HC35_RW_end" : 402.9,
                "HC36_LW_start" : 328.3, "HC36_LW_end" : 338.3, "HC36_RW_start" : 301.0, "HC36_RW_end" : 311.0,
                "HC37_LW_start" : 321.4, "HC37_LW_end" : 331.4, "HC37_RW_start" : 303.8, "HC37_RW_end" : 313.8,
                
                "PD01_OFF_LW_start" : 241.0, "PD01_OFF_LW_end" : 251.0, "PD01_OFF_RW_start" : 207.8, "PD01_OFF_RW_end" : 217.8,
                "PD03_OFF_LW_start" : 1155.3, "PD03_OFF_LW_end" : 1165.3, "PD03_OFF_RW_start" : 1117.0, "PD03_OFF_RW_end" : 1127.0,
                "PD04_OFF_LW_start" : 368.7, "PD04_OFF_LW_end" : 378.7, "PD04_OFF_RW_start" : 324.3, "PD04_OFF_RW_end" : 334.3,
                "PD05_OFF_LW_start" : 345.4, "PD05_OFF_LW_end" : 355.4, "PD05_OFF_RW_start" : 314.5, "PD05_OFF_RW_end" : 324.5,
                "PD09_OFF_LW_start" : 428.5, "PD09_OFF_LW_end" : 438.5, "PD09_OFF_RW_start" : 383.3, "PD09_OFF_RW_end" : 393.3,
                "PD13_OFF_LW_start" : 475.3, "PD13_OFF_LW_end" : 485.3, "PD13_OFF_RW_start" : 481.6, "PD13_OFF_RW_end" : 491.6,
                "PD16_OFF_LW_start" : 355.44, "PD16_OFF_LW_end" : 365.44, "PD16_OFF_RW_start" : 330.7, "PD16_OFF_RW_end" : 340.7,
                "PD17_OFF_LW_start" : 404.5, "PD17_OFF_LW_end" : 414.5, "PD17_OFF_RW_start" : 418.2, "PD17_OFF_RW_end" : 428.2,
                "PD22_OFF_LW_start" : 342.6, "PD22_OFF_LW_end" : 352.6, "PD22_OFF_RW_start" : 311.0, "PD22_OFF_RW_end" : 321.0,
                "PD25_OFF_LW_start" : 350.8, "PD25_OFF_LW_end" : 360.8, "PD25_OFF_RW_start" : 360.7, "PD25_OFF_RW_end" : 370.7,
                "PD29_OFF_LW_start" : 365.4, "PD29_OFF_LW_end" : 375.4, "PD29_OFF_RW_start" : 307.1, "PD29_OFF_RW_end" : 317.1,
                "PD31_OFF_LW_start" : 315.0, "PD31_OFF_LW_end" : 325.0, "PD31_OFF_RW_start" : 291.5, "PD31_OFF_RW_end" : 301.5,
                "PD33_OFF_LW_start" : 399.6, "PD33_OFF_LW_end" : 409.6, "PD33_OFF_RW_start" : 357.5, "PD33_OFF_RW_end" : 367.5,
                "PD34_OFF_LW_start" : 389.35, "PD34_OFF_LW_end" : 399.35, "PD34_OFF_RW_start" : 355.1, "PD34_OFF_RW_end" : 365.1,
                "PD36_OFF_LW_start" : 326.3, "PD36_OFF_LW_end" : 336.3, "PD36_OFF_RW_start" : 297.55, "PD36_OFF_RW_end" : 307.55,
                "PD37_OFF_LW_start" : 293.3, "PD37_OFF_LW_end" : 303.3, "PD37_OFF_RW_start" : 267.6, "PD37_OFF_RW_end" : 277.6,
                "PD38_OFF_LW_start" : 322.26, "PD38_OFF_LW_end" : 332.26, "PD38_OFF_RW_start" : 287.6, "PD38_OFF_RW_end" : 297.6,
                "PD39_OFF_LW_start" : 307.8, "PD39_OFF_LW_end" : 317.8, "PD39_OFF_RW_start" : 281.0, "PD39_OFF_RW_end" : 291.0,
                
                "PD01_ON_LW_start" : 414.0, "PD01_ON_LW_end" : 424.0, "PD01_ON_RW_start" : 374.2, "PD01_ON_RW_end" : 384.2,
                "PD03_ON_LW_start" : 249.0, "PD03_ON_LW_end" : 259.0, "PD03_ON_RW_start" : 221.9, "PD03_ON_RW_end" : 231.9,
                "PD04_ON_LW_start" : 367.0, "PD04_ON_LW_end" : 377.0, "PD04_ON_RW_start" : 337.8, "PD04_ON_RW_end" : 347.8,
                "PD05_ON_LW_start" : 247.785, "PD05_ON_LW_end" : 257.785, "PD05_ON_RW_start" : 209.0, "PD05_ON_RW_end" : 219.0,
                "PD08_ON_LW_start" : 351.96, "PD08_ON_LW_end" : 361.96, "PD08_ON_RW_start" : 331.484, "PD08_ON_RW_end" : 341.484,
                "PD09_ON_LW_start" : 295.49, "PD09_ON_LW_end" : 305.49, "PD09_ON_RW_start" : 266.22, "PD09_ON_RW_end" : 276.22,
                "PD13_ON_LW_start" : 232.9, "PD13_ON_LW_end" : 242.9, "PD13_ON_RW_start" : 246.65, "PD13_ON_RW_end" : 256.65,
#                 "PD16_ON_LW_start" : 191.75, "PD16_ON_LW_end" : 201.75, "PD16_ON_RW_start" : 0.0, "PD16_ON_RW_end" : 10.0,
                "PD17_ON_LW_start" : 241.4, "PD17_ON_LW_end" : 251.4, "PD17_ON_RW_start" : 213.7, "PD17_ON_RW_end" : 223.7,
                "PD22_ON_LW_start" : 291.2, "PD22_ON_LW_end" : 301.2, "PD22_ON_RW_start" : 217.8, "PD22_ON_RW_end" : 227.8,
                "PD25_ON_LW_start" : 229.95, "PD25_ON_LW_end" : 239.95, "PD25_ON_RW_start" : 197.1, "PD25_ON_RW_end" : 207.1,
                "PD29_ON_LW_start" : 212.75, "PD29_ON_LW_end" : 222.75, "PD29_ON_RW_start" : 194.0, "PD29_ON_RW_end" : 204.0,
                "PD31_ON_LW_start" : 394.4, "PD31_ON_LW_end" : 404.4, "PD31_ON_RW_start" : 376.15, "PD31_ON_RW_end" : 386.15,
                "PD33_ON_LW_start" : 257.4, "PD33_ON_LW_end" : 267.4, "PD33_ON_RW_start" : 223.75, "PD33_ON_RW_end" : 233.75,
                "PD34_ON_LW_start" : 454.0, "PD34_ON_LW_end" : 464.0, "PD34_ON_RW_start" : 432.0, "PD34_ON_RW_end" : 442.0,
                "PD36_ON_LW_start" : 277.35, "PD36_ON_LW_end" : 287.35, "PD36_ON_RW_start" : 244.2, "PD36_ON_RW_end" : 254.2,
               # "PD37_ON_LW_start" : 287.5, "PD37_ON_LW_end" : 297.5, "PD37_ON_RW_start" : 0.0, "PD37_ON_RW_end" : 10.0,
                "PD38_ON_LW_start" : 229.84, "PD38_ON_LW_end" : 239.84, "PD38_ON_RW_start" : 207.84, "PD38_ON_RW_end" : 217.84,
                "PD39_ON_LW_start" : 222.4, "PD39_ON_LW_end" : 232.4, "PD39_ON_RW_start" : 196.58, "PD39_ON_RW_end" : 206.58}
    
    timeslots_full_duration = {"CA01_LW_start" : 358.0, "CA01_LW_end" : 367.0, "CA01_RW_start" : 318.7, "CA01_RW_end" : 327.3, 
                 "CA02_LW_start" : 235.6, "CA02_LW_end" : 243.0, "CA02_RW_start" : 206.5, "CA02_RW_end" : 216.0,
                "CA03_LW_start" : 552.0, "CA03_LW_end" : 565.6, "CA03_RW_start" : 503.9, "CA03_RW_end" : 520.3,
                "CA11_LW_start" : 244.3, "CA11_LW_end" : 256.1, "CA11_RW_start" : 216.7, "CA11_RW_end" : 227.1,
                "CA13_LW_start" : 355.5, "CA13_LW_end" : 368.0, "CA13_RW_start" : 326.3, "CA13_RW_end" : 334.6,
                "CA15_LW_start" : 356.4, "CA15_LW_end" : 368.9, "CA15_RW_start" : 330.6, "CA15_RW_end" : 345.9,
                "CA16_LW_start" : 281.5, "CA16_LW_end" : 293.8, "CA16_RW_start" : 254.9, "CA16_RW_end" : 267.3,
                "CA25_LW_start" : 324.6, "CA25_LW_end" : 338.0, "CA25_RW_start" : 301.7, "CA25_RW_end" : 315.1,
                "CA29_LW_start" : 251.5, "CA29_LW_end" : 262.6, "CA29_RW_start" : 266.5, "CA29_RW_end" : 274.4,
                "CA37_LW_start" : 505.4, "CA37_LW_end" : 516.5, "CA37_RW_start" : 459.0, "CA37_RW_end" : 471.0,
                "CA39_LW_start" : 435.3, "CA39_LW_end" : 444.2, "CA39_RW_start" : 381.2, "CA39_RW_end" : 389.0,
                "CA40_LW_start" : 363.0, "CA40_LW_end" : 375.0, "CA40_RW_start" : 327.1, "CA40_RW_end" : 336.95,
                "CA41_LW_start" : 296.4, "CA41_LW_end" : 306.7, "CA41_RW_start" : 258.5, "CA41_RW_end" : 268.66,
                "CA44_LW_start" : 247.5, "CA44_LW_end" : 255.6, "CA44_RW_start" : 217.7, "CA44_RW_end" : 225.6,
                "CA46_LW_start" : 346.7, "CA46_LW_end" : 355.9, "CA46_RW_start" : 313.6, "CA46_RW_end" : 327.0,
                "CA52_LW_start" : 413.6, "CA52_LW_end" : 424.0, "CA52_RW_start" : 418.3, "CA52_RW_end" : 430.2,
                "CA55_LW_start" : 391.6, "CA55_LW_end" : 407.4, "CA55_RW_start" : 362.5, "CA55_RW_end" : 373.8,
                "CA56_LW_start" : 293.4, "CA56_LW_end" : 309.9, "CA56_RW_start" : 289.9, "CA56_RW_end" : 302.5,
                "CA59_LW_start" : 294.3, "CA59_LW_end" : 304.2, "CA59_RW_start" : 298.5, "CA59_RW_end" : 308.65,
                "HC01_LW_start" : 650.5, "HC01_LW_end" : 662.8, "HC01_RW_start" : 621.0, "HC01_RW_end" : 634.3,
                "HC08_LW_start" : 294.4, "HC08_LW_end" : 302.1, "HC08_RW_start" : 278.6, "HC08_RW_end" : 287.7,
                "HC09_LW_start" : 214.7, "HC09_LW_end" : 222.5, "HC09_RW_start" : 194.2, "HC09_RW_end" : 204.5,
                "HC11_LW_start" : 300.3, "HC11_LW_end" : 308.5, "HC11_RW_start" : 289.5, "HC11_RW_end" : 298.8,
                "HC12_LW_start" : 358.0, "HC12_LW_end" : 368.0, "HC12_RW_start" : 332.0, "HC12_RW_end" : 341.4,
                "HC13_LW_start" : 356.3, "HC13_LW_end" : 367.3, "HC13_RW_start" : 318.0, "HC13_RW_end" : 330.8,
                "HC14_LW_start" : 293.3, "HC14_LW_end" : 302.8, "HC14_RW_start" : 269.9, "HC14_RW_end" : 281.6,
                "HC17_LW_start" : 448.8, "HC17_LW_end" : 457.6, "HC17_RW_start" : 422.7, "HC17_RW_end" : 432.9,
                "HC20_LW_start" : 439.0, "HC20_LW_end" : 446.0, "HC20_RW_start" : 384.0, "HC20_RW_end" : 390.0,
                "HC22_LW_start" : 550.5, "HC22_LW_end" : 563.0, "HC22_RW_start" : 422.0, "HC22_RW_end" : 431.2,
                "HC23_LW_start" : 325.8, "HC23_LW_end" : 333.9, "HC23_RW_start" : 303.1, "HC23_RW_end" : 310.6,
                "HC25_LW_start" : 482.3, "HC25_LW_end" : 490.4, "HC25_RW_start" : 463.8, "HC25_RW_end" : 470.24,
                "HC27_LW_start" : 379.95, "HC27_LW_end" : 389.7, "HC27_RW_start" : 365.9, "HC27_RW_end" : 376.37,
#                 "HC28_LW_start" : 0.0, "HC28_LW_end" : 10.0, "HC28_RW_start" : 358.6, "HC28_RW_end" : 368.6,
                "HC30_LW_start" : 347.6, "HC30_LW_end" : 360.8, "HC30_RW_start" : 295.0, "HC30_RW_end" : 306.6,
                "HC33_LW_start" : 360.8, "HC33_LW_end" : 370.3, "HC33_RW_start" : 344.9, "HC33_RW_end" : 353.6,
                "HC35_LW_start" : 398.3, "HC35_LW_end" : 407.9, "HC35_RW_start" : 393.0, "HC35_RW_end" : 402.2,
                "HC36_LW_start" : 328.4, "HC36_LW_end" : 337.7, "HC36_RW_start" : 301.0, "HC36_RW_end" : 311.0,
                "HC37_LW_start" : 321.5, "HC37_LW_end" : 331.1, "HC37_RW_start" : 303.8, "HC37_RW_end" : 311.6,
                
                "PD01_OFF_LW_start" : 241.0, "PD01_OFF_LW_end" : 251.5, "PD01_OFF_RW_start" : 207.8, "PD01_OFF_RW_end" : 222.3,
                "PD03_OFF_LW_start" : 1155.3, "PD03_OFF_LW_end" : 1173.1, "PD03_OFF_RW_start" : 1116.9, "PD03_OFF_RW_end" : 1135.9,
                "PD04_OFF_LW_start" : 368.8, "PD04_OFF_LW_end" : 381.8, "PD04_OFF_RW_start" : 324.3, "PD04_OFF_RW_end" : 334.9,
                "PD05_OFF_LW_start" : 345.3, "PD05_OFF_LW_end" : 361.3, "PD05_OFF_RW_start" : 314.5, "PD05_OFF_RW_end" : 330.0,
                "PD09_OFF_LW_start" : 428.5, "PD09_OFF_LW_end" : 443.6, "PD09_OFF_RW_start" : 383.3, "PD09_OFF_RW_end" : 401.4,
                "PD13_OFF_LW_start" : 475.3, "PD13_OFF_LW_end" : 493.3, "PD13_OFF_RW_start" : 481.6, "PD13_OFF_RW_end" : 493.4,
                "PD16_OFF_LW_start" : 355.44, "PD16_OFF_LW_end" : 372.9, "PD16_OFF_RW_start" : 330.7, "PD16_OFF_RW_end" : 341.6,
                "PD17_OFF_LW_start" : 404.5, "PD17_OFF_LW_end" : 417.1, "PD17_OFF_RW_start" : 418.2, "PD17_OFF_RW_end" : 429.7,
                "PD22_OFF_LW_start" : 342.6, "PD22_OFF_LW_end" : 354.2, "PD22_OFF_RW_start" : 311.0, "PD22_OFF_RW_end" : 329.7,
                "PD25_OFF_LW_start" : 350.8, "PD25_OFF_LW_end" : 364.8, "PD25_OFF_RW_start" : 360.7, "PD25_OFF_RW_end" : 373.5,
                "PD29_OFF_LW_start" : 365.4, "PD29_OFF_LW_end" : 380.4, "PD29_OFF_RW_start" : 307.1, "PD29_OFF_RW_end" : 318.1,
                "PD31_OFF_LW_start" : 315.0, "PD31_OFF_LW_end" : 326.4, "PD31_OFF_RW_start" : 291.5, "PD31_OFF_RW_end" : 301.5,
                "PD33_OFF_LW_start" : 399.6, "PD33_OFF_LW_end" : 410.6, "PD33_OFF_RW_start" : 357.5, "PD33_OFF_RW_end" : 366.3,
                "PD34_OFF_LW_start" : 389.35, "PD34_OFF_LW_end" : 401.8, "PD34_OFF_RW_start" : 355.1, "PD34_OFF_RW_end" : 368.9,
                "PD36_OFF_LW_start" : 326.3, "PD36_OFF_LW_end" : 337.55, "PD36_OFF_RW_start" : 297.55, "PD36_OFF_RW_end" : 309.6,
                "PD37_OFF_LW_start" : 293.3, "PD37_OFF_LW_end" : 304.1, "PD37_OFF_RW_start" : 267.6, "PD37_OFF_RW_end" : 277.6,
                "PD38_OFF_LW_start" : 322.26, "PD38_OFF_LW_end" : 333.5, "PD38_OFF_RW_start" : 287.6, "PD38_OFF_RW_end" : 296.7,
                "PD39_OFF_LW_start" : 307.8, "PD39_OFF_LW_end" : 321.25, "PD39_OFF_RW_start" : 281.0, "PD39_OFF_RW_end" : 297.3,
                
                "PD01_ON_LW_start" : 414.0, "PD01_ON_LW_end" : 432.2, "PD01_ON_RW_start" : 374.2, "PD01_ON_RW_end" : 394.2,
                "PD03_ON_LW_start" : 249.0, "PD03_ON_LW_end" : 260.9, "PD03_ON_RW_start" : 221.9, "PD03_ON_RW_end" : 235.3,
                "PD04_ON_LW_start" : 367.0, "PD04_ON_LW_end" : 379.7, "PD04_ON_RW_start" : 337.8, "PD04_ON_RW_end" : 348.9,
                "PD05_ON_LW_start" : 244.2, "PD05_ON_LW_end" : 263.4, "PD05_ON_RW_start" : 209.0, "PD05_ON_RW_end" : 229.6,
                "PD08_ON_LW_start" : 351.96, "PD08_ON_LW_end" : 366.45, "PD08_ON_RW_start" : 331.484, "PD08_ON_RW_end" : 343.25,
                "PD09_ON_LW_start" : 295.49, "PD09_ON_LW_end" : 309.6, "PD09_ON_RW_start" : 266.22, "PD09_ON_RW_end" : 277.6,
                "PD13_ON_LW_start" : 232.9, "PD13_ON_LW_end" : 246.4, "PD13_ON_RW_start" : 246.65, "PD13_ON_RW_end" : 255.8,
#                 "PD16_ON_LW_start" : 191.75, "PD16_ON_LW_end" : 199.66, "PD16_ON_RW_start" : 0.0, "PD16_ON_RW_end" : 10.0,
                "PD17_ON_LW_start" : 241.4, "PD17_ON_LW_end" : 252.0, "PD17_ON_RW_start" : 213.7, "PD17_ON_RW_end" : 224.2,
                "PD22_ON_LW_start" : 291.2, "PD22_ON_LW_end" : 301.55, "PD22_ON_RW_start" : 217.8, "PD22_ON_RW_end" : 229.1,
                "PD25_ON_LW_start" : 229.95, "PD25_ON_LW_end" : 241.2, "PD25_ON_RW_start" : 197.1, "PD25_ON_RW_end" : 209.3,
                "PD29_ON_LW_start" : 212.75, "PD29_ON_LW_end" : 224.2, "PD29_ON_RW_start" : 194.0, "PD29_ON_RW_end" : 207.2,
                "PD31_ON_LW_start" : 394.4, "PD31_ON_LW_end" : 408.4, "PD31_ON_RW_start" : 376.15, "PD31_ON_RW_end" : 387.4,
                "PD33_ON_LW_start" : 257.4, "PD33_ON_LW_end" : 266.9, "PD33_ON_RW_start" : 223.75, "PD33_ON_RW_end" : 232.66,
                "PD34_ON_LW_start" : 454.0, "PD34_ON_LW_end" : 463.2, "PD34_ON_RW_start" : 432.0, "PD34_ON_RW_end" : 441.9,
                "PD36_ON_LW_start" : 277.35, "PD36_ON_LW_end" : 289.5, "PD36_ON_RW_start" : 244.2, "PD36_ON_RW_end" : 257.3,
               # "PD37_ON_LW_start" : 287.5, "PD37_ON_LW_end" : 297.5, "PD37_ON_RW_start" : 0.0, "PD37_ON_RW_end" : 10.0,
                "PD38_ON_LW_start" : 229.84, "PD38_ON_LW_end" : 239.2, "PD38_ON_RW_start" : 207.84, "PD38_ON_RW_end" : 219.84,
                "PD39_ON_LW_start" : 222.4, "PD39_ON_LW_end" : 235.2, "PD39_ON_RW_start" : 196.58, "PD39_ON_RW_end" : 210.5}
    
    # Extract gyro data
    if (timeslot_choice == "full"):
        gyro_data_LW, gyro_FS_LW, gyro_timestamps_LW = getSensorData(data_LW, timeslots_full_duration[filename + "_LW_start"], timeslots_full_duration[filename + "_LW_end"]) 
        gyro_data_RW, gyro_FS_RW, gyro_timestamps_RW = getSensorData(data_RW, timeslots_full_duration[filename + "_RW_start"], timeslots_full_duration[filename + "_RW_end"]) 
    else:
        gyro_data_LW, gyro_FS_LW, gyro_timestamps_LW = getSensorData(data_LW, timeslots_10_seconds[filename + "_LW_start"], timeslots_10_seconds[filename + "_LW_end"]) 
        gyro_data_RW, gyro_FS_RW, gyro_timestamps_RW = getSensorData(data_RW, timeslots_10_seconds[filename + "_RW_start"], timeslots_10_seconds[filename + "_RW_end"]) 
    
    # Read clinical data from JSON files
    clinical_data = readJSONFile("Data/" + filename + ".txt")  
    
    # Identify diagnosis
    if (filename.count("PD") == 1 and filename.count("OFF") == 1):
        diagnosis = 0
    elif  (filename.count("PD") == 1 and filename.count("ON") == 1):
        diagnosis = 1
    elif  (filename.count("CA") == 1):
        diagnosis = 2
    else:
        diagnosis = 3
    
    if clinical_data['years'] == "NA":
        years = np.nan
    else:
        years = clinical_data['years']
        
    # Create subject
    s = Subject(filename, diagnosis, clinical_data['typist'], clinical_data['side'], years, clinical_data['UPDRS-3_4a'], clinical_data['UPDRS-3_4b'], clinical_data['UPDRS-3_5a'], clinical_data['UPDRS-3_5b'], clinical_data['UPDRS-3_6a'], clinical_data['UPDRS-3_6b'], clinical_data['serial'], clinical_data['hand'], header_LW, header_RW, gyro_data_LW, gyro_FS_LW, gyro_timestamps_LW, gyro_data_RW, gyro_FS_RW, gyro_timestamps_RW)
    return s

class Subject:
    """ 
    This class contains all collected information about the subject from the wearable sensor data collected during the supination pronation movements of the hands. 
      
    Attributes: 
        subject_id (string): the id of the subject that is noted in the filename
        side: side affected
        years: number of years the symptons were present
        UPDRS-3_4a: UPDRS score for finger tapping test 3.4 right hand
        UPDRS-3_4b: UPDRS score for finger tapping test 3.4 left hand
        serial (string): the serial number of the subject
        hand (string): dominant hand
    """
    def __init__(self, subject_id, diagnosis, typist, side, years, UPDRS3_4a, UPDRS3_4b, UPDRS3_5a, UPDRS3_5b, UPDRS3_6a, UPDRS3_6b, serial, handedness, header_LW, header_RW, gyro_data_LW, gyro_FS_LW, gyro_timestamps_LW, gyro_data_RW, gyro_FS_RW, gyro_timestamps_RW):
        # General info
        self.subject_id = subject_id
        self.diagnosis = diagnosis
        self.serial = serial
        self.handedness = handedness
        self.side = side
        self.years = years
        self.typist = typist
        self.header_LW = header_LW
        self.header_RW = header_RW
        self.gyro_data_LW = gyro_data_LW
        self.gyro_FS_LW = gyro_FS_LW
        self.gyro_timestamps_LW = gyro_timestamps_LW
        self.gyro_data_RW = gyro_data_RW
        self.gyro_FS_RW = gyro_FS_RW
        self.gyro_timestamps_RW = gyro_timestamps_RW
        
        if (handedness == 'right'):
            self.UPDRS_dom = UPDRS3_6a 
            self.UPDRS_ndom = UPDRS3_6b 
        else:
            self.UPDRS_dom = UPDRS3_6b 
            self.UPDRS_ndom = UPDRS3_6a 
        
    def toDataframe(self):
        data = [self.subject_id, self.diagnosis, self.typist, self.side, self.years, self.UPDRS_dom, self.UPDRS_ndom, self.serial, self.handedness, self.side, self.years, self.header_LW, self.header_RW, self.gyro_data_LW, self.gyro_FS_LW, self.gyro_timestamps_LW, self.gyro_data_RW, self.gyro_FS_RW, self.gyro_timestamps_RW]
        features = ["subject_id", "diagnosis", "typist", "side", "years", "UPDRS_dom", "UPDRS_ndom", "serial", "handedness", "side", "years", "header_LW", "header_RW", "gyro_data_LW", "gyro_FS_LW", "gyro_timestamps_LW", "gyro_data_RW", "gyro_FS_RW", "gyro_timestamps_RW"] 
        data_t = pd.DataFrame(data).transpose()
        data_t.columns = features
        return data_t
        
class Subjects:
    
    def __init__(self, timeslot_choice):
        self.PD_OFF_ids = ["PD01_OFF", "PD03_OFF", "PD04_OFF", "PD05_OFF", "PD09_OFF", "PD13_OFF", "PD16_OFF", "PD17_OFF", "PD22_OFF", "PD25_OFF", "PD29_OFF", "PD31_OFF", "PD33_OFF", "PD34_OFF", "PD36_OFF", "PD37_OFF", "PD38_OFF", "PD39_OFF"] # "PD21_OFF", "PD08_OFF"
        self.PD_ON_ids = ["PD01_ON", "PD03_ON", "PD04_ON", "PD05_ON", "PD08_ON", "PD09_ON", "PD13_ON", "PD17_ON", "PD22_ON", "PD25_ON", "PD29_ON", "PD31_ON", "PD33_ON", "PD34_ON", "PD36_ON", "PD38_ON", "PD39_ON"] # "PD21_ON" "PD37_ON" "PD16_ON"
        self.HC_ids = ["HC01", "HC08", "HC09", "HC11", "HC12", "HC13", "HC14", "HC17", "HC20", "HC22", "HC23", "HC25", "HC27", "HC30", "HC33", "HC35", "HC36", "HC37"] # "HC19" "HC28"
        self.CA_ids = ["CA01", "CA02", "CA03", "CA11", "CA13", "CA15", "CA16", "CA25", "CA29", "CA37", "CA39", "CA40", "CA41", "CA44", "CA46", "CA52", "CA55", "CA56", "CA59"]
        self.subject_groups = [self.PD_OFF_ids, self.PD_ON_ids, self.HC_ids, self.CA_ids]
        self.features =  ["subject_id", "diagnosis", "typist", "side", "years", "UPDRS_dom", "UPDRS_ndom", "serial", "handedness", "side", "years", "header_LW", "header_RW", "gyro_data_LW", "gyro_FS_LW", "gyro_timestamps_LW", "gyro_data_RW", "gyro_FS_RW", "gyro_timestamps_RW"] 
        self.df = pd.DataFrame(columns = self.features)
        self.subjects = []
        
        for group in self.subject_groups:
            for id in group:
                s = createSubjectFromData(id, timeslot_choice)
                self.df = self.df.append(s.toDataframe())
                self.subjects.append(s)
                
    def getPD_OFF(self):
        return self.df[self.df['subject_id'].str.contains('PD') & self.df['subject_id'].str.contains('OFF')]
        
    def getPD_ON(self):
        return self.df[self.df['subject_id'].str.contains('PD') & self.df['subject_id'].str.contains('ON')]
        
    def getHC(self):
        return self.df[self.df['subject_id'].str.contains('HC')]
        
    def getCA(self):
        return self.df[self.df['subject_id'].str.contains('CA')]
    
class Wear4PD():
    def __init__(self, savepath="./models/", timeslot_choice = "full"):
        """
        Description:
            init method for class
        """
        # load subjects on initialization; subjects with flight and dwell times (top 5% outliers + first and last 10 taps removed)
        self.subjects = Subjects(timeslot_choice).subjects
        self.savepath = savepath
        self.timeslot_choice = timeslot_choice
        self.loadUPDRS()
        self.params = {}
    
    def preprocess(self, df):
        # normalize data
        contcols = [c for c in df.columns if c != "subject_id" and c != "UPDRS" and c != "diagnosis" and c != "side" and c != "typist" and c != "handedness" and c != "years" and c != "dominant" and c != "affected"]
        df[contcols] = MinMaxScaler().fit_transform(df[contcols])
        df = df.fillna(0.0)
       
        return df
    
    def loadUPDRS(self):
        df = pd.DataFrame(columns=['subject_id', 'handedness', 'diagnosis', 'UPDRS', 'typist', 'years', 'dominant', 'affected', 'mean_amplitude', 'var_amplitude', 'decrement_amplitude', 'main_freq', 'var_periods', 'decrement_periods', 'movements', 'hesitations', 'freezes', 'f1']) # add features: hesitations, halts, wavelets
            
        for s in self.subjects:
            print(s.subject_id)

            if (s.handedness == "left"):
                # left hand y-axis
                integral_LW, amplitudes_LW, durations_LW = getAmplitudes(s.gyro_timestamps_LW, s.gyro_data_LW[:,0])
                
                ## Amplitude Features:
                #'mean_amplitude_supination'
                mean_amplitude_supination_dom = np.mean(amplitudes_LW[amplitudes_LW > 0])
                #'mean_amplitude_pronation' 
                mean_amplitude_pronation_dom = np.mean(amplitudes_LW[amplitudes_LW < 0])
                #'mean_amplitude'
                mean_amplitude_dom = np.mean(np.abs(amplitudes_LW))
                #'var_amplitude_supination'
                var_amplitude_supination_dom = np.var(amplitudes_LW[amplitudes_LW > 0])
                #'var_amplitude_pronation' 
                var_amplitude_pronation_dom = np.var(amplitudes_LW[amplitudes_LW < 0])
                #'var_amplitude'
                var_amplitude_dom = np.var(np.abs(amplitudes_LW))
                #'decrement_amplitude_supination'
                polyfit = np.polyfit(range(len(amplitudes_LW[amplitudes_LW > 0])), amplitudes_LW[amplitudes_LW > 0], 1)
                decrement_amplitude_supination_dom = polyfit[0]
                #'decrement_amplitude_pronation' 
                polyfit = np.polyfit(range(len(amplitudes_LW[amplitudes_LW < 0])), amplitudes_LW[amplitudes_LW < 0], 1)
                decrement_amplitude_pronation_dom = polyfit[0]
                #'decrement_amplitude'
                polyfit = np.polyfit(range(len(amplitudes_LW)), np.abs(amplitudes_LW), 1)
                decrement_amplitude_dom = polyfit[0]
                
                ## Frequency/Speed Features:
                #'main_freq'
                z = fftpack.fft(s.gyro_data_LW[:, 0])
                freqs = fftpack.fftfreq(len(s.gyro_data_LW)) * s.gyro_FS_LW
                main_freq_dom = freqs[np.argmax(np.abs(z))]
                #'var_periods'
                var_periods_dom = np.var(durations_LW)
                #'decrement_periods'
                polyfit = np.polyfit(range(len(durations_LW)), durations_LW, 1)
                decrement_periods_dom = polyfit[0]
                #'movements'
                movements_dom = len(durations_LW)
                
                ## Hesitations and Halts
                dat = s.gyro_data_LW[:,0]
                t0 = s.gyro_timestamps_LW[0]
                dt = s.gyro_timestamps_LW[1] - s.gyro_timestamps_LW[0]
                t = s.gyro_timestamps_LW

                # Normalize signal
                p = np.polyfit(s.gyro_timestamps_LW - s.gyro_timestamps_LW[0], s.gyro_data_LW[:,0], 1)
                dat_notrend = s.gyro_data_LW[:,0] - np.polyval(p, s.gyro_timestamps_LW - s.gyro_timestamps_LW[0])
                std = dat_notrend.std()  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend / std  # Normalized dataset

                w = 1. # central frequency
                freq = np.linspace(1, s.gyro_FS_LW/2, 100)
                widths = w*s.gyro_FS_LW / (2*freq*np.pi)
                cwtm = signal.cwt(dat_norm, signal.morlet2, widths, w=w) # continuous wavelet transform
                
                (x, y) = np.where(np.abs(cwtm) == np.amax(np.abs(cwtm))) # get index of frequency characteristic
                f1_dom = freq[x]
                
                # Compute cross-sectional area by summing the CWT coefficients perpendicular to the time axis
                # Normalize with respect to its maximum value and is expressed as a percentage
                CSA_T = np.sum(np.abs(cwtm), 0)/np.max(np.sum(np.abs(cwtm), 0)) * 100

                # Compute thresholds
                hesitation_threshold = 0.5*np.mean(CSA_T)
                freeze_threshold = 0.25*np.mean(CSA_T)

                hesitations_dom = findHesitationsFreezing(CSA_T, hesitation_threshold, freq[x], dt)
                freezes_dom =  findHesitationsFreezing(CSA_T, freeze_threshold, freq[x], dt)

                del integral_LW, amplitudes_LW, durations_LW 
                
                # right hand y-axis
                integral_RW, amplitudes_RW, durations_RW = getAmplitudes(s.gyro_timestamps_RW, s.gyro_data_RW[:,0])
                
                ## Amplitude Features:
                #'mean_amplitude_supination'
                mean_amplitude_supination_ndom = np.mean(amplitudes_RW[amplitudes_RW > 0])
                #'mean_amplitude_pronation' 
                mean_amplitude_pronation_ndom = np.mean(amplitudes_RW[amplitudes_RW < 0])
                #'mean_amplitude'
                mean_amplitude_ndom = np.mean(np.abs(amplitudes_RW))
                #'var_amplitude_supination'
                var_amplitude_supination_ndom = np.var(amplitudes_RW[amplitudes_RW > 0])
                #'var_amplitude_pronation' 
                var_amplitude_pronation_ndom = np.var(amplitudes_RW[amplitudes_RW < 0])
                #'var_amplitude'
                var_amplitude_ndom = np.var(np.abs(amplitudes_RW))
                #'decrement_amplitude_supination'
                polyfit = np.polyfit(range(len(amplitudes_RW[amplitudes_RW > 0])), amplitudes_RW[amplitudes_RW > 0], 1)
                decrement_amplitude_supination_ndom = polyfit[0]
                #'decrement_amplitude_pronation' 
                polyfit = np.polyfit(range(len(amplitudes_RW[amplitudes_RW < 0])), amplitudes_RW[amplitudes_RW < 0], 1)
                decrement_amplitude_pronation_ndom = polyfit[0]
                #'decrement_amplitude'
                polyfit = np.polyfit(range(len(amplitudes_RW)), np.abs(amplitudes_RW), 1)
                decrement_amplitude_ndom = polyfit[0]
                
                ## Frequency/Speed Features:
                #'main_freq'
                z = fftpack.fft(s.gyro_data_RW[:, 0])
                freqs = fftpack.fftfreq(len(s.gyro_data_RW)) * s.gyro_FS_RW
                main_freq_ndom = freqs[np.argmax(np.abs(z))]
                #'var_periods'
                var_periods_ndom = np.var(durations_RW)
                #'decrement_periods'
                polyfit = np.polyfit(range(len(durations_RW)), durations_RW, 1)
                decrement_periods_ndom = polyfit[0]
                #'movements'
                movements_ndom = len(durations_RW)
                
                ## Hesitations and Halts
                dat = s.gyro_data_RW[:,0]
                t0 = s.gyro_timestamps_RW[0]
                dt = s.gyro_timestamps_RW[1] - s.gyro_timestamps_RW[0]
                t = s.gyro_timestamps_RW

                # Normalize signal
                p = np.polyfit(s.gyro_timestamps_RW - s.gyro_timestamps_RW[0], s.gyro_data_RW[:,0], 1)
                dat_notrend = s.gyro_data_RW[:,0] - np.polyval(p, s.gyro_timestamps_RW - s.gyro_timestamps_RW[0])
                std = dat_notrend.std()  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend / std  # Normalized dataset

                w = 1. # central frequency
                freq = np.linspace(1, s.gyro_FS_RW/2, 100)
                widths = w*s.gyro_FS_RW / (2*freq*np.pi)
                cwtm = signal.cwt(dat_norm, signal.morlet2, widths, w=w) # continuous wavelet transform
                
                (x, y) = np.where(np.abs(cwtm) == np.amax(np.abs(cwtm))) # get index of frequency characteristic
                f1_ndom = freq[x]
                
                # Compute cross-sectional area by summing the CWT coefficients perpendicular to the time axis
                # Normalize with respect to its maximum value and is expressed as a percentage
                CSA_T = np.sum(np.abs(cwtm), 0)/np.max(np.sum(np.abs(cwtm), 0)) * 100

                # Compute thresholds
                hesitation_threshold = 0.5*np.mean(CSA_T)
                freeze_threshold = 0.25*np.mean(CSA_T)

                hesitations_ndom = findHesitationsFreezing(CSA_T, hesitation_threshold, freq[x], dt)
                freezes_ndom =  findHesitationsFreezing(CSA_T, freeze_threshold, freq[x], dt)
                
                del integral_RW, amplitudes_RW, durations_RW 
                
                # Right affected
                if (s.side == 1):
                    affected_dom = False
                    affected_ndom = True
                # Left affected
                elif (s.side == 2):
                    affected_dom = True
                    affected_ndom = False
                # Both sides affected
                elif (s.side == 3):
                    affected_dom = True
                    affected_ndom = True
                # No information
                else:
                    affected_dom = False
                    affected_ndom = False
            else:
                # left hand y-axis
                integral_LW, amplitudes_LW, durations_LW = getAmplitudes(s.gyro_timestamps_LW, s.gyro_data_LW[:,0])
                
                ## Amplitude Features:
                #'mean_amplitude_supination'
                mean_amplitude_supination_ndom = np.mean(amplitudes_LW[amplitudes_LW > 0])
                #'mean_amplitude_pronation' 
                mean_amplitude_pronation_ndom = np.mean(amplitudes_LW[amplitudes_LW < 0])
                #'mean_amplitude'
                mean_amplitude_ndom = np.mean(np.abs(amplitudes_LW))
                #'var_amplitude_supination'
                var_amplitude_supination_ndom = np.var(amplitudes_LW[amplitudes_LW > 0])
                #'var_amplitude_pronation' 
                var_amplitude_pronation_ndom = np.var(amplitudes_LW[amplitudes_LW < 0])
                #'var_amplitude'
                var_amplitude_ndom = np.var(np.abs(amplitudes_LW))
                #'decrement_amplitude_supination'
                polyfit = np.polyfit(range(len(amplitudes_LW[amplitudes_LW > 0])), amplitudes_LW[amplitudes_LW > 0], 1)
                decrement_amplitude_supination_ndom = polyfit[0]
                #'decrement_amplitude_pronation' 
                polyfit = np.polyfit(range(len(amplitudes_LW[amplitudes_LW < 0])), amplitudes_LW[amplitudes_LW < 0], 1)
                decrement_amplitude_pronation_ndom = polyfit[0]
                #'decrement_amplitude'
                polyfit = np.polyfit(range(len(amplitudes_LW)), np.abs(amplitudes_LW), 1)
                decrement_amplitude_ndom = polyfit[0]
                
                ## Frequency/Speed Features:
                #'main_freq'
                z = fftpack.fft(s.gyro_data_LW[:, 0])
                freqs = fftpack.fftfreq(len(s.gyro_data_LW)) * s.gyro_FS_LW
                main_freq_ndom = freqs[np.argmax(np.abs(z))]
                #'var_periods'
                var_periods_ndom = np.var(durations_LW)
                #'decrement_periods'
                polyfit = np.polyfit(range(len(durations_LW)), durations_LW, 1)
                decrement_periods_ndom = polyfit[0]
                #'movements'
                movements_ndom = len(durations_LW)
                
                ## Hesitations and Halts
                dat = s.gyro_data_LW[:,0]
                t0 = s.gyro_timestamps_LW[0]
                dt = s.gyro_timestamps_LW[1] - s.gyro_timestamps_LW[0]
                t = s.gyro_timestamps_LW

                # Normalize signal
                p = np.polyfit(s.gyro_timestamps_LW - s.gyro_timestamps_LW[0], s.gyro_data_LW[:,0], 1)
                dat_notrend = s.gyro_data_LW[:,0] - np.polyval(p, s.gyro_timestamps_LW - s.gyro_timestamps_LW[0])
                std = dat_notrend.std()  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend / std  # Normalized dataset

                w = 1. # central frequency
                freq = np.linspace(1, s.gyro_FS_LW/2, 100)
                widths = w*s.gyro_FS_LW / (2*freq*np.pi)
                cwtm = signal.cwt(dat_norm, signal.morlet2, widths, w=w) # continuous wavelet transform
                
                (x, y) = np.where(np.abs(cwtm) == np.amax(np.abs(cwtm))) # get index of frequency characteristic
                f1_ndom = freq[x]
                
                # Compute cross-sectional area by summing the CWT coefficients perpendicular to the time axis
                # Normalize with respect to its maximum value and is expressed as a percentage
                CSA_T = np.sum(np.abs(cwtm), 0)/np.max(np.sum(np.abs(cwtm), 0)) * 100

                # Compute thresholds
                hesitation_threshold = 0.5*np.mean(CSA_T)
                freeze_threshold = 0.25*np.mean(CSA_T)

                hesitations_ndom = findHesitationsFreezing(CSA_T, hesitation_threshold, freq[x], dt)
                freezes_ndom =  findHesitationsFreezing(CSA_T, freeze_threshold, freq[x], dt)

                del integral_LW, amplitudes_LW, durations_LW 
                
                # right hand y-axis
                integral_RW, amplitudes_RW, durations_RW = getAmplitudes(s.gyro_timestamps_RW, s.gyro_data_RW[:,0])
    
                ## Amplitude Features:
                #'mean_amplitude_supination'
                mean_amplitude_supination_dom = np.mean(amplitudes_RW[amplitudes_RW > 0])
                #'mean_amplitude_pronation' 
                mean_amplitude_pronation_dom = np.mean(amplitudes_RW[amplitudes_RW < 0])
                #'mean_amplitude'
                mean_amplitude_dom = np.mean(np.abs(amplitudes_RW))
                #'var_amplitude_supination'
                var_amplitude_supination_dom = np.var(amplitudes_RW[amplitudes_RW > 0])
                #'var_amplitude_pronation' 
                var_amplitude_pronation_dom = np.var(amplitudes_RW[amplitudes_RW < 0])
                #'var_amplitude'
                var_amplitude_dom = np.var(np.abs(amplitudes_RW))
                #'decrement_amplitude_supination'
                polyfit = np.polyfit(range(len(amplitudes_RW[amplitudes_RW > 0])), amplitudes_RW[amplitudes_RW > 0], 1)
                decrement_amplitude_supination_dom = polyfit[0]
                #'decrement_amplitude_pronation' 
                polyfit = np.polyfit(range(len(amplitudes_RW[amplitudes_RW < 0])), amplitudes_RW[amplitudes_RW < 0], 1)
                decrement_amplitude_pronation_dom = polyfit[0]
                #'decrement_amplitude'
                polyfit = np.polyfit(range(len(amplitudes_RW)), np.abs(amplitudes_RW), 1)
                decrement_amplitude_dom = polyfit[0]
                
                ## Frequency/Speed Features:
                #'main_freq'
                z = fftpack.fft(s.gyro_data_RW[:, 0])
                freqs = fftpack.fftfreq(len(s.gyro_data_RW)) * s.gyro_FS_RW
                main_freq_dom = freqs[np.argmax(np.abs(z))]
                #'var_periods'
                var_periods_dom = np.var(durations_RW)
                #'decrement_periods'
                polyfit = np.polyfit(range(len(durations_RW)), durations_RW, 1)
                decrement_periods_dom = polyfit[0]
                #'movements'
                movements_dom = len(durations_RW)
                
                ## Hesitations and Halts
                dat = s.gyro_data_RW[:,0]
                t0 = s.gyro_timestamps_RW[0]
                dt = s.gyro_timestamps_RW[1] - s.gyro_timestamps_RW[0]
                t = s.gyro_timestamps_RW

                # Normalize signal
                p = np.polyfit(s.gyro_timestamps_RW - s.gyro_timestamps_RW[0], s.gyro_data_RW[:,0], 1)
                dat_notrend = s.gyro_data_RW[:,0] - np.polyval(p, s.gyro_timestamps_RW - s.gyro_timestamps_RW[0])
                std = dat_notrend.std()  # Standard deviation
                var = std ** 2  # Variance
                dat_norm = dat_notrend / std  # Normalized dataset

                w = 1. # central frequency
                freq = np.linspace(1, s.gyro_FS_RW/2, 100)
                widths = w*s.gyro_FS_RW / (2*freq*np.pi)
                cwtm = signal.cwt(dat_norm, signal.morlet2, widths, w=w) # continuous wavelet transform
                
                (x, y) = np.where(np.abs(cwtm) == np.amax(np.abs(cwtm))) # get index of frequency characteristic
                f1_dom = freq[x]
                
                # Compute cross-sectional area by summing the CWT coefficients perpendicular to the time axis
                # Normalize with respect to its maximum value and is expressed as a percentage
                CSA_T = np.sum(np.abs(cwtm), 0)/np.max(np.sum(np.abs(cwtm), 0)) * 100

                # Compute thresholds
                hesitation_threshold = 0.5*np.mean(CSA_T)
                freeze_threshold = 0.25*np.mean(CSA_T)

                hesitations_dom = findHesitationsFreezing(CSA_T, hesitation_threshold, freq[x], dt)
                freezes_dom =  findHesitationsFreezing(CSA_T, freeze_threshold, freq[x], dt)

                del integral_RW, amplitudes_RW, durations_RW 
                
                # Right affected
                if (s.side == 1):
                    affected_dom = True
                    affected_ndom = False
                # Left affected
                elif (s.side == 2):
                    affected_dom = False
                    affected_ndom = True
                # Both sides affected
                elif (s.side == 3):
                    affected_dom = True
                    affected_ndom = True
                # No information
                else:
                    affected_dom = False
                    affected_ndom = False

            # append dom hand
            df = df.append({'subject_id' : s.subject_id, 
                            'handedness' : s.handedness, 
                            'diagnosis' : s.diagnosis, 
                            'UPDRS' : s.UPDRS_dom, 
                            'typist' : s.typist, 
                            'years' : s.years, 
                            'dominant' : True, 
                            'affected' : affected_dom,
                            'mean_amplitude' : mean_amplitude_dom, 
                            'var_amplitude' : var_amplitude_dom, 
                            'decrement_amplitude' : decrement_amplitude_dom, 
                            'main_freq' : main_freq_dom, 
                            'var_periods' : var_periods_dom, 
                            'decrement_periods' : decrement_periods_dom, 
                            'movements' : movements_dom,
                            'hesitations' : hesitations_dom, 
                            'freezes' : freezes_dom, 
                            'f1' : f1_dom}, ignore_index=True) 
            # append ndom hand        
            df = df.append({'subject_id' : s.subject_id, 
                            'handedness' : s.handedness, 
                            'diagnosis' : s.diagnosis, 
                            'UPDRS' : s.UPDRS_ndom, 
                            'typist' : s.typist, 
                            'years' : s.years, 
                            'dominant' : False, 
                            'affected' : affected_ndom,
                            'mean_amplitude' : mean_amplitude_ndom, 
                            'var_amplitude' : var_amplitude_ndom, 
                            'decrement_amplitude' : decrement_amplitude_ndom, 
                            'main_freq' : main_freq_ndom, 
                            'var_periods' : var_periods_ndom, 
                            'decrement_periods' : decrement_periods_ndom, 
                            'movements' : movements_ndom,
                            'hesitations' : hesitations_ndom, 
                            'freezes' : freezes_ndom, 
                            'f1' : f1_ndom}, ignore_index=True) 
        
        # Pre-process the data
        self.df_raw = df.copy(deep=True)
        self.df = self.preprocess(df)
        
        self.X = self.df.drop(columns=["subject_id", "UPDRS", "diagnosis", "handedness", "years", "dominant", "affected"]).to_numpy()
        self.y = self.df['UPDRS'].to_numpy()
        self.label_dict = dict(zip([0, 1, 2, 3], ['UPDRS_0', 'UPDRS_1', 'UPDRS_2','UPDRS_3']))   
    
    def save(self, model, modelname):
        """
        Description:
            method to save trained model
        
        Arguments:
            model - trained model
        """
        with open(self.savepath + modelname, 'wb') as file:
            pickle.dump(model, file)
            
    def gridsearch(self, param_grid, clf, scoring=None, k=5, n_jobs=4, verbose=2, obj = "BA"):
        """
        Description:
            method to perform Grid Search CV
        Arguments:
            param_grid - [dictionary] parameter grid settings
            clf - [sklearn] classifier model for grid search
            k - [int] k-fold value
            n_jobs - [int] number of jobs (Parallelization)
        Returns:
            cv_clf - [sklearn] trained model after grid search
        """
        cv_clf = GridSearchCV(clf, param_grid, scoring=scoring, refit=obj, n_jobs=n_jobs, cv=k, verbose=verbose)
        cv_clf.fit(self.X_train, self.y_train)
        return cv_clf


    def get_params(self, key):
        return self.params[key]
    
    def set_params(self, key, params):
        self.params[key] = params
            
    def plotIntegrals(self):
        for s in self.subjects:
            if (s.handedness == "left"):
                # Right affected
                if (s.side == 1):
                    right_side = " (affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                    left_side = " (not affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                # Left affected
                elif (s.side == 2):
                    right_side = " (not affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                    left_side = " (affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                # Both sides affected
                elif (s.side == 3):
                    right_side = " (affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                    left_side = " (affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                # No information
                else:
                    right_side = " (not affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                    left_side = " (not affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
            else:
                # Right affected
                if (s.side == 1):
                    right_side = " (affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                    left_side = " (not affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                # Left affected
                elif (s.side == 2):
                    right_side = " (not affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                    left_side = " (affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                # Both sides affected
                elif (s.side == 3):
                    right_side = " (affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                    left_side = " (affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                # No information
                else:
                    right_side = " (not affected, dominant, UPDRS " + str(s.UPDRS_dom) + ")"
                    left_side = " (not affected, non-dominant, UPDRS " + str(s.UPDRS_ndom) + ")"
                    
            integral_RW, absAmp_RW, durations_RW = getAmplitudes(s.gyro_timestamps_RW, s.gyro_data_RW[:,0])
            integral_LW, absAmp_LW, durations_LW = getAmplitudes(s.gyro_timestamps_LW, s.gyro_data_LW[:,0])
            (maxi_RW,), (mini_RW,) = segmentIntegrals(integral_RW)
            (maxi_LW,), (mini_LW,) = segmentIntegrals(integral_LW)

            fig, axs = plt.subplots(2, 2, figsize = (20,10))
            fig.suptitle(s.subject_id + ' Integrals', fontsize=18)
            # left hand y-axis raw
            axs[0, 0].plot(s.gyro_timestamps_LW, s.gyro_data_LW[:,0])
            axs[0, 0].set_ylabel('Angular Velocity [deg/s]', fontsize = 14)
            axs[0, 0].set_title('Left Hand Y-Axis' + left_side + ", Nr of Periods: " + str(len(durations_LW)/2), fontsize=16)

            # right hand y-axis raw
            axs[0, 1].plot(s.gyro_timestamps_RW, s.gyro_data_RW[:,0])
            axs[0, 1].set_title('Right Hand Y-Axis' + right_side + ", Nr of Periods: " + str(len(durations_RW)/2), fontsize=16)

            # left hand y-axis integral
            axs[1, 0].plot(s.gyro_timestamps_LW, integral_LW)
            axs[1, 0].plot(s.gyro_timestamps_LW[mini_LW], integral_LW[mini_LW], "or")
            axs[1, 0].plot(s.gyro_timestamps_LW[maxi_LW], integral_LW[maxi_LW], "og")
            axs[1, 0].set_ylabel('Angular Displacement [deg]', fontsize = 14)
            axs[1, 0].set_xlabel('Time [s]', fontsize = 14)
            axs[1, 0].set_title('Left Hand Y-Axis Integral' + left_side + ", Nr of Periods: " + str(len(durations_LW)/2), fontsize=16)

            # right hand y-axis integral
            axs[1, 1].plot(s.gyro_timestamps_RW, integral_RW)
            axs[1, 1].plot(s.gyro_timestamps_RW[mini_RW], integral_RW[mini_RW], "or")
            axs[1, 1].plot(s.gyro_timestamps_RW[maxi_RW], integral_RW[maxi_RW], "og")
            axs[1, 1].set_xlabel('Time [s]', fontsize = 14)
            axs[1, 1].set_title('Right Hand Y-Axis Integral' + right_side + ", Nr of Periods: " + str(len(durations_RW)/2), fontsize=16)

            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

            fig.savefig('Plots/' + s.subject_id + '_Integrals.png')

def run(clf, data, n_runs=30, output=None):    
    """
    Run the evaluation of a classifier for n times.

    Args:
        clf    = [Classifier] classifier to evaluate 
        data   = [Tadpole] Tadpole dataset
        n_runs = [int] number of runs
        output = [string] save path of the output

    Returns [pd.DataFrame]:
        Scores from the evaluation of the classifier.
    """
    evaluator = Evaluator(clf, data, n_runs=n_runs)
    evaluator.evaluate()
    if output:
        evaluator.export_to_csv(output)
    return evaluator.get_scores()
        