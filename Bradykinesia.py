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


def readTestFile(filename):
    """Reads JSON file and returns the data as a Python dictionary.
    
    Parameters:
        filename(string): name of the JSON file
    
    Returns: 
        dict: data of JSON file
    """
    f = open(filename,) 
    data = json.load(f) 
    
    return data

def createSubjectFromData(filename):
    """Given a dictionary of the data, an object of the class Subject is initiated and returned.
    
    Parameters:
        filename(dict): filename of the JSON file
    
    Returns: 
        Subject object: initiated object of the Subject class containing the data
    """
    data = readTestFile(filename)  
    
    if (filename.count("PD") == 1 and filename.count("OFF") == 1):
        diagnosis = 0
    elif  (filename.count("PD") == 1 and filename.count("ON") == 1):
        diagnosis = 1
    elif  (filename.count("CA") == 1):
        diagnosis = 2
    else:
        diagnosis = 3
    s = Subject(filename, diagnosis, data['typist'], data['side'], data['years'], data['UPDRS-3_4a'], data['UPDRS-3_4b'], data['UPDRS-3_5a'], data['UPDRS-3_5b'], data['UPDRS-3_6a'], data['UPDRS-3_6b'], data['serial'], data['hand'], data['tm'], data['d']['0'], data['d']['1'], data['d']['2'], data['d']['3'], data['d']['4'], data['d']['5'])
    return s

def getOtherTargetKey(target1, target2, current):
    """Returns the key code of the target key that is not the current key.
    
    Parameters:
        target1(int): key code of the target key 1 of the FTT
        target2(int): key code of the target key 2 of the FTT
        current(int): current target key
        
    Returns: 
        int: key code of the target key
    """
    if current == target1:
        return target2
    else:
        return target1

def getOppositeKeyPos(pos):
    """Returns the opposite key position. 
    
    Parameters:
        pos(int): current key position
    
    Returns: 
        int: opposite key position
    """
    if pos == 0:
        return 1
    else:
        return 0

def computeFlightTimesWithTwoTargets(target1, target2, datalist, time, starttime = 0):
    """Returns the flight times of the FTT with two target keys in a list.
    
    Parameters:
        target1(int): key code of target key 1 of the FT
        target2(int): key code of target key 2 of the FT
        datalist(list): data list which contains the information about when which key was pressed/released
        time(int): time interval
        starttime(int): start of the time interval
    
    Returns: 
        list: list of flight times
        int: nr of errors
    """
    ft = []
    nrOfErrors = 0
    start = datalist[0]['e'] + starttime
    end = start + time  

    # Start with the first release of a target key
    i = 0
    while (datalist[i]['p'] != 1 or (datalist[i]['k'] != target1 and datalist[i]['k'] != target2) or datalist[i]['e'] < start):
        if (datalist[i]['k'] != target1 and datalist[i]['k'] != target2):
            nrOfErrors = nrOfErrors + 1
        i = i + 1
        if (i > len(datalist)-1):
            break
    if (i > len(datalist)-1):
        return [], 0
    else:
        # Set current target, pos, and time of first release
        currentTargetKey = datalist[i]['k']
        currentKeyPos = datalist[i]['p']
        currentKeyTime = datalist[i]['e']

        # Loop through data of key presses/releases
        for d in datalist[i+1:]:

            # If next key is pressed
            if (d['p'] == 0 and d['e'] < end):
                # and if key is the other target key
                if (d['k'] == getOtherTargetKey(target1, target2, currentTargetKey)):
                    # compute flight time and append it to list
                    ft.append(d['e'] - currentKeyTime)

                    # Update current target key and time
                    currentTargetKey = d['k']
                    currentKeyTime = d['e']

                # if the key is not a target key
                elif (d['k'] != target1 and d['k'] != target2):
                    # increase number of errors by 1
                    nrOfErrors = nrOfErrors + 1

            elif (d['p'] == 1 and d['e'] < end):
                # and if key is the other target key
                if (d['k'] == currentTargetKey):
                    currentKeyTime = d['e']
                # if the key is not a target key
                elif (d['k'] != target1 and d['k'] != target2):
                    # increase number of errors by 1
                    nrOfErrors = nrOfErrors + 1      
        return ft, nrOfErrors

def computeFlightTimesWithOneTarget(target, datalist, time, starttime = 0):
    """Returns the flight times of the FTT with one target key in a list.
    
    Parameters:
        target(int): key code of the target key of the FT
        datalist(list): data list which contains the information about when which key was pressed/released
        time(int): time interval
        starttime(int): start time of the time interval
    
    Returns: 
        list: list of flight times
        int: number of errors
    """
    ft = []
    nrOfErrors = 0
    errorFT = False
    
    start = datalist[0]['e'] + starttime
    end = start + time 
    
    # Start with the first release of a target key
    i = 0
    while (datalist[i]['p'] != 1 or datalist[i]['k'] != target or datalist[i]['e'] < start):
        if (datalist[i]['k'] != target):
            nrOfErrors = nrOfErrors + 1
        i = i + 1
        if (i > len(datalist)-1):
            break
    if (i > len(datalist)-1):
        return [], 0
    else:
        # Set current time of first release
        currentKeyTime = datalist[i]['e']   

        # Loop through data of key presses/releases
        for d in datalist[i+1:]:
            # If next key is pressed
            if (d['p'] == 0 and d['e'] < end):
                # and if key is the target key
                if (d['k'] == target):
                    # compute flight time and append it to list if not error FT
                    ft.append(d['e'] - currentKeyTime)
                    # Update current time
                    currentKeyTime = d['e']
                # if the key is not the target key
                elif (d['k'] != target):
                    # increase number of errors by 1
                    nrOfErrors = nrOfErrors + 1
            elif (d['p'] == 1 and d['e'] < end):
                # and if key is the target key
                if (d['k'] == target):
                    # Update current time
                    currentKeyTime = d['e']
                # if the key is not the target key
                elif (d['k'] != target):
                    # increase number of errors by 1
                    nrOfErrors = nrOfErrors + 1
        return ft, nrOfErrors
    
def getDS(datalist, target, time, adjacent, target2 = None):
    """
    Returns the dysmetria score (DS; a measure of the average accuracy of key strikes where the central keyscores 1, adjacent keys are 2, and all other keys are 3)
    
    Parameters:
        target(int): key code of the target key of the FT
        datalist(list): data list which contains the information about when which key was pressed/released
        time(int): time interval
        adjacent(list): list of adjacent keys
        target2(int): optional second target key
    
    Returns: 
        float: DS 
    """
    err = []
    nrOfTaps = 0
    
    start = datalist[0]['e']
    end = start + time 
    
    # Loop through data of key presses/releases
    for d in datalist:
        # If next key is pressed
        if (d['p'] == 0 and d['e'] < end):
            # increase nr of taps
            nrOfTaps = nrOfTaps + 1
            
            # and if key is one of the target key(s)
            if (d['k'] == target or d['k'] == target2):
                # append 1
                err.append(1)
                
            # if the key is one of the adjacent keys
            elif (d['k'] in adjacent):
                # append 2
                err.append(2)
                
            else:
                # append 3
                err.append(3)
                
    # return sum of err list divided by total number of taps          
    return np.sum(err)/nrOfTaps

def computeDwellTimes(datalist, target, time, target2 = None):
    """Returns the dwell times of the FTT in a list.
    
    Parameters:
        datalist(list): data list which contains the information about when which key was pressed/released
        target(int): key code of the target key of the DT
        interval(int): interval time in msec
        target2(int): key code of optional second target
    
    
    Returns: 
        list: list of dwell times with or without 'x' marking the ends of intervals
    """
    dt = []
    i = 0
    
    start = datalist[0]['e']
    end = start + time    
    
    for kPress in datalist:
        if (kPress['p'] == 0 and (kPress['k'] == target or kPress['k'] == target2) and (kPress['e'] < end)):
            for kRelease in datalist[i + 1:]:
                if ((kRelease['p'] == 1) and (kPress['k'] == kRelease['k'])):
                    dt.append(kRelease['e'] - kPress['e'])
                    break
        i = i + 1
                
    return dt

def reject_outliers(data, m=3, remove = 0):
    """Rejects outliers and returns the number of found outliers

        Parameters:
            data(list): data list of DTs/FTs
            m(int): number of standard deviations
            remove(int): number of elements to remove from the start and end of list

        Returns:
            list: data list without outliers
            int: number of outliers
    """
    npdata = np.array(data)
    out = np.count_nonzero((abs(npdata - np.mean(npdata)) < m * np.std(npdata)) == 0)
    if (remove > 0):
        new_data = npdata[abs(npdata - np.mean(npdata)) < m * np.std(npdata)][remove:-remove]
    else:
        new_data = npdata[abs(npdata - np.mean(npdata)) < m * np.std(npdata)]
    return new_data, out
                
def getSequenceEffectScore(fts):
    """Returns sequence effect score accounting for recovery.

        Parameters:
            fts(list): list of flight times

        Returns:
            float: sequence effect score
    """
#     sums = []
#     sum = 0
#     i = 0
#     for ft in fts:
#         if (i < 3):
#             sum = sum + ft
#             i = i + 1
#         else:
#             sums.append(sum)
#             sum = 0
#             i = 0
            
#     diff = [x - sums[i - 1] for i, x in enumerate(sums)][1:]
#     return np.var(diff)
    slopes = []
    x = []
    i = 0
    for ft in fts:
        if (i < 5):
            x.append(ft)
            i = i + 1
        else:
            slope, _, _, _, _ = stats.linregress(range(5), x)
            slopes.append(slope)
            x = []
            i = 0
            
    return np.var(slopes)

def computeVelocityScore2(fts):
    """Returns velocity score based on BRAIN test (based on every FT separately).

        Parameters:
            fts(list): list of flight times

        Returns:
            float: velocity score
    """
    vs_dom =  [(16.0/(x + 0.00001)) for x in fts]
    vs_perc_dom = [((x/float(vs_dom[0])) * 100.0) - 100.0 for x in vs_dom]
    vs_slope_dom, _, _, _, _ = stats.linregress(range(len(vs_perc_dom)), vs_perc_dom)
    return vs_slope_dom #np.mean(vs_perc_dom)

def computeVelocityScore(target1, keypresses, target2 = None):
    """Returns velocity score based on BRAIN test (with time intervals).

        Parameters:
            fts(list): list of flight times

        Returns:
            float: velocity score
    """
    start = 0
    step = 2000
    secs = step/1000
    vss = []
    if (target2 == None):
        v, _ = computeFlightTimesWithOneTarget(target1, keypresses, step, starttime = start)
        start = start + step
        velocity = len(v) / secs
        baselineVS = velocity + 0.000000000001
        while (start < 60000):
            v, _ = computeFlightTimesWithOneTarget(target1, keypresses, step, starttime = start)
            velocity = len(v) / secs
            change = (velocity / baselineVS) * 100.0
            vss.append(change)
            start = start + step
    else:
        v, _ = computeFlightTimesWithTwoTargets(target1, target2, keypresses, step, starttime = start)
        start = start + step
        velocity = len(v) / secs
        baselineVS = velocity + 0.000000000001
        while (start < 60000):
            v, _ = computeFlightTimesWithTwoTargets(target1, target2, keypresses, step, starttime = start)
            velocity = len(v) / secs
            change = (velocity / baselineVS) * 100.0
            vss.append(change)
            start = start + step
    slope, intercept, _, _, std_err = stats.linregress(range(len(vss)), vss)
    return slope, intercept, std_err

class Subject:
    """ 
    This class contains all collected information about the subject from the FTT.
    """
    def __init__(self, subject_id, diagnosis, typist, side, years, UPDRS3_4a, UPDRS3_4b, UPDRS3_5a, UPDRS3_5b, UPDRS3_6a, UPDRS3_6b, serial, hand, tm, qp_dom, qp_ndom, mn_dom, mn_ndom, m_dom, m_ndom):
        # General info
        self.subject_id = subject_id
        self.diagnosis = diagnosis
        self.serial = serial
        self.hand = hand
        self.tm = tm
        self.side = side
        self.years = years
        self.typist = typist
        
        # Amount of UPDRS scores
        n_a = np.count_nonzero(~np.isnan([UPDRS3_4a, UPDRS3_5a, UPDRS3_6a]))
        n_b = np.count_nonzero(~np.isnan([UPDRS3_4b, UPDRS3_5b, UPDRS3_6b]))
        
        if (hand == 'right'):
            self.UPDRS_dom = np.round(np.sum([i for i in [UPDRS3_4a, UPDRS3_5a, UPDRS3_6a] if ~np.isnan(i)]) / n_a)
            self.UPDRS_ndom = np.round(np.sum([i for i in [UPDRS3_4b, UPDRS3_5b, UPDRS3_6b] if ~np.isnan(i)]) / n_b)
        else:
            self.UPDRS_dom = np.round(np.sum([i for i in [UPDRS3_4b, UPDRS3_5b, UPDRS3_6b] if ~np.isnan(i)]) / n_b)
            self.UPDRS_ndom = np.round(np.sum([i for i in [UPDRS3_4a, UPDRS3_5a, UPDRS3_6a] if ~np.isnan(i)]) / n_a)
        
        # List of key presses/releases of the different tests
        self.qp_dom = qp_dom
        self.qp_ndom = qp_ndom
        self.mn_dom = mn_dom
        self.mn_ndom = mn_ndom
        self.m_dom = m_dom
        self.m_ndom = m_ndom
        
        # Flight times and errors
        self.qp_dom_ft, self.qp_dom_err = computeFlightTimesWithTwoTargets(80, 81, qp_dom, 60000)
        self.qp_ndom_ft, self.qp_ndom_err = computeFlightTimesWithTwoTargets(80, 81, qp_ndom, 60000)
        self.mn_dom_ft, self.mn_dom_err = computeFlightTimesWithTwoTargets(77, 78, mn_dom, 60000)
        self.mn_ndom_ft, self.mn_ndom_err = computeFlightTimesWithTwoTargets(77, 78, mn_ndom, 60000)
        self.m_dom_ft, self.m_dom_err = computeFlightTimesWithOneTarget(77, m_dom, 60000)
        self.m_ndom_ft, self.m_ndom_err = computeFlightTimesWithOneTarget(77, m_ndom, 60000)
        
        # Velocity score per 2 second intervals
        self.qp_dom_vs, _, _ = computeVelocityScore(80, qp_dom, 81)
        self.qp_ndom_vs, _, _ = computeVelocityScore(80, qp_ndom, 81)
        self.mn_dom_vs, _, _ = computeVelocityScore(77, mn_dom, 78)
        self.mn_ndom_vs, _, _ = computeVelocityScore(77, mn_ndom, 78)
        self.m_dom_vs, _, _ = computeVelocityScore(77, m_dom)
        self.m_ndom_vs, _, _ = computeVelocityScore(77, m_ndom)
        
        # Dwell times
        self.qp_dom_dt = computeDwellTimes(qp_dom, 80, 60000, 81)
        self.qp_ndom_dt = computeDwellTimes(qp_ndom, 80, 60000, 81)
        self.mn_dom_dt = computeDwellTimes(mn_dom, 77, 60000, 78)
        self.mn_ndom_dt = computeDwellTimes(mn_ndom, 77, 60000, 78)
        self.m_dom_dt = computeDwellTimes(m_dom, 77, 60000)
        self.m_ndom_dt = computeDwellTimes(m_ndom, 77, 60000)
        
        # Flight times and dwell times for first 10 seconds
        self.qp_dom_ft_10, self.qp_dom_err_10 = computeFlightTimesWithTwoTargets(80, 81, qp_dom, 10000)
        self.qp_ndom_ft_10, self.qp_ndom_err_10 = computeFlightTimesWithTwoTargets(80, 81, qp_ndom, 10000)
        self.mn_dom_ft_10, self.mn_dom_err_10 = computeFlightTimesWithTwoTargets(77, 78, mn_dom, 10000)
        self.mn_ndom_ft_10, self.mn_ndom_err_10 = computeFlightTimesWithTwoTargets(77, 78, mn_ndom, 10000)
        self.m_dom_ft_10, self.m_dom_err_10 = computeFlightTimesWithOneTarget(77, m_dom, 10000)
        self.m_ndom_ft_10, self.m_ndom_err_10 = computeFlightTimesWithOneTarget(77, m_ndom, 10000)
        self.qp_dom_dt_10 = computeDwellTimes(qp_dom, 80, 10000, 81)
        self.qp_ndom_dt_10 = computeDwellTimes(qp_ndom, 80, 10000, 81)
        self.mn_dom_dt_10 = computeDwellTimes(mn_dom, 77, 10000, 78)
        self.mn_ndom_dt_10 = computeDwellTimes(mn_ndom, 77, 10000, 78)
        self.m_dom_dt_10 = computeDwellTimes(m_dom, 77, 10000)
        self.m_ndom_dt_10 = computeDwellTimes(m_ndom, 77, 10000)
        
        # Hasan 2019 features
        self.qp_dom_ft_30, _ = computeFlightTimesWithTwoTargets(80, 81, qp_dom, 30000)
        self.qp_ndom_ft_30, _ = computeFlightTimesWithTwoTargets(80, 81, qp_ndom, 30000)
        self.qp_dom_dt_30 = computeDwellTimes(qp_dom, 80, 30000, 81)
        self.qp_ndom_dt_30 = computeDwellTimes(qp_ndom, 80, 30000, 81)
        self.qp_dom_DS_30 = getDS(qp_dom, 80, 30000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_ndom_DS_30 = getDS(qp_ndom, 80, 30000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_dom_VS_30 = computeVelocityScore2(self.qp_dom_ft_30)
        self.qp_ndom_VS_30 = computeVelocityScore2(self.qp_ndom_ft_30)
        
        self.qp_dom_DS_60 = getDS(qp_dom, 80, 60000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_ndom_DS_60 = getDS(qp_ndom, 80, 60000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_dom_VS_60 = computeVelocityScore2(self.qp_dom_ft)
        self.qp_ndom_VS_60 = computeVelocityScore2(self.qp_ndom_ft)
        
        self.qp_dom_DS_10 = getDS(qp_dom, 80, 10000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_ndom_DS_10 = getDS(qp_ndom, 80, 10000, [79, 48, 45, 91, 59, 76, 9, 49, 50, 87, 65, 20], 81)
        self.qp_dom_VS_10 = computeVelocityScore2(self.qp_dom_ft_10)
        self.qp_ndom_VS_10 = computeVelocityScore2(self.qp_ndom_ft_10)
        
        self.mn_dom_DS_60 = getDS(mn_dom, 77, 60000, [74, 75, 44, 32, 66, 72], 78)
        self.mn_ndom_DS_60 = getDS(mn_ndom, 77, 60000, [74, 75, 44, 32, 66, 72], 78)
        self.m_dom_DS_60 = getDS(m_dom, 77, 60000, [78, 74, 75, 32, 44])
        self.m_ndom_DS_60 = getDS(m_ndom, 77, 60000, [78, 74, 75, 32, 44])

        # Preprocessing of FT and DT: reject outliers >2SD
        self.qp_dom_dt, self.qp_dom_out_dt = reject_outliers(self.qp_dom_dt, m=2, remove = 0)
        self.qp_ndom_dt, self.qp_ndom_out_dt = reject_outliers(self.qp_ndom_dt, m=2, remove = 0)
        self.mn_dom_dt, self.mn_dom_out_dt = reject_outliers(self.mn_dom_dt, m=2, remove = 0)
        self.mn_ndom_dt, self.mn_ndom_out_dt = reject_outliers(self.mn_ndom_dt, m=2, remove = 0)
        self.m_dom_dt, self.m_dom_out_dt = reject_outliers(self.m_dom_dt, m=2, remove = 0)
        self.m_ndom_dt, self.m_ndom_out_dt = reject_outliers(self.m_ndom_dt, m=2, remove = 0)
        
        self.qp_dom_ft, self.qp_dom_out_ft = reject_outliers(self.qp_dom_ft, m=2, remove = 0)
        self.qp_ndom_ft, self.qp_ndom_out_ft = reject_outliers(self.qp_ndom_ft, m=2, remove = 0)
        self.mn_dom_ft, self.mn_dom_out_ft = reject_outliers(self.mn_dom_ft, m=2, remove = 0)
        self.mn_ndom_ft, self.mn_ndom_out_ft = reject_outliers(self.mn_ndom_ft, m=2, remove = 0)
        self.m_dom_ft, self.m_dom_out_ft = reject_outliers(self.m_dom_ft, m=2, remove = 0)
        self.m_ndom_ft, self.m_ndom_out_ft = reject_outliers(self.m_ndom_ft, m=2, remove = 0)
        
    def toDataframe(self):
        """
            Transforms all attributes into a dataframe
        """
        data = [self.subject_id, self.diagnosis, self.typist, self.side, self.years, self.UPDRS_dom, self.UPDRS_ndom, self.serial, self.hand, self.tm, self.qp_dom, self.qp_ndom, self.mn_dom, self.mn_ndom, self.m_dom, self.m_ndom, self.qp_dom_ft, self.qp_ndom_ft, self.mn_dom_ft, self.mn_ndom_ft, self.m_dom_ft, self.m_ndom_ft, self.qp_dom_err, self.qp_ndom_err, self.mn_dom_err, self.mn_ndom_err, self.m_dom_err, self.m_ndom_err, self.qp_dom_dt, self.qp_ndom_dt, self.mn_dom_dt, self.mn_ndom_dt, self.m_dom_dt, self.m_ndom_dt, self.qp_dom_ft_10, self.qp_ndom_ft_10, self.mn_dom_ft_10, self.mn_ndom_ft_10, self.m_dom_ft_10, self.m_ndom_ft_10, self.qp_dom_dt_10, self.qp_ndom_dt_10, self.mn_dom_dt_10, self.mn_ndom_dt_10, self.m_dom_dt_10, self.m_ndom_dt_10, self.qp_dom_err_10, self.qp_ndom_err_10, self.mn_dom_err_10, self.mn_ndom_err_10, self.m_dom_err_10, self.m_ndom_err_10, self.qp_dom_vs, self.qp_ndom_vs, self.mn_dom_vs, self.mn_ndom_vs, self.m_dom_vs, self.m_ndom_vs, self.qp_dom_ft_30, self.qp_ndom_ft_30, self.qp_dom_dt_30, self.qp_ndom_dt_30, self.qp_dom_DS_30, self.qp_ndom_DS_30, self.qp_dom_VS_30, self.qp_ndom_VS_30, self.qp_dom_DS_60, self.qp_ndom_DS_60, self.qp_dom_VS_60, self.qp_ndom_VS_60, self.qp_dom_DS_10, self.qp_ndom_DS_10, self.qp_dom_VS_10, self.qp_ndom_VS_10, self.qp_dom_out_dt, self.qp_ndom_out_dt, self.mn_dom_out_dt, self.mn_ndom_out_dt, self.m_dom_out_dt, self.m_ndom_out_dt, self.qp_dom_out_ft, self.qp_ndom_out_ft, self.mn_dom_out_ft, self.mn_ndom_out_ft, self.m_dom_out_ft, self.m_ndom_out_ft, self.mn_dom_DS_60, self.mn_ndom_DS_60, self.m_dom_DS_60, self.m_ndom_DS_60]
        features = ["subject_id", "diagnosis", "typist", "side", "years", "UPDRS_dom", "UPDRS_ndom", "serial", "hand", "tm", "qp_dom", "qp_ndom", "mn_dom", "mn_ndom", "m_dom", "m_ndom", "qp_dom_ft", "qp_ndom_ft", "mn_dom_ft", "mn_ndom_ft", "m_dom_ft", "m_ndom_ft", "qp_dom_err", "qp_ndom_err", "mn_dom_err", "mn_ndom_err", "m_dom_err", "m_ndom_err", "qp_dom_dt", "qp_ndom_dt", "mn_dom_dt", "mn_ndom_dt", "m_dom_dt", "m_ndom_dt", "qp_dom_ft_10", "qp_ndom_ft_10", "mn_dom_ft_10", "mn_ndom_ft_10", "m_dom_ft_10", "m_ndom_ft_10", "qp_dom_dt_10", "qp_ndom_dt_10", "mn_dom_dt_10", "mn_ndom_dt_10", "m_dom_dt_10", "m_ndom_dt_10", "qp_dom_err_10", "qp_ndom_err_10", "mn_dom_err_10", "mn_ndom_err_10", "m_dom_err_10", "m_ndom_err_10", "qp_dom_vs", "qp_ndom_vs", "mn_dom_vs", "mn_ndom_vs", "m_dom_vs", "m_ndom_vs", "qp_dom_ft_30", "qp_ndom_ft_30", "qp_dom_dt_30", "qp_ndom_dt_30", "qp_dom_DS_30", "qp_ndom_DS_30", "qp_dom_VS_30", "qp_ndom_VS_30", "qp_dom_DS_60", "qp_ndom_DS_60", "qp_dom_VS_60", "qp_ndom_VS_60", "qp_dom_DS_10", "qp_ndom_DS_10", "qp_dom_VS_10", "qp_ndom_VS_10", "qp_dom_hesitations_dt", "qp_ndom_hesitations_dt", "mn_dom_hesitations_dt", "mn_ndom_hesitations_dt", "m_dom_hesitations_dt", "m_ndom_hesitations_dt", "qp_dom_hesitations_ft", "qp_ndom_hesitations_ft", "mn_dom_hesitations_ft", "mn_ndom_hesitations_ft", "m_dom_hesitations_ft", "m_ndom_hesitations_ft", "mn_dom_DS_60", "mn_ndom_DS_60", "m_dom_DS_60", "m_ndom_DS_60"]
        data_t = pd.DataFrame(data).transpose()
        data_t.columns = features
        return data_t
        
class Subjects:
    """
        Loads all subjects.
    """
    def __init__(self):
        self.PD_OFF_ids = ["PD01_OFF", "PD03_OFF", "PD04_OFF", "PD05_OFF", "PD08_OFF", "PD09_OFF", "PD13_OFF", "PD16_OFF", "PD17_OFF", "PD22_OFF", "PD25_OFF", "PD29_OFF", "PD31_OFF", "PD33_OFF", "PD34_OFF", "PD36_OFF", "PD37_OFF", "PD38_OFF", "PD39_OFF"] # "PD21_OFF"
        self.PD_ON_ids = ["PD01_ON", "PD03_ON", "PD04_ON", "PD05_ON", "PD08_ON", "PD09_ON", "PD13_ON", "PD16_ON", "PD17_ON", "PD21_ON", "PD22_ON", "PD25_ON", "PD29_ON", "PD31_ON", "PD33_ON", "PD34_ON", "PD36_ON", "PD37_ON", "PD38_ON", "PD39_ON"]
        self.HC_ids = ["HC01", "HC08", "HC09", "HC11", "HC12", "HC13", "HC14", "HC17", "HC19", "HC20", "HC22", "HC23", "HC25", "HC27", "HC28", "HC30", "HC33", "HC35", "HC36", "HC37"]
        self.CA_ids = ["CA01", "CA02", "CA03", "CA11", "CA13", "CA15", "CA16", "CA25", "CA29", "CA37", "CA39", "CA40", "CA41", "CA44", "CA46", "CA52", "CA55", "CA56", "CA59"]
        self.subject_groups = [self.PD_OFF_ids, self.PD_ON_ids, self.HC_ids, self.CA_ids]
        self.features =  ["subject_id", "diagnosis", "typist", "side", "years", "UPDRS_dom", "UPDRS_ndom", "serial", "hand", "tm", "qp_dom", "qp_ndom", "mn_dom", "mn_ndom", "m_dom", "m_ndom", "qp_dom_ft", "qp_ndom_ft", "mn_dom_ft", "mn_ndom_ft", "m_dom_ft", "m_ndom_ft", "qp_dom_err", "qp_ndom_err", "mn_dom_err", "mn_ndom_err", "m_dom_err", "m_ndom_err", "qp_dom_dt", "qp_ndom_dt", "mn_dom_dt", "mn_ndom_dt", "m_dom_dt", "m_ndom_dt", "qp_dom_ft_10", "qp_ndom_ft_10", "mn_dom_ft_10", "mn_ndom_ft_10", "m_dom_ft_10", "m_ndom_ft_10", "qp_dom_dt_10", "qp_ndom_dt_10", "mn_dom_dt_10", "mn_ndom_dt_10", "m_dom_dt_10", "m_ndom_dt_10", "qp_dom_err_10", "qp_ndom_err_10", "mn_dom_err_10", "mn_ndom_err_10", "m_dom_err_10", "m_ndom_err_10", "qp_dom_vs", "qp_ndom_vs", "mn_dom_vs", "mn_ndom_vs", "m_dom_vs", "m_ndom_vs", "qp_dom_ft_30", "qp_ndom_ft_30", "qp_dom_dt_30", "qp_ndom_dt_30", "qp_dom_DS_30", "qp_ndom_DS_30", "qp_dom_VS_30", "qp_ndom_VS_30", "qp_dom_DS_60", "qp_ndom_DS_60", "qp_dom_VS_60", "qp_ndom_VS_60", "qp_dom_DS_10", "qp_ndom_DS_10", "qp_dom_VS_10", "qp_ndom_VS_10", "qp_dom_hesitations_dt", "qp_ndom_hesitations_dt", "mn_dom_hesitations_dt", "mn_ndom_hesitations_dt", "m_dom_hesitations_dt", "m_ndom_hesitations_dt", "qp_dom_hesitations_ft", "qp_ndom_hesitations_ft", "mn_dom_hesitations_ft", "mn_ndom_hesitations_ft", "m_dom_hesitations_ft", "m_ndom_hesitations_ft", "mn_dom_DS_60", "mn_ndom_DS_60", "m_dom_DS_60", "m_ndom_DS_60"]
        self.df = pd.DataFrame(columns = self.features)
        self.subjects = []
        
        for group in self.subject_groups:
            for id in group:
                s = createSubjectFromData("Data/" + id + ".txt")
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