# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:46:32 2017

@author: laoyang
"""
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import findpeaks
import numpy as np

def readData(bcg ,ecg ,start = 0,t = 10000):
    #bcgin = pd.read_csv('test2.csv')
    bcgin = pd.read_csv(bcg)

    ecgin = pd.read_csv(ecg) 

    fs = 125
    fsp = 125
    Nyquist = fs/2

    # bcg = [elem for elem in bcgin.iloc[:,0]]

    bcg = [elem for elem in bcgin.iloc[start:start+fs*t,1]]
    bcgRawMean = sum(bcg)/len(bcg)
    bcg = [elem-bcgRawMean for elem in bcg]

    #bcgtime = [elem for elem in bcgin.iloc[start:start+fs*t,0]]

    #高通滤波 1Hz
    b, a = signal.butter(2, 1/Nyquist ,'high')
    bcg = signal.filtfilt(b, a, bcg)

    #低通滤波 12Hz
    b2, a2 = signal.butter(2, 10/Nyquist )
    bcg = signal.filtfilt(b2, a2, bcg)

    #获取ecg
    #ecgtime = [elem for elem in ecgin.iloc[start:start+fs*t,0]]
    ecg = [elem for elem in ecgin.iloc[start:start+fs*t,1]]

#     plt.figure(figsize=(50,10))
#     plt.plot(ecg)

    #获取ecg尖峰和尖峰位置
    ecgPeak,ecgLoc = findpeaks.findpeaks(ecg, mph=200, mpd=fsp*0.5)
    ecgInterval = (np.diff(ecgLoc) )#/ fs) * 1000

    for i  in range(len(ecgLoc)):
        ecgLoc[i] = ecgLoc[i]*fs/fsp
        
    return bcg,ecgInterval,ecgLoc


def read_bcg(bcg, slice_len = 250):
    #read raw bcg data
    bcgin = pd.read_csv(bcg)
    bcg = [elem for elem in bcgin.iloc[:,1]]
    bcg = np.array(bcg)/4000 - 0.5
    
    #filter
    b, a = signal.butter(2, 1/62.5 ,'high')
    bcg = signal.filtfilt(b, a, bcg)
    b2, a2 = signal.butter(2, 10/62.5 )
    bcg = signal.filtfilt(b2, a2, bcg)
    
    #normalization
    bcg = list(bcg.astype(np.float32))
    
    #slice
    bcgs=[]
    for i in range(len(bcg)-slice_len):
        bcgs.append(bcg[i:i+slice_len])
        
    bcgs = np.array(bcgs)
    
    return bcg,bcgs