# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:19:42 2017

@author: laoyang
"""
import numpy as np
import findpeaks

windowWidthMax = 150 #窗口大小
windowWidthMin = 60 

# sig = 30 
sig = 30 
u = 20 


def findInterval(bcg,samplingLen = 100 ,cashLen = 10):
    #samplingLen:采点间隔   cashLen:取均值的个数   k_median:k中值滤波   medianLen:采用中值滤波的阈值
    from math import sqrt as sqrt
    from math import e as exp
    from math import pi as pi

    heartInterval = [] #最后得到的心率间隔
    combinedList=[]  
    peakLoc = 0
    intervalList = []

    peaks = []
    location = []
    
    for i in range(windowWidthMax,len(bcg)-windowWidthMax,samplingLen):
        combinedList = findpeaks.findpeaks(interval(bcg[i-windowWidthMax:i],bcg[i:i+windowWidthMax]))

        peaks = combinedList[0].tolist()[:3]
        location = combinedList[1][:3]
#         print(location)

        if len(intervalList) < cashLen:

            peakLoc = location[peaks.index(max(peaks))]
            intervalList.append(peakLoc)
        else:
            u = sum(intervalList)/len(intervalList)

            
            peaks2 = []
            for j in range(len(peaks)):
                peaks2.append((1/(sqrt(2*pi)*sig)*exp**((-(location[j]-u)**2/(2*sig**2))))*peaks[j]*1000)

            peakLoc = location[peaks2.index(max(peaks2))]
            
#             peaks2 = np.abs((location - u)).tolist()
#             peakLoc = location[peaks2.index(min(peaks2))]
            
            
            intervalList = intervalList[1:]
            intervalList.append(peakLoc)

        heartInterval.append(peakLoc+windowWidthMin)
        #heartInterval.append(60/((peakLoc+windowWidthMin)/fs))

    return heartInterval

# k中值滤波
def MedianFilter(heartInterval,k_median=3 ,medianLen = 10):
    k = int(k_median/2)
    for i in range(k,len(heartInterval)-k):
        median = np.median(heartInterval[(i-k):(i+k+1)])
        if  abs(heartInterval[i]-median)>medianLen:
            heartInterval[i] = median

    return heartInterval;

def interval(dataL,dataR): 
    combined = []

    corr_pdf = fusion(dataL,dataR,Corr)
    amdf_pdf = fusion(dataL,dataR,AMDF)
    map_pdf = fusion(dataL,dataR,MAP)

    for i in range(windowWidthMax-windowWidthMin):
        combined.append(corr_pdf[i]*amdf_pdf[i]*map_pdf[i])

    return combined

#融合
def fusion(dataL,dataR,estimatorFunction):
    estimator = []
    sum = 0
    pdf = []
    
    for i in range(windowWidthMin,windowWidthMax):  #0.4s-1.5s
        estimator.append(estimatorFunction(i,dataL,dataR))
        
    Min = min(estimator)
    for i in range(windowWidthMax-windowWidthMin):
        estimator[i] = estimator[i] - Min
    
    for i in range(windowWidthMax-windowWidthMin):
        sum += estimator[i]

    for i in range(windowWidthMax-windowWidthMin):
        pdf.append(estimator[i]/sum)
    
    return pdf

#改进的自相关 Modified autocorrelation
def Corr(m=0,dataL=[],dataR=[]):  #50<m<200 ，m窗口
    sum = 0
    sum = np.dot(dataL[-m:],np.transpose(dataR[:m]))

    return sum/m

#Modified average mafnitude difference function
def AMDF(m=0,dataL=[],dataR=[]):  
    sum = 0
    sum = np.sum(np.abs(dataL[-m:]-dataR[:m]))
    
    return 1/(sum/m)

#maximum amplitude pairs
def MAP(m=0,dataL=[],dataR=[]):  
    max = np.max(dataL[-m:]+dataR[:m])

    return max