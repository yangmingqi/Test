# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:19:34 2017

@author: laoyang
"""
from keras import regularizers
from keras.models import Sequential
from keras.models import Model #泛型模型  
from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, UpSampling1D
import findinterval
import readData
import matplotlib.pyplot as plt 
import numpy as np

fsp = 125
windowWidthMax = 150
samplingLen = 30

bcg,ecgInterval,ecgLoc = readData.readData(bcg='data/20170927/ch.bcg.cal.csv',ecg='data/20170927/ch.ppg.cal.csv')

heartInterval = findinterval.findInterval(bcg,samplingLen=30,cashLen=5) 
heartInterval = np.array(heartInterval) / 150 - .5
#heartInterval = MedianFilter(heartInterval,k_median=7 ,medianLen = 10)

ecgRate = 60/(ecgInterval/fsp)

#plt.figure()
#plt.plot(range(windowWidthMax,len(bcg)-windowWidthMax,samplingLen),heartInterval)
#plt.plot(ecgLoc[1:],ecgInterval,color='r')



#slice
slice_len = 24
Intervals = []

slice_Interval=[]
for i in range(len(heartInterval)-slice_len):
    Intervals.append(heartInterval[i:i+slice_len])
Intervals = np.array(Intervals)
#Intervals = np.reshape(Intervals,(len(Intervals),slice_len,1))

input_bcg = Input(shape=(slice_len,))  

#encoded = Convolution1D(1, 3 , activation='relu', border_mode='same')(input_bcg)
#encoded = MaxPooling1D(2, border_mode='same')(encoded)
##encoded = Convolution1D(10, 3 , activation='relu', border_mode='same',
##                        activity_regularizer=regularizers.l1(10e-6))(encoded)
##encoded = MaxPooling1D(2, border_mode='same')(encoded)
#
#decoded = Convolution1D(1, 3 , activation='relu', border_mode='same')(encoded)
#decoded = UpSampling1D(2)(decoded)
##decoded = Convolution1D(10, 3 , activation='relu', border_mode='same')(decoded)
##decoded = UpSampling1D(2)(decoded)
#decoded = Convolution1D(1, 3, activation='tanh', border_mode='same')(decoded)
#encoded = Dense(20, activation='relu')(input_bcg)
encoded = Dense(5, activation='relu',
                activity_regularizer=regularizers.l1(8e-6))(input_bcg)
#decoded = Dense(20, activation='relu')(encoded)  
decoded = Dense(slice_len, activation='tanh')(encoded)  

# 构建自编码模型  
autoencoder = Model(inputs=input_bcg, outputs=decoded)  
  
# 构建编码模型  
encoder = Model(inputs=input_bcg, outputs=encoded)  
  
# compile autoencoder  
autoencoder.compile(optimizer='adam', loss='mse')  
  
# training  
autoencoder.fit(Intervals, Intervals, epochs=1000, batch_size=256, shuffle=True)  
  
result = autoencoder.predict(Intervals)
encoded_result = encoder.predict(Intervals) 


# 构造粘贴的训练集输入
result = np.reshape(result,(len(Intervals),slice_len))
train_data = []
for i in range(slice_len,len(result)):
    train_slice = []
    for j in range(slice_len):
        train_slice.append(result[i-j][j])
    train_data.append(train_slice)
train_data = np.array(train_data)

#粘贴
input_slice = Input(shape=(slice_len,)) 
connection = Dense(10, activation='relu')(input_slice) 
connection = Dense(5, activation='relu')(connection) 
#connection = Dense(10, activation='relu')(connection) 
connection = Dense(1, activation='tanh')(connection) 

connection = Model(inputs=input_slice, outputs=connection)  
connection.compile(optimizer='adam', loss='mse') 
connection.fit(train_data, np.array(heartInterval[slice_len:-slice_len]), epochs=500, batch_size=256, shuffle=True)

connection_result = connection.predict(train_data)[:,0]
#connection_result = connection.predict(train_data[:,range(0,250,2)])[:,0]
plt.figure()
plt.plot(range(windowWidthMax,len(bcg)-windowWidthMax,samplingLen),heartInterval)
plt.plot(ecgLoc[1:],ecgInterval/150-0.5,color='r')
plt.plot(range(windowWidthMax+slice_len*samplingLen,len(bcg)-windowWidthMax-slice_len*samplingLen,samplingLen),
         connection_result)
