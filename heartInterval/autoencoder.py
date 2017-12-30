import numpy as np  
#np.random.seed(1337)  # for reproducibility  

from keras import regularizers
from keras.models import Sequential
from keras.models import Model #泛型模型  
from keras.layers import Input, Dense, Convolution1D, MaxPooling1D, UpSampling1D 
import matplotlib.pyplot as plt  
import pandas as pd
from scipy import signal
import findinterval
import readData
from sklearn.cluster import KMeans  
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D
import random

slice_len = 240

bcg,bcgs = readData.read_bcg(bcg='data/20170927/ch.bcg.cal.csv',slice_len=slice_len)

_,bcgs2 = readData.read_bcg(bcg='data/20170927/hq.bcg.cal.csv',slice_len=slice_len)

_,bcgs3 = readData.read_bcg(bcg='data/20170927/st.bcg.cal.csv',slice_len=slice_len)
_,bcgs4 = readData.read_bcg(bcg='data/20170927/hy.bcg.cal.csv',slice_len=slice_len)
_,bcgs5 = readData.read_bcg(bcg='data/20170927/st2.bcg.cal.csv',slice_len=slice_len)
_,bcgs6 = readData.read_bcg(bcg='data/20170927/xy.bcg.cal.csv',slice_len=slice_len)

_,bcgs7 = readData.read_bcg(bcg='data/20170928/ch.bcg.cal.csv',slice_len=slice_len)
_,bcgs8 = readData.read_bcg(bcg='data/20170928/gq.bcg.cal.csv',slice_len=slice_len)
_,bcgs9 = readData.read_bcg(bcg='data/20170928/hq.bcg.cal.csv',slice_len=slice_len)
_,bcgs10 = readData.read_bcg(bcg='data/20170928/rx2.bcg.cal.csv',slice_len=slice_len)
_,bcgs11 = readData.read_bcg(bcg='data/20170928/st.bcg.cal.csv',slice_len=slice_len)

bcgs_train = np.concatenate((bcgs,bcgs2,bcgs3,bcgs4,bcgs5,bcgs6,bcgs7,bcgs8,bcgs9,bcgs10,bcgs11))
bcgs_train = np.reshape(bcgs_train,(len(bcgs_train),slice_len,1))
  
# this is our input placeholder  
input_bcg = Input(shape=(slice_len,1))  
  
#encoded = Convolution1D(10, 3 , activation='relu', border_mode='same')(input_bcg)
#encoded = MaxPooling1D(2, border_mode='same')(encoded)
encoded = Convolution1D(1, 200 , activation='relu', border_mode='same')(input_bcg)
encoded = MaxPooling1D(2, border_mode='same')(encoded)
encoded = Convolution1D(1, 3 , activation='relu', border_mode='same')(encoded)
encoded = MaxPooling1D(2, border_mode='same')(encoded)

decoded = Convolution1D(1, 3 , activation='relu', border_mode='same')(encoded)
decoded = UpSampling1D(2)(decoded)
decoded = Convolution1D(1, 200 , activation='relu', border_mode='same')(decoded)
decoded = UpSampling1D(2)(decoded)
#decoded = Convolution1D(10, 3 , activation='relu', border_mode='same')(decoded)
#decoded = UpSampling1D(2)(decoded)
decoded = Convolution1D(1, 3, activation='tanh', border_mode='same')(decoded)
# 编码层  
#encoded = Dense(100, activation='relu')(input_bcg) 
#encoded = Dense(50, activation='relu')(encoded)  
#encoded = Dense(100, activation='relu',
#                activity_regularizer=regularizers.l1(10e-7))(input_bcg)  #10e-7
#encoded = Dense(5, activation='tanh')(encoded)  
  
# 解码层  
#decoded = Dense(10, activation='relu')(encoded)  
#decoded = Dense(50, activation='relu')(decoded) 
#decoded = Dense(100, activation='relu')(decoded)  
#decoded = Dense(slice_len, activation='tanh')(encoded)  
  
# 构建自编码模型  
autoencoder = Model(inputs=input_bcg, outputs=decoded)  
  
# 构建编码模型  
encoder = Model(inputs=input_bcg, outputs=encoded)  
  
# compile autoencoder  
autoencoder.compile(optimizer='adam', loss='mse')  
  
# training  
autoencoder.fit(bcgs_train, bcgs_train, epochs=1, batch_size=256, shuffle=True)  
  
#result = autoencoder.predict(bcgs)
encoded_result = encoder.predict(np.reshape(bcgs,(len(bcgs),slice_len,1))) 

#input_bcg2 = Input(shape=(100,))  
#encoded2 = Dense(50, activation='relu')(input_bcg2)
#decoded2 = Dense(100, activation='tanh')(encoded2) 
#autoencoder2 = Model(inputs=input_bcg2, outputs=decoded2)  
#autoencoder2.compile(optimizer='adam', loss='mse')  
#autoencoder2.fit(encoded_result, encoded_result, epochs=2, batch_size=256, shuffle=True) 

#result = encoder.predict(bcgs)
#result = autoencoder2.predict(result)


#plt.figure()
#plt.scatter(range(len(encoded_result)),encoded_result)
#
## K-means聚类，将正常数据与体动分开
#estimator = KMeans(n_clusters=2)#构造聚类器
#estimator.fit(encoded_result)
#label_pred = estimator.labels_ #获取聚类标签
#plt.figure()
#plt.scatter(range(len(encoded_result)),encoded_result,c=label_pred, s=50, cmap='rainbow')
#
## 混合高斯模型聚类，将正常数据与体动分开
#clf = mixture.GMM(n_components=2)
#clf.fit(encoded_result)
#labels_clf = clf.fit_predict(encoded_result)
#plt.figure()
#plt.scatter(range(len(encoded_result)),encoded_result,c=labels_clf, s=50, cmap='rainbow')
#error_loc = np.where(labels_clf == 1)
#plt.figure()
#for i in range(len(error_loc)):
#    plt.plot([error_loc[i],error_loc[i]],[60,100],'red')

## 编码2D散点图
#plt.figure()
#plt.scatter(encoded_result[:,0],encoded_result[:,1])
## 编码3D散点图
#plt.figure()
#ax=plt.subplot(111,projection='3d')
#ax.scatter(range(len(encoded_result[:,0])),encoded_result[:,0],encoded_result[:,1])

## L1范数
#l1 = np.abs(encoded_result)
##l1 = np.min(l1,axis=1) # 对每一行求和
#l1 = l1[:,0] + l1[:,1] 
#plt.figure()
#plt.plot(l1)
#
#L1 = []
#for i in l1:
#    L1.append([i])

# 混合高斯模型    
#clf = mixture.GMM(n_components=2)
#clf.fit(encoded_result)
#labels_clf = clf.fit_predict(encoded_result)
## 2D散点图
#plt.figure()
#plt.scatter(encoded_result[:,0],encoded_result[:,1],c=labels_clf, cmap='rainbow')
## 3D散点图
#plt.figure()
#ax=plt.subplot(111,projection='3d')
#ax.scatter(range(len(encoded_result[:,0])),encoded_result[:,0],encoded_result[:,1],c=labels_clf[:] ,cmap='rainbow')


# 用均值拼接
#n_mean = 1
#mean_result = [0]*(slice_len - n_mean)
#for i in range(0,len(result)-1,n_mean):
#    n = [0]*i
#    mean_result = np.append(mean_result,[0]*n_mean) + np.append(n,result[i])
#    if i%5000==0:
#        print(i)
#mean_result = mean_result[slice_len:len(bcg)-slice_len]/slice_len
#plt.figure()
#plt.plot(mean_result)




#拼接
#bcg_result = []
#for i in range(int(len(bcg)/slice_len)):
#    bcg_result = np.append(bcg_result,result[slice_len*i])
#plt.figure(figsize=(50,10))
#plt.plot(bcg_result)
    


result = autoencoder.predict(np.reshape(bcgs,(len(bcgs),slice_len,1)))
result = np.reshape(result,(len(result),slice_len))
# 将异常点设置为缺失值nan
#result[labels_clf==0] = np.nan

# 构造粘贴的训练集输入
train_data = []
#y_slice = []
for i in range(slice_len,len(result)):
    train_slice = []
    for j in range(slice_len):
        train_slice.append(result[i-j][j])
        
    train_data.append(train_slice)
#    y_slice.append([bcg[i:i+500]])
#train_slice = np.array(train_slice)
#y_slice = np.array(y_slice)  
train_data = np.array(train_data)

#mean_connection = np.mean(train_data,axis=1)
#plt.figure()                       
#plt.plot(train_data)


#radom_train = np.array([])
radom_train = []
#for i in range(len(train_data)):
#    train_slice = []
#    train_slice = list(train_data[i][~np.isnan(train_data[i])])
#    
#    if len(train_slice) >= int(slice_len/2):
#        radom_slice = random.sample(train_slice, int(slice_len/2))
#    else :
#        radom_slice = train_slice * int(int(slice_len/2)/len(train_slice))
#        radom_slice = radom_slice + random.sample(train_slice, int(slice_len/2)%len(train_slice))
#        print(len(radom_slice))
#    
#    radom_train.append(radom_slice)

error = []
for i in range(len(train_data)):
    train_slice = []
    train_slice = list(train_data[i][~np.isnan(train_data[i])])
    if 150 <len(train_slice) < 200:
        error.append(i)
    radom_slice = train_slice * int(slice_len/len(train_slice))
    radom_slice = radom_slice + random.sample(train_slice, slice_len%len(train_slice))
    
    radom_train.append(radom_slice)
    
#radom_len = 150
#for i in range(len(train_data)):
#    radom_slice = random.sample(list(train_data[i]),radom_len)
#    radom_train.append(radom_slice)

radom_train = np.array(radom_train)



input_slice = Input(shape=(slice_len,)) 
connection = Dense(100, activation='relu')(input_slice) 
connection = Dense(50, activation='relu')(connection) 
connection = Dense(10, activation='relu')(connection) 
connection = Dense(1, activation='tanh')(connection) 

connection = Model(inputs=input_slice, outputs=connection)  
connection.compile(optimizer='adam', loss='mse') 
connection.fit(train_data, np.array(bcg[slice_len:-slice_len]), epochs=10, batch_size=256, shuffle=True)

connection_result = connection.predict(train_data)[:,0]
#connection_result = connection.predict(train_data[:,range(0,250,2)])[:,0]
plt.figure()
plt.plot(connection_result)

#model = Sequential()
#model.add(Dense(500,activation='relu',input_shape=(250,250)))
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#model.fit(train_slice, y_slice, epochs=10, batch_size=32)



#result = autoencoder.predict(bcgs)
## 将异常点设置为缺失值nan
#result[labels_clf==0] = 0
#
#n_mean = 10
#hiddenLayer_len = 250
#mean_connection = [0]*(hiddenLayer_len - n_mean)
#for i in range(0,len(encoded_result)-1,n_mean):
#    n = [0]*i
#    mean_connection = np.append(mean_connection,[0]*n_mean) + np.append(n,result[i])
#    if i%5000==0:
#        print(i)
#mean_connection = mean_connection[hiddenLayer_len:len(bcg)-hiddenLayer_len]/hiddenLayer_len
#plt.figure()                       
#plt.plot(mean_connection)
#
#c=findinterval.findInterval(encoded_result[:,0],samplingLen=30,cashLen=5)
#plt.figure()
#plt.plot(bcg)




b=findinterval.findInterval(np.array(bcg[slice_len:len(bcg)-slice_len]),samplingLen=30,cashLen=5)
b=findinterval.MedianFilter(b,k_median=15 ,medianLen = 10)
a=findinterval.findInterval(connection_result,samplingLen=30,cashLen=5)

#plt.figure(figsize=(50,10))
#plt.plot(b)
#plt.plot(a)


fsp = 125
windowWidthMax = 150
samplingLen = 30

bcg2,ecgInterval,ecgLoc = readData.readData(bcg='data/20170927/ch.bcg.cal.csv',ecg='data/20170927/ch.ppg.cal.csv')
heartInterval = findinterval.findInterval(bcg2,samplingLen=30,cashLen=5)
#heartInterval = MedianFilter(heartInterval,k_median=7 ,medianLen = 10)

ecgRate = 60/(ecgInterval/fsp)

plt.figure()
plt.plot(range(windowWidthMax,len(bcg)-windowWidthMax,samplingLen),heartInterval)
plt.plot(ecgLoc[1:],ecgInterval,color='r')
#for i in range(len(error_loc)):
#    plt.plot([error_loc[i],error_loc[i]],[60,100],'black')
plt.plot(range(windowWidthMax+slice_len,len(bcg)-windowWidthMax-slice_len,samplingLen),a)
