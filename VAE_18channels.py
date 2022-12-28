# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:07:14 2022

@author: sdd380
"""


seed_value= 1
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# 1: Loading required libararies.
# import tensorflow as tf

import csv
import numpy
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from os.path import dirname, join as pjoin
import scipy.io as sio
# import os
import scipy.io as spio
from matplotlib import cm as CM
#from matplotlib import mlab as ML
# import numpy as np
from keras.callbacks import EarlyStopping


import visualkeras
import tensorflow as tf
# model = ...
windowLength = 200
numberOfColumns = 18
epochLength = 200
latentFeatures = 3
Createinitialize = False
useInitializeWeights = True

trainModel = False
pathToInitWeights = "https://github.com/SinaDavid/VAE/Init_weights_18Channel_Small/"
pathToTrainedWeights = "https://github.com/SinaDavid/VAE/trained_weights_18Channel_Small/"

   
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D
from collections import defaultdict
import visualkeras
from PIL import ImageFont

from matplotlib.lines import Line2D
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,LeaveOneGroupOut,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import pickle
import pandas as pd
import seaborn as sns
plt.close('all')

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt((((predictions - targets)/targets) ** 2).mean())

def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*28*28
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        # print('print recon loss: ' + str(reconstruction_loss_batch) + 'print kl loss:' + str(kl_loss_batch))
        # pickle.dump(test, locals())
        # f = open('testing123.pckl', 'wb')
        # pickle.dump(reconstruction_loss_batch, f)
        # f.close()
        return reconstruction_loss_batch  + kl_loss_batch
        
    return total_loss

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random




#### model #####

## Encoder part ##
tf.compat.v1.disable_eager_execution()
input_data = tf.keras.layers.Input(shape=(epochLength, numberOfColumns))
encoder = tf.keras.layers.Conv1D(20, 50,activation='relu')(input_data)
encoder = tf.keras.layers.Conv1D(10, 40,activation='relu')(encoder)
encoder = tf.keras.layers.Conv1D(2, 30,activation='relu')(encoder)
encoder = tf.keras.layers.Flatten()(encoder)
encoder = tf.keras.layers.Dense(80)(encoder)
distribution_mean = tf.keras.layers.Dense(latentFeatures, name='mean')(encoder)
distribution_variance = tf.keras.layers.Dense(latentFeatures, name='log_variance')(encoder)
latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])
encoder_model = tf.keras.Model(input_data, latent_encoding)
encoder_model.summary()


## Decoder part ##
decoder_input = tf.keras.layers.Input(shape=(latentFeatures)) 
decoder = tf.keras.layers.Dense(80)(decoder_input)
decoder = tf.keras.layers.Reshape((1, 80))(decoder)
decoder = tf.keras.layers.Conv1DTranspose(2, 30, activation='relu')(decoder)
decoder = tf.keras.layers.Conv1DTranspose(10, 40, activation='relu')(decoder)
decoder = tf.keras.layers.Conv1DTranspose(20, 50, activation='relu')(decoder)
decoder_output = tf.keras.layers.Conv1DTranspose(18, 83)(decoder)
decoder_output = tf.keras.layers.LeakyReLU(alpha=0.1)(decoder_output)
decoder_model = tf.keras.Model(decoder_input, decoder_output)
decoder_model.summary()
### end of Model ###



#### getting the weights for initializing stuff
if Createinitialize:
    decoderWeightsInit = []
    encoderWeightsInit = []
    for indx in range(0,len(decoder_model.layers)):
        decoderWeightsInit.append(decoder_model.layers[indx].get_weights())
    
    for indx in range(0,len(encoder_model.layers)):
        encoderWeightsInit.append(encoder_model.layers[indx].get_weights())
        
    f = open(pathToInitWeights + 'decoderInitWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'wb')
    pickle.dump(decoderWeightsInit, f)
    f.close()
    
    f = open(pathToInitWeights + 'encoderInitWeightsBottleneck'+ str(latentFeatures) +  '_.pckl', 'wb')
    pickle.dump(encoderWeightsInit, f)
    f.close()

### if training the model use the iniialized untrained weights and biases.

if useInitializeWeights and trainModel:
    f = open(pathToInitWeights + 'decoderInitWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'rb')
    decoderWeightstesting = pickle.load(f)
    f.close() 
    f = open(pathToInitWeights + 'encoderInitWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'rb')
    encoderWeightstesting = pickle.load(f)
    f.close()
    for indx in range(0,len(decoder_model.layers)):
        a = decoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            decoder_model.layers[indx].set_weights([decoderWeightstesting[indx][0],decoderWeightstesting[indx][1]])
            print('weights initialized for layer: ' + str(indx))

    for indx in range(0,len(encoder_model.layers)):
        a = encoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            encoder_model.layers[indx].set_weights([encoderWeightstesting[indx][0],encoderWeightstesting[indx][1]])  
            print('weights initialized for layer: ' + str(indx))


    


### Loading data ####
path = "https://github.com/SinaDavid/VAE/3dPreparedData/"
# path = "C:\\Users\\sdd380\\surfdrive\\Projects\\VAE_Stroke\\BACKUP\\SSI_Stroke\\3dPreparedData\\"
groupFile = 'stored_3D_groupsplit_withoutS01722_latentfeatures_4_frequency_50.npy'
yFile = 'stored_y_3D_adapted_withoutS01722_latentfeatures_4_frequency_50.npy'
data = 'stored_3D_other_data_withoutS01722_latentfeatures_4_frequency_50.npy'

other_data = np.load(path + data)
y_adapted =  np.load(path + yFile)
groupsplit = np.load(path + groupFile)


###### here Sina excludes the outliers


exludefiles = np.loadtxt("https://github.com/SinaDavid/VAE//exclude_files.txt")
int_array = np.int_(exludefiles)

other_data = np.delete(other_data,int_array,axis=0)
y_adapted = np.delete(y_adapted,int_array,axis=0)
groupsplit = np.delete(groupsplit,int_array,axis=0)
# trackgroup2 = np.delete(trackGroup1,int_array,axis=0)
# Trialnum = np.delete(TrialE,int_array,axis=0)
###### continue code ####

gss = GroupShuffleSplit(n_splits=2, train_size=.65, random_state=1)
gss.get_n_splits()

for train_idx, test_idx in gss.split(other_data, y_adapted, groupsplit):
     print("TRAIN:", train_idx, "TEST:", test_idx)
    

train_data = other_data[train_idx,:,:]
test_data = other_data[test_idx,:,:]

train_y = y_adapted[train_idx]
test_y = y_adapted[test_idx]


encoded = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = tf.keras.models.Model(input_data, decoded)
autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
print("\nautoenoder summary")
autoencoder.summary()
if trainModel:
    history = autoencoder.fit(train_data,
                              train_data,
                              epochs=40,
                              batch_size=64,
                              callbacks=[EarlyStopping(monitor='loss', patience=25)],
                              validation_data=(test_data, test_data))
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
  ### Now storing the trained weights ###  
    decoderWeightstrained = []
    encoderWeightstrained = []
    for indx in range(0,len(decoder_model.layers)):
        decoderWeightstrained.append(decoder_model.layers[indx].get_weights())
    
    for indx in range(0,len(encoder_model.layers)):
        encoderWeightstrained.append(encoder_model.layers[indx].get_weights())
        
    f = open(pathToTrainedWeights + 'decoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'wb')
    pickle.dump(decoderWeightstrained, f)
    f.close()   
    f = open(pathToTrainedWeights + 'encoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'wb')
    pickle.dump(encoderWeightstrained, f)
    f.close()

else:
    decoderWeightstrained = []
    encoderWeightstrained = []
    f = open(pathToTrainedWeights + 'decoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'rb')
    decoderWeightstrained = pickle.load(f)
    f.close()   
    f = open(pathToTrainedWeights + 'encoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_.pckl', 'rb')
    encoderWeightstrained = pickle.load(f)
    f.close()
    for indx in range(0,len(decoder_model.layers)):
        a = decoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            decoder_model.layers[indx].set_weights([decoderWeightstrained[indx][0],decoderWeightstrained[indx][1]])
    
    for indx in range(0,len(encoder_model.layers)):
        a = encoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            encoder_model.layers[indx].set_weights([encoderWeightstrained[indx][0],encoderWeightstrained[indx][1]])  

    






case =170
fig3, axes = plt.subplots(6, 2, figsize=(15, 5))#, sharey=True
axes[0,0].plot(test_data[case,0:200,0:3])
axes[1,0].plot(test_data[case,0:200,3:6])
axes[2,0].plot(test_data[case,0:200,6:9])
axes[3,0].plot(test_data[case,0:200,9:12])
axes[4,0].plot(test_data[case,0:200,12:15])
axes[5,0].plot(test_data[case,0:200,15:18])
axes[0, 0].set_title('original timeseries')

visualDecodedData = np.expand_dims(test_data[case], axis=0)
visualDecodedData = autoencoder.predict(visualDecodedData)
axes[0,1].plot(visualDecodedData[0,0:200,0:3])
axes[1,1].plot(visualDecodedData[0,0:200,3:6])
axes[2,1].plot(visualDecodedData[0,0:200,6:9])
axes[3,1].plot(visualDecodedData[0,0:200,9:12])
axes[4,1].plot(visualDecodedData[0,0:200,12:15])
axes[5,1].plot(visualDecodedData[0,0:200,15:18])
axes[0, 1].set_title('Reconstructed timeseries')


sprong = 100
originalData = []
Reconstructed = []
fig2, axes = plt.subplots(6, 2, figsize=(15, 5))#, sharey=True
for indx in range(0,6): 
    originalData.append(test_data[indx*sprong])
    axes[indx,0].plot(test_data[indx*sprong])
    axes[0, 0].set_title('original timeseries')
    test = np.expand_dims(test_data[indx*sprong], axis=0)
    testPredicted = autoencoder.predict(test)
    testPredicted = testPredicted[0,:,:]
    Reconstructed.append(testPredicted)
    axes[indx,1].plot(testPredicted)
    axes[0, 1].set_title('Reconstructed timeseries')
    
SD_org = np.transpose(np.zeros(6))
SD_dec = np.zeros(6) 
signalRange = np.zeros(6)

for i in range(0,len(test_data)):
    test_new = test_data[i]
    test_new_dec = np.expand_dims(test_new, axis=0)
    test_dec = autoencoder.predict(test_new_dec)[0]
    test_new = np.expand_dims(test_new, axis=0)[0] 
    # calculate orginal SD_org
    sd_temp = []#0,0,0,0,0,0
    sd_temp_Dec = []
    range_temp = []
    for ii in range(0,6):
        sd_temp = np.hstack((sd_temp, np.std(test_new[0:200,ii])))
        sd_temp_Dec = np.hstack((sd_temp_Dec, np.std(test_dec[0:200,ii])))
        range_temp = np.hstack((range_temp,np.ptp(test_dec[0:200,ii])))
    SD_org = np.vstack((SD_org,sd_temp))
    SD_dec = np.vstack((SD_dec,sd_temp_Dec))
    signalRange = np.vstack((signalRange,range_temp))
    
encoded = []
for i in range(0,len(test_data)):
    # z.append(testy[i])
    test_new = test_data[i]
    test_new = np.expand_dims(test_new, axis=0)
    op = encoder_model.predict(test_new)
    encoded.append(op[0])


if latentFeatures == 3:
    xx = []
    yy = []
    zz = []
    z = []
    groupcolor = []
    for i in range(0,len(np.array(encoded))):
        xx.append(np.array(encoded)[i][0])
        yy.append(np.array(encoded)[i][1])
        zz.append(np.array(encoded)[i][2]) 
        z.append(test_y[i]) # Fall risk / group

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    df1 = pd.DataFrame()
    df1['xx'] = xx
    df1['yy'] = yy
    df1['zz'] = zz
    df1['z'] = ["fall risk-"+str(k) for k in z]
    df1['groupcolor'] = ["subject-"+str(k) for k in test_data]
    df1.loc[df1['z'] == 'fall risk-[1]', 'z'] = 'green'
    df1.loc[df1['z'] == 'fall risk-[0]', 'z'] = 'blue'
    df1.loc[df1['z'] == 'fall risk-[2]', 'z'] = 'green'
    df1.loc[df1['z'] == 'fall risk-[3]', 'z'] = 'green' 

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    # plt = fig.add_subplot(projection='3d')
    ax1.scatter(df1['xx'], df1['yy'], df1['zz'], c=df1['z'],label=df1['z'],edgecolors='white')
    # plt.title('3D Variational autoencoder / decoder')
    ax1.set_xlabel('Latent feature 1')
    ax1.set_ylabel('Latent feature 2')
    ax1.set_zlabel('Latent feature 3')



#### Additional validation ######


#############################################################################
### training & test data  correlation / rmse and nrmse plus visualisation ###
#############################################################################

error_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
correlationtemp = []
correlation_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Norerror_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Norerrortemp = []
errortemp = []
# predictedtrainAll = [0,0,0,0,0,0]

# float_list=[]
# output_path = "C:/Users/sdd380/surfdrive3/Projects/VAE_Stroke/PythonOutputTrain/"
# fileformat=".csv"    
# subject="Trial"

for indx in range(0,len(train_data)):
    tempDimensiontrain = np.expand_dims(train_data[indx,:,:], axis=0) 
    predictedtrain = autoencoder.predict(tempDimensiontrain)
    predictedtrain = predictedtrain.reshape(200,numberOfColumns)
    
    # filename= subject + str(groupsplit[train_idx[indx]]) + '_' + str(indx)
    # print(filename)
    # float_list=predictedtrain
    # a = numpy.asarray(float_list)
    # numpy.savetxt(output_path + filename + fileformat , a, delimiter=",")  
    
    # predictedtrainAll = np.vstack((predictedtrainAll,predictedtrain))
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,numberOfColumns):
        errortemp.append(rmse(predictedtrain[:,indx2],train_data[indx,:,indx2]))
        Norerrortemp.append(nrmse(predictedtrain[:,indx2],train_data[indx,:,indx2]))
        correlationtemp.append(np.corrcoef((predictedtrain[:,indx2],train_data[indx,:,indx2]))[0,1])
    error_train = np.vstack((error_train,errortemp))
    Norerror_train = np.vstack((Norerror_train,Norerrortemp))
    correlation_train = np.vstack((correlation_train,correlationtemp))
    errortemp = []
    correlationtemp = []
    Norerrortemp = []

error_train = np.delete(error_train, (0), axis=0)
Norerror_train = np.delete(Norerror_train, (0), axis=0)
correlation_train = np.delete(correlation_train, (0), axis=0)
trainMeanRMSE = np.mean(error_train,axis=0)

boxplottraindata_RMSE = [error_train[:,0],error_train[:,1],error_train[:,2],error_train[:,3],error_train[:,4],error_train[:,5],error_train[:,6],error_train[:,7],error_train[:,8],error_train[:,9],error_train[:,10],error_train[:,11],error_train[:,12],error_train[:,13],error_train[:,14],error_train[:,15],error_train[:,16],error_train[:,17]]
boxplottraindata_nRMSE = [Norerror_train[:,0],Norerror_train[:,1],Norerror_train[:,2],Norerror_train[:,3],Norerror_train[:,4],Norerror_train[:,5],Norerror_train[:,6],Norerror_train[:,7],Norerror_train[:,8],Norerror_train[:,9],Norerror_train[:,10],Norerror_train[:,11],Norerror_train[:,12],Norerror_train[:,13],Norerror_train[:,14],Norerror_train[:,15],Norerror_train[:,16],Norerror_train[:,17]]


################################
### Repeat for test data ###
################################
# float_list=[]
# output_path = "C:/Users/sdd380/surfdrive3/Projects/VAE_Stroke/PythonInputTest/"
# fileformat=".csv"    
# subject="Trial"


error_test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
errortemp = []
correlationtemp = []
correlation_test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Norerror_test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
Norerrortemp = []
for indx in range(0,len(test_data)):
    tempDimensiontest = np.expand_dims(test_data[indx,:,:], axis=0) 
    predictedtest = autoencoder.predict(tempDimensiontest)
    predictedtest = predictedtest.reshape(200,numberOfColumns)
    
    # filename= trackgroup2[test_idx[indx]] + '_' + str(indx)
    # trial = Trialnum[test_idx[indx]]
    # print(filename)
    # float_list=predictedtest
    # a = numpy.asarray(float_list)
    # numpy.savetxt(output_path + filename + subject + str(trial) + fileformat , a, delimiter=",")  
    
    
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,numberOfColumns):
        errortemp.append(rmse(predictedtest[:,indx2],test_data[indx,:,indx2]))
        Norerrortemp.append(nrmse(predictedtest[:,indx2],test_data[indx,:,indx2]))
        correlationtemp.append(np.corrcoef((predictedtest[:,indx2],test_data[indx,:,indx2]))[0,1])
    error_test = np.vstack((error_test,errortemp))
    correlation_test = np.vstack((correlation_test,correlationtemp))
    Norerror_test = np.vstack((Norerror_test,Norerrortemp))
    errortemp = []
    correlationtemp = []
    errortemp = []
    Norerrortemp = []

error_test = np.delete(error_test, (0), axis=0)
testMeanRMSE = np.mean(error_test,axis=0)
Norerror_test = np.delete(Norerror_test, (0), axis=0)


boxplottestdata_RMSE = [error_test[:,0],error_test[:,1],error_test[:,2],error_test[:,3],error_test[:,4],error_test[:,5],error_test[:,6],error_test[:,7],error_test[:,8],error_test[:,9],error_test[:,10],error_test[:,11],error_test[:,12],error_test[:,13],error_test[:,14],error_test[:,15],error_test[:,16],error_test[:,17]]
boxplottestdata_nRMSE = [Norerror_test[:,0],Norerror_test[:,1],Norerror_test[:,2],Norerror_test[:,3],Norerror_test[:,4],Norerror_test[:,5],Norerror_test[:,6],Norerror_test[:,7],Norerror_test[:,8],Norerror_test[:,9],Norerror_test[:,10],Norerror_test[:,11],Norerror_test[:,12],Norerror_test[:,13],Norerror_test[:,14],Norerror_test[:,15],Norerror_test[:,16],Norerror_test[:,17]]


boxplottraindata = [correlation_train[:,0],correlation_train[:,1],correlation_train[:,2],correlation_train[:,3],correlation_train[:,4],correlation_train[:,5],correlation_train[:,6],correlation_train[:,7],correlation_train[:,8],correlation_train[:,9],correlation_train[:,10],correlation_train[:,11],correlation_train[:,12],correlation_train[:,13],correlation_train[:,14],correlation_train[:,15],correlation_train[:,16],correlation_train[:,17]]
boxplottestdata = [correlation_test[:,0],correlation_test[:,1],correlation_test[:,2],correlation_test[:,3],correlation_test[:,4],correlation_test[:,5],correlation_test[:,6],correlation_test[:,7],correlation_test[:,8],correlation_test[:,9],correlation_test[:,10],correlation_test[:,11],correlation_test[:,12],correlation_test[:,13],correlation_test[:,14],correlation_test[:,15],correlation_test[:,16],correlation_test[:,17]]

##### START THE PLOTTING ###############################

# Correlation #

fig = plt.figure(24)
ax1 = fig.add_subplot(211)
ax1.title.set_text('Correlation boxplot: training dataset')

bplot1 = ax1.boxplot(boxplottraindata,patch_artist=True,showfliers=False)

# ax7.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
ax2 = fig.add_subplot(212)
ax2.title.set_text('Correlation boxplot: testing dataset')
bplot2 = ax2.boxplot(boxplottestdata,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
plt.show()




# RMSE + NRMSE #

ylim = 60
ymin =0

fig = plt.figure(18)
ax1 = fig.add_subplot(221)
ax1.title.set_text('RMSE boxplot: training dataset')

bplot1 = ax1.boxplot(boxplottraindata_RMSE,patch_artist=True,showfliers=False)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.ylabel('degrees')
ax2 = fig.add_subplot(222)
ax2.title.set_text('Normalized RMSE boxplot: training dataset')
bplot2 = ax2.boxplot(boxplottraindata_nRMSE,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen']#,'pink', 'lightblue', 'lightgreen'
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('percentage')

ax3 = fig.add_subplot(223)
ax3.title.set_text('RMSE boxplot: test dataset')
bplot3 = ax3.boxplot(boxplottestdata_RMSE,patch_artist=True,showfliers=False)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.ylabel('degrees')

ax4 = fig.add_subplot(224)
ax4.title.set_text('Normalized RMSE boxplot: training dataset')
bplot4 = ax4.boxplot(boxplottestdata_nRMSE,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen']#,'pink', 'lightblue', 'lightgreen'
for bplot in (bplot1, bplot2,bplot3,bplot4):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('percentage')

plt.show()




### plot 1 subplt


fig = plt.figure(20)
ax1 = fig.add_subplot(121)
ax1.title.set_text('RMSE external validation')
# ax1.bar([0,1,2,3,4,5],externMeanRMSE)
# plt.ylim(top=ylim) #ymax is your value
# plt.ylim(bottom=ymin) #ymin is your value
# plt.xlabel('Column number')
# plt.ylabel('degrees')

ax2 = fig.add_subplot(121)
ax2.title.set_text('RMSE training validation')
ax2.bar([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],trainMeanRMSE)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('degrees')

ax3 = fig.add_subplot(122)
ax3.title.set_text('RMSE test validation')
ax3.bar([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],testMeanRMSE)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('degrees')
## put on discord    


# from matplotlib.backends.backend_pdf import PdfPages

# def save_multi_image(filename):
#     pp = PdfPages(filename)
#     fig_nums = plt.get_fignums()
#     figs = [plt.figure(n) for n in fig_nums]
#     for fig in figs:
#         fig.savefig(pp, format='pdf')
#     pp.close()


# filename = "80.pdf"
# save_multi_image(filename)


