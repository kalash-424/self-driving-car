# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 00:09:18 2022

@author: kalash bhagwat
"""
import data_prep as dp
import random
import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


#### GENERATE BATCH
def generateBatch(imagespath, steeringlist, batchsize, TrainFlag=True):
    while True:
        imgbatch = []
        steeringbatch = []
        
        for i in range(batchsize):
            index = random.randint(0, len(imagespath)-1)
            if TrainFlag:
                img, steering = dp.augmentImage(imagespath[index], steeringlist[index])
            else:
                img = mpimg.imread(imagespath[index])
                steering = steeringlist[index]
            img = dp.preProcessImg(img)
            imgbatch.append(img)
            steeringbatch.append(steering)
        
        yield (np.asarray(imgbatch), np.asarray(steeringbatch))
        
        
        
def createModel():
    
    #### NVIDIA's MODEL
    model = Sequential()  
    
    model.add(Convolution2D(24,(5,5),(2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2), activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2), activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(learning_rate=0.0001), loss='mse')
    
    print(model.summary())
    return model

