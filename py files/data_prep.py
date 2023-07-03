# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 00:06:31 2022

@author: kalash bhagwat
"""
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from imgaug import augmenters as imga
import cv2


####DATA PROCESSING
def getname(filepath):
    return filepath.split('\\')[-1]


def importDatainfo(path,datafile):
    colomns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, datafile), names = colomns)
    data['Center'] = data['Center'].apply(getname)
    print('Total images imported: ', data.shape[0])
    return data
     
 
def balancedata(data, display=True):
    nBins = 31
    samplesperbin = 5100              ####CUTOFF CAN BE CHANGED ACCORDING TO DATA
    hist, bins = np.histogram(data['Steering'], nBins)
    
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()
    
    removeIndexList = []
    for i in range(nBins):
        bindatalist = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data['Steering'][j] <= bins[i+1]:
                bindatalist.append(j)
            
        bindatalist = shuffle(bindatalist)
        bindatalist = bindatalist[samplesperbin:]
        removeIndexList.extend(bindatalist)
    print('Removed images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining images: ', len(data))
    
    if display:
        hist, x = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()
    
    return data


def load_Data(path, data):
    imagespath = []
    steering = []
    
    for i in range(len(data)):
        indexeddata = data.iloc[i]
        imagespath.append(os.path.join(path, img_data_path, indexeddata[0]))
        steering.append(float(indexeddata[3]))
    
    imagespath = np.asarray(imagespath)
    steering = np.asarray(steering)
    
    return imagespath, steering



####IMAGE PROCESSING 
def augmentImage(imgpath, steering):
    img = mpimg.imread(imgpath)
    
    ##PANNING
    if np.random.rand() < 0.5:
        pan = imga.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    
    ##ZOOMING
    if np.random.rand() < 0.5:
        zoom = imga.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    
    ##BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = imga.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    
    ##FLIPPING
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering
    
    return img, steering


def preProcessImg(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)   
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))    #NVIDIA's PARAMETERS
    img = img/255
    
    return img



#### DATA LOCATION
path = 'myData'
datafile = 'driving_log.csv'
img_data_path = 'IMG'

data = importDatainfo(path, datafile)

data = balancedata(data, display=True)

####transfer data for training
def loadData():
    return load_Data(path, data)






















