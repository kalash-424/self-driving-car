# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:00:12 2022

@author: kalash bhagwat
"""

import data_prep as dp
import model as md
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



#### GETTING THE USEFUL DATA
imagespath, steering = dp.loadData()


####SPLITTING THE VALUES FOR TRAINING AND TESTING
xTrain, xVal, yTrain, yVal = train_test_split(imagespath, steering, test_size=0.2, random_state=5)
print('Total training images: ', len(xTrain))
print('Total validation images: ', len(xVal))



#### CREATE AND TRAIN THE MODEL 
model = md.createModel()
history = model.fit(md.generateBatch(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=27,
                    validation_data=md.generateBatch(xVal,yVal,100,0),validation_steps=200)


#### SAVE MODEL
model.save('Model.h5') 
print('model saved.')


#### SAVE MODEL HISTORY
np.save('train_history.npy',history.history)


#### PLOTTING RESULTS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Training", "Validation"])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epochs')
plt.show()
























