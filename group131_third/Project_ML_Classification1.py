#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:18:28 2021

@author: Mar
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


#Data
data_xtest = np.load('Xtest_Classification_Part1-2.npy')
data_xtrain = np.load('Xtrain_Classification_Part1-2.npy')
data_ytrain = np.load('Ytrain_Classification_Part1-2.npy')

data_xtrain = np.reshape(data_xtrain, (6470, 50, 50, 1))
data_xtest = np.reshape(data_xtest, (1164, 50, 50, 1))



#Get image and classification
img = plt.imshow(data_xtrain[6], cmap='gray')
print(data_ytrain[0]) #1 is female



#Normalize the pixels to be values between 0 and 1
data_xtrain = data_xtrain / 255
data_xtest = data_xtest / 255



#Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

datagen.fit(data_xtrain)



#One hot encoding class variables
data_ytrain = to_categorical(data_ytrain)



#Stop training when accuracy has stopped improving
callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, 
                         verbose=0, mode='auto', baseline=None, 
                         restore_best_weights=True)



#Model architecture
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid')) #sigmoid is for binary data

#Compile model
model.compile(loss='binary_crossentropy',  #Even with one hot encoding?
              optimizer='adam',            #RMSprop? or adam?
              metrics=['accuracy'])

model.summary()



#Fit model on training data
hist = model.fit(datagen.flow(data_xtrain, data_ytrain, 
          batch_size=64, subset='training'),
                 validation_data=datagen.flow(data_xtrain, data_ytrain, 
        batch_size=32, subset='validation'), 
        epochs=60)

#val_loss: 0.2537 - val_accuracy: 0.9019



#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'lower right')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()



#Testing the results
img = plt.imshow(data_xtest[51], cmap='gray')
predictions = model.predict(np.array(data_xtest))
print(predictions[8])



#Sorting predictions
final_predictions = np.array([]);
for i in range(len(predictions)):
    if predictions[i][0] > predictions[i][1]:
        final_predictions = np.append(final_predictions, 0)
    else:
        final_predictions = np.append(final_predictions, 1)

np.save('data3.npy', final_predictions)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    