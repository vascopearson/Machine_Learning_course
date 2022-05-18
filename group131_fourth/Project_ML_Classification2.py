#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:18:29 2021

@author: Mar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, RandomWidth, RandomRotation, RandomZoom
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tensorflow.keras.optimizers import Adam

#Data
data_xtest = np.load('Xtest_Classification_Part2-2.npy')
data_xtrain = np.load('Xtrain_Classification_Part2-2.npy')
data_ytrain = np.load('Ytrain_Classification_Part2-2.npy')

data_xtrain = np.reshape(data_xtrain, (7366, 50, 50, 1))
data_xtest = np.reshape(data_xtest, (1290, 50, 50, 1))



#Get image and classification
#img = plt.imshow(data_xtrain[7], cmap='gray')
print(data_ytrain[7]) #0 Caucasian, 1 African, 2 Asian or 3 Indian.



#Normalize the pixels to be values between 0 and 1
data_xtrain = data_xtrain / 255
data_xtest = data_xtest / 255



#One hot encoding class variables
data_ytrain = to_categorical(data_ytrain)



# #Get the validation set
X_train, X_val, y_train, y_val = train_test_split(data_xtrain, data_ytrain, 
                                                    test_size=0.1, 
                                                    random_state=42)



# Data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

datagen.fit(X_train)


#Stop training when loss has stopped improving
callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, 
                          verbose=1, mode='auto', baseline=None, 
                          restore_best_weights=True)


opt = Adam(lr=0.0005)

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
model.add(Dense(4, activation='softmax')) #sigmoid is for binary data

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,            #RMSprop? or adam?
              metrics=['accuracy'])

model.summary()



#Fit model on training data
hist = model.fit(datagen.flow(X_train, y_train, 
          batch_size=32), 
                  validation_data=(X_val, y_val), epochs=100, callbacks=callback)
# hist = model.fit(data_xtrain, data_ytrain, batch_size=128, 
#                  validation_split=0.2, epochs=32)

#val_loss: 0.3854 - val_accuracy: 0.8670



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
#img = plt.imshow(data_xtest[700], cmap='gray')
predictions = model.predict(np.array(data_xtest))
print(predictions[700])



#Final predictions
predictions = model.predict(np.array(data_xtest))
final_predictions = np.array([]);
for i in range(len(predictions)):
    pred = max(predictions[i][0], predictions[i][1], 
              predictions[i][2], predictions[i][3])
    if (pred == predictions[i][0]):
        final_predictions = np.append(final_predictions, 0)
    elif (pred == predictions[i][1]):
        final_predictions = np.append(final_predictions, 1)
    elif (pred == predictions[i][2]):
        final_predictions = np.append(final_predictions, 2)
    elif (pred == predictions[i][3]):
        final_predictions = np.append(final_predictions, 3)
np.save('data4.npy', final_predictions)
        

#Function to reverse one hot encoding
y_val_real = np.argmax(y_val, axis=1)
#Balanced accuracy
predictions_val = model.predict(np.array(X_val))

final_predictions_val = np.array([]);
for i in range(len(predictions_val)):
    pred = max(predictions_val[i][0], predictions_val[i][1], 
              predictions_val[i][2], predictions_val[i][3])
    if (pred == predictions_val[i][0]):
        final_predictions_val = np.append(final_predictions_val, 0)
    elif (pred == predictions_val[i][1]):
        final_predictions_val = np.append(final_predictions_val, 1)
    elif (pred == predictions_val[i][2]):
        final_predictions_val = np.append(final_predictions_val, 2)
    elif (pred == predictions_val[i][3]):
        final_predictions_val = np.append(final_predictions_val, 3)

        
balanced_accuracy_score(y_val_real, final_predictions_val)
accuracy_score(y_val_real, final_predictions_val)
