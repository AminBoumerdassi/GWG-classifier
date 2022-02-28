# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:58:08 2022

@author: Amin Boumerdassi

This script contains the CNN and trains it on glitch data and
pre-encoded labels. It then saves the trained model along with
the test data for use in create_confusion_matrix.py. 

First run create_glitch_array.py and create_glitch_labels.py.
"""
#Import relevant modules
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from numpy import shape, load, save
from sklearn.model_selection import train_test_split

#Load O1+2 glitch data from .npy
dsampled_glitch_data= load("glitch_data_4096_pts.npy")

#Load pre-encoded y labels from .npy
ylabel= load("glitch_labels_encoded.npy")

#May be useful to normalise the data in some way e.g. divide by 1e-21    
    
#Reshape x data to fit into the CNN (labels already in the right dimensions)
dimensions= shape(dsampled_glitch_data)
dsampled_glitch_data= dsampled_glitch_data.reshape(dimensions[0],dimensions[1],-1)

#Creating test-train split
X_train, X_test, y_train, y_test = train_test_split(dsampled_glitch_data, ylabel, test_size=0.33, random_state=38)

#Defining the CNN
model = keras.Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu',input_shape=(dimensions[1],1)))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(shape(ylabel)[1], activation='softmax'))#no. of nodes depends on no. of *used* classes  
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"), metrics=['accuracy'])

#Fitting the CNN
history = model.fit(X_train, y_train, epochs=7, batch_size=25, verbose=1, validation_data=(X_test, y_test))

#Evaluate CNN performance
_, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print("Validation accuracy: {}".format(accuracy))

#Save test data & model for use in confusion matrix generation
save("glitch_data_{:}_pts_X_TEST.npy".format(dimensions[1]), X_test)
save("glitch_data_{:}_pts_y_TEST.npy".format(dimensions[1]), y_test)
model.save("glitch_classifier_model1")
