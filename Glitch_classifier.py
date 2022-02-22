# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:58:08 2022

@author: Amin Boumerdassi
"""
#Import relevant modules
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from numpy import loadtxt, array, append, shape
from sklearn.preprocessing import LabelEncoder
import glob
from downsampler import downsample
from sklearn.model_selection import train_test_split


#Import glitch data
glitch_directory = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o1_text_files\\'

#Locating a small number of glitches for debugging
glitch_files= glob.glob(glitch_directory+'Tomte'+ '*.txt')+glob.glob(glitch_directory+'Whistle'+ '*.txt')+glob.glob(glitch_directory+'Air_Compressor'+'*.txt')

glitch_data= array([loadtxt(glitch) for glitch in glitch_files])

#May be useful to normalise the data in some way e.g. divide by 1e-21

#Downsample the data
npts=5000
dsampled_glitch_data= array([downsample(i,npts) for i in glitch_data])

#Define training labels
classes= array(['Air_Compressor', 'Blip', 'Extremely_Loud', 'Koi_Fish',
       'Low_Frequency_Burst', 'Low_Frequency_Lines', 'Power_Line',
       'Repeating_Blips', 'Scattered_Light', 'Scratchy', 'Tomte', 'Whistle'])
glitch_labels= array([])

for i in glitch_files:
    glitch_labels= append(glitch_labels,[label for label in classes if label in i])
    
#Encoding labels
num=3#No. of unique classes as *loaded*. May be less than len(classes)
label_encoder= LabelEncoder()
classes_encoded= label_encoder.fit_transform(glitch_labels)
ylabel= to_categorical(classes_encoded, num_classes=num)    
    
    
#Reshape x data to fit into the CNN (labels already in the right dimensions)
dsampled_glitch_data= dsampled_glitch_data.reshape(len(dsampled_glitch_data),npts,-1)

#Creating test-train split
X_train, X_test, y_train, y_test = train_test_split(dsampled_glitch_data, ylabel, test_size=0.33, random_state=38)

#Defining the CNN
model = keras.Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu',input_shape=(npts,1)))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num, activation='softmax'))#no. of nodes depends on no. of *used* classes
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"), metrics=['accuracy'])

#Fitting the CNN
history = model.fit(X_train, y_train, epochs=20, batch_size=7, verbose=1, validation_data=(X_test, y_test))#Dimension issues on this line

#Evaluate CNN performance
_, accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
print("Validation accuracy: {}".format(accuracy))

