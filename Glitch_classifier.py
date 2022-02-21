# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:58:08 2022

@author: Amin Boumerdassi
"""
#Import relevant modules
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from numpy import loadtxt, array, append, shape
from sklearn.preprocessing import LabelEncoder
import glob
from downsampler import downsample


#Import glitch data
glitch_directory = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o1_text_files\\'
glitch_files= glob.glob(glitch_directory+"Tomte"+ '*.txt')
glitch_data= array([loadtxt(glitch) for glitch in glitch_files])

#May be useful to normalise the data in some way e.g. divide by 1e-21

#Downsample the data
npts=5000
#dsampled_glitch_data= array([])
dsampled_glitch_data= array([downsample(i,npts) for i in glitch_data])

#for i in glitch_data:
#    dsampled_glitch_data= append(dsampled_glitch_data, downsample(i,npts))

#Define training labels
classes= array(['Air_Compressor', 'Blip', 'Extremely_Loud', 'Koi_Fish',
       'Low_Frequency_Burst', 'Low_Frequency_Lines', 'Power_Line',
       'Repeating_Blips', 'Scattered_Light', 'Scratchy', 'Tomte', 'Whistle'])
glitch_labels= array([])

for i in glitch_files:
    glitch_labels= append(glitch_labels,[label for label in classes if label in i])
    
#Encoding labels
label_encoder= LabelEncoder()
classes_encoded= label_encoder.fit_transform(glitch_labels)
ylabel= to_categorical(classes_encoded)

#Reshape array to fit into the CNN
#dimensions= [1,len(glitch_data[1])]
glitch_data.reshape(10,1,16384,1)
