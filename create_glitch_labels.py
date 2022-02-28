# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 22:20:57 2022

@author: Amin Boumerdassi

This script will read the label from each filename and save a one-hot encoding
of these.
"""
from numpy import array, append, save
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#Loading glitch dir list
file= open("glitch_dir_list.pkl", "rb")
glitch_files= pickle.load(file)
file.close()

#Creating and saving glitch labels
classes= array(['Air_Compressor', 'Blip_', 'Extremely_Loud', 'Koi_Fish','Light_Modulation',
       'Low_Frequency_Burst', 'Low_Frequency_Lines','Paired_Doves', 'Power_Line',
       'Repeating_Blips', 'Scattered_Light', 'Scratchy', 'Tomte', 'Whistle'])

glitch_labels= array([])

for i in glitch_files:
    glitch_labels= append(glitch_labels,[label for label in classes if label in i])

#Encode class list
label_encoder= LabelEncoder()
class_lst_encoded= label_encoder.fit_transform(classes)
     
#Encoding labels
num=14#No. of unique classes as *loaded*. May be less than len(classes)
classes_encoded= label_encoder.fit_transform(glitch_labels)
ylabel= to_categorical(classes_encoded, num_classes=num)  
save("glitch_labels_encoded.npy", ylabel)
save("classes_encoded.npy", class_lst_encoded)
