# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:10:12 2022

@author: Amin Boumerdassi

This is the script that generates confusion matrices based off of a saved model and
test data from glitch_classifier.py.

First run all previous Python scripts.
"""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from numpy import load, argmax, array, round_
import matplotlib.pyplot as plt

#Load model, test data, and number-encoded classes
model = keras.models.load_model(r"glitch_classifier_model1")
X_test = load(r"glitch_data_4096_pts_X_TEST.npy")#Alter as appropriate
y_test = load(r"glitch_data_4096_pts_y_TEST.npy")#Alter as appropriate
classes_encoded = load("classes_encoded.npy")

#Test model on test data, and undo one-hot encoding of labels
y_pred=model.predict(X_test) 
y_test = argmax(y_test,axis=1)#undoing one-hot encoding of labels
y_pred = argmax(y_pred,axis=1)

classes= array(['Air Compr.', 'Blip', 'Extr. Loud', 'Koi Fish','Light Mod.',
       'LF Burst', 'LF Lines','Paired Doves', 'Power Line',
       'R. Blips', 'Scatt. Light', 'Scratchy', 'Tomte', 'Whistle'])

#Generate confusion matrix - round values for readability
cm = confusion_matrix(y_test, y_pred, labels = classes_encoded, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=round_(cm,decimals=2), display_labels = classes)
disp.plot()
plt.xticks(fontsize=10, rotation=90)
plt.figure(figsize=(60, 60), dpi=80)
plt.show()

#Save confusion matrix by interactive use in Spyder - plt.savefig() doesn't
#quite work for confusion matrices

#plt.savefig("test_confusion_matrix.jpg")