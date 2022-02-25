# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:54:39 2022

@author: Amin Boumerdassi

This script will use the specified O1+2 directories to locate all glitch files
and append them to an array. Then these are downsampled and saved as a numpy 
file. Then, the glitch directory list is saved as a pkl file for use in
create_glitch_labels.py
"""
import glob
from numpy import save, array, loadtxt
from downsampler import downsample
import pickle

#Creating and saving glitch data arrays
glitch_directory1 = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o1_text_files\\'

glitch_directory2 = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o2_text_files\\'

glitch_files= glob.glob(glitch_directory1+'*.txt')+glob.glob(glitch_directory2+'*.txt')
glitch_data= array([loadtxt(glitch) for glitch in glitch_files])

#Downsample the data  
npts=4096#no. of points to d_sample to
dsampled_glitch_data= array([downsample(i,npts) for i in glitch_data])

#Save glitch data
save("glitch_data_{:}_pts.npy".format(npts), dsampled_glitch_data)

#Save glitch file directory list
file= open("glitch_dir_list.pkl", "wb")
pickle.dump(glitch_files, file)
file.close()
    

