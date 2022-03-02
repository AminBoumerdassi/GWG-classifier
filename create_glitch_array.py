# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:54:39 2022

@author: Amin Boumerdassi

This script will load the glitch directory list to locate all glitch files
and append them to an array. Then these are downsampled and saved as a numpy 
file.
"""
from numpy import save, array, loadtxt
from downsampler import downsample
import pickle

#Load glitch dir list
file= open("glitch_dir_list.pkl", "rb")
glitch_files= pickle.load(file)
file.close()

glitch_data= array([loadtxt(glitch) for glitch in glitch_files])

#Downsample the data  
npts=1024#no. of points to d_sample to
dsampled_glitch_data= array([downsample(i,npts) for i in glitch_data])

#Save downsampled glitch data
save("glitch_data_{:}_pts.npy".format(npts), dsampled_glitch_data)
