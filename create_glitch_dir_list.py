# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 00:35:49 2022

@author: Amin Boumerdassi

This script specifies the glitch directories and then locates all the glitch
files in each directory. This list of file directories is then stored as a 
pickle file for use in "create_glitch_array.py"
"""
import glob
import pickle

#Define glitch directories
glitch_directory1 = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o1_text_files\\'

glitch_directory2 = 'C:\\Users\\aminb\\Desktop\\4th year project\\Glitch files\\o2_text_files\\'

#Locate all relevant glitch files
glitch_files= glob.glob(glitch_directory1+'*.txt')+glob.glob(glitch_directory2+'*.txt')

#Save glitch file directory list
file= open("glitch_dir_list.pkl", "wb")
pickle.dump(glitch_files, file)
file.close()