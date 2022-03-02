# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:17:49 2022

@author: Amin Boumerdassi
"""
from scipy.interpolate import interp1d
from numpy import arange, linspace

def downsample(array, npts):#Downsamples an array to some no. of elements "npts"
    interpolated = interp1d(arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    return interpolated(linspace(0, len(array), npts))
