#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 03:13:55 2021

@author: manuel
"""

import os
import ants
import numpy as np
import rawpy
import skimage.io as io
from matplotlib import pyplot as plt
from PIL import Image


# Directories
source_dir = os.getcwd() # current working directory
project_dir = os.path.dirname(source_dir) # where the dataset folder should be
dataset_dir = os.path.join(project_dir, 'train12') 

input_file = '../dataset/train12/copd1/copd1_eBHCT.img'
imageSize = (512, 512, 121)

Dim_size=np.array(imageSize, dtype=np.int16)

f = open(input_file,'rb') #only opens the file for reading
img_arr=np.fromfile(f,dtype=np.uint16)
img_arr=img_arr.reshape(Dim_size[0],Dim_size[1],Dim_size[2])

plt.imshow(img_arr[:,:,60].astype(np.init8))



