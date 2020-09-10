# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:27:15 2020

@author: lenovo
"""
import scipy.io as scio
from matplotlib.image import  imread
import numpy as np
import os
root_path = '../CoNSeP/Test/'

Image_path = os.path.join(root_path,'Images')
Inst_path = os.path.join(root_path,'Labels')

image_list = os.listdir(Image_path)
inst_list = os.listdir(Inst_path)
data = np.empty((0,1000,1000,4))
for i,inst in enumerate(inst_list):
    in_path = os.path.join(Inst_path,inst)
    im_path = os.path.join(Image_path,image_list[i])
    inst_row = scio.loadmat(in_path)
    img_row = imread(im_path)
    img_row = np.dstack((img_row[...,:3],inst_row['type_map']))
    print(img_row.shape)
    data = np.vstack((data,np.expand_dims(img_row,axis = 0)))
np.save('ConSeP_test.npy',data)
# image_list = sorted(image_list)