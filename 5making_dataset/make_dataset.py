# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:22:02 2021

@author: yuki
"""




#https://qiita.com/skyfish20ch/items/ef8b7e0db4a6903c730b
import matplotlib.pyplot as plt
import sys
from keras.preprocessing.image import array_to_img,img_to_array,load_img
import numpy as np
import os
import glob

X_train=[]
Y_train=[]

X_test=[]
Y_test=[]

files = glob.glob("./128resized_images/*.jpg", recursive=True)
n=0



for image in files:
    print(image)

    temp_img=load_img(image)
    temp_img_array=img_to_array(temp_img)
    print(temp_img_array.shape)
    X_train.append(temp_img_array)
    n=n+1



"""
#動かない
def data_append(image):
    print(image)
    global n
    temp_img=load_img(image)
    temp_img_array=img_to_array(temp_img)
    print(temp_img_array.shape)
    X_train.append(temp_img_array)
    n=n+1
    

from multiprocessing import Pool
if __name__ == '__main__':
    p = Pool()
    p.map(data_append, files)
"""




print("書き込み中")
np.savez("./gan.npz",x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test)