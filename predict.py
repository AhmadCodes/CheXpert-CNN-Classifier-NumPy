#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 04:13:58 2019

@author: ahmad
"""
import numpy as np
from classifier import CNN
import pickle
import skimage
#%%


#%%


def readandresize(fileNames, directory,dims = (1,200,200)):
    '''
    Input:
    ______
        fileNames:
            list of file names that are to be read from the directory
        directory:
            the directory in which file names are present
        dims:
            tuple of image dimension in the order of depth, height, width
        
    Output:
    ______
        returns a set of images of dimensions:
            len(fileNames), depth, height, width

    '''
    
    n_x, d_x, h_x, w_x = len(fileNames), dims[0], dims[1], dims[2]
    img = np.zeros((n_x, d_x, h_x, w_x))
    xdim = dims[1]
    ydim = dims[2]
    for i,fN in enumerate(fileNames):
        path_ = directory + fN
        im = skimage.io.imread(path_)
        im = skimage.transform.resize(im,(xdim,ydim))
        im = im[...,np.newaxis].transpose(2,0,1)
        img[i,:d_x,:,:] = im/255.0

    return img

#img =  readandresize([p],'')


#%% prediction function
    
with open(r"weights/params.pkl", "rb") as f:
    wdict = pickle.load(f)

params_cache= wdict['params']
V_cache= wdict['velecity']
cnnLayersInfo = wdict['LayersInfo']

#d='/home/ahmad/Study/Deep Learning/Project 2 V2/'
global cnn
cnn = CNN(cnnLayersInfo, W_init_ = False, params_cache=params_cache,
                 V_cache=V_cache, input_dims = (1,1,200,200), readDirectory = '')



def predict(abs_path):
    
    
    global cnn
    
    img =  readandresize([abs_path],'')
    pr = cnn.prediction(img)

    return pr[:,0]