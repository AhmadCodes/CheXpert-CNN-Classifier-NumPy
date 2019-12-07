#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:05:39 2019

@author: ahmad
"""
import pandas as pd
import numpy as np
from classifier import CNN

#%%

askpath = "insert path to the directory containing the folder: CheXpert-v1.0-small\n"
alt = "If the folder is already present in working directory, insert 'd'\n"


d = input(askpath+alt)
d = d+'/'
if d == 'd/' or d=='D/':
    d = ''      
if type(d) is str:
    print("thank you for your response, training will start shortly\n")
else:
    print("unknown type of input, training may not work successfully")



dirr = d+'CheXpert-v1.0-small/'

#%%
def readlabels(dirr,total_ims_to_load = 1, total_images_on_disk = 223414):
               
    chexpert_train = pd.read_csv(dirr+'train.csv')
    chexpert_train = chexpert_train.fillna(0)
    paths = chexpert_train.values
    paths = paths[:,0]
                           
    sequenced_idxs = np.arange(total_images_on_disk)
    permuted_idxs = np.random.permutation(sequenced_idxs)
    required_idxs = permuted_idxs[:total_ims_to_load]
    
    names = paths[required_idxs] 
    names = names.tolist()
    lbls = chexpert_train.iloc[:,5:].values
    lbls[lbls<0] = 1
    labels = lbls[required_idxs,:].astype('uint8')
    total_imgs = lbls.shape[0]
    return names, labels, total_imgs



filenames, labels, total_imgs= readlabels(dirr,total_ims_to_load = 223414)
labels = labels.T
#%% Make and train CNN model
                
l1hyp  = {'conv_pad':1,'f_size':3,'f_stride':1,'n_f':32, 'mxp_size':2,'mxp_stride':2}
l1opt = {'maxpooling': True, 'batch_normalize': not True}
l1attr = {'layer_num': 1, 'layer_type': "Convolution", 'layer_name' : 'conv1'}
L1 = {'layer_hyperparams':l1hyp,'layer_options' : l1opt, 'layer_attr':l1attr}



l2hyp  = {'conv_pad':1,'f_size':3,'f_stride':1,'n_f':64, 'mxp_size':2,'mxp_stride':2}
l2opt = {'maxpooling': True, 'batch_normalize': not True}
l2attr = {'layer_num': 2, 'layer_type': "Convolution", 'layer_name' : 'conv2'}
L2 = {'layer_hyperparams':l2hyp,'layer_options' : l2opt, 'layer_attr':l2attr}


l3hyp  = {'conv_pad':1,'f_size':3,'f_stride':1,'n_f':128, 'mxp_size':2,'mxp_stride':2}
l3opt = {'maxpooling': True, 'batch_normalize': not True}
l3attr = {'layer_num': 3, 'layer_type': "Convolution", 'layer_name' : 'conv3'}
L3 = {'layer_hyperparams':l3hyp,'layer_options' : l3opt, 'layer_attr':l3attr}

L = [L1,L2,L3] #,L2,L3          ,L3        


cnn = CNN(L,readDirectory=d)

cnn.train(X_file_names = filenames, y = labels, n_epochs = 5, alpha_init=0.1,
          mini_batch_size=5,imageload = True,momentum = 0.99,lamda = 0.001)



#%% save weights
import pickle
cnn_weights = {}
cnn_weights['params'] = cnn.params
cnn_weights['velecity'] = cnn.velocity
cnn_weights['LayersInfo'] = cnn.cnnLayersInfo
with open("weights/params10e5bs.pkl","wb")as f:
    pickle.dump(cnn_weights,f)


