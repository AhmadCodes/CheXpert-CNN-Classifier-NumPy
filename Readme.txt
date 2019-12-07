*/

This software is open-sourced. You may use if for educational and research
purposes only. 

If you have any questions, contact the author @ amdali97@gmail.com

/*


 ==============================================================================
        For training perform the following steps 
 ==============================================================================

        Simple run routine:
        ___________________
1- Make sure classifier.py is present in working directory of python.
2- Run the file named 'train.py'
3- Upon prompt, enter the path to directory containing the folder named as:
    CheXpert-v1.0-small
4- this folder must contain complete chexpert dataset, i.e. 223414 training 
    images
    
        Modifying the architecture:
        __________________________
Modification of the achitecture needs some basic understanding of python lists
    and dictionaries.

-> A Layer information dictionary can be made as shown in the folliwing example:


 
l1hyp  = {'conv_pad':1,'f_size':3,'f_stride':1,'n_f':32, 'mxp_size':2,'mxp_stride':2}
l1opt = {'maxpooling': True, 'batch_normalize': True}
l1attr = {'layer_num': 1, 'layer_type': "Convolution", 'layer_name' : 'conv1'}

L1 = {'layer_hyperparams':l1hyp,'layer_options' : l1opt, 'layer_attr':l1attr}



Where L1 is a dictionary of 3 more dictionaries containing inforamation about 
layer hyperparameters, layer options and layer attributes.

-> These series of "layer information dictionaries" can be appended to a list in the
    following fashion:
    

L = [L1,L2,L3]


Where each L1,L2,L3 . . . Ln   is a "layer information dictionary".
And n is the number of convolutional layers

->The list L can be passed to the cnn objion during object initialization as show:


cnn = CNN(L,readDirectory=d)


 ==============================================================================
         Test Time
 ==============================================================================
 
1- make sure a folder named 'weights' containing model weights is present in the
    working directory. Model weights are named as params.pkl
2- Run predict.py
3- use the predict() function as shown:
    predict(absolute_path)
4- predict() function returns a (14,) vector while its input is an absolute path 
    to an image. 