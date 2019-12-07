#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:49:51 2019

@author: ahmad
"""
import numpy as np
from builtins import range
from numba import jit
import numexpr as ne
import skimage


#%%
# =============================================================================
# Helper functions of vectorized implementation of Convolution
# =============================================================================
@jit
def get_im2col_indxs(Xin_shape, hf, wf, padding=1, s=1):
    
    
    # Determine input and output sizes
    m, d, h, w = Xin_shape
    h_o = (h + 2 * padding - hf) // s + 1
    w_o = (w+ 2 * padding - wf) // s + 1

    #check dimension conformity
    assert (h + 2 * padding - hf) % s == 0 , 'height does not conform' 
    assert (w + 2 * padding - wf) % s == 0 , 'width does not conform'
    
    #determine preliminary indices for indexing the column
    # So that matrix multiplication is gives vectorized
    #implementation of convolution
    
#    i0 = np.repeat(np.arange(field_height), field_width)
#    i0 = np.tile(i0, C)
#    i1 = stride * np.repeat(np.arange(out_height), out_width)
#    j0 = np.tile(np.arange(field_width), field_height * C)
#    j1 = stride * np.tile(np.arange(out_width), out_height)
#    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#
#    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
#    
#    
    r = np.repeat(np.arange(hf), wf)
    r = np.tile(r, d)
    
    r1 = s * np.repeat(np.arange(h_o), w_o)
    c = np.tile(np.arange(wf), hf * d)
    
    c1 = s * np.tile(np.arange(w_o), h_o)
    
    rre = r.reshape(-1, 1)
    r1re = r1.reshape(1, -1)
    c1re = c1.reshape(1, -1)
    cre = c.reshape(-1, 1)
    rowidxs = rre + r1re
    colidxs = cre + c1re
    depidxs = np.repeat(np.arange(d), hf * wf)
    depidxs = depidxs.reshape(-1, 1)

    return depidxs, rowidxs, colidxs

@jit
def im2col_indxs(Xin, hf, wf, padding=1, stride=1):

    # Zero-pad the input
    p = padding
    s = stride
    padtuple = ((0, 0), (0, 0), (p, p), (p, p))
    xpad = np.pad(Xin, padtuple, mode='constant')

    depidxs, rowidxs, colidxs = get_im2col_indxs(Xin.shape, hf, wf, p, s)
    
    # transform the imge according to the columns recieved from the above
    # function. this transformation allows the matrix multplication to 
    # be a vectorized implementation of convolution
    transformed = xpad[:, depidxs, rowidxs, colidxs]
    d = Xin.shape[1]
    transformed = transformed.transpose(1, 2, 0)
    reshp = hf * wf * d
    transformed = transformed.reshape(reshp, -1)
    return transformed

@jit
def col2im_indxs(cols, x_shape, field_height=3, field_width=3, padding=1,
                stride=1):

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indxs(x_shape, field_height, field_width, padding,
                                stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


#%%============================================================================
# CNN Class
# =============================================================================

class CNN:
    '''
    Properties:
        Solver is 'Nestrov's Momentum'
        Supports multple layers
        Fully connected network with 1 hidden layer at the end.
        supports mini batch gradient descent
        supports L1 regularization.
        
        
    '''
    def __init__(self, cnnLayersInfo, W_init_ = True, params_cache=None,
                 V_cache=None, input_dims = (1,1,200,200), n_FC_hidden = 200,
                 readDirectory = ''):
        '''
        Initialize Layers and and other attributes
        
        Input:
        _____
        cnnLayersInfo:
            list with each element having layer information and hyperparameters
            each element is a dictionary. Each item contains three elements,
            namely: layer_attr, layer options layer_hyperparams. these three
            elements are dictionaries of the following format:
                {'layer_hyperparams':{'conv_pad':1,'f_size':3,'f_stride':1,
                                        'n_f':64 'mxp_size':2,'mxp_stride':2},
                'layer_options' : {'maxpooling': True, 'batch_normalize': True},    
                'layer_attr' : {'layer_num': 1, 'layer_type': "Convolution", 
                              'layer_name' : 'conv'}
                }
        W_init:
            bool variable that defines whether or not Wights have to be initia-
            lized. weights will be initialized according to 'f_size', 'n_f',
            'f_stride' keys' values in cnnLayers and input_dims.
        params_cache:
            dictionary of parameters that can be used for transfer learning
        V_cache:
            dictionary of velocities (used for momentum update) that conforms
            with params_cache
        input_dims:
            this tuple corresponds to the input dimensions in the following seq-
            uence: num_examples, depths, height, width.
        '''
        # setting class attributes
        self.input_dims = input_dims
        self.cnnLayersInfo = cnnLayersInfo
        self.n_layers = len(cnnLayersInfo)
        self.readDir = readDirectory
        self.W_init_ = W_init_
        self.loss_history = []
        self.acc_history = []
        self.n_FC_hidden = n_FC_hidden
        
        #Initializ weights if init_ flag is true
        if W_init_:
            self.params,self.velocity = self.param_init()
        else:
            self.params,self.velocity  = params_cache, V_cache
            
        if (W_init_ is not True) and  (params_cache is None):
            issue='W_init was set to False while param_cache was not provided.'
            expectation = ' Expected param_cache to be a dictionary.'
            solution = ' Input weights dict as W_cache or set W_init to True'
            raise Exception(issue+expectation+solution)
            
        
    def param_init(self):
        '''
        Returns a dictionary containing Xavier initialized weights for CNN
        
        '''
        L = self.cnnLayersInfo
        input_dims = self.input_dims
        params_cache = {}
        velocity_cache = {}
        n_x,d_x,h_x,w_x = input_dims
        n_f = L[0]['layer_hyperparams']['n_f']
        d_f = d_x
        h_f = L[0]['layer_hyperparams']['f_size']
        w_f = L[0]['layer_hyperparams']['f_size']
        maxpooling = L[0]['layer_options']['maxpooling']
        batch_normalize = L[0]['layer_options']['batch_normalize']
        fan_in = d_f* h_f* w_f
        W = np.random.randn(n_f, d_f, h_f, w_f)/np.sqrt(fan_in/2)
        params_cache['WL0'] = W
        params_cache['bL0'] = np.zeros((n_f,1))/np.sqrt(fan_in/2)
        
        vW,vb = np.zeros((n_f, d_f, h_f, w_f)), np.zeros((n_f,1))
        velocity_cache['WL0'] = vW
        velocity_cache['bL0'] = vb

        mxp_s = L[0]['layer_hyperparams']['mxp_stride']
        mxp_f = L[0]['layer_hyperparams']['mxp_size'] 
        maxpoolInfo = {'f':mxp_f, 's':mxp_s}
        p = L[0]['layer_hyperparams']['conv_pad']
        s_f = L[0]['layer_hyperparams']['f_stride']
        convInfo = {'p':p, 'f':h_f, 's':s_f, 'n_f':n_f}
        if batch_normalize:    
            gamma = np.ones((d_x,h_x,w_x))#np.random.randn(d_x,h_x,w_x)/np.sqrt(2/(d_x*h_x*w_x))
            Vgamma = np.zeros((d_x,h_x,w_x))
            params_cache['gammaL0'] = gamma
            velocity_cache['gammaL0'] = Vgamma
            beta = np.zeros((d_x,h_x,w_x))#np.random.randn(d_x,h_x,w_x)/np.sqrt(2/(d_x*h_x*w_x))
            Vbeta = np.zeros((d_x,h_x,w_x))
            params_cache['betaL0'] = beta
            velocity_cache['betaL0'] = Vbeta
        
        for i,l in enumerate(L[1:]):
            num_str = str(i+1)
            
            maxpooling = l['layer_options']['maxpooling']
            batch_normalize = l['layer_options']['batch_normalize']
            
            d_f = n_f
            n_f = l['layer_hyperparams']['n_f']
            h_f = l['layer_hyperparams']['f_size']
            w_f = l['layer_hyperparams']['f_size']
            
            fan_in = d_f* h_f* w_f
            W = np.random.randn(n_f, d_f, h_f, w_f)/np.sqrt(fan_in/2)
            params_cache['WL'+num_str] = W
            params_cache['bL'+num_str] = np.zeros((n_f,1))/np.sqrt(fan_in/2)
            
            vW,vb = np.zeros((n_f, d_f, h_f, w_f)), np.zeros((n_f,1))
            velocity_cache['WL'+num_str] = vW
            velocity_cache['bL'+num_str] = vb
            
            mxp_s = l['layer_hyperparams']['mxp_stride']
            mxp_f = l['layer_hyperparams']['mxp_size'] 
            maxpoolInfo = {'f':mxp_f, 's':mxp_s}
            p = l['layer_hyperparams']['conv_pad']
            s_f = l['layer_hyperparams']['f_stride']
            convInfo = {'p':p, 'f':h_f, 's':s_f, 'n_f':n_f}


            conv_dim, input_dims = self.determine_layer_output_size(input_dims, 
                                                    maxpoolInfo, convInfo)
            n_x,d_x,h_x,w_x = conv_dim
            n_b,d_b,h_b,w_b = input_dims
            if batch_normalize:
                gamma = np.zeros((d_b,h_b,w_b))#np.random.randn(d_b,h_b,w_b)/np.sqrt(2/(d_b*h_b*w_b))
                Vgamma = np.zeros((d_b,h_b,w_b))
                params_cache['gammaL'+num_str] = gamma
                velocity_cache['gammaL'+num_str] = Vgamma
                beta = np.ones((d_b,h_b,w_b))#np.random.randn(d_b,h_b,w_b)/np.sqrt(2/(d_b*h_b*w_b))
                Vbeta = np.zeros((d_b,h_b,w_b))
                params_cache['betaL'+num_str] = beta
                velocity_cache['betaL'+num_str] = Vbeta
            if not maxpooling:
                input_dims = conv_dim
            
          
        return params_cache , velocity_cache
    
    
    def determine_layer_output_size(self,prev_dims, maxpoolInfo = {'f':2, 's':2},
                       convInfo = {'p':1, 'f':3, 's':1, 'n_f':1}):
        '''
        Returns a tuple of the feature map sizes by using the formula:
        (N-F+2p)/Stride + 1

        Input:
        _____
            prev_dims:
                dimensions of the maps that are being inputted to the layer.
                dimensions are in the following order:
                    n_examples, depth, height, width
            maxpoolingInfo:
                Dictionary containing information about the maxpooling
                operation. Dictionary should be of the following format:
                    {'f':2, 's':2}
                Where 'f' corresponds to the height and width of the filter.
                's' corresponds to the stride of the filter.
            convInfo:
                Dictionary containing information about the convolution
                operation. Dictionary should be of the following format:
                    {'p':1, 'f':2, 's':2, 'n_f':1})
                Where 'f' corresponds to the height and width of the filter,
                's' corresponds to the stride of the filter and p correspon-
                ds to the zero padding.
                
        Output:
        ______
            tuple of two arrays. first contains information about dimensions 
            after convution and the 2nd contains information about the dim-
            ensions after maxpooling operation in the following sequence:
            n_examples, depth, height, width
            
            
        '''
        nf,conv_s,f_size = convInfo['n_f'], convInfo['s'] ,convInfo['f']
        p  = convInfo['p'] 
        n_x,d_x,h_x,w_x = prev_dims

        d_c = nf
        h_c = (h_x-f_size+2*p)/conv_s + 1
        w_c = (w_x-f_size+2*p)/conv_s + 1
        
        conv_dim = np.uint32((n_x,d_c,h_c,w_c))

        mxp_stride,mxp_size = maxpoolInfo['s'] ,maxpoolInfo['f'] 
        n_x,d_x,h_x,w_x = conv_dim

        d_m = nf
        h_m = (h_x-mxp_size)/mxp_stride + 1
        w_m = (w_x-f_size+2*p)/mxp_stride + 1
        
        mxp_dim = np.uint32((n_x,d_m,h_m,w_m))

        return (conv_dim, mxp_dim)


    # =============================================================================
    # forward and backward for convolution
    # =============================================================================


    @jit
    def conv_fwdprop(self,Input, Filters, biases, stride=1, padding=0):
        '''
        Return convolved feature map after vectorized convolution operation.

        Inputs:
        ______
            Input:
                input maps to the conv layer conforming to the dimesnions:
                n_examples, depth (channels), height, width.
            Filters:
                the filters to be convolved witht the input maps. The dimensions 
                should conform with the following order: n_filters, depth, 
                height, width.
            biases:
                biases for the convolution operation. shape: (n_filters,1)

            padding: 
                the number of zero padding on each side of the input feature maps.
                the new shape of input feature maps after padding will be 
                height = height +2*p, width = width +2*p
        Outputs:
        _______
            out_maps:
                the feature maps that are convolved with the filters. 
                shape out output is (n_out, d_out, h_out, w_out)
                n_out remains the same as n_examples, 
                d_out = n_filters
                h_out = (h_in+2*p - h_f)/stride + 1
                w_out = (w_in+2*p - w_f)/stride + 1
            fwd_cache : 
                Contains information about the  forward pass, the cache is a tuple
                having the the format: (Input, Filters, biases, s, p, Input2col)   
        '''

        fwd_cache = Filters, biases, stride, padding
        #Determine input and output sizes
        p = padding
        s = stride
        n_f, d_f, h_f, w_f = Filters.shape
        n_x, d_x, h_x, w_x = Input.shape
        h_o = (h_x - h_f + 2 * p) / s + 1
        w_o = (w_x - w_f + 2 * p) / s + 1
        
        #check the conformity of Filter sizes with input sizes
        condition1 = (h_x - h_f + 2 * p) % s == 0
        condition2 = (w_x - w_f + 2 * p) % s == 0
        if not condition1 or not condition2:
            raise Exception('Input and filter dimensions do not conform')
        #convert to array for indexing
        h_o, w_o = ( np.uint16(h_o), np.uint16(w_o)  )

        #transform image to col for vectorized implementatio of convolution
        Input2col = im2col_indxs(Input, h_f, w_f, padding=p, stride=s)
        #reshape filter for vectorized implementation of convolution
        Filter2col = Filters.reshape(n_f, -1)

        # Convolve and then reshape so that input and output maps conform    
        out_col = Filter2col @ Input2col + biases
        out_arr = out_col.reshape(n_f, h_o, w_o, n_x)
        out_maps = out_arr.transpose(3, 0, 1, 2)

        fwd_cache = (Input, Filters, biases, s, p, Input2col)

        return out_maps, fwd_cache

    @jit
    def conv_backprop(self,dZ, fwd_cache):
        '''
        Returns gradients when backpropagated throught the convolution.
        Uses vectorized implementation for backpropagation.
        dJ/dW (dFilters) = dJ/dZ (dZ) * dZ/dW (Input)
        dJ/db (dbiases) = dJ/dZ (dZ) * [1]
        dJ/dA (dX_map) = dJ/dZ (dZ) * dZ/dA (Filters)

        Inputs:
        ______
            dZ:
                upstream derivative to the convolution, 
                conforming to the dimesnions:
                    n_examples, depth (channels), height, width.
            fwd_cache:
                Contains information about the  forward pass, the cache is a tuple
                having the the format: (Input, Filters, biases, s, p, Input2col)
        
        Outputs:
        _______
            dX_map:
                computed by dJ/dA = dJ/dZ (dZ) * dZ/dA (Filters)
                shape: conforming to dZ
            dFilters:
                dJ/dW (dFilters) = dJ/dZ (dZ) * dZ/dW (Input)
                shape conforming to the filters.

            dbiases   
                computed by dJ/db = dJ/dZ (dZ) * [1]
                shape conforming to the biases.

        '''
        # Unpack the cache and determine sizes
        Input, Filters, biases, s, p, Input2col = fwd_cache
        n_f, d_f, h_f, w_f = Filters.shape

        #compute dJ/db
        dbiases = np.sum(dZ, axis=(0, 2, 3))
        dbiases = dbiases.reshape(n_f, -1)
        
        #compute dJ/dW
        dZ_transposed = dZ.transpose(1, 2, 3, 0)
        dZ_reshaped = dZ_transposed.reshape(n_f, -1)
        dFilters = dZ_reshaped @ Input2col.T
        dFilters = dFilters.reshape((n_f, d_f, h_f, w_f))
        #compute dJ/dA
        Filters_reshaped = Filters.reshape(n_f, -1)
        dX2col = Filters_reshaped.T @ dZ_reshaped
        
        #transform the image back to origincal form
        n,c,h,w = Input.shape
        dX_map = col2im_indxs(dX2col, (n,c,h,w), h_f, w_f, padding=p, stride=s)

        return dX_map, dFilters, dbiases
    # =============================================================================
    # forward and backward for ReLU
    # =============================================================================
    def relu(self,fm, leaky = True, alphaa = 0.001):
        if not leaky: 
            return ne.evaluate('fm*(fm>0)')
        if leaky:
            return ne.evaluate('fm*(fm>0) + fm*(fm<0)*alphaa')
        
        
    def relu_derivative(self, x, leaky = True, alphaa = 0.001):
        if not leaky:    
            return ne.evaluate('(x>0)+0')
        if leaky:
            return ne.evaluate('(x>0)*1 + alphaa*(x<0)')
    # =============================================================================
    # forward and backward for MaxPooling
    # =============================================================================
    @jit
    def batch_norm_fwd(self,X, gamma, beta, eps):
        """ 
        Performs batch normalization across all the mini batches. this method is
        used for zero centering and performs normalization for better variance in
        either direction
        Equations for batch normalization:
            Xhat = (X-mu)/sqrt(var)
            X_out = gamma*Xhat + beta
            Where gamma and beta are learnable parameters
                
        Inputs:
        _________
        X: 
            shape : (n_batches, depth, height, width)
        gamma: 
            the learnable parameter associated with the layer. 
            shape: (depth, height, width)
        beta: 
            the learnable parameter associated with the layer. 
            shape: (depth, height, width)
        eps : 
            very small scaler value.
        
        Output:
        _______
        out: 
            batch normalized set of feature maps.
            shape: (n_examples, depth, height, width)
        """

        m = X.shape[0]  # n_examples
        mu = (1./m)*np.sum(X, axis = 0)  # averaging across all examples

        xhat_numerator = (X - mu)

        
        #determinin standard deviation of the data
        var = (1./m)*np.sum(xhat_numerator**2, axis = 0)
        std_dev = np.sqrt(var + eps)
        
        #determining x_hat (batch normalized feature maps)
        xhat_denomenator = 1./std_dev
        X_hat =  xhat_numerator * xhat_denomenator

        gamma_x = gamma * X_hat
        X_out = gamma_x + beta 
        
        batch_norm_cache = {"x_hat" : X_hat, "xmu" : xhat_numerator, "i_var" : xhat_denomenator,
                 "sqrt_var" : std_dev, "var" : var, "eps" : eps,
                 "gamma": gamma, "beta":beta}
        return X_out, batch_norm_cache
    
    @jit
    def batch_norm_backprop(self,dZ, cache):
        """ 
        Returns the backpropagated feature maps as well as derivatives
        with respect to gamma and beta for their update.
        Equations used:
            dJ/dBeta = dJ/dZ (i.e. dZ) * dZ/d_beta (i.e. matrix of ones) 
            dJ/dGamma = dJ/dZ * dZ/d_gamma
        
        Inputs:
        _______
        dZ:
            Upstream derivative before the backpropagation of batch normalization
            shape:
                (num_examples, depth, height, width).
        cache: dictionary of elements used in the forward propagation of the batch normalization.
        
        Outputs:
        ________
            (d_x, dBeta, dGamma): tuple of elements containing derivatives w.r.t the input
            of batch_normalization and the learnable parameters beta and gamma respectively.
        """
        x_hat, xmu, i_var,sqrt_var = cache["x_hat"], cache["xmu"], cache["i_var"], cache["sqrt_var"]
        var, eps, xmu, gamma = cache["var"], cache["eps"], cache["xmu"], cache["gamma"]
        m = dZ.shape[0]  # number of examples
    
        #determine  dJ/dBeta = dJ/dZ*dZ/d_beta  (Chain Rule)
        dBeta = 1 * np.sum(dZ, axis = 0)
        
        #determining dGamma
        dgamma_x = 1 * dZ
        dGamma = np.sum(x_hat * dgamma_x, axis = 0)
        
        # Determining d_X (derivative of loss wrt the input of batch normalization)
        d_Xhat = dgamma_x * gamma
        divar = np.sum(d_Xhat * xmu, axis = 0) #sum across all examples
        d_X_mu = d_Xhat * i_var
        d_std_dev = divar * (-1./(sqrt_var**2))
        d_Var = d_std_dev * (1./np.sqrt(var + eps)) * (1/2) 
        d_xmu_squared = d_Var * (1./m) * np.ones(dZ.shape) 
        d_X_mu += d_xmu_squared * 2 * xmu  
        dmu = np.sum(-1. * d_X_mu , axis=0)  
        d_x = d_X_mu * 1
        d_x += dmu * (1./m) * np.ones(dZ.shape)

        return (d_x, dBeta, dGamma)
    
    
    @jit
    def maxpool_fwd(self,X, pool_height = 2, pool_width = 2, stride = 2):
        '''
        Return maxpooled feature map after vectorized maxpooling operation.

        Inputs:
        ______
            X:
                input maps to the conv layer conforming to the dimesnions:
                n_examples, depth (channels), height, width.
            
            pool_height:
                the height of maxpoool window.
            
            pool_width:
                the width of maxpoool window.
            
            stride: 
              the stride of the maxpooling window
              
        Outputs:
        _______
            out:
                the feature maps that are max pooled
            cache : 
                Contains information about the  forward pass, the cache is a tuple
                having the the format: (X, x2cols, x_max_idxs, mxp_param) 
        '''

        n_exmpls, depth, h_x, w_x = X.shape        
        h_f, w_f, s = pool_height, pool_width, stride
        condition1 = (h_x - h_f ) % s == 0
        condition2 = (w_x - w_f ) % s == 0
        if not condition1 or not condition2:
            raise Exception('Input and filter dimensions do not conform')
            
            
        h_o = int((h_x - pool_height) / stride + 1)
        w_o = int((w_x - pool_width) / stride + 1)

        x_hat = X.reshape(n_exmpls * depth, 1, h_x, w_x)
        x2cols = im2col_indxs(x_hat, h_f, w_f, padding=0, stride=s)
        
        # determine the max indexes
        x_max_idxs = np.argmax(x2cols, axis=0)
        
        # determine the max 
        x2cols_max = x2cols[x_max_idxs, np.arange(x2cols.shape[1])]
        
        # reshape and transpose according to original shape and format
        out = x2cols_max.reshape(h_o, w_o, n_exmpls, depth)
        out = out.transpose(2, 3, 0, 1)
        
        #make cache and return the outputs
        mxp_param = {}
        mxp_param['stride'] = stride
        mxp_param['pool_height'], mxp_param['pool_width'] = pool_height, pool_width
          
        mxp_cache = (X, x2cols, x_max_idxs, mxp_param)
        
        return out, mxp_cache
    
    @jit
    def maxpool_backprop(self,dout, cache):
        '''
        Return backpropagated feature maps of the derivative.
        this method uses the indexes of the max's to determine where the dout
        should be stored
        
        Inputs:
        _______
            dout:
                the derivatives wrt feature maps that are max pooled
            cache : 
                Contains information about the  forward pass, the cache is a tuple
                having the the format: (X, x2cols, x_max_idxs, mxp_param)
                
        Output:
        ________
            dX:
                the sparse matrix containing the upstream derivative spreaded out
                at the location where max's existed
        '''
        # determining shapes
        X, x2cols, x_max_idxs, mxp_params = cache
        n_x, d_x, h_x, w_x = X.shape
        h_f, w_f, s = mxp_params['pool_height'], mxp_params['pool_width'], mxp_params['stride']
        
        #tranposing for eqating to dX2cols
        dout_hat = dout.transpose(2, 3, 0, 1)
        dout_hat = dout_hat.flatten()
        
        #input the dout_hat to the locations where x_max_idxs are present
        dX2cols = np.zeros_like(x2cols)
        dX2cols[x_max_idxs, np.arange(dX2cols.shape[1])] = dout_hat
        
        # determine the derivative wrt X
        dX = col2im_indxs(dX2cols, (n_x * d_x, 1, h_x, w_x), h_f, w_f, padding=0, stride=s)
        dX = dX.reshape(X.shape)
        
        return dX


    # =============================================================================
    # #    Convolution Layer    
    # =============================================================================


    @jit
    def convLayer(self,input_feature_map, filters, biases,
                layer_hyperparams={'conv_pad':1,'f_stride':1,'mxp_size':2,
                                    'mxp_stride':2},
                layer_options = {'maxpooling':True, 'batch_nomalize':True},
                layer_attr = {'layer_num': 1, 'layer_type': "Convolution", 
                                'layer_name' : 'conv'}):
        '''
        Function of the convolution operation for the convolution layers.
        
        size after convolution:
            N_out = (input_shape - f_size + 2*padding)/stride+1
        
        Inputs:
        ______
        input_feature_map:
             ndarray of input maps to the conv layer.
             shape: n_x, d_x, h_x, w_x
        filters:
            shape = n_filters, d_filter, h_filter, w_filter
         biases:
             shape = n_filters, 1
        Outputs:
        ______
            output_feature_maps: 
                the feature maps outputted from the  
                shape : n_examples, n_filters, N_out, N_out
            Layer_cache:
                dictionary of caches containing information about the layer operations
        '''
        
        # determine layer attributes
        p = layer_hyperparams['conv_pad']
        maxpooling = layer_options['maxpooling'] 
        b_normalize = layer_options['batch_normalize'] 
        
        # determine input and output shapes
        n_filters, d_f, h_f, w_f = np.uint32(filters.shape)
        n_examples, d_x, h_x, w_x = np.uint32((input_feature_map.shape))
        stride=layer_hyperparams['f_stride']
        mp_size = layer_hyperparams['mxp_size']
        mp_stride = layer_hyperparams['mxp_stride']
        h_conv = (h_x-h_f+2*p)/stride+1
        w_conv = (w_x-w_f+2*p)/stride+1
        h_out = (h_conv-mp_size)/mp_stride + 1
        w_out = (w_conv-mp_size)/mp_stride + 1
        
        # check for conformity of sizes
        if not d_x == d_f:
            l_n = layer_attr['layer_num']
            l_t = layer_attr['layer_type']
            expectation = ' Expected %d but got %d'%(d_x,d_f)
            raise Exception('In %s layer %d, d_x != d_f.'%(l_t,l_n)+expectation)
        if not h_conv.is_integer() or not h_conv.is_integer():
            l_n = layer_attr['layer_num']
            l_t = layer_attr['layer_type']
            raise Exception('for convolution, sizes do not conform in %s layer %d'%(l_t,l_n))
        if not h_out.is_integer() or not h_out.is_integer():
            l_n = layer_attr['layer_num']
            l_t = layer_attr['layer_type']
            raise Exception('for maxpooling, sizes do not conform in %s layer %d'%(l_t,l_n))

        
        #convert values to integers for indexing    
        h_out, w_out = int(h_out), int(w_out)
        h_conv, w_conv = int(h_conv), int(w_conv)
    
        
        
        # initialize arrays so values can be stored
        conv_shape = (n_examples, n_filters, h_conv, w_conv)
        out_shape = (n_examples,n_filters,h_out,w_out )
        conv_feature_maps = np.zeros((conv_shape))
        relued_feature_maps = np.zeros((conv_shape))
        output_feature_maps = np.zeros((out_shape))

        # get feature maps after convolution and relu
        conv_feature_maps, conv_cache = self.conv_fwdprop(input_feature_map, 
                                                    filters, biases, padding=p)
#        print('conv_feature_maps.shape = ',conv_feature_maps.shape)
        if b_normalize:
            l_num = layer_attr['layer_num']
            gamma = self.params['gammaL'+str(l_num-1)]
            beta = self.params['betaL'+str(l_num-1)]
            eps = 1e-3
            conv_feature_maps, bnorm_cache = self.batch_norm_fwd(conv_feature_maps, 
                                                                      gamma, beta, eps)
        else:
            bnorm_cache = 0
        
        relued_feature_maps = self.relu(conv_feature_maps, leaky = False)
        
        if maxpooling: #if there is maxpooling in the layer
            output_feature_maps, mxpcache = self.maxpool_fwd(relued_feature_maps)
        else: # else output the convolution-output
            output_feature_maps = relued_feature_maps
            mxpcache = 0
        
        #setting a cache. to be used in backpropagation
        Layer_cache = {'convd_maps':conv_feature_maps, 'conv_cache':conv_cache, 
                    'mxp_cache': mxpcache, 
                    'bnorm_cache': bnorm_cache,
                    'maxpooling':maxpooling,
                    'batch_normalize': b_normalize,
                    'layer_hyperparams':layer_hyperparams, 
                    'layer_attr':layer_attr}
        
        return output_feature_maps, Layer_cache

        
    # ==========================================================================
    #  Fully Connected Layer                               
    # ==========================================================================
    
    def init_FC_params(self,act_layer, n_classes=14):
        
        '''
        initialize weights and biases for the fully connected layer 
        according to the size of the last layer.
        '''
        
        h1 = self.n_FC_hidden
        X_size = act_layer.shape[0]
        W0 = np.random.randn(h1, X_size)/np.sqrt(X_size)
        b0 = np.random.randn(h1,1)/np.sqrt(X_size)
        W1 = np.random.randn(n_classes, h1)/np.sqrt(h1)
        b1 = np.random.randn(n_classes,1)/np.sqrt(h1)
        return W0,b0,W1,b1
    
    
    def straighten(self,act_map):
        '''
        used for flattening the last map (before fully connected layer)
        '''
        m,d,h,w = act_map.shape
        act_layer = act_map.transpose(1,2,3,0).reshape(d*h*w,m)
        return act_layer
    
    
    def Logistic(self,act_layer,w_cache):
        
        '''
        Forward propagation through the fully connected layer. the output of FC 
        is a Logistic non linearity on the scores of last layer
        '''
        
        W0,b0,W1,b1 = w_cache['Wfc0'], w_cache['bfc0'], w_cache['Wfc1'], w_cache['bfc1']
        Z0 = np.dot(W0,act_layer) + b0
        A0 = self.relu(Z0, leaky = False)
        Z1 = np.dot(W1,A0) 
        Z1 = Z1 + b1 
#        A1 = np.exp(Z1 - np.max(Z1))/( np.sum(np.exp(Z1  - np.max(Z1))) )
        
        A1 = 1/(1+np.exp(-Z1))
        return A1,Z1,A0,Z0
    
    
    def CE_loss(self,out,label,W_cache,lamda = 0):
        '''
        Determine Cross-Entropy loss
        '''

        m = label.shape[-1]
        loss = -np.multiply(label,np.log(out)) - np.multiply((1-label),np.log(1-out))
        loss = np.sum(loss)/m
        param_norm = self.squared_parameter_norm(W_cache)
        reg_loss = 0.5*lamda*param_norm*(1/m)
        loss=loss + reg_loss
        return loss
    
    
    def squared_parameter_norm(self,W_cache):
        '''
        required for calculating regularization cost
        '''
        W = W_cache
        norm_W = 0;
        for wkey in W.keys():
            norm_W += np.sum(W[wkey]*W[wkey])
        return norm_W
    
    # =============================================================================
    # Forward Propagate    
    # =============================================================================
    def forwardprop(self,X,paramsdict, FC_params_init = False):
        '''
        Forward  propagate throught all the CNN layers. the number of layers in FC
        layers are fixed so no need to loop through each layer when number of layers
        are known.
        
        Inputs:
        _____
            X: 
                input image
            paramsdict:
                dictionary of parameters that are used for forward propagation.
        Outputs:
        _______
            layers_outputs:
                List of layer outputs from CNN layers 
            layers_caches:
                this list contains dictionaries of information of every list when 
                forward propagating
            FC_cache:
                information about the fully connected layer and the intermediate 
                outputs of the fully connected layer
            
        '''
        
        L = self.cnnLayersInfo
        layers_caches=[]
#        layers_outputs = []
        FM = X
        for i,l in enumerate(L):
            n_layer = str(i)
            
            layer_hyperparams = l['layer_hyperparams']
            layer_options = l['layer_options']
            layer_attr = l['layer_attr']
            
            filters,biases = paramsdict['WL'+n_layer], paramsdict['bL'+n_layer] 
            
            FM,L_cache = self.convLayer(FM,filters,biases,layer_attr = layer_attr,
                                        layer_options = layer_options, 
                                        layer_hyperparams = layer_hyperparams) 
               
            layers_caches.append(L_cache)
#            layers_outputs.append(FM)
        
        
        act_layer = self.straighten(FM)
        
        if FC_params_init:
            self.params['Wfc0'],self.params['bfc0'],self.params['Wfc1'],self.params['bfc1'] = self.init_FC_params(act_layer,14)
            FC_params_init = False
#            
#            self.params['Wfc1'] = fcW1 
#            self.params['bfc1'] = fcb1 
#            self.params['Wfc0'] = fcW0
#            self.params['bfc0'] = fcb0
            self.velocity['Wfc1'] = np.zeros(self.params['Wfc1'].shape) 
            self.velocity['bfc1'] = np.zeros(self.params['bfc1'].shape) 
            self.velocity['Wfc0'] = np.zeros(self.params['Wfc0'].shape)
            self.velocity['bfc0'] = np.zeros(self.params['bfc0'].shape)
        
        FC_cache = {}
        FC_cache['A1'],FC_cache['Z1'],FC_cache['A0'],FC_cache['Z0'] = self.Logistic(act_layer,self.params)
         
        FC_cache['X'] = act_layer
        
        return FM.shape, layers_caches, FC_cache
            
    @jit
    def backprop(self,conv_layers_sizes, conv_L_caches,FC_cache,params_dict,y):
        der_cache = {}
        m,dlast,hlast,wlast = conv_layers_sizes
        
        dZ1 = (FC_cache['A1']-y)/m
    
        #derivatives wrt weights between last layer and hidden layer
        dWfc1 =  dZ1.dot(FC_cache['A0'].T)/m
        dbfc1 =  np.sum(dZ1, axis=1)/m
        dbfc1  = dbfc1[...,np.newaxis]
        
        #derivatives wrt hidden layer and input layer of FC
        dA0 = params_dict['Wfc1'].T.dot(dZ1)
    
        dZ0 = np.multiply(dA0, self.relu_derivative(FC_cache['Z0'], leaky = False))
        
        #derivatives wrt weights between hidden layer and input layer of FC
        dWfc0 =  dZ0.dot(FC_cache['X'].T)/m
        dbfc0 =  np.sum(dZ0, axis=1)/m
        dbfc0  = dbfc0[...,np.newaxis]
        
        dA_layer = params_dict['Wfc0'].T.dot(dZ0) #reshaping gives der wrt last f_maps of convs
        
        #reshaping the gradients from FC neural network
    #    end_map_size = asd['conv3'].shape
        dL_conv3  = dA_layer.reshape((dlast,hlast,wlast,m))
        dL_convlast = dL_conv3.transpose(3,0,1,2)
        
        der_cache['Wfc1'] = dWfc1
        der_cache['Wfc0'] = dWfc0
        der_cache['bfc1'] = dbfc1
        der_cache['bfc0'] = dbfc0
        
        n_layers = self.n_layers
        i = n_layers -1
        
        dA_conv= self.maxpool_backprop(dL_convlast ,conv_L_caches[i]['mxp_cache'])
        
    
        dZ_conv = np.multiply(dA_conv,
                               self.relu_derivative(conv_L_caches[i]['convd_maps'], leaky = False))
        b_norm = conv_L_caches[i]['batch_normalize']

        if b_norm:
            dZ_conv, d_beta, d_gamma =  self.batch_norm_backprop(dZ_conv, 
                                                                  conv_L_caches[i]['bnorm_cache'])
            der_cache['gammaL'+str(i)] = d_gamma
            der_cache['betaL'+str(i)] = d_beta

    
        dXconv, dWconv, dbconv = self.conv_backprop(dZ_conv, 
                                              conv_L_caches[i]['conv_cache'])
        
        num_l_str = str(i)
        
        der_cache['WL'+num_l_str] = dWconv
        der_cache['bL'+num_l_str] = dbconv
        
        
        #finding derivatives through convolution layers
        for i in range(n_layers-2, -1, -1):
            dA_conv= self.maxpool_backprop(dXconv ,conv_L_caches[i]['mxp_cache'])
#            dA_conv += np.random.randn(dA_conv.shape) *0.001
            dZ_conv = np.multiply(dA_conv,
                                   self.relu_derivative(conv_L_caches[i]['convd_maps'], leaky = False))
            b_norm = conv_L_caches[i]['batch_normalize']

            if b_norm:
                dZ_conv, d_beta, d_gamma =  self.batch_norm_backprop(dZ_conv, 
                                                                      conv_L_caches[i]['bnorm_cache'])
                der_cache['gammaL'+str(i)] = d_gamma
                der_cache['betaL'+str(i)] = d_beta

            
            dXconv, dWconv, dbconv = self.conv_backprop(dZ_conv, 
                                                  conv_L_caches[i]['conv_cache'])     
#            dXconv += np.random.randn(dXconv.shape)*0.001
            num_l_str = str(i)
            
            der_cache['WL'+num_l_str] = dWconv
            der_cache['bL'+num_l_str] = dbconv
        
        
        return der_cache
    
    
    def train(self, X_file_names, y, n_epochs = 10, alpha_init = 0.000001, momentum = 0.9,
              lamda = 0.1, mini_batch_size = 5, imageload=True):
        
        init_flg = self.W_init_
        directory = self.readDir
        
        names = X_file_names

        
        n_batches = y.shape[-1] / mini_batch_size
        n_batches = int(np.ceil(n_batches))
        i = 0
        alpha = alpha_init
        paramsdict = self.params
        mini_size = mini_batch_size
        for epoch in range(n_epochs):
            batch_iterable = self.get_next_names(names, y, mini_size)
            
            for mini_batch in range(n_batches):
                n,l = next(batch_iterable)
                if imageload:
                    X= self.read_images(n, directory)
                else: 
                    X = n
                
                label= l
                m = label.shape[-1]
                
                if init_flg: 
                    layers_outputs_shape , layers_caches, FC_cache= self.forwardprop(X,
                                                                         paramsdict, 
                                                                         FC_params_init = True)
                    init_flg = False
                else:
                    layers_outputs_shape , layers_caches, FC_cache= self.forwardprop(X,
                                                                         paramsdict, 
                                                                         FC_params_init = False)
                
                l = self.CE_loss(FC_cache['A1'],label,paramsdict)
                acc,cacc = self.Evaluate(X,label)
                
                print('epoch: %d/%d, batch: %d/%d, loss: %.2f, Acc: %.2f'%(epoch+1,n_epochs,
                                                                mini_batch+1,n_batches,
                                                                l,cacc))
                self.loss_history.append(l)
                
                
                self.acc_history.append(cacc)

                der_cache = self.backprop(layers_outputs_shape, layers_caches,FC_cache,paramsdict,label)
                
                
                del layers_caches, FC_cache
                self.params, self.velocity = self.parameter_update_momentum(W_cache = paramsdict,
                                                        der_cache = der_cache,V_cache=self.velocity,
                                                        m = m,lamda = lamda,alpha = alpha,
                                                        beta = momentum)
                i +=1
 
    
    def parameter_update_momentum(self,W_cache,der_cache,V_cache,m = 1,
                                  lamda = 0,alpha = 0.0001,beta = 0.9):
        
        for i,wkey in enumerate(W_cache.keys()):
            if wkey[:4] == 'beta' or wkey[:5] == 'gamma':
                upd = alpha/m*der_cache[wkey] 
                W_cache[wkey] = W_cache[wkey] - upd
                
            else:
                reg_term = (lamda/(2*m))*W_cache[wkey]
                oldV = V_cache[wkey]
                V_cache[wkey] = beta*V_cache[wkey] - (alpha)*(der_cache[wkey]+reg_term)
                W_cache[wkey] = -beta*oldV + (1+beta)*V_cache[wkey] 
    
        return W_cache, V_cache    
    
    
    def get_next_names(self,names, y, mini_batch_size):
        for idx in range(0, len(names), mini_batch_size):
            idx1 = idx + mini_batch_size
            batch_names = names[idx:idx1]
            batch_labels = y[:,idx:idx1]
            yield batch_names,batch_labels
     
        
    def read_images(self,fileNames, directory,dims = (1,200,200)):
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
        
        for i,fN in enumerate(fileNames):
            path_ = directory + fN
            im = skimage.io.imread(path_)
            im = skimage.transform.resize(im,(200,200))
            im = im[...,np.newaxis].transpose(2,0,1)
            img[i,:d_x,:,:] = im/255.0
    
        return img
       
    
    def predict_proba(self,X):
        '''
        X shape :  m,d,h,w (d,h,w should conform with input)
        
        '''
        paramsdict = self.params
        a,a,aa = self.forwardprop(X,paramsdict)
        out = aa['A1']        
        return out
    
    
    def prediction(self,X):
        '''
        X shape :  m,d,h,w (d,h,w should conform with input)
        
        '''
        paramsdict = self.params
        a,a,aa = self.forwardprop(X,paramsdict)
        out = aa['A1']        
        return np.round(out)
        
    
    def Evaluate(self,X_eval, y_eval):
        '''
        Inputs:
        __________
            X_eval:
                ndarray of examples. 
                Shape: (number of examples, depth, height, width)
            y_eval: 
                ndarray of ground truth labels. 
                Shape: (number of classes, number of examples)
        Output:
        ______
            accuracy:
                a scaler of containing accuracy measure.  
    
        '''
    
        ypred = self.prediction(X_eval)
        acc = np.zeros(y_eval.shape[0]) + y_eval.shape[1] # asssume 100% accuracy
        
        for k in range(y_eval.shape[0]): #loop through each class
            for m in range(0, y_eval.shape[1]): # loop through each example
                if (ypred[k,m] != y_eval[k,m]): # if the predicted and actual are not equal
                    acc[k] = acc[k]-1           # then reduce the true_positive counter
        
        accuracy = acc/y_eval.shape[1] * 100 # calculate accuracy in percentage
        c_accuracy = np.sum(accuracy)/y_eval.shape[0]
        return accuracy, c_accuracy
        