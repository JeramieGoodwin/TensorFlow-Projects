# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:36:31 2018

@author: jg568_000
"""

import numpy as np
import json
import time
import tensorflow as tf
import keras
from keras.layers import *
from lifelines.utils import concordance_index


class TFSurv:
    def __init__(self, n_in, learning_rate, 
                  hidden_layers_sizes = None, lr_decay = 0.0,
                  momentum = 0.9, L2_reg = 0.0, L1_reg = 0.0,
                  activation = 'rectify', dropout = None, 
                  batch_norm = False, standardize = False
                  ):
        """
        This class implements and trains a DeepSurv model.
        Parameters:
            n_in: number of input nodes.
            learning_rate: learning rate for training.
            lr_decay: coefficient for Power learning rate decay.
            L2_reg: coefficient for L2 weight decay regularization. Used to help
                prevent the model from overfitting.
            L1_reg: coefficient for L1 weight decay regularization
            momentum: coefficient for momentum. Can be 0 or None to disable.
            hidden_layer_sizes: a list of integers to determine the size of
                each hidden layer.
            activation: a lasagne activation class.
                Default: lasagne.nonlinearities.rectify
            batch_norm: True or False. Include batch normalization layers.
            dropout: if not None or 0, the percentage of dropout to include
                after each hidden layer. Default: None
            standardize: True or False. Include standardization layer after
                input layer.
        """
        
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, n_in], name='X')
        self.E = tf.placeholder(dtype = tf.float32, name='E')
        
             
        self.standardize = standardize
        
        if activation == 'rectify':
            activation_fn = tf.nn.relu
        elif activation == 'selu':
            activation_fn = tf.nn.selu
        else:
            raise IllegalArgumentException("Unknown activation function %s" % activation)
        
        out = x
        in_size = n_in
        
        for i in hidden_layers_sizes:
            if activation_fn == tf.nn.relu:
                weights = tf.Variable(tf.glorot_uniform_initializer((in_size,i), dtype=tf.float32))
            else:
                weights = tf.Variable(tf.truncated_normal((in_size, i)),dtype = tf.float32)
                
            out = tf.layers(out, weights)
            
            if batch_norm:
                batch_mean, batch_var = tf.nn.moments(out,[0])
                out = tf.nn.batch_normalization(x=out, mean=batch_mean,
                                                var=batch_var)
            if not dropout is None:
                out = tf.nn.dropout(x=out, keep_prob=dropout)
                
            
                
            
            
                
        
        
        
        
