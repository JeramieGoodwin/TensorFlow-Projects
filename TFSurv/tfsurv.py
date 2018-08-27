# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:36:31 2018

@author: jg568_000
"""

import numpy as np
import json
import time
import tensorflow as tf
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
        
        # Default standardization values: mean=0, std=1
        self.offset = np.zeros(shape=n_in, dtype=np.float32)
        self.scale = np.ones(shape=n_in, dtype=np.float32)
        
        nn = tf.keras.layers.Input(shape=(None,n_in))
        
        self.standardize = standardize
        
        if activation == 'rectify':
            activation_fn = tf.nn.relu
        elif activation == 'selu':
            activation_fn = tf.nn.selu
        else:
            raise ValueError("Unknown activation function %s" % activation)
        
        in_size = n_in
        
        # Hidden layers
        for i in hidden_layers_sizes:
            if activation_fn == tf.nn.relu:
                weights = tf.Variable(tf.glorot_uniform_initializer((in_size,i), dtype=tf.float32))
            else:
                weights = tf.Variable(tf.truncated_normal((in_size, i)),dtype = tf.float32)
                
            nn = tf.layers.Dense(weights)(nn)
            
            if batch_norm:
               nn = tf.keras.layers.BatchNormalization()(nn)
            if not dropout is None:
               nn = tf.keras.layers.Dropout(dropout)(nn)
        
        # Output layer
        weights = tf.Variable(tf.truncated_normal((in_size, 1)), dtype=tf.float32)
        bias = tf.Variable(tf.zeros(1), dtype=tf.float32)
        nn = tf.keras.layers.Dense(weights,activation=activation_fn)(nn)
        
        
    def _negative_log_likelihood(self, E, deterministic = False):
        """Return the negative average log-likelihood of the prediction
        of this model under a given target distribution.
        math::
            \frac{1}{N_D} \sum_{i \in D}[F(x_i,\theta) - log(\sum_{j \in R_i} e^F(x_j,\theta))] - \lambda P(\theta)
        where:
            D is the set of observed events
            N_D is the number of observed events
            R_i is the set of examples that are still alive at time of death t_j
            F(x,\theta) = log hazard rate
        Note: We assume that there are no tied event times
        Parameters:
            E (n,): TensorVector that corresponds to a vector that gives the censor variable for each example
        deterministic: True or False. Determines if the output of the network
            is calculated determinsitically.
        Returns:
            neg_likelihood: Theano expression that computes negative partial Cox likelihood
            """
        risk = self.risk(deterministic)
        hazard_ratio = tf.exp(risk)
        log_risk = tf.log(tf.cumsum(hazard_ratio))
        uncensored_likelihood = risk.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        num_observed_events = np.sum(E)
        neg_likelihood = -tf.reduc_sum(censored_likelihood) / num_observed_events
        return neg_likelihood
