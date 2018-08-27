# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:37:19 2018

@author: jeramie.goodwin
"""

import numpy as np
import json
import time
import tensorflow as tf
import keras
import keras.backend as K
from lifelines.utils import concordance_index
import pandas as pd


def build_data_dict(df):
    x = df[[c for c in df.columns if 'Variable' in c]]
    e = df[[c for c in df.columns if 'Event' in c]]
    t = df[[c for c in df.columns if 'Time' in c]]
    
    data_dict = {'x':x, 
                 'e':e,
                 't':t
                 }
    return data_dict

def neg_log_likelihood(**data):
    x = data['x']
    e  =data['e']
    
    hazard_ratio = K.exp(x)
    log_risk = K.log(K.cumsum(hazard_ratio))
    uncensored_likelihood = x - log_risk
    censored_likelihood = uncensored_likelihood * e
    loss = -K.sum(censored_likelihood)
    
    return loss

def base_model():
    in_data = keras.layers.Input(shape=[None,None,4])
    net = keras.layers.Dense(activation='relu')(in_data)
    net = keras.layers.Dropout(0.4)(net)
    net.compile(loss=neg_log_likelihood, 
                  optimizer='adam', 
                  metrics=[concordance_index])
    return net 


url = r'https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/notebooks/example_data.csv'
data = pd.read_csv(url)
data_dict = build_data_dict(data)

model = base_model()