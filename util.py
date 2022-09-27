# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:22:39 2021

@author: narut
"""

import numpy as np

# activation functions

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_prime(x):
    return np.heaviside(x, 1)

def linear(x):
    return x

def linear_prime(x):
    return 1

def sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1+np.exp(x)), 1/(1+np.exp(-x)))
    return sig

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# error functions

def rms(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def accuracy(y_true, y_pred):
    correct = 0.0
    for i in range(len(y_true)):
        if np.argmax(y_true[i]) == np.argmax(y_pred[i]):
            correct += 1.0            
    return correct / len(y_true)

# loss functions

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2) * (0.5)
 
def sse_prime(y_true, y_pred):
    return y_pred - y_true

def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum( y_true * np.log(y_pred+1e-9))

def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
    
    