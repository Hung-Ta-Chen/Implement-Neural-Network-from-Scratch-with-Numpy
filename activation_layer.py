# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:58:35 2021

@author: narut
"""

from layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.act = activation
        self.act_prime = activation_prime
        
    def forward(self, input_data):
        self.input = input_data
        self.output = self.act(self.input)
        return self.output
    
    def backprop(self, output_grad, learning_rate):
        input_grad = np.multiply((self.act_prime(self.input)), output_grad)
        return input_grad