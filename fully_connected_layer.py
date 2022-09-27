# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:45:25 2021

@author: narut
"""
from layer import Layer
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        
    def forward(self, input_data):
        self.input = input_data
        self.output = np.matmul(input_data, self.weight) + self.bias
        return self.output
        
    def backprop(self, output_grad, learning_rate):        
        input_grad = np.matmul(output_grad, self.weight.T)
        weight_grad = np.matmul(self.input.T, output_grad)
        bias_grad = output_grad
        
        self.weight -= learning_rate * weight_grad
        self.bias -= learning_rate * bias_grad
        return input_grad
        

        