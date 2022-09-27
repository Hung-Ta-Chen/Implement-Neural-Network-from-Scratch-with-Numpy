# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:48:14 2021

@author: narut
"""

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        raise NotImplementedError("Forward prop not implemented")

    def backprop(self, output_grad):
        raise NotImplementedError("Backward prop not implemented")
        
