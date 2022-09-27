# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 01:45:11 2021

@author: narut
"""
import numpy as np
from fully_connected_layer import FullyConnectedLayer

class Network:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss_func = loss
        self.loss_prime = loss_prime
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def predict(self, input_data):
        sample_size = len(input_data)
        result = []
        
        for i in range(sample_size):
            output = input_data[i].reshape(1, -1)
            
            for layer in self.layers:
                output = layer.forward(output)
                
            result.append(output)
        
        return np.array(result)
    
    def train(self, x_train, y_train, epoch_num, lr):
        sample_size = len(x_train)
        error_list = []
        print("==== Start training ====")
        
        #training
        for epoch in range(epoch_num):
            err = 0
            output_list = []
            
            for i in range(sample_size):
                output = x_train[i].reshape(1, -1)
                
                #forward
                for layer in self.layers:
                    output = layer.forward(output)
         
                output_list.append(output)
                err += self.loss_func(y_train[i], output)
                
                #backprop
                grad = self.loss_prime(y_train[i], output)
                for layer in reversed(self.layers):
                    grad = layer.backprop(grad, lr)
                    
            
            err /= sample_size
            error_list.append(err)
            
            print('epoch %d/%d   loss=%f' % (epoch+1, epoch_num, err))
        
        return error_list
    
    def train_latent(self, x_train, y_train, epoch_num, lr, feature_size, interval):
        sample_size = len(x_train)
        error_list = []
        latent = np.zeros((int(epoch_num) // int(interval), int(sample_size), int(feature_size)))
        print("==== Start training ====")
        
        #training
        for epoch in range(epoch_num):
            err = 0
            output_list = []
            
            for i in range(sample_size):
                output = x_train[i].reshape(1, -1)
                
                #forward
                for layer in self.layers:
                    output = layer.forward(output)
                    
                    if (epoch + 1) % interval == 0:
                        if output.shape == (1, feature_size):
                            if isinstance(layer, FullyConnectedLayer):
                                latent[(((epoch+1)//int(interval))-1), i, :] = output
                            
         
                output_list.append(output)
                err += self.loss_func(y_train[i], output)
                
                #backprop
                grad = self.loss_prime(y_train[i], output)
                for layer in reversed(self.layers):
                    grad = layer.backprop(grad, lr)
                    
            
            err /= sample_size
            error_list.append(err)
            
            print('epoch %d/%d   loss=%f' % (epoch+1, epoch_num, err))
        
        return error_list, latent
    