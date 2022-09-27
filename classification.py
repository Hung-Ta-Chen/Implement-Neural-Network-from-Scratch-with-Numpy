# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 05:04:53 2021

@author: narut
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fully_connected_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from network import Network
from util import *
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    np.set_printoptions(threshold=sys.maxsize)
    pd.set_option('display.max_columns', None)
    
    # Read data from csv file
    df = pd.read_csv("./ionosphere_data.csv", header=None)
    # One hot encoding
    df[35] = list(map(lambda x: 1 if x == 'g' else 0, df[34]))
    df[36] = list(map(lambda x: 1 if x == 'b' else 0, df[34]))
    df = df.drop([34], axis=1)
    
    data = df.to_numpy()
    np.random.shuffle(data)
    
    data_train = data[0:280, :]
    data_test = data[280: , :]
    
    x_train = data_train[:, 0:34]
    y_train = data_train[:, 34:36]
    x_test = data_test[:, 0:34]
    y_test = data_test[:, 34:36]
    
    ## Build the network
    nn = Network(cross_entropy, cross_entropy_prime)
    nn.add_layer(FullyConnectedLayer(34, 30))
    nn.add_layer(ActivationLayer(tanh, tanh_prime))
    nn.add_layer(FullyConnectedLayer(30, 3))
    nn.add_layer(ActivationLayer(tanh, tanh_prime))
    nn.add_layer(FullyConnectedLayer(3, 2))
    nn.add_layer(ActivationLayer(sigmoid, sigmoid_prime))
    plt.rcParams["figure.figsize"] = (8, 3.5)
    
    
    ## Fit
    error = nn.train(x_train, y_train, 300, 0.01)
    # Plot learning curve
    plt.plot(error)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Learning curve")
    plt.show()
    print(" ")
    
    ## Plot latent features
    # 3D latent feature
    """
    error, latent = nn.train_latent(x_train, y_train, 300, 0.01, 3, 20)
    # Plot learning curve
    plt.plot(error)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Learning curve")
    plt.show()
    print(" ")    
    
    target = ["class0", "class1"]
    for i in range(latent.shape[0]):
        latent_1 = []
        latent_0 = []
        for j in range(latent.shape[1]):
            if np.array_equal(y_train[j, :], np.array([1, 0])):
                latent_0.append(latent[i, j, :])
            elif np.array_equal(y_train[j, :], np.array([0, 1])):
                latent_1.append(latent[i, j, :])
            else:
                print("bs")
                
        latent_1 = np.array(latent_1)
        latent_0 = np.array(latent_0)       
        
        # Plot 3d 
        ax2 = plt.axes(projection='3d')
        ax2.scatter(latent_0[:, 0], latent_0[:, 1], latent_0[:, 2], c = 'r', s = 10)
        ax2.scatter(latent_1[:, 0], latent_1[:, 1], latent_1[:, 2], c = 'b', s = 10)
        ax2.legend(target)
        ax2.grid()
        plt.title("Latent feature in {}th epoch".format( 20* (i+1) ))
        plt.show()
    
    # 2D latent feature
    error, latent = nn.train_latent(x_train, y_train, 300, 0.01, 2, 20)
    # Plot learning curve
    plt.plot(error)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Learning curve")
    plt.show()
    print(" ")    
    plt.rcParams["figure.figsize"] = (4, 3.5)
    
    target = ["class0", "class1"]
    for i in range(latent.shape[0]):
        latent_1 = []
        latent_0 = []
        for j in range(latent.shape[1]):
            if np.array_equal(y_train[j, :], np.array([1, 0])):
                latent_0.append(latent[i, j, :])
            elif np.array_equal(y_train[j, :], np.array([0, 1])):
                latent_1.append(latent[i, j, :])
            else:
                print("bs")
                
        latent_1 = np.array(latent_1)
        latent_0 = np.array(latent_0)       
        
        # Plot 3d 
        fig , ax = plt.subplots()
        plt.plot(latent_0[:, 0], latent_0[:, 1], label = "label", color = "red", linestyle="", marker="o", markersize=5)
        plt.plot(latent_1[:, 0], latent_1[:, 1], label = "label", color = "blue", linestyle="", marker="o", markersize=5)
        plt.title("Latent feature in {}th epoch".format( 20* (i+1) ))
     
        ax.legend(target)
        plt.show()
    """
     
    
    ## Predict
    # Use training data 
    train_result = nn.predict(x_train)
    train_result = np.squeeze(train_result, axis = 1)
    #print(train_result)
    train_acc = accuracy(y_train, train_result)
    print("<< Predicted result with training data >>")
    print("Accuracy = {}".format(train_acc))
    plt.rcParams["figure.figsize"] = (10, 3.5)

    
    fig , ax = plt.subplots()
    plt.plot(range(len(y_train)), np.argmax(y_train, axis = 1), label = "label", color = "blue", linestyle="", marker="o")
    plt.plot(range(len(train_result)), np.argmax(train_result, axis = 1), label = "predict", color = "orange", linestyle="" ,marker="o")
    plt.title("Prediction for training data")
    plt.ylabel("Class")
    plt.xlabel("nth case")
    leg = ax.legend(loc='upper left') 
    plt.show()
    print(" ")
    
    # Use testing data 
    test_result = nn.predict(x_test)
    test_result = np.squeeze(test_result, axis = 1)
    #print(train_result)
    test_acc = accuracy(y_test, test_result)
    print("<< Predicted result with testing data >>")
    print("Accuracy = {}".format(test_acc))
    fig , ax = plt.subplots()
    #plt.rcParams["figure.figsize"] = (8, 3.5)
    plt.plot(range(len(y_test)), np.argmax(y_test, axis = 1), label = "label", color = "blue", linestyle="", marker="o")
    plt.plot(range(len(test_result)), np.argmax(test_result, axis = 1), label = "predict", color = "orange", linestyle="" ,marker="o")
    plt.title("Prediction for testing data")
    plt.ylabel("Class")
    plt.xlabel("nth case")
    leg = ax.legend(loc='upper left') 
    plt.show()
    print(" ")
    