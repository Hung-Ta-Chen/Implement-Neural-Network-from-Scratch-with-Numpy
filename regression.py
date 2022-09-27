# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:20:11 2021

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
    df = pd.read_csv("./energy_efficiency_data.csv")
    print("Correlation:", df.corr(method='pearson'))
    print("")
    print("")
    
    # One hot encoding
    df["Orientation2"] = list(map(lambda x: 1 if x == 2 else 0, df["Orientation"]))
    df["Orientation3"] = list(map(lambda x: 1 if x == 3 else 0, df["Orientation"]))
    df["Orientation4"] = list(map(lambda x: 1 if x == 4 else 0, df["Orientation"]))
    df["Orientation5"] = list(map(lambda x: 1 if x == 5 else 0, df["Orientation"]))  
    df["Distribution0"] = list(map(lambda x: 1 if x == 0 else 0, df["Glazing Area Distribution"]))
    df["Distribution1"] = list(map(lambda x: 1 if x == 1 else 0, df["Glazing Area Distribution"]))
    df["Distribution2"] = list(map(lambda x: 1 if x == 2 else 0, df["Glazing Area Distribution"]))
    df["Distribution3"] = list(map(lambda x: 1 if x == 3 else 0, df["Glazing Area Distribution"]))
    df["Distribution4"] = list(map(lambda x: 1 if x == 4 else 0, df["Glazing Area Distribution"]))
    df["Distribution5"] = list(map(lambda x: 1 if x == 5 else 0, df["Glazing Area Distribution"]))
    df = df.drop(['Orientation', 'Glazing Area Distribution'], axis=1)
    
    columns = df.columns.tolist()
    columns = columns[0:6] + columns[8:18] + columns[6:8]
    df = df[columns]
    # Normalize the data
    df.iloc[:,0:6] = df.iloc[:,0:6].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    #df.to_csv("./test.csv")
    
    # Split training data and test data
    data = df.to_numpy()
    np.random.shuffle(data)
    
    data_train = data[0:576, :]
    data_test = data[576: , :]
    
    x_train = data_train[:, 0:16]
    #x_train = data_train[:, np.r_[0:1, 2:16]]
    #x_train = data_train[:, 3:5]
    y_train = data_train[:, 16]
    x_test = data_test[:, 0:16]
    #x_test = data_test[:, np.r_[0:1, 2:16]]
    #x_test = data_test[:, 3:5]
    y_test = data_test[:, 16]
    
    
    
    ## Build the network
    nn = Network(sse, sse_prime)
    nn.add_layer(FullyConnectedLayer(16, 20))
    #nn.add_layer(FullyConnectedLayer(2, 25))
    nn.add_layer(ActivationLayer(tanh, tanh_prime))
    nn.add_layer(FullyConnectedLayer(20, 20))
    nn.add_layer(ActivationLayer(tanh, tanh_prime))
    nn.add_layer(FullyConnectedLayer(20, 1))
    nn.add_layer(ActivationLayer(linear, linear_prime))
    
    
    ## Fit
    error = nn.train(x_train, y_train, 500, 0.01)
    # Plot learning curve
    plt.plot(error)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Learning curve")
    plt.show()
    print(" ")
      
    
    ## Predict
    # Use training data 
    train_result = nn.predict(x_train)
    train_result = np.squeeze(train_result)
    train_rms = rms(y_train, train_result)
    
    print("<< Predicted result with training data >>")
    print("RMS = {}".format(train_rms))
    fig , ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (8, 3.5)
    plt.plot(range(len(y_train)), y_train, label = "label", color = "blue", linewidth = 0.5)
    plt.plot(range(len(train_result)), train_result, label = "predict", color = "orange", linewidth = 0.5)
    plt.title("Prediction for training data")
    plt.ylabel("heating load")
    plt.xlabel("nth case")
    leg = ax.legend(loc='upper left') 
    plt.show()
    print(" ")
    
    
    # Use testing data
    test_result = nn.predict(x_test)
    test_result = np.squeeze(test_result)
    test_rms = rms(y_test, test_result)
    
    print("<< Predicted result with testing data >>")
    print("RMS = {}".format(test_rms))
    fig , ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (8, 3.5)
    plt.plot(range(len(y_test)), y_test, label = "label", color = "blue", linewidth = 0.5)
    plt.plot(range(len(test_result)), test_result, label = "predict", color = "orange", linewidth = 0.5)
    plt.title("Prediction for testing data")
    plt.ylabel("heating load")
    plt.xlabel("nth case")
    leg = ax.legend(loc='upper left') 
    plt.show()
    
    
    
    
    
    
