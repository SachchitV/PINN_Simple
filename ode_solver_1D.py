# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:19:02 2020

@author: sachchit
"""

# Importing Basic Libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation

import numpy as np

tf.keras.backend.set_floatx('float64')


# Defining BentIdentity Function which will be used as activation function
# to avoid saturation
def bentIdentity(x):
    return x + (tf.sqrt(x*x+1)-1)/2
    

class ODESolver1DBase(object):
    # This is the base class of solver which handles the backend of solving 
    # Equations. User have to create child class of this class and at least 
    # implement loss_and_grad() method.
    
    def __init__(self, inDim = 1, outDim = 1, nHiddenLayer = 5, nodePerLayer = 10):
        # Number of Hidden Layer
        self.nLayers = nHiddenLayer
        
        # Neural Network Model is simple sequential model
        self.nnModel = tf.keras.Sequential()
        
        # Add first hidden Layer
        self.nnModel.add(layers.Dense(nodePerLayer, 
                                      activation=Activation(bentIdentity), 
                                      input_dim=inDim,
                                      kernel_initializer='random_uniform',
                                      bias_initializer='zeros'))
        
        # Add rest of the hidden layer
        for i in range(self.nLayers-1):
            self.nnModel.add(layers.Dense(nodePerLayer, 
                                          activation=Activation(bentIdentity),
                                          kernel_initializer='random_uniform',
                                          bias_initializer='zeros'))
        
        # Add Output Layer
        self.nnModel.add(layers.Dense(outDim, 
                                       kernel_initializer='random_uniform',
                                       bias_initializer='zeros'))
        
        # Storing History of Loss function to see convergence
        self.lossHistory = []
        
    def __call__(self, x):
        # Retun output of neural network with x input
        return self.nnModel(x)
    
    def loss_and_grad(self, x):
        # This function have to be implemented in Child Class
        pass
    
    def solve(self, inputs):
        # This is solver
        inputs = tf.convert_to_tensor(np.array([inputs]).transpose())
        
        # Selecting optimizer and batch size
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)
        optimizer.batch_size = 1001
        
        # Optimization loop
        for i in range(300):
            # Calculate loss function and gradient
            lossValue, grads = self.loss_and_grad(inputs)

            # Store Loss Value for each Iteration           
            self.lossHistory.append(lossValue.numpy())
            
            # Print for showing progress
            print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                                      lossValue.numpy()))
            
            # Nudge the weights of neural network towards convergence (hopefully)
            optimizer.apply_gradients(zip(grads, self.nnModel.trainable_variables))
            
    # def get_weights(self):
    #     for layer in self.nnModel.layers:
    #         print(layer.get_weights())
