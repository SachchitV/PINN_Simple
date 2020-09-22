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

def swish(x):
    return x * tf.math.sigmoid(x)
    

class ODESolver1DBase(object):
    # This is the base class of solver which handles the backend of solving 
    # Equations. User have to create child class of this class and at least 
    # implement loss_and_grad() method.
    
    def __init__(self, 
                 inDim = 1, 
                 outDim = 1, 
                 nHiddenLayer = 5, 
                 nodePerLayer = 10, 
                 nIter = 1000,
                 learningRate = 0.001,
                 batchSize = 1001,
                 activation = Activation(swish),
                 kernelInitializer = tf.keras.initializers.he_uniform()):
        # Number of Hidden Layer
        self.nLayers = nHiddenLayer
        
        # Neural Network Model is simple sequential model
        self.nnModel = tf.keras.Sequential()
        
        # Add first hidden Layer
        self.nnModel.add(layers.Dense(nodePerLayer, 
                                      activation=activation, 
                                      input_dim=inDim,
                                      kernel_initializer=kernelInitializer,
                                      bias_initializer='zeros'))
        
        # Add rest of the hidden layer
        for i in range(self.nLayers-1):
            self.nnModel.add(layers.Dense(nodePerLayer, 
                                          activation=activation,
                                          kernel_initializer=kernelInitializer,
                                          bias_initializer='zeros'))
        
        # Add Output Layer
        self.nnModel.add(layers.Dense(outDim, 
                                       kernel_initializer=kernelInitializer,
                                       bias_initializer='zeros'))
        
        # Optimization iterations
        self.nIter = nIter
        
        # Optimization Learning Rate
        self.learningRate = learningRate
        
        # Optimization Batch Size
        self.batchSize = batchSize
        
        # Storing History of Loss function to see convergence
        self.lossHistory = []
        
    def __call__(self, x):
        # Retun output of neural network with x input
        return self.nnModel(x)
    
    def loss_and_grad(self, x):
        # This function have to be implemented in Child Class
        pass
    
    def solve(self, trainSet):
        # Selecting optimizer and batch size
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        
        # Create Batch
        batchList = self.create_batch(trainSet)
        nBatch = len(batchList)
        
        # Optimization loop
        for i in range(self.nIter):
            for batch in batchList:
                # Calculate loss function and gradient
                lossValue, grads = self.loss_and_grad(batch)
    
                # Store Loss Value for each Iteration           
                self.lossHistory.append(lossValue.numpy())
                
                # Print for showing progress
                print("Epoch: {}, BatchNo: {}, Loss: {}".format(i+1,
                                                                optimizer.iterations.numpy()-i*nBatch+1,
                                                                lossValue.numpy()))
                
                # Nudge the weights of neural network towards convergence (hopefully)
                optimizer.apply_gradients(zip(grads, self.nnModel.trainable_variables))
                
    def create_batch(self, trainSet):
        nBatch = int(trainSet.shape[0]/self.batchSize)+1
        
        batchList = []
        
        for i in range(nBatch):
            start = i*self.batchSize
            end = (i+1)*self.batchSize
            if end > trainSet.shape[0]:
                end = trainSet.shape[0]
            
            if start != end:
                batchList.append(trainSet[start:end])
        
        return batchList
            
    def get_weights(self):
        for layer in self.nnModel.layers:
            print(layer.get_weights())