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

# GPU configuration for newer TensorFlow versions
# Suppress CUDA warnings when no GPU is available
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) available")
    else:
        print("No GPU detected, using CPU")
except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(f"GPU configuration error: {e}")
    print("Falling back to CPU")

tf.keras.backend.set_floatx('float64')


# Defining BentIdentity Function which will be used as activation function
# to avoid saturation
def bentIdentity(x):
    return x + (tf.sqrt(x*x+1)-1)/2

def swish(x):
    return x * tf.math.sigmoid(x)
    

class PINNBaseModel(object):
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
        
        # Selecting optimizer and batch size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        
        # Optimization Batch Size
        self.batchSize = batchSize
        
        # Storing History of Loss function to see convergence
        self.lossHistory = []
        self.minLoss = -1.
        self.minLossWeights = self.nnModel.get_weights()
        
    def __call__(self, x):
        # Retun output of neural network with x input
        return self.nnModel(x)
    
    def train_step(self, x):
        # This function have to be implemented in Child Class
        pass
    
    def solve(self, trainSet):
               
        # Create Batch
        batchList = self.create_batch(trainSet)
        nBatch = len(batchList)
        
        # Optimization loop
        for i in range(self.nIter):
            for batch in batchList:
                # Calculate loss function and gradient
                lossValue = self.train_step(batch)
    
                # Store Loss Value for each Iteration           
                self.lossHistory.append(lossValue.numpy())
                
                # Print for showing progress
                tf.print("Epoch: {}, BatchNo: {}, Loss: {}".format(i+1,
                                                                self.optimizer.iterations.numpy()-i*nBatch+1,
                                                                self.lossHistory[-1]))
            
            # Store First loss as minimum loss
            if self.minLoss < 0:
                self.minLoss = self.lossHistory[-1]
            
            # Store minimum loss
            if (self.minLoss > self.lossHistory[-1]):
                self.minLoss = self.lossHistory[-1]
                self.minLossWeights = self.nnModel.get_weights()
        
        self.nnModel.set_weights(self.minLossWeights)
        
        tf.print("\n\n")
        tf.print("------------------------------")
        tf.print("Minimum Loss: %.5E"%self.minLoss)
        tf.print("------------------------------")
    
    @tf.function
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
    
    @tf.function        
    def get_weights(self):
        for layer in self.nnModel.layers:
            print(layer.get_weights())