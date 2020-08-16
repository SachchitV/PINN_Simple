# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:26:55 2020

@author: sachchit
"""

import matplotlib.pyplot as plt

from ode_solver_1D import ODESolver1DBase, tf, np

class ZeroOrderODE(ODESolver1DBase):
    # Here we are trying to Approximat f(x) = y = x^2
    # Silly, but helps in understanding implementation
    
    def loss_and_grad(self, x):
        # All magic happens here. Needs to be written Carefully
        # It have to return single value of Loss Function
        # and also the Gradient of Loss function with respect to all
        # trainable variables
        
        with tf.GradientTape() as g:
            yHat = self.nnModel(x)
            
            # Loss Function
            # Here we are actually defining equations
            currentLoss = tf.reduce_sum((yHat - x**2)**2)/len(x)
            
        return currentLoss, g.gradient(currentLoss, self.nnModel.trainable_variables)
    
model = ZeroOrderODE(inDim = 1, 
                     outDim = 1, 
                     nHiddenLayer = 10, 
                     nodePerLayer = 50, 
                     nIter = 500,
                     learningRate = 0.001,
                     batchSize=20)

# Input Matrix (aka Training)
trainMin = -20
trainMax = 20
nTrain = 20
scale = trainMax - trainMin
trainSet = trainMin + scale*tf.constant(np.random.rand(nTrain,1))

model.solve(trainSet)


# Testing Set
nTest = 50
scale = trainMax - trainMin
testSet = trainMin + scale*np.random.rand(nTest,1)

plt.figure(0)
plt.scatter(testSet[:,0], model(tf.convert_to_tensor(testSet)))
plt.scatter(testSet[:,0], testSet[:,0]**2)
plt.figure(1)
plt.plot(model.lossHistory)
plt.yscale('log')
