# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:42:20 2020

@author: sachchit
"""

import matplotlib.pyplot as plt

from pinn_base_model import PINNBaseModel, tf, np, swish, bentIdentity, Activation

class ZeroOrderODE(PINNBaseModel):
    # Here we are trying to Approximat f(x) = y = x + sin(4*pi*x)
    # Silly, but helps in understanding implementation
    
    def loss_and_grad(self, x):
        # All magic happens here. Needs to be written Carefully
        # It have to return single value of Loss Function
        # and also the Gradient of Loss function with respect to all
        # trainable variables
        
        with tf.GradientTape() as lossTape:
            yHat = self.nnModel(x)
            
            # Loss Function
            # Here we are actually defining equations
            currentLoss = tf.reduce_sum((yHat - (x + tf.sin(4*np.pi*x)))**2)/len(x)
            
            # print(x,x + tf.sin(4*np.pi*x))
            
        return currentLoss, lossTape.gradient(currentLoss, self.nnModel.trainable_variables)
    
model = ZeroOrderODE(inDim = 1, 
                     outDim = 1, 
                     nHiddenLayer = 10, 
                     nodePerLayer = 50, 
                     nIter = 500,
                     learningRate = 0.001,
                     batchSize = 50,
                     activation = Activation(swish),
                     kernelInitializer = tf.keras.initializers.he_uniform())

# Input Matrix (aka Training)
trainMin = 0
trainMax = 1
nTrain = 50
scale = trainMax - trainMin
trainSet = trainMin + scale*tf.constant(np.random.rand(nTrain,1))

model.solve(trainSet)


# Testing Set
nTest = 50
scale = trainMax - trainMin
testSet = trainMin + scale*np.random.rand(nTest,1)
xTest = testSet[:,0]

# Comparision with actual function
plt.figure(0)
plt.scatter(xTest, model(tf.convert_to_tensor(testSet)), label="Neural Net")
plt.scatter(xTest, (xTest + np.sin(4*np.pi*xTest)), label="Actual")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=2, mode="expand")

# Convergence History
plt.figure(1)
plt.plot(model.lossHistory)
plt.yscale('log')
plt.xlabel("Optimizer Iteration")
plt.ylabel("Loss Value")
