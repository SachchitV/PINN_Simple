# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:31:46 2020

@author: sachchit
"""

import matplotlib.pyplot as plt

from pinn_base_model import PINNBaseModel, tf, np, swish, bentIdentity, Activation

class FirstOrderODE(PINNBaseModel):
    # Here we are trying to Approximat df(x)/dx = dy/dx = 1/x
    # Silly, but helps in understanding implementation
    
    @tf.function
    def train_step(self, x):
        # All magic happens here. Needs to be written Carefully
        # It have to return single value of Loss Function
        # and also the Gradient of Loss function with respect to all
        # trainable variables
        
        with tf.GradientTape() as lossTape:
            with tf.GradientTape() as g:
                g.watch(x)
                yHat = self.nnModel(x)
                
            dyHatdx = g.gradient(yHat, x)
            xInit = tf.convert_to_tensor(np.asarray([[1]]))
            yHatInitCondition = self.nnModel(xInit)
            
            # Loss Function
            # Here we are actually defining equations
            currentLoss = tf.reduce_sum((dyHatdx - 1/x)**2)/x.shape[0] + tf.reduce_sum((yHatInitCondition-0)**2)
            
        # Nudge the weights of neural network towards convergence (hopefully)
        grads = lossTape.gradient(currentLoss, self.nnModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nnModel.trainable_variables))
            
        return currentLoss
    
model = FirstOrderODE(inDim = 1, 
                     outDim = 1, 
                     nHiddenLayer = 10, 
                     nodePerLayer = 50, 
                     nIter = 500,
                     learningRate = 0.001,
                     batchSize = 50,
                     activation = Activation(swish),
                     kernelInitializer = tf.keras.initializers.he_uniform())

# Input Matrix (aka Training)
trainMin = 0.5
trainMax = 10
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
plt.scatter(xTest, np.log(xTest), label="Actual")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=2, mode="expand")

# Convergence History
plt.figure(1)
plt.plot(model.lossHistory)
plt.yscale('log')
plt.xlabel("Optimizer Iteration")
plt.ylabel("Loss Value")

plt.show()