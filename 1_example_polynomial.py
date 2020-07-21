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
    

model = ZeroOrderODE()

train_set = np.linspace(-20,20,101,dtype=np.float64)

model.solve(train_set)


plt.figure(0)
plt.scatter(train_set, model(tf.convert_to_tensor(np.array([train_set]).transpose())))
plt.scatter(train_set, train_set**2)
plt.figure(1)
plt.plot(model.lossHistory)
