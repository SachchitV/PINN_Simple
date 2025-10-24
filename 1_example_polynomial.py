# -*- coding: utf-8 -*-
"""
PyTorch version of polynomial example
Created based on TensorFlow version from Wed Jul 22 00:26:55 2020

@author: sachchit
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

from pinn_base_model import PINNBaseModel, swish, bentIdentity, device

class ZeroOrderODE(PINNBaseModel):
    # Here we are trying to Approximate f(x) = y = x^2
    # Silly, but helps in understanding implementation
    
    def train_step(self, x):
        # All magic happens here. Needs to be written Carefully
        # It have to return single value of Loss Function
        # and also the Gradient of Loss function with respect to all
        # trainable variables
        
        # Ensure x is on the correct device
        if not x.is_cuda and device.type == 'cuda':
            x = x.to(device)
        
        # Neural network prediction
        yHat = self.forward(x)
        
        # Loss Function
        # Here we are actually defining equations
        currentLoss = torch.mean((yHat - x**2)**2)
        
        return currentLoss

# Create model with same parameters as TensorFlow version
model = ZeroOrderODE(inDim=1, 
                     outDim=1, 
                     nHiddenLayer=10, 
                     nodePerLayer=50, 
                     nIter=500,
                     learningRate=0.001,
                     batchSize=20,
                     activation=swish,
                     kernelInitializer='he_uniform')

# Move model to device
model = model.to(device)

# Input Matrix (aka Training)
trainMin = -20
trainMax = 20
nTrain = 20
scale = trainMax - trainMin
trainSet = trainMin + scale * torch.tensor(np.random.rand(nTrain, 1), dtype=torch.float64, device=device)

# Train the model
model.solve(trainSet)

# Testing Set
nTest = 50
scale = trainMax - trainMin
testSet = trainMin + scale * np.random.rand(nTest, 1)
testSet_tensor = torch.tensor(testSet, dtype=torch.float64, device=device)
xTest = testSet[:, 0]

# Get predictions
model.eval()
with torch.no_grad():
    predictions = model(testSet_tensor)

# Convert to numpy for plotting
y_pred = predictions.cpu().numpy().flatten()
y_true = xTest**2

# Comparison with actual function
plt.figure(0)
plt.scatter(xTest, y_pred, label="Neural Net")
plt.scatter(xTest, y_true, label="Actual")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=2, mode="expand")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network vs Actual Function: f(x) = x²')
plt.grid(True, alpha=0.3)

# Convergence History
plt.figure(1)
plt.plot(model.lossHistory)
plt.yscale('log')
plt.xlabel("Optimizer Iteration")
plt.ylabel("Loss Value")
plt.title('Training Convergence')
plt.grid(True, alpha=0.3)

plt.show()

# Print performance metrics
mse = np.mean((y_pred - y_true)**2)
mae = np.mean(np.abs(y_pred - y_true))
print(f"\nPerformance Metrics:")
print(f"Mean Squared Error: {mse:.6e}")
print(f"Mean Absolute Error: {mae:.6e}")
print(f"Final Training Loss: {model.lossHistory[-1]:.6e}")
