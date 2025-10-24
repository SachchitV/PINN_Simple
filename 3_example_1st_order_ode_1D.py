# -*- coding: utf-8 -*-
"""
PyTorch version of first-order ODE example
Created based on TensorFlow version from Sun Sep 27 14:31:46 2020

@author: sachchit
"""

import matplotlib.pyplot as plt
import torch
import numpy as np

from pinn_base_model import PINNBaseModel, swish, bentIdentity, device

class FirstOrderODE(PINNBaseModel):
    # Here we are trying to Approximate df(x)/dx = dy/dx = 1/x
    # Silly, but helps in understanding implementation
    
    def train_step(self, x):
        # All magic happens here. Needs to be written Carefully
        # It have to return single value of Loss Function
        # and also the Gradient of Loss function with respect to all
        # trainable variables
        
        # Ensure x is on the correct device
        if not x.is_cuda and device.type == 'cuda':
            x = x.to(device)
        
        # Enable gradient computation for x
        x.requires_grad_(True)
        
        # Neural network prediction
        yHat = self.forward(x)
        
        # Compute derivative dy/dx using automatic differentiation
        dyHatdx = torch.autograd.grad(yHat, x, 
                                    grad_outputs=torch.ones_like(yHat), 
                                    create_graph=True, 
                                    retain_graph=True)[0]
        
        # Initial condition: y(1) = 0
        xInit = torch.tensor([[1.0]], dtype=torch.float64, device=device)
        yHatInitCondition = self.forward(xInit)
        
        # Loss Function
        # Here we are actually defining equations
        # PDE loss: dy/dx = 1/x
        pde_loss = torch.mean((dyHatdx - 1/x)**2)
        
        # Initial condition loss: y(1) = 0
        ic_loss = torch.mean((yHatInitCondition - 0)**2)
        
        # Total loss
        currentLoss = pde_loss + ic_loss
        
        return currentLoss

# Create model with same parameters as TensorFlow version
model = FirstOrderODE(inDim=1, 
                      outDim=1, 
                      nHiddenLayer=10, 
                      nodePerLayer=50, 
                      nIter=100,
                      learningRate=0.001,
                      batchSize=50,
                      activation=swish,
                      kernelInitializer='he_uniform')

# Move model to device
model = model.to(device)

# Input Matrix (aka Training)
trainMin = 0.5
trainMax = 10
nTrain = 50
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
y_true = np.log(xTest)  # The solution of dy/dx = 1/x is y = ln(x) + C, with y(1) = 0, so C = 0

# Comparison with actual function
plt.figure(0)
plt.scatter(xTest, y_pred, label="Neural Net")
plt.scatter(xTest, y_true, label="Actual")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=2, mode="expand")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network vs Actual Function: dy/dx = 1/x, y(1) = 0')
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

# Additional analysis: Check derivative
print(f"\nDerivative Analysis:")
x_test_deriv = torch.tensor([[2.0], [3.0], [5.0]], dtype=torch.float64, device=device)
x_test_deriv.requires_grad_(True)
y_test = model(x_test_deriv)
dy_dx = torch.autograd.grad(y_test, x_test_deriv, 
                           grad_outputs=torch.ones_like(y_test), 
                           create_graph=True)[0]

print(f"x values: {x_test_deriv.cpu().numpy().flatten()}")
print(f"Predicted dy/dx: {dy_dx.cpu().numpy().flatten()}")
print(f"True dy/dx (1/x): {1/x_test_deriv.cpu().numpy().flatten()}")
print(f"Derivative error: {torch.abs(dy_dx - 1/x_test_deriv).cpu().numpy().flatten()}")
