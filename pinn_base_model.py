# -*- coding: utf-8 -*-
"""
PyTorch version of PINN Base Model
Created based on TensorFlow version from Sat Feb  8 15:19:02 2020

@author: sachchit
"""

# Importing Basic Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU configuration
if torch.cuda.is_available():
    print(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    # Enable memory growth for PyTorch
    torch.backends.cudnn.benchmark = True
else:
    print("No GPU detected, using CPU")

# Set default dtype
torch.set_default_dtype(torch.float64)

# Defining BentIdentity Function which will be used as activation function
# to avoid saturation
def bentIdentity(x):
    return x + (torch.sqrt(x*x+1)-1)/2

def swish(x):
    return x * torch.sigmoid(x)

class PINNBaseModel(nn.Module):
    """
    PyTorch version of PINN Base Model
    This is the base class of solver which handles the backend of solving 
    Equations. User have to create child class of this class and at least 
    implement train_step() method.
    """
    
    def __init__(self, 
                 inDim=1, 
                 outDim=1, 
                 nHiddenLayer=5, 
                 nodePerLayer=10,
                 nIter=1000,
                 learningRate=0.001,
                 batchSize=1001,
                 activation=swish,
                 kernelInitializer='he_uniform'):
        super(PINNBaseModel, self).__init__()
        
        # Store hyperparameters
        self.nLayers = nHiddenLayer
        self.nIter = nIter
        self.learningRate = learningRate
        self.batchSize = batchSize
        
        # Create neural network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(inDim, nodePerLayer))
        
        # Hidden layers
        for _ in range(nHiddenLayer - 1):
            self.layers.append(nn.Linear(nodePerLayer, nodePerLayer))
        
        # Output layer
        self.layers.append(nn.Linear(nodePerLayer, outDim))
        
        self.activation = activation
        
        # Initialize weights
        self._initialize_weights(kernelInitializer)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        
        # Training history
        self.lossHistory = []
        self.minLoss = float('inf')
        self.minLossWeights = None
        
    def _initialize_weights(self, init_type):
        """Initialize network weights"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_type == 'he_uniform':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif init_type == 'he_normal':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                else:
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x
    
    def __call__(self, x):
        """Return output of neural network with x input"""
        return self.forward(x)
    
    def train_step(self, x):
        """
        This method should be implemented in child classes.
        It defines the loss function and physics constraints.
        """
        raise NotImplementedError("train_step must be implemented in child class")
    
    def solve(self, trainSet):
        """
        Main training loop
        """
        self.train()  # Set to training mode
        
        # Convert to PyTorch tensor if needed
        if not isinstance(trainSet, torch.Tensor):
            trainSet = torch.tensor(trainSet, dtype=torch.float64, device=device)
        
        # Ensure model is on the correct device
        self.to(device)
        
        # Create data loader
        dataset = TensorDataset(trainSet)
        dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        
        print("Starting training...")
        print("=" * 50)
        
        for epoch in range(self.nIter):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (batch_data,) in enumerate(dataloader):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Compute loss
                loss = self.train_step(batch_data)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch: {epoch+1}/{self.nIter}, "
                          f"Batch: {batch_idx+1}/{len(dataloader)}, "
                          f"Loss: {loss.item():.6e}")
            
            # Average loss for this epoch
            avg_loss = epoch_loss / num_batches
            self.lossHistory.append(avg_loss)
            
            # Keep track of best weights
            if avg_loss < self.minLoss:
                self.minLoss = avg_loss
                self.minLossWeights = {name: param.clone() for name, param in self.named_parameters()}
        
        # Load best weights
        if self.minLossWeights is not None:
            for name, param in self.named_parameters():
                param.data = self.minLossWeights[name]
        
        print("=" * 50)
        print(f"Training completed!")
        print(f"Minimum Loss: {self.minLoss:.6e}")
        print("=" * 50)
    
    def create_batch(self, trainSet):
        """Create batches from training set (for compatibility with TensorFlow version)"""
        if not isinstance(trainSet, torch.Tensor):
            trainSet = torch.tensor(trainSet, dtype=torch.float64, device=device)
        
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
        """Print weights for debugging (for compatibility with TensorFlow version)"""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i}:")
                print(f"  Weight: {layer.weight.data}")
                print(f"  Bias: {layer.bias.data}")
    
    def save_model(self, filepath):
        """Save model with proper device handling"""
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model with proper device handling"""
        self.load_state_dict(torch.load(filepath, map_location=device))
        self.to(device)
        self.eval()
        print(f"Model loaded from {filepath} and moved to {device}")