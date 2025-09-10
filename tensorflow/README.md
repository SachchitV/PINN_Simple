# TensorFlow Physics-Informed Neural Networks (PINNs)

This directory contains the original TensorFlow implementation of Physics-Informed Neural Networks for solving various mathematical problems.

## 📁 Files Overview

### Core Files
- **`pinn_base_model.py`** - TensorFlow base model class for PINNs
- **`1_example_polynomial.py`** - Learn f(x) = x²
- **`2_example_trignometric.py`** - Learn f(x) = x + sin(4πx)
- **`3_example_1st_order_ode_1D.py`** - Solve dy/dx = 1/x with y(1) = 0

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate your virtual environment
source ~/venvs/ai_research/bin/activate

# Install TensorFlow dependencies
pip install tensorflow>=2.0.0 numpy matplotlib
```

### 2. Run Examples
```bash
# Navigate to tensorflow directory
cd tensorflow

# Run individual examples
python 1_example_polynomial.py
python 2_example_trignometric.py
python 3_example_1st_order_ode_1D.py
```

## 🔧 Key Features

### TensorFlow Advantages
- **Graph Compilation**: `@tf.function` for optimized execution
- **Automatic Differentiation**: `tf.GradientTape()` for gradients
- **GPU Acceleration**: Automatic device placement
- **Keras Integration**: Easy model building with `tf.keras`

### Architecture
- **Sequential Model**: Uses `tf.keras.Sequential()` for network construction
- **Custom Activations**: Swish and Bent Identity functions
- **Weight Initialization**: He uniform initialization
- **Adam Optimizer**: Adaptive learning rate optimization

## 📊 Examples

### 1. Polynomial Function (f(x) = x²)
- **Domain**: [-20, 20]
- **Training Points**: 20
- **Architecture**: 10 hidden layers, 50 nodes each
- **Purpose**: Learn a simple quadratic function

### 2. Trigonometric Function (f(x) = x + sin(4πx))
- **Domain**: [0, 1]
- **Training Points**: 50
- **Architecture**: 10 hidden layers, 50 nodes each
- **Purpose**: Learn a function with oscillatory behavior

### 3. First-Order ODE (dy/dx = 1/x)
- **Domain**: [0.5, 10]
- **Training Points**: 50
- **Architecture**: 10 hidden layers, 50 nodes each
- **Purpose**: Solve differential equation with initial condition y(1) = 0

## 🏗️ Implementation Details

### Base Model (`PINNBaseModel`)
```python
class PINNBaseModel(object):
    def __init__(self, inDim, outDim, nHiddenLayer, nodePerLayer, ...):
        # Build sequential model
        # Set up optimizer
        # Initialize training parameters
    
    def train_step(self, x):
        # Must be implemented in child classes
        # Define loss function and physics constraints
    
    def solve(self, trainSet):
        # Main training loop with batching
```

### Example Implementation
```python
class ZeroOrderODE(PINNBaseModel):
    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as lossTape:
            yHat = self.nnModel(x)
            currentLoss = tf.reduce_sum((yHat - x**2)**2)/x.shape[0]
        
        grads = lossTape.gradient(currentLoss, self.nnModel.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nnModel.trainable_variables))
        return currentLoss
```

## 🔄 Training Process

1. **Data Generation**: Create training points in specified domain
2. **Model Initialization**: Build neural network with specified architecture
3. **Training Loop**: 
   - Forward pass through network
   - Compute loss function (physics constraint)
   - Backward pass (automatic differentiation)
   - Update weights (Adam optimizer)
4. **Convergence**: Track loss history and select best weights

## 📈 Performance

### Training Characteristics
- **Convergence**: Typically converges within 500 epochs
- **Loss Reduction**: Logarithmic decrease in loss over training
- **Accuracy**: High accuracy on test data
- **Stability**: Consistent results across runs

### Memory Usage
- **Efficient**: TensorFlow's graph optimization
- **GPU**: Automatic GPU memory management
- **Batching**: Configurable batch sizes for memory control

## 🛠️ Customization

### Modifying Examples
1. **Change Function**: Modify the target function in `train_step()`
2. **Adjust Domain**: Change `trainMin` and `trainMax` values
3. **Architecture**: Modify `nHiddenLayer` and `nodePerLayer`
4. **Training**: Adjust `nIter`, `learningRate`, `batchSize`

### Adding New Examples
1. Create new class inheriting from `PINNBaseModel`
2. Implement `train_step()` with your physics constraints
3. Define loss function using TensorFlow operations
4. Run training with `model.solve(trainSet)`

## 🐛 Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch size if out of memory
2. **Convergence**: Adjust learning rate or increase iterations
3. **Gradient Issues**: Check loss function implementation
4. **Device Placement**: TensorFlow handles this automatically

### Debug Tips
- Use `tf.print()` for debugging output
- Check gradient norms with `tf.norm()`
- Visualize loss history with matplotlib
- Use `tf.debugging.check_numerics()` for NaN detection

## 📚 References

- Original implementation by sachchit
- TensorFlow documentation: https://www.tensorflow.org/
- Physics-Informed Neural Networks: https://www.sciencedirect.com/science/article/pii/S0021999118307125

## 📄 License

Same license as the main project.
