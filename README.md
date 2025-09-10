# PINN_Simple

A comprehensive implementation of Physics-Informed Neural Networks (PINNs) for approximating 1D functions and solving ordinary differential equations. This project provides both TensorFlow and PyTorch implementations with a foundational framework to understand and explore PINNs through basic examples.

## 🏗️ Project Structure

```
PINN_Simple/
├── tensorflow/                 # TensorFlow implementation
│   ├── pinn_base_model.py     # TensorFlow base model
│   ├── 1_example_polynomial.py
│   ├── 2_example_trignometric.py
│   ├── 3_example_1st_order_ode_1D.py
│   └── README.md              # TensorFlow documentation
├── pinn_base_model.py         # PyTorch base model (main)
├── 1_example_polynomial.py    # PyTorch examples (main)
├── 2_example_trignometric.py
├── 3_example_1st_order_ode_1D.py
├── pytorch_pinn_example.ipynb # Comprehensive Jupyter notebook
├── test_pytorch_examples.py   # Test suite
├── README_PyTorch.md          # PyTorch documentation
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### PyTorch Version (Recommended)
```bash
# Setup environment
source ~/venvs/ai_research/bin/activate
pip install -r requirements.txt

# Run examples
python 1_example_polynomial.py
python 2_example_trignometric.py
python 3_example_1st_order_ode_1D.py

# Or use the Jupyter notebook
jupyter notebook pytorch_pinn_example.ipynb
```

### TensorFlow Version
```bash
# Navigate to tensorflow directory
cd tensorflow

# Run examples
python 1_example_polynomial.py
python 2_example_trignometric.py
python 3_example_1st_order_ode_1D.py
```

## 📊 Examples

### 1. Polynomial Function - f(x) = x²
- **Domain**: [-20, 20]
- **Purpose**: Learn a simple quadratic function
- **Complexity**: Basic

### 2. Trigonometric Function - f(x) = x + sin(4πx)
- **Domain**: [0, 1]
- **Purpose**: Learn a function with oscillatory behavior
- **Complexity**: Medium

### 3. First-order ODE - dy/dx = 1/x
- **Domain**: [0.5, 10]
- **Initial Condition**: y(1) = 0
- **Purpose**: Solve differential equation
- **Complexity**: Advanced

## 🔧 Framework Comparison

| Feature | TensorFlow | PyTorch |
|---------|------------|---------|
| **Ease of Use** | Good | Excellent |
| **Debugging** | Limited | Full Python |
| **Performance** | Fast | Fast |
| **Flexibility** | Static Graphs | Dynamic Graphs |
| **Learning Curve** | Steep | Gentle |

## 📚 Documentation

- **Main PyTorch**: See `README_PyTorch.md` for detailed PyTorch documentation
- **TensorFlow**: See `tensorflow/README.md` for TensorFlow-specific docs
- **Jupyter Notebook**: `pytorch_pinn_example.ipynb` for interactive exploration

## 🧪 Testing

Run the test suite to verify all examples work:
```bash
python test_pytorch_examples.py
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+ (for main examples)
- TensorFlow 2.0+ (for tensorflow/ examples)
- NumPy, Matplotlib, Jupyter

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation
https://maziarraissi.github.io/PINNs/

    @article{raissi2019physics,
      title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
      journal={Journal of Computational Physics},
      volume={378},
      pages={686--707},
      year={2019},
      publisher={Elsevier}
    }

    @article{raissi2017physicsI,
      title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10561},
      year={2017}
    }

    @article{raissi2017physicsII,
      title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10566},
      year={2017}
    }
