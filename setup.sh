#!/bin/bash

# Setup script for PINN_Simple project (PyTorch-focused)
echo "🚀 Setting up PINN_Simple project..."
echo "================================================"

# Activate ai_research virtual environment
echo "📦 Activating ai_research virtual environment..."
source ~/venvs/ai_research/bin/activate

# Verify virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Error: Failed to activate ai_research virtual environment"
    echo "Please make sure the virtual environment exists at ~/venvs/ai_research"
    exit 1
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check PyTorch installation (main framework)
echo "🔥 Checking PyTorch installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'🚀 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'💾 CUDA version: {torch.version.cuda}')
    print(f'🧠 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Check TensorFlow installation (for tensorflow/ examples)
echo "🔧 Checking TensorFlow installation..."
python -c "
import tensorflow as tf
print(f'✅ TensorFlow version: {tf.__version__}')
print(f'🖥️  Device: {\"GPU\" if tf.config.list_physical_devices(\"GPU\") else \"CPU\"}')
"

# Check other dependencies
echo "📚 Checking other dependencies..."
python -c "
import numpy as np
import matplotlib
import jupyter
print(f'✅ NumPy version: {np.__version__}')
print(f'✅ Matplotlib version: {matplotlib.__version__}')
print(f'✅ Jupyter version: {jupyter.__version__}')
"

# Test PyTorch examples
echo "🧪 Testing PyTorch examples..."
echo "Running polynomial example test..."
python -c "
import sys
sys.path.append('.')
from pinn_base_model import PINNBaseModel, device
print(f'✅ PyTorch base model imports successfully')
print(f'✅ Device: {device}')
"

echo ""
echo "🎉 Setup complete!"
echo "================================================"
echo "📖 You can now run:"
echo ""
echo "🔥 PyTorch Examples (Main):"
echo "  python 1_example_polynomial.py"
echo "  python 2_example_trignometric.py"
echo "  python 3_example_1st_order_ode_1D.py"
echo ""
echo "📓 Jupyter Notebook:"
echo "  jupyter notebook pytorch_pinn_example.ipynb"
echo ""
echo "🧪 Test Suite:"
echo "  python test_pytorch_examples.py"
echo ""
echo "🔧 TensorFlow Examples (Legacy):"
echo "  cd tensorflow"
echo "  python 1_example_polynomial.py"
echo ""
echo "📚 Documentation:"
echo "  - Main: README.md"
echo "  - PyTorch: README_PyTorch.md"
echo "  - TensorFlow: tensorflow/README.md"
echo "================================================"
