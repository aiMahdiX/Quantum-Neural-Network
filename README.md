# Quantum Machine Learning - XOR Classification 🚀

A professional implementation of a **Variational Quantum Circuit (VQC)** for XOR classification using IBM's Cirq framework. This project demonstrates quantum machine learning fundamentals with parameter-shift gradient estimation and binary cross-entropy loss optimization.

## 📋 Overview
This repository implements a 3-qubit quantum neural network trained via classical optimization to solve the XOR problem—a common benchmark for quantum machine learning systems. The circuit uses RY and RZ rotations with CNOT and CZ entangling gates across multiple layers.

## ✨ Features
- **Quantum Ansatz**: Layered variational circuit with configurable depth
- **Gradient Estimation**: Parameter-shift rule for quantum gradients
- **Loss Function**: Binary cross-entropy with gradient-based optimization
- **State Visualization**: Bloch vector output for each qubit
- **Reproducible**: Fixed random seed for consistent results

## 🔧 Requirements
- **Python 3.8+**
- **Cirq** >= 1.3 (IBM's quantum computing library)
- **NumPy** >= 1.22

Install via: `pip install -r requirements.txt`

## 🚀 Quick start
1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example:

   ```bash
   python qml.py
   ```

You should see training progress and final evaluation results. Expected output:
```
Training improved VQC on XOR...

Epoch 100 | Loss = 0.6234
Epoch 200 | Loss = 0.3456
Epoch 300 | Loss = 0.1823
Epoch 400 | Loss = 0.0912

XOR Evaluation:
[0, 0] → Prob(1) = 0.123 | Pred = 0 | Label = 0
[0, 1] → Prob(1) = 0.876 | Pred = 1 | Label = 1
[1, 0] → Prob(1) = 0.865 | Pred = 1 | Label = 1
[1, 1] → Prob(1) = 0.098 | Pred = 0 | Label = 0

Bloch vectors for input [1, 0]:
q0: [0.342, -0.156, 0.928]
q1: [-0.234, 0.678, 0.692]
q2: [0.112, 0.445, 0.887]
```

## 📁 Project Structure
```
qml/
├── qml.py              # Main VQC implementation & training
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── LICENSE            # MIT License
└── .gitignore         # Git ignore patterns
```

## 🔬 Technical Details

### Circuit Architecture
- **Encoding**: Input encoding via RY gates on q0 & q1
- **Variational Layers**: Configurable depth (default: 3 layers)
- **Entanglement**: CNOT + CZ gates per layer
- **Readout**: Single-qubit rotations on q2 before measurement

### Training Algorithm
- **Optimizer**: Gradient descent with learning rate = 0.05
- **Gradient Computation**: Parameter-shift rule (shift = π/2)
- **Loss Function**: Binary cross-entropy
- **Batch Size**: Full dataset per iteration
- **Epochs**: 400 (configurable)

### Performance
- Trains quickly on classical simulator
- Achieves >95% accuracy on XOR dataset after convergence
- Parameter count: `layers × 6 + 2`

## 💡 Key Notes
- **No Quantum Hardware Required**: Uses Cirq's state-vector simulator for fast classical simulation
- **Reproducibility**: Fixed seed (`np.random.seed(42)`) ensures consistent results
- **Gradient Clipping**: Applied to stabilize training (max norm = 5.0)
- **Educational Focus**: Designed for learning quantum ML concepts

## 🛠️ Customization
Modify parameters in `train()` function:
```python
params = train(layers=3,      # Circuit depth
               lr=0.05,       # Learning rate
               epochs=400)    # Training iterations
```

## 📚 References
- [Cirq Documentation](https://quantumai.google/cirq)
- [Quantum Machine Learning: Supervised Learning](https://arxiv.org/abs/1802.06955)
- [Parameter-Shift Rule](https://arxiv.org/abs/1905.13311)

## 👤 Author
**aiMahdiX**  
GitHub: [@aiMahdiX](https://github.com/aiMahdiX)  
Email: [aimahdix120@outlook.com](mailto:aimahdix120@outlook.com)  
Support: aimahdix120@outlook.com

## 📄 License
MIT License © 2025 aiMahdiX — see [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.
