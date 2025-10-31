# 2D Ising Model Coarse-Graining with Neural Networks

## Project Overview
This project implements a neural network that learns to perform coarse-graining operations on 2D Ising model configurations. The goal is to demonstrate how deep learning can reduce degrees of freedom in physical systems while preserving essential macroscopic features.

## Project Structure
```
Project/
├── data-gen.py            # Data generation script (Ising configurations)
├── simulator.py           # Monte Carlo Ising model simulator
├── models.py              # Neural network architectures (CNN, MLP)
├── train_model.py         # Complete training and evaluation pipeline
├── save_results.py        # Results saving utilities
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Generated Ising configurations (.npz, .pkl)
├── models/                # Trained neural network models (.pth)
├── figures/               # Visualization outputs (.png)
└── results/               # Structured results (JSON, CSV, pickle, txt)
```

## Physics Background
The 2D Ising model is a mathematical model used to describe ferromagnetism in statistical mechanics. Each site on a 2D lattice has a spin that can be either +1 (up) or -1 (down). The model exhibits a phase transition at the critical temperature T_c ≈ 2.269.

## Implementation Details

### 1. Data Generation (`data-gen.py` + `simulator.py`)
- **Monte Carlo Simulation**: Uses the Metropolis algorithm to generate equilibrium configurations
- **Temperature Range**: Samples across low, critical, and high temperature regimes (8 temperatures)
- **Coarse-Graining**: Creates 8×8 targets from 16×16 configurations using block averaging
- **Periodic Boundary Conditions**: Ensures proper lattice physics
- **Dataset**: 1,600 configurations (200 per temperature)

### 2. Neural Network Architectures (`models.py`)
- **CNN Model**: Convolutional layers with spatial awareness (~10K parameters)
  - 3 convolutional layers with batch normalization
  - Adaptive pooling for size reduction
  - Tanh activation for spin-like outputs in [-1, 1]
- **MLP Model**: Fully connected alternative for comparison (~306K parameters)
  - Dense layers with dropout regularization
  - Reshaping for spatial output

### 3. Training Pipeline (`train_model.py`)
- **Data Splitting**: Train (70%) / Validation (10%) / Test (20%) split
- **Loss Function**: Mean Squared Error for continuous outputs
- **Optimization**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Result Saving**: Automatically saves metrics, history, and analysis (`save_results.py`)

## Key Features

### Human-Written Code Style
- **Clear Comments**: Every function and complex operation explained
- **Modular Design**: Separate files for different functionalities
- **Error Handling**: Graceful handling of missing dependencies
- **Realistic Implementation**: Includes practical considerations like GPU usage

### Physical Validation
- **Magnetization Preservation**: Ensures macroscopic properties are maintained
- **Spatial Correlation Analysis**: Quantifies domain structure preservation
- **Temperature Dependence**: Tests across different physical regimes

### Comprehensive Visualization
- **Configuration Comparisons**: Side-by-side original, target, and predicted
- **Training Curves**: Loss evolution and convergence monitoring
- **Error Analysis**: Spatial error maps and statistical distributions
- **Physical Properties**: Magnetization and correlation preservation

## Installation and Usage

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib scikit-learn numba
```
or use `uv sync`, if you have the uv package installed


### Running the Project

1. **Generate Data**:
   ```bash
   python data-gen.py
   ```
   This creates Ising model configurations at various temperatures and saves them to `data/`.

2. **Train Models**:
   ```bash
   python train_model.py
   ```
   This trains both CNN and MLP models, compares them, and generates:
   - Trained models in `models/`
   - Visualizations in `figures/`
   - Structured results in `results/`

3. **Alternative**: Run individual components:
   ```bash
   # Test simulator only
   python simulator.py
   
   # Test neural network components
   python models.py
   ```

### Expected Outputs
- **Data files**: `data/ising_data_16x16_1600configs.npz`
- **Trained models**: `models/best_cnn_model.pth` and `models/best_mlp_model.pth`
- **Visualizations**: Various plots in `figures/` directory
- **Results data**: Structured results in `results/` directory
- **Analysis**: Comprehensive performance metrics and report

### Using the Results

After training, the `results/` directory contains structured output files:

**Quick Review** (Human-readable text files):
```bash
cat results/model_comparison.txt    # Side-by-side model performance
cat results/configuration.txt        # Training setup details
```

**Detailed Analysis** (Load in Python):
```python
import json
import pandas as pd
import pickle

# Load metrics summary
with open('results/metrics_summary.json') as f:
    metrics = json.load(f)
print(f"Best model: {metrics['best_model']}")
print(f"CNN MSE: {metrics['models']['CNN']['mse']:.6f}")

# Load and plot training history
history = pd.read_csv('results/training_history.csv')
history.plot(x='epoch', y=['cnn_val_loss', 'mlp_val_loss'])

# Load complete results for custom analysis
with open('results/detailed_results.pkl', 'rb') as f:
    results = pickle.load(f)
    cnn_history = results['cnn_training_history']
    metadata = results['metadata']
```

**Available Result Files**:
- `metrics_summary.json` - Overall performance metrics (MSE, MAE, RMSE)
- `training_history.csv` - Epoch-by-epoch training/validation losses
- `model_comparison.txt` - Human-readable performance comparison
- `configuration.txt` - Dataset and training configuration
- `detailed_results.pkl` - Complete results for custom analysis
- `REPORT.md` - Comprehensive project report

## Results Interpretation

### Model Performance
- **MSE/MAE**: Quantitative measures of prediction accuracy
- **Spatial Correlation**: How well domain structures are preserved
- **Magnetization RMSE**: Conservation of macroscopic properties

### Physical Insights
- **Critical Behavior**: How well the model handles near-critical fluctuations
- **Domain Preservation**: Maintenance of magnetic domain structures
- **Scale Separation**: Effective reduction from 256 to 64 degrees of freedom

## Educational Value

### Physics Concepts
- Phase transitions in condensed matter
- Monte Carlo methods in statistical mechanics
- Coarse-graining and renormalization ideas
- Order parameters and critical phenomena

### Machine Learning Concepts
- Convolutional neural networks for spatial data
- Loss functions for regression problems
- Overfitting and regularization techniques
- Model comparison and validation

### Computational Physics
- Efficient simulation of physical systems
- Data preprocessing for ML applications
- Visualization of high-dimensional data
- Performance optimization considerations

## Extensions and Future Work

### Model Improvements
- **Attention Mechanisms**: Focus on critical regions
- **Residual Connections**: Preserve fine-scale information
- **Variational Approaches**: Learn probability distributions
- **Physics-Informed Loss**: Incorporate energy constraints

### Physical Extensions
- **3D Ising Model**: Extension to higher dimensions
- **Different Lattices**: Triangular, hexagonal geometries
- **External Fields**: Non-zero magnetic field effects
- **Quantum Models**: Extend to quantum spin systems

### Analysis Enhancements
- **Finite-Size Scaling**: Study system size effects
- **Critical Exponents**: Extract universal properties
- **Real-Space RG**: Compare to theoretical renormalization
- **Transfer Learning**: Apply to other physical systems

## Acknowledgments
This project demonstrates the intersection of machine learning and computational physics, showing how neural networks can learn to perform sophisticated physical transformations that traditionally require analytical understanding.

The implementation follows best practices for both machine learning and computational physics, making it suitable for educational purposes while maintaining research-quality standards.
