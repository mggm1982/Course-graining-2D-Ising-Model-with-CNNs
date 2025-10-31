import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import (
    IsingDataset, CoarseGrainingCNN, CoarseGrainingMLP, 
    CoarseGrainingTrainer, evaluate_model
)
from save_results import save_training_results


def load_data():
    """Load the generated Ising model data"""
    data_file = 'data/ising_data_16x16_1600configs.npz'
    metadata_file = 'data/metadata.pkl'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError("Data file not found. Please run main.py first to generate data.")
    
    print("Loading Ising model data...")
    data = np.load(data_file)
    fine_configs = data['fine_configs']
    coarse_targets = data['coarse_targets']
    temperatures = data['temperatures']
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {len(fine_configs)} configurations")
    print(f"Fine-grained shape: {fine_configs.shape}")
    print(f"Coarse-grained shape: {coarse_targets.shape}")
    
    return fine_configs, coarse_targets, temperatures, metadata

def prepare_data_loaders(fine_configs, coarse_targets, batch_size=32, 
                        test_size=0.2, val_size=0.1):
    """
    Prepare PyTorch data loaders for training.
    
    Returns:
    tuple: (train_loader, val_loader, test_loader, test_data)
    """
    print("\nPreparing data loaders...")
    
    # Split the data
    train_val_fine, test_fine, train_val_coarse, test_coarse = train_test_split(
        fine_configs, coarse_targets, test_size=test_size, random_state=42,
        stratify=None  # Can't stratify continuous data
    )
    
    # Further split training data into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    train_fine, val_fine, train_coarse, val_coarse = train_test_split(
        train_val_fine, train_val_coarse, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"Training samples: {len(train_fine)}")
    print(f"Validation samples: {len(val_fine)}")
    print(f"Test samples: {len(test_fine)}")
    
    # Create datasets
    train_dataset = IsingDataset(train_fine, train_coarse)
    val_dataset = IsingDataset(val_fine, val_coarse)
    test_dataset = IsingDataset(test_fine, test_coarse)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store test data for detailed analysis
    test_data = {
        'fine': test_fine,
        'coarse': test_coarse
    }
    
    return train_loader, val_loader, test_loader, test_data

def train_cnn_model(train_loader, val_loader, num_epochs=100, lr=1e-3):
    """Train the CNN model for coarse-graining"""
    print("\n" + "=" * 60)
    print("TRAINING CNN MODEL")
    print("=" * 60)
    
    # Create model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CoarseGrainingCNN(input_size=16, output_size=8)
    trainer = CoarseGrainingTrainer(model, device=device, model_name='cnn')
    
    print(f"Model architecture: CNN")
    print(f"Input size: 16x16")
    print(f"Output size: 8x8")
    print(f"Device: {device}")
    
    # Train the model
    trainer.train_model(
        train_loader, val_loader, 
        num_epochs=num_epochs, 
        lr=lr,
        patience=15
    )
    
    # Plot training history
    fig = trainer.plot_training_history()
    plt.savefig('figures/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return trainer

def train_mlp_model(train_loader, val_loader, num_epochs=100, lr=1e-3):
    """Train the MLP model for comparison"""
    print("\n" + "=" * 60)
    print("TRAINING MLP MODEL")
    print("=" * 60)
    
    # Create model and trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CoarseGrainingMLP(input_size=16, output_size=8)
    trainer = CoarseGrainingTrainer(model, device=device, model_name='mlp')
    
    print(f"Model architecture: MLP")
    print(f"Input size: 16x16 = 256")
    print(f"Output size: 8x8 = 64")
    print(f"Device: {device}")
    
    # Train the model
    trainer.train_model(
        train_loader, val_loader, 
        num_epochs=num_epochs, 
        lr=lr,
        patience=15
    )
    
    # Plot training history
    fig = trainer.plot_training_history()
    plt.savefig('figures/mlp_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return trainer

def comprehensive_evaluation(cnn_trainer, mlp_trainer, test_loader, test_data):
    """
    Perform comprehensive evaluation of both models.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluate CNN
    print("Evaluating CNN:")
    cnn_results = evaluate_model(cnn_trainer, test_loader, device)
    
    print("\nEvaluating MLP:")
    mlp_results = evaluate_model(mlp_trainer, test_loader, device)
    
    # Compare models
    print("\n" + "-" * 40)
    print("MODEL COMPARISON")
    print("-" * 40)
    print(f"CNN - MSE: {cnn_results['mse']:.6f}, MAE: {cnn_results['mae']:.6f}")
    print(f"MLP - MSE: {mlp_results['mse']:.6f}, MAE: {mlp_results['mae']:.6f}")
    
    better_model = "CNN" if cnn_results['mse'] < mlp_results['mse'] else "MLP"
    print(f"Better model: {better_model}")
    
    # Visualize sample predictions
    visualize_predictions(cnn_results, mlp_results, "CNN vs MLP")
    
    return cnn_results, mlp_results

def visualize_predictions(cnn_results, mlp_results, title):
    """
    Create detailed visualizations of model predictions.
    """
    n_samples = min(5, len(cnn_results['sample_inputs']))
    
    for i in range(n_samples):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Original fine-grained configuration
        fine_config = cnn_results['sample_inputs'][i][0]  # Remove channel dimension
        target_config = cnn_results['sample_targets'][i][0]
        cnn_pred = cnn_results['sample_predictions'][i][0]
        mlp_pred = mlp_results['sample_predictions'][i][0]
        
        # Calculate magnetization to identify phase
        magnetization = np.mean(fine_config)
        if magnetization > 0.7:
            phase = "Ordered ↑"
        elif magnetization < -0.7:
            phase = "Ordered ↓"
        elif abs(magnetization) < 0.3:
            phase = "Critical/Disordered"
        else:
            phase = f"Mixed (M={magnetization:.2f})"
        
        # First row: CNN predictions
        axes[0, 0].imshow(fine_config, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title(f'Fine-grained (16x16)\nM={magnetization:.3f}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_config, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title('Target (8x8)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cnn_pred, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 2].set_title('CNN Prediction')
        axes[0, 2].axis('off')
        
        # CNN error map
        cnn_error = np.abs(cnn_pred - target_config)
        cnn_mse = np.mean(cnn_error**2)
        im_cnn = axes[0, 3].imshow(cnn_error, cmap='Reds', vmin=0, vmax=2)
        axes[0, 3].set_title(f'CNN Error\nMSE={cnn_mse:.4f}')
        axes[0, 3].axis('off')
        plt.colorbar(im_cnn, ax=axes[0, 3], shrink=0.6)
        
        # Second row: MLP predictions
        axes[1, 0].imshow(fine_config, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title(f'Fine-grained (16x16)\nM={magnetization:.3f}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(target_config, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title('Target (8x8)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(mlp_pred, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 2].set_title('MLP Prediction')
        axes[1, 2].axis('off')
        
        # MLP error map
        mlp_error = np.abs(mlp_pred - target_config)
        mlp_mse = np.mean(mlp_error**2)
        im_mlp = axes[1, 3].imshow(mlp_error, cmap='Reds', vmin=0, vmax=2)
        axes[1, 3].set_title(f'MLP Error\nMSE={mlp_mse:.4f}')
        axes[1, 3].axis('off')
        plt.colorbar(im_mlp, ax=axes[1, 3], shrink=0.6)
        
        plt.suptitle(f'{title} - Sample {i+1}: {phase}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'figures/prediction_comparison_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()

def analyze_physical_properties(trainer, test_data):
    """
    Analyze how well the model preserves physical properties.
    """
    print("\n" + "=" * 60)
    print("PHYSICAL PROPERTIES ANALYSIS")
    print("=" * 60)
    
    fine_configs = test_data['fine']
    coarse_targets = test_data['coarse']
    
    # Get model predictions
    predictions = trainer.predict(fine_configs)
    predictions = predictions.squeeze()  # Remove channel dimension if present
    
    # Calculate magnetizations
    fine_magnetizations = [np.mean(config) for config in fine_configs]
    target_magnetizations = [np.mean(config) for config in coarse_targets]
    pred_magnetizations = [np.mean(config) for config in predictions]
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Magnetization preservation
    axes[0].scatter(fine_magnetizations, target_magnetizations, alpha=0.6, 
                   label='Target (Block Avg)', s=20)
    axes[0].scatter(fine_magnetizations, pred_magnetizations, alpha=0.6, 
                   label='NN Prediction', s=20)
    axes[0].plot([-1, 1], [-1, 1], 'r--', alpha=0.7, label='Perfect')
    axes[0].set_xlabel('Fine-grained Magnetization')
    axes[0].set_ylabel('Coarse-grained Magnetization')
    axes[0].set_title('Magnetization Preservation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error in magnetization
    mag_error = np.array(pred_magnetizations) - np.array(target_magnetizations)
    axes[1].hist(mag_error, bins=30, alpha=0.7, density=True)
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Magnetization Error (Predicted - Target)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Magnetization Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Domain preservation (correlation analysis)
    correlations = []
    for i in range(len(predictions)):
        corr = np.corrcoef(coarse_targets[i].flatten(), predictions[i].flatten())[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    axes[2].hist(correlations, bins=30, alpha=0.7, density=True)
    axes[2].axvline(np.mean(correlations), color='red', linestyle='--', 
                   alpha=0.7, label=f'Mean: {np.mean(correlations):.3f}')
    axes[2].set_xlabel('Spatial Correlation')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Spatial Pattern Preservation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/physical_properties_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Magnetization RMSE: {np.sqrt(np.mean(mag_error**2)):.6f}")
    print(f"Mean spatial correlation: {np.mean(correlations):.6f}")
    print(f"Correlation std: {np.std(correlations):.6f}")

def main():
    """Main training and evaluation pipeline"""
    
    print("2D Ising Model Coarse-Graining: Neural Network Training")
    print("=" * 60)
    
    # Load data
    try:
        fine_configs, coarse_targets, temperatures, metadata = load_data()
    except FileNotFoundError:
        print("Data not found. Please run main.py first to generate the data.")
        return
    
    # Prepare data loaders
    train_loader, val_loader, test_loader, test_data = prepare_data_loaders(
        fine_configs, coarse_targets, batch_size=32
    )
    
    # Train CNN model
    cnn_trainer = train_cnn_model(train_loader, val_loader, num_epochs=50, lr=1e-3)
    
    # Train MLP model for comparison
    mlp_trainer = train_mlp_model(train_loader, val_loader, num_epochs=50, lr=1e-3)
    
    # Comprehensive evaluation
    cnn_results, mlp_results = comprehensive_evaluation(
        cnn_trainer, mlp_trainer, test_loader, test_data
    )
    
    # Save comprehensive results to results folder
    print("\nSaving training and evaluation results...")
    save_training_results(
        cnn_results, mlp_results, 
        cnn_trainer, mlp_trainer,
        metadata=metadata,
        output_dir='results'
    )
    
    # Choose the better model for detailed analysis
    better_trainer = cnn_trainer if cnn_results['mse'] < mlp_results['mse'] else mlp_trainer
    better_name = "CNN" if cnn_results['mse'] < mlp_results['mse'] else "MLP"
    
    print(f"\nPerforming detailed analysis with {better_name} model...")
    analyze_physical_properties(better_trainer, test_data)
    
    # Save both best models to the models folder
    os.makedirs('models', exist_ok=True)
    
    cnn_model_path = 'models/best_cnn_model.pth'
    mlp_model_path = 'models/best_mlp_model.pth'
    
    torch.save(cnn_trainer.model.state_dict(), cnn_model_path)
    torch.save(mlp_trainer.model.state_dict(), mlp_model_path)
    
    print(f"\nBest CNN model saved to {cnn_model_path}")
    print(f"Best MLP model saved to {mlp_model_path}")
    
    # Clean up temporary model files
    print("\nCleaning up temporary model files...")
    cnn_trainer.cleanup_temp_models()
    mlp_trainer.cleanup_temp_models()
    
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
    print("Results saved in:")
    print("- results/: Metrics, training history, and analysis data")
    print("- figures/: All visualization plots")
    print("- models/: Best CNN and MLP models")
    print("- data/: Generated Ising model configurations")

if __name__ == "__main__":
    main()
