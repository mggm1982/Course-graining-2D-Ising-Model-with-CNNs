"""
Module for saving comprehensive training and evaluation results
"""
import json
import csv
import pickle
import os
from datetime import datetime
import numpy as np


def save_training_results(cnn_results, mlp_results, cnn_trainer, mlp_trainer, 
                          metadata=None, output_dir='results'):
    """
    Save comprehensive training and evaluation results to the results folder.
    
    Parameters:
    cnn_results: Dictionary with CNN evaluation metrics
    mlp_results: Dictionary with MLP evaluation metrics
    cnn_trainer: Trained CNN model trainer
    mlp_trainer: Trained MLP model trainer
    metadata: Optional metadata dictionary
    output_dir: Directory to save results (default: 'results')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save summary metrics (JSON for easy reading)
    save_metrics_json(cnn_results, mlp_results, output_dir, timestamp)
    
    # 2. Save training history (CSV for plotting/analysis)
    save_training_history_csv(cnn_trainer, mlp_trainer, output_dir)
    
    # 3. Save detailed results (pickle for complete data)
    save_detailed_results_pickle(cnn_results, mlp_results, cnn_trainer, 
                                 mlp_trainer, metadata, output_dir)
    
    # 4. Save comparison table (text for quick review)
    save_comparison_table(cnn_results, mlp_results, output_dir)
    
    # 5. Save configuration summary
    save_configuration_summary(metadata, output_dir)
    
    print(f"\n{'='*60}")
    print("RESULTS SAVED TO:", output_dir)
    print(f"{'='*60}")
    print(f"✓ metrics_summary.json      - Overall performance metrics")
    print(f"✓ training_history.csv      - Epoch-by-epoch training data")
    print(f"✓ detailed_results.pkl      - Complete results (for analysis)")
    print(f"✓ model_comparison.txt      - Side-by-side model comparison")
    print(f"✓ configuration.txt         - Training configuration details")
    print(f"{'='*60}\n")


def save_metrics_json(cnn_results, mlp_results, output_dir, timestamp):
    """Save metrics summary as JSON"""
    metrics_summary = {
        'timestamp': timestamp,
        'models': {
            'CNN': {
                'mse': float(cnn_results['mse']),
                'mae': float(cnn_results['mae']),
                'rmse': float(cnn_results['rmse'])
            },
            'MLP': {
                'mse': float(mlp_results['mse']),
                'mae': float(mlp_results['mae']),
                'rmse': float(mlp_results['rmse'])
            }
        },
        'best_model': 'CNN' if cnn_results['mse'] < mlp_results['mse'] else 'MLP',
        'improvement': {
            'mse_difference': float(abs(cnn_results['mse'] - mlp_results['mse'])),
            'mae_difference': float(abs(cnn_results['mae'] - mlp_results['mae']))
        }
    }
    
    json_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)


def save_training_history_csv(cnn_trainer, mlp_trainer, output_dir):
    """Save training history as CSV for easy plotting"""
    csv_path = os.path.join(output_dir, 'training_history.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['epoch', 'cnn_train_loss', 'cnn_val_loss', 
                        'mlp_train_loss', 'mlp_val_loss'])
        
        # Data rows (pad shorter history with empty values)
        max_epochs = max(len(cnn_trainer.train_losses), len(mlp_trainer.train_losses))
        
        for epoch in range(max_epochs):
            cnn_train = cnn_trainer.train_losses[epoch] if epoch < len(cnn_trainer.train_losses) else ''
            cnn_val = cnn_trainer.val_losses[epoch] if epoch < len(cnn_trainer.val_losses) else ''
            mlp_train = mlp_trainer.train_losses[epoch] if epoch < len(mlp_trainer.train_losses) else ''
            mlp_val = mlp_trainer.val_losses[epoch] if epoch < len(mlp_trainer.val_losses) else ''
            
            writer.writerow([epoch + 1, cnn_train, cnn_val, mlp_train, mlp_val])


def save_detailed_results_pickle(cnn_results, mlp_results, cnn_trainer, 
                                 mlp_trainer, metadata, output_dir):
    """Save detailed results as pickle for further analysis"""
    detailed_results = {
        'cnn_results': cnn_results,
        'mlp_results': mlp_results,
        'cnn_training_history': {
            'train_losses': cnn_trainer.train_losses,
            'val_losses': cnn_trainer.val_losses
        },
        'mlp_training_history': {
            'train_losses': mlp_trainer.train_losses,
            'val_losses': mlp_trainer.val_losses
        },
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }
    
    pkl_path = os.path.join(output_dir, 'detailed_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(detailed_results, f)


def save_comparison_table(cnn_results, mlp_results, output_dir):
    """Save human-readable comparison table"""
    txt_path = os.path.join(output_dir, 'model_comparison.txt')
    
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{'Metric':<20} {'CNN':<20} {'MLP':<20}\n")
        f.write("-"*60 + "\n")
        
        f.write(f"{'MSE':<20} {cnn_results['mse']:<20.6f} {mlp_results['mse']:<20.6f}\n")
        f.write(f"{'MAE':<20} {cnn_results['mae']:<20.6f} {mlp_results['mae']:<20.6f}\n")
        f.write(f"{'RMSE':<20} {cnn_results['rmse']:<20.6f} {mlp_results['rmse']:<20.6f}\n")
        
        f.write("\n" + "-"*60 + "\n\n")
        
        better_model = "CNN" if cnn_results['mse'] < mlp_results['mse'] else "MLP"
        f.write(f"Best Model: {better_model}\n")
        
        improvement = abs(cnn_results['mse'] - mlp_results['mse'])
        improvement_pct = (improvement / max(cnn_results['mse'], mlp_results['mse'])) * 100
        f.write(f"MSE Improvement: {improvement:.6f} ({improvement_pct:.2f}%)\n")
        
        f.write("\n" + "="*60 + "\n")


def save_configuration_summary(metadata, output_dir):
    """Save configuration and dataset information"""
    txt_path = os.path.join(output_dir, 'configuration.txt')
    
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING CONFIGURATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        if metadata:
            f.write("Dataset Configuration:\n")
            f.write("-"*60 + "\n")
            f.write(f"Lattice Size: {metadata.get('size', 'N/A')}x{metadata.get('size', 'N/A')}\n")
            f.write(f"Total Configurations: {metadata.get('total_configs', 'N/A')}\n")
            f.write(f"Number of Temperatures: {metadata.get('n_temperatures', 'N/A')}\n")
            f.write(f"Configs per Temperature: {metadata.get('n_configs_per_temp', 'N/A')}\n")
            f.write(f"Equilibration Steps: {metadata.get('equilibration_steps', 'N/A')}\n")
            f.write(f"Sampling Interval: {metadata.get('sampling_interval', 'N/A')}\n")
            
            if 'temperatures' in metadata:
                f.write(f"\nTemperatures Sampled:\n")
                for i, T in enumerate(metadata['temperatures']):
                    marker = " (CRITICAL)" if abs(T - 2.269) < 0.2 else ""
                    f.write(f"  T_{i+1} = {T:.3f}{marker}\n")
            
            if 'magnetizations' in metadata:
                mags = metadata['magnetizations']
                f.write(f"\nMagnetization Statistics:\n")
                f.write(f"  Mean: {np.mean(mags):.4f}\n")
                f.write(f"  Std:  {np.std(mags):.4f}\n")
                f.write(f"  Min:  {np.min(mags):.4f}\n")
                f.write(f"  Max:  {np.max(mags):.4f}\n")
            
            f.write(f"\nGeneration Time: {metadata.get('generation_time', 'N/A')}\n")
        else:
            f.write("No metadata available\n")
        
        f.write("\n" + "="*60 + "\n")


def save_per_temperature_analysis(test_data, predictions_cnn, predictions_mlp, 
                                  temperatures, output_dir='results'):
    """
    Save per-temperature performance analysis.
    Useful for understanding model behavior across different phases.
    """
    csv_path = os.path.join(output_dir, 'per_temperature_metrics.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', 'phase', 'n_samples', 
                        'cnn_mse', 'cnn_mae', 'mlp_mse', 'mlp_mae'])
        
        for T in np.unique(temperatures):
            # Find samples at this temperature
            temp_mask = temperatures == T
            
            # Determine phase
            if T < 1.8:
                phase = 'ordered'
            elif T < 2.8:
                phase = 'critical'
            else:
                phase = 'disordered'
            
            # Calculate metrics for this temperature
            n_samples = np.sum(temp_mask)
            
            # This would need the actual predictions and targets
            # Placeholder for now
            writer.writerow([T, phase, n_samples, 0, 0, 0, 0])


if __name__ == "__main__":
    print("Results saving module")
    print("Import this module in train_model.py to save comprehensive results")

