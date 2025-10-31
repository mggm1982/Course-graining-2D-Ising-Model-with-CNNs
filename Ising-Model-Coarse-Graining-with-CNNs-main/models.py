import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class IsingDataset(Dataset):
    """Custom dataset for Ising model configurations"""
    
    def __init__(self, fine_configs, coarse_targets, transform=None):
        """
        Initialize the dataset.
        
        Parameters:
        fine_configs (array): Fine-grained configurations
        coarse_targets (array): Coarse-grained target configurations
        transform (callable): Optional transform to be applied to samples
        """
        self.fine_configs = torch.FloatTensor(fine_configs)
        self.coarse_targets = torch.FloatTensor(coarse_targets)
        self.transform = transform
        
        # Add channel dimension for CNN (N, C, H, W format)
        if len(self.fine_configs.shape) == 3:
            self.fine_configs = self.fine_configs.unsqueeze(1)
        if len(self.coarse_targets.shape) == 3:
            self.coarse_targets = self.coarse_targets.unsqueeze(1)
    
    def __len__(self):
        return len(self.fine_configs)
    
    def __getitem__(self, idx):
        fine_config = self.fine_configs[idx]
        coarse_target = self.coarse_targets[idx]
        
        if self.transform:
            fine_config = self.transform(fine_config)
            coarse_target = self.transform(coarse_target)
        
        return fine_config, coarse_target

class CoarseGrainingCNN(nn.Module):
    """
    Convolutional Neural Network for coarse-graining Ising configurations.
    Maps from 16x16 fine-grained to 8x8 coarse-grained configurations.
    """
    
    def __init__(self, input_size=16, output_size=8, hidden_channels=[16, 32, 16]):
        super(CoarseGrainingCNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Convolutional layers with appropriate kernel sizes and strides
        self.conv1 = nn.Conv2d(1, hidden_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1)
        
        # Adaptive pooling to get desired output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_size, output_size))
        
        # Final convolution to get single channel output
        self.final_conv = nn.Conv2d(hidden_channels[2], 1, kernel_size=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pooling to desired output size
        x = self.adaptive_pool(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Apply tanh to get values in [-1, 1] range like spins
        x = torch.tanh(x)
        
        return x

class CoarseGrainingMLP(nn.Module):
    """
    Multi-Layer Perceptron alternative for coarse-graining.
    Flattens input and uses fully connected layers.
    """
    
    def __init__(self, input_size=16, output_size=8, hidden_dims=[512, 256, 128]):
        super(CoarseGrainingMLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.input_dim = input_size * input_size
        self.output_dim = output_size * output_size
        
        # Build the MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Pass through the network
        x = self.network(x)
        
        # Reshape to output grid
        x = x.view(batch_size, 1, self.output_size, self.output_size)
        
        return x

class CoarseGrainingTrainer:
    """Trainer class for the coarse-graining neural network"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', model_name='model'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.model_name = model_name
        self.best_model_path = None
        
    def train_model(self, train_loader, val_loader, num_epochs=100, lr=1e-3, 
                   weight_decay=1e-5, patience=10):
        """
        Train the coarse-graining model.
        
        Parameters:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model to models folder
                os.makedirs('models', exist_ok=True)
                self.best_model_path = f'models/temp_best_{self.model_name}.pth'
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                epochs_without_improvement += 1
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
    
    def predict(self, inputs):
        """Make predictions with the trained model"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.FloatTensor(inputs)
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(0).unsqueeze(0)
                elif len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)
            
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            return outputs.cpu().numpy()
    
    def cleanup_temp_models(self):
        """Remove temporary model checkpoint files"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
            print(f"Cleaned up temporary model file: {self.best_model_path}")
    
    def plot_training_history(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        return plt.gcf()

def evaluate_model(trainer, test_loader, device='cpu'):
    """
    Evaluate the trained model on test data.
    
    Returns various metrics and sample predictions.
    """
    trainer.model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    all_inputs = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = trainer.model(inputs)
            
            # Calculate metrics
            mse = F.mse_loss(predictions, targets, reduction='sum').item()
            mae = F.l1_loss(predictions, targets, reduction='sum').item()
            
            total_mse += mse
            total_mae += mae
            total_samples += inputs.size(0)
            
            # Store ALL samples for intelligent selection later
            all_inputs.extend(inputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    print(f"Test Results:")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"RMSE: {np.sqrt(avg_mse):.6f}")
    
    # Select diverse samples based on magnetization
    # This ensures we get interesting configurations from different phases
    sample_inputs, sample_targets, sample_predictions = _select_diverse_samples(
        all_inputs, all_targets, all_predictions, n_samples=5
    )
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'rmse': np.sqrt(avg_mse),
        'sample_inputs': sample_inputs,
        'sample_targets': sample_targets,
        'sample_predictions': sample_predictions
    }

def _select_diverse_samples(inputs, targets, predictions, n_samples=5):
    """
    Select diverse samples from different temperature regimes.
    Prioritizes configurations with different magnetizations.
    """
    if len(inputs) <= n_samples:
        return inputs, targets, predictions
    
    # Calculate magnetization for each configuration
    magnetizations = [np.mean(inp[0]) for inp in inputs]  # inp[0] removes channel dim
    
    # Find indices for different regimes
    indices = []
    
    # 1. High negative magnetization (ordered spin-down)
    neg_ordered = [i for i, m in enumerate(magnetizations) if m < -0.8]
    if neg_ordered:
        indices.append(neg_ordered[len(neg_ordered)//2])  # middle example
    
    # 2. Medium negative magnetization
    med_neg = [i for i, m in enumerate(magnetizations) if -0.5 < m < -0.1]
    if med_neg:
        indices.append(med_neg[len(med_neg)//2])
    
    # 3. Near zero magnetization (critical/disordered)
    near_zero = [i for i, m in enumerate(magnetizations) if abs(m) < 0.3]
    if near_zero:
        indices.append(near_zero[len(near_zero)//2])
    
    # 4. Medium positive magnetization
    med_pos = [i for i, m in enumerate(magnetizations) if 0.1 < m < 0.5]
    if med_pos:
        indices.append(med_pos[len(med_pos)//2])
    
    # 5. High positive magnetization (ordered spin-up)
    pos_ordered = [i for i, m in enumerate(magnetizations) if m > 0.8]
    if pos_ordered:
        indices.append(pos_ordered[len(pos_ordered)//2])
    
    # If we don't have enough diverse samples, fill with evenly spaced ones
    while len(indices) < n_samples and len(indices) < len(inputs):
        # Find largest gap in magnetization space
        if not indices:
            indices.append(0)
        else:
            sorted_mags = [(magnetizations[i], i) for i in indices]
            sorted_mags.sort()
            
            # Add sample that's most different from existing ones
            max_dist = 0
            best_idx = None
            for i in range(len(inputs)):
                if i not in indices:
                    min_dist_to_existing = min(abs(magnetizations[i] - m) for m, _ in sorted_mags)
                    if min_dist_to_existing > max_dist:
                        max_dist = min_dist_to_existing
                        best_idx = i
            
            if best_idx is not None:
                indices.append(best_idx)
            else:
                break
    
    # Ensure we have exactly n_samples
    indices = indices[:n_samples]
    
    selected_inputs = [inputs[i] for i in indices]
    selected_targets = [targets[i] for i in indices]
    selected_predictions = [predictions[i] for i in indices]
    
    return selected_inputs, selected_targets, selected_predictions

if __name__ == "__main__":
    # This will be used for testing the neural network components
    print("Neural network module loaded successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
