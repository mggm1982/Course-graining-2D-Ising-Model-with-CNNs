import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime

# Import our custom modules
from simulator import IsingSimulator, create_coarse_grained_target, visualize_configurations

def setup_project_directories():
    """Create necessary directories for the project"""
    directories = ['data', 'models', 'results', 'figures']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Project directories created successfully!")

def generate_training_data(size=16, n_temps=8, n_configs_per_temp=200, 
                          equilibration_steps=1000, sampling_interval=10, 
                          save_data=True):
    """
    Generate comprehensive training data for the coarse-graining neural network.
    
    Parameters:
    size (int): Size of the square lattice
    n_temps (int): Number of different temperatures to sample
    n_configs_per_temp (int): Number of configurations per temperature
    equilibration_steps (int): Monte Carlo steps for equilibration
    sampling_interval (int): Steps between configuration samples
    save_data (bool): Whether to save the generated data
    
    Returns:
    tuple: (fine_configs, coarse_targets, temperatures, metadata)
    """
    print("=" * 60)
    print("GENERATING TRAINING DATA")
    print("=" * 60)
    
    # Initialize the Ising simulator
    simulator = IsingSimulator(size=size)
    
    # Define temperature range around critical temperature
    T_c = 2.269  # Critical temperature for 2D Ising model
    temperatures = np.concatenate([
        np.linspace(0.5, T_c - 0.5, n_temps//3),  # Low temperature (ordered)
        np.linspace(T_c - 0.3, T_c + 0.3, n_temps//3),  # Near critical
        np.linspace(T_c + 0.5, 4.0, n_temps//3)  # High temperature (disordered)
    ])
    
    if len(temperatures) < n_temps:
        temperatures = np.linspace(0.5, 4.0, n_temps)
    
    print(f"Sampling at {len(temperatures)} temperatures:")
    for i, T in enumerate(temperatures):
        marker = " (CRITICAL)" if abs(T - T_c) < 0.1 else ""
        print(f"  T_{i+1} = {T:.3f}{marker}")
    
    # Generate configurations
    fine_configs, temp_labels = simulator.generate_configurations(
        temperatures, 
        n_configs_per_temp=n_configs_per_temp,
        equilibration_steps=equilibration_steps,
        sampling_interval=sampling_interval
    )
    
    print(f"\nGenerating coarse-grained targets...")
    
    # Create coarse-grained targets using block averaging
    coarse_targets = []
    for i, config in enumerate(fine_configs):
        coarse_target = create_coarse_grained_target(config, block_size=2)
        coarse_targets.append(coarse_target)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(fine_configs)} configurations")
    
    coarse_targets = np.array(coarse_targets)
    
    # Calculate some statistics
    magnetizations = [np.mean(config) for config in fine_configs]
    coarse_magnetizations = [np.mean(config) for config in coarse_targets]
    
    metadata = {
        'size': size,
        'n_temperatures': len(temperatures),
        'temperatures': temperatures,
        'n_configs_per_temp': n_configs_per_temp,
        'total_configs': len(fine_configs),
        'equilibration_steps': equilibration_steps,
        'sampling_interval': sampling_interval,
        'temperature_labels': temp_labels,
        'magnetizations': magnetizations,
        'coarse_magnetizations': coarse_magnetizations,
        'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"\nData generation completed!")
    print(f"Total configurations: {len(fine_configs)}")
    print(f"Fine-grained shape: {fine_configs.shape}")
    print(f"Coarse-grained shape: {coarse_targets.shape}")
    print(f"Magnetization range: [{np.min(magnetizations):.3f}, {np.max(magnetizations):.3f}]")
    
    if save_data:
        # Save the data
        data_filename = f'data/ising_data_{size}x{size}_{len(fine_configs)}configs.npz'
        np.savez_compressed(data_filename, 
                           fine_configs=fine_configs,
                           coarse_targets=coarse_targets,
                           temperatures=temperatures,
                           temp_labels=temp_labels)
        
        # Save metadata
        with open('data/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Data saved to: {data_filename}")
        print(f"Metadata saved to: data/metadata.pkl")
    
    return fine_configs, coarse_targets, temperatures, metadata

def analyze_data_distribution(fine_configs, coarse_targets, temperatures, metadata):
    """
    Analyze and visualize the distribution of generated data.
    """
    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    
    # Create analysis plots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Magnetization vs Temperature
    plt.subplot(2, 3, 1)
    temp_labels = metadata['temperature_labels']
    magnetizations = metadata['magnetizations']
    
    # Plot magnetization vs temperature
    for temp in np.unique(temperatures):
        temp_mask = temp_labels == temp
        temp_mags = np.array(magnetizations)[temp_mask]
        plt.scatter([temp] * len(temp_mags), np.abs(temp_mags), alpha=0.6, s=10)
    
    plt.axvline(x=2.269, color='red', linestyle='--', alpha=0.7, label='T_c = 2.269')
    plt.xlabel('Temperature')
    plt.ylabel('|Magnetization|')
    plt.title('Magnetization vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Magnetization histogram
    plt.subplot(2, 3, 2)
    plt.hist(magnetizations, bins=50, alpha=0.7, density=True)
    plt.xlabel('Magnetization')
    plt.ylabel('Density')
    plt.title('Magnetization Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Sample configurations at different temperatures
    sample_temps = [temperatures[0], temperatures[len(temperatures)//2], temperatures[-1]]
    for i, temp in enumerate(sample_temps):
        plt.subplot(2, 3, 4 + i)
        
        # Find a configuration at this temperature
        temp_indices = np.where(temp_labels == temp)[0]
        if len(temp_indices) > 0:
            config_idx = temp_indices[len(temp_indices)//2]
            config = fine_configs[config_idx]
            
            plt.imshow(config, cmap='RdBu', vmin=-1, vmax=1)
            plt.title(f'T = {temp:.2f}')
            plt.axis('off')
    
    # 3. Coarse-graining quality check
    plt.subplot(2, 3, 3)
    coarse_mags = metadata['coarse_magnetizations']
    plt.scatter(magnetizations, coarse_mags, alpha=0.6, s=10)
    plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.7, label='Perfect correlation')
    plt.xlabel('Fine-grained Magnetization')
    plt.ylabel('Coarse-grained Magnetization')
    plt.title('Coarse-graining Preservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/data_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Show a few example configurations from diverse temperature regimes
    print("\nExample configurations:")
    n_examples = 3
    
    # Select diverse samples based on magnetization
    # This ensures we get interesting configurations from different phases
    magnetizations = np.array(metadata['magnetizations'])
    temp_labels = metadata['temperature_labels']
    
    # Find indices for different regimes
    example_indices = []
    
    # 1. Ordered phase (high |magnetization|)
    ordered_idx = np.where(np.abs(magnetizations) > 0.8)[0]
    if len(ordered_idx) > 0:
        example_indices.append(ordered_idx[len(ordered_idx)//2])
    
    # 2. Critical/disordered phase (low |magnetization|)
    critical_idx = np.where(np.abs(magnetizations) < 0.3)[0]
    if len(critical_idx) > 0:
        example_indices.append(critical_idx[len(critical_idx)//2])
    
    # 3. Intermediate phase
    intermediate_idx = np.where((np.abs(magnetizations) >= 0.3) & (np.abs(magnetizations) <= 0.8))[0]
    if len(intermediate_idx) > 0:
        example_indices.append(intermediate_idx[len(intermediate_idx)//2])
    
    # If we don't have enough diverse samples, use evenly spaced indices
    if len(example_indices) < n_examples:
        example_indices = [0, len(fine_configs)//2, len(fine_configs)-1]
    
    for idx_num, i in enumerate(example_indices[:n_examples]):
        mag = magnetizations[i]
        temp = temp_labels[i]
        print(f"  Example {idx_num+1}: T={temp:.3f}, M={mag:.3f}")
        fig = visualize_configurations(fine_configs[i], coarse_targets[i])
        plt.savefig(f'figures/example_config_{idx_num+1}.png', dpi=150, bbox_inches='tight')
        plt.show()

def prepare_data_for_training(fine_configs, coarse_targets, test_size=0.2, val_size=0.1):
    """
    Prepare data for neural network training.
    
    Returns:
    tuple: (train_fine, train_coarse, val_fine, val_coarse, test_fine, test_coarse)
    """
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 60)
    
    # First split: separate test set
    train_val_fine, test_fine, train_val_coarse, test_coarse = train_test_split(
        fine_configs, coarse_targets, test_size=test_size, random_state=42
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
    train_fine, val_fine, train_coarse, val_coarse = train_test_split(
        train_val_fine, train_val_coarse, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"Training set: {len(train_fine)} configurations")
    print(f"Validation set: {len(val_fine)} configurations")
    print(f"Test set: {len(test_fine)} configurations")
    
    return train_fine, train_coarse, val_fine, val_coarse, test_fine, test_coarse

def main():
    """Main function to run the complete workflow"""
    print("2D Ising Model Coarse-Graining with Neural Networks")
    print("=" * 60)
    
    # Setup project structure
    setup_project_directories()
    
    # Check if data already exists
    data_file = 'data/ising_data_16x16_1600configs.npz'
    if os.path.exists(data_file) and os.path.exists('data/metadata.pkl'):
        print("Loading existing data...")
        data = np.load(data_file)
        fine_configs = data['fine_configs']
        coarse_targets = data['coarse_targets']
        temperatures = data['temperatures']
        
        with open('data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loaded {len(fine_configs)} configurations")
    else:
        # Generate new data
        fine_configs, coarse_targets, temperatures, metadata = generate_training_data(
            size=16,
            n_temps=8,
            n_configs_per_temp=200,
            equilibration_steps=1000,
            sampling_interval=10
        )
    
    # Analyze the data
    analyze_data_distribution(fine_configs, coarse_targets, temperatures, metadata)
    
    # Prepare data for training (this would normally import scikit-learn)
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Install required packages: pip install torch torchvision numpy matplotlib scikit-learn numba")
    print("2. Run the neural network training script")
    print("3. Evaluate model performance")
    
    return fine_configs, coarse_targets, temperatures, metadata

if __name__ == "__main__":
    # Import scikit-learn only when needed (for splitting)
    try:
        from sklearn.model_selection import train_test_split
        main()
    except ImportError:
        print("Scikit-learn not found. Running data generation only...")
        
        # Setup and generate data without train/test split
        setup_project_directories()
        fine_configs, coarse_targets, temperatures, metadata = generate_training_data()
        
        print("\nPlease install scikit-learn to enable train/test splitting:")
        print("pip install scikit-learn")
