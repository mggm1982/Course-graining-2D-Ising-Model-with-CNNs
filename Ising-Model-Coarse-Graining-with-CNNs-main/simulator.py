"""
2D Ising Model Simulator using Monte Carlo (Metropolis Algorithm)
This module generates configurations of the 2D Ising model at different temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os

class IsingSimulator:
    def __init__(self, size=16, J=1.0, h=0.0):
        """
        Initialize the 2D Ising model simulator.
        
        Parameters:
        size (int): Size of the square lattice (size x size)
        J (float): Coupling constant (positive for ferromagnetic)
        h (float): External magnetic field
        """
        self.size = size
        self.J = J
        self.h = h
        self.spins = None
        self.reset_lattice()
        
    def reset_lattice(self, random_state=None):
        """Initialize the lattice with random spins (+1 or -1)"""
        if random_state is not None:
            np.random.seed(random_state)
        self.spins = np.random.choice([-1, 1], size=(self.size, self.size))
    
    @staticmethod
    @jit(nopython=True)
    def calculate_energy_change(spins, i, j, J, h):
        """
        Calculate the energy change if we flip spin at position (i, j).
        Uses periodic boundary conditions.
        """
        size = spins.shape[0]
        current_spin = spins[i, j]
        
        # Sum of neighboring spins (with periodic boundary conditions)
        neighbors_sum = (spins[(i-1) % size, j] + 
                        spins[(i+1) % size, j] + 
                        spins[i, (j-1) % size] + 
                        spins[i, (j+1) % size])
        
        # Energy change = 2 * current_spin * (J * neighbors_sum + h)
        delta_E = 2 * current_spin * (J * neighbors_sum + h)
        return delta_E
    
    def monte_carlo_step(self, temperature):
        """
        Perform one Monte Carlo step (attempt to flip each spin once on average).
        """
        beta = 1.0 / temperature if temperature > 0 else float('inf')
        
        # Attempt N random spin flips where N = total number of spins
        for _ in range(self.size * self.size):
            i = np.random.randint(0, self.size)
            j = np.random.randint(0, self.size)
            
            delta_E = self.calculate_energy_change(self.spins, i, j, self.J, self.h)
            
            # Accept the flip if it lowers energy or with Boltzmann probability
            if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
                self.spins[i, j] *= -1
    
    def equilibrate(self, temperature, steps=1000):
        """
        Equilibrate the system at given temperature.
        """
        for _ in range(steps):
            self.monte_carlo_step(temperature)
    
    def calculate_magnetization(self):
        """Calculate the total magnetization of the system"""
        return np.sum(self.spins) / (self.size * self.size)
    
    def calculate_energy(self):
        """Calculate the total energy of the system"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                # Nearest neighbor interactions (count each pair once)
                energy -= self.J * self.spins[i, j] * (
                    self.spins[(i+1) % self.size, j] + 
                    self.spins[i, (j+1) % self.size]
                )
                # External field contribution
                energy -= self.h * self.spins[i, j]
        return energy
    
    def generate_configurations(self, temperatures, n_configs_per_temp=100, 
                              equilibration_steps=1000, sampling_interval=10):
        """
        Generate configurations at different temperatures.
        
        Parameters:
        temperatures (array): Array of temperatures to sample
        n_configs_per_temp (int): Number of configurations per temperature
        equilibration_steps (int): Steps to equilibrate before sampling
        sampling_interval (int): Steps between samples
        
        Returns:
        configurations (array): Array of spin configurations
        temp_labels (array): Temperature labels for each configuration
        """
        configurations = []
        temp_labels = []
        
        print("Generating Ising model configurations...")
        
        for temp_idx, T in enumerate(temperatures):
            print(f"Temperature {temp_idx+1}/{len(temperatures)}: T = {T:.3f}")
            
            # Start from a random configuration
            self.reset_lattice()
            
            # Equilibrate at this temperature
            self.equilibrate(T, equilibration_steps)
            
            # Sample configurations
            temp_configs = []
            for config_idx in range(n_configs_per_temp):
                # Perform some Monte Carlo steps between samples
                for _ in range(sampling_interval):
                    self.monte_carlo_step(T)
                
                # Store the configuration
                temp_configs.append(self.spins.copy())
                temp_labels.append(T)
                
                if (config_idx + 1) % 20 == 0:
                    print(f"  Generated {config_idx + 1}/{n_configs_per_temp} configurations")
            
            configurations.extend(temp_configs)
        
        return np.array(configurations), np.array(temp_labels)

def create_coarse_grained_target(fine_config, block_size=2):
    """
    Create coarse-grained configuration using block averaging.
    
    Parameters:
    fine_config (array): Fine-grained configuration (size x size)
    block_size (int): Size of blocks for coarse-graining
    
    Returns:
    coarse_config (array): Coarse-grained configuration
    """
    size = fine_config.shape[0]
    coarse_size = size // block_size
    coarse_config = np.zeros((coarse_size, coarse_size))
    
    for i in range(coarse_size):
        for j in range(coarse_size):
            # Extract the block
            block = fine_config[i*block_size:(i+1)*block_size, 
                              j*block_size:(j+1)*block_size]
            
            # Use majority voting (or average and threshold)
            block_average = np.mean(block)
            coarse_config[i, j] = 1 if block_average > 0 else -1
    
    return coarse_config

def visualize_configurations(fine_config, coarse_target, coarse_predicted=None):
    """
    Visualize fine-grained and coarse-grained configurations.
    """
    if coarse_predicted is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(fine_config, cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('Fine-grained (Original)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        
        im2 = axes[1].imshow(coarse_target, cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Coarse-grained (Target)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        
        im3 = axes[2].imshow(coarse_predicted, cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Coarse-grained (Predicted)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        
        # Add colorbar
        plt.colorbar(im1, ax=axes, shrink=0.6, label='Spin value')
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = axes[0].imshow(fine_config, cmap='RdBu', vmin=-1, vmax=1)
        axes[0].set_title('Fine-grained (16x16)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        
        im2 = axes[1].imshow(coarse_target, cmap='RdBu', vmin=-1, vmax=1)
        axes[1].set_title('Coarse-grained (8x8)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        
        # Add colorbar
        plt.colorbar(im1, ax=axes, shrink=0.6, label='Spin value')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    simulator = IsingSimulator(size=16)
    
    # Critical temperature for 2D Ising model
    T_c = 2.269
    temperatures = np.linspace(0.5, 4.0, 8)
    
    # Generate some test configurations
    configs, temp_labels = simulator.generate_configurations(
        temperatures, n_configs_per_temp=10, 
        equilibration_steps=500, sampling_interval=5
    )
    
    # Create coarse-grained targets
    coarse_targets = []
    for config in configs:
        coarse_target = create_coarse_grained_target(config)
        coarse_targets.append(coarse_target)
    
    coarse_targets = np.array(coarse_targets)
    
    # Visualize a few examples
    for i in range(3):
        fig = visualize_configurations(configs[i], coarse_targets[i])
        plt.savefig(f'example_config_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"Generated {len(configs)} configurations")
    print(f"Fine-grained shape: {configs.shape}")
    print(f"Coarse-grained shape: {coarse_targets.shape}")
