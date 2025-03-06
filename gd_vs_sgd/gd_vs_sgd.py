#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic data for a simple linear regression problem
def generate_data(n=100, noise=0.5):
    x = np.linspace(-10, 10, n)
    # True parameters: w=2, b=3
    y_true = 2 * x + 3
    y = y_true + noise * np.random.randn(n)
    return x, y

# Mean Squared Error (MSE) loss function
def mse_loss(x, y, w, b):
    y_pred = w * x + b
    return np.mean((y - y_pred) ** 2)

# Gradient of the MSE loss with respect to parameters
def gradient(x, y, w, b):
    n = len(x)
    y_pred = w * x + b
    dw = -2/n * np.sum(x * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    return dw, db

# Gradient Descent algorithm
def gradient_descent(x, y, learning_rate=0.01, iterations=200):
    # Initialize parameters with the same random values for both algorithms
    np.random.seed(42)  # Reset seed to get same initialization
    w = np.random.randn()
    b = np.random.randn()
    
    # Store initial values for comparison
    w_init = w
    b_init = b
    
    # To store history for visualization
    w_history = [w]
    b_history = [b]
    loss_history = [mse_loss(x, y, w, b)]
    
    for i in range(iterations):
        # Calculate gradients
        dw, db = gradient(x, y, w, b)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Store history
        w_history.append(w)
        b_history.append(b)
        loss_history.append(mse_loss(x, y, w, b))
    
    print(f"GD - Initial: w={w_init:.4f}, b={b_init:.4f}")
    print(f"GD - Final: w={w:.4f}, b={b:.4f}")
    
    return w, b, w_history, b_history, loss_history

# Stochastic Gradient Descent algorithm
def sgd(x, y, learning_rate=0.01, iterations=200, batch_size=1):
    # Initialize parameters with the same random values for both algorithms
    np.random.seed(42)  # Reset seed to get same initialization
    w = np.random.randn()
    b = np.random.randn()
    
    # Store initial values for comparison
    w_init = w
    b_init = b
    
    # To store history for visualization
    w_history = [w]
    b_history = [b]
    loss_history = [mse_loss(x, y, w, b)]
    
    indices = list(range(len(x)))
    
    for i in range(iterations):
        # Shuffle indices for randomness in batch selection
        random.shuffle(indices)
        
        # Select a mini-batch
        batch_indices = indices[:batch_size]
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        
        # Calculate gradients for the mini-batch
        n = len(x_batch)
        y_pred = w * x_batch + b
        dw = -2/n * np.sum(x_batch * (y_batch - y_pred))
        db = -2/n * np.sum(y_batch - y_pred)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Store history
        w_history.append(w)
        b_history.append(b)
        loss_history.append(mse_loss(x, y, w, b))
    
    print(f"SGD - Initial: w={w_init:.4f}, b={b_init:.4f}")
    print(f"SGD - Final: w={w:.4f}, b={b:.4f}")
    
    return w, b, w_history, b_history, loss_history

# Create contour plot for loss landscape
def create_loss_contour(x, y, w_range=(-1, 4), b_range=(0, 6), resolution=100):
    w_values = np.linspace(w_range[0], w_range[1], resolution)
    b_values = np.linspace(b_range[0], b_range[1], resolution)
    W, B = np.meshgrid(w_values, b_values)
    
    Z = np.zeros_like(W)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = mse_loss(x, y, W[i, j], B[i, j])
    
    return W, B, Z

# Calculate the true optimal parameters analytically
def calculate_optimal_parameters(x, y):
    # For linear regression, the closed-form solution is:
    # w = (X^T X)^(-1) X^T y
    X = np.column_stack((np.ones_like(x), x))
    params = np.linalg.inv(X.T @ X) @ X.T @ y
    b_opt, w_opt = params
    return w_opt, b_opt

# Main function to run and visualize both algorithms
def compare_gd_sgd():
    # Generate data
    x, y = generate_data(n=100, noise=1.0)
    
    # Calculate optimal parameters
    w_opt, b_opt = calculate_optimal_parameters(x, y)
    print(f"Optimal parameters: w={w_opt:.4f}, b={b_opt:.4f}")
    
    # Run algorithms (with more iterations and adjusted learning rates)
    w_gd, b_gd, w_history_gd, b_history_gd, loss_history_gd = gradient_descent(x, y, learning_rate=0.01, iterations=200)
    w_sgd, b_sgd, w_history_sgd, b_history_sgd, loss_history_sgd = sgd(x, y, learning_rate=0.005, iterations=1000, batch_size=1)
    
    print(f"GD distance from optimal: {np.sqrt((w_gd-w_opt)**2 + (b_gd-b_opt)**2):.6f}")
    print(f"SGD distance from optimal: {np.sqrt((w_sgd-w_opt)**2 + (b_sgd-b_opt)**2):.6f}")
    
    # Create loss landscape for visualization
    w_min = min(min(w_history_gd), min(w_history_sgd), w_opt) - 0.5
    w_max = max(max(w_history_gd), max(w_history_sgd), w_opt) + 0.5
    b_min = min(min(b_history_gd), min(b_history_sgd), b_opt) - 0.5
    b_max = max(max(b_history_gd), max(b_history_sgd), b_opt) + 0.5
    
    W, B, Z = create_loss_contour(x, y, w_range=(w_min, w_max), b_range=(b_min, b_max))
    
    # Create figure with two subplots side by side
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    
    # Plot for Gradient Descent
    ax1.set_title('Gradient Descent Convergence', fontsize=14)
    cs1 = ax1.contour(W, B, Z, levels=50, cmap='viridis')
    line_gd, = ax1.plot(w_history_gd, b_history_gd, 'r-o', linewidth=2, markersize=3, label='GD Path')
    ax1.plot(w_opt, b_opt, 'g*', markersize=12, label='Optimal Solution')
    ax1.set_xlabel('Weight (w)', fontsize=12)
    ax1.set_ylabel('Bias (b)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot for Stochastic Gradient Descent
    ax2.set_title('Stochastic Gradient Descent Convergence', fontsize=14)
    cs2 = ax2.contour(W, B, Z, levels=50, cmap='viridis')
    line_sgd, = ax2.plot(w_history_sgd, b_history_sgd, 'b-o', linewidth=2, markersize=1, label='SGD Path')
    ax2.plot(w_opt, b_opt, 'g*', markersize=12, label='Optimal Solution')
    ax2.set_xlabel('Weight (w)', fontsize=12)
    ax2.set_ylabel('Bias (b)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Ensure both plots have the same scale
    ax1.set_xlim(w_min, w_max)
    ax1.set_ylim(b_min, b_max)
    ax2.set_xlim(w_min, w_max)
    ax2.set_ylim(b_min, b_max)
    
    plt.colorbar(cs1, ax=ax1, label='Loss (MSE)')
    plt.colorbar(cs2, ax=ax2, label='Loss (MSE)')
    
    plt.tight_layout()
    plt.savefig('gd_vs_sgd.png', dpi=300)
    plt.show()
    
    # Plot the loss history
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history_gd, 'r-', linewidth=2, label='Gradient Descent')
    plt.plot(loss_history_sgd, 'b-', linewidth=2, label='Stochastic Gradient Descent')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Loss Convergence Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=300)
    plt.show()

# Run the comparison
if __name__ == "__main__":
    compare_gd_sgd()
