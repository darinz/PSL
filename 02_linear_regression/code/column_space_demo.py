"""
Column Space Demonstration
==========================

This module demonstrates the concept of column space in linear regression,
showing how the design matrix spans a subspace of the observation space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_column_space():
    """Demonstrate the concept of column space"""
    
    # Create a simple design matrix
    X = np.array([[1, 2], [1, 4], [1, 6]])  # n=3, p=1
    print("Design matrix X:")
    print(X)
    
    # Different coefficient vectors
    beta1 = np.array([1, 2])
    beta2 = np.array([0, 1])
    beta3 = np.array([-1, 0.5])
    
    # Compute different points in the column space
    y1 = X @ beta1
    y2 = X @ beta2
    y3 = X @ beta3
    
    print("\nColumn space examples:")
    print(f"X * {beta1} = {y1}")
    print(f"X * {beta2} = {y2}")
    print(f"X * {beta3} = {y3}")
    
    # Visualize column space in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the column vectors
    ax.quiver(0, 0, 0, X[0, 0], X[1, 0], X[2, 0], color='blue', arrow_length_ratio=0.1, label='Column 1 (intercept)')
    ax.quiver(0, 0, 0, X[0, 1], X[1, 1], X[2, 1], color='red', arrow_length_ratio=0.1, label='Column 2 (predictor)')
    
    # Plot some points in the column space
    ax.scatter(y1[0], y1[1], y1[2], color='green', s=100, label=f'X*{beta1}')
    ax.scatter(y2[0], y2[1], y2[2], color='orange', s=100, label=f'X*{beta2}')
    ax.scatter(y3[0], y3[1], y3[2], color='purple', s=100, label=f'X*{beta3}')
    
    # Plot the plane spanned by the columns
    # Generate points on the plane
    t = np.linspace(-2, 2, 20)
    s = np.linspace(-2, 2, 20)
    T, S = np.meshgrid(t, s)
    
    plane_x = T * X[0, 0] + S * X[0, 1]
    plane_y = T * X[1, 0] + S * X[1, 1]
    plane_z = T * X[2, 0] + S * X[2, 1]
    
    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.3, color='gray')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Column Space of Design Matrix X')
    ax.legend()
    
    plt.show()
    
    return X, y1, y2, y3

if __name__ == "__main__":
    # Demonstrate column space
    X, y1, y2, y3 = demonstrate_column_space()
