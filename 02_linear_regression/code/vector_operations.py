import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_vector_operations():
    """Demonstrate basic vector operations"""
    
    # Define vectors
    a = np.array([1, 2, 0])
    b = np.array([3, 1, 1])
    
    # Scalar multiplication
    scaled_a = 2 * a
    scaled_b = 3 * b
    
    # Vector addition
    result = scaled_a + scaled_b
    
    print("Vector a:", a)
    print("Vector b:", b)
    print("2a:", scaled_a)
    print("3b:", scaled_b)
    print("2a + 3b:", result)
    
    # Vector norms (lengths)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_result = np.linalg.norm(result)
    
    print(f"\nNorms:")
    print(f"||a|| = {norm_a:.3f}")
    print(f"||b|| = {norm_b:.3f}")
    print(f"||2a + 3b|| = {norm_result:.3f}")
    
    return a, b, result

def plot_vectors_3d(a, b, result):
    """Plot vectors in 3D space"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vectors as arrows from origin
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='blue', arrow_length_ratio=0.1, label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='red', arrow_length_ratio=0.1, label='b')
    ax.quiver(0, 0, 0, result[0], result[1], result[2], color='green', arrow_length_ratio=0.1, label='2a + 3b')
    
    # Plot scaled vectors
    ax.quiver(0, 0, 0, 2*a[0], 2*a[1], 2*a[2], color='lightblue', alpha=0.5, arrow_length_ratio=0.1, label='2a')
    ax.quiver(0, 0, 0, 3*b[0], 3*b[1], 3*b[2], color='lightcoral', alpha=0.5, arrow_length_ratio=0.1, label='3b')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Operations in 3D Space')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = max(result)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.show()

# Run demonstration
a, b, result = demonstrate_vector_operations()

# Create visualization
plot_vectors_3d(a, b, result)
