"""
Distance Measures Implementation
===============================

This module provides a comprehensive implementation of distance measures
for clustering analysis, including numerical, categorical, and text-based measures.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns

class DistanceMeasures:
    """Comprehensive implementation of distance measures for clustering analysis."""
    
    def __init__(self):
        pass
    
    def euclidean_distance(self, x, z):
        """Compute Euclidean distance between two points."""
        return np.sqrt(np.sum((x - z) ** 2))
    
    def manhattan_distance(self, x, z):
        """Compute Manhattan distance between two points."""
        return np.sum(np.abs(x - z))
    
    def minkowski_distance(self, x, z, p=2):
        """Compute Minkowski distance with parameter p."""
        return np.power(np.sum(np.power(np.abs(x - z), p)), 1/p)
    
    def chebyshev_distance(self, x, z):
        """Compute Chebyshev distance (L∞ norm)."""
        return np.max(np.abs(x - z))
    
    def hamming_distance(self, x, z):
        """Compute Hamming distance between two arrays."""
        return np.sum(x != z)
    
    def jaccard_distance(self, set_a, set_b):
        """Compute Jaccard distance between two sets."""
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return 1 - intersection / union if union > 0 else 0
    
    def cosine_distance(self, x, z):
        """Compute cosine distance between two vectors."""
        dot_product = np.dot(x, z)
        norm_x = np.linalg.norm(x)
        norm_z = np.linalg.norm(z)
        return 1 - dot_product / (norm_x * norm_z) if norm_x > 0 and norm_z > 0 else 1
    
    def edit_distance(self, s, t):
        """Compute Levenshtein edit distance between two strings."""
        m, n = len(s), len(t)
        dp = np.zeros((m + 1, n + 1))
        
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == t[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        
        return int(dp[m, n])
    
    def compute_distance_matrix(self, X, metric='euclidean'):
        """Compute pairwise distance matrix for a dataset."""
        if metric == 'euclidean':
            return squareform(pdist(X, metric='euclidean'))
        elif metric == 'manhattan':
            return squareform(pdist(X, metric='manhattan'))
        elif metric == 'cosine':
            return cosine_distances(X)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def classical_mds(self, D, k=None):
        """Perform classical MDS on distance matrix D."""
        n = D.shape[0]
        if k is None:
            k = n
        
        # Step 1: Double centering
        D_squared = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_squared @ H
        
        # Step 2: Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)
        
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Step 3: Reconstruction
        X_reconstructed = eigenvecs[:, :k] @ np.sqrt(np.diag(eigenvals[:k]))
        
        return X_reconstructed, eigenvals, eigenvecs
    
    def plot_distance_comparison(self, X, metrics=['euclidean', 'manhattan', 'cosine']):
        """Compare different distance measures on the same dataset."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            D = self.compute_distance_matrix(X, metric)
            sns.heatmap(D, ax=axes[i], cmap='viridis', square=True)
            axes[i].set_title(f'{metric.capitalize()} Distance')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('Sample Index')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_distance_distributions(self, X, metrics=['euclidean', 'manhattan', 'cosine']):
        """Analyze the distribution of distances for different metrics."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            D = self.compute_distance_matrix(X, metric)
            # Get upper triangular part (excluding diagonal)
            distances = D[np.triu_indices_from(D, k=1)]
            
            axes[i].hist(distances, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric.capitalize()} Distance Distribution')
            axes[i].set_xlabel('Distance')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(np.mean(distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distances):.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

def demonstrate_distance_measures():
    """Demonstrate various distance measures with examples."""
    dm = DistanceMeasures()
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 3)  # 50 samples, 3 features
    
    print("=== Distance Measures Demonstration ===\n")
    
    # Numerical distance examples
    x1, x2 = X[0], X[1]
    print(f"Sample points: x1 = {x1}, x2 = {x2}")
    print(f"Euclidean distance: {dm.euclidean_distance(x1, x2):.4f}")
    print(f"Manhattan distance: {dm.manhattan_distance(x1, x2):.4f}")
    print(f"Minkowski distance (p=3): {dm.minkowski_distance(x1, x2, p=3):.4f}")
    print(f"Chebyshev distance: {dm.chebyshev_distance(x1, x2):.4f}")
    print(f"Cosine distance: {dm.cosine_distance(x1, x2):.4f}")
    
    # Categorical distance examples
    set_a = {'apple', 'banana', 'cherry', 'date'}
    set_b = {'apple', 'banana', 'elderberry'}
    print(f"\nSet A: {set_a}")
    print(f"Set B: {set_b}")
    print(f"Jaccard distance: {dm.jaccard_distance(set_a, set_b):.4f}")
    
    # String distance examples
    s1, s2 = "karolin", "kathrin"
    print(f"\nString 1: '{s1}'")
    print(f"String 2: '{s2}'")
    print(f"Edit distance: {dm.edit_distance(s1, s2)}")
    
    # Distance matrix analysis
    print(f"\nComputing distance matrices for {X.shape[0]} samples...")
    dm.plot_distance_comparison(X)
    dm.analyze_distance_distributions(X)
    
    # MDS demonstration
    print("\n=== Multidimensional Scaling Demo ===")
    D = dm.compute_distance_matrix(X, 'euclidean')
    X_mds, eigenvals, eigenvecs = dm.classical_mds(D, k=2)
    
    print(f"Original data shape: {X.shape}")
    print(f"MDS reconstructed shape: {X_mds.shape}")
    print(f"Top 5 eigenvalues: {eigenvals[:5]}")
    
    # Plot MDS results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    plt.title('Original Data (First 2 Dimensions)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_mds[:, 0], X_mds[:, 1], alpha=0.7)
    plt.title('MDS Reconstruction (2D)')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    
    plt.tight_layout()
    plt.show()

def analyze_distance_properties():
    """Analyze properties of different distance measures."""
    dm = DistanceMeasures()
    
    # Generate data with different characteristics
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.randn(100, 3)
    
    # Data with outliers
    X_outliers = np.random.randn(100, 3)
    X_outliers[0] = [10, 10, 10]  # Add outlier
    
    # High-dimensional data
    X_high_dim = np.random.randn(50, 20)
    
    print("=== Distance Measure Properties Analysis ===\n")
    
    # Compare distance distributions
    datasets = {
        'Normal': X_normal,
        'With Outliers': X_outliers,
        'High Dimensional': X_high_dim
    }
    
    metrics = ['euclidean', 'manhattan', 'cosine']
    
    for name, X in datasets.items():
        print(f"\n{name} Data ({X.shape[0]} samples, {X.shape[1]} features):")
        
        for metric in metrics:
            D = dm.compute_distance_matrix(X, metric)
            distances = D[np.triu_indices_from(D, k=1)]
            
            print(f"  {metric.capitalize()}: mean={np.mean(distances):.3f}, "
                  f"std={np.std(distances):.3f}, "
                  f"min={np.min(distances):.3f}, "
                  f"max={np.max(distances):.3f}")
    
    # Visualize distance distributions
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(15, 12))
    
    for i, (name, X) in enumerate(datasets.items()):
        for j, metric in enumerate(metrics):
            D = dm.compute_distance_matrix(X, metric)
            distances = D[np.triu_indices_from(D, k=1)]
            
            axes[i, j].hist(distances, bins=30, alpha=0.7, edgecolor='black')
            axes[i, j].set_title(f'{name} - {metric.capitalize()}')
            axes[i, j].set_xlabel('Distance')
            axes[i, j].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def demonstrate_mds_applications():
    """Demonstrate MDS applications with different distance measures."""
    dm = DistanceMeasures()
    
    # Generate data with known structure
    np.random.seed(42)
    
    # Create data with 3 clusters
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([0, 5])
    
    X_clustered = np.vstack([cluster1, cluster2, cluster3])
    labels = np.repeat([0, 1, 2], 30)
    
    print("=== MDS Applications Demo ===\n")
    
    # Test different distance measures
    metrics = ['euclidean', 'manhattan', 'cosine']
    
    fig, axes = plt.subplots(2, len(metrics), figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        D = dm.compute_distance_matrix(X_clustered, metric)
        X_mds, eigenvals, eigenvecs = dm.classical_mds(D, k=2)
        
        # Original data
        axes[0, i].scatter(X_clustered[:, 0], X_clustered[:, 1], 
                          c=labels, cmap='viridis', alpha=0.7)
        axes[0, i].set_title(f'Original Data - {metric.capitalize()}')
        axes[0, i].set_xlabel('Feature 1')
        axes[0, i].set_ylabel('Feature 2')
        
        # MDS reconstruction
        axes[1, i].scatter(X_mds[:, 0], X_mds[:, 1], 
                          c=labels, cmap='viridis', alpha=0.7)
        axes[1, i].set_title(f'MDS Reconstruction - {metric.capitalize()}')
        axes[1, i].set_xlabel('MDS Dimension 1')
        axes[1, i].set_ylabel('MDS Dimension 2')
        
        print(f"{metric.capitalize()} MDS - Top 3 eigenvalues: {eigenvals[:3]}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Demonstrating Distance Measures...")
    demonstrate_distance_measures()
    
    print("\nAnalyzing Distance Properties...")
    analyze_distance_properties()
    
    print("\nDemonstrating MDS Applications...")
    demonstrate_mds_applications()
