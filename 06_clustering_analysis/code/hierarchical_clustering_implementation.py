"""
Hierarchical Clustering Implementation
====================================

This module provides comprehensive implementations of hierarchical clustering
methods, including various linkage criteria, dendrogram visualization, and
comparison tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns

class HierarchicalClustering:
    """Comprehensive hierarchical clustering implementation."""
    
    def __init__(self, method='complete', metric='euclidean'):
        """
        Initialize hierarchical clustering.
        
        Parameters:
        -----------
        method : str, default='complete'
            Linkage method: 'single', 'complete', 'average', 'ward'
        metric : str, default='euclidean'
            Distance metric for computing pairwise distances
        """
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.distance_matrix = None
        
    def fit(self, X):
        """
        Fit hierarchical clustering to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Compute distance matrix
        self.distance_matrix = pdist(X, metric=self.metric)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(self.distance_matrix, method=self.method)
        
        return self
    
    def get_clusters(self, n_clusters=None, height=None):
        """
        Extract clusters from the dendrogram.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to extract
        height : float, optional
            Height at which to cut the dendrogram
            
        Returns:
        --------
        labels : array
            Cluster labels for each sample
        """
        if n_clusters is not None:
            return fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')
        elif height is not None:
            return fcluster(self.linkage_matrix, t=height, criterion='distance')
        else:
            raise ValueError("Must specify either n_clusters or height")
    
    def plot_dendrogram(self, max_d=None, title=None):
        """
        Plot the dendrogram.
        
        Parameters:
        -----------
        max_d : float, optional
            Maximum distance for truncating the dendrogram
        title : str, optional
            Title for the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Create dendrogram
        dendrogram(
            self.linkage_matrix,
            max_d=max_d,
            leaf_rotation=90,
            leaf_font_size=10,
            show_leaf_counts=True
        )
        
        plt.title(title or f'Hierarchical Clustering Dendrogram ({self.method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    def cophenetic_correlation(self):
        """
        Compute cophenetic correlation coefficient.
        
        Returns:
        --------
        correlation : float
            Cophenetic correlation coefficient
        """
        c, coph_dists = cophenet(self.linkage_matrix, self.distance_matrix)
        return c
    
    def compare_linkage_methods(self, X, methods=['single', 'complete', 'average', 'ward']):
        """
        Compare different linkage methods.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        methods : list, default=['single', 'complete', 'average', 'ward']
            List of linkage methods to compare
            
        Returns:
        --------
        results : dict
            Dictionary containing results for each method
        """
        results = {}
        
        for method in methods:
            # Fit clustering
            hc = HierarchicalClustering(method=method)
            hc.fit(X)
            
            # Compute cophenetic correlation
            cophenetic_corr = hc.cophenetic_correlation()
            
            # Compute silhouette scores for different K
            silhouette_scores = []
            for k in range(2, min(11, len(X))):
                labels = hc.get_clusters(n_clusters=k)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
            
            results[method] = {
                'cophenetic_correlation': cophenetic_corr,
                'silhouette_scores': silhouette_scores,
                'linkage_matrix': hc.linkage_matrix
            }
        
        return results
    
    def plot_comparison(self, X, methods=['single', 'complete', 'average', 'ward']):
        """
        Plot comparison of different linkage methods.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        methods : list, default=['single', 'complete', 'average', 'ward']
            List of linkage methods to compare
        """
        results = self.compare_linkage_methods(X, methods)
        
        # Plot dendrograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, method in enumerate(methods):
            # Plot dendrogram
            hc = HierarchicalClustering(method=method)
            hc.fit(X)
            
            dendrogram(hc.linkage_matrix, ax=axes[i], leaf_rotation=90, leaf_font_size=8)
            axes[i].set_title(f'{method.capitalize()} Linkage')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('Distance')
        
        plt.tight_layout()
        plt.show()
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        for method in methods:
            scores = results[method]['silhouette_scores']
            plt.plot(range(2, len(scores) + 2), scores, marker='o', label=method.capitalize())
        
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for Different Linkage Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print cophenetic correlations
        print("Cophenetic Correlation Coefficients:")
        for method in methods:
            print(f"{method.capitalize()}: {results[method]['cophenetic_correlation']:.4f}")

def demonstrate_hierarchical_clustering():
    """
    Demonstrate hierarchical clustering with various examples.
    """
    print("=== Hierarchical Clustering Demonstration ===\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Create three well-separated clusters
    cluster1 = np.random.normal([0, 0], [1, 1], (n_samples//3, 2))
    cluster2 = np.random.normal([6, 6], [1, 1], (n_samples//3, 2))
    cluster3 = np.random.normal([3, 9], [1, 1], (n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"Data shape: {X.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}\n")
    
    # Initialize hierarchical clustering
    hc = HierarchicalClustering(method='complete')
    hc.fit(X)
    
    # Plot dendrogram
    print("Plotting dendrogram...")
    hc.plot_dendrogram(title="Complete Linkage Dendrogram")
    
    # Extract clusters
    labels_3 = hc.get_clusters(n_clusters=3)
    labels_5 = hc.get_clusters(n_clusters=5)
    
    # Visualize cluster assignments
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels_3, cmap='viridis', alpha=0.7)
    axes[0].set_title('3 Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels_5, cmap='viridis', alpha=0.7)
    axes[1].set_title('5 Clusters')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    # Compare linkage methods
    print("\nComparing different linkage methods...")
    hc.plot_comparison(X)
    
    # Cophenetic correlation
    print(f"\nCophenetic correlation: {hc.cophenetic_correlation():.4f}")
    
    # Evaluate clustering quality
    silhouette_3 = silhouette_score(X, labels_3)
    silhouette_5 = silhouette_score(X, labels_5)
    print(f"Silhouette score (3 clusters): {silhouette_3:.4f}")
    print(f"Silhouette score (5 clusters): {silhouette_5:.4f}")
    
    return hc, X, labels_3, labels_5

def analyze_linkage_methods():
    """
    Analyze the behavior of different linkage methods on various data types.
    """
    print("=== Linkage Methods Analysis ===\n")
    
    np.random.seed(42)
    
    # Test 1: Well-separated clusters
    print("Test 1: Well-separated clusters")
    cluster1 = np.random.normal([0, 0], [0.5, 0.5], (50, 2))
    cluster2 = np.random.normal([4, 4], [0.5, 0.5], (50, 2))
    cluster3 = np.random.normal([0, 4], [0.5, 0.5], (50, 2))
    X_well_separated = np.vstack([cluster1, cluster2, cluster3])
    
    hc_well = HierarchicalClustering(method='complete')
    hc_well.fit(X_well_separated)
    print(f"Cophenetic correlation: {hc_well.cophenetic_correlation():.4f}")
    
    # Test 2: Overlapping clusters
    print("\nTest 2: Overlapping clusters")
    cluster1 = np.random.normal([0, 0], [1.5, 1.5], (50, 2))
    cluster2 = np.random.normal([2, 2], [1.5, 1.5], (50, 2))
    cluster3 = np.random.normal([0, 2], [1.5, 1.5], (50, 2))
    X_overlapping = np.vstack([cluster1, cluster2, cluster3])
    
    hc_overlapping = HierarchicalClustering(method='complete')
    hc_overlapping.fit(X_overlapping)
    print(f"Cophenetic correlation: {hc_overlapping.cophenetic_correlation():.4f}")
    
    # Test 3: Chain-like structure
    print("\nTest 3: Chain-like structure")
    t = np.linspace(0, 4*np.pi, 100)
    X_chain = np.column_stack([np.cos(t) + np.random.normal(0, 0.1, 100),
                              np.sin(t) + np.random.normal(0, 0.1, 100)])
    
    hc_chain = HierarchicalClustering(method='single')
    hc_chain.fit(X_chain)
    print(f"Cophenetic correlation: {hc_chain.cophenetic_correlation():.4f}")
    
    # Visualize all test cases
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Well-separated clusters
    axes[0, 0].scatter(X_well_separated[:, 0], X_well_separated[:, 1], alpha=0.7)
    axes[0, 0].set_title('Well-separated Clusters')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Overlapping clusters
    axes[0, 1].scatter(X_overlapping[:, 0], X_overlapping[:, 1], alpha=0.7)
    axes[0, 1].set_title('Overlapping Clusters')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # Chain-like structure
    axes[0, 2].scatter(X_chain[:, 0], X_chain[:, 1], alpha=0.7)
    axes[0, 2].set_title('Chain-like Structure')
    axes[0, 2].set_xlabel('Feature 1')
    axes[0, 2].set_ylabel('Feature 2')
    
    # Dendrograms
    dendrogram(hc_well.linkage_matrix, ax=axes[1, 0], leaf_rotation=90, leaf_font_size=8)
    axes[1, 0].set_title('Well-separated Dendrogram')
    
    dendrogram(hc_overlapping.linkage_matrix, ax=axes[1, 1], leaf_rotation=90, leaf_font_size=8)
    axes[1, 1].set_title('Overlapping Dendrogram')
    
    dendrogram(hc_chain.linkage_matrix, ax=axes[1, 2], leaf_rotation=90, leaf_font_size=8)
    axes[1, 2].set_title('Chain Dendrogram')
    
    plt.tight_layout()
    plt.show()

def demonstrate_cluster_extraction():
    """
    Demonstrate different ways to extract clusters from hierarchical clustering.
    """
    print("=== Cluster Extraction Demonstration ===\n")
    
    # Generate data
    np.random.seed(42)
    n_samples = 80
    
    # Create clusters with different densities
    cluster1 = np.random.normal([0, 0], [0.8, 0.8], (n_samples//4, 2))
    cluster2 = np.random.normal([4, 0], [0.8, 0.8], (n_samples//4, 2))
    cluster3 = np.random.normal([2, 4], [1.2, 1.2], (n_samples//2, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Fit hierarchical clustering
    hc = HierarchicalClustering(method='ward')
    hc.fit(X)
    
    # Extract clusters at different levels
    labels_2 = hc.get_clusters(n_clusters=2)
    labels_3 = hc.get_clusters(n_clusters=3)
    labels_4 = hc.get_clusters(n_clusters=4)
    
    # Extract clusters at different heights
    height_1 = 2.0
    height_2 = 3.5
    labels_h1 = hc.get_clusters(height=height_1)
    labels_h2 = hc.get_clusters(height=height_2)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Number-based extraction
    scatter1 = axes[0, 0].scatter(X[:, 0], X[:, 1], c=labels_2, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('2 Clusters')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    scatter2 = axes[0, 1].scatter(X[:, 0], X[:, 1], c=labels_3, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('3 Clusters')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    scatter3 = axes[0, 2].scatter(X[:, 0], X[:, 1], c=labels_4, cmap='viridis', alpha=0.7)
    axes[0, 2].set_title('4 Clusters')
    axes[0, 2].set_xlabel('Feature 1')
    axes[0, 2].set_ylabel('Feature 2')
    plt.colorbar(scatter3, ax=axes[0, 2])
    
    # Height-based extraction
    scatter4 = axes[1, 0].scatter(X[:, 0], X[:, 1], c=labels_h1, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title(f'Height = {height_1}')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    plt.colorbar(scatter4, ax=axes[1, 0])
    
    scatter5 = axes[1, 1].scatter(X[:, 0], X[:, 1], c=labels_h2, cmap='viridis', alpha=0.7)
    axes[1, 1].set_title(f'Height = {height_2}')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    plt.colorbar(scatter5, ax=axes[1, 1])
    
    # Dendrogram with cut lines
    dendrogram(hc.linkage_matrix, ax=axes[1, 2], leaf_rotation=90, leaf_font_size=8)
    axes[1, 2].axhline(y=height_1, color='red', linestyle='--', alpha=0.7, label=f'Height = {height_1}')
    axes[1, 2].axhline(y=height_2, color='orange', linestyle='--', alpha=0.7, label=f'Height = {height_2}')
    axes[1, 2].set_title('Dendrogram with Cut Lines')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    print("Cluster Statistics:")
    print(f"2 clusters: {len(np.unique(labels_2))} unique labels")
    print(f"3 clusters: {len(np.unique(labels_3))} unique labels")
    print(f"4 clusters: {len(np.unique(labels_4))} unique labels")
    print(f"Height {height_1}: {len(np.unique(labels_h1))} unique labels")
    print(f"Height {height_2}: {len(np.unique(labels_h2))} unique labels")

if __name__ == "__main__":
    print("Demonstrating Hierarchical Clustering...")
    
    # Basic demonstration
    hc, X, labels_3, labels_5 = demonstrate_hierarchical_clustering()
    
    # Analyze linkage methods
    analyze_linkage_methods()
    
    # Demonstrate cluster extraction
    demonstrate_cluster_extraction()
