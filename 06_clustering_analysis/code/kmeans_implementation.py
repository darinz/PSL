"""
K-means and K-medoids Implementation
===================================

This module provides comprehensive implementations of K-means and K-medoids
clustering algorithms with various enhancements and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

class KMeansClustering:
    """Comprehensive K-means implementation with various enhancements."""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = None
    
    def fit(self, X):
        """Fit K-means to the data."""
        best_inertia = float('inf')
        best_labels = None
        best_centers = None
        best_n_iter = 0
        
        for init in range(self.n_init):
            # Initialize centroids
            centroids = self._kmeans_plus_plus_init(X)
            
            # Run single K-means
            labels, centers, inertia, n_iter = self._kmeans_single_run(X, centroids)
            
            # Update best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
                best_n_iter = n_iter
        
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def _kmeans_plus_plus_init(self, X):
        """K-means++ initialization."""
        n, p = X.shape
        centroids = np.zeros((self.n_clusters, p))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n)]
        
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.min([np.sum((X - centroids[i])**2, axis=1) 
                               for i in range(k)], axis=0)
            
            # Choose next centroid with probability proportional to distance^2
            probs = distances / distances.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.random()
            idx = np.where(cumprobs >= r)[0][0]
            centroids[k] = X[idx]
        
        return centroids
    
    def _kmeans_single_run(self, X, initial_centroids):
        """Single run of K-means algorithm."""
        n, p = X.shape
        centroids = initial_centroids.copy()
        
        for iteration in range(self.max_iter):
            old_centroids = centroids.copy()
            
            # Assignment step
            distances = np.array([np.sum((X - centroids[k])**2, axis=1) 
                                 for k in range(self.n_clusters)])
            labels = np.argmin(distances, axis=0)
            
            # Update step
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    centroids[k] = np.mean(X[labels == k], axis=0)
            
            # Check convergence
            if np.max(np.linalg.norm(centroids - old_centroids, axis=1)) < self.tol:
                break
        
        # Compute final inertia
        inertia = sum(np.sum((X[labels == k] - centroids[k])**2) 
                     for k in range(self.n_clusters))
        
        return labels, centroids, inertia, iteration + 1
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        distances = np.array([np.sum((X - self.cluster_centers_[k])**2, axis=1) 
                             for k in range(self.n_clusters)])
        return np.argmin(distances, axis=0)
    
    def plot_clusters(self, X, title="K-means Clustering"):
        """Visualize clustering results."""
        plt.figure(figsize=(10, 8))
        
        # Plot data points colored by cluster
        scatter = plt.scatter(X[:, 0], X[:, 1], c=self.labels_, 
                             cmap='viridis', alpha=0.7, s=50)
        
        # Plot cluster centers
        plt.scatter(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_clustering(self, X):
        """Evaluate clustering quality using multiple metrics."""
        metrics = {}
        
        # Inertia (within-cluster sum of squares)
        metrics['inertia'] = self.inertia_
        
        # Silhouette score
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette'] = silhouette_score(X, self.labels_)
        else:
            metrics['silhouette'] = 0
        
        # Number of iterations
        metrics['n_iterations'] = self.n_iter_
        
        # Cluster sizes
        unique, counts = np.unique(self.labels_, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique, counts))
        
        return metrics

class KMedoidsClustering:
    """K-medoids implementation using PAM algorithm."""
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.medoids_ = None
        self.inertia_ = None
    
    def fit(self, D):
        """Fit K-medoids using distance matrix D."""
        n = D.shape[0]
        np.random.seed(self.random_state)
        
        # Initialize medoids randomly
        medoids = np.random.choice(n, self.n_clusters, replace=False)
        
        for iteration in range(self.max_iter):
            old_medoids = medoids.copy()
            
            # Assignment step
            labels = np.argmin(D[:, medoids], axis=1)
            
            # Swap step
            labels, medoids = self._pam_swap_phase(D, labels, medoids)
            
            # Check convergence
            if np.array_equal(medoids, old_medoids):
                break
        
        # Compute final cost
        inertia = sum(D[i, medoids[labels[i]]] for i in range(n))
        
        self.labels_ = labels
        self.medoids_ = medoids
        self.inertia_ = inertia
        
        return self
    
    def _pam_swap_phase(self, D, labels, medoids):
        """PAM swap phase implementation."""
        n, K = D.shape[0], len(medoids)
        improved = True
        
        while improved:
            improved = False
            
            for k in range(K):
                current_medoid = medoids[k]
                
                # Try swapping with each non-medoid point
                for i in range(n):
                    if i in medoids:
                        continue
                    
                    # Temporarily swap
                    temp_medoids = medoids.copy()
                    temp_medoids[k] = i
                    
                    # Compute new assignments and cost
                    temp_labels = np.argmin(D[:, temp_medoids], axis=1)
                    temp_cost = sum(D[j, temp_medoids[temp_labels[j]]] 
                                   for j in range(n))
                    
                    # Current cost
                    current_cost = sum(D[j, medoids[labels[j]]] 
                                       for j in range(n))
                    
                    # If improvement, make swap permanent
                    if temp_cost < current_cost:
                        medoids = temp_medoids
                        labels = temp_labels
                        improved = True
                        break
        
        return labels, medoids

def random_init(X, K):
    """Random initialization: randomly select K data points as centroids."""
    n = X.shape[0]
    indices = np.random.choice(n, K, replace=False)
    return X[indices]

def kmeans_plus_plus_init(X, K):
    """K-means++ initialization for better initial centroids."""
    n, p = X.shape
    centroids = np.zeros((K, p))
    
    # Choose first centroid randomly
    centroids[0] = X[np.random.randint(n)]
    
    for k in range(1, K):
        # Compute distances to nearest centroid
        distances = np.min([np.sum((X - centroids[i])**2, axis=1) 
                           for i in range(k)], axis=0)
        
        # Choose next centroid with probability proportional to distance^2
        probs = distances / distances.sum()
        cumprobs = np.cumsum(probs)
        r = np.random.random()
        idx = np.where(cumprobs >= r)[0][0]
        centroids[k] = X[idx]
    
    return centroids

def kmeans_multiple_runs(X, K, n_runs=10):
    """Run K-means multiple times and return best clustering."""
    best_inertia = float('inf')
    best_labels = None
    best_centroids = None
    
    for run in range(n_runs):
        # Initialize centroids
        centroids = kmeans_plus_plus_init(X, K)
        
        # Run K-means
        labels, centroids, inertia = kmeans_single_run(X, K, centroids)
        
        # Update best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    
    return best_labels, best_centroids, best_inertia

def kmeans_single_run(X, K, initial_centroids):
    """Single run of K-means algorithm."""
    n, p = X.shape
    centroids = initial_centroids.copy()
    max_iter = 300
    tol = 1e-4
    
    for iteration in range(max_iter):
        old_centroids = centroids.copy()
        
        # Assignment step
        distances = np.array([np.sum((X - centroids[k])**2, axis=1) 
                             for k in range(K)])
        labels = np.argmin(distances, axis=0)
        
        # Update step
        for k in range(K):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        
        # Check convergence
        if np.max(np.linalg.norm(centroids - old_centroids, axis=1)) < tol:
            break
    
    # Compute final inertia
    inertia = sum(np.sum((X[labels == k] - centroids[k])**2) 
                 for k in range(K))
    
    return labels, centroids, inertia

def kmeans_with_dimension_reduction(X, K, method='pca', d=None):
    """K-means with dimension reduction preprocessing."""
    if d is None:
        d = min(K + 1, X.shape[1])  # Rule of thumb
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=d)
    elif method == 'random':
        from sklearn.random_projection import GaussianRandomProjection
        reducer = GaussianRandomProjection(n_components=d)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reduce dimensions
    X_reduced = reducer.fit_transform(X)
    
    # Run K-means on reduced data
    labels, centroids_reduced, inertia = kmeans_multiple_runs(X_reduced, K)
    
    # Transform centroids back to original space
    centroids = reducer.inverse_transform(centroids_reduced)
    
    return labels, centroids, inertia, reducer

def manhattan_centroid(X_cluster):
    """Compute centroid for Manhattan distance (median)."""
    return np.median(X_cluster, axis=0)

def cosine_centroid(X_cluster):
    """Compute centroid for cosine distance."""
    mean_vec = np.mean(X_cluster, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec

def mixed_distance(x, y, weights=[0.4, 0.6]):
    """Mixed distance: L1 for numerical, Hamming for categorical."""
    numerical_dist = np.sum(np.abs(x[:2] - y[:2]))  # First 2 features
    categorical_dist = np.sum(x[2:] != y[2:])       # Remaining features
    return weights[0] * numerical_dist + weights[1] * categorical_dist

def mixed_centroid(X_cluster):
    """Compute centroid for mixed distance measure."""
    # Numerical features: median
    numerical_centroid = np.median(X_cluster[:, :2], axis=0)
    
    # Categorical features: mode
    categorical_centroid = []
    for j in range(2, X_cluster.shape[1]):
        values, counts = np.unique(X_cluster[:, j], return_counts=True)
        mode_idx = np.argmax(counts)
        categorical_centroid.append(values[mode_idx])
    
    return np.concatenate([numerical_centroid, categorical_centroid])

def pam_swap_phase(D, labels, medoids):
    """PAM swap phase: try swapping medoids with non-medoids."""
    n, K = D.shape[0], len(medoids)
    improved = True
    
    while improved:
        improved = False
        
        for k in range(K):
            current_medoid = medoids[k]
            
            # Try swapping with each non-medoid point
            for i in range(n):
                if i in medoids:
                    continue
                
                # Temporarily swap
                temp_medoids = medoids.copy()
                temp_medoids[k] = i
                
                # Compute new assignments and cost
                temp_labels = np.argmin(D[:, temp_medoids], axis=1)
                temp_cost = sum(D[j, temp_medoids[temp_labels[j]]] 
                               for j in range(n))
                
                # Current cost
                current_cost = sum(D[j, medoids[labels[j]]] 
                                  for j in range(n))
                
                # If improvement, make swap permanent
                if temp_cost < current_cost:
                    medoids = temp_medoids
                    labels = temp_labels
                    improved = True
                    break
    
    return labels, medoids

def demonstrate_kmeans():
    """Demonstrate K-means clustering with various examples."""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    
    # Create three clusters
    cluster1 = np.random.normal([0, 0], [1, 1], (n_samples//3, 2))
    cluster2 = np.random.normal([4, 4], [1, 1], (n_samples//3, 2))
    cluster3 = np.random.normal([2, 6], [1, 1], (n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("=== K-means Clustering Demonstration ===\n")
    
    # Test different numbers of clusters
    for K in [2, 3, 4, 5]:
        print(f"Testing K = {K} clusters...")
        
        # Fit K-means
        kmeans = KMeansClustering(n_clusters=K, n_init=10)
        kmeans.fit(X)
        
        # Evaluate results
        metrics = kmeans.evaluate_clustering(X)
        print(f"  Inertia: {metrics['inertia']:.2f}")
        print(f"  Silhouette Score: {metrics['silhouette']:.3f}")
        print(f"  Iterations: {metrics['n_iterations']}")
        print(f"  Cluster Sizes: {metrics['cluster_sizes']}")
        print()
        
        # Plot results
        kmeans.plot_clusters(X, f"K-means with K={K}")
    
    # Compare with sklearn implementation
    print("Comparing with sklearn implementation...")
    sklearn_kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    sklearn_kmeans.fit(X)
    
    print(f"Sklearn inertia: {sklearn_kmeans.inertia_:.2f}")
    print(f"Our inertia: {kmeans.inertia_:.2f}")
    print(f"Results match: {abs(sklearn_kmeans.inertia_ - kmeans.inertia_) < 1e-6}")

def demonstrate_kmedoids():
    """Demonstrate K-medoids clustering."""
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    
    # Compute distance matrix
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    
    print("=== K-medoids Clustering Demonstration ===\n")
    
    # Fit K-medoids
    kmedoids = KMedoidsClustering(n_clusters=3, random_state=42)
    kmedoids.fit(D)
    
    print(f"Final cost: {kmedoids.inertia_:.2f}")
    print(f"Medoids: {kmedoids.medoids_}")
    print(f"Cluster sizes: {dict(zip(*np.unique(kmedoids.labels_, return_counts=True)))}")
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmedoids.labels_, 
                         cmap='viridis', alpha=0.7, s=50)
    plt.scatter(X[kmedoids.medoids_, 0], X[kmedoids.medoids_, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Medoids')
    plt.title("K-medoids Clustering")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter)
    plt.show()

def analyze_initialization_methods():
    """Compare different initialization methods."""
    np.random.seed(42)
    
    # Generate challenging data
    n_samples = 200
    cluster1 = np.random.normal([0, 0], [0.5, 0.5], (n_samples//2, 2))
    cluster2 = np.random.normal([3, 3], [0.5, 0.5], (n_samples//2, 2))
    X = np.vstack([cluster1, cluster2])
    
    print("=== Initialization Method Comparison ===\n")
    
    # Test random initialization
    inertias_random = []
    for _ in range(20):
        centroids = random_init(X, 2)
        labels, _, inertia = kmeans_single_run(X, 2, centroids)
        inertias_random.append(inertia)
    
    # Test K-means++ initialization
    inertias_plus_plus = []
    for _ in range(20):
        centroids = kmeans_plus_plus_init(X, 2)
        labels, _, inertia = kmeans_single_run(X, 2, centroids)
        inertias_plus_plus.append(inertia)
    
    print(f"Random initialization - Mean inertia: {np.mean(inertias_random):.2f}")
    print(f"Random initialization - Std inertia: {np.std(inertias_random):.2f}")
    print(f"K-means++ initialization - Mean inertia: {np.mean(inertias_plus_plus):.2f}")
    print(f"K-means++ initialization - Std inertia: {np.std(inertias_plus_plus):.2f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([inertias_random, inertias_plus_plus], 
                labels=['Random', 'K-means++'])
    plt.ylabel('Inertia')
    plt.title('Initialization Method Comparison')
    plt.grid(True, alpha=0.3)
    plt.show()

def demonstrate_dimension_reduction():
    """Demonstrate K-means with dimension reduction."""
    np.random.seed(42)
    
    # Generate high-dimensional data
    n_samples = 300
    n_features = 20
    
    # Create 3 clusters in high-dimensional space
    cluster1 = np.random.normal([0] * n_features, [1] * n_features, (n_samples//3, n_features))
    cluster2 = np.random.normal([3] * n_features, [1] * n_features, (n_samples//3, n_features))
    cluster3 = np.random.normal([6] * n_features, [1] * n_features, (n_samples//3, n_features))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("=== Dimension Reduction for K-means ===\n")
    
    # Compare different methods
    methods = ['pca', 'random']
    
    for method in methods:
        print(f"Testing {method.upper()} dimension reduction...")
        
        labels, centroids, inertia, reducer = kmeans_with_dimension_reduction(X, 3, method=method, d=3)
        
        print(f"  Reduced dimensions: {reducer.n_components_}")
        print(f"  Final inertia: {inertia:.2f}")
        print(f"  Explained variance (PCA): {getattr(reducer, 'explained_variance_ratio_', None)}")
        print()

if __name__ == "__main__":
    print("Demonstrating K-means Clustering...")
    demonstrate_kmeans()
    
    print("\nDemonstrating K-medoids Clustering...")
    demonstrate_kmedoids()
    
    print("\nAnalyzing Initialization Methods...")
    analyze_initialization_methods()
    
    print("\nDemonstrating Dimension Reduction...")
    demonstrate_dimension_reduction()
