# 6.2. K-means and K-medoids

## 6.2.1. Introduction to K-means Clustering

K-means is one of the most popular and widely-used clustering algorithms in machine learning and data science. It belongs to the family of **partitioning clustering algorithms** that divide a dataset into K non-overlapping clusters, where each data point belongs to exactly one cluster.

### Problem Formulation

Given a dataset $`X = \{x_1, x_2, \ldots, x_n\}`$ where each $`x_i \in \mathbb{R}^p`$ is a p-dimensional vector, the goal is to partition the data into K clusters $`C_1, C_2, \ldots, C_K`$ such that:

1. $`C_i \cap C_j = \emptyset`$ for $`i \neq j`$ (clusters are disjoint)
2. $`\bigcup_{i=1}^K C_i = X`$ (all points are assigned to clusters)
3. Points within the same cluster are similar to each other
4. Points in different clusters are dissimilar

### Mathematical Foundation

The K-means algorithm aims to minimize the **within-cluster sum of squares (WCSS)** or **inertia**:

```math
\Omega(z_{1:n}, m_{1:K}) = \sum_{i=1}^n \|x_i - m_{z_i}\|^2
```

where:
- $`z_i \in \{1, 2, \ldots, K\}`$ is the cluster assignment for data point $`x_i`$
- $`m_k \in \mathbb{R}^p`$ is the centroid (center) of cluster $`k`$
- $`\| \cdot \|`$ denotes the Euclidean norm

This can be rewritten as a double summation over clusters and observations:

```math
\Omega(z_{1:n}, m_{1:K}) = \sum_{k=1}^K \sum_{i: z_i=k} \|x_i - m_k\|^2
```

### Geometric Interpretation

The objective function measures the total squared Euclidean distance from each data point to its assigned cluster centroid. Minimizing this function is equivalent to finding the optimal partition that minimizes the total "spread" within clusters.

## 6.2.2. The K-means Algorithm

### Algorithm Overview

K-means is an **iterative optimization algorithm** that alternates between two steps:

1. **Assignment Step**: Assign each data point to the nearest centroid
2. **Update Step**: Recompute centroids as the mean of all points in each cluster

### Detailed Algorithm Steps

#### Step 0: Initialization
Choose K initial cluster centroids $`m_1^{(0)}, m_2^{(0)}, \ldots, m_K^{(0)}`$. Common initialization strategies include:

- **Random initialization**: Randomly select K data points as initial centroids
- **K-means++**: Probabilistic initialization that spreads initial centroids
- **Forgy method**: Randomly assign points to clusters and compute centroids

#### Step 1: Assignment (E-step)
For each data point $`x_i`$, assign it to the cluster with the nearest centroid:

```math
z_i^{(t+1)} = \arg\min_{k \in \{1,\ldots,K\}} \|x_i - m_k^{(t)}\|^2
```

This step minimizes the objective function with respect to cluster assignments while keeping centroids fixed.

#### Step 2: Update (M-step)
For each cluster $`k`$, update the centroid as the mean of all points assigned to that cluster:

```math
m_k^{(t+1)} = \frac{1}{|C_k^{(t+1)}|} \sum_{i: z_i^{(t+1)} = k} x_i
```

where $`C_k^{(t+1)} = \{x_i : z_i^{(t+1)} = k\}`$ is the set of points assigned to cluster $`k`$ at iteration $`t+1`$.

#### Convergence
Repeat Steps 1 and 2 until convergence, which occurs when:
- No data points change cluster assignments, OR
- Centroids stop moving significantly, OR
- Maximum number of iterations is reached

### Convergence Properties

**Theorem**: The K-means algorithm converges to a local minimum of the objective function.

**Proof Sketch**:
1. The assignment step can only decrease or maintain the objective function value
2. The update step (computing means) minimizes the objective function for fixed assignments
3. Since the objective function is bounded below by 0, the algorithm must converge

**Important Note**: K-means converges to a **local minimum**, not necessarily the global minimum. The final clustering depends heavily on the initial centroid positions.

### Computational Complexity

- **Time Complexity**: $`O(I \cdot n \cdot K \cdot p)`$ where:
  - $`I`$ = number of iterations
  - $`n`$ = number of data points
  - $`K`$ = number of clusters
  - $`p`$ = number of features
- **Space Complexity**: $`O(n \cdot p + K \cdot p)`$

## 6.2.3. Initialization Strategies

### Random Initialization
```python
def random_init(X, K):
    """Random initialization: randomly select K data points as centroids."""
    n = X.shape[0]
    indices = np.random.choice(n, K, replace=False)
    return X[indices]
```

### K-means++ Initialization
K-means++ improves upon random initialization by spreading initial centroids:

1. Choose first centroid uniformly at random
2. For each subsequent centroid:
   - Compute distances from each point to nearest existing centroid
   - Choose next centroid with probability proportional to squared distance

```python
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
```

## 6.2.4. Local Minima and Multiple Initializations

### The Local Minimum Problem

K-means can converge to suboptimal solutions due to poor initialization. Consider this example:

**Scenario**: 4 points in 2D space forming a rectangle
- Points: (0,0), (0,1), (2,0), (2,1)
- Desired: 2 clusters with points (0,0), (0,1) and (2,0), (2,1)
- Poor initialization: centroids at (0,0) and (0,1) → suboptimal clustering

### Solution: Multiple Initializations

Run K-means multiple times with different initializations and choose the best result:

```python
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
```

## 6.2.5. Dimension Reduction for K-means

### Computational Challenges

The computational cost of K-means scales with the number of features $`p`$. For high-dimensional data, this can be prohibitive. Dimension reduction techniques can help:

### Principal Component Analysis (PCA)

PCA reduces dimensionality while preserving variance:

```math
X_{\text{reduced}} = X \cdot W
```

where $`W \in \mathbb{R}^{p \times d}`$ contains the top $`d`$ principal components.

**Properties**:
- Preserves pairwise distances on average
- Captures data-specific patterns
- Computationally efficient

### Random Projection

Based on the Johnson-Lindenstrauss lemma, random projection preserves distances approximately:

```math
X_{\text{reduced}} = X \cdot R
```

where $`R \in \mathbb{R}^{p \times d}`$ is a random matrix with entries from $`N(0, 1/d)`$.

**Properties**:
- Data-agnostic projection matrix
- Less sensitive to original dimension
- May not capture data-specific patterns as well as PCA

### Implementation

```python
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
```

## 6.2.6. Alternative Distance Measures

### Beyond Euclidean Distance

K-means can be generalized to use other distance measures, but this requires modifications to the update step.

### Generalized Objective Function

```math
\Omega(z_{1:n}, m_{1:K}) = \sum_{k=1}^K \sum_{i: z_i=k} d(x_i, m_k)
```

where $`d(\cdot, \cdot)`$ is a general distance measure.

### Challenges with Non-Euclidean Distances

1. **Assignment Step**: Still straightforward - assign to nearest centroid
2. **Update Step**: Computing the "mean" becomes non-trivial

### Examples of Alternative Distance Measures

#### Manhattan Distance (L1)
For Manhattan distance, the optimal centroid is the **median** of cluster points:

```python
def manhattan_centroid(X_cluster):
    """Compute centroid for Manhattan distance (median)."""
    return np.median(X_cluster, axis=0)
```

#### Cosine Distance
For cosine distance, the optimal centroid is the **normalized mean**:

```python
def cosine_centroid(X_cluster):
    """Compute centroid for cosine distance."""
    mean_vec = np.mean(X_cluster, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec
```

#### Mixed Distance Measures
For data with mixed types (numerical + categorical):

```python
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
```

## 6.2.7. The K-medoids Algorithm

### Motivation

When using non-Euclidean distances, computing centroids can be computationally expensive or even impossible. K-medoids addresses this by restricting cluster centers to actual data points.

### Problem Formulation

Given a distance matrix $`D \in \mathbb{R}^{n \times n}`$ and number of clusters $`K`$, find:
- Cluster assignments $`z_1, z_2, \ldots, z_n`$
- Medoids (cluster centers) $`m_1, m_2, \ldots, m_K`$ where each $`m_k`$ is a data point

### Objective Function

```math
\Omega(z_{1:n}, m_{1:K}) = \sum_{k=1}^K \sum_{i: z_i=k} D_{i, m_k}
```

where $`D_{i, m_k}`$ is the distance between data point $`i`$ and medoid $`m_k`$.

### PAM (Partitioning Around Medoids) Algorithm

#### Step 1: Initialization
Randomly select K data points as initial medoids.

#### Step 2: Assignment
Assign each data point to the nearest medoid:

```math
z_i = \arg\min_{k \in \{1,\ldots,K\}} D_{i, m_k}
```

#### Step 3: Update (Swap Phase)
For each medoid $`m_k`$ and non-medoid point $`x_i`$:
1. Temporarily swap $`m_k`$ and $`x_i``
2. Compute total cost of new configuration
3. If cost decreases, make the swap permanent

```python
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
```

### Computational Complexity

- **Time Complexity**: $`O(I \cdot K \cdot (n-K) \cdot n)`$ where $`I`$ is number of iterations
- **Space Complexity**: $`O(n^2)`$ for storing distance matrix

### Advantages and Disadvantages

**Advantages**:
- Works with any distance measure
- More robust to outliers than K-means
- Medoids are actual data points (interpretable)

**Disadvantages**:
- Computationally more expensive than K-means
- Requires precomputed distance matrix
- May not scale well to large datasets

## 6.2.8. Python Implementation

```python
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

# Example usage and demonstration
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

if __name__ == "__main__":
    demonstrate_kmeans()
    demonstrate_kmedoids()
```

## 6.2.9. R Implementation

```r
# K-means and K-medoids Implementation in R
library(stats)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)

KMeansClustering <- setRefClass("KMeansClustering",
  fields = list(
    n_clusters = "numeric",
    max_iter = "numeric",
    tol = "numeric",
    n_init = "numeric",
    labels = "numeric",
    cluster_centers = "matrix",
    inertia = "numeric",
    n_iter = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_clusters = 3, max_iter = 300, tol = 1e-4, n_init = 10) {
      n_clusters <<- n_clusters
      max_iter <<- max_iter
      tol <<- tol
      n_init <<- n_init
    },
    
    kmeans_plus_plus_init = function(X) {
      n <- nrow(X)
      p <- ncol(X)
      centroids <- matrix(0, n_clusters, p)
      
      # Choose first centroid randomly
      centroids[1, ] <- X[sample(n, 1), ]
      
      for (k in 2:n_clusters) {
        # Compute distances to nearest centroid
        distances <- sapply(1:n, function(i) {
          min(sapply(1:(k-1), function(j) {
            sum((X[i, ] - centroids[j, ])^2)
          }))
        })
        
        # Choose next centroid with probability proportional to distance^2
        probs <- distances / sum(distances)
        cumprobs <- cumsum(probs)
        r <- runif(1)
        idx <- which(cumprobs >= r)[1]
        centroids[k, ] <- X[idx, ]
      }
      
      centroids
    },
    
    kmeans_single_run = function(X, initial_centroids) {
      n <- nrow(X)
      p <- ncol(X)
      centroids <- initial_centroids
      
      for (iteration in 1:max_iter) {
        old_centroids <- centroids
        
        # Assignment step
        distances <- sapply(1:n_clusters, function(k) {
          rowSums((X - matrix(centroids[k, ], n, p, byrow = TRUE))^2)
        })
        labels <- apply(distances, 1, which.min)
        
        # Update step
        for (k in 1:n_clusters) {
          if (sum(labels == k) > 0) {
            centroids[k, ] <- colMeans(X[labels == k, , drop = FALSE])
          }
        }
        
        # Check convergence
        if (max(sqrt(rowSums((centroids - old_centroids)^2))) < tol) {
          break
        }
      }
      
      # Compute final inertia
      inertia <- sum(sapply(1:n_clusters, function(k) {
        if (sum(labels == k) > 0) {
          sum(rowSums((X[labels == k, , drop = FALSE] - 
                       matrix(centroids[k, ], sum(labels == k), p, byrow = TRUE))^2))
        } else {
          0
        }
      }))
      
      list(labels = labels, centroids = centroids, inertia = inertia, n_iter = iteration)
    },
    
    fit = function(X) {
      best_inertia <- Inf
      best_labels <- NULL
      best_centroids <- NULL
      best_n_iter <- 0
      
      for (init in 1:n_init) {
        # Initialize centroids
        centroids <- kmeans_plus_plus_init(X)
        
        # Run single K-means
        result <- kmeans_single_run(X, centroids)
        
        # Update best result
        if (result$inertia < best_inertia) {
          best_inertia <- result$inertia
          best_labels <- result$labels
          best_centroids <- result$centroids
          best_n_iter <- result$n_iter
        }
      }
      
      labels <<- best_labels
      cluster_centers <<- best_centroids
      inertia <<- best_inertia
      n_iter <<- best_n_iter
      
      invisible(.self)
    },
    
    predict = function(X) {
      distances <- sapply(1:n_clusters, function(k) {
        rowSums((X - matrix(cluster_centers[k, ], nrow(X), ncol(X), byrow = TRUE))^2)
      })
      apply(distances, 1, which.min)
    },
    
    plot_clusters = function(X, title = "K-means Clustering") {
      df <- data.frame(
        x = X[, 1],
        y = X[, 2],
        cluster = factor(labels)
      )
      
      centroids_df <- data.frame(
        x = cluster_centers[, 1],
        y = cluster_centers[, 2],
        cluster = factor(1:n_clusters)
      )
      
      ggplot() +
        geom_point(data = df, aes(x = x, y = y, color = cluster), 
                   alpha = 0.7, size = 2) +
        geom_point(data = centroids_df, aes(x = x, y = y), 
                   color = "red", shape = 4, size = 4, stroke = 2) +
        labs(title = title, x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
    },
    
    evaluate_clustering = function(X) {
      metrics <- list()
      
      # Inertia
      metrics$inertia <- inertia
      
      # Silhouette score
      if (length(unique(labels)) > 1) {
        metrics$silhouette <- mean(silhouette(labels, dist(X))[, 3])
      } else {
        metrics$silhouette <- 0
      }
      
      # Number of iterations
      metrics$n_iterations <- n_iter
      
      # Cluster sizes
      cluster_counts <- table(labels)
      metrics$cluster_sizes <- as.list(cluster_counts)
      
      metrics
    }
  )
)

KMedoidsClustering <- setRefClass("KMedoidsClustering",
  fields = list(
    n_clusters = "numeric",
    max_iter = "numeric",
    labels = "numeric",
    medoids = "numeric",
    inertia = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_clusters = 3, max_iter = 300) {
      n_clusters <<- n_clusters
      max_iter <<- max_iter
    },
    
    pam_swap_phase = function(D, labels, medoids) {
      n <- nrow(D)
      K <- length(medoids)
      improved <- TRUE
      
      while (improved) {
        improved <- FALSE
        
        for (k in 1:K) {
          current_medoid <- medoids[k]
          
          # Try swapping with each non-medoid point
          for (i in 1:n) {
            if (i %in% medoids) next
            
            # Temporarily swap
            temp_medoids <- medoids
            temp_medoids[k] <- i
            
            # Compute new assignments and cost
            temp_labels <- apply(D[, temp_medoids], 1, which.min)
            temp_cost <- sum(sapply(1:n, function(j) {
              D[j, temp_medoids[temp_labels[j]]]
            }))
            
            # Current cost
            current_cost <- sum(sapply(1:n, function(j) {
              D[j, medoids[labels[j]]]
            }))
            
            # If improvement, make swap permanent
            if (temp_cost < current_cost) {
              medoids <- temp_medoids
              labels <- temp_labels
              improved <- TRUE
              break
            }
          }
        }
      }
      
      list(labels = labels, medoids = medoids)
    },
    
    fit = function(D) {
      n <- nrow(D)
      
      # Initialize medoids randomly
      medoids <- sample(n, n_clusters)
      
      for (iteration in 1:max_iter) {
        old_medoids <- medoids
        
        # Assignment step
        labels <- apply(D[, medoids], 1, which.min)
        
        # Swap step
        result <- pam_swap_phase(D, labels, medoids)
        labels <- result$labels
        medoids <- result$medoids
        
        # Check convergence
        if (all(medoids == old_medoids)) break
      }
      
      # Compute final cost
      inertia <- sum(sapply(1:n, function(i) {
        D[i, medoids[labels[i]]]
      }))
      
      labels <<- labels
      medoids <<- medoids
      inertia <<- inertia
      
      invisible(.self)
    }
  )
)

# Example usage and demonstration
demonstrate_kmeans <- function() {
  cat("=== K-means Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  n_samples <- 300
  
  # Create three clusters
  cluster1 <- matrix(rnorm(n_samples/3 * 2, mean = c(0, 0), sd = 1), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/3 * 2, mean = c(4, 4), sd = 1), ncol = 2)
  cluster3 <- matrix(rnorm(n_samples/3 * 2, mean = c(2, 6), sd = 1), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Test different numbers of clusters
  for (K in c(2, 3, 4, 5)) {
    cat("Testing K =", K, "clusters...\n")
    
    # Fit K-means
    kmeans <- KMeansClustering$new(n_clusters = K, n_init = 10)
    kmeans$fit(X)
    
    # Evaluate results
    metrics <- kmeans$evaluate_clustering(X)
    cat("  Inertia:", round(metrics$inertia, 2), "\n")
    cat("  Silhouette Score:", round(metrics$silhouette, 3), "\n")
    cat("  Iterations:", metrics$n_iterations, "\n")
    cat("  Cluster Sizes:", unlist(metrics$cluster_sizes), "\n\n")
    
    # Plot results
    print(kmeans$plot_clusters(X, paste("K-means with K=", K)))
  }
  
  # Compare with built-in kmeans
  cat("Comparing with built-in kmeans function...\n")
  builtin_kmeans <- kmeans(X, centers = 3, nstart = 10)
  cat("Built-in inertia:", round(builtin_kmeans$tot.withinss, 2), "\n")
  cat("Our inertia:", round(kmeans$inertia, 2), "\n")
}

demonstrate_kmedoids <- function() {
  cat("=== K-medoids Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  X <- matrix(rnorm(50 * 2), ncol = 2)
  
  # Compute distance matrix
  D <- as.matrix(dist(X))
  
  # Fit K-medoids
  kmedoids <- KMedoidsClustering$new(n_clusters = 3)
  kmedoids$fit(D)
  
  cat("Final cost:", round(kmedoids$inertia, 2), "\n")
  cat("Medoids:", kmedoids$medoids, "\n")
  cat("Cluster sizes:", table(kmedoids$labels), "\n")
  
  # Visualize results
  df <- data.frame(
    x = X[, 1],
    y = X[, 2],
    cluster = factor(kmedoids$labels)
  )
  
  centroids_df <- data.frame(
    x = X[kmedoids$medoids, 1],
    y = X[kmedoids$medoids, 2],
    cluster = factor(1:kmedoids$n_clusters)
  )
  
  p <- ggplot() +
    geom_point(data = df, aes(x = x, y = y, color = cluster), 
               alpha = 0.7, size = 2) +
    geom_point(data = centroids_df, aes(x = x, y = y), 
               color = "red", shape = 4, size = 4, stroke = 2) +
    labs(title = "K-medoids Clustering", x = "Feature 1", y = "Feature 2") +
    theme_minimal() +
    scale_color_viridis_d()
  
  print(p)
}

# Run demonstrations
demonstrate_kmeans()
demonstrate_kmedoids()
```

## 6.2.10. Summary and Best Practices

### Key Takeaways

1. **K-means is a local optimization algorithm** that converges to local minima
2. **Initialization matters** - use K-means++ for better results
3. **Multiple runs are essential** to find good solutions
4. **K-medoids is more robust** but computationally expensive
5. **Dimension reduction** can significantly improve performance

### Algorithm Selection Guidelines

**Use K-means when:**
- Data is numerical and Euclidean distance is appropriate
- Computational efficiency is important
- Data is well-separated and roughly spherical

**Use K-medoids when:**
- Working with non-Euclidean distances
- Robustness to outliers is important
- Interpretable cluster centers are needed

### Common Pitfalls

1. **Poor initialization**: Can lead to suboptimal local minima
2. **Wrong number of clusters**: Use elbow method or silhouette analysis
3. **Non-spherical clusters**: K-means assumes spherical clusters
4. **Scale sensitivity**: Standardize features before clustering
5. **Outliers**: Can significantly affect centroid positions

### Advanced Topics

- **Kernel K-means**: Extend to non-linear cluster boundaries
- **Fuzzy K-means**: Allow soft cluster assignments
- **Hierarchical K-means**: Combine with hierarchical clustering
- **Online K-means**: Process data in streaming fashion
- **Spectral clustering**: Use eigenvectors for clustering