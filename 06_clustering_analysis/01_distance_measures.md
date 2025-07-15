# 6.1. Distance Measures

## 6.1.1. Introduction to Distance Measures

In clustering analysis, the fundamental objective is to group similar objects together while separating dissimilar ones. The choice of distance or similarity measure is crucial as it directly influences how "similarity" is quantified and, consequently, the quality of the resulting clusters.

### Mathematical Definition of Distance

A distance measure $`d(x, z)`$ is a function that quantifies the dissimilarity between two points $`x`$ and $`z`$ in a metric space. For a function to be considered a proper distance metric, it must satisfy the following four axioms:

1. **Non-negativity**: $`d(x, z) \geq 0`$ for all $`x, z`$
2. **Identity of indiscernibles**: $`d(x, z) = 0`$ if and only if $`x = z`$
3. **Symmetry**: $`d(x, z) = d(z, x)`$ for all $`x, z`$
4. **Triangle inequality**: $`d(x, y) \leq d(x, z) + d(z, y)`$ for all $`x, y, z`$

The triangle inequality ensures that the distance between two points represents the shortest possible path, preventing counterintuitive situations where going through an intermediate point could be shorter than the direct path.

### Types of Distance Measures

Distance measures can be broadly categorized based on the data type they're designed for:

- **Numerical data**: Euclidean, Manhattan, Minkowski, Chebyshev distances
- **Categorical data**: Hamming, Jaccard, Dice distances  
- **Mixed data**: Gower's distance, Mahalanobis distance
- **Text data**: Cosine, Jaccard, Edit distances

## 6.1.2. Numerical Distance Measures

### Euclidean Distance (L2 Norm)

The most commonly used distance measure for numerical data is the Euclidean distance, which represents the "ordinary" straight-line distance between two points.

```math
d_{\text{Euclidean}}(x, z) = \sqrt{\sum_{i=1}^{p} (x_i - z_i)^2}
```

**Properties:**
- Invariant to rotation and translation
- Sensitive to scale differences between features
- Assumes all features are equally important
- Computationally efficient

**Geometric interpretation**: The Euclidean distance represents the length of the straight line connecting two points in p-dimensional space.

### Manhattan Distance (L1 Norm)

Also known as the "city block" or "taxicab" distance, it measures distance as if you could only move along the axes.

```math
d_{\text{Manhattan}}(x, z) = \sum_{i=1}^{p} |x_i - z_i|
```

**Properties:**
- Less sensitive to outliers than Euclidean distance
- Useful when movement is constrained to grid-like paths
- Often preferred in high-dimensional spaces due to robustness

### Minkowski Distance (Lp Norm)

A generalization of both Euclidean and Manhattan distances:

```math
d_{\text{Minkowski}}(x, z) = \left(\sum_{i=1}^{p} |x_i - z_i|^p\right)^{1/p}
```

**Special cases:**
- $`p = 1`$: Manhattan distance
- $`p = 2`$: Euclidean distance
- $`p \to \infty`$: Chebyshev distance (L∞ norm)

### Chebyshev Distance (L∞ Norm)

Also called the "maximum metric" or "chessboard distance":

```math
d_{\text{Chebyshev}}(x, z) = \max_{i=1,\ldots,p} |x_i - z_i|
```

**Properties:**
- Measures the maximum difference along any single dimension
- Useful when the worst-case difference is most important
- Common in game theory and optimization problems

## 6.1.3. Categorical Distance Measures

### Hamming Distance

The Hamming distance counts the number of positions at which corresponding elements differ:

```math
d_{\text{Hamming}}(x, z) = \sum_{i=1}^{p} \mathbb{I}(x_i \neq z_i)
```

where $`\mathbb{I}(\cdot)`$ is the indicator function.

**Applications:**
- DNA sequence analysis
- Error detection in binary codes
- Text similarity for strings of equal length

**Example:**
- Strings: "karolin" vs "kathrin" → Hamming distance = 3
- Binary: 1011101 vs 1001001 → Hamming distance = 2

### Jaccard Distance

For set-based data, the Jaccard distance measures dissimilarity between two sets:

```math
d_{\text{Jaccard}}(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cup B| - |A \cap B|}{|A \cup B|}
```

**Properties:**
- Ranges from 0 (identical sets) to 1 (disjoint sets)
- Useful for text analysis, recommendation systems
- Handles sets of different sizes naturally

**Example:**
- Set A: {A, C, D, E}
- Set B: {A, D, E}
- Intersection: {A, D, E} (size 3)
- Union: {A, C, D, E} (size 4)
- Jaccard distance: $`1 - \frac{3}{4} = \frac{1}{4}`$

### Dice Distance

Similar to Jaccard but gives more weight to common elements:

```math
d_{\text{Dice}}(A, B) = 1 - \frac{2|A \cap B|}{|A| + |B|}
```

## 6.1.4. Text and Vector Distance Measures

### Cosine Distance

Measures the cosine of the angle between two vectors:

```math
d_{\text{Cosine}}(x, z) = 1 - \frac{\sum_{i=1}^{p} x_i z_i}{\sqrt{\sum_{i=1}^{p} x_i^2} \sqrt{\sum_{i=1}^{p} z_i^2}} = 1 - \cos(\theta)
```

**Properties:**
- Invariant to vector magnitude (only direction matters)
- Ranges from 0 (parallel vectors) to 2 (opposite vectors)
- Excellent for text analysis and document similarity
- Handles high-dimensional sparse data well

### Edit Distance (Levenshtein Distance)

The minimum number of single-character edits required to change one string into another:

```math
d_{\text{Edit}}(s, t) = \min\{d_{i,j}\}
```

where $`d_{i,j}`$ is computed using dynamic programming:

```math
d_{i,j} = \begin{cases}
\max(i, j) & \text{if } \min(i, j) = 0 \\
\min\begin{cases}
d_{i-1, j} + 1 & \text{(deletion)} \\
d_{i, j-1} + 1 & \text{(insertion)} \\
d_{i-1, j-1} + \mathbb{I}(s_i \neq t_j) & \text{(substitution)}
\end{cases} & \text{otherwise}
\end{cases}
```

## 6.1.5. Distance Matrix Computation

Given a data matrix $`X \in \mathbb{R}^{n \times p}`$, we can compute the pairwise distance matrix $`D \in \mathbb{R}^{n \times n}`$:

```math
D_{ij} = d(x_i, x_j)
```

**Properties of D:**
- Symmetric: $`D_{ij} = D_{ji}`$
- Zero diagonal: $`D_{ii} = 0`$
- Non-negative: $`D_{ij} \geq 0`$

## 6.1.6. Multidimensional Scaling (MDS)

### Problem Statement

Given an $`n \times n`$ distance matrix $`D`$, can we reconstruct the original data points $`x_1, \ldots, x_n`$ in some coordinate system?

**Key insight**: Distances are invariant to translation, rotation, and reflection. Therefore, we can only recover the data up to these transformations.

### Classical MDS Algorithm

#### Step 1: Double Centering Transformation

Transform the squared distance matrix $`D^{(2)}`$ (where $`D_{ij}^{(2)} = D_{ij}^2`$):

```math
\tilde{D} = -\frac{1}{2} \left(I - \frac{1}{n}11^T\right) D^{(2)} \left(I - \frac{1}{n}11^T\right)
```

where $`I`$ is the identity matrix and $`1`$ is a vector of ones.

**Intuition**: This transformation centers the data and converts squared distances to inner products.

#### Step 2: Eigendecomposition

Decompose $`\tilde{D}`$:

```math
\tilde{D} = U \Lambda U^T
```

where $`U`$ contains eigenvectors and $`\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`$ contains eigenvalues.

#### Step 3: Reconstruction

The reconstructed data matrix is:

```math
X = U \Lambda^{1/2}
```

**Dimensionality reduction**: Use only the top $`k`$ eigenvalues and corresponding eigenvectors:

```math
X_k = U_k \Lambda_k^{1/2}
```

where $`U_k`$ contains the first $`k`$ eigenvectors and $`\Lambda_k`$ contains the first $`k`$ eigenvalues.

### Properties of MDS

1. **Exact reconstruction**: If $`D`$ is Euclidean, classical MDS provides exact reconstruction
2. **Dimensionality reduction**: Can reduce to any desired dimension $`k \leq p`$
3. **Stress minimization**: Minimizes the stress function:

```math
\text{Stress} = \sqrt{\frac{\sum_{i,j} (d_{ij} - \hat{d}_{ij})^2}{\sum_{i,j} d_{ij}^2}}
```

where $`\hat{d}_{ij}`$ are the reconstructed distances.

## 6.1.7. Practical Considerations

### Data Preprocessing

1. **Standardization**: For numerical data, standardize features to have zero mean and unit variance
2. **Normalization**: Scale features to [0,1] range for bounded distances
3. **Missing values**: Handle missing values through imputation or specialized distance measures

### Distance Measure Selection

**Guidelines:**
- **Euclidean**: Default choice for continuous numerical data
- **Manhattan**: Robust to outliers, good for high-dimensional data
- **Cosine**: Text data, high-dimensional sparse data
- **Jaccard**: Categorical data, set-based data
- **Hamming**: Binary data, DNA sequences
- **Edit**: String data, DNA sequences

### Computational Complexity

- **Distance matrix computation**: $`O(n^2 p)`$ for $`n`$ samples and $`p`$ features
- **MDS eigendecomposition**: $`O(n^3)`$
- **Memory requirements**: $`O(n^2)`$ for storing distance matrix

## 6.1.8. Python Implementation

```python
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

# Example usage and demonstration
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

if __name__ == "__main__":
    demonstrate_distance_measures()
```

## 6.1.9. R Implementation

```r
# Distance Measures and Multidimensional Scaling in R
library(stats)
library(ggplot2)
library(dplyr)
library(proxy)
library(MASS)

DistanceMeasures <- setRefClass("DistanceMeasures",
  methods = list(
    
    euclidean_distance = function(x, z) {
      sqrt(sum((x - z)^2))
    },
    
    manhattan_distance = function(x, z) {
      sum(abs(x - z))
    },
    
    minkowski_distance = function(x, z, p = 2) {
      (sum(abs(x - z)^p))^(1/p)
    },
    
    chebyshev_distance = function(x, z) {
      max(abs(x - z))
    },
    
    hamming_distance = function(x, z) {
      sum(x != z)
    },
    
    jaccard_distance = function(set_a, set_b) {
      intersection <- length(intersect(set_a, set_b))
      union <- length(union(set_a, set_b))
      if (union == 0) return(0)
      1 - intersection / union
    },
    
    cosine_distance = function(x, z) {
      dot_product <- sum(x * z)
      norm_x <- sqrt(sum(x^2))
      norm_z <- sqrt(sum(z^2))
      if (norm_x == 0 || norm_z == 0) return(1)
      1 - dot_product / (norm_x * norm_z)
    },
    
    edit_distance = function(s, t) {
      # Simple implementation using adist
      adist(s, t)[1, 1]
    },
    
    compute_distance_matrix = function(X, metric = "euclidean") {
      if (metric == "euclidean") {
        as.matrix(dist(X, method = "euclidean"))
      } else if (metric == "manhattan") {
        as.matrix(dist(X, method = "manhattan"))
      } else if (metric == "cosine") {
        # Use proxy package for cosine distance
        as.matrix(proxy::dist(X, method = "cosine"))
      } else {
        stop(paste("Unsupported metric:", metric))
      }
    },
    
    classical_mds = function(D, k = NULL) {
      n <- nrow(D)
      if (is.null(k)) k <- n
      
      # Step 1: Double centering
      D_squared <- D^2
      H <- diag(n) - matrix(1, n, n) / n
      B <- -0.5 * H %*% D_squared %*% H
      
      # Step 2: Eigendecomposition
      eigen_decomp <- eigen(B)
      eigenvals <- eigen_decomp$values
      eigenvecs <- eigen_decomp$vectors
      
      # Sort in descending order
      idx <- order(eigenvals, decreasing = TRUE)
      eigenvals <- eigenvals[idx]
      eigenvecs <- eigenvecs[, idx]
      
      # Step 3: Reconstruction
      X_reconstructed <- eigenvecs[, 1:k] %*% diag(sqrt(eigenvals[1:k]))
      
      list(
        coordinates = X_reconstructed,
        eigenvalues = eigenvals,
        eigenvectors = eigenvecs
      )
    },
    
    plot_distance_comparison = function(X, metrics = c("euclidean", "manhattan", "cosine")) {
      plots <- list()
      
      for (i in seq_along(metrics)) {
        metric <- metrics[i]
        D <- compute_distance_matrix(X, metric)
        
        # Convert to long format for ggplot
        D_long <- as.data.frame(D) %>%
          mutate(row_id = row_number()) %>%
          tidyr::gather(key = "col_id", value = "distance", -row_id) %>%
          mutate(col_id = as.numeric(col_id))
        
        p <- ggplot(D_long, aes(x = col_id, y = row_id, fill = distance)) +
          geom_tile() +
          scale_fill_viridis_c() +
          labs(title = paste(toupper(metric), "Distance"),
               x = "Sample Index", y = "Sample Index") +
          theme_minimal() +
          theme(axis.text = element_text(size = 8))
        
        plots[[i]] <- p
      }
      
      # Combine plots
      do.call(gridExtra::grid.arrange, c(plots, ncol = length(metrics)))
    },
    
    analyze_distance_distributions = function(X, metrics = c("euclidean", "manhattan", "cosine")) {
      plots <- list()
      
      for (i in seq_along(metrics)) {
        metric <- metrics[i]
        D <- compute_distance_matrix(X, metric)
        
        # Get upper triangular part (excluding diagonal)
        distances <- D[upper.tri(D)]
        
        p <- ggplot(data.frame(distance = distances), aes(x = distance)) +
          geom_histogram(bins = 30, alpha = 0.7, fill = "steelblue", color = "black") +
          geom_vline(xintercept = mean(distances), color = "red", linestyle = "dashed") +
          labs(title = paste(toupper(metric), "Distance Distribution"),
               x = "Distance", y = "Frequency") +
          annotate("text", x = mean(distances), y = Inf, 
                   label = paste("Mean:", round(mean(distances), 3)),
                   vjust = 2, hjust = -0.1, color = "red") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      do.call(gridExtra::grid.arrange, c(plots, ncol = length(metrics)))
    }
  )
)

# Example usage and demonstration
demonstrate_distance_measures <- function() {
  cat("=== Distance Measures Demonstration ===\n\n")
  
  dm <- DistanceMeasures$new()
  
  # Generate sample data
  set.seed(42)
  X <- matrix(rnorm(50 * 3), ncol = 3)  # 50 samples, 3 features
  
  # Numerical distance examples
  x1 <- X[1, ]
  x2 <- X[2, ]
  cat("Sample points:\n")
  cat("x1 =", paste(round(x1, 3), collapse = ", "), "\n")
  cat("x2 =", paste(round(x2, 3), collapse = ", "), "\n\n")
  
  cat("Distance measures:\n")
  cat("Euclidean distance:", round(dm$euclidean_distance(x1, x2), 4), "\n")
  cat("Manhattan distance:", round(dm$manhattan_distance(x1, x2), 4), "\n")
  cat("Minkowski distance (p=3):", round(dm$minkowski_distance(x1, x2, 3), 4), "\n")
  cat("Chebyshev distance:", round(dm$chebyshev_distance(x1, x2), 4), "\n")
  cat("Cosine distance:", round(dm$cosine_distance(x1, x2), 4), "\n\n")
  
  # Categorical distance examples
  set_a <- c("apple", "banana", "cherry", "date")
  set_b <- c("apple", "banana", "elderberry")
  cat("Set A:", paste(set_a, collapse = ", "), "\n")
  cat("Set B:", paste(set_b, collapse = ", "), "\n")
  cat("Jaccard distance:", round(dm$jaccard_distance(set_a, set_b), 4), "\n\n")
  
  # String distance examples
  s1 <- "karolin"
  s2 <- "kathrin"
  cat("String 1:", s1, "\n")
  cat("String 2:", s2, "\n")
  cat("Edit distance:", dm$edit_distance(s1, s2), "\n\n")
  
  # Distance matrix analysis
  cat("Computing distance matrices for", nrow(X), "samples...\n")
  dm$plot_distance_comparison(X)
  dm$analyze_distance_distributions(X)
  
  # MDS demonstration
  cat("\n=== Multidimensional Scaling Demo ===\n")
  D <- dm$compute_distance_matrix(X, "euclidean")
  mds_result <- dm$classical_mds(D, k = 2)
  
  cat("Original data shape:", dim(X), "\n")
  cat("MDS reconstructed shape:", dim(mds_result$coordinates), "\n")
  cat("Top 5 eigenvalues:", round(mds_result$eigenvalues[1:5], 4), "\n")
  
  # Plot MDS results
  par(mfrow = c(1, 2))
  
  plot(X[, 1], X[, 2], main = "Original Data (First 2 Dimensions)",
       xlab = "Feature 1", ylab = "Feature 2", pch = 19, col = "blue")
  
  plot(mds_result$coordinates[, 1], mds_result$coordinates[, 2],
       main = "MDS Reconstruction (2D)",
       xlab = "MDS Dimension 1", ylab = "MDS Dimension 2", 
       pch = 19, col = "red")
  
  par(mfrow = c(1, 1))
}

# Run demonstration
demonstrate_distance_measures()
```

## 6.1.10. Summary and Best Practices

### Key Takeaways

1. **Distance measure selection is crucial** for clustering quality
2. **Different data types require different measures**
3. **Preprocessing matters** - standardize numerical data
4. **Computational efficiency** varies significantly between measures
5. **MDS provides powerful dimensionality reduction** from distance matrices

### Decision Framework

**For numerical data:**
- Start with Euclidean distance
- Use Manhattan for robustness to outliers
- Consider cosine for high-dimensional data

**For categorical data:**
- Use Jaccard for set-based data
- Use Hamming for binary data
- Use edit distance for strings

**For mixed data:**
- Use Gower's distance
- Or standardize and use numerical measures

### Common Pitfalls

1. **Scale sensitivity**: Features with different scales dominate Euclidean distance
2. **Curse of dimensionality**: Distances become less meaningful in high dimensions
3. **Computational complexity**: Distance matrix computation scales quadratically
4. **Missing values**: Require specialized handling

### Advanced Topics

- **Kernel methods**: Extend distance measures to non-Euclidean spaces
- **Metric learning**: Learn optimal distance measures from data
- **Approximate methods**: Use techniques like locality-sensitive hashing for large datasets
- **Non-metric distances**: Relax metric axioms for specific applications
