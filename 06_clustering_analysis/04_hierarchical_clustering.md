# 6.4. Hierarchical Clustering

## 6.4.1. Introduction to Hierarchical Clustering

Hierarchical clustering is a fundamental clustering approach that builds a **hierarchy of clusters** without requiring the user to specify the number of clusters $`K`$ in advance. Unlike K-means, which produces a flat partition of the data, hierarchical clustering creates a **tree-like structure** (dendrogram) that shows the relationships between clusters at different levels of granularity.

### Problem Formulation

Given a dataset $`X = \{x_1, x_2, \ldots, x_n\}`$ where each $`x_i \in \mathbb{R}^p`$, and a distance matrix $`D \in \mathbb{R}^{n \times n}`$ where $`D_{ij} = d(x_i, x_j)`$, hierarchical clustering aims to:

1. Build a hierarchy of nested clusters
2. Provide a dendrogram visualization
3. Allow cluster extraction at any desired level

### Key Advantages

- **No predefined K**: Unlike K-means, no need to specify number of clusters upfront
- **Hierarchical structure**: Natural representation of data relationships
- **Flexible distance measures**: Can use any distance metric
- **Visual interpretation**: Dendrogram provides intuitive cluster visualization
- **Nested clusters**: Clusters at level $`K`$ are always refinements of clusters at level $`K-1`$

## 6.4.2. Types of Hierarchical Clustering

### Agglomerative (Bottom-Up) Clustering

**Most common approach**: Start with each observation as its own cluster and iteratively merge the closest pairs.

**Algorithm**:
1. Initialize: $`n`$ clusters, each containing one observation
2. Iterate: Merge the two closest clusters
3. Terminate: When all observations are in one cluster

### Divisive (Top-Down) Clustering

**Less common**: Start with all observations in one cluster and recursively split.

**Algorithm**:
1. Initialize: One cluster containing all observations
2. Iterate: Split the cluster that maximizes some criterion
3. Terminate: When each observation is its own cluster

## 6.4.3. Linkage Criteria

The choice of **linkage criterion** determines how to measure distance between clusters and significantly affects the resulting cluster structure.

### Single Linkage (Nearest Neighbor)

Distance between clusters $`A`$ and $`B`$ is the minimum distance between any point in $`A`$ and any point in $`B`$:

```math
d_{\text{single}}(A, B) = \min_{x \in A, y \in B} d(x, y)
```

**Properties**:
- Tends to produce "chaining" - long, stringy clusters
- Sensitive to noise and outliers
- Can handle non-elliptical cluster shapes
- Computationally efficient

**Example**: If cluster A contains points (1,1) and (1,2), and cluster B contains (5,1), then $`d_{\text{single}}(A, B) = \min\{d((1,1), (5,1)), d((1,2), (5,1))\} = \min\{4, \sqrt{17}\} = 4`$

### Complete Linkage (Farthest Neighbor)

Distance is the maximum distance between any point in $`A`$ and any point in $`B`$:

```math
d_{\text{complete}}(A, B) = \max_{x \in A, y \in B} d(x, y)
```

**Properties**:
- Tends to produce compact, spherical clusters
- More robust to noise than single linkage
- Can break large clusters
- Computationally efficient

**Example**: Using the same clusters as above, $`d_{\text{complete}}(A, B) = \max\{4, \sqrt{17}\} = \sqrt{17}`$

### Average Linkage (UPGMA - Unweighted Pair Group Method with Arithmetic Mean)

Distance is the average of all pairwise distances:

```math
d_{\text{average}}(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} d(x, y)
```

**Properties**:
- Balances single and complete linkage
- Less sensitive to outliers than single linkage
- More flexible cluster shapes than complete linkage
- Computationally efficient

### Ward's Linkage

Minimizes the increase in total within-cluster variance. The distance between clusters $`A`$ and $`B`$ is:

```math
d_{\text{ward}}(A, B) = \frac{|A||B|}{|A| + |B|} \|m_A - m_B\|^2
```

where $`m_A`$ and $`m_B`$ are the centroids of clusters $`A`$ and $`B`$.

**Properties**:
- Tends to produce clusters of similar sizes
- Minimizes within-cluster variance
- Sensitive to outliers
- Computationally efficient

### Weighted Average Linkage (WPGMA)

Similar to average linkage but gives equal weight to each cluster regardless of size:

```math
d_{\text{weighted}}(A, B) = \frac{1}{2} \left( \frac{1}{|A|} \sum_{x \in A} d(x, m_B) + \frac{1}{|B|} \sum_{y \in B} d(y, m_A) \right)
```

## 6.4.4. The Agglomerative Algorithm in Detail

### Algorithm Steps

**Input**: Distance matrix $`D \in \mathbb{R}^{n \times n}`$, linkage method

**Output**: Linkage matrix $`Z \in \mathbb{R}^{(n-1) \times 4}`$

**Algorithm**:

1. **Initialization**:
   - Set $`C_i = \{x_i\}`$ for $`i = 1, 2, \ldots, n`$ (each point is its own cluster)
   - Set $`\mathcal{C} = \{C_1, C_2, \ldots, C_n\}`$ (set of all clusters)

2. **Iterative Merging**:
   For $`t = 1, 2, \ldots, n-1`$:
   - Find clusters $`C_i, C_j \in \mathcal{C}`$ that minimize $`d(C_i, C_j)`$ according to the chosen linkage method
   - Merge $`C_i`$ and $`C_j`$ into new cluster $`C_{n+t} = C_i \cup C_j`$
   - Update $`\mathcal{C} = \mathcal{C} \setminus \{C_i, C_j\} \cup \{C_{n+t}\}`$
   - Store merge information in $`Z[t, :] = [i, j, d(C_i, C_j), |C_{n+t}|]`$

3. **Termination**: When $`|\mathcal{C}| = 1`$

### Linkage Matrix Structure

The linkage matrix $`Z`$ has $`n-1`$ rows and 4 columns:
- $`Z[i, 0]`$: Index of first cluster merged at step $`i`$
- $`Z[i, 1]`$: Index of second cluster merged at step $`i`$
- $`Z[i, 2]`$: Distance between the merged clusters
- $`Z[i, 3]`$: Number of observations in the new cluster

## 6.4.5. Dendrograms and Visualization

### Dendrogram Structure

A **dendrogram** is a tree diagram that visualizes the hierarchical clustering process:

- **Leaves**: Individual observations (bottom of tree)
- **Internal nodes**: Merges of clusters
- **Height**: Distance at which clusters are merged
- **Branches**: Connections between clusters

### Mathematical Properties

**Monotonicity**: The height (distance) at which clusters are merged never decreases as you move up the dendrogram:

```math
Z[i, 2] \leq Z[i+1, 2] \quad \text{for all } i
```

**Nestedness**: The set of clusters at each level is a refinement of the set at the previous level.

### Cluster Extraction

To extract $`K`$ clusters from the dendrogram:

1. **Height-based cutting**: Cut at a specific height $`h`$
2. **Number-based cutting**: Cut to get exactly $`K`$ clusters

**Mathematical formulation**: For height-based cutting, cluster $`C`$ contains all observations $`x_i`$ such that the path from $`x_i`$ to the root has maximum height $`\leq h`$.

## 6.4.6. Computational Complexity

### Time Complexity

- **Single/Complete/Average linkage**: $`O(n^2 \log n)`$ with efficient implementations
- **Ward's linkage**: $`O(n^2 \log n)`$
- **Naive implementation**: $`O(n^3)`$

### Space Complexity

- **Distance matrix**: $`O(n^2)`$
- **Linkage matrix**: $`O(n)`$
- **Total**: $`O(n^2)`$

### Optimizations

1. **Nearest neighbor chains**: Reduces time complexity for single linkage
2. **Sparse distance matrices**: For high-dimensional data
3. **Approximate methods**: For very large datasets

## 6.4.7. Comparison of Linkage Methods

### Visual Comparison

**Single Linkage**: Produces "chaining" - long, stringy clusters that can connect distant points through intermediate points.

**Complete Linkage**: Produces compact, spherical clusters that are more robust to noise.

**Average Linkage**: Balances the extremes, producing clusters of moderate compactness.

**Ward's Linkage**: Produces clusters of similar sizes, minimizing within-cluster variance.

### Mathematical Comparison

For clusters $`A`$ and $`B`$ with centroids $`m_A`$ and $`m_B`$:

```math
d_{\text{single}}(A, B) \leq d_{\text{average}}(A, B) \leq d_{\text{complete}}(A, B)
```

Ward's linkage is not directly comparable as it uses a different distance measure.

## 6.4.8. Python Implementation

```python
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
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.distance_matrix = None
        
    def fit(self, X):
        """Fit hierarchical clustering to the data."""
        # Compute distance matrix
        self.distance_matrix = pdist(X, metric=self.metric)
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(self.distance_matrix, method=self.method)
        
        return self
    
    def get_clusters(self, n_clusters=None, height=None):
        """Extract clusters from the dendrogram."""
        if n_clusters is not None:
            return fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust')
        elif height is not None:
            return fcluster(self.linkage_matrix, t=height, criterion='distance')
        else:
            raise ValueError("Must specify either n_clusters or height")
    
    def plot_dendrogram(self, max_d=None, title=None):
        """Plot the dendrogram."""
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
        """Compute cophenetic correlation coefficient."""
        c, coph_dists = cophenet(self.linkage_matrix, self.distance_matrix)
        return c
    
    def compare_linkage_methods(self, X, methods=['single', 'complete', 'average', 'ward']):
        """Compare different linkage methods."""
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
        """Plot comparison of different linkage methods."""
        results = self.compare_linkage_methods(X, methods)
        
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
    """Demonstrate hierarchical clustering with various examples."""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Create three well-separated clusters
    cluster1 = np.random.normal([0, 0], [1, 1], (n_samples//3, 2))
    cluster2 = np.random.normal([6, 6], [1, 1], (n_samples//3, 2))
    cluster3 = np.random.normal([3, 9], [1, 1], (n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("=== Hierarchical Clustering Demonstration ===\n")
    
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

if __name__ == "__main__":
    demonstrate_hierarchical_clustering()
```

## 6.4.9. R Implementation

```r
# Hierarchical Clustering Implementation in R
library(stats)
library(cluster)
library(dendextend)
library(ggplot2)
library(dplyr)

HierarchicalClustering <- setRefClass("HierarchicalClustering",
  fields = list(
    method = "character",
    metric = "character",
    linkage_matrix = "matrix",
    distance_matrix = "dist"
  ),
  
  methods = list(
    
    initialize = function(method = "complete", metric = "euclidean") {
      method <<- method
      metric <<- metric
    },
    
    fit = function(X) {
      # Compute distance matrix
      distance_matrix <<- dist(X, method = metric)
      
      # Perform hierarchical clustering
      hc <- hclust(distance_matrix, method = method)
      linkage_matrix <<- hc$merge
      
      invisible(.self)
    },
    
    get_clusters = function(n_clusters = NULL, height = NULL) {
      if (!is.null(n_clusters)) {
        cutree(hclust(distance_matrix, method = method), k = n_clusters)
      } else if (!is.null(height)) {
        cutree(hclust(distance_matrix, method = method), h = height)
      } else {
        stop("Must specify either n_clusters or height")
      }
    },
    
    plot_dendrogram = function(title = NULL) {
      hc <- hclust(distance_matrix, method = method)
      
      plot(hc, 
           main = title %||% paste("Hierarchical Clustering Dendrogram (", method, " linkage)"),
           xlab = "Sample Index", 
           ylab = "Distance",
           sub = "")
    },
    
    cophenetic_correlation = function() {
      hc <- hclust(distance_matrix, method = method)
      cor(distance_matrix, cophenetic(hc))
    },
    
    compare_linkage_methods = function(X, methods = c("single", "complete", "average", "ward.D")) {
      results <- list()
      
      for (method in methods) {
        # Fit clustering
        hc_temp <- HierarchicalClustering$new(method = method)
        hc_temp$fit(X)
        
        # Compute cophenetic correlation
        cophenetic_corr <- hc_temp$cophenetic_correlation()
        
        # Compute silhouette scores for different K
        silhouette_scores <- numeric(9)  # K = 2 to 10
        for (k in 2:10) {
          labels <- hc_temp$get_clusters(n_clusters = k)
          if (length(unique(labels)) > 1) {
            silhouette_scores[k-1] <- mean(silhouette(labels, hc_temp$distance_matrix)[, 3])
          }
        }
        
        results[[method]] <- list(
          cophenetic_correlation = cophenetic_corr,
          silhouette_scores = silhouette_scores,
          hclust_obj = hclust(hc_temp$distance_matrix, method = method)
        )
      }
      
      results
    },
    
    plot_comparison = function(X, methods = c("single", "complete", "average", "ward.D")) {
      results <- compare_linkage_methods(X, methods)
      
      # Plot dendrograms
      par(mfrow = c(2, 2))
      for (method in methods) {
        plot(results[[method]]$hclust_obj, 
             main = paste(toupper(method), "Linkage"),
             xlab = "Sample Index", 
             ylab = "Distance")
      }
      par(mfrow = c(1, 1))
      
      # Plot silhouette scores
      silhouette_data <- data.frame()
      for (method in methods) {
        scores <- results[[method]]$silhouette_scores
        silhouette_data <- rbind(silhouette_data, 
                                data.frame(
                                  K = 2:10,
                                  Score = scores,
                                  Method = method
                                ))
      }
      
      p <- ggplot(silhouette_data, aes(x = K, y = Score, color = Method)) +
        geom_line() +
        geom_point() +
        labs(title = "Silhouette Scores for Different Linkage Methods",
             x = "Number of Clusters (K)",
             y = "Silhouette Score") +
        theme_minimal() +
        scale_color_viridis_d()
      
      print(p)
      
      # Print cophenetic correlations
      cat("Cophenetic Correlation Coefficients:\n")
      for (method in methods) {
        cat(sprintf("%s: %.4f\n", toupper(method), results[[method]]$cophenetic_correlation))
      }
    }
  )
)

# Example usage and demonstration
demonstrate_hierarchical_clustering <- function() {
  cat("=== Hierarchical Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  n_samples <- 100
  
  # Create three well-separated clusters
  cluster1 <- matrix(rnorm(n_samples/3 * 2, mean = c(0, 0), sd = 1), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/3 * 2, mean = c(6, 6), sd = 1), ncol = 2)
  cluster3 <- matrix(c(rnorm(n_samples/3, mean = 3, sd = 1), 
                       rnorm(n_samples/3, mean = 9, sd = 1)), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Initialize hierarchical clustering
  hc <- HierarchicalClustering$new(method = "complete")
  hc$fit(X)
  
  # Plot dendrogram
  cat("Plotting dendrogram...\n")
  hc$plot_dendrogram("Complete Linkage Dendrogram")
  
  # Extract clusters
  labels_3 <- hc$get_clusters(n_clusters = 3)
  labels_5 <- hc$get_clusters(n_clusters = 5)
  
  # Visualize cluster assignments
  par(mfrow = c(1, 2))
  
  plot(X[, 1], X[, 2], col = labels_3, pch = 19, 
       main = "3 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  plot(X[, 1], X[, 2], col = labels_5, pch = 19, 
       main = "5 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  par(mfrow = c(1, 1))
  
  # Compare linkage methods
  cat("\nComparing different linkage methods...\n")
  hc$plot_comparison(X)
  
  # Cophenetic correlation
  cat(sprintf("\nCophenetic correlation: %.4f\n", hc$cophenetic_correlation()))
  
  # Evaluate clustering quality
  silhouette_3 <- mean(silhouette(labels_3, hc$distance_matrix)[, 3])
  silhouette_5 <- mean(silhouette(labels_5, hc$distance_matrix)[, 3])
  cat(sprintf("Silhouette score (3 clusters): %.4f\n", silhouette_3))
  cat(sprintf("Silhouette score (5 clusters): %.4f\n", silhouette_5))
}

# Run demonstration
demonstrate_hierarchical_clustering()
```

## 6.4.10. Summary and Best Practices

### Key Takeaways

1. **Hierarchical clustering builds a tree structure** without requiring predefined K
2. **Linkage method choice is crucial** - affects cluster shape and quality
3. **Dendrograms provide visual insight** into data structure
4. **Computational cost scales quadratically** with dataset size
5. **Nested structure** allows flexible cluster extraction

### Linkage Method Selection

**Use Single Linkage when:**
- Clusters have irregular shapes
- You want to detect chaining patterns
- Computational efficiency is important

**Use Complete Linkage when:**
- You want compact, spherical clusters
- Data is noisy or has outliers
- You prefer more balanced cluster sizes

**Use Average Linkage when:**
- You want a balanced approach
- Clusters have moderate compactness
- You're unsure about cluster shapes

**Use Ward's Linkage when:**
- You want clusters of similar sizes
- Minimizing within-cluster variance is important
- Data is relatively clean

### Common Pitfalls

1. **Chaining in single linkage**: Can connect distant points through intermediate points
2. **Computational complexity**: May not scale to very large datasets
3. **Sensitivity to noise**: Outliers can affect cluster structure
4. **Irreversible merges**: Once clusters are merged, they cannot be split

### Advanced Topics

- **Dynamic time warping**: For time series data
- **Fast hierarchical clustering**: Approximate methods for large datasets
- **Consensus clustering**: Combining multiple hierarchical clusterings
- **Bootstrap hierarchical clustering**: Assessing cluster stability
