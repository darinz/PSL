# 9.5. Fisher Discriminant Analysis

## 9.5.0. Introduction and Motivation

Fisher Discriminant Analysis (FDA), also known as Fisher's Linear Discriminant Analysis, is a fundamental supervised dimensionality reduction technique that finds optimal projection directions to maximize class separation. Unlike unsupervised methods like PCA, FDA leverages class label information to find directions that are most discriminative for classification.

### Key Concepts

**Supervised vs Unsupervised Dimensionality Reduction**:
- **Unsupervised** (e.g., PCA): Uses only feature data $`X`$, ignores labels $`Y`$
- **Supervised** (e.g., FDA): Uses both features $`X`$ and labels $`Y`$ to find discriminative directions

### Fisher's Intuition

Fisher's key insight was to find a projection direction that:
1. **Maximizes** the separation between class means (between-class variance)
2. **Minimizes** the spread within each class (within-class variance)

This leads to the famous **Fisher criterion**:

```math
J(\mathbf{a}) = \frac{\text{Between-class variance}}{\text{Within-class variance}} = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}}
```

Where $`\mathbf{a}`$ is the projection direction we seek to find.

## 9.5.1. Mathematical Foundation

### The Fisher Criterion

Let's formalize Fisher's objective. Given a projection direction $`\mathbf{a} \in \mathbb{R}^p`$, we want to maximize:

```math
J(\mathbf{a}) = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}}
```

### Between-Class Scatter Matrix ($`\mathbf{B}`$)

The between-class scatter matrix measures how far apart the class means are:

```math
\mathbf{B} = \frac{1}{K-1} \sum_{k=1}^K n_k (\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})^T
```

Where:
- $`\boldsymbol{\mu}_k`$ is the mean of class $`k`$
- $`\bar{\boldsymbol{\mu}} = \frac{1}{n} \sum_{k=1}^K n_k \boldsymbol{\mu}_k`$ is the overall mean
- $`n_k`$ is the number of samples in class $`k`$
- $`K`$ is the number of classes

**Intuition**: $`\mathbf{B}`$ captures the variance of class centers around the overall mean.

### Within-Class Scatter Matrix ($`\mathbf{W}`$)

The within-class scatter matrix measures the spread within each class:

```math
\mathbf{W} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
```

**Intuition**: $`\mathbf{W}`$ is essentially the pooled covariance matrix, measuring how tightly points cluster around their class means.

### Geometric Interpretation

Every data point $`\mathbf{x}_i`$ can be decomposed as:

```math
\mathbf{x}_i = \underbrace{(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})}_{\text{Within-class deviation}} + \underbrace{\boldsymbol{\mu}_{y_i}}_{\text{Class center}}
```

Where:
- $`(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})`$ represents the deviation from the class mean (captured by $`\mathbf{W}`$)
- $`\boldsymbol{\mu}_{y_i}`$ represents the class center (captured by $`\mathbf{B}`$)

## 9.5.2. The Generalized Eigenvalue Problem

### Optimization Formulation

Maximizing the Fisher criterion leads to a **generalized eigenvalue problem**:

```math
\mathbf{B} \mathbf{a} = \lambda \mathbf{W} \mathbf{a}
```

This can be rewritten as:

```math
\mathbf{W}^{-1} \mathbf{B} \mathbf{a} = \lambda \mathbf{a}
```

### Solution Properties

1. **Number of Directions**: We can find at most $`K-1`$ non-zero eigenvalues because $`\text{rank}(\mathbf{B}) \leq K-1`$

2. **Eigenvalue Interpretation**: The eigenvalues $`\lambda_i`$ represent the ratio of between-class to within-class variance along each direction

3. **Optimal Directions**: The eigenvectors $`\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_{K-1}`$ are the optimal projection directions

### Mathematical Derivation

To find the maximum of $`J(\mathbf{a})`$, we set the gradient to zero:

```math
\nabla_{\mathbf{a}} J(\mathbf{a}) = \frac{2\mathbf{B}\mathbf{a}(\mathbf{a}^T\mathbf{W}\mathbf{a}) - 2\mathbf{W}\mathbf{a}(\mathbf{a}^T\mathbf{B}\mathbf{a})}{(\mathbf{a}^T\mathbf{W}\mathbf{a})^2} = 0
```

This simplifies to:

```math
\mathbf{B}\mathbf{a} = \frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} \mathbf{W}\mathbf{a}
```

Recognizing that $`\frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} = J(\mathbf{a})`$ is the eigenvalue $`\lambda`$, we get:

```math
\mathbf{B}\mathbf{a} = \lambda \mathbf{W}\mathbf{a}
```

## 9.5.3. Connection to Linear Discriminant Analysis

### Equivalence Under Normality Assumptions

When we assume:
1. Classes follow multivariate normal distributions
2. All classes share the same covariance matrix $`\boldsymbol{\Sigma}`$

Then FDA and LDA produce **equivalent subspaces**:

```math
\mathbf{W} \approx \boldsymbol{\Sigma} \quad \text{and} \quad \mathbf{B} \approx \boldsymbol{\Sigma}_B
```

Where $`\boldsymbol{\Sigma}_B`$ is the between-class covariance matrix in LDA.

### Key Differences

| Aspect | FDA | LDA |
|--------|-----|-----|
| **Assumptions** | No distributional assumptions | Multivariate normal, equal covariance |
| **Objective** | Maximize class separation | Minimize classification error |
| **Output** | Projection directions | Classification rule |
| **Flexibility** | More general | More restrictive |

### Practical Implementation

In practice, FDA directions can be extracted from LDA:

```python
# FDA directions from LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
fda_directions = lda.scalings_  # These are the FDA directions
```

## 9.5.4. Supervised Dimension Reduction

### Why Supervised?

FDA is **supervised** because it uses class labels $`Y`$ to find discriminative directions. This is fundamentally different from PCA:

- **PCA**: Directions maximize variance regardless of class labels
- **FDA**: Directions maximize class separation

### Example: Toy Data Visualization

Consider a 2D dataset with 3 classes:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

def generate_toy_data(n_samples=300, random_state=42):
    """
    Generate toy data for FDA demonstration
    """
    np.random.seed(random_state)
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0: centered at (0, 0)
    X0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_per_class)
    
    # Class 1: centered at (3, 2)
    X1 = np.random.multivariate_normal([3, 2], [[1, 0.5], [0.5, 1]], n_per_class)
    
    # Class 2: centered at (1, 4)
    X2 = np.random.multivariate_normal([1, 4], [[1, 0.5], [0.5, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    return X, y

def compare_pca_fda():
    """
    Compare PCA and FDA on toy data
    """
    X, y = generate_toy_data()
    
    # Apply PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)
    
    # Apply FDA (via LDA)
    lda = LinearDiscriminantAnalysis()
    X_fda = lda.fit_transform(X, y)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCA projection
    for i in range(3):
        mask = y == i
        axes[0, 1].scatter(X_pca[mask], np.zeros_like(X_pca[mask]), alpha=0.7, label=f'Class {i}')
    axes[0, 1].set_title('PCA Projection (1D)')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.1, 0.1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # FDA projection
    for i in range(3):
        mask = y == i
        axes[1, 0].scatter(X_fda[mask], np.zeros_like(X_fda[mask]), alpha=0.7, label=f'Class {i}')
    axes[1, 0].set_title('FDA Projection (1D)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.1, 0.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Projection directions
    pca_direction = pca.components_[0]
    fda_direction = lda.scalings_[:, 0]
    
    # Normalize for visualization
    pca_direction = pca_direction / np.linalg.norm(pca_direction)
    fda_direction = fda_direction / np.linalg.norm(fda_direction)
    
    for i in range(3):
        mask = y == i
        axes[1, 1].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    
    # Plot projection directions
    origin = np.array([0, 0])
    axes[1, 1].quiver(origin[0], origin[1], pca_direction[0], pca_direction[1], 
                     color='red', scale=5, label='PCA Direction', linewidth=3)
    axes[1, 1].quiver(origin[0], origin[1], fda_direction[0], fda_direction[1], 
                     color='green', scale=5, label='FDA Direction', linewidth=3)
    
    axes[1, 1].set_title('Projection Directions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print separation metrics
    print("Class Separation Analysis:")
    print("-" * 40)
    
    # Calculate separation for PCA
    pca_separation = calculate_separation(X_pca, y)
    print(f"PCA Separation: {pca_separation:.4f}")
    
    # Calculate separation for FDA
    fda_separation = calculate_separation(X_fda, y)
    print(f"FDA Separation: {fda_separation:.4f}")
    
    return pca, lda

def calculate_separation(X_proj, y):
    """
    Calculate Fisher's separation criterion for projected data
    """
    classes = np.unique(y)
    overall_mean = np.mean(X_proj)
    
    # Between-class variance
    between_var = 0
    for c in classes:
        class_mean = np.mean(X_proj[y == c])
        n_class = np.sum(y == c)
        between_var += n_class * (class_mean - overall_mean) ** 2
    
    # Within-class variance
    within_var = 0
    for c in classes:
        class_data = X_proj[y == c]
        class_mean = np.mean(class_data)
        within_var += np.sum((class_data - class_mean) ** 2)
    
    return between_var / within_var if within_var > 0 else 0

# Run comparison
if __name__ == "__main__":
    pca, lda = compare_pca_fda()
```

### Extension to Regression

FDA can be extended to regression problems by discretizing the continuous response:

```python
def fda_for_regression(X, y, n_bins=10):
    """
    Apply FDA to regression by discretizing the response
    """
    # Discretize y into bins
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    
    # Apply FDA
    lda = LinearDiscriminantAnalysis()
    X_fda = lda.fit_transform(X, y_binned)
    
    return X_fda, lda
```

## 9.5.5. Implementation from Scratch

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

class FisherDiscriminantAnalysis:
    """
    Fisher Discriminant Analysis implementation from scratch
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.explained_variance_ratio_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit FDA model
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Set number of components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # Calculate class means and overall mean
        class_means = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_means[i] = np.mean(X[class_mask], axis=0)
            class_counts[i] = np.sum(class_mask)
        
        overall_mean = np.average(class_means, weights=class_counts, axis=0)
        
        # Calculate between-class scatter matrix
        B = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            diff = class_means[i] - overall_mean
            B += class_counts[i] * np.outer(diff, diff)
        B /= (n_classes - 1)
        
        # Calculate within-class scatter matrix
        W = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            diff = class_data - class_means[i]
            W += diff.T @ diff
        W /= (n_samples - n_classes)
        
        # Solve generalized eigenvalue problem: B * a = λ * W * a
        # This is equivalent to: W^(-1) * B * a = λ * a
        try:
            W_inv = np.linalg.inv(W)
            eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
            
            # Sort eigenvalues in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select top components
            self.scalings_ = eigenvecs[:, :self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components]
            
        except np.linalg.LinAlgError:
            # Handle singular W matrix
            print("Warning: Singular within-class scatter matrix. Using regularization.")
            W_reg = W + 1e-6 * np.eye(n_features)
            W_inv = np.linalg.inv(W_reg)
            eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
            
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            self.scalings_ = eigenvecs[:, :self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data using FDA projection
        """
        if self.scalings_ is None:
            raise ValueError("Model must be fitted before transform")
        
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """
        Fit FDA and transform data
        """
        return self.fit(X, y).transform(X)
    
    def get_discriminant_directions(self):
        """
        Return the discriminant directions (eigenvectors)
        """
        return self.scalings_

def demonstrate_fda_scratch():
    """
    Demonstrate FDA implementation from scratch
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0
    X0 = np.random.multivariate_normal([0, 0, 0, 0], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    # Class 1
    X1 = np.random.multivariate_normal([3, 2, 1, 0], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    # Class 2
    X2 = np.random.multivariate_normal([1, 4, 2, 3], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=2)
    X_fda = fda.fit_transform(X, y)
    
    # Compare with sklearn LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (first 2 dimensions)
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0].set_title('Original Data (First 2 Dimensions)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Our FDA projection
    for i in range(3):
        mask = y == i
        axes[1].scatter(X_fda[mask, 0], X_fda[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[1].set_title('Our FDA Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sklearn LDA projection
    for i in range(3):
        mask = y == i
        axes[2].scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[2].set_title('Sklearn LDA Projection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("FDA Results:")
    print("-" * 30)
    print(f"Number of components: {fda.n_components}")
    print(f"Explained variance ratios: {fda.explained_variance_ratio_}")
    print(f"Discriminant directions shape: {fda.scalings_.shape}")
    
    # Calculate separation
    separation_our = calculate_separation(X_fda, y)
    separation_sklearn = calculate_separation(X_lda, y)
    
    print(f"\nSeparation Analysis:")
    print(f"Our FDA: {separation_our:.4f}")
    print(f"Sklearn LDA: {separation_sklearn:.4f}")
    
    return fda, lda

# Run demonstration
if __name__ == "__main__":
    fda, lda = demonstrate_fda_scratch()
```

### R Implementation

```r
# Fisher Discriminant Analysis in R
library(MASS)
library(ggplot2)
library(gridExtra)

# Custom FDA implementation
fisher_discriminant_analysis <- function(X, y, n_components = NULL) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Set number of components
  if (is.null(n_components)) {
    n_components <- min(n_classes - 1, n_features)
  }
  
  # Calculate class means and counts
  class_means <- matrix(0, nrow = n_classes, ncol = n_features)
  class_counts <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_means[i,] <- colMeans(X[class_mask,, drop = FALSE])
    class_counts[i] <- sum(class_mask)
  }
  
  # Overall mean
  overall_mean <- colSums(class_means * class_counts) / sum(class_counts)
  
  # Between-class scatter matrix
  B <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    diff <- class_means[i,] - overall_mean
    B <- B + class_counts[i] * outer(diff, diff)
  }
  B <- B / (n_classes - 1)
  
  # Within-class scatter matrix
  W <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    diff <- sweep(class_data, 2, class_means[i,], "-")
    W <- W + t(diff) %*% diff
  }
  W <- W / (n_samples - n_classes)
  
  # Solve generalized eigenvalue problem
  tryCatch({
    W_inv <- solve(W)
    eigen_result <- eigen(W_inv %*% B)
    
    # Sort eigenvalues and eigenvectors
    idx <- order(eigen_result$values, decreasing = TRUE)
    eigenvals <- eigen_result$values[idx]
    eigenvecs <- eigen_result$vectors[, idx]
    
    # Select components
    scalings <- eigenvecs[, 1:n_components, drop = FALSE]
    explained_variance_ratio <- eigenvals[1:n_components]
    
  }, error = function(e) {
    # Handle singular matrix
    cat("Warning: Singular within-class scatter matrix. Using regularization.\n")
    W_reg <- W + 1e-6 * diag(n_features)
    W_inv <- solve(W_reg)
    eigen_result <- eigen(W_inv %*% B)
    
    idx <- order(eigen_result$values, decreasing = TRUE)
    eigenvals <- eigen_result$values[idx]
    eigenvecs <- eigen_result$vectors[, idx]
    
    scalings <- eigenvecs[, 1:n_components, drop = FALSE]
    explained_variance_ratio <- eigenvals[1:n_components]
  })
  
  return(list(
    scalings = scalings,
    explained_variance_ratio = explained_variance_ratio,
    classes = classes,
    n_components = n_components
  ))
}

# Transform function
transform_fda <- function(model, X) {
  return(X %*% model$scalings)
}

# Demonstrate FDA
demonstrate_fda_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 300
  n_features <- 4
  
  # Generate 3 classes
  n_per_class <- n_samples %/% 3
  
  # Class 0
  X0 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 0, 0), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  # Class 1
  X1 <- MASS::mvrnorm(n_per_class, mu = c(3, 2, 1, 0), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  # Class 2
  X2 <- MASS::mvrnorm(n_per_class, mu = c(1, 4, 2, 3), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  X <- rbind(X0, X1, X2)
  y <- rep(c(0, 1, 2), each = n_per_class)
  
  # Apply our FDA
  fda_model <- fisher_discriminant_analysis(X, y, n_components = 2)
  X_fda <- transform_fda(fda_model, X)
  
  # Apply MASS LDA
  lda_model <- lda(X, y)
  X_lda <- predict(lda_model, X)$x
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  df_fda <- data.frame(
    x1 = X_fda[,1],
    x2 = X_fda[,2],
    class = factor(y)
  )
  
  df_lda <- data.frame(
    x1 = X_lda[,1],
    x2 = X_lda[,2],
    class = factor(y)
  )
  
  # Plot original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Original Data (First 2 Dimensions)", color = "Class") +
    theme_minimal()
  
  # Plot our FDA
  p2 <- ggplot(df_fda, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Our FDA Projection", color = "Class") +
    theme_minimal()
  
  # Plot MASS LDA
  p3 <- ggplot(df_lda, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "MASS LDA Projection", color = "Class") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  # Print results
  cat("FDA Results:\n")
  cat("-" * 30, "\n")
  cat("Number of components:", fda_model$n_components, "\n")
  cat("Explained variance ratios:", fda_model$explained_variance_ratio, "\n")
  cat("Discriminant directions shape:", dim(fda_model$scalings), "\n")
  
  return(list(fda_model = fda_model, lda_model = lda_model))
}

# Run demonstration
results <- demonstrate_fda_r()
```

## 9.5.6. Risk of Overfitting

### The Overfitting Problem

When $`p \gg n`$ (high-dimensional data with few samples), FDA can overfit severely. This happens because:

1. **Perfect Separation**: With $`p \geq n`$, we can always find directions that perfectly separate classes
2. **Random Features**: Even random noise can appear discriminative in high dimensions
3. **Limited Degrees of Freedom**: The within-class scatter matrix becomes singular

### Example: Overfitting Demonstration

```python
def demonstrate_overfitting():
    """
    Demonstrate FDA overfitting in high dimensions
    """
    np.random.seed(42)
    
    # Generate high-dimensional data with random features
    n_samples = 20
    n_features = 50  # Much larger than n_samples
    
    # Random features
    X = np.random.randn(n_samples, n_features)
    
    # Binary labels
    y = np.random.randint(0, 2, n_samples)
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=1)
    X_fda = fda.fit_transform(X, y)
    
    # Calculate separation
    separation = calculate_separation(X_fda, y)
    
    print(f"High-dimensional FDA Results:")
    print(f"n_samples: {n_samples}")
    print(f"n_features: {n_features}")
    print(f"Separation: {separation:.4f}")
    print(f"Perfect separation achieved: {separation > 100}")
    
    # Visualize projection
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda[mask], np.zeros_like(X_fda[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.title('FDA Projection (Random Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with low-dimensional case
    X_low = X[:, :5]  # Use only first 5 features
    fda_low = FisherDiscriminantAnalysis(n_components=1)
    X_fda_low = fda_low.fit_transform(X_low, y)
    separation_low = calculate_separation(X_fda_low, y)
    
    plt.subplot(1, 2, 2)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda_low[mask], np.zeros_like(X_fda_low[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.title(f'FDA Projection (5 Features)\nSeparation: {separation_low:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return separation, separation_low

# Run overfitting demonstration
high_sep, low_sep = demonstrate_overfitting()
```

### Mitigation Strategies

#### 1. Regularization

```python
def regularized_fda(X, y, alpha=0.1, n_components=None):
    """
    Regularized FDA with shrinkage
    """
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    
    if n_components is None:
        n_components = min(n_classes - 1, n_features)
    
    # Calculate scatter matrices
    B, W = calculate_scatter_matrices(X, y)
    
    # Regularize W
    W_reg = W + alpha * np.eye(n_features)
    
    # Solve eigenvalue problem
    W_inv = np.linalg.inv(W_reg)
    eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
    
    # Sort and select
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    scalings = eigenvecs[:, :n_components]
    
    return scalings, eigenvals[:n_components]

def calculate_scatter_matrices(X, y):
    """
    Calculate between-class and within-class scatter matrices
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape
    
    # Class means and counts
    class_means = np.zeros((n_classes, n_features))
    class_counts = np.zeros(n_classes)
    
    for i, c in enumerate(classes):
        class_mask = y == c
        class_means[i] = np.mean(X[class_mask], axis=0)
        class_counts[i] = np.sum(class_mask)
    
    overall_mean = np.average(class_means, weights=class_counts, axis=0)
    
    # Between-class scatter
    B = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        diff = class_means[i] - overall_mean
        B += class_counts[i] * np.outer(diff, diff)
    B /= (n_classes - 1)
    
    # Within-class scatter
    W = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        class_mask = y == c
        class_data = X[class_mask]
        diff = class_data - class_means[i]
        W += diff.T @ diff
    W /= (n_samples - n_classes)
    
    return B, W
```

#### 2. Feature Selection

```python
def fda_with_feature_selection(X, y, n_features=10, n_components=None):
    """
    FDA with feature selection to reduce dimensionality
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select most discriminative features
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=n_components)
    X_fda = fda.fit_transform(X_selected, y)
    
    return X_fda, fda, selector
```

#### 3. Cross-Validation

```python
def cross_validate_fda(X, y, n_splits=5):
    """
    Cross-validate FDA to assess generalization
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    separations = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit FDA on training data
        fda = FisherDiscriminantAnalysis()
        fda.fit(X_train, y_train)
        
        # Transform test data
        X_test_fda = fda.transform(X_test)
        
        # Calculate separation on test data
        separation = calculate_separation(X_test_fda, y_test)
        separations.append(separation)
    
    return np.mean(separations), np.std(separations)
```

## 9.5.7. Real-World Applications

### Example 1: Face Recognition

```python
def face_recognition_fda():
    """
    FDA for face recognition (simplified example)
    """
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    
    # Load face dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=39)  # 40 classes - 1
    X_train_fda = fda.fit_transform(X_train, y_train)
    X_test_fda = fda.transform(X_test)
    
    # Classify using k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_fda, y_train)
    y_pred = knn.predict(X_test_fda)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Face Recognition Accuracy: {accuracy:.4f}")
    
    # Visualize first few discriminant directions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        direction = fda.scalings_[:, i].reshape(64, 64)
        axes[row, col].imshow(direction, cmap='RdBu_r')
        axes[row, col].set_title(f'Direction {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fda, accuracy
```

### Example 2: Gene Expression Analysis

```python
def gene_expression_fda():
    """
    FDA for gene expression classification
    """
    # Simulate gene expression data
    np.random.seed(42)
    n_samples = 100
    n_genes = 1000
    
    # Generate data with some discriminative genes
    X = np.random.randn(n_samples, n_genes)
    
    # Add discriminative signal to first 50 genes
    X[:50, :50] += 2  # Class 0
    X[50:, :50] -= 2  # Class 1
    
    y = np.hstack([np.zeros(50), np.ones(50)])
    
    # Apply FDA with feature selection
    X_fda, fda, selector = fda_with_feature_selection(X, y, n_features=100, n_components=1)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Original data (first 2 genes)
    plt.subplot(1, 3, 1)
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    plt.xlabel('Gene 1')
    plt.ylabel('Gene 2')
    plt.title('Original Data (First 2 Genes)')
    plt.legend()
    
    # Selected features
    plt.subplot(1, 3, 2)
    selected_features = selector.get_support()
    plt.bar(range(100), selected_features[:100])
    plt.xlabel('Gene Index')
    plt.ylabel('Selected')
    plt.title('Feature Selection')
    
    # FDA projection
    plt.subplot(1, 3, 3)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda[mask], np.zeros_like(X_fda[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.xlabel('FDA Component')
    plt.ylabel('')
    plt.title('FDA Projection')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fda, selector
```

## 9.5.8. Summary and Best Practices

### Key Takeaways

1. **FDA Objective**: Maximize between-class variance while minimizing within-class variance
2. **Supervised Nature**: Uses class labels to find discriminative directions
3. **Dimensionality Reduction**: Naturally reduces to $`K-1`$ dimensions
4. **Connection to LDA**: Equivalent under normality assumptions

### Best Practices

1. **Data Preprocessing**:
   - Standardize features
   - Handle missing values
   - Check for multicollinearity

2. **Dimensionality Management**:
   - Use regularization when $`p \gg n`$
   - Apply feature selection
   - Cross-validate results

3. **Model Validation**:
   - Check for overfitting
   - Use cross-validation
   - Monitor separation metrics

4. **Interpretation**:
   - Examine discriminant directions
   - Analyze explained variance ratios
   - Visualize projections

### When to Use FDA

**Use FDA when**:
- You need supervised dimensionality reduction
- Classes are well-separated
- Interpretability is important
- You want to reduce to $`K-1`$ dimensions

**Consider alternatives when**:
- Classes overlap significantly (use other methods)
- You need more than $`K-1`$ dimensions
- Data is non-linear (use kernel methods)

### Limitations

1. **Linear Assumption**: Only finds linear projections
2. **Overfitting Risk**: Can overfit in high dimensions
3. **Normality Assumption**: Implicit in the formulation
4. **Limited Dimensions**: Maximum $`K-1`$ components

Fisher Discriminant Analysis remains a powerful and interpretable method for supervised dimensionality reduction, providing a solid foundation for understanding the relationship between classes in high-dimensional data.
