# 7.1. Model-based Clustering

## 7.1.1. Introduction to Model-based Clustering

Model-based clustering is a principled approach to clustering that frames the problem as **mixture model estimation**. Unlike distance-based methods (K-means, hierarchical clustering), model-based clustering assumes that the data is generated from a mixture of probability distributions, where each cluster corresponds to a component of the mixture.

### Problem Formulation

Given a dataset $`X = \{x_1, x_2, \ldots, x_n\}`$ where each $`x_i \in \mathbb{R}^p`$, we assume the data is generated from a **finite mixture model**:

```math
f(x) = \sum_{k=1}^K \pi_k f_k(x; \theta_k)
```

where:
- $`K`$ is the number of mixture components (clusters)
- $`\pi_k`$ is the mixing weight for component $`k`$ ($`\pi_k \geq 0`$ and $`\sum_{k=1}^K \pi_k = 1`$)
- $`f_k(x; \theta_k)`$ is the probability density function of component $`k`$ with parameters $`\theta_k`$
- $`\theta = \{\pi_1, \ldots, \pi_K, \theta_1, \ldots, \theta_K\}`$ are the model parameters

### Key Advantages

- **Probabilistic framework**: Provides uncertainty quantification
- **Model selection**: Can use information criteria (AIC, BIC) to select K
- **Flexible distributions**: Can model different cluster shapes
- **Soft assignments**: Provides posterior probabilities of cluster membership
- **Theoretical foundation**: Based on well-established statistical theory

## 7.1.2. Gaussian Mixture Models (GMM)

The most common choice for model-based clustering is the **Gaussian Mixture Model (GMM)**, where each component follows a multivariate normal distribution:

```math
f_k(x; \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)
```

The complete GMM is:

```math
f(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)
```

### Parameter Interpretation

- $`\mu_k \in \mathbb{R}^p`$: Mean vector of component $`k`$ (cluster center)
- $`\Sigma_k \in \mathbb{R}^{p \times p}`$: Covariance matrix of component $`k`$ (cluster shape and orientation)
- $`\pi_k`$: Mixing weight (prior probability of belonging to cluster $`k`$)

### Cluster Assignment

Given the fitted model, we can assign observations to clusters using:

1. **Hard assignment**: $`z_i = \arg\max_k P(z_i = k | x_i)`$
2. **Soft assignment**: $`P(z_i = k | x_i) = \frac{\pi_k f_k(x_i; \theta_k)}{\sum_{l=1}^K \pi_l f_l(x_i; \theta_l)}`$

## 7.1.3. Maximum Likelihood Estimation

### Likelihood Function

The log-likelihood function for the mixture model is:

```math
\ell(\theta) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k f_k(x_i; \theta_k)
```

### Challenges in MLE

1. **Non-convex optimization**: The likelihood function has multiple local maxima
2. **Singularities**: When $`\sigma_k \to 0`$, the likelihood becomes unbounded
3. **Label switching**: The likelihood is invariant to component relabeling
4. **Computational complexity**: Direct optimization is computationally expensive

### Solution: Expectation-Maximization (EM) Algorithm

The EM algorithm provides an efficient way to find the MLE for mixture models.

## 7.1.4. The EM Algorithm for GMM

### Algorithm Overview

The EM algorithm alternates between two steps:

1. **E-step (Expectation)**: Compute posterior probabilities
2. **M-step (Maximization)**: Update model parameters

### E-step: Computing Posterior Probabilities

For each observation $`x_i`$ and component $`k`$, compute the posterior probability:

```math
\gamma_{ik} = P(z_i = k | x_i, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x_i; \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{l=1}^K \pi_l^{(t)} \mathcal{N}(x_i; \mu_l^{(t)}, \Sigma_l^{(t)})}
```

### M-step: Updating Parameters

Given the posterior probabilities, update the parameters:

**Mixing weights**:
```math
\pi_k^{(t+1)} = \frac{1}{n} \sum_{i=1}^n \gamma_{ik}
```

**Mean vectors**:
```math
\mu_k^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}
```

**Covariance matrices**:
```math
\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^T}{\sum_{i=1}^n \gamma_{ik}}
```

### Convergence

The algorithm converges when the log-likelihood improvement falls below a threshold:

```math
|\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})| < \epsilon
```

## 7.1.5. Model Selection

### Information Criteria

To select the optimal number of components $`K`$, we can use:

**Akaike Information Criterion (AIC)**:
```math
\text{AIC}(K) = 2\ell(\hat{\theta}_K) - 2p_K
```

**Bayesian Information Criterion (BIC)**:
```math
\text{BIC}(K) = 2\ell(\hat{\theta}_K) - p_K \log n
```

where $`p_K`$ is the number of parameters in a $`K`$-component model.

### Parameter Count

For a $`K`$-component GMM in $`p`$ dimensions:
- $`K-1`$ mixing weights (one is constrained by $`\sum \pi_k = 1`$)
- $`Kp`$ mean parameters
- $`K \cdot \frac{p(p+1)}{2}`$ covariance parameters (symmetric matrices)
- Total: $`p_K = K-1 + Kp + K \cdot \frac{p(p+1)}{2}`$

## 7.1.6. Covariance Structure Constraints

Different covariance structures can be imposed to control model complexity:

### Spherical (Equal Volume)
```math
\Sigma_k = \sigma_k^2 I
```

### Diagonal (Equal Shape)
```math
\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kp}^2)
```

### Tied (Equal Orientation)
```math
\Sigma_k = \lambda_k D
```

### Full (Unconstrained)
```math
\Sigma_k \text{ is any positive definite matrix}
```

## 7.1.7. Old Faithful Geyser Data Example

The Old Faithful Geyser data contains measurements of eruption duration and waiting time between eruptions. This data naturally forms clusters due to the geyser's bimodal behavior.

### Data Description
- **Duration**: Length of eruption in minutes
- **Waiting**: Time between eruptions in minutes
- **Natural clusters**: Short eruptions with short waits vs. long eruptions with long waits

### Model Fitting Results

**2-Component GMM**: Captures the main bimodal structure
- Component 1: Short eruptions, short waits
- Component 2: Long eruptions, long waits

**3-Component GMM**: Captures additional structure
- Component 1: Short eruptions, short waits
- Component 2: Long eruptions, long waits  
- Component 3: Intermediate eruptions, variable waits

## 7.1.8. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

class ModelBasedClustering:
    """Comprehensive model-based clustering implementation."""
    
    def __init__(self, n_components=2, covariance_type='full', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self.bic_scores = []
        self.aic_scores = []
        
    def fit(self, X):
        """Fit Gaussian Mixture Model to the data."""
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10
        )
        self.gmm.fit(X)
        return self
    
    def predict(self, X):
        """Predict cluster labels."""
        return self.gmm.predict(X)
    
    def predict_proba(self, X):
        """Predict cluster membership probabilities."""
        return self.gmm.predict_proba(X)
    
    def score(self, X):
        """Compute log-likelihood."""
        return self.gmm.score(X)
    
    def bic(self, X):
        """Compute BIC score."""
        return self.gmm.bic(X)
    
    def aic(self, X):
        """Compute AIC score."""
        return self.gmm.aic(X)
    
    def plot_clusters(self, X, title=None):
        """Visualize clustering results."""
        labels = self.predict(X)
        probas = self.predict_proba(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot cluster assignments
        scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0].set_title('Hard Cluster Assignments')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot uncertainty (max probability)
        max_proba = np.max(probas, axis=1)
        scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=max_proba, cmap='plasma', alpha=0.7)
        axes[1].set_title('Cluster Assignment Uncertainty')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.suptitle(title or f'GMM Clustering (K={self.n_components})')
        plt.tight_layout()
        plt.show()
    
    def plot_contours(self, X, title=None):
        """Plot GMM contours and data."""
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Compute GMM density
        density = np.exp(self.gmm.score_samples(grid))
        density = density.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Contour plot of mixture density
        plt.contour(xx, yy, density, levels=20, alpha=0.6, colors='black')
        plt.contourf(xx, yy, density, levels=20, alpha=0.3, cmap='viridis')
        
        # Plot data points
        labels = self.predict(X)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                             alpha=0.7, edgecolors='black', s=50)
        
        # Plot component means
        plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Component Means')
        
        plt.title(title or f'GMM Density Contours (K={self.n_components})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter)
        plt.show()
    
    def model_selection(self, X, K_range=range(1, 11)):
        """Perform model selection using BIC and AIC."""
        bic_scores = []
        aic_scores = []
        log_likelihoods = []
        
        for k in K_range:
            if k == 1:
                # Single component case
                bic_scores.append(np.inf)
                aic_scores.append(np.inf)
                log_likelihoods.append(-np.inf)
                continue
                
            gmm = GaussianMixture(n_components=k, covariance_type=self.covariance_type,
                                 random_state=self.random_state, n_init=10)
            gmm.fit(X)
            
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            log_likelihoods.append(gmm.score(X))
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(K_range, log_likelihoods, marker='o')
        axes[0].set_title('Log-Likelihood')
        axes[0].set_xlabel('Number of Components (K)')
        axes[0].set_ylabel('Log-Likelihood')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(K_range, bic_scores, marker='o', color='red')
        axes[1].set_title('BIC Score')
        axes[1].set_xlabel('Number of Components (K)')
        axes[1].set_ylabel('BIC')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(K_range, aic_scores, marker='o', color='green')
        axes[2].set_title('AIC Score')
        axes[2].set_xlabel('Number of Components (K)')
        axes[2].set_ylabel('AIC')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal K
        optimal_bic_k = K_range[np.argmin(bic_scores)]
        optimal_aic_k = K_range[np.argmin(aic_scores)]
        
        print(f"Optimal K (BIC): {optimal_bic_k}")
        print(f"Optimal K (AIC): {optimal_aic_k}")
        
        return {
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'log_likelihoods': log_likelihoods,
            'optimal_bic_k': optimal_bic_k,
            'optimal_aic_k': optimal_aic_k
        }

def load_old_faithful_data():
    """Load and preprocess Old Faithful Geyser data."""
    # Generate synthetic Old Faithful data (similar to the real data)
    np.random.seed(42)
    n_samples = 272
    
    # Component 1: Short eruptions, short waits
    n1 = int(0.6 * n_samples)
    duration1 = np.random.normal(2.0, 0.3, n1)
    waiting1 = np.random.normal(54, 8, n1)
    
    # Component 2: Long eruptions, long waits
    n2 = n_samples - n1
    duration2 = np.random.normal(4.3, 0.4, n2)
    waiting2 = np.random.normal(80, 12, n2)
    
    # Combine data
    duration = np.concatenate([duration1, duration2])
    waiting = np.concatenate([waiting1, waiting2])
    
    # Add some noise and intermediate cases
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    duration[noise_idx] += np.random.normal(0, 0.5, len(noise_idx))
    waiting[noise_idx] += np.random.normal(0, 10, len(noise_idx))
    
    return np.column_stack([duration, waiting])

def demonstrate_model_based_clustering():
    """Demonstrate model-based clustering with Old Faithful data."""
    
    # Load data
    X = load_old_faithful_data()
    
    print("=== Model-Based Clustering Demonstration ===\n")
    print(f"Dataset shape: {X.shape}")
    print(f"Features: Duration (minutes), Waiting time (minutes)")
    
    # Model selection
    print("\nPerforming model selection...")
    mbc = ModelBasedClustering()
    results = mbc.model_selection(X, K_range=range(1, 8))
    
    # Fit optimal model (using BIC)
    optimal_k = results['optimal_bic_k']
    print(f"\nFitting optimal model with K={optimal_k}...")
    
    mbc_optimal = ModelBasedClustering(n_components=optimal_k)
    mbc_optimal.fit(X)
    
    # Visualize results
    mbc_optimal.plot_clusters(X, f"Old Faithful Data - {optimal_k} Components")
    mbc_optimal.plot_contours(X, f"Old Faithful Data - {optimal_k} Components")
    
    # Compare with different K values
    print("\nComparing different numbers of components...")
    for k in [2, 3, 4]:
        mbc_k = ModelBasedClustering(n_components=k)
        mbc_k.fit(X)
        
        # Evaluate clustering
        labels = mbc_k.predict(X)
        silhouette = silhouette_score(X, labels)
        bic = mbc_k.bic(X)
        
        print(f"K={k}: Silhouette={silhouette:.3f}, BIC={bic:.1f}")
        
        # Plot
        mbc_k.plot_clusters(X, f"Old Faithful Data - {k} Components")
        mbc_k.plot_contours(X, f"Old Faithful Data - {k} Components")
    
    # Analyze component parameters
    print(f"\nComponent parameters for K={optimal_k}:")
    for k in range(optimal_k):
        print(f"Component {k+1}:")
        print(f"  Mixing weight: {mbc_optimal.gmm.weights_[k]:.3f}")
        print(f"  Mean: {mbc_optimal.gmm.means_[k]}")
        print(f"  Covariance:\n{mbc_optimal.gmm.covariances_[k]}")

if __name__ == "__main__":
    demonstrate_model_based_clustering()
```

## 7.1.9. R Implementation

```r
# Model-Based Clustering Implementation in R
library(mclust)
library(ggplot2)
library(dplyr)
library(gridExtra)

ModelBasedClustering <- setRefClass("ModelBasedClustering",
  fields = list(
    n_components = "numeric",
    covariance_type = "character",
    gmm = "ANY",
    bic_scores = "numeric",
    aic_scores = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_components = 2, covariance_type = "VVV") {
      n_components <<- n_components
      covariance_type <<- covariance_type
    },
    
    fit = function(X) {
      # Fit Gaussian Mixture Model using mclust
      gmm <<- Mclust(X, G = n_components, modelNames = covariance_type)
      invisible(.self)
    },
    
    predict = function(X) {
      predict(gmm, X)$classification
    },
    
    predict_proba = function(X) {
      predict(gmm, X)$z
    },
    
    score = function(X) {
      # Log-likelihood
      sum(logLik(gmm))
    },
    
    bic = function(X) {
      BIC(gmm)
    },
    
    aic = function(X) {
      AIC(gmm)
    },
    
    plot_clusters = function(X, title = NULL) {
      labels <- predict(X)
      probas <- predict_proba(X)
      max_proba <- apply(probas, 1, max)
      
      # Create data frames for plotting
      df_clusters <- data.frame(
        x = X[, 1],
        y = X[, 2],
        cluster = factor(labels)
      )
      
      df_uncertainty <- data.frame(
        x = X[, 1],
        y = X[, 2],
        uncertainty = 1 - max_proba
      )
      
      # Plot cluster assignments
      p1 <- ggplot(df_clusters, aes(x = x, y = y, color = cluster)) +
        geom_point(alpha = 0.7) +
        labs(title = "Hard Cluster Assignments",
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
      
      # Plot uncertainty
      p2 <- ggplot(df_uncertainty, aes(x = x, y = y, color = uncertainty)) +
        geom_point(alpha = 0.7) +
        labs(title = "Cluster Assignment Uncertainty",
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_c()
      
      # Combine plots
      grid.arrange(p1, p2, ncol = 2, 
                   top = title %||% paste("GMM Clustering (K=", n_components, ")"))
    },
    
    plot_contours = function(X, title = NULL) {
      # Create grid for contour plot
      x_range <- range(X[, 1])
      y_range <- range(X[, 2])
      
      x_grid <- seq(x_range[1] - 1, x_range[2] + 1, length.out = 100)
      y_grid <- seq(y_range[1] - 1, y_range[2] + 1, length.out = 100)
      
      grid_points <- expand.grid(x = x_grid, y = y_grid)
      
      # Compute density
      density_values <- predict(gmm, as.matrix(grid_points))$z
      total_density <- rowSums(density_values)
      
      grid_points$density <- total_density
      
      # Plot
      p <- ggplot() +
        geom_contour(data = grid_points, aes(x = x, y = y, z = density), 
                     bins = 20, alpha = 0.6) +
        geom_contour_filled(data = grid_points, aes(x = x, y = y, z = density), 
                           alpha = 0.3) +
        geom_point(data = data.frame(x = X[, 1], y = X[, 2], 
                                    cluster = factor(predict(X))),
                  aes(x = x, y = y, color = cluster), alpha = 0.7) +
        geom_point(data = data.frame(x = gmm$parameters$mean[1, ], 
                                    y = gmm$parameters$mean[2, ]),
                  aes(x = x, y = y), color = "red", shape = 4, size = 3) +
        labs(title = title %||% paste("GMM Density Contours (K=", n_components, ")"),
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
      
      print(p)
    },
    
    model_selection = function(X, K_range = 1:10) {
      bic_scores <- numeric(length(K_range))
      aic_scores <- numeric(length(K_range))
      log_likelihoods <- numeric(length(K_range))
      
      for (i in seq_along(K_range)) {
        k <- K_range[i]
        if (k == 1) {
          bic_scores[i] <- Inf
          aic_scores[i] <- Inf
          log_likelihoods[i] <- -Inf
          next
        }
        
        gmm_temp <- Mclust(X, G = k, modelNames = covariance_type)
        bic_scores[i] <- BIC(gmm_temp)
        aic_scores[i] <- AIC(gmm_temp)
        log_likelihoods[i] <- sum(logLik(gmm_temp))
      }
      
      # Create plots
      df_results <- data.frame(
        K = K_range,
        BIC = bic_scores,
        AIC = aic_scores,
        LogLik = log_likelihoods
      )
      
      p1 <- ggplot(df_results, aes(x = K, y = LogLik)) +
        geom_line() + geom_point() +
        labs(title = "Log-Likelihood", x = "Number of Components (K)", y = "Log-Likelihood") +
        theme_minimal()
      
      p2 <- ggplot(df_results, aes(x = K, y = BIC)) +
        geom_line(color = "red") + geom_point(color = "red") +
        labs(title = "BIC Score", x = "Number of Components (K)", y = "BIC") +
        theme_minimal()
      
      p3 <- ggplot(df_results, aes(x = K, y = AIC)) +
        geom_line(color = "green") + geom_point(color = "green") +
        labs(title = "AIC Score", x = "Number of Components (K)", y = "AIC") +
        theme_minimal()
      
      grid.arrange(p1, p2, p3, ncol = 3)
      
      # Find optimal K
      optimal_bic_k <- K_range[which.min(bic_scores)]
      optimal_aic_k <- K_range[which.min(aic_scores)]
      
      cat("Optimal K (BIC):", optimal_bic_k, "\n")
      cat("Optimal K (AIC):", optimal_aic_k, "\n")
      
      list(
        bic_scores = bic_scores,
        aic_scores = aic_scores,
        log_likelihoods = log_likelihoods,
        optimal_bic_k = optimal_bic_k,
        optimal_aic_k = optimal_aic_k
      )
    }
  )
)

# Load Old Faithful data
load_old_faithful_data <- function() {
  # Use built-in faithful data
  data(faithful)
  as.matrix(faithful)
}

# Example usage and demonstration
demonstrate_model_based_clustering <- function() {
  cat("=== Model-Based Clustering Demonstration ===\n\n")
  
  # Load data
  X <- load_old_faithful_data()
  cat("Dataset shape:", dim(X), "\n")
  cat("Features: Duration (minutes), Waiting time (minutes)\n")
  
  # Model selection
  cat("\nPerforming model selection...\n")
  mbc <- ModelBasedClustering$new()
  results <- mbc$model_selection(X, K_range = 1:8)
  
  # Fit optimal model
  optimal_k <- results$optimal_bic_k
  cat(sprintf("\nFitting optimal model with K=%d...\n", optimal_k))
  
  mbc_optimal <- ModelBasedClustering$new(n_components = optimal_k)
  mbc_optimal$fit(X)
  
  # Visualize results
  mbc_optimal$plot_clusters(X, sprintf("Old Faithful Data - %d Components", optimal_k))
  mbc_optimal$plot_contours(X, sprintf("Old Faithful Data - %d Components", optimal_k))
  
  # Compare different K values
  cat("\nComparing different numbers of components...\n")
  for (k in c(2, 3, 4)) {
    mbc_k <- ModelBasedClustering$new(n_components = k)
    mbc_k$fit(X)
    
    # Evaluate clustering
    labels <- mbc_k$predict(X)
    silhouette <- mean(silhouette(labels, dist(X))[, 3])
    bic <- mbc_k$bic(X)
    
    cat(sprintf("K=%d: Silhouette=%.3f, BIC=%.1f\n", k, silhouette, bic))
    
    # Plot
    mbc_k$plot_clusters(X, sprintf("Old Faithful Data - %d Components", k))
    mbc_k$plot_contours(X, sprintf("Old Faithful Data - %d Components", k))
  }
  
  # Analyze component parameters
  cat(sprintf("\nComponent parameters for K=%d:\n", optimal_k))
  for (k in 1:optimal_k) {
    cat(sprintf("Component %d:\n", k))
    cat(sprintf("  Mixing weight: %.3f\n", mbc_optimal$gmm$parameters$pro[k]))
    cat(sprintf("  Mean: [%.2f, %.2f]\n", 
                mbc_optimal$gmm$parameters$mean[1, k],
                mbc_optimal$gmm$parameters$mean[2, k]))
    cat("  Covariance:\n")
    print(mbc_optimal$gmm$parameters$variance$sigma[, , k])
  }
}

# Run demonstration
demonstrate_model_based_clustering()
```

## 7.1.10. Summary and Best Practices

### Key Takeaways

1. **Model-based clustering provides a probabilistic framework** for clustering
2. **Gaussian Mixture Models are the most common choice** for continuous data
3. **EM algorithm efficiently finds MLE** for mixture model parameters
4. **Information criteria (BIC, AIC) help select optimal K**
5. **Soft assignments provide uncertainty quantification**

### Model Selection Guidelines

**Use BIC when:**
- You want to penalize model complexity more heavily
- Sample size is large
- You prefer simpler models

**Use AIC when:**
- You want to balance fit and complexity
- Sample size is small
- You prefer more complex models

### Common Pitfalls

1. **Local optima**: EM can converge to suboptimal solutions
2. **Singularities**: Components can collapse to single points
3. **Label switching**: Component labels may not be consistent across runs
4. **Overfitting**: Too many components can lead to overfitting

### Advanced Topics

- **Non-Gaussian mixtures**: For non-normal data (Poisson, etc.)
- **Regularization**: To prevent singularities
- **Bayesian mixtures**: For uncertainty in K
- **Semi-supervised learning**: Incorporating labeled data
