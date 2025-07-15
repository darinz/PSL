# 9.6. Naive Bayes Classifiers

## 9.6.0. Introduction and Motivation

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with a strong (naive) assumption of conditional independence between features. Despite its simplicity, Naive Bayes often performs surprisingly well and is widely used in text classification, spam filtering, medical diagnosis, and many other applications.

### The "Naive" Assumption

The "Naive" part comes from the **conditional independence assumption**: given the class label, all features are assumed to be independent of each other. This is often violated in real-world data, but the method still works well in practice.

### Why Naive Bayes Works

1. **Computational Efficiency**: Independence assumption dramatically reduces parameter count
2. **Robustness**: Works well even when independence assumption is violated
3. **Interpretability**: Easy to understand and explain
4. **Small Sample Performance**: Works well with limited training data

## 9.6.1. Mathematical Foundation

### Bayes' Theorem

The foundation of Naive Bayes is Bayes' theorem:

```math
P(Y=k | X=x) = \frac{P(X=x | Y=k) \cdot P(Y=k)}{P(X=x)}
```

Where:
- $`P(Y=k | X=x)`$ is the **posterior probability** of class $`k`$ given features $`x`$
- $`P(X=x | Y=k)`$ is the **likelihood** of features $`x`$ given class $`k`$
- $`P(Y=k)`$ is the **prior probability** of class $`k`$
- $`P(X=x)`$ is the **evidence** (normalizing constant)

### The Decision Function

For classification, we want to find the class that maximizes the posterior probability:

```math
\hat{y} = \arg\max_k P(Y=k | X=x)
```

Since $`P(X=x)`$ is the same for all classes, we can ignore it and maximize:

```math
\hat{y} = \arg\max_k P(X=x | Y=k) \cdot P(Y=k)
```

Or equivalently, using logarithms to avoid numerical underflow:

```math
\hat{y} = \arg\max_k \log P(X=x | Y=k) + \log P(Y=k)
```

### The Naive Independence Assumption

The key assumption is that features are conditionally independent given the class:

```math
P(X=x | Y=k) = P(X_1=x_1 | Y=k) \cdot P(X_2=x_2 | Y=k) \cdots P(X_p=x_p | Y=k)
```

This allows us to factorize the joint likelihood into a product of individual feature likelihoods:

```math
f_k(x) = f_{k1}(x_1) \times f_{k2}(x_2) \times \cdots \times f_{kp}(x_p)
```

Where $`f_{kj}(x_j)`$ is the probability density (or mass) function for feature $`j`$ in class $`k`$.

## 9.6.2. Parameter Estimation

### Prior Probabilities

The prior probability of class $`k`$ is estimated as:

```math
\hat{\pi}_k = P(Y=k) = \frac{n_k}{n}
```

Where $`n_k`$ is the number of samples in class $`k`$ and $`n`$ is the total number of samples.

### Likelihood Estimation

The estimation of $`f_{kj}(x_j)`$ depends on the type of features:

#### 1. Discrete Features (Categorical)

For discrete features, we use empirical probabilities:

```math
\hat{f}_{kj}(x_j) = P(X_j = x_j | Y = k) = \frac{\text{count}(X_j = x_j, Y = k)}{\text{count}(Y = k)}
```

#### 2. Continuous Features (Numerical)

For continuous features, we have two options:

**Parametric Approach (Gaussian Naive Bayes)**:
```math
f_{kj}(x_j) = \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
```

Where:
- $`\mu_{kj} = \frac{1}{n_k} \sum_{i: y_i=k} x_{ij}`$ (mean of feature $`j`$ in class $`k`$)
- $`\sigma_{kj}^2 = \frac{1}{n_k-1} \sum_{i: y_i=k} (x_{ij} - \mu_{kj})^2`$ (variance of feature $`j`$ in class $`k`$)

**Non-parametric Approach (Kernel Density Estimation)**:
```math
f_{kj}(x_j) = \frac{1}{n_k h} \sum_{i: y_i=k} K\left(\frac{x_j - x_{ij}}{h}\right)
```

Where $`K`$ is a kernel function (e.g., Gaussian) and $`h`$ is the bandwidth.

### Parameter Count

For **parametric Naive Bayes** with $`p`$ features and $`K`$ classes:
- **Means**: $`K \times p`$ parameters
- **Variances**: $`K \times p`$ parameters  
- **Priors**: $`K`$ parameters
- **Total**: $`2Kp + K`$ parameters

This is much smaller than the $`K \times 2^p`$ parameters needed without the independence assumption.

## 9.6.3. Classification Decision Function

### Log-Likelihood Formulation

To avoid numerical underflow, we work with logarithms. The decision function becomes:

```math
d_k(x) = \log P(Y=k) + \sum_{j=1}^p \log f_{kj}(x_j)
```

### Gaussian Naive Bayes Decision Function

For Gaussian Naive Bayes, the decision function is:

```math
\begin{split}
d_k(x) &= \log \pi_k + \sum_{j=1}^p \log f_{kj}(x_j) \\
&= \log \pi_k + \sum_{j=1}^p \log \left(\frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)\right) \\
&= \log \pi_k + \sum_{j=1}^p \left(-\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma_{kj}^2) - \frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right) \\
&= \log \pi_k - \frac{p}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^p \log(\sigma_{kj}^2) - \frac{1}{2}\sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
\end{split}
```

### Numerical Stability Issues

The key insight is that we can drop constant terms that don't depend on the class:

```math
d_k(x) = \log \pi_k - \frac{1}{2}\sum_{j=1}^p \log(\sigma_{kj}^2) - \frac{1}{2}\sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
```

**Critical Issue**: When $`x_j`$ is far from $`\mu_{kj}`$, the exponential term becomes very small, leading to numerical underflow. Some implementations truncate these values, which can lead to incorrect predictions.

## 9.6.4. Implementation from Scratch

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implementation from scratch
    """
    
    def __init__(self, feature_type='gaussian'):
        self.feature_type = feature_type
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.variances_ = None
        
    def fit(self, X, y):
        """
        Fit Naive Bayes classifier
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        # Estimate parameters for each class
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            n_class = np.sum(class_mask)
            
            # Prior probability
            self.priors_[i] = n_class / n_samples
            
            # Mean and variance for each feature
            self.means_[i] = np.mean(class_data, axis=0)
            self.variances_[i] = np.var(class_data, axis=0, ddof=1)
            
            # Add small constant to avoid zero variance
            self.variances_[i] = np.maximum(self.variances_[i], 1e-9)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        log_proba = self.predict_log_proba(X)
        # Convert log probabilities to probabilities
        proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
        return proba / np.sum(proba, axis=1, keepdims=True)
    
    def predict_log_proba(self, X):
        """
        Predict log class probabilities (numerically stable)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # Log prior
            log_proba[:, i] = np.log(self.priors_[i])
            
            # Log likelihood for each feature
            for j in range(n_features):
                mu = self.means_[i, j]
                sigma2 = self.variances_[i, j]
                
                # Gaussian log-likelihood
                log_likelihood = -0.5 * np.log(2 * np.pi * sigma2) - \
                                0.5 * (X[:, j] - mu)**2 / sigma2
                
                log_proba[:, i] += log_likelihood
        
        return log_proba
    
    def score(self, X, y):
        """
        Return accuracy score
        """
        return accuracy_score(y, self.predict(X))

def demonstrate_naive_bayes():
    """
    Demonstrate Naive Bayes with synthetic data
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0: centered at (0, 0, 0, 0)
    X0 = np.random.multivariate_normal([0, 0, 0, 0], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    # Class 1: centered at (2, 2, 0, 0)
    X1 = np.random.multivariate_normal([2, 2, 0, 0], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    # Class 2: centered at (0, 0, 2, 2)
    X2 = np.random.multivariate_normal([0, 0, 2, 2], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit our implementation
    nb_scratch = NaiveBayesClassifier()
    nb_scratch.fit(X_train, y_train)
    
    # Fit sklearn implementation
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    
    # Compare predictions
    y_pred_scratch = nb_scratch.predict(X_test)
    y_pred_sklearn = nb_sklearn.predict(X_test)
    
    print("Naive Bayes Results:")
    print("-" * 40)
    print(f"Our Implementation Accuracy: {nb_scratch.score(X_test, y_test):.4f}")
    print(f"Sklearn Implementation Accuracy: {nb_sklearn.score(X_test, y_test):.4f}")
    
    # Compare parameters
    print(f"\nParameter Comparison:")
    print(f"Our means shape: {nb_scratch.means_.shape}")
    print(f"Sklearn means shape: {nb_sklearn.theta_.shape}")
    print(f"Our variances shape: {nb_scratch.variances_.shape}")
    print(f"Sklearn variances shape: {nb_sklearn.var_.shape}")
    
    # Visualize decision boundaries (first 2 features)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[0].set_title('Original Data (Features 0 & 1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Create test points (use mean values for other features)
    test_points = np.zeros((10000, 4))
    test_points[:, 0] = xx.ravel()
    test_points[:, 1] = yy.ravel()
    test_points[:, 2] = np.mean(X[:, 2])  # Use mean of feature 2
    test_points[:, 3] = np.mean(X[:, 3])  # Use mean of feature 3
    
    Z = nb_scratch.predict(test_points)
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    for i in range(3):
        mask = y == i
        axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[1].set_title('Decision Boundaries (Our Implementation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Feature importance (based on variance ratios)
    feature_importance = np.zeros(n_features)
    for j in range(n_features):
        # Calculate ratio of between-class to within-class variance
        overall_mean = np.mean(X[:, j])
        between_var = np.sum([np.sum(y == c) * (np.mean(X[y == c, j]) - overall_mean)**2 
                             for c in np.unique(y)])
        within_var = np.sum([np.sum((X[y == c, j] - np.mean(X[y == c, j]))**2) 
                            for c in np.unique(y)])
        feature_importance[j] = between_var / within_var if within_var > 0 else 0
    
    axes[2].bar(range(n_features), feature_importance)
    axes[2].set_title('Feature Importance (Variance Ratio)')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Between/Within Variance Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return nb_scratch, nb_sklearn

# Run demonstration
if __name__ == "__main__":
    nb_scratch, nb_sklearn = demonstrate_naive_bayes()
```

### R Implementation

```r
# Naive Bayes Classifier in R
library(e1071)
library(ggplot2)
library(gridExtra)

# Custom Naive Bayes implementation
naive_bayes_scratch <- function(X, y) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Initialize parameters
  priors <- rep(0, n_classes)
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  variances <- matrix(0, nrow = n_classes, ncol = n_features)
  
  # Estimate parameters for each class
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    n_class <- sum(class_mask)
    
    # Prior probability
    priors[i] <- n_class / n_samples
    
    # Mean and variance for each feature
    means[i,] <- colMeans(class_data)
    variances[i,] <- apply(class_data, 2, var)
    
    # Add small constant to avoid zero variance
    variances[i,] <- pmax(variances[i,], 1e-9)
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    variances = variances
  ))
}

# Prediction function
predict_naive_bayes <- function(model, X) {
  X <- as.matrix(X)
  n_samples <- nrow(X)
  n_classes <- length(model$classes)
  
  log_proba <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Log prior
    log_proba[, i] <- log(model$priors[i])
    
    # Log likelihood for each feature
    for (j in 1:ncol(X)) {
      mu <- model$means[i, j]
      sigma2 <- model$variances[i, j]
      
      # Gaussian log-likelihood
      log_likelihood <- -0.5 * log(2 * pi * sigma2) - 
                       0.5 * (X[, j] - mu)^2 / sigma2
      
      log_proba[, i] <- log_proba[, i] + log_likelihood
    }
  }
  
  # Return predicted classes
  predictions <- model$classes[apply(log_proba, 1, which.max)]
  return(predictions)
}

# Demonstrate Naive Bayes
demonstrate_naive_bayes_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 1000
  n_features <- 4
  
  # Generate 3 classes
  n_per_class <- n_samples %/% 3
  
  # Class 0
  X0 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 0, 0), 
                      Sigma = diag(4))
  
  # Class 1
  X1 <- MASS::mvrnorm(n_per_class, mu = c(2, 2, 0, 0), 
                      Sigma = diag(4))
  
  # Class 2
  X2 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 2, 2), 
                      Sigma = diag(4))
  
  X <- rbind(X0, X1, X2)
  y <- rep(c(0, 1, 2), each = n_per_class)
  
  # Split data
  train_idx <- sample(1:nrow(X), 0.7 * nrow(X))
  X_train <- X[train_idx,]
  y_train <- y[train_idx]
  X_test <- X[-train_idx,]
  y_test <- y[-train_idx]
  
  # Fit our implementation
  nb_model <- naive_bayes_scratch(X_train, y_train)
  y_pred_scratch <- predict_naive_bayes(nb_model, X_test)
  
  # Fit e1071 implementation
  nb_e1071 <- naiveBayes(X_train, y_train)
  y_pred_e1071 <- predict(nb_e1071, X_test)
  
  # Calculate accuracy
  accuracy_scratch <- mean(y_pred_scratch == y_test)
  accuracy_e1071 <- mean(y_pred_e1071 == y_test)
  
  cat("Naive Bayes Results:\n")
  cat("-" * 40, "\n")
  cat("Our Implementation Accuracy:", round(accuracy_scratch, 4), "\n")
  cat("e1071 Implementation Accuracy:", round(accuracy_e1071, 4), "\n")
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  # Plot original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = "Original Data (Features 1 & 2)", color = "Class") +
    theme_minimal()
  
  # Feature importance
  feature_importance <- rep(0, n_features)
  for (j in 1:n_features) {
    overall_mean <- mean(X[, j])
    between_var <- sum(sapply(unique(y), function(c) {
      sum(y == c) * (mean(X[y == c, j]) - overall_mean)^2
    }))
    within_var <- sum(sapply(unique(y), function(c) {
      sum((X[y == c, j] - mean(X[y == c, j]))^2)
    }))
    feature_importance[j] <- between_var / within_var
  }
  
  df_importance <- data.frame(
    feature = 1:n_features,
    importance = feature_importance
  )
  
  p2 <- ggplot(df_importance, aes(x = feature, y = importance)) +
    geom_bar(stat = "identity") +
    labs(title = "Feature Importance (Variance Ratio)", 
         x = "Feature Index", y = "Between/Within Variance Ratio") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(nb_model = nb_model, nb_e1071 = nb_e1071))
}

# Run demonstration
results <- demonstrate_naive_bayes_r()
```

## 9.6.5. Numerical Stability Issues

### The Problem

When computing probabilities for points far from the class means, the Gaussian PDF becomes extremely small:

```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
```

For large $`|x - \mu|`$, this approaches zero, causing numerical underflow.

### Demonstration of the Issue

```python
def demonstrate_numerical_issues():
    """
    Demonstrate numerical stability issues in Naive Bayes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate data with extreme values
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.normal(0, 1, 100)
    
    # Extreme data
    X_extreme = np.random.normal(10, 1, 100)
    
    # Compute Gaussian PDF
    def gaussian_pdf(x, mu, sigma):
        return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    def gaussian_log_pdf(x, mu, sigma):
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2
    
    # Test points
    test_points = np.linspace(-5, 15, 1000)
    
    # Compute probabilities
    pdf_normal = gaussian_pdf(test_points, 0, 1)
    pdf_extreme = gaussian_pdf(test_points, 10, 1)
    
    log_pdf_normal = gaussian_log_pdf(test_points, 0, 1)
    log_pdf_extreme = gaussian_log_pdf(test_points, 10, 1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # PDF for normal data
    axes[0, 0].plot(test_points, pdf_normal)
    axes[0, 0].set_title('Gaussian PDF (μ=0, σ=1)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PDF for extreme data
    axes[0, 1].plot(test_points, pdf_extreme)
    axes[0, 1].set_title('Gaussian PDF (μ=10, σ=1)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log PDF for normal data
    axes[1, 0].plot(test_points, log_pdf_normal)
    axes[1, 0].set_title('Gaussian Log-PDF (μ=0, σ=1)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('log f(x)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log PDF for extreme data
    axes[1, 1].plot(test_points, log_pdf_extreme)
    axes[1, 1].set_title('Gaussian Log-PDF (μ=10, σ=1)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('log f(x)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate numerical issues
    print("Numerical Stability Analysis:")
    print("-" * 40)
    
    # Test with extreme point
    extreme_point = 20
    mu = 0
    sigma = 1
    
    pdf_value = gaussian_pdf(extreme_point, mu, sigma)
    log_pdf_value = gaussian_log_pdf(extreme_point, mu, sigma)
    
    print(f"Point: {extreme_point}")
    print(f"Mean: {mu}, Std: {sigma}")
    print(f"PDF value: {pdf_value:.2e}")
    print(f"Log-PDF value: {log_pdf_value:.4f}")
    print(f"Recovered PDF: {np.exp(log_pdf_value):.2e}")
    
    return pdf_value, log_pdf_value

# Run numerical stability demonstration
pdf_val, log_pdf_val = demonstrate_numerical_issues()
```

### Solutions

#### 1. Use Log-Probabilities (Recommended)

Always work with log-probabilities to avoid underflow:

```python
def safe_naive_bayes_predict(X, model):
    """
    Safe Naive Bayes prediction using log-probabilities
    """
    log_proba = model.predict_log_proba(X)
    return model.classes_[np.argmax(log_proba, axis=1)]
```

#### 2. Add Regularization

Add small constants to prevent zero variances:

```python
def regularized_naive_bayes(X, y, epsilon=1e-9):
    """
    Naive Bayes with regularization
    """
    # ... existing code ...
    
    # Regularize variances
    self.variances_ = np.maximum(self.variances_, epsilon)
    
    return self
```

#### 3. Truncation (Not Recommended)

Some packages truncate very small probabilities, but this can lead to incorrect predictions:

```python
def truncated_naive_bayes(X, model, threshold=1e-10):
    """
    Naive Bayes with truncation (not recommended)
    """
    proba = model.predict_proba(X)
    proba = np.maximum(proba, threshold)  # Truncate small values
    return model.classes_[np.argmax(proba, axis=1)]
```

## 9.6.6. Variants of Naive Bayes

### 1. Gaussian Naive Bayes

For continuous features, assumes Gaussian distribution:

```python
class GaussianNaiveBayes(NaiveBayesClassifier):
    def __init__(self):
        super().__init__(feature_type='gaussian')
```

### 2. Multinomial Naive Bayes

For discrete count data (e.g., text classification):

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        
    def fit(self, X, y):
        # Count features for each class
        # Apply Laplace smoothing
        # Estimate class-conditional probabilities
        pass
```

### 3. Bernoulli Naive Bayes

For binary features:

```python
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of feature being 1 for each class
        # Apply Laplace smoothing
        pass
```

### 4. Categorical Naive Bayes

For categorical features:

```python
class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of each category for each class
        # Apply Laplace smoothing
        pass
```

## 9.6.7. Real-World Applications

### Example 1: Text Classification

```python
def text_classification_example():
    """
    Naive Bayes for text classification
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Sample text data
    texts = [
        "great movie amazing acting",
        "terrible film waste of time", 
        "excellent performance brilliant",
        "boring plot disappointing",
        "fantastic story wonderful",
        "awful acting bad script",
        "outstanding film superb",
        "poor quality terrible",
        "incredible movie perfect",
        "horrible waste bad"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fit Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    
    print("Text Classification Results:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, 
                               target_names=['Negative', 'Positive']))
    
    # Feature importance
    feature_names = vectorizer.get_feature_names_out()
    log_probs = nb.feature_log_prob_
    
    # Show most discriminative words
    positive_words = log_probs[1] - log_probs[0]
    negative_words = log_probs[0] - log_probs[1]
    
    print("\nMost Positive Words:")
    pos_indices = np.argsort(positive_words)[-5:]
    for idx in pos_indices:
        print(f"  {feature_names[idx]}: {positive_words[idx]:.3f}")
    
    print("\nMost Negative Words:")
    neg_indices = np.argsort(negative_words)[-5:]
    for idx in neg_indices:
        print(f"  {feature_names[idx]}: {negative_words[idx]:.3f}")
    
    return nb, vectorizer
```

### Example 2: Medical Diagnosis

```python
def medical_diagnosis_example():
    """
    Naive Bayes for medical diagnosis
    """
    # Simulate medical data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, blood_pressure, cholesterol, glucose
    age = np.random.normal(50, 15, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    glucose = np.random.normal(100, 20, n_samples)
    
    X = np.column_stack([age, blood_pressure, cholesterol, glucose])
    
    # Disease risk based on features
    risk_score = (age * 0.1 + (blood_pressure - 120) * 0.05 + 
                  (cholesterol - 200) * 0.02 + (glucose - 100) * 0.03 +
                  np.random.normal(0, 0.1, n_samples))
    
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit Naive Bayes
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Medical Diagnosis Results:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 'Glucose']
    feature_importance = np.zeros(4)
    
    for j in range(4):
        overall_mean = np.mean(X[:, j])
        between_var = np.sum([np.sum(y == c) * (np.mean(X[y == c, j]) - overall_mean)**2 
                             for c in np.unique(y)])
        within_var = np.sum([np.sum((X[y == c, j] - np.mean(X[y == c, j]))**2) 
                            for c in np.unique(y)])
        feature_importance[j] = between_var / within_var if within_var > 0 else 0
    
    # Plot feature importance
    plt.figure(figsize=(10, 4))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance in Medical Diagnosis')
    plt.ylabel('Between/Within Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return nb, feature_importance
```

## 9.6.8. Advantages and Limitations

### Advantages

1. **Simplicity**: Easy to understand and implement
2. **Speed**: Fast training and prediction
3. **Small Sample Performance**: Works well with limited data
4. **Interpretability**: Clear probabilistic interpretation
5. **Handles Missing Data**: Can handle missing features gracefully

### Limitations

1. **Independence Assumption**: Often violated in real data
2. **Feature Scaling**: Sensitive to feature scaling
3. **Zero Frequency Problem**: Can't handle unseen feature values
4. **Continuous Features**: Assumes specific distributions
5. **Correlated Features**: Performance degrades with correlated features

### When to Use Naive Bayes

**Use Naive Bayes when**:
- You have limited training data
- Features are approximately independent
- You need fast training and prediction
- Interpretability is important
- You're doing text classification

**Consider alternatives when**:
- Features are highly correlated
- You have complex feature interactions
- You need high accuracy (consider ensemble methods)
- You have large amounts of training data

## 9.6.9. Summary and Best Practices

### Key Takeaways

1. **Independence Assumption**: The core assumption that makes Naive Bayes "naive"
2. **Log-Probabilities**: Always use log-probabilities for numerical stability
3. **Parameter Count**: Only $`2Kp + K`$ parameters needed
4. **Variants**: Choose the right variant for your data type

### Best Practices

1. **Data Preprocessing**:
   - Handle missing values appropriately
   - Scale features if using Gaussian Naive Bayes
   - Apply Laplace smoothing for discrete features

2. **Model Selection**:
   - Use Gaussian NB for continuous features
   - Use Multinomial NB for count data
   - Use Bernoulli NB for binary features

3. **Numerical Stability**:
   - Always work with log-probabilities
   - Add small constants to prevent zero variances
   - Avoid truncation of small probabilities

4. **Evaluation**:
   - Use cross-validation for small datasets
   - Check for feature independence violations
   - Monitor for numerical issues

### Implementation Checklist

- [ ] Choose appropriate Naive Bayes variant
- [ ] Handle missing values
- [ ] Apply feature scaling if needed
- [ ] Use log-probabilities for numerical stability
- [ ] Add regularization to prevent zero variances
- [ ] Validate independence assumption
- [ ] Cross-validate model performance

Naive Bayes remains a powerful and interpretable classification method that provides an excellent baseline for many machine learning problems, especially when computational efficiency and interpretability are important.
