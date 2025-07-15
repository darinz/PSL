# 9.3. Quadratic Discriminant Analysis

## 9.3.1. Introduction to QDA

Quadratic Discriminant Analysis (QDA) is a powerful classification method that models each class as a multivariate Gaussian distribution with its own mean vector and covariance matrix. Unlike Linear Discriminant Analysis (LDA), which assumes all classes share the same covariance structure, QDA allows for class-specific covariance matrices, making it more flexible for capturing complex decision boundaries.

### Key Characteristics of QDA

1. **Class-Specific Covariances**: Each class has its own covariance matrix $`\Sigma_k`$
2. **Quadratic Decision Boundaries**: The decision function is quadratic in the feature vector
3. **Generative Model**: Models the joint distribution $`P(X, Y)`$ through class-conditional densities
4. **Bayes Optimal**: Under Gaussian assumptions, QDA provides the Bayes optimal classifier

### When to Use QDA

- Classes have different covariance structures
- Sufficient data to estimate class-specific covariances reliably
- Non-linear decision boundaries are needed
- High-dimensional data with enough samples per class

## 9.3.2. Mathematical Foundation

### Multivariate Gaussian Distribution

For each class $`k`$, we assume the feature vector $`X`$ follows a multivariate normal distribution:

```math
X \mid Y = k \sim \mathcal{N}(\mu_k, \Sigma_k)
```

where:
- $`\mu_k \in \mathbb{R}^p`$ is the mean vector for class $`k`$
- $`\Sigma_k \in \mathbb{R}^{p \times p}`$ is the covariance matrix for class $`k`$

### Parameter Notation

Let's define the precision matrix (inverse covariance) as $`\Theta_k = \Sigma_k^{-1}`$:

```math
\mu_k = \begin{pmatrix} 
\mu_{k,1} \\ 
\mu_{k,2} \\ 
\vdots \\ 
\mu_{k,p} 
\end{pmatrix}_{p \times 1}, \quad
\Theta_k = \Sigma_k^{-1} = \begin{pmatrix} 
\theta_{k,11} & \cdots & \theta_{k,1p} \\ 
\vdots & \ddots & \vdots \\ 
\theta_{k,p1} & \cdots & \theta_{k,pp} 
\end{pmatrix}_{p \times p}
```

### Class-Conditional Density Function

The probability density function for class $`k`$ is:

```math
f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)
```

The quadratic term in the exponent can be expanded as:

```math
(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) = \sum_{j=1}^p \sum_{l=1}^p \theta_{k,jl} (x_j - \mu_{k,j}) (x_l - \mu_{k,l})
```

### Bayes Decision Rule

Using Bayes' theorem, the posterior probability is:

```math
P(Y = k \mid X = x) \propto \pi_k f_k(x) \propto e^{-d_k(x)/2}
```

where $`d_k(x)`$ is the **quadratic discriminant function**:

```math
\begin{split}
d_k(x) &= 2[-\log f_k(x) - \log \pi_k] - \text{Constant} \\
&= (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log|\Sigma_k| - 2\log \pi_k
\end{split}
```

### Components of the Discriminant Function

The function $`d_k(x)`$ consists of three terms:

1. **Mahalanobis Distance**: $`(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)`$ - measures the distance from $`x`$ to class center $`\mu_k`$ in the metric defined by $`\Sigma_k^{-1}`$

2. **Log Determinant**: $`\log|\Sigma_k|`$ - penalizes classes with larger covariance matrices (more spread out)

3. **Prior Term**: $`-2\log \pi_k`$ - incorporates class prior probabilities

### Decision Rule

The optimal classification rule is:

```math
\hat{y} = \arg\min_k d_k(x)
```

## 9.3.3. Parameter Estimation

### Maximum Likelihood Estimation

Given training data $`\{(x_i, y_i)\}_{i=1}^n`$, we estimate parameters using maximum likelihood:

#### Class Priors
```math
\hat{\pi}_k = \frac{n_k}{n}
```
where $`n_k = \sum_{i=1}^n \mathbb{I}(y_i = k)`$ is the number of samples in class $`k`$.

#### Class Means
```math
\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i
```

#### Class Covariances
```math
\hat{\Sigma}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T
```

### Numerical Stability

When $`\Sigma_k`$ is singular or near-singular (common in high dimensions), we use regularization:

```math
\hat{\Sigma}_k^{reg} = \hat{\Sigma}_k + \epsilon I_p
```

where $`\epsilon > 0`$ is a small constant (e.g., $`10^{-6}`$).

## 9.3.4. Implementation: QDA from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA

class QuadraticDiscriminantAnalysis:
    """Quadratic Discriminant Analysis implementation"""
    
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = None
        self.precision_matrices_ = None
        
    def fit(self, X, y):
        """Fit QDA model"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)
        
        # Initialize arrays
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = np.zeros((n_classes, n_features, n_features))
        self.precision_matrices_ = np.zeros((n_classes, n_features, n_features))
        
        # Estimate parameters for each class
        for i, k in enumerate(self.classes_):
            # Get samples from class k
            X_k = X[y == k]
            n_k = len(X_k)
            
            # Estimate prior
            self.priors_[i] = n_k / n_samples
            
            # Estimate mean
            self.means_[i] = np.mean(X_k, axis=0)
            
            # Estimate covariance with regularization
            cov_k = np.cov(X_k, rowvar=False)
            self.covariances_[i] = cov_k + self.regularization * np.eye(n_features)
            
            # Compute precision matrix (inverse covariance)
            try:
                self.precision_matrices_[i] = np.linalg.inv(self.covariances_[i])
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                self.precision_matrices_[i] = np.linalg.pinv(self.covariances_[i])
        
        return self
    
    def _compute_discriminant_function(self, X):
        """Compute quadratic discriminant function for all classes"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        discriminant_values = np.zeros((n_samples, n_classes))
        
        for i, k in enumerate(self.classes_):
            # Compute Mahalanobis distance
            diff = X - self.means_[i]
            mahalanobis_dist = np.sum(diff @ self.precision_matrices_[i] * diff, axis=1)
            
            # Compute log determinant
            log_det = np.log(np.linalg.det(self.covariances_[i]))
            
            # Compute prior term
            prior_term = -2 * np.log(self.priors_[i])
            
            # Combine all terms
            discriminant_values[:, i] = mahalanobis_dist + log_det + prior_term
        
        return discriminant_values
    
    def predict_proba(self, X):
        """Compute posterior probabilities"""
        discriminant_values = self._compute_discriminant_function(X)
        
        # Convert to probabilities using softmax
        # Subtract minimum for numerical stability
        discriminant_values -= np.min(discriminant_values, axis=1, keepdims=True)
        exp_values = np.exp(-0.5 * discriminant_values)
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """Predict class labels"""
        discriminant_values = self._compute_discriminant_function(X)
        return self.classes_[np.argmin(discriminant_values, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy score"""
        return accuracy_score(y, self.predict(X))
    
    def decision_function(self, X):
        """Compute decision function values"""
        return -0.5 * self._compute_discriminant_function(X)

# Create synthetic data for demonstration
def create_qda_demo_data(n_samples=1000, random_state=42):
    """Create synthetic data with different covariance structures"""
    np.random.seed(random_state)
    
    # Three classes with different means and covariances
    means = [
        np.array([0, 0]),
        np.array([3, 3]),
        np.array([-2, 2])
    ]
    
    covs = [
        np.array([[1, 0.5], [0.5, 1]]),      # Positive correlation
        np.array([[1, -0.5], [-0.5, 1]]),    # Negative correlation
        np.array([[0.5, 0], [0, 2]])         # Different variances
    ]
    
    X_list = []
    y_list = []
    
    for k, (mean, cov) in enumerate(zip(means, covs)):
        n_k = n_samples // 3
        X_k = np.random.multivariate_normal(mean, cov, n_k)
        X_list.append(X_k)
        y_list.append(np.full(n_k, k))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

# Generate data and fit QDA
X, y = create_qda_demo_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Fit our QDA implementation
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Make predictions
y_pred = qda.predict(X_test)
y_proba = qda.predict_proba(X_test)

print("QDA Results:")
print(f"Accuracy: {qda.score(X_test, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Compare with sklearn implementation
sklearn_qda = SklearnQDA()
sklearn_qda.fit(X_train, y_train)
sklearn_accuracy = sklearn_qda.score(X_test, y_test)

print(f"\nSklearn QDA Accuracy: {sklearn_accuracy:.3f}")
print(f"Our QDA Accuracy: {qda.score(X_test, y_test):.3f}")
```

```r
# R implementation
library(MASS)
library(ggplot2)
library(caret)

# Create synthetic data for QDA demonstration
create_qda_demo_data <- function(n_samples = 1000, random_state = 42) {
  set.seed(random_state)
  
  # Three classes with different means and covariances
  means <- list(
    c(0, 0),
    c(3, 3),
    c(-2, 2)
  )
  
  covs <- list(
    matrix(c(1, 0.5, 0.5, 1), nrow = 2),      # Positive correlation
    matrix(c(1, -0.5, -0.5, 1), nrow = 2),    # Negative correlation
    matrix(c(0.5, 0, 0, 2), nrow = 2)         # Different variances
  )
  
  X_list <- list()
  y_list <- list()
  
  for (k in 1:3) {
    n_k <- n_samples %/% 3
    X_k <- mvrnorm(n_k, mu = means[[k]], Sigma = covs[[k]])
    X_list[[k]] <- X_k
    y_list[[k]] <- rep(k - 1, n_k)
  }
  
  X <- do.call(rbind, X_list)
  y <- factor(unlist(y_list))
  
  return(list(X = X, y = y))
}

# Generate data
data <- create_qda_demo_data()
X <- data$X
y <- data$y

# Split data
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Fit QDA using MASS package
qda_model <- qda(X_train, y_train)

# Make predictions
qda_predictions <- predict(qda_model, X_test)
qda_pred_class <- qda_predictions$class
qda_pred_proba <- qda_predictions$posterior

# Evaluate performance
accuracy <- mean(qda_pred_class == y_test)
cat("QDA Accuracy:", accuracy, "\n")

# Confusion matrix
confusion_matrix <- table(Predicted = qda_pred_class, Actual = y_test)
print("Confusion Matrix:")
print(confusion_matrix)
```

## 9.3.5. Decision Boundaries and Visualization

### Understanding QDA Decision Boundaries

QDA produces quadratic decision boundaries because the discriminant function $`d_k(x)`$ is quadratic in $`x`$. For two classes, the decision boundary is where $`d_1(x) = d_2(x)`$.

```python
def plot_qda_decision_boundaries(X, y, qda_model, title="QDA Decision Boundaries"):
    """Plot QDA decision boundaries and data"""
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on mesh grid
    Z = qda_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(12, 5))
    
    # Decision boundaries
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k', cmap='viridis')
    plt.title(f'{title} - Decision Boundaries')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    
    # Posterior probabilities
    plt.subplot(1, 2, 2)
    Z_proba = qda_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)
    plt.contourf(xx, yy, Z_proba, alpha=0.4, cmap='RdBu_r')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k', cmap='viridis')
    plt.title(f'{title} - Posterior Probabilities')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Plot decision boundaries
plot_qda_decision_boundaries(X_test, y_test, qda, "QDA")

# Compare with LDA decision boundaries
def compare_qda_lda_boundaries(X_train, y_train, X_test, y_test):
    """Compare QDA and LDA decision boundaries"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Fit both models
    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()
    
    qda.fit(X_train, y_train)
    lda.fit(X_train, y_train)
    
    # Create mesh grid
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # QDA decision boundaries
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_qda = Z_qda.reshape(xx.shape)
    axes[0, 0].contourf(xx, yy, Z_qda, alpha=0.4, cmap='viridis')
    axes[0, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    axes[0, 0].set_title('QDA Decision Boundaries')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # LDA decision boundaries
    Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lda = Z_lda.reshape(xx.shape)
    axes[0, 1].contourf(xx, yy, Z_lda, alpha=0.4, cmap='viridis')
    axes[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    axes[0, 1].set_title('LDA Decision Boundaries')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # QDA posterior probabilities
    Z_qda_proba = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_qda_proba = Z_qda_proba.reshape(xx.shape)
    axes[1, 0].contourf(xx, yy, Z_qda_proba, alpha=0.4, cmap='RdBu_r')
    axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    axes[1, 0].set_title('QDA Posterior Probabilities')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    
    # LDA posterior probabilities
    Z_lda_proba = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_lda_proba = Z_lda_proba.reshape(xx.shape)
    axes[1, 1].contourf(xx, yy, Z_lda_proba, alpha=0.4, cmap='RdBu_r')
    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    axes[1, 1].set_title('LDA Posterior Probabilities')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracies
    print("Model Comparison:")
    print(f"QDA Accuracy: {qda.score(X_test, y_test):.3f}")
    print(f"LDA Accuracy: {lda.score(X_test, y_test):.3f}")

# Compare decision boundaries
compare_qda_lda_boundaries(X_train, y_train, X_test, y_test)
```

## 9.3.6. Model Analysis and Diagnostics

### Parameter Analysis

```python
def analyze_qda_parameters(qda_model, feature_names=None):
    """Analyze QDA model parameters"""
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(qda_model.means_.shape[1])]
    
    n_classes = len(qda_model.classes_)
    n_features = len(feature_names)
    
    # Create parameter summary
    print("QDA Model Parameters:")
    print("=" * 50)
    
    for i, k in enumerate(qda_model.classes_):
        print(f"\nClass {k}:")
        print(f"  Prior Probability: {qda_model.priors_[i]:.3f}")
        print(f"  Mean Vector: {qda_model.means_[i]}")
        print(f"  Covariance Matrix:")
        print(qda_model.covariances_[i])
        print(f"  Log Determinant: {np.log(np.linalg.det(qda_model.covariances_[i])):.3f}")
    
    # Visualize parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class priors
    axes[0, 0].bar(qda_model.classes_, qda_model.priors_)
    axes[0, 0].set_title('Class Prior Probabilities')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Prior Probability')
    
    # Class means
    for i, k in enumerate(qda_model.classes_):
        axes[0, 1].bar(np.arange(n_features) + i*0.25, qda_model.means_[i], 
                      width=0.25, label=f'Class {k}', alpha=0.8)
    axes[0, 1].set_title('Class Mean Vectors')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].legend()
    axes[0, 1].set_xticks(np.arange(n_features) + 0.25)
    axes[0, 1].set_xticklabels(feature_names)
    
    # Covariance matrices (heatmaps)
    for i, k in enumerate(qda_model.classes_):
        sns.heatmap(qda_model.covariances_[i], annot=True, fmt='.2f', 
                   ax=axes[1, i], cmap='viridis')
        axes[1, i].set_title(f'Covariance Matrix - Class {k}')
        axes[1, i].set_xticklabels(feature_names)
        axes[1, i].set_yticklabels(feature_names)
    
    plt.tight_layout()
    plt.show()

# Analyze QDA parameters
analyze_qda_parameters(qda)
```

### Mahalanobis Distance Analysis

```python
def analyze_mahalanobis_distances(X, y, qda_model):
    """Analyze Mahalanobis distances for each class"""
    n_classes = len(qda_model.classes_)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))
    if n_classes == 1:
        axes = [axes]
    
    for i, k in enumerate(qda_model.classes_):
        # Get samples from class k
        X_k = X[y == k]
        
        # Compute Mahalanobis distances
        diff = X_k - qda_model.means_[i]
        mahal_dist = np.sum(diff @ qda_model.precision_matrices_[i] * diff, axis=1)
        
        # Plot histogram
        axes[i].hist(mahal_dist, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Mahalanobis Distances - Class {k}')
        axes[i].set_xlabel('Mahalanobis Distance')
        axes[i].set_ylabel('Frequency')
        
        # Add theoretical chi-squared distribution
        df = X.shape[1]  # degrees of freedom = number of features
        x_chi2 = np.linspace(0, np.max(mahal_dist), 100)
        y_chi2 = len(mahal_dist) * (x_chi2[1] - x_chi2[0]) * chi2.pdf(x_chi2, df)
        axes[i].plot(x_chi2, y_chi2, 'r-', linewidth=2, label=f'χ²({df})')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Analyze Mahalanobis distances
from scipy.stats import chi2
analyze_mahalanobis_distances(X_test, y_test, qda)
```

## 9.3.7. High-Dimensional QDA

### Challenges in High Dimensions

When the number of features $`p`$ is large relative to the sample size, QDA faces several challenges:

1. **Curse of Dimensionality**: Need $`O(p^2)`$ parameters per class
2. **Singular Covariance**: Covariance matrices become singular
3. **Overfitting**: Model complexity increases with $`p^2``

### Regularization Techniques

```python
class RegularizedQDA:
    """Regularized QDA for high-dimensional data"""
    
    def __init__(self, regularization='diagonal', alpha=0.1):
        self.regularization = regularization
        self.alpha = alpha
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = None
        
    def fit(self, X, y):
        """Fit regularized QDA"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)
        
        # Initialize arrays
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = np.zeros((n_classes, n_features, n_features))
        
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            n_k = len(X_k)
            
            # Estimate parameters
            self.priors_[i] = n_k / n_samples
            self.means_[i] = np.mean(X_k, axis=0)
            
            # Regularize covariance matrix
            cov_k = np.cov(X_k, rowvar=False)
            
            if self.regularization == 'diagonal':
                # Diagonal regularization
                self.covariances_[i] = np.diag(np.diag(cov_k)) + self.alpha * np.eye(n_features)
            elif self.regularization == 'shrinkage':
                # Shrinkage regularization
                target = np.trace(cov_k) / n_features * np.eye(n_features)
                self.covariances_[i] = (1 - self.alpha) * cov_k + self.alpha * target
            else:
                # Ridge regularization
                self.covariances_[i] = cov_k + self.alpha * np.eye(n_features)
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        discriminant_values = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, k in enumerate(self.classes_):
            diff = X - self.means_[i]
            inv_cov = np.linalg.inv(self.covariances_[i])
            mahal_dist = np.sum(diff @ inv_cov * diff, axis=1)
            log_det = np.log(np.linalg.det(self.covariances_[i]))
            prior_term = -2 * np.log(self.priors_[i])
            
            discriminant_values[:, i] = mahal_dist + log_det + prior_term
        
        return self.classes_[np.argmin(discriminant_values, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy score"""
        return accuracy_score(y, self.predict(X))

# Test regularized QDA on high-dimensional data
def test_high_dimensional_qda():
    """Test QDA performance in high dimensions"""
    np.random.seed(42)
    
    # Generate high-dimensional data
    n_samples = 200
    n_features = 50
    n_classes = 3
    
    # Create sparse covariance matrices
    means = [np.random.randn(n_features) for _ in range(n_classes)]
    covs = []
    
    for k in range(n_classes):
        # Create sparse precision matrix
        precision = np.eye(n_features)
        for i in range(n_features):
            for j in range(i+1, min(i+3, n_features)):
                if np.random.random() < 0.3:
                    precision[i, j] = precision[j, i] = 0.5
        covs.append(np.linalg.inv(precision))
    
    # Generate data
    X_list = []
    y_list = []
    for k in range(n_classes):
        n_k = n_samples // n_classes
        X_k = np.random.multivariate_normal(means[k], covs[k], n_k)
        X_list.append(X_k)
        y_list.append(np.full(n_k, k))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test different regularization methods
    methods = {
        'Standard QDA': QuadraticDiscriminantAnalysis(),
        'Diagonal QDA': RegularizedQDA(regularization='diagonal', alpha=0.1),
        'Shrinkage QDA': RegularizedQDA(regularization='shrinkage', alpha=0.1),
        'Ridge QDA': RegularizedQDA(regularization='ridge', alpha=0.1)
    }
    
    results = {}
    for name, method in methods.items():
        try:
            method.fit(X_train, y_train)
            accuracy = method.score(X_test, y_test)
            results[name] = accuracy
            print(f"{name}: {accuracy:.3f}")
        except Exception as e:
            print(f"{name}: Failed - {e}")
            results[name] = 0
    
    return results

# Test high-dimensional QDA
high_dim_results = test_high_dimensional_qda()
```

## 9.3.8. Model Selection and Validation

### Cross-Validation for QDA

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

def qda_cross_validation(X, y, cv=5):
    """Perform cross-validation for QDA"""
    # Standard QDA
    qda = QuadraticDiscriminantAnalysis()
    cv_scores = cross_val_score(qda, X, y, cv=cv)
    
    print("QDA Cross-Validation Results:")
    print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"Individual CV Scores: {cv_scores}")
    
    return cv_scores

# Perform cross-validation
cv_scores = qda_cross_validation(X, y, cv=5)

# Grid search for regularization parameter
def qda_grid_search(X, y):
    """Grid search for optimal regularization parameter"""
    param_grid = {
        'regularization': [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    }
    
    qda = QuadraticDiscriminantAnalysis()
    grid_search = GridSearchCV(qda, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    print("Grid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    return grid_search

# Perform grid search
grid_search = qda_grid_search(X, y)
```

## 9.3.9. Real-World Applications

### Example: Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def qda_iris_example():
    """QDA on the classic Iris dataset"""
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Use only two classes for binary classification
    mask = y != 2
    X_binary = X[mask]
    y_binary = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = qda.score(X_test_scaled, y_test)
    print(f"Iris Dataset QDA Accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    y_pred = qda.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('QDA Confusion Matrix - Iris Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return qda, accuracy

# Run Iris example
iris_qda, iris_accuracy = qda_iris_example()
```

### Example: Credit Risk Assessment

```python
def qda_credit_risk_example():
    """QDA for credit risk assessment"""
    # Create synthetic credit data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: income, credit_score, debt_ratio, employment_years
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    debt_ratio = np.random.beta(2, 5, n_samples) * 2
    employment_years = np.random.exponential(5, n_samples)
    
    X = np.column_stack([income, credit_score, debt_ratio, employment_years])
    
    # Generate target based on features
    risk_score = (0.3 * (income - 50000) / 20000 + 
                  0.4 * (credit_score - 700) / 100 + 
                  0.2 * (debt_ratio - 1) + 
                  0.1 * (employment_years - 5) / 5)
    
    risk_score += np.random.normal(0, 0.2, n_samples)
    y = (risk_score > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = qda.score(X_test_scaled, y_test)
    y_pred = qda.predict(X_test_scaled)
    y_proba = qda.predict_proba(X_test_scaled)
    
    print("Credit Risk Assessment Results:")
    print(f"QDA Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    # Feature importance analysis
    feature_names = ['Income', 'Credit Score', 'Debt Ratio', 'Employment Years']
    
    plt.figure(figsize=(12, 4))
    
    # Compare means between classes
    plt.subplot(1, 2, 1)
    for i, class_label in enumerate([0, 1]):
        class_mask = y_train == class_label
        class_means = np.mean(X_train_scaled[class_mask], axis=0)
        plt.bar(np.arange(len(feature_names)) + i*0.35, class_means, 
               width=0.35, label=f'Class {class_label}', alpha=0.8)
    
    plt.title('Feature Means by Class')
    plt.xlabel('Features')
    plt.ylabel('Standardized Mean')
    plt.xticks(np.arange(len(feature_names)) + 0.175, feature_names, rotation=45)
    plt.legend()
    
    # Covariance comparison
    plt.subplot(1, 2, 2)
    for i, class_label in enumerate([0, 1]):
        cov_diag = np.diag(qda.covariances_[i])
        plt.bar(np.arange(len(feature_names)) + i*0.35, cov_diag, 
               width=0.35, label=f'Class {class_label}', alpha=0.8)
    
    plt.title('Feature Variances by Class')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.xticks(np.arange(len(feature_names)) + 0.175, feature_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return qda, accuracy

# Run credit risk example
credit_qda, credit_accuracy = qda_credit_risk_example()
```

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of Quadratic Discriminant Analysis. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and handling of common challenges in high-dimensional settings.
