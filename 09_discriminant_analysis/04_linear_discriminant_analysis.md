# 9.4. Linear Discriminant Analysis

## 9.4.0. Introduction and Motivation

Linear Discriminant Analysis (LDA) is a fundamental classification method that extends the principles of discriminant analysis by making a key simplifying assumption: **all classes share the same covariance matrix**. This assumption transforms the quadratic decision boundaries of QDA into linear ones, making LDA both computationally efficient and interpretable.

### Key Advantages of LDA:
1. **Computational Efficiency**: Linear decision boundaries are faster to compute
2. **Dimensionality Reduction**: Natural ability to reduce features to (K-1) dimensions
3. **Robustness**: Less prone to overfitting in high-dimensional settings
4. **Interpretability**: Linear coefficients provide clear feature importance

### When to Use LDA:
- When classes have similar covariance structures
- When you need dimensionality reduction
- When interpretability is important
- When computational efficiency matters

## 9.4.1. Mathematical Foundation

### From QDA to LDA: The Key Assumption

In our previous discussion on Quadratic Discriminant Analysis (QDA), the discriminant function plays a pivotal role in making classification decisions. The QDA discriminant function is:

```math
d_k(x) = (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k) + \log |\Sigma_k| - 2 \log \pi_k
```

**Key Insight**: If we make the assumption that all groups share the same covariance matrix ($`\Sigma_k = \Sigma`$ for all k), the discriminant function simplifies dramatically:

```math
d_k(x) = (x-\mu_k)^T \Sigma^{-1} (x-\mu_k) + \log |\Sigma| - 2 \log \pi_k
```

### Understanding the Linear Transformation

The first term $(x-\mu_k)^T \Sigma^{-1} (x-\mu_k)$ is the **Mahalanobis distance** between point $`x`$ and class center $`\mu_k`$. Let's expand this term to see why it becomes linear:

```math
\begin{split}
(x-\mu_k)^T \Sigma^{-1} (x-\mu_k) &= x^T \Sigma^{-1} x - 2x^T \Sigma^{-1} \mu_k + \mu_k^T \Sigma^{-1} \mu_k \\
&= \textcolor{gray}{x^T \Sigma^{-1} x} - 2x^T \Sigma^{-1} \mu_k + \mu_k^T \Sigma^{-1} \mu_k
\end{split}
```

**Critical Observation**: The term $`x^T \Sigma^{-1} x`$ (highlighted in gray) is **common to all classes** and doesn't affect the classification decision. When comparing discriminant functions across classes, this term cancels out.

### The Linear Discriminant Function

After removing the common quadratic term, the discriminant function becomes **linear in x**:

```math
d_k(x) = -2x^T \Sigma^{-1} \mu_k + \mu_k^T \Sigma^{-1} \mu_k + \log |\Sigma| - 2 \log \pi_k
```

This can be rewritten as:

```math
d_k(x) = w_k^T x + b_k
```

Where:
- $`w_k = -2\Sigma^{-1}\mu_k`$ (linear coefficients)
- $`b_k = \mu_k^T \Sigma^{-1} \mu_k + \log |\Sigma| - 2 \log \pi_k`$ (bias term)

### Decision Boundary

For binary classification (K=2), the decision boundary occurs when $`d_1(x) = d_2(x)`$:

```math
\begin{split}
w_1^T x + b_1 &= w_2^T x + b_2 \\
(w_1 - w_2)^T x + (b_1 - b_2) &= 0 \\
w^T x + b &= 0
\end{split}
```

This is a **linear decision boundary** in the feature space.

## 9.4.2. Parameter Estimation

### Maximum Likelihood Estimation

The parameters of LDA are estimated using maximum likelihood:

#### 1. Class Priors ($`\pi_k`$)
```math
\hat{\pi}_k = \frac{n_k}{n}
```
Where $`n_k`$ is the number of samples in class k, and $`n`$ is the total number of samples.

#### 2. Class Means ($`\mu_k`$)
```math
\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i
```

#### 3. Shared Covariance Matrix ($`\Sigma`$)
The **pooled sample covariance** combines information from all classes:

```math
\hat{\Sigma} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T
```

**Intuition**: This is a weighted average of the within-class covariance matrices, where each class contributes proportionally to its sample size.

### Numerical Stability: Handling Singular Covariance

When $`p > n-K`$ (high-dimensional data), $`\hat{\Sigma}`$ may be singular. Several solutions exist:

#### 1. Regularization (Ridge-like)
```math
\hat{\Sigma}_{\text{reg}} = \hat{\Sigma} + \epsilon I
```
Where $`\epsilon`$ is a small positive constant.

#### 2. Generalized Inverse (SVD-based)
```math
\hat{\Sigma} = U \begin{pmatrix} D & 0 \\ 0 & 0 \end{pmatrix} U^T
```

```math
\hat{\Sigma}^{-1} = U \begin{pmatrix} D^{-1} & 0 \\ 0 & 0 \end{pmatrix} U^T
```

Where $`D`$ contains the non-zero eigenvalues.

## 9.4.3. Dimensionality Reduction: Reduced Rank LDA

### The Natural Dimensionality Reduction

LDA provides a natural way to reduce dimensionality from $`p`$ to $`K-1`$ dimensions. This is one of its most powerful features.

### Geometric Intuition

Let's start with the simplified case where $`\Sigma = I`$ (identity matrix):

```math
d_k(x) = \|x - \mu_k\|^2 - 2 \log \pi_k
```

**Key Insight**: The K class centers $`\{\mu_1, \mu_2, \ldots, \mu_K\}`$ span at most a $(K-1)$-dimensional subspace.

### Mathematical Derivation

Without loss of generality, assume the mean of all class centers is at the origin:
```math
\frac{1}{K} \sum_{k=1}^K \mu_k = 0
```

For any point $`x`$, we can decompose it as:
```math
x = x_1 + x_2
```

Where:
- $`x_1`$ lies in the $(K-1)$-dimensional subspace spanned by the class centers
- $`x_2`$ lies in the orthogonal complement (dimension $`p-K+1`$)

The squared distance becomes:
```math
\|x - \mu_k\|^2 = \|x_1 + x_2 - \mu_k\|^2 = \|x_1 - \mu_k\|^2 + \|x_2\|^2
```

**Critical Observation**: $`\|x_2\|^2`$ is constant across all classes and doesn't affect classification decisions.

### The LDA Projection

The optimal projection direction is given by the eigenvectors of $`\Sigma^{-1}\Sigma_B`$, where:

```math
\Sigma_B = \sum_{k=1}^K \pi_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^T
```

is the **between-class scatter matrix**, and $`\bar{\mu} = \sum_{k=1}^K \pi_k \mu_k`$ is the overall mean.

### Binary Classification Example

For K=2 (binary classification), LDA reduces to a single dimension:

**Original 2D Space**: Data points in $`\mathbb{R}^2`$
**LDA Projection**: All points projected onto a single line
**Decision**: Classify based on position along this line

This is equivalent to finding the optimal linear separator in the original space.

## 9.4.4. Practical Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

class LinearDiscriminantAnalysisFromScratch:
    """
    Linear Discriminant Analysis implementation from scratch
    """
    
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        """
        Fit LDA model
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get unique classes and their counts
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = np.sum(y == c) / n_samples
            
        # Calculate class means
        self.means_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            self.means_[i] = np.mean(X[y == c], axis=0)
            
        # Calculate pooled covariance matrix
        self.covariance_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_samples = X[y == c]
            class_mean = self.means_[i]
            diff = class_samples - class_mean
            self.covariance_ += diff.T @ diff
            
        self.covariance_ /= (n_samples - n_classes)
        
        # Add regularization for numerical stability
        self.covariance_ += self.regularization * np.eye(n_features)
        
        # Calculate coefficients and intercepts
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        
        cov_inv = np.linalg.inv(self.covariance_)
        for i in range(n_classes):
            self.coef_[i] = -2 * cov_inv @ self.means_[i]
            self.intercept_[i] = (self.means_[i] @ cov_inv @ self.means_[i] + 
                                 np.log(np.linalg.det(self.covariance_)) - 
                                 2 * np.log(self.priors_[i]))
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        discriminant_scores = self.decision_function(X)
        return self.classes_[np.argmax(discriminant_scores, axis=1)]
    
    def decision_function(self, X):
        """
        Compute discriminant scores
        """
        X = np.asarray(X)
        return X @ self.coef_.T + self.intercept_
    
    def transform(self, X, n_components=None):
        """
        Transform data using LDA projection
        """
        if n_components is None:
            n_components = len(self.classes_) - 1
            
        # Calculate between-class scatter matrix
        overall_mean = np.average(self.means_, weights=self.priors_, axis=0)
        between_scatter = np.zeros((X.shape[1], X.shape[1]))
        
        for i, c in enumerate(self.classes_):
            diff = self.means_[i] - overall_mean
            between_scatter += self.priors_[i] * np.outer(diff, diff)
            
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eigh(
            np.linalg.inv(self.covariance_) @ between_scatter
        )
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top components
        projection_matrix = eigenvecs[:, :n_components]
        
        return X @ projection_matrix

# Generate synthetic data
def generate_lda_data(n_samples=1000, n_features=2, n_classes=3, random_state=42):
    """
    Generate synthetic data suitable for LDA
    """
    np.random.seed(random_state)
    
    # Generate class means
    means = np.random.randn(n_classes, n_features) * 2
    
    # Generate shared covariance matrix
    A = np.random.randn(n_features, n_features)
    covariance = A @ A.T + np.eye(n_features)
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        class_samples = np.random.multivariate_normal(
            means[i], covariance, samples_per_class
        )
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    return np.vstack(X), np.array(y)

# Example usage and comparison
def demonstrate_lda():
    """
    Demonstrate LDA with synthetic data
    """
    # Generate data
    X, y = generate_lda_data(n_samples=900, n_features=2, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit our implementation
    lda_scratch = LinearDiscriminantAnalysisFromScratch()
    lda_scratch.fit(X_train, y_train)
    
    # Fit sklearn implementation
    lda_sklearn = LinearDiscriminantAnalysis()
    lda_sklearn.fit(X_train, y_train)
    
    # Compare predictions
    y_pred_scratch = lda_scratch.predict(X_test)
    y_pred_sklearn = lda_sklearn.predict(X_test)
    
    print("Accuracy Comparison:")
    print(f"Our Implementation: {accuracy_score(y_test, y_pred_scratch):.4f}")
    print(f"Sklearn Implementation: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[0].set_title('Original Data')
    axes[0].legend()
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = lda_scratch.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    for i in range(3):
        mask = y == i
        axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[1].set_title('Decision Boundaries')
    axes[1].legend()
    
    # LDA projection
    X_transformed = lda_scratch.transform(X)
    for i in range(3):
        mask = y == i
        axes[2].scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       alpha=0.6, label=f'Class {i}')
    axes[2].set_title('LDA Projection (2D → 2D)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return lda_scratch, lda_sklearn

# Run demonstration
if __name__ == "__main__":
    lda_scratch, lda_sklearn = demonstrate_lda()
```

### R Implementation

```r
# Linear Discriminant Analysis in R
library(MASS)
library(ggplot2)
library(gridExtra)

# Generate synthetic data for LDA
generate_lda_data <- function(n_samples = 1000, n_features = 2, n_classes = 3, seed = 42) {
  set.seed(seed)
  
  # Generate class means
  means <- matrix(rnorm(n_classes * n_features, 0, 2), nrow = n_classes)
  
  # Generate shared covariance matrix
  A <- matrix(rnorm(n_features^2), nrow = n_features)
  covariance <- A %*% t(A) + diag(n_features)
  
  # Generate samples
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  y <- rep(0, n_samples)
  samples_per_class <- n_samples %/% n_classes
  
  for (i in 1:n_classes) {
    start_idx <- (i-1) * samples_per_class + 1
    end_idx <- i * samples_per_class
    class_samples <- MASS::mvrnorm(samples_per_class, means[i,], covariance)
    X[start_idx:end_idx,] <- class_samples
    y[start_idx:end_idx] <- i
  }
  
  return(list(X = X, y = y))
}

# Custom LDA implementation
lda_from_scratch <- function(X, y, regularization = 1e-6) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Calculate class priors
  priors <- sapply(classes, function(c) sum(y == c) / n_samples)
  
  # Calculate class means
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  for (i in 1:n_classes) {
    means[i,] <- colMeans(X[y == classes[i],, drop = FALSE])
  }
  
  # Calculate pooled covariance matrix
  covariance <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_samples <- X[y == classes[i],, drop = FALSE]
    class_mean <- means[i,]
    diff <- sweep(class_samples, 2, class_mean, "-")
    covariance <- covariance + t(diff) %*% diff
  }
  covariance <- covariance / (n_samples - n_classes)
  
  # Add regularization
  covariance <- covariance + regularization * diag(n_features)
  
  # Calculate coefficients and intercepts
  cov_inv <- solve(covariance)
  coef <- matrix(0, nrow = n_classes, ncol = n_features)
  intercept <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    coef[i,] <- -2 * cov_inv %*% means[i,]
    intercept[i] <- t(means[i,]) %*% cov_inv %*% means[i,] + 
                   log(det(covariance)) - 2 * log(priors[i])
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    covariance = covariance,
    coef = coef,
    intercept = intercept
  ))
}

# Prediction function
predict_lda <- function(model, X) {
  discriminant_scores <- X %*% t(model$coef) + matrix(model$intercept, 
                                                     nrow = nrow(X), 
                                                     ncol = length(model$intercept), 
                                                     byrow = TRUE)
  predictions <- apply(discriminant_scores, 1, which.max)
  return(model$classes[predictions])
}

# Demonstrate LDA
demonstrate_lda_r <- function() {
  # Generate data
  data <- generate_lda_data(n_samples = 900, n_features = 2, n_classes = 3)
  X <- data$X
  y <- data$y
  
  # Fit our implementation
  lda_model <- lda_from_scratch(X, y)
  
  # Fit MASS implementation
  lda_mass <- lda(X, y)
  
  # Make predictions
  y_pred_scratch <- predict_lda(lda_model, X)
  y_pred_mass <- predict(lda_mass, X)$class
  
  # Calculate accuracy
  accuracy_scratch <- mean(y_pred_scratch == y)
  accuracy_mass <- mean(y_pred_mass == y)
  
  cat("Accuracy Comparison:\n")
  cat("Our Implementation:", round(accuracy_scratch, 4), "\n")
  cat("MASS Implementation:", round(accuracy_mass, 4), "\n")
  
  # Create visualizations
  df <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y),
    pred_scratch = factor(y_pred_scratch),
    pred_mass = factor(y_pred_mass)
  )
  
  # Original data
  p1 <- ggplot(df, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = "Original Data", color = "True Class") +
    theme_minimal()
  
  # Predictions from our implementation
  p2 <- ggplot(df, aes(x = x1, y = x2, color = pred_scratch)) +
    geom_point(alpha = 0.6) +
    labs(title = "Our LDA Predictions", color = "Predicted Class") +
    theme_minimal()
  
  # Predictions from MASS
  p3 <- ggplot(df, aes(x = x1, y = x2, color = pred_mass)) +
    geom_point(alpha = 0.6) +
    labs(title = "MASS LDA Predictions", color = "Predicted Class") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(lda_model = lda_model, lda_mass = lda_mass))
}

# Run demonstration
results <- demonstrate_lda_r()
```

## 9.4.5. Advanced Topics

### 9.4.5.1. Regularized LDA

For high-dimensional data, we can add regularization to the covariance estimation:

```python
def regularized_lda(X, y, alpha=0.1):
    """
    Regularized LDA with shrinkage parameter alpha
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Calculate pooled covariance
    covariance = calculate_pooled_covariance(X, y)
    
    # Regularization: convex combination with identity matrix
    identity = np.eye(n_features)
    regularized_cov = (1 - alpha) * covariance + alpha * identity
    
    return regularized_cov
```

### 9.4.5.2. Kernel LDA

For non-linear decision boundaries, we can apply the kernel trick:

```python
def kernel_lda(X, y, kernel='rbf', gamma=1.0):
    """
    Kernel LDA implementation
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Compute kernel matrix
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    
    # Apply LDA in kernel space
    # (Implementation details omitted for brevity)
    pass
```

### 9.4.5.3. Multi-class LDA

For K > 2 classes, LDA finds K-1 discriminant directions:

```python
def multiclass_lda(X, y):
    """
    Multi-class LDA with dimensionality reduction
    """
    n_classes = len(np.unique(y))
    n_components = min(n_classes - 1, X.shape[1])
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(X, y)
    
    return X_transformed, lda
```

## 9.4.6. Model Evaluation and Diagnostics

### Performance Metrics

```python
def evaluate_lda_model(X_train, X_test, y_train, y_test):
    """
    Comprehensive LDA model evaluation
    """
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                               confusion_matrix, roc_auc_score)
    
    # Fit model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Predictions
    y_pred = lda.predict(X_test)
    y_pred_proba = lda.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
```

### Model Diagnostics

```python
def lda_diagnostics(X, y, lda_model):
    """
    Diagnostic plots for LDA
    """
    # 1. Check normality assumption
    from scipy import stats
    
    residuals = []
    for i, class_label in enumerate(lda_model.classes_):
        class_mask = y == class_label
        class_residuals = X[class_mask] - lda_model.means_[i]
        residuals.extend(class_residuals.flatten())
    
    # Q-Q plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot for Normality Check")
    
    # 2. Check homoscedasticity
    plt.subplot(1, 3, 2)
    X_transformed = lda_model.transform(X)
    plt.scatter(X_transformed[:, 0], residuals[:len(X_transformed)], alpha=0.5)
    plt.xlabel("First LDA Component")
    plt.ylabel("Residuals")
    plt.title("Homoscedasticity Check")
    
    # 3. Feature importance
    plt.subplot(1, 3, 3)
    feature_importance = np.abs(lda_model.coef_[0])
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel("Feature Index")
    plt.ylabel("|Coefficient|")
    plt.title("Feature Importance")
    
    plt.tight_layout()
    plt.show()
```

## 9.4.7. Real-World Applications

### Example 1: Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# LDA with cross-validation
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=5)

print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Dimensionality reduction
X_transformed = lda.fit_transform(X, y)
print(f"Original dimensions: {X.shape[1]}")
print(f"LDA dimensions: {X_transformed.shape[1]}")
```

### Example 2: Credit Risk Classification

```python
def credit_risk_lda():
    """
    LDA for credit risk assessment
    """
    # Simulate credit data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: income, debt, credit_score, age
    income = np.random.lognormal(10, 0.5, n_samples)
    debt = np.random.lognormal(8, 0.3, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    age = np.random.normal(35, 10, n_samples)
    
    X = np.column_stack([income, debt, credit_score, age])
    
    # Risk classification (0: low risk, 1: high risk)
    risk_score = (income * 0.3 + debt * (-0.4) + credit_score * 0.2 + age * 0.1 + 
                  np.random.normal(0, 0.1, n_samples))
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Credit Risk Classification Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Income', 'Debt', 'Credit Score', 'Age']
    importance = np.abs(lda.coef_[0])
    
    plt.figure(figsize=(10, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importance in Credit Risk LDA")
    plt.ylabel("|Coefficient|")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return lda, accuracy
```

## 9.4.8. Risk of Overfitting

### Understanding the Overfitting Problem

When $`p \gg K`$ (high-dimensional data with few classes), LDA can overfit because:

1. **Limited Degrees of Freedom**: The pooled covariance matrix has limited degrees of freedom
2. **Curse of Dimensionality**: In high dimensions, the "empty space" phenomenon makes distance measures less reliable
3. **Sample Size Requirements**: Need sufficient samples per class for reliable covariance estimation

### Mitigation Strategies

#### 1. Regularization
```python
def regularized_lda_cv(X, y, alphas=np.logspace(-4, 1, 20)):
    """
    Cross-validated regularized LDA
    """
    from sklearn.model_selection import GridSearchCV
    
    # Create custom LDA with regularization
    class RegularizedLDA:
        def __init__(self, alpha=0.1):
            self.alpha = alpha
            
        def fit(self, X, y):
            # Implementation with regularization
            pass
    
    # Grid search
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(RegularizedLDA(), param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_
```

#### 2. Feature Selection
```python
def lda_with_feature_selection(X, y, n_features=10):
    """
    LDA with feature selection
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Select top features
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_selected, y)
    
    return lda, selector
```

#### 3. Cross-Validation
```python
def robust_lda_evaluation(X, y, n_splits=5):
    """
    Robust LDA evaluation with cross-validation
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        score = lda.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## 9.4.9. Summary and Best Practices

### Key Takeaways

1. **LDA Assumptions**: 
   - Classes follow multivariate normal distributions
   - All classes share the same covariance matrix
   - Features are independent given the class

2. **Advantages**:
   - Computationally efficient
   - Natural dimensionality reduction
   - Interpretable coefficients
   - Works well with limited data

3. **Limitations**:
   - Assumes linear decision boundaries
   - Sensitive to violations of normality
   - Can overfit in high dimensions

### Best Practices

1. **Data Preprocessing**:
   - Standardize features (mean=0, std=1)
   - Check for multicollinearity
   - Handle missing values appropriately

2. **Model Validation**:
   - Use cross-validation for small datasets
   - Check normality assumptions
   - Monitor for overfitting

3. **Hyperparameter Tuning**:
   - Regularization parameter for high-dimensional data
   - Number of components for dimensionality reduction

4. **Interpretation**:
   - Examine feature coefficients
   - Visualize decision boundaries
   - Analyze class separation in reduced dimensions

### When to Use LDA

**Use LDA when**:
- Classes have similar covariance structures
- You need dimensionality reduction
- Interpretability is important
- You have limited training data
- Linear decision boundaries are appropriate

**Consider alternatives when**:
- Classes have very different covariance structures (use QDA)
- Non-linear decision boundaries are needed (use SVM, Random Forest)
- High-dimensional data with complex patterns (use deep learning)

LDA remains a fundamental and powerful classification method that provides an excellent balance between simplicity, interpretability, and performance for many real-world problems.
