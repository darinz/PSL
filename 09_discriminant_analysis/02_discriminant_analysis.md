# 9.2. Discriminant Analysis

## 9.2.1. Introduction to Discriminant Analysis

Discriminant analysis is a family of classification methods that model the distribution of features within each class and use Bayes' theorem to make predictions. Unlike discriminative methods that directly model $`P(Y|X)`$, discriminant analysis is a **generative approach** that models the joint distribution $`P(X, Y)`$ by decomposing it into class-conditional densities and class priors.

### Generative vs. Discriminative Approaches

| Approach | Models | Example Methods |
|----------|--------|-----------------|
| **Generative** | $`P(X, Y) = P(Y) \cdot P(X \mid Y)`$ | LDA, QDA, Naive Bayes |
| **Discriminative** | $`P(Y \mid X)`$ directly | Logistic Regression, SVM |

### Mathematical Foundation

The key insight of discriminant analysis is to decompose the joint distribution:

```math
p(x, y) = p(y) \cdot p(x \mid y)
```

where:
- $`p(y)`$ is the **class prior** (marginal distribution of classes)
- $`p(x \mid y)`$ is the **class-conditional density** (distribution of features given class)

This decomposition allows us to:
1. Estimate class priors from class frequencies in the data
2. Model class-conditional densities using parametric or non-parametric methods
3. Apply Bayes' theorem to compute posterior probabilities

### Types of Discriminant Analysis

We will explore three main approaches:

1. **Quadratic Discriminant Analysis (QDA)**: Assumes different covariance matrices for each class
2. **Linear Discriminant Analysis (LDA)**: Assumes shared covariance matrix across classes
3. **Naive Bayes**: Assumes conditional independence of features given class

## 9.2.2. Bayes' Theorem and Optimal Classification

### Derivation of Bayes' Theorem

The optimal classifier maximizes the posterior probability $`P(Y=k \mid X=x)`$. Using Bayes' theorem:

```math
P(Y = k \mid X=x) = \frac{P(X=x, Y=k)}{P(X=x)} = \frac{P(X=x \mid Y=k) \cdot P(Y=k)}{P(X=x)}
```

Let's define:
- $`f_k(x) = p(x \mid Y=k)`$: class-conditional density function
- $`\pi_k = P(Y=k)`$: class prior probability

Then:

```math
P(Y = k \mid X=x) = \frac{\pi_k f_k(x)}{P(X=x)} \propto \pi_k f_k(x)
```

Since $`P(X=x)`$ is constant across all classes, the optimal classifier is:

```math
\hat{y} = \arg\max_k P(Y=k \mid X=x) = \arg\max_k \pi_k f_k(x)
```

### Log-Likelihood Formulation

For numerical stability and computational efficiency, we often work with log-likelihoods:

```math
\hat{y} = \arg\max_k \log(\pi_k f_k(x)) = \arg\max_k [\log \pi_k + \log f_k(x)]
```

Or equivalently, minimizing the negative log-likelihood:

```math
\hat{y} = \arg\min_k [-\log \pi_k - \log f_k(x)]
```

### Implementation: Bayes Classifier Framework

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import multivariate_normal
import pandas as pd

class BayesClassifier:
    """Base class for Bayes classifiers"""
    
    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.conditional_densities_ = None
    
    def fit(self, X, y):
        """Fit the Bayes classifier"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        
        # Estimate class priors
        self.priors_ = np.zeros(n_classes)
        for i, k in enumerate(self.classes_):
            self.priors_[i] = np.sum(y == k) / n_samples
        
        # Estimate class-conditional densities
        self._fit_conditional_densities(X, y)
        
        return self
    
    def _fit_conditional_densities(self, X, y):
        """Estimate class-conditional densities (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def predict_proba(self, X):
        """Compute posterior probabilities"""
        if self.conditional_densities_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        # Compute log-likelihoods for each class
        for i, k in enumerate(self.classes_):
            log_probs[:, i] = (np.log(self.priors_[i]) + 
                              self.conditional_densities_[i].logpdf(X))
        
        # Convert to probabilities (softmax)
        # Subtract max for numerical stability
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        """Compute accuracy score"""
        return accuracy_score(y, self.predict(X))

# Example usage
def create_gaussian_mixture_data(n_samples=1000, random_state=42):
    """Create synthetic data from Gaussian mixture"""
    np.random.seed(random_state)
    
    # Three classes with different means and covariances
    means = [
        np.array([0, 0]),
        np.array([3, 3]),
        np.array([-2, 2])
    ]
    
    covs = [
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[1, -0.5], [-0.5, 1]]),
        np.array([[0.5, 0], [0, 0.5]])
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

# Create dataset
X, y = create_gaussian_mixture_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Dataset shape:", X.shape)
print("Class distribution:")
print(pd.Series(y).value_counts(normalize=True))
```

```r
# R implementation
library(MASS)
library(ggplot2)
library(caret)

# Create synthetic Gaussian mixture data
create_gaussian_mixture_data <- function(n_samples = 1000, random_state = 42) {
  set.seed(random_state)
  
  # Three classes with different means and covariances
  means <- list(
    c(0, 0),
    c(3, 3),
    c(-2, 2)
  )
  
  covs <- list(
    matrix(c(1, 0.5, 0.5, 1), nrow = 2),
    matrix(c(1, -0.5, -0.5, 1), nrow = 2),
    matrix(c(0.5, 0, 0, 0.5), nrow = 2)
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

# Create dataset
data <- create_gaussian_mixture_data()
X <- data$X
y <- data$y

# Split data
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

cat("Dataset shape:", nrow(X), "x", ncol(X), "\n")
cat("Class distribution:\n")
print(table(y) / length(y))
```

## 9.2.3. Quadratic Discriminant Analysis (QDA)

### Mathematical Formulation

QDA assumes that each class follows a multivariate Gaussian distribution with its own mean and covariance matrix:

```math
f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)
```

The decision function becomes:

```math
\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k
```

### Parameter Estimation

For each class $`k`$:

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$ where $`n_k`$ is the number of samples in class $`k`$
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$
3. **Class covariance**: $`\hat{\Sigma}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$

### Implementation: QDA

```python
class QuadraticDiscriminantAnalysis(BayesClassifier):
    """Quadratic Discriminant Analysis"""
    
    def _fit_conditional_densities(self, X, y):
        """Fit Gaussian densities for each class"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = np.zeros((n_classes, n_features, n_features))
        self.conditional_densities_ = []
        
        for i, k in enumerate(self.classes_):
            # Get samples from class k
            X_k = X[y == k]
            
            # Estimate mean
            self.means_[i] = np.mean(X_k, axis=0)
            
            # Estimate covariance
            self.covariances_[i] = np.cov(X_k, rowvar=False)
            
            # Create multivariate normal distribution
            density = multivariate_normal(
                mean=self.means_[i], 
                cov=self.covariances_[i]
            )
            self.conditional_densities_.append(density)
    
    def decision_function(self, X):
        """Compute decision function values for each class"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decisions = np.zeros((n_samples, n_classes))
        
        for i, k in enumerate(self.classes_):
            # Compute quadratic discriminant function
            diff = X - self.means_[i]
            inv_cov = np.linalg.inv(self.covariances_[i])
            
            # Quadratic term
            quad_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            
            # Log determinant term
            log_det_term = -0.5 * np.log(np.linalg.det(self.covariances_[i]))
            
            # Prior term
            prior_term = np.log(self.priors_[i])
            
            decisions[:, i] = quad_term + log_det_term + prior_term
        
        return decisions

# Fit QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Make predictions
qda_predictions = qda.predict(X_test)
qda_probabilities = qda.predict_proba(X_test)

print("QDA Results:")
print(f"Accuracy: {qda.score(X_test, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, qda_predictions))

# Visualize decision boundaries
def plot_decision_boundaries_qda(X, y, qda_model, title="QDA Decision Boundaries"):
    """Plot QDA decision boundaries"""
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on mesh grid
    Z = qda_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k', cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

# Plot QDA decision boundaries
plot_decision_boundaries_qda(X_test, y_test, qda)
```

## 9.2.4. Linear Discriminant Analysis (LDA)

### Mathematical Formulation

LDA assumes that all classes share the same covariance matrix $`\Sigma`$:

```math
f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right)
```

The decision function becomes linear:

```math
\delta_k(x) = \mu_k^T \Sigma^{-1} x - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k
```

### Parameter Estimation

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$
3. **Shared covariance**: $`\hat{\Sigma} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$

### Implementation: LDA

```python
class LinearDiscriminantAnalysis(BayesClassifier):
    """Linear Discriminant Analysis"""
    
    def _fit_conditional_densities(self, X, y):
        """Fit Gaussian densities with shared covariance"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)
        
        self.means_ = np.zeros((n_classes, n_features))
        self.conditional_densities_ = []
        
        # Estimate class means
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.means_[i] = np.mean(X_k, axis=0)
        
        # Estimate shared covariance matrix
        self.shared_covariance_ = np.zeros((n_features, n_features))
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            diff = X_k - self.means_[i]
            self.shared_covariance_ += diff.T @ diff
        
        self.shared_covariance_ /= (n_samples - n_classes)
        
        # Create multivariate normal distributions with shared covariance
        for i in range(n_classes):
            density = multivariate_normal(
                mean=self.means_[i], 
                cov=self.shared_covariance_
            )
            self.conditional_densities_.append(density)
    
    def decision_function(self, X):
        """Compute linear decision function values"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decisions = np.zeros((n_samples, n_classes))
        
        # Precompute inverse covariance
        inv_cov = np.linalg.inv(self.shared_covariance_)
        
        for i, k in enumerate(self.classes_):
            # Linear discriminant function
            linear_term = self.means_[i] @ inv_cov @ X.T
            constant_term = -0.5 * self.means_[i] @ inv_cov @ self.means_[i]
            prior_term = np.log(self.priors_[i])
            
            decisions[:, i] = linear_term + constant_term + prior_term
        
        return decisions

# Fit LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Make predictions
lda_predictions = lda.predict(X_test)
lda_probabilities = lda.predict_proba(X_test)

print("LDA Results:")
print(f"Accuracy: {lda.score(X_test, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, lda_predictions))

# Compare QDA vs LDA
def compare_qda_lda(X_train, y_train, X_test, y_test):
    """Compare QDA and LDA performance"""
    # Fit both models
    qda = QuadraticDiscriminantAnalysis()
    lda = LinearDiscriminantAnalysis()
    
    qda.fit(X_train, y_train)
    lda.fit(X_train, y_train)
    
    # Predictions
    qda_pred = qda.predict(X_test)
    lda_pred = lda.predict(X_test)
    
    # Results
    print("Model Comparison:")
    print(f"QDA Accuracy: {qda.score(X_test, y_test):.3f}")
    print(f"LDA Accuracy: {lda.score(X_test, y_test):.3f}")
    
    # Visualize decision boundaries
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # QDA boundaries
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # QDA
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_qda = Z_qda.reshape(xx.shape)
    ax1.contourf(xx, yy, Z_qda, alpha=0.4, cmap='viridis')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    ax1.set_title('QDA Decision Boundaries')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # LDA
    Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lda = Z_lda.reshape(xx.shape)
    ax2.contourf(xx, yy, Z_lda, alpha=0.4, cmap='viridis')
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
    ax2.set_title('LDA Decision Boundaries')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

# Compare models
compare_qda_lda(X_train, y_train, X_test, y_test)
```

## 9.2.5. Naive Bayes

### Mathematical Formulation

Naive Bayes assumes conditional independence of features given the class:

```math
f_k(x) = \prod_{j=1}^p f_{kj}(x_j)
```

where $`f_{kj}(x_j)`$ is the marginal density of feature $`j`$ in class $`k`$.

The decision function becomes:

```math
\delta_k(x) = \log \pi_k + \sum_{j=1}^p \log f_{kj}(x_j)
```

### Implementation: Gaussian Naive Bayes

```python
class GaussianNaiveBayes(BayesClassifier):
    """Gaussian Naive Bayes classifier"""
    
    def _fit_conditional_densities(self, X, y):
        """Fit independent Gaussian densities for each feature and class"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.conditional_densities_ = []
        
        for i, k in enumerate(self.classes_):
            X_k = X[y == k]
            
            # Estimate means and variances for each feature
            self.means_[i] = np.mean(X_k, axis=0)
            self.variances_[i] = np.var(X_k, axis=0, ddof=1)
            
            # Create independent Gaussian distributions
            density = multivariate_normal(
                mean=self.means_[i],
                cov=np.diag(self.variances_[i])  # Diagonal covariance matrix
            )
            self.conditional_densities_.append(density)
    
    def decision_function(self, X):
        """Compute naive Bayes decision function values"""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decisions = np.zeros((n_samples, n_classes))
        
        for i, k in enumerate(self.classes_):
            # Compute log-likelihood for each feature independently
            log_likelihood = 0
            for j in range(X.shape[1]):
                # Log density of univariate Gaussian
                diff = X[:, j] - self.means_[i, j]
                log_likelihood += (-0.5 * np.log(2 * np.pi * self.variances_[i, j]) - 
                                  0.5 * diff**2 / self.variances_[i, j])
            
            decisions[:, i] = log_likelihood + np.log(self.priors_[i])
        
        return decisions

# Fit Gaussian Naive Bayes
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Make predictions
gnb_predictions = gnb.predict(X_test)
gnb_probabilities = gnb.predict_proba(X_test)

print("Gaussian Naive Bayes Results:")
print(f"Accuracy: {gnb.score(X_test, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, gnb_predictions))
```

## 9.2.6. Fisher's Discriminant Analysis (FDA)

### Mathematical Foundation

FDA finds a linear projection that maximizes the ratio of between-class variance to within-class variance:

```math
J(w) = \frac{w^T S_B w}{w^T S_W w}
```

where:
- $`S_B = \sum_{k=1}^K n_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^T`$ is the between-class scatter matrix
- $`S_W = \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^T`$ is the within-class scatter matrix
- $`\bar{\mu} = \frac{1}{n} \sum_{i=1}^n x_i`$ is the overall mean

### Solution

The optimal projection vector is the eigenvector corresponding to the largest eigenvalue of $`S_W^{-1} S_B`$:

```math
S_W^{-1} S_B w = \lambda w
```

### Implementation: FDA

```python
class FishersDiscriminantAnalysis:
    """Fisher's Discriminant Analysis"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X, y):
        """Fit FDA"""
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Compute overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Compute class means
        class_means = np.zeros((n_classes, n_features))
        for i, k in enumerate(classes):
            class_means[i] = np.mean(X[y == k], axis=0)
        
        # Compute between-class scatter matrix
        S_B = np.zeros((n_features, n_features))
        for i, k in enumerate(classes):
            n_k = np.sum(y == k)
            diff = class_means[i] - overall_mean
            S_B += n_k * np.outer(diff, diff)
        
        # Compute within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        for i, k in enumerate(classes):
            X_k = X[y == k]
            diff = X_k - class_means[i]
            S_W += diff.T @ diff
        
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eigh(np.linalg.inv(S_W) @ S_B)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Store results
        self.eigenvalues_ = eigenvals
        self.eigenvectors_ = eigenvecs
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # Compute explained variance ratio
        self.explained_variance_ratio_ = eigenvals[:self.n_components] / np.sum(eigenvals)
        
        return self
    
    def transform(self, X):
        """Transform data using FDA projection"""
        return X @ self.eigenvectors_[:, :self.n_components]
    
    def fit_transform(self, X, y):
        """Fit FDA and transform data"""
        return self.fit(X, y).transform(X)

# Apply FDA for dimensionality reduction
fda = FishersDiscriminantAnalysis(n_components=2)
X_train_fda = fda.fit_transform(X_train, y_train)
X_test_fda = fda.transform(X_test)

print("FDA Results:")
print(f"Explained variance ratio: {fda.explained_variance_ratio_}")
print(f"Eigenvalues: {fda.eigenvalues_[:2]}")

# Visualize FDA projection
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.8, edgecolors='k')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# FDA projection
plt.subplot(1, 2, 2)
plt.scatter(X_test_fda[:, 0], X_test_fda[:, 1], c=y_test, alpha=0.8, edgecolors='k')
plt.title('FDA Projection')
plt.xlabel('First Discriminant')
plt.ylabel('Second Discriminant')

plt.tight_layout()
plt.show()

# Apply LDA on FDA-transformed data
lda_fda = LinearDiscriminantAnalysis()
lda_fda.fit(X_train_fda, y_train)
fda_lda_accuracy = lda_fda.score(X_test_fda, y_test)

print(f"LDA on FDA-transformed data accuracy: {fda_lda_accuracy:.3f}")
```

## 9.2.7. Model Comparison and Selection

### Theoretical Comparison

| Method | Assumptions | Decision Boundary | Complexity |
|--------|-------------|-------------------|------------|
| **QDA** | Different covariances | Quadratic | $`O(p^2)`$ |
| **LDA** | Shared covariance | Linear | $`O(p^2)`$ |
| **Naive Bayes** | Feature independence | Piecewise linear | $`O(p)`$ |

### Practical Comparison

```python
def comprehensive_model_comparison(X_train, y_train, X_test, y_test):
    """Comprehensive comparison of discriminant analysis methods"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import time
    
    models = {
        'Our QDA': QuadraticDiscriminantAnalysis(),
        'Our LDA': LinearDiscriminantAnalysis(),
        'Our Naive Bayes': GaussianNaiveBayes(),
        'Sklearn LDA': SklearnLDA(),
        'Sklearn QDA': SklearnQDA(),
        'Sklearn Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        # Time the fitting
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fit_time': fit_time,
            'predict_time': predict_time
        }
    
    # Create comparison table
    df_results = pd.DataFrame(results).T
    print("Model Comparison Results:")
    print(df_results.round(4))
    
    return results, df_results

# Run comprehensive comparison
results, df_results = comprehensive_model_comparison(X_train, y_train, X_test, y_test)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
metrics = ['accuracy', 'precision', 'recall', 'f1']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = [results[name][metric] for name in results.keys()]
    names = list(results.keys())
    
    bars = ax.bar(range(len(values)), values, alpha=0.8)
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## 9.2.8. Practical Considerations

### Model Selection Guidelines

1. **Use LDA when**:
   - Classes have similar covariance structures
   - Sample size is small relative to number of features
   - Linear decision boundaries are appropriate

2. **Use QDA when**:
   - Classes have different covariance structures
   - Sufficient data to estimate class-specific covariances
   - Non-linear decision boundaries are needed

3. **Use Naive Bayes when**:
   - Features are approximately independent given class
   - High-dimensional data with limited samples
   - Fast prediction is required

### Regularization and Robustness

```python
def regularized_lda(X_train, y_train, X_test, y_test, alpha=0.1):
    """Regularized LDA with shrinkage"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Regularized LDA with shrinkage
    lda_reg = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=alpha)
    lda_reg.fit(X_train, y_train)
    
    accuracy = lda_reg.score(X_test, y_test)
    print(f"Regularized LDA (α={alpha}) accuracy: {accuracy:.3f}")
    
    return lda_reg

# Test regularization
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.1)
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.5)
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.9)
```

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of discriminant analysis methods. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and comparison of different approaches.
