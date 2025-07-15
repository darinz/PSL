# 11.1. Introduction to Support Vector Machines (SVM)

## Introduction

Support Vector Machines (SVM) are powerful supervised learning algorithms that excel at classification tasks by finding optimal hyperplanes that separate different classes. SVMs are particularly effective in high-dimensional spaces and are known for their robustness and theoretical foundations.

## Key Concepts

### 1. **Margin Maximization**
The fundamental idea behind SVM is to find a hyperplane that maximizes the margin - the distance between the hyperplane and the nearest data points from each class.

### 2. **Support Vectors**
Only a subset of training points, called support vectors, determine the optimal hyperplane. These are the points that lie on or near the margin boundaries.

### 3. **Kernel Trick**
SVMs can handle nonlinear classification by implicitly mapping data to higher-dimensional spaces using kernel functions.

## Linear SVM: Separable Case

### Problem Setup

Consider a binary classification problem with linearly separable data. We have:
- Training data: $`\{(\mathbf{x}_i, y_i)\}_{i=1}^n`$ where $`\mathbf{x}_i \in \mathbb{R}^p`$ and $`y_i \in \{-1, +1\}`$
- Goal: Find a hyperplane $`f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = 0`$ that separates the classes

### Geometric Intuition

The hyperplane divides the space into two regions:
- $`f(\mathbf{x}) > 0`$ for class +1
- $`f(\mathbf{x}) < 0`$ for class -1

The margin is the distance between two parallel hyperplanes:
- $`f(\mathbf{x}) = +1`$ (positive margin boundary)
- $`f(\mathbf{x}) = -1`$ (negative margin boundary)

### Mathematical Formulation

The margin width is $`\frac{2}{\|\mathbf{w}\|}`$. To maximize the margin, we minimize $`\|\mathbf{w}\|`$:

```math
\begin{align*}
&\min_{\mathbf{w}, b} \quad \frac{1}{2} \|\mathbf{w}\|^2 \\
&\text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i = 1, \ldots, n
\end{align*}
```

### Constraint Interpretation

The constraints ensure that:
- Points with $`y_i = +1`$ satisfy $`\mathbf{w}^T \mathbf{x}_i + b \geq 1`$
- Points with $`y_i = -1`$ satisfy $`\mathbf{w}^T \mathbf{x}_i + b \leq -1`$

Points that satisfy $`y_i (\mathbf{w}^T \mathbf{x}_i + b) = 1`$ lie exactly on the margin boundaries and are called **support vectors**.

### Dual Formulation

Using Lagrange multipliers, we can derive the dual problem:

```math
\begin{align*}
&\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
&\text{subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0 \quad \forall i
\end{align*}
```

### Solution Properties

1. **Complementarity Condition**: $`\alpha_i [y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1] = 0`$
2. **Support Vectors**: Points with $`\alpha_i > 0`$ are support vectors
3. **Weight Vector**: $`\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i`$
4. **Bias Term**: $`b = y_i - \mathbf{w}^T \mathbf{x}_i`$ for any support vector

## Linear SVM: Non-Separable Case

### Problem Motivation

When data is not linearly separable, we introduce slack variables $`\xi_i \geq 0`$ to allow some points to violate the margin constraints.

### Mathematical Formulation

```math
\begin{align*}
&\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \\
&\text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \quad \forall i \\
&\quad \quad \quad \quad \xi_i \geq 0 \quad \forall i
\end{align*}
```

### Parameter C
The parameter $`C`$ controls the trade-off between:
- Maximizing the margin (smaller $`\|\mathbf{w}\|`$)
- Minimizing classification errors (smaller $`\sum \xi_i`$)

### Dual Formulation

```math
\begin{align*}
&\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
&\text{subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C \quad \forall i
\end{align*}
```

## Nonlinear SVM and Kernel Trick

### Feature Space Mapping

To handle nonlinear decision boundaries, we map data to a higher-dimensional feature space:

```math
\Phi : \mathbb{R}^p \to \mathcal{H}, \quad \mathbf{x} \mapsto \Phi(\mathbf{x})
```

### Kernel Function

Instead of explicitly computing $`\Phi(\mathbf{x})`$, we use a kernel function:

```math
K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle
```

### Dual Problem with Kernel

```math
\begin{align*}
&\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \\
&\text{subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C \quad \forall i
\end{align*}
```

### Decision Function

```math
f(\mathbf{x}) = \sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b
```

### Popular Kernels

1. **Linear Kernel**: $`K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j`$
2. **Polynomial Kernel**: $`K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d`$
3. **RBF Kernel**: $`K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)`$
4. **Sigmoid Kernel**: $`K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)`$

## SVM as Regularization Method

### Hinge Loss

SVM can be viewed as minimizing the hinge loss with L2 regularization:

```math
L(y, f(\mathbf{x})) = \max(0, 1 - y f(\mathbf{x}))
```

### Optimization Problem

```math
\min_{\mathbf{w}, b} \quad \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)) + \frac{\lambda}{2} \|\mathbf{w}\|^2
```

### Comparison with Other Loss Functions

1. **0-1 Loss**: $`L(y, f) = \mathbb{I}[y f \leq 0]`$
2. **Hinge Loss**: $`L(y, f) = \max(0, 1 - y f)`$
3. **Logistic Loss**: $`L(y, f) = \log(1 + e^{-y f})`$
4. **Squared Loss**: $`L(y, f) = (1 - y f)^2``

## Implementation and Demonstration

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, make_circles
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns

class SVMDemo:
    def __init__(self):
        self.models = {}
        self.data = {}
        
    def generate_separable_data(self, n_samples=100, random_state=42):
        """Generate linearly separable data"""
        np.random.seed(random_state)
        
        # Generate two classes with clear separation
        n_class1 = n_samples // 2
        n_class2 = n_samples - n_class1
        
        # Class 1: centered at (2, 2)
        class1 = np.random.randn(n_class1, 2) + np.array([2, 2])
        
        # Class 2: centered at (-2, -2)
        class2 = np.random.randn(n_class2, 2) + np.array([-2, -2])
        
        X = np.vstack([class1, class2])
        y = np.hstack([np.ones(n_class1), -np.ones(n_class2)])
        
        self.data['separable'] = {'X': X, 'y': y}
        return X, y
    
    def generate_nonseparable_data(self, n_samples=100, random_state=42):
        """Generate non-linearly separable data"""
        np.random.seed(random_state)
        
        # Generate circular data
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
        
        self.data['nonseparable'] = {'X': X, 'y': y}
        return X, y
    
    def generate_overlapping_data(self, n_samples=100, random_state=42):
        """Generate overlapping data for soft margin demonstration"""
        np.random.seed(random_state)
        
        # Generate overlapping classes
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, 
                                 random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
        
        self.data['overlapping'] = {'X': X, 'y': y}
        return X, y
    
    def fit_linear_svm(self, X, y, C=1.0):
        """Fit linear SVM"""
        model = SVC(kernel='linear', C=C, random_state=42)
        model.fit(X, y)
        return model
    
    def fit_rbf_svm(self, X, y, C=1.0, gamma='scale'):
        """Fit RBF kernel SVM"""
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X, y)
        return model
    
    def visualize_decision_boundary(self, X, y, model, title="SVM Decision Boundary"):
        """Visualize SVM decision boundary"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh points
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and regions
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                           edgecolors='black', s=50, alpha=0.8)
        
        # Highlight support vectors
        if hasattr(model, 'support_vectors_'):
            ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                      s=200, facecolors='none', edgecolors='red', linewidth=2,
                      label=f'Support Vectors ({len(model.support_vectors_)})')
            ax.legend()
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.show()
    
    def demonstrate_separable_case(self):
        """Demonstrate linear SVM on separable data"""
        print("=== Linear SVM: Separable Case ===\n")
        
        # Generate data
        X, y = self.generate_separable_data()
        
        # Fit SVM
        model = self.fit_linear_svm(X, y)
        
        # Evaluate
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of support vectors: {len(model.support_vectors_)}")
        print(f"Support vector ratio: {len(model.support_vectors_)/len(X):.2f}")
        
        # Visualize
        self.visualize_decision_boundary(X, y, model, "Linear SVM: Separable Case")
        
        return model
    
    def demonstrate_nonseparable_case(self):
        """Demonstrate RBF SVM on non-separable data"""
        print("=== Nonlinear SVM: Non-Separable Case ===\n")
        
        # Generate data
        X, y = self.generate_nonseparable_data()
        
        # Fit linear SVM (should perform poorly)
        linear_model = self.fit_linear_svm(X, y)
        linear_accuracy = accuracy_score(y, linear_model.predict(X))
        
        # Fit RBF SVM
        rbf_model = self.fit_rbf_svm(X, y)
        rbf_accuracy = accuracy_score(y, rbf_model.predict(X))
        
        print(f"Linear SVM Accuracy: {linear_accuracy:.4f}")
        print(f"RBF SVM Accuracy: {rbf_accuracy:.4f}")
        print(f"RBF SVM Support Vectors: {len(rbf_model.support_vectors_)}")
        
        # Visualize both
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (model, title) in enumerate([(linear_model, "Linear SVM"), 
                                          (rbf_model, "RBF SVM")]):
            ax = axes[i]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            ax.set_title(f"{title}\nAccuracy: {accuracy_score(y, model.predict(X)):.3f}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return rbf_model
    
    def demonstrate_soft_margin(self):
        """Demonstrate soft margin SVM with different C values"""
        print("=== Soft Margin SVM ===\n")
        
        # Generate overlapping data
        X, y = self.generate_overlapping_data()
        
        # Try different C values
        C_values = [0.1, 1.0, 10.0, 100.0]
        models = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, C in enumerate(C_values):
            model = self.fit_linear_svm(X, y, C=C)
            models[C] = model
            
            ax = axes[i // 2, i % 2]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            
            ax.set_title(f"C = {C}\nAccuracy: {accuracy:.3f}, SVs: {n_sv}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("Summary:")
        for C, model in models.items():
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            print(f"C = {C:>6}: Accuracy = {accuracy:.3f}, Support Vectors = {n_sv}")
        
        return models
    
    def demonstrate_kernels(self):
        """Demonstrate different kernel functions"""
        print("=== Kernel Comparison ===\n")
        
        # Generate non-separable data
        X, y = self.generate_nonseparable_data()
        
        # Define kernels to test
        kernels = [
            ('linear', {'kernel': 'linear'}),
            ('poly', {'kernel': 'poly', 'degree': 3}),
            ('rbf', {'kernel': 'rbf'}),
            ('sigmoid', {'kernel': 'sigmoid'})
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (name, params) in enumerate(kernels):
            model = SVC(C=1.0, random_state=42, **params)
            model.fit(X, y)
            
            ax = axes[i // 2, i % 2]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            
            ax.set_title(f"{name.upper()} Kernel\nAccuracy: {accuracy:.3f}, SVs: {n_sv}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_hyperparameter_tuning(self):
        """Demonstrate hyperparameter tuning with GridSearchCV"""
        print("=== Hyperparameter Tuning ===\n")
        
        # Generate data
        X, y = self.generate_nonseparable_data(n_samples=200)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        # Grid search
        grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Test set accuracy: {grid_search.score(X_test, y_test):.3f}")
        
        # Visualize best model
        best_model = grid_search.best_estimator_
        self.visualize_decision_boundary(X, y, best_model, 
                                       f"Best SVM (C={best_model.C}, gamma={best_model.gamma})")
        
        return grid_search

# Run demonstrations
demo = SVMDemo()

# 1. Separable case
separable_model = demo.demonstrate_separable_case()

# 2. Non-separable case
nonseparable_model = demo.demonstrate_nonseparable_case()

# 3. Soft margin
soft_margin_models = demo.demonstrate_soft_margin()

# 4. Kernel comparison
demo.demonstrate_kernels()

# 5. Hyperparameter tuning
tuned_model = demo.demonstrate_hyperparameter_tuning()

# Additional analysis: Support vector analysis
print("\n=== Support Vector Analysis ===")
for name, data in demo.data.items():
    X, y = data['X'], data['y']
    model = demo.fit_rbf_svm(X, y)
    
    n_sv = len(model.support_vectors_)
    sv_ratio = n_sv / len(X)
    
    print(f"{name.capitalize()} data:")
    print(f"  Total samples: {len(X)}")
    print(f"  Support vectors: {n_sv}")
    print(f"  SV ratio: {sv_ratio:.3f}")
    print()
```

### R Implementation

```r
# Support Vector Machines in R

# Load required libraries
library(e1071)
library(ggplot2)
library(gridExtra)
library(kernlab)

# Generate separable data
generate_separable_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  n_class1 <- n_samples %/% 2
  n_class2 <- n_samples - n_class1
  
  # Class 1: centered at (2, 2)
  class1 <- matrix(rnorm(n_class1 * 2), n_class1, 2) + matrix(c(2, 2), n_class1, 2, byrow = TRUE)
  
  # Class 2: centered at (-2, -2)
  class2 <- matrix(rnorm(n_class2 * 2), n_class2, 2) + matrix(c(-2, -2), n_class2, 2, byrow = TRUE)
  
  X <- rbind(class1, class2)
  y <- c(rep(1, n_class1), rep(-1, n_class2))
  
  return(list(X = X, y = y))
}

# Generate non-separable data
generate_nonseparable_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate circular data
  theta <- runif(n_samples, 0, 2 * pi)
  r <- runif(n_samples, 0.5, 1.5)
  
  # Inner circle (class 1)
  n_inner <- n_samples %/% 2
  inner_theta <- theta[1:n_inner]
  inner_r <- runif(n_inner, 0.5, 1.0)
  
  # Outer circle (class -1)
  outer_theta <- theta[(n_inner + 1):n_samples]
  outer_r <- runif(n_samples - n_inner, 1.2, 1.8)
  
  X_inner <- cbind(inner_r * cos(inner_theta), inner_r * sin(inner_theta))
  X_outer <- cbind(outer_r * cos(outer_theta), outer_r * sin(outer_theta))
  
  X <- rbind(X_inner, X_outer)
  y <- c(rep(1, n_inner), rep(-1, n_samples - n_inner))
  
  return(list(X = X, y = y))
}

# Generate overlapping data
generate_overlapping_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate overlapping classes
  X <- matrix(rnorm(n_samples * 2), n_samples, 2)
  y <- ifelse(X[, 1] + X[, 2] > 0, 1, -1)
  
  # Add some noise
  y[sample(n_samples, n_samples %/% 10)] <- -y[sample(n_samples, n_samples %/% 10)]
  
  return(list(X = X, y = y))
}

# Visualize decision boundary
visualize_decision_boundary <- function(X, y, model, title = "SVM Decision Boundary") {
  # Create grid
  x_min <- min(X[, 1]) - 0.5
  x_max <- max(X[, 1]) + 0.5
  y_min <- min(X[, 2]) - 0.5
  y_max <- max(X[, 2]) + 0.5
  
  grid_x <- seq(x_min, x_max, length.out = 100)
  grid_y <- seq(y_min, y_max, length.out = 100)
  grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
  
  # Predict on grid
  grid_data$pred <- predict(model, grid_data)
  
  # Create plot
  p <- ggplot() +
    geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)), 
                 breaks = 0, color = "black", size = 1) +
    geom_point(data = data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y)), 
               aes(x = X1, y = X2, color = y), size = 3, alpha = 0.8) +
    scale_color_manual(values = c("-1" = "blue", "1" = "red")) +
    labs(title = title, x = "Feature 1", y = "Feature 2") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(p)
}

# Demonstrate separable case
demonstrate_separable_case <- function() {
  cat("=== Linear SVM: Separable Case ===\n\n")
  
  # Generate data
  data <- generate_separable_data()
  X <- data$X
  y <- data$y
  
  # Fit SVM
  model <- svm(X, y, kernel = "linear", scale = FALSE)
  
  # Evaluate
  y_pred <- predict(model, X)
  accuracy <- mean(y_pred == y)
  
  cat("Accuracy:", accuracy, "\n")
  cat("Number of support vectors:", length(model$index), "\n")
  cat("Support vector ratio:", length(model$index)/length(y), "\n\n")
  
  # Visualize
  p <- visualize_decision_boundary(X, y, model, "Linear SVM: Separable Case")
  print(p)
  
  return(model)
}

# Demonstrate non-separable case
demonstrate_nonseparable_case <- function() {
  cat("=== Nonlinear SVM: Non-Separable Case ===\n\n")
  
  # Generate data
  data <- generate_nonseparable_data()
  X <- data$X
  y <- data$y
  
  # Fit linear SVM
  linear_model <- svm(X, y, kernel = "linear", scale = FALSE)
  linear_accuracy <- mean(predict(linear_model, X) == y)
  
  # Fit RBF SVM
  rbf_model <- svm(X, y, kernel = "radial", scale = FALSE)
  rbf_accuracy <- mean(predict(rbf_model, X) == y)
  
  cat("Linear SVM Accuracy:", linear_accuracy, "\n")
  cat("RBF SVM Accuracy:", rbf_accuracy, "\n")
  cat("RBF SVM Support Vectors:", length(rbf_model$index), "\n\n")
  
  # Visualize both
  p1 <- visualize_decision_boundary(X, y, linear_model, "Linear SVM")
  p2 <- visualize_decision_boundary(X, y, rbf_model, "RBF SVM")
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(rbf_model)
}

# Demonstrate soft margin
demonstrate_soft_margin <- function() {
  cat("=== Soft Margin SVM ===\n\n")
  
  # Generate overlapping data
  data <- generate_overlapping_data()
  X <- data$X
  y <- data$y
  
  # Try different C values
  C_values <- c(0.1, 1.0, 10.0, 100.0)
  models <- list()
  plots <- list()
  
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- svm(X, y, kernel = "linear", cost = C, scale = FALSE)
    models[[i]] <- model
    
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    
    title <- paste("C =", C, "\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_decision_boundary(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print summary
  cat("Summary:\n")
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- models[[i]]
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    cat(sprintf("C = %6.1f: Accuracy = %.3f, Support Vectors = %d\n", C, accuracy, n_sv))
  }
  
  return(models)
}

# Demonstrate kernels
demonstrate_kernels <- function() {
  cat("=== Kernel Comparison ===\n\n")
  
  # Generate non-separable data
  data <- generate_nonseparable_data()
  X <- data$X
  y <- data$y
  
  # Define kernels to test
  kernels <- c("linear", "polynomial", "radial", "sigmoid")
  plots <- list()
  
  for (i in seq_along(kernels)) {
    kernel <- kernels[i]
    model <- svm(X, y, kernel = kernel, scale = FALSE)
    
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    
    title <- paste(toupper(kernel), "Kernel\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_decision_boundary(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
}

# Demonstrate hyperparameter tuning
demonstrate_hyperparameter_tuning <- function() {
  cat("=== Hyperparameter Tuning ===\n\n")
  
  # Generate data
  data <- generate_nonseparable_data(n_samples = 200)
  X <- data$X
  y <- data$y
  
  # Create data frame for tuning
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y))
  
  # Tune parameters
  tuned_model <- tune(svm, y ~ ., data = df, 
                     ranges = list(cost = c(0.1, 1, 10, 100),
                                  gamma = c(0.1, 0.5, 1, 2)),
                     kernel = "radial")
  
  cat("Best parameters:\n")
  print(tuned_model$best.parameters)
  cat("Best performance:", tuned_model$best.performance, "\n")
  
  # Visualize best model
  best_model <- tuned_model$best.model
  p <- visualize_decision_boundary(X, y, best_model, 
                                  "Best SVM (Tuned Parameters)")
  print(p)
  
  return(tuned_model)
}

# Run demonstrations
separable_model <- demonstrate_separable_case()
nonseparable_model <- demonstrate_nonseparable_case()
soft_margin_models <- demonstrate_soft_margin()
demonstrate_kernels()
tuned_model <- demonstrate_hyperparameter_tuning()

# Support vector analysis
cat("\n=== Support Vector Analysis ===\n")
data_types <- list(
  separable = generate_separable_data(),
  nonseparable = generate_nonseparable_data(),
  overlapping = generate_overlapping_data()
)

for (name in names(data_types)) {
  data <- data_types[[name]]
  model <- svm(data$X, data$y, kernel = "radial", scale = FALSE)
  
  n_sv <- length(model$index)
  sv_ratio <- n_sv / length(data$y)
  
  cat(name, "data:\n")
  cat("  Total samples:", length(data$y), "\n")
  cat("  Support vectors:", n_sv, "\n")
  cat("  SV ratio:", round(sv_ratio, 3), "\n\n")
}
```

## Key Insights

### 1. **Margin Maximization**
- SVM finds the hyperplane that maximizes the margin between classes
- This leads to better generalization and robustness
- Only support vectors influence the decision boundary

### 2. **Sparsity**
- The solution depends only on support vectors
- This makes SVM memory efficient and robust to outliers
- Non-support vectors can be moved without affecting the classifier

### 3. **Kernel Trick**
- Allows handling nonlinear decision boundaries
- Computationally efficient through implicit feature mapping
- Popular kernels: linear, polynomial, RBF, sigmoid

### 4. **Regularization**
- Parameter C controls the trade-off between margin and errors
- Larger C: smaller margin, fewer errors
- Smaller C: larger margin, more errors

### 5. **Theoretical Foundations**
- Based on structural risk minimization
- Strong theoretical guarantees
- Connection to regularization theory

## Applications

### 1. **Text Classification**
- Document categorization
- Spam detection
- Sentiment analysis

### 2. **Image Recognition**
- Face detection
- Object recognition
- Handwritten digit recognition

### 3. **Bioinformatics**
- Protein classification
- Gene expression analysis
- Disease diagnosis

### 4. **Finance**
- Credit scoring
- Fraud detection
- Market prediction

## Summary

Support Vector Machines are powerful classification algorithms that:

1. **Maximize margin** for better generalization
2. **Use support vectors** for sparse, robust solutions
3. **Handle nonlinearity** through kernel functions
4. **Provide regularization** through parameter C
5. **Have strong theoretical foundations** in statistical learning theory

SVMs are particularly effective for high-dimensional data and when the number of support vectors is small relative to the dataset size.