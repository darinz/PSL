# 11.4. Nonlinear SVMs

## 11.4.1. Linear SVM Recap

Before discussing the extension from a linear SVM to a non-linear SVM, let's briefly review the linear SVM, which we have covered extensively. In the linear SVM, we start with our primal problem, which involves terms like the slope $`\beta`$, intercept $`\beta_0`$, and the regularization parameter $`C`$. We solve the dual problem with the Lagrangian multipliers $`\lambda_1`$ to $`\lambda_n`$. The original parameters $`\beta`$ and $`\beta_0`$ can be found using the KKT condition, and they depend on a small set of support vectors.

**Key Insight**: The dual formulation reveals that we only need the Lagrange multipliers $`\lambda_i`$ and support vectors for prediction, not the explicit $`\beta`$ and $`\beta_0`$ parameters.

### Linear SVM Prediction

The decision function for linear SVM is:
```math
f(x) = \sum_{i=1}^n \lambda_i y_i x_i^T x + \beta_0
```

This shows that prediction only requires:
1. **Support vectors** $`x_i`$ (where $`\lambda_i > 0`$)
2. **Lagrange multipliers** $`\lambda_i`$
3. **Intercept** $`\beta_0`$

### Why This Matters for Nonlinear Extension

The fact that we only need inner products $`x_i^T x`$ in the prediction phase is crucial for the kernel trick. This allows us to replace linear inner products with nonlinear kernel functions.

## 11.4.2. Embedding and Feature Space Transformation

### The Need for Nonlinearity

Linear SVMs can only create linear decision boundaries. However, many real-world classification problems require nonlinear decision boundaries. Consider the classic XOR problem:

```python
import numpy as np
import matplotlib.pyplot as plt

# XOR-like data
X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([-1, -1, 1, 1])

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', s=100, label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', s=100, label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XOR Problem - Not Linearly Separable')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

This data cannot be separated by a linear boundary, motivating the need for nonlinear methods.

### Feature Space Embedding

To handle nonlinear problems, we transform the data into a higher-dimensional feature space where it becomes linearly separable:

```math
\Phi : \mathcal{X} \rightarrow \mathcal{F}, \quad \Phi(x) = (\phi_1(x), \phi_2(x), \ldots, \phi_d(x))
```

where $`\mathcal{X}`$ is the original input space and $`\mathcal{F}`$ is the feature space.

### Example: Polynomial Features

For a 2D input $`x = (x_1, x_2)`$, a quadratic transformation could be:
```math
\Phi(x) = (1, x_1, x_2, x_1^2, x_2^2, x_1 x_2)
```

This transforms 2D data into 6D space, where linear separation becomes possible.

### The Curse of Dimensionality

While embedding can make data linearly separable, it comes with computational costs:
- **Memory**: Storing high-dimensional feature vectors
- **Computation**: Computing inner products in high dimensions
- **Overfitting**: Risk of overfitting in high-dimensional spaces

## 11.4.3. The Kernel Trick

### The Key Insight

The kernel trick allows us to compute inner products in the feature space without explicitly computing the feature transformation:

```math
K(x_i, x_j) = \langle \Phi(x_i), \Phi(x_j) \rangle_{\mathcal{F}}
```

### Why This Works

In the dual SVM formulation, we only need inner products between data points. The kernel function computes these inner products directly in the original space.

### Mathematical Foundation

The kernel function must satisfy the **Mercer condition**:
```math
\int \int K(x, y) f(x) f(y) dx dy \geq 0
```
for all square-integrable functions $`f`$.

This ensures that $`K`$ corresponds to an inner product in some feature space.

### Popular Kernel Functions

#### 1. Linear Kernel
```math
K(x_i, x_j) = x_i^T x_j
```
Equivalent to no transformation (linear SVM).

#### 2. Polynomial Kernel
```math
K(x_i, x_j) = (\gamma x_i^T x_j + r)^d
```
where $`\gamma`$ is the scaling parameter, $`r`$ is the bias term, and $`d`$ is the degree.

#### 3. Radial Basis Function (RBF) Kernel
```math
K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
```
where $`\gamma > 0`$ controls the influence of each training point.

#### 4. Sigmoid Kernel
```math
K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)
```
Similar to neural network activation functions.

### Kernel Matrix Properties

The kernel matrix $`K_{ij} = K(x_i, x_j)`$ must be:
- **Symmetric**: $`K_{ij} = K_{ji}`$
- **Positive semi-definite**: $`\alpha^T K \alpha \geq 0`$ for all $`\alpha`$

## 11.4.4. Nonlinear SVM Formulation

### Dual Problem with Kernels

The dual problem for nonlinear SVM becomes:
```math
\begin{aligned}
\max_{\lambda} \quad & \sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \lambda_i \lambda_j y_i y_j K(x_i, x_j) \\
\text{subject to} \quad & \sum_{i=1}^n \lambda_i y_i = 0 \\
& 0 \leq \lambda_i \leq C, \quad i = 1, 2, \ldots, n
\end{aligned}
```

### Decision Function

The decision function becomes:
```math
f(x) = \sum_{i=1}^n \lambda_i y_i K(x_i, x) + \beta_0
```

### Computing the Intercept

For nonlinear SVM, the intercept is computed as:
```math
\beta_0 = y_i - \sum_{j=1}^n \lambda_j y_j K(x_j, x_i)
```
for any support vector $`x_i`$.

## 11.4.5. Loss + Penalty Framework

### Primal Formulation

The primal problem in the feature space is:
```math
\min_{\beta, \beta_0} \quad \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n [1 - y_i(\beta^T \Phi(x_i) + \beta_0)]_+
```

### Representer Theorem

The representer theorem states that the solution can be written as:
```math
\beta = \sum_{i=1}^n \alpha_i \Phi(x_i)
```

where $`\alpha_i = \lambda_i y_i`$.

### Dual Formulation with Kernels

Substituting the representer form into the primal:
```math
\min_{\alpha} \quad \frac{1}{2}\alpha^T K \alpha + C\sum_{i=1}^n [1 - y_i \sum_{j=1}^n \alpha_j K(x_i, x_j)]_+
```

This shows that the penalty term becomes $`\frac{1}{2}\alpha^T K \alpha`$, a generalized ridge penalty.

## 11.4.6. Implementation and Examples

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
import cvxopt
from cvxopt import matrix, solvers

class KernelSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, degree=3, coef0=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_vectors = None
        self.lambda_values = None
        self.beta_0 = None
        
    def kernel_function(self, X1, X2):
        """Compute kernel matrix between X1 and X2"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # Compute pairwise distances
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            K = np.exp(-self.gamma * (X1_norm + X2_norm - 2 * np.dot(X1, X2.T)))
            return K
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_function(X, X)
        
        # Prepare the quadratic programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        
        # Constraints: 0 <= lambda_i <= C
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))
        
        A = matrix(y.reshape(1, -1))
        b = matrix(0.0)
        
        # Solve the quadratic programming problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Extract Lagrange multipliers
        self.lambda_values = np.array(solution['x']).flatten()
        
        # Find support vectors
        support_vector_indices = self.lambda_values > 1e-5
        self.support_vectors = X[support_vector_indices]
        support_vector_lambdas = self.lambda_values[support_vector_indices]
        support_vector_y = y[support_vector_indices]
        
        # Compute beta_0
        self.beta_0 = np.mean(support_vector_y - 
                             np.sum(support_vector_lambdas.reshape(-1, 1) * 
                                   support_vector_y.reshape(-1, 1) * 
                                   self.kernel_function(self.support_vectors, self.support_vectors), axis=0))
        
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def decision_function(self, X):
        if self.support_vectors is None:
            return np.zeros(X.shape[0])
        
        K = self.kernel_function(X, self.support_vectors)
        support_vector_lambdas = self.lambda_values[self.lambda_values > 1e-5]
        support_vector_y = np.array([1 if i == 1 else -1 for i in range(len(self.lambda_values)) if self.lambda_values[i] > 1e-5])
        
        return np.sum(support_vector_lambdas.reshape(-1, 1) * 
                     support_vector_y.reshape(-1, 1) * K.T, axis=0) + self.beta_0

# Generate non-linear data
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
y = 2 * y - 1  # Convert to {-1, 1}

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare different kernels
kernels = ['linear', 'poly', 'rbf']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, kernel in enumerate(kernels):
    # Fit SVM
    svm = KernelSVM(C=1.0, kernel=kernel, gamma=1.0)
    svm.fit(X_scaled, y)
    
    # Plotting
    ax = axes[i]
    
    # Plot data points
    ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
              c='red', label='Class 1', alpha=0.6)
    ax.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], 
              c='blue', label='Class -1', alpha=0.6)
    
    # Plot decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='black')
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, 
                colors=['blue', 'white', 'red'])
    
    # Highlight support vectors
    if svm.support_vectors is not None:
        ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=100, linewidth=1, facecolors='none', edgecolors='k', 
                   label='Support Vectors')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Kernel SVM ({kernel.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print model information
for kernel in kernels:
    svm = KernelSVM(C=1.0, kernel=kernel, gamma=1.0)
    svm.fit(X_scaled, y)
    n_support_vectors = len(svm.support_vectors) if svm.support_vectors is not None else 0
    print(f"{kernel.upper()} kernel: {n_support_vectors} support vectors")
```

### R Implementation

```r
library(e1071)
library(ggplot2)

# Generate non-linear data
set.seed(42)
n <- 100
theta <- runif(n, 0, 2*pi)
r <- runif(n, 0.5, 1.5)
X <- cbind(r * cos(theta), r * sin(theta))
y <- ifelse(r < 1, 1, -1)

# Function to fit and plot kernel SVM
plot_kernel_svm <- function(X, y, kernel_type, gamma=1, degree=3) {
  # Fit SVM
  svm_model <- svm(X, y, kernel=kernel_type, gamma=gamma, degree=degree, scale=FALSE)
  
  # Create prediction grid
  x_min <- min(X[,1]) - 0.5
  x_max <- max(X[,1]) + 0.5
  y_min <- min(X[,2]) - 0.5
  y_max <- max(X[,2]) + 0.5
  
  grid_points <- expand.grid(
    x1 = seq(x_min, x_max, length.out=50),
    x2 = seq(y_min, y_max, length.out=50)
  )
  
  # Make predictions
  grid_points$pred <- predict(svm_model, grid_points)
  
  # Plot
  p <- ggplot() +
    geom_point(data=data.frame(X, y=factor(y)), 
               aes(x=X1, y=X2, color=y), size=2) +
    geom_contour(data=grid_points, 
                 aes(x=x1, y=x2, z=as.numeric(pred)), 
                 breaks=c(0.5), color="black", size=1) +
    geom_point(data=data.frame(X[svm_model$index,]), 
               aes(x=X1, y=X2), shape=21, size=3, 
               fill="transparent", color="black") +
    labs(title=paste("Kernel SVM (", toupper(kernel_type), ")", sep=""), 
         x="Feature 1", y="Feature 2") +
    theme_minimal()
  
  return(list(plot=p, model=svm_model))
}

# Compare different kernels
kernels <- c("linear", "polynomial", "radial")
plots <- lapply(kernels, function(k) plot_kernel_svm(X, y, k))

# Display plots
for(i in 1:length(plots)) {
  print(plots[[i]]$plot)
  cat("Kernel =", kernels[i], ": Number of support vectors =", 
      length(plots[[i]]$model$index), "\n")
}
```

## 11.4.7. Kernel Selection and Parameter Tuning

### Kernel Selection Guidelines

1. **Linear Kernel**: When data is linearly separable or nearly so
2. **Polynomial Kernel**: When features have multiplicative interactions
3. **RBF Kernel**: Most commonly used, works well for most problems
4. **Sigmoid Kernel**: Similar to neural networks, less commonly used

### Parameter Tuning

#### For RBF Kernel
- **$`\gamma`$**: Controls the influence of each training point
  - Large $`\gamma`$: Narrow Gaussian, may overfit
  - Small $`\gamma`$: Wide Gaussian, may underfit

#### For Polynomial Kernel
- **$`d`$**: Degree of polynomial
- **$`\gamma`$**: Scaling parameter
- **$`r`$**: Bias term

### Cross-Validation for Kernel Selection

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Define parameter grids for different kernels
param_grids = {
    'linear': {'C': [0.1, 1, 10, 100]},
    'poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': [0.1, 1, 10]},
    'rbf': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
}

best_scores = {}
best_params = {}

for kernel, param_grid in param_grids.items():
    svm = SVC(kernel=kernel, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    best_scores[kernel] = grid_search.best_score_
    best_params[kernel] = grid_search.best_params_
    
    print(f"{kernel.upper()} kernel:")
    print(f"  Best score: {grid_search.best_score_:.3f}")
    print(f"  Best parameters: {grid_search.best_params_}")

# Find best kernel
best_kernel = max(best_scores, key=best_scores.get)
print(f"\nBest kernel: {best_kernel.upper()} with score {best_scores[best_kernel]:.3f}")
```

## 11.4.8. The Kernel Machine Perspective

### Alternative Viewpoint

Instead of thinking about feature transformations, we can view kernel SVM as a **similarity-based classifier**:

1. **Training**: Each training point becomes a "prototype"
2. **Prediction**: New points are classified based on similarity to prototypes
3. **Weights**: Lagrange multipliers determine the importance of each prototype

### Connection to k-Nearest Neighbors

Kernel SVM can be seen as a weighted version of k-NN:
- **k-NN**: Equal weights for k nearest neighbors
- **Kernel SVM**: Learned weights (Lagrange multipliers) for all training points

### Representer Theorem

The representer theorem guarantees that the optimal solution has the form:
```math
f(x) = \sum_{i=1}^n \alpha_i K(x_i, x) + \beta_0
```

This means we never need to explicitly compute the feature transformation $`\Phi(x)`$.

## 11.4.9. Reproducing Kernel Hilbert Space (RKHS)

### Mathematical Foundation

An RKHS is a Hilbert space of functions where:
1. **Evaluation functionals are continuous**
2. **Reproducing property**: $`f(x) = \langle f, K(x, \cdot) \rangle`$

### Properties of RKHS

1. **Fixed function space**: Independent of training data
2. **Finite representation**: Optimal solution uses only training points
3. **Regularization**: Natural penalty term $`\|f\|^2_{\mathcal{H}}`$

### Connection to SVM

The SVM objective in RKHS is:
```math
\min_{f \in \mathcal{H}} \quad \frac{1}{n}\sum_{i=1}^n [1 - y_i f(x_i)]_+ + \frac{1}{2C}\|f\|^2_{\mathcal{H}}
```

The representer theorem ensures the solution has the finite form above.

## 11.4.10. Advantages and Limitations

### Advantages

1. **Nonlinear Decision Boundaries**: Can handle complex classification problems
2. **Flexible Kernels**: Can choose kernel based on domain knowledge
3. **Sparse Solution**: Only support vectors matter
4. **Theoretical Foundation**: Based on solid mathematical theory
5. **Global Optimum**: Convex optimization problem

### Limitations

1. **Kernel Selection**: Need to choose appropriate kernel and parameters
2. **Computational Cost**: $`O(n^3)`$ training time, $`O(n_{sv})`$ prediction time
3. **Memory Requirements**: Need to store kernel matrix
4. **Interpretability**: Less interpretable than linear models
5. **Sensitivity to Parameters**: Performance depends heavily on kernel parameters

## 11.4.11. Summary

Nonlinear SVMs extend linear SVMs through the kernel trick:

1. **Feature Space Embedding**: Transform data to higher dimensions
2. **Kernel Trick**: Compute inner products without explicit transformation
3. **Kernel Functions**: RBF, polynomial, linear, sigmoid
4. **Dual Formulation**: Solve optimization in dual space
5. **Representer Theorem**: Finite representation using training points

Key insights:
- **Kernel trick**: Avoid explicit feature transformation
- **Mercer condition**: Ensures valid inner product
- **Support vectors**: Only critical points matter
- **Parameter tuning**: Essential for good performance
- **RKHS**: Mathematical foundation for kernel methods

This framework provides a powerful and flexible approach to nonlinear classification, setting the foundation for many modern machine learning algorithms.
