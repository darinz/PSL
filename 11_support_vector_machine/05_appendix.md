# 11.5. Appendix

## 11.5.1. SVM Mathematical Foundations

### Convex Optimization Review

Support Vector Machines are based on convex optimization principles. A convex optimization problem has the form:

```math
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, 2, \ldots, m \\
& h_j(x) = 0, \quad j = 1, 2, \ldots, p
\end{aligned}
```

where $`f(x)`$ is convex, $`g_i(x)`$ are convex, and $`h_j(x)`$ are affine.

**Key Properties**:
- **Global optimum**: Any local minimum is also global
- **KKT conditions**: Necessary and sufficient for optimality
- **Duality**: Primal and dual problems are related

### Lagrangian Duality

For the constrained optimization problem:
```math
\min_{x} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \quad i = 1, 2, \ldots, m
```

The Lagrangian is:
```math
L(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x)
```

**Dual function**:
```math
g(\lambda) = \inf_x L(x, \lambda)
```

**Dual problem**:
```math
\max_{\lambda \geq 0} g(\lambda)
```

### Strong Duality

For convex problems with Slater's condition, strong duality holds:
```math
\min_x \max_{\lambda \geq 0} L(x, \lambda) = \max_{\lambda \geq 0} \min_x L(x, \lambda)
```

This is why we can solve the dual problem instead of the primal.

## 11.5.2. Reproducing Kernel Hilbert Space (RKHS)

### Hilbert Space Basics

A **Hilbert space** $`\mathcal{H}`$ is a complete inner product space. Key properties:

1. **Inner product**: $`\langle f, g \rangle`$ for $`f, g \in \mathcal{H}`$
2. **Norm**: $`\|f\| = \sqrt{\langle f, f \rangle}`$
3. **Completeness**: Every Cauchy sequence converges

### Reproducing Property

An RKHS has the **reproducing property**:
```math
f(x) = \langle f, K(x, \cdot) \rangle_{\mathcal{H}}
```

where $`K(x, \cdot)`$ is the reproducing kernel.

### Kernel Construction

Given a positive definite kernel $`K(x, y)`$, we can construct an RKHS:

1. **Pre-Hilbert space**: Span of $`\{K(x_i, \cdot)\}`$
2. **Inner product**: $`\langle K(x_i, \cdot), K(x_j, \cdot) \rangle = K(x_i, x_j)`$
3. **Completion**: Add limit points to get full RKHS

### Representer Theorem

**Theorem**: Let $`\mathcal{H}`$ be an RKHS with kernel $`K`$. For any loss function $`L`$ and regularization term $`\Omega`$, the minimizer of:
```math
\min_{f \in \mathcal{H}} \sum_{i=1}^n L(y_i, f(x_i)) + \Omega(\|f\|_{\mathcal{H}})
```

has the form:
```math
f(x) = \sum_{i=1}^n \alpha_i K(x_i, x)
```

**Proof Sketch**:
1. Decompose $`f = f_s + f_\perp`$ where $`f_s`$ is in the span of $`\{K(x_i, \cdot)\}`$
2. Show $`f_\perp`$ doesn't affect the objective
3. Conclude optimal solution lies in the span

### Implementation of RKHS Concepts

```python
import numpy as np
from scipy.spatial.distance import cdist

class RKHS:
    def __init__(self, kernel='rbf', gamma=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.alpha = None
        
    def kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'rbf':
            # RBF kernel: K(x,y) = exp(-gamma ||x-y||^2)
            dist_sq = cdist(X1, X2, metric='sqeuclidean')
            return np.exp(-self.gamma * dist_sq)
        elif self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y, lambda_reg=0.1):
        """Fit using representer theorem"""
        self.X_train = X
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_function(X, X)
        
        # Add regularization
        K_reg = K + lambda_reg * np.eye(n_samples)
        
        # Solve linear system: K_reg * alpha = y
        self.alpha = np.linalg.solve(K_reg, y)
        
    def predict(self, X):
        """Make predictions"""
        if self.X_train is None:
            raise ValueError("Model not fitted yet")
        
        K_test = self.kernel_function(X, self.X_train)
        return np.dot(K_test, self.alpha)

# Example usage
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)

rkhs = RKHS(kernel='rbf', gamma=1.0)
rkhs.fit(X, y, lambda_reg=0.01)

# Test predictions
X_test = np.random.randn(20, 2)
y_pred = rkhs.predict(X_test)
print("Predictions shape:", y_pred.shape)
```

## 11.5.3. Mercer's Theorem and Kernel Properties

### Mercer's Theorem

**Mercer's Theorem**: Let $`K(x, y)`$ be a continuous symmetric function on $`[a, b] \times [a, b]`$. If:
```math
\int_a^b \int_a^b K(x, y) f(x) f(y) dx dy \geq 0
```

for all $`f \in L^2[a, b]`$, then $`K`$ can be expanded as:
```math
K(x, y) = \sum_{i=1}^{\infty} \lambda_i \phi_i(x) \phi_i(y)
```

where $`\lambda_i \geq 0`$ and $`\{\phi_i\}`$ form an orthonormal basis.

### Implications for SVM

1. **Feature map**: $`\Phi(x) = (\sqrt{\lambda_1} \phi_1(x), \sqrt{\lambda_2} \phi_2(x), \ldots)`$
2. **Inner product**: $`K(x, y) = \langle \Phi(x), \Phi(y) \rangle`$
3. **Positive definiteness**: Kernel matrix is positive semi-definite

### Kernel Matrix Properties

```python
import numpy as np
from scipy.linalg import eigh

def check_kernel_properties(K):
    """Check if K satisfies kernel properties"""
    n = K.shape[0]
    
    # Check symmetry
    is_symmetric = np.allclose(K, K.T)
    print(f"Symmetric: {is_symmetric}")
    
    # Check positive semi-definiteness
    eigenvals = eigh(K, eigvals_only=True)
    is_psd = np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    print(f"Positive semi-definite: {is_psd}")
    print(f"Eigenvalues: {eigenvals[:5]}...")  # Show first 5
    
    # Check trace
    trace = np.trace(K)
    print(f"Trace: {trace:.3f}")
    
    return is_symmetric and is_psd

# Test different kernels
X = np.random.randn(50, 3)

# Linear kernel
K_linear = np.dot(X, X.T)
print("Linear kernel:")
check_kernel_properties(K_linear)

# RBF kernel
gamma = 1.0
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
K_rbf = np.exp(-gamma * dist_sq)
print("\nRBF kernel:")
check_kernel_properties(K_rbf)

# Polynomial kernel
K_poly = (np.dot(X, X.T) + 1)**2
print("\nPolynomial kernel:")
check_kernel_properties(K_poly)
```

## 11.5.4. Advanced SVM Topics

### Multi-Class SVM

#### One-vs-One (OVO)
Train $`\binom{K}{2}`$ binary classifiers and use voting:

```python
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

def ovo_svm_example():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate multi-class data
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                             n_clusters_per_class=1, n_redundant=0, 
                             random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Train OVO SVM
    ovo_svm = OneVsOneClassifier(SVC(kernel='rbf', random_state=42))
    ovo_svm.fit(X_train, y_train)
    
    # Evaluate
    train_score = ovo_svm.score(X_train, y_train)
    test_score = ovo_svm.score(X_test, y_test)
    
    print(f"OVO SVM - Train accuracy: {train_score:.3f}")
    print(f"OVO SVM - Test accuracy: {test_score:.3f}")
    
    return ovo_svm

ovo_svm_example()
```

#### One-vs-Rest (OVR)
Train $`K`$ binary classifiers:

```python
from sklearn.multiclass import OneVsRestClassifier

def ovr_svm_example():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate multi-class data
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                             n_clusters_per_class=1, n_redundant=0, 
                             random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Train OVR SVM
    ovr_svm = OneVsRestClassifier(SVC(kernel='rbf', random_state=42))
    ovr_svm.fit(X_train, y_train)
    
    # Evaluate
    train_score = ovr_svm.score(X_train, y_train)
    test_score = ovr_svm.score(X_test, y_test)
    
    print(f"OVR SVM - Train accuracy: {train_score:.3f}")
    print(f"OVR SVM - Test accuracy: {test_score:.3f}")
    
    return ovr_svm

ovr_svm_example()
```

### Support Vector Regression (SVR)

SVR extends SVM to regression problems:

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

def svr_example():
    # Generate regression data
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    
    # Fit SVR models with different kernels
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_linear = SVR(kernel='linear', C=100, epsilon=0.1)
    svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1)
    
    # Fit models
    svr_rbf.fit(X, y)
    svr_linear.fit(X, y)
    svr_poly.fit(X, y)
    
    # Predictions
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_rbf = svr_rbf.predict(X_test)
    y_linear = svr_linear.predict(X_test)
    y_poly = svr_poly.predict(X_test)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, c='black', label='data')
    plt.plot(X_test, y_rbf, c='red', label='RBF')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(X, y, c='black', label='data')
    plt.plot(X_test, y_linear, c='blue', label='Linear')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, c='black', label='data')
    plt.plot(X_test, y_poly, c='green', label='Polynomial')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

svr_example()
```

## 11.5.5. Computational Considerations

### Large-Scale SVM

For large datasets, standard SVM becomes computationally expensive. Solutions:

#### 1. Sequential Minimal Optimization (SMO)

```python
def simplified_smo(X, y, C=1.0, max_iter=1000, tol=1e-3):
    """Simplified SMO algorithm"""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0
    
    # Precompute kernel matrix
    K = np.dot(X, X.T)
    
    for iteration in range(max_iter):
        alpha_pairs_changed = 0
        
        for i in range(n_samples):
            # Calculate error
            Ei = np.sum(alpha * y * K[i, :]) + b - y[i]
            
            # Check KKT conditions
            if ((y[i] * Ei < -tol and alpha[i] < C) or 
                (y[i] * Ei > tol and alpha[i] > 0)):
                
                # Choose second alpha randomly
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                
                Ej = np.sum(alpha * y * K[j, :]) + b - y[j]
                
                # Save old alphas
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                
                # Compute bounds
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                # Compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                # Update alpha[j]
                alpha[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha[i]
                alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
                
                # Update b
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                b = (b1 + b2) / 2
                
                alpha_pairs_changed += 1
        
        if alpha_pairs_changed == 0:
            break
    
    return alpha, b

# Example usage
X = np.random.randn(100, 2)
y = np.sign(X[:, 0] + X[:, 1])

alpha, b = simplified_smo(X, y, C=1.0)
print(f"Converged with {len(alpha[alpha > 1e-5])} support vectors")
```

#### 2. Kernel Approximation

```python
from sklearn.kernel_approximation import RBFSampler, Nystroem

def kernel_approximation_example():
    from sklearn.datasets import make_circles
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Standard SVM
    svm_standard = SVC(kernel='rbf', gamma=1.0, random_state=42)
    svm_standard.fit(X_train, y_train)
    score_standard = svm_standard.score(X_test, y_test)
    
    # RBF approximation
    rbf_feature = RBFSampler(gamma=1.0, n_components=100, random_state=42)
    X_train_rbf = rbf_feature.fit_transform(X_train)
    X_test_rbf = rbf_feature.transform(X_test)
    
    svm_rbf_approx = SVC(kernel='linear', random_state=42)
    svm_rbf_approx.fit(X_train_rbf, y_train)
    score_rbf_approx = svm_rbf_approx.score(X_test_rbf, y_test)
    
    # Nystroem approximation
    nystroem = Nystroem(kernel='rbf', gamma=1.0, n_components=100, random_state=42)
    X_train_nystroem = nystroem.fit_transform(X_train)
    X_test_nystroem = nystroem.transform(X_test)
    
    svm_nystroem = SVC(kernel='linear', random_state=42)
    svm_nystroem.fit(X_train_nystroem, y_train)
    score_nystroem = svm_nystroem.score(X_test_nystroem, y_test)
    
    print(f"Standard SVM accuracy: {score_standard:.3f}")
    print(f"RBF approximation accuracy: {score_rbf_approx:.3f}")
    print(f"Nystroem approximation accuracy: {score_nystroem:.3f}")

kernel_approximation_example()
```

## 11.5.6. Theoretical Bounds and Generalization

### VC Dimension

The VC dimension of SVM with RBF kernel is infinite, but generalization is controlled by margin.

### Margin-Based Bounds

For SVM with margin $`\gamma`$ and $`R`$ as the radius of the data:

**Theorem**: With probability at least $`1 - \delta`$:
```math
R(f) \leq \hat{R}(f) + \sqrt{\frac{4}{\gamma^2} \log\left(\frac{2en}{\gamma}\right) + \log\left(\frac{4}{\delta}\right)}{n}}
```

where $`R(f)`$ is the true risk and $`\hat{R}(f)`$ is the empirical risk.

### Implementation of Margin Analysis

```python
def margin_analysis(X, y, svm_model):
    """Analyze margin and support vectors"""
    # Get support vectors
    support_vectors = svm_model.support_vectors_
    support_vector_indices = svm_model.support_
    
    # Compute margin
    w = svm_model.coef_[0]
    margin = 2 / np.linalg.norm(w)
    
    # Compute distances to decision boundary
    decision_values = svm_model.decision_function(X)
    distances = np.abs(decision_values) / np.linalg.norm(w)
    
    # Find minimum margin
    min_margin = np.min(distances)
    
    print(f"Margin: {margin:.4f}")
    print(f"Minimum margin: {min_margin:.4f}")
    print(f"Number of support vectors: {len(support_vectors)}")
    print(f"Support vector ratio: {len(support_vectors)/len(X):.3f}")
    
    return margin, min_margin, support_vectors

# Example usage
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = 2 * y - 1  # Convert to {-1, 1}

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X, y)

margin, min_margin, support_vectors = margin_analysis(X, y, svm)
```

## 11.5.7. Summary

This appendix covers advanced topics in Support Vector Machines:

1. **Mathematical Foundations**: Convex optimization, Lagrangian duality
2. **RKHS Theory**: Reproducing kernels, representer theorem
3. **Mercer's Theorem**: Kernel properties and feature maps
4. **Multi-class SVM**: OVO and OVR strategies
5. **Support Vector Regression**: Extension to regression problems
6. **Computational Methods**: SMO, kernel approximation
7. **Theoretical Bounds**: VC dimension, margin-based generalization

Key insights:
- **Duality**: Enables efficient optimization
- **Kernels**: Provide nonlinear capabilities
- **Margin**: Controls generalization
- **Support vectors**: Determine the solution
- **Computational efficiency**: Critical for large-scale applications

These concepts provide the theoretical foundation for understanding and implementing SVMs effectively.

## References

1. **lec_W11_appendix_SVM**: [SVM Mathematical Appendix](./lec_W11_appendix_SVM.pdf)
2. **lec_W11_appendix_RKHS**: [RKHS Theory Appendix](./lec_W11_appendix_RKHS.pdf)