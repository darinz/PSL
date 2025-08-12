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

Linear SVMs can only create linear decision boundaries. However, many real-world classification problems require nonlinear decision boundaries. Consider the classic XOR problem, which demonstrates the fundamental limitation of linear classifiers.

The XOR problem shows that some data patterns cannot be separated by a linear boundary, motivating the need for nonlinear methods. The implementation demonstrates this concept using the `generate_xor_data()` function in both Python and R code files.

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

The implementation and demonstration of nonlinear SVM concepts is provided in separate code files for both Python and R. These files contain comprehensive examples covering all the theoretical concepts discussed above.

### Python Implementation

The complete Python implementation is available in the file `code/nonlinear_svms_implementation.py`. This file includes:

- **KernelSVM class from scratch** using quadratic programming with cvxopt
- **Data generation functions** for different types of nonlinear data (circles, moons, XOR)
- **Kernel function implementations** for linear, polynomial, RBF, and sigmoid kernels
- **Decision boundary visualization** with support vector highlighting
- **Kernel comparison demonstrations** showing different kernel performances
- **Parameter effects analysis** showing how γ affects RBF kernel behavior
- **Cross-validation for kernel selection** using GridSearchCV
- **Representer theorem demonstration** showing finite representation
- **Advantages and limitations analysis** with practical demonstrations
- **Comprehensive demonstrations** of all nonlinear SVM concepts

To run the Python demonstrations:

```python
# Import and run the main demonstration
from code.nonlinear_svms_implementation import main
results = main()
```

### R Implementation

The complete R implementation is available in the file `code/r_nonlinear_svms_implementation.R`. This file includes:

- **Data generation functions** for nonlinear data patterns
- **SVM fitting and visualization** using e1071 package with different kernels
- **Kernel function demonstrations** showing mathematical properties
- **Parameter effects analysis** across different γ values
- **Cross-validation for kernel selection** using tune function
- **Representer theorem verification** showing finite representation
- **Advantages and limitations analysis** with practical demonstrations
- **Kernel performance comparison** across different data types
- **Comprehensive demonstrations** of all nonlinear SVM concepts

To run the R demonstrations:

```r
# Source and run the main demonstration
source("code/r_nonlinear_svms_implementation.R")
results <- main_r()
```

### Key Demonstrations

Both implementations provide comprehensive demonstrations of:

1. **Kernel Comparison**: Shows how different kernels handle nonlinear data
2. **Kernel Functions**: Demonstrates mathematical properties of different kernels
3. **Parameter Effects**: Illustrates how γ controls kernel behavior
4. **Cross-Validation**: Demonstrates systematic kernel and parameter selection
5. **Representer Theorem**: Shows finite representation using support vectors
6. **Advantages and Limitations**: Compares kernel performance across data types
7. **XOR Problem**: Demonstrates the need for nonlinear classification
8. **Practical Considerations**: Shows when to use different kernels

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

Cross-validation is essential for selecting the optimal kernel and parameters. The implementation demonstrates systematic parameter selection using grid search with cross-validation.

The cross-validation implementation is available in both Python and R code files:

**Python**: The `demonstrate_cross_validation()` function in `code/nonlinear_svms_implementation.py` shows:
- Grid search with `GridSearchCV` from scikit-learn
- Systematic exploration of parameter space for different kernels
- Cross-validation accuracy plotting
- Best kernel and parameter identification
- Performance comparison across kernel types

**R**: The `demonstrate_cross_validation()` function in `code/r_nonlinear_svms_implementation.R` shows:
- Grid search with `tune()` function from e1071
- Cross-validation error analysis for different kernels
- Parameter space exploration
- Best model selection and comparison

Both implementations demonstrate how to systematically find the optimal kernel and parameters that provide the best classification performance for the given dataset.

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
