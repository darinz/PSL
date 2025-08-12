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

The complete implementation of Quadratic Discriminant Analysis is provided in the following code files:

**Python Implementation:** [`code/qda_implementation.py`](code/qda_implementation.py)

**R Implementation:** [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These files contain:

- Complete QDA class implementation with parameter estimation
- Regularized QDA for high-dimensional data
- Decision boundary visualization functions
- Model comparison utilities (QDA vs LDA)
- Parameter analysis and diagnostics
- Cross-validation and model selection
- Real-world examples (Iris dataset, credit risk assessment)

The Python implementation includes a custom `QuadraticDiscriminantAnalysis` class that mirrors the scikit-learn API, while the R implementation provides both MASS package integration and custom functions for educational purposes.

## 9.3.5. Decision Boundaries and Visualization

### Understanding QDA Decision Boundaries

QDA produces quadratic decision boundaries because the discriminant function $`d_k(x)`$ is quadratic in $`x`$. For two classes, the decision boundary is where $`d_1(x) = d_2(x)`$.

The visualization functions for decision boundaries are implemented in the code files:

**Python:** See `plot_qda_decision_boundaries()` and `compare_qda_lda_boundaries()` functions in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `plot_qda_decision_boundaries()` and `compare_qda_lda_boundaries()` functions in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions create mesh grids over the feature space, compute predictions for each grid point, and visualize both decision boundaries and posterior probabilities. The comparison function demonstrates the key differences between QDA's quadratic boundaries and LDA's linear boundaries.

## 9.3.6. Model Analysis and Diagnostics

### Parameter Analysis

The parameter analysis functions are implemented in the code files:

**Python:** See `analyze_qda_parameters()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `analyze_qda_parameters()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions provide comprehensive analysis of QDA model parameters including:
- Class prior probabilities
- Class mean vectors
- Covariance matrices with heatmap visualizations
- Log determinants for each class

### Mahalanobis Distance Analysis

The Mahalanobis distance analysis functions are implemented in the code files:

**Python:** See `analyze_mahalanobis_distances()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `analyze_mahalanobis_distances()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions compute and visualize the distribution of Mahalanobis distances for each class, comparing them with the theoretical chi-squared distribution to assess model fit and identify potential outliers.

## 9.3.7. High-Dimensional QDA

### Challenges in High Dimensions

When the number of features $`p`$ is large relative to the sample size, QDA faces several challenges:

1. **Curse of Dimensionality**: Need $`O(p^2)`$ parameters per class
2. **Singular Covariance**: Covariance matrices become singular
3. **Overfitting**: Model complexity increases with $`p^2``

### Regularization Techniques

The regularized QDA implementation and high-dimensional testing functions are provided in the code files:

**Python:** See `RegularizedQDA` class and `test_high_dimensional_qda()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `test_high_dimensional_qda()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

The `RegularizedQDA` class implements three regularization strategies:
- **Diagonal regularization**: Forces covariance matrices to be diagonal
- **Shrinkage regularization**: Shrinks covariance matrices toward a target (scaled identity)
- **Ridge regularization**: Adds a small constant to the diagonal for numerical stability

The high-dimensional testing function generates synthetic data with sparse covariance structures and compares the performance of different regularization approaches.

## 9.3.8. Model Selection and Validation

### Cross-Validation for QDA

The cross-validation and model selection functions are implemented in the code files:

**Python:** See `qda_cross_validation()` and `qda_grid_search()` functions in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_cross_validation()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions provide:
- **Cross-validation**: Evaluates QDA performance using k-fold cross-validation
- **Grid search**: Finds optimal regularization parameters using cross-validation
- **Model selection**: Compares different QDA configurations systematically

## 9.3.9. Real-World Applications

### Example: Iris Dataset

The Iris dataset example is implemented in the code files:

**Python:** See `qda_iris_example()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_iris_example()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

This example demonstrates QDA on the classic Iris dataset, using only two classes for binary classification. It includes data preprocessing, model fitting, evaluation with confusion matrices, and visualization of results.

### Example: Credit Risk Assessment

The credit risk assessment example is implemented in the code files:

**Python:** See `qda_credit_risk_example()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_credit_risk_example()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

This example creates synthetic credit data with features like income, credit score, debt ratio, and employment years. It demonstrates how QDA can be used for risk assessment, including feature analysis and visualization of class-specific parameter differences.

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of Quadratic Discriminant Analysis. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and handling of common challenges in high-dimensional settings.

---

**Navigation:**
- **Next Topic:** [Linear Discriminant Analysis](04_linear_discriminant_analysis.md) - Shared covariance assumption and linear decision boundaries
- **Previous Topic:** [Discriminant Analysis](02_discriminant_analysis.md) - Bayes' theorem application and joint distribution factorization
