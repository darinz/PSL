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

The complete implementation of Linear Discriminant Analysis is provided in the following code files:

**Python Implementation:** [`code/lda_implementation.py`](code/lda_implementation.py)

**R Implementation:** [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

These files contain:

- Complete LDA class implementation with parameter estimation
- Regularized LDA for high-dimensional data
- Decision boundary visualization functions
- Model comparison utilities (custom vs. library implementations)
- Parameter analysis and diagnostics
- Cross-validation and model selection
- Real-world examples (Iris dataset, credit risk assessment)
- Dimensionality reduction capabilities

The Python implementation includes a custom `LinearDiscriminantAnalysisFromScratch` class that mirrors the scikit-learn API, while the R implementation provides both MASS package integration and custom functions for educational purposes.

## 9.4.5. Advanced Topics

### 9.4.5.1. Regularized LDA

For high-dimensional data, we can add regularization to the covariance estimation. The implementation is provided in the code files:

**Python:** See `regularized_lda()` function and `RegularizedLDA` class in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `regularized_lda()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

This approach applies a convex combination between the estimated covariance matrix and the identity matrix, controlled by a shrinkage parameter α.

### 9.4.5.2. Kernel LDA

For non-linear decision boundaries, we can apply the kernel trick. The implementation is provided in the code files:

**Python:** See `kernel_lda()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

This approach computes a kernel matrix and applies LDA in the kernel space, allowing for non-linear decision boundaries.

### 9.4.5.3. Multi-class LDA

For K > 2 classes, LDA finds K-1 discriminant directions. The implementation is provided in the code files:

**Python:** See `multiclass_lda()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `multiclass_lda()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

This naturally provides dimensionality reduction from p features to (K-1) discriminant components.

## 9.4.6. Model Evaluation and Diagnostics

### Performance Metrics

The comprehensive model evaluation functions are implemented in the code files:

**Python:** See `evaluate_lda_model()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `evaluate_lda_model()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

These functions provide:
- Accuracy, precision, recall, and F1-score
- Confusion matrix analysis
- ROC AUC for binary classification
- Multi-class evaluation metrics

### Model Diagnostics

The diagnostic functions are implemented in the code files:

**Python:** See `lda_diagnostics()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `lda_diagnostics()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

These functions provide:
- Q-Q plots for normality assumption checking
- Homoscedasticity analysis
- Feature importance visualization
- Residual analysis

## 9.4.7. Real-World Applications

### Example 1: Iris Dataset

The Iris dataset example is implemented in the code files:

**Python:** See `iris_lda_example()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `iris_lda_example()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

This example demonstrates LDA on the classic Iris dataset, including cross-validation and dimensionality reduction from 4 features to 2 discriminant components.

### Example 2: Credit Risk Classification

The credit risk assessment example is implemented in the code files:

**Python:** See `credit_risk_lda()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `credit_risk_lda()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

This example creates synthetic credit data with features like income, debt, credit score, and age, demonstrating how LDA can be used for risk assessment with feature importance analysis.

## 9.4.8. Risk of Overfitting

### Understanding the Overfitting Problem

When $`p \gg K`$ (high-dimensional data with few classes), LDA can overfit because:

1. **Limited Degrees of Freedom**: The pooled covariance matrix has limited degrees of freedom
2. **Curse of Dimensionality**: In high dimensions, the "empty space" phenomenon makes distance measures less reliable
3. **Sample Size Requirements**: Need sufficient samples per class for reliable covariance estimation

### Mitigation Strategies

#### 1. Regularization

The cross-validated regularized LDA implementation is provided in the code files:

**Python:** See `regularized_lda_cv()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

This function performs grid search over regularization parameters using cross-validation to find the optimal shrinkage parameter.

#### 2. Feature Selection

The LDA with feature selection implementation is provided in the code files:

**Python:** See `lda_with_feature_selection()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

This function uses ANOVA F-test to select the most discriminative features before applying LDA.

#### 3. Cross-Validation

The robust LDA evaluation implementation is provided in the code files:

**Python:** See `robust_lda_evaluation()` function in [`code/lda_implementation.py`](code/lda_implementation.py)

**R:** See `robust_lda_evaluation()` function in [`code/r_lda_implementation.R`](code/r_lda_implementation.R)

These functions provide stratified k-fold cross-validation for reliable performance estimation.

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
