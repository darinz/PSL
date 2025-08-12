# 3.4. Lasso Regression

## Introduction

Lasso (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani in 1996, is a powerful regularization technique that combines variable selection with coefficient shrinkage. Unlike ridge regression, lasso can produce exactly zero coefficients, making it particularly valuable for sparse modeling and automatic feature selection.

![Lasso Regression Solution Paths](../_images/w4_solution_path.png)

*Figure: Solution paths for lasso regression. Shows how coefficients change as the regularization parameter varies.*

## 3.4.1 Mathematical Foundation

### The Lasso Objective Function

Lasso regression modifies the standard least squares objective by adding an L1 penalty on the coefficient vector:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1
```

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the response vector
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector
- $\lambda \geq 0$ is the regularization parameter
- $`\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|`$ is the L1 norm

### Key Properties of the L1 Penalty

1. **Non-differentiability**: The L1 penalty is not differentiable at zero
2. **Sparsity**: Can produce exactly zero coefficients
3. **Convexity**: The L1 penalty is convex, making optimization tractable
4. **Scale sensitivity**: Unlike L2 penalty, L1 is sensitive to predictor scaling

### Orthogonal Design Matrix Case

When the design matrix $\mathbf{X}$ is orthogonal (i.e., $\mathbf{X}^T\mathbf{X} = \mathbf{I}_p$), the lasso problem can be decomposed into $p$ independent one-dimensional problems.

First, let's decompose the residual sum of squares:

```math
\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} + \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2
```

Using the Pythagorean theorem and orthogonality:

```math
\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}}\|^2_2 + \|\mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2
```

The cross-product term vanishes because the residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}}$ is orthogonal to the column space of $\mathbf{X}$.

Therefore, the lasso objective becomes:

```math
\begin{align*}
\hat{\boldsymbol{\beta}}_{\text{lasso}} &= \arg\min_{\boldsymbol{\beta}} \left[\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[\|\mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta})^T\mathbf{X}^T\mathbf{X}(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \sum_{j=1}^p \left[(\beta_j - \hat{\beta}_j^{\text{OLS}})^2 + \lambda|\beta_j|\right]
\end{align*}
```

This decomposition allows us to solve for each $\beta_j$ independently.

## 3.4.2 The Soft-Thresholding Operator

### One-Dimensional Lasso Problem

For each component, we need to solve:

```math
\min_{x} (x - a)^2 + \lambda|x|
```

where $a = \hat{\beta}_j^{\text{OLS}}$ and $x = \beta_j$.

### Subgradient Analysis

Since the absolute value function is not differentiable at zero, we use subgradients. The subgradient of $|x|$ at $x = 0$ is any value in $[-1, 1]$.

The optimality condition is:

```math
2(x^* - a) + \lambda z^* = 0
```

where $z^*$ is the subgradient of $|x|$ at $x^*$.

### Solution: Soft-Thresholding

The solution is given by the soft-thresholding operator:

```math
x^* = S_{\lambda/2}(a) = \text{sign}(a)(|a| - \lambda/2)_+ = \begin{cases}
a - \lambda/2, & \text{if } a > \lambda/2 \\
0, & \text{if } |a| \leq \lambda/2 \\
a + \lambda/2, & \text{if } a < -\lambda/2
\end{cases}
```

where $(x)_+ = \max(x, 0)$ is the positive part function.

### Component-Wise Lasso Solution

For orthogonal design matrices, the lasso solution is:

```math
\hat{\beta}_j^{\text{lasso}} = \begin{cases}
\text{sign}(\hat{\beta}_j^{\text{OLS}})(|\hat{\beta}_j^{\text{OLS}}| - \lambda/2), & \text{if } |\hat{\beta}_j^{\text{OLS}}| > \lambda/2 \\
0, & \text{if } |\hat{\beta}_j^{\text{OLS}}| \leq \lambda/2
\end{cases}
```

### Geometric Interpretation

The soft-thresholding operator can be understood geometrically:

1. **Shrinkage**: Coefficients are shrunk toward zero by $\lambda/2$
2. **Thresholding**: Coefficients smaller than $\lambda/2$ in magnitude are set to zero
3. **Sign preservation**: The sign of non-zero coefficients is preserved

![Lasso Duality and Geometry](../_images/w3_lasso_duality.png)

*Figure: Geometric interpretation of the lasso constraint and solution. The diamond-shaped constraint region leads to sparse solutions.*

## 3.4.3 Lasso vs Ridge: Geometric Comparison

### Constrained Optimization Formulation

Both lasso and ridge can be formulated as constrained optimization problems:

**Lasso:**
```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_1 \leq t
```

**Ridge:**
```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_2^2 \leq t
```

### Geometric Interpretation

The constraint regions have different shapes:

1. **L1 ball (Lasso)**: Diamond-shaped in 2D, octahedron in 3D
2. **L2 ball (Ridge)**: Circular in 2D, spherical in 3D

The solution occurs where the contours of the RSS (ellipsoids) touch the constraint boundary.

### Key Differences

| Property | Lasso | Ridge |
|----------|-------|-------|
| Constraint shape | Diamond (L1 ball) | Circle (L2 ball) |
| Corner solutions | Yes (sparse) | No (dense) |
| Variable selection | Automatic | Manual |
| Coefficient shrinkage | Non-linear | Linear |
| Multicollinearity | Groups one variable | Groups all variables |

### Thresholding Mechanisms

1. **Hard thresholding (Subset selection)**: Coefficients are either kept at OLS value or set to zero
2. **Soft thresholding (Lasso)**: Coefficients are shrunk toward zero, with some set to exactly zero
3. **Linear shrinkage (Ridge)**: All coefficients are shrunk proportionally, rarely reaching zero

## 3.4.4 Coordinate Descent Algorithm

### Algorithm Overview

For general design matrices, lasso doesn't have a closed-form solution. The coordinate descent algorithm updates one coefficient at a time while keeping others fixed.

### Algorithm Steps

1. **Initialize**: $\boldsymbol{\beta}^{(0)} = \mathbf{0}$
2. **For iteration $k = 1, 2, \ldots$**:
   - For $j = 1, 2, \ldots, p$:
     - Compute partial residual: $r_j = \mathbf{y} - \sum_{l \neq j} \mathbf{x}_l \beta_l^{(k)}$
     - Compute univariate OLS: $\tilde{\beta}_j = \mathbf{x}_j^T r_j / \|\mathbf{x}_j\|^2$
     - Apply soft-thresholding: $\beta_j^{(k+1)} = S_{\lambda/(2\|\mathbf{x}_j\|^2)}(\tilde{\beta}_j)$
3. **Convergence**: Stop when coefficients change by less than tolerance

### Mathematical Derivation

For coordinate $j$, the objective function becomes:

```math
\min_{\beta_j} \|\mathbf{r}_j - \mathbf{x}_j\beta_j\|^2_2 + \lambda|\beta_j|
```

where $\mathbf{r}_j = \mathbf{y} - \sum_{l \neq j} \mathbf{x}_l\beta_l$ is the partial residual.

The solution is:

```math
\beta_j = S_{\lambda/(2\|\mathbf{x}_j\|^2)}\left(\frac{\mathbf{x}_j^T\mathbf{r}_j}{\|\mathbf{x}_j\|^2}\right)
```

### Convergence Properties

1. **Monotonicity**: The objective function decreases at each iteration
2. **Convergence**: The algorithm converges to a global minimum
3. **Finite convergence**: For some problems, convergence occurs in finitely many steps

## 3.4.5 Uniqueness and Solution Properties

### Uniqueness Conditions

The lasso solution is unique when:

1. **Full-rank design matrix**: $\text{rank}(\mathbf{X}) = p$
2. **Sufficient observations**: $n \geq p$
3. **Strict convexity**: The objective function is strictly convex

### Non-uniqueness Scenarios

When $p > n$ or $\mathbf{X}$ is not full-rank:

1. **Multiple solutions**: Different coefficient vectors may give the same fitted values
2. **Unique fitted values**: The predicted values $\hat{\mathbf{y}}$ are always unique
3. **Unique L1 norm**: The L1 norm of the solution is always unique

### Solution Characterization

For any lasso solution $\hat{\boldsymbol{\beta}}$:

1. **Optimality conditions**: Must satisfy the subgradient conditions
2. **Support recovery**: The set of non-zero coefficients is well-defined
3. **Sign consistency**: The signs of non-zero coefficients are consistent across solutions

## 3.4.6 Practical Implementation

### Python Implementation

See the complete Python implementation in [`code/lasso_regression_detailed.py`](code/lasso_regression_detailed.py) which demonstrates comprehensive lasso regression with coordinate descent, soft thresholding, and variable selection analysis.

### R Implementation

See the complete R implementation in [`code/lasso_regression_detailed.R`](code/lasso_regression_detailed.R) which demonstrates comprehensive lasso regression with coordinate descent and variable selection using the glmnet package.

## 3.4.7 Advanced Topics

### Elastic Net

Elastic net combines L1 and L2 penalties:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|^2_2
```

This provides a compromise between lasso and ridge, offering both variable selection and group selection.

### Group Lasso

For grouped variables, group lasso uses:

```math
P(\boldsymbol{\beta}) = \sum_{g=1}^G \|\boldsymbol{\beta}_g\|_2
```

where $\boldsymbol{\beta}_g$ represents coefficients for group $g$.

### Adaptive Lasso

Adaptive lasso uses weighted L1 penalty:

```math
P(\boldsymbol{\beta}) = \sum_{j=1}^p w_j |\beta_j|
```

where weights $w_j$ are typically based on initial OLS estimates.

### Lasso for Classification

Lasso can be extended to classification using logistic regression with L1 penalty:

```math
\min_{\boldsymbol{\beta}} \sum_{i=1}^n \log(1 + e^{-y_i\mathbf{x}_i^T\boldsymbol{\beta}}) + \lambda\|\boldsymbol{\beta}\|_1
```

## 3.4.8 Model Selection and Validation

### Choosing the Regularization Parameter

1. **Cross-validation**: Most common approach
2. **Information criteria**: AIC, BIC with effective degrees of freedom
3. **Stability selection**: Assess variable selection stability
4. **Bayesian methods**: Empirical Bayes, hierarchical models

### Variable Selection Stability

Lasso's variable selection can be unstable. Stability selection addresses this by:

1. Running lasso on multiple subsamples
2. Computing selection frequencies
3. Selecting variables with high selection probability

### Model Diagnostics

1. **Residual analysis**: Check for model adequacy
2. **Influence diagnostics**: Identify influential observations
3. **Variable importance**: Assess coefficient stability
4. **Prediction intervals**: Quantify uncertainty

## 3.4.9 Practical Guidelines

### When to Use Lasso

**Use lasso when:**
- You want automatic variable selection
- The true model is sparse
- Interpretability is important
- You have many predictors relative to sample size
- You want a sparse model

**Consider alternatives when:**
- Predictors are highly correlated (use elastic net)
- You want to keep all variables (use ridge)
- You have grouped variables (use group lasso)
- The true model is dense

### Best Practices

1. **Always standardize predictors** before applying lasso
2. **Use cross-validation** to select the regularization parameter
3. **Check variable selection stability** across different samples
4. **Validate on a holdout set** to assess generalization
5. **Consider the bias-variance tradeoff** when interpreting results

### Common Pitfalls

1. **Not standardizing data**: Can lead to inconsistent results
2. **Over-regularization**: Choosing λ too large can remove important variables
3. **Under-regularization**: Choosing λ too small may not address overfitting
4. **Ignoring multicollinearity**: Can affect variable selection
5. **Not validating variable selection**: Can lead to spurious findings

## Summary

Lasso regression is a powerful regularization technique that combines variable selection with coefficient shrinkage. Its key features are:

1. **Sparsity**: Can produce exactly zero coefficients through soft thresholding
2. **Variable selection**: Automatic feature selection
3. **Convex optimization**: Computationally tractable
4. **Geometric interpretation**: L1 constraint leads to corner solutions
5. **Coordinate descent**: Efficient algorithm for general design matrices

Lasso is particularly valuable in high-dimensional settings where sparsity is expected, providing both prediction accuracy and interpretability through automatic variable selection.

---

**Navigation:**
- **Next Topic:** [Discussion and Comparison](05_discussion.md) - Comparing variable selection and regularization methods
- **Previous Topic:** [Ridge Regression](03_ridge_regression.md) - L2 regularization and coefficient shrinkage
