# 3.3. Ridge Regression

## Introduction

Ridge regression, introduced by Hoerl and Kennard in 1970, is one of the most fundamental regularization techniques in statistical learning. It addresses the bias-variance tradeoff by introducing an L2 penalty on the regression coefficients, leading to more stable and often more accurate predictions than ordinary least squares (OLS).

![Ridge Regression Solution Paths](../_images/w4_solution_path.png)

*Figure: Solution paths for ridge regression. Shows how coefficients change as the regularization parameter varies.*

## 3.3.1 Mathematical Foundation

### The Ridge Regression Objective Function

Ridge regression modifies the standard least squares objective by adding a penalty term proportional to the squared L2 norm of the coefficient vector:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|^2_2
```

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the response vector
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector
- $\lambda \geq 0$ is the regularization parameter

### Derivation of the Ridge Estimator

To find the ridge estimator, we take the derivative of the objective function with respect to $\boldsymbol{\beta}$ and set it to zero:

```math
\frac{\partial}{\partial \boldsymbol{\beta}} \left[\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|^2_2\right] = 0
```

This gives us:

```math
-2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta} = 0
```

Rearranging:

```math
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} + \lambda\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
```

```math
(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
```

Therefore, the ridge estimator is:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

### The Augmented Data Interpretation

An elegant interpretation of ridge regression is through the concept of augmented data. We can view ridge regression as ordinary least squares applied to an augmented dataset.

Consider the augmented response vector and design matrix:

```math
\tilde{\mathbf{y}} = \begin{pmatrix} \mathbf{y} \\ \mathbf{0}_p \end{pmatrix}, \quad \tilde{\mathbf{X}} = \begin{pmatrix} \mathbf{X} \\ \sqrt{\lambda}\mathbf{I}_p \end{pmatrix}
```

The augmented model becomes:

```math
\tilde{\mathbf{y}} = \tilde{\mathbf{X}}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
```

The residual sum of squares for this augmented model is:

```math
\|\tilde{\mathbf{y}} - \tilde{\mathbf{X}}\boldsymbol{\beta}\|^2_2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda\|\boldsymbol{\beta}\|^2_2
```

This is exactly the ridge regression objective function! The OLS solution for the augmented model is:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\tilde{\mathbf{X}}^T\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{y}}
```

Computing the components:

```math
\tilde{\mathbf{X}}^T\tilde{\mathbf{X}} = \begin{pmatrix} \mathbf{X}^T & \sqrt{\lambda}\mathbf{I}_p \end{pmatrix} \begin{pmatrix} \mathbf{X} \\ \sqrt{\lambda}\mathbf{I}_p \end{pmatrix} = \mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}
```

```math
\tilde{\mathbf{X}}^T\tilde{\mathbf{y}} = \begin{pmatrix} \mathbf{X}^T & \sqrt{\lambda}\mathbf{I}_p \end{pmatrix} \begin{pmatrix} \mathbf{y} \\ \mathbf{0}_p \end{pmatrix} = \mathbf{X}^T\mathbf{y}
```

Therefore:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

### Key Properties of the Ridge Estimator

1. **Existence**: The ridge estimator always exists, even when $\mathbf{X}^T\mathbf{X}$ is singular
2. **Uniqueness**: The solution is unique for any $\lambda > 0$
3. **Continuity**: The estimator is continuous in $\lambda$
4. **Limiting behavior**: 
   - As $\lambda \to 0$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$
   - As $\lambda \to \infty$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$

## 3.3.2 The Shrinkage Effect

### Orthogonal Design Matrix Case

To understand the shrinkage effect, let's first consider the special case where the design matrix $\mathbf{X}$ has orthonormal columns (i.e., $\mathbf{X}^T\mathbf{X} = \mathbf{I}$).

In this case:

```math
\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{X}^T\mathbf{y}
```

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} = \frac{1}{1 + \lambda}\mathbf{X}^T\mathbf{y} = \frac{1}{1 + \lambda}\hat{\boldsymbol{\beta}}_{\text{OLS}}
```

The ridge estimator is a scaled version of the OLS estimator, with scaling factor $\frac{1}{1 + \lambda} < 1$ for $\lambda > 0$.

For predictions:

```math
\hat{\mathbf{y}}_{\text{OLS}} = \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{OLS}}
```

```math
\hat{\mathbf{y}}_{\text{ridge}} = \mathbf{X}\hat{\boldsymbol{\beta}}_{\text{ridge}} = \frac{1}{1 + \lambda}\mathbf{X}\hat{\boldsymbol{\beta}}_{\text{OLS}} = \frac{1}{1 + \lambda}\hat{\mathbf{y}}_{\text{OLS}}
```

### General Case: Singular Value Decomposition

For the general case, we use the singular value decomposition (SVD) of $\mathbf{X}$:

```math
\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T
```

where:
- $\mathbf{U} \in \mathbb{R}^{n \times p}$ has orthonormal columns
- $\mathbf{D} \in \mathbb{R}^{p \times p}$ is diagonal with singular values $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$
- $\mathbf{V} \in \mathbb{R}^{p \times p}$ is orthogonal

The OLS estimator can be written as:

```math
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^T\mathbf{y}
```

The ridge estimator becomes:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \mathbf{V}(\mathbf{D}^2 + \lambda\mathbf{I})^{-1}\mathbf{D}\mathbf{U}^T\mathbf{y}
```

In terms of the principal components, let $\boldsymbol{\alpha} = \mathbf{V}^T\boldsymbol{\beta}$. Then:

```math
\hat{\boldsymbol{\alpha}}_{\text{OLS}} = \mathbf{D}^{-1}\mathbf{U}^T\mathbf{y}, \quad \hat{\alpha}_j^{\text{OLS}} = \frac{1}{d_j}\mathbf{u}_j^T\mathbf{y}
```

```math
\hat{\boldsymbol{\alpha}}_{\text{ridge}} = \frac{d_j}{d_j^2 + \lambda}\mathbf{U}^T\mathbf{y}, \quad \hat{\alpha}_j^{\text{ridge}} = \frac{d_j^2}{d_j^2 + \lambda}\hat{\alpha}_j^{\text{OLS}}
```

The shrinkage factor for the $j$-th component is $\frac{d_j^2}{d_j^2 + \lambda}$:
- Components with large singular values (strong signal) are shrunk less
- Components with small singular values (weak signal or noise) are shrunk more

### Geometric Interpretation

The shrinkage effect can be understood geometrically:

1. **OLS**: Minimizes the distance from $\mathbf{y}$ to the column space of $\mathbf{X}$
2. **Ridge**: Minimizes this distance while also penalizing the norm of $\boldsymbol{\beta}$

The ridge solution is the projection of $\mathbf{y}$ onto a shrunken version of the column space, where the shrinkage is more pronounced in directions corresponding to small singular values.

## 3.3.3 Why Shrinkage Works: Bias-Variance Tradeoff

### Theoretical Motivation

While the OLS estimator is unbiased, it may have high variance, especially when:
- The number of predictors is large relative to sample size
- Predictors are highly correlated (multicollinearity)
- The design matrix is ill-conditioned

Ridge regression introduces bias but reduces variance, potentially leading to lower mean squared error (MSE).

### Simple Example: One-Dimensional Estimation

Consider estimating a parameter $\theta$ from $Z_1, \ldots, Z_n \sim N(\theta, \sigma^2)$.

The sample mean $\bar{Z}$ is unbiased with variance $\sigma^2/n$.

Consider the shrunken estimator $\frac{1}{2}\bar{Z}$:

```math
\text{Bias}\left(\frac{1}{2}\bar{Z}\right) = \mathbb{E}\left(\frac{1}{2}\bar{Z}\right) - \theta = \frac{\theta}{2} - \theta = -\frac{\theta}{2}
```

```math
\text{Var}\left(\frac{1}{2}\bar{Z}\right) = \frac{1}{4}\text{Var}(\bar{Z}) = \frac{\sigma^2}{4n}
```

The MSE is:

```math
\text{MSE}\left(\frac{1}{2}\bar{Z}\right) = \text{Bias}^2 + \text{Var} = \frac{\theta^2}{4} + \frac{\sigma^2}{4n}
```

Comparing with the MSE of $\bar{Z}$:

```math
\text{MSE}(\bar{Z}) = \frac{\sigma^2}{n}
```

The shrunken estimator has lower MSE when:

```math
\frac{\theta^2}{4} + \frac{\sigma^2}{4n} < \frac{\sigma^2}{n}
```

```math
\theta^2 < \frac{3\sigma^2}{n}
```

This demonstrates that shrinkage can be beneficial when the true parameter is small relative to the noise level.

### Ridge Regression MSE Analysis

For ridge regression, the bias and variance are:

```math
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
```

```math
\text{Var}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}
```

The total MSE is the sum of squared bias and trace of variance:

```math
\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \|\text{Bias}\|^2 + \text{tr}(\text{Var})
```

## 3.3.4 Degrees of Freedom

### Definition and Motivation

The degrees of freedom (df) of a statistical method measures its effective complexity. For linear methods that produce fitted values $\hat{\mathbf{y}} = \mathbf{A}\mathbf{y}$, the degrees of freedom is defined as:

```math
\text{df} = \text{tr}(\mathbf{A})
```

This definition has several interpretations:
1. **Variance inflation**: Measures how much the method inflates the variance of predictions
2. **Model complexity**: Represents the effective number of parameters
3. **Optimism**: Quantifies the optimism in in-sample performance

### Degrees of Freedom for Ridge Regression

For ridge regression, the fitted values are:

```math
\hat{\mathbf{y}}_{\text{ridge}} = \mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{S}_\lambda\mathbf{y}
```

where $\mathbf{S}_\lambda = \mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T$ is the ridge smoother matrix.

Using the SVD decomposition:

```math
\mathbf{S}_\lambda = \mathbf{U}\mathbf{D}(\mathbf{D}^2 + \lambda\mathbf{I})^{-1}\mathbf{D}\mathbf{U}^T = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}\mathbf{u}_j\mathbf{u}_j^T
```

The degrees of freedom is:

```math
\text{df}(\lambda) = \text{tr}(\mathbf{S}_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
```

### Properties of Ridge Degrees of Freedom

1. **Range**: $0 < \text{df}(\lambda) < p$
2. **Monotonicity**: $\text{df}(\lambda)$ decreases as $\lambda$ increases
3. **Limiting behavior**:
   - $\lambda \to 0$: $\text{df}(\lambda) \to p$ (full complexity)
   - $\lambda \to \infty$: $\text{df}(\lambda) \to 0$ (no complexity)
4. **Fractional values**: Unlike subset selection, ridge can have fractional degrees of freedom

## 3.3.5 Practical Implementation

### Python Implementation

See the complete Python implementation in [`code/ridge_regression_detailed.py`](code/ridge_regression_detailed.py) which demonstrates comprehensive ridge regression with multicollinearity handling, SVD analysis, and augmented data interpretation.

### R Implementation

See the complete R implementation in [`code/ridge_regression_detailed.R`](code/ridge_regression_detailed.R) which demonstrates comprehensive ridge regression with multicollinearity handling using the glmnet package.

## 3.3.6 Advanced Topics

### Bayesian Interpretation

Ridge regression can be interpreted as a Bayesian estimator with a Gaussian prior:

```math
\boldsymbol{\beta} \sim N(0, \tau^2\mathbf{I})
```

The posterior mean is:

```math
\mathbb{E}[\boldsymbol{\beta}|\mathbf{y}] = (\mathbf{X}^T\mathbf{X} + \sigma^2/\tau^2\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

This is equivalent to ridge regression with $\lambda = \sigma^2/\tau^2$.

### Ridge Regression with Different Penalties

Generalized ridge regression allows different penalties for different coefficients:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \boldsymbol{\beta}^T\mathbf{D}\boldsymbol{\beta}
```

where $\mathbf{D}$ is a diagonal matrix with penalty weights.

### Ridge Regression for Classification

Ridge regression can be extended to classification problems using logistic regression with L2 penalty:

```math
\min_{\boldsymbol{\beta}} \sum_{i=1}^n \log(1 + e^{-y_i\mathbf{x}_i^T\boldsymbol{\beta}}) + \lambda\|\boldsymbol{\beta}\|^2_2
```

## 3.3.7 Model Selection and Validation

### Choosing the Regularization Parameter

1. **Cross-validation**: Most common approach
2. **Generalized cross-validation**: Approximates leave-one-out CV
3. **Information criteria**: AIC, BIC with effective degrees of freedom
4. **Bayesian methods**: Empirical Bayes, hierarchical models

### Generalized Cross-Validation

GCV provides an efficient approximation to leave-one-out cross-validation:

```math
\text{GCV}(\lambda) = \frac{\|\mathbf{y} - \hat{\mathbf{y}}_{\text{ridge}}\|^2_2}{[n - \text{df}(\lambda)]^2}
```

### Model Diagnostics

1. **Residual analysis**: Check for model adequacy
2. **Influence diagnostics**: Identify influential observations
3. **Multicollinearity**: Assess correlation structure
4. **Prediction intervals**: Quantify uncertainty

## 3.3.8 Practical Guidelines

### When to Use Ridge Regression

**Use ridge regression when:**
- You have many predictors relative to sample size
- Predictors are highly correlated
- The design matrix is ill-conditioned
- You want to keep all variables in the model
- Primary goal is prediction accuracy

**Consider alternatives when:**
- You want automatic variable selection (use lasso)
- You have domain knowledge about variable importance
- The true model is sparse
- Interpretability is crucial

### Best Practices

1. **Always standardize predictors** before applying ridge regression
2. **Use cross-validation** to select the regularization parameter
3. **Check for influential observations** that might affect the solution
4. **Validate assumptions** about the error distribution
5. **Consider the bias-variance tradeoff** when interpreting results

### Common Pitfalls

1. **Not standardizing data**: Can lead to inconsistent results
2. **Over-regularization**: Choosing λ too large can introduce excessive bias
3. **Under-regularization**: Choosing λ too small may not address overfitting
4. **Ignoring multicollinearity**: Can affect coefficient interpretation
5. **Not validating on holdout set**: Can lead to overoptimistic performance estimates

## Summary

Ridge regression is a powerful regularization technique that addresses the bias-variance tradeoff through L2 penalization. It provides stable coefficient estimates, handles multicollinearity effectively, and often improves prediction accuracy compared to ordinary least squares. The key insights are:

1. **Shrinkage**: Ridge shrinks coefficients toward zero, with more shrinkage for directions with smaller singular values
2. **Bias-variance tradeoff**: Introduces bias but reduces variance, potentially lowering MSE
3. **Degrees of freedom**: Provides a continuous measure of model complexity
4. **Geometric interpretation**: Can be viewed as projection onto a shrunken subspace
5. **Bayesian interpretation**: Equivalent to maximum a posteriori estimation with Gaussian prior

Proper implementation requires careful attention to data preprocessing, parameter selection, and model validation. Ridge regression is particularly valuable in high-dimensional settings with correlated predictors.