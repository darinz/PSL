# 3.2. Regularization

## Introduction

Regularization is a fundamental concept in statistical learning that addresses the bias-variance tradeoff by introducing penalty terms to the objective function. In this comprehensive lecture, we'll explore the theoretical foundations, mathematical formulations, and practical implementations of regularization methods.

## 3.2.1 The Regularization Framework

### Motivation and Problem Setup

Regularization emerges from the fundamental challenge in statistical learning: balancing model complexity with generalization performance. When we have many predictors relative to the sample size, or when predictors are highly correlated, the standard least squares estimator can suffer from:

1. **High variance**: Small changes in data lead to large changes in coefficient estimates
2. **Overfitting**: The model captures noise rather than true signal
3. **Poor generalization**: Good in-sample performance but poor out-of-sample prediction

### Mathematical Foundation

Consider the standard linear regression model:

```math
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
```

where:
- $\mathbf{y} \in \mathbb{R}^n$ is the response vector
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the design matrix
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the coefficient vector
- $\boldsymbol{\varepsilon} \sim N(0, \sigma^2\mathbf{I})$ is the error vector

The ordinary least squares (OLS) estimator minimizes:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2
```

### The Regularization Objective Function

Regularization introduces a penalty term to control model complexity:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \cdot P(\boldsymbol{\beta})
```

where:
- $\lambda \geq 0$ is the regularization parameter (controls penalty strength)
- $P(\boldsymbol{\beta})$ is the penalty function that encodes our prior beliefs about the coefficient structure

## 3.2.2 L0 Regularization: Subset Selection Revisited

### Mathematical Formulation

The L0 penalty counts the number of non-zero coefficients:

```math
P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_0 = \sum_{j=1}^p \mathbf{1}_{\{\beta_j \neq 0\}}
```

This leads to the objective function:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_0
```

### Connection to Information Criteria

The L0 penalty is closely related to information criteria like AIC and BIC:

**AIC (Akaike Information Criterion):**
```math
\text{AIC} = n\log(\text{RSS}/n) + 2p
```

**BIC (Bayesian Information Criterion):**
```math
\text{BIC} = n\log(\text{RSS}/n) + \log(n)p
```

where RSS is the residual sum of squares.

### Computational Challenges

The L0 penalty creates a non-convex optimization problem that is NP-hard. The solution requires exploring all $2^p$ possible subsets, which becomes computationally infeasible for large $p$.

## 3.2.3 L2 Regularization: Ridge Regression

### Mathematical Formulation

Ridge regression uses the L2 penalty:

```math
P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|^2_2 = \sum_{j=1}^p \beta_j^2
```

The objective function becomes:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|^2_2
```

### Closed-Form Solution

The ridge estimator has a closed-form solution:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
```

### Geometric Interpretation

Ridge regression can be interpreted as:
1. **Bayesian prior**: Assuming $\boldsymbol{\beta} \sim N(0, \tau^2\mathbf{I})$ where $\lambda = \sigma^2/\tau^2$
2. **Constrained optimization**: Minimizing RSS subject to $\|\boldsymbol{\beta}\|^2_2 \leq t$
3. **Shrinkage**: Pulling coefficients toward zero

### Bias-Variance Tradeoff

The ridge estimator introduces bias but reduces variance:

```math
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
```

```math
\text{Var}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}
```

## 3.2.4 L1 Regularization: Lasso Regression

### Mathematical Formulation

Lasso regression uses the L1 penalty:

```math
P(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|
```

The objective function becomes:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1
```

### Key Properties

1. **Sparsity**: Lasso can produce exactly zero coefficients, performing automatic variable selection
2. **Convexity**: The L1 penalty is convex, making optimization tractable
3. **Non-differentiability**: The L1 penalty is not differentiable at zero

### Geometric Interpretation

Lasso can be viewed as constrained optimization:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_1 \leq t
```

The L1 constraint creates a diamond-shaped feasible region that can intersect the contours of the RSS at corners, leading to sparse solutions.

### Soft Thresholding

For orthogonal design matrices, the lasso solution has a simple form:

```math
\hat{\beta}_j = S(\hat{\beta}_j^{\text{OLS}}, \lambda) = \text{sign}(\hat{\beta}_j^{\text{OLS}}) \cdot \max(|\hat{\beta}_j^{\text{OLS}}| - \lambda, 0)
```

where $S$ is the soft thresholding operator.

## 3.2.5 Data Preprocessing and Standardization

### The Scaling Problem

Regularization methods are sensitive to the scale of predictors. Consider two scenarios:

1. **Price in dollars vs thousands of dollars**: $X_1 = 1000X_1'$
2. **Location shifts**: $X_1 = X_1' + c$

These transformations can dramatically affect coefficient estimates and model performance.

### Standardization Solution

To ensure consistent results, we standardize the data:

**For predictors:**
```math
\tilde{X}_{ij} = \frac{X_{ij} - \bar{X}_j}{s_j}
```

where:
- $\bar{X}_j = \frac{1}{n}\sum_{i=1}^n X_{ij}$ is the sample mean
- $s_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_{ij} - \bar{X}_j)^2}$ is the sample standard deviation

**For response:**
```math
\tilde{y}_i = \frac{y_i - \bar{y}}{s_y}
```

### Coefficient Transformation

After fitting the model on standardized data, we transform coefficients back to the original scale:

```math
\hat{\beta}_j^{\text{original}} = \hat{\beta}_j^{\text{standardized}} \cdot \frac{s_y}{s_j}
```

```math
\hat{\beta}_0^{\text{original}} = \bar{y} - \sum_{j=1}^p \hat{\beta}_j^{\text{original}} \bar{X}_j
```

## 3.2.6 Practical Implementation

### Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [2, -1.5, 1, -0.8, 0.6]  # Only first 5 coefficients are non-zero
y = X @ true_beta + 0.5 * np.random.randn(n)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Fit models
lambda_values = np.logspace(-3, 3, 50)

# Ridge regression
ridge_scores = []
ridge_coefs = []

for alpha in lambda_values:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    ridge_scores.append(-scores.mean())
    
    ridge.fit(X_train_scaled, y_train_scaled)
    ridge_coefs.append(ridge.coef_)

# Lasso regression
lasso_scores = []
lasso_coefs = []

for alpha in lambda_values:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    scores = cross_val_score(lasso, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    lasso_scores.append(-scores.mean())
    
    lasso.fit(X_train_scaled, y_train_scaled)
    lasso_coefs.append(lasso.coef_)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Cross-validation scores
axes[0, 0].semilogx(lambda_values, ridge_scores, 'b-', label='Ridge')
axes[0, 0].semilogx(lambda_values, lasso_scores, 'r-', label='Lasso')
axes[0, 0].set_xlabel('Regularization Parameter (λ)')
axes[0, 0].set_ylabel('Cross-Validation MSE')
axes[0, 0].set_title('Cross-Validation Performance')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Coefficient paths
ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

axes[0, 1].semilogx(lambda_values, ridge_coefs)
axes[0, 1].set_xlabel('Regularization Parameter (λ)')
axes[0, 1].set_ylabel('Coefficient Values')
axes[0, 1].set_title('Ridge: Coefficient Paths')
axes[0, 1].grid(True)

axes[1, 0].semilogx(lambda_values, lasso_coefs)
axes[1, 0].set_xlabel('Regularization Parameter (λ)')
axes[1, 0].set_ylabel('Coefficient Values')
axes[1, 0].set_title('Lasso: Coefficient Paths')
axes[1, 0].grid(True)

# Sparsity comparison
ridge_nonzero = np.sum(ridge_coefs != 0, axis=1)
lasso_nonzero = np.sum(lasso_coefs != 0, axis=1)

axes[1, 1].semilogx(lambda_values, ridge_nonzero, 'b-', label='Ridge')
axes[1, 1].semilogx(lambda_values, lasso_nonzero, 'r-', label='Lasso')
axes[1, 1].set_xlabel('Regularization Parameter (λ)')
axes[1, 1].set_ylabel('Number of Non-zero Coefficients')
axes[1, 1].set_title('Sparsity Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Optimal lambda selection
best_ridge_idx = np.argmin(ridge_scores)
best_lasso_idx = np.argmin(lasso_scores)

print(f"Best Ridge λ: {lambda_values[best_ridge_idx]:.4f}")
print(f"Best Lasso λ: {lambda_values[best_lasso_idx]:.4f}")

# Final model evaluation
best_ridge = Ridge(alpha=lambda_values[best_ridge_idx])
best_lasso = Lasso(alpha=lambda_values[best_lasso_idx])

best_ridge.fit(X_train_scaled, y_train_scaled)
best_lasso.fit(X_train_scaled, y_train_scaled)

# Transform coefficients back to original scale
ridge_coef_original = best_ridge.coef_ * scaler_y.scale_ / scaler_X.scale_
lasso_coef_original = best_lasso.coef_ * scaler_y.scale_ / scaler_X.scale_

ridge_intercept = scaler_y.mean_ - np.sum(ridge_coef_original * scaler_X.mean_)
lasso_intercept = scaler_y.mean_ - np.sum(lasso_coef_original * scaler_X.mean_)

print("\nRidge Regression Results:")
print(f"Intercept: {ridge_intercept:.4f}")
print(f"Non-zero coefficients: {np.sum(ridge_coef_original != 0)}")
print(f"Test R²: {r2_score(y_test, ridge_intercept + X_test @ ridge_coef_original):.4f}")

print("\nLasso Regression Results:")
print(f"Intercept: {lasso_intercept:.4f}")
print(f"Non-zero coefficients: {np.sum(lasso_coef_original != 0)}")
print(f"Test R²: {r2_score(y_test, lasso_intercept + X_test @ lasso_coef_original):.4f}")
```

### R Implementation

```r
# Load libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Generate synthetic data
set.seed(42)
n <- 100
p <- 20
X <- matrix(rnorm(n * p), n, p)
true_beta <- rep(0, p)
true_beta[1:5] <- c(2, -1.5, 1, -0.8, 0.6)
y <- X %*% true_beta + 0.5 * rnorm(n)

# Split data
train_idx <- sample(1:n, 0.7 * n)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Fit ridge regression
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0, standardize = TRUE)
ridge_fit <- glmnet(X_train, y_train, alpha = 0, lambda = ridge_cv$lambda.min)

# Fit lasso regression
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, standardize = TRUE)
lasso_fit <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min)

# Plot coefficient paths
par(mfrow = c(1, 2))

# Ridge coefficient paths
plot(ridge_cv$glmnet.fit, xvar = "lambda", main = "Ridge: Coefficient Paths")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = 2)

# Lasso coefficient paths
plot(lasso_cv$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = 2)

# Model comparison
ridge_pred <- predict(ridge_fit, newx = X_test)
lasso_pred <- predict(lasso_fit, newx = X_test)

ridge_r2 <- 1 - sum((y_test - ridge_pred)^2) / sum((y_test - mean(y_test))^2)
lasso_r2 <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)

cat("Ridge R²:", round(ridge_r2, 4), "\n")
cat("Lasso R²:", round(lasso_r2, 4), "\n")
cat("Ridge non-zero coefficients:", sum(coef(ridge_fit) != 0), "\n")
cat("Lasso non-zero coefficients:", sum(coef(lasso_fit) != 0), "\n")
```

## 3.2.7 Theoretical Properties

### Ridge Regression Properties

1. **Bias**: Ridge introduces bias but reduces variance
2. **Multicollinearity**: Ridge handles multicollinearity effectively
3. **Stability**: Ridge estimates are more stable than OLS
4. **No sparsity**: Ridge rarely produces exactly zero coefficients

### Lasso Properties

1. **Sparsity**: Lasso can produce exactly zero coefficients
2. **Variable selection**: Automatic feature selection
3. **Interpretability**: Sparse models are often more interpretable
4. **Group selection**: Lasso may not handle grouped variables well

### Comparison of Penalties

| Property | L0 | L1 (Lasso) | L2 (Ridge) |
|----------|----|------------|------------|
| Sparsity | Yes | Yes | No |
| Convexity | No | Yes | Yes |
| Computational cost | NP-hard | Polynomial | Polynomial |
| Variable selection | Yes | Yes | No |
| Group selection | Yes | No | No |

## 3.2.8 Advanced Topics

### Elastic Net

Elastic net combines L1 and L2 penalties:

```math
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|^2_2
```

This provides a compromise between ridge and lasso, offering both shrinkage and variable selection.

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

## 3.2.9 Model Selection and Validation

### Cross-Validation

Use cross-validation to select the optimal regularization parameter:

```python
from sklearn.model_selection import GridSearchCV

# Ridge with cross-validation
ridge_cv = GridSearchCV(Ridge(), 
                       param_grid={'alpha': np.logspace(-3, 3, 50)},
                       cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train_scaled)

print(f"Best Ridge α: {ridge_cv.best_params_['alpha']:.4f}")
```

### Information Criteria

For model comparison, consider:

1. **AIC**: $\text{AIC} = n\log(\text{RSS}/n) + 2p$
2. **BIC**: $\text{BIC} = n\log(\text{RSS}/n) + \log(n)p$
3. **Adjusted R²**: $R^2_{adj} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)}$

## 3.2.10 Practical Guidelines

### When to Use Ridge vs Lasso

**Use Ridge when:**
- Predictors are highly correlated
- You want to keep all variables
- Primary goal is prediction accuracy
- Sample size is small relative to number of predictors

**Use Lasso when:**
- You want automatic variable selection
- Interpretability is important
- You suspect many coefficients are exactly zero
- You want a sparse model

### Best Practices

1. **Always standardize predictors** before applying regularization
2. **Use cross-validation** to select the regularization parameter
3. **Consider the bias-variance tradeoff** when choosing λ
4. **Validate on a holdout set** to assess generalization performance
5. **Interpret coefficients carefully** in the context of standardization

### Common Pitfalls

1. **Not standardizing data**: Can lead to inconsistent results
2. **Over-regularization**: Choosing λ too large can introduce excessive bias
3. **Under-regularization**: Choosing λ too small may not address overfitting
4. **Ignoring multicollinearity**: Can affect coefficient interpretation
5. **Not validating assumptions**: Regularization doesn't eliminate the need for model diagnostics

## Summary

Regularization provides a powerful framework for addressing the bias-variance tradeoff in statistical learning. Ridge regression offers stability and handles multicollinearity, while lasso provides automatic variable selection and sparsity. The choice between methods depends on the specific problem context, goals, and data characteristics. Proper implementation requires careful attention to data preprocessing, parameter selection, and model validation.
