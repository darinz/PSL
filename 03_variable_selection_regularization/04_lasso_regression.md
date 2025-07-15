# 3.4. Lasso Regression

## Introduction

Lasso (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani in 1996, is a powerful regularization technique that combines variable selection with coefficient shrinkage. Unlike ridge regression, lasso can produce exactly zero coefficients, making it particularly valuable for sparse modeling and automatic feature selection.

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
- $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$ is the L1 norm

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

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

# Generate synthetic data with sparse true coefficients
np.random.seed(42)
n, p = 100, 20

# Create design matrix
X = np.random.randn(n, p)
# Add some correlation between predictors
X[:, 1] = 0.3 * X[:, 0] + 0.7 * np.random.randn(n)
X[:, 2] = 0.2 * X[:, 0] + 0.8 * np.random.randn(n)

# True coefficients (sparse: only first 5 are non-zero)
true_beta = np.zeros(p)
true_beta[:5] = [3, -2, 1.5, -1, 0.8]

# Generate response
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

# Implement coordinate descent for lasso
def coordinate_descent_lasso(X, y, lambda_val, max_iter=1000, tol=1e-6):
    """Coordinate descent algorithm for lasso regression"""
    n, p = X.shape
    beta = np.zeros(p)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(p):
            # Compute partial residual
            r_j = y - X @ beta + X[:, j] * beta[j]
            
            # Compute univariate OLS
            x_j_norm_sq = np.sum(X[:, j]**2)
            if x_j_norm_sq > 0:
                beta_ols = np.dot(X[:, j], r_j) / x_j_norm_sq
                
                # Apply soft thresholding
                threshold = lambda_val / (2 * x_j_norm_sq)
                if abs(beta_ols) <= threshold:
                    beta[j] = 0
                else:
                    beta[j] = np.sign(beta_ols) * (abs(beta_ols) - threshold)
        
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    
    return beta

# Implement soft thresholding operator
def soft_threshold(x, threshold):
    """Soft thresholding operator"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Demonstrate soft thresholding
x_vals = np.linspace(-3, 3, 100)
thresholds = [0.5, 1.0, 1.5]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for threshold in thresholds:
    y_vals = soft_threshold(x_vals, threshold)
    plt.plot(x_vals, y_vals, label=f'λ = {threshold}')
plt.plot(x_vals, x_vals, 'k--', alpha=0.5, label='Identity')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Soft Thresholding Operator')
plt.legend()
plt.grid(True)

# Lasso with different lambda values
lambda_values = np.logspace(-3, 1, 50)
lasso_coefs = []
lasso_preds = []
lasso_nonzero = []

for alpha in lambda_values:
    # Fit lasso using sklearn
    lasso = Lasso(alpha=alpha, max_iter=2000)
    lasso.fit(X_train_scaled, y_train_scaled)
    
    # Store results
    lasso_coefs.append(lasso.coef_)
    y_pred = lasso.predict(X_test_scaled)
    lasso_preds.append(y_pred)
    lasso_nonzero.append(np.sum(lasso.coef_ != 0))

lasso_coefs = np.array(lasso_coefs)
lasso_preds = np.array(lasso_preds)

# Compare with coordinate descent
lambda_test = 0.1
lasso_sklearn = Lasso(alpha=lambda_test, max_iter=2000)
lasso_sklearn.fit(X_train_scaled, y_train_scaled)

lasso_cd = coordinate_descent_lasso(X_train_scaled, y_train_scaled, lambda_test)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Coefficient paths
axes[0, 0].semilogx(lambda_values, lasso_coefs)
axes[0, 0].set_xlabel('Regularization Parameter (λ)')
axes[0, 0].set_ylabel('Coefficient Values')
axes[0, 0].set_title('Lasso: Coefficient Paths')
axes[0, 0].grid(True)

# Number of non-zero coefficients
axes[0, 1].semilogx(lambda_values, lasso_nonzero, 'r-')
axes[0, 1].set_xlabel('Regularization Parameter (λ)')
axes[0, 1].set_ylabel('Number of Non-zero Coefficients')
axes[0, 1].set_title('Lasso: Sparsity')
axes[0, 1].grid(True)

# Cross-validation for optimal lambda
cv_scores = []
for alpha in lambda_values:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    scores = cross_val_score(lasso, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

best_idx = np.argmin(cv_scores)
best_lambda = lambda_values[best_idx]

axes[0, 2].semilogx(lambda_values, cv_scores, 'g-')
axes[0, 2].axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
axes[0, 2].set_xlabel('Regularization Parameter (λ)')
axes[0, 2].set_ylabel('Cross-Validation MSE')
axes[0, 2].set_title('Cross-Validation Performance')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Compare OLS vs Lasso coefficients
ols_coefs = linalg.lstsq(X_train_scaled, y_train_scaled, rcond=None)[0]
best_lasso = Lasso(alpha=best_lambda, max_iter=2000)
best_lasso.fit(X_train_scaled, y_train_scaled)

x_pos = np.arange(len(ols_coefs))
width = 0.35

axes[1, 0].bar(x_pos - width/2, ols_coefs, width, label='OLS', alpha=0.7)
axes[1, 0].bar(x_pos + width/2, best_lasso.coef_, width, label=f'Lasso (λ={best_lambda:.4f})', alpha=0.7)
axes[1, 0].set_xlabel('Predictor Index')
axes[1, 0].set_ylabel('Coefficient Value')
axes[1, 0].set_title('OLS vs Lasso Coefficients')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Compare sklearn vs coordinate descent
axes[1, 1].scatter(lasso_sklearn.coef_, lasso_cd, alpha=0.7)
axes[1, 1].plot([lasso_sklearn.coef_.min(), lasso_sklearn.coef_.max()], 
                [lasso_sklearn.coef_.min(), lasso_sklearn.coef_.max()], 'r--')
axes[1, 1].set_xlabel('Sklearn Lasso Coefficients')
axes[1, 1].set_ylabel('Coordinate Descent Coefficients')
axes[1, 1].set_title('Implementation Comparison')
axes[1, 1].grid(True)

# Prediction comparison
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
ols_pred = scaler_y.inverse_transform((X_test_scaled @ ols_coefs).reshape(-1, 1)).ravel()
lasso_pred = scaler_y.inverse_transform((X_test_scaled @ best_lasso.coef_).reshape(-1, 1)).ravel()

axes[1, 2].scatter(y_test_original, ols_pred, alpha=0.6, label=f'OLS (R²={r2_score(y_test_original, ols_pred):.3f})')
axes[1, 2].scatter(y_test_original, lasso_pred, alpha=0.6, label=f'Lasso (R²={r2_score(y_test_original, lasso_pred):.3f})')
axes[1, 2].plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 'k--', alpha=0.5)
axes[1, 2].set_xlabel('True Values')
axes[1, 2].set_ylabel('Predicted Values')
axes[1, 2].set_title('Prediction Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

# Print results
print(f"Best Lasso λ: {best_lambda:.4f}")
print(f"Non-zero coefficients: {np.sum(best_lasso.coef_ != 0)}")
print(f"OLS Test R²: {r2_score(y_test_original, ols_pred):.4f}")
print(f"Lasso Test R²: {r2_score(y_test_original, lasso_pred):.4f}")
print(f"OLS Test MSE: {mean_squared_error(y_test_original, ols_pred):.4f}")
print(f"Lasso Test MSE: {mean_squared_error(y_test_original, lasso_pred):.4f}")

# Compare implementations
print(f"\nImplementation comparison (λ={lambda_test}):")
print("Sklearn Lasso:", lasso_sklearn.coef_[:5])
print("Coordinate Descent:", lasso_cd[:5])
print("Difference:", np.max(np.abs(lasso_sklearn.coef_ - lasso_cd)))

# Demonstrate variable selection
print(f"\nVariable selection results:")
print("True non-zero coefficients:", np.sum(true_beta != 0))
print("Lasso non-zero coefficients:", np.sum(best_lasso.coef_ != 0))
print("Correctly identified non-zero:", np.sum((true_beta != 0) & (best_lasso.coef_ != 0)))
print("Correctly identified zero:", np.sum((true_beta == 0) & (best_lasso.coef_ == 0)))
```

### R Implementation

```r
# Load libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Generate synthetic data with sparse true coefficients
set.seed(42)
n <- 100
p <- 20

# Create design matrix
X <- matrix(rnorm(n * p), n, p)
X[, 2] <- 0.3 * X[, 1] + 0.7 * rnorm(n)
X[, 3] <- 0.2 * X[, 1] + 0.8 * rnorm(n)

# True coefficients (sparse)
true_beta <- rep(0, p)
true_beta[1:5] <- c(3, -2, 1.5, -1, 0.8)

# Generate response
y <- X %*% true_beta + 0.5 * rnorm(n)

# Split data
train_idx <- sample(1:n, 0.7 * n)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Standardize data
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                       scale = attr(X_train_scaled, "scaled:scale"))
y_train_scaled <- scale(y_train)
y_test_scaled <- scale(y_test, center = attr(y_train_scaled, "scaled:center"), 
                       scale = attr(y_train_scaled, "scaled:scale"))

# Implement coordinate descent for lasso
coordinate_descent_lasso <- function(X, y, lambda_val, max_iter = 1000, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  
  for (iteration in 1:max_iter) {
    beta_old <- beta
    
    for (j in 1:p) {
      # Compute partial residual
      r_j <- y - X %*% beta + X[, j] * beta[j]
      
      # Compute univariate OLS
      x_j_norm_sq <- sum(X[, j]^2)
      if (x_j_norm_sq > 0) {
        beta_ols <- sum(X[, j] * r_j) / x_j_norm_sq
        
        # Apply soft thresholding
        threshold <- lambda_val / (2 * x_j_norm_sq)
        if (abs(beta_ols) <= threshold) {
          beta[j] <- 0
        } else {
          beta[j] <- sign(beta_ols) * (abs(beta_ols) - threshold)
        }
      }
    }
    
    # Check convergence
    if (max(abs(beta - beta_old)) < tol) break
  }
  
  return(beta)
}

# Soft thresholding operator
soft_threshold <- function(x, threshold) {
  return(sign(x) * pmax(abs(x) - threshold, 0))
}

# Demonstrate soft thresholding
x_vals <- seq(-3, 3, length.out = 100)
thresholds <- c(0.5, 1.0, 1.5)

# Create plot data
plot_data <- data.frame(
  x = rep(x_vals, length(thresholds)),
  y = unlist(lapply(thresholds, function(t) soft_threshold(x_vals, t))),
  threshold = rep(paste("λ =", thresholds), each = length(x_vals))
)

ggplot(plot_data, aes(x = x, y = y, color = threshold)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5) +
  labs(title = "Soft Thresholding Operator", x = "Input", y = "Output") +
  theme_minimal()

# Lasso with cross-validation
lasso_cv <- cv.glmnet(X_train_scaled, y_train_scaled, alpha = 1, standardize = FALSE)
lasso_fit <- glmnet(X_train_scaled, y_train_scaled, alpha = 1, lambda = lasso_cv$lambda.min)

# Compare with coordinate descent
lambda_test <- 0.1
lasso_cd <- coordinate_descent_lasso(X_train_scaled, y_train_scaled, lambda_test)

# Plot coefficient paths
plot(lasso_cv$glmnet.fit, xvar = "lambda", main = "Lasso: Coefficient Paths")
abline(v = log(lasso_cv$lambda.min), col = "red", lty = 2)

# Compare with OLS
ols_coefs <- coef(lm(y_train_scaled ~ X_train_scaled - 1))
lasso_coefs <- as.vector(coef(lasso_fit))[-1]  # Remove intercept

# Create comparison plot
coef_comparison <- data.frame(
  predictor = 1:p,
  ols = ols_coefs,
  lasso = lasso_coefs
)

ggplot(coef_comparison, aes(x = predictor)) +
  geom_bar(aes(y = ols, fill = "OLS"), stat = "identity", alpha = 0.7, width = 0.4) +
  geom_bar(aes(y = lasso, fill = "Lasso"), stat = "identity", alpha = 0.7, width = 0.4, 
           position = position_nudge(x = 0.4)) +
  scale_fill_manual(values = c("OLS" = "blue", "Lasso" = "red")) +
  labs(title = "OLS vs Lasso Coefficients", x = "Predictor Index", y = "Coefficient Value") +
  theme_minimal()

# Prediction comparison
ols_pred <- X_test_scaled %*% ols_coefs
lasso_pred <- predict(lasso_fit, newx = X_test_scaled)

ols_r2 <- 1 - sum((y_test_scaled - ols_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)
lasso_r2 <- 1 - sum((y_test_scaled - lasso_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)

cat("OLS Test R²:", round(ols_r2, 4), "\n")
cat("Lasso Test R²:", round(lasso_r2, 4), "\n")
cat("Lasso non-zero coefficients:", sum(lasso_coefs != 0), "\n")
cat("Best λ:", round(lasso_cv$lambda.min, 4), "\n")

# Compare implementations
cat("Implementation comparison (λ =", lambda_test, "):\n")
cat("Glmnet Lasso:", coef(lasso_fit, s = lambda_test)[-1][1:5], "\n")
cat("Coordinate Descent:", lasso_cd[1:5], "\n")
```

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
