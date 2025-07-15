# 3.3. Ridge Regression

## Introduction

Ridge regression, introduced by Hoerl and Kennard in 1970, is one of the most fundamental regularization techniques in statistical learning. It addresses the bias-variance tradeoff by introducing an L2 penalty on the regression coefficients, leading to more stable and often more accurate predictions than ordinary least squares (OLS).

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

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

# Generate synthetic data with multicollinearity
np.random.seed(42)
n, p = 100, 10

# Create correlated predictors
X = np.random.randn(n, p)
# Add correlation between predictors
X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n)
X[:, 2] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n)

# True coefficients (only first 3 are non-zero)
true_beta = np.zeros(p)
true_beta[:3] = [2, -1.5, 1]

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

# Compute SVD
U, d, Vt = linalg.svd(X_train_scaled, full_matrices=False)

# Ridge regression with different lambda values
lambda_values = np.logspace(-3, 3, 50)
ridge_coefs = []
ridge_preds = []
ridge_dfs = []

for alpha in lambda_values:
    # Fit ridge regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train_scaled)
    
    # Store coefficients
    ridge_coefs.append(ridge.coef_)
    
    # Store predictions
    y_pred = ridge.predict(X_test_scaled)
    ridge_preds.append(y_pred)
    
    # Compute degrees of freedom
    df = np.sum(d**2 / (d**2 + alpha))
    ridge_dfs.append(df)

ridge_coefs = np.array(ridge_coefs)
ridge_preds = np.array(ridge_preds)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Coefficient paths
axes[0, 0].semilogx(lambda_values, ridge_coefs)
axes[0, 0].set_xlabel('Regularization Parameter (λ)')
axes[0, 0].set_ylabel('Coefficient Values')
axes[0, 0].set_title('Ridge: Coefficient Paths')
axes[0, 0].grid(True)

# Degrees of freedom
axes[0, 1].semilogx(lambda_values, ridge_dfs, 'b-')
axes[0, 1].set_xlabel('Regularization Parameter (λ)')
axes[0, 1].set_ylabel('Degrees of Freedom')
axes[0, 1].set_title('Ridge: Degrees of Freedom')
axes[0, 1].grid(True)

# Shrinkage factors for different singular values
shrinkage_factors = d**2 / (d**2[:, None] + lambda_values)
for i in range(min(5, len(d))):
    axes[0, 2].semilogx(lambda_values, shrinkage_factors[i], label=f'd_{i+1}={d[i]:.2f}')
axes[0, 2].set_xlabel('Regularization Parameter (λ)')
axes[0, 2].set_ylabel('Shrinkage Factor')
axes[0, 2].set_title('Shrinkage Factors by Singular Value')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Cross-validation for optimal lambda
cv_scores = []
for alpha in lambda_values:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train_scaled, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

best_idx = np.argmin(cv_scores)
best_lambda = lambda_values[best_idx]

axes[1, 0].semilogx(lambda_values, cv_scores, 'r-')
axes[1, 0].axvline(best_lambda, color='red', linestyle='--', label=f'Best λ = {best_lambda:.4f}')
axes[1, 0].set_xlabel('Regularization Parameter (λ)')
axes[1, 0].set_ylabel('Cross-Validation MSE')
axes[1, 0].set_title('Cross-Validation Performance')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Compare OLS vs Ridge coefficients
ols_coefs = linalg.lstsq(X_train_scaled, y_train_scaled, rcond=None)[0]
best_ridge = Ridge(alpha=best_lambda)
best_ridge.fit(X_train_scaled, y_train_scaled)

x_pos = np.arange(len(ols_coefs))
width = 0.35

axes[1, 1].bar(x_pos - width/2, ols_coefs, width, label='OLS', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, best_ridge.coef_, width, label=f'Ridge (λ={best_lambda:.4f})', alpha=0.7)
axes[1, 1].set_xlabel('Predictor Index')
axes[1, 1].set_ylabel('Coefficient Value')
axes[1, 1].set_title('OLS vs Ridge Coefficients')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Prediction comparison
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
ols_pred = scaler_y.inverse_transform((X_test_scaled @ ols_coefs).reshape(-1, 1)).ravel()
ridge_pred = scaler_y.inverse_transform((X_test_scaled @ best_ridge.coef_).reshape(-1, 1)).ravel()

axes[1, 2].scatter(y_test_original, ols_pred, alpha=0.6, label=f'OLS (R²={r2_score(y_test_original, ols_pred):.3f})')
axes[1, 2].scatter(y_test_original, ridge_pred, alpha=0.6, label=f'Ridge (R²={r2_score(y_test_original, ridge_pred):.3f})')
axes[1, 2].plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'k--', alpha=0.5)
axes[1, 2].set_xlabel('True Values')
axes[1, 2].set_ylabel('Predicted Values')
axes[1, 2].set_title('Prediction Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.show()

# Print results
print(f"Best Ridge λ: {best_lambda:.4f}")
print(f"Degrees of Freedom: {ridge_dfs[best_idx]:.2f}")
print(f"OLS Test R²: {r2_score(y_test_original, ols_pred):.4f}")
print(f"Ridge Test R²: {r2_score(y_test_original, ridge_pred):.4f}")
print(f"OLS Test MSE: {mean_squared_error(y_test_original, ols_pred):.4f}")
print(f"Ridge Test MSE: {mean_squared_error(y_test_original, ridge_pred):.4f}")

# Demonstrate the augmented data interpretation
def ridge_via_augmented_data(X, y, lambda_val):
    """Implement ridge regression using the augmented data approach"""
    n, p = X.shape
    
    # Create augmented data
    X_aug = np.vstack([X, np.sqrt(lambda_val) * np.eye(p)])
    y_aug = np.concatenate([y, np.zeros(p)])
    
    # Solve using OLS on augmented data
    beta_aug = linalg.lstsq(X_aug, y_aug, rcond=None)[0]
    
    return beta_aug

# Compare methods
lambda_test = 1.0
ridge_sklearn = Ridge(alpha=lambda_test)
ridge_sklearn.fit(X_train_scaled, y_train_scaled)

ridge_augmented = ridge_via_augmented_data(X_train_scaled, y_train_scaled, lambda_test)

print(f"\nCoefficient comparison (λ={lambda_test}):")
print("Sklearn Ridge:", ridge_sklearn.coef_[:3])
print("Augmented Data:", ridge_augmented[:3])
print("Difference:", np.max(np.abs(ridge_sklearn.coef_ - ridge_augmented)))
```

### R Implementation

```r
# Load libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Generate synthetic data with multicollinearity
set.seed(42)
n <- 100
p <- 10

# Create correlated predictors
X <- matrix(rnorm(n * p), n, p)
X[, 2] <- 0.8 * X[, 1] + 0.2 * rnorm(n)
X[, 3] <- 0.7 * X[, 1] + 0.3 * rnorm(n)

# True coefficients
true_beta <- rep(0, p)
true_beta[1:3] <- c(2, -1.5, 1)

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

# Compute SVD
svd_result <- svd(X_train_scaled)
U <- svd_result$u
d <- svd_result$d
V <- svd_result$v

# Ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train_scaled, y_train_scaled, alpha = 0, standardize = FALSE)
ridge_fit <- glmnet(X_train_scaled, y_train_scaled, alpha = 0, lambda = ridge_cv$lambda.min)

# Compute degrees of freedom
df_ridge <- sum(d^2 / (d^2 + ridge_cv$lambda.min))

# Plot coefficient paths
plot(ridge_cv$glmnet.fit, xvar = "lambda", main = "Ridge: Coefficient Paths")
abline(v = log(ridge_cv$lambda.min), col = "red", lty = 2)

# Compare with OLS
ols_coefs <- coef(lm(y_train_scaled ~ X_train_scaled - 1))
ridge_coefs <- as.vector(coef(ridge_fit))[-1]  # Remove intercept

# Create comparison plot
coef_comparison <- data.frame(
  predictor = 1:p,
  ols = ols_coefs,
  ridge = ridge_coefs
)

ggplot(coef_comparison, aes(x = predictor)) +
  geom_bar(aes(y = ols, fill = "OLS"), stat = "identity", alpha = 0.7, width = 0.4) +
  geom_bar(aes(y = ridge, fill = "Ridge"), stat = "identity", alpha = 0.7, width = 0.4, 
           position = position_nudge(x = 0.4)) +
  scale_fill_manual(values = c("OLS" = "blue", "Ridge" = "red")) +
  labs(title = "OLS vs Ridge Coefficients", x = "Predictor Index", y = "Coefficient Value") +
  theme_minimal()

# Prediction comparison
ols_pred <- X_test_scaled %*% ols_coefs
ridge_pred <- predict(ridge_fit, newx = X_test_scaled)

ols_r2 <- 1 - sum((y_test_scaled - ols_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)
ridge_r2 <- 1 - sum((y_test_scaled - ridge_pred)^2) / sum((y_test_scaled - mean(y_test_scaled))^2)

cat("OLS Test R²:", round(ols_r2, 4), "\n")
cat("Ridge Test R²:", round(ridge_r2, 4), "\n")
cat("Ridge Degrees of Freedom:", round(df_ridge, 2), "\n")
cat("Best λ:", round(ridge_cv$lambda.min, 4), "\n")
```

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