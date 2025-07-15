# 5.4. Smoothing Splines

## 5.4.1. Introduction to Smoothing Splines

Smoothing splines represent an elegant solution to the knot selection problem in regression splines. Instead of manually choosing knot locations, smoothing splines place knots at every unique data point and use regularization to control the smoothness of the fit. This approach eliminates the arbitrariness of knot placement while providing a principled way to balance fit and smoothness.

### The Knot Selection Problem

In regression splines, we face the challenge of selecting:
1. **Number of knots**: Too few knots may underfit, too many may overfit
2. **Knot locations**: Poor placement can lead to suboptimal fits
3. **Model complexity**: Balancing flexibility with generalization

Smoothing splines address these issues by:
- Placing knots at every unique data point
- Using a roughness penalty to control smoothness
- Automatically selecting the optimal level of smoothing

### Mathematical Framework

Given data points $`(x_i, y_i)_{i=1}^n`$ where $`x_1 < x_2 < \cdots < x_n`$ are unique, we seek to estimate a smooth function $`f(x)`$ that minimizes the penalized residual sum of squares:

```math
\text{RSS}_\lambda(f) = \sum_{i=1}^n [y_i - f(x_i)]^2 + \lambda \int_{x_1}^{x_n} [f''(x)]^2 dx
```

The objective function has two components:
1. **Data fidelity term**: $`\sum_{i=1}^n [y_i - f(x_i)]^2`$ ensures the function fits the data well
2. **Roughness penalty**: $`\lambda \int_{x_1}^{x_n} [f''(x)]^2 dx`$ penalizes "wiggliness" in the function

### The Smoothing Parameter λ

The parameter $`\lambda`$ controls the trade-off between fit and smoothness:
- **Large λ**: Emphasizes smoothness, may underfit the data
- **Small λ**: Emphasizes fit, may overfit the data
- **Optimal λ**: Balances fit and smoothness, typically chosen by cross-validation

## 5.4.2. Theoretical Foundation: The Roughness Penalty Approach

### The Infinite-Dimensional Optimization Problem

Consider the space $`S[a,b]`$ of all smooth functions defined on $`[a,b]`$. The smoothing spline problem is to find:

```math
\hat{f} = \arg\min_{f \in S[a,b]} \text{RSS}_\lambda(f)
```

This is an infinite-dimensional optimization problem, but it has a remarkable finite-dimensional solution.

### The Fundamental Theorem

**Theorem**: The minimizer of the penalized residual sum of squares over the infinite-dimensional function space $`S[a,b]`$ is a natural cubic spline with knots at the unique data points $`x_1, x_2, \ldots, x_n`$.

```math
\min_{f \in S[a,b]} \text{RSS}_\lambda(f) = \min_{g \in \text{NCS}_n} \text{RSS}_\lambda(g)
```

where $`\text{NCS}_n`$ denotes the family of natural cubic splines with knots at $`x_1, x_2, \ldots, x_n`$.

### Proof Sketch

The proof relies on two key insights:

1. **Interpolation Property**: For any function $`f \in S[a,b]`$, there exists a natural cubic spline $`g`$ with knots at $`x_1, x_2, \ldots, x_n`$ such that:
```math
f(x_i) = g(x_i), \quad i = 1, 2, \ldots, n
```

2. **Minimum Curvature Property**: Among all functions that interpolate the data points, the natural cubic spline minimizes the integrated squared second derivative:
```math
\int_{x_1}^{x_n} [g''(x)]^2 dx \leq \int_{x_1}^{x_n} [f''(x)]^2 dx
```

This result reduces the infinite-dimensional optimization problem to a finite-dimensional one.

## 5.4.3. Finite-Dimensional Formulation

### Basis Function Representation

Since the optimal function is a natural cubic spline with $`n`$ knots, it can be represented as:

```math
f(x) = \sum_{i=1}^n \beta_i h_i(x)
```

where $`\{h_i(x)\}_{i=1}^n`$ are the natural cubic spline basis functions with knots at $`x_1, x_2, \ldots, x_n`$.

### Matrix Formulation

The penalized objective function becomes:

```math
\text{RSS}_\lambda(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{H}\boldsymbol{\beta}\|^2 + \lambda \boldsymbol{\beta}^T \boldsymbol{\Omega} \boldsymbol{\beta}
```

where:
- $`\mathbf{y} = (y_1, y_2, \ldots, y_n)^T`$ is the response vector
- $`\mathbf{H}`$ is the $`n \times n`$ design matrix with $`H_{ij} = h_j(x_i)`$
- $`\boldsymbol{\beta} = (\beta_1, \beta_2, \ldots, \beta_n)^T`$ is the coefficient vector
- $`\boldsymbol{\Omega}`$ is the penalty matrix with $`\Omega_{ij} = \int_{x_1}^{x_n} h_i''(x) h_j''(x) dx`$

### Solution

The solution is given by:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{H}^T\mathbf{H} + \lambda \boldsymbol{\Omega})^{-1} \mathbf{H}^T\mathbf{y}
```

This is equivalent to ridge regression with a non-identity penalty matrix.

### The Smoother Matrix

The fitted values can be expressed as:

```math
\hat{\mathbf{y}} = \mathbf{S}_\lambda \mathbf{y}
```

where $`\mathbf{S}_\lambda = \mathbf{H}(\mathbf{H}^T\mathbf{H} + \lambda \boldsymbol{\Omega})^{-1} \mathbf{H}^T`$ is the smoother matrix.

## 5.4.4. The Demmler-Reinsch Basis

### Double Orthogonality

A particularly useful basis is the Demmler-Reinsch (DR) basis, which has the property that both the basis functions and their second derivatives are orthogonal:

```math
\int_{x_1}^{x_n} h_i(x) h_j(x) dx = \delta_{ij}
```

```math
\int_{x_1}^{x_n} h_i''(x) h_j''(x) dx = d_i \delta_{ij}
```

where $`d_i`$ are the eigenvalues of the penalty matrix $`\boldsymbol{\Omega}`$.

### Eigenvalue Structure

The eigenvalues $`d_i`$ have a specific structure:
- $`d_1 = d_2 = 0`$ (corresponding to linear functions)
- $`d_3 \leq d_4 \leq \cdots \leq d_n`$ (increasing eigenvalues)

This structure reflects that linear functions are not penalized, while higher-order variations are increasingly penalized.

### Shrinkage Representation

In the DR basis, the solution can be written as:

```math
\hat{\beta}_i = \frac{1}{1 + \lambda d_i} \tilde{\beta}_i
```

where $`\tilde{\beta}_i`$ are the ordinary least squares coefficients. This shows that:
- Linear terms ($`i = 1, 2`$) are not shrunk ($`d_1 = d_2 = 0`$)
- Higher-order terms are increasingly shrunk as $`d_i`$ increases

## 5.4.5. Effective Degrees of Freedom

### Definition

The effective degrees of freedom (EDF) of a smoothing spline is defined as:

```math
\text{EDF}(\lambda) = \text{tr}(\mathbf{S}_\lambda) = \sum_{i=1}^n \frac{1}{1 + \lambda d_i}
```

### Properties

1. **Range**: $`2 \leq \text{EDF}(\lambda) \leq n`$
   - $`\text{EDF}(0) = n`$ (no smoothing, interpolating spline)
   - $`\text{EDF}(\infty) = 2`$ (linear fit)

2. **Interpretation**: EDF measures the effective number of parameters in the model

3. **Non-integer values**: Unlike traditional degrees of freedom, EDF can be fractional

### Relationship to λ

The relationship between $`\lambda`$ and EDF is monotonic but nonlinear. In practice, it's often more intuitive to specify the desired EDF rather than $`\lambda`$.

## 5.4.6. Complete Smoothing Spline Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy import stats

class SmoothingSpline:
    def __init__(self, lambda_val=None, df=None, cv=True):
        """
        Smoothing Spline Implementation
        
        Parameters:
        lambda_val: smoothing parameter
        df: effective degrees of freedom
        cv: whether to use cross-validation for lambda selection
        """
        self.lambda_val = lambda_val
        self.df = df
        self.cv = cv
        self.X = None
        self.y = None
        self.beta = None
        self.smoother_matrix = None
        self.edf = None
        
    def create_natural_spline_basis(self, X):
        """
        Create natural cubic spline basis matrix
        """
        n = len(X)
        H = np.zeros((n, n))
        
        # Create basis functions using scipy
        for i in range(n):
            # Create unit vector for basis function i
            unit_vector = np.zeros(n)
            unit_vector[i] = 1.0
            
            # Create natural cubic spline
            spline = CubicSpline(X, unit_vector, bc_type='natural')
            H[:, i] = spline(X)
        
        return H
    
    def create_penalty_matrix(self, X):
        """
        Create penalty matrix for natural cubic splines
        """
        n = len(X)
        Omega = np.zeros((n, n))
        
        # For natural cubic splines, the penalty matrix can be constructed
        # using the second derivatives of the basis functions
        for i in range(n):
            for j in range(n):
                # This is a simplified version - in practice, use specialized algorithms
                if i == j:
                    Omega[i, j] = 1.0
                else:
                    Omega[i, j] = 0.0
        
        # Ensure the first two rows/columns are zero (linear terms not penalized)
        Omega[0, :] = 0
        Omega[1, :] = 0
        Omega[:, 0] = 0
        Omega[:, 1] = 0
        
        return Omega
    
    def fit(self, X, y):
        """Fit smoothing spline"""
        # Sort data by X
        sort_idx = np.argsort(X)
        self.X = X[sort_idx]
        self.y = y[sort_idx]
        
        n = len(self.X)
        
        # Create basis matrix and penalty matrix
        H = self.create_natural_spline_basis(self.X)
        Omega = self.create_penalty_matrix(self.X)
        
        # Select lambda if not provided
        if self.lambda_val is None:
            if self.df is not None:
                # Find lambda that gives desired degrees of freedom
                self.lambda_val = self.find_lambda_for_df(H, Omega, self.df)
            elif self.cv:
                # Use cross-validation to select lambda
                self.lambda_val = self.select_lambda_cv(H, Omega)
            else:
                # Default lambda
                self.lambda_val = 1.0
        
        # Solve for coefficients
        self.beta = solve(H.T @ H + self.lambda_val * Omega, H.T @ self.y)
        
        # Compute smoother matrix
        self.smoother_matrix = H @ solve(H.T @ H + self.lambda_val * Omega, H.T)
        
        # Compute effective degrees of freedom
        self.edf = np.trace(self.smoother_matrix)
        
        return self
    
    def find_lambda_for_df(self, H, Omega, target_df):
        """Find lambda that gives desired degrees of freedom"""
        def objective(lambda_val):
            S = H @ solve(H.T @ H + lambda_val * Omega, H.T)
            edf = np.trace(S)
            return (edf - target_df)**2
        
        # Use binary search to find optimal lambda
        lambda_min, lambda_max = 1e-6, 1e6
        for _ in range(20):
            lambda_mid = np.sqrt(lambda_min * lambda_max)
            if objective(lambda_mid) < 1e-6:
                break
            if np.trace(H @ solve(H.T @ H + lambda_mid * Omega, H.T)) > target_df:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid
        
        return lambda_mid
    
    def select_lambda_cv(self, H, Omega):
        """Select lambda using cross-validation"""
        lambda_candidates = np.logspace(-3, 3, 20)
        cv_scores = []
        
        for lambda_val in lambda_candidates:
            S = H @ solve(H.T @ H + lambda_val * Omega, H.T)
            # Leave-one-out cross-validation
            y_pred = S @ self.y
            residuals = self.y - y_pred
            # Adjust for leverage
            leverage = np.diag(S)
            adjusted_residuals = residuals / (1 - leverage)
            cv_score = np.mean(adjusted_residuals**2)
            cv_scores.append(cv_score)
        
        return lambda_candidates[np.argmin(cv_scores)]
    
    def predict(self, X_new):
        """Make predictions"""
        if self.beta is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create basis matrix for new points
        H_new = self.create_natural_spline_basis(X_new)
        
        # Predict using fitted coefficients
        return H_new @ self.beta
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        if self.beta is None:
            raise ValueError("Model must be fitted before getting spline function")
        
        def spline_func(x):
            return self.predict(x)
        return spline_func

def demonstrate_smoothing_splines():
    """Demonstrate smoothing spline fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 50)
    
    # Test different lambda values
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    models = {}
    
    for lambda_val in lambda_values:
        model = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model.fit(X, y)
        models[f'λ={lambda_val}'] = model
    
    # Test different degrees of freedom
    df_values = [3, 5, 8, 12, 20, 30]
    models_df = {}
    
    for df in df_values:
        model = SmoothingSpline(df=df, cv=False)
        model.fit(X, y)
        models_df[f'DF={df}'] = model
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effect of lambda
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Smoothing Parameter λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of degrees of freedom
    plt.subplot(3, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models_df.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Degrees of Freedom')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Lambda vs EDF
    plt.subplot(3, 2, 3)
    lambda_list = []
    edf_list = []
    
    for name, model in models.items():
        lambda_val = float(name.split('=')[1])
        lambda_list.append(lambda_val)
        edf_list.append(model.edf)
    
    plt.semilogx(lambda_list, edf_list, 'bo-')
    plt.xlabel('λ')
    plt.ylabel('Effective Degrees of Freedom')
    plt.title('λ vs Effective Degrees of Freedom')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cross-validation
    plt.subplot(3, 2, 4)
    lambda_candidates = np.logspace(-3, 3, 20)
    cv_scores = []
    
    for lambda_val in lambda_candidates:
        model = SmoothingSpline(lambda_val=lambda_val, cv=False)
        model.fit(X, y)
        
        # Compute LOOCV score
        y_pred = model.smoother_matrix @ y
        residuals = y - y_pred
        leverage = np.diag(model.smoother_matrix)
        adjusted_residuals = residuals / (1 - leverage)
        cv_score = np.mean(adjusted_residuals**2)
        cv_scores.append(cv_score)
    
    plt.semilogx(lambda_candidates, cv_scores, 'ro-')
    plt.xlabel('λ')
    plt.ylabel('LOOCV Score')
    plt.title('Cross-Validation for λ Selection')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(3, 2, 5)
    best_model = models['λ=1.0']  # Choose a reasonable model
    y_pred = best_model.smoother_matrix @ y
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Smoother matrix
    plt.subplot(3, 2, 6)
    plt.imshow(best_model.smoother_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Smoother Matrix S_λ')
    plt.xlabel('j')
    plt.ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return models, models_df

def analyze_noisy_data():
    """Analyze smoothing splines on noisy data"""
    # Generate noisy data with different noise levels
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    models = {}
    
    for noise in noise_levels:
        y = y_true + np.random.normal(0, noise, 100)
        
        # Fit smoothing spline with cross-validation
        model = SmoothingSpline(cv=True)
        model.fit(X, y)
        models[f'Noise={noise}'] = model
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X)
        plt.plot(X, y_plot, label=f'{name}, λ={model.lambda_val:.3f}', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Smoothing Splines on Noisy Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    noise_list = []
    lambda_list = []
    edf_list = []
    
    for name, model in models.items():
        noise = float(name.split('=')[1])
        noise_list.append(noise)
        lambda_list.append(model.lambda_val)
        edf_list.append(model.edf)
    
    plt.plot(noise_list, lambda_list, 'bo-', label='λ')
    plt.xlabel('Noise Level')
    plt.ylabel('Selected λ')
    plt.title('λ Selection vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(noise_list, edf_list, 'ro-', label='EDF')
    plt.xlabel('Noise Level')
    plt.ylabel('Effective Degrees of Freedom')
    plt.title('EDF vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show smoother matrix for one model
    model = models['Noise=0.5']
    plt.imshow(model.smoother_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Smoother Matrix (Noise=0.5)')
    plt.xlabel('j')
    plt.ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return models

# Run demonstrations
if __name__ == "__main__":
    print("Demonstrating Smoothing Splines...")
    models, models_df = demonstrate_smoothing_splines()
    
    print("\nAnalyzing Noisy Data...")
    noisy_models = analyze_noisy_data()
```

### R Implementation

```r
# Smoothing Spline Implementation in R
library(splines)
library(ggplot2)
library(dplyr)

# Function to fit smoothing spline
fit_smoothing_spline <- function(X, y, lambda = NULL, df = NULL, cv = TRUE) {
  if (is.null(lambda) && is.null(df)) {
    if (cv) {
      # Use cross-validation to select lambda
      spline_model <- smooth.spline(X, y, cv = TRUE)
    } else {
      # Use default lambda
      spline_model <- smooth.spline(X, y, lambda = 1.0)
    }
  } else if (!is.null(df)) {
    # Use specified degrees of freedom
    spline_model <- smooth.spline(X, y, df = df)
  } else {
    # Use specified lambda
    spline_model <- smooth.spline(X, y, lambda = lambda)
  }
  
  return(spline_model)
}

# Function to demonstrate smoothing splines
demonstrate_smoothing_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 50)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(50, 0, 0.5)
  
  # Test different lambda values
  lambda_values <- c(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
  models <- list()
  
  for (lambda_val in lambda_values) {
    models[[paste0("λ=", lambda_val)]] <- fit_smoothing_spline(X, y, lambda = lambda_val, cv = FALSE)
  }
  
  # Test different degrees of freedom
  df_values <- c(3, 5, 8, 12, 20, 30)
  models_df <- list()
  
  for (df in df_values) {
    models_df[[paste0("DF=", df)]] <- fit_smoothing_spline(X, y, df = df, cv = FALSE)
  }
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Smoothing Parameter λ", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], X_plot)$y
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  p2 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Degrees of Freedom", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models_df)) {
    y_pred <- predict(models_df[[name]], X_plot)$y
    p2 <- p2 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Lambda vs EDF
  lambda_list <- numeric(0)
  edf_list <- numeric(0)
  
  for (name in names(models)) {
    lambda_val <- as.numeric(gsub("λ=", "", name))
    edf <- models[[name]]$df
    
    lambda_list <- c(lambda_list, lambda_val)
    edf_list <- c(edf_list, edf)
  }
  
  p3 <- ggplot(data.frame(Lambda = lambda_list, EDF = edf_list), aes(Lambda, EDF)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    scale_x_log10() +
    labs(title = "λ vs Effective Degrees of Freedom", x = "λ", y = "EDF") +
    theme_minimal()
  
  # Cross-validation
  lambda_candidates <- 10^seq(-3, 3, length.out = 20)
  cv_scores <- numeric(0)
  
  for (lambda_val in lambda_candidates) {
    model <- fit_smoothing_spline(X, y, lambda = lambda_val, cv = FALSE)
    # Compute LOOCV score (simplified)
    y_pred <- predict(model, X)$y
    cv_score <- mean((y - y_pred)^2)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  p4 <- ggplot(data.frame(Lambda = lambda_candidates, CV_Score = cv_scores), 
               aes(Lambda, CV_Score)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    scale_x_log10() +
    labs(title = "Cross-Validation for λ Selection", x = "λ", y = "CV Score") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  return(list(models = models, models_df = models_df))
}

# Function to analyze noisy data
analyze_noisy_data_r <- function() {
  # Generate noisy data with different noise levels
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  
  noise_levels <- c(0.1, 0.5, 1.0, 2.0)
  models <- list()
  
  for (noise in noise_levels) {
    y <- y_true + rnorm(100, 0, noise)
    
    # Fit smoothing spline with cross-validation
    model <- fit_smoothing_spline(X, y, cv = TRUE)
    models[[paste0("Noise=", noise)]] <- model
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Smoothing Splines on Noisy Data", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], X)$y
    p1 <- p1 + geom_line(data = data.frame(X = X, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  noise_list <- numeric(0)
  lambda_list <- numeric(0)
  edf_list <- numeric(0)
  
  for (name in names(models)) {
    noise <- as.numeric(gsub("Noise=", "", name))
    lambda <- models[[name]]$lambda
    edf <- models[[name]]$df
    
    noise_list <- c(noise_list, noise)
    lambda_list <- c(lambda_list, lambda)
    edf_list <- c(edf_list, edf)
  }
  
  p2 <- ggplot(data.frame(Noise = noise_list, Lambda = lambda_list), 
               aes(Noise, Lambda)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "λ Selection vs Noise Level", x = "Noise Level", y = "Selected λ") +
    theme_minimal()
  
  p3 <- ggplot(data.frame(Noise = noise_list, EDF = edf_list), 
               aes(Noise, EDF)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    labs(title = "EDF vs Noise Level", x = "Noise Level", y = "Effective Degrees of Freedom") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  return(models)
}

# Run demonstrations
cat("Demonstrating Smoothing Splines in R...\n")
results <- demonstrate_smoothing_splines_r()

cat("\nAnalyzing Noisy Data in R...\n")
noisy_results <- analyze_noisy_data_r()
```

## 5.4.7. Advanced Topics

### Leave-One-Out Cross-Validation

The LOOCV score can be computed efficiently using the smoother matrix:

```python
def compute_loocv_score(model, X, y):
    """
    Compute leave-one-out cross-validation score
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    leverage = np.diag(model.smoother_matrix)
    
    # Adjust residuals for leverage
    adjusted_residuals = residuals / (1 - leverage)
    loocv_score = np.mean(adjusted_residuals**2)
    
    return loocv_score
```

### Generalized Cross-Validation

GCV provides a computationally efficient approximation to LOOCV:

```python
def compute_gcv_score(model, X, y):
    """
    Compute generalized cross-validation score
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    edf = np.trace(model.smoother_matrix)
    n = len(y)
    
    # GCV score
    gcv_score = np.mean(residuals**2) / (1 - edf/n)**2
    
    return gcv_score
```

### Confidence Intervals

```python
def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for smoothing spline predictions
    """
    # Get predictions
    y_pred = model.predict(X_new)
    
    # Compute residuals
    y_fit = model.smoother_matrix @ y
    residuals = y - y_fit
    sigma_hat = np.std(residuals)
    
    # Compute leverage for new points
    # This is a simplified version
    leverage_new = np.diag(model.smoother_matrix)[:len(X_new)]
    
    # Standard error of prediction
    se_pred = sigma_hat * np.sqrt(leverage_new)
    
    # Confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, len(y) - model.edf)
    
    ci_lower = y_pred - t_critical * se_pred
    ci_upper = y_pred + t_critical * se_pred
    
    return y_pred, ci_lower, ci_upper
```

### Weighted Smoothing Splines

For heteroscedastic data, we can use weighted smoothing splines:

```python
def fit_weighted_smoothing_spline(X, y, weights, lambda_val=None):
    """
    Fit weighted smoothing spline
    """
    n = len(X)
    W = np.diag(weights)
    
    # Create basis and penalty matrices
    H = create_natural_spline_basis(X)
    Omega = create_penalty_matrix(X)
    
    # Solve weighted problem
    if lambda_val is None:
        lambda_val = 1.0
    
    beta = solve(H.T @ W @ H + lambda_val * Omega, H.T @ W @ y)
    
    return beta, H, Omega
```

## 5.4.8. Model Diagnostics and Validation

### Comprehensive Diagnostics

```python
def smoothing_spline_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for smoothing splines
    """
    y_pred = model.smoother_matrix @ y
    residuals = y - y_pred
    leverage = np.diag(model.smoother_matrix)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Predictor
    axes[0, 2].scatter(X, residuals, alpha=0.6)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Residuals vs X')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Leverage plot
    axes[1, 0].scatter(range(len(leverage)), leverage, alpha=0.6)
    axes[1, 0].axhline(y=2*model.edf/len(y), color='r', linestyle='--', 
                       label='2*EDF/n threshold')
    axes[1, 0].set_xlabel('Observation Index')
    axes[1, 0].set_ylabel('Leverage')
    axes[1, 0].set_title('Leverage Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location plot
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Smoother matrix
    axes[1, 2].imshow(model.smoother_matrix, cmap='viridis')
    axes[1, 2].set_title('Smoother Matrix')
    axes[1, 2].set_xlabel('j')
    axes[1, 2].set_ylabel('i')
    
    plt.tight_layout()
    plt.show()
    
    return residuals, leverage
```

## Summary

Smoothing splines provide an elegant solution to the knot selection problem through:

1. **Automatic Knot Placement**: Knots at every unique data point
2. **Roughness Penalty**: Controls smoothness via integrated squared second derivative
3. **Finite-Dimensional Solution**: Infinite-dimensional problem reduces to ridge regression
4. **Effective Degrees of Freedom**: Measures model complexity
5. **Cross-Validation**: Automatic selection of smoothing parameter
6. **Theoretical Foundation**: Optimal solution is a natural cubic spline

The mathematical framework ensures optimal estimation, while the computational implementation provides both efficiency and interpretability. Smoothing splines eliminate the arbitrariness of knot selection while maintaining flexibility and smoothness.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- Wahba, G. (1990). Spline models for observational data. SIAM.
- Green, P. J., & Silverman, B. W. (1994). Nonparametric regression and generalized linear models: a roughness penalty approach. CRC Press.
