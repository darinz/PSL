# 5.2. Cubic Splines

## 5.2.1. Introduction to Splines

Cubic splines represent a powerful approach to nonlinear regression that addresses many limitations of polynomial regression. Unlike global polynomials, splines use piecewise polynomial functions that provide local flexibility while maintaining smoothness across the entire domain.

### Mathematical Framework

Consider a one-dimensional predictor variable $`x`$ and response variable $`y`$. A spline function $`f(x)`$ is defined as a piecewise polynomial function over a partition of the domain into intervals.

**Definition**: A cubic spline is a function $`f(x)`$ that satisfies:
1. $`f(x)`$ is a cubic polynomial on each interval $`[x_i, x_{i+1}]`$
2. $`f(x)`$ is continuous at each knot $`x_i`$
3. $`f'(x)`$ is continuous at each knot $`x_i`$
4. $`f''(x)`$ is continuous at each knot $`x_i`$

### Piecewise Polynomial Structure

Given knots $`\xi_1 < \xi_2 < \cdots < \xi_m`$, the cubic spline can be expressed as:

```math
f(x) = \begin{cases}
p_1(x) & \text{if } x \in [\xi_0, \xi_1] \\
p_2(x) & \text{if } x \in [\xi_1, \xi_2] \\
\vdots & \vdots \\
p_{m+1}(x) & \text{if } x \in [\xi_m, \xi_{m+1}]
\end{cases}
```

where each $`p_i(x)`$ is a cubic polynomial:

```math
p_i(x) = a_i + b_i x + c_i x^2 + d_i x^3
```

### Continuity Conditions

At each knot $`\xi_i`$, the following continuity conditions must be satisfied:

```math
p_i(\xi_i) = p_{i+1}(\xi_i) \quad \text{(function continuity)}
```

```math
p_i'(\xi_i) = p_{i+1}'(\xi_i) \quad \text{(first derivative continuity)}
```

```math
p_i''(\xi_i) = p_{i+1}''(\xi_i) \quad \text{(second derivative continuity)}
```

## 5.2.2. Mathematical Construction of Cubic Splines

### Basis Function Representation

Cubic splines can be represented using a set of basis functions. The most common representation uses the truncated power basis:

```math
f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \sum_{i=1}^m \beta_{i+3}(x - \xi_i)_+^3
```

where $`(x - \xi_i)_+^3`$ is the truncated power function:

```math
(x - \xi_i)_+^3 = \begin{cases}
0 & \text{if } x < \xi_i \\
(x - \xi_i)^3 & \text{if } x \geq \xi_i
\end{cases}
```

### Degrees of Freedom Calculation

For a cubic spline with $`m`$ knots:

- **Total parameters**: $`4(m+1)`$ (4 coefficients for each of $`m+1`$ intervals)
- **Continuity constraints**: $`3m`$ (3 constraints at each of $`m`$ knots)
- **Effective degrees of freedom**: $`4(m+1) - 3m = m + 4`$

### Matrix Formulation

The cubic spline can be expressed in matrix form as:

```math
\mathbf{y} = \mathbf{B}\boldsymbol{\beta} + \boldsymbol{\epsilon}
```

where $`\mathbf{B}`$ is the basis matrix with columns corresponding to the basis functions:

```math
\mathbf{B} = \begin{pmatrix}
1 & x_1 & x_1^2 & x_1^3 & (x_1 - \xi_1)_+^3 & \cdots & (x_1 - \xi_m)_+^3 \\
1 & x_2 & x_2^2 & x_2^3 & (x_2 - \xi_1)_+^3 & \cdots & (x_2 - \xi_m)_+^3 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & x_n^3 & (x_n - \xi_1)_+^3 & \cdots & (x_n - \xi_m)_+^3
\end{pmatrix}
```

## 5.2.3. Natural Cubic Splines

### Definition and Properties

A natural cubic spline is a cubic spline with additional constraints at the boundary knots:

```math
f''(\xi_1) = f''(\xi_m) = 0
```

This constraint forces the spline to be linear in the extreme intervals, reducing the degrees of freedom from $`m + 4`$ to $`m`$.

### Mathematical Justification

The natural cubic spline minimizes the integrated squared second derivative:

```math
\int_{\xi_1}^{\xi_m} [f''(x)]^2 dx
```

subject to the interpolation constraints $`f(x_i) = y_i`$ for all data points.

### Basis Functions for Natural Cubic Splines

The basis functions for natural cubic splines are more complex and typically use B-splines or the natural spline basis:

```math
N_1(x) = 1, \quad N_2(x) = x, \quad N_{i+2}(x) = d_i(x) - d_{m-1}(x)
```

where $`d_i(x)`$ are the cubic spline basis functions.

## 5.2.4. Complete Cubic Spline Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class CubicSplineRegression:
    def __init__(self, knots=None, natural=False):
        """
        Cubic Spline Regression
        
        Parameters:
        knots: array of knot positions
        natural: whether to use natural cubic splines
        """
        self.knots = knots
        self.natural = natural
        self.spline = None
        self.basis_matrix = None
        self.coefficients = None
        
    def create_truncated_power_basis(self, X):
        """
        Create truncated power basis for cubic splines
        """
        n_samples = len(X)
        n_knots = len(self.knots)
        
        # Basis matrix: [1, x, x^2, x^3, (x-xi_1)_+^3, ..., (x-xi_m)_+^3]
        basis_matrix = np.zeros((n_samples, n_knots + 4))
        
        # Polynomial terms
        basis_matrix[:, 0] = 1
        basis_matrix[:, 1] = X
        basis_matrix[:, 2] = X**2
        basis_matrix[:, 3] = X**3
        
        # Truncated power terms
        for i, knot in enumerate(self.knots):
            basis_matrix[:, i + 4] = np.maximum(0, X - knot)**3
        
        return basis_matrix
    
    def create_natural_spline_basis(self, X):
        """
        Create natural cubic spline basis
        """
        n_samples = len(X)
        n_knots = len(self.knots)
        
        # For natural splines, we need to construct the basis differently
        # This is a simplified version - in practice, use specialized libraries
        basis_matrix = np.zeros((n_samples, n_knots))
        
        # Use scipy's natural cubic spline
        self.spline = CubicSpline(self.knots, np.zeros(n_knots), bc_type='natural')
        
        # Create basis functions by evaluating at different points
        for i in range(n_knots):
            # Create a unit vector at knot i
            unit_vector = np.zeros(n_knots)
            unit_vector[i] = 1.0
            
            # Create spline with this unit vector
            temp_spline = CubicSpline(self.knots, unit_vector, bc_type='natural')
            basis_matrix[:, i] = temp_spline(X)
        
        return basis_matrix
    
    def fit(self, X, y):
        """Fit cubic spline regression"""
        if self.knots is None:
            # Use quantiles of X as knots
            self.knots = np.percentile(X, np.linspace(0, 100, 6))[1:-1]
        
        if self.natural:
            self.basis_matrix = self.create_natural_spline_basis(X)
        else:
            self.basis_matrix = self.create_truncated_power_basis(X)
        
        # Fit linear regression on basis functions
        model = LinearRegression()
        model.fit(self.basis_matrix, y)
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.natural:
            basis_matrix = self.create_natural_spline_basis(X)
        else:
            basis_matrix = self.create_truncated_power_basis(X)
        
        return self.intercept + basis_matrix @ self.coefficients
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        if self.natural:
            # For natural splines, use scipy's CubicSpline
            return self.spline
        else:
            # For regular cubic splines, create a function
            def spline_func(x):
                basis = self.create_truncated_power_basis(x)
                return self.intercept + basis @ self.coefficients
            return spline_func

def demonstrate_cubic_splines():
    """Demonstrate cubic spline regression"""
    # Generate synthetic data with nonlinear relationship
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Define knots
    knots = np.array([2, 4, 6, 8])
    
    # Fit different types of splines
    splines = {}
    
    # Regular cubic spline
    spline_reg = CubicSplineRegression(knots=knots, natural=False)
    spline_reg.fit(X, y)
    splines['Regular'] = spline_reg
    
    # Natural cubic spline
    spline_nat = CubicSplineRegression(knots=knots, natural=True)
    spline_nat.fit(X, y)
    splines['Natural'] = spline_nat
    
    # Scipy cubic spline for comparison
    from scipy.interpolate import CubicSpline
    scipy_spline = CubicSpline(X, y, bc_type='natural')
    splines['Scipy'] = scipy_spline
    
    # Evaluate and plot
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and spline fits
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, spline in splines.items():
        if name == 'Scipy':
            y_plot = spline(X_plot)
        else:
            y_plot = spline.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'{name} Spline', linewidth=2)
    
    # Mark knots
    for knot in knots:
        plt.axvline(x=knot, color='gray', linestyle=':', alpha=0.7)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Basis functions for regular spline
    plt.subplot(2, 3, 2)
    basis_matrix = spline_reg.create_truncated_power_basis(X_plot)
    
    for i in range(basis_matrix.shape[1]):
        plt.plot(X_plot, basis_matrix[:, i], label=f'Basis {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('Basis Function Value')
    plt.title('Truncated Power Basis Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: First derivatives
    plt.subplot(2, 3, 3)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_deriv = spline.derivative()(X_plot)
        else:
            # Numerical derivative
            h = 1e-6
            y_plot_plus = spline.predict(X_plot + h)
            y_plot_minus = spline.predict(X_plot - h)
            y_deriv = (y_plot_plus - y_plot_minus) / (2*h)
        
        plt.plot(X_plot, y_deriv, label=f'{name} Spline')
    
    plt.xlabel('X')
    plt.ylabel('First Derivative')
    plt.title('First Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Second derivatives
    plt.subplot(2, 3, 4)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_deriv2 = spline.derivative(2)(X_plot)
        else:
            # Numerical second derivative
            h = 1e-6
            y_plot_plus = spline.predict(X_plot + h)
            y_plot_minus = spline.predict(X_plot - h)
            y_plot = spline.predict(X_plot)
            y_deriv2 = (y_plot_plus - 2*y_plot + y_plot_minus) / h**2
        
        plt.plot(X_plot, y_deriv2, label=f'{name} Spline')
    
    plt.xlabel('X')
    plt.ylabel('Second Derivative')
    plt.title('Second Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(2, 3, 5)
    for name, spline in splines.items():
        if name == 'Scipy':
            y_pred = spline(X)
        else:
            y_pred = spline.predict(X)
        
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, label=f'{name} Spline')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Knot placement effect
    plt.subplot(2, 3, 6)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    # Different knot placements
    knot_configs = {
        'Few knots': np.array([3, 7]),
        'Many knots': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'Optimal knots': np.array([2, 4, 6, 8])
    }
    
    for name, knots in knot_configs.items():
        spline = CubicSplineRegression(knots=knots, natural=True)
        spline.fit(X, y)
        y_plot = spline.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Knot Placement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return splines

# Run demonstration
if __name__ == "__main__":
    spline_models = demonstrate_cubic_splines()
```

### R Implementation

```r
# Cubic Spline Implementation in R
library(splines)
library(ggplot2)
library(dplyr)

# Function to create truncated power basis
create_truncated_power_basis <- function(X, knots) {
  n_samples <- length(X)
  n_knots <- length(knots)
  
  # Basis matrix: [1, x, x^2, x^3, (x-xi_1)_+^3, ..., (x-xi_m)_+^3]
  basis_matrix <- matrix(0, nrow = n_samples, ncol = n_knots + 4)
  
  # Polynomial terms
  basis_matrix[, 1] <- 1
  basis_matrix[, 2] <- X
  basis_matrix[, 3] <- X^2
  basis_matrix[, 4] <- X^3
  
  # Truncated power terms
  for (i in 1:n_knots) {
    basis_matrix[, i + 4] <- pmax(0, X - knots[i])^3
  }
  
  return(basis_matrix)
}

# Function to fit cubic spline regression
fit_cubic_spline <- function(X, y, knots, natural = FALSE) {
  if (natural) {
    # Use natural cubic splines
    spline_model <- lm(y ~ ns(X, knots = knots))
  } else {
    # Use regular cubic splines
    spline_model <- lm(y ~ bs(X, knots = knots, degree = 3))
  }
  
  return(spline_model)
}

# Function to demonstrate cubic splines
demonstrate_cubic_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Define knots
  knots <- c(2, 4, 6, 8)
  
  # Fit different types of splines
  splines <- list()
  
  # Regular cubic spline using B-splines
  splines$Regular <- lm(y ~ bs(X, knots = knots, degree = 3))
  
  # Natural cubic spline
  splines$Natural <- lm(y ~ ns(X, knots = knots))
  
  # Smoothing spline
  splines$Smoothing <- smooth.spline(X, y, cv = TRUE)
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    geom_vline(xintercept = knots, linetype = "dotted", color = "gray", alpha = 0.7) +
    labs(title = "Cubic Spline Fits", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X_plot)$y
    } else {
      y_pred <- predict(splines[[name]], data.frame(X = X_plot))
    }
    
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Basis functions plot
  basis_matrix <- create_truncated_power_basis(X_plot, knots)
  basis_df <- data.frame(
    X = rep(X_plot, ncol(basis_matrix)),
    Basis = rep(paste("Basis", 1:ncol(basis_matrix)), each = length(X_plot)),
    Value = as.vector(basis_matrix)
  )
  
  p2 <- ggplot(basis_df, aes(X, Value, color = Basis)) +
    geom_line() +
    labs(title = "Truncated Power Basis Functions", x = "X", y = "Basis Function Value") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Residuals plot
  residuals_df <- data.frame(
    Predicted = numeric(0),
    Residuals = numeric(0),
    Type = character(0)
  )
  
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X)$y
    } else {
      y_pred <- predict(splines[[name]])
    }
    
    residuals_df <- rbind(residuals_df, data.frame(
      Predicted = y_pred,
      Residuals = y - y_pred,
      Type = name
    ))
  }
  
  p3 <- ggplot(residuals_df, aes(Predicted, Residuals, color = Type)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals", x = "Predicted Values", y = "Residuals") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  # Model comparison
  cat("Model Comparison:\n")
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X)$y
    } else {
      y_pred <- predict(splines[[name]])
    }
    
    mse <- mean((y - y_pred)^2)
    r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
    
    cat(sprintf("%s Spline: MSE = %.4f, R² = %.4f\n", name, mse, r2))
  }
  
  return(splines)
}

# Run demonstration
spline_models_r <- demonstrate_cubic_splines_r()
```

## 5.2.5. Advanced Topics

### B-Spline Basis

B-splines provide a more numerically stable basis for cubic splines:

```python
def create_bspline_basis(X, knots, degree=3):
    """
    Create B-spline basis functions
    """
    from scipy.interpolate import BSpline
    
    # Extend knots for B-splines
    n_knots = len(knots)
    extended_knots = np.r_[(knots[0],)*(degree+1), knots, (knots[-1],)*(degree+1)]
    
    # Create B-spline basis
    basis_matrix = np.zeros((len(X), n_knots + degree - 1))
    
    for i in range(n_knots + degree - 1):
        # Create unit vector for basis function i
        coeffs = np.zeros(n_knots + degree - 1)
        coeffs[i] = 1.0
        
        # Create B-spline
        bspline = BSpline(extended_knots, coeffs, degree)
        basis_matrix[:, i] = bspline(X)
    
    return basis_matrix
```

### Smoothing Splines

Smoothing splines minimize the penalized objective function:

```math
\sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int [f''(x)]^2 dx
```

where $`\lambda`$ controls the trade-off between fit and smoothness.

```python
def fit_smoothing_spline(X, y, lambda_val=1.0):
    """
    Fit smoothing spline using penalized least squares
    """
    from scipy.interpolate import CubicSpline
    from scipy.sparse import diags
    
    n = len(X)
    
    # Create natural cubic spline basis
    # This is a simplified version - in practice, use specialized algorithms
    spline = CubicSpline(X, y, bc_type='natural')
    
    # Create penalty matrix (simplified)
    # The actual implementation is more complex
    penalty_matrix = np.eye(n) * lambda_val
    
    # Solve penalized least squares
    # This is a conceptual implementation
    return spline
```

### Knot Selection

Optimal knot placement is crucial for spline performance:

```python
def select_optimal_knots(X, y, max_knots=10, method='quantile'):
    """
    Select optimal knot positions
    """
    if method == 'quantile':
        # Use quantiles of X
        knots = np.percentile(X, np.linspace(0, 100, max_knots + 2))[1:-1]
    elif method == 'uniform':
        # Uniform spacing
        knots = np.linspace(X.min(), X.max(), max_knots + 2)[1:-1]
    elif method == 'cross_validation':
        # Use cross-validation to select optimal number of knots
        best_score = float('inf')
        best_knots = None
        
        for n_knots in range(2, max_knots + 1):
            knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LinearRegression
            
            # Create basis matrix
            basis_matrix = create_truncated_power_basis(X, knots)
            
            # Cross-validation
            cv_scores = cross_val_score(LinearRegression(), basis_matrix, y, cv=5, 
                                      scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
            
            if score < best_score:
                best_score = score
                best_knots = knots
        
        knots = best_knots
    
    return knots
```

## 5.2.6. Model Diagnostics and Validation

### Spline Diagnostics

```python
def analyze_spline_diagnostics(spline_model, X, y):
    """
    Analyze spline model diagnostics
    """
    y_pred = spline_model.predict(X)
    residuals = y - y_pred
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Residuals vs Predictor
    axes[1, 0].scatter(X, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs X')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return residuals
```

## Summary

Cubic splines provide a flexible and powerful approach to nonlinear regression through:

1. **Piecewise Structure**: Local polynomial fits with global smoothness
2. **Continuity Constraints**: Smooth transitions at knot points
3. **Basis Representations**: Multiple basis function options (truncated power, B-splines)
4. **Natural Splines**: Linear behavior at boundaries
5. **Knot Selection**: Critical for model performance

The mathematical foundations ensure optimal smoothness, while the algorithmic design provides both computational efficiency and interpretability. Cubic splines address many limitations of polynomial regression while maintaining local flexibility.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- de Boor, C. (2001). A practical guide to splines. Springer Science & Business Media.
- Wahba, G. (1990). Spline models for observational data. SIAM.
