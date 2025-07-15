# 5.5. Local Regression

## 5.5.1. Introduction to Local Regression

Local regression, also known as locally weighted scatterplot smoothing (LOWESS) or locally weighted polynomial regression, is a nonparametric method for fitting smooth curves to data. Unlike global methods that fit a single function to all data points, local regression fits simple models (typically polynomials) to localized subsets of the data, then combines these local fits to create a smooth global curve.

### Key Concepts

1. **Local Neighborhood**: For each prediction point, we consider only nearby data points
2. **Weighted Regression**: Data points closer to the prediction point receive higher weights
3. **Polynomial Basis**: Local fits use low-degree polynomials (usually linear or quadratic)
4. **Smoothing Parameter**: Controls the size of the local neighborhood

### Mathematical Framework

Given data points $`(x_i, y_i)_{i=1}^n`$, we seek to estimate the function $`f(x)`$ at any point $`x_0`$ by fitting a local polynomial to nearby data points.

The local regression estimate at $`x_0`$ is:

```math
\hat{f}(x_0) = \hat{\beta}_0(x_0)
```

where $`\hat{\beta}_0(x_0)`$ is the intercept from the weighted least squares fit:

```math
(\hat{\beta}_0(x_0), \hat{\beta}_1(x_0), \ldots, \hat{\beta}_p(x_0)) = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^n w_i(x_0) [y_i - \sum_{j=0}^p \beta_j (x_i - x_0)^j]^2
```

### Weight Function

The weight function $`w_i(x_0)`$ determines the influence of each data point on the local fit:

```math
w_i(x_0) = K\left(\frac{|x_i - x_0|}{h(x_0)}\right)
```

where:
- $`K(\cdot)`$ is a kernel function
- $`h(x_0)`$ is the bandwidth (smoothing parameter)

Common kernel functions include:

**Tricube Kernel**:
```math
K(u) = \begin{cases}
(1 - |u|^3)^3 & \text{if } |u| < 1 \\
0 & \text{otherwise}
\end{cases}
```

**Gaussian Kernel**:
```math
K(u) = \exp\left(-\frac{u^2}{2}\right)
```

**Epanechnikov Kernel**:
```math
K(u) = \begin{cases}
\frac{3}{4}(1 - u^2) & \text{if } |u| < 1 \\
0 & \text{otherwise}
\end{cases}
```

## 5.5.2. Bandwidth Selection

### Fixed Bandwidth

The simplest approach uses a constant bandwidth $`h`$ for all prediction points:

```math
h(x_0) = h \quad \text{for all } x_0
```

### Variable Bandwidth

More sophisticated methods use variable bandwidths:

**Nearest Neighbor Bandwidth**:
```math
h(x_0) = \text{distance to the } k\text{-th nearest neighbor}
```

**Adaptive Bandwidth**:
```math
h(x_0) = h \cdot \left(\frac{f(x_0)}{g}\right)^{-\alpha}
```

where $`f(x_0)`$ is a pilot estimate of the density and $`g`$ is the geometric mean.

### Cross-Validation for Bandwidth Selection

The optimal bandwidth can be selected by minimizing the cross-validation score:

```math
\text{CV}(h) = \frac{1}{n}\sum_{i=1}^n [y_i - \hat{f}^{(-i)}(x_i)]^2
```

where $`\hat{f}^{(-i)}(x_i)`$ is the estimate at $`x_i`$ using all data except observation $`i``.

## 5.5.3. Complete Local Regression Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy import stats

class LocalRegression:
    def __init__(self, degree=1, bandwidth=None, kernel='tricube', 
                 nn_frac=0.3, robust=False, iterations=3):
        """
        Local Regression Implementation
        
        Parameters:
        degree: polynomial degree for local fits
        bandwidth: fixed bandwidth (if None, use nearest neighbor)
        kernel: kernel function ('tricube', 'gaussian', 'epanechnikov')
        nn_frac: fraction of points for nearest neighbor bandwidth
        robust: whether to use robust fitting (LOWESS)
        iterations: number of iterations for robust fitting
        """
        self.degree = degree
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.nn_frac = nn_frac
        self.robust = robust
        self.iterations = iterations
        self.X = None
        self.y = None
        
    def kernel_function(self, u):
        """Compute kernel weights"""
        if self.kernel == 'tricube':
            return np.where(np.abs(u) < 1, (1 - np.abs(u)**3)**3, 0)
        elif self.kernel == 'gaussian':
            return np.exp(-u**2 / 2)
        elif self.kernel == 'epanechnikov':
            return np.where(np.abs(u) < 1, 0.75 * (1 - u**2), 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def compute_bandwidth(self, X, x0):
        """Compute bandwidth for prediction point x0"""
        if self.bandwidth is not None:
            return self.bandwidth
        
        # Nearest neighbor bandwidth
        n_neighbors = max(1, int(self.nn_frac * len(X)))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X.reshape(-1, 1))
        
        distances, _ = nn.kneighbors([[x0]])
        return distances[0, -1]
    
    def local_fit(self, X, y, x0, bandwidth):
        """Fit local polynomial at point x0"""
        # Compute distances and weights
        distances = np.abs(X - x0)
        u = distances / bandwidth
        weights = self.kernel_function(u)
        
        # Remove points with zero weight
        mask = weights > 0
        if np.sum(mask) < self.degree + 1:
            return np.nan
        
        X_local = X[mask]
        y_local = y[mask]
        weights_local = weights[mask]
        
        # Create polynomial basis
        X_poly = np.ones((len(X_local), self.degree + 1))
        for d in range(1, self.degree + 1):
            X_poly[:, d] = (X_local - x0)**d
        
        # Weighted least squares
        W = np.diag(weights_local)
        XWX = X_poly.T @ W @ X_poly
        XWy = X_poly.T @ W @ y_local
        
        try:
            beta = np.linalg.solve(XWX, XWy)
            return beta[0]  # Return intercept (prediction at x0)
        except np.linalg.LinAlgError:
            return np.nan
    
    def robust_weights(self, residuals):
        """Compute robust weights for LOWESS"""
        # Bisquare weight function
        u = residuals / (6 * np.median(np.abs(residuals)))
        return np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
    
    def fit(self, X, y):
        """Fit local regression model"""
        self.X = np.array(X)
        self.y = np.array(y)
        return self
    
    def predict(self, X_new):
        """Make predictions"""
        if self.X is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_new = np.array(X_new)
        predictions = np.zeros(len(X_new))
        
        for i, x0 in enumerate(X_new):
            if self.robust:
                # Robust fitting (LOWESS)
                pred = self._robust_predict(x0)
            else:
                # Standard local regression
                bandwidth = self.compute_bandwidth(self.X, x0)
                pred = self.local_fit(self.X, self.y, x0, bandwidth)
            
            predictions[i] = pred
        
        return predictions
    
    def _robust_predict(self, x0):
        """Robust prediction using LOWESS algorithm"""
        # Initial fit
        bandwidth = self.compute_bandwidth(self.X, x0)
        pred = self.local_fit(self.X, self.y, x0, bandwidth)
        
        if np.isnan(pred):
            return np.nan
        
        # Iterative robust fitting
        for _ in range(self.iterations):
            # Compute residuals
            all_preds = np.array([self.local_fit(self.X, self.y, xi, bandwidth) 
                                 for xi in self.X])
            residuals = self.y - all_preds
            
            # Compute robust weights
            robust_weights = self.robust_weights(residuals)
            
            # Refit with robust weights
            distances = np.abs(self.X - x0)
            u = distances / bandwidth
            kernel_weights = self.kernel_function(u)
            combined_weights = kernel_weights * robust_weights
            
            # Weighted local fit
            mask = combined_weights > 0
            if np.sum(mask) < self.degree + 1:
                break
            
            X_local = self.X[mask]
            y_local = self.y[mask]
            weights_local = combined_weights[mask]
            
            X_poly = np.ones((len(X_local), self.degree + 1))
            for d in range(1, self.degree + 1):
                X_poly[:, d] = (X_local - x0)**d
            
            W = np.diag(weights_local)
            XWX = X_poly.T @ W @ X_poly
            XWy = X_poly.T @ W @ y_local
            
            try:
                beta = np.linalg.solve(XWX, XWy)
                pred = beta[0]
            except np.linalg.LinAlgError:
                break
        
        return pred

def demonstrate_local_regression():
    """Demonstrate local regression fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Test different parameters
    models = {}
    
    # Different bandwidths
    for nn_frac in [0.1, 0.3, 0.5]:
        model = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
        model.fit(X, y)
        models[f'NN={nn_frac}'] = model
    
    # Different degrees
    for degree in [0, 1, 2]:
        model = LocalRegression(degree=degree, nn_frac=0.3, robust=False)
        model.fit(X, y)
        models[f'Degree={degree}'] = model
    
    # Robust vs non-robust
    model_robust = LocalRegression(degree=1, nn_frac=0.3, robust=True)
    model_robust.fit(X, y)
    models['Robust'] = model_robust
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Effect of bandwidth
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        if 'NN=' in name:
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Bandwidth (Nearest Neighbor Fraction)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of polynomial degree
    plt.subplot(3, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        if 'Degree=' in name:
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Effect of Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Robust vs non-robust
    plt.subplot(3, 2, 3)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    # Non-robust
    model_std = models['Degree=1']
    y_plot_std = model_std.predict(X_plot)
    plt.plot(X_plot, y_plot_std, label='Standard', linewidth=2)
    
    # Robust
    y_plot_rob = model_robust.predict(X_plot)
    plt.plot(X_plot, y_plot_rob, label='Robust (LOWESS)', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robust vs Non-Robust Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cross-validation for bandwidth selection
    plt.subplot(3, 2, 4)
    nn_fractions = np.linspace(0.05, 0.8, 20)
    cv_scores = []
    
    for nn_frac in nn_fractions:
        model = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
        model.fit(X, y)
        
        # Leave-one-out cross-validation
        cv_preds = []
        for i in range(len(X)):
            X_cv = np.delete(X, i)
            y_cv = np.delete(y, i)
            model_cv = LocalRegression(degree=1, nn_frac=nn_frac, robust=False)
            model_cv.fit(X_cv, y_cv)
            pred = model_cv.predict([X[i]])[0]
            cv_preds.append(pred)
        
        cv_score = mean_squared_error(y, cv_preds)
        cv_scores.append(cv_score)
    
    plt.plot(nn_fractions, cv_scores, 'bo-')
    plt.xlabel('Nearest Neighbor Fraction')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Bandwidth Selection via Cross-Validation')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Kernel functions
    plt.subplot(3, 2, 5)
    u = np.linspace(-2, 2, 100)
    
    kernels = ['tricube', 'gaussian', 'epanechnikov']
    for kernel in kernels:
        model = LocalRegression(kernel=kernel)
        weights = model.kernel_function(u)
        plt.plot(u, weights, label=kernel.capitalize(), linewidth=2)
    
    plt.xlabel('u')
    plt.ylabel('K(u)')
    plt.title('Kernel Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Local weights at a point
    plt.subplot(3, 2, 6)
    x0 = 5.0
    model = models['NN=0.3']
    bandwidth = model.compute_bandwidth(X, x0)
    
    distances = np.abs(X - x0)
    u = distances / bandwidth
    weights = model.kernel_function(u)
    
    plt.scatter(X, weights, alpha=0.6)
    plt.axvline(x=x0, color='r', linestyle='--', label=f'x₀ = {x0}')
    plt.xlabel('X')
    plt.ylabel('Weight')
    plt.title(f'Local Weights at x₀ = {x0}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models

def analyze_outliers():
    """Analyze local regression with outliers"""
    # Generate data with outliers
    np.random.seed(42)
    X = np.linspace(0, 10, 80)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 80)
    
    # Add outliers
    outlier_indices = [20, 40, 60]
    y[outlier_indices] += 3 * np.random.normal(0, 1, len(outlier_indices))
    
    # Fit models
    model_std = LocalRegression(degree=1, nn_frac=0.3, robust=False)
    model_robust = LocalRegression(degree=1, nn_frac=0.3, robust=True)
    
    model_std.fit(X, y)
    model_robust.fit(X, y)
    
    # Predictions
    X_plot = np.linspace(0, 10, 200)
    y_plot_std = model_std.predict(X_plot)
    y_plot_rob = model_robust.predict(X_plot)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.scatter(X[outlier_indices], y[outlier_indices], 
               color='red', s=100, label='Outliers', zorder=5)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.plot(X_plot, y_plot_std, label='Standard Local Regression', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Standard Local Regression with Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.scatter(X[outlier_indices], y[outlier_indices], 
               color='red', s=100, label='Outliers', zorder=5)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.plot(X_plot, y_plot_rob, label='Robust Local Regression', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robust Local Regression (LOWESS) with Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(X_plot, y_plot_std, label='Standard', linewidth=2)
    plt.plot(X_plot, y_plot_rob, label='Robust', linewidth=2)
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison of Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show residuals
    y_pred_std = model_std.predict(X)
    y_pred_rob = model_robust.predict(X)
    
    residuals_std = y - y_pred_std
    residuals_rob = y - y_pred_rob
    
    plt.scatter(y_pred_std, residuals_std, alpha=0.6, label='Standard')
    plt.scatter(y_pred_rob, residuals_rob, alpha=0.6, label='Robust')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model_std, model_robust

# Run demonstrations
if __name__ == "__main__":
    print("Demonstrating Local Regression...")
    models = demonstrate_local_regression()
    
    print("\nAnalyzing Outliers...")
    model_std, model_robust = analyze_outliers()
```

### R Implementation

```r
# Local Regression Implementation in R
library(ggplot2)
library(dplyr)

# Function to compute kernel weights
kernel_weights <- function(u, kernel = "tricube") {
  if (kernel == "tricube") {
    return(ifelse(abs(u) < 1, (1 - abs(u)^3)^3, 0))
  } else if (kernel == "gaussian") {
    return(exp(-u^2 / 2))
  } else if (kernel == "epanechnikov") {
    return(ifelse(abs(u) < 1, 0.75 * (1 - u^2), 0))
  } else {
    stop("Unknown kernel")
  }
}

# Function to fit local regression
fit_local_regression <- function(X, y, x0, bandwidth, degree = 1, kernel = "tricube") {
  # Compute distances and weights
  distances <- abs(X - x0)
  u <- distances / bandwidth
  weights <- kernel_weights(u, kernel)
  
  # Remove points with zero weight
  mask <- weights > 0
  if (sum(mask) < degree + 1) {
    return(NA)
  }
  
  X_local <- X[mask]
  y_local <- y[mask]
  weights_local <- weights[mask]
  
  # Create polynomial basis
  X_poly <- matrix(1, nrow = length(X_local), ncol = degree + 1)
  for (d in 1:degree) {
    X_poly[, d + 1] <- (X_local - x0)^d
  }
  
  # Weighted least squares
  W <- diag(weights_local)
  XWX <- t(X_poly) %*% W %*% X_poly
  XWy <- t(X_poly) %*% W %*% y_local
  
  tryCatch({
    beta <- solve(XWX, XWy)
    return(beta[1])  # Return intercept
  }, error = function(e) {
    return(NA)
  })
}

# Function to compute bandwidth
compute_bandwidth <- function(X, x0, nn_frac = 0.3) {
  n_neighbors <- max(1, round(nn_frac * length(X)))
  distances <- abs(X - x0)
  sorted_distances <- sort(distances)
  return(sorted_distances[n_neighbors])
}

# Function to predict using local regression
predict_local_regression <- function(X, y, X_new, bandwidth = NULL, 
                                   nn_frac = 0.3, degree = 1, kernel = "tricube") {
  predictions <- numeric(length(X_new))
  
  for (i in seq_along(X_new)) {
    x0 <- X_new[i]
    
    if (is.null(bandwidth)) {
      h <- compute_bandwidth(X, x0, nn_frac)
    } else {
      h <- bandwidth
    }
    
    pred <- fit_local_regression(X, y, x0, h, degree, kernel)
    predictions[i] <- pred
  }
  
  return(predictions)
}

# Function to demonstrate local regression
demonstrate_local_regression_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Test different parameters
  X_plot <- seq(0, 10, length.out = 200)
  
  # Different bandwidths
  nn_fractions <- c(0.1, 0.3, 0.5)
  predictions_nn <- list()
  
  for (nn_frac in nn_fractions) {
    pred <- predict_local_regression(X, y, X_plot, nn_frac = nn_frac, degree = 1)
    predictions_nn[[paste0("NN=", nn_frac)]] <- pred
  }
  
  # Different degrees
  degrees <- c(0, 1, 2)
  predictions_degree <- list()
  
  for (degree in degrees) {
    pred <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = degree)
    predictions_degree[[paste0("Degree=", degree)]] <- pred
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Bandwidth", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for different bandwidths
  for (name in names(predictions_nn)) {
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = predictions_nn[[name]]), 
                         aes(X, y), color = name, size = 1)
  }
  
  p2 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Polynomial Degree", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for different degrees
  for (name in names(predictions_degree)) {
    p2 <- p2 + geom_line(data = data.frame(X = X_plot, y = predictions_degree[[name]]), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Cross-validation for bandwidth selection
  nn_fractions_cv <- seq(0.05, 0.8, length.out = 20)
  cv_scores <- numeric(length(nn_fractions_cv))
  
  for (i in seq_along(nn_fractions_cv)) {
    nn_frac <- nn_fractions_cv[i]
    
    # Leave-one-out cross-validation
    cv_preds <- numeric(length(X))
    for (j in seq_along(X)) {
      X_cv <- X[-j]
      y_cv <- y[-j]
      pred <- predict_local_regression(X_cv, y_cv, X[j], nn_frac = nn_frac, degree = 1)
      cv_preds[j] <- pred
    }
    
    cv_scores[i] <- mean((y - cv_preds)^2, na.rm = TRUE)
  }
  
  p3 <- ggplot(data.frame(NN_Fraction = nn_fractions_cv, CV_Score = cv_scores), 
               aes(NN_Fraction, CV_Score)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "Bandwidth Selection via Cross-Validation", 
         x = "Nearest Neighbor Fraction", y = "Cross-Validation MSE") +
    theme_minimal()
  
  # Kernel functions
  u <- seq(-2, 2, length.out = 100)
  kernels <- c("tricube", "gaussian", "epanechnikov")
  kernel_data <- data.frame(
    u = rep(u, length(kernels)),
    weight = c(kernel_weights(u, "tricube"), 
               kernel_weights(u, "gaussian"), 
               kernel_weights(u, "epanechnikov")),
    kernel = rep(kernels, each = length(u))
  )
  
  p4 <- ggplot(kernel_data, aes(u, weight, color = kernel)) +
    geom_line(size = 1) +
    labs(title = "Kernel Functions", x = "u", y = "K(u)") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  return(list(predictions_nn = predictions_nn, 
              predictions_degree = predictions_degree,
              cv_scores = cv_scores))
}

# Function to analyze outliers
analyze_outliers_r <- function() {
  # Generate data with outliers
  set.seed(42)
  X <- seq(0, 10, length.out = 80)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(80, 0, 0.3)
  
  # Add outliers
  outlier_indices <- c(20, 40, 60)
  y[outlier_indices] <- y[outlier_indices] + 3 * rnorm(length(outlier_indices))
  
  # Fit models
  X_plot <- seq(0, 10, length.out = 200)
  
  # Standard local regression
  y_pred_std <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = 1)
  
  # For robust fitting, we would need to implement LOWESS
  # For now, use standard local regression
  y_pred_rob <- y_pred_std  # Placeholder
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_point(data = data.frame(X = X[outlier_indices], y = y[outlier_indices]), 
               aes(X, y), color = "red", size = 3) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    geom_line(data = data.frame(X = X_plot, y = y_pred_std), aes(X, y), 
              color = "blue", size = 1) +
    labs(title = "Local Regression with Outliers", x = "X", y = "Y") +
    theme_minimal()
  
  print(p1)
  
  return(list(y_pred_std = y_pred_std, y_pred_rob = y_pred_rob))
}

# Run demonstrations
cat("Demonstrating Local Regression in R...\n")
results <- demonstrate_local_regression_r()

cat("\nAnalyzing Outliers in R...\n")
outlier_results <- analyze_outliers_r()
```

## 5.5.4. Advanced Topics

### Confidence Intervals

```python
def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for local regression predictions
    """
    predictions = model.predict(X_new)
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_preds = np.zeros((n_bootstrap, len(X_new)))
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Fit model on bootstrap sample
        model_boot = LocalRegression(degree=model.degree, 
                                   nn_frac=model.nn_frac, 
                                   kernel=model.kernel)
        model_boot.fit(X_boot, y_boot)
        
        # Predict
        bootstrap_preds[i, :] = model_boot.predict(X_new)
    
    # Compute confidence intervals
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_preds, alpha/2 * 100, axis=0)
    ci_upper = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
    
    return predictions, ci_lower, ci_upper
```

### Variable Bandwidth

```python
def adaptive_bandwidth(X, y, x0, pilot_bandwidth=0.3, alpha=0.5):
    """
    Compute adaptive bandwidth using pilot estimate
    """
    # Pilot estimate
    pilot_model = LocalRegression(degree=1, nn_frac=pilot_bandwidth)
    pilot_model.fit(X, y)
    pilot_pred = pilot_model.predict(X)
    
    # Compute residuals
    residuals = np.abs(y - pilot_pred)
    
    # Local standard deviation
    distances = np.abs(X - x0)
    weights = pilot_model.kernel_function(distances / pilot_bandwidth)
    local_std = np.sqrt(np.average(residuals**2, weights=weights))
    
    # Global standard deviation
    global_std = np.std(residuals)
    
    # Adaptive bandwidth
    adaptive_factor = (local_std / global_std)**alpha
    base_bandwidth = pilot_model.compute_bandwidth(X, x0)
    
    return base_bandwidth * adaptive_factor
```

## 5.5.5. Model Diagnostics

```python
def local_regression_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for local regression
    """
    y_pred = model.predict(X)
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

Local regression provides a flexible approach to nonparametric regression through:

1. **Local Neighborhood**: Fits simple models to nearby data points
2. **Weighted Regression**: Uses kernel weights based on distance
3. **Polynomial Basis**: Local fits use low-degree polynomials
4. **Bandwidth Control**: Balances bias and variance
5. **Robust Fitting**: LOWESS handles outliers effectively
6. **Adaptive Methods**: Variable bandwidths for heteroscedastic data

The method is particularly useful for:
- Data with unknown functional form
- Heteroscedastic errors
- Outlier-prone data
- Exploratory data analysis

Local regression provides a good balance between flexibility and interpretability, making it a valuable tool in the nonparametric regression toolkit.

## References

- Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. Journal of the American Statistical Association, 74(368), 829-836.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- Fan, J., & Gijbels, I. (1996). Local polynomial modelling and its applications. CRC Press.