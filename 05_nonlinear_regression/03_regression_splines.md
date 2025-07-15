# 5.3. Regression Splines

## 5.3.1. Introduction to Regression Splines

Regression splines represent a powerful framework for fitting smooth, flexible functions to data by combining the local flexibility of piecewise polynomials with the global smoothness of spline functions. Unlike polynomial regression, which uses a single high-degree polynomial across the entire domain, regression splines use low-degree polynomials (typically cubic) in local regions while ensuring smooth transitions at the boundaries.

### Mathematical Framework

Given data points $`(x_i, y_i)_{i=1}^n`$ where $`x`$ is one-dimensional, we seek to model the relationship:

```math
y_i = f(x_i) + \epsilon_i, \quad i = 1, 2, \ldots, n
```

where $`f(x)`$ is a smooth function and $`\epsilon_i \sim N(0, \sigma^2)`$ are independent errors.

### Basis Function Representation

The spline function $`f(x)`$ is represented as a linear combination of basis functions:

```math
f(x) = \sum_{j=1}^p \beta_j h_j(x)
```

where $`\{h_j(x)\}_{j=1}^p`$ are the basis functions and $`\{\beta_j\}_{j=1}^p`$ are the coefficients to be estimated.

For cubic splines with $`m`$ knots, we have $`p = m + 4`$ basis functions:

```math
h_1(x) = 1, \quad h_2(x) = x, \quad h_3(x) = x^2, \quad h_4(x) = x^3
```

```math
h_{j+4}(x) = (x - \xi_j)_+^3, \quad j = 1, 2, \ldots, m
```

For natural cubic splines with $`m`$ knots, we have $`p = m`$ basis functions.

### Matrix Formulation

The regression model can be expressed in matrix form as:

```math
\mathbf{y} = \mathbf{H}\boldsymbol{\beta} + \boldsymbol{\epsilon}
```

where:
- $`\mathbf{y} = (y_1, y_2, \ldots, y_n)^T`$ is the response vector
- $`\mathbf{H}`$ is the $`n \times p`$ design matrix with elements $`H_{ij} = h_j(x_i)`$
- $`\boldsymbol{\beta} = (\beta_1, \beta_2, \ldots, \beta_p)^T`$ is the coefficient vector
- $`\boldsymbol{\epsilon} = (\epsilon_1, \epsilon_2, \ldots, \epsilon_n)^T`$ is the error vector

The design matrix $`\mathbf{H}`$ has the form:

```math
\mathbf{H} = \begin{pmatrix}
h_1(x_1) & h_2(x_1) & \cdots & h_p(x_1) \\
h_1(x_2) & h_2(x_2) & \cdots & h_p(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
h_1(x_n) & h_2(x_n) & \cdots & h_p(x_n)
\end{pmatrix}
```

### Parameter Estimation

The coefficients are estimated by minimizing the sum of squared errors:

```math
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{H}\boldsymbol{\beta}\|^2
```

The solution is given by the normal equations:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^T\mathbf{y}
```

## 5.3.2. Degrees of Freedom and Model Complexity

### Degrees of Freedom Definition

In the context of regression splines, degrees of freedom (DF) refers to the effective number of parameters in the model, which is related to the number of knots and the type of spline:

- **Cubic splines**: $`\text{DF} = m + 4`$ where $`m`$ is the number of knots
- **Natural cubic splines**: $`\text{DF} = m`$ where $`m`$ is the number of knots

### Model Selection Criteria

Several criteria can be used to select the optimal number of knots:

#### Akaike Information Criterion (AIC)

```math
\text{AIC} = n\log(\text{RSS}/n) + 2p
```

where RSS is the residual sum of squares and $`p`$ is the number of parameters.

#### Bayesian Information Criterion (BIC)

```math
\text{BIC} = n\log(\text{RSS}/n) + p\log(n)
```

#### Cross-Validation

```math
\text{CV} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{f}^{(-i)}(x_i))^2
```

where $`\hat{f}^{(-i)}`$ is the fitted function using all data except observation $`i`$.

## 5.3.3. Knot Selection Strategies

### Fixed Knot Placement

#### Quantile-Based Placement

Place knots at quantiles of the predictor variable:

```math
\xi_j = Q_x\left(\frac{j}{m+1}\right), \quad j = 1, 2, \ldots, m
```

where $`Q_x(p)`$ is the $`p`$-th quantile of $`x`$.

#### Uniform Placement

Place knots uniformly across the range:

```math
\xi_j = x_{\min} + \frac{j}{m+1}(x_{\max} - x_{\min}), \quad j = 1, 2, \ldots, m
```

### Adaptive Knot Selection

#### Stepwise Selection

1. Start with a small number of knots
2. Add knots one at a time at locations that minimize RSS
3. Use cross-validation to determine when to stop

#### Penalized Selection

Use regularization methods like Lasso or Ridge regression:

```math
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{H}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1
```

where $`\lambda`$ controls the amount of regularization.

## 5.3.4. Complete Regression Spline Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy import stats

class RegressionSpline:
    def __init__(self, df=None, knots=None, spline_type='cubic', 
                 regularization=None, lambda_val=1.0):
        """
        Regression Spline Implementation
        
        Parameters:
        df: degrees of freedom (number of basis functions)
        knots: array of knot positions
        spline_type: 'cubic' or 'natural'
        regularization: 'ridge', 'lasso', or None
        lambda_val: regularization parameter
        """
        self.df = df
        self.knots = knots
        self.spline_type = spline_type
        self.regularization = regularization
        self.lambda_val = lambda_val
        self.model = None
        self.basis_matrix = None
        self.coefficients = None
        self.intercept = None
        
    def create_basis_matrix(self, X):
        """
        Create basis matrix for regression splines
        """
        n_samples = len(X)
        
        if self.spline_type == 'cubic':
            # Cubic spline basis
            if self.knots is not None:
                n_knots = len(self.knots)
                n_basis = n_knots + 4
            else:
                # Use df to determine number of knots
                n_knots = self.df - 4
                self.knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
                n_basis = self.df
        else:
            # Natural cubic spline basis
            if self.knots is not None:
                n_knots = len(self.knots)
                n_basis = n_knots
            else:
                n_knots = self.df
                self.knots = np.percentile(X, np.linspace(0, 100, n_knots + 2))[1:-1]
                n_basis = self.df
        
        basis_matrix = np.zeros((n_samples, n_basis))
        
        if self.spline_type == 'cubic':
            # Polynomial terms
            basis_matrix[:, 0] = 1
            basis_matrix[:, 1] = X
            basis_matrix[:, 2] = X**2
            basis_matrix[:, 3] = X**3
            
            # Truncated power terms
            for i, knot in enumerate(self.knots):
                basis_matrix[:, i + 4] = np.maximum(0, X - knot)**3
        else:
            # Natural cubic spline basis using scipy
            for i in range(n_knots):
                # Create unit vector for basis function i
                unit_vector = np.zeros(n_knots)
                unit_vector[i] = 1.0
                
                # Create natural cubic spline
                temp_spline = CubicSpline(self.knots, unit_vector, bc_type='natural')
                basis_matrix[:, i] = temp_spline(X)
        
        return basis_matrix
    
    def fit(self, X, y):
        """Fit regression spline model"""
        self.basis_matrix = self.create_basis_matrix(X)
        
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.lambda_val)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.lambda_val)
        else:
            self.model = LinearRegression()
        
        self.model.fit(self.basis_matrix, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        basis_matrix = self.create_basis_matrix(X)
        return self.intercept + basis_matrix @ self.coefficients
    
    def get_spline_function(self):
        """Get the fitted spline function"""
        def spline_func(x):
            basis = self.create_basis_matrix(x)
            return self.intercept + basis @ self.coefficients
        return spline_func

def demonstrate_regression_splines():
    """Demonstrate regression spline fitting"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y_true = 2 + 3*np.sin(X) + 0.5*X
    y = y_true + np.random.normal(0, 0.3, 100)
    
    # Test different degrees of freedom
    df_values = [4, 6, 8, 10, 12, 15]
    models = {}
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='cubic')
        model.fit(X, y)
        models[f'DF={df}'] = model
    
    # Evaluate models
    X_plot = np.linspace(0, 10, 200)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Spline fits
    plt.subplot(3, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.plot(X, y_true, 'k--', label='True Function', linewidth=2)
    
    for name, model in models.items():
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Model comparison
    plt.subplot(3, 2, 2)
    df_list = []
    mse_list = []
    r2_list = []
    
    for name, model in models.items():
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        df_val = int(name.split('=')[1])
        df_list.append(df_val)
        mse_list.append(mse)
        r2_list.append(r2)
    
    plt.plot(df_list, mse_list, 'bo-', label='MSE')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance vs DF')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cross-validation
    plt.subplot(3, 2, 3)
    cv_scores = []
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='cubic')
        cv_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-cv_score.mean())
    
    plt.plot(df_values, cv_scores, 'ro-', label='CV MSE')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Basis functions
    plt.subplot(3, 2, 4)
    model = models['DF=8']
    basis_matrix = model.create_basis_matrix(X_plot)
    
    for i in range(basis_matrix.shape[1]):
        plt.plot(X_plot, basis_matrix[:, i], label=f'Basis {i+1}')
    
    plt.xlabel('X')
    plt.ylabel('Basis Function Value')
    plt.title('Basis Functions (DF=8)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals
    plt.subplot(3, 2, 5)
    best_model = models['DF=8']  # Choose a reasonable model
    y_pred = best_model.predict(X)
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Regularization comparison
    plt.subplot(3, 2, 6)
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    
    for lambda_val in lambda_values:
        model_ridge = RegressionSpline(df=12, regularization='ridge', lambda_val=lambda_val)
        model_ridge.fit(X, y)
        y_plot = model_ridge.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'λ={lambda_val}')
    
    plt.scatter(X, y, alpha=0.6, label='Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ridge Regularization Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models

def analyze_birthrate_data():
    """Analyze birthrate data with regression splines"""
    # Generate birthrate-like data (simulated)
    np.random.seed(42)
    years = np.arange(1960, 2020)
    # Simulate birthrate with some trend and noise
    birthrate = 20 - 0.1*(years - 1960) + 2*np.sin(2*np.pi*(years - 1960)/20) + np.random.normal(0, 0.5, len(years))
    
    # Test different degrees of freedom
    df_values = [3, 5, 7, 10, 15, 20]
    models = {}
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='natural')
        model.fit(years, birthrate)
        models[f'DF={df}'] = model
    
    # Cross-validation to select optimal df
    cv_scores = []
    for df in df_values:
        model = RegressionSpline(df=df, spline_type='natural')
        cv_score = cross_val_score(model, years, birthrate, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-cv_score.mean())
    
    optimal_df = df_values[np.argmin(cv_scores)]
    print(f"Optimal degrees of freedom: {optimal_df}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(years, birthrate, alpha=0.7, label='Data')
    
    years_plot = np.linspace(1960, 2020, 200)
    for name, model in models.items():
        y_plot = model.predict(years_plot)
        plt.plot(years_plot, y_plot, label=name, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Birthrate')
    plt.title('Birthrate Data: Spline Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df_values, cv_scores, 'bo-')
    plt.axvline(x=optimal_df, color='r', linestyle='--', label=f'Optimal DF={optimal_df}')
    plt.xlabel('Degrees of Freedom')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Model Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    best_model = models[f'DF={optimal_df}']
    y_pred = best_model.predict(years)
    residuals = birthrate - y_pred
    
    plt.scatter(years, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return models, optimal_df

# Run demonstrations
if __name__ == "__main__":
    print("Demonstrating Regression Splines...")
    models = demonstrate_regression_splines()
    
    print("\nAnalyzing Birthrate Data...")
    birthrate_models, optimal_df = analyze_birthrate_data()
```

### R Implementation

```r
# Regression Spline Implementation in R
library(splines)
library(ggplot2)
library(dplyr)
library(caret)

# Function to fit regression spline with specified degrees of freedom
fit_regression_spline <- function(X, y, df, spline_type = "cubic") {
  if (spline_type == "cubic") {
    # Use B-splines for cubic splines
    model <- lm(y ~ bs(X, df = df))
  } else {
    # Use natural cubic splines
    model <- lm(y ~ ns(X, df = df))
  }
  return(model)
}

# Function to demonstrate regression splines
demonstrate_regression_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Test different degrees of freedom
  df_values <- c(4, 6, 8, 10, 12, 15)
  models <- list()
  
  for (df in df_values) {
    models[[paste0("DF=", df)]] <- fit_regression_spline(X, y, df, "cubic")
  }
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Regression Spline Fits", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], data.frame(X = X_plot))
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  df_list <- numeric(0)
  mse_list <- numeric(0)
  r2_list <- numeric(0)
  
  for (name in names(models)) {
    y_pred <- predict(models[[name]])
    mse <- mean((y - y_pred)^2)
    r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
    
    df_val <- as.numeric(gsub("DF=", "", name))
    df_list <- c(df_list, df_val)
    mse_list <- c(mse_list, mse)
    r2_list <- c(r2_list, r2)
  }
  
  p2 <- ggplot(data.frame(DF = df_list, MSE = mse_list), aes(DF, MSE)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "Model Performance vs DF", x = "Degrees of Freedom", y = "Mean Squared Error") +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
  
  # Cross-validation
  cv_scores <- numeric(0)
  for (df in df_values) {
    cv_score <- mean(cv.glm(data.frame(X = X, y = y), 
                           glm(y ~ bs(X, df = df)), K = 5)$delta)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  p3 <- ggplot(data.frame(DF = df_values, CV_MSE = cv_scores), aes(DF, CV_MSE)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    labs(title = "Cross-Validation Performance", x = "Degrees of Freedom", y = "CV MSE") +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  # Model comparison table
  cat("Model Comparison:\n")
  comparison_df <- data.frame(
    DF = df_list,
    MSE = mse_list,
    R2 = r2_list,
    CV_MSE = cv_scores
  )
  print(comparison_df)
  
  return(models)
}

# Function to analyze birthrate data
analyze_birthrate_data_r <- function() {
  # Generate birthrate-like data (simulated)
  set.seed(42)
  years <- 1960:2019
  birthrate <- 20 - 0.1*(years - 1960) + 2*sin(2*pi*(years - 1960)/20) + rnorm(length(years), 0, 0.5)
  
  # Test different degrees of freedom
  df_values <- c(3, 5, 7, 10, 15, 20)
  models <- list()
  
  for (df in df_values) {
    models[[paste0("DF=", df)]] <- fit_regression_spline(years, birthrate, df, "natural")
  }
  
  # Cross-validation to select optimal df
  cv_scores <- numeric(0)
  for (df in df_values) {
    cv_score <- mean(cv.glm(data.frame(years = years, birthrate = birthrate), 
                           glm(birthrate ~ ns(years, df = df)), K = 5)$delta)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  optimal_df <- df_values[which.min(cv_scores)]
  cat(sprintf("Optimal degrees of freedom: %d\n", optimal_df))
  
  # Create plots
  years_plot <- seq(1960, 2020, length.out = 200)
  
  p1 <- ggplot() +
    geom_point(data = data.frame(years = years, birthrate = birthrate), 
               aes(years, birthrate), alpha = 0.7) +
    labs(title = "Birthrate Data: Spline Fits", x = "Year", y = "Birthrate") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], data.frame(years = years_plot))
    p1 <- p1 + geom_line(data = data.frame(years = years_plot, y = y_pred), 
                         aes(years, y), color = name, size = 1)
  }
  
  p2 <- ggplot(data.frame(DF = df_values, CV_MSE = cv_scores), aes(DF, CV_MSE)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    geom_vline(xintercept = optimal_df, color = "red", linestyle = "dashed") +
    labs(title = "Model Selection", x = "Degrees of Freedom", y = "Cross-Validation MSE") +
    theme_minimal()
  
  # Residuals analysis
  best_model <- models[[paste0("DF=", optimal_df)]]
  y_pred <- predict(best_model)
  residuals <- birthrate - y_pred
  
  p3 <- ggplot(data.frame(years = years, residuals = residuals), aes(years, residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals", x = "Year", y = "Residuals") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  return(list(models = models, optimal_df = optimal_df))
}

# Run demonstrations
cat("Demonstrating Regression Splines in R...\n")
models_r <- demonstrate_regression_splines_r()

cat("\nAnalyzing Birthrate Data in R...\n")
birthrate_results <- analyze_birthrate_data_r()
```

## 5.3.5. Advanced Topics

### Model Selection with Information Criteria

```python
def select_optimal_df_information_criteria(X, y, max_df=20, spline_type='cubic'):
    """
    Select optimal degrees of freedom using information criteria
    """
    df_values = range(3, max_df + 1)
    aic_scores = []
    bic_scores = []
    
    for df in df_values:
        model = RegressionSpline(df=df, spline_type=spline_type)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        rss = np.sum((y - y_pred)**2)
        n = len(y)
        
        # AIC
        aic = n * np.log(rss/n) + 2 * df
        aic_scores.append(aic)
        
        # BIC
        bic = n * np.log(rss/n) + df * np.log(n)
        bic_scores.append(bic)
    
    optimal_df_aic = df_values[np.argmin(aic_scores)]
    optimal_df_bic = df_values[np.argmin(bic_scores)]
    
    return optimal_df_aic, optimal_df_bic, aic_scores, bic_scores
```

### Regularization with Ridge and Lasso

```python
def compare_regularization_methods(X, y, df=10):
    """
    Compare different regularization methods
    """
    lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    results = {}
    
    for lambda_val in lambda_values:
        # Ridge regression
        model_ridge = RegressionSpline(df=df, regularization='ridge', lambda_val=lambda_val)
        model_ridge.fit(X, y)
        
        # Lasso regression
        model_lasso = RegressionSpline(df=df, regularization='lasso', lambda_val=lambda_val)
        model_lasso.fit(X, y)
        
        # Evaluate
        y_pred_ridge = model_ridge.predict(X)
        y_pred_lasso = model_lasso.predict(X)
        
        mse_ridge = mean_squared_error(y, y_pred_ridge)
        mse_lasso = mean_squared_error(y, y_pred_lasso)
        
        results[lambda_val] = {
            'ridge_mse': mse_ridge,
            'lasso_mse': mse_lasso,
            'ridge_coef': model_ridge.coefficients,
            'lasso_coef': model_lasso.coefficients
        }
    
    return results
```

### Confidence Intervals

```python
def compute_confidence_intervals(model, X, y, X_new, confidence=0.95):
    """
    Compute confidence intervals for spline predictions
    """
    # Get predictions
    y_pred = model.predict(X_new)
    
    # Compute residuals
    y_fit = model.predict(X)
    residuals = y - y_fit
    sigma_hat = np.std(residuals)
    
    # Compute leverage
    basis_matrix = model.create_basis_matrix(X)
    basis_new = model.create_basis_matrix(X_new)
    
    H = basis_matrix @ np.linalg.inv(basis_matrix.T @ basis_matrix) @ basis_matrix.T
    leverage = np.diag(H)
    
    # Standard error of prediction
    se_pred = sigma_hat * np.sqrt(1 + np.sum(basis_new**2, axis=1))
    
    # Confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, len(y) - model.df)
    
    ci_lower = y_pred - t_critical * se_pred
    ci_upper = y_pred + t_critical * se_pred
    
    return y_pred, ci_lower, ci_upper
```

## 5.3.6. Model Diagnostics and Validation

### Comprehensive Diagnostics

```python
def comprehensive_spline_diagnostics(model, X, y):
    """
    Comprehensive diagnostics for regression splines
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    
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
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location plot
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cook's Distance
    basis_matrix = model.create_basis_matrix(X)
    H = basis_matrix @ np.linalg.inv(basis_matrix.T @ basis_matrix) @ basis_matrix.T
    leverage = np.diag(H)
    cooks_d = residuals**2 * leverage / (model.df * np.var(residuals) * (1 - leverage)**2)
    
    axes[1, 2].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
    axes[1, 2].axhline(y=4/len(X), color='r', linestyle='--', label='4/n threshold')
    axes[1, 2].set_xlabel('Observation Index')
    axes[1, 2].set_ylabel("Cook's Distance")
    axes[1, 2].set_title("Cook's Distance")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return residuals, cooks_d
```

## Summary

Regression splines provide a powerful and flexible approach to nonlinear regression through:

1. **Basis Function Representation**: Linear combination of spline basis functions
2. **Degrees of Freedom Control**: Direct control over model complexity
3. **Knot Selection Strategies**: Multiple approaches for optimal knot placement
4. **Model Selection**: Information criteria and cross-validation for optimal DF selection
5. **Regularization**: Ridge and Lasso methods for coefficient shrinkage
6. **Comprehensive Diagnostics**: Multiple diagnostic plots and tests

The mathematical framework ensures optimal estimation, while the computational implementation provides both efficiency and interpretability. Regression splines address the limitations of polynomial regression while maintaining local flexibility and global smoothness.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- de Boor, C. (2001). A practical guide to splines. Springer Science & Business Media.
- Wood, S. N. (2017). Generalized additive models: an introduction with R. CRC press.
