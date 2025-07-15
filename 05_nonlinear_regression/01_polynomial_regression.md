# 5.1. Polynomial Regression

## 5.1.1. Introduction to Polynomial Regression

Polynomial regression is a form of nonlinear regression that models the relationship between a dependent variable and one or more independent variables as an $`n`$-th degree polynomial. While the relationship between variables is nonlinear, the model remains linear in the parameters, making it a special case of multiple linear regression.

### Mathematical Framework

Consider a polynomial regression model of degree $`d`$ with a single predictor variable:

```math
y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \epsilon_i
```

where:
- $`y_i`$ is the response variable for observation $`i`$
- $`x_i`$ is the predictor variable
- $`\beta_0, \beta_1, \ldots, \beta_d`$ are the polynomial coefficients
- $`\epsilon_i`$ is the error term, typically assumed to be $`\epsilon_i \sim N(0, \sigma^2)`$

### Matrix Formulation

The polynomial regression model can be expressed in matrix form as:

```math
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
```

where:

```math
\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad
\mathbf{X} = \begin{pmatrix} 
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{pmatrix}, \quad
\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d \end{pmatrix}, \quad
\boldsymbol{\epsilon} = \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{pmatrix}
```

### Parameter Estimation

The least squares estimator for the polynomial coefficients is:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

The fitted values are:

```math
\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}
```

where $`\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T`$ is the hat matrix.

### Degrees of Freedom

For a polynomial of degree $`d`$, the model has $`d + 1`$ parameters (including the intercept). The residual degrees of freedom are:

```math
\text{df}_{\text{residual}} = n - (d + 1)
```

![Degrees of Freedom and Model Flexibility](../_images/w5_ss_DR.png)

*Figure: Illustration of how degrees of freedom affect the flexibility of a polynomial regression model.*

## 5.1.2. Basis Functions and Orthogonal Polynomials

### Standard Polynomial Basis

The standard polynomial basis functions are:

```math
\phi_0(x) = 1, \quad \phi_1(x) = x, \quad \phi_2(x) = x^2, \quad \ldots, \quad \phi_d(x) = x^d
```

### Orthogonal Polynomials

To avoid multicollinearity issues, orthogonal polynomials are often used. The Gram-Schmidt orthogonalization process creates orthogonal basis functions:

```math
p_0(x) = 1
```

```math
p_1(x) = x - \frac{\sum_{i=1}^n x_i}{n}
```

```math
p_j(x) = (x - \alpha_j)p_{j-1}(x) - \beta_j p_{j-2}(x)
```

where $`\alpha_j`$ and $`\beta_j`$ are chosen to ensure orthogonality.

### Implementation of Orthogonal Polynomials

```python
import numpy as np
from scipy.special import legendre
from sklearn.preprocessing import PolynomialFeatures

def create_orthogonal_polynomials(X, degree):
    """
    Create orthogonal polynomial features
    
    Parameters:
    X: predictor variable (n_samples,)
    degree: polynomial degree
    
    Returns:
    X_poly: orthogonal polynomial features (n_samples, degree+1)
    """
    n_samples = len(X)
    X_poly = np.zeros((n_samples, degree + 1))
    
    # Normalize X to [-1, 1] for better numerical stability
    X_norm = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    
    for d in range(degree + 1):
        # Use Legendre polynomials (orthogonal on [-1, 1])
        poly = legendre(d)
        X_poly[:, d] = poly(X_norm)
    
    return X_poly

def create_standard_polynomials(X, degree):
    """
    Create standard polynomial features
    
    Parameters:
    X: predictor variable (n_samples,)
    degree: polynomial degree
    
    Returns:
    X_poly: polynomial features (n_samples, degree+1)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X.reshape(-1, 1))
```

## 5.1.3. Model Selection and Degree Selection

### Information Criteria

#### Akaike Information Criterion (AIC)

```math
\text{AIC} = 2k - 2\ln(L)
```

where $`k = d + 1`$ is the number of parameters and $`L`$ is the likelihood.

For normal errors, AIC becomes:

```math
\text{AIC} = n\ln(\text{RSS}/n) + 2(d + 1)
```

#### Bayesian Information Criterion (BIC)

```math
\text{BIC} = \ln(n)k - 2\ln(L)
```

For normal errors:

```math
\text{BIC} = n\ln(\text{RSS}/n) + (d + 1)\ln(n)
```

### Cross-Validation

The $`k`$-fold cross-validation score is:

```math
\text{CV}(d) = \frac{1}{k}\sum_{i=1}^k \frac{1}{n_i}\sum_{j \in \text{fold}_i} (y_j - \hat{y}_j^{(-i)})^2
```

where $`\hat{y}_j^{(-i)}`$ is the prediction for observation $`j`$ using the model trained on all folds except fold $`i`$.

### Forward and Backward Selection

#### Forward Selection Algorithm

```python
def forward_polynomial_selection(X, y, max_degree=10, criterion='aic'):
    """
    Forward selection for polynomial degree
    
    Parameters:
    X: predictor variable
    y: response variable
    max_degree: maximum degree to consider
    criterion: 'aic', 'bic', or 'cv'
    
    Returns:
    best_degree: optimal polynomial degree
    scores: scores for each degree
    """
    n = len(y)
    scores = []
    
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        X_poly = create_standard_polynomials(X, degree)
        
        if criterion in ['aic', 'bic']:
            # Fit model
            beta_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
            y_hat = X_poly @ beta_hat
            rss = np.sum((y - y_hat)**2)
            
            if criterion == 'aic':
                score = n * np.log(rss/n) + 2 * (degree + 1)
            else:  # bic
                score = n * np.log(rss/n) + (degree + 1) * np.log(n)
        else:  # cv
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
        
        scores.append(score)
    
    best_degree = np.argmin(scores) + 1
    return best_degree, scores

def backward_polynomial_selection(X, y, max_degree=10, criterion='aic'):
    """
    Backward selection for polynomial degree
    """
    n = len(y)
    scores = []
    
    for degree in range(max_degree, 0, -1):
        # Create polynomial features
        X_poly = create_standard_polynomials(X, degree)
        
        if criterion in ['aic', 'bic']:
            # Fit model
            beta_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
            y_hat = X_poly @ beta_hat
            rss = np.sum((y - y_hat)**2)
            
            if criterion == 'aic':
                score = n * np.log(rss/n) + 2 * (degree + 1)
            else:  # bic
                score = n * np.log(rss/n) + (degree + 1) * np.log(n)
        else:  # cv
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LinearRegression
            
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
        
        scores.append(score)
    
    # Reverse to get ascending order
    scores = scores[::-1]
    best_degree = max_degree - np.argmin(scores)
    return best_degree, scores
```

## 5.1.4. Complete Polynomial Regression Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd

class PolynomialRegression:
    def __init__(self, degree=2, use_orthogonal=False):
        """
        Polynomial Regression Model
        
        Parameters:
        degree: polynomial degree
        use_orthogonal: whether to use orthogonal polynomials
        """
        self.degree = degree
        self.use_orthogonal = use_orthogonal
        self.model = LinearRegression()
        self.poly_features = None
        self.X_poly = None
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit polynomial regression model"""
        if self.use_orthogonal:
            self.X_poly = create_orthogonal_polynomials(X, self.degree)
        else:
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=True)
            self.X_poly = self.poly_features.fit_transform(X.reshape(-1, 1))
        
        # Fit linear regression
        self.model.fit(self.X_poly, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.use_orthogonal:
            X_poly = create_orthogonal_polynomials(X, self.degree)
        else:
            X_poly = self.poly_features.transform(X.reshape(-1, 1))
        
        return self.model.predict(X_poly)
    
    def get_polynomial_equation(self):
        """Get the polynomial equation as a string"""
        if self.use_orthogonal:
            return "Orthogonal polynomial coefficients: " + str(self.coefficients)
        
        equation = f"y = {self.intercept:.4f}"
        for i, coef in enumerate(self.coefficients[1:], 1):
            if coef >= 0:
                equation += f" + {coef:.4f}x^{i}"
            else:
                equation += f" - {abs(coef):.4f}x^{i}"
        
        return equation
    
    def calculate_metrics(self, X, y):
        """Calculate model performance metrics"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Adjusted R-squared
        n = len(y)
        p = self.degree + 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # AIC and BIC
        rss = np.sum((y - y_pred)**2)
        aic = n * np.log(rss/n) + 2 * p
        bic = n * np.log(rss/n) + p * np.log(n)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'Adjusted R²': adj_r2,
            'AIC': aic,
            'BIC': bic
        }

def demonstrate_polynomial_regression():
    """Demonstrate polynomial regression with synthetic data"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 2 + 3*X - 0.5*X**2 + 0.1*X**3
    y = y_true + np.random.normal(0, 0.5, 100)
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 4, 5, 6]
    models = {}
    metrics = {}
    
    for degree in degrees:
        # Fit model
        model = PolynomialRegression(degree=degree)
        model.fit(X, y)
        models[degree] = model
        
        # Calculate metrics
        metrics[degree] = model.calculate_metrics(X, y)
        
        print(f"Degree {degree}:")
        print(f"  Equation: {model.get_polynomial_equation()}")
        print(f"  R²: {metrics[degree]['R²']:.4f}")
        print(f"  AIC: {metrics[degree]['AIC']:.4f}")
        print(f"  BIC: {metrics[degree]['BIC']:.4f}")
        print()
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and fitted curves
    plt.subplot(2, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    X_plot = np.linspace(-3, 3, 200)
    
    for degree in [1, 2, 3]:
        y_plot = models[degree].predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polynomial Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: R² vs Degree
    plt.subplot(2, 3, 2)
    r2_values = [metrics[d]['R²'] for d in degrees]
    plt.plot(degrees, r2_values, 'bo-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R²')
    plt.title('R² vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: AIC vs Degree
    plt.subplot(2, 3, 3)
    aic_values = [metrics[d]['AIC'] for d in degrees]
    plt.plot(degrees, aic_values, 'ro-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('AIC')
    plt.title('AIC vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: BIC vs Degree
    plt.subplot(2, 3, 4)
    bic_values = [metrics[d]['BIC'] for d in degrees]
    plt.plot(degrees, bic_values, 'go-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('BIC')
    plt.title('BIC vs Degree')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals for degree 3
    plt.subplot(2, 3, 5)
    y_pred_3 = models[3].predict(X)
    residuals_3 = y - y_pred_3
    plt.scatter(y_pred_3, residuals_3, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals (Degree 3)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Overfitting demonstration
    plt.subplot(2, 3, 6)
    plt.scatter(X, y, alpha=0.6, label='Data')
    
    for degree in [3, 6]:
        y_plot = models[degree].predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Overfitting Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models, metrics

# Run demonstration
if __name__ == "__main__":
    models, metrics = demonstrate_polynomial_regression()
```

### R Implementation

```r
# Polynomial Regression Implementation in R
library(ggplot2)
library(dplyr)

# Function to create polynomial features
create_polynomial_features <- function(X, degree) {
  X_poly <- matrix(1, nrow = length(X), ncol = degree + 1)
  for (d in 1:degree) {
    X_poly[, d + 1] <- X^d
  }
  return(X_poly)
}

# Function to fit polynomial regression
fit_polynomial_regression <- function(X, y, degree) {
  X_poly <- create_polynomial_features(X, degree)
  
  # Fit linear regression
  model <- lm(y ~ X_poly - 1)  # -1 removes intercept since we include it in X_poly
  
  return(list(
    model = model,
    coefficients = coef(model),
    fitted_values = fitted(model),
    residuals = residuals(model)
  ))
}

# Function to calculate model metrics
calculate_polynomial_metrics <- function(model, X, y, degree) {
  y_pred <- fitted(model)
  
  # Basic metrics
  mse <- mean((y - y_pred)^2)
  rmse <- sqrt(mse)
  r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
  
  # Adjusted R-squared
  n <- length(y)
  p <- degree + 1
  adj_r2 <- 1 - (1 - r2) * (n - 1) / (n - p - 1)
  
  # AIC and BIC
  rss <- sum((y - y_pred)^2)
  aic <- n * log(rss/n) + 2 * p
  bic <- n * log(rss/n) + p * log(n)
  
  return(list(
    MSE = mse,
    RMSE = rmse,
    R2 = r2,
    Adjusted_R2 = adj_r2,
    AIC = aic,
    BIC = bic
  ))
}

# Function to demonstrate polynomial regression
demonstrate_polynomial_regression_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(-3, 3, length.out = 100)
  y_true <- 2 + 3*X - 0.5*X^2 + 0.1*X^3
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Test different polynomial degrees
  degrees <- 1:6
  models <- list()
  metrics <- list()
  
  for (degree in degrees) {
    # Fit model
    model_result <- fit_polynomial_regression(X, y, degree)
    models[[degree]] <- model_result
    
    # Calculate metrics
    metrics[[degree]] <- calculate_polynomial_metrics(
      model_result$model, X, y, degree
    )
    
    cat("Degree", degree, ":\n")
    cat("  R²:", round(metrics[[degree]]$R2, 4), "\n")
    cat("  AIC:", round(metrics[[degree]]$AIC, 4), "\n")
    cat("  BIC:", round(metrics[[degree]]$BIC, 4), "\n\n")
  }
  
  # Create visualization
  X_plot <- seq(-3, 3, length.out = 200)
  
  # Data frame for plotting
  plot_data <- data.frame(
    X = rep(X_plot, length(degrees)),
    Degree = rep(degrees, each = length(X_plot)),
    Y = NA
  )
  
  # Calculate predictions for each degree
  for (degree in degrees) {
    X_poly_plot <- create_polynomial_features(X_plot, degree)
    y_plot <- X_poly_plot %*% models[[degree]]$coefficients
    plot_data$Y[plot_data$Degree == degree] <- y_plot
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = plot_data[plot_data$Degree %in% c(1, 2, 3), ], 
              aes(X, Y, color = factor(Degree))) +
    labs(title = "Polynomial Fits", x = "X", y = "Y", color = "Degree") +
    theme_minimal()
  
  # Metrics plots
  metrics_df <- data.frame(
    Degree = degrees,
    R2 = sapply(metrics, function(m) m$R2),
    AIC = sapply(metrics, function(m) m$AIC),
    BIC = sapply(metrics, function(m) m$BIC)
  )
  
  p2 <- ggplot(metrics_df, aes(Degree, R2)) +
    geom_line() + geom_point() +
    labs(title = "R² vs Degree") +
    theme_minimal()
  
  p3 <- ggplot(metrics_df, aes(Degree, AIC)) +
    geom_line() + geom_point() +
    labs(title = "AIC vs Degree") +
    theme_minimal()
  
  p4 <- ggplot(metrics_df, aes(Degree, BIC)) +
    geom_line() + geom_point() +
    labs(title = "BIC vs Degree") +
    theme_minimal()
  
  # Residuals plot
  residuals_df <- data.frame(
    Predicted = models[[3]]$fitted_values,
    Residuals = models[[3]]$residuals
  )
  
  p5 <- ggplot(residuals_df, aes(Predicted, Residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals (Degree 3)") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  print(p5)
  
  return(list(models = models, metrics = metrics))
}

# Run demonstration
results_r <- demonstrate_polynomial_regression_r()
```

## 5.1.5. Model Diagnostics and Validation

### Residual Analysis

```python
def analyze_polynomial_residuals(model, X, y):
    """
    Analyze residuals for polynomial regression
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
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    
    # Jarque-Bera test for normality
    jb_stat, jb_p = jarque_bera(residuals)
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
    
    return residuals
```

### Cross-Validation for Model Selection

```python
def cross_validate_polynomial_degree(X, y, max_degree=10, cv_folds=5):
    """
    Cross-validation for polynomial degree selection
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for degree in range(1, max_degree + 1):
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            model = PolynomialRegression(degree=degree)
            model.fit(X_train, y_train)
            
            # Predict and calculate MSE
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            fold_scores.append(mse)
        
        cv_scores.append(np.mean(fold_scores))
    
    # Find optimal degree
    optimal_degree = np.argmin(cv_scores) + 1
    
    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_degree + 1), cv_scores, 'bo-')
    plt.axvline(x=optimal_degree, color='r', linestyle='--', 
                label=f'Optimal degree: {optimal_degree}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation MSE')
    plt.title('Cross-Validation for Polynomial Degree Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_degree, cv_scores
```

## 5.1.6. Limitations and Alternatives

### Limitations of Polynomial Regression

1. **Overfitting**: High-degree polynomials can fit noise in the data
2. **Extrapolation Issues**: Polynomials behave poorly outside the training range
3. **Global Nature**: Assumes the same relationship holds across the entire domain
4. **Interpretability**: High-degree terms are difficult to interpret

### Mathematical Analysis of Limitations

#### Extrapolation Problem

For a polynomial $`f(x) = \sum_{i=0}^d \beta_i x^i`$, the behavior as $`x \to \infty`$ is dominated by the highest degree term:

```math
\lim_{x \to \infty} f(x) = \lim_{x \to \infty} \beta_d x^d
```

This leads to explosive growth or decay outside the training range.

#### Runge's Phenomenon

For equally spaced interpolation points, high-degree polynomials can exhibit oscillatory behavior:

```math
f(x) = \frac{1}{1 + 25x^2}
```

The interpolating polynomial of degree $`n`$ at $`n+1`$ equally spaced points can have maximum error growing exponentially with $`n`$.

### Alternatives to Polynomial Regression

1. **Spline Regression**: Piecewise polynomials with continuity constraints
2. **Local Polynomial Regression**: Fitting polynomials in local neighborhoods
3. **Kernel Regression**: Non-parametric smoothing methods
4. **Basis Expansion**: Using other basis functions (Fourier, wavelet, etc.)

## Summary

Polynomial regression provides a flexible approach to modeling nonlinear relationships while maintaining linearity in parameters. Key concepts include:

1. **Mathematical Foundation**: Linear in parameters, nonlinear in predictors
2. **Model Selection**: AIC, BIC, and cross-validation for degree selection
3. **Orthogonal Polynomials**: Avoiding multicollinearity issues
4. **Diagnostics**: Residual analysis and model validation
5. **Limitations**: Overfitting, extrapolation issues, and global assumptions

The method serves as a foundation for more advanced nonlinear regression techniques like splines and local polynomial methods.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. Springer.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to linear regression analysis. John Wiley & Sons.
