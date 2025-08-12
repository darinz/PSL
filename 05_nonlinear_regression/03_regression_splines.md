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

**Complete Implementation:** [regression_spline_implementation.py](code/regression_spline_implementation.py)

The Python implementation includes:

- **RegressionSpline Class**: Complete implementation with support for cubic and natural splines, regularization (Ridge/Lasso), and automatic knot placement
- **Basis Functions**: Truncated power basis for cubic splines and natural cubic spline basis using scipy
- **Model Selection**: Cross-validation for optimal degrees of freedom selection
- **Comprehensive Visualization**: 6-panel demonstration including spline fits, model comparison, cross-validation, basis functions, residuals, and regularization effects
- **Real Data Analysis**: Birthrate data analysis with natural splines and model diagnostics

Key features:
- Support for both cubic and natural cubic splines
- Automatic knot placement using quantiles
- Ridge and Lasso regularization options
- Cross-validation for model selection
- Comprehensive diagnostic tools and model comparison
- Integration with scipy for robust spline implementation

### R Implementation

**Complete Implementation:** [r_regression_splines.R](code/r_regression_splines.R)

The R implementation includes:

- **fit_regression_spline()**: Flexible function supporting cubic and natural splines with specified degrees of freedom
- **demonstrate_regression_splines_r()**: Comprehensive demonstration with ggplot2 visualizations and model comparison
- **analyze_birthrate_data_r()**: Real data analysis with natural splines and cross-validation
- **demonstrate_advanced_features_r()**: Advanced features including spline type comparison
- **compare_with_other_methods_r()**: Comparison with linear regression, polynomial regression, and other methods

Key features:
- Integration with R's built-in `bs()` and `ns()` functions for robust spline implementation
- ggplot2-based visualizations for publication-quality plots
- Cross-validation for optimal degrees of freedom selection
- Model comparison and diagnostics
- Support for both cubic and natural splines
- Comprehensive demonstration functions with synthetic and real data analysis

## 5.3.5. Advanced Topics

### Advanced Utilities

**Complete Implementation:** [advanced_regression_utilities.py](code/advanced_regression_utilities.py)

The advanced utilities include:

- **Model Selection Functions**: 
  - `select_optimal_df_information_criteria()`: AIC and BIC for optimal degrees of freedom selection
  - `compare_regularization_methods()`: Ridge and Lasso regularization comparison
- **Confidence Intervals**: `compute_confidence_intervals()` for prediction uncertainty quantification
- **Comprehensive Diagnostics**: `comprehensive_spline_diagnostics()` with 6-panel diagnostic plots
- **Advanced Demonstrations**: `demonstrate_advanced_features()` showing information criteria, regularization, and model comparison

Key features:
- Information criteria (AIC/BIC) for model selection
- Ridge and Lasso regularization with coefficient comparison
- Confidence interval calculation with proper standard error estimation
- Comprehensive diagnostic suite including residuals, leverage, and Cook's distance
- Advanced visualization and model comparison tools

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
