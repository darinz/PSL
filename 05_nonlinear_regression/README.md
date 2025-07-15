# Nonlinear Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This module covers advanced, modernized methods for modeling nonlinear relationships in regression problems. The content has been expanded and clarified for accessibility, with detailed mathematical derivations, code explanations, and improved formatting using inline ($`...`$) and display math (```math) LaTeX. Where possible, image-based equations and text have been converted to selectable, copyable LaTeX in the markdown files for clarity and accessibility.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand the limitations** of linear models and when nonlinear methods are needed
- **Implement polynomial regression** with proper degree selection
- **Construct and interpret spline models** including cubic and natural splines
- **Apply regression splines** with knot selection and basis functions
- **Use smoothing splines** with roughness penalties and cross-validation
- **Implement local regression** methods for flexible curve fitting
- **Choose appropriate methods** for different nonlinear patterns
- **Implement these techniques** in both R and Python

## Topics Covered

### 5.1 Polynomial Regression
- **Introduction to Polynomial Regression**: Extending linear models to capture nonlinear trends
- **Model Specification**: Design matrix construction and basis functions
- **Degree Selection**: Forward and backward approaches for choosing polynomial degree
- **Orthogonal Polynomials**: Using poly() function for numerical stability
- **Global vs Local Limitations**: Understanding when polynomials fail
- **Expanded mathematical derivations and LaTeX formatting**

**Key Concepts:**
- Polynomial basis expansion
- Degree selection strategies
- Orthogonal polynomial construction
- Global function assumptions
- Tail behavior considerations

### 5.2 Cubic Splines
- **Introduction to Splines**: Piecewise polynomial functions
- **Cubic Spline Definition**: Continuity and smoothness conditions
- **Basis Functions**: Truncated power basis and B-splines
- **Natural Cubic Splines**: Linear behavior in boundary regions
- **Degrees of Freedom**: Understanding spline complexity
- **Enhanced code and LaTeX math explanations**

**Key Concepts:**
- Piecewise polynomial construction
- Smoothness constraints at knots
- Basis function representation
- Natural spline boundary conditions
- Degrees of freedom calculation

### 5.3 Regression Splines
- **Basis Expansion Approach**: Linear combination of spline basis functions
- **Knot Selection**: Quantile-based placement and number of knots
- **Model Fitting**: Ordinary least squares with spline basis
- **Cross-Validation**: Selecting optimal number of knots
- **B-spline Implementation**: Using software packages
- **Expanded code and math explanations**

**Key Concepts:**
- Basis function representation
- Knot placement strategies
- Model selection criteria
- Cross-validation for knot selection
- Software implementation

### 5.4 Smoothing Splines
- **Introduction to Smoothing Splines**: Knots at every data point
- **Roughness Penalty**: Second derivative penalty for smoothness
- **Ridge Regression Connection**: Penalized least squares formulation
- **DR Basis**: Double orthogonality and shrinkage interpretation
- **Effective Degrees of Freedom**: Trace of smoother matrix
- **Lambda Selection**: Cross-validation and GCV methods
- **Expanded code and LaTeX math explanations**

**Key Concepts:**
- Roughness penalty approach
- Infinite-dimensional optimization
- Natural cubic spline solution
- Shrinkage interpretation
- Effective degrees of freedom
- Leave-one-out cross-validation

### 5.5 Local Regression
- **Local Smoothers**: Point-wise estimation approach
- **Kernel Methods**: Weighted local polynomial fitting
- **Bandwidth Selection**: Controlling local vs global behavior
- **Implementation**: Using statistical software packages
- **Expanded code and math explanations**

**Key Concepts:**
- Local polynomial fitting
- Kernel weighting functions
- Bandwidth parameter selection
- Point-wise estimation

## Recent Enhancements

- **Expanded Explanations:** All modules now feature clearer, more detailed explanations of mathematical concepts and algorithms.
- **LaTeX Math Formatting:** All math is now formatted using inline ($`...`$) and display (```math) LaTeX for readability and copy-paste support.
- **Code Examples:** Python and R code snippets are provided and explained for all major algorithms.
- **Image-to-Text Conversion:** PNG images containing math or text have been transcribed into markdown with LaTeX where possible, improving accessibility.
- **Visual Aids:** Diagrams and figures are referenced and described in context to support conceptual understanding.

## Code Examples

### Python Implementation

#### Polynomial Regression (`Python_W5_PolynomialRegression.py`)
```python
# Manual polynomial terms
X1 = np.power.outer(age, np.arange(1, 4))
M1 = sm.OLS(wage, sm.add_constant(X1)).fit()

# Orthogonal polynomials
X2, alpha, norm2 = poly(age, 3)
M2 = sm.OLS(wage, sm.add_constant(X2)).fit()

# Prediction with orthogonal polynomials
age_new = poly_predict(age_new, 3, alpha, norm2)
fit2.predict(age_new)
```

#### Regression Splines (`Python_W5_RegressionSpline.py`)
```python
# B-spline basis functions
def bs(x, df=None, knots=None, boundary_knots=None, degree=3):
    # Implementation of B-spline basis
    
# Natural spline basis functions  
def ns(x, df=None, knots=None, boundary_knots=None):
    # Implementation of natural spline basis

# Example usage
F = bs(x, knots=myknots, include_intercept=True)
model = LinearRegression().fit(F, y)
```

#### Smoothing Splines (`Python_W5_SmoothingSpline.html`)
- Interactive HTML notebook with smoothing spline implementation
- Cross-validation for lambda selection
- Effective degrees of freedom calculation
- Performance comparison with other methods

#### Local Regression (`Python_W5_LocalSmoother.html`)
- Interactive HTML notebook with local regression methods
- Kernel smoothing implementation
- Bandwidth selection techniques
- Comparison with global methods

### R Implementation

#### Polynomial Regression (`Rcode_W5_PolynomialRegression.R`)
- Polynomial regression using `poly()` function
- Degree selection with forward/backward approaches
- Model comparison and validation
- Prediction and visualization

#### Regression Splines (`Rcode_W5_RegressionSpline.R`)
- B-spline and natural spline implementation
- Knot selection and placement strategies
- Cross-validation for model selection
- Basis function visualization

#### Smoothing Splines (`Rcode_W5_SmoothingSpline.html`)
- Interactive HTML notebook with R implementation
- Smoothing spline fitting and validation
- Lambda selection methods
- Performance evaluation

#### Local Regression (`Rcode_W5_LocalSmoother.html`)
- Interactive HTML notebook with local regression
- Kernel smoothing methods
- Bandwidth optimization
- Model comparison

## Key Mathematical Concepts

### Polynomial Regression
- **Model**: $`y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d + \varepsilon`$
- **Design Matrix**: $`X = [1, x, x^2, ..., x^d]`$
- **Orthogonal Polynomials**: $`\text{poly}(x, d)`$ for numerical stability

### Cubic Splines
- **Piecewise Definition**: $`f(x)`$ is a cubic polynomial on each interval
- **Smoothness**: Continuous up to second derivatives at knots
- **Degrees of Freedom**: $`m+4`$ for cubic splines, $`m`$ for natural splines
- **Basis Functions**: $`h_0(x) = 1, h_1(x) = x, h_2(x) = x^2, h_3(x) = x^3, h_{i+3}(x) = (x-\xi_i)_+^3`$

### Regression Splines
- **Basis Expansion**: $`y = \sum \beta_i h_i(x) + \varepsilon`$
- **Matrix Form**: $`y = F\beta + \varepsilon`$
- **Estimation**: $`\hat{\beta} = \arg\min \|y - F\beta\|^2`$

### Smoothing Splines
- **Objective Function**: $`\text{RSS}_\lambda(g) = \sum [y_i - g(x_i)]^2 + \lambda \int [g''(x)]^2 dx`$
- **Solution**: Natural cubic spline with knots at data points
- **Matrix Form**: $`\min \|y - F\beta\|^2 + \lambda \beta^T \Omega \beta`$
- **Effective DF**: $`\operatorname{tr}(S_\lambda)`$ where $`S_\lambda = F(F^T F + \lambda \Omega)^{-1} F^T`$

## Practical Applications

### When to Use Each Method

1. **Polynomial Regression**:
   - Simple nonlinear trends
   - Theoretical basis for polynomial relationship
   - Quick exploratory analysis
   - When global function is appropriate

2. **Regression Splines**:
   - Complex nonlinear patterns
   - Need for local flexibility
   - Control over smoothness
   - When knot placement is known or can be optimized

3. **Smoothing Splines**:
   - Automatic smoothness control
   - No need to specify knots
   - Maximum flexibility with regularization
   - When data-driven smoothness is desired

4. **Local Regression**:
   - Highly localized patterns
   - Non-stationary relationships
   - Exploratory data analysis
   - When global assumptions fail

### Data Characteristics

- **Sample Size**: Larger samples support more complex models
- **Noise Level**: Higher noise requires more regularization
- **Pattern Complexity**: Complex patterns need flexible methods
- **Computational Resources**: Smoothing splines can be expensive
- **Interpretability**: Polynomials and splines are more interpretable

## Model Comparison

### Advantages of Nonlinear Methods
- **Flexibility**: Can capture complex relationships
- **Local Adaptability**: Methods like splines adapt to local patterns
- **Automatic Feature Engineering**: Basis functions handle nonlinearity
- **Regularization**: Methods like smoothing splines prevent overfitting

### Limitations
- **Computational Cost**: More expensive than linear methods
- **Parameter Tuning**: Multiple hyperparameters to select
- **Interpretability**: Less interpretable than linear models
- **Extrapolation**: Poor performance outside training range

## Visualization and Analysis

The code examples include comprehensive visualizations:

- **Polynomial Fits**: Different degree polynomials on same data
- **Spline Basis Functions**: Visual representation of basis functions
- **Knot Placement**: Effect of knot location on spline fit
- **Smoothing Parameter**: Effect of lambda on smoothness
- **Cross-Validation Plots**: Model selection using CV
- **Local Regression**: Point-wise estimation and confidence bands

## Getting Started

1. **Review the theoretical foundations** in the markdown files
2. **Start with polynomial regression** to understand basis expansion
3. **Experiment with regression splines** for local flexibility
4. **Try smoothing splines** for automatic smoothness control
5. **Explore local regression** for highly localized patterns
6. **Compare methods** on your own datasets
7. **Practice parameter tuning** and model selection
8. **Reference expanded math/code explanations and LaTeX formatting throughout**

## Additional Resources

- **Textbooks**: Elements of Statistical Learning (ESL), Introduction to Statistical Learning (ISL)
- **Papers**: Original spline and local regression papers
- **Software**: scipy, statsmodels, mgcv, loess packages
- **Online Resources**: Statistical learning course materials and tutorials

## Contributing

Feel free to contribute improvements to the code examples or documentation. This module is designed to be a comprehensive resource for learning nonlinear regression methods in statistical learning. 