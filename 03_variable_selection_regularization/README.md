# Variable Selection and Regularization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This module covers essential, modernized techniques for variable selection and regularization in statistical learning, focusing on methods to handle high-dimensional data and improve model performance through feature selection and coefficient shrinkage. The content has been expanded and clarified for accessibility, with detailed mathematical derivations, code explanations, and improved formatting using inline ($`...`$) and display math (```math) LaTeX. Where possible, image-based equations and text have been converted to selectable, copyable LaTeX in the markdown files for clarity and accessibility.

## Learning Objectives

By the end of this module, you will be able to:

- **Understand the motivation** behind variable selection and regularization
- **Compare different model selection criteria** (AIC, BIC, Mallow's Cp)
- **Implement subset selection methods** (best subset, forward, backward, stepwise)
- **Apply regularization techniques** (Ridge and Lasso regression)
- **Interpret the geometric and optimization perspectives** of regularization
- **Choose appropriate methods** for different data scenarios
- **Implement these techniques** in both R and Python

## Topics Covered

### 3.1 Subset Selection
- **Why Subset Selection**: Understanding the bias-variance trade-off and the need for variable selection
- **Selection Criteria**: AIC, BIC, and Mallow's Cp for model comparison
- **AIC vs BIC**: Philosophical differences and practical implications
- **Search Algorithms**: Level-wise search, greedy algorithms (forward, backward, stepwise)
- **Variable Screening**: Handling cases where p > n
- **Expanded mathematical derivations and LaTeX formatting**

### 3.2 Regularization Framework
- **Unified Objective Function**: Framing variable selection as optimization problems
- **L0, L1, and L2 Penalties**: Understanding different regularization approaches
- **Data Preprocessing**: Centering and scaling for consistent results
- **Scale Invariance**: Ensuring methods work regardless of variable scaling
- **Enhanced geometric and optimization explanations**

### 3.3 Ridge Regression
- **Introduction**: L2 penalty and quadratic optimization
- **Shrinkage Effect**: Understanding coefficient shrinkage in orthogonal and non-orthogonal cases
- **SVD Perspective**: Using singular value decomposition to understand ridge behavior
- **Degree of Freedom**: Effective degrees of freedom and model complexity
- **Expanded code and math explanations**

### 3.4 Lasso Regression
- **Introduction**: L1 penalty and sparse solutions
- **Soft Thresholding**: Understanding the one-dimensional lasso solution
- **Lasso vs Ridge**: Geometric and optimization perspectives
- **Coordinate Descent**: Algorithm for solving lasso problems
- **Uniqueness**: Conditions for unique lasso solutions
- **Expanded code and LaTeX math explanations**

### 3.5 Discussion and Comparison
- **Method Selection**: When to use each approach
- **Simulation Studies**: Comparing methods on different data scenarios
- **Practical Guidelines**: Choosing methods based on data characteristics
- **Visual and LaTeX math enhancements**

## Code Examples

### Python Implementation

#### Subset Selection (`Python_W3_VarSel_SubsetSelection.py`)
```python
# Best subset selection with AIC/BIC
def bestsubset(X, Y):
    """Find best subset for each model size"""
    # Implementation details...

# Stepwise selection
def stepAIC(X, Y, features=None, AIC=True):
    """Stepwise selection based on AIC or BIC"""
    # Implementation details...
```

#### Ridge and Lasso (`Python_W3_VarSel_RidgeLasso.py`)
```python
# Ridge regression with cross-validation
ridgecv = RidgeCV(alphas=alphas, cv=10, scoring='neg_mean_squared_error')
ridgecv.fit(X_train, Y_train)

# Lasso regression with cross-validation
lassocv = LassoCV(alphas=alphas, cv=10)
lassocv.fit(X_train, Y_train)
```

### R Implementation

#### Subset Selection (`R_W3_VarSel_SubsetSelection.R`)
- Best subset selection using `leaps` package
- AIC/BIC model selection
- Stepwise selection procedures

#### Ridge and Lasso (`Rcode_W3_VarSel_RidgeLasso.R`)
- Ridge regression with `glmnet`
- Lasso regression with cross-validation
- Coefficient path analysis
- Model comparison and evaluation

## Key Mathematical Concepts

### Model Selection Criteria
- **AIC**: $`-2\log(\text{likelihood}) + 2p`$
- **BIC**: $`-2\log(\text{likelihood}) + \log(n)p`$
- **Mallow's Cp**: $`\text{RSS} + 2\sigma^2 p`$

### Regularization Objectives
- **Ridge**: $`\min \|y - X\beta\|^2 + \lambda\|\beta\|^2`$
- **Lasso**: $`\min \|y - X\beta\|^2 + \lambda\|\beta\|_1`$

### Soft Thresholding (Lasso)
```math
\hat{\beta}_j^{lasso} = \operatorname{sign}(\hat{\beta}_j^{LS})\left(|\hat{\beta}_j^{LS}| - \frac{\lambda}{2}\right)_+
```

## Practical Applications

### When to Use Each Method

1. **Subset Selection**:
   - Small to moderate number of predictors (p < 40)
   - Need for interpretable models
   - Computational resources available

2. **Ridge Regression**:
   - Multicollinearity present
   - All variables potentially relevant
   - Need for stable coefficient estimates

3. **Lasso Regression**:
   - High-dimensional data (p > n)
   - Sparse solutions desired
   - Variable selection needed

### Data Scenarios

- **X1**: Small, curated feature set → Full model often sufficient
- **X2**: Correlated but relevant features → Ridge/PCR effective
- **X3**: Many noise features → Lasso preferred for variable selection

## Visualization and Analysis

The code examples include comprehensive visualizations:

- **Coefficient Paths**: How coefficients change with regularization strength
- **Cross-Validation Paths**: Model selection using CV
- **Model Comparison Plots**: AIC/BIC vs model size
- **Performance Evaluation**: Test MSE comparisons

## Getting Started

1. **Review the theoretical foundations** in the markdown files
2. **Run the Python examples** to understand implementation
3. **Experiment with the R code** for additional insights
4. **Try the simulation study** to compare methods
5. **Apply to your own datasets** to gain practical experience
6. **Reference expanded math/code explanations and LaTeX formatting throughout**

## Additional Resources

- **Textbooks**: Elements of Statistical Learning (ESL)
- **Papers**: Original lasso and ridge regression papers
- **Software**: scikit-learn, glmnet, leaps packages
- **Online Resources**: Statistical learning course materials

## Contributing

Feel free to contribute improvements to the code examples or documentation. This module is designed to be a comprehensive resource for learning variable selection and regularization techniques in statistical learning. 