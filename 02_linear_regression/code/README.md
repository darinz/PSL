# Linear Regression Code Examples

This folder contains Python code examples that demonstrate the concepts covered in the Linear Regression section of the Statistical Learning course.

## Files Overview

### Core Linear Regression

- **`least_squares_estimation.py`** - Demonstrates least squares estimation using the normal equation.

- **`linear_regression_analysis.py`** - Complete linear regression analysis including coefficient estimation, standard errors, and goodness-of-fit measures.

- **`diagnostic_plots.py`** - Creates diagnostic plots for linear regression including residuals vs fitted, Q-Q plots, scale-location plots, and residuals vs leverage.

- **`statistical_inference.py`** - Performs statistical inference including confidence intervals, hypothesis tests, and F-tests for linear regression.

### Model Assessment

- **`cross_validation_assessment.py`** - Demonstrates cross-validation assessment for linear regression using scikit-learn.

- **`data_preprocessing.py`** - Shows data preprocessing techniques including centering and scaling for linear regression.

- **`multicollinearity_check.py`** - Computes Variance Inflation Factors (VIF) to detect multicollinearity.

### Model Selection

- **`forward_selection.py`** - Implements forward stepwise selection for variable selection in linear regression.

- **`polynomial_regression.py`** - Demonstrates polynomial regression to capture non-linear relationships.

### Geometric Interpretation

- **`vector_operations.py`** - Demonstrates basic vector operations in 3D space with visualization.

- **`orthogonal_projection.py`** - Shows orthogonal projection properties in linear regression.

- **`hat_matrix.py`** - Demonstrates the hat matrix and its properties including leverage values.

## Usage

Each Python file can be run independently to demonstrate the specific concept it covers. Most files include:

- Data generation or loading
- Model fitting and evaluation
- Visualization of results
- Print statements showing key metrics and properties

## Dependencies

The code examples require the following Python packages:
- numpy
- matplotlib
- scikit-learn
- scipy

## Running the Examples

To run any example, simply execute the Python file:

```bash
python filename.py
```

For example:
```bash
python least_squares_estimation.py
```

## Key Concepts Demonstrated

### Mathematical Foundations
- Normal equations and least squares estimation
- Matrix operations and linear algebra
- Orthogonal projection and geometric interpretation
- Hat matrix properties and leverage

### Statistical Inference
- Coefficient estimation and standard errors
- Confidence intervals and hypothesis testing
- F-tests for overall model significance
- Diagnostic plots and residual analysis

### Model Assessment
- R-squared and adjusted R-squared
- Cross-validation for model evaluation
- Multicollinearity detection
- Data preprocessing techniques

### Model Selection
- Forward stepwise selection
- Polynomial regression for non-linear relationships
- Variable selection criteria

## Notes

- All examples use synthetic data for reproducibility
- Random seeds are set where appropriate to ensure consistent results
- The code is designed to be educational and well-commented
- Visualizations are included to help understand geometric concepts
- Mathematical properties are verified numerically where possible
