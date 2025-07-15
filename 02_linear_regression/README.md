# Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This folder contains comprehensive, modernized materials on linear regression, covering both theoretical foundations and practical implementation. The content has been expanded and clarified for accessibility, with detailed mathematical derivations, code explanations, and improved formatting using inline ($`...`$) and display math (```math) LaTeX. Where possible, image-based equations and text have been converted to selectable, copyable LaTeX in the markdown files for clarity and accessibility.

## Contents

### Theory Documents

1. **`01_mulitple_linear_regression.md`** – Core concepts of multiple linear regression
   - Matrix representation and notation
   - Least squares principle and normal equations
   - Classical vs. modern settings (n >> p vs p >> n)
   - Fitted values, residuals, and variance estimation
   - Expanded mathematical derivations and LaTeX formatting

2. **`02_geometric_interpretation.md`** – Geometric understanding of linear regression
   - Vector spaces and linear subspaces
   - Projection and orthogonality
   - R² (coefficient of determination) interpretation
   - Geometric properties of least squares
   - Enhanced visual and LaTeX math explanations

3. **`03_practical_issues.md`** – Real-world implementation considerations
   - Data analysis with R/Python
   - Coefficient interpretation and multicollinearity
   - Hypothesis testing (F-tests and t-tests)
   - Model diagnostics and validation
   - Expanded code and practical examples

### Code Examples

#### Python Implementation

1. **`Python_W2_LinearRegression_1.py`** – Basic linear regression analysis
   - Data loading and exploration using Prostate cancer dataset
   - Multiple implementation approaches (sklearn, statsmodels, numpy)
   - Model interpretation and evaluation
   - Prediction with new data
   - Impact of irrelevant variables
   - Step-by-step code explanations

2. **`Python_W2_LinearRegression_2.py`** – Advanced concepts and analysis
   - Training vs test error analysis (bias-variance tradeoff)
   - Coefficient interpretation in simple vs multiple regression
   - Partial regression coefficients (Frisch-Waugh-Lovell theorem)
   - Hypothesis testing and model comparison
   - Collinearity detection and analysis
   - Expanded code and math explanations

#### R Implementation

**`Rcode_W2_LinearRegression.R`** – Comprehensive R analysis
- Data loading and exploratory analysis
- Linear model fitting and interpretation
- Manual calculation verification
- Prediction methods
- Training vs test error analysis
- Coefficient interpretation and partial effects
- Hypothesis testing and model diagnostics
- Enhanced code comments and explanations

### Visualizations

The `img/` folder contains supporting visualizations:
- `w2_coef_X_corr.png` – Coefficient correlation analysis
- `w2_projection.png` – Geometric projection concepts
- `w2_example_subspace.png` – Linear subspace examples
- `w2_Xr.png` – Design matrix and residuals
- `w2_LS.png` – Least squares visualization
- `w2_large_p_small_n.png` – High-dimensional setting
- `w2_large_n_small_p.png` – Classical setting

> **Note:** Where images previously contained mathematical expressions or text, these have been transcribed into markdown with LaTeX for clarity and accessibility. Visuals are now referenced in context to support the expanded explanations.

## Recent Enhancements

- **Expanded Explanations:** All modules now feature clearer, more detailed explanations of mathematical concepts and algorithms.
- **LaTeX Math Formatting:** All math is now formatted using inline ($`...`$) and display (```math) LaTeX for readability and copy-paste support.
- **Code Examples:** Python and R code snippets are provided and explained for all major algorithms.
- **Image-to-Text Conversion:** PNG images containing math or text have been transcribed into markdown with LaTeX where possible, improving accessibility.
- **Visual Aids:** Diagrams and figures are referenced and described in context to support conceptual understanding.

## Learning Objectives

By working through this material, you will:

### Theoretical Understanding
- **Matrix Formulation:** Understand the matrix representation of linear regression
- **Geometric Interpretation:** Visualize regression as projection in vector spaces
- **Least Squares:** Comprehend the mathematical foundation of OLS estimation
- **Model Assessment:** Learn to interpret R², residuals, and model diagnostics

### Practical Skills
- **Data Analysis:** Load, explore, and preprocess real datasets
- **Model Fitting:** Implement linear regression using multiple approaches
- **Prediction:** Make predictions and assess model performance
- **Diagnostics:** Detect and handle multicollinearity and other issues

### Advanced Concepts
- **Bias-Variance Tradeoff:** Understand overfitting through training vs test error
- **Partial Effects:** Apply the Frisch-Waugh-Lovell theorem
- **Hypothesis Testing:** Perform F-tests and t-tests for model validation
- **Model Comparison:** Compare nested models and assess significance

## Getting Started

### Prerequisites
- Basic knowledge of linear algebra and statistics
- Familiarity with Python (numpy, pandas, sklearn, matplotlib) or R
- Understanding of basic probability concepts

### Recommended Order
1. Start with `01_mulitple_linear_regression.md` for theoretical foundations
2. Read `02_geometric_interpretation.md` for geometric insights
3. Review `03_practical_issues.md` for implementation considerations
4. Run the Python or R code examples to practice implementation
5. Study the visualizations in the `img/` folder
6. Reference expanded math/code explanations and LaTeX formatting throughout

### Key Datasets
- **Prostate Cancer Dataset:** Used throughout the examples
  - Source: Elements of Statistical Learning website
  - Purpose: Examine correlation between PSA levels and clinical measures
  - Variables: 8 predictors + 1 response (log PSA)

## Key Concepts Covered

### Mathematical Foundations
- Multiple linear regression model: $`y = X\beta + \varepsilon`$
- Least squares estimation: $`\hat{\beta} = (X^T X)^{-1} X^T y`$
- Normal equations and matrix algebra
- Degrees of freedom and variance estimation

### Geometric Insights
- Column space of design matrix
- Orthogonal projection of response vector
- Pythagorean theorem in regression context
- R² as ratio of explained to total variance

### Practical Considerations
- Data preprocessing and exploration
- Model interpretation and coefficient meaning
- Multicollinearity detection and handling
- Training vs test error analysis
- Hypothesis testing and model validation

## Code Features

### Python Implementation
- **Multiple Approaches:** sklearn, statsmodels, and manual numpy implementation
- **Comprehensive Analysis:** Data exploration, model fitting, diagnostics
- **Educational Focus:** Step-by-step explanations and manual calculations
- **Visualization:** Correlation matrices, error analysis plots, diagnostic plots

### R Implementation
- **Base R Functions:** Uses `lm()`, `summary()`, `predict()` functions
- **Manual Verification:** Educational manual calculations to verify R's output
- **Diagnostic Tools:** Leverages R's comprehensive diagnostic capabilities
- **Statistical Testing:** F-tests, t-tests, and model comparison methods

## Expected Outcomes

After completing this material, you should be able to:
- Formulate and solve linear regression problems mathematically
- Implement regression analysis in both Python and R
- Interpret model coefficients and assess model fit
- Handle common practical issues in regression modeling
- Understand the geometric and algebraic foundations of least squares
- Apply hypothesis testing and model validation techniques

## Further Reading

The materials reference concepts from:
- **Elements of Statistical Learning** (ESL) – Hastie, Tibshirani, Friedman
- **Introduction to Statistical Learning** (ISL) – James, Witten, Hastie, Tibshirani
- **Applied Linear Regression Models** – Kutner, Nachtsheim, Neter

## Contributing

This material is part of a comprehensive statistical learning course. For questions or suggestions, please refer to the main course documentation. 