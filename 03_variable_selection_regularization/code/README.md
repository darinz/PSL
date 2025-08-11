# Variable Selection and Regularization Code Examples

This folder contains Python code examples that demonstrate the concepts covered in the Variable Selection and Regularization section of the Statistical Learning course.

## Files Overview

### Subset Selection

- **`error_decomposition.py`** - Demonstrates error decomposition in variable selection with comprehensive visualization and analysis.

- **`model_selection_criteria.py`** - Demonstrates model selection criteria (AIC, BIC, Mallow's Cp) with comprehensive analysis and visualization.

- **`aic_bic_comparison.py`** - Demonstrates AIC vs BIC comparison across different sample sizes with comprehensive analysis and visualization.

- **`screening_stepwise.py`** - Demonstrates screening and stepwise selection for high-dimensional variable selection with comprehensive analysis and visualization.

- **`search_algorithms.py`** - Demonstrates different search algorithms for variable selection (exhaustive, forward, backward, stepwise) with comprehensive analysis and visualization.

### Regularization

- **`regularization_comparison.py`** - Demonstrates ridge vs lasso regularization comparison with comprehensive analysis and visualization.

- **`regularization_comparison.R`** - Demonstrates ridge vs lasso regularization comparison using the glmnet package with comprehensive analysis and visualization.

- **`cross_validation_selection.py`** - Demonstrates cross-validation for regularization parameter selection with comprehensive analysis and visualization.

- **`ridge_regression_detailed.py`** - Demonstrates comprehensive ridge regression with multicollinearity handling, SVD analysis, and augmented data interpretation.

- **`ridge_regression_detailed.R`** - Demonstrates comprehensive ridge regression with multicollinearity handling using the glmnet package.

### Existing Files

- **`Python_W3_VarSel_SubsetSelection.py`** - Original subset selection implementation.

- **`Python_W3_VarSel_RidgeLasso.py`** - Original ridge and lasso regression implementation.

- **`R_W3_VarSel_SubsetSelection.R`** - R implementation of subset selection.

- **`Rcode_W3_VarSel_RidgeLasso.R`** - R implementation of ridge and lasso regression.

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
- pandas
- itertools
- statsmodels (for some examples)

## Running the Examples

To run any example, simply execute the Python file:

```bash
python filename.py
```

For example:
```bash
python error_decomposition.py
```

## Key Concepts Demonstrated

### Error Decomposition
- Training vs test error analysis
- Bias-variance tradeoff in variable selection
- Overfitting demonstration
- Theoretical error decomposition

### Model Selection Criteria
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Mallow's Cp
- Comparison of different selection criteria
- Penalty term analysis

### AIC vs BIC Comparison
- Sample size effects on model selection
- Penalty term behavior
- Theoretical insights
- Practical decision guidelines

## Notes

- All examples use synthetic data for reproducibility
- Random seeds are set where appropriate to ensure consistent results
- The code is designed to be educational and well-commented
- Visualizations are included to help understand the concepts
- Mathematical properties are verified numerically where possible
