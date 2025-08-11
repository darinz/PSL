# Introduction Code Examples

This folder contains Python code examples that demonstrate the concepts covered in the Introduction section of the Statistical Learning course.

## Files Overview

### Core Concepts

- **`curse_of_dimensionality_demo.py`** - Demonstrates how kNN performance degrades as the number of features increases, illustrating the curse of dimensionality.

- **`ridge_regression_grid_search.py`** - Shows how to use GridSearchCV for hyperparameter tuning with Ridge regression.

### Bias-Variance Tradeoff

- **`polynomial_regression_bias_variance.py`** - Demonstrates polynomial regression with different degrees showing bias-variance tradeoff.

- **`bias_variance_decomposition.py`** - Calculates and visualizes the bias-variance decomposition for different polynomial degrees.

- **`double_descent_phenomenon.py`** - Demonstrates the double descent phenomenon in linear regression as the number of features increases.

- **`regularization_effect.py`** - Shows the effect of different regularization strengths on Ridge and Lasso regression performance.

- **`cross_validation_model_selection.py`** - Demonstrates how to use cross-validation for hyperparameter tuning with Ridge regression.

- **`bagging_vs_single_model.py`** - Compares the performance of a single decision tree versus a bagging ensemble.

- **`early_stopping_neural_network.py`** - Demonstrates early stopping in neural networks to prevent overfitting.

- **`complexity_bounds.py`** - Calculates complexity bounds for different types of models.

- **`simple_vs_complex_models.py`** - Demonstrates when to use simple vs complex models based on dataset size.

- **`systematic_model_selection.py`** - Shows systematic model selection with increasing complexity.

### Learning Theory

- **`cross_validation_learning_theory.py`** - Demonstrates 5-fold cross-validation with Ridge regression for different regularization strengths.

### Linear Regression and kNN

- **`knn_k_selection.py`** - Demonstrates how to select the optimal k value for kNN using cross-validation.

- **`linear_vs_polynomial_regression.py`** - Shows how linear regression fails on non-linear data while polynomial regression can capture the relationship.

- **`knn_nonlinear_boundary.py`** - Demonstrates how kNN can capture complex non-linear decision boundaries.

- **`curse_of_dimensionality_demo.py`** - Shows how distances become less meaningful as dimensionality increases.

- **`gaussian_mixture_model.py`** - Demonstrates fitting a Gaussian Mixture Model to multi-modal data.

- **`knn_implementation.py`** - Provides a function to evaluate kNN for different k values.

- **`linear_regression_implementation.py`** - Provides a function to evaluate linear regression for classification tasks.

- **`performance_comparison.py`** - Provides functions to visualize and compare the performance of kNN and linear regression.

### Bayes Rule

- **`bayes_classifier_simple.py`** - Demonstrates the Bayes classifier for simple Gaussian distributions.

- **`bayes_classifier_mixture.py`** - Demonstrates the Bayes classifier for mixture Gaussian distributions.

- **`bayes_decision_boundary.py`** - Shows how to visualize the Bayes decision boundary for different scenarios.

## Usage

Each Python file can be run independently to demonstrate the specific concept it covers. Most files include:

- Data generation or loading
- Model fitting and evaluation
- Visualization of results
- Print statements showing key metrics

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
python curse_of_dimensionality_demo.py
```

## Notes

- All examples use synthetic data for reproducibility
- Random seeds are set where appropriate to ensure consistent results
- The code is designed to be educational and well-commented
- Visualizations are included to help understand the concepts
