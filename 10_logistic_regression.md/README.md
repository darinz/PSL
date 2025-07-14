# Logistic Regression Module

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## Overview

This module provides a comprehensive introduction to logistic regression, a fundamental classification technique in statistical learning. Logistic regression models the probability of a binary outcome using a logistic function, making it one of the most widely used methods for binary classification problems.

## Module Structure

### 1. Setup and Fundamentals
- **File**: `01_setup.md`
- **Topics**: Binary classification, link functions, logit transformation, loss function design
- **Key Concepts**: Probability modeling, linear model constraints, sigmoid function

### 2. Maximum Likelihood Estimation
- **File**: `02_mle.md`
- **Topics**: MLE derivation, Newton-Raphson algorithm, Hessian matrix, reweighted least squares
- **Key Concepts**: Likelihood function, iterative optimization, convergence properties

### 3. Separable Data Challenges
- **File**: `03_seperable_data.md`
- **Topics**: Well-separated data, convergence issues, decision boundaries, regularization limitations
- **Key Concepts**: Perfect separation, coefficient growth, model interpretation

### 4. Retrospective Sampling
- **File**: `04_retrospective_sampling_data.md`
- **Topics**: Multinomial extension, nonlinear models, sampling bias, population inference
- **Key Concepts**: Sampling methodology, coefficient consistency, intercept adjustment

## Prerequisites

- Basic understanding of linear algebra and calculus
- Familiarity with probability theory and statistical inference
- Knowledge of optimization concepts
- Experience with Python or R programming

## Learning Objectives

By the end of this module, students will be able to:

1. **Understand the mathematical foundations** of logistic regression
2. **Derive the maximum likelihood estimator** using iterative algorithms
3. **Handle edge cases** such as separable data and convergence issues
4. **Apply logistic regression** to real-world classification problems
5. **Interpret model coefficients** and assess model performance
6. **Address sampling biases** in retrospective studies

## Code Examples

### Python Implementation
- **File**: `Python_W10_LogisticReg.html`
- **Content**: Complete Python implementation with scikit-learn
- **Features**: Model fitting, prediction, evaluation metrics

### R Implementation
- **File**: `Rcode_W10_LogisticReg.html`
- **Content**: R implementation using glm() function
- **Features**: Statistical analysis, diagnostic plots, model validation

### Phoneme Classification Example
- **Python**: `Python_W10_LogisticReg_Phoneme.html`
- **R**: `Rcode_W10_LogisticReg_Phoneme.html`
- **Application**: Real-world phoneme classification problem

## Mathematical Framework

### Core Model
The logistic regression model is defined as:

$$P(Y=1|X=x) = \frac{\exp(x^T\beta)}{1 + \exp(x^T\beta)}$$

### Logit Transformation
The logit link function transforms probabilities to unconstrained values:

$$\text{logit}(\eta(x)) = \log\frac{\eta(x)}{1 - \eta(x)} = x^T\beta$$

### Likelihood Function
For binary outcomes, the likelihood is:

$$L(\beta) = \prod_{i=1}^n \sigma(x_i^T\beta)^{y_i}(1 - \sigma(x_i^T\beta))^{1-y_i}$$

## Key Algorithms

### Newton-Raphson Method
The iterative algorithm for finding MLE:

$$\beta^{(t+1)} = \beta^{(t)} - H^{-1}(\beta^{(t)}) \nabla l(\beta^{(t)})$$

### Reweighted Least Squares
Alternative formulation for computational efficiency:

$$\beta^{(t+1)} = (X^TW^{(t)}X)^{-1}X^TW^{(t)}z^{(t)}$$

## Practical Considerations

### Convergence Issues
- **Separable Data**: When classes are perfectly separated, coefficients may grow without bound
- **Solution**: Use regularization or accept that decision boundary is well-defined despite convergence warnings

### Model Assessment
- **Goodness of Fit**: Deviance, AIC, BIC
- **Classification Performance**: Accuracy, precision, recall, ROC curves
- **Residual Analysis**: Deviance residuals, Pearson residuals

### Sampling Considerations
- **Retrospective Sampling**: Adjust for sampling bias in case-control studies
- **Coefficient Consistency**: Main effects remain consistent, intercept requires adjustment

## Applications

Logistic regression is widely used in:

- **Medical Research**: Disease prediction, risk assessment
- **Marketing**: Customer churn prediction, response modeling
- **Finance**: Credit scoring, fraud detection
- **Social Sciences**: Survey analysis, behavioral prediction

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
3. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*. Chapman & Hall.

## Related Modules

- **Linear Regression**: Foundation for understanding regression concepts
- **Regularization**: Ridge and Lasso regression for high-dimensional data
- **Discriminant Analysis**: Alternative classification approaches
- **Model Selection**: Techniques for choosing optimal models

## Acknowledgments

This module builds upon classical statistical theory and modern machine learning practices. Special thanks to the statistical learning community for developing robust implementations and comprehensive documentation.

---

*For questions or contributions, please refer to the main course documentation or contact the course instructors.* 