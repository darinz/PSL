# Logistic Regression Module

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

## Overview

This module provides a comprehensive, modernized introduction to logistic regression, a fundamental classification technique in statistical learning.

## Module Structure

### 1. Setup and Fundamentals
- **File**: `01_setup.md`
- **Topics**: Binary classification, link functions, logit transformation, loss function design
- **Key Concepts**: Probability modeling, linear model constraints, sigmoid function
- **Expanded mathematical derivations and LaTeX formatting**

### 2. Maximum Likelihood Estimation
- **File**: `02_mle.md`
- **Topics**: MLE derivation, Newton-Raphson algorithm, Hessian matrix, reweighted least squares
- **Key Concepts**: Likelihood function, iterative optimization, convergence properties
- **Expanded code and math explanations**

### 3. Separable Data Challenges
- **File**: `03_seperable_data.md`
- **Topics**: Well-separated data, convergence issues, decision boundaries, regularization limitations
- **Key Concepts**: Perfect separation, coefficient growth, model interpretation
- **Enhanced visual and LaTeX math explanations**

### 4. Retrospective Sampling
- **File**: `04_retrospective_sampling_data.md`
- **Topics**: Multinomial extension, nonlinear models, sampling bias, population inference
- **Key Concepts**: Sampling methodology, coefficient consistency, intercept adjustment
- **Expanded code and math explanations**

## Recent Enhancements

- **Expanded Explanations:** All modules now feature clearer, more detailed explanations of mathematical concepts and algorithms.
- **LaTeX Math Formatting:** All math is now formatted using inline ($`...`$) and display (```math) LaTeX for readability and copy-paste support.
- **Code Examples:** Python and R code snippets are provided and explained for all major algorithms.
- **Image-to-Text Conversion:** PNG images containing math or text have been transcribed into markdown with LaTeX where possible, improving accessibility.
- **Visual Aids:** Diagrams and figures are referenced and described in context to support conceptual understanding.

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

$`P(Y=1|X=x) = \frac{\exp(x^T\beta)}{1 + \exp(x^T\beta)}`$

### Logit Transformation
The logit link function transforms probabilities to unconstrained values:

$`\text{logit}(\eta(x)) = \log\frac{\eta(x)}{1 - \eta(x)} = x^T\beta`$

### Likelihood Function
For binary outcomes, the likelihood is:

$`L(\beta) = \prod_{i=1}^n \sigma(x_i^T\beta)^{y_i}(1 - \sigma(x_i^T\beta))^{1-y_i}`