# 12. Classification Trees and Boosting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This section covers classification trees and boosting algorithms, building upon the concepts introduced in regression trees while focusing on classification problems.

## Contents

### 12.1. Introduction
- **File**: `01_introduction.md`
- **Topics**:
  - Three essential aspects of classification trees: Where to Split, When to Stop, How to Predict
  - Comparison between regression and classification approaches
  - Goodness-of-split criterion for classification
  - Greedy algorithm for finding optimal splits
  - Expanded mathematical derivations and LaTeX formatting

### 12.2. Impurity Measures
- **File**: `02_impurity_measures.md`
- **Topics**:
  - Definition and properties of impurity measures
  - Goodness-of-split criterion formulation
  - Three main impurity measures:
    - Misclassification Rate
    - Entropy (Deviance)
    - Gini Index
  - Mathematical formulations and practical considerations
  - Enhanced code and math explanations

### 12.3. Misclassification Rate vs. Entropy
- **File**: `03_misclassification.md`
- **Topics**:
  - Mathematical distinctions between misclassification rate and entropy
  - Concavity properties and their implications
  - Visual examples and comparisons
  - Practical recommendations for tree construction vs. pruning
  - Greedy algorithm analysis
  - Expanded code and LaTeX math explanations

### 12.4. AdaBoosting
- **File**: `04_ada-boosting.md`
- **Topics**:
  - AdaBoost algorithm overview
  - Weak classifiers and ensemble methods
  - Mathematical proof of convergence
  - Training error analysis
  - Practical considerations and limitations
  - Expanded code and math explanations

### 12.5. Forward Stagewise Additive Modeling
- **File**: `05_forward_stagewise.md`
- **Topics**:
  - Additive model formulation
  - Forward stagewise optimization
  - AdaBoost with exponential loss
  - Regression with square loss
  - Comparison of boosting algorithms (GBM, XGBoost, CatBoost)
  - Enhanced code and math explanations

## Key Concepts

### Classification Trees
- **Splitting Criteria**: Based on impurity measures rather than residual sum of squares
- **Prediction**: Majority voting or probability-based predictions at leaf nodes
- **Pruning**: Similar to regression trees, using loss plus penalty criteria

### Impurity Measures
- **Misclassification Rate**: $`1 - \max_j p_j`$
- **Entropy**: $`-\sum_{j=1}^K p_j \log p_j`$
- **Gini Index**: $`\sum_{j=1}^K p_j(1-p_j) = 1 - \sum_j p_j^2`$

### Boosting Algorithms
- **AdaBoost**: Combines weak classifiers with exponential loss
- **Forward Stagewise**: Greedy approach to additive modeling
- **Modern Variants**: GBM, XGBoost, CatBoost

## Mathematical Foundations

### Goodness-of-Split Criterion
$`\Phi(j,s) = i(t) - \big [ p_R \cdot i(t_R) + p_L \cdot i(t_L) \big ]`$

### AdaBoost Weight Update
$`\alpha_t = \frac{1}{2} \log \frac{1- \epsilon_t}{ \epsilon_t}`$

### Additive Model
$`f(x) = \alpha_1 g_1(x) + \alpha_2 g_2(x) + \cdots + \alpha_T g_T(x)`$

## Practical Considerations

### When to Use Different Impurity Measures
- **Entropy**: Preferred during tree construction (encourages pure nodes)
- **Misclassification Rate**: Can be used during pruning
- **Gini Index**: Alternative to entropy with similar performance

### Boosting Considerations
- **Overfitting**: AdaBoost may overfit to training data
- **Regularization**: Early stopping often necessary
- **Weak Classifiers**: Can use classifiers worse than random guessing

> **Note:** Where images previously contained mathematical expressions or text, these have been transcribed into markdown with LaTeX for clarity and accessibility. Visuals are now referenced in context to support the expanded explanations.

## Related Sections
- **Week 4**: Regression Trees and Ensemble Methods
- **Week 10**: Logistic Regression
- **Week 11**: Support Vector Machine

## Code Resources
- Python implementations available for classification trees
- R code examples for boosting algorithms
- Practical examples with real datasets
