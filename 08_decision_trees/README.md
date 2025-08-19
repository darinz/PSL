# Decision Trees

[![Topic](https://img.shields.io/badge/Topic-Decision%20Trees-blue.svg)]()

> **Good Resource**: Before diving into this section, it's recommended to review Chapter 1 on Decision Trees from [A Course in Machine Learning](https://github.com/darinz/ML/blob/a7d4a50c31fc0301ecc1e90c3638d4ad6e4fbbbb/reference/ciml-v0_99-all.pdf) for foundational concepts.

## Overview

This directory contains comprehensive materials on decision trees, a fundamental machine learning algorithm that creates hierarchical decision structures by recursively partitioning the input space. Decision trees serve as the foundation for more advanced ensemble methods like Random Forests and Gradient Boosting.

## Learning Objectives

By the end of this module, you will understand:

- **Fundamental Concepts**: How decision trees partition input space into regions
- **Tree Construction**: The recursive splitting process and greedy optimization
- **Loss Functions**: Entropy, Gini impurity, and squared error for classification and regression
- **Regularization**: Techniques to prevent overfitting (pruning, depth limits, leaf size)
- **Limitations**: Axis-aligned boundaries and lack of additive structure
- **Ensemble Methods**: Introduction to Random Forests and Boosting

## Course Materials

### Core Content
- **`01_introduction.md`** - Comprehensive introduction to decision trees covering:
  - Nonlinear classification capabilities
  - Recursive region splitting
  - Loss functions (entropy, Gini impurity, squared error)
  - Regularization techniques
  - Computational complexity
  - Limitations and ensemble methods

### Reference Materials
- **`N-01-2_CS229_Decision-Trees-Notes.pdf`** - Stanford CS229 lecture notes on decision trees
- **`N-02_lecture11-decision-trees.pdf`** - Lecture 11 slides on decision trees
- **`N-03_lecture11-decision-tree-overfitting.pdf`** - Lecture on overfitting in decision trees
- **`N-04_lecture11-boosting.pdf`** - Lecture on boosting algorithms
- **`N-05_cs229-boosting_slides.pdf`** - CS229 boosting slides
- **`N-06_boosting.pdf`** - Additional boosting reference material

### Visual Aids
- **`img/nonlinear-classifier.png`** - Illustration of decision trees as nonlinear classifiers
- **`img/split_1.png`**, **`img/split_2.png`**, **`img/split_3.png`** - Step-by-step visualization of recursive splitting
- **`img/lack_structure.png`** - Demonstration of decision trees' limitations with diagonal boundaries

### Code Examples
- **`code/boosting_example.m`** - MATLAB implementation of boosting algorithm

## Key Concepts

### 1. Input Space Partitioning
Decision trees divide the input space $\mathcal{X}$ into disjoint regions:

$$\mathcal{X} = \bigcup_{i=0}^{n} R_i$$

where $R_i \cap R_j = \emptyset$ for $i \neq j$

### 2. Recursive Splitting
Each split creates two child regions based on a feature threshold:

$$R_1 = \{X \mid X_j < t, X \in R_p\}$$
$$R_2 = \{X \mid X_j \geq t, X \in R_p\}$$

### 3. Loss Functions

#### Classification (Entropy Loss)
$$L_{\text{cross}}(R) = - \sum_c \hat{p}_c \log_2 \hat{p}_c$$

#### Classification (Gini Impurity)
$$I_G(\hat{p}) = \sum_{i=1}^{c} \hat{p}_i (1 - \hat{p}_i)$$

#### Regression (Squared Error)
$$L_{\text{squared}}(R) = \frac{\sum_{i \in R} (y_i - \hat{y})^2}{|R|}$$

### 4. Information Gain
The quality of a split is measured by:

$$L(R_p) - \frac{|R_1|L(R_1) + |R_2|L(R_2)}{|R_1| + |R_2|}$$

### 5. Regularization Techniques
- **Minimum leaf size**: Prevent splits on small regions
- **Maximum depth**: Limit tree complexity
- **Maximum nodes**: Control total tree size

## Computational Complexity

- **Training**: $O(nfd)$ where $n$ = examples, $f$ = features, $d$ = depth
- **Testing**: $O(d)$ (or $O(\log n)$ for balanced trees)

## Limitations

### Axis-Aligned Boundaries
Decision trees create rectangular regions and struggle with:
- Diagonal decision boundaries
- Non-axis-aligned patterns
- Additive feature interactions

### Overfitting Tendency
- High variance, low bias
- Can grow very deep and complex
- Require careful regularization

## Ensemble Methods

### Random Forests
- Multiple decorrelated trees
- Feature and sample bagging
- Reduces overfitting through diversity

### Boosting
- Sequential weak learner addition
- Adaptive sample reweighting
- Can achieve zero training loss theoretically
- Modern implementations: XGBoost, LightGBM

## Practical Applications

### Classification
- Medical diagnosis
- Credit scoring
- Customer segmentation
- Fraud detection

### Regression
- Housing price prediction
- Sales forecasting
- Risk assessment
- Quality control

## Implementation Resources

### Python Libraries
- **scikit-learn**: `DecisionTreeClassifier`, `DecisionTreeRegressor`
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Microsoft's gradient boosting
- **CatBoost**: Yandex's gradient boosting

### R Packages
- **rpart**: Recursive partitioning
- **tree**: Classification and regression trees
- **party**: Conditional inference trees
- **randomForest**: Random forests

### MATLAB
- **Statistics and Machine Learning Toolbox**: `fitctree`, `fitrtree`
- **TreeBagger**: Random forests implementation

## Related Course Modules

- **[Regression Trees](../04_regression_trees/)** - Tree-based regression methods
- **[Classification Trees](../12_classification_trees/)** - Tree-based classification
- **[Variable Selection](../03_variable_selection_regularization/)** - Feature importance in trees
- **[Random Forest](../04_regression_trees/)** - Ensemble tree methods

## Learning Path

1. **Start with fundamentals**: Read `01_introduction.md` for core concepts
2. **Review visual examples**: Study the splitting process through images
3. **Explore reference materials**: Deepen understanding with CS229 notes
4. **Practice implementation**: Use the boosting example code
5. **Apply to real problems**: Experiment with different datasets
6. **Study ensemble methods**: Understand Random Forests and Boosting

## Assessment

After completing this module, you should be able to:
- Explain how decision trees partition input space
- Describe the greedy splitting algorithm
- Compare different loss functions for classification and regression
- Identify when decision trees are appropriate vs. their limitations
- Understand the relationship between trees and ensemble methods
- Implement basic decision tree algorithms

---

*This module provides a solid foundation in decision trees, preparing you for more advanced ensemble methods and practical machine learning applications.* 