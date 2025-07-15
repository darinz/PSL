# Practical Statistical Learning (PSL)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-Math%20Typesetting-red.svg)](https://www.latex-project.org/)

A comprehensive collection of course materials covering fundamental concepts in statistical learning, machine learning, and data analysis. This repository contains lecture notes, code examples, and practical implementations in both Python and R.

## Table of Contents

- [Overview](#overview)
- [Course Structure](#course-structure)
- [Topics Covered](#topics-covered)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

This repository contains comprehensive materials for learning statistical learning concepts, from basic principles to advanced techniques. The course covers both theoretical foundations and practical implementations, with code examples in Python and R.

## Complementary Learning Resources

For a comprehensive introduction to machine learning that complements this statistical learning material, see:

- **[Machine Learning Repository](https://github.com/darinz/Machine-Learning)**: A comprehensive resource covering supervised and unsupervised learning, learning theory, reinforcement learning, and modern applications including deep learning, transformers, and foundation models. This repository provides:
  - Linear models and generative learning algorithms
  - Advanced classification techniques
  - Deep learning fundamentals and neural networks
  - Clustering and EM algorithms with variational methods
  - Dimensionality reduction techniques (PCA, ICA)
  - Self-supervised learning and foundation models
  - Reinforcement learning and control systems
  - Modern applications in NLP, computer vision, and robotics

The Machine Learning repository offers a broader perspective on algorithmic approaches and modern machine learning techniques, while this Statistical Learning repository focuses on the statistical foundations and classical methods. Together, they provide a complete learning path from statistical theory to modern machine learning practice.

## Course Structure

The course is organized into the following modules:

### 1. Introduction
- Learning theory fundamentals
- Bias-variance tradeoff
- Least squares and k-nearest neighbors
- Bayes rule applications

### 2. Linear Regression
- Multiple linear regression
- Geometric interpretation
- Practical issues and solutions

### 3. Variable Selection and Regularization
- Subset selection methods
- Ridge regression
- Lasso regression
- Regularization techniques

### 4. Regression Trees and Ensemble Methods
- Regression trees
- Random forests
- Gradient boosting machines (GBM)

### 5. Nonlinear Regression
- Polynomial regression
- Cubic splines
- Regression splines
- Smoothing splines
- Local regression

### 6. Clustering Analysis
- Distance measures
- K-means and K-medoids
- Choice of K
- Hierarchical clustering

### 7. Latent Structure Models
- Model-based clustering
- Mixture models
- EM algorithm
- Latent Dirichlet Allocation (LDA)
- Hidden Markov Models (HMM)

### 8. Discriminant Analysis
- Introduction to classification
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Fisher Discriminant Analysis
- Naive Bayes classifiers

### 9. Logistic Regression
- Binary and multinomial logistic regression
- Maximum likelihood estimation
- Separable and non-separable data
- Retrospective sampling

### 10. Support Vector Machines (SVM)
- Linear SVMs (separable and non-separable cases)
- Max-margin principle and duality
- KKT conditions
- Soft margin SVM and slack variables
- Nonlinear SVMs and the kernel trick
- RKHS and kernel machines
- Practical considerations and applications
- [See module: `11_support_vector_machine/`](./11_support_vector_machine/)

### 11. Classification Trees and Boosting
- Classification trees with impurity measures
- Misclassification rate vs. entropy comparison
- AdaBoost algorithm and convergence proof
- Forward stagewise additive modeling
- Modern boosting algorithms (GBM, XGBoost, CatBoost)
- [See module: `12_classification_trees/`](./12_classification_trees/)

### 12. Recommender Systems
- Content-based recommendation methods
- Collaborative filtering (user-based and item-based)
- Latent factor models and matrix decomposition
- Challenges and evaluation strategies
- Deep learning approaches for recommendations
- Real-world applications (Netflix, Spotify, Amazon)
- [See module: `13_recommender_system/`](./13_recommender_system/)

## Topics Covered

### Core Concepts
- **Supervised Learning**: Regression and classification problems
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Model Selection**: Cross-validation, regularization, and hyperparameter tuning
- **Statistical Inference**: Hypothesis testing, confidence intervals, and p-values

### Algorithms and Methods
- **Linear Methods**: Linear regression, logistic regression, LDA, SVM
- **Tree-Based Methods**: Decision trees, random forests, boosting, classification trees
- **Regularization**: Ridge, Lasso, elastic net
- **Clustering**: K-means, hierarchical clustering, model-based clustering
- **Dimensionality Reduction**: PCA, factor analysis
- **Advanced Models**: HMM, LDA, mixture models, SVMs with kernels
- **Ensemble Methods**: AdaBoost, gradient boosting, modern variants (XGBoost, CatBoost)
- **Recommender Systems**: Content-based, collaborative filtering, latent factor models, deep learning approaches

### Programming Languages
- **Python**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **R**: Base R, tidyverse, caret, ggplot2

## Getting Started

### Prerequisites

Before starting this course, you should have:

- Basic knowledge of calculus and linear algebra
- Familiarity with probability and statistics
- Programming experience in Python or R
- Understanding of data structures and algorithms

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Statistical-Learning.git
   cd Statistical-Learning
   ```

2. **Set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Set up R environment**:
   ```r
   # Install required R packages
   install.packages(c("tidyverse", "caret", "ggplot2", "dplyr", "tidyr"))
   ```

### Required Packages

#### Python
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- statsmodels

#### R
- tidyverse
- caret
- ggplot2
- dplyr
- tidyr
- MASS
- e1071

## Usage

### Running Code Examples

1. **Python Examples**:
   ```bash
   # Navigate to specific module
   cd 01_introduction
   
   # Run Python script
   python Python_W1_SimulationStudy.py
   
   # Or use Jupyter notebook
   jupyter notebook
   ```

2. **R Examples**:
   ```r
   # Set working directory
   setwd("01_introduction")
   
   # Source R script
   source("Rcode_W1_SimulationStudy.R")
   ```

### Reading Materials

Each module contains:
- **Markdown files**: Detailed theoretical explanations
- **Code files**: Practical implementations
- **Images**: Visualizations and diagrams
- **README files**: Module-specific instructions

## Contributing

We welcome contributions to improve the course materials:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Ensure all code follows the style guidelines
- Add appropriate documentation and comments
- Include tests for new functionality
- Update README files when adding new modules
- Verify that all mathematical notation is correctly formatted

## References

### Textbooks
- **Elements of Statistical Learning** (ESL) - Trevor Hastie, Robert Tibshirani, Jerome Friedman
- **Introduction to Statistical Learning** (ISL) - Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani
- **Pattern Recognition and Machine Learning** - Christopher Bishop

### Online Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [R Documentation](https://www.r-project.org/)
- [CRAN Task Views](https://cran.r-project.org/web/views/)

### Academic Papers
- Original papers for algorithms covered in the course
- Recent developments in statistical learning
- Applications in various domains