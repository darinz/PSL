# Introduction to Statistical Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This module presents comprehensive, modernized materials covering the foundational concepts of statistical learning, from basic problem types to advanced theoretical frameworks.

## Module Contents

### Core Concepts
- **[01_introduction.md](01_introduction.md)** – Types of statistical learning problems, supervised vs unsupervised learning, and the fundamental challenges
- **[02_learning_theory.md](02_learning_theory.md)** – Mathematical foundations of learning theory, statistical decision theory, and the bias-variance decomposition
- **[03_bias_variance.md](03_bias_variance.md)** – Deep dive into the bias-variance tradeoff with practical examples and strategies

### Fundamental Algorithms
- **[04_ls_and_knn.md](04_ls_and_knn.md)** – Linear regression and k-Nearest Neighbors: parametric vs non-parametric approaches
- **[05_bayes_rule.md](05_bayes_rule.md)** – Bayes classification rule: the theoretical optimal classifier

### Practical Implementation
- **[Python_W1_SimulationStudy.py](Python_W1_SimulationStudy.py)** – Python implementation of simulation studies
- **[Rcode_W1_SimulationStudy.R](Rcode_W1_SimulationStudy.R)** – R implementation of simulation studies

### Supplementary Materials
- **[img/](img/)** – Supporting images and diagrams for visual learning

## Learning Objectives

Upon completion of this module, students will be able to:

### 1. Problem Classification
- **Supervised Learning:** Regression (continuous targets) vs Classification (categorical targets)
- **Unsupervised Learning:** Clustering, dimensionality reduction, pattern discovery
- **Advanced Paradigms:** Semi-supervised, active, and transfer learning

### 2. Theoretical Foundations
- **Statistical Decision Theory:** Optimal prediction under uncertainty
- **Learning Theory:** Generalization bounds and model selection
- **Bias-Variance Tradeoff:** The fundamental tension in model complexity

### 3. Algorithm Understanding
- **Linear Regression:** Parametric approach with interpretable coefficients
- **k-Nearest Neighbors:** Non-parametric, instance-based learning
- **Bayes Classifier:** Theoretical optimal decision rule

### 4. Practical Skills
- **Model Selection:** Cross-validation and regularization techniques
- **Performance Evaluation:** Understanding training vs test error
- **Implementation:** Hands-on experience with Python and R

## Recommended Study Sequence

1. **Begin with [01_introduction.md](01_introduction.md)** for foundational concepts
2. **Proceed to [02_learning_theory.md](02_learning_theory.md)** for mathematical framework
3. **Examine [03_bias_variance.md](03_bias_variance.md)** for the core tradeoff concept
4. **Investigate [04_ls_and_knn.md](04_ls_and_knn.md)** for concrete algorithms
5. **Conclude with [05_bayes_rule.md](05_bayes_rule.md)** for theoretical optimality
6. **Reinforce concepts through simulation exercises and expanded code/math examples**

## Key Mathematical Concepts

### Supervised Learning Framework
```math
f^* = \arg\min_{f \in \mathcal{F}} \mathbb{E}_{(X,Y)}[L(Y, f(X))]
```

### Bias-Variance Decomposition
```math
\mathbb{E}[(Y - \hat{f}(X))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
```

### Bayes Classifier
```math
f^*(x) = \arg\max_{k} P(Y = k \mid X = x)
```

### k-Nearest Neighbors
```math
\hat{f}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
```

### Linear Regression
```math
\hat{\beta} = (X^T X)^{-1} X^T y
```

## Computational Resources

### Python Implementation
The `Python_W1_SimulationStudy.py` script provides:
- Bias-variance tradeoff visualization
- Model complexity analysis
- Cross-validation for model selection
- Performance comparison between algorithms

### R Implementation
The `Rcode_W1_SimulationStudy.R` script offers:
- Statistical analysis and visualization
- Hypothesis testing for model comparison
- Advanced plotting capabilities
- Integration with statistical learning packages

## Advanced Topic Connections

This introduction establishes the foundation for:
- **Regularization Methods:** Ridge, Lasso, Elastic Net
- **Ensemble Methods:** Bagging, Boosting, Random Forests
- **Neural Networks:** Understanding overfitting and generalization
- **Model Selection:** Cross-validation and information criteria
- **Statistical Inference:** Confidence intervals and hypothesis testing

## Study Recommendations

1. **Implement algorithms:** Apply theoretical concepts through practical coding exercises and expanded code examples
2. **Utilize visualizations:** Leverage provided diagrams and new LaTeX math to build intuitive understanding
3. **Experiment with parameters:** Modify simulation parameters to observe effects
4. **Bridge theory and practice:** Connect mathematical formulations to empirical performance
5. **Compare methodologies:** Analyze how different algorithms address similar problems
6. **Reference expanded explanations:** Use the improved markdown and math formatting for deeper understanding

## Contributing

To contribute improvements to this module:
- Report issues through the main repository
- Suggest enhancements to explanations
- Propose additional examples or visualizations
- Improve computational implementations

---

*This material is based on the [PSL Online Notes](https://liangfgithub.github.io/PSL/) and other foundational statistical learning resources.* 