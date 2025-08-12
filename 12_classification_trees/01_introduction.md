# 12.1. Introduction

Classification trees are a fundamental machine learning technique that extends the concept of decision trees from regression to classification problems. Just as in our previous discussion about regression trees, when it comes to classification trees, we must also focus on three essential aspects:

## 12.1.1. The Three Key Components

### 1. Where to Split

This involves deciding on the variable (denoted as $`j`$) and the split value ($`s`$) that divides our data into two parts, based on whether $`X_j < s`$ or not.

**Mathematical Formulation**: For a feature $`j`$ and split point $`s`$, we create two regions:
```math
R_1(j, s) = \{X | X_j \leq s\} \quad \text{and} \quad R_2(j, s) = \{X | X_j > s\}
```

**Key Considerations**:
- **Feature Selection**: Which variable provides the best split?
- **Split Point**: What threshold value maximizes separation?
- **Binary Splits**: Each split creates exactly two child nodes

### 2. When to Stop

As previously discussed, the general strategy is to initially construct a large tree and then employ a pruning process based on a loss plus penalty criteria. This strategy helps prevent overfitting.

**Stopping Criteria**:
- **Minimum node size**: Stop when node contains fewer than $`n_{\min}`$ samples
- **Maximum depth**: Stop when tree reaches maximum depth $`d_{\max}`$
- **Pure nodes**: Stop when all samples in node belong to same class
- **Minimum improvement**: Stop when split improvement is below threshold

**Pruning Strategy**:
```math
\text{Cost}(T) = \text{Loss}(T) + \alpha \cdot \text{Complexity}(T)
```

where $`\alpha`$ is the regularization parameter controlling tree size.

### 3. How to Predict at Each Leaf Node

Depending on whether we are dealing with regression or classification, we adopt different approaches for making predictions at leaf nodes.

#### Regression Trees
For regression, at each leaf node, we calculate the average Y value based on the training samples within that node:
```math
\hat{y}_{\text{leaf}} = \frac{1}{n_{\text{leaf}}} \sum_{i \in \text{leaf}} y_i
```

#### Classification Trees
For classification, we apply a similar concept. When a leaf node contains observations from $`K`$ classes, we can either:

**Majority Voting**:
```math
\hat{y}_{\text{leaf}} = \arg\max_{k} n_k
```

where $`n_k`$ is the number of samples of class $`k`$ in the leaf.

**Class Probabilities**:
```math
P(y = k | \text{leaf}) = \frac{n_k}{n_{\text{leaf}}}
```

where $`n_{\text{leaf}} = \sum_{k=1}^K n_k`$ is the total number of samples in the leaf.

## 12.1.2. Goodness-of-Split Criterion

### Regression vs Classification

In the context of regression, this often involves calculating the reduction in residual sum of squares. Specifically, we consider a node $`T`$:

**Regression Split Criterion**:
```math
\Delta \text{RSS} = \text{RSS}(T) - \left[\text{RSS}(T_L) + \text{RSS}(T_R)\right]
```

where:
- $`\text{RSS}(T) = \sum_{i \in T} (y_i - \bar{y}_T)^2`$
- $`\text{RSS}(T_L) = \sum_{i \in T_L} (y_i - \bar{y}_{T_L})^2`$
- $`\text{RSS}(T_R) = \sum_{i \in T_R} (y_i - \bar{y}_{T_R})^2`$

**Classification Split Criterion**:
For classification, we use impurity measures instead of RSS:

```math
\Delta I = I(T) - \left[\frac{n_L}{n_T} I(T_L) + \frac{n_R}{n_T} I(T_R)\right]
```

where $`I(T)`$ is the impurity measure for node $`T`$.

### Common Impurity Measures

#### 1. Gini Impurity
```math
I_{\text{Gini}}(T) = 1 - \sum_{k=1}^K p_k^2
```

where $`p_k = \frac{n_k}{n_T}`$ is the proportion of class $`k`$ in node $`T`$.

#### 2. Entropy
```math
I_{\text{Entropy}}(T) = -\sum_{k=1}^K p_k \log_2(p_k)
```

#### 3. Misclassification Error
```math
I_{\text{Error}}(T) = 1 - \max_k p_k
```

### Properties of Impurity Measures

1. **Range**: All measures are in $`[0, 1]`$ for binary classification
2. **Minimum**: Achieved when node is pure (all samples same class)
3. **Maximum**: Achieved when classes are equally distributed
4. **Differentiability**: Gini and Entropy are differentiable, Error is not

## 12.1.3. The Greedy Algorithm

The process of searching for the best split follows a basic greedy algorithm:

### Algorithm Steps

1. **Start at root node** with all training data
2. **For each feature** $`j = 1, 2, \ldots, p`$:
   - Sort unique values of feature $`j`$
   - **For each split point** $`s`$ (midpoint between consecutive values):
     - Split data: $`X_j \leq s`$ vs $`X_j > s``
     - Calculate impurity reduction $`\Delta I`$
3. **Select best split**: Choose $`(j^*, s^*)`$ that maximizes $`\Delta I`$
4. **Create child nodes**: Split data according to best split
5. **Recurse**: Apply algorithm to each child node

### Computational Complexity

- **Time**: $`O(p \cdot n \log n)`$ per node (sorting dominates)
- **Space**: $`O(n)`$ for storing node data
- **Total**: $`O(p \cdot n \log n \cdot \text{number of nodes})`$

## 12.1.4. Implementation and Examples

The implementation of classification trees is provided in separate code files for both Python and R. These implementations demonstrate the core concepts of classification trees including impurity measures, tree building, and decision boundaries.

**Python Implementation**: The complete classification tree implementation is available in `code/introduction_implementation.py` and includes:
- **`ClassificationTree` class** with custom implementation of all impurity measures (Gini, Entropy, Misclassification Error)
- **Tree building algorithm** with recursive splitting and stopping criteria
- **Impurity measure comparison** between Gini and Entropy criteria
- **Tree structure visualization** with different depths
- **Stopping criteria demonstration** showing the effects of different parameters
- **Greedy algorithm step-by-step demonstration** showing how splits are chosen
- **Advantages and limitations analysis** with different data patterns
- **Decision boundary visualization** and accuracy analysis

**R Implementation**: The complete classification tree implementation is available in `code/r_introduction_implementation.R` and includes:
- **Basic tree demonstration** using rpart with tree visualization
- **Impurity measures analysis** using Gini criterion (rpart default)
- **Tree structure analysis** with different depths and node counts
- **Stopping criteria demonstration** with various parameter configurations
- **Greedy algorithm demonstration** showing split selection process
- **Advantages and limitations analysis** with different data patterns
- **Decision boundary visualization** using ggplot2

To run the classification tree demonstrations:

```python
# Python
from code.introduction_implementation import main
results = main()
```

```r
# R
source("code/r_introduction_implementation.R")
results <- main_r()
```

The implementations demonstrate how classification trees extend regression trees by using impurity measures instead of RSS, and how the greedy algorithm efficiently finds optimal splits to create interpretable decision boundaries.

## 12.1.5. Advantages and Limitations

### Advantages

1. **Interpretability**: Easy to understand and visualize
2. **No Assumptions**: No assumptions about data distribution
3. **Handles Mixed Data**: Can handle both numerical and categorical features
4. **Feature Importance**: Natural feature selection through splits
5. **Robust**: Insensitive to monotone transformations

### Limitations

1. **Instability**: Small changes in data can lead to very different trees
2. **Overfitting**: Tendency to overfit without proper regularization
3. **Axis-Aligned**: Can only create axis-aligned decision boundaries
4. **Greedy**: Local optimization may miss global optimum
5. **High Variance**: Individual trees have high variance

## 12.1.6. Summary

Classification trees extend regression trees to classification problems by:

1. **Impurity Measures**: Using Gini, entropy, or misclassification error instead of RSS
2. **Prediction Methods**: Majority voting or class probabilities at leaf nodes
3. **Split Criteria**: Maximizing impurity reduction
4. **Greedy Algorithm**: Same recursive splitting approach

Key insights:
- **Impurity measures** control split quality
- **Stopping criteria** prevent overfitting
- **Greedy approach** is computationally efficient
- **Tree structure** provides interpretability

This foundation sets the stage for more advanced tree-based methods like random forests and gradient boosting, which address many of the limitations of single classification trees.

---

**Navigation:**
- **Next Topic:** [Impurity Measures](02_impurity_measures.md) - Mathematical foundations and properties of impurity measures for classification
- **Previous Topic:** [Classification Trees Overview](README.md) - Overview of classification trees and boosting algorithms
