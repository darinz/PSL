# 4.1. Regression Trees

## 4.1.1. Introduction to Regression Trees

Regression trees represent a fundamental approach to non-parametric regression that partitions the feature space into rectangular regions and fits a simple model (typically a constant) in each region. This week, we'll delve into tree-based models for regression, starting with single regression trees before progressing to ensemble methods like Random Forests (based on bagging) and Gradient Boosting Machines (GBM, based on boosting techniques).

### Mathematical Framework

Consider a regression problem with:
- **Input features**: $`X = (X_1, X_2, \ldots, X_p) \in \mathbb{R}^p`$
- **Response variable**: $`Y \in \mathbb{R}`$
- **Training data**: $`\{(x_i, y_i)\}_{i=1}^n`$ where $`x_i = (x_{i1}, x_{i2}, \ldots, x_{ip})`$

A regression tree model can be expressed as:

```math
f(x) = \sum_{m=1}^M c_m \cdot I(x \in R_m)
```

where:
- $`R_m`$ represents the $`m`$-th rectangular region (leaf node)
- $`c_m`$ is the constant prediction for region $`R_m`$
- $`I(\cdot)`$ is the indicator function
- $`M`$ is the number of leaf nodes

### Tree Structure and Terminology

Regression trees are constructed by recursively partitioning the feature space $`\mathbb{R}^p`$ into two sub-regions, beginning with the entire space. Each partition is defined by a **split rule** of the form:

```math
\text{Split Rule: } X_j \leq s
```

where:
- $`X_j`$ is the $`j`$-th feature variable
- $`s`$ is the split threshold

**Tree Components:**
- **Root Node**: The entire feature space
- **Internal Nodes**: Nodes with children (split points)
- **Leaf Nodes**: Terminal nodes (rectangular regions)
- **Branches**: Connections between nodes

### Boston Housing Example

Consider the Boston Housing dataset with two features: longitude and latitude. The regression tree partitions the 2D space into rectangular regions, where each region corresponds to a leaf node with a constant prediction.

**Visualization Description:**
- **Right plot**: Scatter plot of houses by longitude and latitude, with grayscale indicating price (darker = more expensive)
- **Left plot**: Tree structure showing recursive splits on longitude and latitude features

**Example Tree Structure:**
```
Root: All houses
├── Longitude ≤ -71.1
│   ├── Latitude ≤ 42.3 → Price = 3.1 (log scale)
│   └── Latitude > 42.3 → Price = 3.5 (log scale)
└── Longitude > -71.1
    ├── Latitude ≤ 42.2 → Price = 3.8 (log scale)
    └── Latitude > 42.2 → Price = 4.2 (log scale)
```

![Boston Housing: Tree Partitioning of Feature Space](../_images/w4_plot_housing_lon_alt.png)

*Figure: Partitioning of the Boston Housing data by longitude and latitude. Each region corresponds to a leaf node in the regression tree.*

### Advantages of Tree-Based Models

1. **Interpretability**: Tree structure is easily explainable to non-technical audiences
2. **Automatic Variable Selection**: Only relevant features are used for splitting
3. **Interaction Detection**: Natural handling of feature interactions at different tree levels
4. **Invariance to Monotonic Transformations**: Tree structure remains unchanged under monotonic transformations of features
5. **Handling Mixed Data Types**: Naturally handles both numerical and categorical variables
6. **Robustness to Outliers**: Less sensitive to outliers compared to linear models

**Mathematical Invariance Property:**
If $`g(\cdot)`$ is a strictly monotonic function, then splitting on $`X_j \leq s`$ is equivalent to splitting on $`g(X_j) \leq g(s)`$.

## 4.1.2. Tree Construction Algorithm

### Mathematical Foundation

The goal is to find the optimal tree structure that minimizes the prediction error. For regression trees, we typically minimize the **Residual Sum of Squares (RSS)**:

```math
\text{RSS} = \sum_{i=1}^n (y_i - f(x_i))^2
```

### Three Core Questions

1. **Where to Split**: Which feature and threshold to use for partitioning
2. **When to Stop**: When to stop growing the tree
3. **How to Predict**: What constant value to assign to each leaf node

### Assigning Predictions to Leaf Nodes

For a leaf node $`R_m`$ containing observations $`\{i: x_i \in R_m\}`$, the optimal constant prediction is the mean of the response values:

```math
c_m = \frac{1}{|R_m|} \sum_{i: x_i \in R_m} y_i
```

This minimizes the RSS within the leaf node.

### Split Criterion: RSS Reduction

For each potential split $(j, s)$, we calculate the reduction in RSS:

```math
\Delta \text{RSS}(j, s) = \text{RSS}_{\text{before}} - \text{RSS}_{\text{after}}
```

where:
- $`\text{RSS}_{\text{before}} = \sum_{i=1}^n (y_i - \bar{y})^2`$ (using overall mean)
- $`\text{RSS}_{\text{after}} = \text{RSS}_{\text{left}} + \text{RSS}_{\text{right}}`$

The left and right RSS are calculated as:

```math
\text{RSS}_{\text{left}} = \sum_{i: x_{ij} \leq s} (y_i - \bar{y}_{\text{left}})^2
```

```math
\text{RSS}_{\text{right}} = \sum_{i: x_{ij} > s} (y_i - \bar{y}_{\text{right}})^2
```

where $`\bar{y}_{\text{left}}`$ and $`\bar{y}_{\text{right}}`$ are the means of the left and right child nodes.

### Greedy Tree Building Algorithm

The greedy tree building algorithm recursively partitions the feature space by finding the optimal split at each node. The implementation includes functions for finding the best split, building nodes recursively, and handling stopping criteria.

**Python Implementation:** [tree_building.py](code/tree_building.py)

The algorithm includes:
- `build_regression_tree()`: Main function to build the complete tree
- `find_best_split()`: Find optimal feature and threshold for splitting
- `build_node()`: Recursively build tree nodes
- Utility functions for prediction and tree analysis

### Handling Categorical Variables

For categorical variables with $`m`$ levels, the optimal split can be found efficiently by:

1. **Sorting levels by response mean**: Calculate $`\bar{y}_k`$ for each level $`k`$
2. **Considering only adjacent splits**: Only $`m-1`$ splits need to be evaluated

**Mathematical Justification:**
The optimal split minimizes within-group variance. By sorting levels by their response means, adjacent levels have similar means, making them natural candidates for grouping.

**Python Implementation:** [tree_building.py](code/tree_building.py) - `find_categorical_split()` function

The implementation efficiently handles categorical variables by:
- Calculating mean response for each level
- Sorting levels by response means
- Evaluating only adjacent splits to find the optimal partition

### Handling Missing Values

Tree-based methods offer several strategies for handling missing values:

1. **Surrogate Splits**: Use correlated variables as backup splits
2. **Missing as Separate Category**: Treat missing values as a distinct category
3. **Majority Rule**: Assign missing values to the larger child node
4. **Imputation**: Fill missing values before tree construction

**Python Implementation:** [tree_building.py](code/tree_building.py) - `find_surrogate_splits()` function

The surrogate split implementation:
- Finds correlated variables that can substitute for the primary split
- Calculates correlation between primary split and potential surrogate splits
- Returns sorted list of surrogate splits by correlation strength
- Uses a minimum correlation threshold (0.5) to ensure quality substitutes

### Stopping Criteria

Common stopping criteria include:

1. **Minimum samples per leaf**: $`|R_m| \geq \text{min\_samples\_leaf}`$
2. **Maximum tree depth**: $`\text{depth} \leq \text{max\_depth}`$
3. **Minimum RSS reduction**: $`\Delta \text{RSS} \geq \text{min\_improvement}`$
4. **Maximum leaf nodes**: $`M \leq \text{max\_leaves}`$

## 4.1.3. Tree Pruning: Complexity Cost

### Overfitting Problem

Large trees can overfit the training data, leading to poor generalization. Pruning addresses this by removing unnecessary splits while maintaining predictive performance.

### Cost-Complexity Pruning

The cost-complexity measure balances fit and complexity:

```math
R_\alpha(T) = \text{RSS}(T) + \alpha |T|
```

where:
- $`\text{RSS}(T) = \sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} (y_i - \bar{y}_m)^2`$
- $`|T|`$ is the number of leaf nodes
- $`\alpha \geq 0`$ is the complexity parameter

**Interpretation:**
- $`\alpha = 0`$: No penalty for complexity (full tree)
- $`\alpha \to \infty`$: Infinite penalty (single node tree)
- Larger $`\alpha`$ produces simpler trees

### Mathematical Properties

For a given $`\alpha`$, the optimal subtree $`T_\alpha`$ minimizes $`R_\alpha(T)`$:

```math
T_\alpha = \arg\min_{T \subseteq T_0} R_\alpha(T)
```

where $`T_0`$ is the full tree.

**Uniqueness Property:**
If multiple subtrees achieve the same minimum cost, there exists a unique smallest optimal subtree (the intersection of all optimal subtrees).

## 4.1.4. Weakest Link Pruning Algorithm

### Alpha Calculation

For each internal node $`t`$, we calculate the threshold $`\alpha_t`$ at which the split becomes unprofitable:

```math
\alpha_t = \frac{\text{RSS}(t) - \text{RSS}(T_t)}{|T_t| - 1}
```

where:
- $`\text{RSS}(t)`$ is the RSS when node $`t`$ is a leaf
- $`\text{RSS}(T_t)`$ is the RSS of the subtree rooted at $`t`$
- $`|T_t|`$ is the number of leaf nodes in the subtree

**Interpretation:**
$`\alpha_t`$ represents the "price" we pay per additional leaf node for the improvement in RSS.

### Algorithm Steps

1. **Initialize**: Start with full tree $`T_0`$, set $`\alpha = 0`$
2. **Calculate alphas**: For each internal node $`t`$, compute $`\alpha_t`$
3. **Find weakest link**: Identify node $`t^*`$ with smallest $`\alpha_t`$
4. **Prune**: Remove the subtree rooted at $`t^*`$, making $`t^*`$ a leaf
5. **Update**: Recalculate $`\alpha_t`$ for affected nodes
6. **Repeat**: Continue until only root remains

**Python Implementation:** [tree_pruning.py](code/tree_pruning.py) - `weakest_link_pruning()` function

The weakest link pruning algorithm includes:
- `calculate_alpha()`: Calculate the alpha threshold for each node
- `find_weakest_link()`: Find the node with the smallest alpha value
- `prune_node()`: Recursively prune the target node from the tree
- Main loop that generates a sequence of pruned trees with increasing alpha values

### Solution Path

The algorithm generates a sequence of trees $`T_0, T_1, \ldots, T_k`$ corresponding to increasing $`\alpha`$ values:

```math
0 = \alpha_0 < \alpha_1 < \alpha_2 < \cdots < \alpha_k
```

Each tree $`T_i`$ is optimal for $`\alpha \in [\alpha_i, \alpha_{i+1})`$.

## 4.1.5. Cross-Validation for Alpha Selection

### Problem Statement

Given the sequence of pruned trees, we need to select the optimal $`\alpha`$ value that minimizes prediction error.

### Cross-Validation Procedure

1. **Generate beta values**: For each interval $`[\alpha_i, \alpha_{i+1})`$, compute $`\beta_i = \sqrt{\alpha_i \cdot \alpha_{i+1}}`$

2. **K-fold cross-validation**: For each fold $`k = 1, 2, \ldots, K`$:
   - Train tree on $`K-1`$ folds
   - Generate pruned tree sequence
   - Evaluate each tree on the held-out fold

3. **Select optimal alpha**: Choose $`\alpha`$ that minimizes cross-validation error

**Python Implementation:** [tree_pruning.py](code/tree_pruning.py) - `cross_validate_alpha()` function

The cross-validation implementation:
- Uses K-fold cross-validation to evaluate different alpha values
- Generates all possible alpha values from the pruning sequence
- Evaluates each alpha value across all folds
- Returns the optimal alpha that minimizes cross-validation error

### One Standard Error Rule

Instead of selecting the minimum CV error, we can use the one standard error rule:

**Python Implementation:** [tree_pruning.py](code/tree_pruning.py) - `one_se_rule()` function

The one standard error rule:
- Calculates the standard error of cross-validation errors
- Selects the largest alpha within one standard error of the minimum
- Provides a more conservative choice that balances complexity and performance

## 4.1.6. Complete Implementation Examples

### Python Implementation

**Complete Implementation:** [complete_implementation.py](code/complete_implementation.py)

The complete Python implementation includes:

- **RegressionTree Class**: A comprehensive class with methods for building, predicting, and pruning trees
- **Tree Building**: Recursive tree construction with configurable stopping criteria
- **Prediction**: Efficient prediction for both single samples and arrays
- **Pruning**: Cost-complexity pruning implementation
- **Demonstration**: Complete example using Boston housing dataset
- **Visualization**: Tree structure analysis and performance evaluation
- **Model Comparison**: Comparison with linear regression models

Key features:
- Configurable parameters (max_depth, min_samples_split, min_samples_leaf)
- Comprehensive error handling and validation
- Built-in visualization and analysis tools
- Integration with scikit-learn for data loading and evaluation

### R Implementation

**Complete R Implementation:** [r_implementation.R](code/r_implementation.R)

The R implementation provides:

- **Tree Building**: Functions for building regression trees using the `rpart` package
- **Cross-Validation**: Implementation of cross-validation for optimal complexity parameter selection
- **Pruning**: Automatic pruning using the complexity parameter (CP)
- **Visualization**: Tree structure plotting and performance analysis
- **Demonstrations**: Multiple examples including Boston housing data and synthetic data
- **Performance Analysis**: Comprehensive evaluation metrics and residual analysis
- **Model Comparison**: Comparison with linear regression models

Key features:
- Uses `rpart` package for efficient tree construction
- Built-in cross-validation and pruning capabilities
- Comprehensive visualization tools with `rpart.plot`
- Integration with `MASS` package for dataset access
- Modular function design for easy customization

### Visualization and Analysis

**Python Implementation:** [complete_implementation.py](code/complete_implementation.py) - Visualization functions

The visualization and analysis tools include:

- **Tree Structure Analysis**: Functions to analyze tree depth, node count, and feature importance
- **Performance Metrics**: Comprehensive evaluation including MSE, RMSE, MAE, and R²
- **Residual Analysis**: Diagnostic plots for model validation
- **Feature Importance**: Analysis of which features are most frequently used for splitting
- **Model Comparison**: Tools to compare tree performance with other models

Key visualization features:
- Tree statistics and structure analysis
- Residual plots and distribution analysis
- Q-Q plots for normality assessment
- Performance comparison visualizations

## Summary

Regression trees provide a powerful, interpretable approach to non-parametric regression. Key concepts include:

1. **Tree Structure**: Recursive binary partitioning of feature space
2. **Split Criterion**: RSS reduction for optimal splits
3. **Pruning**: Cost-complexity pruning to prevent overfitting
4. **Cross-Validation**: Selection of optimal complexity parameter
5. **Handling Special Cases**: Categorical variables, missing values

The mathematical foundations ensure optimality, while the greedy algorithm provides computational efficiency. The pruning process balances model complexity with predictive performance, making regression trees a versatile tool for both exploration and prediction.

## Code Files Summary

The following code files provide complete implementations of the concepts discussed in this chapter:

### Python Implementation
- **[tree_building.py](code/tree_building.py)**: Greedy tree building algorithm, categorical variable handling, and surrogate splits
- **[tree_pruning.py](code/tree_pruning.py)**: Weakest link pruning algorithm, cross-validation for alpha selection, and one standard error rule
- **[complete_implementation.py](code/complete_implementation.py)**: Complete RegressionTree class with building, prediction, pruning, visualization, and analysis tools

### R Implementation
- **[r_implementation.R](code/r_implementation.R)**: Complete R implementation using rpart package with tree building, pruning, cross-validation, and visualization

Each file includes comprehensive examples, demonstrations, and analysis tools to help understand and apply regression tree concepts in practice.

## References

- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and regression trees. CRC press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.

---

**Navigation:**
- **Next Topic:** [Random Forest](02_random_forest.md) - Ensemble methods and bootstrap aggregation
- **Previous Topic:** [Regression Trees Overview](README.md) - Overview of tree-based regression methods
