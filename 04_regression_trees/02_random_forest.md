# 4.2. Random Forest

## 4.2.1. Introduction to Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mean prediction of the individual trees. This approach addresses the high variance problem inherent in single decision trees by leveraging the power of ensemble methods.

![Random Forest Ensemble](../_images/w4_forest.png)

*Figure: A random forest is an ensemble of many decision trees, each trained on a different bootstrap sample of the data.*

### Mathematical Framework

Consider a regression problem with:
- **Input features**: $`X = (X_1, X_2, \ldots, X_p) \in \mathbb{R}^p`$
- **Response variable**: $`Y \in \mathbb{R}`$
- **Training data**: $`\{(x_i, y_i)\}_{i=1}^n`$

A Random Forest model can be expressed as:

```math
f_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)
```

where:
- $`f_b(x)`$ is the prediction of the $`b`$-th tree
- $`B`$ is the number of trees in the forest
- Each tree $`f_b`$ is trained on a bootstrap sample with feature subsampling

### Why Ensemble Methods?

Single decision trees suffer from high variance due to their greedy, top-down construction. Small changes in the training data can lead to dramatically different tree structures. Ensemble methods address this by:

1. **Variance Reduction**: Averaging multiple trees reduces prediction variance
2. **Bias-Variance Trade-off**: Maintains low bias while reducing variance
3. **Robustness**: Less sensitive to noise and outliers

**Mathematical Justification:**
For independent trees with variance $`\sigma^2`$, the ensemble variance is $`\sigma^2/B`$. However, trees are typically correlated, so the actual variance reduction is less dramatic but still significant.

## 4.2.2. Bootstrap Sampling and Bagging

### Bootstrap Sampling

Bootstrap sampling is a resampling technique that creates multiple datasets by sampling with replacement from the original training data.

**Mathematical Definition:**
Given training data $`\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n`$, a bootstrap sample $`\mathcal{D}^{(b)}`$ is created by:
1. Randomly selecting $`n`$ samples with replacement from $`\mathcal{D}`$
2. Some samples may appear multiple times, others may not appear at all

**Expected Number of Unique Samples:**
In a bootstrap sample of size $`n`$, the expected number of unique samples is:

```math
E[\text{unique samples}] = n \left(1 - \left(1 - \frac{1}{n}\right)^n\right) \approx n(1 - e^{-1}) \approx 0.632n
```

This means approximately 36.8% of the original samples are not included in each bootstrap sample.

### Out-of-Bag (OOB) Samples

The samples not included in a bootstrap sample are called **Out-of-Bag (OOB)** samples. These serve as a natural validation set for each tree.

**OOB Estimation:**
For each observation $`(x_i, y_i)`$, we can compute the OOB prediction by averaging predictions from trees where $`(x_i, y_i)`$ was not in the bootstrap sample:

```math
f_{\text{OOB}}(x_i) = \frac{1}{|\mathcal{T}_i|} \sum_{b \in \mathcal{T}_i} f_b(x_i)
```

where $`\mathcal{T}_i`$ is the set of trees where observation $`i`$ is OOB.

### Bootstrap Aggregation (Bagging)

Bagging combines predictions from multiple trees trained on bootstrap samples:

```math
f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)
```

where each tree $`f_b`$ is trained on bootstrap sample $`\mathcal{D}^{(b)}`$.

**Python Implementation:** [bootstrap_bagging.py](code/bootstrap_bagging.py)

The bootstrap sampling and bagging implementation includes:
- `bootstrap_sample()`: Create bootstrap samples with replacement
- `bagging_regression()`: Implement bagging for regression trees
- `predict_bagging()`: Make predictions using bagging ensemble
- `calculate_oob_score()`: Calculate Out-of-Bag score for validation
- Demonstration functions for bootstrap sampling properties and ensemble size analysis

## 4.2.3. Random Forest Algorithm

### Feature Subsampling

Random Forest extends bagging by introducing feature subsampling at each split. This decorrelates the trees and improves ensemble performance.

**Algorithm:**
1. For $`b = 1, 2, \ldots, B`$:
   - Draw bootstrap sample $`\mathcal{D}^{(b)}`$ from training data
   - Grow tree $`f_b`$ to maximum depth using the following rule:
     - At each split, randomly select $`m \leq p`$ features
     - Find the best split among the selected features
   - Output ensemble prediction: $`f_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x)`$

### Feature Subsampling Parameters

The number of features to consider at each split ($`m`$) is a key hyperparameter:

- **Classification**: $`m = \sqrt{p}`$ (square root of total features)
- **Regression**: $`m = p/3`$ (one-third of total features)
- **Alternative**: $`m = \log_2(p)`$ (logarithm of total features)

**Mathematical Justification:**
Feature subsampling serves two purposes:
1. **Decorrelation**: Reduces correlation between trees
2. **Computational Efficiency**: Reduces training time per tree

### Complete Random Forest Implementation

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py)

The complete Random Forest implementation includes:

- **RandomForestRegressor Class**: A comprehensive class with methods for training, prediction, and evaluation
- **Feature Subsampling**: Automatic feature selection at each split using various strategies (sqrt, log2, fraction)
- **Bootstrap Sampling**: Optional bootstrap sampling for training each tree
- **Out-of-Bag Estimation**: Built-in OOB score calculation for validation
- **Feature Importance**: Automatic calculation of feature importance scores
- **Demonstration Functions**: Complete examples with synthetic data and evaluation metrics

Key features:
- Configurable hyperparameters (n_trees, max_features, max_depth, etc.)
- Support for different feature subsampling strategies
- Built-in OOB estimation for model validation
- Comprehensive feature importance analysis
- Integration with scikit-learn for data handling and evaluation

## 4.2.4. Variable Importance Measures

Random Forest provides two main approaches for measuring variable importance:

### 1. RSS-Based Importance

This measure quantifies the total reduction in RSS attributable to each feature across all trees.

**Mathematical Definition:**
For feature $`j`$, the importance is calculated as:

```math
\text{Importance}_j = \frac{1}{B} \sum_{b=1}^B \sum_{t \in \mathcal{T}_b^{(j)}} \Delta \text{RSS}_t
```

where:
- $`\mathcal{T}_b^{(j)}`$ is the set of nodes in tree $`b`$ that split on feature $`j`$
- $`\Delta \text{RSS}_t`$ is the RSS reduction at node $`t`$

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `calculate_rss_importance()` function

The RSS-based importance calculation aggregates feature importances from all trees in the ensemble.

### 2. Permutation Importance

This measure evaluates the increase in prediction error when a feature is randomly permuted.

**Algorithm:**
1. Calculate baseline prediction error using OOB samples
2. For each feature $`j`$:
   - Permute feature $`j`$ in OOB samples
   - Recalculate prediction error
   - Importance = (permuted error - baseline error)
3. Average importance across all trees

**Mathematical Definition:**
```math
\text{Permutation Importance}_j = \frac{1}{B} \sum_{b=1}^B \left(\text{Err}_{\text{perm}}^{(b)} - \text{Err}_{\text{baseline}}^{(b)}\right)
```

where $`\text{Err}_{\text{perm}}^{(b)}`$ is the OOB error after permuting feature $`j`$ in tree $`b`$.

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `calculate_permutation_importance()` function

The permutation importance implementation evaluates feature importance by measuring the increase in prediction error when features are randomly permuted.

### Handling High-Cardinality Variables

High-cardinality categorical variables can appear artificially important due to their increased partitioning power. To address this:

1. **Feature Engineering**: Create meaningful aggregations
2. **Regularization**: Use feature subsampling more aggressively
3. **Alternative Importance Measures**: Use permutation importance instead of RSS-based importance

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `handle_high_cardinality_features()` function

The high-cardinality feature handling implementation creates binary features for the most frequent categories and removes the original feature to prevent artificial importance inflation.

## 4.2.5. Hyperparameter Tuning

### Key Hyperparameters

1. **n_trees**: Number of trees in forest
2. **max_features**: Number of features to consider at each split
3. **max_depth**: Maximum depth of trees
4. **min_samples_split**: Minimum samples required to split
5. **min_samples_leaf**: Minimum samples required at leaf node

### Grid Search Implementation

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `tune_random_forest()` and `demonstrate_hyperparameter_tuning()` functions

The hyperparameter tuning implementation includes:
- Grid search over key hyperparameters (n_trees, max_features, max_depth, etc.)
- Cross-validation for robust parameter selection
- Integration with scikit-learn's GridSearchCV
- Complete demonstration with synthetic data and evaluation

## 4.2.6. R Implementation

**Complete R Implementation:** [r_random_forest.R](code/r_random_forest.R)

The R implementation provides:

- **Random Forest Training**: Complete implementation using the `randomForest` package
- **Bootstrap Sampling**: Demonstration of bootstrap sampling properties
- **Bagging**: Comparison between single trees and bagging ensembles
- **Variable Importance**: Built-in permutation importance calculation
- **Hyperparameter Tuning**: Grid search implementation with cross-validation
- **Visualization**: Comprehensive plotting functions for model analysis
- **Ensemble Size Analysis**: Analysis of the effect of ensemble size on performance
- **Partial Dependence Plots**: Tools for understanding feature effects
- **Confidence Intervals**: Prediction intervals using tree quantiles

Key features:
- Uses `randomForest` package for efficient implementation
- Built-in OOB estimation and variable importance
- Comprehensive visualization tools
- Integration with `MASS` package for dataset access
- Modular function design for easy customization and analysis

## 4.2.7. Advanced Topics

### Partial Dependence Plots

Partial dependence plots show the marginal effect of a feature on predictions.

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `partial_dependence_plot()` function

The partial dependence plot implementation:
- Generates a range of feature values
- Calculates average predictions for each value
- Visualizes the marginal effect of the feature on model predictions

### Confidence Intervals

Random Forest can provide prediction intervals using quantiles of tree predictions.

**Python Implementation:** [random_forest_implementation.py](code/random_forest_implementation.py) - `predict_with_intervals()` function

The confidence interval implementation:
- Collects predictions from all trees in the ensemble
- Calculates quantiles to determine confidence bounds
- Returns mean predictions with lower and upper confidence bounds

## Summary

Random Forest is a powerful ensemble method that addresses the high variance problem of single decision trees through:

1. **Bootstrap Aggregation**: Reduces variance by averaging multiple trees
2. **Feature Subsampling**: Decorrelates trees and improves ensemble diversity
3. **Out-of-Bag Estimation**: Provides unbiased error estimates
4. **Variable Importance**: Offers insights into feature relevance
5. **Robustness**: Handles outliers and noise effectively

The mathematical foundations ensure optimal performance, while the algorithmic design provides computational efficiency and interpretability through variable importance measures.

## Code Files Summary

The following code files provide complete implementations of the concepts discussed in this chapter:

### Python Implementation
- **[bootstrap_bagging.py](code/bootstrap_bagging.py)**: Bootstrap sampling, bagging implementation, and ensemble size analysis
- **[random_forest_implementation.py](code/random_forest_implementation.py)**: Complete Random Forest implementation with feature subsampling, variable importance, hyperparameter tuning, and advanced features

### R Implementation
- **[r_random_forest.R](code/r_random_forest.R)**: Complete R implementation using randomForest package with training, evaluation, visualization, and analysis tools

Each file includes comprehensive examples, demonstrations, and analysis tools to help understand and apply Random Forest concepts in practice.

## References

- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- Liaw, A., & Wiener, M. (2002). Classification and regression by randomForest. R news, 2(3), 18-22.

---

**Navigation:**
- **Next Topic:** [Gradient Boosting Machines](03_gbm.md) - Sequential ensemble learning with gradient descent
- **Previous Topic:** [Regression Trees](01_regression_trees.md) - Understanding tree structure and recursive binary splitting
