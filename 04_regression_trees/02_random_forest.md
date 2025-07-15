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

**Implementation:**
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def bootstrap_sample(X, y, n_samples=None):
    """
    Create a bootstrap sample from the data
    
    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target vector (n_samples,)
    n_samples: number of samples to draw (default: n_samples)
    
    Returns:
    X_boot, y_boot: bootstrap sample
    """
    if n_samples is None:
        n_samples = len(y)
    
    # Sample with replacement
    indices = np.random.choice(len(y), size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]
    
    return X_boot, y_boot

def bagging_regression(X, y, n_trees=100, max_depth=None, min_samples_split=2):
    """
    Implement bagging for regression trees
    
    Parameters:
    X: feature matrix
    y: target vector
    n_trees: number of trees in ensemble
    max_depth: maximum depth of each tree
    min_samples_split: minimum samples required to split
    
    Returns:
    trees: list of trained trees
    """
    trees = []
    
    for b in range(n_trees):
        # Create bootstrap sample
        X_boot, y_boot = bootstrap_sample(X, y)
        
        # Train tree on bootstrap sample
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=b
        )
        tree.fit(X_boot, y_boot)
        trees.append(tree)
    
    return trees

def predict_bagging(trees, X):
    """
    Make predictions using bagging ensemble
    
    Parameters:
    trees: list of trained trees
    X: feature matrix for prediction
    
    Returns:
    predictions: ensemble predictions
    """
    predictions = np.zeros(len(X))
    
    for tree in trees:
        predictions += tree.predict(X)
    
    return predictions / len(trees)
```

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

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class RandomForestRegressor:
    def __init__(self, n_trees=100, max_features='sqrt', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True, 
                 random_state=None):
        """
        Random Forest Regressor
        
        Parameters:
        n_trees: number of trees in forest
        max_features: number of features to consider at each split
        max_depth: maximum depth of trees
        min_samples_split: minimum samples required to split
        min_samples_leaf: minimum samples required at leaf node
        bootstrap: whether to use bootstrap samples
        random_state: random seed
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        
    def _get_max_features(self, n_features):
        """Determine number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = len(y)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Train Random Forest"""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        feature_importances = np.zeros(n_features)
        
        for b in range(self.n_trees):
            # Create bootstrap sample
            if self.bootstrap:
                X_boot, y_boot = self._bootstrap_sample(X, y)
            else:
                X_boot, y_boot = X, y
            
            # Train tree with feature subsampling
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                random_state=b
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Accumulate feature importances
            feature_importances += tree.feature_importances_
        
        # Average feature importances
        self.feature_importances_ = feature_importances / self.n_trees
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = np.zeros(len(X))
        
        for tree in self.trees:
            predictions += tree.predict(X)
        
        return predictions / self.n_trees
    
    def get_oob_score(self, X, y):
        """Calculate Out-of-Bag score"""
        oob_predictions = np.zeros(len(y))
        oob_counts = np.zeros(len(y))
        
        for b, tree in enumerate(self.trees):
            # Find OOB samples for this tree
            if self.bootstrap:
                # This is a simplified version - in practice, you'd track OOB samples during training
                indices = np.random.choice(len(y), size=len(y), replace=True)
                oob_mask = ~np.isin(np.arange(len(y)), indices)
            else:
                oob_mask = np.ones(len(y), dtype=bool)
            
            if np.any(oob_mask):
                oob_pred = tree.predict(X[oob_mask])
                oob_predictions[oob_mask] += oob_pred
                oob_counts[oob_mask] += 1
        
        # Average OOB predictions
        valid_oob = oob_counts > 0
        oob_predictions[valid_oob] /= oob_counts[valid_oob]
        
        # Calculate OOB score
        oob_score = r2_score(y[valid_oob], oob_predictions[valid_oob])
        return oob_score

# Example usage
def demonstrate_random_forest():
    """Demonstrate Random Forest on synthetic data"""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                          noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    rf = RandomForestRegressor(n_trees=100, max_features='sqrt', 
                              max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    oob_score = rf.get_oob_score(X_train, y_train)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"OOB Score: {oob_score:.4f}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance_pairs = list(enumerate(rf.feature_importances_))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature_idx, importance) in enumerate(feature_importance_pairs[:10]):
        print(f"  Feature {feature_idx}: {importance:.4f}")
    
    return rf

# Run demonstration
if __name__ == "__main__":
    rf_model = demonstrate_random_forest()
```

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

**Implementation:**
```python
def calculate_rss_importance(trees, X, y):
    """
    Calculate RSS-based feature importance
    """
    n_features = X.shape[1]
    importance = np.zeros(n_features)
    
    for tree in trees:
        # Get feature importances from sklearn tree
        importance += tree.feature_importances_
    
    return importance / len(trees)
```

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

**Implementation:**
```python
def calculate_permutation_importance(rf_model, X, y, n_repeats=5):
    """
    Calculate permutation-based feature importance
    
    Parameters:
    rf_model: trained Random Forest model
    X: feature matrix
    y: target vector
    n_repeats: number of times to repeat permutation
    
    Returns:
    importance: permutation importance scores
    """
    n_features = X.shape[1]
    importance = np.zeros(n_features)
    
    # Calculate baseline OOB error
    baseline_error = 1 - rf_model.get_oob_score(X, y)
    
    for j in range(n_features):
        feature_importance = 0
        
        for repeat in range(n_repeats):
            # Create copy of data with permuted feature
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])
            
            # Calculate error with permuted feature
            # This is a simplified version - in practice, you'd need to track OOB samples
            perm_error = 1 - rf_model.get_oob_score(X_perm, y)
            
            feature_importance += (perm_error - baseline_error)
        
        importance[j] = feature_importance / n_repeats
    
    return importance
```

### Handling High-Cardinality Variables

High-cardinality categorical variables can appear artificially important due to their increased partitioning power. To address this:

1. **Feature Engineering**: Create meaningful aggregations
2. **Regularization**: Use feature subsampling more aggressively
3. **Alternative Importance Measures**: Use permutation importance instead of RSS-based importance

**Example:**
```python
def handle_high_cardinality_features(X, y, categorical_features, max_categories=10):
    """
    Handle high-cardinality categorical features
    
    Parameters:
    X: feature matrix
    y: target vector
    categorical_features: list of categorical feature indices
    max_categories: maximum number of categories to keep
    
    Returns:
    X_processed: processed feature matrix
    """
    X_processed = X.copy()
    
    for feature_idx in categorical_features:
        unique_values, counts = np.unique(X[:, feature_idx], return_counts=True)
        
        if len(unique_values) > max_categories:
            # Keep top categories by frequency
            top_categories = unique_values[np.argsort(counts)[-max_categories:]]
            
            # Create binary features for top categories
            for i, category in enumerate(top_categories):
                X_processed = np.column_stack([
                    X_processed, 
                    (X[:, feature_idx] == category).astype(int)
                ])
            
            # Remove original feature
            X_processed = np.delete(X_processed, feature_idx, axis=1)
    
    return X_processed
```

## 4.2.5. Hyperparameter Tuning

### Key Hyperparameters

1. **n_trees**: Number of trees in forest
2. **max_features**: Number of features to consider at each split
3. **max_depth**: Maximum depth of trees
4. **min_samples_split**: Minimum samples required to split
5. **min_samples_leaf**: Minimum samples required at leaf node

### Grid Search Implementation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

def tune_random_forest(X, y):
    """
    Tune Random Forest hyperparameters using grid search
    """
    param_grid = {
        'n_trees': [50, 100, 200],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Random Forest model
    rf = RandomForestRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", -grid_search.best_score_)
    
    return grid_search.best_estimator_

# Example usage
def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning"""
    from sklearn.datasets import make_regression
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                          noise=0.1, random_state=42)
    
    # Tune hyperparameters
    best_rf = tune_random_forest(X, y)
    
    # Evaluate on test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return best_rf
```

## 4.2.6. R Implementation

```r
# Random Forest Implementation in R
library(randomForest)
library(ggplot2)
library(dplyr)

# Function to demonstrate Random Forest
demonstrate_random_forest_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Prepare data
  X <- Boston[, -ncol(Boston)]
  y <- Boston$medv
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Train Random Forest
  rf_model <- randomForest(
    medv ~ ., 
    data = data.frame(X_train, medv = y_train),
    ntree = 100,
    mtry = sqrt(ncol(X_train)),  # sqrt(p) for regression
    importance = TRUE,
    keep.forest = TRUE
  )
  
  # Make predictions
  predictions <- predict(rf_model, X_test)
  
  # Calculate metrics
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Test MSE:", round(mse, 4), "\n")
  cat("Test R²:", round(r2, 4), "\n")
  cat("OOB MSE:", round(rf_model$mse[length(rf_model$mse)], 4), "\n")
  
  # Variable importance
  importance_df <- data.frame(
    feature = rownames(importance(rf_model)),
    importance = importance(rf_model)[, "%IncMSE"]
  ) %>%
    arrange(desc(importance))
  
  print("Variable Importance (Permutation):")
  print(importance_df)
  
  # Visualize results
  par(mfrow = c(2, 2))
  
  # Predictions vs actual
  plot(y_test, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Random Forest Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  # Variable importance plot
  varImpPlot(rf_model, main = "Variable Importance")
  
  # OOB error vs number of trees
  plot(rf_model$mse, type = "l", xlab = "Number of Trees",
       ylab = "OOB MSE", main = "OOB Error vs Number of Trees")
  
  # Residuals
  residuals <- y_test - predictions
  plot(predictions, residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residual Plot")
  abline(h = 0, col = "red", lty = 2)
  
  return(rf_model)
}

# Function to tune hyperparameters
tune_random_forest_r <- function() {
  library(caret)
  
  # Load data
  data(Boston, package = "MASS")
  
  # Define parameter grid
  param_grid <- expand.grid(
    mtry = c(2, 4, 6, 8),
    ntree = c(50, 100, 200)
  )
  
  # Control for cross-validation
  control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
  
  # Train models
  results <- list()
  for (i in 1:nrow(param_grid)) {
    cat("Training model", i, "of", nrow(param_grid), "\n")
    
    rf_model <- randomForest(
      medv ~ .,
      data = Boston,
      mtry = param_grid$mtry[i],
      ntree = param_grid$ntree[i]
    )
    
    # Cross-validation score
    cv_scores <- numeric(5)
    for (fold in 1:5) {
      # Simple CV implementation
      test_indices <- sample(1:nrow(Boston), size = nrow(Boston) %/% 5)
      train_data <- Boston[-test_indices, ]
      test_data <- Boston[test_indices, ]
      
      fold_model <- randomForest(
        medv ~ .,
        data = train_data,
        mtry = param_grid$mtry[i],
        ntree = param_grid$ntree[i]
      )
      
      predictions <- predict(fold_model, test_data)
      cv_scores[fold] <- mean((test_data$medv - predictions)^2)
    }
    
    results[[i]] <- mean(cv_scores)
  }
  
  # Find best parameters
  best_idx <- which.min(unlist(results))
  best_params <- param_grid[best_idx, ]
  
  cat("Best parameters:\n")
  cat("mtry:", best_params$mtry, "\n")
  cat("ntree:", best_params$ntree, "\n")
  cat("Best CV MSE:", results[[best_idx]], "\n")
  
  return(best_params)
}

# Run demonstrations
rf_model_r <- demonstrate_random_forest_r()
best_params_r <- tune_random_forest_r()
```

## 4.2.7. Advanced Topics

### Partial Dependence Plots

Partial dependence plots show the marginal effect of a feature on predictions:

```python
def partial_dependence_plot(rf_model, X, feature_idx, feature_names=None):
    """
    Create partial dependence plot for a feature
    """
    feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
    
    # Generate feature values
    feature_values = np.linspace(X[:, feature_idx].min(), 
                                X[:, feature_idx].max(), 50)
    
    # Calculate partial dependence
    pd_values = []
    for val in feature_values:
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        predictions = rf_model.predict(X_temp)
        pd_values.append(np.mean(predictions))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, pd_values, 'b-', linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return feature_values, pd_values
```

### Confidence Intervals

Random Forest can provide prediction intervals using quantiles of tree predictions:

```python
def predict_with_intervals(rf_model, X, confidence=0.95):
    """
    Make predictions with confidence intervals
    """
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in rf_model.trees])
    
    # Calculate quantiles
    alpha = 1 - confidence
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    mean_pred = np.mean(tree_predictions, axis=0)
    lower_bound = np.quantile(tree_predictions, lower_quantile, axis=0)
    upper_bound = np.quantile(tree_predictions, upper_quantile, axis=0)
    
    return mean_pred, lower_bound, upper_bound
```

## Summary

Random Forest is a powerful ensemble method that addresses the high variance problem of single decision trees through:

1. **Bootstrap Aggregation**: Reduces variance by averaging multiple trees
2. **Feature Subsampling**: Decorrelates trees and improves ensemble diversity
3. **Out-of-Bag Estimation**: Provides unbiased error estimates
4. **Variable Importance**: Offers insights into feature relevance
5. **Robustness**: Handles outliers and noise effectively

The mathematical foundations ensure optimal performance, while the algorithmic design provides computational efficiency and interpretability through variable importance measures.

## References

- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- Liaw, A., & Wiener, M. (2002). Classification and regression by randomForest. R news, 2(3), 18-22.
