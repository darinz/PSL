# 4.3. Gradient Boosting Machines (GBM)

## 4.3.1. Introduction to Boosting

Gradient Boosting Machines (GBM) represent a powerful ensemble learning technique that builds strong predictive models by combining multiple weak learners in a sequential manner. Unlike Random Forest, which builds trees independently and averages their predictions, GBM builds trees sequentially, with each tree correcting the errors of its predecessors.

### Mathematical Framework

Consider a regression problem with:
- **Input features**: $`X = (X_1, X_2, \ldots, X_p) \in \mathbb{R}^p`$
- **Response variable**: $`Y \in \mathbb{R}`$
- **Training data**: $`\{(x_i, y_i)\}_{i=1}^n`$

The GBM model is an additive model of the form:

```math
F(x) = \sum_{t=1}^T f_t(x)
```

where:
- $`f_t(x)`$ is the $`t`$-th weak learner (typically a regression tree)
- $`T`$ is the number of boosting iterations
- Each $`f_t`$ is trained to predict the residuals from the previous iteration

### Loss Function and Optimization

GBM minimizes a loss function $`L(y, F(x))`$ by finding the optimal additive expansion. For regression, the most common loss function is the squared error:

```math
L(y, F(x)) = \frac{1}{2}(y - F(x))^2
```

The optimization problem is:

```math
\min_{F} \sum_{i=1}^n L(y_i, F(x_i))
```

### Forward Stagewise Additive Modeling

GBM uses a forward stagewise approach to solve this optimization problem:

1. **Initialize**: $`F_0(x) = \arg\min_{\gamma} \sum_{i=1}^n L(y_i, \gamma)`$
2. **For** $`t = 1, 2, \ldots, T`$:
   - Compute residuals: $`r_{it} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x_i) = F_{t-1}(x_i)}`$
   - Fit weak learner $`f_t`$ to residuals $`\{r_{it}\}_{i=1}^n`$
   - Update: $`F_t(x) = F_{t-1}(x) + \eta f_t(x)`$

where $`\eta`$ is the learning rate (shrinkage parameter).

## 4.3.2. Mathematical Derivation

### Gradient Descent Interpretation

GBM can be viewed as gradient descent in function space. At each iteration, we compute the negative gradient of the loss function with respect to the current model:

```math
r_{it} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \bigg|_{F(x_i) = F_{t-1}(x_i)}
```

For squared error loss:
```math
L(y, F(x)) = \frac{1}{2}(y - F(x))^2
```

The gradient is:
```math
\frac{\partial L(y, F(x))}{\partial F(x)} = -(y - F(x))
```

Therefore, the residuals are simply:
```math
r_{it} = y_i - F_{t-1}(x_i)
```

### Tree Fitting to Residuals

At each iteration, we fit a regression tree to the residuals. The tree minimizes:

```math
\sum_{i=1}^n (r_{it} - f_t(x_i))^2
```

This is equivalent to finding the best split that minimizes the sum of squared errors within each leaf node.

### Learning Rate and Regularization

The learning rate $`\eta`$ controls the contribution of each tree:

```math
F_t(x) = F_{t-1}(x) + \eta f_t(x)
```

A smaller learning rate requires more trees but can lead to better generalization. The optimal learning rate is typically found through cross-validation.

## 4.3.3. Complete GBM Implementation

### Python Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, 
                 random_state=None):
        """
        Gradient Boosting Regressor
        
        Parameters:
        n_estimators: number of boosting iterations
        learning_rate: learning rate (shrinkage)
        max_depth: maximum depth of trees
        min_samples_split: minimum samples required to split
        min_samples_leaf: minimum samples required at leaf node
        subsample: fraction of samples used for each tree
        random_state: random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        """Train Gradient Boosting model"""
        np.random.seed(self.random_state)
        
        n_samples = len(y)
        
        # Initialize with mean of target
        self.initial_prediction = np.mean(y)
        F = np.full(n_samples, self.initial_prediction)
        
        self.trees = []
        self.train_scores = []
        
        for t in range(self.n_estimators):
            # Compute residuals (negative gradients)
            residuals = y - F
            
            # Subsample data if specified
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, size=n_subsample, replace=False)
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=t
            )
            tree.fit(X_sub, residuals_sub)
            
            # Update predictions
            tree_pred = tree.predict(X)
            F += self.learning_rate * tree_pred
            
            # Store tree
            self.trees.append(tree)
            
            # Calculate training score
            train_score = mean_squared_error(y, F)
            self.train_scores.append(train_score)
            
            # Early stopping (optional)
            if t > 10 and abs(self.train_scores[-1] - self.train_scores[-2]) < 1e-6:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def staged_predict(self, X):
        """Make predictions at each stage"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            yield predictions.copy()

# Example usage
def demonstrate_gbm():
    """Demonstrate GBM on synthetic data"""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                          noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train GBM
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gbm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gbm.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(gbm.train_scores)
    plt.xlabel('Iteration')
    plt.ylabel('Training MSE')
    plt.title('Training Progress')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('GBM Predictions')
    
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    return gbm

# Run demonstration
if __name__ == "__main__":
    gbm_model = demonstrate_gbm()
```

### Advanced GBM Features

```python
class AdvancedGBMRegressor(GradientBoostingRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, random_state=None):
        """
        Advanced GBM with additional features
        """
        super().__init__(n_estimators, learning_rate, max_depth, 
                        min_samples_split, min_samples_leaf, subsample, random_state)
        self.colsample_bytree = colsample_bytree
        self.feature_importances_ = None
        
    def fit(self, X, y, validation_data=None):
        """Train with validation monitoring"""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.initial_prediction = np.mean(y)
        F = np.full(n_samples, self.initial_prediction)
        
        self.trees = []
        self.train_scores = []
        self.val_scores = []
        feature_importances = np.zeros(n_features)
        
        for t in range(self.n_estimators):
            # Compute residuals
            residuals = y - F
            
            # Subsample data
            if self.subsample < 1.0:
                n_subsample = int(self.subsample * n_samples)
                indices = np.random.choice(n_samples, size=n_subsample, replace=False)
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals
            
            # Feature subsampling
            if self.colsample_bytree < 1.0:
                n_features_sub = int(self.colsample_bytree * n_features)
                feature_indices = np.random.choice(n_features, size=n_features_sub, replace=False)
                X_sub = X_sub[:, feature_indices]
            else:
                feature_indices = np.arange(n_features)
            
            # Fit tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=t
            )
            tree.fit(X_sub, residuals_sub)
            
            # Update predictions
            if self.colsample_bytree < 1.0:
                X_pred = X[:, feature_indices]
                tree_pred = tree.predict(X_pred)
            else:
                tree_pred = tree.predict(X)
            
            F += self.learning_rate * tree_pred
            
            # Store tree and feature indices
            self.trees.append((tree, feature_indices))
            
            # Update feature importances
            if hasattr(tree, 'feature_importances_'):
                feature_importances[feature_indices] += tree.feature_importances_
            
            # Calculate scores
            train_score = mean_squared_error(y, F)
            self.train_scores.append(train_score)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.predict(X_val)
                val_score = mean_squared_error(y_val, y_val_pred)
                self.val_scores.append(val_score)
        
        # Average feature importances
        self.feature_importances_ = feature_importances / self.n_estimators
        
        return self
    
    def predict(self, X):
        """Make predictions with feature subsampling"""
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree, feature_indices in self.trees:
            X_sub = X[:, feature_indices]
            predictions += self.learning_rate * tree.predict(X_sub)
        
        return predictions
```

## 4.3.4. Hyperparameter Tuning

### Key Hyperparameters

1. **n_estimators**: Number of boosting iterations
2. **learning_rate**: Shrinkage parameter (typically 0.01-0.3)
3. **max_depth**: Maximum depth of trees (typically 3-8)
4. **subsample**: Fraction of samples used per tree
5. **colsample_bytree**: Fraction of features used per tree

### Grid Search Implementation

```python
from sklearn.model_selection import GridSearchCV, train_test_split

def tune_gbm_hyperparameters(X, y):
    """
    Tune GBM hyperparameters using grid search
    """
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    best_score = float('inf')
    best_params = None
    best_model = None
    
    # Grid search
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        
                        # Train model
                        gbm = AdvancedGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            random_state=42
                        )
                        
                        gbm.fit(X_train, y_train, validation_data=(X_val, y_val))
                        
                        # Evaluate
                        y_val_pred = gbm.predict(X_val)
                        val_score = mean_squared_error(y_val, y_val_pred)
                        
                        if val_score < best_score:
                            best_score = val_score
                            best_params = {
                                'n_estimators': n_estimators,
                                'learning_rate': learning_rate,
                                'max_depth': max_depth,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree
                            }
                            best_model = gbm
    
    print("Best parameters:", best_params)
    print("Best validation MSE:", best_score)
    
    return best_model, best_params

# Example usage
def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning"""
    from sklearn.datasets import make_regression
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                          noise=0.1, random_state=42)
    
    # Tune hyperparameters
    best_gbm, best_params = tune_gbm_hyperparameters(X, y)
    
    # Evaluate on test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    best_gbm.fit(X_train, y_train)
    y_pred = best_gbm.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return best_gbm
```

## 4.3.5. R Implementation

```r
# GBM Implementation in R
library(gbm)
library(ggplot2)
library(dplyr)

# Function to demonstrate GBM
demonstrate_gbm_r <- function() {
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
  
  # Train GBM
  gbm_model <- gbm(
    medv ~ .,
    data = data.frame(X_train, medv = y_train),
    distribution = "gaussian",
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = FALSE
  )
  
  # Find optimal number of trees
  best_iter <- gbm.perf(gbm_model, method = "cv")
  
  # Make predictions
  predictions <- predict(gbm_model, X_test, n.trees = best_iter)
  
  # Calculate metrics
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Test MSE:", round(mse, 4), "\n")
  cat("Test R²:", round(r2, 4), "\n")
  cat("Optimal trees:", best_iter, "\n")
  
  # Variable importance
  importance_df <- summary(gbm_model, n.trees = best_iter, plotit = FALSE)
  
  print("Variable Importance:")
  print(importance_df)
  
  # Visualize results
  par(mfrow = c(2, 2))
  
  # CV error vs number of trees
  plot(gbm_model$cv.error, type = "l", xlab = "Number of Trees",
       ylab = "CV Error", main = "Cross-Validation Error")
  abline(v = best_iter, col = "red", lty = 2)
  
  # Predictions vs actual
  plot(y_test, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "GBM Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  # Variable importance plot
  barplot(importance_df$rel.inf, names.arg = importance_df$var,
          main = "Variable Importance", las = 2)
  
  # Residuals
  residuals <- y_test - predictions
  plot(predictions, residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residual Plot")
  abline(h = 0, col = "red", lty = 2)
  
  return(gbm_model)
}

# Function to tune hyperparameters
tune_gbm_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Define parameter grid
  param_grid <- expand.grid(
    n.trees = c(50, 100, 200),
    interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1, 0.2),
    bag.fraction = c(0.5, 0.8, 1.0)
  )
  
  # Train models
  results <- list()
  for (i in 1:nrow(param_grid)) {
    cat("Training model", i, "of", nrow(param_grid), "\n")
    
    gbm_model <- gbm(
      medv ~ .,
      data = Boston,
      distribution = "gaussian",
      n.trees = param_grid$n.trees[i],
      interaction.depth = param_grid$interaction.depth[i],
      shrinkage = param_grid$shrinkage[i],
      bag.fraction = param_grid$bag.fraction[i],
      cv.folds = 5,
      verbose = FALSE
    )
    
    # Get CV error
    best_iter <- gbm.perf(gbm_model, method = "cv", plotit = FALSE)
    cv_error <- gbm_model$cv.error[best_iter]
    
    results[[i]] <- cv_error
  }
  
  # Find best parameters
  best_idx <- which.min(unlist(results))
  best_params <- param_grid[best_idx, ]
  
  cat("Best parameters:\n")
  cat("n.trees:", best_params$n.trees, "\n")
  cat("interaction.depth:", best_params$interaction.depth, "\n")
  cat("shrinkage:", best_params$shrinkage, "\n")
  cat("bag.fraction:", best_params$bag.fraction, "\n")
  cat("Best CV MSE:", results[[best_idx]], "\n")
  
  return(best_params)
}

# Run demonstrations
gbm_model_r <- demonstrate_gbm_r()
best_params_r <- tune_gbm_r()
```

## 4.3.6. Comparison with Random Forest

### Key Differences

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| **Training** | Parallel | Sequential |
| **Bias-Variance** | Low bias, high variance | Low bias, low variance |
| **Overfitting** | Less prone | More prone |
| **Tuning** | Fewer parameters | More parameters |
| **Speed** | Faster training | Slower training |
| **Interpretability** | Good | Moderate |

### Mathematical Comparison

**Random Forest Variance:**
```math
\text{Var}(\hat{f}_{\text{RF}}) = \frac{\sigma^2}{B} + \rho \sigma^2 \left(1 - \frac{1}{B}\right)
```

**Gradient Boosting Variance:**
```math
\text{Var}(\hat{f}_{\text{GBM}}) = \sigma^2 \sum_{t=1}^T \eta^2 (1 - \rho)^t
```

where $`\rho`$ is the correlation between trees and $`\eta`$ is the learning rate.

### Performance Comparison Code

```python
def compare_rf_gbm(X, y):
    """
    Compare Random Forest and GBM performance
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # GBM
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Cross-validation scores
    rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    gbm_scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Train final models
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    
    # Test predictions
    rf_pred = rf.predict(X_test)
    gbm_pred = gbm.predict(X_test)
    
    # Results
    results = {
        'Random Forest': {
            'CV MSE': -rf_scores.mean(),
            'CV Std': rf_scores.std(),
            'Test MSE': mean_squared_error(y_test, rf_pred),
            'Test R²': r2_score(y_test, rf_pred)
        },
        'Gradient Boosting': {
            'CV MSE': -gbm_scores.mean(),
            'CV Std': gbm_scores.std(),
            'Test MSE': mean_squared_error(y_test, gbm_pred),
            'Test R²': r2_score(y_test, gbm_pred)
        }
    }
    
    print("Performance Comparison:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results
```

## 4.3.7. Advanced Topics

### Early Stopping

```python
def gbm_with_early_stopping(X_train, y_train, X_val, y_val, patience=10):
    """
    GBM with early stopping based on validation performance
    """
    gbm = GradientBoostingRegressor(
        n_estimators=1000,  # Large number
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    best_val_score = float('inf')
    best_iteration = 0
    patience_counter = 0
    
    # Initialize
    F = np.full(len(X_train), np.mean(y_train))
    val_predictions = np.full(len(X_val), np.mean(y_train))
    
    for t in range(gbm.n_estimators):
        # Fit tree to residuals
        residuals = y_train - F
        
        tree = DecisionTreeRegressor(max_depth=3, random_state=t)
        tree.fit(X_train, residuals)
        
        # Update predictions
        tree_pred_train = tree.predict(X_train)
        tree_pred_val = tree.predict(X_val)
        
        F += gbm.learning_rate * tree_pred_train
        val_predictions += gbm.learning_rate * tree_pred_val
        
        # Check validation performance
        val_score = mean_squared_error(y_val, val_predictions)
        
        if val_score < best_val_score:
            best_val_score = val_score
            best_iteration = t
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at iteration {t}")
            break
    
    return best_iteration, best_val_score
```

### Feature Importance Analysis

```python
def analyze_gbm_feature_importance(gbm_model, X, feature_names=None):
    """
    Analyze feature importance in GBM
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Calculate feature importance based on RSS reduction
    importance = np.zeros(X.shape[1])
    
    for tree in gbm_model.trees:
        if hasattr(tree, 'feature_importances_'):
            importance += tree.feature_importances_
    
    importance /= len(gbm_model.trees)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('GBM Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df
```

## Summary

Gradient Boosting Machines provide a powerful approach to regression through:

1. **Sequential Learning**: Each tree corrects the errors of previous trees
2. **Gradient Descent**: Optimizes loss function in function space
3. **Regularization**: Learning rate and subsampling prevent overfitting
4. **Flexibility**: Can handle various loss functions and weak learners
5. **Performance**: Often achieves state-of-the-art results with proper tuning

The mathematical foundations ensure optimal convergence, while the algorithmic design provides both computational efficiency and predictive power. GBM requires careful hyperparameter tuning but can outperform Random Forest when properly configured.

## References

- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189-1232.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).
