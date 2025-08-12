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

**Complete GBM Implementation:** [gbm_implementation.py](code/gbm_implementation.py)

The basic GBM implementation includes:

- **GradientBoostingRegressor Class**: Complete implementation with training, prediction, and evaluation
- **Sequential Learning**: Each tree corrects the errors of previous trees
- **Subsampling**: Optional data subsampling for regularization
- **Training Progress Monitoring**: Tracks training scores during iterations
- **Demonstration Function**: Complete example with synthetic data and visualization

Key features:
- Configurable hyperparameters (n_estimators, learning_rate, max_depth, etc.)
- Built-in training progress monitoring
- Optional early stopping based on convergence
- Comprehensive visualization of training progress and predictions

### Advanced GBM Features

**Advanced GBM Implementation:** [advanced_gbm.py](code/advanced_gbm.py)

The advanced GBM implementation includes:

- **AdvancedGBMRegressor Class**: Extended implementation with feature subsampling and validation monitoring
- **Feature Subsampling**: Column-wise subsampling for additional regularization
- **Validation Monitoring**: Built-in validation score tracking during training
- **Feature Importance**: Automatic calculation of feature importance scores
- **Enhanced Prediction**: Support for feature subsampling in predictions

Key advanced features:
- Column-wise subsampling (colsample_bytree parameter)
- Validation data monitoring during training
- Comprehensive feature importance analysis
- Enhanced regularization capabilities

## 4.3.4. Hyperparameter Tuning

### Key Hyperparameters

1. **n_estimators**: Number of boosting iterations
2. **learning_rate**: Shrinkage parameter (typically 0.01-0.3)
3. **max_depth**: Maximum depth of trees (typically 3-8)
4. **subsample**: Fraction of samples used per tree
5. **colsample_bytree**: Fraction of features used per tree

### Grid Search Implementation

**Python Implementation:** [advanced_gbm.py](code/advanced_gbm.py) - `tune_gbm_hyperparameters()` and `demonstrate_hyperparameter_tuning()` functions

The hyperparameter tuning implementation includes:

- **Grid Search**: Comprehensive search over key hyperparameters
- **Validation Strategy**: Proper train/validation split for parameter selection
- **Parameter Grid**: Covers n_estimators, learning_rate, max_depth, subsample, and colsample_bytree
- **Best Model Selection**: Automatic selection of best performing model
- **Complete Demonstration**: End-to-end example with synthetic data

Key features:
- Systematic exploration of hyperparameter space
- Validation-based model selection
- Integration with advanced GBM implementation
- Comprehensive evaluation and reporting

## 4.3.5. R Implementation

**Complete R Implementation:** [r_gbm.R](code/r_gbm.R)

The R implementation provides:

- **GBM Training**: Complete implementation using the `gbm` package
- **Cross-Validation**: Built-in cross-validation for optimal tree selection
- **Hyperparameter Tuning**: Grid search implementation for parameter optimization
- **Variable Importance**: Built-in variable importance calculation and visualization
- **Model Comparison**: Comparison with Random Forest performance
- **Early Stopping**: Demonstration of early stopping based on CV performance
- **Learning Rate Analysis**: Analysis of learning rate effects on model performance

Key features:
- Uses `gbm` package for efficient implementation
- Built-in cross-validation and optimal tree selection
- Comprehensive hyperparameter tuning
- Integration with `MASS` package for dataset access
- Advanced visualization and analysis tools
- Modular function design for easy customization

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
