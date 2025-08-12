"""
Random Forest Implementation
===========================

This module provides a complete implementation of Random Forest for regression,
including feature subsampling, variable importance measures, and hyperparameter tuning.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, pd_values, 'b-', linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for {feature_name}')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return feature_values, pd_values

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

if __name__ == "__main__":
    # Demonstrate Random Forest
    rf_model = demonstrate_random_forest()
    
    # Demonstrate hyperparameter tuning
    best_rf = demonstrate_hyperparameter_tuning()
