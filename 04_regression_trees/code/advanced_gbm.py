"""
Advanced Gradient Boosting Machines (GBM) Features
================================================

This module provides advanced GBM features including feature subsampling,
hyperparameter tuning, early stopping, and comparison with Random Forest.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class AdvancedGBMRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 colsample_bytree=1.0, random_state=None):
        """
        Advanced GBM with additional features
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None
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

def gbm_with_early_stopping(X_train, y_train, X_val, y_val, patience=10):
    """
    GBM with early stopping based on validation performance
    """
    gbm = AdvancedGBMRegressor(
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

def analyze_gbm_feature_importance(gbm_model, X, feature_names=None):
    """
    Analyze feature importance in GBM
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Calculate feature importance based on RSS reduction
    importance = np.zeros(X.shape[1])
    
    for tree, feature_indices in gbm_model.trees:
        if hasattr(tree, 'feature_importances_'):
            importance[feature_indices] += tree.feature_importances_
    
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

def compare_rf_gbm(X, y):
    """
    Compare Random Forest and GBM performance
    """
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
    gbm = AdvancedGBMRegressor(
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

if __name__ == "__main__":
    # Demonstrate hyperparameter tuning
    best_gbm = demonstrate_hyperparameter_tuning()
    
    # Demonstrate early stopping
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                          noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_iter, best_score = gbm_with_early_stopping(X_train, y_train, X_val, y_val)
    
    # Compare with Random Forest
    results = compare_rf_gbm(X, y)
