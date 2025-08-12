"""
Gradient Boosting Machines (GBM) Implementation
==============================================

This module provides a complete implementation of Gradient Boosting Machines
for regression, including basic and advanced features.
"""

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

if __name__ == "__main__":
    gbm_model = demonstrate_gbm()
