"""
Complete Regression Tree Implementation
=====================================

This module provides a complete implementation of regression trees including
building, pruning, and evaluation with comprehensive examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class RegressionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        """Build the regression tree"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return {'type': 'leaf', 'prediction': np.mean(y)}
        
        # Find best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return {'type': 'leaf', 'prediction': np.mean(y)}
        
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Check minimum samples per leaf
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            return {'type': 'leaf', 'prediction': np.mean(y)}
        
        # Create internal node
        node = {
            'type': 'internal',
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    def _find_best_split(self, X, y):
        """Find the best split for current node"""
        best_split = None
        best_score = float('inf')
        
        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for threshold in unique_values[:-1]:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                # Calculate RSS
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                rss_left = np.sum((y_left - np.mean(y_left))**2)
                rss_right = np.sum((y_right - np.mean(y_right))**2)
                rss_total = rss_left + rss_right
                
                if rss_total < best_score:
                    best_score = rss_total
                    best_split = (feature, threshold)
        
        return best_split
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.tree))
        return np.array(predictions)
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node['type'] == 'leaf':
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def prune(self, alpha):
        """Prune tree using cost-complexity pruning"""
        self.tree = self._prune_node(self.tree, alpha)
        return self
    
    def _prune_node(self, node, alpha):
        """Recursively prune node"""
        if node['type'] == 'leaf':
            return node
        
        # Prune children
        node['left'] = self._prune_node(node['left'], alpha)
        node['right'] = self._prune_node(node['right'], alpha)
        
        # Check if we should prune this node
        if node['left']['type'] == 'leaf' and node['right']['type'] == 'leaf':
            # Calculate alpha for this node
            # This is a simplified version - in practice, you'd need the full data
            return {'type': 'leaf', 'prediction': (node['left']['prediction'] + 
                                                 node['right']['prediction']) / 2}
        
        return node

def demonstrate_regression_tree():
    """Demonstrate regression tree on Boston housing data"""
    # Load data
    boston = load_boston()
    X, y = boston.data, boston.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit tree
    tree = RegressionTree(max_depth=5, min_samples_split=10)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Tree Predictions')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test - y_pred, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    return tree

def visualize_tree_structure(tree, feature_names=None):
    """Visualize tree structure"""
    def count_nodes(node):
        if node['type'] == 'leaf':
            return 1
        return 1 + count_nodes(node['left']) + count_nodes(node['right'])
    
    def get_depth(node, current_depth=0):
        if node['type'] == 'leaf':
            return current_depth
        return max(get_depth(node['left'], current_depth + 1),
                  get_depth(node['right'], current_depth + 1))
    
    n_nodes = count_nodes(tree.tree)
    max_depth = get_depth(tree.tree)
    
    print(f"Tree Statistics:")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  Maximum depth: {max_depth}")
    print(f"  Number of leaves: {n_nodes // 2 + 1}")
    
    # Feature importance (simplified)
    feature_usage = {}
    
    def count_feature_usage(node):
        if node['type'] == 'internal':
            feature = node['feature']
            feature_usage[feature] = feature_usage.get(feature, 0) + 1
            count_feature_usage(node['left'])
            count_feature_usage(node['right'])
    
    count_feature_usage(tree.tree)
    
    if feature_names:
        print("\nFeature Importance:")
        for feature, count in sorted(feature_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature_names[feature]}: {count} splits")
    
    return feature_usage

def analyze_tree_performance(X, y, tree):
    """Analyze tree performance"""
    predictions = tree.predict(X)
    residuals = y - predictions
    
    # Performance metrics
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    print(f"Performance Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Residual analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def compare_with_linear_model(X_train, X_test, y_train, y_test):
    """Compare tree with linear regression"""
    from sklearn.linear_model import LinearRegression
    
    # Fit linear model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    
    # Fit tree model
    tree_model = RegressionTree(max_depth=5, min_samples_split=10)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    
    # Compare performance
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    r2_tree = r2_score(y_test, y_pred_tree)
    
    print("Model Comparison:")
    print(f"{'Model':<15} {'MSE':<10} {'R²':<8}")
    print("-" * 35)
    print(f"{'Linear':<15} {mse_linear:<10.4f} {r2_linear:<8.4f}")
    print(f"{'Tree':<15} {mse_tree:<10.4f} {r2_tree:<8.4f}")
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_linear, alpha=0.6, label='Linear')
    plt.scatter(y_test, y_pred_tree, alpha=0.6, label='Tree')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions Comparison')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test - y_pred_linear, alpha=0.6, label='Linear')
    plt.scatter(y_test, y_test - y_pred_tree, alpha=0.6, label='Tree')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return linear_model, tree_model

if __name__ == "__main__":
    # Demonstrate regression tree
    tree = demonstrate_regression_tree()
    
    # Load Boston data for analysis
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Analyze tree structure
    print("\n=== TREE STRUCTURE ANALYSIS ===")
    feature_usage = visualize_tree_structure(tree, boston.feature_names)
    
    # Analyze performance
    print("\n=== PERFORMANCE ANALYSIS ===")
    performance = analyze_tree_performance(X_test, y_test, tree)
    
    # Compare with linear model
    print("\n=== MODEL COMPARISON ===")
    linear_model, tree_model = compare_with_linear_model(X_train, X_test, y_train, y_test)
