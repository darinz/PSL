"""
Bootstrap Sampling and Bagging for Random Forest
===============================================

This module demonstrates bootstrap sampling and bagging techniques
used in Random Forest ensemble methods.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

def calculate_oob_score(trees, X, y):
    """
    Calculate Out-of-Bag (OOB) score for bagging ensemble
    
    Parameters:
    trees: list of trained trees
    X: feature matrix
    y: target vector
    
    Returns:
    oob_score: R² score using OOB predictions
    """
    n_samples = len(y)
    oob_predictions = np.zeros(n_samples)
    oob_counts = np.zeros(n_samples)
    
    for tree in trees:
        # Find OOB samples for this tree
        # This is a simplified version - in practice, you'd track OOB samples during training
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_mask = ~np.isin(np.arange(n_samples), indices)
        
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

def demonstrate_bootstrap_sampling():
    """Demonstrate bootstrap sampling properties"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    print("=== BOOTSTRAP SAMPLING DEMONSTRATION ===")
    print(f"Original dataset: {n_samples} samples")
    
    # Create multiple bootstrap samples
    n_bootstrap = 10
    unique_samples_list = []
    
    for i in range(n_bootstrap):
        X_boot, y_boot = bootstrap_sample(X, y)
        unique_samples = len(np.unique(np.arange(n_samples)[np.random.choice(n_samples, size=n_samples, replace=True)]))
        unique_samples_list.append(unique_samples)
        
        print(f"Bootstrap {i+1}: {unique_samples} unique samples ({unique_samples/n_samples*100:.1f}%)")
    
    # Theoretical expectation
    theoretical_unique = n_samples * (1 - np.exp(-1))
    print(f"\nTheoretical expectation: {theoretical_unique:.1f} unique samples ({theoretical_unique/n_samples*100:.1f}%)")
    print(f"Average observed: {np.mean(unique_samples_list):.1f} unique samples ({np.mean(unique_samples_list)/n_samples*100:.1f}%)")
    
    return X, y

def demonstrate_bagging():
    """Demonstrate bagging for regression"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.5, n_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=== BAGGING DEMONSTRATION ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train single tree
    single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    single_pred = single_tree.predict(X_test)
    single_mse = mean_squared_error(y_test, single_pred)
    single_r2 = r2_score(y_test, single_pred)
    
    print(f"\nSingle Tree:")
    print(f"  MSE: {single_mse:.4f}")
    print(f"  R²: {single_r2:.4f}")
    
    # Train bagging ensemble
    trees = bagging_regression(X_train, y_train, n_trees=50, max_depth=10)
    bagging_pred = predict_bagging(trees, X_test)
    bagging_mse = mean_squared_error(y_test, bagging_pred)
    bagging_r2 = r2_score(y_test, bagging_pred)
    
    print(f"\nBagging Ensemble (50 trees):")
    print(f"  MSE: {bagging_mse:.4f}")
    print(f"  R²: {bagging_r2:.4f}")
    print(f"  MSE Improvement: {(single_mse - bagging_mse) / single_mse * 100:.1f}%")
    
    # Calculate OOB score
    oob_score = calculate_oob_score(trees, X_train, y_train)
    print(f"  OOB Score: {oob_score:.4f}")
    
    return trees, X_train, y_train, X_test, y_test

def analyze_ensemble_size_effect(X_train, y_train, X_test, y_test):
    """Analyze the effect of ensemble size on performance"""
    
    ensemble_sizes = [1, 5, 10, 25, 50, 100]
    mse_scores = []
    r2_scores = []
    
    print("\n=== ENSEMBLE SIZE ANALYSIS ===")
    
    for n_trees in ensemble_sizes:
        if n_trees == 1:
            # Single tree
            tree = DecisionTreeRegressor(max_depth=10, random_state=42)
            tree.fit(X_train, y_train)
            pred = tree.predict(X_test)
        else:
            # Bagging ensemble
            trees = bagging_regression(X_train, y_train, n_trees=n_trees, max_depth=10)
            pred = predict_bagging(trees, X_test)
        
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        
        print(f"{n_trees:3d} trees: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(ensemble_sizes, mse_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Trees')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE vs Ensemble Size')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(ensemble_sizes, r2_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Trees')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² vs Ensemble Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return ensemble_sizes, mse_scores, r2_scores

if __name__ == "__main__":
    # Demonstrate bootstrap sampling
    X, y = demonstrate_bootstrap_sampling()
    
    # Demonstrate bagging
    trees, X_train, y_train, X_test, y_test = demonstrate_bagging()
    
    # Analyze ensemble size effect
    ensemble_sizes, mse_scores, r2_scores = analyze_ensemble_size_effect(X_train, y_train, X_test, y_test)
