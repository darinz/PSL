"""
Regression Tree Building Algorithm
=================================

This module demonstrates the greedy algorithm for building regression trees,
including handling categorical variables and missing values.
"""

import numpy as np

def build_regression_tree(X, y, max_depth=None, min_samples_split=2):
    """
    Build a regression tree using greedy algorithm
    
    Parameters:
    X: feature matrix (n_samples, n_features)
    y: target vector (n_samples,)
    max_depth: maximum tree depth
    min_samples_split: minimum samples required to split
    
    Returns:
    tree: dictionary representing the tree structure
    """
    def find_best_split(X, y):
        best_split = None
        best_score = float('inf')
        
        for j in range(X.shape[1]):  # For each feature
            unique_values = np.unique(X[:, j])
            for s in unique_values[:-1]:  # For each potential split
                left_mask = X[:, j] <= s
                right_mask = ~left_mask
                
                if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
                    continue
                
                # Calculate RSS reduction
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                rss_left = np.sum((y_left - np.mean(y_left))**2)
                rss_right = np.sum((y_right - np.mean(y_right))**2)
                rss_total = rss_left + rss_right
                
                if rss_total < best_score:
                    best_score = rss_total
                    best_split = (j, s)
        
        return best_split
    
    def build_node(X, y, depth):
        # Stopping criteria
        if (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
            return {'type': 'leaf', 'prediction': np.mean(y)}
        
        # Find best split
        split = find_best_split(X, y)
        if split is None:
            return {'type': 'leaf', 'prediction': np.mean(y)}
        
        j, s = split
        left_mask = X[:, j] <= s
        right_mask = ~left_mask
        
        # Create internal node
        node = {
            'type': 'internal',
            'feature': j,
            'threshold': s,
            'left': build_node(X[left_mask], y[left_mask], depth + 1),
            'right': build_node(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    return build_node(X, y, 0)

def find_categorical_split(X_cat, y):
    """
    Find optimal split for categorical variable
    
    Parameters:
    X_cat: categorical feature vector
    y: target vector
    
    Returns:
    best_split: set of levels for left child
    """
    # Calculate mean response for each level
    levels = np.unique(X_cat)
    level_means = {}
    for level in levels:
        mask = X_cat == level
        level_means[level] = np.mean(y[mask])
    
    # Sort levels by mean response
    sorted_levels = sorted(levels, key=lambda x: level_means[x])
    
    # Try splits between adjacent levels
    best_split = None
    best_score = float('inf')
    
    for i in range(len(sorted_levels) - 1):
        left_levels = set(sorted_levels[:i+1])
        left_mask = np.isin(X_cat, list(left_levels))
        right_mask = ~left_mask
        
        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            continue
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        rss_left = np.sum((y_left - np.mean(y_left))**2)
        rss_right = np.sum((y_right - np.mean(y_right))**2)
        rss_total = rss_left + rss_right
        
        if rss_total < best_score:
            best_score = rss_total
            best_split = left_levels
    
    return best_split

def find_surrogate_splits(X, y, primary_split, missing_mask):
    """
    Find surrogate splits for missing values
    
    Parameters:
    X: feature matrix
    y: target vector
    primary_split: (feature_index, threshold) of primary split
    missing_mask: boolean mask indicating missing values
    
    Returns:
    surrogate_splits: list of (feature, threshold, correlation) tuples
    """
    j, s = primary_split
    surrogate_splits = []
    
    for k in range(X.shape[1]):
        if k == j:
            continue
        
        # Calculate correlation with primary split
        primary_values = (X[:, j] <= s).astype(int)
        k_values = X[:, k]
        
        # Find best split on feature k that mimics primary split
        unique_values = np.unique(k_values)
        best_correlation = 0
        best_threshold = None
        
        for threshold in unique_values[:-1]:
            k_split = (k_values <= threshold).astype(int)
            correlation = np.corrcoef(primary_values, k_split)[0, 1]
            
            if abs(correlation) > abs(best_correlation):
                best_correlation = correlation
                best_threshold = threshold
        
        if abs(best_correlation) > 0.5:  # Minimum correlation threshold
            surrogate_splits.append((k, best_threshold, best_correlation))
    
    # Sort by correlation strength
    surrogate_splits.sort(key=lambda x: abs(x[2]), reverse=True)
    return surrogate_splits

def predict_tree(tree, X):
    """
    Make predictions using a regression tree
    
    Parameters:
    tree: tree structure dictionary
    X: feature matrix
    
    Returns:
    predictions: array of predictions
    """
    predictions = []
    for x in X:
        predictions.append(_predict_single(x, tree))
    return np.array(predictions)

def _predict_single(x, node):
    """
    Predict for a single sample
    
    Parameters:
    x: feature vector
    node: current node in tree
    
    Returns:
    prediction: predicted value
    """
    if node['type'] == 'leaf':
        return node['prediction']
    
    if x[node['feature']] <= node['threshold']:
        return _predict_single(x, node['left'])
    else:
        return _predict_single(x, node['right'])

def print_tree_structure(tree, feature_names=None, depth=0):
    """
    Print tree structure for visualization
    
    Parameters:
    tree: tree structure dictionary
    feature_names: list of feature names
    depth: current depth in tree
    """
    indent = "  " * depth
    
    if tree['type'] == 'leaf':
        print(f"{indent}Leaf: prediction = {tree['prediction']:.4f}")
    else:
        feature_name = feature_names[tree['feature']] if feature_names else f"X{tree['feature']}"
        print(f"{indent}Split: {feature_name} <= {tree['threshold']:.4f}")
        print(f"{indent}├── Left:")
        print_tree_structure(tree['left'], feature_names, depth + 1)
        print(f"{indent}└── Right:")
        print_tree_structure(tree['right'], feature_names, depth + 1)

def count_tree_nodes(tree):
    """
    Count total number of nodes in tree
    
    Parameters:
    tree: tree structure dictionary
    
    Returns:
    count: total number of nodes
    """
    if tree['type'] == 'leaf':
        return 1
    return 1 + count_tree_nodes(tree['left']) + count_tree_nodes(tree['right'])

def get_tree_depth(tree):
    """
    Get maximum depth of tree
    
    Parameters:
    tree: tree structure dictionary
    
    Returns:
    depth: maximum depth
    """
    if tree['type'] == 'leaf':
        return 0
    return 1 + max(get_tree_depth(tree['left']), get_tree_depth(tree['right']))

def demonstrate_tree_building():
    """Demonstrate tree building with synthetic data"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    print("=== REGRESSION TREE BUILDING DEMONSTRATION ===")
    print(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Build tree
    tree = build_regression_tree(X, y, max_depth=3, min_samples_split=5)
    
    # Tree statistics
    n_nodes = count_tree_nodes(tree)
    max_depth = get_tree_depth(tree)
    
    print(f"\nTree Statistics:")
    print(f"  Total nodes: {n_nodes}")
    print(f"  Maximum depth: {max_depth}")
    print(f"  Number of leaves: {n_nodes // 2 + 1}")
    
    # Print tree structure
    print(f"\nTree Structure:")
    feature_names = ['Feature_1', 'Feature_2', 'Feature_3']
    print_tree_structure(tree, feature_names)
    
    # Make predictions
    predictions = predict_tree(tree, X)
    mse = np.mean((y - predictions)**2)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    
    print(f"\nModel Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return tree, X, y

if __name__ == "__main__":
    # Demonstrate tree building
    tree, X, y = demonstrate_tree_building()
    
    # Demonstrate categorical split
    print("\n=== CATEGORICAL SPLIT DEMONSTRATION ===")
    
    # Create categorical data
    categories = ['A', 'B', 'C', 'D']
    X_cat = np.random.choice(categories, size=50)
    y_cat = np.random.normal(0, 1, 50)
    
    # Add some relationship
    for i, cat in enumerate(X_cat):
        if cat == 'A':
            y_cat[i] += 2
        elif cat == 'B':
            y_cat[i] += 1
        elif cat == 'C':
            y_cat[i] -= 1
        else:  # D
            y_cat[i] -= 2
    
    best_split = find_categorical_split(X_cat, y_cat)
    print(f"Best categorical split: {best_split}")
    
    # Demonstrate surrogate splits
    print("\n=== SURROGATE SPLITS DEMONSTRATION ===")
    
    # Create data with missing values
    X_surrogate = np.random.randn(100, 3)
    y_surrogate = np.random.normal(0, 1, 100)
    
    # Add some correlation between features
    X_surrogate[:, 1] = 0.7 * X_surrogate[:, 0] + 0.3 * np.random.normal(0, 1, 100)
    
    primary_split = (0, 0.5)  # Split on feature 0 at threshold 0.5
    missing_mask = np.random.choice([True, False], size=100, p=[0.1, 0.9])
    
    surrogate_splits = find_surrogate_splits(X_surrogate, y_surrogate, primary_split, missing_mask)
    print(f"Surrogate splits found: {len(surrogate_splits)}")
    for i, (feature, threshold, correlation) in enumerate(surrogate_splits[:3]):
        print(f"  {i+1}. Feature {feature} <= {threshold:.3f} (correlation: {correlation:.3f})")
