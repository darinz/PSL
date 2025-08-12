"""
Tree Pruning: Weakest Link Algorithm
===================================

This module demonstrates the weakest link pruning algorithm for regression trees,
including cost-complexity pruning and cross-validation for alpha selection.
"""

import numpy as np
from sklearn.model_selection import KFold

def weakest_link_pruning(tree, X, y):
    """
    Perform weakest link pruning
    
    Parameters:
    tree: tree structure dictionary
    X: feature matrix
    y: target vector
    
    Returns:
    trees: list of pruned trees
    alphas: list of corresponding alpha values
    """
    def calculate_alpha(node, X_node, y_node):
        """Calculate alpha for a node"""
        if node['type'] == 'leaf':
            return float('inf')
        
        # RSS if this node were a leaf
        rss_leaf = np.sum((y_node - np.mean(y_node))**2)
        
        # RSS of subtree
        left_mask = X_node[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        y_left = y_node[left_mask]
        y_right = y_node[right_mask]
        
        rss_left = np.sum((y_left - np.mean(y_left))**2)
        rss_right = np.sum((y_right - np.mean(y_right))**2)
        rss_subtree = rss_left + rss_right
        
        # Count leaves in subtree
        n_leaves = count_leaves(node)
        
        alpha = (rss_leaf - rss_subtree) / (n_leaves - 1)
        return alpha
    
    def count_leaves(node):
        """Count number of leaves in subtree"""
        if node['type'] == 'leaf':
            return 1
        return count_leaves(node['left']) + count_leaves(node['right'])
    
    def find_weakest_link(node, X_node, y_node):
        """Find node with smallest alpha"""
        if node['type'] == 'leaf':
            return None, float('inf')
        
        alpha = calculate_alpha(node, X_node, y_node)
        weakest_node = node
        weakest_alpha = alpha
        
        # Check children
        left_mask = X_node[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        left_weakest, left_alpha = find_weakest_link(
            node['left'], X_node[left_mask], y_node[left_mask]
        )
        right_weakest, right_alpha = find_weakest_link(
            node['right'], X_node[right_mask], y_node[right_mask]
        )
        
        if left_alpha < weakest_alpha:
            weakest_node = left_weakest
            weakest_alpha = left_alpha
        if right_alpha < weakest_alpha:
            weakest_node = right_weakest
            weakest_alpha = right_alpha
        
        return weakest_node, weakest_alpha
    
    def prune_node(node, target_node, X_node, y_node):
        """Prune the target node from the tree"""
        if node['type'] == 'leaf':
            return node
        
        if node is target_node:
            return {'type': 'leaf', 'prediction': np.mean(y_node)}
        
        left_mask = X_node[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        node['left'] = prune_node(
            node['left'], target_node, X_node[left_mask], y_node[left_mask]
        )
        node['right'] = prune_node(
            node['right'], target_node, X_node[right_mask], y_node[right_mask]
        )
        
        return node
    
    # Generate sequence of pruned trees
    trees = [tree]
    alphas = [0.0]
    
    current_tree = tree.copy()
    
    while True:
        weakest_node, alpha = find_weakest_link(current_tree, X, y)
        
        if weakest_node is None:
            break
        
        current_tree = prune_node(current_tree, weakest_node, X, y)
        trees.append(current_tree.copy())
        alphas.append(alpha)
    
    return trees, alphas

def cross_validate_alpha(X, y, cv_folds=5, max_depth=10, min_samples_split=2):
    """
    Cross-validation for alpha selection
    
    Parameters:
    X: feature matrix
    y: target vector
    cv_folds: number of cross-validation folds
    max_depth: maximum tree depth
    min_samples_split: minimum samples to split
    
    Returns:
    optimal_alpha: optimal alpha value
    alphas: all alpha values tested
    cv_errors: cross-validation errors
    """
    from .tree_building import build_regression_tree, predict_tree
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Generate all possible alpha values
    all_alphas = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build and prune tree
        tree = build_regression_tree(X_train, y_train, max_depth, min_samples_split)
        trees, alphas = weakest_link_pruning(tree, X_train, y_train)
        all_alphas.extend(alphas)
    
    # Unique alpha values, sorted
    unique_alphas = sorted(set(all_alphas))
    
    # Cross-validation errors
    cv_errors = []
    
    for alpha in unique_alphas:
        fold_errors = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and prune tree
            tree = build_regression_tree(X_train, y_train, max_depth, min_samples_split)
            trees, alphas = weakest_link_pruning(tree, X_train, y_train)
            
            # Find optimal tree for this alpha
            optimal_tree = None
            for i, a in enumerate(alphas):
                if a <= alpha:
                    optimal_tree = trees[i]
            
            if optimal_tree is None:
                optimal_tree = trees[0]  # Use full tree
            
            # Predict and calculate error
            predictions = predict_tree(optimal_tree, X_val)
            mse = np.mean((y_val - predictions)**2)
            fold_errors.append(mse)
        
        cv_errors.append(np.mean(fold_errors))
    
    # Find optimal alpha
    optimal_idx = np.argmin(cv_errors)
    optimal_alpha = unique_alphas[optimal_idx]
    
    return optimal_alpha, unique_alphas, cv_errors

def one_se_rule(alphas, cv_errors):
    """
    One standard error rule for alpha selection
    
    Parameters:
    alphas: list of alpha values
    cv_errors: list of cross-validation errors
    
    Returns:
    optimal_alpha: alpha selected by one SE rule
    """
    cv_errors = np.array(cv_errors)
    min_error = np.min(cv_errors)
    min_idx = np.argmin(cv_errors)
    
    # Calculate standard error
    se = np.std(cv_errors) / np.sqrt(len(cv_errors))
    threshold = min_error + se
    
    # Find largest alpha within one SE of minimum
    for i in range(min_idx, -1, -1):
        if cv_errors[i] <= threshold:
            return alphas[i]
    
    return alphas[0]

def calculate_cost_complexity(tree, X, y, alpha):
    """
    Calculate cost-complexity measure
    
    Parameters:
    tree: tree structure dictionary
    X: feature matrix
    y: target vector
    alpha: complexity parameter
    
    Returns:
    cost_complexity: R_alpha(T) value
    """
    def calculate_rss(node, X_node, y_node):
        """Calculate RSS for a subtree"""
        if node['type'] == 'leaf':
            return np.sum((y_node - np.mean(y_node))**2)
        
        left_mask = X_node[:, node['feature']] <= node['threshold']
        right_mask = ~left_mask
        
        rss_left = calculate_rss(node['left'], X_node[left_mask], y_node[left_mask])
        rss_right = calculate_rss(node['right'], X_node[right_mask], y_node[right_mask])
        
        return rss_left + rss_right
    
    def count_leaves(node):
        """Count number of leaves in tree"""
        if node['type'] == 'leaf':
            return 1
        return count_leaves(node['left']) + count_leaves(node['right'])
    
    rss = calculate_rss(tree, X, y)
    n_leaves = count_leaves(tree)
    
    return rss + alpha * n_leaves

def demonstrate_pruning():
    """Demonstrate tree pruning with synthetic data"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 4)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.5, n_samples)
    
    print("=== TREE PRUNING DEMONSTRATION ===")
    print(f"Dataset: {n_samples} samples, {X.shape[1]} features")
    
    # Build full tree
    from .tree_building import build_regression_tree, predict_tree, count_tree_nodes
    
    full_tree = build_regression_tree(X, y, max_depth=8, min_samples_split=5)
    n_nodes_full = count_tree_nodes(full_tree)
    
    print(f"\nFull tree nodes: {n_nodes_full}")
    
    # Perform pruning
    trees, alphas = weakest_link_pruning(full_tree, X, y)
    
    print(f"\nPruning sequence:")
    print(f"  Number of trees: {len(trees)}")
    print(f"  Alpha range: [{alphas[0]:.6f}, {alphas[-1]:.6f}]")
    
    # Evaluate each tree
    print(f"\nTree evaluation:")
    print(f"{'Alpha':<12} {'Nodes':<8} {'MSE':<10} {'R²':<8}")
    print("-" * 40)
    
    for i, (tree, alpha) in enumerate(zip(trees, alphas)):
        predictions = predict_tree(tree, X)
        mse = np.mean((y - predictions)**2)
        r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
        n_nodes = count_tree_nodes(tree)
        
        print(f"{alpha:<12.6f} {n_nodes:<8} {mse:<10.4f} {r2:<8.4f}")
    
    # Cross-validation for alpha selection
    print(f"\n=== CROSS-VALIDATION FOR ALPHA SELECTION ===")
    
    optimal_alpha, all_alphas, cv_errors = cross_validate_alpha(
        X, y, cv_folds=5, max_depth=6, min_samples_split=10
    )
    
    print(f"Optimal alpha (min CV error): {optimal_alpha:.6f}")
    
    # One SE rule
    optimal_alpha_se = one_se_rule(all_alphas, cv_errors)
    print(f"Optimal alpha (one SE rule): {optimal_alpha_se:.6f}")
    
    # Find corresponding tree
    optimal_tree = None
    for tree, alpha in zip(trees, alphas):
        if alpha <= optimal_alpha:
            optimal_tree = tree
    
    if optimal_tree is None:
        optimal_tree = trees[0]
    
    # Final evaluation
    predictions = predict_tree(optimal_tree, X)
    mse = np.mean((y - predictions)**2)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    n_nodes = count_tree_nodes(optimal_tree)
    
    print(f"\nFinal pruned tree:")
    print(f"  Alpha: {optimal_alpha:.6f}")
    print(f"  Nodes: {n_nodes}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return trees, alphas, optimal_tree, optimal_alpha

def visualize_pruning_sequence(trees, alphas, X, y):
    """Visualize pruning sequence"""
    
    import matplotlib.pyplot as plt
    from .tree_building import predict_tree, count_tree_nodes
    
    # Calculate metrics for each tree
    nodes = []
    mse_values = []
    r2_values = []
    
    for tree in trees:
        predictions = predict_tree(tree, X)
        mse = np.mean((y - predictions)**2)
        r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
        n_nodes = count_tree_nodes(tree)
        
        nodes.append(n_nodes)
        mse_values.append(mse)
        r2_values.append(r2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Alpha vs nodes
    axes[0, 0].plot(alphas, nodes, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Alpha')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Tree Size vs Alpha')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha vs MSE
    axes[0, 1].plot(alphas, mse_values, 'ro-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Alpha')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_title('MSE vs Alpha')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alpha vs R²
    axes[1, 0].plot(alphas, r2_values, 'go-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Alpha')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² vs Alpha')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Nodes vs R²
    axes[1, 1].plot(nodes, r2_values, 'mo-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Number of Nodes')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('R² vs Tree Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Demonstrate pruning
    trees, alphas, optimal_tree, optimal_alpha = demonstrate_pruning()
    
    # Generate data for visualization
    np.random.seed(42)
    X_viz = np.random.randn(100, 3)
    y_viz = 2 * X_viz[:, 0] + 1.5 * X_viz[:, 1] - 0.8 * X_viz[:, 2] + np.random.normal(0, 0.5, 100)
    
    # Build and prune tree for visualization
    from .tree_building import build_regression_tree
    
    full_tree_viz = build_regression_tree(X_viz, y_viz, max_depth=5, min_samples_split=5)
    trees_viz, alphas_viz = weakest_link_pruning(full_tree_viz, X_viz, y_viz)
    
    # Visualize pruning sequence
    visualize_pruning_sequence(trees_viz, alphas_viz, X_viz, y_viz)
