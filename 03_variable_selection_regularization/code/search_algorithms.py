import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from itertools import combinations
import time

def demonstrate_search_algorithms():
    """Demonstrate different search algorithms for variable selection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data
    n = 100
    p = 8
    X = np.random.randn(n, p)

    # True model: only first 4 variables matter
    beta_true = np.zeros(p)
    beta_true[:4] = [1.5, -0.8, 0.6, -0.4]
    f_true = X @ beta_true
    y = f_true + np.random.normal(0, 0.5, n)

    print("=== TRUE MODEL ===")
    print(f"True coefficients: {beta_true}")
    print(f"True model variables: {list(range(4))}")

    # Function to calculate model score (AIC)
    def calculate_score(X_subset, y):
        """Calculate AIC for a given model"""
        if X_subset.shape[1] == 0:
            return np.inf
        
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        
        rss = np.sum((y - y_pred)**2)
        p = X_subset.shape[1]
        aic = n * np.log(rss/n) + 2*p
        
        return aic

    # 1. Exhaustive Search (Best Subset)
    print("\n=== EXHAUSTIVE SEARCH ===")
    start_time = time.time()

    best_score_exhaustive = np.inf
    best_model_exhaustive = None
    all_scores = []

    for p_subset in range(1, p + 1):
        for subset in combinations(range(p), p_subset):
            X_subset = X[:, subset]
            score = calculate_score(X_subset, y)
            all_scores.append((score, subset))
            
            if score < best_score_exhaustive:
                best_score_exhaustive = score
                best_model_exhaustive = subset

    exhaustive_time = time.time() - start_time
    print(f"Best model: {best_model_exhaustive}")
    print(f"Best score: {best_score_exhaustive:.2f}")
    print(f"Computation time: {exhaustive_time:.3f} seconds")

    # 2. Forward Selection
    print("\n=== FORWARD SELECTION ===")
    start_time = time.time()

    current_vars = set()
    best_score_forward = np.inf
    forward_history = []

    for step in range(p):
        best_score_step = np.inf
        best_var_step = None
        
        for var in range(p):
            if var not in current_vars:
                test_vars = list(current_vars) + [var]
                X_test = X[:, test_vars]
                score = calculate_score(X_test, y)
                
                if score < best_score_step:
                    best_score_step = score
                    best_var_step = var
        
        if best_score_step < best_score_forward:
            current_vars.add(best_var_step)
            best_score_forward = best_score_step
            forward_history.append((best_score_step, list(current_vars)))
        else:
            break

    forward_time = time.time() - start_time
    print(f"Best model: {list(current_vars)}")
    print(f"Best score: {best_score_forward:.2f}")
    print(f"Computation time: {forward_time:.3f} seconds")

    # 3. Backward Elimination
    print("\n=== BACKWARD ELIMINATION ===")
    start_time = time.time()

    current_vars = set(range(p))
    best_score_backward = calculate_score(X, y)
    backward_history = [(best_score_backward, list(current_vars))]

    for step in range(p - 1):
        best_score_step = np.inf
        best_var_step = None
        
        for var in current_vars:
            test_vars = list(current_vars - {var})
            X_test = X[:, test_vars]
            score = calculate_score(X_test, y)
            
            if score < best_score_step:
                best_score_step = score
                best_var_step = var
        
        if best_score_step < best_score_backward:
            current_vars.remove(best_var_step)
            best_score_backward = best_score_step
            backward_history.append((best_score_step, list(current_vars)))
        else:
            break

    backward_time = time.time() - start_time
    print(f"Best model: {list(current_vars)}")
    print(f"Best score: {best_score_backward:.2f}")
    print(f"Computation time: {backward_time:.3f} seconds")

    # 4. Stepwise Selection
    print("\n=== STEPWISE SELECTION ===")
    start_time = time.time()

    current_vars = set(range(p))
    best_score_stepwise = calculate_score(X, y)
    stepwise_history = [(best_score_stepwise, list(current_vars))]
    improved = True

    while improved:
        improved = False
        
        # Backward step
        for var in list(current_vars):
            test_vars = list(current_vars - {var})
            X_test = X[:, test_vars]
            score = calculate_score(X_test, y)
            
            if score < best_score_stepwise:
                current_vars.remove(var)
                best_score_stepwise = score
                stepwise_history.append((best_score_stepwise, list(current_vars)))
                improved = True
                break
        
        # Forward step
        if not improved:
            for var in range(p):
                if var not in current_vars:
                    test_vars = list(current_vars) + [var]
                    X_test = X[:, test_vars]
                    score = calculate_score(X_test, y)
                    
                    if score < best_score_stepwise:
                        current_vars.add(var)
                        best_score_stepwise = score
                        stepwise_history.append((best_score_stepwise, list(current_vars)))
                        improved = True
                        break

    stepwise_time = time.time() - start_time
    print(f"Best model: {list(current_vars)}")
    print(f"Best score: {best_score_stepwise:.2f}")
    print(f"Computation time: {stepwise_time:.3f} seconds")

    # Comparison
    print("\n=== ALGORITHM COMPARISON ===")
    comparison_df = pd.DataFrame({
        'Algorithm': ['Exhaustive', 'Forward', 'Backward', 'Stepwise'],
        'Best Score': [best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise],
        'Best Model': [str(best_model_exhaustive), str(list(current_vars)), str(list(current_vars)), str(list(current_vars))],
        'Time (s)': [exhaustive_time, forward_time, backward_time, stepwise_time],
        'Optimal': [best_score_exhaustive == min(best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise)] * 4
    })

    print(comparison_df.to_string(index=False))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Score progression for each algorithm
    steps_forward = range(1, len(forward_history) + 1)
    scores_forward = [h[0] for h in forward_history]

    steps_backward = range(1, len(backward_history) + 1)
    scores_backward = [h[0] for h in backward_history]

    steps_stepwise = range(1, len(stepwise_history) + 1)
    scores_stepwise = [h[0] for h in stepwise_history]

    axes[0, 0].plot(steps_forward, scores_forward, 'bo-', label='Forward', linewidth=2)
    axes[0, 0].plot(steps_backward, scores_backward, 'ro-', label='Backward', linewidth=2)
    axes[0, 0].plot(steps_stepwise, scores_stepwise, 'go-', label='Stepwise', linewidth=2)
    axes[0, 0].axhline(y=best_score_exhaustive, color='black', linestyle='--', alpha=0.7, label='Exhaustive')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('AIC Score')
    axes[0, 0].set_title('Score Progression')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Model size progression
    model_sizes_forward = [len(h[1]) for h in forward_history]
    model_sizes_backward = [len(h[1]) for h in backward_history]
    model_sizes_stepwise = [len(h[1]) for h in stepwise_history]

    axes[0, 1].plot(steps_forward, model_sizes_forward, 'bo-', label='Forward', linewidth=2)
    axes[0, 1].plot(steps_backward, model_sizes_backward, 'ro-', label='Backward', linewidth=2)
    axes[0, 1].plot(steps_stepwise, model_sizes_stepwise, 'go-', label='Stepwise', linewidth=2)
    axes[0, 1].axhline(y=len(best_model_exhaustive), color='black', linestyle='--', alpha=0.7, label='Exhaustive')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Model Size')
    axes[0, 1].set_title('Model Size Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Computation time comparison
    algorithms = ['Exhaustive', 'Forward', 'Backward', 'Stepwise']
    times = [exhaustive_time, forward_time, backward_time, stepwise_time]

    axes[1, 0].bar(algorithms, times, color=['red', 'blue', 'green', 'orange'])
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Computation Time Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # Score comparison
    scores = [best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise]
    colors = ['red' if s == min(scores) else 'gray' for s in scores]

    axes[1, 1].bar(algorithms, scores, color=colors)
    axes[1, 1].set_ylabel('AIC Score')
    axes[1, 1].set_title('Final Score Comparison')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Key insights
    print("\n=== KEY INSIGHTS ===")
    print("1. Exhaustive search finds the global optimum but is computationally expensive")
    print("2. Greedy algorithms are much faster but may find local optima")
    print("3. Stepwise selection often provides a good compromise")
    print("4. All algorithms found similar model sizes in this example")
    
    return comparison_df, best_model_exhaustive, forward_history, backward_history, stepwise_history

# Run demonstration
comparison_df, best_model_exhaustive, forward_history, backward_history, stepwise_history = demonstrate_search_algorithms()
