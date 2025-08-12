"""
Rank Deficiency Analysis
========================

This module demonstrates rank deficiency in linear regression,
showing how to detect and handle cases where the design matrix
does not have full column rank.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_rank_deficiency():
    """Demonstrate and analyze rank deficiency"""
    
    print("=== Rank Deficiency Analysis ===")
    
    # 1. Perfect collinearity example
    print("\n1. Perfect Collinearity Example")
    temp_c = np.array([0, 10, 20, 30, 40])
    temp_f = 9/5 * temp_c + 32
    X_collinear = np.column_stack([np.ones(5), temp_c, temp_f])
    y_collinear = np.array([1, 2, 3, 4, 5])
    
    print("Design matrix:")
    print(X_collinear)
    
    # Check rank
    rank = np.linalg.matrix_rank(X_collinear)
    print(f"Rank of X: {rank}")
    print(f"Number of columns: {X_collinear.shape[1]}")
    print(f"Rank deficient: {rank < X_collinear.shape[1]}")
    
    # Check correlation
    correlation = np.corrcoef(temp_c, temp_f)[0, 1]
    print(f"Correlation between temp_c and temp_f: {correlation:.6f}")
    
    # Try different solution methods
    try:
        # Direct inverse (will fail)
        beta_direct = np.linalg.inv(X_collinear.T @ X_collinear) @ X_collinear.T @ y_collinear
        print("Direct inverse solution:", beta_direct)
    except np.linalg.LinAlgError:
        print("Direct inverse failed due to rank deficiency")
    
    # Pseudoinverse (works)
    beta_pinv = np.linalg.pinv(X_collinear) @ y_collinear
    print("Pseudoinverse solution:", beta_pinv)
    
    # Sklearn solution
    model = LinearRegression()
    model.fit(X_collinear, y_collinear)
    print("Sklearn solution:", np.concatenate([[model.intercept_], model.coef_]))
    
    # 2. Redundant variables example
    print("\n2. Redundant Variables Example")
    age_young = np.array([30, 25, 40, 35, 45])
    age_middle = np.array([45, 50, 35, 40, 30])
    age_old = 100 - age_young - age_middle
    X_redundant = np.column_stack([np.ones(5), age_young, age_middle, age_old])
    y_redundant = np.array([1, 2, 3, 4, 5])
    
    print("Design matrix:")
    print(X_redundant)
    
    # Check rank
    rank = np.linalg.matrix_rank(X_redundant)
    print(f"Rank of X: {rank}")
    print(f"Number of columns: {X_redundant.shape[1]}")
    print(f"Rank deficient: {rank < X_redundant.shape[1]}")
    
    # Check the linear relationship
    check_sum = age_young + age_middle + age_old
    print(f"Sum of age variables: {check_sum}")
    
    return {
        'collinear_X': X_collinear,
        'collinear_y': y_collinear,
        'redundant_X': X_redundant,
        'redundant_y': y_redundant
    }

def detect_and_handle_rank_deficiency(X, y):
    """Detect and handle rank deficiency"""
    
    print("\n=== Rank Deficiency Detection and Handling ===")
    
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    # 1. Check rank
    rank = np.linalg.matrix_rank(X)
    print(f"Matrix rank: {rank}")
    print(f"Number of columns: {p_plus_1}")
    print(f"Rank deficient: {rank < p_plus_1}")
    
    if rank < p_plus_1:
        print("⚠️  Rank deficiency detected!")
        
        # 2. Check condition number
        eigenvals = np.linalg.eigvals(X.T @ X)
        condition_number = np.max(eigenvals) / np.min(eigenvals[eigenvals > 1e-10])
        print(f"Condition number: {condition_number:.2e}")
        
        if condition_number > 1e12:
            print("⚠️  High condition number - numerical instability likely")
        
        # 3. Find linear dependencies
        print("\nLinear dependency analysis:")
        for i in range(1, p_plus_1):
            # Try to predict column i from other columns
            X_others = np.delete(X, i, axis=1)
            X_i = X[:, i].reshape(-1, 1)
            
            try:
                beta_i = np.linalg.lstsq(X_others, X_i.flatten(), rcond=None)[0]
                y_pred_i = X_others @ beta_i
                rss_i = np.sum((X_i.flatten() - y_pred_i)**2)
                r_squared_i = 1 - rss_i / np.sum((X_i.flatten() - np.mean(X_i))**2)
                
                if r_squared_i > 0.99:
                    print(f"  Column {i} is highly predictable from others (R² = {r_squared_i:.4f})")
            except:
                pass
    
    # 4. Solutions
    print("\nSolution approaches:")
    
    # Pseudoinverse solution
    beta_pinv = np.linalg.pinv(X) @ y
    print(f"Pseudoinverse solution: {beta_pinv}")
    
    # Ridge regression (regularization)
    lambda_ridge = 0.01
    beta_ridge = np.linalg.inv(X.T @ X + lambda_ridge * np.eye(p_plus_1)) @ X.T @ y
    print(f"Ridge regression solution (λ={lambda_ridge}): {beta_ridge}")
    
    # Compare fitted values
    y_hat_pinv = X @ beta_pinv
    y_hat_ridge = X @ beta_ridge
    y_hat_diff = np.linalg.norm(y_hat_pinv - y_hat_ridge)
    print(f"Difference in fitted values: {y_hat_diff:.10f}")
    
    return {
        'rank': rank,
        'condition_number': condition_number if rank < p_plus_1 else None,
        'beta_pinv': beta_pinv,
        'beta_ridge': beta_ridge
    }

if __name__ == "__main__":
    # Analyze rank deficiency
    rank_deficiency_results = analyze_rank_deficiency()
    
    # Detect and handle rank deficiency
    rank_analysis = detect_and_handle_rank_deficiency(
        rank_deficiency_results['collinear_X'], 
        rank_deficiency_results['collinear_y']
    )
