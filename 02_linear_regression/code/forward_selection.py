import numpy as np
import matplotlib.pyplot as plt

def least_squares_estimation(X, y):
    """Compute least squares estimates using the normal equation"""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

def forward_selection(X, y, max_predictors=None):
    """
    Forward stepwise selection
    
    Parameters:
    X: design matrix (without intercept)
    y: response vector
    max_predictors: maximum number of predictors to include
    
    Returns:
    selected_predictors: indices of selected predictors
    """
    n, p = X.shape
    if max_predictors is None:
        max_predictors = p
    
    selected = []
    remaining = list(range(p))
    
    for step in range(max_predictors):
        best_score = -np.inf
        best_predictor = None
        
        for j in remaining:
            # Add predictor j to current model
            current_predictors = selected + [j]
            X_current = np.column_stack([np.ones(n), X[:, current_predictors]])
            
            # Fit model and compute score
            beta_hat = least_squares_estimation(X_current, y)
            y_hat = X_current @ beta_hat
            rss = np.sum((y - y_hat)**2)
            
            # Use adjusted R-squared as selection criterion
            tss = np.sum((y - np.mean(y))**2)
            r_squared = 1 - rss / tss
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - len(current_predictors) - 1)
            
            if adj_r_squared > best_score:
                best_score = adj_r_squared
                best_predictor = j
        
        if best_predictor is not None:
            selected.append(best_predictor)
            remaining.remove(best_predictor)
            print(f"Step {step+1}: Added predictor {best_predictor}, Adj R² = {best_score:.4f}")
        else:
            break
    
    return selected

# Example usage
def create_example_data(n=100, p=5, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.5, 0.5, 0.0, 0.0])  # Last two predictors are noise
    y = X @ beta_true + noise_std * np.random.randn(n)
    return X, y, beta_true

# Perform forward selection
X, y, beta_true = create_example_data()
X_no_intercept = X[:, 1:]  # Remove intercept column
selected_predictors = forward_selection(X_no_intercept, y)
print(f"\nSelected predictors: {selected_predictors}")
