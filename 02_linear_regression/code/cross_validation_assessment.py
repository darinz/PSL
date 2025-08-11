import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def cross_validation_assessment(X, y, cv_folds=5):
    """
    Perform cross-validation assessment
    
    Parameters:
    X: design matrix (without intercept)
    y: response vector
    cv_folds: number of cross-validation folds
    
    Returns:
    Dictionary with CV results
    """
    # Create linear regression model
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    cv_mse = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    
    return {
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'cv_mse_mean': np.mean(cv_mse),
        'cv_mse_std': np.std(cv_mse)
    }

# Example usage
def create_example_data(n=100, p=2, noise_std=0.5):
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.5])
    y = X @ beta_true + noise_std * np.random.randn(n)
    return X, y, beta_true

# Perform cross-validation (using X without intercept column)
X, y, beta_true = create_example_data()
X_no_intercept = X[:, 1:]  # Remove intercept column for sklearn
cv_results = cross_validation_assessment(X_no_intercept, y)

print("=== Cross-Validation Results ===")
print(f"CV R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
print(f"CV MSE: {cv_results['cv_mse_mean']:.4f} ± {cv_results['cv_mse_std']:.4f}")
