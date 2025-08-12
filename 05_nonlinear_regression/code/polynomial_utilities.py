"""
Polynomial Regression Utilities
==============================

This module provides utility functions for polynomial regression including
orthogonal polynomials, model selection algorithms, and feature creation.
"""

import numpy as np
from scipy.special import legendre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def create_orthogonal_polynomials(X, degree):
    """
    Create orthogonal polynomial features
    
    Parameters:
    X: predictor variable (n_samples,)
    degree: polynomial degree
    
    Returns:
    X_poly: orthogonal polynomial features (n_samples, degree+1)
    """
    n_samples = len(X)
    X_poly = np.zeros((n_samples, degree + 1))
    
    # Normalize X to [-1, 1] for better numerical stability
    X_norm = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    
    for d in range(degree + 1):
        # Use Legendre polynomials (orthogonal on [-1, 1])
        poly = legendre(d)
        X_poly[:, d] = poly(X_norm)
    
    return X_poly

def create_standard_polynomials(X, degree):
    """
    Create standard polynomial features
    
    Parameters:
    X: predictor variable (n_samples,)
    degree: polynomial degree
    
    Returns:
    X_poly: polynomial features (n_samples, degree+1)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X.reshape(-1, 1))

def forward_polynomial_selection(X, y, max_degree=10, criterion='aic'):
    """
    Forward selection for polynomial degree
    
    Parameters:
    X: predictor variable
    y: response variable
    max_degree: maximum degree to consider
    criterion: 'aic', 'bic', or 'cv'
    
    Returns:
    best_degree: optimal polynomial degree
    scores: scores for each degree
    """
    n = len(y)
    scores = []
    
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        X_poly = create_standard_polynomials(X, degree)
        
        if criterion in ['aic', 'bic']:
            # Fit model
            beta_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
            y_hat = X_poly @ beta_hat
            rss = np.sum((y - y_hat)**2)
            
            if criterion == 'aic':
                score = n * np.log(rss/n) + 2 * (degree + 1)
            else:  # bic
                score = n * np.log(rss/n) + (degree + 1) * np.log(n)
        else:  # cv
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
        
        scores.append(score)
    
    best_degree = np.argmin(scores) + 1
    return best_degree, scores

def backward_polynomial_selection(X, y, max_degree=10, criterion='aic'):
    """
    Backward selection for polynomial degree
    """
    n = len(y)
    scores = []
    
    for degree in range(max_degree, 0, -1):
        # Create polynomial features
        X_poly = create_standard_polynomials(X, degree)
        
        if criterion in ['aic', 'bic']:
            # Fit model
            beta_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
            y_hat = X_poly @ beta_hat
            rss = np.sum((y - y_hat)**2)
            
            if criterion == 'aic':
                score = n * np.log(rss/n) + 2 * (degree + 1)
            else:  # bic
                score = n * np.log(rss/n) + (degree + 1) * np.log(n)
        else:  # cv
            model = LinearRegression()
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
            score = -cv_scores.mean()
        
        scores.append(score)
    
    # Reverse to get ascending order
    scores = scores[::-1]
    best_degree = max_degree - np.argmin(scores)
    return best_degree, scores

def demonstrate_model_selection():
    """Demonstrate model selection algorithms"""
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y_true = 2 + 3*X - 0.5*X**2 + 0.1*X**3
    y = y_true + np.random.normal(0, 0.5, 100)
    
    print("=== FORWARD SELECTION ===")
    best_degree_forward, scores_forward = forward_polynomial_selection(X, y, max_degree=8, criterion='aic')
    print(f"Best degree (AIC): {best_degree_forward}")
    
    best_degree_forward_bic, scores_forward_bic = forward_polynomial_selection(X, y, max_degree=8, criterion='bic')
    print(f"Best degree (BIC): {best_degree_forward_bic}")
    
    best_degree_forward_cv, scores_forward_cv = forward_polynomial_selection(X, y, max_degree=8, criterion='cv')
    print(f"Best degree (CV): {best_degree_forward_cv}")
    
    print("\n=== BACKWARD SELECTION ===")
    best_degree_backward, scores_backward = backward_polynomial_selection(X, y, max_degree=8, criterion='aic')
    print(f"Best degree (AIC): {best_degree_backward}")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, 9), scores_forward, 'bo-', label='Forward Selection')
    plt.axvline(x=best_degree_forward, color='red', linestyle='--', label=f'Best: {best_degree_forward}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('AIC Score')
    plt.title('Forward Selection (AIC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, 9), scores_forward_bic, 'go-', label='Forward Selection')
    plt.axvline(x=best_degree_forward_bic, color='red', linestyle='--', label=f'Best: {best_degree_forward_bic}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('BIC Score')
    plt.title('Forward Selection (BIC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, 9), scores_forward_cv, 'ro-', label='Forward Selection')
    plt.axvline(x=best_degree_forward_cv, color='red', linestyle='--', label=f'Best: {best_degree_forward_cv}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('CV MSE')
    plt.title('Forward Selection (CV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'forward_aic': (best_degree_forward, scores_forward),
        'forward_bic': (best_degree_forward_bic, scores_forward_bic),
        'forward_cv': (best_degree_forward_cv, scores_forward_cv),
        'backward_aic': (best_degree_backward, scores_backward)
    }

if __name__ == "__main__":
    results = demonstrate_model_selection()
