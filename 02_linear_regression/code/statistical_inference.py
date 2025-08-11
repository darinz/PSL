import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def least_squares_estimation(X, y):
    """Compute least squares estimates using the normal equation"""
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        print("Warning: X^T X is not full rank. Solution may not be unique.")
    beta_hat = np.linalg.inv(XtX) @ X.T @ y
    return beta_hat

def statistical_inference(X, y, results, alpha=0.05):
    """
    Perform statistical inference for linear regression
    
    Parameters:
    X: design matrix
    y: response vector
    results: regression results dictionary
    alpha: significance level
    
    Returns:
    Dictionary with inference results
    """
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    df = n - p - 1
    
    # Critical values
    t_critical = stats.t.ppf(1 - alpha/2, df)
    f_critical = stats.f.ppf(1 - alpha, p, df)
    
    # Confidence intervals
    ci_lower = results['coefficients'] - t_critical * results['standard_errors']
    ci_upper = results['coefficients'] + t_critical * results['standard_errors']
    
    # Significance indicators
    significant = results['p_values'] < alpha
    
    # F-test for overall model
    y_mean = np.mean(y)
    TSS = np.sum((y - y_mean)**2)
    MSR = (TSS - results['RSS']) / p  # Mean square regression
    MSE = results['RSS'] / df  # Mean square error
    f_stat = MSR / MSE
    f_p_value = 1 - stats.f.cdf(f_stat, p, df)
    
    return {
        'confidence_intervals': list(zip(ci_lower, ci_upper)),
        'significant_coefficients': significant,
        'f_statistic': f_stat,
        'f_p_value': f_p_value,
        'model_significant': f_p_value < alpha
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

def linear_regression_analysis(X, y):
    """Complete linear regression analysis"""
    n, p_plus_1 = X.shape
    p = p_plus_1 - 1
    
    beta_hat = least_squares_estimation(X, y)
    y_hat = X @ beta_hat
    residuals = y - y_hat
    RSS = np.sum(residuals**2)
    sigma2_hat = RSS / (n - p - 1)
    
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(sigma2_hat * np.diag(XtX_inv))
    t_stats = beta_hat / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    return {
        'coefficients': beta_hat,
        'standard_errors': se_beta,
        'p_values': p_values,
        'RSS': RSS
    }

# Perform statistical inference
X, y, beta_true = create_example_data()
results = linear_regression_analysis(X, y)
inference_results = statistical_inference(X, y, results)

print("=== Statistical Inference ===")
print("Confidence Intervals (95%):")
for i, (lower, upper) in enumerate(inference_results['confidence_intervals']):
    print(f"  β_{i}: [{lower:.4f}, {upper:.4f}]")

print(f"\nF-statistic: {inference_results['f_statistic']:.4f}")
print(f"F-test p-value: {inference_results['f_p_value']:.4f}")
print(f"Model significant: {inference_results['model_significant']}")
