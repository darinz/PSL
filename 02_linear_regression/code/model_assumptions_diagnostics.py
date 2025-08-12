"""
Model Assumptions and Diagnostics in Linear Regression
====================================================

This module demonstrates how to check linear regression assumptions
and perform diagnostic analysis including outlier detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

def generate_data_with_violations():
    """Generate data with various assumption violations"""
    
    np.random.seed(42)
    n = 100
    
    # Generate base data
    X = np.random.uniform(0, 10, n)
    
    # Different scenarios
    scenarios = {}
    
    # 1. Normal data (no violations)
    y_normal = 2 + 0.5 * X + np.random.normal(0, 0.5, n)
    scenarios['Normal'] = {'X': X, 'y': y_normal, 'description': 'No assumption violations'}
    
    # 2. Non-linear relationship
    y_nonlinear = 2 + 0.5 * X + 0.1 * X**2 + np.random.normal(0, 0.5, n)
    scenarios['Non-linear'] = {'X': X, 'y': y_nonlinear, 'description': 'Non-linear relationship'}
    
    # 3. Heteroscedasticity
    y_hetero = 2 + 0.5 * X + np.random.normal(0, 0.1 + 0.1 * X, n)
    scenarios['Heteroscedastic'] = {'X': X, 'y': y_hetero, 'description': 'Non-constant variance'}
    
    # 4. Non-normal errors
    y_nonnormal = 2 + 0.5 * X + np.random.exponential(0.5, n) - 0.5
    scenarios['Non-normal'] = {'X': X, 'y': y_nonnormal, 'description': 'Non-normal errors'}
    
    # 5. Outliers
    y_outliers = 2 + 0.5 * X + np.random.normal(0, 0.5, n)
    y_outliers[0] = 20  # Add outlier
    y_outliers[10] = -5  # Add outlier
    scenarios['Outliers'] = {'X': X, 'y': y_outliers, 'description': 'Contains outliers'}
    
    return scenarios

def check_linearity(X, y, model, scenario_name):
    """Check linearity assumption"""
    
    print(f"\n=== LINEARITY CHECK: {scenario_name} ===")
    
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred
    
    # Plot residuals vs fitted values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'Residuals vs Fitted Values\n({scenario_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Plot residuals vs X
    axes[1].scatter(X, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'Residuals vs X\n({scenario_name})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test for non-linearity using polynomial terms
    X_poly = np.column_stack([X, X**2])
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    
    # F-test for quadratic term
    rss_linear = np.sum(residuals**2)
    rss_poly = np.sum((y - model_poly.predict(X_poly))**2)
    f_stat = ((rss_linear - rss_poly) / 1) / (rss_poly / (len(y) - 3))
    p_value = 1 - stats.f.cdf(f_stat, 1, len(y) - 3)
    
    print(f"F-test for quadratic term: F = {f_stat:.3f}, p = {p_value:.3f}")
    print(f"Non-linearity detected: {'Yes' if p_value < 0.05 else 'No'}")

def check_normality(residuals, scenario_name):
    """Check normality assumption"""
    
    print(f"\n=== NORMALITY CHECK: {scenario_name} ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title(f'Q-Q Plot of Residuals\n({scenario_name})')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(residuals, bins=20, density=True, alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2)
    axes[1].set_title(f'Histogram of Residuals\n({scenario_name})')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Density')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Shapiro-Wilk test
    statistic, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test: statistic = {statistic:.3f}, p-value = {p_value:.3f}")
    print(f"Normality assumption violated: {'Yes' if p_value < 0.05 else 'No'}")

def check_homoscedasticity(X, y, model, scenario_name):
    """Check homoscedasticity assumption"""
    
    print(f"\n=== HOMOSCEDASTICITY CHECK: {scenario_name} ===")
    
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred
    
    # Plot residuals vs fitted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Fitted Values\n({scenario_name})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Breusch-Pagan test for heteroscedasticity
    # Simplified version: test correlation between squared residuals and fitted values
    squared_residuals = residuals**2
    correlation = np.corrcoef(y_pred, squared_residuals)[0, 1]
    
    # Test significance of correlation
    t_stat = correlation * np.sqrt((len(y) - 2) / (1 - correlation**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    
    print(f"Correlation between fitted values and squared residuals: {correlation:.3f}")
    print(f"Test statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
    print(f"Heteroscedasticity detected: {'Yes' if p_value < 0.05 else 'No'}")

def calculate_leverage(X):
    """Calculate leverage (hat values)"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    H = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    return np.diag(H)

def calculate_cooks_distance(X, y, model):
    """Calculate Cook's distance for each observation"""
    n = len(y)
    p = X.shape[1] + 1  # +1 for intercept
    residuals = y - model.predict(X.reshape(-1, 1))
    mse = np.sum(residuals**2) / (n - p)
    
    cooks_d = []
    for i in range(n):
        # Remove observation i
        X_i = np.delete(X, i)
        y_i = np.delete(y, i)
        
        # Fit model without observation i
        model_i = LinearRegression()
        model_i.fit(X_i.reshape(-1, 1), y_i)
        
        # Calculate Cook's distance
        beta_diff = model.coef_[0] - model_i.coef_[0]
        cooks_d_i = (beta_diff**2 * np.sum((X - np.mean(X))**2)) / (p * mse)
        cooks_d.append(cooks_d_i)
    
    return np.array(cooks_d)

def detect_outliers(X, y, model, scenario_name):
    """Detect and analyze outliers"""
    
    print(f"\n=== OUTLIER DETECTION: {scenario_name} ===")
    
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred
    
    # Calculate diagnostics
    leverage = calculate_leverage(X)
    cooks_d = calculate_cooks_distance(X, y, model)
    
    # Standardized residuals
    mse = np.sum(residuals**2) / (len(y) - 2)
    standardized_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
    # Studentized residuals
    studentized_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
    # Thresholds
    leverage_threshold = 2 * (X.shape[1] + 1) / len(y)
    cooks_threshold = 4 / len(y)
    
    # Identify outliers
    high_leverage = leverage > leverage_threshold
    high_cooks = cooks_d > cooks_threshold
    high_residuals = np.abs(standardized_residuals) > 2
    
    print(f"Leverage threshold: {leverage_threshold:.3f}")
    print(f"Cook's distance threshold: {cooks_threshold:.3f}")
    print(f"Standardized residual threshold: ±2")
    
    print(f"\nOutlier summary:")
    print(f"  High leverage points: {np.sum(high_leverage)}")
    print(f"  High Cook's distance points: {np.sum(high_cooks)}")
    print(f"  High standardized residuals: {np.sum(high_residuals)}")
    
    # Visualize diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Leverage
    axes[0, 0].scatter(range(len(leverage)), leverage, alpha=0.6)
    axes[0, 0].axhline(y=leverage_threshold, color='r', linestyle='--', label=f'Threshold: {leverage_threshold:.3f}')
    axes[0, 0].set_xlabel('Observation')
    axes[0, 0].set_ylabel('Leverage')
    axes[0, 0].set_title(f'Leverage Plot\n({scenario_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cook's distance
    axes[0, 1].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
    axes[0, 1].axhline(y=cooks_threshold, color='r', linestyle='--', label=f'Threshold: {cooks_threshold:.3f}')
    axes[0, 1].set_xlabel('Observation')
    axes[0, 1].set_ylabel("Cook's Distance")
    axes[0, 1].set_title(f"Cook's Distance Plot\n({scenario_name})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Standardized residuals
    axes[1, 0].scatter(range(len(standardized_residuals)), standardized_residuals, alpha=0.6)
    axes[1, 0].axhline(y=2, color='r', linestyle='--', label='Threshold: ±2')
    axes[1, 0].axhline(y=-2, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Observation')
    axes[1, 0].set_ylabel('Standardized Residuals')
    axes[1, 0].set_title(f'Standardized Residuals Plot\n({scenario_name})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Leverage vs residuals
    axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.6)
    axes[1, 1].axhline(y=2, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=-2, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=leverage_threshold, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title(f'Leverage vs Standardized Residuals\n({scenario_name})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return leverage, cooks_d, standardized_residuals, high_leverage, high_cooks, high_residuals

def comprehensive_diagnostics():
    """Run comprehensive diagnostics on all scenarios"""
    
    print("=== COMPREHENSIVE MODEL DIAGNOSTICS ===")
    
    # Generate data with various violations
    scenarios = generate_data_with_violations()
    
    results = {}
    
    for scenario_name, data in scenarios.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {scenario_name}")
        print(f"Description: {data['description']}")
        print(f"{'='*50}")
        
        X, y = data['X'], data['y']
        
        # Fit model
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        
        # Check assumptions
        check_linearity(X, y, model, scenario_name)
        
        y_pred = model.predict(X.reshape(-1, 1))
        residuals = y - y_pred
        
        check_normality(residuals, scenario_name)
        check_homoscedasticity(X, y, model, scenario_name)
        
        # Detect outliers
        leverage, cooks_d, standardized_residuals, high_leverage, high_cooks, high_residuals = detect_outliers(X, y, model, scenario_name)
        
        # Store results
        results[scenario_name] = {
            'model': model,
            'residuals': residuals,
            'leverage': leverage,
            'cooks_d': cooks_d,
            'standardized_residuals': standardized_residuals,
            'high_leverage': high_leverage,
            'high_cooks': high_cooks,
            'high_residuals': high_residuals
        }
    
    return results

def suggest_remedies(scenario_name, violations):
    """Suggest remedies for assumption violations"""
    
    print(f"\n=== REMEDIES FOR {scenario_name.upper()} ===")
    
    if 'non-linear' in violations:
        print("Non-linearity detected:")
        print("  - Add polynomial terms (X², X³)")
        print("  - Use splines or other non-linear transformations")
        print("  - Consider non-linear models")
    
    if 'non-normal' in violations:
        print("Non-normal errors detected:")
        print("  - Transform the response variable (log, square root)")
        print("  - Use robust regression methods")
        print("  - Consider non-parametric methods")
    
    if 'heteroscedastic' in violations:
        print("Heteroscedasticity detected:")
        print("  - Use weighted least squares")
        print("  - Transform variables")
        print("  - Use robust standard errors")
    
    if 'outliers' in violations:
        print("Outliers detected:")
        print("  - Investigate outliers (data errors vs. real observations)")
        print("  - Use robust regression methods")
        print("  - Consider removing influential outliers (with justification)")
        print("  - Report results with and without outliers")

if __name__ == "__main__":
    # Run comprehensive diagnostics
    results = comprehensive_diagnostics()
    
    # Summary of findings
    print("\n" + "="*60)
    print("SUMMARY OF DIAGNOSTIC FINDINGS")
    print("="*60)
    
    for scenario_name, result in results.items():
        print(f"\n{scenario_name}:")
        print(f"  High leverage points: {np.sum(result['high_leverage'])}")
        print(f"  High Cook's distance points: {np.sum(result['high_cooks'])}")
        print(f"  High standardized residuals: {np.sum(result['high_residuals'])}")
        
        # Identify violations
        violations = []
        if np.sum(result['high_leverage']) > 0:
            violations.append('outliers')
        if np.sum(result['high_residuals']) > 0:
            violations.append('non-normal')
        
        if violations:
            suggest_remedies(scenario_name, violations)
        else:
            print("  No major violations detected.")
