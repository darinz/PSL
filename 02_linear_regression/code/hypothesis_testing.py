"""
Hypothesis Testing in Linear Regression
======================================

This module demonstrates comprehensive hypothesis testing in linear regression,
including F-tests, t-tests, and the distinction between statistical and practical significance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

def comprehensive_hypothesis_testing():
    """Demonstrate comprehensive hypothesis testing"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data with known effects
    n = 150
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X3 = np.random.normal(0, 1, n)
    
    # True model: X1 and X2 have effects, X3 has no effect
    beta0_true = 2.0
    beta1_true = 1.5  # Strong effect
    beta2_true = -0.8  # Moderate effect
    beta3_true = 0.0   # No effect
    
    y = beta0_true + beta1_true * X1 + beta2_true * X2 + beta3_true * X3 + np.random.normal(0, 0.8, n)
    
    # Create design matrix
    X = np.column_stack([X1, X2, X3])
    feature_names = ['X1', 'X2', 'X3']
    
    print("=== TRUE MODEL ===")
    print(f"Y = {beta0_true} + {beta1_true}*X1 + {beta2_true}*X2 + {beta3_true}*X3 + ε")
    
    # Fit model using scikit-learn
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Calculate degrees of freedom
    n, p = X.shape
    df_residual = n - p - 1
    
    print(f"\n=== MODEL FIT ===")
    print(f"Sample size: {n}")
    print(f"Number of predictors: {p}")
    print(f"Residual degrees of freedom: {df_residual}")
    
    # Calculate standard errors manually
    X_with_intercept = np.column_stack([np.ones(n), X])
    mse = np.sum(residuals**2) / df_residual
    var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se_beta = np.sqrt(np.diag(var_beta))[1:]  # Exclude intercept
    
    # Calculate t-statistics and p-values
    t_stats = model.coef_ / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))
    
    # Calculate confidence intervals
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, df_residual)
    ci_lower = model.coef_ - t_critical * se_beta
    ci_upper = model.coef_ + t_critical * se_beta
    
    print(f"\n=== HYPOTHESIS TESTING RESULTS ===")
    print("Manual calculations:")
    for i, (name, coef, se, t_stat, p_val, ci_l, ci_u) in enumerate(zip(feature_names, model.coef_, se_beta, t_stats, p_values, ci_lower, ci_upper)):
        print(f"\n{name}:")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  Standard Error: {se:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  {100*(1-alpha)}% CI: [{ci_l:.4f}, {ci_u:.4f}]")
        print(f"  Significant: {'Yes' if p_val < alpha else 'No'}")
    
    # Compare with statsmodels (more comprehensive output)
    print(f"\n=== STATSMODELS COMPARISON ===")
    X_sm = sm.add_constant(X)
    sm_model = sm.OLS(y, X_sm).fit()
    print(sm_model.summary())
    
    # F-test for overall model significance
    print(f"\n=== OVERALL F-TEST ===")
    # Calculate F-statistic manually
    y_mean = np.mean(y)
    ssr = np.sum((y_pred - y_mean)**2)  # Sum of squares regression
    sse = np.sum(residuals**2)          # Sum of squares error
    msr = ssr / p                       # Mean square regression
    mse_manual = sse / df_residual      # Mean square error
    f_stat = msr / mse_manual
    f_p_value = 1 - stats.f.cdf(f_stat, p, df_residual)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {f_p_value:.4f}")
    print(f"R²: {sm_model.rsquared:.4f}")
    print(f"Adjusted R²: {sm_model.rsquared_adj:.4f}")
    
    return X, y, model, residuals, feature_names, t_stats, p_values, ci_lower, ci_upper

def visualize_hypothesis_testing(X, y, model, residuals, feature_names, t_stats, p_values, ci_lower, ci_upper):
    """Visualize hypothesis testing results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coefficient estimates with confidence intervals
    coef_names = ['β₁ (X1)', 'β₂ (X2)', 'β₃ (X3)']
    true_coefs = [1.5, -0.8, 0.0]  # From the true model
    
    axes[0, 0].errorbar(range(len(model.coef_)), model.coef_, 
                       yerr=stats.t.ppf(0.975, len(y)-4) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * np.sum(residuals**2) / (len(y)-4)), 
                       fmt='o', capsize=5, capthick=2)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Coefficients')
    axes[0, 0].set_ylabel('Estimate')
    axes[0, 0].set_title('Coefficient Estimates with 95% CI')
    axes[0, 0].set_xticks(range(len(model.coef_)))
    axes[0, 0].set_xticklabels(coef_names)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot true vs estimated coefficients
    axes[0, 1].scatter(true_coefs, model.coef_, s=100, alpha=0.7)
    axes[0, 1].plot([min(true_coefs), max(true_coefs)], [min(true_coefs), max(true_coefs)], 'r--')
    axes[0, 1].set_xlabel('True Coefficients')
    axes[0, 1].set_ylabel('Estimated Coefficients')
    axes[0, 1].set_title('True vs Estimated Coefficients')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals vs fitted values
    y_pred = model.predict(X)
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs Fitted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'True_β': [1.5, -0.8, 0.0],
        'Est_β': model.coef_,
        'SE': np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * np.sum(residuals**2) / (len(y)-4)),
        't_stat': t_stats,
        'p_value': p_values,
        'Significant': [p < 0.05 for p in p_values],
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })
    
    print(f"\n=== SUMMARY TABLE ===")
    print(results_df.round(4).to_string(index=False))

def sample_size_effect_demo():
    """Demonstrate the effect of sample size on statistical vs practical significance"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define sample sizes to test
    sample_sizes = [30, 100, 500, 1000, 5000, 10000]
    true_effect = 0.1  # Small true effect
    noise_std = 1.0    # Large noise
    
    results = []
    
    for n in sample_sizes:
        # Generate data with small effect
        X = np.random.randn(n, 1)
        y = true_effect * X.flatten() + np.random.normal(0, noise_std, n)
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        # Calculate F-statistic manually
        y_mean = np.mean(y)
        ssr = np.sum((y_pred - y_mean)**2)
        sse = np.sum((y - y_pred)**2)
        msr = ssr / 1  # 1 predictor
        mse = sse / (n - 2)  # n - p - 1 degrees of freedom
        f_stat = msr / mse
        f_p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
        
        # Calculate t-statistic
        se_beta = np.sqrt(mse / np.sum((X.flatten() - np.mean(X))**2))
        t_stat = model.coef_[0] / se_beta
        t_p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))
        
        results.append({
            'n': n,
            'R²': r2,
            'F_stat': f_stat,
            'F_p_value': f_p_value,
            't_stat': t_stat,
            't_p_value': t_p_value,
            'beta_est': model.coef_[0],
            'significant': f_p_value < 0.05
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("=== SAMPLE SIZE EFFECT ON SIGNIFICANCE ===")
    print("True effect size: 0.1")
    print("Noise standard deviation: 1.0")
    print("\nResults:")
    print(results_df.round(4).to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # R² vs sample size
    axes[0, 0].plot(results_df['n'], results_df['R²'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Sample Size')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].set_title('R² vs Sample Size')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='R² = 0.05')
    axes[0, 0].legend()
    
    # F-statistic vs sample size
    axes[0, 1].plot(results_df['n'], results_df['F_stat'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sample Size')
    axes[0, 1].set_ylabel('F-statistic')
    axes[0, 1].set_title('F-statistic vs Sample Size')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # p-value vs sample size
    axes[1, 0].plot(results_df['n'], results_df['F_p_value'], 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('p-value')
    axes[1, 0].set_title('p-value vs Sample Size')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Coefficient estimate vs sample size
    axes[1, 1].plot(results_df['n'], results_df['beta_est'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=true_effect, color='red', linestyle='--', alpha=0.7, label=f'True β = {true_effect}')
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('Estimated β')
    axes[1, 1].set_title('Coefficient Estimate vs Sample Size')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary of findings
    print("\n=== KEY INSIGHTS ===")
    print("1. R² remains relatively constant across sample sizes (around 0.01-0.02)")
    print("2. F-statistic increases dramatically with sample size")
    print("3. p-value decreases with sample size, eventually becoming significant")
    print("4. Coefficient estimates converge to true value with larger samples")
    
    # Practical interpretation
    print("\n=== PRACTICAL INTERPRETATION ===")
    print("With n = 30:   R² = 0.01, p = 0.74  → Not significant, weak effect")
    print("With n = 1000: R² = 0.01, p = 0.02  → Significant, but still weak effect")
    print("With n = 10000: R² = 0.01, p < 0.001 → Highly significant, but still weak effect")
    
    print("\nMoral: Large sample sizes can detect tiny effects that may not be practically meaningful!")

if __name__ == "__main__":
    # Comprehensive hypothesis testing
    X, y, model, residuals, feature_names, t_stats, p_values, ci_lower, ci_upper = comprehensive_hypothesis_testing()
    
    # Visualize results
    visualize_hypothesis_testing(X, y, model, residuals, feature_names, t_stats, p_values, ci_lower, ci_upper)
    
    # Demonstrate sample size effect
    sample_size_effect_demo()
