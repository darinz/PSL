import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def demonstrate_confounding_effect():
    """Demonstrate confounding effect in simple vs multiple regression"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate correlated predictors with known correlation
    n = 1000
    rho = 0.8  # Correlation between X1 and X2

    # Generate X1 and X2 with specified correlation
    X1 = np.random.normal(0, 1, n)
    X2 = rho * X1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

    # True model: Y depends only on X2, not X1
    beta0_true = 2.0
    beta1_true = 0.0  # No direct effect of X1
    beta2_true = 1.5  # Strong effect of X2
    sigma = 0.5

    y = beta0_true + beta1_true * X1 + beta2_true * X2 + np.random.normal(0, sigma, n)

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': y
    })

    print("=== TRUE MODEL ===")
    print(f"Y = {beta0_true} + {beta1_true}*X1 + {beta2_true}*X2 + ε")
    print(f"Correlation between X1 and X2: {np.corrcoef(X1, X2)[0,1]:.3f}")

    # Simple Linear Regression: Y ~ X1
    print("\n=== SIMPLE LINEAR REGRESSION: Y ~ X1 ===")
    slr_model = LinearRegression()
    slr_model.fit(X1.reshape(-1, 1), y)
    slr_coef = slr_model.coef_[0]
    slr_intercept = slr_model.intercept_
    slr_r2 = r2_score(y, slr_model.predict(X1.reshape(-1, 1)))

    print(f"Estimated model: Y = {slr_intercept:.3f} + {slr_coef:.3f}*X1")
    print(f"R² = {slr_r2:.3f}")
    print(f"Bias in β1: {slr_coef - beta1_true:.3f}")

    # Multiple Linear Regression: Y ~ X1 + X2
    print("\n=== MULTIPLE LINEAR REGRESSION: Y ~ X1 + X2 ===")
    mlr_model = LinearRegression()
    X_both = np.column_stack([X1, X2])
    mlr_model.fit(X_both, y)
    mlr_coefs = mlr_model.coef_
    mlr_intercept = mlr_model.intercept_
    mlr_r2 = r2_score(y, mlr_model.predict(X_both))

    print(f"Estimated model: Y = {mlr_intercept:.3f} + {mlr_coefs[0]:.3f}*X1 + {mlr_coefs[1]:.3f}*X2")
    print(f"R² = {mlr_r2:.3f}")
    print(f"β1 bias: {mlr_coefs[0] - beta1_true:.3f}")
    print(f"β2 bias: {mlr_coefs[1] - beta2_true:.3f}")

    # Theoretical bias calculation
    cov_x1x2 = np.cov(X1, X2)[0,1]
    var_x1 = np.var(X1)
    theoretical_bias = beta2_true * cov_x1x2 / var_x1
    print(f"Theoretical bias: {theoretical_bias:.3f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot: X1 vs Y
    axes[0, 0].scatter(X1, y, alpha=0.6)
    axes[0, 0].plot(X1, slr_model.predict(X1.reshape(-1, 1)), 'r-', linewidth=2)
    axes[0, 0].set_xlabel('X1')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Simple Regression: Y ~ X1')
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot: X2 vs Y
    axes[0, 1].scatter(X2, y, alpha=0.6)
    # Fit Y ~ X2 for comparison
    model_x2 = LinearRegression()
    model_x2.fit(X2.reshape(-1, 1), y)
    axes[0, 1].plot(X2, model_x2.predict(X2.reshape(-1, 1)), 'g-', linewidth=2)
    axes[0, 1].set_xlabel('X2')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Simple Regression: Y ~ X2')
    axes[0, 1].grid(True, alpha=0.3)

    # Scatter plot: X1 vs X2
    axes[1, 0].scatter(X1, X2, alpha=0.6)
    axes[1, 0].set_xlabel('X1')
    axes[1, 0].set_ylabel('X2')
    axes[1, 0].set_title('Correlation between X1 and X2')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals comparison
    slr_residuals = y - slr_model.predict(X1.reshape(-1, 1))
    mlr_residuals = y - mlr_model.predict(X_both)

    axes[1, 1].scatter(slr_model.predict(X1.reshape(-1, 1)), slr_residuals, 
                       alpha=0.6, label='SLR', color='red')
    axes[1, 1].scatter(mlr_model.predict(X_both), mlr_residuals, 
                       alpha=0.6, label='MLR', color='blue')
    axes[1, 1].axhline(y=0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary table
    summary_df = pd.DataFrame({
        'Model': ['Simple (Y~X1)', 'Multiple (Y~X1+X2)'],
        'β1': [slr_coef, mlr_coefs[0]],
        'β2': [np.nan, mlr_coefs[1]],
        'R²': [slr_r2, mlr_r2],
        'β1 Bias': [slr_coef - beta1_true, mlr_coefs[0] - beta1_true]
    })
    
    print("\n=== SUMMARY TABLE ===")
    print(summary_df)
    
    return slr_model, mlr_model, df

# Run demonstration
slr_model, mlr_model, df = demonstrate_confounding_effect()
