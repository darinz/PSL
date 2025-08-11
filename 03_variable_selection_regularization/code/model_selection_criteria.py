import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations
import statsmodels.api as sm

def demonstrate_model_selection_criteria():
    """Demonstrate model selection criteria (AIC, BIC, Mallow's Cp)"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data with known true model
    n = 100
    p_total = 8
    X = np.random.randn(n, p_total)

    # True model: only first 4 variables matter
    beta_true = np.zeros(p_total)
    beta_true[:4] = [1.5, -0.8, 0.6, -0.4]
    f_true = X @ beta_true
    y = f_true + np.random.normal(0, 0.5, n)

    # Fit full model to get sigma^2 estimate
    full_model = LinearRegression()
    full_model.fit(X, y)
    y_pred_full = full_model.predict(X)
    sigma2_full = np.sum((y - y_pred_full)**2) / (n - p_total - 1)

    print("=== TRUE MODEL ===")
    print(f"True coefficients: {beta_true}")
    print(f"Estimated σ² from full model: {sigma2_full:.4f}")

    # Function to calculate model selection criteria
    def calculate_criteria(X_subset, y, sigma2_full, n):
        """Calculate AIC, BIC, and Mallow's Cp for a given model"""
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        
        p = X_subset.shape[1]
        rss = np.sum((y - y_pred)**2)
        
        # AIC (for normal errors)
        aic = n * np.log(rss/n) + 2*p
        
        # BIC (for normal errors)
        bic = n * np.log(rss/n) + np.log(n)*p
        
        # Mallow's Cp
        cp = rss/sigma2_full - n + 2*p
        
        return aic, bic, cp, rss

    # Generate all possible model combinations
    all_models = []
    criteria_results = []

    for p in range(1, p_total + 1):
        for subset in combinations(range(p_total), p):
            X_subset = X[:, subset]
            aic, bic, cp, rss = calculate_criteria(X_subset, y, sigma2_full, n)
            
            all_models.append(list(subset))
            criteria_results.append({
                'p': p,
                'aic': aic,
                'bic': bic,
                'cp': cp,
                'rss': rss,
                'variables': subset
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(criteria_results)

    # Find best models according to each criterion
    best_aic_idx = results_df['aic'].idxmin()
    best_bic_idx = results_df['bic'].idxmin()
    best_cp_idx = results_df['cp'].idxmin()

    print("\n=== MODEL SELECTION RESULTS ===")
    print("Best model by AIC:")
    print(f"  Variables: {results_df.loc[best_aic_idx, 'variables']}")
    print(f"  AIC: {results_df.loc[best_aic_idx, 'aic']:.2f}")
    print(f"  p: {results_df.loc[best_aic_idx, 'p']}")

    print("\nBest model by BIC:")
    print(f"  Variables: {results_df.loc[best_bic_idx, 'variables']}")
    print(f"  BIC: {results_df.loc[best_bic_idx, 'bic']:.2f}")
    print(f"  p: {results_df.loc[best_bic_idx, 'p']}")

    print("\nBest model by Mallow's Cp:")
    print(f"  Variables: {results_df.loc[best_cp_idx, 'variables']}")
    print(f"  Cp: {results_df.loc[best_cp_idx, 'cp']:.2f}")
    print(f"  p: {results_df.loc[best_cp_idx, 'p']}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # AIC vs number of predictors
    for p in range(1, p_total + 1):
        mask = results_df['p'] == p
        if mask.sum() > 0:
            best_aic_p = results_df[mask]['aic'].min()
            axes[0, 0].scatter(p, best_aic_p, c='blue', s=100, alpha=0.7)

    axes[0, 0].set_xlabel('Number of Predictors (p)')
    axes[0, 0].set_ylabel('Best AIC')
    axes[0, 0].set_title('AIC vs Model Size')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

    # BIC vs number of predictors
    for p in range(1, p_total + 1):
        mask = results_df['p'] == p
        if mask.sum() > 0:
            best_bic_p = results_df[mask]['bic'].min()
            axes[0, 1].scatter(p, best_bic_p, c='green', s=100, alpha=0.7)

    axes[0, 1].set_xlabel('Number of Predictors (p)')
    axes[0, 1].set_ylabel('Best BIC')
    axes[0, 1].set_title('BIC vs Model Size')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

    # Mallow's Cp vs number of predictors
    for p in range(1, p_total + 1):
        mask = results_df['p'] == p
        if mask.sum() > 0:
            best_cp_p = results_df[mask]['cp'].min()
            axes[1, 0].scatter(p, best_cp_p, c='red', s=100, alpha=0.7)

    axes[1, 0].set_xlabel('Number of Predictors (p)')
    axes[1, 0].set_ylabel('Best Mallow\'s Cp')
    axes[1, 0].set_title('Mallow\'s Cp vs Model Size')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

    # RSS vs number of predictors
    for p in range(1, p_total + 1):
        mask = results_df['p'] == p
        if mask.sum() > 0:
            best_rss_p = results_df[mask]['rss'].min()
            axes[1, 1].scatter(p, best_rss_p, c='purple', s=100, alpha=0.7)

    axes[1, 1].set_xlabel('Number of Predictors (p)')
    axes[1, 1].set_ylabel('Best RSS')
    axes[1, 1].set_title('RSS vs Model Size')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

    plt.tight_layout()
    plt.show()

    # Compare criteria behavior
    print("\n=== CRITERIA COMPARISON ===")
    print("Model sizes selected by each criterion:")
    print(f"AIC: {results_df.loc[best_aic_idx, 'p']} predictors")
    print(f"BIC: {results_df.loc[best_bic_idx, 'p']} predictors")
    print(f"Mallow's Cp: {results_df.loc[best_cp_idx, 'p']} predictors")
    print(f"True model: 4 predictors")

    # Show penalty comparison
    print(f"\nPenalty comparison (n={n}, log(n)={np.log(n):.2f}):")
    print(f"AIC penalty per parameter: 2")
    print(f"BIC penalty per parameter: {np.log(n):.2f}")
    print(f"Mallow's Cp penalty per parameter: 2 (scaled by σ²)")
    
    return results_df, best_aic_idx, best_bic_idx, best_cp_idx

# Run demonstration
results_df, best_aic_idx, best_bic_idx, best_cp_idx = demonstrate_model_selection_criteria()
