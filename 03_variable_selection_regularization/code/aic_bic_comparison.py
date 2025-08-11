import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations

def demonstrate_aic_bic_comparison():
    """Demonstrate AIC vs BIC comparison across different sample sizes"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data with different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000]
    p_total = 10

    results = []

    for n in sample_sizes:
        # Generate data
        X = np.random.randn(n, p_total)
        
        # True model: only first 3 variables matter
        beta_true = np.zeros(p_total)
        beta_true[:3] = [1.5, -0.8, 0.6]
        f_true = X @ beta_true
        y = f_true + np.random.normal(0, 0.5, n)
        
        # Calculate criteria for all possible models
        for p in range(1, min(6, p_total + 1)):  # Limit to 5 predictors for computational efficiency
            for subset in combinations(range(p_total), p):
                X_subset = X[:, subset]
                model = LinearRegression()
                model.fit(X_subset, y)
                y_pred = model.predict(X_subset)
                
                rss = np.sum((y - y_pred)**2)
                
                # Calculate AIC and BIC
                aic = n * np.log(rss/n) + 2*p
                bic = n * np.log(rss/n) + np.log(n)*p
                
                # Check if this is the true model (first 3 variables)
                is_true_model = set(subset) == set(range(3))
                
                results.append({
                    'n': n,
                    'p': p,
                    'aic': aic,
                    'bic': bic,
                    'is_true_model': is_true_model,
                    'variables': subset
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Analysis by sample size
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, n in enumerate(sample_sizes):
        mask = results_df['n'] == n
        
        # Find best models by each criterion
        best_aic_idx = results_df[mask]['aic'].idxmin()
        best_bic_idx = results_df[mask]['bic'].idxmin()
        
        best_aic_p = results_df.loc[best_aic_idx, 'p']
        best_bic_p = results_df.loc[best_bic_idx, 'p']
        
        # Plot AIC vs BIC for this sample size
        axes[0, i].scatter(results_df[mask]['aic'], results_df[mask]['bic'], 
                          c=results_df[mask]['is_true_model'], cmap='viridis', alpha=0.6)
        axes[0, i].scatter(results_df.loc[best_aic_idx, 'aic'], results_df.loc[best_aic_idx, 'bic'], 
                          c='red', s=200, marker='*', label='Best AIC')
        axes[0, i].scatter(results_df.loc[best_bic_idx, 'aic'], results_df.loc[best_bic_idx, 'bic'], 
                          c='blue', s=200, marker='s', label='Best BIC')
        
        axes[0, i].set_xlabel('AIC')
        axes[0, i].set_ylabel('BIC')
        axes[0, i].set_title(f'n = {n}\nAIC: {best_aic_p} vars, BIC: {best_bic_p} vars')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot penalty comparison
        p_values = range(1, 6)
        aic_penalties = [2*p for p in p_values]
        bic_penalties = [np.log(n)*p for p in p_values]
        
        axes[1, i].plot(p_values, aic_penalties, 'ro-', label='AIC Penalty', linewidth=2)
        axes[1, i].plot(p_values, bic_penalties, 'bo-', label='BIC Penalty', linewidth=2)
        axes[1, i].set_xlabel('Number of Parameters (p)')
        axes[1, i].set_ylabel('Penalty')
        axes[1, i].set_title(f'Penalty Comparison (n={n})')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("=== AIC vs BIC COMPARISON SUMMARY ===")
    for n in sample_sizes:
        mask = results_df['n'] == n
        best_aic_idx = results_df[mask]['aic'].idxmin()
        best_bic_idx = results_df[mask]['bic'].idxmin()
        
        aic_p = results_df.loc[best_aic_idx, 'p']
        bic_p = results_df.loc[best_bic_idx, 'p']
        aic_true = results_df.loc[best_aic_idx, 'is_true_model']
        bic_true = results_df.loc[best_bic_idx, 'is_true_model']
        
        print(f"\nn = {n}:")
        print(f"  AIC selects {aic_p} variables (true model: {aic_true})")
        print(f"  BIC selects {bic_p} variables (true model: {bic_true})")
        print(f"  BIC penalty factor: {np.log(n):.2f}")

    # Theoretical analysis
    print(f"\n=== THEORETICAL INSIGHTS ===")
    print("As sample size increases:")
    print("- AIC penalty remains constant at 2 per parameter")
    print("- BIC penalty increases with log(n)")
    print("- BIC becomes more conservative with larger samples")
    print("- AIC maintains prediction focus regardless of sample size")
    
    return results_df

# Run demonstration
results_df = demonstrate_aic_bic_comparison()
