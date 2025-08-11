import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

def demonstrate_screening_stepwise():
    """Demonstrate screening and stepwise selection for high-dimensional variable selection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate high-dimensional data
    n = 200
    p = 1000
    X = np.random.randn(n, p)
    
    # True model: only first 5 variables matter
    beta_true = np.zeros(p)
    beta_true[:5] = [1.5, -0.8, 0.6, -0.4, 0.3]
    f_true = X @ beta_true
    y = f_true + np.random.normal(0, 0.5, n)

    print("=== HIGH-DIMENSIONAL VARIABLE SELECTION ===")
    print(f"Sample size: {n}")
    print(f"Number of predictors: {p}")
    print(f"True model variables: {list(range(5))}")
    print(f"True model size: 5")

    # 1. Correlation-based screening
    print("\n=== CORRELATION-BASED SCREENING ===")

    # Calculate correlations
    correlations = np.corrcoef(X.T, y)[:-1, -1]
    corr_abs = np.abs(correlations)

    # Select top K variables
    K = n // 3  # n/3 variables
    top_k_indices = np.argsort(corr_abs)[-K:]

    print(f"Selected {K} variables based on correlation")
    print(f"Top 10 correlations: {corr_abs[top_k_indices[-10:]]}")
    print(f"True variables in top {K}: {sum(i < 5 for i in top_k_indices)}/5")

    # 2. Univariate regression screening
    print("\n=== UNIVARIATE REGRESSION SCREENING ===")

    p_values = []
    for j in range(p):
        # Simple linear regression
        model = LinearRegression()
        model.fit(X[:, j:j+1], y)
        y_pred = model.predict(X[:, j:j+1])
        
        # Calculate p-value
        rss = np.sum((y - y_pred)**2)
        tss = np.sum((y - np.mean(y))**2)
        r_squared = 1 - rss/tss
        
        if r_squared < 1:
            f_stat = (r_squared / 1) / ((1 - r_squared) / (n - 2))
            p_val = 1 - stats.f.cdf(f_stat, 1, n - 2)
        else:
            p_val = 0
        
        p_values.append(p_val)

    p_values = np.array(p_values)
    alpha_screen = 0.05
    significant_vars = np.where(p_values < alpha_screen)[0]

    print(f"Variables significant at α = {alpha_screen}: {len(significant_vars)}")
    print(f"True variables significant: {sum(i < 5 for i in significant_vars)}/5")

    # 3. Mutual information screening
    print("\n=== MUTUAL INFORMATION SCREENING ===")

    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # Select top K variables
    top_k_mi_indices = np.argsort(mi_scores)[-K:]

    print(f"Selected {K} variables based on mutual information")
    print(f"Top 10 MI scores: {mi_scores[top_k_mi_indices[-10:]]}")
    print(f"True variables in top {K}: {sum(i < 5 for i in top_k_mi_indices)}/5")

    # 4. Combined screening approach
    print("\n=== COMBINED SCREENING APPROACH ===")

    # Combine all screening methods
    all_screened = set(top_k_indices) | set(significant_vars) | set(top_k_mi_indices)
    screened_vars = list(all_screened)

    print(f"Combined screening selected {len(screened_vars)} variables")
    print(f"True variables in combined set: {sum(i < 5 for i in screened_vars)}/5")

    # 5. Stepwise selection on screened variables
    print("\n=== STEPWISE SELECTION ON SCREENED VARIABLES ===")

    if len(screened_vars) > 0:
        X_screened = X[:, screened_vars]
        
        # Function to calculate AIC
        def calculate_aic(X_subset, y):
            if X_subset.shape[1] == 0:
                return np.inf
            
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            
            rss = np.sum((y - y_pred)**2)
            p = X_subset.shape[1]
            aic = n * np.log(rss/n) + 2*p
            
            return aic
        
        # Stepwise selection
        current_vars = set(range(X_screened.shape[1]))
        best_score = calculate_aic(X_screened, y)
        improved = True
        
        while improved and len(current_vars) > 1:
            improved = False
            
            # Try removing each variable
            for var in list(current_vars):
                test_vars = list(current_vars - {var})
                X_test = X_screened[:, test_vars]
                score = calculate_aic(X_test, y)
                
                if score < best_score:
                    current_vars.remove(var)
                    best_score = score
                    improved = True
                    break
        
        final_vars = [screened_vars[i] for i in current_vars]
        print(f"Final model size: {len(final_vars)}")
        print(f"Final variables: {final_vars}")
        print(f"True variables in final model: {sum(i < 5 for i in final_vars)}/5")
        print(f"Final AIC: {best_score:.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Correlation distribution
    axes[0, 0].hist(corr_abs, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=corr_abs[top_k_indices[-1]], color='red', linestyle='--', 
                       label=f'Threshold ({K} variables)')
    axes[0, 0].set_xlabel('Absolute Correlation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Correlation Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # P-value distribution
    axes[0, 1].hist(p_values, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=alpha_screen, color='red', linestyle='--', 
                       label=f'α = {alpha_screen}')
    axes[0, 1].set_xlabel('P-value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('P-value Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Mutual information distribution
    axes[0, 2].hist(mi_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=mi_scores[top_k_mi_indices[-1]], color='red', linestyle='--', 
                       label=f'Threshold ({K} variables)')
    axes[0, 2].set_xlabel('Mutual Information')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Mutual Information Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Screening comparison
    screening_methods = ['Correlation', 'P-value', 'MI', 'Combined']
    true_vars_found = [
        sum(i < 5 for i in top_k_indices),
        sum(i < 5 for i in significant_vars),
        sum(i < 5 for i in top_k_mi_indices),
        sum(i < 5 for i in screened_vars)
    ]

    axes[1, 0].bar(screening_methods, true_vars_found, color=['blue', 'green', 'red', 'purple'])
    axes[1, 0].set_ylabel('True Variables Found')
    axes[1, 0].set_title('Screening Performance')
    axes[1, 0].set_ylim(0, 5)
    axes[1, 0].grid(True, alpha=0.3)

    # Variable importance comparison
    true_correlations = corr_abs[:5]
    true_p_values = p_values[:5]
    true_mi_scores = mi_scores[:5]

    x_pos = np.arange(5)
    width = 0.25

    axes[1, 1].bar(x_pos - width, true_correlations, width, label='Correlation', alpha=0.7)
    axes[1, 1].bar(x_pos, -np.log10(true_p_values), width, label='-log10(p-value)', alpha=0.7)
    axes[1, 1].bar(x_pos + width, true_mi_scores, width, label='MI Score', alpha=0.7)
    axes[1, 1].set_xlabel('True Variable Index')
    axes[1, 1].set_ylabel('Importance Score')
    axes[1, 1].set_title('True Variable Importance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Final model comparison
    if len(screened_vars) > 0:
        final_model_vars = final_vars if 'final_vars' in locals() else []
        final_true_vars = sum(i < 5 for i in final_model_vars)
        
        axes[1, 2].pie([final_true_vars, 5 - final_true_vars, len(final_model_vars) - final_true_vars], 
                       labels=['True Variables', 'Missed True', 'False Positives'],
                       colors=['green', 'red', 'orange'], autopct='%1.1f%%')
        axes[1, 2].set_title('Final Model Composition')

    plt.tight_layout()
    plt.show()

    # Summary
    print("\n=== SCREENING SUMMARY ===")
    print("Screening Method Performance:")
    print(f"Correlation screening: {true_vars_found[0]}/5 true variables")
    print(f"P-value screening: {true_vars_found[1]}/5 true variables")
    print(f"MI screening: {true_vars_found[2]}/5 true variables")
    print(f"Combined screening: {true_vars_found[3]}/5 true variables")

    if len(screened_vars) > 0:
        print(f"\nFinal model after stepwise: {len(final_vars)} variables")
        print(f"True variables in final model: {sum(i < 5 for i in final_vars)}/5")

    print(f"\nKey Insights:")
    print("1. Screening reduces computational complexity from O(2^p) to O(2^K)")
    print("2. Different screening methods may select different variables")
    print("3. Combined screening often captures more true variables")
    print("4. Stepwise selection on screened variables provides final model")
    
    return screened_vars, final_vars if 'final_vars' in locals() else []

# Run demonstration
screened_vars, final_vars = demonstrate_screening_stepwise()
