"""
Collinearity Analysis in Linear Regression
=========================================

This module demonstrates how to detect and handle collinearity (multicollinearity)
in linear regression, including VIF calculation and correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_vif(X):
    """Calculate VIF for each predictor"""
    n_features = X.shape[1]
    vif = []
    
    for i in range(n_features):
        # Regress X_i on all other predictors
        X_others = np.delete(X, i, axis=1)
        X_i = X[:, i]
        
        model = LinearRegression()
        model.fit(X_others, X_i)
        r2 = r2_score(X_i, model.predict(X_others))
        
        vif_i = 1 / (1 - r2) if r2 < 1 else np.inf
        vif.append(vif_i)
    
    return vif

def demonstrate_collinearity():
    """Demonstrate collinearity detection and effects"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate collinear data
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = 0.95 * X1 + np.random.normal(0, 0.1, n)  # Very high correlation
    X3 = np.random.normal(0, 1, n)  # Independent
    X4 = 0.8 * X1 + 0.3 * X3 + np.random.normal(0, 0.2, n)  # Moderate correlation
    
    # Create design matrix
    X = np.column_stack([X1, X2, X3, X4])
    feature_names = ['X1', 'X2', 'X3', 'X4']
    
    # True model
    beta0_true = 2.0
    beta1_true = 1.5
    beta2_true = -0.8
    beta3_true = 0.4
    beta4_true = 0.2
    
    y = beta0_true + beta1_true * X1 + beta2_true * X2 + beta3_true * X3 + beta4_true * X4 + np.random.normal(0, 0.5, n)
    
    print("=== TRUE MODEL ===")
    print(f"Y = {beta0_true} + {beta1_true}*X1 + {beta2_true}*X2 + {beta3_true}*X3 + {beta4_true}*X4 + ε")
    
    return X, y, feature_names, [beta1_true, beta2_true, beta3_true, beta4_true]

def analyze_correlation_matrix(X, feature_names):
    """Analyze correlation matrix of predictors"""
    
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
    
    print("Correlation Matrix:")
    print(corr_df.round(3))
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = abs(corr_matrix[i, j])
            if corr_val > 0.7:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))
    
    if high_corr_pairs:
        print("\nHigh correlations (|r| > 0.7):")
        for var1, var2, corr_val in high_corr_pairs:
            print(f"  {var1} vs {var2}: {corr_val:.3f}")
    else:
        print("\nNo high correlations detected.")
    
    return corr_df, high_corr_pairs

def calculate_and_interpret_vif(X, feature_names):
    """Calculate and interpret VIF values"""
    
    print("\n=== VARIANCE INFLATION FACTOR (VIF) ===")
    
    vif_values = calculate_vif(X)
    
    print("VIF values:")
    for name, vif in zip(feature_names, vif_values):
        if np.isinf(vif):
            print(f"  {name}: ∞ (Perfect collinearity)")
        else:
            print(f"  {name}: {vif:.2f}")
    
    # Interpret VIF values
    print("\nVIF Interpretation:")
    for name, vif in zip(feature_names, vif_values):
        if np.isinf(vif):
            print(f"  {name}: Perfect collinearity - remove this variable")
        elif vif > 10:
            print(f"  {name}: High collinearity (VIF > 10) - consider removing")
        elif vif > 5:
            print(f"  {name}: Moderate collinearity (VIF > 5) - monitor closely")
        else:
            print(f"  {name}: Low collinearity (VIF < 5) - acceptable")
    
    return vif_values

def fit_models_and_compare(X, y, feature_names, true_coefs):
    """Fit models with and without collinear variables"""
    
    print("\n=== MODEL COMPARISON ===")
    
    # Full model (with all variables)
    model_full = LinearRegression()
    model_full.fit(X, y)
    y_pred_full = model_full.predict(X)
    r2_full = r2_score(y, y_pred_full)
    
    print("Full Model (all variables):")
    print(f"  R²: {r2_full:.4f}")
    for name, coef, true_coef in zip(feature_names, model_full.coef_, true_coefs):
        print(f"  {name}: {coef:.4f} (true: {true_coef:.4f}, diff: {abs(coef - true_coef):.4f})")
    
    # Model without X2 (highly collinear with X1)
    X_reduced = np.delete(X, 1, axis=1)  # Remove X2
    feature_names_reduced = [name for i, name in enumerate(feature_names) if i != 1]
    true_coefs_reduced = [coef for i, coef in enumerate(true_coefs) if i != 1]
    
    model_reduced = LinearRegression()
    model_reduced.fit(X_reduced, y)
    y_pred_reduced = model_reduced.predict(X_reduced)
    r2_reduced = r2_score(y, y_pred_reduced)
    
    print(f"\nReduced Model (without {feature_names[1]}):")
    print(f"  R²: {r2_reduced:.4f}")
    print(f"  R² difference: {r2_full - r2_reduced:.4f}")
    for name, coef, true_coef in zip(feature_names_reduced, model_reduced.coef_, true_coefs_reduced):
        print(f"  {name}: {coef:.4f} (true: {true_coef:.4f}, diff: {abs(coef - true_coef):.4f})")
    
    return model_full, model_reduced, r2_full, r2_reduced

def visualize_collinearity(X, feature_names, corr_df, vif_values):
    """Visualize collinearity analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correlation heatmap
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Matrix of Predictors')
    
    # VIF bar plot
    vif_clean = [v if not np.isinf(v) else 20 for v in vif_values]  # Cap at 20 for visualization
    bars = axes[0, 1].bar(feature_names, vif_clean, color=['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_clean])
    axes[0, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='VIF = 5')
    axes[0, 1].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='VIF = 10')
    axes[0, 1].set_ylabel('VIF')
    axes[0, 1].set_title('Variance Inflation Factor')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add labels for infinite VIF values
    for i, (bar, vif) in enumerate(zip(bars, vif_values)):
        if np.isinf(vif):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           '∞', ha='center', va='bottom', fontweight='bold')
    
    # Scatter plot of most collinear variables
    axes[1, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[1, 0].set_xlabel(feature_names[0])
    axes[1, 0].set_ylabel(feature_names[1])
    axes[1, 0].set_title(f'{feature_names[0]} vs {feature_names[1]} (r = {corr_df.iloc[0, 1]:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot of independent variables
    axes[1, 1].scatter(X[:, 0], X[:, 2], alpha=0.6)
    axes[1, 1].set_xlabel(feature_names[0])
    axes[1, 1].set_ylabel(feature_names[2])
    axes[1, 1].set_title(f'{feature_names[0]} vs {feature_names[2]} (r = {corr_df.iloc[0, 2]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_collinearity_effects():
    """Demonstrate the effects of collinearity on coefficient estimates"""
    
    print("\n=== COLLINEARITY EFFECTS DEMONSTRATION ===")
    
    # Generate data with different levels of collinearity
    np.random.seed(42)
    n = 100
    
    # Low collinearity
    X1_low = np.random.normal(0, 1, n)
    X2_low = 0.2 * X1_low + np.random.normal(0, 0.98, n)  # r ≈ 0.2
    
    # High collinearity
    X1_high = np.random.normal(0, 1, n)
    X2_high = 0.9 * X1_high + np.random.normal(0, 0.44, n)  # r ≈ 0.9
    
    # True coefficients
    beta1_true = 1.5
    beta2_true = -0.8
    
    # Generate response variables
    y_low = beta1_true * X1_low + beta2_true * X2_low + np.random.normal(0, 0.5, n)
    y_high = beta1_true * X1_high + beta2_true * X2_high + np.random.normal(0, 0.5, n)
    
    # Fit models
    X_low = np.column_stack([X1_low, X2_low])
    X_high = np.column_stack([X1_high, X2_high])
    
    model_low = LinearRegression()
    model_low.fit(X_low, y_low)
    
    model_high = LinearRegression()
    model_high.fit(X_high, y_high)
    
    # Calculate correlations
    corr_low = np.corrcoef(X1_low, X2_low)[0, 1]
    corr_high = np.corrcoef(X1_high, X2_high)[0, 1]
    
    # Calculate VIF
    vif_low = calculate_vif(X_low)
    vif_high = calculate_vif(X_high)
    
    print("Low Collinearity (r = {:.3f}):".format(corr_low))
    print(f"  VIF: {vif_low[0]:.2f}, {vif_low[1]:.2f}")
    print(f"  β1: {model_low.coef_[0]:.3f} (true: {beta1_true})")
    print(f"  β2: {model_low.coef_[1]:.3f} (true: {beta2_true})")
    
    print("\nHigh Collinearity (r = {:.3f}):".format(corr_high))
    print(f"  VIF: {vif_high[0]:.2f}, {vif_high[1]:.2f}")
    print(f"  β1: {model_high.coef_[0]:.3f} (true: {beta1_true})")
    print(f"  β2: {model_high.coef_[1]:.3f} (true: {beta2_true})")
    
    # Visualize the difference
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Low collinearity
    axes[0].scatter(X1_low, X2_low, alpha=0.6)
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_title(f'Low Collinearity (r = {corr_low:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # High collinearity
    axes[1].scatter(X1_high, X2_high, alpha=0.6)
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_title(f'High Collinearity (r = {corr_high:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def suggest_remedies(vif_values, feature_names, high_corr_pairs):
    """Suggest remedies for collinearity"""
    
    print("\n=== COLLINEARITY REMEDIES ===")
    
    high_vif_vars = [name for name, vif in zip(feature_names, vif_values) if vif > 10 or np.isinf(vif)]
    
    if high_vif_vars:
        print("Variables with high VIF (>10):")
        for var in high_vif_vars:
            print(f"  - {var}")
        
        print("\nSuggested remedies:")
        print("1. Remove redundant variables:")
        for var in high_vif_vars:
            print(f"   - Consider removing {var}")
        
        print("\n2. Combine related variables:")
        if high_corr_pairs:
            for var1, var2, corr in high_corr_pairs:
                print(f"   - Create composite variable from {var1} and {var2} (r = {corr:.3f})")
        
        print("\n3. Use regularization methods:")
        print("   - Ridge regression (L2 penalty)")
        print("   - Lasso regression (L1 penalty)")
        print("   - Elastic net (combination)")
        
        print("\n4. Collect more data:")
        print("   - More observations can help reduce collinearity effects")
        
        print("\n5. Transform variables:")
        print("   - Center and scale variables")
        print("   - Use principal components analysis (PCA)")
    else:
        print("No severe collinearity detected. VIF values are acceptable.")

if __name__ == "__main__":
    # Demonstrate collinearity
    X, y, feature_names, true_coefs = demonstrate_collinearity()
    
    # Analyze correlation matrix
    corr_df, high_corr_pairs = analyze_correlation_matrix(X, feature_names)
    
    # Calculate and interpret VIF
    vif_values = calculate_and_interpret_vif(X, feature_names)
    
    # Fit models and compare
    model_full, model_reduced, r2_full, r2_reduced = fit_models_and_compare(X, y, feature_names, true_coefs)
    
    # Visualize results
    visualize_collinearity(X, feature_names, corr_df, vif_values)
    
    # Demonstrate collinearity effects
    demonstrate_collinearity_effects()
    
    # Suggest remedies
    suggest_remedies(vif_values, feature_names, high_corr_pairs)
