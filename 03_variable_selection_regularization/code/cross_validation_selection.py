import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def demonstrate_cross_validation_selection():
    """Demonstrate cross-validation for regularization parameter selection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    n, p = 100, 20
    X = np.random.randn(n, p)
    true_beta = np.zeros(p)
    true_beta[:5] = [2, -1.5, 1, -0.8, 0.6]  # Only first 5 coefficients are non-zero
    y = X @ true_beta + 0.5 * np.random.randn(n)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # 1. Ridge with cross-validation
    print("=== RIDGE REGRESSION WITH CROSS-VALIDATION ===")
    
    ridge_cv = GridSearchCV(Ridge(), 
                           param_grid={'alpha': np.logspace(-3, 3, 50)},
                           cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train_scaled)

    print(f"Best Ridge α: {ridge_cv.best_params_['alpha']:.4f}")
    print(f"Best CV score: {-ridge_cv.best_score_:.4f}")

    # 2. Lasso with cross-validation
    print("\n=== LASSO REGRESSION WITH CROSS-VALIDATION ===")
    
    lasso_cv = GridSearchCV(Lasso(max_iter=2000), 
                           param_grid={'alpha': np.logspace(-3, 3, 50)},
                           cv=5, scoring='neg_mean_squared_error')
    lasso_cv.fit(X_train_scaled, y_train_scaled)

    print(f"Best Lasso α: {lasso_cv.best_params_['alpha']:.4f}")
    print(f"Best CV score: {-lasso_cv.best_score_:.4f}")

    # 3. Compare models
    print("\n=== MODEL COMPARISON ===")
    
    # Fit best models
    best_ridge = Ridge(alpha=ridge_cv.best_params_['alpha'])
    best_lasso = Lasso(alpha=lasso_cv.best_params_['alpha'], max_iter=2000)

    best_ridge.fit(X_train_scaled, y_train_scaled)
    best_lasso.fit(X_train_scaled, y_train_scaled)

    # Transform coefficients back to original scale
    ridge_coef_original = best_ridge.coef_ * scaler_y.scale_ / scaler_X.scale_
    lasso_coef_original = best_lasso.coef_ * scaler_y.scale_ / scaler_X.scale_

    ridge_intercept = scaler_y.mean_ - np.sum(ridge_coef_original * scaler_X.mean_)
    lasso_intercept = scaler_y.mean_ - np.sum(lasso_coef_original * scaler_X.mean_)

    # Test performance
    ridge_pred = ridge_intercept + X_test @ ridge_coef_original
    lasso_pred = lasso_intercept + X_test @ lasso_coef_original

    ridge_mse = mean_squared_error(y_test, ridge_pred)
    lasso_mse = mean_squared_error(y_test, lasso_pred)

    ridge_r2 = r2_score(y_test, ridge_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)

    print("Ridge Regression Results:")
    print(f"  Test MSE: {ridge_mse:.4f}")
    print(f"  Test R²: {ridge_r2:.4f}")
    print(f"  Non-zero coefficients: {np.sum(ridge_coef_original != 0)}")

    print("\nLasso Regression Results:")
    print(f"  Test MSE: {lasso_mse:.4f}")
    print(f"  Test R²: {lasso_r2:.4f}")
    print(f"  Non-zero coefficients: {np.sum(lasso_coef_original != 0)}")

    # 4. Information Criteria
    print("\n=== INFORMATION CRITERIA ===")
    
    def calculate_aic_bic(X, y, coef, intercept):
        """Calculate AIC and BIC for a model"""
        y_pred = intercept + X @ coef
        rss = np.sum((y - y_pred)**2)
        p = np.sum(coef != 0)  # Number of non-zero coefficients
        
        aic = n * np.log(rss/n) + 2*p
        bic = n * np.log(rss/n) + np.log(n)*p
        
        return aic, bic

    ridge_aic, ridge_bic = calculate_aic_bic(X_test, y_test, ridge_coef_original, ridge_intercept)
    lasso_aic, lasso_bic = calculate_aic_bic(X_test, y_test, lasso_coef_original, lasso_intercept)

    print("Ridge:")
    print(f"  AIC: {ridge_aic:.2f}")
    print(f"  BIC: {ridge_bic:.2f}")

    print("\nLasso:")
    print(f"  AIC: {lasso_aic:.2f}")
    print(f"  BIC: {lasso_bic:.2f}")

    # 5. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # CV scores vs alpha
    alphas = np.logspace(-3, 3, 50)
    
    ridge_scores = -ridge_cv.cv_results_['mean_test_score']
    lasso_scores = -lasso_cv.cv_results_['mean_test_score']

    axes[0, 0].semilogx(alphas, ridge_scores, 'b-', label='Ridge', linewidth=2)
    axes[0, 0].semilogx(alphas, lasso_scores, 'r-', label='Lasso', linewidth=2)
    axes[0, 0].axvline(x=ridge_cv.best_params_['alpha'], color='blue', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=lasso_cv.best_params_['alpha'], color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Regularization Parameter (α)')
    axes[0, 0].set_ylabel('Cross-Validation MSE')
    axes[0, 0].set_title('Cross-Validation Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Coefficient comparison
    x_pos = np.arange(p)
    width = 0.35

    axes[0, 1].bar(x_pos - width/2, ridge_coef_original, width, label='Ridge', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, lasso_coef_original, width, label='Lasso', alpha=0.7)
    axes[0, 1].set_xlabel('Variable Index')
    axes[0, 1].set_ylabel('Coefficient Value')
    axes[0, 1].set_title('Coefficient Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # True vs predicted
    axes[1, 0].scatter(y_test, ridge_pred, alpha=0.6, label=f'Ridge (R²={ridge_r2:.3f})')
    axes[1, 0].scatter(y_test, lasso_pred, alpha=0.6, label=f'Lasso (R²={lasso_r2:.3f})')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.7)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title('True vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Model comparison summary
    comparison_data = {
        'Metric': ['MSE', 'R²', 'Non-zero Coefs', 'AIC', 'BIC'],
        'Ridge': [ridge_mse, ridge_r2, np.sum(ridge_coef_original != 0), ridge_aic, ridge_bic],
        'Lasso': [lasso_mse, lasso_r2, np.sum(lasso_coef_original != 0), lasso_aic, lasso_bic]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create a table plot
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=comparison_df.values, 
                            colLabels=comparison_df.columns,
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Model Comparison Summary')

    plt.tight_layout()
    plt.show()

    # Key insights
    print("\n=== KEY INSIGHTS ===")
    print("1. Cross-validation helps select optimal regularization parameters")
    print("2. Ridge regression keeps all variables but shrinks coefficients")
    print("3. Lasso regression performs automatic variable selection")
    print("4. Information criteria provide additional model comparison metrics")
    print("5. Standardization is crucial for fair comparison between methods")
    
    return ridge_cv, lasso_cv, ridge_coef_original, lasso_coef_original, comparison_df

# Run demonstration
ridge_cv, lasso_cv, ridge_coef_original, lasso_coef_original, comparison_df = demonstrate_cross_validation_selection()
