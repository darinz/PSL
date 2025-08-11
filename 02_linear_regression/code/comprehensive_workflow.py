import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from scipy import stats

def comprehensive_linear_regression_workflow():
    """Demonstrate a comprehensive linear regression workflow"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate realistic sample data with known relationships
    n = 200
    X1 = np.random.normal(0, 1, n)  # Predictor 1
    X2 = 0.3 * X1 + np.random.normal(0, 0.9, n)  # Correlated predictor
    X3 = np.random.normal(0, 1, n)  # Independent predictor

    # True model with some noise
    y_true = 2.5 + 1.8 * X1 - 0.6 * X2 + 0.4 * X3
    noise = np.random.normal(0, 0.8, n)
    y = y_true + noise

    # Create design matrix
    X = np.column_stack([X1, X2, X3])
    feature_names = ['X1', 'X2', 'X3']

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit scikit-learn model
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)

    # Fit statsmodels for detailed statistics
    X_train_sm = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_train_sm).fit()

    # Generate predictions
    y_train_pred = sk_model.predict(X_train)
    y_test_pred = sk_model.predict(X_test)

    # Model evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Cross-validation
    cv_scores = cross_val_score(sk_model, X_train, y_train, cv=5, scoring='r2')

    print("=== LINEAR REGRESSION MODEL RESULTS ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print("\n=== COEFFICIENT ESTIMATES ===")
    print("scikit-learn results:")
    for name, coef in zip(feature_names, sk_model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {sk_model.intercept_:.4f}")

    print("\nstatsmodels detailed results:")
    print(sm_model.summary())

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals vs Predicted
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    axes[0, 1].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix of Predictors')
    plt.show()
    
    return sk_model, sm_model, (X_train, X_test, y_train, y_test)

# Run the comprehensive workflow
sk_model, sm_model, data = comprehensive_linear_regression_workflow()
