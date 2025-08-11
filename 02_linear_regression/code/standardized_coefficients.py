import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def demonstrate_standardized_coefficients():
    """Demonstrate standardized vs unstandardized coefficients"""
    
    # Generate data
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, n)  # Standard normal
    X2 = np.random.normal(0, 10, n)  # Different scale
    y = 2 + 1.5 * X1 - 0.8 * X2 + np.random.normal(0, 0.5, n)

    # Unstandardized coefficients
    X = np.column_stack([X1, X2])
    model = LinearRegression()
    model.fit(X, y)

    print("Unstandardized coefficients:")
    for i, coef in enumerate(model.coef_):
        print(f"β{i+1} = {coef:.3f}")

    # Standardized coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_scaled = LinearRegression()
    model_scaled.fit(X_scaled, y)

    print("\nStandardized coefficients:")
    for i, coef in enumerate(model_scaled.coef_):
        print(f"β*{i+1} = {coef:.3f}")

    # Manual calculation of standardized coefficients
    y_std = np.std(y)
    X_stds = np.std(X, axis=0)
    standardized_coefs = model.coef_ * X_stds / y_std
    print("\nManually calculated standardized coefficients:")
    for i, coef in enumerate(standardized_coefs):
        print(f"β*{i+1} = {coef:.3f}")
    
    return model, model_scaled, standardized_coefs

# Run demonstration
model, model_scaled, standardized_coefs = demonstrate_standardized_coefficients()
