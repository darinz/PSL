import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def demonstrate_error_decomposition():
    """Demonstrate error decomposition in variable selection"""
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data with known true function
    n = 100
    p_max = 20
    X = np.random.randn(n, p_max)

    # True model: only first 5 variables matter
    beta_true = np.zeros(p_max)
    beta_true[:5] = [1.5, -0.8, 0.6, -0.4, 0.3]
    f_true = X @ beta_true
    y = f_true + np.random.normal(0, 0.5, n)

    # Function to calculate errors for different model sizes
    def calculate_errors(X, y, p_values):
        train_errors = []
        test_errors = []
        
        for p in p_values:
            # Use only first p predictors
            X_p = X[:, :p]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_p, y, test_size=0.3, random_state=42
            )
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate errors
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        
        return np.array(train_errors), np.array(test_errors)

    # Calculate errors for different model sizes
    p_values = range(1, p_max + 1)
    train_errors, test_errors = calculate_errors(X, y, p_values)

    # Theoretical decomposition
    sigma2_est = np.var(y - f_true)  # Estimate of sigma^2
    unavoidable_error = sigma2_est
    bias_squared = np.array([np.mean((f_true - X[:, :p] @ beta_true[:p])**2) for p in p_values])
    dimensional_error = np.array([p * sigma2_est for p in p_values])

    # Expected errors
    expected_train_error = unavoidable_error - dimensional_error + bias_squared
    expected_test_error = unavoidable_error + dimensional_error + bias_squared

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Observed errors
    axes[0, 0].plot(p_values, train_errors, 'bo-', label='Observed Train Error', linewidth=2)
    axes[0, 0].plot(p_values, test_errors, 'ro-', label='Observed Test Error', linewidth=2)
    axes[0, 0].set_xlabel('Number of Predictors (p)')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].set_title('Observed Training vs Test Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='True Model Size')

    # Expected errors
    axes[0, 1].plot(p_values, expected_train_error, 'b--', label='Expected Train Error', linewidth=2)
    axes[0, 1].plot(p_values, expected_test_error, 'r--', label='Expected Test Error', linewidth=2)
    axes[0, 1].set_xlabel('Number of Predictors (p)')
    axes[0, 1].set_ylabel('Expected Error')
    axes[0, 1].set_title('Theoretical Error Decomposition')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='True Model Size')

    # Error components
    axes[1, 0].plot(p_values, unavoidable_error * np.ones_like(p_values), 'g-', 
                    label='Unavoidable Error', linewidth=2)
    axes[1, 0].plot(p_values, bias_squared, 'm-', label='Bias²', linewidth=2)
    axes[1, 0].plot(p_values, dimensional_error, 'c-', label='Dimensional Error', linewidth=2)
    axes[1, 0].set_xlabel('Number of Predictors (p)')
    axes[1, 0].set_ylabel('Error Component')
    axes[1, 0].set_title('Error Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Overfitting demonstration
    axes[1, 1].plot(p_values, test_errors - train_errors, 'ko-', linewidth=2)
    axes[1, 1].set_xlabel('Number of Predictors (p)')
    axes[1, 1].set_ylabel('Test Error - Train Error')
    axes[1, 1].set_title('Overfitting Gap')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Print key insights
    print("=== ERROR DECOMPOSITION INSIGHTS ===")
    print(f"Unavoidable Error: {unavoidable_error:.4f}")
    print(f"Optimal model size (observed): {p_values[np.argmin(test_errors)]}")
    print(f"Optimal model size (theoretical): {p_values[np.argmin(expected_test_error)]}")
    print(f"True model size: 5")

    print(f"\nOverfitting gap at p={p_max}: {test_errors[-1] - train_errors[-1]:.4f}")
    print(f"Theoretical gap: {2 * p_max * sigma2_est:.4f}")
    
    return train_errors, test_errors, p_values, beta_true

# Run demonstration
train_errors, test_errors, p_values, beta_true = demonstrate_error_decomposition()
