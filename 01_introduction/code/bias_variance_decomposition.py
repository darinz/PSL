import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Calculate bias and variance for different polynomial degrees
def calculate_bias_variance(X, y_true, y_noisy, degrees):
    bias_squared = []
    variance = []
    total_error = []
    
    for degree in degrees:
        # Generate multiple datasets by adding noise
        predictions = []
        for _ in range(100):
            y_sample = y_true + 0.3 * np.random.randn(len(y_true))
            
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y_sample)
            pred = model.predict(X_poly)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias^2
        mean_pred = np.mean(predictions, axis=0)
        bias_sq = np.mean((mean_pred - y_true)**2)
        
        # Calculate variance
        var = np.mean(np.var(predictions, axis=0))
        
        # Calculate total error
        total = bias_sq + var
        
        bias_squared.append(bias_sq)
        variance.append(var)
        total_error.append(total)
    
    return bias_squared, variance, total_error

# Generate data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X).flatten()
y_noisy = y_true + 0.3 * np.random.randn(100)

degrees = range(1, 16)
bias_sq, var, total = calculate_bias_variance(X, y_true, y_noisy, degrees)

# Plot bias-variance decomposition
plt.figure(figsize=(12, 8))
plt.plot(degrees, bias_sq, 'b-', label='Bias²', linewidth=2)
plt.plot(degrees, var, 'r-', label='Variance', linewidth=2)
plt.plot(degrees, total, 'g-', label='Total Error', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('Error')
plt.title('Bias-Variance Decomposition')
plt.legend()
plt.grid(True)
plt.show()
