import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X).flatten()
y_noisy = y_true + 0.3 * np.random.randn(100)

# Fit polynomials of different degrees
degrees = [1, 3, 5, 10, 15]
models = []
predictions = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y_noisy)
    
    models.append(model)
    predictions.append(model.predict(X_poly))

# Plot results
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees):
    plt.subplot(2, 3, i+1)
    plt.scatter(X, y_noisy, alpha=0.5, label='Data')
    plt.plot(X, y_true, 'g-', label='True Function', linewidth=2)
    plt.plot(X, predictions[i], 'r-', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
