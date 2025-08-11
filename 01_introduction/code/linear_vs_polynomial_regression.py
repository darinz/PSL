import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 0.5 * X.flatten()**2 + np.random.normal(0, 1, 100)

# Fit linear regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Fit polynomial regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly = poly_model.predict(X_poly)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_linear, 'r-', linewidth=2, label='Linear Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression on Non-linear Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_poly, 'g-', linewidth=2, label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
