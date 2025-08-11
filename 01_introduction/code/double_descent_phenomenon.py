import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Demonstrate double descent with linear regression
n_samples = 50
n_features_range = range(10, 200, 10)
test_errors = []

for n_features in n_features_range:
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate test error
    y_pred = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred)
    test_errors.append(test_error)

plt.figure(figsize=(10, 6))
plt.plot(n_features_range, test_errors, 'b-', linewidth=2)
plt.axvline(x=n_samples, color='r', linestyle='--', label='n = p')
plt.xlabel('Number of Features')
plt.ylabel('Test Error')
plt.title('Double Descent Phenomenon')
plt.legend()
plt.grid(True)
plt.show()
