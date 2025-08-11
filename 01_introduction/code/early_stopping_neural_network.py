import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.randn(100, 20)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)

# Split data into train, validation, and test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# Train with early stopping
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

mlp.fit(X_train, y_train)
print(f"Best validation score: {mlp.best_validation_score_:.4f}")
print(f"Number of iterations: {mlp.n_iter_}")
