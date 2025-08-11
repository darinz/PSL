from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def evaluate_linear_regression(X_train, y_train, X_test, y_test):
    """Evaluate linear regression for classification"""
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    train_pred_proba = lr.predict(X_train)
    test_pred_proba = lr.predict(X_test)
    
    # Convert to binary predictions
    train_pred = (train_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    # Calculate errors
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)
    
    return train_error, test_error

# Generate example data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Example usage
train_error, test_error = evaluate_linear_regression(X_train, y_train, X_test, y_test)

print(f"Linear Regression Results:")
print(f"Training Error: {train_error:.4f}")
print(f"Test Error: {test_error:.4f}")
