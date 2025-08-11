from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def evaluate_knn(X_train, y_train, X_test, y_test, k_values):
    """Evaluate kNN for different k values"""
    train_errors = []
    test_errors = []
    
    for k in k_values:
        # Fit kNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Calculate errors
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)
        
        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return train_errors, test_errors

# Generate example data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Example usage
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201]
train_errors, test_errors = evaluate_knn(X_train, y_train, X_test, y_test, k_values)

# Print results
print("kNN Evaluation Results:")
for k, train_err, test_err in zip(k_values, train_errors, test_errors):
    print(f"k={k:3d}: Train Error={train_err:.4f}, Test Error={test_err:.4f}")
