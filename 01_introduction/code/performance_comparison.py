import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def plot_performance_comparison(k_values, knn_train_errors, knn_test_errors, 
                               lr_train_error, lr_test_error):
    """Plot performance comparison between kNN and linear regression"""
    plt.figure(figsize=(12, 5))
    
    # kNN performance
    plt.subplot(1, 2, 1)
    plt.plot(k_values, knn_train_errors, 'b-', label='Training Error', linewidth=2)
    plt.plot(k_values, knn_test_errors, 'r-', label='Test Error', linewidth=2)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Error Rate')
    plt.title('kNN Performance')
    plt.legend()
    plt.grid(True)
    
    # Linear regression performance
    plt.subplot(1, 2, 2)
    plt.bar(['Training', 'Test'], [lr_train_error, lr_test_error], 
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Error Rate')
    plt.title('Linear Regression Performance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Generate example data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate kNN
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101]
knn_train_errors = []
knn_test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    
    knn_train_errors.append(1 - accuracy_score(y_train, train_pred))
    knn_test_errors.append(1 - accuracy_score(y_test, test_pred))

# Evaluate linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

train_pred_proba = lr.predict(X_train)
test_pred_proba = lr.predict(X_test)

train_pred = (train_pred_proba > 0.5).astype(int)
test_pred = (test_pred_proba > 0.5).astype(int)

lr_train_error = 1 - accuracy_score(y_train, train_pred)
lr_test_error = 1 - accuracy_score(y_test, test_pred)

# Plot comparison
plot_performance_comparison(k_values, knn_train_errors, knn_test_errors, 
                           lr_train_error, lr_test_error)
