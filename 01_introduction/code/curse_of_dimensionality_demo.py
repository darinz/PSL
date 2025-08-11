import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
def generate_data(n_samples, n_features, n_classes=2):
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

# Test kNN performance across dimensions
n_samples = 1000
n_features_list = [1, 2, 5, 10, 20, 50, 100]
accuracies = []

for n_features in n_features_list:
    X, y = generate_data(n_samples, n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(n_features_list, accuracies, 'bo-')
plt.xlabel('Number of Features')
plt.ylabel('Test Accuracy')
plt.title('kNN Performance vs. Dimensionality')
plt.grid(True)
plt.show()
