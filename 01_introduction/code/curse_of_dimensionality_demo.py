import numpy as np
import matplotlib.pyplot as plt

# Demonstrate curse of dimensionality
n_samples = 1000
dimensions = [1, 2, 5, 10, 20, 50, 100]
distances = []

for p in dimensions:
    X = np.random.randn(n_samples, p)
    
    # Calculate distances from first point to all others
    dists = np.sqrt(np.sum((X - X[0])**2, axis=1))
    
    # Calculate coefficient of variation
    cv = np.std(dists) / np.mean(dists)
    distances.append(cv)

plt.figure(figsize=(10, 6))
plt.plot(dimensions, distances, 'bo-', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('Coefficient of Variation of Distances')
plt.title('Curse of Dimensionality: Distance Concentration')
plt.grid(True)
plt.show()

# Print results
for p, cv in zip(dimensions, distances):
    print(f"Dimensions: {p}, CV: {cv:.4f}")
