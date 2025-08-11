import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate mixture data
np.random.seed(42)
n_samples = 1000

# Component parameters
means = [[0, 0], [4, 4], [0, 4]]
covariances = [np.eye(2), np.eye(2), np.eye(2)]
weights = [0.4, 0.3, 0.3]

# Generate data
X = np.zeros((n_samples, 2))
for i in range(n_samples):
    # Choose component
    component = np.random.choice(3, p=weights)
    # Generate from chosen component
    X[i] = np.random.multivariate_normal(means[component], covariances[component])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Generated Mixture Data')
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1, 2, 2)
# Generate points for contour plot
x = np.linspace(-2, 6, 100)
y = np.linspace(-2, 6, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XY = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

# Calculate density
density = np.exp(gmm.score_samples(XY))
density = density.reshape(X_grid.shape)

plt.contour(X_grid, Y_grid, density, levels=20)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Fitted Mixture Model')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()
