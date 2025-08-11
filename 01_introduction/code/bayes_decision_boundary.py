import numpy as np
import matplotlib.pyplot as plt

def bayes_classifier(x, mu0, mu1, sigma2, p):
    """Compute Bayes classifier for simple Gaussian case"""
    f0 = np.exp(-0.5 * np.sum((x - mu0)**2) / sigma2) / (2*np.pi*sigma2)
    f1 = np.exp(-0.5 * np.sum((x - mu1)**2) / sigma2) / (2*np.pi*sigma2)
    
    numerator = p * f1
    denominator = p * f1 + (1-p) * f0
    
    prob = numerator / denominator
    decision = 1 if prob > 0.5 else 0
    
    return prob, decision

def plot_bayes_decision_boundary(mu0, mu1, sigma2, p, title="Bayes Decision Boundary"):
    """Plot the Bayes decision boundary for Example 1"""
    # Create grid
    x_min, x_max = -3, 5
    y_min, y_max = -3, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Compute decisions for each grid point
    decisions = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x = np.array([xx[i,j], yy[i,j]])
            _, decision = bayes_classifier(x, mu0, mu1, sigma2, p)
            decisions[i,j] = decision
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, decisions, alpha=0.3, levels=[0, 0.5, 1])
    plt.contour(xx, yy, decisions, levels=[0.5], colors='red', linewidths=2)
    
    # Plot class means
    plt.plot(mu0[0], mu0[1], 'bo', markersize=10, label='Class 0 Mean')
    plt.plot(mu1[0], mu1[1], 'ro', markersize=10, label='Class 1 Mean')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage
mu0 = np.array([0, 0])
mu1 = np.array([2, 2])
sigma2 = 1.0
p = 0.5

# Plot decision boundary
plot_bayes_decision_boundary(mu0, mu1, sigma2, p)
