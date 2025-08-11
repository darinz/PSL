import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def bayes_classifier(x, mu0, mu1, sigma2, p):
    """
    Compute Bayes classifier for simple Gaussian case
    
    Parameters:
    x: input point (2D array)
    mu0: mean of class 0 (2D array)
    mu1: mean of class 1 (2D array)
    sigma2: variance (scalar)
    p: prior probability of class 1
    
    Returns:
    prob: probability of class 1
    decision: predicted class (0 or 1)
    """
    # Compute class-conditional densities
    f0 = np.exp(-0.5 * np.sum((x - mu0)**2) / sigma2) / (2*np.pi*sigma2)
    f1 = np.exp(-0.5 * np.sum((x - mu1)**2) / sigma2) / (2*np.pi*sigma2)
    
    # Apply Bayes' theorem
    numerator = p * f1
    denominator = p * f1 + (1-p) * f0
    
    # Return probability and decision
    prob = numerator / denominator
    decision = 1 if prob > 0.5 else 0
    
    return prob, decision

# Example usage
mu0 = np.array([0, 0])
mu1 = np.array([2, 2])
sigma2 = 1.0
p = 0.5

# Test point
x_test = np.array([1, 1])
prob, decision = bayes_classifier(x_test, mu0, mu1, sigma2, p)
print(f"Probability of class 1: {prob:.3f}")
print(f"Predicted class: {decision}")
