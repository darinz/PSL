import numpy as np

def mixture_bayes_classifier(x, mu0_list, mu1_list, sigma2, p, weights):
    """
    Compute Bayes classifier for mixture Gaussian case
    
    Parameters:
    x: input point (2D array)
    mu0_list: list of means for class 0 components
    mu1_list: list of means for class 1 components
    sigma2: variance (scalar)
    p: prior probability of class 1
    weights: mixture weights
    
    Returns:
    prob: probability of class 1
    decision: predicted class (0 or 1)
    """
    # Compute mixture densities
    f0 = sum(w * np.exp(-0.5 * np.sum((x - mu)**2) / sigma2) 
             for w, mu in zip(weights, mu0_list)) / (2*np.pi*sigma2)
    f1 = sum(w * np.exp(-0.5 * np.sum((x - mu)**2) / sigma2) 
             for w, mu in zip(weights, mu1_list)) / (2*np.pi*sigma2)
    
    # Apply Bayes' theorem
    numerator = p * f1
    denominator = p * f1 + (1-p) * f0
    
    prob = numerator / denominator
    decision = 1 if prob > 0.5 else 0
    
    return prob, decision

def generate_mixture_parameters(n_components=10):
    """Generate random mixture parameters"""
    np.random.seed(42)
    mu0_list = [np.random.randn(2) for _ in range(n_components)]
    mu1_list = [np.random.randn(2) + np.array([2, 2]) for _ in range(n_components)]
    weights = np.random.dirichlet(np.ones(n_components))
    return mu0_list, mu1_list, weights

# Example usage for mixture case
mu0_list, mu1_list, weights = generate_mixture_parameters()
x_test = np.array([1, 1])
prob, decision = mixture_bayes_classifier(x_test, mu0_list, mu1_list, 1.0, 0.5, weights)
print(f"Mixture probability of class 1: {prob:.3f}")
print(f"Predicted class: {decision}")
