"""
Mixture Models Implementation
============================

This module provides comprehensive implementations of mixture models,
including visualization, KL divergence computation, and the EM algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.integrate import quad
from sklearn.mixture import GaussianMixture
import seaborn as sns

def visualize_mixture_model():
    """
    Visualize a two-component Gaussian mixture model.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters for a two-component Gaussian mixture
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 4, 1.5
    pi1 = 0.6
    pi2 = 1 - pi1

    # Generate data
    n_samples = 1000
    z = np.random.choice([0, 1], size=n_samples, p=[pi1, pi2])
    x = np.zeros(n_samples)

    x[z == 0] = np.random.normal(mu1, sigma1, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(mu2, sigma2, size=np.sum(z == 1))

    # Plot the mixture
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram of data
    ax1.hist(x, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Histogram of Mixture Data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')

    # True mixture density
    x_range = np.linspace(-3, 8, 1000)
    true_density = pi1 * norm.pdf(x_range, mu1, sigma1) + pi2 * norm.pdf(x_range, mu2, sigma2)
    ax1.plot(x_range, true_density, 'r-', linewidth=2, label='True Mixture Density')
    ax1.legend()

    # Individual components
    ax2.plot(x_range, pi1 * norm.pdf(x_range, mu1, sigma1), 'b--', 
             label=f'Component 1 (π={pi1:.1f})')
    ax2.plot(x_range, pi2 * norm.pdf(x_range, mu2, sigma2), 'g--', 
             label=f'Component 2 (π={pi2:.1f})')
    ax2.plot(x_range, true_density, 'r-', linewidth=2, label='Mixture')
    ax2.set_title('Mixture Components')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Generated {n_samples} samples from mixture model")
    print(f"Component 1: μ={mu1}, σ={sigma1}, π={pi1}")
    print(f"Component 2: μ={mu2}, σ={sigma2}, π={pi2}")

class TwoComponentGaussianMixture:
    """
    Two-component Gaussian mixture model implementation.
    """
    
    def __init__(self, mu1=0, mu2=4, sigma1=1, sigma2=1.5, pi=0.6):
        """
        Initialize the two-component Gaussian mixture model.
        
        Parameters:
        -----------
        mu1, mu2 : float
            Means of the two Gaussian components
        sigma1, sigma2 : float
            Standard deviations of the two Gaussian components
        pi : float
            Mixing weight for the first component
        """
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        
    def generate_data(self, n_samples=1000):
        """
        Generate data from the mixture model.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        x : array
            Generated data
        z : array
            Latent component assignments
        """
        z = np.random.choice([0, 1], size=n_samples, p=[self.pi, 1-self.pi])
        x = np.zeros(n_samples)
        
        x[z == 0] = np.random.normal(self.mu1, self.sigma1, size=np.sum(z == 0))
        x[z == 1] = np.random.normal(self.mu2, self.sigma2, size=np.sum(z == 1))
        
        return x, z
    
    def pdf(self, x):
        """
        Compute the probability density function.
        
        Parameters:
        -----------
        x : array
            Points at which to evaluate the PDF
            
        Returns:
        --------
        density : array
            Probability density values
        """
        return (self.pi * norm.pdf(x, self.mu1, self.sigma1) + 
                (1-self.pi) * norm.pdf(x, self.mu2, self.sigma2))
    
    def plot_mixture(self, x, z=None):
        """
        Plot the mixture model and data.
        
        Parameters:
        -----------
        x : array
            Data to plot
        z : array, optional
            True component assignments
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(x, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        x_range = np.linspace(x.min()-1, x.max()+1, 1000)
        ax1.plot(x_range, self.pdf(x_range), 'r-', linewidth=2, label='True Mixture')
        ax1.set_title('Data and True Mixture Density')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Components
        ax2.plot(x_range, self.pi * norm.pdf(x_range, self.mu1, self.sigma1), 
                'b--', label=f'Component 1 (π={self.pi:.1f})')
        ax2.plot(x_range, (1-self.pi) * norm.pdf(x_range, self.mu2, self.sigma2), 
                'g--', label=f'Component 2 (π={1-self.pi:.1f})')
        ax2.plot(x_range, self.pdf(x_range), 'r-', linewidth=2, label='Mixture')
        ax2.set_title('Mixture Components')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def demonstrate_two_component_mixture():
    """
    Demonstrate the two-component Gaussian mixture model.
    """
    # Example usage
    np.random.seed(42)
    gmm = TwoComponentGaussianMixture()
    x, z = gmm.generate_data(1000)
    gmm.plot_mixture(x, z)

    # Fit using sklearn for comparison
    sklearn_gmm = GaussianMixture(n_components=2, random_state=42)
    sklearn_gmm.fit(x.reshape(-1, 1))

    print("True parameters:")
    print(f"μ1={gmm.mu1}, μ2={gmm.mu2}, σ1={gmm.sigma1}, σ2={gmm.sigma2}, π={gmm.pi}")
    print("\nSklearn estimated parameters:")
    print(f"μ1={sklearn_gmm.means_[0,0]:.3f}, μ2={sklearn_gmm.means_[1,0]:.3f}")
    print(f"σ1={np.sqrt(sklearn_gmm.covariances_[0,0,0]):.3f}, σ2={np.sqrt(sklearn_gmm.covariances_[1,0,0]):.3f}")
    print(f"π1={sklearn_gmm.weights_[0]:.3f}, π2={sklearn_gmm.weights_[1]:.3f}")

def kl_divergence_discrete(p, q):
    """
    Compute KL divergence for discrete distributions.
    
    Parameters:
    -----------
    p, q : array
        Probability mass functions
        
    Returns:
    --------
    kl_div : float
        KL divergence KL(p||q)
    """
    # Ensure probabilities sum to 1
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    return np.sum(p * np.log(p / q))

def kl_divergence_continuous(p_func, q_func, x_range):
    """
    Compute KL divergence for continuous distributions using numerical integration.
    
    Parameters:
    -----------
    p_func, q_func : callable
        Probability density functions
    x_range : tuple
        Range of integration (min, max)
        
    Returns:
    --------
    kl_div : float
        KL divergence KL(p||q)
    """
    def integrand(x):
        p_val = p_func(x)
        q_val = q_func(x)
        # Avoid log(0)
        if p_val > 0 and q_val > 0:
            return p_val * np.log(p_val / q_val)
        return 0
    
    result, _ = quad(integrand, x_range[0], x_range[-1])
    return result

def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two Gaussian distributions.
    
    Parameters:
    -----------
    mu1, mu2 : float
        Means of the two Gaussians
    sigma1, sigma2 : float
        Standard deviations of the two Gaussians
        
    Returns:
    --------
    kl_div : float
        KL divergence KL(N(mu1,sigma1)||N(mu2,sigma2))
    """
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5)

def demonstrate_kl_divergence():
    """
    Demonstrate KL divergence computation.
    """
    # Example: KL divergence between Gaussians
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 1, 1.5

    # Analytical solution
    kl_analytical = kl_divergence_gaussian(mu1, sigma1, mu2, sigma2)

    # Numerical solution
    def gaussian_pdf(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    kl_numerical = kl_divergence_continuous(
        lambda x: gaussian_pdf(x, mu1, sigma1),
        lambda x: gaussian_pdf(x, mu2, sigma2),
        [-5, 5]
    )

    print(f"KL divergence between N({mu1},{sigma1}) and N({mu2},{sigma2})")
    print(f"Analytical: {kl_analytical:.6f}")
    print(f"Numerical:  {kl_numerical:.6f}")

    # Visualize the distributions
    x = np.linspace(-4, 6, 1000)
    p1 = gaussian_pdf(x, mu1, sigma1)
    p2 = gaussian_pdf(x, mu2, sigma2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, p1, 'b-', label=f'N({mu1},{sigma1})')
    plt.plot(x, p2, 'r-', label=f'N({mu2},{sigma2})')
    plt.fill_between(x, p1, p2, alpha=0.3, color='gray')
    plt.title(f'Gaussian Distributions (KL divergence: {kl_analytical:.4f})')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

class EMGaussianMixture:
    """
    Gaussian mixture model fitted using the EM algorithm.
    """
    
    def __init__(self, n_components=2, max_iter=100, tol=1e-6):
        """
        Initialize the EM Gaussian mixture model.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.responsibilities_ = None
        
    def fit(self, X):
        """
        Fit the Gaussian mixture model using EM algorithm.
        
        Parameters:
        -----------
        X : array
            Training data
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples = len(X)
        
        # Initialize parameters randomly
        self._initialize_parameters(X)
        
        log_likelihoods = []
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            self.responsibilities_ = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if len(log_likelihoods) > 1:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        n_samples = len(X)
        
        # Random means
        self.means_ = np.random.choice(X, size=self.n_components, replace=False)
        
        # Random covariances
        self.covariances_ = np.array([np.var(X)] * self.n_components)
        
        # Random weights
        self.weights_ = np.random.dirichlet(np.ones(self.n_components))
    
    def _e_step(self, X):
        """E-step: Compute responsibilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = (self.weights_[k] * 
                                   norm.pdf(X, self.means_[k], np.sqrt(self.covariances_[k])))
        
        # Normalize
        row_sums = responsibilities.sum(axis=1)
        responsibilities = responsibilities / row_sums[:, np.newaxis]
        
        return responsibilities
    
    def _m_step(self, X):
        """M-step: Update parameters"""
        n_samples = len(X)
        
        for k in range(self.n_components):
            # Update weights
            self.weights_[k] = np.mean(self.responsibilities_[:, k])
            
            # Update means
            self.means_[k] = (np.sum(self.responsibilities_[:, k] * X) / 
                             np.sum(self.responsibilities_[:, k]))
            
            # Update covariances
            self.covariances_[k] = (np.sum(self.responsibilities_[:, k] * 
                                          (X - self.means_[k])**2) / 
                                   np.sum(self.responsibilities_[:, k]))
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood"""
        likelihood = np.zeros(len(X))
        
        for k in range(self.n_components):
            likelihood += (self.weights_[k] * 
                         norm.pdf(X, self.means_[k], np.sqrt(self.covariances_[k])))
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def predict_proba(self, X):
        """Predict component probabilities"""
        return self._e_step(X)
    
    def predict(self, X):
        """Predict component assignments"""
        return np.argmax(self.predict_proba(X), axis=1)

def demonstrate_em_algorithm():
    """
    Demonstrate the EM algorithm for fitting Gaussian mixture models.
    """
    np.random.seed(42)

    # Generate data from true mixture
    true_gmm = TwoComponentGaussianMixture()
    X, true_z = true_gmm.generate_data(1000)

    # Fit using EM
    em_gmm = EMGaussianMixture(n_components=2)
    em_gmm.fit(X)

    # Compare results
    print("True parameters:")
    print(f"μ1={true_gmm.mu1}, μ2={true_gmm.mu2}")
    print(f"σ1={true_gmm.sigma1}, σ2={true_gmm.sigma2}")
    print(f"π={true_gmm.pi}")

    print("\nEM estimated parameters:")
    print(f"μ1={em_gmm.means_[0]:.3f}, μ2={em_gmm.means_[1]:.3f}")
    print(f"σ1={np.sqrt(em_gmm.covariances_[0]):.3f}, σ2={np.sqrt(em_gmm.covariances_[1]):.3f}")
    print(f"π1={em_gmm.weights_[0]:.3f}, π2={em_gmm.weights_[1]:.3f}")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Data and fitted mixture
    ax1.hist(X, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x_range = np.linspace(X.min()-1, X.max()+1, 1000)

    # True mixture
    true_density = true_gmm.pdf(x_range)
    ax1.plot(x_range, true_density, 'r-', linewidth=2, label='True Mixture')

    # Fitted mixture
    fitted_density = (em_gmm.weights_[0] * norm.pdf(x_range, em_gmm.means_[0], np.sqrt(em_gmm.covariances_[0])) +
                     em_gmm.weights_[1] * norm.pdf(x_range, em_gmm.means_[1], np.sqrt(em_gmm.covariances_[1])))
    ax1.plot(x_range, fitted_density, 'g--', linewidth=2, label='Fitted Mixture')

    ax1.set_title('Data and Mixture Densities')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Responsibilities
    predicted_z = em_gmm.predict(X)
    ax2.scatter(X, em_gmm.responsibilities_[:, 0], alpha=0.6, s=20)
    ax2.set_title('Responsibilities (Component 1)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('P(Z=1|x)')

    plt.tight_layout()
    plt.show()

def fit_multiple_initializations(X, n_components=2, n_init=10):
    """
    Fit GMM with multiple initializations and return the best result.
    
    Parameters:
    -----------
    X : array
        Training data
    n_components : int
        Number of mixture components
    n_init : int
        Number of different initializations to try
        
    Returns:
    --------
    best_gmm : EMGaussianMixture
        Best fitted model
    best_log_likelihood : float
        Best log-likelihood achieved
    """
    best_log_likelihood = -np.inf
    best_gmm = None
    
    for i in range(n_init):
        gmm = EMGaussianMixture(n_components=n_components)
        gmm.fit(X)
        
        log_likelihood = gmm._compute_log_likelihood(X)
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_gmm = gmm
    
    return best_gmm, best_log_likelihood

def demonstrate_multiple_initializations():
    """
    Demonstrate multiple initializations for robust EM fitting.
    """
    np.random.seed(42)
    
    # Generate data
    true_gmm = TwoComponentGaussianMixture()
    X, true_z = true_gmm.generate_data(1000)
    
    # Fit with multiple initializations
    best_gmm, best_ll = fit_multiple_initializations(X, n_components=2, n_init=10)
    print(f"Best log-likelihood: {best_ll:.3f}")
    
    return best_gmm, X

if __name__ == "__main__":
    print("Demonstrating Mixture Models...")
    
    # Basic mixture model visualization
    print("\n1. Basic Mixture Model Visualization")
    visualize_mixture_model()
    
    # Two-component mixture demonstration
    print("\n2. Two-Component Mixture Model")
    demonstrate_two_component_mixture()
    
    # KL divergence demonstration
    print("\n3. KL Divergence Computation")
    demonstrate_kl_divergence()
    
    # EM algorithm demonstration
    print("\n4. EM Algorithm")
    demonstrate_em_algorithm()
    
    # Multiple initializations
    print("\n5. Multiple Initializations")
    best_gmm, X = demonstrate_multiple_initializations()
