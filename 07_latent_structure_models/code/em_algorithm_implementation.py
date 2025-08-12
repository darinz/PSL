"""
EM Algorithm Implementation
==========================

This module provides comprehensive implementations of the Expectation-Maximization
algorithm, including basic EM, convergence monitoring, K-means comparison,
and variational EM variants.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

class EMAlgorithm:
    """
    Basic EM algorithm implementation for Gaussian mixture models.
    """
    
    def __init__(self, max_iter=100, tol=1e-6):
        """
        Initialize the EM algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihoods = []
        
    def fit(self, X, initial_params=None):
        """
        Fit the model using EM algorithm.
        
        Parameters:
        -----------
        X : array
            Training data
        initial_params : dict, optional
            Initial parameters for the model
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if len(self.log_likelihoods) > 1:
                if abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        # For a two-component Gaussian mixture
        n_samples = len(X)
        
        # Random means
        means = np.random.choice(X, size=2, replace=False)
        
        # Random variances
        variances = np.array([np.var(X)] * 2)
        
        # Random mixing weights
        weights = np.random.dirichlet([1, 1])
        
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _e_step(self, X):
        """E-step: Compute responsibilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, 2))
        
        for k in range(2):
            responsibilities[:, k] = (self.params['weights'][k] * 
                                   norm.pdf(X, self.params['means'][k], 
                                           np.sqrt(self.params['variances'][k])))
        
        # Normalize
        row_sums = responsibilities.sum(axis=1)
        responsibilities = responsibilities / row_sums[:, np.newaxis]
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M-step: Update parameters"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(responsibilities[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(responsibilities[:, k] * X) / 
                                     np.sum(responsibilities[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(responsibilities[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(responsibilities[:, k]))
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood"""
        likelihood = np.zeros(len(X))
        
        for k in range(2):
            likelihood += (self.params['weights'][k] * 
                         norm.pdf(X, self.params['means'][k], 
                                 np.sqrt(self.params['variances'][k])))
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def predict_proba(self, X):
        """Predict component probabilities"""
        return self._e_step(X)
    
    def predict(self, X):
        """Predict component assignments"""
        return np.argmax(self.predict_proba(X), axis=1)

def demonstrate_basic_em():
    """
    Demonstrate the basic EM algorithm.
    """
    np.random.seed(42)

    # Generate data from a two-component Gaussian mixture
    n_samples = 1000
    z = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    x = np.zeros(n_samples)

    x[z == 0] = np.random.normal(0, 1, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(4, 1.5, size=np.sum(z == 1))

    # Fit using EM
    em = EMAlgorithm(max_iter=100, tol=1e-6)
    em.fit(x)

    # Compare with sklearn
    sklearn_gmm = GaussianMixture(n_components=2, random_state=42)
    sklearn_gmm.fit(x.reshape(-1, 1))

    print("EM Algorithm Results:")
    print(f"Means: {em.params['means']}")
    print(f"Variances: {em.params['variances']}")
    print(f"Weights: {em.params['weights']}")

    print("\nSklearn Results:")
    print(f"Means: {sklearn_gmm.means_.flatten()}")
    print(f"Variances: {sklearn_gmm.covariances_.flatten()}")
    print(f"Weights: {sklearn_gmm.weights_}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(em.log_likelihoods)
    plt.title('EM Algorithm Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return em, x

def monitor_em_convergence(X, n_components=2, n_runs=5):
    """
    Monitor EM convergence across multiple runs.
    
    Parameters:
    -----------
    X : array
        Training data
    n_components : int
        Number of mixture components
    n_runs : int
        Number of runs to perform
        
    Returns:
    --------
    results : list
        List of results from each run
    """
    results = []
    
    for run in range(n_runs):
        em = EMAlgorithm(max_iter=200, tol=1e-8)
        em.fit(X)
        
        results.append({
            'run': run + 1,
            'final_ll': em.log_likelihoods[-1],
            'iterations': len(em.log_likelihoods),
            'params': em.params.copy(),
            'log_likelihoods': em.log_likelihoods.copy()
        })
    
    return results

def demonstrate_convergence_monitoring():
    """
    Demonstrate convergence monitoring across multiple runs.
    """
    np.random.seed(42)
    
    # Generate data
    n_samples = 1000
    z = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    x = np.zeros(n_samples)
    x[z == 0] = np.random.normal(0, 1, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(4, 1.5, size=np.sum(z == 1))

    # Monitor convergence
    convergence_results = monitor_em_convergence(x, n_components=2, n_runs=5)

    print("Convergence Results:")
    for result in convergence_results:
        print(f"Run {result['run']}: Final LL = {result['final_ll']:.3f}, "
              f"Iterations = {result['iterations']}")

    # Plot convergence for all runs
    plt.figure(figsize=(12, 8))

    for result in convergence_results:
        plt.plot(result['log_likelihoods'], label=f'Run {result["run"]}', alpha=0.7)

    plt.title('EM Algorithm Convergence (Multiple Runs)')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return convergence_results

def compare_em_kmeans(X, n_components=2):
    """
    Compare EM algorithm with K-means.
    
    Parameters:
    -----------
    X : array
        Training data
    n_components : int
        Number of components/clusters
        
    Returns:
    --------
    em_result : EMAlgorithm
        Fitted EM model
    kmeans_result : KMeans
        Fitted K-means model
    em_labels : array
        EM cluster labels
    kmeans_labels : array
        K-means cluster labels
    """
    # EM Algorithm
    em = EMAlgorithm(max_iter=100, tol=1e-6)
    em.fit(X)
    em_labels = em.predict(X)
    em_responsibilities = em.predict_proba(X)
    
    # K-means
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Compare results
    print("EM Algorithm Results:")
    print(f"Means: {em.params['means']}")
    print(f"Variances: {em.params['variances']}")
    print(f"Weights: {em.params['weights']}")
    
    print("\nK-means Results:")
    print(f"Centers: {kmeans.cluster_centers_.flatten()}")
    print(f"Inertia: {kmeans.inertia_:.3f}")
    
    # Compare assignments
    ari_score = adjusted_rand_score(em_labels, kmeans_labels)
    print(f"\nAdjusted Rand Index: {ari_score:.3f}")
    
    return em, kmeans, em_labels, kmeans_labels

def demonstrate_em_kmeans_comparison():
    """
    Demonstrate comparison between EM algorithm and K-means.
    """
    np.random.seed(42)
    
    # Generate data
    n_samples = 1000
    z = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    x = np.zeros(n_samples)
    x[z == 0] = np.random.normal(0, 1, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(4, 1.5, size=np.sum(z == 1))

    # Compare EM and K-means
    em_result, kmeans_result, em_labels, kmeans_labels = compare_em_kmeans(x)

    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # EM responsibilities
    scatter1 = ax1.scatter(x, em_result.predict_proba(x)[:, 0], c=em_labels, alpha=0.6)
    ax1.set_title('EM Algorithm: Responsibilities')
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(Z=1|x)')

    # K-means assignments
    scatter2 = ax2.scatter(x, np.zeros_like(x), c=kmeans_labels, alpha=0.6)
    ax2.set_title('K-means: Hard Assignments')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cluster')

    # Histogram comparison
    ax3.hist(x[em_labels == 0], bins=30, alpha=0.7, label='EM Cluster 0', density=True)
    ax3.hist(x[em_labels == 1], bins=30, alpha=0.7, label='EM Cluster 1', density=True)
    ax3.set_title('EM Algorithm Clusters')
    ax3.legend()

    ax4.hist(x[kmeans_labels == 0], bins=30, alpha=0.7, label='K-means Cluster 0', density=True)
    ax4.hist(x[kmeans_labels == 1], bins=30, alpha=0.7, label='K-means Cluster 1', density=True)
    ax4.set_title('K-means Clusters')
    ax4.legend()

    plt.tight_layout()
    plt.show()
    
    return em_result, kmeans_result, em_labels, kmeans_labels

class VariationalEM:
    """
    Variational EM algorithm implementation.
    """
    
    def __init__(self, max_iter=100, tol=1e-6):
        """
        Initialize the variational EM algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.free_energies = []
        
    def fit(self, X, initial_params=None):
        """
        Fit using variational EM.
        
        Parameters:
        -----------
        X : array
            Training data
        initial_params : dict, optional
            Initial parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        # Initialize variational distribution
        self.q = np.ones((n_samples, 2)) * 0.5  # Uniform initialization
        
        for iteration in range(self.max_iter):
            # E-step: Update variational distribution
            self._update_variational(X)
            
            # M-step: Update parameters
            self._update_parameters(X)
            
            # Compute free energy
            free_energy = self._compute_free_energy(X)
            self.free_energies.append(free_energy)
            
            # Check convergence
            if len(self.free_energies) > 1:
                if abs(self.free_energies[-1] - self.free_energies[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        means = np.random.choice(X, size=2, replace=False)
        variances = np.array([np.var(X)] * 2)
        weights = np.random.dirichlet([1, 1])
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _update_variational(self, X):
        """Update variational distribution (E-step)"""
        n_samples = len(X)
        
        for i in range(n_samples):
            # Compute unnormalized responsibilities
            log_resp = np.zeros(2)
            for k in range(2):
                log_resp[k] = (np.log(self.params['weights'][k]) + 
                             norm.logpdf(X[i], self.params['means'][k], 
                                       np.sqrt(self.params['variances'][k])))
            
            # Normalize using log-sum-exp trick
            max_log = np.max(log_resp)
            exp_log_resp = np.exp(log_resp - max_log)
            self.q[i] = exp_log_resp / np.sum(exp_log_resp)
    
    def _update_parameters(self, X):
        """Update parameters (M-step)"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(self.q[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(self.q[:, k] * X) / 
                                     np.sum(self.q[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(self.q[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(self.q[:, k]))
    
    def _compute_free_energy(self, X):
        """Compute free energy objective"""
        n_samples = len(X)
        free_energy = 0
        
        for i in range(n_samples):
            for k in range(2):
                if self.q[i, k] > 0:
                    # Log-likelihood term
                    log_likelihood = (np.log(self.params['weights'][k]) + 
                                    norm.logpdf(X[i], self.params['means'][k], 
                                              np.sqrt(self.params['variances'][k])))
                    
                    # Entropy term
                    entropy = -np.log(self.q[i, k])
                    
                    free_energy += self.q[i, k] * (log_likelihood - entropy)
        
        return free_energy
    
    def predict_proba(self, X):
        """Predict component probabilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            log_resp = np.zeros(2)
            for k in range(2):
                log_resp[k] = (np.log(self.params['weights'][k]) + 
                             norm.logpdf(X[i], self.params['means'][k], 
                                       np.sqrt(self.params['variances'][k])))
            
            max_log = np.max(log_resp)
            exp_log_resp = np.exp(log_resp - max_log)
            responsibilities[i] = exp_log_resp / np.sum(exp_log_resp)
        
        return responsibilities

class FactorizedVariationalEM:
    """
    Factorized variational EM algorithm implementation.
    """
    
    def __init__(self, max_iter=100, tol=1e-6):
        """
        Initialize the factorized variational EM algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.free_energies = []
        
    def fit(self, X, initial_params=None):
        """
        Fit using factorized variational EM.
        
        Parameters:
        -----------
        X : array
            Training data
        initial_params : dict, optional
            Initial parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        # Initialize factorized variational distribution
        self.q_factors = np.ones((n_samples, 2)) * 0.5
        
        for iteration in range(self.max_iter):
            # Update each factor independently
            self._update_factors(X)
            
            # Update parameters
            self._update_parameters(X)
            
            # Compute free energy
            free_energy = self._compute_free_energy(X)
            self.free_energies.append(free_energy)
            
            # Check convergence
            if len(self.free_energies) > 1:
                if abs(self.free_energies[-1] - self.free_energies[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        means = np.random.choice(X, size=2, replace=False)
        variances = np.array([np.var(X)] * 2)
        weights = np.random.dirichlet([1, 1])
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _update_factors(self, X):
        """Update factorized variational distribution"""
        n_samples = len(X)
        
        for i in range(n_samples):
            # Compute expected log-likelihood for each component
            expected_log_likelihood = np.zeros(2)
            
            for k in range(2):
                # Prior term
                expected_log_likelihood[k] = np.log(self.params['weights'][k])
                
                # Likelihood term
                expected_log_likelihood[k] += norm.logpdf(X[i], self.params['means'][k], 
                                                        np.sqrt(self.params['variances'][k]))
            
            # Update factor using softmax
            max_log = np.max(expected_log_likelihood)
            exp_log = np.exp(expected_log_likelihood - max_log)
            self.q_factors[i] = exp_log / np.sum(exp_log)
    
    def _update_parameters(self, X):
        """Update parameters using factorized approximation"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(self.q_factors[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(self.q_factors[:, k] * X) / 
                                     np.sum(self.q_factors[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(self.q_factors[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(self.q_factors[:, k]))
    
    def _compute_free_energy(self, X):
        """Compute free energy with factorized approximation"""
        n_samples = len(X)
        free_energy = 0
        
        for i in range(n_samples):
            for k in range(2):
                if self.q_factors[i, k] > 0:
                    # Expected log-likelihood
                    expected_ll = (np.log(self.params['weights'][k]) + 
                                 norm.logpdf(X[i], self.params['means'][k], 
                                           np.sqrt(self.params['variances'][k])))
                    
                    # Entropy of factor
                    entropy = -np.log(self.q_factors[i, k])
                    
                    free_energy += self.q_factors[i, k] * (expected_ll - entropy)
        
        return free_energy

def demonstrate_variational_em():
    """
    Demonstrate variational EM algorithms.
    """
    np.random.seed(42)
    
    # Generate data
    n_samples = 1000
    z = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    x = np.zeros(n_samples)
    x[z == 0] = np.random.normal(0, 1, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(4, 1.5, size=np.sum(z == 1))

    # Standard EM
    em = EMAlgorithm(max_iter=100, tol=1e-6)
    em.fit(x)

    # Variational EM
    vem = VariationalEM(max_iter=100, tol=1e-6)
    vem.fit(x)

    # Factorized Variational EM
    fvem = FactorizedVariationalEM(max_iter=100, tol=1e-6)
    fvem.fit(x)

    print("Standard EM Results:")
    print(f"Means: {em.params['means']}")
    print(f"Variances: {em.params['variances']}")
    print(f"Weights: {em.params['weights']}")

    print("\nVariational EM Results:")
    print(f"Means: {vem.params['means']}")
    print(f"Variances: {vem.params['variances']}")
    print(f"Weights: {vem.params['weights']}")

    print("\nFactorized Variational EM Results:")
    print(f"Means: {fvem.params['means']}")
    print(f"Variances: {fvem.params['variances']}")
    print(f"Weights: {fvem.params['weights']}")

    # Plot free energy convergence
    plt.figure(figsize=(10, 6))
    plt.plot(vem.free_energies, label='Variational EM')
    plt.plot(fvem.free_energies, label='Factorized VEM')
    plt.title('Variational EM: Free Energy Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Free Energy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Compare different EM variants
    em_variants = {
        'Standard EM': em,
        'Variational EM': vem,
        'Factorized VEM': fvem
    }

    print("\nComparison of EM Variants:")
    for name, variant in em_variants.items():
        if hasattr(variant, 'log_likelihoods'):
            print(f"{name}: Final LL = {variant.log_likelihoods[-1]:.3f}")
        else:
            print(f"{name}: Final Free Energy = {variant.free_energies[-1]:.3f}")
    
    return em, vem, fvem, x

if __name__ == "__main__":
    print("Demonstrating EM Algorithm...")
    
    # Basic EM demonstration
    print("\n1. Basic EM Algorithm")
    em, x = demonstrate_basic_em()
    
    # Convergence monitoring
    print("\n2. Convergence Monitoring")
    convergence_results = demonstrate_convergence_monitoring()
    
    # EM vs K-means comparison
    print("\n3. EM vs K-means Comparison")
    em_result, kmeans_result, em_labels, kmeans_labels = demonstrate_em_kmeans_comparison()
    
    # Variational EM demonstration
    print("\n4. Variational EM")
    em, vem, fvem, x = demonstrate_variational_em()
