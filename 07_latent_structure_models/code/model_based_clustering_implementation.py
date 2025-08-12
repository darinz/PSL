"""
Model-Based Clustering Implementation
===================================

This module provides comprehensive implementations of model-based clustering
using Gaussian Mixture Models, including model selection, visualization,
and analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

class ModelBasedClustering:
    """Comprehensive model-based clustering implementation."""
    
    def __init__(self, n_components=2, covariance_type='full', random_state=None):
        """
        Initialize model-based clustering.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of mixture components (clusters)
        covariance_type : str, default='full'
            Type of covariance parameters: 'full', 'tied', 'diag', 'spherical'
        random_state : int, default=None
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = None
        self.bic_scores = []
        self.aic_scores = []
        
    def fit(self, X):
        """
        Fit Gaussian Mixture Model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10
        )
        self.gmm.fit(X)
        return self
    
    def predict(self, X):
        """
        Predict cluster labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Predicted cluster labels
        """
        return self.gmm.predict(X)
    
    def predict_proba(self, X):
        """
        Predict cluster membership probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_components)
            Cluster membership probabilities
        """
        return self.gmm.predict_proba(X)
    
    def score(self, X):
        """
        Compute log-likelihood.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood of the data
        """
        return self.gmm.score(X)
    
    def bic(self, X):
        """
        Compute BIC score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        bic : float
            Bayesian Information Criterion score
        """
        return self.gmm.bic(X)
    
    def aic(self, X):
        """
        Compute AIC score.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        aic : float
            Akaike Information Criterion score
        """
        return self.gmm.aic(X)
    
    def plot_clusters(self, X, title=None):
        """
        Visualize clustering results.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to visualize
        title : str, optional
            Title for the plot
        """
        labels = self.predict(X)
        probas = self.predict_proba(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot cluster assignments
        scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[0].set_title('Hard Cluster Assignments')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot uncertainty (max probability)
        max_proba = np.max(probas, axis=1)
        scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=max_proba, cmap='plasma', alpha=0.7)
        axes[1].set_title('Cluster Assignment Uncertainty')
        axes[1].set_xlabel('Feature 1')
        axes[1].set_ylabel('Feature 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.suptitle(title or f'GMM Clustering (K={self.n_components})')
        plt.tight_layout()
        plt.show()
    
    def plot_contours(self, X, title=None):
        """
        Plot GMM contours and data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to visualize
        title : str, optional
            Title for the plot
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Compute GMM density
        density = np.exp(self.gmm.score_samples(grid))
        density = density.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Contour plot of mixture density
        plt.contour(xx, yy, density, levels=20, alpha=0.6, colors='black')
        plt.contourf(xx, yy, density, levels=20, alpha=0.3, cmap='viridis')
        
        # Plot data points
        labels = self.predict(X)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                             alpha=0.7, edgecolors='black', s=50)
        
        # Plot component means
        plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Component Means')
        
        plt.title(title or f'GMM Density Contours (K={self.n_components})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter)
        plt.show()
    
    def model_selection(self, X, K_range=range(1, 11)):
        """
        Perform model selection using BIC and AIC.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data for model selection
        K_range : range, default=range(1, 11)
            Range of K values to test
            
        Returns:
        --------
        results : dict
            Dictionary containing model selection results
        """
        bic_scores = []
        aic_scores = []
        log_likelihoods = []
        
        for k in K_range:
            if k == 1:
                # Single component case
                bic_scores.append(np.inf)
                aic_scores.append(np.inf)
                log_likelihoods.append(-np.inf)
                continue
                
            gmm = GaussianMixture(n_components=k, covariance_type=self.covariance_type,
                                 random_state=self.random_state, n_init=10)
            gmm.fit(X)
            
            bic_scores.append(gmm.bic(X))
            aic_scores.append(gmm.aic(X))
            log_likelihoods.append(gmm.score(X))
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(K_range, log_likelihoods, marker='o')
        axes[0].set_title('Log-Likelihood')
        axes[0].set_xlabel('Number of Components (K)')
        axes[0].set_ylabel('Log-Likelihood')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(K_range, bic_scores, marker='o', color='red')
        axes[1].set_title('BIC Score')
        axes[1].set_xlabel('Number of Components (K)')
        axes[1].set_ylabel('BIC')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(K_range, aic_scores, marker='o', color='green')
        axes[2].set_title('AIC Score')
        axes[2].set_xlabel('Number of Components (K)')
        axes[2].set_ylabel('AIC')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal K
        optimal_bic_k = K_range[np.argmin(bic_scores)]
        optimal_aic_k = K_range[np.argmin(aic_scores)]
        
        print(f"Optimal K (BIC): {optimal_bic_k}")
        print(f"Optimal K (AIC): {optimal_aic_k}")
        
        return {
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'log_likelihoods': log_likelihoods,
            'optimal_bic_k': optimal_bic_k,
            'optimal_aic_k': optimal_aic_k
        }
    
    def analyze_components(self, X):
        """
        Analyze component parameters and characteristics.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to analyze
            
        Returns:
        --------
        analysis : dict
            Dictionary containing component analysis
        """
        labels = self.predict(X)
        probas = self.predict_proba(X)
        
        analysis = {
            'component_sizes': [],
            'component_weights': self.gmm.weights_,
            'component_means': self.gmm.means_,
            'component_covariances': self.gmm.covariances_,
            'component_uncertainty': []
        }
        
        for k in range(self.n_components):
            # Component size
            size = np.sum(labels == k)
            analysis['component_sizes'].append(size)
            
            # Component uncertainty (average probability for assigned points)
            component_mask = labels == k
            if np.any(component_mask):
                avg_prob = np.mean(probas[component_mask, k])
                analysis['component_uncertainty'].append(avg_prob)
            else:
                analysis['component_uncertainty'].append(0.0)
        
        return analysis

def load_old_faithful_data():
    """
    Load and preprocess Old Faithful Geyser data.
    
    Returns:
    --------
    X : array, shape (n_samples, 2)
        Old Faithful data with duration and waiting time
    """
    # Generate synthetic Old Faithful data (similar to the real data)
    np.random.seed(42)
    n_samples = 272
    
    # Component 1: Short eruptions, short waits
    n1 = int(0.6 * n_samples)
    duration1 = np.random.normal(2.0, 0.3, n1)
    waiting1 = np.random.normal(54, 8, n1)
    
    # Component 2: Long eruptions, long waits
    n2 = n_samples - n1
    duration2 = np.random.normal(4.3, 0.4, n2)
    waiting2 = np.random.normal(80, 12, n2)
    
    # Combine data
    duration = np.concatenate([duration1, duration2])
    waiting = np.concatenate([waiting1, waiting2])
    
    # Add some noise and intermediate cases
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    duration[noise_idx] += np.random.normal(0, 0.5, len(noise_idx))
    waiting[noise_idx] += np.random.normal(0, 10, len(noise_idx))
    
    return np.column_stack([duration, waiting])

def demonstrate_model_based_clustering():
    """
    Demonstrate model-based clustering with Old Faithful data.
    """
    print("=== Model-Based Clustering Demonstration ===\n")
    
    # Load data
    X = load_old_faithful_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: Duration (minutes), Waiting time (minutes)")
    
    # Model selection
    print("\nPerforming model selection...")
    mbc = ModelBasedClustering()
    results = mbc.model_selection(X, K_range=range(1, 8))
    
    # Fit optimal model (using BIC)
    optimal_k = results['optimal_bic_k']
    print(f"\nFitting optimal model with K={optimal_k}...")
    
    mbc_optimal = ModelBasedClustering(n_components=optimal_k)
    mbc_optimal.fit(X)
    
    # Visualize results
    mbc_optimal.plot_clusters(X, f"Old Faithful Data - {optimal_k} Components")
    mbc_optimal.plot_contours(X, f"Old Faithful Data - {optimal_k} Components")
    
    # Compare with different K values
    print("\nComparing different numbers of components...")
    for k in [2, 3, 4]:
        mbc_k = ModelBasedClustering(n_components=k)
        mbc_k.fit(X)
        
        # Evaluate clustering
        labels = mbc_k.predict(X)
        silhouette = silhouette_score(X, labels)
        bic = mbc_k.bic(X)
        
        print(f"K={k}: Silhouette={silhouette:.3f}, BIC={bic:.1f}")
        
        # Plot
        mbc_k.plot_clusters(X, f"Old Faithful Data - {k} Components")
        mbc_k.plot_contours(X, f"Old Faithful Data - {k} Components")
    
    # Analyze component parameters
    print(f"\nComponent parameters for K={optimal_k}:")
    analysis = mbc_optimal.analyze_components(X)
    for k in range(optimal_k):
        print(f"Component {k+1}:")
        print(f"  Size: {analysis['component_sizes'][k]}")
        print(f"  Mixing weight: {analysis['component_weights'][k]:.3f}")
        print(f"  Mean: {analysis['component_means'][k]}")
        print(f"  Average assignment probability: {analysis['component_uncertainty'][k]:.3f}")
        print(f"  Covariance:\n{analysis['component_covariances'][k]}")
    
    return mbc_optimal, X, results

def compare_covariance_types():
    """
    Compare different covariance types for GMM.
    """
    print("=== Covariance Type Comparison ===\n")
    
    # Load data
    X = load_old_faithful_data()
    
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    results = {}
    
    for cov_type in covariance_types:
        print(f"Testing {cov_type} covariance...")
        
        # Model selection
        mbc = ModelBasedClustering(covariance_type=cov_type)
        model_results = mbc.model_selection(X, K_range=range(2, 7))
        
        # Fit optimal model
        optimal_k = model_results['optimal_bic_k']
        mbc_optimal = ModelBasedClustering(n_components=optimal_k, covariance_type=cov_type)
        mbc_optimal.fit(X)
        
        # Evaluate
        labels = mbc_optimal.predict(X)
        silhouette = silhouette_score(X, labels)
        bic = mbc_optimal.bic(X)
        
        results[cov_type] = {
            'optimal_k': optimal_k,
            'silhouette': silhouette,
            'bic': bic,
            'log_likelihood': mbc_optimal.score(X)
        }
        
        print(f"  Optimal K: {optimal_k}")
        print(f"  Silhouette: {silhouette:.3f}")
        print(f"  BIC: {bic:.1f}")
        print(f"  Log-likelihood: {mbc_optimal.score(X):.1f}")
        
        # Plot
        mbc_optimal.plot_clusters(X, f"Old Faithful Data - {cov_type} Covariance")
    
    # Summary
    print("\n=== Summary ===")
    for cov_type, result in results.items():
        print(f"{cov_type}: K={result['optimal_k']}, Silhouette={result['silhouette']:.3f}, BIC={result['bic']:.1f}")
    
    return results

def demonstrate_uncertainty_analysis():
    """
    Demonstrate uncertainty analysis in model-based clustering.
    """
    print("=== Uncertainty Analysis ===\n")
    
    # Load data
    X = load_old_faithful_data()
    
    # Fit models with different K
    models = {}
    for k in [2, 3, 4]:
        mbc = ModelBasedClustering(n_components=k)
        mbc.fit(X)
        models[k] = mbc
    
    # Analyze uncertainty
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, k in enumerate([2, 3, 4]):
        mbc = models[k]
        probas = mbc.predict_proba(X)
        max_proba = np.max(probas, axis=1)
        uncertainty = 1 - max_proba
        
        # Plot uncertainty distribution
        axes[i//2, i%2].hist(uncertainty, bins=20, alpha=0.7, edgecolor='black')
        axes[i//2, i%2].set_title(f'Uncertainty Distribution (K={k})')
        axes[i//2, i%2].set_xlabel('Uncertainty (1 - max probability)')
        axes[i//2, i%2].set_ylabel('Frequency')
        axes[i//2, i%2].grid(True, alpha=0.3)
        
        print(f"K={k}: Mean uncertainty = {np.mean(uncertainty):.3f}")
    
    # Plot uncertainty vs data
    mbc_3 = models[3]
    probas_3 = mbc_3.predict_proba(X)
    max_proba_3 = np.max(probas_3, axis=1)
    uncertainty_3 = 1 - max_proba_3
    
    scatter = axes[1, 1].scatter(X[:, 0], X[:, 1], c=uncertainty_3, cmap='plasma', alpha=0.7)
    axes[1, 1].set_title('Uncertainty in Data Space (K=3)')
    axes[1, 1].set_xlabel('Duration')
    axes[1, 1].set_ylabel('Waiting Time')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return models

if __name__ == "__main__":
    print("Demonstrating Model-Based Clustering...")
    
    # Basic demonstration
    mbc_optimal, X, results = demonstrate_model_based_clustering()
    
    # Compare covariance types
    covariance_results = compare_covariance_types()
    
    # Uncertainty analysis
    uncertainty_models = demonstrate_uncertainty_analysis()
