"""
Choice of K Implementation
==========================

This module provides comprehensive implementations of methods for determining
the optimal number of clusters K, including gap statistics, silhouette analysis,
and prediction strength.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import seaborn as sns

def gap_statistic(X, K_range, B=10, method='uniform', random_state=42):
    """
    Compute gap statistic for determining optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    K_range : list
        Range of K values to test
    B : int, default=10
        Number of reference datasets
    method : str, default='uniform'
        Method for generating reference data: 'uniform' or 'pca'
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    gap_scores : array
        Gap statistic values for each K
    gap_errors : array
        Standard errors for gap statistics
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    gap_scores = []
    gap_errors = []
    
    # Compute observed sum of squares for each K
    ss_obs = []
    for K in K_range:
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        kmeans.fit(X)
        ss_obs.append(kmeans.inertia_)
    
    # Generate reference datasets
    if method == 'uniform':
        # Uniform sampling over the range of observed data
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        reference_data = []
        for b in range(B):
            ref_sample = np.random.uniform(min_vals, max_vals, (n_samples, n_features))
            reference_data.append(ref_sample)
    elif method == 'pca':
        # PCA-based sampling
        pca = PCA().fit(X)
        X_pca = pca.transform(X)
        min_vals = X_pca.min(axis=0)
        max_vals = X_pca.max(axis=0)
        reference_data = []
        for b in range(B):
            ref_sample_pca = np.random.uniform(min_vals, max_vals, (n_samples, n_features))
            ref_sample = pca.inverse_transform(ref_sample_pca)
            reference_data.append(ref_sample)
    
    # Compute reference sum of squares for each K and each reference dataset
    ss_ref = np.zeros((len(K_range), B))
    for b, ref_data in enumerate(reference_data):
        for i, K in enumerate(K_range):
            kmeans = KMeans(n_clusters=K, n_init=10, random_state=random_state)
            kmeans.fit(ref_data)
            ss_ref[i, b] = kmeans.inertia_
    
    # Compute gap statistic
    for i, K in enumerate(K_range):
        log_ss_ref = np.log(ss_ref[i, :])
        log_ss_obs = np.log(ss_obs[i])
        
        gap = np.mean(log_ss_ref) - log_ss_obs
        gap_scores.append(gap)
        
        # Standard error
        se = np.std(log_ss_ref) * np.sqrt(1 + 1/B)
        gap_errors.append(se)
    
    return np.array(gap_scores), np.array(gap_errors)

def find_optimal_k_gap(gap_scores, gap_errors, K_range):
    """
    Find optimal K using gap statistic with one-standard-error rule.
    
    Parameters:
    -----------
    gap_scores : array
        Gap statistic values
    gap_errors : array
        Standard errors for gap statistics
    K_range : list
        Range of K values tested
    
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters
    """
    # Find K where gap(K) >= gap(K+1) - se(K+1)
    for i in range(len(K_range) - 1):
        if gap_scores[i] >= gap_scores[i + 1] - gap_errors[i + 1]:
            return K_range[i]
    
    # If no clear elbow, return K with maximum gap
    return K_range[np.argmax(gap_scores)]

def silhouette_analysis(X, K_range, random_state=42):
    """
    Perform silhouette analysis for determining optimal K.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    K_range : list
        Range of K values to test
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    silhouette_scores : array
        Average silhouette scores for each K
    silhouette_samples_dict : dict
        Individual silhouette scores for each K
    """
    silhouette_scores = []
    silhouette_samples_dict = {}
    
    for K in K_range:
        kmeans = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Store individual silhouette scores
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        silhouette_samples_dict[K] = sample_silhouette_values
    
    return np.array(silhouette_scores), silhouette_samples_dict

def prediction_strength(X, K, n_splits=5, threshold=0.8, random_state=42):
    """
    Compute prediction strength for a given K.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    K : int
        Number of clusters
    n_splits : int, default=5
        Number of data splits for averaging
    threshold : float, default=0.8
        Threshold for prediction strength
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    ps_score : float
        Average prediction strength score
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    ps_scores = []
    
    for split in range(n_splits):
        # Split data randomly
        idx = np.random.permutation(n_samples)
        split_point = n_samples // 2
        A, B = X[idx[:split_point]], X[idx[split_point:]]
        
        # Cluster B
        kmeans_B = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        labels_B = kmeans_B.fit_predict(B)
        
        # Cluster A and predict labels for B
        kmeans_A = KMeans(n_clusters=K, n_init=10, random_state=random_state)
        kmeans_A.fit(A)
        labels_B_pred = kmeans_A.predict(B)
        
        # Compute prediction strength for each cluster
        ps_j = []
        for j in range(K):
            members = np.where(labels_B == j)[0]
            if len(members) < 2:
                continue
            
            # Count pairs that agree in both clusterings
            pairs = [(i, l) for idx, i in enumerate(members) for l in members[idx+1:]]
            agree = sum(labels_B_pred[i] == labels_B_pred[l] for i, l in pairs)
            ps_j.append(agree / len(pairs))
        
        if ps_j:
            ps_scores.append(min(ps_j))
    
    return np.mean(ps_scores) if ps_scores else 0.0

def compute_prediction_strength_range(X, K_range, n_splits=5, threshold=0.8, random_state=42):
    """
    Compute prediction strength for a range of K values.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    K_range : list
        Range of K values to test
    n_splits : int, default=5
        Number of data splits for averaging
    threshold : float, default=0.8
        Threshold for prediction strength
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    ps_scores : array
        Prediction strength scores for each K
    """
    ps_scores = []
    for K in K_range:
        ps_score = prediction_strength(X, K, n_splits, threshold, random_state)
        ps_scores.append(ps_score)
    
    return np.array(ps_scores)

def find_optimal_k_prediction_strength(ps_scores, K_range, threshold=0.8):
    """
    Find optimal K using prediction strength.
    
    Parameters:
    -----------
    ps_scores : array
        Prediction strength scores
    K_range : list
        Range of K values tested
    threshold : float, default=0.8
        Threshold for prediction strength
    
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters
    """
    # Find the largest K where PS(K) >= threshold
    for i in range(len(K_range) - 1, -1, -1):
        if ps_scores[i] >= threshold:
            return K_range[i]
    
    return K_range[0]  # Default to smallest K if none meet threshold

def plot_gap_statistic(K_range, gap_scores, gap_errors, optimal_k=None):
    """
    Plot gap statistic results.
    
    Parameters:
    -----------
    K_range : list
        Range of K values tested
    gap_scores : array
        Gap statistic values
    gap_errors : array
        Standard errors for gap statistics
    optimal_k : int, optional
        Optimal K value to highlight
    """
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(K_range, gap_scores, yerr=gap_errors, marker='o', capsize=5)
    
    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal K = {optimal_k}')
    
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic for Optimal K Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_silhouette_analysis(K_range, silhouette_scores, silhouette_samples_dict, optimal_k=None):
    """
    Plot silhouette analysis results.
    
    Parameters:
    -----------
    K_range : list
        Range of K values tested
    silhouette_scores : array
        Average silhouette scores
    silhouette_samples_dict : dict
        Individual silhouette scores for each K
    optimal_k : int, optional
        Optimal K value to highlight
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot average silhouette scores
    axes[0].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
    if optimal_k is not None:
        axes[0].axvline(x=optimal_k, color='red', linestyle='--', 
                       label=f'Optimal K = {optimal_k}')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Average Silhouette Score')
    axes[0].set_title('Silhouette Analysis for Optimal K Selection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot silhouette distribution for optimal K (if provided)
    if optimal_k is not None and optimal_k in silhouette_samples_dict:
        sample_silhouette_values = silhouette_samples_dict[optimal_k]
        y_lower = 10
        
        for i in range(optimal_k):
            cluster_silhouette_values = sample_silhouette_values[sample_silhouette_values == i]
            cluster_silhouette_values.sort()
            size_cluster_i = len(cluster_silhouette_values)
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(i / optimal_k)
            axes[1].fill_betweenx(np.arange(y_lower, y_upper),
                                 0, cluster_silhouette_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
        
        axes[1].axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
        axes[1].set_xlabel("Silhouette Coefficient")
        axes[1].set_ylabel("Cluster")
        axes[1].set_title(f'Silhouette Plot for K = {optimal_k}')
    
    plt.tight_layout()
    plt.show()

def plot_prediction_strength(K_range, ps_scores, threshold=0.8, optimal_k=None):
    """
    Plot prediction strength results.
    
    Parameters:
    -----------
    K_range : list
        Range of K values tested
    ps_scores : array
        Prediction strength scores
    threshold : float, default=0.8
        Threshold line
    optimal_k : int, optional
        Optimal K value to highlight
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(K_range, ps_scores, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Threshold = {threshold}')
    
    if optimal_k is not None:
        plt.axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal K = {optimal_k}')
    
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Prediction Strength')
    plt.title('Prediction Strength for Optimal K Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def comprehensive_k_selection(X, K_range, methods=['gap', 'silhouette', 'prediction_strength'], 
                           random_state=42, threshold=0.8):
    """
    Perform comprehensive K selection using multiple methods.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    K_range : list
        Range of K values to test
    methods : list, default=['gap', 'silhouette', 'prediction_strength']
        Methods to use for K selection
    random_state : int, default=42
        Random seed for reproducibility
    threshold : float, default=0.8
        Threshold for prediction strength
    
    Returns:
    --------
    results : dict
        Results from all methods including optimal K values
    """
    results = {}
    
    if 'gap' in methods:
        print("Computing gap statistic...")
        gap_scores, gap_errors = gap_statistic(X, K_range, random_state=random_state)
        optimal_k_gap = find_optimal_k_gap(gap_scores, gap_errors, K_range)
        results['gap'] = {
            'scores': gap_scores,
            'errors': gap_errors,
            'optimal_k': optimal_k_gap
        }
        print(f"Gap statistic optimal K: {optimal_k_gap}")
    
    if 'silhouette' in methods:
        print("Computing silhouette analysis...")
        silhouette_scores, silhouette_samples_dict = silhouette_analysis(X, K_range, random_state=random_state)
        optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
        results['silhouette'] = {
            'scores': silhouette_scores,
            'samples_dict': silhouette_samples_dict,
            'optimal_k': optimal_k_silhouette
        }
        print(f"Silhouette optimal K: {optimal_k_silhouette}")
    
    if 'prediction_strength' in methods:
        print("Computing prediction strength...")
        ps_scores = compute_prediction_strength_range(X, K_range, random_state=random_state)
        optimal_k_ps = find_optimal_k_prediction_strength(ps_scores, K_range, threshold)
        results['prediction_strength'] = {
            'scores': ps_scores,
            'optimal_k': optimal_k_ps
        }
        print(f"Prediction strength optimal K: {optimal_k_ps}")
    
    return results

def demonstrate_k_selection():
    """
    Demonstrate K selection methods with synthetic data.
    """
    print("=== K Selection Methods Demonstration ===\n")
    
    # Generate synthetic data with known clusters
    np.random.seed(42)
    n_samples = 300
    
    # Create three well-separated clusters
    cluster1 = np.random.normal([0, 0], [1, 1], (n_samples//3, 2))
    cluster2 = np.random.normal([6, 6], [1, 1], (n_samples//3, 2))
    cluster3 = np.random.normal([3, 9], [1, 1], (n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Define K range to test
    K_range = list(range(2, 11))
    
    print(f"Testing K values: {K_range}")
    print(f"Data shape: {X.shape}")
    print(f"True number of clusters: 3\n")
    
    # Perform comprehensive K selection
    results = comprehensive_k_selection(X, K_range)
    
    # Plot results
    print("\nGenerating plots...")
    
    if 'gap' in results:
        plot_gap_statistic(K_range, results['gap']['scores'], 
                          results['gap']['errors'], results['gap']['optimal_k'])
    
    if 'silhouette' in results:
        plot_silhouette_analysis(K_range, results['silhouette']['scores'],
                               results['silhouette']['samples_dict'], 
                               results['silhouette']['optimal_k'])
    
    if 'prediction_strength' in results:
        plot_prediction_strength(K_range, results['prediction_strength']['scores'],
                               optimal_k=results['prediction_strength']['optimal_k'])
    
    # Summary
    print("\n=== Summary ===")
    for method, result in results.items():
        print(f"{method.capitalize()}: Optimal K = {result['optimal_k']}")
    
    return results

def compare_methods_on_different_data():
    """
    Compare K selection methods on different types of data.
    """
    print("=== Comparing K Selection Methods ===\n")
    
    np.random.seed(42)
    K_range = list(range(2, 11))
    
    # Test 1: Well-separated clusters
    print("Test 1: Well-separated clusters")
    cluster1 = np.random.normal([0, 0], [0.5, 0.5], (100, 2))
    cluster2 = np.random.normal([4, 4], [0.5, 0.5], (100, 2))
    cluster3 = np.random.normal([0, 4], [0.5, 0.5], (100, 2))
    X_well_separated = np.vstack([cluster1, cluster2, cluster3])
    
    results_well = comprehensive_k_selection(X_well_separated, K_range)
    
    # Test 2: Overlapping clusters
    print("\nTest 2: Overlapping clusters")
    cluster1 = np.random.normal([0, 0], [1.5, 1.5], (100, 2))
    cluster2 = np.random.normal([2, 2], [1.5, 1.5], (100, 2))
    cluster3 = np.random.normal([0, 2], [1.5, 1.5], (100, 2))
    X_overlapping = np.vstack([cluster1, cluster2, cluster3])
    
    results_overlapping = comprehensive_k_selection(X_overlapping, K_range)
    
    # Test 3: No clear structure
    print("\nTest 3: No clear structure")
    X_no_structure = np.random.normal(0, 1, (300, 2))
    
    results_no_structure = comprehensive_k_selection(X_no_structure, K_range)
    
    # Summary comparison
    print("\n=== Method Comparison Summary ===")
    datasets = {
        'Well-separated': results_well,
        'Overlapping': results_overlapping,
        'No structure': results_no_structure
    }
    
    for dataset_name, results in datasets.items():
        print(f"\n{dataset_name}:")
        for method, result in results.items():
            print(f"  {method.capitalize()}: K = {result['optimal_k']}")
    
    return datasets

if __name__ == "__main__":
    print("Demonstrating K Selection Methods...")
    results = demonstrate_k_selection()
    
    print("\nComparing Methods on Different Data Types...")
    comparison_results = compare_methods_on_different_data()
