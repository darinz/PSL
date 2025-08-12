import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


def compare_complexity():
    """
    Compare computational complexity of discriminant analysis methods
    """
    # Parameters
    p_values = np.arange(10, 101, 10)  # Feature dimensions
    K = 3  # Number of classes
    
    # Parameter counts
    qda_params = K * p_values**2 + K * p_values + K
    lda_params = p_values**2 + K * p_values + K
    nb_params = 2 * K * p_values + K
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, qda_params, 'o-', label='QDA', linewidth=2)
    plt.plot(p_values, lda_params, 's-', label='LDA', linewidth=2)
    plt.plot(p_values, nb_params, '^-', label='Naive Bayes', linewidth=2)
    
    plt.xlabel('Number of Features (p)')
    plt.ylabel('Number of Parameters')
    plt.title('Parameter Complexity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Print numerical results
    print("Parameter Complexity Comparison:")
    print("-" * 50)
    for i, p in enumerate(p_values):
        print(f"p={p:3d}: QDA={qda_params[i]:6d}, LDA={lda_params[i]:6d}, NB={nb_params[i]:6d}")
    
    return qda_params, lda_params, nb_params


class DiscriminantAnalysisComparison:
    """
    Comprehensive comparison of discriminant analysis methods
    """
    
    def __init__(self):
        self.methods = {
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'Naive Bayes': GaussianNB()
        }
        self.results = {}
        
    def generate_data(self, n_samples=1000, n_features=10, n_classes=3, 
                     n_informative=8, n_redundant=2, random_state=42):
        """
        Generate synthetic data for comparison
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    def compare_methods(self, X, y, cv=5):
        """
        Compare all methods using cross-validation
        """
        for name, method in self.methods.items():
            scores = cross_val_score(method, X, y, cv=cv, scoring='accuracy')
            self.results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
        return self.results
    
    def visualize_results(self):
        """
        Visualize comparison results
        """
        methods = list(self.results.keys())
        means = [self.results[m]['mean_score'] for m in methods]
        stds = [self.results[m]['std_score'] for m in methods]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = axes[0].bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Cross-validation Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Color bars based on performance
        colors = ['green' if m == max(means) else 'lightblue' for m in means]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Box plot
        scores_data = [self.results[m]['scores'] for m in methods]
        axes[1].boxplot(scores_data, labels=methods)
        axes[1].set_title('Score Distribution')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("Method Comparison Results:")
        print("-" * 50)
        for method in methods:
            result = self.results[method]
            print(f"{method:15s}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
    
    def analyze_decision_boundaries(self, X, y):
        """
        Analyze decision boundaries for 2D data
        """
        # Use only first 2 features for visualization
        X_2d = X[:, :2]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (name, method) in enumerate(self.methods.items()):
            # Fit method
            method.fit(X_2d, y)
            
            # Create mesh for decision boundaries
            x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
            y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            # Predict on mesh
            Z = method.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundaries
            axes[i].contourf(xx, yy, Z, alpha=0.3)
            
            # Plot data points
            for j in range(len(np.unique(y))):
                mask = y == j
                axes[i].scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.7, label=f'Class {j}')
            
            axes[i].set_title(f'{name} Decision Boundaries')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def parameter_efficiency_analysis(self, X, y):
        """
        Analyze parameter efficiency of different methods
        """
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        # Calculate parameter counts
        qda_params = n_classes * n_features**2 + n_classes * n_features + n_classes
        lda_params = n_features**2 + n_classes * n_features + n_classes
        nb_params = 2 * n_classes * n_features + n_classes
        
        # Calculate decision parameters
        qda_decision_params = n_classes * (n_features + 1)  # Quadratic terms
        lda_decision_params = n_features + 1  # Linear terms
        nb_decision_params = n_features + 1  # Linear in log space
        
        # Create comparison table
        comparison_data = {
            'Method': ['QDA', 'LDA', 'Naive Bayes'],
            'Total Parameters': [qda_params, lda_params, nb_params],
            'Decision Parameters': [qda_decision_params, lda_decision_params, nb_decision_params],
            'Efficiency Ratio': [qda_decision_params/qda_params, lda_decision_params/lda_params, nb_decision_params/nb_params]
        }
        
        df = pd.DataFrame(comparison_data)
        
        print("Parameter Efficiency Analysis:")
        print("-" * 60)
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Visualize efficiency
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parameter counts
        methods = df['Method']
        total_params = df['Total Parameters']
        decision_params = df['Decision Parameters']
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0].bar(x - width/2, total_params, width, label='Total Parameters', alpha=0.7)
        axes[0].bar(x + width/2, decision_params, width, label='Decision Parameters', alpha=0.7)
        axes[0].set_xlabel('Method')
        axes[0].set_ylabel('Number of Parameters')
        axes[0].set_title('Parameter Count Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Efficiency ratio
        efficiency = df['Efficiency Ratio']
        bars = axes[1].bar(methods, efficiency, alpha=0.7)
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Efficiency Ratio')
        axes[1].set_title('Parameter Efficiency (Higher is Better)')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiency):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{eff:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return df


def demonstrate_comparison():
    """
    Demonstrate comprehensive comparison
    """
    # Create comparison object
    comparison = DiscriminantAnalysisComparison()
    
    # Generate data
    X, y = comparison.generate_data(n_samples=1000, n_features=10, n_classes=3)
    
    # Compare methods
    results = comparison.compare_methods(X, y)
    
    # Visualize results
    comparison.visualize_results()
    
    # Analyze decision boundaries
    comparison.analyze_decision_boundaries(X, y)
    
    # Parameter efficiency analysis
    efficiency_df = comparison.parameter_efficiency_analysis(X, y)
    
    return comparison, results, efficiency_df


def analyze_scalability():
    """
    Analyze computational scalability of discriminant analysis methods
    """
    # Parameters
    n_samples_list = [100, 500, 1000, 2000, 5000]
    n_features = 50
    n_classes = 3
    
    methods = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Naive Bayes': GaussianNB()
    }
    
    timing_results = {name: [] for name in methods.keys()}
    
    for n_samples in n_samples_list:
        # Generate data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=42
        )
        
        for name, method in methods.items():
            # Time fitting
            start_time = time.time()
            method.fit(X, y)
            fit_time = time.time() - start_time
            
            # Time prediction
            start_time = time.time()
            method.predict(X)
            pred_time = time.time() - start_time
            
            timing_results[name].append((fit_time, pred_time))
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fitting time
    for name in methods.keys():
        fit_times = [t[0] for t in timing_results[name]]
        axes[0].plot(n_samples_list, fit_times, 'o-', label=name, linewidth=2)
    
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_ylabel('Fitting Time (seconds)')
    axes[0].set_title('Fitting Time Scalability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Prediction time
    for name in methods.keys():
        pred_times = [t[1] for t in timing_results[name]]
        axes[1].plot(n_samples_list, pred_times, 'o-', label=name, linewidth=2)
    
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Prediction Time (seconds)')
    axes[1].set_title('Prediction Time Scalability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print timing results
    print("Scalability Analysis Results:")
    print("-" * 60)
    print(f"{'Samples':>8} {'Method':>15} {'Fit (s)':>10} {'Pred (s)':>10}")
    print("-" * 60)
    
    for i, n_samples in enumerate(n_samples_list):
        for name in methods.keys():
            fit_time, pred_time = timing_results[name][i]
            print(f"{n_samples:8d} {name:15s} {fit_time:10.4f} {pred_time:10.4f}")
    
    return timing_results


def binary_lda_analysis():
    """
    Detailed analysis of binary LDA case
    """
    # Generate binary classification data
    np.random.seed(42)
    n_samples = 1000
    n_features = 2
    
    # Class 0: centered at (0, 0)
    X0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
    
    # Class 1: centered at (2, 2)
    X1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Extract parameters
    means = lda.means_
    covariance = lda.covariance_
    priors = lda.priors_
    
    # Calculate decision boundary parameters
    beta = np.linalg.inv(covariance) @ (means[1] - means[0])
    beta0 = -0.5 * (means[1] @ np.linalg.inv(covariance) @ means[1] - 
                    means[0] @ np.linalg.inv(covariance) @ means[0]) + np.log(priors[1]/priors[0])
    
    print("Binary LDA Analysis:")
    print("-" * 40)
    print(f"Class means: {means}")
    print(f"Shared covariance:\n{covariance}")
    print(f"Class priors: {priors}")
    print(f"Decision boundary coefficient: {beta}")
    print(f"Decision boundary intercept: {beta0:.4f}")
    
    # Visualize decision boundary
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Data and decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Calculate decision function
    Z = beta[0] * xx + beta[1] * yy + beta0
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    axes[0].contour(xx, yy, Z, levels=[0], colors='red', linewidths=2, label='Decision Boundary')
    axes[0].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['blue', 'orange'])
    
    # Plot data points
    for i in range(2):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    
    axes[0].set_title('Binary LDA Decision Boundary')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Parameter efficiency visualization
    total_params = n_features**2 + 2*n_features + 1  # covariance + means + prior
    decision_params = n_features + 1  # beta + beta0
    
    efficiency_data = ['Total Parameters', 'Decision Parameters']
    param_counts = [total_params, decision_params]
    
    bars = axes[1].bar(efficiency_data, param_counts, alpha=0.7)
    axes[1].set_title('Parameter Efficiency in Binary LDA')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, param_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return lda, beta, beta0


def method_selection_guidelines():
    """
    Demonstrate method selection guidelines with examples
    """
    # Create different scenarios
    scenarios = {
        'Low-dimensional, normal data': {
            'n_features': 2,
            'n_samples': 500,
            'n_classes': 3,
            'n_informative': 2,
            'n_redundant': 0,
            'n_clusters_per_class': 1,
            'recommended': 'QDA'
        },
        'High-dimensional, normal data': {
            'n_features': 50,
            'n_samples': 1000,
            'n_classes': 3,
            'n_informative': 20,
            'n_redundant': 30,
            'n_clusters_per_class': 1,
            'recommended': 'LDA'
        },
        'Limited training data': {
            'n_features': 10,
            'n_samples': 100,
            'n_classes': 2,
            'n_informative': 8,
            'n_redundant': 2,
            'n_clusters_per_class': 1,
            'recommended': 'Naive Bayes'
        }
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n{scenario_name}:")
        print("-" * 50)
        
        # Generate data
        X, y = make_classification(
            n_samples=params['n_samples'],
            n_features=params['n_features'],
            n_classes=params['n_classes'],
            n_informative=params['n_informative'],
            n_redundant=params['n_redundant'],
            n_clusters_per_class=params['n_clusters_per_class'],
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Compare methods
        methods = {
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis(),
            'Naive Bayes': GaussianNB()
        }
        
        scenario_results = {}
        for name, method in methods.items():
            scores = cross_val_score(method, X, y, cv=5, scoring='accuracy')
            scenario_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
        
        # Print results
        print(f"Recommended method: {params['recommended']}")
        print("Performance comparison:")
        for method, result in scenario_results.items():
            marker = "★" if method == params['recommended'] else " "
            print(f"{marker} {method:12s}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        
        results[scenario_name] = scenario_results
    
    return results


def limitations_analysis():
    """
    Analyze limitations of discriminant analysis methods
    """
    limitations = {
        'Distributional Assumptions': {
            'description': 'Most methods assume normality',
            'impact': 'Performance degrades with non-normal data',
            'mitigation': 'Use non-parametric methods or data transformation'
        },
        'Linear Decision Boundaries': {
            'description': 'LDA and FDA are limited to linear separators',
            'impact': 'Cannot capture complex non-linear relationships',
            'mitigation': 'Use QDA, kernel methods, or non-linear classifiers'
        },
        'Parameter Inefficiency': {
            'description': 'Many parameters for simple decision rules',
            'impact': 'Computational cost and overfitting risk',
            'mitigation': 'Use direct methods like logistic regression'
        },
        'Curse of Dimensionality': {
            'description': 'Performance degrades in high dimensions',
            'impact': 'Poor generalization with many features',
            'mitigation': 'Feature selection, regularization, or dimensionality reduction'
        },
        'Feature Independence': {
            'description': 'Naive Bayes assumes independence',
            'impact': 'Performance loss with correlated features',
            'mitigation': 'Feature engineering or use other methods'
        }
    }
    
    print("Limitations of Discriminant Analysis:")
    print("=" * 80)
    
    for limitation, details in limitations.items():
        print(f"\n{limitation}:")
        print(f"  Description: {details['description']}")
        print(f"  Impact: {details['impact']}")
        print(f"  Mitigation: {details['mitigation']}")
    
    # Demonstrate some limitations with examples
    print("\n" + "=" * 80)
    print("Demonstrating Limitations:")
    print("=" * 80)
    
    # 1. Non-normal data example
    print("\n1. Non-normal Data Example:")
    np.random.seed(42)
    X_nonnormal = np.random.exponential(1, (500, 2))
    y_nonnormal = (X_nonnormal[:, 0] + X_nonnormal[:, 1] > 2).astype(int)
    
    lda_nonnormal = LinearDiscriminantAnalysis()
    nb_nonnormal = GaussianNB()
    
    lda_score = cross_val_score(lda_nonnormal, X_nonnormal, y_nonnormal, cv=5).mean()
    nb_score = cross_val_score(nb_nonnormal, X_nonnormal, y_nonnormal, cv=5).mean()
    
    print(f"LDA accuracy on non-normal data: {lda_score:.4f}")
    print(f"Naive Bayes accuracy on non-normal data: {nb_score:.4f}")
    
    # 2. High-dimensional data example
    print("\n2. High-dimensional Data Example:")
    X_highdim, y_highdim = make_classification(
        n_samples=100, n_features=100, n_classes=2, 
        n_informative=10, n_redundant=90, random_state=42
    )
    
    lda_highdim = LinearDiscriminantAnalysis()
    nb_highdim = GaussianNB()
    
    lda_score_high = cross_val_score(lda_highdim, X_highdim, y_highdim, cv=5).mean()
    nb_score_high = cross_val_score(nb_highdim, X_highdim, y_highdim, cv=5).mean()
    
    print(f"LDA accuracy on high-dimensional data: {lda_score_high:.4f}")
    print(f"Naive Bayes accuracy on high-dimensional data: {nb_score_high:.4f}")
    
    return limitations


def main():
    """
    Main function to demonstrate summary analysis
    """
    print("Discriminant Analysis Summary Demonstration")
    print("=" * 60)
    
    # 1. Complexity comparison
    print("\n1. Parameter Complexity Analysis:")
    qda_p, lda_p, nb_p = compare_complexity()
    
    # 2. Comprehensive comparison
    print("\n2. Comprehensive Method Comparison:")
    comparison, results, efficiency_df = demonstrate_comparison()
    
    # 3. Scalability analysis
    print("\n3. Computational Scalability Analysis:")
    timing_results = analyze_scalability()
    
    # 4. Binary LDA analysis
    print("\n4. Binary LDA Detailed Analysis:")
    lda_binary, beta, beta0 = binary_lda_analysis()
    
    # 5. Method selection guidelines
    print("\n5. Method Selection Guidelines:")
    selection_results = method_selection_guidelines()
    
    # 6. Limitations analysis
    print("\n6. Limitations Analysis:")
    limitations = limitations_analysis()
    
    return {
        'complexity': (qda_p, lda_p, nb_p),
        'comparison': comparison,
        'results': results,
        'efficiency': efficiency_df,
        'timing': timing_results,
        'binary_lda': (lda_binary, beta, beta0),
        'selection': selection_results,
        'limitations': limitations
    }


if __name__ == "__main__":
    main()
