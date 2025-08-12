import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.preprocessing import StandardScaler
import cvxopt
from cvxopt import matrix, solvers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns


class KernelSVM:
    """Support Vector Machine implementation with kernel functions"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, degree=3, coef0=0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.support_vectors = None
        self.lambda_values = None
        self.beta_0 = None
        
    def kernel_function(self, X1, X2):
        """Compute kernel matrix between X1 and X2"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # Compute pairwise distances efficiently
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            K = np.exp(-self.gamma * (X1_norm + X2_norm - 2 * np.dot(X1, X2.T)))
            return K
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """Fit kernel SVM using quadratic programming"""
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_function(X, X)
        
        # Prepare the quadratic programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        
        # Constraints: 0 <= lambda_i <= C
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))
        
        A = matrix(y.reshape(1, -1))
        b = matrix(0.0)
        
        # Solve the quadratic programming problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Extract Lagrange multipliers
        self.lambda_values = np.array(solution['x']).flatten()
        
        # Find support vectors
        support_vector_indices = self.lambda_values > 1e-5
        self.support_vectors = X[support_vector_indices]
        support_vector_lambdas = self.lambda_values[support_vector_indices]
        support_vector_y = y[support_vector_indices]
        
        # Compute beta_0
        self.beta_0 = np.mean(support_vector_y - 
                             np.sum(support_vector_lambdas.reshape(-1, 1) * 
                                   support_vector_y.reshape(-1, 1) * 
                                   self.kernel_function(self.support_vectors, self.support_vectors), axis=0))
        
    def predict(self, X):
        """Predict class labels"""
        return np.sign(self.decision_function(X))
    
    def decision_function(self, X):
        """Compute decision function values"""
        if self.support_vectors is None:
            return np.zeros(X.shape[0])
        
        K = self.kernel_function(X, self.support_vectors)
        support_vector_lambdas = self.lambda_values[self.lambda_values > 1e-5]
        support_vector_y = np.array([y_val for i, y_val in enumerate(self.lambda_values) if self.lambda_values[i] > 1e-5])
        
        return np.sum(support_vector_lambdas.reshape(-1, 1) * 
                     support_vector_y.reshape(-1, 1) * K.T, axis=0) + self.beta_0


def generate_xor_data(n_samples=100, random_state=42):
    """Generate XOR-like data"""
    np.random.seed(random_state)
    
    # Generate XOR pattern with noise
    X = np.random.randn(n_samples, 2) * 0.3
    y = np.ones(n_samples)
    
    # XOR pattern: (0,0) and (1,1) are class -1, (0,1) and (1,0) are class 1
    for i in range(n_samples):
        if (X[i, 0] < 0 and X[i, 1] < 0) or (X[i, 0] > 0 and X[i, 1] > 0):
            y[i] = -1
        else:
            y[i] = 1
    
    return X, y


def generate_nonlinear_data(data_type='circles', n_samples=100, random_state=42):
    """Generate different types of nonlinear data"""
    if data_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
    elif data_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
    elif data_type == 'xor':
        X, y = generate_xor_data(n_samples=n_samples, random_state=random_state)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return X, y


def visualize_kernel_svm(X, y, svm, title="Kernel SVM"):
    """Visualize kernel SVM decision boundary"""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
               c='red', label='Class 1', alpha=0.6, s=50)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], 
               c='blue', label='Class -1', alpha=0.6, s=50)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and regions
    plt.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='black', linewidths=2)
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, 
                colors=['blue', 'white', 'red'])
    
    # Highlight support vectors
    if svm.support_vectors is not None:
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=100, linewidth=2, facecolors='none', edgecolors='k', 
                   label=f'Support Vectors ({len(svm.support_vectors)})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_kernel_comparison():
    """Demonstrate different kernel functions on nonlinear data"""
    print("=== Kernel Comparison ===\n")
    
    # Generate non-linear data
    X, y = generate_nonlinear_data('circles', n_samples=100)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Class distribution: {np.bincount(y + 1)}")
    
    # Compare different kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, kernel in enumerate(kernels):
        # Fit SVM
        svm = KernelSVM(C=1.0, kernel=kernel, gamma=1.0, degree=3)
        svm.fit(X_scaled, y)
        
        # Plotting
        ax = axes[i // 2, i % 2]
        
        # Plot data points
        ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
                  c='red', label='Class 1', alpha=0.6, s=30)
        ax.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], 
                  c='blue', label='Class -1', alpha=0.6, s=30)
        
        # Plot decision boundary
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='black', linewidths=2)
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, 
                   colors=['blue', 'white', 'red'])
        
        # Highlight support vectors
        if svm.support_vectors is not None:
            ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                      s=100, linewidth=2, facecolors='none', edgecolors='k')
        
        # Calculate metrics
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        n_sv = len(svm.support_vectors)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{kernel.upper()} Kernel\nAccuracy: {accuracy:.3f}, SVs: {n_sv}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Summary:")
    for kernel in kernels:
        svm = KernelSVM(C=1.0, kernel=kernel, gamma=1.0, degree=3)
        svm.fit(X_scaled, y)
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        n_sv = len(svm.support_vectors)
        print(f"{kernel.upper():>10} kernel: Accuracy = {accuracy:.3f}, Support Vectors = {n_sv}")
    
    return kernels, X_scaled, y


def demonstrate_kernel_functions():
    """Demonstrate different kernel functions and their properties"""
    print("\n=== Kernel Functions Demonstration ===\n")
    
    # Generate sample data
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    
    # Test different kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_values = [0.1, 1.0, 10.0]
    
    print("Kernel function values for x1 = [1, 2], x2 = [3, 4]:")
    print("-" * 60)
    
    for kernel in kernels:
        print(f"\n{kernel.upper()} Kernel:")
        for gamma in gamma_values:
            svm = KernelSVM(kernel=kernel, gamma=gamma, degree=3)
            k_value = svm.kernel_function(x1, x2)[0, 0]
            print(f"  γ = {gamma:>4}: K(x1, x2) = {k_value:.4f}")
    
    # Visualize kernel functions
    x = np.linspace(-3, 3, 100)
    y = np.zeros_like(x)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, kernel in enumerate(kernels):
        ax = axes[i // 2, i % 2]
        
        # Compute kernel values
        X1 = np.column_stack([x, y])
        X2 = np.array([[0, 0]])  # Reference point at origin
        
        svm = KernelSVM(kernel=kernel, gamma=1.0, degree=3)
        k_values = svm.kernel_function(X1, X2).flatten()
        
        ax.plot(x, k_values, 'b-', linewidth=2)
        ax.set_xlabel('Distance from origin')
        ax.set_ylabel('Kernel value')
        ax.set_title(f'{kernel.upper()} Kernel')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.show()


def demonstrate_parameter_effects():
    """Demonstrate the effects of different kernel parameters"""
    print("\n=== Parameter Effects ===\n")
    
    # Generate data
    X, y = generate_nonlinear_data('circles', n_samples=100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different gamma values for RBF kernel
    gamma_values = [0.1, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, gamma in enumerate(gamma_values):
        # Fit SVM
        svm = KernelSVM(C=1.0, kernel='rbf', gamma=gamma)
        svm.fit(X_scaled, y)
        
        # Plotting
        ax = axes[i // 2, i % 2]
        
        # Plot data points
        ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
                  c='red', alpha=0.6, s=30)
        ax.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], 
                  c='blue', alpha=0.6, s=30)
        
        # Plot decision boundary
        x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
        y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='black', linewidths=2)
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, 
                   colors=['blue', 'white', 'red'])
        
        # Highlight support vectors
        if svm.support_vectors is not None:
            ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                      s=100, linewidth=2, facecolors='none', edgecolors='k')
        
        # Calculate metrics
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        n_sv = len(svm.support_vectors)
        
        ax.set_title(f'RBF Kernel, γ = {gamma}\nAccuracy: {accuracy:.3f}, SVs: {n_sv}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("RBF Kernel Parameter Analysis:")
    for gamma in gamma_values:
        svm = KernelSVM(C=1.0, kernel='rbf', gamma=gamma)
        svm.fit(X_scaled, y)
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        n_sv = len(svm.support_vectors)
        print(f"  γ = {gamma:>6}: Accuracy = {accuracy:.3f}, Support Vectors = {n_sv}")


def demonstrate_cross_validation():
    """Demonstrate cross-validation for kernel selection and parameter tuning"""
    print("\n=== Cross-Validation for Kernel Selection ===\n")
    
    # Generate data
    X, y = generate_nonlinear_data('circles', n_samples=200)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grids for different kernels
    param_grids = {
        'linear': {'C': [0.1, 1, 10, 100]},
        'poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': [0.1, 1, 10]},
        'rbf': {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10]},
        'sigmoid': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    }
    
    best_scores = {}
    best_params = {}
    
    for kernel, param_grid in param_grids.items():
        print(f"Testing {kernel.upper()} kernel...")
        svm = SVC(kernel=kernel, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_scaled, y)
        
        best_scores[kernel] = grid_search.best_score_
        best_params[kernel] = grid_search.best_params_
        
        print(f"  Best score: {grid_search.best_score_:.3f}")
        print(f"  Best parameters: {grid_search.best_params_}")
    
    # Find best kernel
    best_kernel = max(best_scores, key=best_scores.get)
    print(f"\nBest kernel: {best_kernel.upper()} with score {best_scores[best_kernel]:.3f}")
    
    # Plot results
    kernels = list(best_scores.keys())
    scores = list(best_scores.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(kernels, scores, color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Kernel Type')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Kernel Performance Comparison')
    plt.ylim(0, 1)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_scores, best_params


def demonstrate_representer_theorem():
    """Demonstrate the representer theorem in practice"""
    print("\n=== Representer Theorem Demonstration ===\n")
    
    # Generate data
    X, y = generate_nonlinear_data('circles', n_samples=50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit kernel SVM
    svm = KernelSVM(C=1.0, kernel='rbf', gamma=1.0)
    svm.fit(X_scaled, y)
    
    # Test points
    test_points = np.array([[0, 0], [1, 1], [-1, -1], [0.5, -0.5]])
    
    print("Representer Theorem Verification:")
    print("f(x) = Σ α_i K(x_i, x) + β_0")
    print("-" * 50)
    
    for i, test_point in enumerate(test_points):
        # Compute decision function using representer form
        support_vector_lambdas = svm.lambda_values[svm.lambda_values > 1e-5]
        support_vector_y = np.array([y_val for j, y_val in enumerate(svm.lambda_values) if svm.lambda_values[j] > 1e-5])
        
        # Compute kernel values
        K = svm.kernel_function(test_point.reshape(1, -1), svm.support_vectors)
        
        # Compute decision function
        decision_value = np.sum(support_vector_lambdas.reshape(-1, 1) * 
                               support_vector_y.reshape(-1, 1) * K.T, axis=0) + svm.beta_0
        
        print(f"Test point {i+1} {test_point}: f(x) = {decision_value[0]:.4f}")
    
    # Visualize the representer form
    print(f"\nNumber of support vectors: {len(svm.support_vectors)}")
    print(f"Total training points: {len(X_scaled)}")
    print(f"Sparsity ratio: {len(svm.support_vectors)/len(X_scaled):.3f}")


def demonstrate_advantages_limitations():
    """Demonstrate advantages and limitations of kernel SVM"""
    print("\n=== Advantages and Limitations ===\n")
    
    # Generate different types of data
    datasets = {
        'Linear': generate_nonlinear_data('circles', n_samples=100, random_state=42),
        'Nonlinear': generate_nonlinear_data('moons', n_samples=100, random_state=42),
        'XOR': generate_nonlinear_data('xor', n_samples=100, random_state=42)
    }
    
    kernels = ['linear', 'rbf']
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"{name} Data:")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results[name] = {}
        
        for kernel in kernels:
            # Fit SVM
            svm = KernelSVM(C=1.0, kernel=kernel, gamma=1.0)
            svm.fit(X_scaled, y)
            
            # Calculate metrics
            accuracy = accuracy_score(y, svm.predict(X_scaled))
            n_sv = len(svm.support_vectors)
            
            results[name][kernel] = {
                'accuracy': accuracy,
                'n_support_vectors': n_sv
            }
            
            print(f"  {kernel.upper():>8} kernel: Accuracy = {accuracy:.3f}, SVs = {n_sv}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    datasets_list = list(results.keys())
    linear_accuracies = [results[d]['linear']['accuracy'] for d in datasets_list]
    rbf_accuracies = [results[d]['rbf']['accuracy'] for d in datasets_list]
    
    x = np.arange(len(datasets_list))
    width = 0.35
    
    axes[0].bar(x - width/2, linear_accuracies, width, label='Linear', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, rbf_accuracies, width, label='RBF', color='red', alpha=0.7)
    axes[0].set_xlabel('Dataset Type')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets_list)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Support vectors comparison
    linear_svs = [results[d]['linear']['n_support_vectors'] for d in datasets_list]
    rbf_svs = [results[d]['rbf']['n_support_vectors'] for d in datasets_list]
    
    axes[1].bar(x - width/2, linear_svs, width, label='Linear', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, rbf_svs, width, label='RBF', color='red', alpha=0.7)
    axes[1].set_xlabel('Dataset Type')
    axes[1].set_ylabel('Number of Support Vectors')
    axes[1].set_title('Support Vectors Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets_list)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Main demonstration of nonlinear SVM concepts"""
    print("Support Vector Machines: Nonlinear SVMs Implementation")
    print("=" * 60)
    
    # 1. Kernel comparison
    print("\n1. Kernel Comparison:")
    kernels, X, y = demonstrate_kernel_comparison()
    
    # 2. Kernel functions demonstration
    print("\n2. Kernel Functions Demonstration:")
    demonstrate_kernel_functions()
    
    # 3. Parameter effects
    print("\n3. Parameter Effects:")
    demonstrate_parameter_effects()
    
    # 4. Cross-validation for kernel selection
    print("\n4. Cross-Validation for Kernel Selection:")
    best_scores, best_params = demonstrate_cross_validation()
    
    # 5. Representer theorem demonstration
    print("\n5. Representer Theorem Demonstration:")
    demonstrate_representer_theorem()
    
    # 6. Advantages and limitations
    print("\n6. Advantages and Limitations:")
    advantages_results = demonstrate_advantages_limitations()
    
    print("\n=== Key Insights ===")
    print("1. Kernel trick allows nonlinear classification without explicit feature transformation")
    print("2. RBF kernel is most commonly used and works well for most problems")
    print("3. Parameter γ controls the influence of each training point")
    print("4. Cross-validation is essential for kernel and parameter selection")
    print("5. Representer theorem ensures finite representation using support vectors")
    print("6. Kernel SVM provides sparse solutions with only support vectors mattering")
    print("7. Computational cost scales with number of support vectors")
    print("8. Kernel selection depends on data characteristics and domain knowledge")
    
    return {
        'kernels': kernels,
        'X': X,
        'y': y,
        'best_scores': best_scores,
        'best_params': best_params,
        'advantages_results': advantages_results
    }


if __name__ == "__main__":
    main()
