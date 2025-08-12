import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC, SVR
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.datasets import make_classification, make_circles, make_blobs
from sklearn.model_selection import train_test_split
import seaborn as sns


class RKHS:
    """Reproducing Kernel Hilbert Space implementation"""
    
    def __init__(self, kernel='rbf', gamma=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.alpha = None
        
    def kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'rbf':
            # RBF kernel: K(x,y) = exp(-gamma ||x-y||^2)
            dist_sq = cdist(X1, X2, metric='sqeuclidean')
            return np.exp(-self.gamma * dist_sq)
        elif self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y, lambda_reg=0.1):
        """Fit using representer theorem"""
        self.X_train = X
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.kernel_function(X, X)
        
        # Add regularization
        K_reg = K + lambda_reg * np.eye(n_samples)
        
        # Solve linear system: K_reg * alpha = y
        self.alpha = np.linalg.solve(K_reg, y)
        
    def predict(self, X):
        """Make predictions"""
        if self.X_train is None:
            raise ValueError("Model not fitted yet")
        
        K_test = self.kernel_function(X, self.X_train)
        return np.dot(K_test, self.alpha)


def demonstrate_rkhs():
    """Demonstrate RKHS concepts"""
    print("=== RKHS Demonstration ===\n")
    
    # Generate example data
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(100)
    
    # Fit RKHS model
    rkhs = RKHS(kernel='rbf', gamma=1.0)
    rkhs.fit(X, y, lambda_reg=0.01)
    
    # Test predictions
    X_test = np.random.randn(20, 2)
    y_pred = rkhs.predict(X_test)
    
    print("Predictions shape:", y_pred.shape)
    print("Representer theorem form: f(x) = Σ α_i K(x_i, x)")
    print(f"Number of training points: {len(X)}")
    print(f"Number of coefficients: {len(rkhs.alpha)}")
    
    # Visualize the fit
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.colorbar(label='y')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Training Data')
    
    # Predictions on test data
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', s=50)
    plt.colorbar(label='Predicted y')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Test Predictions')
    
    plt.tight_layout()
    plt.show()
    
    return rkhs


def check_kernel_properties(K):
    """Check if K satisfies kernel properties"""
    n = K.shape[0]
    
    # Check symmetry
    is_symmetric = np.allclose(K, K.T)
    print(f"Symmetric: {is_symmetric}")
    
    # Check positive semi-definiteness
    eigenvals = eigh(K, eigvals_only=True)
    is_psd = np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    print(f"Positive semi-definite: {is_psd}")
    print(f"Eigenvalues: {eigenvals[:5]}...")  # Show first 5
    
    # Check trace
    trace = np.trace(K)
    print(f"Trace: {trace:.3f}")
    
    return is_symmetric and is_psd


def demonstrate_mercer_theorem():
    """Demonstrate Mercer's theorem and kernel properties"""
    print("\n=== Mercer's Theorem Demonstration ===\n")
    
    # Test different kernels
    X = np.random.randn(50, 3)
    
    # Linear kernel
    K_linear = np.dot(X, X.T)
    print("Linear kernel:")
    check_kernel_properties(K_linear)
    
    # RBF kernel
    gamma = 1.0
    dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
    K_rbf = np.exp(-gamma * dist_sq)
    print("\nRBF kernel:")
    check_kernel_properties(K_rbf)
    
    # Polynomial kernel
    K_poly = (np.dot(X, X.T) + 1)**2
    print("\nPolynomial kernel:")
    check_kernel_properties(K_poly)
    
    # Visualize kernel matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(K_linear, cmap='viridis')
    axes[0].set_title('Linear Kernel Matrix')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(K_rbf, cmap='viridis')
    axes[1].set_title('RBF Kernel Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(K_poly, cmap='viridis')
    axes[2].set_title('Polynomial Kernel Matrix')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return K_linear, K_rbf, K_poly


def ovo_svm_example():
    """Demonstrate One-vs-One multi-class SVM"""
    print("\n=== One-vs-One SVM Example ===\n")
    
    # Generate multi-class data
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                             n_clusters_per_class=1, n_redundant=0, 
                             random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Train OVO SVM
    ovo_svm = OneVsOneClassifier(SVC(kernel='rbf', random_state=42))
    ovo_svm.fit(X_train, y_train)
    
    # Evaluate
    train_score = ovo_svm.score(X_train, y_train)
    test_score = ovo_svm.score(X_test, y_test)
    
    print(f"OVO SVM - Train accuracy: {train_score:.3f}")
    print(f"OVO SVM - Test accuracy: {test_score:.3f}")
    print(f"Number of binary classifiers: {len(ovo_svm.estimators_)}")
    
    # Visualize decision boundaries
    plt.figure(figsize=(10, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')
    plt.colorbar(scatter)
    
    # Test data
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Test Data')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return ovo_svm


def ovr_svm_example():
    """Demonstrate One-vs-Rest multi-class SVM"""
    print("\n=== One-vs-Rest SVM Example ===\n")
    
    # Generate multi-class data
    X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                             n_clusters_per_class=1, n_redundant=0, 
                             random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Train OVR SVM
    ovr_svm = OneVsRestClassifier(SVC(kernel='rbf', random_state=42))
    ovr_svm.fit(X_train, y_train)
    
    # Evaluate
    train_score = ovr_svm.score(X_train, y_train)
    test_score = ovr_svm.score(X_test, y_test)
    
    print(f"OVR SVM - Train accuracy: {train_score:.3f}")
    print(f"OVR SVM - Test accuracy: {test_score:.3f}")
    print(f"Number of binary classifiers: {len(ovr_svm.estimators_)}")
    
    return ovr_svm


def svr_example():
    """Demonstrate Support Vector Regression"""
    print("\n=== Support Vector Regression Example ===\n")
    
    # Generate regression data
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
    
    # Fit SVR models with different kernels
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_linear = SVR(kernel='linear', C=100, epsilon=0.1)
    svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1)
    
    # Fit models
    svr_rbf.fit(X, y)
    svr_linear.fit(X, y)
    svr_poly.fit(X, y)
    
    # Predictions
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_rbf = svr_rbf.predict(X_test)
    y_linear = svr_linear.predict(X_test)
    y_poly = svr_poly.predict(X_test)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, c='black', label='data', alpha=0.6)
    plt.plot(X_test, y_rbf, c='red', label='RBF', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR with RBF Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X, y, c='black', label='data', alpha=0.6)
    plt.plot(X_test, y_linear, c='blue', label='Linear', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR with Linear Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, c='black', label='data', alpha=0.6)
    plt.plot(X_test, y_poly, c='green', label='Polynomial', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR with Polynomial Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return svr_rbf, svr_linear, svr_poly


def simplified_smo(X, y, C=1.0, max_iter=1000, tol=1e-3):
    """Simplified SMO algorithm"""
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)
    b = 0.0
    
    # Precompute kernel matrix
    K = np.dot(X, X.T)
    
    for iteration in range(max_iter):
        alpha_pairs_changed = 0
        
        for i in range(n_samples):
            # Calculate error
            Ei = np.sum(alpha * y * K[i, :]) + b - y[i]
            
            # Check KKT conditions
            if ((y[i] * Ei < -tol and alpha[i] < C) or 
                (y[i] * Ei > tol and alpha[i] > 0)):
                
                # Choose second alpha randomly
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                
                Ej = np.sum(alpha * y * K[j, :]) + b - y[j]
                
                # Save old alphas
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                
                # Compute bounds
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                # Compute eta
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                # Update alpha[j]
                alpha[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                alpha[j] = np.clip(alpha[j], L, H)
                
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                # Update alpha[i]
                alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
                
                # Update b
                b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                b = (b1 + b2) / 2
                
                alpha_pairs_changed += 1
        
        if alpha_pairs_changed == 0:
            break
    
    return alpha, b


def demonstrate_smo():
    """Demonstrate SMO algorithm"""
    print("\n=== SMO Algorithm Demonstration ===\n")
    
    # Generate data
    X = np.random.randn(100, 2)
    y = np.sign(X[:, 0] + X[:, 1])
    
    # Run SMO
    alpha, b = simplified_smo(X, y, C=1.0)
    
    print(f"Converged with {len(alpha[alpha > 1e-5])} support vectors")
    print(f"Bias term: {b:.4f}")
    
    # Visualize support vectors
    support_vector_indices = alpha > 1e-5
    support_vectors = X[support_vector_indices]
    
    plt.figure(figsize=(10, 5))
    
    # All data points
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, s=30)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('All Data Points')
    
    # Support vectors highlighted
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.3, s=20)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               c='red', s=100, marker='o', facecolors='none', linewidth=2, label='Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Support Vectors Highlighted')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return alpha, b


def kernel_approximation_example():
    """Demonstrate kernel approximation methods"""
    print("\n=== Kernel Approximation Example ===\n")
    
    # Generate data
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42)
    
    # Standard SVM
    svm_standard = SVC(kernel='rbf', gamma=1.0, random_state=42)
    svm_standard.fit(X_train, y_train)
    score_standard = svm_standard.score(X_test, y_test)
    
    # RBF approximation
    rbf_feature = RBFSampler(gamma=1.0, n_components=100, random_state=42)
    X_train_rbf = rbf_feature.fit_transform(X_train)
    X_test_rbf = rbf_feature.transform(X_test)
    
    svm_rbf_approx = SVC(kernel='linear', random_state=42)
    svm_rbf_approx.fit(X_train_rbf, y_train)
    score_rbf_approx = svm_rbf_approx.score(X_test_rbf, y_test)
    
    # Nystroem approximation
    nystroem = Nystroem(kernel='rbf', gamma=1.0, n_components=100, random_state=42)
    X_train_nystroem = nystroem.fit_transform(X_train)
    X_test_nystroem = nystroem.transform(X_test)
    
    svm_nystroem = SVC(kernel='linear', random_state=42)
    svm_nystroem.fit(X_train_nystroem, y_train)
    score_nystroem = svm_nystroem.score(X_test_nystroem, y_test)
    
    print(f"Standard SVM accuracy: {score_standard:.3f}")
    print(f"RBF approximation accuracy: {score_rbf_approx:.3f}")
    print(f"Nystroem approximation accuracy: {score_nystroem:.3f}")
    
    # Compare computational complexity
    print(f"\nFeature dimensions:")
    print(f"  Original: {X_train.shape[1]}")
    print(f"  RBF approximation: {X_train_rbf.shape[1]}")
    print(f"  Nystroem approximation: {X_train_nystroem.shape[1]}")
    
    return svm_standard, svm_rbf_approx, svm_nystroem


def margin_analysis(X, y, svm_model):
    """Analyze margin and support vectors"""
    # Get support vectors
    support_vectors = svm_model.support_vectors_
    support_vector_indices = svm_model.support_
    
    # Compute margin
    w = svm_model.coef_[0]
    margin = 2 / np.linalg.norm(w)
    
    # Compute distances to decision boundary
    decision_values = svm_model.decision_function(X)
    distances = np.abs(decision_values) / np.linalg.norm(w)
    
    # Find minimum margin
    min_margin = np.min(distances)
    
    print(f"Margin: {margin:.4f}")
    print(f"Minimum margin: {min_margin:.4f}")
    print(f"Number of support vectors: {len(support_vectors)}")
    print(f"Support vector ratio: {len(support_vectors)/len(X):.3f}")
    
    return margin, min_margin, support_vectors


def demonstrate_margin_analysis():
    """Demonstrate margin analysis"""
    print("\n=== Margin Analysis ===\n")
    
    # Generate data
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Fit SVM
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X, y)
    
    # Analyze margin
    margin, min_margin, support_vectors = margin_analysis(X, y, svm)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Decision boundary and margin
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, s=30)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               c='red', s=100, marker='o', facecolors='none', linewidth=2, label='Support Vectors')
    
    # Plot decision boundary
    w = svm.coef_[0]
    b = svm.intercept_[0]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = w[0] * xx + w[1] * yy + b
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linewidths=1, linestyles='--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Margin: {margin:.3f}')
    plt.legend()
    
    # Margin distribution
    plt.subplot(1, 2, 2)
    decision_values = svm.decision_function(X)
    distances = np.abs(decision_values) / np.linalg.norm(w)
    
    plt.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(margin/2, color='red', linestyle='--', label='Margin/2')
    plt.xlabel('Distance to Decision Boundary')
    plt.ylabel('Frequency')
    plt.title('Margin Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return margin, min_margin, support_vectors


def main():
    """Main demonstration of SVM appendix concepts"""
    print("Support Vector Machines: Appendix Implementation")
    print("=" * 60)
    
    # 1. RKHS demonstration
    print("\n1. RKHS Demonstration:")
    rkhs_model = demonstrate_rkhs()
    
    # 2. Mercer's theorem demonstration
    print("\n2. Mercer's Theorem Demonstration:")
    K_linear, K_rbf, K_poly = demonstrate_mercer_theorem()
    
    # 3. Multi-class SVM examples
    print("\n3. Multi-class SVM Examples:")
    ovo_model = ovo_svm_example()
    ovr_model = ovr_svm_example()
    
    # 4. Support Vector Regression
    print("\n4. Support Vector Regression:")
    svr_models = svr_example()
    
    # 5. SMO algorithm
    print("\n5. SMO Algorithm:")
    alpha, b = demonstrate_smo()
    
    # 6. Kernel approximation
    print("\n6. Kernel Approximation:")
    approx_models = kernel_approximation_example()
    
    # 7. Margin analysis
    print("\n7. Margin Analysis:")
    margin_results = demonstrate_margin_analysis()
    
    print("\n=== Key Insights ===")
    print("1. RKHS provides theoretical foundation for kernel methods")
    print("2. Mercer's theorem ensures valid kernel functions")
    print("3. Multi-class SVM extends binary classification")
    print("4. SVR applies SVM principles to regression")
    print("5. SMO enables efficient SVM training")
    print("6. Kernel approximation scales to large datasets")
    print("7. Margin analysis provides generalization insights")
    print("8. Support vectors determine the optimal solution")
    
    return {
        'rkhs_model': rkhs_model,
        'kernel_matrices': (K_linear, K_rbf, K_poly),
        'multiclass_models': (ovo_model, ovr_model),
        'svr_models': svr_models,
        'smo_results': (alpha, b),
        'approximation_models': approx_models,
        'margin_results': margin_results
    }


if __name__ == "__main__":
    main()
