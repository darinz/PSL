import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import cvxopt
from cvxopt import matrix, solvers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns


class SVM:
    """Support Vector Machine implementation from scratch for separable case"""
    
    def __init__(self, C=1.0):
        self.C = C
        self.support_vectors = None
        self.lambda_values = None
        self.beta = None
        self.beta_0 = None
        
    def fit(self, X, y):
        """Fit SVM using quadratic programming"""
        n_samples, n_features = X.shape
        
        # Prepare the quadratic programming problem
        P = matrix(np.outer(y, y) * np.dot(X, X.T))
        q = matrix(-np.ones(n_samples))
        G = matrix(-np.eye(n_samples))
        h = matrix(np.zeros(n_samples))
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
        
        # Compute beta
        self.beta = np.sum(support_vector_lambdas.reshape(-1, 1) * 
                          support_vector_y.reshape(-1, 1) * self.support_vectors, axis=0)
        
        # Compute beta_0
        self.beta_0 = np.mean(support_vector_y - 
                             np.dot(self.support_vectors, self.beta))
        
    def predict(self, X):
        """Predict class labels"""
        return np.sign(np.dot(X, self.beta) + self.beta_0)
    
    def decision_function(self, X):
        """Compute decision function values"""
        return np.dot(X, self.beta) + self.beta_0
    
    def get_margin(self):
        """Compute the margin width"""
        return 2 / np.linalg.norm(self.beta)


def generate_separable_data(n_samples=100, centers=2, random_state=42):
    """Generate linearly separable data"""
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
                     random_state=random_state, cluster_std=1.0)
    y = 2 * y - 1  # Convert to {-1, 1}
    return X, y


def visualize_svm_decision_boundary(X, y, svm, title="SVM Decision Boundary"):
    """Visualize SVM decision boundary with support vectors"""
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
               c='red', label='Class 1', alpha=0.6, s=50)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], 
               c='blue', label='Class -1', alpha=0.6, s=50)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margin
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, 
               colors=['blue', 'black', 'red'], linewidths=[1, 2, 1])
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


def demonstrate_separable_case():
    """Demonstrate SVM on linearly separable data"""
    print("=== SVM Separable Case Demonstration ===\n")
    
    # Generate separable data
    X, y = generate_separable_data(n_samples=100, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Class distribution: {np.bincount(y + 1)}")
    
    # Fit SVM from scratch
    svm_scratch = SVM()
    svm_scratch.fit(X_scaled, y)
    
    # Fit sklearn SVM for comparison
    svm_sklearn = SVC(kernel='linear', C=1.0, random_state=42)
    svm_sklearn.fit(X_scaled, y)
    
    # Compare results
    print(f"\nScratch SVM Results:")
    print(f"Number of support vectors: {len(svm_scratch.support_vectors)}")
    print(f"Beta: {svm_scratch.beta}")
    print(f"Beta_0: {svm_scratch.beta_0:.4f}")
    print(f"Margin width: {svm_scratch.get_margin():.4f}")
    
    print(f"\nSklearn SVM Results:")
    print(f"Number of support vectors: {len(svm_sklearn.support_vectors_)}")
    print(f"Beta: {svm_sklearn.coef_[0]}")
    print(f"Beta_0: {svm_sklearn.intercept_[0]:.4f}")
    
    # Compare predictions
    y_pred_scratch = svm_scratch.predict(X_scaled)
    y_pred_sklearn = svm_sklearn.predict(X_scaled)
    
    accuracy_scratch = accuracy_score(y, y_pred_scratch)
    accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
    
    print(f"\nAccuracy Comparison:")
    print(f"Scratch SVM: {accuracy_scratch:.4f}")
    print(f"Sklearn SVM: {accuracy_sklearn:.4f}")
    
    # Visualize
    visualize_svm_decision_boundary(X_scaled, y, svm_scratch, 
                                   "SVM Decision Boundary (Scratch Implementation)")
    
    return svm_scratch, svm_sklearn, X_scaled, y


def demonstrate_kkt_conditions(X, y, svm):
    """Demonstrate KKT conditions verification"""
    print("\n=== KKT Conditions Verification ===\n")
    
    # Calculate decision function values
    decision_values = svm.decision_function(X)
    
    # Check primal feasibility: y_i * f(x_i) >= 1
    primal_violations = y * decision_values < 1
    print(f"Primal feasibility violations: {np.sum(primal_violations)}")
    
    # Check complementary slackness: λ_i * (y_i * f(x_i) - 1) = 0
    margin_violations = y * decision_values - 1
    complementary_slackness = svm.lambda_values * margin_violations
    
    print(f"Complementary slackness check:")
    print(f"  Max violation: {np.max(np.abs(complementary_slackness)):.6f}")
    print(f"  Mean violation: {np.mean(np.abs(complementary_slackness)):.6f}")
    
    # Check support vector conditions
    support_vector_indices = svm.lambda_values > 1e-5
    non_support_vector_indices = ~support_vector_indices
    
    print(f"\nSupport Vector Analysis:")
    print(f"  Total points: {len(X)}")
    print(f"  Support vectors: {np.sum(support_vector_indices)}")
    print(f"  Non-support vectors: {np.sum(non_support_vector_indices)}")
    
    # Check that support vectors lie on margin
    sv_margin_values = margin_violations[support_vector_indices]
    print(f"  Support vector margin violations:")
    print(f"    Max: {np.max(np.abs(sv_margin_values)):.6f}")
    print(f"    Mean: {np.mean(np.abs(sv_margin_values)):.6f}")
    
    # Check that non-support vectors are beyond margin
    non_sv_margin_values = margin_violations[non_support_vector_indices]
    print(f"  Non-support vector margin values:")
    print(f"    Min: {np.min(non_sv_margin_values):.6f}")
    print(f"    Mean: {np.mean(non_sv_margin_values):.6f}")


def demonstrate_dual_formulation(X, y, svm):
    """Demonstrate dual formulation properties"""
    print("\n=== Dual Formulation Analysis ===\n")
    
    # Verify dual objective value
    dual_objective = np.sum(svm.lambda_values) - 0.5 * np.sum(
        np.outer(svm.lambda_values, svm.lambda_values) * 
        np.outer(y, y) * np.dot(X, X.T)
    )
    
    # Verify primal objective value
    primal_objective = 0.5 * np.linalg.norm(svm.beta)**2
    
    print(f"Dual objective value: {dual_objective:.6f}")
    print(f"Primal objective value: {primal_objective:.6f}")
    print(f"Strong duality gap: {abs(dual_objective - primal_objective):.6f}")
    
    # Check dual constraints
    dual_constraint1 = np.sum(svm.lambda_values * y)  # Should be 0
    dual_constraint2 = np.sum(svm.lambda_values < 0)  # Should be 0
    
    print(f"\nDual constraint checks:")
    print(f"  Σ λ_i * y_i = 0: {abs(dual_constraint1):.6f}")
    print(f"  λ_i >= 0 violations: {dual_constraint2}")


def demonstrate_margin_analysis(X, y):
    """Demonstrate margin analysis with different data configurations"""
    print("\n=== Margin Analysis ===\n")
    
    # Test different data separations
    separations = [1.0, 2.0, 3.0, 4.0]
    results = []
    
    for sep in separations:
        # Generate data with different separations
        X_sep, y_sep = make_blobs(n_samples=100, centers=2, 
                                 random_state=42, cluster_std=1.0,
                                 center_box=(-sep, sep))
        y_sep = 2 * y_sep - 1
        
        # Scale data
        scaler = StandardScaler()
        X_sep_scaled = scaler.fit_transform(X_sep)
        
        # Fit SVM
        svm = SVM()
        svm.fit(X_sep_scaled, y_sep)
        
        # Calculate metrics
        margin = svm.get_margin()
        n_sv = len(svm.support_vectors)
        accuracy = accuracy_score(y_sep, svm.predict(X_sep_scaled))
        
        results.append({
            'separation': sep,
            'margin': margin,
            'n_support_vectors': n_sv,
            'accuracy': accuracy
        })
        
        print(f"Separation {sep}: Margin = {margin:.4f}, SVs = {n_sv}, Accuracy = {accuracy:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    separations = [r['separation'] for r in results]
    margins = [r['margin'] for r in results]
    n_svs = [r['n_support_vectors'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    axes[0].plot(separations, margins, 'bo-')
    axes[0].set_xlabel('Data Separation')
    axes[0].set_ylabel('Margin Width')
    axes[0].set_title('Margin vs Data Separation')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(separations, n_svs, 'ro-')
    axes[1].set_xlabel('Data Separation')
    axes[1].set_ylabel('Number of Support Vectors')
    axes[1].set_title('Support Vectors vs Data Separation')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(separations, accuracies, 'go-')
    axes[2].set_xlabel('Data Separation')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy vs Data Separation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_computational_complexity():
    """Demonstrate computational complexity analysis"""
    print("\n=== Computational Complexity Analysis ===\n")
    
    # Test different dataset sizes
    sizes = [50, 100, 200, 300, 400]
    results = []
    
    for size in sizes:
        print(f"Testing with {size} samples...")
        
        # Generate data
        X, y = generate_separable_data(n_samples=size, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time the fitting
        import time
        start_time = time.time()
        svm = SVM()
        svm.fit(X_scaled, y)
        fit_time = time.time() - start_time
        
        # Time the prediction
        start_time = time.time()
        y_pred = svm.predict(X_scaled)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        n_sv = len(svm.support_vectors)
        accuracy = accuracy_score(y, y_pred)
        
        results.append({
            'size': size,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'n_support_vectors': n_sv,
            'accuracy': accuracy
        })
        
        print(f"  Fit time: {fit_time:.4f}s")
        print(f"  Predict time: {predict_time:.4f}s")
        print(f"  Support vectors: {n_sv}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Plot complexity analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sizes = [r['size'] for r in results]
    fit_times = [r['fit_time'] for r in results]
    predict_times = [r['predict_time'] for r in results]
    n_svs = [r['n_support_vectors'] for r in results]
    
    # Fit time vs dataset size (should be O(n^3))
    axes[0, 0].plot(sizes, fit_times, 'bo-')
    axes[0, 0].set_xlabel('Dataset Size')
    axes[0, 0].set_ylabel('Fit Time (s)')
    axes[0, 0].set_title('Fit Time vs Dataset Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Predict time vs support vectors (should be O(n_sv * p))
    axes[0, 1].plot(n_svs, predict_times, 'ro-')
    axes[0, 1].set_xlabel('Number of Support Vectors')
    axes[0, 1].set_ylabel('Predict Time (s)')
    axes[0, 1].set_title('Predict Time vs Support Vectors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Support vectors vs dataset size
    axes[1, 0].plot(sizes, n_svs, 'go-')
    axes[1, 0].set_xlabel('Dataset Size')
    axes[1, 0].set_ylabel('Number of Support Vectors')
    axes[1, 0].set_title('Support Vectors vs Dataset Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-log plot for fit time (to check O(n^3) complexity)
    axes[1, 1].loglog(sizes, fit_times, 'mo-')
    axes[1, 1].set_xlabel('Dataset Size')
    axes[1, 1].set_ylabel('Fit Time (s)')
    axes[1, 1].set_title('Fit Time Complexity (Log-Log)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_theoretical_properties():
    """Demonstrate theoretical properties of SVM"""
    print("\n=== Theoretical Properties ===\n")
    
    # Generate data
    X, y = generate_separable_data(n_samples=200, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit SVM
    svm = SVM()
    svm.fit(X_scaled, y)
    
    print("1. Maximum Margin Property:")
    margin = svm.get_margin()
    print(f"   Margin width: {margin:.4f}")
    
    # Check that all points are correctly classified with margin >= 1
    decision_values = svm.decision_function(X_scaled)
    margin_violations = y * decision_values < 1
    print(f"   Points violating margin constraint: {np.sum(margin_violations)}")
    
    print("\n2. Support Vector Property:")
    print(f"   Support vectors lie exactly on margin boundaries")
    sv_margin_values = y[svm.lambda_values > 1e-5] * decision_values[svm.lambda_values > 1e-5]
    print(f"   Support vector margin values: {sv_margin_values}")
    
    print("\n3. Sparsity Property:")
    n_sv = len(svm.support_vectors)
    n_total = len(X_scaled)
    print(f"   Total points: {n_total}")
    print(f"   Support vectors: {n_sv}")
    print(f"   Sparsity ratio: {n_sv/n_total:.3f}")
    
    print("\n4. Dual Formulation:")
    # Verify that beta can be reconstructed from support vectors
    beta_reconstructed = np.sum(
        svm.lambda_values.reshape(-1, 1) * y.reshape(-1, 1) * X_scaled, axis=0
    )
    beta_error = np.linalg.norm(svm.beta - beta_reconstructed)
    print(f"   Beta reconstruction error: {beta_error:.6f}")
    
    return svm


def main():
    """Main demonstration of SVM separable case"""
    print("Support Vector Machines: Separable Case Implementation")
    print("=" * 60)
    
    # 1. Basic separable case demonstration
    print("\n1. Basic Separable Case Demonstration:")
    svm_scratch, svm_sklearn, X, y = demonstrate_separable_case()
    
    # 2. KKT conditions verification
    print("\n2. KKT Conditions Verification:")
    demonstrate_kkt_conditions(X, y, svm_scratch)
    
    # 3. Dual formulation analysis
    print("\n3. Dual Formulation Analysis:")
    demonstrate_dual_formulation(X, y, svm_scratch)
    
    # 4. Margin analysis
    print("\n4. Margin Analysis:")
    margin_results = demonstrate_margin_analysis(X, y)
    
    # 5. Computational complexity
    print("\n5. Computational Complexity Analysis:")
    complexity_results = demonstrate_computational_complexity()
    
    # 6. Theoretical properties
    print("\n6. Theoretical Properties:")
    theoretical_svm = demonstrate_theoretical_properties()
    
    print("\n=== Key Insights ===")
    print("1. SVM finds the optimal hyperplane with maximum margin")
    print("2. Only support vectors determine the decision boundary")
    print("3. KKT conditions provide theoretical foundation")
    print("4. Dual formulation enables kernel trick")
    print("5. Computational complexity is O(n³) for training")
    print("6. Prediction complexity is O(n_sv * p)")
    
    return {
        'svm_scratch': svm_scratch,
        'svm_sklearn': svm_sklearn,
        'X': X,
        'y': y,
        'margin_results': margin_results,
        'complexity_results': complexity_results,
        'theoretical_svm': theoretical_svm
    }


if __name__ == "__main__":
    main()
