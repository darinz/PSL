import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import cvxopt
from cvxopt import matrix, solvers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns


class SoftMarginSVM:
    """Support Vector Machine implementation from scratch for non-separable case"""
    
    def __init__(self, C=1.0):
        self.C = C
        self.support_vectors = None
        self.lambda_values = None
        self.beta = None
        self.beta_0 = None
        
    def fit(self, X, y):
        """Fit soft margin SVM using quadratic programming"""
        n_samples, n_features = X.shape
        
        # Prepare the quadratic programming problem
        P = matrix(np.outer(y, y) * np.dot(X, X.T))
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
    
    def get_slack_variables(self, X, y):
        """Compute slack variables for given data"""
        decision_values = self.decision_function(X)
        slack = np.maximum(0, 1 - y * decision_values)
        return slack


def generate_nonseparable_data(n_samples=100, cluster_std=1.5, noise_ratio=0.1, random_state=42):
    """Generate non-separable data with controlled overlap"""
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std, 
                     random_state=random_state)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Add noise to make it non-separable
    np.random.seed(random_state)
    noise_indices = np.random.choice(len(X), size=int(n_samples * noise_ratio), replace=False)
    y[noise_indices] = -y[noise_indices]
    
    return X, y


def visualize_soft_margin_svm(X, y, svm, title="Soft Margin SVM"):
    """Visualize soft margin SVM decision boundary"""
    plt.figure(figsize=(10, 8))
    
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


def demonstrate_soft_margin_svm():
    """Demonstrate soft margin SVM with different C values"""
    print("=== Soft Margin SVM Demonstration ===\n")
    
    # Generate non-separable data
    X, y = generate_nonseparable_data(n_samples=100, cluster_std=1.5, noise_ratio=0.1)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"Class distribution: {np.bincount(y + 1)}")
    
    # Compare different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, C in enumerate(C_values):
        # Fit SVM
        svm = SoftMarginSVM(C=C)
        svm.fit(X_scaled, y)
        
        # Plotting
        ax = axes[i // 2, i % 2]
        
        # Plot data points
        ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
                  c='red', label='Class 1', alpha=0.6, s=30)
        ax.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], 
                  c='blue', label='Class -1', alpha=0.6, s=30)
        
        # Plot decision boundary
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, 
                  colors=['blue', 'black', 'red'], linewidths=[1, 2, 1])
        ax.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.1, 
                   colors=['blue', 'white', 'red'])
        
        # Highlight support vectors
        if svm.support_vectors is not None:
            ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                      s=100, linewidth=2, facecolors='none', edgecolors='k')
        
        # Calculate metrics
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        margin = svm.get_margin()
        n_sv = len(svm.support_vectors)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'C={C}\nAccuracy: {accuracy:.3f}, Margin: {margin:.3f}, SVs: {n_sv}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Summary:")
    for C in C_values:
        svm = SoftMarginSVM(C=C)
        svm.fit(X_scaled, y)
        accuracy = accuracy_score(y, svm.predict(X_scaled))
        margin = svm.get_margin()
        n_sv = len(svm.support_vectors)
        print(f"C = {C:>6}: Accuracy = {accuracy:.3f}, Margin = {margin:.3f}, Support Vectors = {n_sv}")
    
    return C_values, X_scaled, y


def demonstrate_kkt_conditions(X, y, svm):
    """Demonstrate KKT conditions verification for soft margin SVM"""
    print("\n=== KKT Conditions Verification (Soft Margin) ===\n")
    
    # Calculate decision function values
    decision_values = svm.decision_function(X)
    
    # Check primal feasibility: y_i * f(x_i) >= 1 - ξ_i
    slack_variables = svm.get_slack_variables(X, y)
    primal_violations = y * decision_values < 1 - slack_variables
    print(f"Primal feasibility violations: {np.sum(primal_violations)}")
    
    # Check complementary slackness: λ_i * (y_i * f(x_i) - 1 + ξ_i) = 0
    margin_violations = y * decision_values - 1 + slack_variables
    complementary_slackness = svm.lambda_values * margin_violations
    
    print(f"Complementary slackness check:")
    print(f"  Max violation: {np.max(np.abs(complementary_slackness)):.6f}")
    print(f"  Mean violation: {np.mean(np.abs(complementary_slackness)):.6f}")
    
    # Check support vector classification
    support_vector_indices = svm.lambda_values > 1e-5
    non_support_vector_indices = ~support_vector_indices
    
    print(f"\nSupport Vector Classification:")
    print(f"  Total points: {len(X)}")
    print(f"  Support vectors: {np.sum(support_vector_indices)}")
    print(f"  Non-support vectors: {np.sum(non_support_vector_indices)}")
    
    # Classify support vectors
    margin_sv = (svm.lambda_values > 1e-5) & (svm.lambda_values < svm.C - 1e-5)
    non_margin_sv = (svm.lambda_values > 1e-5) & (svm.lambda_values >= svm.C - 1e-5)
    
    print(f"  Margin support vectors: {np.sum(margin_sv)}")
    print(f"  Non-margin support vectors: {np.sum(non_margin_sv)}")
    
    # Check dual constraints
    dual_constraint1 = np.sum(svm.lambda_values * y)  # Should be 0
    dual_constraint2 = np.sum(svm.lambda_values < 0)  # Should be 0
    dual_constraint3 = np.sum(svm.lambda_values > svm.C + 1e-5)  # Should be 0
    
    print(f"\nDual constraint checks:")
    print(f"  Σ λ_i * y_i = 0: {abs(dual_constraint1):.6f}")
    print(f"  λ_i >= 0 violations: {dual_constraint2}")
    print(f"  λ_i <= C violations: {dual_constraint3}")


def demonstrate_c_parameter_effects():
    """Demonstrate the effects of different C parameter values"""
    print("\n=== C Parameter Effects ===\n")
    
    # Generate data with different overlap levels
    overlap_levels = [0.05, 0.1, 0.2, 0.3]
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    fig, axes = plt.subplots(len(overlap_levels), len(C_values), figsize=(20, 16))
    
    for i, overlap in enumerate(overlap_levels):
        # Generate data with specific overlap
        X, y = generate_nonseparable_data(n_samples=100, cluster_std=1.5, 
                                        noise_ratio=overlap, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for j, C in enumerate(C_values):
            # Fit SVM
            svm = SoftMarginSVM(C=C)
            svm.fit(X_scaled, y)
            
            # Plotting
            ax = axes[i, j]
            
            # Plot data points
            ax.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], 
                      c='red', alpha=0.6, s=20)
            ax.scatter(X_scaled[y == -1][:, 0], X_scaled[y == -1][:, 1], 
                      c='blue', alpha=0.6, s=20)
            
            # Plot decision boundary
            x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
            y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, 
                      colors=['blue', 'black', 'red'], linewidths=[1, 2, 1])
            
            # Highlight support vectors
            if svm.support_vectors is not None:
                ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                          s=80, linewidth=1, facecolors='none', edgecolors='k')
            
            # Calculate metrics
            accuracy = accuracy_score(y, svm.predict(X_scaled))
            margin = svm.get_margin()
            n_sv = len(svm.support_vectors)
            
            ax.set_title(f'Overlap={overlap}, C={C}\nAcc:{accuracy:.2f}, SVs:{n_sv}')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Analyze C parameter effects
    print("C Parameter Analysis:")
    for overlap in overlap_levels:
        print(f"\nOverlap level: {overlap}")
        X, y = generate_nonseparable_data(n_samples=100, cluster_std=1.5, 
                                        noise_ratio=overlap, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for C in C_values:
            svm = SoftMarginSVM(C=C)
            svm.fit(X_scaled, y)
            accuracy = accuracy_score(y, svm.predict(X_scaled))
            margin = svm.get_margin()
            n_sv = len(svm.support_vectors)
            print(f"  C = {C:>6}: Accuracy = {accuracy:.3f}, Margin = {margin:.3f}, SVs = {n_sv}")


def demonstrate_hinge_loss():
    """Demonstrate hinge loss function and its properties"""
    print("\n=== Hinge Loss Demonstration ===\n")
    
    # Define different loss functions
    def hinge_loss(y_true, y_pred):
        return np.maximum(0, 1 - y_true * y_pred)
    
    def logistic_loss(y_true, y_pred):
        return np.log(1 + np.exp(-y_true * y_pred))
    
    def exponential_loss(y_true, y_pred):
        return np.exp(-y_true * y_pred)
    
    # Generate sample data
    y_true = np.array([1, 1, -1, -1, 1, -1])
    y_pred = np.array([0.5, 2.0, -0.5, -2.0, -0.5, 0.5])
    
    # Calculate losses
    hinge_losses = hinge_loss(y_true, y_pred)
    logistic_losses = logistic_loss(y_true, y_pred)
    exponential_losses = exponential_loss(y_true, y_pred)
    
    print("Loss Comparison:")
    print("y_true  y_pred  Hinge  Logistic  Exponential")
    print("-" * 50)
    for i in range(len(y_true)):
        print(f"{y_true[i]:>6}  {y_pred[i]:>6.1f}  {hinge_losses[i]:>6.3f}  {logistic_losses[i]:>9.3f}  {exponential_losses[i]:>12.3f}")
    
    # Plot loss functions
    margin_values = np.linspace(-3, 3, 100)
    y_true_plot = np.ones_like(margin_values)
    
    hinge_plot = hinge_loss(y_true_plot, margin_values)
    logistic_plot = logistic_loss(y_true_plot, margin_values)
    exponential_plot = exponential_loss(y_true_plot, margin_values)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(margin_values, hinge_plot, 'b-', linewidth=2, label='Hinge Loss')
    plt.xlabel('y * f(x)')
    plt.ylabel('Loss')
    plt.title('Hinge Loss Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(margin_values, logistic_plot, 'r-', linewidth=2, label='Logistic Loss')
    plt.xlabel('y * f(x)')
    plt.ylabel('Loss')
    plt.title('Logistic Loss Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(margin_values, exponential_plot, 'g-', linewidth=2, label='Exponential Loss')
    plt.xlabel('y * f(x)')
    plt.ylabel('Loss')
    plt.title('Exponential Loss Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(margin_values, hinge_plot, 'b-', linewidth=2, label='Hinge')
    plt.plot(margin_values, logistic_plot, 'r-', linewidth=2, label='Logistic')
    plt.plot(margin_values, exponential_plot, 'g-', linewidth=2, label='Exponential')
    plt.xlabel('y * f(x)')
    plt.ylabel('Loss')
    plt.title('Loss Function Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate margin-aware property
    print("\nMargin-aware Property of Hinge Loss:")
    margins = [0.5, 1.0, 1.5, 2.0]
    for margin in margins:
        loss = hinge_loss(1, margin)
        print(f"  Margin = {margin}: Loss = {loss:.3f}")


def demonstrate_cross_validation():
    """Demonstrate cross-validation for parameter selection"""
    print("\n=== Cross-Validation for Parameter Selection ===\n")
    
    # Generate data
    X, y = generate_nonseparable_data(n_samples=200, cluster_std=1.5, noise_ratio=0.15)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grid
    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Perform grid search with cross-validation
    param_grid = {'C': C_values}
    svm = SVC(kernel='linear', random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    # Print results
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Plot cross-validation results
    cv_scores = grid_search.cv_results_['mean_test_score']
    cv_std = grid_search.cv_results_['std_test_score']
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(C_values, cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(C_values, cv_scores - cv_std, cv_scores + cv_std, alpha=0.3)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Cross-validation Score vs C Parameter')
    plt.grid(True, alpha=0.3)
    
    # Plot number of support vectors vs C
    plt.subplot(2, 1, 2)
    n_support_vectors = []
    for C in C_values:
        svm_temp = SVC(kernel='linear', C=C, random_state=42)
        svm_temp.fit(X_scaled, y)
        n_support_vectors.append(len(svm_temp.support_vectors_))
    
    plt.semilogx(C_values, n_support_vectors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Number of Support Vectors')
    plt.title('Number of Support Vectors vs C Parameter')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze best model
    best_svm = grid_search.best_estimator_
    best_accuracy = accuracy_score(y, best_svm.predict(X_scaled))
    best_n_sv = len(best_svm.support_vectors_)
    
    print(f"\nBest Model Analysis:")
    print(f"  C = {grid_search.best_params_['C']}")
    print(f"  Cross-validation accuracy = {grid_search.best_score_:.4f}")
    print(f"  Test accuracy = {best_accuracy:.4f}")
    print(f"  Number of support vectors = {best_n_sv}")
    
    return grid_search


def demonstrate_advantages_limitations():
    """Demonstrate advantages and limitations of soft margin SVM"""
    print("\n=== Advantages and Limitations ===\n")
    
    # Generate different types of data
    datasets = {
        'Clean Separable': generate_nonseparable_data(n_samples=100, cluster_std=1.0, noise_ratio=0.0),
        'Noisy Separable': generate_nonseparable_data(n_samples=100, cluster_std=1.0, noise_ratio=0.05),
        'Overlapping': generate_nonseparable_data(n_samples=100, cluster_std=1.5, noise_ratio=0.1),
        'Highly Overlapping': generate_nonseparable_data(n_samples=100, cluster_std=2.0, noise_ratio=0.2)
    }
    
    C_values = [0.1, 1.0, 10.0, 100.0]
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"\n{name} Data:")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results[name] = {}
        
        for C in C_values:
            # Fit SVM
            svm = SoftMarginSVM(C=C)
            svm.fit(X_scaled, y)
            
            # Calculate metrics
            accuracy = accuracy_score(y, svm.predict(X_scaled))
            margin = svm.get_margin()
            n_sv = len(svm.support_vectors)
            slack_sum = np.sum(svm.get_slack_variables(X_scaled, y))
            
            results[name][C] = {
                'accuracy': accuracy,
                'margin': margin,
                'n_support_vectors': n_sv,
                'slack_sum': slack_sum
            }
            
            print(f"  C = {C:>6}: Acc = {accuracy:.3f}, Margin = {margin:.3f}, SVs = {n_sv}, Slack = {slack_sum:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, metric in enumerate(['accuracy', 'margin', 'n_support_vectors', 'slack_sum']):
        ax = axes[i // 2, i % 2]
        
        for name in datasets.keys():
            values = [results[name][C][metric] for C in C_values]
            ax.semilogx(C_values, values, 'o-', label=name, linewidth=2, markersize=6)
        
        ax.set_xlabel('C Parameter')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs C Parameter')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Main demonstration of SVM non-separable case"""
    print("Support Vector Machines: Non-Separable Case Implementation")
    print("=" * 60)
    
    # 1. Basic soft margin demonstration
    print("\n1. Basic Soft Margin SVM Demonstration:")
    C_values, X, y = demonstrate_soft_margin_svm()
    
    # 2. KKT conditions verification
    print("\n2. KKT Conditions Verification:")
    svm = SoftMarginSVM(C=1.0)
    svm.fit(X, y)
    demonstrate_kkt_conditions(X, y, svm)
    
    # 3. C parameter effects
    print("\n3. C Parameter Effects:")
    demonstrate_c_parameter_effects()
    
    # 4. Hinge loss demonstration
    print("\n4. Hinge Loss Demonstration:")
    demonstrate_hinge_loss()
    
    # 5. Cross-validation for parameter selection
    print("\n5. Cross-Validation for Parameter Selection:")
    grid_search = demonstrate_cross_validation()
    
    # 6. Advantages and limitations
    print("\n6. Advantages and Limitations:")
    advantages_results = demonstrate_advantages_limitations()
    
    print("\n=== Key Insights ===")
    print("1. Soft margin SVM handles non-separable data using slack variables")
    print("2. Parameter C controls the trade-off between margin and errors")
    print("3. Support vectors are classified into margin and non-margin types")
    print("4. Hinge loss provides margin-aware error measurement")
    print("5. Cross-validation is essential for parameter selection")
    print("6. Soft margin SVM is robust to noise and overlapping classes")
    print("7. The method scales poorly with dataset size")
    print("8. Feature scaling is important for optimal performance")
    
    return {
        'C_values': C_values,
        'X': X,
        'y': y,
        'grid_search': grid_search,
        'advantages_results': advantages_results
    }


if __name__ == "__main__":
    main()
