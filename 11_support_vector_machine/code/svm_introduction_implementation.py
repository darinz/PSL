import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification, make_circles
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns


class SVMDemo:
    def __init__(self):
        self.models = {}
        self.data = {}
        
    def generate_separable_data(self, n_samples=100, random_state=42):
        """Generate linearly separable data"""
        np.random.seed(random_state)
        
        # Generate two classes with clear separation
        n_class1 = n_samples // 2
        n_class2 = n_samples - n_class1
        
        # Class 1: centered at (2, 2)
        class1 = np.random.randn(n_class1, 2) + np.array([2, 2])
        
        # Class 2: centered at (-2, -2)
        class2 = np.random.randn(n_class2, 2) + np.array([-2, -2])
        
        X = np.vstack([class1, class2])
        y = np.hstack([np.ones(n_class1), -np.ones(n_class2)])
        
        self.data['separable'] = {'X': X, 'y': y}
        return X, y
    
    def generate_nonseparable_data(self, n_samples=100, random_state=42):
        """Generate non-linearly separable data"""
        np.random.seed(random_state)
        
        # Generate circular data
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
        
        self.data['nonseparable'] = {'X': X, 'y': y}
        return X, y
    
    def generate_overlapping_data(self, n_samples=100, random_state=42):
        """Generate overlapping data for soft margin demonstration"""
        np.random.seed(random_state)
        
        # Generate overlapping classes
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, 
                                 random_state=random_state)
        y = 2 * y - 1  # Convert to {-1, 1}
        
        self.data['overlapping'] = {'X': X, 'y': y}
        return X, y
    
    def fit_linear_svm(self, X, y, C=1.0):
        """Fit linear SVM"""
        model = SVC(kernel='linear', C=C, random_state=42)
        model.fit(X, y)
        return model
    
    def fit_rbf_svm(self, X, y, C=1.0, gamma='scale'):
        """Fit RBF kernel SVM"""
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X, y)
        return model
    
    def visualize_decision_boundary(self, X, y, model, title="SVM Decision Boundary"):
        """Visualize SVM decision boundary"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh points
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and regions
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                           edgecolors='black', s=50, alpha=0.8)
        
        # Highlight support vectors
        if hasattr(model, 'support_vectors_'):
            ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                      s=200, facecolors='none', edgecolors='red', linewidth=2,
                      label=f'Support Vectors ({len(model.support_vectors_)})')
            ax.legend()
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.show()
    
    def demonstrate_separable_case(self):
        """Demonstrate linear SVM on separable data"""
        print("=== Linear SVM: Separable Case ===\n")
        
        # Generate data
        X, y = self.generate_separable_data()
        
        # Fit SVM
        model = self.fit_linear_svm(X, y)
        
        # Evaluate
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of support vectors: {len(model.support_vectors_)}")
        print(f"Support vector ratio: {len(model.support_vectors_)/len(X):.2f}")
        
        # Visualize
        self.visualize_decision_boundary(X, y, model, "Linear SVM: Separable Case")
        
        return model
    
    def demonstrate_nonseparable_case(self):
        """Demonstrate RBF SVM on non-separable data"""
        print("=== Nonlinear SVM: Non-Separable Case ===\n")
        
        # Generate data
        X, y = self.generate_nonseparable_data()
        
        # Fit linear SVM (should perform poorly)
        linear_model = self.fit_linear_svm(X, y)
        linear_accuracy = accuracy_score(y, linear_model.predict(X))
        
        # Fit RBF SVM
        rbf_model = self.fit_rbf_svm(X, y)
        rbf_accuracy = accuracy_score(y, rbf_model.predict(X))
        
        print(f"Linear SVM Accuracy: {linear_accuracy:.4f}")
        print(f"RBF SVM Accuracy: {rbf_accuracy:.4f}")
        print(f"RBF SVM Support Vectors: {len(rbf_model.support_vectors_)}")
        
        # Visualize both
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for i, (model, title) in enumerate([(linear_model, "Linear SVM"), 
                                          (rbf_model, "RBF SVM")]):
            ax = axes[i]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            ax.set_title(f"{title}\nAccuracy: {accuracy_score(y, model.predict(X)):.3f}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return rbf_model
    
    def demonstrate_soft_margin(self):
        """Demonstrate soft margin SVM with different C values"""
        print("=== Soft Margin SVM ===\n")
        
        # Generate overlapping data
        X, y = self.generate_overlapping_data()
        
        # Try different C values
        C_values = [0.1, 1.0, 10.0, 100.0]
        models = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, C in enumerate(C_values):
            model = self.fit_linear_svm(X, y, C=C)
            models[C] = model
            
            ax = axes[i // 2, i % 2]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            
            ax.set_title(f"C = {C}\nAccuracy: {accuracy:.3f}, SVs: {n_sv}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("Summary:")
        for C, model in models.items():
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            print(f"C = {C:>6}: Accuracy = {accuracy:.3f}, Support Vectors = {n_sv}")
        
        return models
    
    def demonstrate_kernels(self):
        """Demonstrate different kernel functions"""
        print("=== Kernel Comparison ===\n")
        
        # Generate non-separable data
        X, y = self.generate_nonseparable_data()
        
        # Define kernels to test
        kernels = [
            ('linear', {'kernel': 'linear'}),
            ('poly', {'kernel': 'poly', 'degree': 3}),
            ('rbf', {'kernel': 'rbf'}),
            ('sigmoid', {'kernel': 'sigmoid'})
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (name, params) in enumerate(kernels):
            model = SVC(C=1.0, random_state=42, **params)
            model.fit(X, y)
            
            ax = axes[i // 2, i % 2]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            
            ax.set_title(f"{name.upper()} Kernel\nAccuracy: {accuracy:.3f}, SVs: {n_sv}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_hyperparameter_tuning(self):
        """Demonstrate hyperparameter tuning with GridSearchCV"""
        print("=== Hyperparameter Tuning ===\n")
        
        # Generate data
        X, y = self.generate_nonseparable_data(n_samples=200)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        # Grid search
        grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Test set accuracy: {grid_search.score(X_test, y_test):.3f}")
        
        # Visualize best model
        best_model = grid_search.best_estimator_
        self.visualize_decision_boundary(X, y, best_model, 
                                       f"Best SVM (C={best_model.C}, gamma={best_model.gamma})")
        
        return grid_search
    
    def demonstrate_margin_analysis(self):
        """Demonstrate margin analysis and support vector properties"""
        print("=== Margin Analysis ===\n")
        
        # Generate separable data
        X, y = self.generate_separable_data()
        
        # Fit SVM with different C values
        C_values = [0.1, 1.0, 10.0, 100.0]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, C in enumerate(C_values):
            model = self.fit_linear_svm(X, y, C=C)
            
            ax = axes[i // 2, i % 2]
            
            # Create mesh grid
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
            
            # Plot data points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                      edgecolors='black', s=50, alpha=0.8)
            
            # Highlight support vectors
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            # Calculate margin
            w = model.coef_[0]
            margin = 2 / np.linalg.norm(w)
            
            accuracy = accuracy_score(y, model.predict(X))
            n_sv = len(model.support_vectors_)
            
            ax.set_title(f"C = {C}\nMargin: {margin:.3f}\nAccuracy: {accuracy:.3f}, SVs: {n_sv}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print margin analysis
        print("Margin Analysis:")
        for C in C_values:
            model = self.fit_linear_svm(X, y, C=C)
            w = model.coef_[0]
            margin = 2 / np.linalg.norm(w)
            n_sv = len(model.support_vectors_)
            print(f"C = {C:>6}: Margin = {margin:.3f}, Support Vectors = {n_sv}")
    
    def demonstrate_theoretical_properties(self):
        """Demonstrate theoretical properties of SVM"""
        print("=== Theoretical Properties ===\n")
        
        # Generate data
        X, y = self.generate_separable_data(n_samples=200)
        
        # Fit SVM
        model = self.fit_linear_svm(X, y)
        
        # Extract parameters
        w = model.coef_[0]
        b = model.intercept_[0]
        support_vectors = model.support_vectors_
        support_vector_indices = model.support_
        
        print("SVM Parameters:")
        print(f"Weight vector w: {w}")
        print(f"Bias term b: {b:.4f}")
        print(f"Number of support vectors: {len(support_vectors)}")
        
        # Verify KKT conditions
        print("\nKKT Conditions Verification:")
        
        # Calculate decision function values
        decision_values = model.decision_function(X)
        
        # Check complementary slackness
        alpha = np.abs(model.dual_coef_[0])  # Dual coefficients
        margin_violations = y * decision_values - 1
        
        print(f"Complementary slackness check:")
        print(f"  α_i * (y_i * f(x_i) - 1) should be 0 for all i")
        
        for i, sv_idx in enumerate(support_vector_indices):
            alpha_val = alpha[i] if i < len(alpha) else 0
            margin_val = margin_violations[sv_idx]
            product = alpha_val * margin_val
            print(f"  Support vector {sv_idx}: α = {alpha_val:.4f}, margin = {margin_val:.4f}, product = {product:.6f}")
        
        # Verify weight vector reconstruction
        print(f"\nWeight vector reconstruction:")
        print(f"  w = Σ α_i * y_i * x_i")
        
        w_reconstructed = np.zeros(2)
        for i, sv_idx in enumerate(support_vector_indices):
            alpha_val = alpha[i] if i < len(alpha) else 0
            w_reconstructed += alpha_val * y[sv_idx] * X[sv_idx]
        
        print(f"  Original w: {w}")
        print(f"  Reconstructed w: {w_reconstructed}")
        print(f"  Difference: {np.linalg.norm(w - w_reconstructed):.6f}")
        
        return model
    
    def demonstrate_scalability_analysis(self):
        """Demonstrate scalability properties of SVM"""
        print("=== Scalability Analysis ===\n")
        
        # Test different dataset sizes
        sizes = [50, 100, 200, 500, 1000]
        results = []
        
        for size in sizes:
            print(f"Testing with {size} samples...")
            
            # Generate data
            X, y = self.generate_separable_data(n_samples=size)
            
            # Time the fitting
            import time
            start_time = time.time()
            model = self.fit_linear_svm(X, y)
            fit_time = time.time() - start_time
            
            # Time the prediction
            start_time = time.time()
            y_pred = model.predict(X)
            predict_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            n_sv = len(model.support_vectors_)
            sv_ratio = n_sv / size
            
            results.append({
                'size': size,
                'fit_time': fit_time,
                'predict_time': predict_time,
                'accuracy': accuracy,
                'n_sv': n_sv,
                'sv_ratio': sv_ratio
            })
            
            print(f"  Fit time: {fit_time:.4f}s")
            print(f"  Predict time: {predict_time:.4f}s")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Support vectors: {n_sv} ({sv_ratio:.3f})")
            print()
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sizes = [r['size'] for r in results]
        fit_times = [r['fit_time'] for r in results]
        predict_times = [r['predict_time'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        sv_ratios = [r['sv_ratio'] for r in results]
        
        # Fit time vs dataset size
        axes[0, 0].plot(sizes, fit_times, 'bo-')
        axes[0, 0].set_xlabel('Dataset Size')
        axes[0, 0].set_ylabel('Fit Time (s)')
        axes[0, 0].set_title('Fit Time vs Dataset Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predict time vs dataset size
        axes[0, 1].plot(sizes, predict_times, 'ro-')
        axes[0, 1].set_xlabel('Dataset Size')
        axes[0, 1].set_ylabel('Predict Time (s)')
        axes[0, 1].set_title('Predict Time vs Dataset Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy vs dataset size
        axes[1, 0].plot(sizes, accuracies, 'go-')
        axes[1, 0].set_xlabel('Dataset Size')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Accuracy vs Dataset Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Support vector ratio vs dataset size
        axes[1, 1].plot(sizes, sv_ratios, 'mo-')
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Support Vector Ratio')
        axes[1, 1].set_title('Support Vector Ratio vs Dataset Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results


def demonstrate_svm_introduction():
    """Main demonstration of SVM introduction concepts"""
    print("Support Vector Machines: Introduction and Implementation")
    print("=" * 60)
    
    # Create demonstration object
    demo = SVMDemo()
    
    # 1. Separable case
    print("\n1. Linear SVM: Separable Case")
    separable_model = demo.demonstrate_separable_case()
    
    # 2. Non-separable case
    print("\n2. Nonlinear SVM: Non-Separable Case")
    nonseparable_model = demo.demonstrate_nonseparable_case()
    
    # 3. Soft margin
    print("\n3. Soft Margin SVM")
    soft_margin_models = demo.demonstrate_soft_margin()
    
    # 4. Kernel comparison
    print("\n4. Kernel Comparison")
    demo.demonstrate_kernels()
    
    # 5. Hyperparameter tuning
    print("\n5. Hyperparameter Tuning")
    tuned_model = demo.demonstrate_hyperparameter_tuning()
    
    # 6. Margin analysis
    print("\n6. Margin Analysis")
    demo.demonstrate_margin_analysis()
    
    # 7. Theoretical properties
    print("\n7. Theoretical Properties")
    theoretical_model = demo.demonstrate_theoretical_properties()
    
    # 8. Scalability analysis
    print("\n8. Scalability Analysis")
    scalability_results = demo.demonstrate_scalability_analysis()
    
    # Additional analysis: Support vector analysis
    print("\n=== Support Vector Analysis ===")
    for name, data in demo.data.items():
        X, y = data['X'], data['y']
        model = demo.fit_rbf_svm(X, y)
        
        n_sv = len(model.support_vectors_)
        sv_ratio = n_sv / len(X)
        
        print(f"{name.capitalize()} data:")
        print(f"  Total samples: {len(X)}")
        print(f"  Support vectors: {n_sv}")
        print(f"  SV ratio: {sv_ratio:.3f}")
        print()
    
    print("\n=== Key Insights ===")
    print("1. SVM maximizes margin for better generalization")
    print("2. Only support vectors influence the decision boundary")
    print("3. Kernel trick enables nonlinear classification")
    print("4. Parameter C controls margin vs error trade-off")
    print("5. SVM provides sparse, robust solutions")
    print("6. Theoretical foundations in structural risk minimization")
    
    return {
        'demo': demo,
        'separable_model': separable_model,
        'nonseparable_model': nonseparable_model,
        'soft_margin_models': soft_margin_models,
        'tuned_model': tuned_model,
        'theoretical_model': theoretical_model,
        'scalability_results': scalability_results
    }


if __name__ == "__main__":
    demonstrate_svm_introduction()
