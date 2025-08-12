import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionMLE:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.history = {'log_likelihood': [], 'beta_norm': []}
    
    def sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def log_likelihood(self, beta, X, y):
        """Compute log-likelihood"""
        z = X @ beta
        p = self.sigmoid(z)
        # Add small epsilon to prevent log(0)
        p = np.clip(p, 1e-15, 1-1e-15)
        return np.sum(y * np.log(p) + (1-y) * np.log(1-p))
    
    def gradient(self, beta, X, y):
        """Compute gradient of log-likelihood"""
        z = X @ beta
        p = self.sigmoid(z)
        return X.T @ (y - p)
    
    def hessian(self, beta, X, y):
        """Compute Hessian matrix"""
        z = X @ beta
        p = self.sigmoid(z)
        W = np.diag(p * (1-p))
        return -X.T @ W @ X
    
    def newton_raphson(self, X, y):
        """Newton-Raphson optimization"""
        n_features = X.shape[1]
        beta = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # Compute current predictions
            z = X @ beta
            p = self.sigmoid(z)
            
            # Store history
            ll = self.log_likelihood(beta, X, y)
            self.history['log_likelihood'].append(ll)
            self.history['beta_norm'].append(np.linalg.norm(beta))
            
            # Compute gradient and Hessian
            grad = self.gradient(beta, X, y)
            H = self.hessian(beta, X, y)
            
            # Newton-Raphson update
            try:
                delta = np.linalg.solve(H, grad)
                beta_new = beta - delta
                
                # Check convergence
                if np.linalg.norm(beta_new - beta) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
                    
                beta = beta_new
                
            except np.linalg.LinAlgError:
                print("Hessian is singular, using pseudo-inverse")
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
                beta = beta - delta
        
        self.beta = beta
        return beta
    
    def irls(self, X, y):
        """Iteratively Reweighted Least Squares"""
        n_features = X.shape[1]
        beta = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # Compute current predictions
            z = X @ beta
            p = self.sigmoid(z)
            
            # Store history
            ll = self.log_likelihood(beta, X, y)
            self.history['log_likelihood'].append(ll)
            self.history['beta_norm'].append(np.linalg.norm(beta))
            
            # Compute working response and weights
            working_response = z + (y - p) / (p * (1-p) + 1e-15)
            weights = p * (1-p)
            
            # Weighted least squares update
            W = np.diag(weights)
            try:
                beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ working_response)
                
                # Check convergence
                if np.linalg.norm(beta_new - beta) < self.tol:
                    print(f"IRLS converged after {iteration + 1} iterations")
                    break
                    
                beta = beta_new
                
            except np.linalg.LinAlgError:
                print("Matrix is singular, using pseudo-inverse")
                beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ working_response, rcond=None)[0]
                beta = beta_new
        
        self.beta = beta
        return beta
    
    def fit(self, X, y, method='newton'):
        """Fit the model using specified method"""
        if method == 'newton':
            return self.newton_raphson(X, y)
        elif method == 'irls':
            return self.irls(X, y)
        else:
            raise ValueError("Method must be 'newton' or 'irls'")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.beta is None:
            raise ValueError("Model not fitted yet")
        z = X @ self.beta
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def generate_synthetic_data(n_samples=1000, n_features=3, random_state=42):
    """Generate synthetic data for logistic regression"""
    np.random.seed(random_state)
    
    # True parameters
    true_beta = np.array([-2.0, 1.5, -0.8])
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = 1  # Add intercept
    
    # Generate probabilities and outcomes
    z = X @ true_beta
    p = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, p)
    
    return X, y, true_beta


def demonstrate_mle_methods():
    """Demonstrate MLE methods for logistic regression"""
    # Generate synthetic data
    X, y, true_beta = generate_synthetic_data()
    
    print("Synthetic Data Summary:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"True parameters: {true_beta}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Fit models using different methods
    methods = ['newton', 'irls']
    models = {}
    
    for method in methods:
        print(f"\n=== Fitting with {method.upper()} method ===")
        model = LogisticRegressionMLE(max_iter=50, tol=1e-6)
        beta_hat = model.fit(X, y, method=method)
        models[method] = model
        
        print(f"Estimated parameters: {beta_hat}")
        print(f"True parameters: {true_beta}")
        print(f"Parameter difference: {np.linalg.norm(beta_hat - true_beta):.6f}")
    
    # Compare with sklearn
    print("\n=== Comparing with sklearn ===")
    sklearn_model = LogisticRegression(fit_intercept=False, max_iter=1000)
    sklearn_model.fit(X, y)
    sklearn_beta = sklearn_model.coef_[0]
    
    print(f"Sklearn parameters: {sklearn_beta}")
    print(f"Sklearn vs Newton difference: {np.linalg.norm(sklearn_beta - models['newton'].beta):.6f}")
    print(f"Sklearn vs IRLS difference: {np.linalg.norm(sklearn_beta - models['irls'].beta):.6f}")
    
    return models, sklearn_model, sklearn_beta, true_beta


def visualize_convergence(models):
    """Visualize convergence of different methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    methods = list(models.keys())
    
    for i, method in enumerate(methods):
        model = models[method]
        
        # Log-likelihood convergence
        axes[0, i].plot(model.history['log_likelihood'])
        axes[0, i].set_title(f'{method.upper()} - Log-Likelihood Convergence')
        axes[0, i].set_xlabel('Iteration')
        axes[0, i].set_ylabel('Log-Likelihood')
        axes[0, i].grid(True)
        
        # Parameter norm convergence
        axes[1, i].plot(model.history['beta_norm'])
        axes[1, i].set_title(f'{method.upper()} - Parameter Norm Convergence')
        axes[1, i].set_xlabel('Iteration')
        axes[1, i].set_ylabel('||β||')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_parameters(models, sklearn_beta, true_beta):
    """Compare parameters across different methods"""
    # Create comparison DataFrame
    param_df = pd.DataFrame({
        'True': true_beta,
        'Newton': models['newton'].beta,
        'IRLS': models['irls'].beta,
        'Sklearn': sklearn_beta
    })
    
    print("\n=== Parameter Comparison ===")
    print(param_df)
    
    # Calculate parameter differences
    print("\n=== Parameter Differences ===")
    for method, model in models.items():
        diff = np.linalg.norm(model.beta - true_beta)
        print(f"{method.upper()} vs True: {diff:.6f}")
    
    sklearn_diff = np.linalg.norm(sklearn_beta - true_beta)
    print(f"Sklearn vs True: {sklearn_diff:.6f}")
    
    return param_df


def evaluate_models(models, X, y):
    """Evaluate model performance"""
    print("\n=== Model Evaluation ===")
    results = {}
    
    for method, model in models.items():
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        results[method] = accuracy
        print(f"{method.upper()} Accuracy: {accuracy:.4f}")
    
    return results


def visualize_decision_boundaries(models, X, y):
    """Visualize decision boundaries for 2D case"""
    if X.shape[1] != 3:  # Not 2D case (including intercept)
        print("Decision boundary visualization only available for 2D case")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = list(models.keys())
    
    for i, method in enumerate(methods):
        model = models[method]
        
        # Create grid
        x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        x2_min, x2_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                              np.linspace(x2_min, x2_max, 100))
        
        # Predict on grid
        X_grid = np.c_[np.ones(xx1.size), xx1.ravel(), xx2.ravel()]
        Z = model.predict_proba(X_grid).reshape(xx1.shape)
        
        # Plot
        contour = axes[i].contour(xx1, xx2, Z, levels=[0.5], colors='red', linewidths=2)
        scatter = axes[i].scatter(X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.6)
        axes[i].set_title(f'{method.upper()} Decision Boundary')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def demonstrate_gradient_hessian():
    """Demonstrate gradient and Hessian computation"""
    # Generate small dataset for demonstration
    X, y, true_beta = generate_synthetic_data(n_samples=100, n_features=3)
    
    # Initialize model
    model = LogisticRegressionMLE()
    
    # Test at different parameter values
    test_betas = [
        np.zeros(3),
        true_beta,
        true_beta + np.random.randn(3) * 0.1
    ]
    
    print("=== Gradient and Hessian Demonstration ===")
    for i, beta in enumerate(test_betas):
        print(f"\nTest {i+1}: β = {beta}")
        
        # Compute gradient and Hessian
        grad = model.gradient(beta, X, y)
        H = model.hessian(beta, X, y)
        
        print(f"Gradient norm: {np.linalg.norm(grad):.6f}")
        print(f"Hessian eigenvalues: {np.linalg.eigvals(H)}")
        print(f"Hessian is negative semi-definite: {np.all(np.linalg.eigvals(H) <= 0)}")
    
    return X, y, true_beta


def analyze_convergence_properties():
    """Analyze convergence properties of different methods"""
    # Test different starting points
    X, y, true_beta = generate_synthetic_data(n_samples=500, n_features=3)
    
    starting_points = [
        np.zeros(3),
        np.random.randn(3) * 0.1,
        np.random.randn(3) * 1.0,
        true_beta + np.random.randn(3) * 0.5
    ]
    
    print("=== Convergence Analysis ===")
    
    for i, start_beta in enumerate(starting_points):
        print(f"\nStarting point {i+1}: {start_beta}")
        
        # Test Newton-Raphson
        model_newton = LogisticRegressionMLE(max_iter=20, tol=1e-6)
        model_newton.beta = start_beta.copy()
        beta_newton = model_newton.newton_raphson(X, y)
        
        # Test IRLS
        model_irls = LogisticRegressionMLE(max_iter=20, tol=1e-6)
        model_irls.beta = start_beta.copy()
        beta_irls = model_irls.irls(X, y)
        
        print(f"Newton iterations: {len(model_newton.history['log_likelihood'])}")
        print(f"IRLS iterations: {len(model_irls.history['log_likelihood'])}")
        print(f"Newton final log-likelihood: {model_newton.history['log_likelihood'][-1]:.6f}")
        print(f"IRLS final log-likelihood: {model_irls.history['log_likelihood'][-1]:.6f}")
    
    return X, y, true_beta


def demonstrate_numerical_stability():
    """Demonstrate numerical stability issues and solutions"""
    # Generate data with potential numerical issues
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create features with high correlation (potential singularity)
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = 1  # Intercept
    X[:, 2] = X[:, 1] + np.random.randn(n_samples) * 0.01  # High correlation
    
    # True parameters with some large values
    true_beta = np.array([-5.0, 10.0, -9.5, 2.0, -1.0])
    
    # Generate outcomes
    z = X @ true_beta
    p = 1 / (1 + np.exp(-z))
    y = np.random.binomial(1, p)
    
    print("=== Numerical Stability Demonstration ===")
    print(f"Feature correlation: {np.corrcoef(X[:, 1], X[:, 2])[0, 1]:.6f}")
    print(f"Logit range: [{z.min():.2f}, {z.max():.2f}]")
    print(f"Probability range: [{p.min():.6f}, {p.max():.6f}]")
    
    # Test different methods
    methods = ['newton', 'irls']
    
    for method in methods:
        print(f"\n{method.upper()} method:")
        try:
            model = LogisticRegressionMLE(max_iter=50, tol=1e-6)
            beta_hat = model.fit(X, y, method=method)
            print(f"Converged successfully")
            print(f"Parameter difference: {np.linalg.norm(beta_hat - true_beta):.6f}")
        except Exception as e:
            print(f"Failed: {e}")
    
    return X, y, true_beta


def compare_with_other_optimizers():
    """Compare MLE methods with other optimization approaches"""
    X, y, true_beta = generate_synthetic_data(n_samples=500, n_features=3)
    
    # Define negative log-likelihood for scipy.optimize
    def neg_log_likelihood(beta):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-15, 1-1e-15)
        return -np.sum(y * np.log(p) + (1-y) * np.log(1-p))
    
    def neg_gradient(beta):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        return -X.T @ (y - p)
    
    print("=== Optimization Method Comparison ===")
    
    # Test different optimization methods
    optimizers = {
        'L-BFGS-B': 'L-BFGS-B',
        'BFGS': 'BFGS',
        'CG': 'CG',
        'Newton-CG': 'Newton-CG'
    }
    
    results = {}
    
    for name, method in optimizers.items():
        print(f"\n{name}:")
        try:
            result = minimize(neg_log_likelihood, np.zeros(3), 
                            method=method, jac=neg_gradient,
                            options={'maxiter': 100})
            
            if result.success:
                print(f"Converged in {result.nit} iterations")
                print(f"Final function value: {result.fun:.6f}")
                print(f"Parameter difference: {np.linalg.norm(result.x - true_beta):.6f}")
                results[name] = result.x
            else:
                print(f"Failed to converge: {result.message}")
        except Exception as e:
            print(f"Error: {e}")
    
    return results, true_beta


def main():
    """Main function to demonstrate MLE implementation"""
    print("Maximum Likelihood Estimation for Logistic Regression")
    print("=" * 60)
    
    # 1. Demonstrate MLE methods
    print("\n1. MLE Methods Demonstration:")
    models, sklearn_model, sklearn_beta, true_beta = demonstrate_mle_methods()
    
    # 2. Visualize convergence
    print("\n2. Convergence Visualization:")
    visualize_convergence(models)
    
    # 3. Compare parameters
    print("\n3. Parameter Comparison:")
    param_df = compare_parameters(models, sklearn_beta, true_beta)
    
    # 4. Evaluate models
    print("\n4. Model Evaluation:")
    X, y, _ = generate_synthetic_data()
    results = evaluate_models(models, X, y)
    
    # 5. Visualize decision boundaries
    print("\n5. Decision Boundary Visualization:")
    visualize_decision_boundaries(models, X, y)
    
    # 6. Demonstrate gradient and Hessian
    print("\n6. Gradient and Hessian Demonstration:")
    X_grad, y_grad, true_beta_grad = demonstrate_gradient_hessian()
    
    # 7. Analyze convergence properties
    print("\n7. Convergence Analysis:")
    X_conv, y_conv, true_beta_conv = analyze_convergence_properties()
    
    # 8. Demonstrate numerical stability
    print("\n8. Numerical Stability Demonstration:")
    X_stab, y_stab, true_beta_stab = demonstrate_numerical_stability()
    
    # 9. Compare with other optimizers
    print("\n9. Optimization Method Comparison:")
    opt_results, true_beta_opt = compare_with_other_optimizers()
    
    return {
        'models': models,
        'sklearn_model': sklearn_model,
        'param_df': param_df,
        'results': results,
        'optimization_results': opt_results,
        'true_beta': true_beta
    }


if __name__ == "__main__":
    main()
