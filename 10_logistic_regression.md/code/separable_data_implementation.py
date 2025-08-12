import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SeparableDataDemo:
    def __init__(self):
        # Create separable toy data
        self.X = np.array([
            [1, 1],    # Red point 1
            [2, 2],    # Red point 2
            [-1, -1],  # Blue point 1
            [-2, -2]   # Blue point 2
        ])
        self.y = np.array([1, 1, 0, 0])  # 1 for red, 0 for blue
        
    def sigmoid(self, z):
        """Sigmoid function with numerical stability"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def log_likelihood(self, beta):
        """Compute log-likelihood for given coefficients"""
        z = self.X @ beta
        p = self.sigmoid(z)
        p = np.clip(p, 1e-15, 1-1e-15)  # Prevent log(0)
        
        ll = 0
        for i in range(len(self.y)):
            if self.y[i] == 1:
                ll += np.log(p[i])
            else:
                ll += np.log(1 - p[i])
        return ll
    
    def compute_probabilities(self, beta):
        """Compute probabilities for all points"""
        z = self.X @ beta
        return self.sigmoid(z)
    
    def analyze_coefficients(self, beta_values):
        """Analyze behavior for different coefficient values"""
        results = []
        
        for beta_val in beta_values:
            beta = np.array([beta_val, beta_val])
            
            # Compute probabilities
            probs = self.compute_probabilities(beta)
            
            # Compute log-likelihood
            ll = self.log_likelihood(beta)
            
            # Compute accuracy
            predictions = (probs >= 0.5).astype(int)
            accuracy = accuracy_score(self.y, predictions)
            
            results.append({
                'beta': beta_val,
                'probabilities': probs,
                'log_likelihood': ll,
                'accuracy': accuracy
            })
        
        return results
    
    def visualize_data_and_boundary(self, beta=None):
        """Visualize data points and decision boundary"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data points
        red_points = self.X[self.y == 1]
        blue_points = self.X[self.y == 0]
        
        ax.scatter(red_points[:, 0], red_points[:, 1], c='red', s=100, label='Class 1', alpha=0.7)
        ax.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', s=100, label='Class 0', alpha=0.7)
        
        # Plot decision boundary if beta is provided
        if beta is not None:
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            
            # Create grid
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            # Compute decision boundary
            Z = beta[0] * xx + beta[1] * yy
            Z = Z.reshape(xx.shape)
            
            # Plot contour
            ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2, label='Decision Boundary')
        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('Separable Data with Decision Boundary')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        plt.show()
    
    def demonstrate_convergence_issue(self):
        """Demonstrate the convergence issue with sklearn"""
        print("=== Demonstrating Convergence Issue ===\n")
        
        # Try different solvers and max_iter values
        solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        max_iters = [100, 1000, 10000]
        
        for solver in solvers:
            print(f"Solver: {solver}")
            for max_iter in max_iters:
                try:
                    model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=42)
                    model.fit(self.X, self.y)
                    
                    # Check if coefficients are reasonable
                    coef_norm = np.linalg.norm(model.coef_[0])
                    
                    if coef_norm > 100:
                        print(f"  Max iter {max_iter}: Coefficients explode! Norm: {coef_norm:.2f}")
                    else:
                        print(f"  Max iter {max_iter}: Coefficients stable. Norm: {coef_norm:.2f}")
                        
                except Exception as e:
                    print(f"  Max iter {max_iter}: Failed - {str(e)}")
            print()


def demonstrate_separable_data():
    """Main demonstration of separable data problem"""
    # Create demonstration
    demo = SeparableDataDemo()
    
    # Analyze different coefficient values
    beta_values = [0.1, 1, 5, 10, 50, 100, 500]
    results = demo.analyze_coefficients(beta_values)
    
    print("=== Coefficient Analysis ===\n")
    print("Beta\tLog-Likelihood\tAccuracy\tProbabilities")
    print("-" * 60)
    for result in results:
        beta = result['beta']
        ll = result['log_likelihood']
        acc = result['accuracy']
        probs = result['probabilities']
        
        print(f"{beta}\t{ll:.6f}\t{acc:.3f}\t{probs}")
    
    return demo, results


def visualize_decision_boundaries(demo):
    """Visualize decision boundaries for different coefficients"""
    # Visualize data
    demo.visualize_data_and_boundary()
    
    # Show decision boundaries for different coefficients
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    beta_values_plot = [0.1, 1, 5, 10, 50, 100]
    
    for i, beta_val in enumerate(beta_values_plot):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        beta = np.array([beta_val, beta_val])
        
        # Plot data points
        red_points = demo.X[demo.y == 1]
        blue_points = demo.X[demo.y == 0]
        
        ax.scatter(red_points[:, 0], red_points[:, 1], c='red', s=50, alpha=0.7)
        ax.scatter(blue_points[:, 0], blue_points[:, 1], c='blue', s=50, alpha=0.7)
        
        # Plot decision boundary
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                           np.linspace(y_min, y_max, 50))
        
        Z = beta[0] * xx + beta[1] * yy
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1)
        ax.set_title(f'β = ({beta_val}, {beta_val})')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_log_likelihood_convergence(demo):
    """Plot log-likelihood vs coefficient magnitude"""
    # Plot log-likelihood vs coefficient magnitude
    beta_magnitudes = np.linspace(0.1, 100, 100)
    log_likelihoods = []
    
    for mag in beta_magnitudes:
        beta = np.array([mag, mag])
        ll = demo.log_likelihood(beta)
        log_likelihoods.append(ll)
    
    plt.figure(figsize=(10, 6))
    plt.plot(beta_magnitudes, log_likelihoods)
    plt.xlabel('Coefficient Magnitude (β₁ = β₂)')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood vs Coefficient Magnitude')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Perfect Fit (LL = 0)')
    plt.legend()
    plt.show()


def demonstrate_regularization_limitations():
    """Demonstrate that regularization doesn't solve the problem"""
    # Create separable data
    X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    y = np.array([1, 1, 0, 0])
    
    # Try different regularization strengths
    C_values = [1.0, 0.1, 0.01, 0.001]  # C = 1/lambda
    
    print("=== Regularization Analysis ===\n")
    print("C (1/λ)\tCoefficient Norm\tConverged")
    print("-" * 40)
    
    for C in C_values:
        try:
            model = LogisticRegression(C=C, max_iter=10000, random_state=42)
            model.fit(X, y)
            
            coef_norm = np.linalg.norm(model.coef_[0])
            converged = model.n_iter_ < 10000
            
            print(f"{C}\t{coef_norm:.2f}\t\t{converged}")
            
        except Exception as e:
            print(f"{C}\tFailed\t\t{str(e)}")
    
    print("\nEven with strong regularization, coefficients can still explode!")


def demonstrate_bayesian_solution():
    """Demonstrate Bayesian approach to separable data"""
    try:
        import pymc3 as pm
        
        # Create separable data
        X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
        y = np.array([1, 1, 0, 0])
        
        print("=== Bayesian Logistic Regression ===\n")
        
        with pm.Model() as model:
            # Informative priors to constrain parameter space
            beta = pm.Normal('beta', mu=0, sd=1, shape=2)
            
            # Likelihood
            p = pm.math.sigmoid(pm.math.dot(X, beta))
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
            
            # Sample from posterior
            trace = pm.sample(1000, tune=1000, return_inferencedata=False)
        
        # Extract posterior means
        beta_mean = trace['beta'].mean(axis=0)
        print(f"Posterior mean of β: {beta_mean}")
        print(f"Posterior std of β: {trace['beta'].std(axis=0)}")
        
        return trace, beta_mean
        
    except ImportError:
        print("PyMC3 not available. Install with: pip install pymc3")
        return None, None


def demonstrate_firth_method():
    """Demonstrate Firth's method for preventing separation"""
    def firth_logistic(X, y, max_iter=100, tol=1e-6):
        n, p = X.shape
        beta = np.zeros(p)
        
        for iteration in range(max_iter):
            # Compute current probabilities
            z = X @ beta
            p_probs = 1 / (1 + np.exp(-z))
            
            # Compute weights and working response
            W = np.diag(p_probs * (1-p_probs))
            z_working = z + (y - p_probs) / (p_probs * (1-p_probs) + 1e-15)
            
            # Add Jeffreys prior correction
            H = X.T @ W @ X
            correction = 0.5 * np.diag(H)
            
            # Update
            try:
                beta_new = np.linalg.solve(H, X.T @ W @ z_working + correction)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                beta_new = np.linalg.lstsq(H, X.T @ W @ z_working + correction, rcond=None)[0]
            
            if np.linalg.norm(beta_new - beta) < tol:
                break
                
            beta = beta_new
        
        return beta
    
    # Create separable data
    X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    y = np.array([1, 1, 0, 0])
    
    print("=== Firth's Method ===\n")
    
    # Standard logistic regression
    try:
        model_standard = LogisticRegression(max_iter=10000, random_state=42)
        model_standard.fit(X, y)
        coef_standard = model_standard.coef_[0]
        print(f"Standard LR coefficients: {coef_standard}")
        print(f"Standard LR coefficient norm: {np.linalg.norm(coef_standard):.2f}")
    except Exception as e:
        print(f"Standard LR failed: {e}")
    
    # Firth's method
    try:
        coef_firth = firth_logistic(X, y)
        print(f"Firth's method coefficients: {coef_firth}")
        print(f"Firth's method coefficient norm: {np.linalg.norm(coef_firth):.2f}")
    except Exception as e:
        print(f"Firth's method failed: {e}")
    
    return coef_firth if 'coef_firth' in locals() else None


def demonstrate_exact_logistic_regression():
    """Demonstrate exact logistic regression for small datasets"""
    try:
        import statsmodels.api as sm
        
        # Create separable data
        X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
        y = np.array([1, 1, 0, 0])
        
        print("=== Exact Logistic Regression ===\n")
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        model = sm.Logit(y, X_with_const)
        
        # Try exact method first
        try:
            result = model.fit(method='exact')
            print("Exact logistic regression results:")
            print(result.summary())
            return result
        except:
            print("Exact method not available, using standard approach")
            result = model.fit()
            print(result.summary())
            return result
            
    except ImportError:
        print("Statsmodels not available. Install with: pip install statsmodels")
        return None


def analyze_mathematical_properties():
    """Analyze mathematical properties of separable data"""
    # Create separable data
    X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    y = np.array([1, 1, 0, 0])
    
    print("=== Mathematical Analysis ===\n")
    
    # Test different coefficient directions
    directions = [
        np.array([1, 1]),      # Diagonal direction
        np.array([1, -1]),     # Anti-diagonal direction
        np.array([2, 1]),      # Asymmetric direction
        np.array([0, 1])       # Vertical direction
    ]
    
    for i, direction in enumerate(directions):
        print(f"Direction {i+1}: {direction}")
        
        # Check separability
        scores = X @ direction
        separable = all(scores[y == 1] > 0) and all(scores[y == 0] < 0)
        print(f"  Separable: {separable}")
        
        if separable:
            margin = min(np.abs(scores))
            print(f"  Margin: {margin:.3f}")
        
        # Test scaling behavior
        scales = [1, 10, 100]
        for scale in scales:
            beta = scale * direction
            z = X @ beta
            p = 1 / (1 + np.exp(-z))
            ll = np.sum(y * np.log(p) + (1-y) * np.log(1-p))
            print(f"  Scale {scale}: LL = {ll:.6f}")
        print()


def demonstrate_practical_implications():
    """Demonstrate practical implications of separable data"""
    print("=== Practical Implications ===\n")
    
    # Create demonstration
    demo = SeparableDataDemo()
    
    # Test prediction with different coefficient magnitudes
    test_points = np.array([[0.5, 0.5], [-0.5, -0.5], [0, 0]])
    
    beta_values = [1, 10, 100]
    
    for beta_val in beta_values:
        beta = np.array([beta_val, beta_val])
        print(f"\nCoefficients: β = ({beta_val}, {beta_val})")
        
        for i, point in enumerate(test_points):
            z = point @ beta
            p = demo.sigmoid(z)
            print(f"  Point {i+1} {point}: P(Y=1) = {p:.6f}")
    
    print("\nKey observations:")
    print("1. Predictions become more extreme as coefficients increase")
    print("2. Decision boundary remains stable")
    print("3. Model confidence increases (probabilities approach 0 or 1)")
    print("4. Standard errors become unreliable")


def main():
    """Main function to demonstrate separable data problem"""
    print("Separable Data Problem in Logistic Regression")
    print("=" * 60)
    
    # 1. Basic demonstration
    print("\n1. Basic Demonstration:")
    demo, results = demonstrate_separable_data()
    
    # 2. Visualize decision boundaries
    print("\n2. Decision Boundary Visualization:")
    visualize_decision_boundaries(demo)
    
    # 3. Demonstrate convergence issue
    print("\n3. Convergence Issue Demonstration:")
    demo.demonstrate_convergence_issue()
    
    # 4. Plot log-likelihood convergence
    print("\n4. Log-Likelihood Convergence:")
    plot_log_likelihood_convergence(demo)
    
    # 5. Demonstrate regularization limitations
    print("\n5. Regularization Limitations:")
    demonstrate_regularization_limitations()
    
    # 6. Demonstrate Bayesian solution
    print("\n6. Bayesian Solution:")
    trace, beta_mean = demonstrate_bayesian_solution()
    
    # 7. Demonstrate Firth's method
    print("\n7. Firth's Method:")
    coef_firth = demonstrate_firth_method()
    
    # 8. Demonstrate exact logistic regression
    print("\n8. Exact Logistic Regression:")
    result_exact = demonstrate_exact_logistic_regression()
    
    # 9. Analyze mathematical properties
    print("\n9. Mathematical Analysis:")
    analyze_mathematical_properties()
    
    # 10. Demonstrate practical implications
    print("\n10. Practical Implications:")
    demonstrate_practical_implications()
    
    print("\n=== Key Observations ===")
    print("1. As coefficients increase, log-likelihood approaches 0 (perfect fit)")
    print("2. All probabilities approach 1 for their respective classes")
    print("3. Decision boundary remains stable despite coefficient explosion")
    print("4. Standard logistic regression solvers may fail to converge")
    print("5. The model is still useful for prediction despite convergence issues")
    print("6. Regularization doesn't solve the fundamental problem")
    print("7. Bayesian methods and Firth's correction provide solutions")
    
    return {
        'demo': demo,
        'results': results,
        'trace': trace,
        'beta_mean': beta_mean,
        'coef_firth': coef_firth,
        'result_exact': result_exact
    }


if __name__ == "__main__":
    main()
