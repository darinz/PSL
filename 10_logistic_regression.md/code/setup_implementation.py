import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def visualize_logit_function():
    """
    Visualize the logit function and its properties
    """
    # Generate probability values
    p = np.linspace(0.01, 0.99, 1000)
    
    # Compute logit values
    logit_p = np.log(p / (1 - p))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Logit function
    axes[0, 0].plot(p, logit_p, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Probability η(x)')
    axes[0, 0].set_ylabel('Logit g(η(x))')
    axes[0, 0].set_title('Logit Function')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    
    # Inverse logit (sigmoid) function
    x = np.linspace(-6, 6, 1000)
    sigmoid_x = 1 / (1 + np.exp(-x))
    
    axes[0, 1].plot(x, sigmoid_x, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Linear Predictor x^T β')
    axes[0, 1].set_ylabel('Probability η(x)')
    axes[0, 1].set_title('Sigmoid Function (Inverse Logit)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Symmetry property
    p_sym = np.linspace(0.1, 0.9, 100)
    logit_p_sym = np.log(p_sym / (1 - p_sym))
    logit_1_minus_p = np.log((1 - p_sym) / p_sym)
    
    axes[1, 0].plot(p_sym, logit_p_sym, 'b-', label='logit(p)', linewidth=2)
    axes[1, 0].plot(p_sym, logit_1_minus_p, 'r--', label='logit(1-p)', linewidth=2)
    axes[1, 0].set_xlabel('Probability p')
    axes[1, 0].set_ylabel('Logit Value')
    axes[1, 0].set_title('Symmetry: logit(p) = -logit(1-p)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Decision boundary visualization
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Example: β = [1, 1, -0.5] (intercept, x1, x2)
    beta = np.array([-0.5, 1, 1])
    Z = 1 / (1 + np.exp(-(beta[0] + beta[1] * X1 + beta[2] * X2)))
    
    contour = axes[1, 1].contourf(X1, X2, Z, levels=20, cmap='RdYlBu_r')
    axes[1, 1].contour(X1, X2, Z, levels=[0.5], colors='black', linewidths=2)
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].set_title('Logistic Regression Decision Boundary')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(contour, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return p, logit_p, x, sigmoid_x


def compare_loss_functions():
    """
    Compare MSE and log-likelihood loss functions
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True parameters
    beta_true = np.array([-1.5, 2.0, -0.8])
    
    # Generate features
    X = np.random.randn(n_samples, 2)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Generate true probabilities
    logits = X_with_intercept @ beta_true
    true_probs = 1 / (1 + np.exp(-logits))
    
    # Generate binary outcomes
    y = np.random.binomial(1, true_probs)
    
    # Define loss functions
    def mse_loss(beta, X, y):
        """Mean squared error loss"""
        probs = 1 / (1 + np.exp(-X @ beta))
        return np.mean((y - probs)**2)
    
    def log_likelihood_loss(beta, X, y):
        """Negative log-likelihood loss"""
        logits = X @ beta
        return -np.mean(y * logits - np.log(1 + np.exp(logits)))
    
    # Test different beta values
    beta_range = np.linspace(-3, 3, 100)
    mse_losses = []
    ll_losses = []
    
    for beta_val in beta_range:
        beta_test = np.array([beta_val, 2.0, -0.8])
        mse_losses.append(mse_loss(beta_test, X_with_intercept, y))
        ll_losses.append(log_likelihood_loss(beta_test, X_with_intercept, y))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE Loss
    axes[0].plot(beta_range, mse_losses, 'b-', linewidth=2)
    axes[0].axvline(x=beta_true[0], color='r', linestyle='--', label='True β₀')
    axes[0].set_xlabel('β₀ (Intercept)')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Mean Squared Error Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log-Likelihood Loss
    axes[1].plot(beta_range, ll_losses, 'g-', linewidth=2)
    axes[1].axvline(x=beta_true[0], color='r', linestyle='--', label='True β₀')
    axes[1].set_xlabel('β₀ (Intercept)')
    axes[1].set_ylabel('Negative Log-Likelihood')
    axes[1].set_title('Negative Log-Likelihood Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("Loss Function Comparison:")
    print("-" * 40)
    print(f"MSE Loss at true β₀: {mse_losses[50]:.6f}")
    print(f"Log-Likelihood Loss at true β₀: {ll_losses[50]:.6f}")
    print(f"MSE Loss gradient (approximate): {abs(mse_losses[51] - mse_losses[49]):.6f}")
    print(f"Log-Likelihood gradient (approximate): {abs(ll_losses[51] - ll_losses[49]):.6f}")
    
    return mse_losses, ll_losses


def logistic_regression_setup_demo():
    """
    Demonstrate the complete setup of logistic regression
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 2
    
    # True parameters
    beta_true = np.array([-1.0, 2.0, -1.5])
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Generate probabilities
    logits = X_with_intercept @ beta_true
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Generate outcomes
    y = np.random.binomial(1, probabilities)
    
    # Visualize the data
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    for i in range(2):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title('Binary Classification Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Probability distribution
    axes[1].hist(probabilities, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('True Probability P(Y=1|X)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of True Probabilities')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Logistic Regression Setup Summary:")
    print("-" * 40)
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"True parameters: {beta_true}")
    print(f"Class balance: {np.mean(y):.3f} (proportion of class 1)")
    
    return X, y, beta_true


def demonstrate_link_function_properties():
    """
    Demonstrate key properties of the logit link function
    """
    # Generate test probabilities
    p_values = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Calculate logit values
    logit_values = np.log(p_values / (1 - p_values))
    
    # Calculate inverse (sigmoid)
    sigmoid_values = 1 / (1 + np.exp(-logit_values))
    
    print("Link Function Properties Demonstration:")
    print("-" * 50)
    print(f"{'Probability':>12} {'Logit':>10} {'Sigmoid':>10}")
    print("-" * 50)
    
    for p, logit, sigmoid in zip(p_values, logit_values, sigmoid_values):
        print(f"{p:12.2f} {logit:10.3f} {sigmoid:10.3f}")
    
    # Demonstrate symmetry
    print("\nSymmetry Property:")
    print("-" * 30)
    for p in [0.1, 0.25, 0.5]:
        logit_p = np.log(p / (1 - p))
        logit_1_minus_p = np.log((1 - p) / p)
        print(f"logit({p:.2f}) = {logit_p:.3f}, logit({1-p:.2f}) = {logit_1_minus_p:.3f}")
        print(f"Sum: {logit_p + logit_1_minus_p:.3f} (should be 0)")
    
    return p_values, logit_values, sigmoid_values


def analyze_decision_boundary():
    """
    Analyze the decision boundary of logistic regression
    """
    # Generate grid of points
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Different parameter sets
    beta_sets = {
        'Linear': np.array([0, 1, 0]),      # x1 = 0
        'Diagonal': np.array([0, 1, 1]),    # x1 + x2 = 0
        'Offset': np.array([-1, 1, 1]),     # x1 + x2 = 1
        'Complex': np.array([-0.5, 2, -1])  # 2x1 - x2 = 0.5
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, beta) in enumerate(beta_sets.items()):
        # Calculate probabilities
        Z = 1 / (1 + np.exp(-(beta[0] + beta[1] * X1 + beta[2] * X2)))
        
        # Plot
        contour = axes[i].contourf(X1, X2, Z, levels=20, cmap='RdYlBu_r')
        axes[i].contour(X1, X2, Z, levels=[0.5], colors='black', linewidths=2, label='Decision Boundary')
        
        # Add decision boundary equation
        if beta[1] != 0 and beta[2] != 0:
            eq = f"{beta[1]:.1f}x₁ + {beta[2]:.1f}x₂ = {-beta[0]:.1f}"
        elif beta[1] != 0:
            eq = f"x₁ = {-beta[0]/beta[1]:.1f}"
        elif beta[2] != 0:
            eq = f"x₂ = {-beta[0]/beta[2]:.1f}"
        else:
            eq = "No boundary"
        
        axes[i].set_xlabel('Feature 1 (x₁)')
        axes[i].set_ylabel('Feature 2 (x₂)')
        axes[i].set_title(f'{name}: {eq}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        plt.colorbar(contour, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return beta_sets


def compare_with_linear_regression():
    """
    Compare logistic regression with linear regression for classification
    """
    # Generate data
    np.random.seed(42)
    n_samples = 200
    
    # True parameters for logistic regression
    beta_true = np.array([-1.0, 2.0, -1.5])
    
    # Generate features
    X = np.random.randn(n_samples, 2)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Generate probabilities and outcomes
    logits = X_with_intercept @ beta_true
    probabilities = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probabilities)
    
    # Fit logistic regression
    lr_model = LogisticRegression(penalty='none', solver='lbfgs')
    lr_model.fit(X, y)
    
    # Fit linear regression (treating binary as continuous)
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    
    # Predictions
    lr_probs = lr_model.predict_proba(X)[:, 1]
    linear_preds = linear_model.predict(X)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data
    for i in range(2):
        mask = y == i
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # True probabilities
    scatter = axes[0, 1].scatter(X[:, 0], X[:, 1], c=probabilities, cmap='RdYlBu_r')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].set_title('True Probabilities')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Logistic regression predictions
    scatter = axes[1, 0].scatter(X[:, 0], X[:, 1], c=lr_probs, cmap='RdYlBu_r')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2')
    axes[1, 0].set_title('Logistic Regression Predictions')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # Linear regression predictions
    scatter = axes[1, 1].scatter(X[:, 0], X[:, 1], c=linear_preds, cmap='RdYlBu_r')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].set_title('Linear Regression Predictions')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("Logistic vs Linear Regression Comparison:")
    print("-" * 50)
    print(f"Logistic Regression - Predictions in [0,1]: {np.min(lr_probs):.3f} to {np.max(lr_probs):.3f}")
    print(f"Linear Regression - Predictions in [0,1]: {np.min(linear_preds):.3f} to {np.max(linear_preds):.3f}")
    print(f"Linear Regression - Predictions outside [0,1]: {np.sum((linear_preds < 0) | (linear_preds > 1))}")
    
    return lr_model, linear_model, lr_probs, linear_preds


def demonstrate_loss_function_properties():
    """
    Demonstrate properties of different loss functions
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # True parameters
    beta_true = np.array([-0.5, 1.0])
    
    # Generate single feature
    X = np.random.randn(n_samples, 1)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Generate true probabilities and outcomes
    logits = X_with_intercept @ beta_true
    true_probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, true_probs)
    
    # Define loss functions
    def mse_loss(beta):
        probs = 1 / (1 + np.exp(-X_with_intercept @ beta))
        return np.mean((y - probs)**2)
    
    def log_likelihood_loss(beta):
        logits = X_with_intercept @ beta
        return -np.mean(y * logits - np.log(1 + np.exp(logits)))
    
    def hinge_loss(beta):
        # Simplified hinge loss for demonstration
        scores = X_with_intercept @ beta
        return np.mean(np.maximum(0, 1 - (2*y - 1) * scores))
    
    # Test different beta values
    beta_range = np.linspace(-2, 2, 50)
    mse_losses = []
    ll_losses = []
    hinge_losses = []
    
    for beta_val in beta_range:
        beta_test = np.array([beta_val, 1.0])
        mse_losses.append(mse_loss(beta_test))
        ll_losses.append(log_likelihood_loss(beta_test))
        hinge_losses.append(hinge_loss(beta_test))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE Loss
    axes[0].plot(beta_range, mse_losses, 'b-', linewidth=2)
    axes[0].axvline(x=beta_true[0], color='r', linestyle='--', label='True β₀')
    axes[0].set_xlabel('β₀ (Intercept)')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Mean Squared Error Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log-Likelihood Loss
    axes[1].plot(beta_range, ll_losses, 'g-', linewidth=2)
    axes[1].axvline(x=beta_true[0], color='r', linestyle='--', label='True β₀')
    axes[1].set_xlabel('β₀ (Intercept)')
    axes[1].set_ylabel('Negative Log-Likelihood')
    axes[1].set_title('Negative Log-Likelihood Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Hinge Loss
    axes[2].plot(beta_range, hinge_losses, 'm-', linewidth=2)
    axes[2].axvline(x=beta_true[0], color='r', linestyle='--', label='True β₀')
    axes[2].set_xlabel('β₀ (Intercept)')
    axes[2].set_ylabel('Hinge Loss')
    axes[2].set_title('Hinge Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find minima
    mse_min_idx = np.argmin(mse_losses)
    ll_min_idx = np.argmin(ll_losses)
    hinge_min_idx = np.argmin(hinge_losses)
    
    print("Loss Function Properties:")
    print("-" * 40)
    print(f"MSE Loss minimum at β₀ = {beta_range[mse_min_idx]:.3f}")
    print(f"Log-Likelihood minimum at β₀ = {beta_range[ll_min_idx]:.3f}")
    print(f"Hinge Loss minimum at β₀ = {beta_range[hinge_min_idx]:.3f}")
    print(f"True β₀ = {beta_true[0]:.3f}")
    
    return mse_losses, ll_losses, hinge_losses


def main():
    """
    Main function to demonstrate logistic regression setup
    """
    print("Logistic Regression Setup Demonstration")
    print("=" * 50)
    
    # 1. Visualize logit function
    print("\n1. Logit Function Visualization:")
    p_vals, logit_vals, x_vals, sigmoid_vals = visualize_logit_function()
    
    # 2. Compare loss functions
    print("\n2. Loss Function Comparison:")
    mse_losses, ll_losses = compare_loss_functions()
    
    # 3. Setup demonstration
    print("\n3. Complete Setup Demonstration:")
    X_data, y_data, beta_true_data = logistic_regression_setup_demo()
    
    # 4. Link function properties
    print("\n4. Link Function Properties:")
    p_values, logit_values, sigmoid_values = demonstrate_link_function_properties()
    
    # 5. Decision boundary analysis
    print("\n5. Decision Boundary Analysis:")
    beta_sets = analyze_decision_boundary()
    
    # 6. Compare with linear regression
    print("\n6. Comparison with Linear Regression:")
    lr_model, linear_model, lr_probs, linear_preds = compare_with_linear_regression()
    
    # 7. Loss function properties
    print("\n7. Loss Function Properties:")
    mse_losses_full, ll_losses_full, hinge_losses = demonstrate_loss_function_properties()
    
    return {
        'logit_data': (p_vals, logit_vals, x_vals, sigmoid_vals),
        'loss_comparison': (mse_losses, ll_losses),
        'setup_data': (X_data, y_data, beta_true_data),
        'link_properties': (p_values, logit_values, sigmoid_values),
        'decision_boundaries': beta_sets,
        'regression_comparison': (lr_model, linear_model, lr_probs, linear_preds),
        'loss_properties': (mse_losses_full, ll_losses_full, hinge_losses)
    }


if __name__ == "__main__":
    main()
