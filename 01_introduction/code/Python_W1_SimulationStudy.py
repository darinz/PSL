"""
PSL: kNN vs Linear Regression Simulation Study
==============================================

This script demonstrates the bias-variance tradeoff through a comprehensive
simulation study comparing k-Nearest Neighbors (kNN) and Linear Regression
on two different data-generating processes.

Example 1: Simple Gaussian Classes (Linear Decision Boundary)
Example 2: Mixture of Gaussians (Non-linear Decision Boundary)

Key Concepts Demonstrated:
- Bias-Variance Tradeoff
- Model Complexity vs. Performance
- Cross-Validation for Model Selection
- Bayes Optimal Classifier
- Overfitting and Underfitting

Author: Statistical Learning Course
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(100)

def generate_simple_gaussian_data(n_per_class=100, n_test_per_class=5000, 
                                 mu1=[1, 0], mu0=[0, 1], sigma=1.0):
    """
    Generate data for Example 1: Simple Gaussian classes with linear decision boundary.
    
    Parameters:
    -----------
    n_per_class : int
        Number of training samples per class
    n_test_per_class : int
        Number of test samples per class
    mu1 : list
        Mean vector for class 1
    mu0 : list
        Mean vector for class 0
    sigma : float
        Standard deviation for both classes
    
    Returns:
    --------
    X_train, y_train, X_test, y_test : arrays
        Training and test data with labels
    """
    print("=" * 60)
    print("EXAMPLE 1: Simple Gaussian Classes")
    print("=" * 60)
    print(f"Class 1 mean: {mu1}, Class 0 mean: {mu0}")
    print(f"Standard deviation: {sigma}")
    print(f"Training samples per class: {n_per_class}")
    print(f"Test samples per class: {n_test_per_class}")
    
    # Generate training data
    # Class 1: N(mu1, sigma^2 * I)
    X_train_class1 = np.random.normal(loc=mu1, scale=sigma, size=(n_per_class, 2))
    # Class 0: N(mu0, sigma^2 * I)
    X_train_class0 = np.random.normal(loc=mu0, scale=sigma, size=(n_per_class, 2))
    
    # Combine training data
    X_train = np.vstack([X_train_class1, X_train_class0])
    y_train = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])
    
    # Generate test data (same process)
    X_test_class1 = np.random.normal(loc=mu1, scale=sigma, size=(n_test_per_class, 2))
    X_test_class0 = np.random.normal(loc=mu0, scale=sigma, size=(n_test_per_class, 2))
    
    X_test = np.vstack([X_test_class1, X_test_class0])
    y_test = np.concatenate([np.ones(n_test_per_class), np.zeros(n_test_per_class)])
    
    return X_train, y_train, X_test, y_test

def visualize_data(X_train, y_train, mu1, mu0, title="Data Visualization"):
    """
    Visualize the training data with class means.
    
    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training labels
    mu1, mu0 : arrays
        True class means
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot training data points
    class1_mask = y_train == 1
    class0_mask = y_train == 0
    
    plt.scatter(X_train[class1_mask, 0], X_train[class1_mask, 1], 
               c='blue', alpha=0.6, s=30, label='Class 1 (Training)')
    plt.scatter(X_train[class0_mask, 0], X_train[class0_mask, 1], 
               c='red', alpha=0.6, s=30, label='Class 0 (Training)')
    
    # Plot true class means
    plt.scatter(mu1[0], mu1[1], marker='+', s=200, c='blue', 
               linewidth=3, label='Class 1 Mean')
    plt.scatter(mu0[0], mu0[1], marker='+', s=200, c='red', 
               linewidth=3, label='Class 0 Mean')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def evaluate_knn(X_train, y_train, X_test, y_test, k_values):
    """
    Evaluate kNN classifier for different k values.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data and labels
    X_test, y_test : arrays
        Test data and labels
    k_values : list
        List of k values to evaluate
    
    Returns:
    --------
    train_errors, test_errors : lists
        Training and test errors for each k
    """
    print("\n" + "=" * 40)
    print("kNN EVALUATION")
    print("=" * 40)
    
    train_errors = []
    test_errors = []
    
    print(f"{'k':>4} {'Train Error':>12} {'Test Error':>12} {'DF':>8}")
    print("-" * 40)
    
    for k in k_values:
        # Create and fit kNN model
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        
        # Calculate errors
        train_error = 1 - knn_model.score(X_train, y_train)
        test_error = 1 - knn_model.score(X_test, y_test)
        
        # Approximate degrees of freedom (n/k)
        df = int(len(y_train) / k)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"{k:>4} {train_error:>12.4f} {test_error:>12.4f} {df:>8}")
    
    return train_errors, test_errors

def cross_validate_knn(X_train, y_train, k_values, cv_folds=5):
    """
    Perform cross-validation to select optimal k for kNN.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data and labels
    k_values : list
        List of k values to evaluate
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    cv_errors : array
        Cross-validation errors for each k
    best_k : int
        Optimal k value
    """
    print(f"\n{'-' * 40}")
    print(f"{cv_folds}-FOLD CROSS-VALIDATION")
    print(f"{'-' * 40}")
    
    cv_errors = []
    
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        # Use negative mean squared error as scoring (higher is better)
        scores = cross_val_score(knn_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        cv_error = 1 - scores.mean()  # Convert accuracy to error
        cv_errors.append(cv_error)
        print(f"k={k:>3}: CV Error = {cv_error:.4f} ± {scores.std():.4f}")
    
    best_k_idx = np.argmin(cv_errors)
    best_k = k_values[best_k_idx]
    
    print(f"\nBest k from CV: {best_k} (CV Error: {cv_errors[best_k_idx]:.4f})")
    
    return cv_errors, best_k

def evaluate_linear_regression(X_train, y_train, X_test, y_test):
    """
    Evaluate linear regression for classification.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data and labels
    X_test, y_test : arrays
        Test data and labels
    
    Returns:
    --------
    train_error, test_error : float
        Training and test errors
    """
    print(f"\n{'-' * 40}")
    print("LINEAR REGRESSION EVALUATION")
    print(f"{'-' * 40}")
    
    # Fit linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Get predictions
    y_train_pred_prob = lr_model.predict(X_train)
    y_test_pred_prob = lr_model.predict(X_test)
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
    y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
    
    # Calculate errors
    train_error = np.mean(y_train != y_train_pred)
    test_error = np.mean(y_test != y_test_pred)
    
    print(f"Training Error: {train_error:.4f}")
    print(f"Test Error: {test_error:.4f}")
    print(f"Model Parameters: {lr_model.coef_} (slopes), {lr_model.intercept_:.4f} (intercept)")
    
    return train_error, test_error

def compute_bayes_error(X_test, y_test, mu1, mu0):
    """
    Compute the Bayes optimal error rate.
    
    Parameters:
    -----------
    X_test, y_test : arrays
        Test data and labels
    mu1, mu0 : arrays
        True class means
    
    Returns:
    --------
    bayes_error : float
        Bayes optimal error rate
    """
    print(f"\n{'-' * 40}")
    print("BAYES OPTIMAL CLASSIFIER")
    print(f"{'-' * 40}")
    
    # Bayes decision rule: predict class 1 if ||x - mu1||^2 < ||x - mu0||^2
    # This simplifies to: 2x^T(mu1 - mu0) > ||mu1||^2 - ||mu0||^2
    
    # Compute decision function
    mu_diff = np.array(mu1) - np.array(mu0)
    threshold = np.dot(mu1, mu1) - np.dot(mu0, mu0)
    
    # Apply Bayes rule
    decisions = 2 * np.dot(X_test, mu_diff) > threshold
    bayes_error = np.mean(y_test != decisions)
    
    print(f"Bayes Error Rate: {bayes_error:.4f}")
    print(f"Decision Rule: 2x^T({mu_diff}) > {threshold:.4f}")
    
    return bayes_error

def plot_performance_comparison(k_values, train_errors_knn, test_errors_knn, 
                               train_error_lr, test_error_lr, bayes_error, 
                               cv_errors=None, best_k_idx=None):
    """
    Create comprehensive performance comparison plot.
    
    Parameters:
    -----------
    k_values : list
        k values used for kNN
    train_errors_knn, test_errors_knn : lists
        kNN training and test errors
    train_error_lr, test_error_lr : float
        Linear regression errors
    bayes_error : float
        Bayes optimal error
    cv_errors : list, optional
        Cross-validation errors
    best_k_idx : int, optional
        Index of best k from CV
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate degrees of freedom for x-axis
    df_values = [int(200 / k) for k in k_values]  # Assuming 200 total samples
    
    # Plot kNN performance
    plt.plot(range(len(k_values)), train_errors_knn, 'b-o', 
             label='kNN Training Error', alpha=0.7, linewidth=2)
    plt.plot(range(len(k_values)), test_errors_knn, 'r-o', 
             label='kNN Test Error', alpha=0.7, linewidth=2)
    
    # Plot linear regression performance (at fixed complexity)
    lr_position = 2  # Arbitrary position for visualization
    plt.scatter(lr_position, train_error_lr, marker='s', s=150, 
               c='blue', label='Linear Regression Training Error', zorder=5)
    plt.scatter(lr_position, test_error_lr, marker='s', s=150, 
               c='red', label='Linear Regression Test Error', zorder=5)
    
    # Plot Bayes error (horizontal line)
    plt.axhline(y=bayes_error, color='purple', linestyle='--', linewidth=2, 
                label=f'Bayes Error ({bayes_error:.4f})')
    
    # Plot CV results if available
    if cv_errors is not None and best_k_idx is not None:
        plt.plot(range(len(k_values)), cv_errors, 'g-o', 
                 label='Cross-Validation Error', alpha=0.7, linewidth=2)
        plt.scatter(best_k_idx, cv_errors[best_k_idx], marker='*', s=200, 
                   c='green', label=f'Best k from CV (k={k_values[best_k_idx]})', zorder=5)
    
    # Customize plot
    plt.xlabel('Model Complexity (Degrees of Freedom)')
    plt.ylabel('Error Rate')
    plt.title('Bias-Variance Tradeoff: kNN vs Linear Regression vs Bayes Optimal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show both k values and degrees of freedom
    plt.xticks(range(len(k_values)), [f'k={k}\nDF={df}' for k, df in zip(k_values, df_values)])
    
    plt.tight_layout()
    plt.show()

def generate_mixture_data(n_per_class=100, n_centers=10, sigma_centers=1.0, 
                         sigma_noise=0.45, base_means=[[1, 0], [0, 1]]):
    """
    Generate data for Example 2: Mixture of Gaussians with non-linear decision boundary.
    
    Parameters:
    -----------
    n_per_class : int
        Number of training samples per class
    n_centers : int
        Number of mixture components per class
    sigma_centers : float
        Standard deviation for generating centers
    sigma_noise : float
        Standard deviation for generating data around centers
    base_means : list
        Base means for each class
    
    Returns:
    --------
    X_train, y_train, centers1, centers0 : arrays
        Training data and mixture centers
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Mixture of Gaussians")
    print("=" * 60)
    print(f"Mixture components per class: {n_centers}")
    print(f"Training samples per class: {n_per_class}")
    print(f"Center spread: {sigma_centers}, Noise level: {sigma_noise}")
    
    # Generate mixture centers for each class
    centers1 = np.random.normal(loc=base_means[0], scale=sigma_centers, size=(n_centers, 2))
    centers0 = np.random.normal(loc=base_means[1], scale=sigma_centers, size=(n_centers, 2))
    
    # Generate training data
    X_train = []
    y_train = []
    
    # Class 1 data
    for i in range(n_per_class):
        # Randomly select a center
        center_idx = np.random.randint(n_centers)
        center = centers1[center_idx]
        # Generate point around the center
        point = np.random.normal(loc=center, scale=sigma_noise, size=2)
        X_train.append(point)
        y_train.append(1)
    
    # Class 0 data
    for i in range(n_per_class):
        center_idx = np.random.randint(n_centers)
        center = centers0[center_idx]
        point = np.random.normal(loc=center, scale=sigma_noise, size=2)
        X_train.append(point)
        y_train.append(0)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    return X_train, y_train, centers1, centers0

def main():
    """
    Main function to run the complete simulation study.
    """
    print("PSL: kNN vs Linear Regression Simulation Study")
    print("=" * 60)
    print("This study demonstrates the bias-variance tradeoff through")
    print("comprehensive comparison of learning algorithms.")
    print("=" * 60)
    
    # Example 1: Simple Gaussian Classes
    print("\n" + "=" * 60)
    print("EXAMPLE 1: SIMPLE GAUSSIAN CLASSES")
    print("=" * 60)
    print("Data generating process:")
    print("- Class 1: N(μ₁=[1,0], σ²I)")
    print("- Class 0: N(μ₀=[0,1], σ²I)")
    print("- Linear decision boundary exists")
    print("- Equal class priors")
    
    # Generate data
    mu1, mu0 = [1, 0], [0, 1]
    X_train, y_train, X_test, y_test = generate_simple_gaussian_data(
        n_per_class=100, n_test_per_class=5000, mu1=mu1, mu0=mu0, sigma=1.0
    )
    
    # Visualize data
    visualize_data(X_train, y_train, mu1, mu0, 
                   "Example 1: Simple Gaussian Classes")
    
    # Define k values for kNN evaluation
    k_values = [151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1]
    
    # Evaluate kNN
    train_errors_knn, test_errors_knn = evaluate_knn(
        X_train, y_train, X_test, y_test, k_values
    )
    
    # Cross-validation for kNN
    cv_errors, best_k = cross_validate_knn(X_train, y_train, k_values)
    best_k_idx = k_values.index(best_k)
    
    # Evaluate linear regression
    train_error_lr, test_error_lr = evaluate_linear_regression(
        X_train, y_train, X_test, y_test
    )
    
    # Compute Bayes error
    bayes_error = compute_bayes_error(X_test, y_test, mu1, mu0)
    
    # Plot performance comparison
    plot_performance_comparison(
        k_values, train_errors_knn, test_errors_knn,
        train_error_lr, test_error_lr, bayes_error,
        cv_errors, best_k_idx
    )
    
    # Example 2: Mixture of Gaussians
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MIXTURE OF GAUSSIANS")
    print("=" * 60)
    print("Data generating process:")
    print("- Each class: mixture of 10 Gaussian components")
    print("- Non-linear decision boundary")
    print("- More complex class-conditional distributions")
    
    # Generate mixture data
    X_train_mix, y_train_mix, centers1, centers0 = generate_mixture_data(
        n_per_class=100, n_centers=10, sigma_centers=1.0, sigma_noise=0.45
    )
    
    # Visualize mixture data
    plt.figure(figsize=(10, 8))
    class1_mask = y_train_mix == 1
    class0_mask = y_train_mix == 0
    
    plt.scatter(X_train_mix[class1_mask, 0], X_train_mix[class1_mask, 1], 
               c='blue', alpha=0.6, s=30, label='Class 1 (Training)')
    plt.scatter(X_train_mix[class0_mask, 0], X_train_mix[class0_mask, 1], 
               c='red', alpha=0.6, s=30, label='Class 0 (Training)')
    
    # Plot mixture centers
    plt.scatter(centers1[:, 0], centers1[:, 1], marker='+', s=150, 
               c='blue', linewidth=2, label='Class 1 Centers')
    plt.scatter(centers0[:, 0], centers0[:, 1], marker='+', s=150, 
               c='red', linewidth=2, label='Class 0 Centers')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Example 2: Mixture of Gaussians')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 60)
    print("SUMMARY AND KEY INSIGHTS")
    print("=" * 60)
    print("1. Bias-Variance Tradeoff:")
    print("   - Small k (high complexity): Low bias, high variance")
    print("   - Large k (low complexity): High bias, low variance")
    print("   - Optimal k balances bias and variance")
    print()
    print("2. Model Comparison:")
    print("   - Linear Regression: Good for Example 1 (linear boundary)")
    print("   - kNN: Adapts to both linear and non-linear boundaries")
    print("   - Bayes Rule: Theoretical optimum (when known)")
    print()
    print("3. Cross-Validation:")
    print("   - Provides realistic estimate of generalization error")
    print("   - Helps select optimal model complexity")
    print("   - More reliable than test set performance for model selection")
    print()
    print("4. Practical Implications:")
    print("   - Choose model complexity based on data characteristics")
    print("   - Use cross-validation for hyperparameter tuning")
    print("   - Consider computational cost vs. performance trade-offs")

if __name__ == "__main__":
    main() 