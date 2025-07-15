# 10.1. Setup

## 10.1.0. Introduction

Logistic Regression is one of the most fundamental and widely-used classification methods in machine learning. Unlike Discriminant Analysis, which follows a generative approach by modeling class-conditional distributions, Logistic Regression takes a **discriminative approach** by directly modeling the posterior probability $`P(Y=1 | X=x)`$.

### Key Concepts

- **Discriminative vs Generative**: Logistic Regression directly models $`P(Y=1 | X=x)`$ without modeling the joint distribution
- **Link Function**: Transforms the constrained probability to an unconstrained space
- **Maximum Likelihood**: Uses log-likelihood as the objective function
- **Linear Decision Boundary**: Results in linear decision boundaries in the feature space

## 10.1.1. The Binary Classification Problem

### Problem Formulation

In binary classification, we have:
- **Features**: $`X \in \mathbb{R}^p`$ (p-dimensional feature vector)
- **Target**: $`Y \in \{0, 1\}`$ (binary outcome)
- **Goal**: Learn a function that predicts $`P(Y=1 | X=x)`$

### Optimal Classifier

From our previous discussions, we know that the **Bayes optimal classifier** for binary classification is:

```math
\hat{y} = \begin{cases} 
1 & \text{if } P(Y=1 | X=x) > 0.5 \\
0 & \text{otherwise}
\end{cases}
```

This means the optimal classifier depends entirely on the **posterior probability**:

```math
\eta(x) = P(Y=1 | X=x)
```

## 10.1.2. Direct Modeling Approach

### The Challenge

We want to directly model $`\eta(x)`$, but there's a fundamental challenge:

**Problem**: $`\eta(x)`$ is constrained to $`[0, 1]`$ (it's a probability), but linear models $`x^T \beta`$ are unconstrained and can output any real value.

**Solution**: Use a **link function** to transform the constrained probability to an unconstrained space.

### Link Function Framework

We model the transformation of $`\eta(x)`$ with a linear function:

```math
g(\eta(x)) = x^T \beta
```

Where:
- $`g(\cdot)`$ is the **link function** (transformation)
- $`x^T \beta`$ is the **linear predictor**
- $`\beta`$ includes the intercept (we assume $`x_0 = 1`$ for the intercept)

### The Inverse Transformation

To get back to probabilities, we apply the inverse link function:

```math
\eta(x) = g^{-1}(x^T \beta)
```

## 10.1.3. The Logit Link Function

### Definition

In Logistic Regression, we use the **logit** (log-odds) link function:

```math
g(\eta(x)) = \text{logit}(\eta(x)) = \log \frac{\eta(x)}{1 - \eta(x)}
```

### Properties of the Logit Function

The logit function has several important properties:

1. **Domain**: $`\eta(x) \in (0, 1)`$ → $`\text{logit}(\eta(x)) \in (-\infty, +\infty)`$
2. **Monotonicity**: Strictly increasing function
3. **Symmetry**: $`\text{logit}(p) = -\text{logit}(1-p)`$

### Key Values

Let's examine the behavior at key probability values:

```math
\begin{align}
\text{When } \eta(x) = 0.5 &: \text{logit}(0.5) = \log \frac{0.5}{0.5} = \log(1) = 0 \\
\text{When } \eta(x) > 0.5 &: \text{logit}(\eta(x)) > 0 \text{ (positive values)} \\
\text{When } \eta(x) < 0.5 &: \text{logit}(\eta(x)) < 0 \text{ (negative values)}
\end{align}
```

### Visualization of the Logit Function

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Run visualization
p_vals, logit_vals, x_vals, sigmoid_vals = visualize_logit_function()
```

## 10.1.4. The Sigmoid Function

### Inverse of the Logit

The inverse of the logit function is the **sigmoid** (logistic) function:

```math
\eta(x) = g^{-1}(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}} = \sigma(x^T \beta)
```

Where $`\sigma(z) = \frac{1}{1 + e^{-z}}`$ is the sigmoid function.

### Properties of the Sigmoid Function

1. **Range**: $`\sigma(z) \in (0, 1)`$ for all $`z \in \mathbb{R}`$
2. **Monotonicity**: Strictly increasing
3. **Symmetry**: $`\sigma(-z) = 1 - \sigma(z)`$
4. **Derivative**: $`\sigma'(z) = \sigma(z)(1 - \sigma(z))`$

### Mathematical Relationship

The complete Logistic Regression model is:

```math
P(Y=1 | X=x) = \eta(x) = \sigma(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}}
```

## 10.1.5. The Data and Parameters

### Data Structure

For each observation $`i = 1, 2, \ldots, n`$, we have:

- **Feature vector**: $`x_i \in \mathbb{R}^p`$ (including intercept $`x_{i0} = 1`$)
- **Binary outcome**: $`y_i \in \{0, 1\}`$
- **True probability**: $`\eta(x_i) = P(Y_i=1 | X_i=x_i)`$

### Unknown Parameters

The unknown parameter vector $`\beta \in \mathbb{R}^p`$ includes:
- $`\beta_0`$: Intercept term
- $`\beta_1, \beta_2, \ldots, \beta_{p-1}`$: Feature coefficients

### The Estimation Problem

Our goal is to estimate $`\beta`$ from the observed data $`\{(x_i, y_i)\}_{i=1}^n`$.

## 10.1.6. Loss Function Selection

### Why Not L2 Loss?

One might consider using the squared error loss:

```math
L_{\text{MSE}}(\beta) = \sum_{i=1}^n (y_i - \eta(x_i))^2
```

However, this has several limitations:

1. **Small Gradients**: Since $`|y_i - \eta(x_i)| \leq 1`$, squaring makes gradients very small
2. **Training Difficulties**: Small gradients make optimization slow and can lead to getting stuck
3. **Non-convexity**: The squared error loss is not convex for logistic regression

### The Log-Likelihood Approach

Instead, we use the **negative log-likelihood** as our loss function:

```math
L(\beta) = -\sum_{i=1}^n \log P(Y_i = y_i | X_i = x_i)
```

### Likelihood Function

For binary outcomes, the likelihood is:

```math
P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i}
```

This can be written more compactly as:

```math
P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i}
```

### Log-Likelihood

Taking the logarithm:

```math
\begin{split}
\log P(Y_i = y_i | X_i = x_i) &= y_i \log \eta(x_i) + (1 - y_i) \log(1 - \eta(x_i)) \\
&= y_i \log \frac{\eta(x_i)}{1 - \eta(x_i)} + \log(1 - \eta(x_i)) \\
&= y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta})
\end{split}
```

### Final Loss Function

The negative log-likelihood loss function is:

```math
L(\beta) = -\sum_{i=1}^n \left[ y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta}) \right]
```

## 10.1.7. Comparison of Loss Functions

### Visual Comparison

```python
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

# Run comparison
mse_losses, ll_losses = compare_loss_functions()
```

## 10.1.8. Advantages of Log-Likelihood

### Why Log-Likelihood is Better

1. **Convexity**: The negative log-likelihood is convex, ensuring global optimality
2. **Proper Gradients**: Provides meaningful gradients for optimization
3. **Statistical Foundation**: Based on maximum likelihood estimation
4. **Interpretability**: Directly related to probability modeling

### Mathematical Properties

The log-likelihood function has several desirable properties:

1. **Convexity**: The Hessian matrix is positive semi-definite
2. **Uniqueness**: Under mild conditions, the maximum likelihood estimator is unique
3. **Asymptotic Properties**: MLE is consistent and asymptotically normal

## 10.1.9. Summary and Next Steps

### What We've Established

1. **Problem Setup**: Binary classification with direct probability modeling
2. **Link Function**: Logit transformation to handle probability constraints
3. **Model Form**: $`P(Y=1 | X=x) = \sigma(x^T \beta)`$
4. **Loss Function**: Negative log-likelihood for optimization

### Key Insights

- **Discriminative Approach**: Direct modeling of $`P(Y=1 | X=x)`$
- **Link Function**: Transforms constrained probabilities to unconstrained space
- **Loss Selection**: Log-likelihood provides better optimization properties than MSE

### Next Steps

In the following sections, we will:
1. **Parameter Estimation**: Derive the maximum likelihood estimator
2. **Optimization**: Implement gradient-based optimization algorithms
3. **Model Evaluation**: Assess model performance and interpretability
4. **Extensions**: Handle multi-class classification and regularization

### Implementation Preview

```python
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

# Run setup demonstration
X_data, y_data, beta_true_data = logistic_regression_setup_demo()
```

This setup provides the foundation for understanding Logistic Regression as a discriminative classification method that directly models posterior probabilities through a carefully chosen link function and loss function.
