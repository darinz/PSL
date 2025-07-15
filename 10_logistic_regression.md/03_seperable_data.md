# 10.3. Separable Data Problem

## Introduction

The separable data problem is a fundamental challenge in logistic regression that occurs when the classes can be perfectly separated by a linear boundary. This seemingly ideal scenario actually creates significant computational and theoretical issues that every practitioner should understand.

## What is Separable Data?

### Definition
Data is said to be **linearly separable** if there exists a hyperplane that perfectly separates the two classes without any misclassifications. Mathematically, this means there exists a vector $\beta$ and scalar $\beta_0$ such that:

```math
\begin{cases}
x_i^T \beta + \beta_0 > 0 & \text{for all } i \text{ where } y_i = 1 \\
x_i^T \beta + \beta_0 < 0 & \text{for all } i \text{ where } y_i = 0
\end{cases}
```

### Toy Example
Consider a simple 2D example with four points:
- **Class 1 (Red)**: $(1, 1)$ and $(2, 2)$
- **Class 0 (Blue)**: $(-1, -1)$ and $(-2, -2)$

This data is perfectly separable by the line $x_1 + x_2 = 0$.

## Mathematical Analysis

### Likelihood Function for Separable Data

For our toy example, let's analyze the likelihood function step by step. We'll assume no intercept ($\beta_0 = 0$) for simplicity.

The logistic regression model is:
```math
P(Y=1|X=x) = \frac{\exp(x^T \beta)}{1 + \exp(x^T \beta)} = \sigma(x^T \beta)
```

For our four data points:
- **Red points**: $x_1 = (1, 1)$, $x_2 = (2, 2)$
- **Blue points**: $x_3 = (-1, -1)$, $x_4 = (-2, -2)$

The likelihood function is:
```math
L(\beta) = \prod_{i=1}^4 P(Y_i = y_i | X_i = x_i)
```

Let's compute this explicitly:

```math
\begin{split}
L(\beta) &= P(Y=1|X=(1,1)) \cdot P(Y=1|X=(2,2)) \cdot P(Y=0|X=(-1,-1)) \cdot P(Y=0|X=(-2,-2)) \\
&= \frac{\exp(\beta_1 + \beta_2)}{1 + \exp(\beta_1 + \beta_2)} \cdot \frac{\exp(2\beta_1 + 2\beta_2)}{1 + \exp(2\beta_1 + 2\beta_2)} \\
&\quad \cdot \frac{1}{1 + \exp(-\beta_1 - \beta_2)} \cdot \frac{1}{1 + \exp(-2\beta_1 - 2\beta_2)}
\end{split}
```

### Log-Likelihood Analysis

Taking the natural logarithm:

```math
\begin{split}
\ell(\beta) &= \log L(\beta) \\
&= \log \frac{\exp(\beta_1 + \beta_2)}{1 + \exp(\beta_1 + \beta_2)} + \log \frac{\exp(2\beta_1 + 2\beta_2)}{1 + \exp(2\beta_1 + 2\beta_2)} \\
&\quad + \log \frac{1}{1 + \exp(-\beta_1 - \beta_2)} + \log \frac{1}{1 + \exp(-2\beta_1 - 2\beta_2)}
\end{split}
```

Simplifying each term:

```math
\begin{split}
\ell(\beta) &= (\beta_1 + \beta_2) - \log(1 + \exp(\beta_1 + \beta_2)) \\
&\quad + (2\beta_1 + 2\beta_2) - \log(1 + \exp(2\beta_1 + 2\beta_2)) \\
&\quad - \log(1 + \exp(-\beta_1 - \beta_2)) \\
&\quad - \log(1 + \exp(-2\beta_1 - 2\beta_2))
\end{split}
```

### Behavior as Coefficients Increase

Let's examine what happens as we increase $\beta_1 = \beta_2 = c$:

```math
\begin{split}
\ell(c, c) &= 2c - \log(1 + \exp(2c)) + 4c - \log(1 + \exp(4c)) \\
&\quad - \log(1 + \exp(-2c)) - \log(1 + \exp(-4c))
\end{split}
```

For large positive $c$:
- $\exp(2c)$ and $\exp(4c)$ dominate, so $\log(1 + \exp(2c)) \approx 2c$ and $\log(1 + \exp(4c)) \approx 4c$
- $\exp(-2c)$ and $\exp(-4c)$ approach 0, so $\log(1 + \exp(-2c)) \approx 0$ and $\log(1 + \exp(-4c)) \approx 0$

Therefore:
```math
\ell(c, c) \approx 2c - 2c + 4c - 4c - 0 - 0 = 0
```

But this is misleading! Let's look at the actual behavior more carefully.

## Detailed Coefficient Analysis

### Case 1: $\beta_1 = \beta_2 = 1$

For the red points:
- $x_1 = (1, 1)$: $x_1^T \beta = 1 + 1 = 2$
- $x_2 = (2, 2)$: $x_2^T \beta = 2 + 2 = 4$

Probabilities:
```math
\begin{split}
P(Y=1|X=(1,1)) &= \frac{\exp(2)}{1 + \exp(2)} = \frac{7.39}{8.39} \approx 0.88 \\
P(Y=1|X=(2,2)) &= \frac{\exp(4)}{1 + \exp(4)} = \frac{54.6}{55.6} \approx 0.982
\end{split}
```

For the blue points:
- $x_3 = (-1, -1)$: $x_3^T \beta = -1 - 1 = -2$
- $x_4 = (-2, -2)$: $x_4^T \beta = -2 - 2 = -4$

Probabilities:
```math
\begin{split}
P(Y=0|X=(-1,-1)) &= \frac{1}{1 + \exp(-2)} = \frac{1}{1 + 0.135} \approx 0.881 \\
P(Y=0|X=(-2,-2)) &= \frac{1}{1 + \exp(-4)} = \frac{1}{1 + 0.018} \approx 0.982
\end{split}
```

### Case 2: $\beta_1 = \beta_2 = 10$

For the red points:
```math
\begin{split}
P(Y=1|X=(1,1)) &= \frac{\exp(20)}{1 + \exp(20)} \approx 0.9999999999 \\
P(Y=1|X=(2,2)) &= \frac{\exp(40)}{1 + \exp(40)} \approx 1.0000000000
\end{split}
```

For the blue points:
```math
\begin{split}
P(Y=0|X=(-1,-1)) &= \frac{1}{1 + \exp(-20)} \approx 0.9999999999 \\
P(Y=0|X=(-2,-2)) &= \frac{1}{1 + \exp(-40)} \approx 1.0000000000
\end{split}
```

### Case 3: $\beta_1 = \beta_2 = 100$

All probabilities approach 1 for their respective classes:
```math
\begin{split}
P(Y=1|X=(1,1)) &\approx 1.0 \\
P(Y=1|X=(2,2)) &\approx 1.0 \\
P(Y=0|X=(-1,-1)) &\approx 1.0 \\
P(Y=0|X=(-2,-2)) &\approx 1.0
\end{split}
```

## The Convergence Problem

### Why Coefficients Grow Without Bound

The key insight is that for separable data, the log-likelihood can be made arbitrarily close to zero (perfect fit) by making the coefficients arbitrarily large. Let's prove this:

For separable data, there exists a direction $\beta^*$ such that:
```math
x_i^T \beta^* > 0 \quad \forall i: y_i = 1 \\
x_i^T \beta^* < 0 \quad \forall i: y_i = 0
```

Then, for any scalar $c > 0$:
```math
\ell(c \beta^*) = \sum_{i: y_i=1} \log \sigma(c x_i^T \beta^*) + \sum_{i: y_i=0} \log(1 - \sigma(c x_i^T \beta^*))
```

As $c \to \infty$:
- For $y_i = 1$: $\sigma(c x_i^T \beta^*) \to 1$, so $\log \sigma(c x_i^T \beta^*) \to 0$
- For $y_i = 0$: $\sigma(c x_i^T \beta^*) \to 0$, so $\log(1 - \sigma(c x_i^T \beta^*)) \to 0$

Therefore:
```math
\lim_{c \to \infty} \ell(c \beta^*) = 0
```

### Decision Boundary Stability

Despite the coefficients growing without bound, the decision boundary remains stable. The decision boundary is defined by:
```math
x^T \beta = 0
```

For any scalar $c > 0$:
```math
x^T (c \beta) = c(x^T \beta) = 0 \iff x^T \beta = 0
```

So the decision boundary $x^T \beta = 0$ is invariant to scaling of $\beta$.

## Implementation and Demonstration

### Python Implementation

```python
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

# Demonstrate convergence issue
demo.demonstrate_convergence_issue()

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

print("=== Key Observations ===")
print("1. As coefficients increase, log-likelihood approaches 0 (perfect fit)")
print("2. All probabilities approach 1 for their respective classes")
print("3. Decision boundary remains stable despite coefficient explosion")
print("4. Standard logistic regression solvers may fail to converge")
print("5. The model is still useful for prediction despite convergence issues")
```

### R Implementation

```r
# Separable Data Problem in Logistic Regression

# Load required libraries
library(ggplot2)
library(gridExtra)

# Create separable toy data
X <- matrix(c(
  1, 1,    # Red point 1
  2, 2,    # Red point 2
  -1, -1,  # Blue point 1
  -2, -2   # Blue point 2
), ncol = 2, byrow = TRUE)

y <- c(1, 1, 0, 0)  # 1 for red, 0 for blue

# Sigmoid function with numerical stability
sigmoid <- function(z) {
  z <- pmin(pmax(z, -500), 500)
  return(1 / (1 + exp(-z)))
}

# Log-likelihood function
log_likelihood <- function(beta) {
  z <- X %*% beta
  p <- sigmoid(z)
  p <- pmin(pmax(p, 1e-15), 1-1e-15)
  
  ll <- 0
  for (i in 1:length(y)) {
    if (y[i] == 1) {
      ll <- ll + log(p[i])
    } else {
      ll <- ll + log(1 - p[i])
    }
  }
  return(ll)
}

# Compute probabilities
compute_probabilities <- function(beta) {
  z <- X %*% beta
  return(sigmoid(z))
}

# Analyze different coefficient values
analyze_coefficients <- function(beta_values) {
  results <- list()
  
  for (i in 1:length(beta_values)) {
    beta_val <- beta_values[i]
    beta <- c(beta_val, beta_val)
    
    # Compute probabilities
    probs <- compute_probabilities(beta)
    
    # Compute log-likelihood
    ll <- log_likelihood(beta)
    
    # Compute accuracy
    predictions <- ifelse(probs >= 0.5, 1, 0)
    accuracy <- mean(predictions == y)
    
    results[[i]] <- list(
      beta = beta_val,
      probabilities = probs,
      log_likelihood = ll,
      accuracy = accuracy
    )
  }
  
  return(results)
}

# Test different coefficient values
beta_values <- c(0.1, 1, 5, 10, 50, 100, 500)
results <- analyze_coefficients(beta_values)

# Display results
cat("=== Coefficient Analysis ===\n\n")
cat("Beta\tLog-Likelihood\tAccuracy\tProbabilities\n")
cat(paste(rep("-", 60), collapse = ""), "\n")

for (result in results) {
  cat(sprintf("%.1f\t%.6f\t%.3f\t[%.3f, %.3f, %.3f, %.3f]\n",
              result$beta, result$log_likelihood, result$accuracy,
              result$probabilities[1], result$probabilities[2],
              result$probabilities[3], result$probabilities[4]))
}

# Visualize data and decision boundaries
visualize_data_and_boundaries <- function() {
  # Create data frame for plotting
  plot_data <- data.frame(
    x1 = X[, 1],
    x2 = X[, 2],
    class = factor(y)
  )
  
  # Create grid for decision boundaries
  x1_range <- seq(-3, 3, length.out = 100)
  x2_range <- seq(-3, 3, length.out = 100)
  grid_data <- expand.grid(x1 = x1_range, x2 = x2_range)
  
  # Plot data points
  p_base <- ggplot(plot_data, aes(x = x1, y = x2, color = class)) +
    geom_point(size = 4, alpha = 0.7) +
    scale_color_manual(values = c("0" = "blue", "1" = "red"),
                       labels = c("Class 0", "Class 1")) +
    labs(title = "Separable Data Points",
         x = "X1", y = "X2") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    coord_fixed(ratio = 1)
  
  print(p_base)
  
  # Create multiple plots with different decision boundaries
  plots <- list()
  beta_values_plot <- c(0.1, 1, 5, 10, 50, 100)
  
  for (i in 1:length(beta_values_plot)) {
    beta_val <- beta_values_plot[i]
    beta <- c(beta_val, beta_val)
    
    # Compute decision boundary
    grid_data$z <- beta[1] * grid_data$x1 + beta[2] * grid_data$x2
    
    p <- ggplot() +
      geom_point(data = plot_data, aes(x = x1, y = x2, color = class), 
                 size = 3, alpha = 0.7) +
      geom_contour(data = grid_data, aes(x = x1, y = x2, z = z), 
                   breaks = 0, color = "black", size = 1) +
      scale_color_manual(values = c("0" = "blue", "1" = "red"),
                         labels = c("Class 0", "Class 1")) +
      labs(title = paste("β = (", beta_val, ", ", beta_val, ")", sep = ""),
           x = "X1", y = "X2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5)) +
      coord_fixed(ratio = 1) +
      xlim(-3, 3) + ylim(-3, 3)
    
    plots[[i]] <- p
  }
  
  # Display plots in a grid
  do.call(grid.arrange, c(plots, ncol = 3))
}

# Demonstrate convergence issue with glm
demonstrate_convergence_issue <- function() {
  cat("\n=== Demonstrating Convergence Issue ===\n\n")
  
  # Try different control parameters
  control_params <- list(
    epsilon = c(1e-8, 1e-10, 1e-12),
    maxit = c(25, 50, 100)
  )
  
  for (epsilon in control_params$epsilon) {
    for (maxit in control_params$maxit) {
      tryCatch({
        model <- glm(y ~ X - 1, family = binomial, 
                     control = list(epsilon = epsilon, maxit = maxit))
        
        coef_norm <- sqrt(sum(coef(model)^2))
        
        if (coef_norm > 100) {
          cat(sprintf("Epsilon: %.0e, Max iter: %d - Coefficients explode! Norm: %.2f\n",
                      epsilon, maxit, coef_norm))
        } else {
          cat(sprintf("Epsilon: %.0e, Max iter: %d - Coefficients stable. Norm: %.2f\n",
                      epsilon, maxit, coef_norm))
        }
      }, error = function(e) {
        cat(sprintf("Epsilon: %.0e, Max iter: %d - Failed: %s\n",
                    epsilon, maxit, e$message))
      })
    }
  }
}

# Plot log-likelihood vs coefficient magnitude
plot_log_likelihood_convergence <- function() {
  beta_magnitudes <- seq(0.1, 100, length.out = 100)
  log_likelihoods <- numeric(length(beta_magnitudes))
  
  for (i in 1:length(beta_magnitudes)) {
    beta <- c(beta_magnitudes[i], beta_magnitudes[i])
    log_likelihoods[i] <- log_likelihood(beta)
  }
  
  plot_data <- data.frame(
    magnitude = beta_magnitudes,
    log_likelihood = log_likelihoods
  )
  
  p <- ggplot(plot_data, aes(x = magnitude, y = log_likelihood)) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed", alpha = 0.7) +
    labs(title = "Log-Likelihood vs Coefficient Magnitude",
         x = "Coefficient Magnitude (β₁ = β₂)",
         y = "Log-Likelihood") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    annotate("text", x = 50, y = -0.5, 
             label = "Perfect Fit (LL = 0)", color = "red")
  
  print(p)
}

# Run demonstrations
visualize_data_and_boundaries()
demonstrate_convergence_issue()
plot_log_likelihood_convergence()

cat("\n=== Key Observations ===\n")
cat("1. As coefficients increase, log-likelihood approaches 0 (perfect fit)\n")
cat("2. All probabilities approach 1 for their respective classes\n")
cat("3. Decision boundary remains stable despite coefficient explosion\n")
cat("4. Standard logistic regression solvers may fail to converge\n")
cat("5. The model is still useful for prediction despite convergence issues\n")
```

## Why Regularization Doesn't Help

### Mathematical Explanation

Regularization adds a penalty term to the log-likelihood:

```math
\ell_{\text{penalized}}(\beta) = \ell(\beta) - \lambda \sum_{j=1}^p |\beta_j|^q
```

Where $q = 1$ for Lasso and $q = 2$ for Ridge.

For separable data, as $\beta \to \infty$:
- $\ell(\beta) \to 0$ (perfect fit)
- But the penalty term $\lambda \sum_{j=1}^p |\beta_j|^q \to \infty$

However, the key insight is that the likelihood improvement dominates the penalty for any finite $\lambda$. Let's prove this:

For separable data, there exists a direction $\beta^*$ such that:
```math
\ell(c \beta^*) \approx -n \log(1 + \exp(-c \epsilon))
```

Where $\epsilon = \min_{i} |x_i^T \beta^*| > 0$.

As $c \to \infty$:
```math
\ell(c \beta^*) \approx -n \exp(-c \epsilon) \to 0
```

The penalty term grows as:
```math
\lambda \sum_{j=1}^p |c \beta_j^*|^q = \lambda c^q \sum_{j=1}^p |\beta_j^*|^q
```

For any finite $\lambda$, there exists a $c$ large enough such that:
```math
|\ell(c \beta^*)| > \lambda c^q \sum_{j=1}^p |\beta_j^*|^q
```

Therefore, the coefficients will still grow without bound, just more slowly.

### Practical Demonstration

```python
# Demonstrate that regularization doesn't solve the problem
from sklearn.linear_model import LogisticRegression
import numpy as np

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
```

## Solutions and Workarounds

### 1. **Bayesian Approach**
Use informative priors to constrain the parameter space:

```python
# Bayesian logistic regression with priors
import pymc3 as pm

with pm.Model() as model:
    # Informative priors
    beta = pm.Normal('beta', mu=0, sd=1, shape=2)
    
    # Likelihood
    p = pm.math.sigmoid(pm.math.dot(X, beta))
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
    
    # Sample from posterior
    trace = pm.sample(1000, tune=1000)
```

### 2. **Firth's Method**
Use Jeffreys prior to prevent separation:

```python
# Firth's logistic regression
def firth_logistic(X, y, max_iter=100, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    
    for iteration in range(max_iter):
        # Compute current probabilities
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        
        # Compute weights and working response
        W = np.diag(p * (1-p))
        z_working = z + (y - p) / (p * (1-p) + 1e-15)
        
        # Add Jeffreys prior correction
        H = X.T @ W @ X
        correction = 0.5 * np.diag(H)
        
        # Update
        beta_new = np.linalg.solve(H, X.T @ W @ z_working + correction)
        
        if np.linalg.norm(beta_new - beta) < tol:
            break
            
        beta = beta_new
    
    return beta
```

### 3. **Exact Logistic Regression**
Use exact methods for small datasets:

```python
# Exact logistic regression (for small datasets)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Add constant for intercept
X_with_const = sm.add_constant(X)
model = sm.Logit(y, X_with_const)

# Use exact method if available
try:
    result = model.fit(method='exact')
    print("Exact logistic regression results:")
    print(result.summary())
except:
    print("Exact method not available, using standard approach")
    result = model.fit()
    print(result.summary())
```

## Summary

The separable data problem in logistic regression is a fundamental issue that occurs when classes can be perfectly separated. Key points:

1. **Mathematical Cause**: Coefficients grow without bound to achieve perfect separation
2. **Practical Impact**: Standard algorithms may fail to converge
3. **Decision Boundary**: Remains stable despite coefficient explosion
4. **Regularization**: Doesn't solve the fundamental problem
5. **Solutions**: Bayesian methods, Firth's correction, or exact methods

Understanding this problem is crucial for practitioners, as it affects both model interpretation and computational stability. While the model may still be useful for prediction, inference on the coefficients becomes problematic.
