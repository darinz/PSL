# 10.2. Maximum Likelihood Estimation (MLE)

## Introduction

Maximum Likelihood Estimation (MLE) is the cornerstone of parameter estimation in logistic regression. Unlike linear regression where we can derive closed-form solutions, logistic regression requires iterative optimization due to the nonlinear nature of the sigmoid function. In this section, we'll derive the MLE step-by-step and implement the optimization algorithms.

## Mathematical Foundation

### Step 1: From Logit to Probability

We start with the logit transformation that connects our linear predictor to the probability:

```math
\log \frac{\eta(x)}{1-\eta(x)} = x^T \beta
```

This equation states that the log-odds of the positive class is a linear function of our features. To work with probabilities directly, we need to solve for $\eta(x)$:

```math
\begin{split}
\log \frac{\eta(x)}{1-\eta(x)} &= x^T \beta \\
\frac{\eta(x)}{1-\eta(x)} &= \exp(x^T \beta) \\
\eta(x) &= \exp(x^T \beta) \cdot (1-\eta(x)) \\
\eta(x) &= \exp(x^T \beta) - \exp(x^T \beta) \cdot \eta(x) \\
\eta(x) + \exp(x^T \beta) \cdot \eta(x) &= \exp(x^T \beta) \\
\eta(x) \cdot (1 + \exp(x^T \beta)) &= \exp(x^T \beta) \\
\eta(x) &= \frac{\exp(x^T \beta)}{1 + \exp(x^T \beta)}
\end{split}
```

### Step 2: Unified Probability Expression

We can express both $P(Y=1|X=x)$ and $P(Y=0|X=x)$ in a unified form using the sigmoid function:

```math
\begin{split}
P(Y=1|X=x) &= \eta(x) = \frac{\exp(x^T \beta)}{1 + \exp(x^T \beta)} = \sigma(x^T \beta) \\
P(Y=0|X=x) &= 1 - \eta(x) = \frac{1}{1 + \exp(x^T \beta)} = 1 - \sigma(x^T \beta)
\end{split}
```

Where $\sigma(z) = \frac{e^z}{1 + e^z}$ is the sigmoid function.

### Step 3: Likelihood Function

For a dataset with $n$ independent observations $(x_i, y_i)$, the likelihood function is:

```math
L(\beta) = \prod_{i=1}^n P(Y_i = y_i | X_i = x_i)
```

Using our unified probability expression:

```math
L(\beta) = \prod_{i=1}^n \sigma(x_i^T \beta)^{y_i} (1 - \sigma(x_i^T \beta))^{1-y_i}
```

### Step 4: Log-Likelihood Function

Taking the natural logarithm (which preserves the maximum and simplifies calculations):

```math
\begin{split}
\ell(\beta) &= \log L(\beta) \\
&= \sum_{i=1}^n \log \left[ \sigma(x_i^T \beta)^{y_i} (1 - \sigma(x_i^T \beta))^{1-y_i} \right] \\
&= \sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \beta) + (1-y_i) \log (1 - \sigma(x_i^T \beta)) \right]
\end{split}
```

This is the **log-likelihood function** that we want to maximize.

## Gradient and Hessian Derivation

### First Derivative (Gradient)

To find the maximum, we set the gradient to zero:

```math
\frac{\partial \ell(\beta)}{\partial \beta} = 0
```

Let's compute this step by step:

```math
\begin{split}
\frac{\partial \ell(\beta)}{\partial \beta} &= \sum_{i=1}^n \frac{\partial}{\partial \beta} \left[ y_i \log \sigma(x_i^T \beta) + (1-y_i) \log (1 - \sigma(x_i^T \beta)) \right] \\
&= \sum_{i=1}^n \left[ y_i \frac{1}{\sigma(x_i^T \beta)} \frac{\partial \sigma(x_i^T \beta)}{\partial \beta} + (1-y_i) \frac{1}{1-\sigma(x_i^T \beta)} \frac{\partial (1-\sigma(x_i^T \beta))}{\partial \beta} \right]
\end{split}
```

Using the chain rule and the fact that $\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z))$:

```math
\begin{split}
\frac{\partial \sigma(x_i^T \beta)}{\partial \beta} &= \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) \cdot x_i \\
\frac{\partial (1-\sigma(x_i^T \beta))}{\partial \beta} &= -\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) \cdot x_i
\end{split}
```

Substituting back:

```math
\begin{split}
\frac{\partial \ell(\beta)}{\partial \beta} &= \sum_{i=1}^n \left[ y_i \frac{1}{\sigma(x_i^T \beta)} \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i + (1-y_i) \frac{1}{1-\sigma(x_i^T \beta)} (-\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta))) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i (1-\sigma(x_i^T \beta)) x_i - (1-y_i) \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i x_i - y_i \sigma(x_i^T \beta) x_i - \sigma(x_i^T \beta) x_i + y_i \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i x_i - \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta))
\end{split}
```

Therefore:

```math
\frac{\partial \ell(\beta)}{\partial \beta} = \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta)) = X^T(y - \hat{y})
```

Where $X$ is the design matrix, $y$ is the vector of observed outcomes, and $\hat{y}$ is the vector of predicted probabilities.

### Second Derivative (Hessian)

The Hessian matrix is:

```math
H(\beta) = \frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T}
```

Computing this:

```math
\begin{split}
\frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T} &= \frac{\partial}{\partial \beta^T} \left[ \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta)) \right] \\
&= \sum_{i=1}^n x_i \frac{\partial}{\partial \beta^T} (y_i - \sigma(x_i^T \beta)) \\
&= \sum_{i=1}^n x_i \left[ -\frac{\partial \sigma(x_i^T \beta)}{\partial \beta^T} \right] \\
&= \sum_{i=1}^n x_i \left[ -\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i^T \right] \\
&= -\sum_{i=1}^n \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i x_i^T
\end{split}
```

In matrix form:

```math
H(\beta) = -X^T W X
```

Where $W$ is a diagonal matrix with $W_{ii} = \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta))$.

## Newton-Raphson Algorithm

Since the gradient equation $\frac{\partial \ell(\beta)}{\partial \beta} = 0$ has no closed-form solution, we use the Newton-Raphson iterative algorithm:

```math
\beta^{(t+1)} = \beta^{(t)} - H(\beta^{(t)})^{-1} \nabla \ell(\beta^{(t)})
```

Substituting our expressions:

```math
\beta^{(t+1)} = \beta^{(t)} + (X^T W^{(t)} X)^{-1} X^T(y - \hat{y}^{(t)})
```

This is equivalent to solving a weighted least squares problem at each iteration.

## Reweighted Least Squares (IRLS) Algorithm

The Newton-Raphson method can be reformulated as an **Iteratively Reweighted Least Squares (IRLS)** algorithm:

### Algorithm Steps:

1. **Initialize**: $\beta^{(0)} = 0$ or use a reasonable starting point
2. **For iteration $t = 0, 1, 2, \ldots$**:
   - Compute predicted probabilities: $\hat{y}_i^{(t)} = \sigma(x_i^T \beta^{(t)})$
   - Compute working response: $z_i^{(t)} = x_i^T \beta^{(t)} + \frac{y_i - \hat{y}_i^{(t)}}{\hat{y}_i^{(t)}(1-\hat{y}_i^{(t)})}$
   - Compute weights: $w_i^{(t)} = \hat{y}_i^{(t)}(1-\hat{y}_i^{(t)})$
   - Update parameters: $\beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} z^{(t)}$
3. **Convergence**: Stop when $||\beta^{(t+1)} - \beta^{(t)}|| < \epsilon$

## Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

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

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 3

# True parameters
true_beta = np.array([-2.0, 1.5, -0.8])

# Generate features
X = np.random.randn(n_samples, n_features)
X[:, 0] = 1  # Add intercept

# Generate probabilities and outcomes
z = X @ true_beta
p = 1 / (1 + np.exp(-z))
y = np.random.binomial(1, p)

print("Synthetic Data Summary:")
print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")
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

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Convergence plots
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

# Model evaluation
print("\n=== Model Evaluation ===")
for method, model in models.items():
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"{method.upper()} Accuracy: {accuracy:.4f}")

# Parameter comparison
print("\n=== Parameter Comparison ===")
param_df = pd.DataFrame({
    'True': true_beta,
    'Newton': models['newton'].beta,
    'IRLS': models['irls'].beta,
    'Sklearn': sklearn_beta
})
print(param_df)

# Decision boundary visualization (for 2D case)
if n_features == 3:  # Including intercept
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
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
```

### R Implementation

```r
# Maximum Likelihood Estimation for Logistic Regression

# Load required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

# Sigmoid function with numerical stability
sigmoid <- function(z) {
  z <- pmin(pmax(z, -500), 500)  # Prevent overflow
  return(1 / (1 + exp(-z)))
}

# Log-likelihood function
log_likelihood <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  p <- pmin(pmax(p, 1e-15), 1-1e-15)  # Prevent log(0)
  return(sum(y * log(p) + (1-y) * log(1-p)))
}

# Gradient function
gradient <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  return(t(X) %*% (y - p))
}

# Hessian function
hessian <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  W <- diag(as.vector(p * (1-p)))
  return(-t(X) %*% W %*% X)
}

# Newton-Raphson optimization
newton_raphson <- function(X, y, max_iter = 100, tol = 1e-6) {
  n_features <- ncol(X)
  beta <- rep(0, n_features)
  history <- list(log_likelihood = numeric(max_iter), 
                  beta_norm = numeric(max_iter))
  
  for (iteration in 1:max_iter) {
    # Compute current predictions
    z <- X %*% beta
    p <- sigmoid(z)
    
    # Store history
    history$log_likelihood[iteration] <- log_likelihood(beta, X, y)
    history$beta_norm[iteration] <- norm(beta, "2")
    
    # Compute gradient and Hessian
    grad <- gradient(beta, X, y)
    H <- hessian(beta, X, y)
    
    # Newton-Raphson update
    tryCatch({
      delta <- solve(H, grad)
      beta_new <- beta - delta
      
      # Check convergence
      if (norm(beta_new - beta, "2") < tol) {
        cat("Newton-Raphson converged after", iteration, "iterations\n")
        break
      }
      
      beta <- beta_new
    }, error = function(e) {
      cat("Hessian is singular, using pseudo-inverse\n")
      delta <- MASS::ginv(H) %*% grad
      beta <<- beta - delta
    })
  }
  
  return(list(beta = beta, history = history))
}

# Iteratively Reweighted Least Squares
irls <- function(X, y, max_iter = 100, tol = 1e-6) {
  n_features <- ncol(X)
  beta <- rep(0, n_features)
  history <- list(log_likelihood = numeric(max_iter), 
                  beta_norm = numeric(max_iter))
  
  for (iteration in 1:max_iter) {
    # Compute current predictions
    z <- X %*% beta
    p <- sigmoid(z)
    
    # Store history
    history$log_likelihood[iteration] <- log_likelihood(beta, X, y)
    history$beta_norm[iteration] <- norm(beta, "2")
    
    # Compute working response and weights
    working_response <- z + (y - p) / (p * (1-p) + 1e-15)
    weights <- p * (1-p)
    
    # Weighted least squares update
    W <- diag(as.vector(weights))
    tryCatch({
      beta_new <- solve(t(X) %*% W %*% X, t(X) %*% W %*% working_response)
      
      # Check convergence
      if (norm(beta_new - beta, "2") < tol) {
        cat("IRLS converged after", iteration, "iterations\n")
        break
      }
      
      beta <- beta_new
    }, error = function(e) {
      cat("Matrix is singular, using pseudo-inverse\n")
      beta_new <- MASS::ginv(t(X) %*% W %*% X) %*% t(X) %*% W %*% working_response
      beta <<- beta_new
    })
  }
  
  return(list(beta = beta, history = history))
}

# Generate synthetic data
set.seed(42)
n_samples <- 1000
n_features <- 3

# True parameters
true_beta <- c(-2.0, 1.5, -0.8)

# Generate features
X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)
X[, 1] <- 1  # Add intercept

# Generate probabilities and outcomes
z <- X %*% true_beta
p <- 1 / (1 + exp(-z))
y <- rbinom(n_samples, 1, p)

cat("Synthetic Data Summary:\n")
cat("Number of samples:", n_samples, "\n")
cat("Number of features:", n_features, "\n")
cat("True parameters:", true_beta, "\n")
cat("Class distribution:", table(y), "\n")

# Fit models using different methods
methods <- c("newton", "irls")
models <- list()

for (method in methods) {
  cat("\n=== Fitting with", toupper(method), "method ===\n")
  
  if (method == "newton") {
    result <- newton_raphson(X, y)
  } else {
    result <- irls(X, y)
  }
  
  models[[method]] <- result
  
  cat("Estimated parameters:", result$beta, "\n")
  cat("True parameters:", true_beta, "\n")
  cat("Parameter difference:", norm(result$beta - true_beta, "2"), "\n")
}

# Compare with glm
cat("\n=== Comparing with glm ===\n")
glm_model <- glm(y ~ X - 1, family = binomial)
glm_beta <- coef(glm_model)

cat("GLM parameters:", glm_beta, "\n")
cat("GLM vs Newton difference:", norm(glm_beta - models$newton$beta, "2"), "\n")
cat("GLM vs IRLS difference:", norm(glm_beta - models$irls$beta, "2"), "\n")

# Visualization
# Convergence plots
convergence_plots <- list()

for (method in methods) {
  history <- models[[method]]$history
  
  # Log-likelihood convergence
  p1 <- ggplot(data.frame(iteration = 1:length(history$log_likelihood), 
                          log_likelihood = history$log_likelihood)) +
    geom_line(aes(x = iteration, y = log_likelihood)) +
    labs(title = paste(toupper(method), "- Log-Likelihood Convergence"),
         x = "Iteration", y = "Log-Likelihood") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Parameter norm convergence
  p2 <- ggplot(data.frame(iteration = 1:length(history$beta_norm), 
                          beta_norm = history$beta_norm)) +
    geom_line(aes(x = iteration, y = beta_norm)) +
    labs(title = paste(toupper(method), "- Parameter Norm Convergence"),
         x = "Iteration", y = "||β||") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  convergence_plots[[method]] <- list(p1, p2)
}

# Display plots
do.call(grid.arrange, c(unlist(convergence_plots, recursive = FALSE), ncol = 2))

# Parameter comparison
param_comparison <- data.frame(
  True = true_beta,
  Newton = models$newton$beta,
  IRLS = models$irls$beta,
  GLM = glm_beta
)
print(param_comparison)

# Model evaluation
cat("\n=== Model Evaluation ===\n")
for (method in methods) {
  beta_hat <- models[[method]]$beta
  z_pred <- X %*% beta_hat
  p_pred <- sigmoid(z_pred)
  y_pred <- ifelse(p_pred >= 0.5, 1, 0)
  accuracy <- mean(y_pred == y)
  cat(toupper(method), "Accuracy:", accuracy, "\n")
}

# Decision boundary visualization (for 2D case)
if (n_features == 3) {
  # Create grid
  x1_range <- range(X[, 2])
  x2_range <- range(X[, 3])
  x1_grid <- seq(x1_range[1] - 0.5, x1_range[2] + 0.5, length.out = 100)
  x2_grid <- seq(x2_range[1] - 0.5, x2_range[2] + 0.5, length.out = 100)
  grid_data <- expand.grid(x1 = x1_grid, x2 = x2_grid)
  
  # Add intercept
  X_grid <- cbind(1, grid_data$x1, grid_data$x2)
  
  # Predict probabilities for each method
  decision_plots <- list()
  
  for (method in methods) {
    beta_hat <- models[[method]]$beta
    z_pred <- X_grid %*% beta_hat
    p_pred <- sigmoid(z_pred)
    
    # Add predictions to grid data
    grid_data$prob <- p_pred
    
    # Create plot
    p <- ggplot() +
      geom_contour(data = grid_data, aes(x = x1, y = x2, z = prob), 
                   breaks = 0.5, color = "red", size = 1) +
      geom_point(data = data.frame(x1 = X[, 2], x2 = X[, 3], y = factor(y)), 
                 aes(x = x1, y = x2, color = y), alpha = 0.6) +
      labs(title = paste(toupper(method), "Decision Boundary"),
           x = "Feature 1", y = "Feature 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    decision_plots[[method]] <- p
  }
  
  # Display decision boundary plots
  do.call(grid.arrange, c(decision_plots, ncol = 2))
}
```

## Key Insights

### 1. **Concavity of Log-Likelihood**
The Hessian matrix $H(\beta) = -X^T W X$ is negative semi-definite because:
- $W$ is diagonal with positive entries $w_i = \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) > 0$
- $X^T W X$ is positive semi-definite
- Therefore, $-X^T W X$ is negative semi-definite

This guarantees that any local maximum is also the global maximum.

### 2. **Connection to Linear Regression**
The gradient equation $X^T(y - \hat{y}) = 0$ is similar to the normal equations in linear regression, but with predicted probabilities instead of linear predictions.

### 3. **Numerical Stability**
- Use `np.clip()` to prevent overflow in sigmoid function
- Add small epsilon to prevent `log(0)` in likelihood computation
- Use pseudo-inverse when Hessian is singular

### 4. **Convergence Properties**
- Newton-Raphson typically converges in 5-10 iterations
- IRLS is more numerically stable but may require more iterations
- Both methods achieve the same optimal solution

### 5. **Computational Complexity**
- Each iteration: $O(np^2 + p^3)$ where $n$ is sample size, $p$ is number of features
- Matrix inversion dominates for large $p$
- Sparse matrix techniques can improve efficiency

## Applications and Extensions

### 1. **Regularized Logistic Regression**
Add L1/L2 penalties to the log-likelihood:

```math
\ell_{\text{penalized}}(\beta) = \ell(\beta) - \lambda \sum_{j=1}^p |\beta_j| \quad \text{(L1)}
```

### 2. **Multinomial Logistic Regression**
Extend to $K > 2$ classes using softmax function:

```math
P(Y=k|X=x) = \frac{\exp(x^T \beta_k)}{\sum_{j=1}^K \exp(x^T \beta_j)}
```

### 3. **Bayesian Logistic Regression**
Use MCMC or variational inference to obtain posterior distributions of parameters.

The MLE approach provides a solid foundation for understanding and implementing logistic regression, with clear connections to both linear regression and modern machine learning techniques.
