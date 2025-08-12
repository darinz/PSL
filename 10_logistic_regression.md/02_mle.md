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

The complete MLE implementation for logistic regression is provided in the code files:

**Python Implementation:** See `LogisticRegressionMLE` class and comprehensive demonstrations in [`code/mle_implementation.py`](code/mle_implementation.py)

**R Implementation:** See optimization functions and demonstrations in [`code/r_mle_implementation.R`](code/r_mle_implementation.R)

These implementations include:

- **LogisticRegressionMLE Class**: Complete implementation with Newton-Raphson and IRLS methods
- **Numerical Stability**: Proper handling of overflow and underflow issues
- **Convergence Tracking**: History tracking for log-likelihood and parameter norms
- **Comprehensive Demonstrations**: 
  - Method comparison (Newton-Raphson vs IRLS)
  - Convergence visualization
  - Parameter comparison with sklearn/glm
  - Decision boundary visualization
  - Gradient and Hessian analysis
  - Numerical stability testing
  - Optimization method comparison

The implementations demonstrate the mathematical foundations while providing practical, robust optimization algorithms for logistic regression parameter estimation.

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

---

**Navigation:**
- **Next Topic:** [Separable Data](03_seperable_data.md) - Handling perfectly separable data and convergence issues
- **Previous Topic:** [Setup and Introduction](01_setup.md) - Mathematical foundations and problem formulation
