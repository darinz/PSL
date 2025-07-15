# 7.3. The EM Algorithm

## 7.3.1. Introduction to the EM Algorithm

The Expectation-Maximization (EM) algorithm is a powerful iterative technique designed to compute Maximum Likelihood Estimation (MLE) in the presence of unobserved latent variables. It's particularly useful when direct maximization of the likelihood function is computationally intractable due to the presence of hidden variables.

### Problem Setup

Consider a scenario where we have:
- **Observed data**: $`\mathbf{x} = (x_1, x_2, \ldots, x_n)`$
- **Latent variables**: $`\mathbf{Z} = (Z_1, Z_2, \ldots, Z_n)`$ (unobserved)
- **Parameters**: $`\theta`$ (to be estimated)

### The Challenge

The marginal log-likelihood of the observed data is:

```math
\log p(\mathbf{x} \mid \theta) = \log \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z} \mid \theta) = \log \sum_{\mathbf{z}} p(\mathbf{z} \mid \theta) p(\mathbf{x} \mid \mathbf{z}, \theta)
```

This expression is difficult to maximize directly because:
1. The sum inside the logarithm makes it non-concave
2. The latent variables $`\mathbf{Z}`$ are unobserved
3. The number of possible values for $`\mathbf{z}`$ can be exponentially large

### The EM Solution

The EM algorithm circumvents this difficulty by working with the **complete data log-likelihood**:

```math
\log p(\mathbf{x}, \mathbf{Z} \mid \theta) = \log p(\mathbf{Z} \mid \theta) + \log p(\mathbf{x} \mid \mathbf{Z}, \theta)
```

This is much easier to work with because it's typically a sum of logarithms rather than the logarithm of a sum.

### Algorithm Overview

The EM algorithm consists of two iterative steps:

1. **E-step (Expectation)**: Compute the expected value of the complete data log-likelihood with respect to the conditional distribution of the latent variables given the observed data and current parameter estimates.

2. **M-step (Maximization)**: Maximize the expected complete data log-likelihood with respect to the parameters.

### Mathematical Formulation

**E-step**: Compute the Q-function
```math
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log p(\mathbf{x}, \mathbf{Z} \mid \theta)\right]
```

**M-step**: Update parameters
```math
\theta^{(t+1)} = \arg\max_{\theta} Q(\theta \mid \theta^{(t)})
```

### Implementation: Basic EM Algorithm

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

class EMAlgorithm:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.log_likelihoods = []
        
    def fit(self, X, initial_params=None):
        """Fit the model using EM algorithm"""
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            
            # Check convergence
            if len(self.log_likelihoods) > 1:
                if abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        # For a two-component Gaussian mixture
        n_samples = len(X)
        
        # Random means
        means = np.random.choice(X, size=2, replace=False)
        
        # Random variances
        variances = np.array([np.var(X)] * 2)
        
        # Random mixing weights
        weights = np.random.dirichlet([1, 1])
        
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _e_step(self, X):
        """E-step: Compute responsibilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, 2))
        
        for k in range(2):
            responsibilities[:, k] = (self.params['weights'][k] * 
                                   norm.pdf(X, self.params['means'][k], 
                                           np.sqrt(self.params['variances'][k])))
        
        # Normalize
        row_sums = responsibilities.sum(axis=1)
        responsibilities = responsibilities / row_sums[:, np.newaxis]
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """M-step: Update parameters"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(responsibilities[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(responsibilities[:, k] * X) / 
                                     np.sum(responsibilities[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(responsibilities[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(responsibilities[:, k]))
    
    def _compute_log_likelihood(self, X):
        """Compute log-likelihood"""
        likelihood = np.zeros(len(X))
        
        for k in range(2):
            likelihood += (self.params['weights'][k] * 
                         norm.pdf(X, self.params['means'][k], 
                                 np.sqrt(self.params['variances'][k])))
        
        return np.sum(np.log(likelihood + 1e-10))
    
    def predict_proba(self, X):
        """Predict component probabilities"""
        return self._e_step(X)
    
    def predict(self, X):
        """Predict component assignments"""
        return np.argmax(self.predict_proba(X), axis=1)

# Example usage
np.random.seed(42)

# Generate data from a two-component Gaussian mixture
n_samples = 1000
z = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
x = np.zeros(n_samples)

x[z == 0] = np.random.normal(0, 1, size=np.sum(z == 0))
x[z == 1] = np.random.normal(4, 1.5, size=np.sum(z == 1))

# Fit using EM
em = EMAlgorithm(max_iter=100, tol=1e-6)
em.fit(x)

# Compare with sklearn
sklearn_gmm = GaussianMixture(n_components=2, random_state=42)
sklearn_gmm.fit(x.reshape(-1, 1))

print("EM Algorithm Results:")
print(f"Means: {em.params['means']}")
print(f"Variances: {em.params['variances']}")
print(f"Weights: {em.params['weights']}")

print("\nSklearn Results:")
print(f"Means: {sklearn_gmm.means_.flatten()}")
print(f"Variances: {sklearn_gmm.covariances_.flatten()}")
print(f"Weights: {sklearn_gmm.weights_}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(em.log_likelihoods)
plt.title('EM Algorithm Convergence')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.grid(True, alpha=0.3)
plt.show()
```

```r
# R implementation
library(mixtools)

EMAlgorithm <- function(max_iter=100, tol=1e-6) {
  list(max_iter=max_iter, tol=tol)
}

fit_em <- function(em, x, initial_params=NULL) {
  n_samples <- length(x)
  
  # Initialize parameters
  if (is.null(initial_params)) {
    params <- list(
      means = sample(x, 2, replace=FALSE),
      variances = rep(var(x), 2),
      weights = rdirichlet(1, c(1, 1))[1,]
    )
  } else {
    params <- initial_params
  }
  
  log_likelihoods <- numeric(0)
  
  for (iteration in 1:em$max_iter) {
    # E-step: Compute responsibilities
    responsibilities <- matrix(0, n_samples, 2)
    
    for (k in 1:2) {
      responsibilities[, k] <- params$weights[k] * dnorm(x, params$means[k], sqrt(params$variances[k]))
    }
    
    # Normalize
    row_sums <- rowSums(responsibilities)
    responsibilities <- responsibilities / row_sums
    
    # M-step: Update parameters
    for (k in 1:2) {
      # Update weights
      params$weights[k] <- mean(responsibilities[, k])
      
      # Update means
      params$means[k] <- sum(responsibilities[, k] * x) / sum(responsibilities[, k])
      
      # Update variances
      params$variances[k] <- sum(responsibilities[, k] * (x - params$means[k])^2) / sum(responsibilities[, k])
    }
    
    # Compute log-likelihood
    likelihood <- rep(0, n_samples)
    for (k in 1:2) {
      likelihood <- likelihood + params$weights[k] * dnorm(x, params$means[k], sqrt(params$variances[k]))
    }
    log_likelihood <- sum(log(likelihood + 1e-10))
    log_likelihoods <- c(log_likelihoods, log_likelihood)
    
    # Check convergence
    if (length(log_likelihoods) > 1) {
      if (abs(log_likelihoods[length(log_likelihoods)] - log_likelihoods[length(log_likelihoods)-1]) < em$tol) {
        cat("Converged after", iteration, "iterations\n")
        break
      }
    }
  }
  
  list(params=params, log_likelihoods=log_likelihoods)
}

# Example usage
set.seed(42)

# Generate data
n_samples <- 1000
z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(0.6, 0.4))
x <- numeric(n_samples)

x[z == 0] <- rnorm(sum(z == 0), 0, 1)
x[z == 1] <- rnorm(sum(z == 1), 4, 1.5)

# Fit using EM
em <- EMAlgorithm(max_iter=100, tol=1e-6)
result <- fit_em(em, x)

# Compare with mixtools
fit <- normalmixEM(x, k=2, maxit=100, epsilon=1e-6)

cat("EM Algorithm Results:\n")
cat("Means:", result$params$means, "\n")
cat("Variances:", result$params$variances, "\n")
cat("Weights:", result$params$weights, "\n")

cat("\nMixtools Results:\n")
cat("Means:", fit$mu, "\n")
cat("Variances:", fit$sigma^2, "\n")
cat("Weights:", fit$lambda, "\n")

# Plot convergence
plot(result$log_likelihoods, type="l", main="EM Algorithm Convergence",
     xlab="Iteration", ylab="Log-Likelihood")
grid()
```

## 7.3.2. Why the EM Algorithm Works

### The Monotonicity Property

A crucial property of the EM algorithm is that it **never decreases the log-likelihood** at each iteration. This ensures convergence to a local maximum.

### Mathematical Proof

Let's prove that the EM algorithm improves the marginal likelihood at each step.

**Step 1**: Define the Q-function
```math
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log p(\mathbf{x}, \mathbf{Z} \mid \theta)\right]
```

**Step 2**: Consider the difference $`Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)})`$

```math
\begin{split}
Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)}) &= \mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log \frac{p(\mathbf{x}, \mathbf{Z} \mid \theta^{(t+1)})}{p(\mathbf{x}, \mathbf{Z} \mid \theta^{(t)})}\right] \\
&= \mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log \frac{p(\mathbf{x} \mid \theta^{(t+1)}) p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t+1)})}{p(\mathbf{x} \mid \theta^{(t)}) p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t)})}\right] \\
&= \log \frac{p(\mathbf{x} \mid \theta^{(t+1)})}{p(\mathbf{x} \mid \theta^{(t)})} - \mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log \frac{p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t)})}{p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t+1)})}\right]
\end{split}
```

**Step 3**: Rearrange to get
```math
\log \frac{p(\mathbf{x} \mid \theta^{(t+1)})}{p(\mathbf{x} \mid \theta^{(t)})} = \underbrace{Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)})}_{\geq 0} + \underbrace{\mathbb{E}_{\mathbf{Z} \mid \mathbf{x}, \theta^{(t)}} \left[\log \frac{p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t)})}{p(\mathbf{Z} \mid \mathbf{x}, \theta^{(t+1)})}\right]}_{\geq 0}
```

The right-hand side is non-negative because:
1. $`Q(\theta^{(t+1)} \mid \theta^{(t)}) - Q(\theta^{(t)} \mid \theta^{(t)}) \geq 0`$ (by definition of M-step)
2. The second term is a KL divergence, which is always non-negative

Therefore, $`\log p(\mathbf{x} \mid \theta^{(t+1)}) \geq \log p(\mathbf{x} \mid \theta^{(t)})`$, proving that the EM algorithm never decreases the log-likelihood.

### Implementation: Convergence Monitoring

```python
def monitor_em_convergence(X, n_components=2, n_runs=5):
    """Monitor EM convergence across multiple runs"""
    results = []
    
    for run in range(n_runs):
        em = EMAlgorithm(max_iter=200, tol=1e-8)
        em.fit(X)
        
        results.append({
            'run': run + 1,
            'final_ll': em.log_likelihoods[-1],
            'iterations': len(em.log_likelihoods),
            'params': em.params.copy()
        })
    
    return results

# Example with convergence monitoring
convergence_results = monitor_em_convergence(x, n_components=2, n_runs=5)

print("Convergence Results:")
for result in convergence_results:
    print(f"Run {result['run']}: Final LL = {result['final_ll']:.3f}, "
          f"Iterations = {result['iterations']}")

# Plot convergence for all runs
plt.figure(figsize=(12, 8))

for i, result in enumerate(convergence_results):
    em = EMAlgorithm()
    em.log_likelihoods = result['log_likelihoods'] if 'log_likelihoods' in result else []
    plt.plot(em.log_likelihoods, label=f'Run {i+1}', alpha=0.7)

plt.title('EM Algorithm Convergence (Multiple Runs)')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 7.3.3. Connection with K-means

### Similarities and Differences

The EM algorithm for Gaussian mixtures and the K-means algorithm share fundamental similarities but differ in important ways:

| Aspect | EM Algorithm | K-means Algorithm |
|--------|--------------|-------------------|
| **Assignment** | Soft (probabilistic) | Hard (deterministic) |
| **Responsibilities** | $`\gamma_i \in [0, 1]`$ | $`\gamma_i \in \{0, 1\}`$ |
| **Objective** | Maximize log-likelihood | Minimize within-cluster variance |
| **Convergence** | Local maximum of likelihood | Local minimum of distortion |

### Mathematical Connection

For a two-component Gaussian mixture with equal variances $`\sigma^2`$, the responsibility ratio is:

```math
\frac{\gamma_i}{1 - \gamma_i} = \frac{\pi}{1-\pi} \times \exp\left(-\frac{1}{2\sigma^2}\left[(x_i - \mu_1)^2 - (x_i - \mu_2)^2\right]\right)
```

### When EM Mimics K-means

As $`\sigma^2 \to 0`$ (very small variance):

1. If $`x_i`$ is closer to $`\mu_1`$: $`\gamma_i \to 1`$
2. If $`x_i`$ is closer to $`\mu_2`$: $`\gamma_i \to 0`$

This makes the EM algorithm behave like K-means with hard assignments.

### Implementation: EM vs K-means Comparison

```python
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def compare_em_kmeans(X, n_components=2):
    """Compare EM algorithm with K-means"""
    
    # EM Algorithm
    em = EMAlgorithm(max_iter=100, tol=1e-6)
    em.fit(X)
    em_labels = em.predict(X)
    em_responsibilities = em.predict_proba(X)
    
    # K-means
    kmeans = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Compare results
    print("EM Algorithm Results:")
    print(f"Means: {em.params['means']}")
    print(f"Variances: {em.params['variances']}")
    print(f"Weights: {em.params['weights']}")
    
    print("\nK-means Results:")
    print(f"Centers: {kmeans.cluster_centers_.flatten()}")
    print(f"Inertia: {kmeans.inertia_:.3f}")
    
    # Compare assignments
    ari_score = adjusted_rand_score(em_labels, kmeans_labels)
    print(f"\nAdjusted Rand Index: {ari_score:.3f}")
    
    return em, kmeans, em_labels, kmeans_labels

# Example comparison
em_result, kmeans_result, em_labels, kmeans_labels = compare_em_kmeans(x)

# Visualize results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# EM responsibilities
scatter1 = ax1.scatter(x, em_result.predict_proba(x)[:, 0], c=em_labels, alpha=0.6)
ax1.set_title('EM Algorithm: Responsibilities')
ax1.set_xlabel('x')
ax1.set_ylabel('P(Z=1|x)')

# K-means assignments
scatter2 = ax2.scatter(x, np.zeros_like(x), c=kmeans_labels, alpha=0.6)
ax2.set_title('K-means: Hard Assignments')
ax2.set_xlabel('x')
ax2.set_ylabel('Cluster')

# Histogram comparison
ax3.hist(x[em_labels == 0], bins=30, alpha=0.7, label='EM Cluster 0', density=True)
ax3.hist(x[em_labels == 1], bins=30, alpha=0.7, label='EM Cluster 1', density=True)
ax3.set_title('EM Algorithm Clusters')
ax3.legend()

ax4.hist(x[kmeans_labels == 0], bins=30, alpha=0.7, label='K-means Cluster 0', density=True)
ax4.hist(x[kmeans_labels == 1], bins=30, alpha=0.7, label='K-means Cluster 1', density=True)
ax4.set_title('K-means Clusters')
ax4.legend()

plt.tight_layout()
plt.show()
```

```r
# R implementation
library(cluster)

compare_em_kmeans <- function(x, n_components=2) {
  # EM Algorithm
  em <- EMAlgorithm(max_iter=100, tol=1e-6)
  em_result <- fit_em(em, x)
  
  # K-means
  kmeans_result <- kmeans(x, centers=n_components, nstart=10)
  
  # Compare results
  cat("EM Algorithm Results:\n")
  cat("Means:", em_result$params$means, "\n")
  cat("Variances:", em_result$params$variances, "\n")
  cat("Weights:", em_result$params$weights, "\n")
  
  cat("\nK-means Results:\n")
  cat("Centers:", kmeans_result$centers, "\n")
  cat("Within SS:", kmeans_result$withinss, "\n")
  
  # Compare assignments
  em_labels <- apply(em_result$responsibilities, 1, which.max) - 1
  ari_score <- adjustedRandIndex(em_labels, kmeans_result$cluster)
  cat("Adjusted Rand Index:", round(ari_score, 3), "\n")
  
  list(em=em_result, kmeans=kmeans_result, em_labels=em_labels, kmeans_labels=kmeans_result$cluster)
}

# Example comparison
comparison <- compare_em_kmeans(x, n_components=2)

# Visualize results
par(mfrow=c(2, 2))

# EM responsibilities
plot(x, comparison$em$responsibilities[,1], col=comparison$em_labels+1, pch=16,
     main="EM Algorithm: Responsibilities", xlab="x", ylab="P(Z=1|x)")

# K-means assignments
plot(x, rep(0, length(x)), col=comparison$kmeans_labels, pch=16,
     main="K-means: Hard Assignments", xlab="x", ylab="Cluster")

# Histogram comparison
hist(x[comparison$em_labels == 0], breaks=30, col="red", alpha=0.7, 
     main="EM Algorithm Clusters", xlab="x", freq=FALSE)
hist(x[comparison$em_labels == 1], breaks=30, col="blue", alpha=0.7, add=TRUE)

hist(x[comparison$kmeans_labels == 0], breaks=30, col="red", alpha=0.7,
     main="K-means Clusters", xlab="x", freq=FALSE)
hist(x[comparison$kmeans_labels == 1], breaks=30, col="blue", alpha=0.7, add=TRUE)
```

## 7.3.4. Alternative View: Variational Perspective

### The Free Energy Objective

The EM algorithm can be viewed as optimizing a **free energy** objective function:

```math
F(q, \theta) = \mathbb{E}_{q(\mathbf{Z})} \left[\log \frac{p(\mathbf{x}, \mathbf{Z} \mid \theta)}{q(\mathbf{Z})}\right]
```

This function can be decomposed as:

```math
F(q, \theta) = \log p(\mathbf{x} \mid \theta) - \text{KL}(q(\mathbf{Z}) \| p(\mathbf{Z} \mid \mathbf{x}, \theta))
```

where:
- $`\log p(\mathbf{x} \mid \theta)`$ is the log-likelihood we want to maximize
- $`\text{KL}(q(\mathbf{Z}) \| p(\mathbf{Z} \mid \mathbf{x}, \theta))`$ is the KL divergence between the variational distribution $`q`$ and the true posterior

### Coordinate Ascent Interpretation

The EM algorithm can be seen as coordinate ascent on $`F(q, \theta)`$:

1. **E-step**: Fix $`\theta`$, optimize $`q`$ to minimize KL divergence
2. **M-step**: Fix $`q`$, optimize $`\theta`$ to maximize log-likelihood

### Implementation: Variational EM

```python
class VariationalEM:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.free_energies = []
        
    def fit(self, X, initial_params=None):
        """Fit using variational EM"""
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        # Initialize variational distribution
        self.q = np.ones((n_samples, 2)) * 0.5  # Uniform initialization
        
        for iteration in range(self.max_iter):
            # E-step: Update variational distribution
            self._update_variational(X)
            
            # M-step: Update parameters
            self._update_parameters(X)
            
            # Compute free energy
            free_energy = self._compute_free_energy(X)
            self.free_energies.append(free_energy)
            
            # Check convergence
            if len(self.free_energies) > 1:
                if abs(self.free_energies[-1] - self.free_energies[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        means = np.random.choice(X, size=2, replace=False)
        variances = np.array([np.var(X)] * 2)
        weights = np.random.dirichlet([1, 1])
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _update_variational(self, X):
        """Update variational distribution (E-step)"""
        n_samples = len(X)
        
        for i in range(n_samples):
            # Compute unnormalized responsibilities
            log_resp = np.zeros(2)
            for k in range(2):
                log_resp[k] = (np.log(self.params['weights'][k]) + 
                             norm.logpdf(X[i], self.params['means'][k], 
                                       np.sqrt(self.params['variances'][k])))
            
            # Normalize using log-sum-exp trick
            max_log = np.max(log_resp)
            exp_log_resp = np.exp(log_resp - max_log)
            self.q[i] = exp_log_resp / np.sum(exp_log_resp)
    
    def _update_parameters(self, X):
        """Update parameters (M-step)"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(self.q[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(self.q[:, k] * X) / 
                                     np.sum(self.q[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(self.q[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(self.q[:, k]))
    
    def _compute_free_energy(self, X):
        """Compute free energy objective"""
        n_samples = len(X)
        free_energy = 0
        
        for i in range(n_samples):
            for k in range(2):
                if self.q[i, k] > 0:
                    # Log-likelihood term
                    log_likelihood = (np.log(self.params['weights'][k]) + 
                                    norm.logpdf(X[i], self.params['means'][k], 
                                              np.sqrt(self.params['variances'][k])))
                    
                    # Entropy term
                    entropy = -np.log(self.q[i, k])
                    
                    free_energy += self.q[i, k] * (log_likelihood - entropy)
        
        return free_energy
    
    def predict_proba(self, X):
        """Predict component probabilities"""
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            log_resp = np.zeros(2)
            for k in range(2):
                log_resp[k] = (np.log(self.params['weights'][k]) + 
                             norm.logpdf(X[i], self.params['means'][k], 
                                       np.sqrt(self.params['variances'][k])))
            
            max_log = np.max(log_resp)
            exp_log_resp = np.exp(log_resp - max_log)
            responsibilities[i] = exp_log_resp / np.sum(exp_log_resp)
        
        return responsibilities

# Example usage
vem = VariationalEM(max_iter=100, tol=1e-6)
vem.fit(x)

print("Variational EM Results:")
print(f"Means: {vem.params['means']}")
print(f"Variances: {vem.params['variances']}")
print(f"Weights: {vem.params['weights']}")

# Plot free energy convergence
plt.figure(figsize=(10, 6))
plt.plot(vem.free_energies)
plt.title('Variational EM: Free Energy Convergence')
plt.xlabel('Iteration')
plt.ylabel('Free Energy')
plt.grid(True, alpha=0.3)
plt.show()
```

## 7.3.5. Variational EM with Factorized Approximations

### Mean Field Approximation

When the exact posterior $`p(\mathbf{Z} \mid \mathbf{x}, \theta)`$ is computationally intractable, we can use a **factorized approximation**:

```math
q(\mathbf{Z}) = \prod_{i=1}^n q_i(Z_i)
```

This is known as the **mean field approximation**.

### Factorized Variational EM

```python
class FactorizedVariationalEM:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.free_energies = []
        
    def fit(self, X, initial_params=None):
        """Fit using factorized variational EM"""
        n_samples = len(X)
        
        # Initialize parameters
        if initial_params is None:
            self.params = self._initialize_parameters(X)
        else:
            self.params = initial_params.copy()
        
        # Initialize factorized variational distribution
        self.q_factors = np.ones((n_samples, 2)) * 0.5
        
        for iteration in range(self.max_iter):
            # Update each factor independently
            self._update_factors(X)
            
            # Update parameters
            self._update_parameters(X)
            
            # Compute free energy
            free_energy = self._compute_free_energy(X)
            self.free_energies.append(free_energy)
            
            # Check convergence
            if len(self.free_energies) > 1:
                if abs(self.free_energies[-1] - self.free_energies[-2]) < self.tol:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        
        return self
    
    def _initialize_parameters(self, X):
        """Initialize parameters randomly"""
        means = np.random.choice(X, size=2, replace=False)
        variances = np.array([np.var(X)] * 2)
        weights = np.random.dirichlet([1, 1])
        return {'means': means, 'variances': variances, 'weights': weights}
    
    def _update_factors(self, X):
        """Update factorized variational distribution"""
        n_samples = len(X)
        
        for i in range(n_samples):
            # Compute expected log-likelihood for each component
            expected_log_likelihood = np.zeros(2)
            
            for k in range(2):
                # Prior term
                expected_log_likelihood[k] = np.log(self.params['weights'][k])
                
                # Likelihood term
                expected_log_likelihood[k] += norm.logpdf(X[i], self.params['means'][k], 
                                                        np.sqrt(self.params['variances'][k]))
            
            # Update factor using softmax
            max_log = np.max(expected_log_likelihood)
            exp_log = np.exp(expected_log_likelihood - max_log)
            self.q_factors[i] = exp_log / np.sum(exp_log)
    
    def _update_parameters(self, X):
        """Update parameters using factorized approximation"""
        n_samples = len(X)
        
        for k in range(2):
            # Update weights
            self.params['weights'][k] = np.mean(self.q_factors[:, k])
            
            # Update means
            self.params['means'][k] = (np.sum(self.q_factors[:, k] * X) / 
                                     np.sum(self.q_factors[:, k]))
            
            # Update variances
            self.params['variances'][k] = (np.sum(self.q_factors[:, k] * 
                                                 (X - self.params['means'][k])**2) / 
                                          np.sum(self.q_factors[:, k]))
    
    def _compute_free_energy(self, X):
        """Compute free energy with factorized approximation"""
        n_samples = len(X)
        free_energy = 0
        
        for i in range(n_samples):
            for k in range(2):
                if self.q_factors[i, k] > 0:
                    # Expected log-likelihood
                    expected_ll = (np.log(self.params['weights'][k]) + 
                                 norm.logpdf(X[i], self.params['means'][k], 
                                           np.sqrt(self.params['variances'][k])))
                    
                    # Entropy of factor
                    entropy = -np.log(self.q_factors[i, k])
                    
                    free_energy += self.q_factors[i, k] * (expected_ll - entropy)
        
        return free_energy

# Example usage
fvem = FactorizedVariationalEM(max_iter=100, tol=1e-6)
fvem.fit(x)

print("Factorized Variational EM Results:")
print(f"Means: {fvem.params['means']}")
print(f"Variances: {fvem.params['variances']}")
print(f"Weights: {fvem.params['weights']}")

# Compare different EM variants
em_variants = {
    'Standard EM': em_result,
    'Variational EM': vem,
    'Factorized VEM': fvem
}

print("\nComparison of EM Variants:")
for name, variant in em_variants.items():
    if hasattr(variant, 'params'):
        print(f"{name}: Final LL = {variant.log_likelihoods[-1] if hasattr(variant, 'log_likelihoods') else 'N/A'}")
    else:
        print(f"{name}: Final LL = {variant.free_energies[-1] if hasattr(variant, 'free_energies') else 'N/A'}")
```

### Advantages of Variational EM

1. **Computational Efficiency**: Factorized approximations can be much faster
2. **Scalability**: Can handle large datasets more efficiently
3. **Flexibility**: Can incorporate constraints on the variational distribution
4. **Theoretical Guarantees**: Provides lower bounds on the log-likelihood

### When to Use Each Variant

- **Standard EM**: When exact posterior is tractable and computational cost is acceptable
- **Variational EM**: When exact posterior is intractable but full variational distribution is manageable
- **Factorized VEM**: When full variational distribution is too complex, use mean field approximation

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of the EM algorithm and its variants. The code examples demonstrate both the theoretical concepts and their practical application.
