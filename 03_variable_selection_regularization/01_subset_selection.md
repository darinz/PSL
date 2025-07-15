# 3.1. Subset Selection

Variable selection, also known as feature selection or subset selection, is a fundamental technique in statistical modeling that addresses the challenge of identifying the most relevant predictors from a potentially large set of candidate variables. This process is crucial for building interpretable, efficient, and generalizable models.

## 3.1.1. Why Subset Selection: The Curse of Dimensionality and Model Complexity

In modern statistical applications, there is often a vast array of potential predictors. Sometimes, the number of predictors $`p`$ can exceed the sample size $`n`$, leading to what is known as the **curse of dimensionality**. In certain applications, the primary objective is to pinpoint a subset of these predictors that have the most significant relevance to the response variable. For such tasks, variable selection becomes indispensable.

### The Fundamental Question: More Variables = Better Predictions?

However, if our sole aim is to achieve accurate predictions without being concerned about the relevance of predictors in our regression model to $`Y`$, do we still need variable selection? Can adding more variables always lead to better predictions?

This question touches on the fundamental trade-off between **model complexity** and **generalization ability**. To understand this trade-off, we need to explore the theoretical foundations of training and test errors in linear regression.

### Mathematical Foundation: Training vs. Test Error

Let's embark on a theoretical exploration of the training and test errors in a linear regression model. Consider a training dataset $`\{(\mathbf{x}_i, y_i)\}_{i=1}^n`$ of size $`n`$. Using this data, we can fit a linear regression model, yielding a least squares estimate $`\hat{\boldsymbol{\beta}}`$.

**Training Error Definition:**
```math
\text{Train Err} = \|\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}}\|^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2
```

where $`\hat{\boldsymbol{\beta}} \in \mathbb{R}^p`$ is the least squares estimate of the regression parameter.

**Test Error Definition:**
Now, consider a separate test dataset $`\{(\mathbf{x}_i, y_i^*)\}_{i=1}^n`$ collected at the same locations $`\mathbf{x}_i`$'s. The test error is:

```math
\text{Test Err} = \|\mathbf{y}^* - \mathbf{X} \hat{\boldsymbol{\beta}}\|^2 = \sum_{i=1}^n (y_i^* - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2
```

### Key Assumptions and Relationships

It is crucial to note that while both $`y_i`$ and $`y_i^*`$ are random and independent of each other, they are assumed to follow the same distribution with:
- **Mean**: $`f(\mathbf{x}_i)`$ (the true underlying function)
- **Variance**: $`\sigma^2`$ (constant error variance)

The estimate $`\hat{\boldsymbol{\beta}}`$ is also random, with its randomness originating from the training data $`\mathbf{y}`$. This means that:
- $`\mathbf{y}`$ and $`\hat{\boldsymbol{\beta}}`$ are **correlated** (both depend on the same training data)
- $`\mathbf{y}^*`$ and $`\hat{\boldsymbol{\beta}}`$ are **independent** (test data is independent of training data)

### Expected Error Decomposition

If we break down the expectations of both errors, they can be segmented into three fundamental components:

```math
\begin{aligned}
\mathbb{E}[\text{Train Err}] &= \text{(Unavoidable Error)} - p\sigma^2 + \text{Bias}^2 \\
\mathbb{E}[\text{Test Err}] &= \text{(Unavoidable Error)} + p\sigma^2 + \text{Bias}^2
\end{aligned}
```

#### Component Analysis:

**1. Unavoidable Error:**
```math
\text{Unavoidable Error} = n\sigma^2
```
This error persists even if we knew the true function $`f`$. It represents the irreducible error due to the inherent noise in the data.

**2. Bias Term:**
```math
\text{Bias}^2 = \sum_{i=1}^n [f(\mathbf{x}_i) - \mathbf{x}_i^T \boldsymbol{\beta}]^2
```
This emerges if the true function $`f`$ deviates from linearity or if we include an incomplete set of predictors in our model.

**3. Dimensional Error Term:**
```math
\text{Dimensional Error} = p\sigma^2
```
This is where things get intriguing. Its sign changes between training and test errors:
- **Positive sign in test error**: Arises because we rely on the estimated $`\hat{\boldsymbol{\beta}}`$ instead of the true $`\boldsymbol{\beta}`$
- **Negative sign in training error**: Can be attributed to the positive correlation between $`\hat{\boldsymbol{\beta}}`$ and $`\mathbf{y}`$

### Mathematical Derivation of Error Decomposition

Let's derive this decomposition step by step:

**For Training Error:**
```math
\begin{aligned}
\mathbb{E}[\text{Train Err}] &= \mathbb{E}[\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{H}\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{H}\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}\|^2 - 2\mathbf{y}^T\mathbf{H}\mathbf{y} + \mathbf{y}^T\mathbf{H}^T\mathbf{H}\mathbf{y}] \\
&= \mathbb{E}[\|\mathbf{y}\|^2 - \mathbf{y}^T\mathbf{H}\mathbf{y}] \\
&= n\sigma^2 + \|f(\mathbf{X})\|^2 - (n\sigma^2 + \|f(\mathbf{X})\|^2 - p\sigma^2) \\
&= n\sigma^2 - p\sigma^2 + \text{Bias}^2
\end{aligned}
```

where $`\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T`$ is the hat matrix.

**For Test Error:**
```math
\begin{aligned}
\mathbb{E}[\text{Test Err}] &= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\boldsymbol{\beta} + \mathbf{X}\boldsymbol{\beta} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\boldsymbol{\beta}\|^2] + \mathbb{E}[\|\mathbf{X}(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})\|^2] \\
&= n\sigma^2 + p\sigma^2 + \text{Bias}^2
\end{aligned}
```

### Practical Implications

This theoretical framework reveals several important insights:

1. **The Bias-Variance Trade-off**: Adding more predictors can reduce bias but increases variance (dimensional error)

2. **Overfitting Risk**: The training error systematically underestimates the test error by $`2p\sigma^2`$

3. **Optimal Model Complexity**: There exists an optimal number of predictors that minimizes test error

4. **Variable Selection Necessity**: Even for pure prediction, variable selection is crucial to avoid overfitting

### Python Example: Demonstrating the Error Decomposition

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate data with known true function
n = 100
p_max = 20
X = np.random.randn(n, p_max)

# True model: only first 5 variables matter
beta_true = np.zeros(p_max)
beta_true[:5] = [1.5, -0.8, 0.6, -0.4, 0.3]
f_true = X @ beta_true
y = f_true + np.random.normal(0, 0.5, n)

# Function to calculate errors for different model sizes
def calculate_errors(X, y, p_values):
    train_errors = []
    test_errors = []
    
    for p in p_values:
        # Use only first p predictors
        X_p = X[:, :p]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_p, y, test_size=0.3, random_state=42
        )
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate errors
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        train_errors.append(train_mse)
        test_errors.append(test_mse)
    
    return np.array(train_errors), np.array(test_errors)

# Calculate errors for different model sizes
p_values = range(1, p_max + 1)
train_errors, test_errors = calculate_errors(X, y, p_values)

# Theoretical decomposition
sigma2_est = np.var(y - f_true)  # Estimate of sigma^2
unavoidable_error = sigma2_est
bias_squared = np.array([np.mean((f_true - X[:, :p] @ beta_true[:p])**2) for p in p_values])
dimensional_error = np.array([p * sigma2_est for p in p_values])

# Expected errors
expected_train_error = unavoidable_error - dimensional_error + bias_squared
expected_test_error = unavoidable_error + dimensional_error + bias_squared

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Observed errors
axes[0, 0].plot(p_values, train_errors, 'bo-', label='Observed Train Error', linewidth=2)
axes[0, 0].plot(p_values, test_errors, 'ro-', label='Observed Test Error', linewidth=2)
axes[0, 0].set_xlabel('Number of Predictors (p)')
axes[0, 0].set_ylabel('Mean Squared Error')
axes[0, 0].set_title('Observed Training vs Test Error')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='True Model Size')

# Expected errors
axes[0, 1].plot(p_values, expected_train_error, 'b--', label='Expected Train Error', linewidth=2)
axes[0, 1].plot(p_values, expected_test_error, 'r--', label='Expected Test Error', linewidth=2)
axes[0, 1].set_xlabel('Number of Predictors (p)')
axes[0, 1].set_ylabel('Expected Error')
axes[0, 1].set_title('Theoretical Error Decomposition')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=5, color='green', linestyle='--', alpha=0.7, label='True Model Size')

# Error components
axes[1, 0].plot(p_values, unavoidable_error * np.ones_like(p_values), 'g-', 
                label='Unavoidable Error', linewidth=2)
axes[1, 0].plot(p_values, bias_squared, 'm-', label='Bias²', linewidth=2)
axes[1, 0].plot(p_values, dimensional_error, 'c-', label='Dimensional Error', linewidth=2)
axes[1, 0].set_xlabel('Number of Predictors (p)')
axes[1, 0].set_ylabel('Error Component')
axes[1, 0].set_title('Error Components')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Overfitting demonstration
axes[1, 1].plot(p_values, test_errors - train_errors, 'ko-', linewidth=2)
axes[1, 1].set_xlabel('Number of Predictors (p)')
axes[1, 1].set_ylabel('Test Error - Train Error')
axes[1, 1].set_title('Overfitting Gap')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print key insights
print("=== ERROR DECOMPOSITION INSIGHTS ===")
print(f"Unavoidable Error: {unavoidable_error:.4f}")
print(f"Optimal model size (observed): {p_values[np.argmin(test_errors)]}")
print(f"Optimal model size (theoretical): {p_values[np.argmin(expected_test_error)]}")
print(f"True model size: 5")

print(f"\nOverfitting gap at p={p_max}: {test_errors[-1] - train_errors[-1]:.4f}")
print(f"Theoretical gap: {2 * p_max * sigma2_est:.4f}")
```

It is crucial to note that whole both y_i and y-star_i are random and independent of each other. They are assumed to follow the same distribution which has a mean of f(x_i) and variance sigma-square. Another random term is beta-hat, whose randomness originates from the data y. This means that y and beta-hat are correlated (therefore, both colored in blue) but y-star (colored in red) and beta-hat are independent.

If we break down the expectations of both errors, they can be segmented into three parts ([see the derivation here](https://liangfgithub.github.io/Notes/lec_W3_VariableSelection_appendix.pdf)):

$$\begin{split}\mathbb{E} [ \text{Train Err}  ] & = \text{(Unavoidable Err)} - p \sigma^2  + \text{Bias} \\
\mathbb{E} [ \text{Test Err}  ] &= \text{(Unavoidable Err)} + p \sigma^2  + \text{Bias}\end{split}$$

1. **Unavoidable Error**: This error persists even if we knew true function f. When the error terms are assumed to be independent with mean zero and variance sigma-square, the unavoidable error is equal to n times sigma-square.

2. **Bias**: This emerges if the true function f deviates from linearity or if, for instance, it involves three predictors, but we include only two in our model.

3. **Dimensional Error term**, $p \sigma^2$: This is where things get intriguing. Its sign changes between training and test errors. The positive sign in the test error arises because of our reliance on the estimated beta instead of the true beta. The negative sign in the training error can be attributed to the positive correlation between beta-hat and y.

In conclusion, whether our primary objective lies in identifying a subset of relevant predictors or merely in enhancing prediction accuracy, it becomes evident that variable selection plays a crucial role.

## 3.1.2. Selection Criteria: Balancing Fit and Complexity

How do we determine which variables to retain and which to discard? This is one of the most fundamental challenges in statistical modeling, requiring a careful balance between model fit and complexity.

### The P-Value Pitfall

One might initially think of using p-values obtained from a linear regression model that includes all variables. In the resulting summary table, each variable is assigned a p-value. A common practice might be to use these p-values, setting a threshold (like 5%), and dropping variables with values exceeding this. But is this approach optimal?

**The Fundamental Problem:**
The crux of the issue is that a variable's p-value is contingent upon the other variables included in the model. Recall that the p-value for a variable assesses its **conditional** contribution in the presence of other variables in the model. If we remove any variable, the entire set of p-values could shift dramatically.

**Mathematical Illustration:**
Consider a model with two correlated predictors $`X_1`$ and $`X_2`$:
```math
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon
```

The p-value for $`\beta_1`$ depends on whether $`X_2`$ is in the model:
- **With $`X_2`$**: Tests $`H_0: \beta_1 = 0`$ conditional on $`X_2`$
- **Without $`X_2`$**: Tests $`H_0: \beta_1 = 0`$ in the presence of omitted variable bias

Thus, simply using a snapshot of p-values from a full model is not recommended.

### Model Scoring Approach

Instead of using p-values, we can assign a score to each model and then utilize an algorithm to determine the best one. Here, 'model' refers to a linear regression model containing a specific subset of variables.

**The Combinatorial Challenge:**
Imagine we have 10 non-intercept predictors. Excluding the intercept, which is always present, our subset of variables will be a combination of these 10 predictors. The potential models can be indexed using binary vectors, with a '1' indicating the presence of a variable and '0' its absence.

**Number of Possible Models:**
```math
\text{Number of models} = 2^p
```

For $`p = 10`$, this gives $`2^{10} = 1024`$ possible models. Even for just 10 predictors, the model possibilities exceed a thousand, underscoring the significance of efficient search algorithms.

### Mathematical Framework for Model Scoring

The score for model selection typically comprises two components:

```math
\text{Model Score} = \text{Goodness of Fit} + \text{Complexity Penalty}
```

**1. Goodness of Fit Measure:**
Often an increasing function of the residual sum of squares (RSS):
```math
\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2
```

**2. Complexity Penalty Term:**
Often an increasing function of $`p`$, the number of non-intercept variables. This penalty discourages overfitting by penalizing model complexity.

### Popular Model Selection Criteria

#### 1. Mallow's $`C_p`$ Statistic

**Definition:**
```math
C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2_{\text{full}}} - n + 2p
```

**Alternative Form:**
```math
C_p = \text{RSS}_p + 2\hat{\sigma}^2_{\text{full}} \times p
```

**Interpretation:**
- $`\text{RSS}_p`$ = Residual sum of squares for the model with $`p`$ predictors
- $`\hat{\sigma}^2_{\text{full}}`$ = Estimated error variance from the full model
- The model with $`C_p \approx p`$ is considered optimal

**Theoretical Foundation:**
Mallow's $`C_p`$ estimates the expected prediction error:
```math
\mathbb{E}[C_p] \approx \mathbb{E}[\text{Test Error}]
```

#### 2. Akaike Information Criterion (AIC)

**Definition:**
```math
\text{AIC} = -2\log L(\hat{\boldsymbol{\beta}}) + 2p
```

**For Linear Regression with Normal Errors:**
```math
\text{AIC} = n\log(\text{RSS}/n) + 2p + \text{constant}
```

**Interpretation:**
- Balances model fit (log-likelihood) with complexity (number of parameters)
- Penalty of 2 per additional parameter
- AIC estimates the relative Kullback-Leibler divergence

#### 3. Bayesian Information Criterion (BIC)

**Definition:**
```math
\text{BIC} = -2\log L(\hat{\boldsymbol{\beta}}) + (\log n)p
```

**For Linear Regression with Normal Errors:**
```math
\text{BIC} = n\log(\text{RSS}/n) + (\log n)p + \text{constant}
```

**Interpretation:**
- Similar to AIC but with stronger penalty for complexity
- Penalty of $`\log n`$ per additional parameter
- BIC estimates the posterior probability of the model

### Mathematical Comparison of Criteria

**Penalty Comparison:**
```math
\begin{aligned}
\text{Mallow's } C_p &: \text{Constant penalty} \\
\text{AIC} &: 2p \text{ penalty} \\
\text{BIC} &: (\log n)p \text{ penalty}
\end{aligned}
```

**Asymptotic Behavior:**
- **AIC**: Penalty remains constant as $`n \to \infty`$
- **BIC**: Penalty grows with $`\log n`$ as $`n \to \infty`$
- **Mallow's $`C_p`$**: Constant penalty, closely related to AIC

### Comprehensive Python Example: Model Selection Criteria

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# Generate data with known true model
n = 100
p_total = 8
X = np.random.randn(n, p_total)

# True model: only first 4 variables matter
beta_true = np.zeros(p_total)
beta_true[:4] = [1.5, -0.8, 0.6, -0.4]
f_true = X @ beta_true
y = f_true + np.random.normal(0, 0.5, n)

# Fit full model to get sigma^2 estimate
full_model = LinearRegression()
full_model.fit(X, y)
y_pred_full = full_model.predict(X)
sigma2_full = np.sum((y - y_pred_full)**2) / (n - p_total - 1)

print("=== TRUE MODEL ===")
print(f"True coefficients: {beta_true}")
print(f"Estimated σ² from full model: {sigma2_full:.4f}")

# Function to calculate model selection criteria
def calculate_criteria(X_subset, y, sigma2_full, n):
    """Calculate AIC, BIC, and Mallow's Cp for a given model"""
    model = LinearRegression()
    model.fit(X_subset, y)
    y_pred = model.predict(X_subset)
    
    p = X_subset.shape[1]
    rss = np.sum((y - y_pred)**2)
    
    # AIC (for normal errors)
    aic = n * np.log(rss/n) + 2*p
    
    # BIC (for normal errors)
    bic = n * np.log(rss/n) + np.log(n)*p
    
    # Mallow's Cp
    cp = rss/sigma2_full - n + 2*p
    
    return aic, bic, cp, rss

# Generate all possible model combinations
all_models = []
criteria_results = []

for p in range(1, p_total + 1):
    for subset in combinations(range(p_total), p):
        X_subset = X[:, subset]
        aic, bic, cp, rss = calculate_criteria(X_subset, y, sigma2_full, n)
        
        all_models.append(list(subset))
        criteria_results.append({
            'p': p,
            'aic': aic,
            'bic': bic,
            'cp': cp,
            'rss': rss,
            'variables': subset
        })

# Convert to DataFrame
results_df = pd.DataFrame(criteria_results)

# Find best models according to each criterion
best_aic_idx = results_df['aic'].idxmin()
best_bic_idx = results_df['bic'].idxmin()
best_cp_idx = results_df['cp'].idxmin()

print("\n=== MODEL SELECTION RESULTS ===")
print("Best model by AIC:")
print(f"  Variables: {results_df.loc[best_aic_idx, 'variables']}")
print(f"  AIC: {results_df.loc[best_aic_idx, 'aic']:.2f}")
print(f"  p: {results_df.loc[best_aic_idx, 'p']}")

print("\nBest model by BIC:")
print(f"  Variables: {results_df.loc[best_bic_idx, 'variables']}")
print(f"  BIC: {results_df.loc[best_bic_idx, 'bic']:.2f}")
print(f"  p: {results_df.loc[best_bic_idx, 'p']}")

print("\nBest model by Mallow's Cp:")
print(f"  Variables: {results_df.loc[best_cp_idx, 'variables']}")
print(f"  Cp: {results_df.loc[best_cp_idx, 'cp']:.2f}")
print(f"  p: {results_df.loc[best_cp_idx, 'p']}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# AIC vs number of predictors
for p in range(1, p_total + 1):
    mask = results_df['p'] == p
    if mask.sum() > 0:
        best_aic_p = results_df[mask]['aic'].min()
        axes[0, 0].scatter(p, best_aic_p, c='blue', s=100, alpha=0.7)

axes[0, 0].set_xlabel('Number of Predictors (p)')
axes[0, 0].set_ylabel('Best AIC')
axes[0, 0].set_title('AIC vs Model Size')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

# BIC vs number of predictors
for p in range(1, p_total + 1):
    mask = results_df['p'] == p
    if mask.sum() > 0:
        best_bic_p = results_df[mask]['bic'].min()
        axes[0, 1].scatter(p, best_bic_p, c='green', s=100, alpha=0.7)

axes[0, 1].set_xlabel('Number of Predictors (p)')
axes[0, 1].set_ylabel('Best BIC')
axes[0, 1].set_title('BIC vs Model Size')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

# Mallow's Cp vs number of predictors
for p in range(1, p_total + 1):
    mask = results_df['p'] == p
    if mask.sum() > 0:
        best_cp_p = results_df[mask]['cp'].min()
        axes[1, 0].scatter(p, best_cp_p, c='red', s=100, alpha=0.7)

axes[1, 0].set_xlabel('Number of Predictors (p)')
axes[1, 0].set_ylabel('Best Mallow\'s Cp')
axes[1, 0].set_title('Mallow\'s Cp vs Model Size')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

# RSS vs number of predictors
for p in range(1, p_total + 1):
    mask = results_df['p'] == p
    if mask.sum() > 0:
        best_rss_p = results_df[mask]['rss'].min()
        axes[1, 1].scatter(p, best_rss_p, c='purple', s=100, alpha=0.7)

axes[1, 1].set_xlabel('Number of Predictors (p)')
axes[1, 1].set_ylabel('Best RSS')
axes[1, 1].set_title('RSS vs Model Size')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='True Model Size')

plt.tight_layout()
plt.show()

# Compare criteria behavior
print("\n=== CRITERIA COMPARISON ===")
print("Model sizes selected by each criterion:")
print(f"AIC: {results_df.loc[best_aic_idx, 'p']} predictors")
print(f"BIC: {results_df.loc[best_bic_idx, 'p']} predictors")
print(f"Mallow's Cp: {results_df.loc[best_cp_idx, 'p']} predictors")
print(f"True model: 4 predictors")

# Show penalty comparison
print(f"\nPenalty comparison (n={n}, log(n)={np.log(n):.2f}):")
print(f"AIC penalty per parameter: 2")
print(f"BIC penalty per parameter: {np.log(n):.2f}")
print(f"Mallow's Cp penalty per parameter: 2 (scaled by σ²)")
```

AIC and BIC are versatile and can be applied to any statistical model. Although one might default to AIC and BIC, it's useful to consider Mallow's Cp, which aligns well with our theoretical understanding of training and test errors. The intent is always to minimize the test error, making Mallow's Cp a robust choice for model selection, especially in the context of linear regression.

## 3.1.3. AIC vs BIC: Philosophical and Practical Differences

AIC and BIC are the two most widely used model selection criteria, but they embody different philosophical approaches and practical considerations. Understanding their differences is crucial for making informed decisions in variable selection.

### Mathematical Comparison

**Key Differences**: Both AIC and BIC serve as model selection criteria, with the primary difference being in their penalty terms. The coefficients, specifically the '2' in AIC and $`\log(n)`$ in BIC, can be thought of as the "cost" associated with adding an additional predictor to the model.

**Mathematical Formulation:**
```math
\begin{aligned}
\text{AIC} &= -2\log L(\hat{\boldsymbol{\beta}}) + 2p \\
\text{BIC} &= -2\log L(\hat{\boldsymbol{\beta}}) + \log(n)p
\end{aligned}
```

**Penalty Comparison:**
- **AIC Penalty**: $`2p`$ (constant per parameter)
- **BIC Penalty**: $`\log(n)p`$ (grows with sample size)

### Asymptotic Behavior Analysis

**Sample Size Dependence:**
As the sample size $`n`$ grows:
- **AIC**: The cost incurred remains constant at 2 per parameter
- **BIC**: The cost increases with $`\log(n)`$ per parameter

**Mathematical Illustration:**
```math
\begin{aligned}
\text{AIC penalty ratio} &= \frac{2(p+1)}{2p} = 1 + \frac{1}{p} \\
\text{BIC penalty ratio} &= \frac{\log(n)(p+1)}{\log(n)p} = 1 + \frac{1}{p}
\end{aligned}
```

While the ratios are mathematically identical, the absolute penalties differ dramatically:
- For $`n = 100`$: BIC penalty = $`\log(100) \approx 4.6`$ per parameter
- For $`n = 1000`$: BIC penalty = $`\log(1000) \approx 6.9`$ per parameter
- For $`n = 10000`$: BIC penalty = $`\log(10000) \approx 9.2`$ per parameter

### Practical Implications

**Model Selection Behavior:**
Given the distinct penalties, it's common to see AIC and BIC favor different models when applied to the same dataset. Generally:
- **AIC tends to select larger models** compared to BIC
- **BIC tends to select more parsimonious models** with fewer predictors

**Mallow's Cp Relationship:**
Mallow's $`C_p`$ aligns closely with AIC because its penalties are constant and don't hinge on $`n`$. However, for many practical purposes, focusing on AIC and BIC might suffice.

### Underlying Philosophies

#### AIC: Prediction-Oriented Approach

**Philosophy**: AIC aims to minimize the predictive error. It prioritizes accurate predictions even if it means including variables that might not necessarily be crucial.

**Theoretical Foundation:**
AIC estimates the relative Kullback-Leibler divergence between the true model and the candidate model:
```math
\text{AIC} \approx 2 \times \text{KL divergence} + \text{constant}
```

**Key Characteristics:**
- **Prediction-focused**: Optimizes for out-of-sample prediction accuracy
- **Less conservative**: Willing to include potentially irrelevant variables
- **Sample size invariant**: Penalty doesn't change with sample size

#### BIC: Model Identification Approach

**Philosophy**: BIC focuses on model parsimony and identifying truly relevant variables. It's more conservative and emphasizes the exclusion of unnecessary predictors.

**Theoretical Foundation:**
BIC approximates the posterior probability of the model (under certain assumptions):
```math
P(\text{Model} | \text{Data}) \propto \exp\left(-\frac{1}{2}\text{BIC}\right)
```

**Key Characteristics:**
- **Model identification-focused**: Aims to find the "true" model
- **More conservative**: Stronger penalty for complexity
- **Sample size dependent**: Penalty increases with sample size

### Error Types in Variable Selection

Two primary errors can arise during variable selection:

1. **Excluding Signals (Type I Error)**: Leaving out variables crucial to $`y`$
2. **Including Noise (Type II Error)**: Incorporating variables that don't significantly impact $`y``

**Impact Analysis:**
While BIC considers both errors equally significant, their impacts on predictions differ:

**Including Irrelevant Variables:**
- An irrelevant variable included in the model will have its influence diminish as the sample size increases
- The estimated coefficient will eventually move towards zero
- **Impact**: Minimal long-term harm to prediction accuracy

**Excluding Relevant Variables:**
- Excluding a relevant variable introduces a bias that persists regardless of sample size
- This bias cannot be overcome by increasing sample size
- **Impact**: Persistent degradation of prediction accuracy

**Result**: AIC has a lighter penalty for adding new predictors because the cost of including noise is lower than the cost of excluding signals.

### Comprehensive Python Example: AIC vs BIC Comparison

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import combinations

# Set random seed for reproducibility
np.random.seed(42)

# Generate data with different sample sizes
sample_sizes = [50, 100, 200, 500, 1000]
p_total = 10

results = []

for n in sample_sizes:
    # Generate data
    X = np.random.randn(n, p_total)
    
    # True model: only first 3 variables matter
    beta_true = np.zeros(p_total)
    beta_true[:3] = [1.5, -0.8, 0.6]
    f_true = X @ beta_true
    y = f_true + np.random.normal(0, 0.5, n)
    
    # Calculate criteria for all possible models
    for p in range(1, min(6, p_total + 1)):  # Limit to 5 predictors for computational efficiency
        for subset in combinations(range(p_total), p):
            X_subset = X[:, subset]
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            
            rss = np.sum((y - y_pred)**2)
            
            # Calculate AIC and BIC
            aic = n * np.log(rss/n) + 2*p
            bic = n * np.log(rss/n) + np.log(n)*p
            
            # Check if this is the true model (first 3 variables)
            is_true_model = set(subset) == set(range(3))
            
            results.append({
                'n': n,
                'p': p,
                'aic': aic,
                'bic': bic,
                'is_true_model': is_true_model,
                'variables': subset
            })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Analysis by sample size
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, n in enumerate(sample_sizes):
    mask = results_df['n'] == n
    
    # Find best models by each criterion
    best_aic_idx = results_df[mask]['aic'].idxmin()
    best_bic_idx = results_df[mask]['bic'].idxmin()
    
    best_aic_p = results_df.loc[best_aic_idx, 'p']
    best_bic_p = results_df.loc[best_bic_idx, 'p']
    
    # Plot AIC vs BIC for this sample size
    axes[0, i].scatter(results_df[mask]['aic'], results_df[mask]['bic'], 
                      c=results_df[mask]['is_true_model'], cmap='viridis', alpha=0.6)
    axes[0, i].scatter(results_df.loc[best_aic_idx, 'aic'], results_df.loc[best_aic_idx, 'bic'], 
                      c='red', s=200, marker='*', label='Best AIC')
    axes[0, i].scatter(results_df.loc[best_bic_idx, 'aic'], results_df.loc[best_bic_idx, 'bic'], 
                      c='blue', s=200, marker='s', label='Best BIC')
    
    axes[0, i].set_xlabel('AIC')
    axes[0, i].set_ylabel('BIC')
    axes[0, i].set_title(f'n = {n}\nAIC: {best_aic_p} vars, BIC: {best_bic_p} vars')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # Plot penalty comparison
    p_values = range(1, 6)
    aic_penalties = [2*p for p in p_values]
    bic_penalties = [np.log(n)*p for p in p_values]
    
    axes[1, i].plot(p_values, aic_penalties, 'ro-', label='AIC Penalty', linewidth=2)
    axes[1, i].plot(p_values, bic_penalties, 'bo-', label='BIC Penalty', linewidth=2)
    axes[1, i].set_xlabel('Number of Parameters (p)')
    axes[1, i].set_ylabel('Penalty')
    axes[1, i].set_title(f'Penalty Comparison (n={n})')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("=== AIC vs BIC COMPARISON SUMMARY ===")
for n in sample_sizes:
    mask = results_df['n'] == n
    best_aic_idx = results_df[mask]['aic'].idxmin()
    best_bic_idx = results_df[mask]['bic'].idxmin()
    
    aic_p = results_df.loc[best_aic_idx, 'p']
    bic_p = results_df.loc[best_bic_idx, 'p']
    aic_true = results_df.loc[best_aic_idx, 'is_true_model']
    bic_true = results_df.loc[best_bic_idx, 'is_true_model']
    
    print(f"\nn = {n}:")
    print(f"  AIC selects {aic_p} variables (true model: {aic_true})")
    print(f"  BIC selects {bic_p} variables (true model: {bic_true})")
    print(f"  BIC penalty factor: {np.log(n):.2f}")

# Theoretical analysis
print(f"\n=== THEORETICAL INSIGHTS ===")
print("As sample size increases:")
print("- AIC penalty remains constant at 2 per parameter")
print("- BIC penalty increases with log(n)")
print("- BIC becomes more conservative with larger samples")
print("- AIC maintains prediction focus regardless of sample size")
```

### Decision Guidelines

**When to Use AIC:**
- Primary goal is **prediction accuracy**
- Sample size is **small to moderate**
- You're willing to include potentially irrelevant variables
- Focus is on **out-of-sample performance**

**When to Use BIC:**
- Primary goal is **model identification**
- Sample size is **large**
- You want a **parsimonious model**
- Focus is on **understanding variable importance**

**In Conclusion**: If your primary goal is prediction, lean towards AIC. But if you're keen on selecting a model with only truly relevant features, BIC is your go-to. The choice should align with your research objectives and the nature of your data.

## 3.1.4. Search Algorithms: Finding the Optimal Model

Once you've chosen your model selection criteria, the next step is to employ a search algorithm to pinpoint the model with the smallest score. This is a critical step that balances computational efficiency with finding the optimal solution.

### The Computational Challenge

**Problem Complexity:**
For $`p`$ predictors, there are $`2^p`$ possible models to evaluate:
```math
\text{Number of models} = \sum_{k=0}^p \binom{p}{k} = 2^p
```

**Computational Growth:**
- $`p = 10`$: 1,024 models
- $`p = 20`$: 1,048,576 models
- $`p = 30`$: 1,073,741,824 models

This exponential growth makes exhaustive search computationally infeasible for large $`p`$.

### Level-wise Search Algorithm (Best Subset Selection)

A popular method is the 'level-wise search algorithm', which works as follows:

#### Algorithm Steps:

**1. Grouping Models by Size:**
Imagine there are $`p`$ potential predictors. Models can then be grouped into $`p`$ groups:
- **Group 1**: Models with 1 predictor ($`\binom{p}{1}`$ models)
- **Group 2**: Models with 2 predictors ($`\binom{p}{2}`$ models)
- ...
- **Group p**: Model with all $`p`$ predictors (1 model)

**2. Identifying Optimal Models Within Groups:**
Given that models within a group share the same size, their penalties are identical. Therefore, within each group, the model with the smallest residual sum of squares is considered optimal for that group.

**Mathematical Formulation:**
For group $`k`$ (models with $`k`$ predictors):
```math
\text{Best model in group } k = \arg\min_{\substack{S \subseteq \{1,\ldots,p\} \\ |S| = k}} \text{RSS}(S)
```

**3. Evaluating Model Scores:**
Next, evaluate the score (residual sum of squares plus the penalty) of these $`p`$ models and select the one with the lowest score:
```math
\text{Optimal model} = \arg\min_{k \in \{1,\ldots,p\}} \text{Score}(\text{Best model in group } k)
```

#### Computational Considerations:

The computational demands at step 2 can be immense, especially when $`p`$ is large. Typically, this algorithm may not be advisable when $`p > 40`$.

**Complexity Analysis:**
- **Time Complexity**: $`O(2^p)`$ in worst case
- **Space Complexity**: $`O(2^p)`$ for storing all models
- **Practical Limit**: $`p \leq 40`$ for reasonable computation time

### Greedy Algorithms: Efficient Alternatives

For significantly large $`p`$ values, employing greedy algorithms is beneficial. These algorithms search for the optimal model following a specific path, sacrificing global optimality for computational efficiency.

#### 1. Forward Selection

**Algorithm:**
1. Start with the null model (only intercept)
2. At each step, add the predictor that most improves the model score
3. Continue until no further improvement is possible

**Mathematical Formulation:**
```math
\begin{aligned}
S_0 &= \emptyset \\
S_{t+1} &= S_t \cup \{j^*\} \\
\text{where } j^* &= \arg\min_{j \notin S_t} \text{Score}(S_t \cup \{j\})
\end{aligned}
```

**Advantages:**
- Computationally efficient: $`O(p^2)`$ complexity
- Works well when true model is sparse
- Can handle $`p > n`$ scenarios

**Disadvantages:**
- Cannot remove variables once added
- May get stuck in local optima
- Sensitive to the order of variable addition

#### 2. Backward Elimination

**Algorithm:**
1. Start with the full model (all predictors)
2. At each step, remove the predictor whose removal most improves the model score
3. Continue until no further improvement is possible

**Mathematical Formulation:**
```math
\begin{aligned}
S_0 &= \{1, 2, \ldots, p\} \\
S_{t+1} &= S_t \setminus \{j^*\} \\
\text{where } j^* &= \arg\min_{j \in S_t} \text{Score}(S_t \setminus \{j\})
\end{aligned}
```

**Advantages:**
- Computationally efficient: $`O(p^2)`$ complexity
- Works well when most variables are relevant
- Can handle multicollinearity better than forward selection

**Disadvantages:**
- Cannot add variables once removed
- May get stuck in local optima
- Requires $`p < n`$ to start

#### 3. Stepwise Algorithm (Forward-Backward)

**Algorithm:**
This is a blend of backward and forward methods:
1. Start with the full model and move backward
2. At each stage, in addition to removing predictors, consider reintroducing ones previously removed
3. The process halts when adding or removing predictors no longer improves the score

**Mathematical Formulation:**
```math
\begin{aligned}
\text{At step } t: \\
\text{Backward step: } S_{t+1} &= S_t \setminus \{j^*\} \text{ if } \text{Score}(S_t \setminus \{j^*\}) < \text{Score}(S_t) \\
\text{Forward step: } S_{t+1} &= S_t \cup \{j^*\} \text{ if } \text{Score}(S_t \cup \{j^*\}) < \text{Score}(S_t)
\end{aligned}
```

**Advantages:**
- More flexible than pure forward or backward
- Can escape local optima
- Often finds better solutions than pure greedy methods

**Disadvantages:**
- More computationally intensive
- Still not guaranteed to find global optimum
- May oscillate between similar models

### Local vs Global Optimality

The nature of greedy algorithms, given their specific path of search, means they may stop at a locally optimal solution rather than a globally optimal one. However, they're faster and often yield solutions that are practically sufficient.

**Mathematical Illustration:**
Consider a landscape of model scores:
```math
\text{Global optimum} = \min_{S \subseteq \{1,\ldots,p\}} \text{Score}(S)
```

Greedy algorithms find:
```math
\text{Local optimum} = \text{Score}(S_{\text{greedy}}) \geq \text{Score}(S_{\text{global}})
```

### Comprehensive Python Example: Search Algorithms

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from itertools import combinations
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
n = 100
p = 8
X = np.random.randn(n, p)

# True model: only first 4 variables matter
beta_true = np.zeros(p)
beta_true[:4] = [1.5, -0.8, 0.6, -0.4]
f_true = X @ beta_true
y = f_true + np.random.normal(0, 0.5, n)

print("=== TRUE MODEL ===")
print(f"True coefficients: {beta_true}")
print(f"True model variables: {list(range(4))}")

# Function to calculate model score (AIC)
def calculate_score(X_subset, y):
    """Calculate AIC for a given model"""
    if X_subset.shape[1] == 0:
        return np.inf
    
    model = LinearRegression()
    model.fit(X_subset, y)
    y_pred = model.predict(X_subset)
    
    rss = np.sum((y - y_pred)**2)
    p = X_subset.shape[1]
    aic = n * np.log(rss/n) + 2*p
    
    return aic

# 1. Exhaustive Search (Best Subset)
print("\n=== EXHAUSTIVE SEARCH ===")
start_time = time.time()

best_score_exhaustive = np.inf
best_model_exhaustive = None
all_scores = []

for p_subset in range(1, p + 1):
    for subset in combinations(range(p), p_subset):
        X_subset = X[:, subset]
        score = calculate_score(X_subset, y)
        all_scores.append((score, subset))
        
        if score < best_score_exhaustive:
            best_score_exhaustive = score
            best_model_exhaustive = subset

exhaustive_time = time.time() - start_time
print(f"Best model: {best_model_exhaustive}")
print(f"Best score: {best_score_exhaustive:.2f}")
print(f"Computation time: {exhaustive_time:.3f} seconds")

# 2. Forward Selection
print("\n=== FORWARD SELECTION ===")
start_time = time.time()

current_vars = set()
best_score_forward = np.inf
forward_history = []

for step in range(p):
    best_score_step = np.inf
    best_var_step = None
    
    for var in range(p):
        if var not in current_vars:
            test_vars = list(current_vars) + [var]
            X_test = X[:, test_vars]
            score = calculate_score(X_test, y)
            
            if score < best_score_step:
                best_score_step = score
                best_var_step = var
    
    if best_score_step < best_score_forward:
        current_vars.add(best_var_step)
        best_score_forward = best_score_step
        forward_history.append((best_score_step, list(current_vars)))
    else:
        break

forward_time = time.time() - start_time
print(f"Best model: {list(current_vars)}")
print(f"Best score: {best_score_forward:.2f}")
print(f"Computation time: {forward_time:.3f} seconds")

# 3. Backward Elimination
print("\n=== BACKWARD ELIMINATION ===")
start_time = time.time()

current_vars = set(range(p))
best_score_backward = calculate_score(X, y)
backward_history = [(best_score_backward, list(current_vars))]

for step in range(p - 1):
    best_score_step = np.inf
    best_var_step = None
    
    for var in current_vars:
        test_vars = list(current_vars - {var})
        X_test = X[:, test_vars]
        score = calculate_score(X_test, y)
        
        if score < best_score_step:
            best_score_step = score
            best_var_step = var
    
    if best_score_step < best_score_backward:
        current_vars.remove(best_var_step)
        best_score_backward = best_score_step
        backward_history.append((best_score_step, list(current_vars)))
    else:
        break

backward_time = time.time() - start_time
print(f"Best model: {list(current_vars)}")
print(f"Best score: {best_score_backward:.2f}")
print(f"Computation time: {backward_time:.3f} seconds")

# 4. Stepwise Selection
print("\n=== STEPWISE SELECTION ===")
start_time = time.time()

current_vars = set(range(p))
best_score_stepwise = calculate_score(X, y)
stepwise_history = [(best_score_stepwise, list(current_vars))]
improved = True

while improved:
    improved = False
    
    # Backward step
    for var in list(current_vars):
        test_vars = list(current_vars - {var})
        X_test = X[:, test_vars]
        score = calculate_score(X_test, y)
        
        if score < best_score_stepwise:
            current_vars.remove(var)
            best_score_stepwise = score
            stepwise_history.append((best_score_stepwise, list(current_vars)))
            improved = True
            break
    
    # Forward step
    if not improved:
        for var in range(p):
            if var not in current_vars:
                test_vars = list(current_vars) + [var]
                X_test = X[:, test_vars]
                score = calculate_score(X_test, y)
                
                if score < best_score_stepwise:
                    current_vars.add(var)
                    best_score_stepwise = score
                    stepwise_history.append((best_score_stepwise, list(current_vars)))
                    improved = True
                    break

stepwise_time = time.time() - start_time
print(f"Best model: {list(current_vars)}")
print(f"Best score: {best_score_stepwise:.2f}")
print(f"Computation time: {stepwise_time:.3f} seconds")

# Comparison
print("\n=== ALGORITHM COMPARISON ===")
comparison_df = pd.DataFrame({
    'Algorithm': ['Exhaustive', 'Forward', 'Backward', 'Stepwise'],
    'Best Score': [best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise],
    'Best Model': [str(best_model_exhaustive), str(list(current_vars)), str(list(current_vars)), str(list(current_vars))],
    'Time (s)': [exhaustive_time, forward_time, backward_time, stepwise_time],
    'Optimal': [best_score_exhaustive == min(best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise)] * 4
})

print(comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Score progression for each algorithm
steps_forward = range(1, len(forward_history) + 1)
scores_forward = [h[0] for h in forward_history]

steps_backward = range(1, len(backward_history) + 1)
scores_backward = [h[0] for h in backward_history]

steps_stepwise = range(1, len(stepwise_history) + 1)
scores_stepwise = [h[0] for h in stepwise_history]

axes[0, 0].plot(steps_forward, scores_forward, 'bo-', label='Forward', linewidth=2)
axes[0, 0].plot(steps_backward, scores_backward, 'ro-', label='Backward', linewidth=2)
axes[0, 0].plot(steps_stepwise, scores_stepwise, 'go-', label='Stepwise', linewidth=2)
axes[0, 0].axhline(y=best_score_exhaustive, color='black', linestyle='--', alpha=0.7, label='Exhaustive')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('AIC Score')
axes[0, 0].set_title('Score Progression')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Model size progression
model_sizes_forward = [len(h[1]) for h in forward_history]
model_sizes_backward = [len(h[1]) for h in backward_history]
model_sizes_stepwise = [len(h[1]) for h in stepwise_history]

axes[0, 1].plot(steps_forward, model_sizes_forward, 'bo-', label='Forward', linewidth=2)
axes[0, 1].plot(steps_backward, model_sizes_backward, 'ro-', label='Backward', linewidth=2)
axes[0, 1].plot(steps_stepwise, model_sizes_stepwise, 'go-', label='Stepwise', linewidth=2)
axes[0, 1].axhline(y=len(best_model_exhaustive), color='black', linestyle='--', alpha=0.7, label='Exhaustive')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Model Size')
axes[0, 1].set_title('Model Size Progression')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Computation time comparison
algorithms = ['Exhaustive', 'Forward', 'Backward', 'Stepwise']
times = [exhaustive_time, forward_time, backward_time, stepwise_time]

axes[1, 0].bar(algorithms, times, color=['red', 'blue', 'green', 'orange'])
axes[1, 0].set_ylabel('Time (seconds)')
axes[1, 0].set_title('Computation Time Comparison')
axes[1, 0].grid(True, alpha=0.3)

# Score comparison
scores = [best_score_exhaustive, best_score_forward, best_score_backward, best_score_stepwise]
colors = ['red' if s == min(scores) else 'gray' for s in scores]

axes[1, 1].bar(algorithms, scores, color=colors)
axes[1, 1].set_ylabel('AIC Score')
axes[1, 1].set_title('Final Score Comparison')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights
print("\n=== KEY INSIGHTS ===")
print("1. Exhaustive search finds the global optimum but is computationally expensive")
print("2. Greedy algorithms are much faster but may find local optima")
print("3. Stepwise selection often provides a good compromise")
print("4. All algorithms found similar model sizes in this example")
```

## 3.1.5. R/Python Code for Subset Selection

- Rcode: [R_W3_VarSel_SubsetSelection](./R_W3_VarSel_SubsetSelection.R)
- Python: [Python_W3_VarSel_SubsetSelection](./Python_W3_VarSel_SubsetSelection.py)

## 3.1.6. Variable Screening: Handling High-Dimensional Data

Among the three model selection procedures — complete, forward, and backward — stepwise is the most computationally intensive. However, compared to forward and backward methods, stepwise is less prone to prematurely settling on a local optimum. The forward method doesn't allow for the removal of variables once they've been added, even if they become less relevant as other predictors are included. Conversely, the backward method can't reintroduce a predictor that might seem unimportant in the presence of other variables but could be beneficial if certain variables are removed.

### Algorithm Comparison Summary

**Computational Complexity:**
```math
\begin{aligned}
\text{Exhaustive Search} &: O(2^p) \\
\text{Forward Selection} &: O(p^2) \\
\text{Backward Elimination} &: O(p^2) \\
\text{Stepwise Selection} &: O(p^2) \text{ (but more iterations)}
\end{aligned}
```

**Recommendation**: If computational resources allow, we recommend the stepwise approach, beginning with the full model, as it provides the best balance between finding good solutions and avoiding local optima.

### The High-Dimensional Challenge: When $`p > n`$

However, what if $`p`$ (number of predictors) exceeds $`n`$ (sample size)? This scenario, known as the **high-dimensional problem**, presents unique challenges.

#### Mathematical Foundation of the Problem

**Perfect Fit Issue:**
For any linear regression model with sample size $`n`$, adding more than $`n-1`$ non-intercept predictors will result in a residual sum of squares of zero. This occurs because:

```math
\text{rank}(\mathbf{X}) \leq \min(n, p)
```

When $`p > n`$, the design matrix $`\mathbf{X}`$ has more columns than rows, leading to:
```math
\text{RSS} = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2 = 0
```

**Criterion Breakdown:**
Consequently, both AIC and BIC metrics become undefined, as the first term of AIC and BIC — being the logarithm of the residual sum of squares — equals negative infinity:

```math
\begin{aligned}
\text{AIC} &= n\log(0/n) + 2p = -\infty + 2p \\
\text{BIC} &= n\log(0/n) + \log(n)p = -\infty + \log(n)p
\end{aligned}
```

#### Solutions for High-Dimensional Data

**1. Model Size Capping:**
Despite the criterion breakdown, search algorithms can still be utilized by setting a cap on the model size. Under the stepwise procedure, for example, when your model reaches this threshold, you can only remove predictors, not add them.

**Mathematical Formulation:**
```math
\text{Maximum model size} = \min(p, n-1)
```

**2. Variable Screening:**
When $`p > n`$, directly using the full model as a starting point isn't feasible. It's recommended to use screening procedures to identify a starting model for the stepwise process.

### Screening Methods

#### 1. Correlation-Based Screening

**Algorithm:**
A simple screening approach is to rank predictors based on their correlation magnitude with the outcome variable $`Y`$ and retain only the top $`K`$ predictors (e.g., $`K = n/3`$).

**Mathematical Formulation:**
```math
\text{Correlation with } Y: \rho_j = \frac{\text{Cov}(X_j, Y)}{\sqrt{\text{Var}(X_j)\text{Var}(Y)}}
```

**Screening Rule:**
```math
S_{\text{screen}} = \{j : |\rho_j| \geq \text{threshold}\}
```

where the threshold is chosen to select approximately $`K`$ variables.

#### 2. Univariate Regression Screening

**Algorithm:**
This method mirrors the process of executing individual simple linear regressions for $`Y`$ against each predictor and ranking them based on p-values.

**Mathematical Formulation:**
For each predictor $`X_j`$:
```math
Y = \beta_0 + \beta_j X_j + \epsilon
```

**Screening Rule:**
```math
S_{\text{screen}} = \{j : p\text{-value}_j \leq \alpha_{\text{screen}}\}
```

where $`\alpha_{\text{screen}}`$ is chosen to control the number of selected variables.

#### 3. Mutual Information Screening

**Algorithm:**
For non-linear relationships, mutual information can be used as a screening criterion.

**Mathematical Formulation:**
```math
I(X_j; Y) = \int\int p(x_j, y) \log\left(\frac{p(x_j, y)}{p(x_j)p(y)}\right) dx_j dy
```

**Screening Rule:**
```math
S_{\text{screen}} = \{j : I(X_j; Y) \geq \text{threshold}\}
```

### Comprehensive Python Example: Variable Screening

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mutual_info_regression
from scipy import stats
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate high-dimensional data (p > n)
n = 50
p = 200

# Generate predictors with some correlation structure
X = np.random.randn(n, p)

# True model: only first 5 variables matter
beta_true = np.zeros(p)
beta_true[:5] = [1.5, -0.8, 0.6, -0.4, 0.3]
f_true = X @ beta_true
y = f_true + np.random.normal(0, 0.5, n)

print("=== HIGH-DIMENSIONAL DATA ===")
print(f"Sample size: {n}")
print(f"Number of predictors: {p}")
print(f"True model variables: {list(range(5))}")
print(f"True model size: 5")

# 1. Correlation-based screening
print("\n=== CORRELATION-BASED SCREENING ===")

# Calculate correlations
correlations = np.corrcoef(X.T, y)[:-1, -1]
corr_abs = np.abs(correlations)

# Select top K variables
K = n // 3  # n/3 variables
top_k_indices = np.argsort(corr_abs)[-K:]

print(f"Selected {K} variables based on correlation")
print(f"Top 10 correlations: {corr_abs[top_k_indices[-10:]]}")
print(f"True variables in top {K}: {sum(i < 5 for i in top_k_indices)}/5")

# 2. Univariate regression screening
print("\n=== UNIVARIATE REGRESSION SCREENING ===")

p_values = []
for j in range(p):
    # Simple linear regression
    model = LinearRegression()
    model.fit(X[:, j:j+1], y)
    y_pred = model.predict(X[:, j:j+1])
    
    # Calculate p-value
    rss = np.sum((y - y_pred)**2)
    tss = np.sum((y - np.mean(y))**2)
    r_squared = 1 - rss/tss
    
    if r_squared < 1:
        f_stat = (r_squared / 1) / ((1 - r_squared) / (n - 2))
        p_val = 1 - stats.f.cdf(f_stat, 1, n - 2)
    else:
        p_val = 0
    
    p_values.append(p_val)

p_values = np.array(p_values)
alpha_screen = 0.05
significant_vars = np.where(p_values < alpha_screen)[0]

print(f"Variables significant at α = {alpha_screen}: {len(significant_vars)}")
print(f"True variables significant: {sum(i < 5 for i in significant_vars)}/5")

# 3. Mutual information screening
print("\n=== MUTUAL INFORMATION SCREENING ===")

# Calculate mutual information
mi_scores = mutual_info_regression(X, y, random_state=42)

# Select top K variables
top_k_mi_indices = np.argsort(mi_scores)[-K:]

print(f"Selected {K} variables based on mutual information")
print(f"Top 10 MI scores: {mi_scores[top_k_mi_indices[-10:]]}")
print(f"True variables in top {K}: {sum(i < 5 for i in top_k_mi_indices)}/5")

# 4. Combined screening approach
print("\n=== COMBINED SCREENING APPROACH ===")

# Combine all screening methods
all_screened = set(top_k_indices) | set(significant_vars) | set(top_k_mi_indices)
screened_vars = list(all_screened)

print(f"Combined screening selected {len(screened_vars)} variables")
print(f"True variables in combined set: {sum(i < 5 for i in screened_vars)}/5")

# 5. Stepwise selection on screened variables
print("\n=== STEPWISE SELECTION ON SCREENED VARIABLES ===")

if len(screened_vars) > 0:
    X_screened = X[:, screened_vars]
    
    # Function to calculate AIC
    def calculate_aic(X_subset, y):
        if X_subset.shape[1] == 0:
            return np.inf
        
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        
        rss = np.sum((y - y_pred)**2)
        p = X_subset.shape[1]
        aic = n * np.log(rss/n) + 2*p
        
        return aic
    
    # Stepwise selection
    current_vars = set(range(X_screened.shape[1]))
    best_score = calculate_aic(X_screened, y)
    improved = True
    
    while improved and len(current_vars) > 1:
        improved = False
        
        # Try removing each variable
        for var in list(current_vars):
            test_vars = list(current_vars - {var})
            X_test = X_screened[:, test_vars]
            score = calculate_aic(X_test, y)
            
            if score < best_score:
                current_vars.remove(var)
                best_score = score
                improved = True
                break
    
    final_vars = [screened_vars[i] for i in current_vars]
    print(f"Final model size: {len(final_vars)}")
    print(f"Final variables: {final_vars}")
    print(f"True variables in final model: {sum(i < 5 for i in final_vars)}/5")
    print(f"Final AIC: {best_score:.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Correlation distribution
axes[0, 0].hist(corr_abs, bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(x=corr_abs[top_k_indices[-1]], color='red', linestyle='--', 
                   label=f'Threshold ({K} variables)')
axes[0, 0].set_xlabel('Absolute Correlation')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Correlation Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# P-value distribution
axes[0, 1].hist(p_values, bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=alpha_screen, color='red', linestyle='--', 
                   label=f'α = {alpha_screen}')
axes[0, 1].set_xlabel('P-value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('P-value Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Mutual information distribution
axes[0, 2].hist(mi_scores, bins=30, alpha=0.7, edgecolor='black')
axes[0, 2].axvline(x=mi_scores[top_k_mi_indices[-1]], color='red', linestyle='--', 
                   label=f'Threshold ({K} variables)')
axes[0, 2].set_xlabel('Mutual Information')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Mutual Information Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Screening comparison
screening_methods = ['Correlation', 'P-value', 'MI', 'Combined']
true_vars_found = [
    sum(i < 5 for i in top_k_indices),
    sum(i < 5 for i in significant_vars),
    sum(i < 5 for i in top_k_mi_indices),
    sum(i < 5 for i in screened_vars)
]

axes[1, 0].bar(screening_methods, true_vars_found, color=['blue', 'green', 'red', 'purple'])
axes[1, 0].set_ylabel('True Variables Found')
axes[1, 0].set_title('Screening Performance')
axes[1, 0].set_ylim(0, 5)
axes[1, 0].grid(True, alpha=0.3)

# Variable importance comparison
true_correlations = corr_abs[:5]
true_p_values = p_values[:5]
true_mi_scores = mi_scores[:5]

x_pos = np.arange(5)
width = 0.25

axes[1, 1].bar(x_pos - width, true_correlations, width, label='Correlation', alpha=0.7)
axes[1, 1].bar(x_pos, -np.log10(true_p_values), width, label='-log10(p-value)', alpha=0.7)
axes[1, 1].bar(x_pos + width, true_mi_scores, width, label='MI Score', alpha=0.7)
axes[1, 1].set_xlabel('True Variable Index')
axes[1, 1].set_ylabel('Importance Score')
axes[1, 1].set_title('True Variable Importance')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Final model comparison
if len(screened_vars) > 0:
    final_model_vars = final_vars if 'final_vars' in locals() else []
    final_true_vars = sum(i < 5 for i in final_model_vars)
    
    axes[1, 2].pie([final_true_vars, 5 - final_true_vars, len(final_model_vars) - final_true_vars], 
                   labels=['True Variables', 'Missed True', 'False Positives'],
                   colors=['green', 'red', 'orange'], autopct='%1.1f%%')
    axes[1, 2].set_title('Final Model Composition')

plt.tight_layout()
plt.show()

# Summary
print("\n=== SCREENING SUMMARY ===")
print("Screening Method Performance:")
print(f"Correlation screening: {true_vars_found[0]}/5 true variables")
print(f"P-value screening: {true_vars_found[1]}/5 true variables")
print(f"MI screening: {true_vars_found[2]}/5 true variables")
print(f"Combined screening: {true_vars_found[3]}/5 true variables")

if len(screened_vars) > 0:
    print(f"\nFinal model after stepwise: {len(final_vars)} variables")
    print(f"True variables in final model: {sum(i < 5 for i in final_vars)}/5")

print(f"\nKey Insights:")
print("1. Screening reduces computational complexity from O(2^p) to O(2^K)")
print("2. Different screening methods may select different variables")
print("3. Combined screening often captures more true variables")
print("4. Stepwise selection on screened variables provides final model")
```

### Key Insights and Recommendations

**1. Screening Benefits:**
- Reduces computational complexity from $`O(2^p)`$ to $`O(2^K)`$
- Makes high-dimensional variable selection feasible
- Provides starting point for more sophisticated methods

**2. Screening Limitations:**
- May miss important variables with weak marginal effects
- Sensitive to correlation structure among predictors
- Different screening methods may select different variables

**3. Best Practices:**
- Use multiple screening methods and combine results
- Apply stepwise selection on screened variables
- Validate final model with cross-validation
- Consider domain knowledge in screening decisions

**4. When to Use Screening:**
- $`p > n`$ scenarios
- Computational constraints
- Initial variable selection in large datasets
- Preprocessing step for more sophisticated methods

Although this elementary procedure might overlook crucial variables, the subsequent stepwise process can potentially reincorporate them into the model, making screening a valuable tool for high-dimensional variable selection.
