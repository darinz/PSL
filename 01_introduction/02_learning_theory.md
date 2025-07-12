# 1.1.4. A Glimpse of Learning Theory

Learning theory provides the mathematical foundation for understanding why and how machine learning algorithms work. It helps us answer fundamental questions about generalization, model selection, and the trade-offs between model complexity and performance.

## The Supervised Learning Framework

### Basic Setup

When we described "how does supervised learning work," we established the following components:

- **Training Data**: $`\{ \mathbf{x}_i, y_i \}_{i=1}^n`$ - a collection of $`n`$ labeled examples
- **Model Function**: $`f: \mathcal{X} \rightarrow \mathcal{Y}`$ - a mapping from input space to output space
- **Loss Function**: $`L(y_i, f(\mathbf{x}_i))`$ - measures the cost of prediction error
- **Training Error**: averaged loss over the training samples

```math
\text{TrainErr}[f] = \frac{1}{n} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i))
```

- **Test Error**: averaged loss over future test samples

```math
\text{TestErr}[f] = \frac{1}{N} \sum_{j=1}^N L(y_j^*, f(\mathbf{x}_j^*))
```

### The Population Perspective

Suppose $`\{x_j^*, y_j^*\}_{j=1}^N`$ is a set of test data with test error given by:

```math
\text{TestErr}[f] = \frac{1}{N} \sum_{j=1}^N \left[ y_j^* - f(x_j^*) \right]^2
```

**Key Insight**: Naturally, we would like to have a very large test set. If $`N \to \infty`$, the average above converges to the population expectation:

```math
\lim_{N \to \infty} \text{TestErr}[f] = \mathbb{E}_{(X^*, Y^*)}[(Y^* - f(X^*))^2]
```

where $`(X^*, Y^*) \sim P(x, y)`$ follows some underlying data distribution.

### The Fundamental Assumption: IID Data

**Critical Assumption**: We assume the training data consists of independent and identically distributed (i.i.d.) samples from the **same unknown distribution** $`P(x, y)`$.

**Why This Matters**: If the training and test data are governed by completely different random processes, then learning becomes impossible. The model learned from one distribution cannot generalize to a fundamentally different one.

**Domain Adaptation**: While there are learning algorithms that try to extract knowledge from one domain and adapt it to others, even these algorithms assume that something meaningful is shared across different domains.

## Statistical Decision Theory

Statistical decision theory provides the theoretical foundation for optimal prediction under uncertainty. It tells us what the best possible predictor looks like when we know the true data-generating process.

### The Risk Function

Assume $`(X, Y) \sim P(x, y)`$ follows some joint distribution. We define a **loss function** to evaluate the prediction accuracy of $`f`$:

- **For Regression**: $`L(y, f(x)) = (y - f(x))^2`$ (squared error)
- **For Classification**: $`L(y, f(x)) = \mathbb{I}[y \neq f(x)]`$ (0-1 loss)

The **risk** (or expected loss) is defined as:

```math
R[f] = \mathbb{E}_{X,Y} L(Y, f(X))
```

**Interpretation**: The risk measures the average prediction error we expect when using function $`f`$ on new data drawn from the true distribution.

### The Optimal Predictor

The optimal function $`f^*`$ minimizes the risk:

```math
f^* = \arg\min_f R[f]
```

The corresponding optimal risk is denoted by $`R^* = R[f^*] = \min_f R[f]`$, often called the **Bayes risk**.

### Deriving the Optimal Predictor

Assume the joint distribution $`P`$ is known. What's the optimal $`f^*`$?

Using the law of iterated expectations, we can rewrite the risk:

```math
R[f] = \mathbb{E}_{X,Y} L(Y, f(X)) = \mathbb{E}_X \left[ \mathbb{E}_{Y|X} L(Y, f(X)) \right]
```

**Key Insight**: Given $`X = x`$, the inner expectation $`\mathbb{E}_{Y|X=x}`$ is over $`Y`$ only. This can be written as:

```math
\mathbb{E}_X \left[ \mathbb{E}_{Y|X} L(Y, f(X)) \right] = \int_x \left[ \int_y L(y, f(x)) p(y|x) dy \right] p(x) dx
```

where $`p(x)`$ is the marginal distribution of $`X`$ and $`p(y|x)`$ is the conditional distribution of $`Y`$ given $`X = x`$.

**The Optimization Problem**: Finding the optimal function $`f`$ that minimizes $`R[f]`$ reduces to solving a series of pointwise optimization problems:

```math
f^*(x) = \arg\min_a \mathbb{E}_{Y|X=x} L(Y, a)
```

We solve this for every $`x`$, and the resulting $`f^*`$ minimizes the overall risk.

### Optimal Predictors for Different Loss Functions

#### Regression with Squared Loss

For regression with squared loss $`L(y, f(x)) = (y - f(x))^2`$:

```math
f^*(x) = \arg\min_a \mathbb{E}_{Y|X=x}(Y - a)^2 = \mathbb{E}[Y \mid X = x]
```

**Interpretation**: The optimal predictor is the conditional expectation of $`Y`$ given $`X = x`$.

**Alternative Loss**: What if we use absolute loss $`L(y, f(x)) = |y - f(x)|`$?

The optimal predictor becomes the conditional median: $`f^*(x) = \text{median}(Y \mid X = x)`$.

#### Classification with 0-1 Loss

For classification with 0-1 loss $`L(y, f(x)) = \mathbb{I}[y \neq f(x)]`$:

```math
f^*(x) = \arg\max_k P(Y = k \mid X = x)
```

For binary classification, this becomes:

```math
f^*(x) = \begin{cases} 
1 & \text{if } P(Y = 1 \mid X = x) > 0.5 \\
0 & \text{otherwise}
\end{cases}
```

This is known as the **Bayes classifier** due to its connection with Bayes' theorem.

## The Reality Gap: Unknown Distribution

In practice, we don't know the true distribution $`P(x, y)`$, so we cannot compute $`f^*`$ directly. Instead, we have:

1. **Training Data**: A set of random samples $`(x_i, y_i)_{i=1}^n`$ from $`P`$
2. **Function Class**: We restrict our search to some family of functions $`\mathcal{F}`$

### The Ideal vs. Reality

**If we knew the true distribution $`P`$**, we would have:

- **Risk of any function**: $`R[f] = \mathbb{E}_{X,Y} L(Y, f(X))`$
- **Optimal function**: $`f^* = \arg\min_f R[f]`$ with optimal risk $`R^* = R[f^*]`$
- **Best function in class**: $`f^*_{\mathcal{F}} = \arg\min_{f \in \mathcal{F}} R[f]`$ with risk $`R^*_{\mathcal{F}} = R[f^*_{\mathcal{F}}]`$

**Given only training data**, we have:

- **Empirical risk**: $`\hat{R}_n[f] = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))`$
- **Empirical optimal function**: $`\hat{f}_{n,\mathcal{F}} = \arg\min_{f \in \mathcal{F}} \hat{R}_n[f]`$

### The Fundamental Question

**Key Question**: How far is $`R[\hat{f}_{n,\mathcal{F}}]`$ from the ideal performance $`R^*`$?

This gap can be decomposed into two components:

```math
R[\hat{f}_{n,\mathcal{F}}] - R^* = \underbrace{R[\hat{f}_{n,\mathcal{F}}] - R[f^*_{\mathcal{F}}]}_{\text{Variance}} + \underbrace{R[f^*_{\mathcal{F}}] - R^*}_{\text{Bias}}
```

**Interpretation**:
- **Bias**: How well the best function in our class $`\mathcal{F}`$ can approximate the true optimal function
- **Variance**: How much our estimated function deviates from the best function in our class due to finite sample size

### Bounding the Variance Term

The variance term can be further decomposed:

```math
R[\hat{f}_{n,\mathcal{F}}] - R[f^*_{\mathcal{F}}] = \underbrace{R[\hat{f}_{n,\mathcal{F}}] - \hat{R}_n[\hat{f}_{n,\mathcal{F}}]}_{\text{Optimization Error}} + \underbrace{\hat{R}_n[\hat{f}_{n,\mathcal{F}}] - \hat{R}_n[f^*_{\mathcal{F}}]}_{\text{Selection Error}} + \underbrace{\hat{R}_n[f^*_{\mathcal{F}}] - R[f^*_{\mathcal{F}}]}_{\text{Estimation Error}}
```

Since $`\hat{f}_{n,\mathcal{F}}`$ minimizes empirical risk, the selection error is non-positive:

```math
\hat{R}_n[\hat{f}_{n,\mathcal{F}}] - \hat{R}_n[f^*_{\mathcal{F}}] \leq 0
```

Therefore:

```math
R[\hat{f}_{n,\mathcal{F}}] - R[f^*_{\mathcal{F}}] \leq R[\hat{f}_{n,\mathcal{F}}] - \hat{R}_n[\hat{f}_{n,\mathcal{F}}] + \hat{R}_n[f^*_{\mathcal{F}}] - R[f^*_{\mathcal{F}}]
```

```math
\leq 2 \max_{f \in \mathcal{F}} |R[f] - \hat{R}_n[f]|
```

**Interpretation**: The variance is controlled by how well the empirical risk approximates the true risk uniformly across the function class $`\mathcal{F}`$.

## Practical Implications

### Model Complexity vs. Sample Size

The bias-variance decomposition reveals fundamental trade-offs:

1. **Complex Models** (large $`\mathcal{F}`$):
   - Low bias (can approximate complex functions)
   - High variance (more parameters to estimate)
   - Require more data to control variance

2. **Simple Models** (small $`\mathcal{F}`$):
   - High bias (limited approximation power)
   - Low variance (fewer parameters)
   - Work well with limited data

### Regularization

Regularization techniques (Ridge, Lasso, etc.) effectively reduce the size of the function class $`\mathcal{F}`$, trading bias for variance:

```math
\hat{f}_{n,\mathcal{F}} = \arg\min_{f \in \mathcal{F}} \left\{ \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \Omega(f) \right\}
```

where $`\Omega(f)`$ is a complexity penalty.

### Cross-Validation

Cross-validation provides an estimate of the generalization error without requiring knowledge of the true distribution:

```math
\text{CV}[f] = \frac{1}{K} \sum_{k=1}^K \frac{1}{|V_k|} \sum_{i \in V_k} L(y_i, f^{-k}(x_i))
```

where $`f^{-k}`$ is trained on data excluding fold $`k`$.

## Summary

Learning theory provides the mathematical framework for understanding:

1. **What is the best possible predictor?** (Bayes classifier/regressor)
2. **How do we approximate it from data?** (Empirical risk minimization)
3. **What are the fundamental limitations?** (Bias-variance trade-off)
4. **How do we control generalization error?** (Regularization, model selection)

This theoretical foundation guides practical decisions in model selection, hyperparameter tuning, and algorithm design.