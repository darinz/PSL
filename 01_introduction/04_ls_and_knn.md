# 1.2.1. Introduction to LS and kNN

In this section, we examine two fundamental supervised learning approaches: k-Nearest Neighbor (kNN) and linear regression. These algorithms represent different paradigms in machine learning - kNN is a non-parametric, instance-based method, while linear regression is a parametric, model-based approach. Understanding their strengths, weaknesses, and mathematical foundations is crucial for building intuition about the bias-variance tradeoff.

## k-Nearest Neighbor (kNN)

The k-Nearest Neighbor algorithm is one of the simplest yet most powerful non-parametric learning methods. It operates on a fundamental principle: similar inputs should have similar outputs. This "local averaging" approach makes no assumptions about the underlying data distribution.

### Mathematical Formulation

For a test point $`x`$, kNN identifies the $`k`$ training samples closest to $`x`$ and uses their target values to make a prediction.

**Distance Metric**: Typically uses Euclidean distance in $`\mathbb{R}^p`$:
```math
d(x_i, x_j) = \sqrt{\sum_{l=1}^p (x_{il} - x_{jl})^2}
```

**Regression**: Output the average of the $`Y`$ values of the $`k`$ nearest neighbors:
```math
\hat{f}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
```
where $`\mathcal{N}_k(x)`$ is the set of indices of the $`k`$ nearest neighbors of $`x`$.

**Classification**: Return the majority vote or probability based on class frequency:
```math
\hat{f}(x) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}[y_i = c]
```

**Probability Estimation**:
```math
P(Y = c | X = x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}[y_i = c]
```

### Example Walkthrough

Consider $`k=5`$ in a binary classification problem. If among the five nearest neighbors:
- 3 have $`Y=1`$
- 2 have $`Y=0`$

Then:
- **Majority vote**: Predict $`Y=1`$
- **Probability estimates**: $`P(Y=1) = 3/5 = 0.6`$, $`P(Y=0) = 2/5 = 0.4`$

### Algorithm Properties

**No Training Phase**: kNN is a "lazy learner" - it simply stores the training data and performs computation only at prediction time.

**Local Approximation**: kNN approximates the true function $`f^*(x)`$ locally around each test point:
```math
\hat{f}(x) \approx \mathbb{E}[Y | X \in \mathcal{B}_k(x)]
```
where $`\mathcal{B}_k(x)`$ is the neighborhood defined by the $`k`$ nearest neighbors.

## Parameters and Tuning

### Neighborhood Size ($`k`$)

The choice of $`k`$ fundamentally affects the bias-variance tradeoff:

**$`k=1`$ (1NN)**:
- Uses only the nearest training sample
- Training error is zero (perfect interpolation)
- Highest variance, lowest bias
- Complexity approximately $`n`$ parameters

**$`k=n`$ (global average)**:
- Uses all training samples equally
- Prediction is constant for all $`x`$ (like fitting only an intercept)
- Lowest variance, highest bias
- Complexity approximately 1 parameter

**General Case**: Complexity is approximately $`n/k`$ parameters.

**Optimal $`k`$**: Typically chosen via cross-validation to balance bias and variance.

### Distance Metrics

**Euclidean Distance** (default for continuous features):
```math
d(x_i, x_j) = \sqrt{\sum_{l=1}^p (x_{il} - x_{jl})^2}
```

**Manhattan Distance** (L1 norm):
```math
d(x_i, x_j) = \sum_{l=1}^p |x_{il} - x_{jl}|
```

**Weighted Distances**:
```math
d(x_i, x_j) = \sqrt{\sum_{l=1}^p w_l (x_{il} - x_{jl})^2}
```

**Domain-Specific Metrics**: For images (pixel similarity), text (cosine similarity), or user preferences.

## Linear Regression

Linear regression is a parametric method that assumes a linear relationship between features and target. It's computationally efficient and provides interpretable results.

### Mathematical Formulation

**Model**: Assume a linear relationship between $`X`$ and $`Y`$:
```math
Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p + \epsilon = X^T \beta + \epsilon
```

where:
- $`\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T`$ is the parameter vector
- $`X = (1, X_1, \ldots, X_p)^T`$ includes the intercept term
- $`\epsilon \sim N(0, \sigma^2)`$ is the error term

**Estimation**: Minimize the sum of squared residuals:
```math
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - x_i^T \beta)^2
```

**Closed-Form Solution**: For $`p`$-dimensional $`X`$, we estimate $`p+1`$ parameters:
```math
\hat{\beta} = (X^T X)^{-1} X^T y
```

**Prediction**: $`\hat{y} = x^T \hat{\beta}`$

### Linear Regression for Classification

**Binary Classification**: Code $`Y`$ as 0 or 1 and use linear regression:
```math
\hat{P}(Y = 1 | X = x) = x^T \hat{\beta}
```

**Decision Rule**: Predict class 1 if $`\hat{P}(Y = 1 | X = x) > 0.5`$:
```math
\hat{f}(x) = \mathbb{I}[x^T \hat{\beta} > 0.5]
```

**Tunable Threshold**: The 0.5 threshold can be adjusted based on class imbalance or cost considerations.

## Pros and Cons Analysis

### Linear Regression Advantages

1. **Computational Efficiency**: $`O(np^2 + p^3)`$ for training, $`O(p)`$ for prediction
2. **Interpretability**: Coefficients have clear meaning
3. **Statistical Inference**: Confidence intervals, hypothesis tests available
4. **Scalability**: Works well with large datasets
5. **Theoretical Foundation**: Well-understood properties

### Linear Regression Drawbacks

1. **Linear Assumption**: May miss non-linear relationships
2. **Invalid Probabilities**: Predictions can be outside $`[0, 1]`$ interval
3. **Squared Loss**: Not optimal for classification performance
4. **Feature Interactions**: Cannot capture complex interactions without manual feature engineering

### kNN Advantages

1. **No Assumptions**: Works with any data distribution
2. **Non-linear**: Can capture complex decision boundaries
3. **Local Adaptation**: Automatically adapts to local structure
4. **Conceptual Simplicity**: Easy to understand and implement

### kNN Drawbacks

1. **Computational Cost**: $`O(n)`$ for each prediction
2. **Curse of Dimensionality**: Performance degrades in high dimensions
3. **No Interpretability**: Black-box predictions
4. **Sensitive to Irrelevant Features**: All features weighted equally

## Model Complexity Analysis

### Degrees of Freedom (DF)

**Definition**: DF measures the effective number of parameters or the flexibility of a model.

**Linear Regression**: 
- **Model DF**: $`p+1`$ (number of coefficients)
- **Residual DF**: $`n-(p+1)`$ (for statistical inference)

**Mathematical Interpretation**: The least squares prediction $`\hat{y}`$ lies in a $`(p+1)`$-dimensional subspace, while the residual vector $`(y-\hat{y})`$ lies in a $`(n-p-1)`$-dimensional subspace.

**kNN**:
- **Approximate DF**: $`n/k`$
- **$`k=1`$**: DF ≈ $`n`$ (highest complexity)
- **$`k=n`$**: DF ≈ $`1`$ (lowest complexity)

### Complexity Comparison

| Algorithm | Parameters | Training Time | Prediction Time | Flexibility |
|-----------|------------|---------------|-----------------|-------------|
| Linear Regression | $`p+1`$ | $`O(np^2 + p^3)`$ | $`O(p)`$ | Low |
| kNN | $`n/k`$ | $`O(1)`$ | $`O(n)`$ | High |

## Theoretical Foundations

### kNN Consistency

Under certain conditions, kNN is consistent (converges to the Bayes classifier):

**Theorem**: If $`k \rightarrow \infty`$ and $`k/n \rightarrow 0`$ as $`n \rightarrow \infty`$, then:
```math
\lim_{n \rightarrow \infty} \mathbb{E}[(\hat{f}_n(X) - f^*(X))^2] = 0
```

**Intuition**: As sample size grows, the neighborhood becomes smaller and more localized, providing better local approximation.

### Linear Regression Optimality

**Gauss-Markov Theorem**: Under the linear model assumptions, the least squares estimator is the Best Linear Unbiased Estimator (BLUE).

**Maximum Likelihood**: Under normality assumption, least squares is equivalent to maximum likelihood estimation.

# 1.2.2. Simulation Study

We now conduct a comprehensive simulation study to compare kNN and linear regression on two carefully designed examples. This study will illustrate the bias-variance tradeoff and help us understand when each method performs well.

## Data Generation Process

### Example 1: Simple Gaussian Classes

**Data Structure**: Binary classification with two classes (0 and 1) in two-dimensional feature space.

**Class 1 Data Generation**:
```math
X | Y = 1 \sim N(\mu_1, \sigma^2 I_2)
```
where $`\mu_1 = (1, 1)^T`$ and $`\sigma^2 = 1`$.

**Class 0 Data Generation**:
```math
X | Y = 0 \sim N(\mu_0, \sigma^2 I_2)
```
where $`\mu_0 = (-1, -1)^T`$ and $`\sigma^2 = 1`$.

**Class Prior**: $`P(Y = 1) = P(Y = 0) = 0.5`$

**Sample Sizes**: 
- Training: 200 samples (100 per class)
- Test: 10,000 samples (5,000 per class)

**Key Characteristics**:
- Linear decision boundary exists
- Equal class priors
- Homoscedastic Gaussian noise
- Well-separated class means

### Example 2: Complex Mixture Distribution

**Data Structure**: Binary classification with mixture distributions for each class.

**Class 1 Data Generation**:
```math
X | Y = 1 \sim \sum_{j=1}^{10} w_j N(\mu_{1j}, \sigma^2 I_2)
```
where $`w_j = 1/10`$ for all $`j`$ and $`\mu_{1j}`$ are 10 different centers.

**Class 0 Data Generation**:
```math
X | Y = 0 \sim \sum_{j=1}^{10} w_j N(\mu_{0j}, \sigma^2 I_2)
```
where $`w_j = 1/10`$ for all $`j`$ and $`\mu_{0j}`$ are 10 different centers.

**Key Characteristics**:
- Non-linear decision boundary
- Complex class-conditional distributions
- Multiple modes per class
- Challenging for linear methods

## Mixture Distribution Theory

### Mathematical Definition

A **mixture distribution** is a probabilistic model representing various subgroups within a larger population. The probability density function is:

```math
f(x) = \sum_{j=1}^k w_j f_j(x)
```

where:
- $`w_j`$ are mixing weights with $`\sum_{j=1}^k w_j = 1`$
- $`f_j(x)`$ are component densities (e.g., Gaussian PDFs)

### Sampling from Mixture Distributions

**Two-Step Process**:

1. **Component Selection**: Draw $`Z \sim \text{Categorical}(w_1, \ldots, w_k)`$
2. **Data Generation**: Draw $`X | Z = j \sim f_j(x)`$

**Mathematical Justification**: This treats the mixture as the marginal of a joint distribution:
```math
f(x, z) = f(z) f(x | z) = w_z f_z(x)
```

**Marginal Distribution**: $`f(x) = \sum_{z=1}^k f(x, z) = \sum_{j=1}^k w_j f_j(x)`$

## Implementation Strategy

### kNN Implementation

1. **Parameter Grid**: Define a set of $`k`$ values (e.g., $`k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201\}`$)

2. **Error Storage**: Initialize vectors to store training and test errors for each $`k`$

3. **Distance Computation**: For each test point, compute distances to all training points

4. **Prediction**: For each $`k`$:
   - Find $`k`$ nearest neighbors
   - Compute majority vote or average
   - Calculate classification error

### Linear Regression Implementation

1. **Data Preparation**: Convert categorical labels (0, 1) to numerical values

2. **Model Fitting**: Fit linear regression using least squares:
```math
\hat{\beta} = (X^T X)^{-1} X^T y
```

3. **Prediction**: Compute $`\hat{P}(Y = 1 | X = x) = x^T \hat{\beta}`$

4. **Classification**: Apply threshold at 0.5:
```math
\hat{f}(x) = \mathbb{I}[x^T \hat{\beta} > 0.5]
```

5. **Error Calculation**: Compute training and test classification errors

### Performance Evaluation

**Metrics**:
- Training Error: $`\frac{1}{n} \sum_{i=1}^n \mathbb{I}[y_i \neq \hat{f}(x_i)]`$
- Test Error: $`\frac{1}{N} \sum_{j=1}^N \mathbb{I}[y_j^* \neq \hat{f}(x_j^*)]`$

**Visualization**: Plot error curves vs. model complexity (k for kNN, fixed complexity for linear regression)

## Expected Results

### Example 1 Predictions

**Linear Regression**: Should perform well due to:
- Linear decision boundary exists
- Gaussian class-conditional distributions
- Well-separated class means

**kNN**: Should also perform well, with optimal $`k`$ likely in the middle range (5-15)

### Example 2 Predictions

**Linear Regression**: Should perform poorly due to:
- Non-linear decision boundary
- Complex mixture distributions
- Multiple modes per class

**kNN**: Should perform better than linear regression, with optimal $`k`$ likely smaller than in Example 1

### Bias-Variance Analysis

**Example 1**: Both methods should have low bias, with kNN showing higher variance for small $`k`$

**Example 2**: Linear regression will have high bias, while kNN can achieve lower bias at the cost of higher variance

This simulation study will provide concrete evidence of the bias-variance tradeoff and help us understand the strengths and limitations of each method.
