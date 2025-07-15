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

**Understanding Distance Metrics:**

1. **Euclidean Distance**: Most common for continuous features
   - Measures "straight-line" distance between points
   - Sensitive to scale of features
   - Assumes all features are equally important

2. **Manhattan Distance (L1 norm)**:
```math
d(x_i, x_j) = \sum_{l=1}^p |x_{il} - x_{jl}|
```
   - Measures "city block" distance
   - Less sensitive to outliers than Euclidean
   - Useful when features have different scales

3. **Weighted Euclidean Distance**:
```math
d(x_i, x_j) = \sqrt{\sum_{l=1}^p w_l (x_{il} - x_{jl})^2}
```
   - Allows different importance for different features
   - Weights $`w_l`$ can be learned or set based on domain knowledge

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

**Understanding the kNN Algorithm:**

1. **Neighborhood Definition**: $`\mathcal{N}_k(x)`$ contains indices of $`k`$ closest training points
2. **Local Averaging**: Prediction is average of neighbors' values
3. **No Training**: Algorithm simply stores training data
4. **Lazy Learning**: Computation happens only at prediction time

### Example Walkthrough

Consider $`k=5`$ in a binary classification problem. If among the five nearest neighbors:
- 3 have $`Y=1`$
- 2 have $`Y=0`$

Then:
- **Majority vote**: Predict $`Y=1`$
- **Probability estimates**: $`P(Y=1) = 3/5 = 0.6`$, $`P(Y=0) = 2/5 = 0.4`$

**Detailed Example: House Price Prediction**

Suppose we have training data with house features (square footage, bedrooms) and prices:
- House 1: (1500 sq ft, 3 beds) → $300,000
- House 2: (1600 sq ft, 3 beds) → $320,000
- House 3: (1400 sq ft, 2 beds) → $280,000
- House 4: (1700 sq ft, 4 beds) → $350,000
- House 5: (1550 sq ft, 3 beds) → $310,000

For a new house: (1520 sq ft, 3 beds)

**Step 1: Calculate distances**
- Distance to House 1: $`\sqrt{(1520-1500)^2 + (3-3)^2} = 20`$
- Distance to House 2: $`\sqrt{(1520-1600)^2 + (3-3)^2} = 80`$
- Distance to House 3: $`\sqrt{(1520-1400)^2 + (3-2)^2} = 120.04`$
- Distance to House 4: $`\sqrt{(1520-1700)^2 + (3-4)^2} = 180.01`$
- Distance to House 5: $`\sqrt{(1520-1550)^2 + (3-3)^2} = 30`$

**Step 2: Find k=3 nearest neighbors**
- House 1 (distance 20)
- House 5 (distance 30)
- House 2 (distance 80)

**Step 3: Predict price**
- $`\hat{y} = \frac{300,000 + 310,000 + 320,000}{3} = 310,000`$

### Algorithm Properties

**No Training Phase**: kNN is a "lazy learner" - it simply stores the training data and performs computation only at prediction time.

**Local Approximation**: kNN approximates the true function $`f^*(x)`$ locally around each test point:
```math
\hat{f}(x) \approx \mathbb{E}[Y | X \in \mathcal{B}_k(x)]
```
where $`\mathcal{B}_k(x)`$ is the neighborhood defined by the $`k`$ nearest neighbors.

**Understanding Local Approximation:**

1. **Neighborhood**: $`\mathcal{B}_k(x)`$ is the region containing the $`k`$ nearest neighbors
2. **Local Expectation**: We estimate the expected value of $`Y`$ in this neighborhood
3. **Assumption**: Points in the neighborhood have similar $`Y`$ values

**Mathematical Properties:**

1. **Consistency**: Under certain conditions, kNN converges to the true function
2. **No Parametric Assumptions**: Works with any data distribution
3. **Adaptive Bandwidth**: Neighborhood size adapts to local density

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

**Mathematical Analysis of k Selection:**

The optimal $`k`$ depends on:
1. **Sample size $`n`$**: Larger $`n`$ allows larger $`k`$
2. **Data dimensionality $`p`$**: Higher $`p`$ requires smaller $`k`$ (curse of dimensionality)
3. **Noise level**: More noise requires larger $`k`$ for smoothing
4. **Local structure**: Complex local patterns require smaller $`k`$

**Example: k Selection via Cross-Validation**

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, random_state=42)

# Test different k values
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101]
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    cv_scores.append(scores.mean())

# Find optimal k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, 'bo-', linewidth=2)
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('kNN: k Selection via Cross-Validation')
plt.legend()
plt.grid(True)
plt.show()
```

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

**Cosine Similarity** (for text data):
```math
\text{cosine}(x_i, x_j) = \frac{x_i^T x_j}{\|x_i\| \|x_j\|}
```

**Mahalanobis Distance** (accounts for feature correlations):
```math
d(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)}
```
where $`\Sigma`$ is the covariance matrix.

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

**Understanding the Linear Model:**

1. **Additive Structure**: Each feature contributes linearly to the prediction
2. **Intercept**: $`\beta_0`$ represents the baseline prediction when all features are zero
3. **Slope Coefficients**: $`\beta_j`$ represents the change in $`Y`$ per unit change in $`X_j`$
4. **Error Term**: $`\epsilon`$ captures unmodeled variation and measurement error

**Estimation**: Minimize the sum of squared residuals:
```math
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - x_i^T \beta)^2
```

**Understanding Least Squares:**

The objective function is:
```math
L(\beta) = \sum_{i=1}^n (y_i - x_i^T \beta)^2 = \|y - X\beta\|^2
```

Taking the gradient and setting to zero:
```math
\nabla L(\beta) = -2X^T(y - X\beta) = 0
```

Solving for $`\beta`$:
```math
X^T X \beta = X^T y
```

**Closed-Form Solution**: For $`p`$-dimensional $`X`$, we estimate $`p+1`$ parameters:
```math
\hat{\beta} = (X^T X)^{-1} X^T y
```

**Prediction**: $`\hat{y} = x^T \hat{\beta}`$

**Understanding the Solution:**

1. **Normal Equations**: $`X^T X \beta = X^T y`$ are the normal equations
2. **Matrix Inversion**: $`(X^T X)^{-1}`$ exists if $`X`$ has full column rank
3. **Projection**: $`\hat{y}`$ is the projection of $`y`$ onto the column space of $`X`$

**Example: Simple Linear Regression**

Consider predicting house price ($`Y`$) from square footage ($`X`$):

Training data:
- House 1: 1500 sq ft → $300,000
- House 2: 2000 sq ft → $400,000
- House 3: 2500 sq ft → $500,000

**Step 1: Set up matrices**
```math
X = \begin{bmatrix} 1 & 1500 \\ 1 & 2000 \\ 1 & 2500 \end{bmatrix}, \quad y = \begin{bmatrix} 300000 \\ 400000 \\ 500000 \end{bmatrix}
```

**Step 2: Compute normal equations**
```math
X^T X = \begin{bmatrix} 3 & 6000 \\ 6000 & 12500000 \end{bmatrix}
```
```math
X^T y = \begin{bmatrix} 1200000 \\ 2450000000 \end{bmatrix}
```

**Step 3: Solve for $`\beta`$**
```math
\hat{\beta} = (X^T X)^{-1} X^T y = \begin{bmatrix} 100000 \\ 160 \end{bmatrix}
```

**Step 4: Prediction equation**
```math
\hat{y} = 100000 + 160x
```

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

**Understanding Linear Classification:**

1. **Linear Decision Boundary**: $`x^T \beta = 0.5`$ defines the decision boundary
2. **Probability Interpretation**: $`x^T \beta`$ can be interpreted as log-odds
3. **Limitations**: Predictions can be outside $`[0,1]`$ interval

**Example: Linear Classification**

Consider classifying emails as spam (1) or not spam (0) based on word frequencies:

Training data:
- Email 1: (0.1, 0.2) → 0 (not spam)
- Email 2: (0.8, 0.1) → 1 (spam)
- Email 3: (0.9, 0.3) → 1 (spam)

**Step 1: Fit linear regression**
```math
\hat{\beta} = \begin{bmatrix} -0.5 \\ 1.5 \\ 0.5 \end{bmatrix}
```

**Step 2: Decision boundary**
```math
-0.5 + 1.5x_1 + 0.5x_2 = 0.5
```
```math
1.5x_1 + 0.5x_2 = 1.0
```

**Step 3: Classification rule**
- Predict spam if $`1.5x_1 + 0.5x_2 > 1.0`$
- Predict not spam otherwise

## Pros and Cons Analysis

### Linear Regression Advantages

1. **Computational Efficiency**: $`O(np^2 + p^3)`$ for training, $`O(p)`$ for prediction
2. **Interpretability**: Coefficients have clear meaning
3. **Statistical Inference**: Confidence intervals, hypothesis tests available
4. **Scalability**: Works well with large datasets
5. **Theoretical Foundation**: Well-understood properties

**Understanding Computational Complexity:**

1. **Training**: $`O(np^2)`$ for matrix multiplication + $`O(p^3)`$ for matrix inversion
2. **Prediction**: $`O(p)`$ for single prediction (just matrix-vector multiplication)
3. **Memory**: $`O(p^2)`$ to store $`(X^T X)^{-1}`$

**Statistical Inference:**

Confidence intervals for coefficients:
```math
\hat{\beta}_j \pm t_{n-p-1, \alpha/2} \cdot \text{SE}(\hat{\beta}_j)
```

where $`\text{SE}(\hat{\beta}_j) = \sqrt{\sigma^2 (X^T X)^{-1}_{jj}}`$

### Linear Regression Drawbacks

1. **Linear Assumption**: May miss non-linear relationships
2. **Invalid Probabilities**: Predictions can be outside $`[0, 1]`$ interval
3. **Squared Loss**: Not optimal for classification performance
4. **Feature Interactions**: Cannot capture complex interactions without manual feature engineering

**Example: Linear vs. Non-linear Relationship**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 0.5 * X.flatten()**2 + np.random.normal(0, 1, 100)

# Fit linear regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Fit polynomial regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly = poly_model.predict(X_poly)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_linear, 'r-', linewidth=2, label='Linear Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression on Non-linear Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_poly, 'g-', linewidth=2, label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### kNN Advantages

1. **No Assumptions**: Works with any data distribution
2. **Non-linear**: Can capture complex decision boundaries
3. **Local Adaptation**: Automatically adapts to local structure
4. **Conceptual Simplicity**: Easy to understand and implement

**Example: kNN Capturing Non-linear Boundaries**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# Fit kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Create mesh for visualization
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict on mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('kNN Decision Boundary on Non-linear Data')
plt.show()
```

### kNN Drawbacks

1. **Computational Cost**: $`O(n)`$ for each prediction
2. **Curse of Dimensionality**: Performance degrades in high dimensions
3. **No Interpretability**: Black-box predictions
4. **Sensitive to Irrelevant Features**: All features weighted equally

**Understanding the Curse of Dimensionality:**

In high dimensions, all points become approximately equidistant:
```math
\lim_{p \rightarrow \infty} \frac{\max_{i,j} d(x_i, x_j) - \min_{i,j} d(x_i, x_j)}{\min_{i,j} d(x_i, x_j)} = 0
```

**Example: Curse of Dimensionality**

```python
# Demonstrate curse of dimensionality
n_samples = 1000
dimensions = [1, 2, 5, 10, 20, 50, 100]
distances = []

for p in dimensions:
    X = np.random.randn(n_samples, p)
    
    # Calculate distances from first point to all others
    dists = np.sqrt(np.sum((X - X[0])**2, axis=1))
    
    # Calculate coefficient of variation
    cv = np.std(dists) / np.mean(dists)
    distances.append(cv)

plt.figure(figsize=(10, 6))
plt.plot(dimensions, distances, 'bo-', linewidth=2)
plt.xlabel('Number of Dimensions')
plt.ylabel('Coefficient of Variation of Distances')
plt.title('Curse of Dimensionality: Distance Concentration')
plt.grid(True)
plt.show()
```

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

**Understanding Degrees of Freedom:**

1. **Linear Regression**: Each parameter contributes one degree of freedom
2. **kNN**: Complexity decreases as $`k`$ increases
3. **Effective Parameters**: kNN has $`n/k`$ effective parameters because each neighbor contributes $`1/k`$ to the prediction

### Complexity Comparison

| Algorithm | Parameters | Training Time | Prediction Time | Flexibility |
|-----------|------------|---------------|-----------------|-------------|
| Linear Regression | $`p+1`$ | $`O(np^2 + p^3)`$ | $`O(p)`$ | Low |
| kNN | $`n/k`$ | $`O(1)`$ | $`O(n)`$ | High |

**Understanding the Trade-offs:**

1. **Training Time**: kNN has no training, linear regression requires matrix operations
2. **Prediction Time**: Linear regression is fast, kNN requires distance computations
3. **Memory**: Linear regression stores $`p+1`$ parameters, kNN stores all training data
4. **Flexibility**: kNN can capture complex patterns, linear regression is limited to linear relationships

## Theoretical Foundations

### kNN Consistency

Under certain conditions, kNN is consistent (converges to the Bayes classifier):

**Theorem**: If $`k \rightarrow \infty`$ and $`k/n \rightarrow 0`$ as $`n \rightarrow \infty`$, then:
```math
\lim_{n \rightarrow \infty} \mathbb{E}[(\hat{f}_n(X) - f^*(X))^2] = 0
```

**Understanding the Conditions:**

1. **$`k \rightarrow \infty`$**: Neighborhood size grows to include more points
2. **$`k/n \rightarrow 0`$**: Neighborhood becomes smaller relative to sample size
3. **Result**: Local approximation becomes more accurate

**Proof Sketch:**

The kNN estimator can be written as:
```math
\hat{f}_n(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
```

As $`n \rightarrow \infty`$ and $`k/n \rightarrow 0`$:
1. The neighborhood $`\mathcal{N}_k(x)`$ becomes smaller
2. Points in the neighborhood become closer to $`x`$
3. $`\hat{f}_n(x) \rightarrow \mathbb{E}[Y|X=x] = f^*(x)`$

**Intuition**: As sample size grows, the neighborhood becomes smaller and more localized, providing better local approximation.

### Linear Regression Optimality

**Gauss-Markov Theorem**: Under the linear model assumptions, the least squares estimator is the Best Linear Unbiased Estimator (BLUE).

**Assumptions for Gauss-Markov:**
1. **Linearity**: $`Y = X^T \beta + \epsilon`$
2. **Random Sampling**: Data is randomly sampled
3. **No Perfect Multicollinearity**: $`X`$ has full column rank
4. **Homoscedasticity**: $`\text{Var}(\epsilon_i) = \sigma^2`$ for all $`i`$
5. **No Autocorrelation**: $`\text{Cov}(\epsilon_i, \epsilon_j) = 0`$ for $`i \neq j`$

**Maximum Likelihood**: Under normality assumption, least squares is equivalent to maximum likelihood estimation.

**Understanding MLE for Linear Regression:**

If $`\epsilon \sim N(0, \sigma^2)`$, then $`Y \sim N(X^T \beta, \sigma^2)`$.

The likelihood function is:
```math
L(\beta, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^T \beta)^2}{2\sigma^2}\right)
```

The log-likelihood is:
```math
\ell(\beta, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - x_i^T \beta)^2
```

Maximizing with respect to $`\beta`$ is equivalent to minimizing the sum of squared residuals.

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

**Understanding the Data Generation:**

1. **Class Separation**: $`\|\mu_1 - \mu_0\| = \sqrt{8} \approx 2.83`$ (well-separated)
2. **Linear Boundary**: The optimal decision boundary is linear
3. **Equal Variance**: Both classes have the same covariance structure
4. **Balanced Classes**: Equal prior probabilities

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

**Understanding Mixture Distributions:**

1. **Multiple Modes**: Each class has 10 different centers
2. **Non-linear Boundary**: The optimal decision boundary is highly non-linear
3. **Complex Structure**: Data cannot be separated by a single hyperplane
4. **Local Patterns**: Different regions have different class distributions

## Mixture Distribution Theory

### Mathematical Definition

A **mixture distribution** is a probabilistic model representing various subgroups within a larger population. The probability density function is:

```math
f(x) = \sum_{j=1}^k w_j f_j(x)
```

where:
- $`w_j`$ are mixing weights with $`\sum_{j=1}^k w_j = 1`$
- $`f_j(x)`$ are component densities (e.g., Gaussian PDFs)

**Understanding Mixture Models:**

1. **Component Densities**: Each $`f_j(x)`$ represents a subpopulation
2. **Mixing Weights**: $`w_j`$ represents the proportion of the population in component $`j`$
3. **Flexibility**: Can model complex, multi-modal distributions
4. **Interpretability**: Each component can have meaningful interpretation

### Sampling from Mixture Distributions

**Two-Step Process**:

1. **Component Selection**: Draw $`Z \sim \text{Categorical}(w_1, \ldots, w_k)`$
2. **Data Generation**: Draw $`X | Z = j \sim f_j(x)`$

**Mathematical Justification**: This treats the mixture as the marginal of a joint distribution:
```math
f(x, z) = f(z) f(x | z) = w_z f_z(x)
```

**Marginal Distribution**: $`f(x) = \sum_{z=1}^k f(x, z) = \sum_{j=1}^k w_j f_j(x)`$

**Example: Gaussian Mixture Model**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate mixture data
np.random.seed(42)
n_samples = 1000

# Component parameters
means = [[0, 0], [4, 4], [0, 4]]
covariances = [np.eye(2), np.eye(2), np.eye(2)]
weights = [0.4, 0.3, 0.3]

# Generate data
X = np.zeros((n_samples, 2))
for i in range(n_samples):
    # Choose component
    component = np.random.choice(3, p=weights)
    # Generate from chosen component
    X[i] = np.random.multivariate_normal(means[component], covariances[component])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Generated Mixture Data')
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1, 2, 2)
# Generate points for contour plot
x = np.linspace(-2, 6, 100)
y = np.linspace(-2, 6, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XY = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

# Calculate density
density = np.exp(gmm.score_samples(XY))
density = density.reshape(X_grid.shape)

plt.contour(X_grid, Y_grid, density, levels=20)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Fitted Mixture Model')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()
```

## Implementation Strategy

### kNN Implementation

1. **Parameter Grid**: Define a set of $`k`$ values (e.g., $`k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201\}`$)

2. **Error Storage**: Initialize vectors to store training and test errors for each $`k`$

3. **Distance Computation**: For each test point, compute distances to all training points

4. **Prediction**: For each $`k`$:
   - Find $`k`$ nearest neighbors
   - Compute majority vote or average
   - Calculate classification error

**Example: kNN Implementation**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def evaluate_knn(X_train, y_train, X_test, y_test, k_values):
    """Evaluate kNN for different k values"""
    train_errors = []
    test_errors = []
    
    for k in k_values:
        # Fit kNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Calculate errors
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)
        
        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return train_errors, test_errors

# Example usage
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201]
train_errors, test_errors = evaluate_knn(X_train, y_train, X_test, y_test, k_values)
```

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

**Example: Linear Regression Implementation**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def evaluate_linear_regression(X_train, y_train, X_test, y_test):
    """Evaluate linear regression for classification"""
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    train_pred_proba = lr.predict(X_train)
    test_pred_proba = lr.predict(X_test)
    
    # Convert to binary predictions
    train_pred = (train_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)
    
    # Calculate errors
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)
    
    return train_error, test_error

# Example usage
train_error, test_error = evaluate_linear_regression(X_train, y_train, X_test, y_test)
```

### Performance Evaluation

**Metrics**:
- Training Error: $`\frac{1}{n} \sum_{i=1}^n \mathbb{I}[y_i \neq \hat{f}(x_i)]`$
- Test Error: $`\frac{1}{N} \sum_{j=1}^N \mathbb{I}[y_j^* \neq \hat{f}(x_j^*)]`$

**Visualization**: Plot error curves vs. model complexity (k for kNN, fixed complexity for linear regression)

**Example: Performance Visualization**

```python
def plot_performance_comparison(k_values, knn_train_errors, knn_test_errors, 
                               lr_train_error, lr_test_error):
    """Plot performance comparison between kNN and linear regression"""
    plt.figure(figsize=(12, 5))
    
    # kNN performance
    plt.subplot(1, 2, 1)
    plt.plot(k_values, knn_train_errors, 'b-', label='Training Error', linewidth=2)
    plt.plot(k_values, knn_test_errors, 'r-', label='Test Error', linewidth=2)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Error Rate')
    plt.title('kNN Performance')
    plt.legend()
    plt.grid(True)
    
    # Linear regression performance
    plt.subplot(1, 2, 2)
    plt.bar(['Training', 'Test'], [lr_train_error, lr_test_error], 
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Error Rate')
    plt.title('Linear Regression Performance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_performance_comparison(k_values, train_errors, test_errors, 
                           lr_train_error, lr_test_error)
```

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

**Understanding the Results:**

1. **Example 1 (Linear Data)**:
   - Linear regression: Low bias, low variance
   - kNN: Low bias, moderate variance (depends on k)
   - Both methods should achieve similar performance

2. **Example 2 (Non-linear Data)**:
   - Linear regression: High bias (cannot capture non-linear patterns)
   - kNN: Low bias, higher variance
   - kNN should outperform linear regression

This simulation study will provide concrete evidence of the bias-variance tradeoff and help us understand the strengths and limitations of each method. The results will demonstrate when each algorithm is most appropriate and how to choose optimal parameters for different types of data.
