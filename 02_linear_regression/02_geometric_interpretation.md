# 2.2. Geometric Interpretation

The geometric interpretation of least squares provides a powerful visual and mathematical framework for understanding linear regression. Instead of focusing on the $`(p+1)`$-dimensional feature space, we work in the $`n`$-dimensional space of observations, where each data point is represented as a vector. This perspective reveals the fundamental structure of linear regression and helps us understand concepts like projection, orthogonality, and the coefficient of determination.

---

## 2.2.1. Basic Concepts in Vector Spaces

**Vectors and Vector Spaces:**
- A vector is an ordered list of numbers that can be visualized as a point in space or as an arrow from the origin to that point.
- Vectors can be two-dimensional, three-dimensional, or more generally, $`n`$-dimensional.
- The set of all $`n`$-dimensional vectors forms a vector space, denoted $`\mathbb{R}^n`$.
- Vectors support two fundamental operations: addition and scalar multiplication.

**Vector Operations:**

```math
\text{Vector addition: } \mathbf{a} + \mathbf{b} = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} + \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix} = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{pmatrix}
```

```math
\text{Scalar multiplication: } c \mathbf{a} = c \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} = \begin{pmatrix} c a_1 \\ c a_2 \\ \vdots \\ c a_n \end{pmatrix}
```

**Example: Vector Addition and Scalar Multiplication**

```math
2 \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix} + 3 \begin{pmatrix} 3 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \\ 0 \end{pmatrix} + \begin{pmatrix} 9 \\ 3 \\ 3 \end{pmatrix} = \begin{pmatrix} 11 \\ 7 \\ 3 \end{pmatrix}
```

This example shows that:
- Scalar multiplication scales a vector by a factor.
- Vector addition combines vectors component-wise.
- The result remains in the same vector space.

**Python Example: Vector Operations**

```python
import numpy as np

# Define vectors
a = np.array([1, 2, 0])
b = np.array([3, 1, 1])

# Scalar multiplication and addition
result = 2 * a + 3 * b
print("2a + 3b =", result)

# Vector norm (length)
norm_a = np.linalg.norm(a)
print("||a|| =", norm_a)
```

**Linear Subspaces:**

A linear subspace is a subset of a vector space that is closed under vector addition and scalar multiplication. Formally, a subset $`S`$ of $`\mathbb{R}^n`$ is a linear subspace if:

1. $`\mathbf{0} \in S`$ (contains the zero vector)
2. If $`\mathbf{u}, \mathbf{v} \in S`$, then $`\mathbf{u} + \mathbf{v} \in S`$
3. If $`\mathbf{u} \in S`$ and $`c`$ is a scalar, then $`c\mathbf{u} \in S`$

**Key Properties:**
- A linear subspace always contains the origin.
- The dimension of a subspace is the number of linearly independent vectors needed to span it.
- In $`\mathbb{R}^2`$, subspaces are lines through the origin.
- In $`\mathbb{R}^3`$, subspaces can be lines or planes through the origin.

**Column Space of $`X`$:**

In regression, we focus on the column space of the design matrix $`X`$:

```math
C(X) = \{ \mathbf{X} \boldsymbol{\beta} : \boldsymbol{\beta} \in \mathbb{R}^{p+1} \}
```

This is the set of all possible linear combinations of the columns of $`X`$. It represents all possible predicted values that can be obtained from the model.

**Python Example: Column Space**

```python
import numpy as np

# Design matrix X (n x (p+1))
X = np.array([[1, 2], [1, 4], [1, 6]])  # n=3, p=1

# Column space: all possible X*beta
beta1 = np.array([1, 2])
beta2 = np.array([0, 1])

y1 = X @ beta1
y2 = X @ beta2

print("Column space examples:")
print("X * [1, 2] =", y1)
print("X * [0, 1] =", y2)
```

---

## 2.2.2. Least Squares and Projection

The least squares optimization problem can be understood geometrically as finding the projection of the response vector $`\mathbf{y}`$ onto the column space of $`X`$.

**The Projection Problem:**

We want to find $`\hat{\boldsymbol{\beta}}`$ such that $`\mathbf{X}\hat{\boldsymbol{\beta}}`$ is as close as possible to $`\mathbf{y}`$ in the Euclidean norm:

```math
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2
```

**Geometric Interpretation:**
- The column space $`C(X)`$ is a subspace of $`\mathbb{R}^n`$.
- The vector $`\mathbf{y}`$ may not lie in $`C(X)`$.
- The least squares solution finds the point in $`C(X)`$ closest to $`\mathbf{y}`$.
- This closest point is the orthogonal projection of $`\mathbf{y}`$ onto $`C(X)`$.

**Orthogonal Decomposition:**

The least squares solution decomposes $`\mathbf{y}`$ into two orthogonal components:

1. **Predicted values:** $`\hat{\mathbf{y}} = \mathbf{X} \hat{\boldsymbol{\beta}}`$ (lies in $`C(X)`$)
2. **Residual vector:** $`\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}`$ (orthogonal to $`C(X)`$)

**Key Properties:**
- $`\hat{\mathbf{y}}`$ and $`\mathbf{r}`$ are orthogonal: $`\hat{\mathbf{y}}^T \mathbf{r} = 0`$
- $`\mathbf{y} = \hat{\mathbf{y}} + \mathbf{r}`$
- $`\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2`$ (Pythagorean theorem)

**Python Example: Projection and Orthogonality**

```python
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(10, 2)
X = np.column_stack([np.ones(10), X])  # Add intercept
beta_true = np.array([2, 1, -0.5])
y = X @ beta_true + np.random.normal(0, 0.1, 10)

# Least squares solution
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat
r = y - y_hat

# Check orthogonality
orthogonality = y_hat @ r
print("Orthogonality check (should be close to 0):", orthogonality)

# Check Pythagorean theorem
norm_y_sq = np.sum(y**2)
norm_yhat_sq = np.sum(y_hat**2)
norm_r_sq = np.sum(r**2)
print("Pythagorean theorem check:", abs(norm_y_sq - (norm_yhat_sq + norm_r_sq)))
```

---

## 2.2.3. $`R^2`$: The Coefficient of Determination

$`R^2`$ (R-squared) is a fundamental measure of model fit that quantifies the proportion of variance in the response variable explained by the predictors.

### Definition of $`R^2`$

```math
R^2 = \frac{\sum_{i=1}^n (\hat{y}_i - \bar{y})^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = \frac{\| \hat{\mathbf{y}} - \bar{\mathbf{y}} \|^2}{\| \mathbf{y} - \bar{\mathbf{y}} \|^2}
```

where $`\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i`$ is the sample mean of the response.

### Geometric Interpretation

The geometric interpretation of $`R^2`$ comes from the Pythagorean theorem applied to centered vectors:

```math
\| \mathbf{y} - \bar{\mathbf{y}} \|^2 = \| \hat{\mathbf{y}} - \bar{\mathbf{y}} \|^2 + \| \mathbf{r} \|^2
```

This decomposition gives us:

- **Total Sum of Squares (TSS):** $`\| \mathbf{y} - \bar{\mathbf{y}} \|^2`$
- **Fitted Sum of Squares (FSS):** $`\| \hat{\mathbf{y}} - \bar{\mathbf{y}} \|^2`$
- **Residual Sum of Squares (RSS):** $`\| \mathbf{r} \|^2`$

**Alternative Expressions:**

```math
R^2 = \frac{\text{FSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}
```

**Key Properties:**
- $`R^2`$ always lies between 0 and 1.
- $`R^2 = 1`$ means perfect fit (all residuals are zero).
- $`R^2 = 0`$ means the model performs no better than predicting the mean.
- In multiple regression, $`R^2`$ is the squared correlation between $`y`$ and $`\hat{y}`$.
- In simple regression, $`R^2`$ is the squared correlation between $`y`$ and $`x`$.

### Invariance Properties

$`R^2`$ has several important invariance properties:

1. **Location invariance:** Adding a constant to $`y`$ does not change $`R^2`$.
2. **Scale invariance:** Multiplying $`y`$ by a constant does not change $`R^2``.
3. **Symmetry in simple regression:** $`R^2`$ is the same whether we predict $`Y`$ from $`X`$ or $`X`$ from $`Y`$.

### Interpretation and Limitations

**Interpretation:**
- High $`R^2`$ (e.g., 0.7 or 0.8) suggests a good fit, but does not guarantee model validity.
- Low $`R^2`$ does not necessarily mean the model is useless; it may still provide useful predictions.

**Limitations:**
- Adding more predictors (even irrelevant ones) can artificially increase $`R^2`$.
- $`R^2`$ does not account for the number of predictors.

**Adjusted $`R^2`$:**

To address the limitation of $`R^2`$ increasing with more predictors, we use adjusted $`R^2`$:

```math
R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)} = 1 - (1 - R^2) \frac{n-1}{n-p-1}
```

Adjusted $`R^2`$ penalizes models with many predictors and can decrease when adding irrelevant variables.

**Python Example: Computing $`R^2`$**

```python
import numpy as np
from sklearn.metrics import r2_score

# Using the data from previous example
y_mean = np.mean(y)
TSS = np.sum((y - y_mean)**2)
RSS = np.sum(r**2)
FSS = TSS - RSS

# Manual computation
R2_manual = FSS / TSS
print("Manual R² =", R2_manual)

# Using sklearn
R2_sklearn = r2_score(y, y_hat)
print("Sklearn R² =", R2_sklearn)

# Adjusted R²
n, p = X.shape
R2_adj = 1 - (1 - R2_manual) * (n - 1) / (n - p - 1)
print("Adjusted R² =", R2_adj)
```

---

## 2.2.4. Linear Transformations of $`X`$

Linear transformations of the design matrix $`X`$ have important implications for the least squares solution.

**Effect on the Fit:**

If we transform $`X`$ to $`X' = XA`$ where $`A`$ is a full-rank matrix, then:

- The column space $`C(X') = C(X)`$ remains the same.
- The fitted values $`\hat{\mathbf{y}}`$ are unchanged.
- The residuals $`\mathbf{r}`$ are unchanged.
- $`R^2`$ is unchanged.
- However, the coefficients $`\boldsymbol{\beta}`$ will change.

**Example: Scaling Predictors**

If we scale a predictor by a factor $`c`$, the corresponding coefficient is scaled by $`1/c`$:

```math
\text{Original: } y = \beta_0 + \beta_1 x_1 + \beta_2 x_2
```

```math
\text{Scaled: } y = \beta_0' + \beta_1' (c x_1) + \beta_2' x_2
```

where $`\beta_1' = \beta_1 / c`$.

**Python Example: Effect of Scaling**

```python
import numpy as np

# Original data
X_orig = np.random.randn(100, 2)
X_orig = np.column_stack([np.ones(100), X_orig])
beta_true = np.array([1, 2, 3])
y = X_orig @ beta_true + np.random.normal(0, 0.1, 100)

# Scaled data (scale first predictor by 2)
X_scaled = X_orig.copy()
X_scaled[:, 1] *= 2

# Fit both models
beta_orig = np.linalg.inv(X_orig.T @ X_orig) @ X_orig.T @ y
beta_scaled = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y

print("Original coefficients:", beta_orig)
print("Scaled coefficients:", beta_scaled)
print("Ratio (should be 2):", beta_orig[1] / beta_scaled[1])
```

---

## 2.2.5. Rank Deficiency

Rank deficiency occurs when the design matrix $`X`$ does not have full column rank, meaning some columns are linear combinations of others.

**Definition:**
$`X`$ is rank deficient if its rank is less than $`p+1`$ (the number of columns).

**Common Causes:**
1. **Perfect collinearity:** Two predictors are perfectly correlated.
2. **Redundant variables:** A predictor is a linear combination of others.
3. **Categorical variables:** Including all levels of a categorical variable with an intercept.

**Examples:**

1. **Same quantity, different units:**
   ```python
   # Temperature in Celsius and Fahrenheit
   temp_c = np.array([0, 10, 20, 30])
   temp_f = 9/5 * temp_c + 32  # Perfect linear relationship
   X = np.column_stack([np.ones(4), temp_c, temp_f])
   ```

2. **Sum to constant:**
   ```python
   # Three age groups that sum to 100%
   age_young = np.array([30, 25, 40])
   age_middle = np.array([45, 50, 35])
   age_old = 100 - age_young - age_middle
   X = np.column_stack([np.ones(3), age_young, age_middle, age_old])
   ```

**Consequences of Rank Deficiency:**

1. **Non-unique solutions:** $`(X^T X)^{-1}`$ does not exist.
2. **Infinite solutions:** There are infinitely many $`\boldsymbol{\beta}`$ that give the same fitted values.
3. **Software behavior:** Different software packages handle rank deficiency differently.

**Software Handling:**

- **R's `lm()`:** Drops redundant columns and marks their coefficients as `NA`.
- **Python's scikit-learn:** Returns the minimum-norm solution using the Moore-Penrose pseudoinverse.
- **NumPy's `np.linalg.lstsq()`:** Also uses the pseudoinverse.

**Python Example: Rank Deficiency**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Create rank-deficient data
X = np.array([[1, 2, 3], [1, 4, 6], [1, 6, 9]])  # Third column = 3 * second column
y = np.array([1, 2, 3])

# Check rank
rank = np.linalg.matrix_rank(X)
print("Rank of X:", rank)
print("Number of columns:", X.shape[1])

# Try different methods
try:
    # Direct inverse (will fail)
    beta_direct = np.linalg.inv(X.T @ X) @ X.T @ y
except np.linalg.LinAlgError:
    print("Direct inverse failed due to rank deficiency")

# Pseudoinverse (works)
beta_pinv = np.linalg.pinv(X) @ y
print("Pseudoinverse solution:", beta_pinv)

# Sklearn (also works)
model = LinearRegression()
model.fit(X, y)
print("Sklearn solution:", model.coef_)
```

**Key Points:**
- Rank deficiency does not invalidate the model entirely.
- The fitted values $`\hat{\mathbf{y}}`$ and residuals $`\mathbf{r}`$ are still unique.
- Only the coefficient interpretation becomes problematic.
- Understanding rank deficiency is crucial for proper model specification and interpretation.

---

*This section provided a comprehensive geometric interpretation of linear regression, covering vector spaces, projection, $`R^2`$, linear transformations, and rank deficiency. The geometric perspective helps us understand the fundamental structure of linear regression and provides intuition for more advanced methods.*
