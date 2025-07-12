# 2. Linear Regression

## 2.1. Multiple Linear Regression

Multiple linear regression (MLR) is a foundational technique in statistical learning that models the relationship between a single response variable and multiple predictor variables. The general form of the model is:

```math
y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + e
```

where:
- $`y`$ is the response (dependent) variable.
- $`x_1, \ldots, x_p`$ are the predictor (independent) variables.
- $`\beta_0, \ldots, \beta_p`$ are the regression coefficients.
- $`e`$ is the error term, capturing the variation in $`y`$ not explained by the predictors.

MLR is widely used because it is simple, interpretable, and computationally efficient. Despite its linearity assumption, many ideas from MLR extend to more complex models.

---

### 2.1.1. Setup: Matrix Representation

To handle multiple predictors efficiently, we use matrix notation. This allows us to write the model for all $`n`$ observations compactly:

```math
y = X\beta + e
```

where:
- $`y`$ is an $`n \times 1`$ vector of observed responses.
- $`X`$ is the $`n \times (p+1)`$ design matrix (including a column of ones for the intercept).
- $`\beta`$ is a $`(p+1) \times 1`$ vector of coefficients.
- $`e`$ is an $`n \times 1`$ vector of errors.

**Expanded matrix form:**

```math
\begin{align*}
\begin{pmatrix}
  y_1 \\
  y_2 \\
  \vdots \\
  y_n
\end{pmatrix}
&=
\begin{pmatrix}
  1 & x_{11} & \cdots & x_{1p} \\
  1 & x_{21} & \cdots & x_{2p} \\
  \vdots & \vdots & \ddots & \vdots \\
  1 & x_{n1} & \cdots & x_{np}
\end{pmatrix}
\begin{pmatrix}
  \beta_0 \\
  \beta_1 \\
  \vdots \\
  \beta_p
\end{pmatrix}
+
\begin{pmatrix}
  e_1 \\
  e_2 \\
  \vdots \\
  e_n
\end{pmatrix}
\end{align*}
```

Or, more compactly:

```math
\mathbf{y}_{n \times 1} = \mathbf{X}_{n \times (p+1)} \boldsymbol{\beta}_{(p+1) \times 1} + \mathbf{e}_{n \times 1}
```

**Interpretation:**
- Each row of $`X`$ corresponds to one observation, with the first column being all ones (for the intercept).
- The model predicts $`y`$ as a linear combination of the predictors plus an error.

#### Classical vs. Modern Settings

- **Classical setting ($`n \gg p`$):**
  - $`X`$ is "tall and skinny" (many more samples than features).
  - $`X^T X`$ is typically invertible, so the solution for $`\beta`$ is unique.
  - Statistical properties (like unbiasedness, variance) are well understood.

- **Modern setting ($`p \gg n`$):**
  - $`X`$ is "short and fat" (more features than samples).
  - $`X^T X`$ is not invertible; there are infinitely many solutions for $`\beta`$.
  - This can lead to overfitting: the model can fit the training data perfectly but generalize poorly.

*For this section, we focus on the classical case: $`n \gg p`$.*

---

### 2.1.2. Least Squares (LS) Principle

The most common method for estimating the coefficients in linear regression is **least squares**. The idea is to choose $`\beta`$ to minimize the total squared difference between the observed and predicted values:

```math
\text{RSS}(\beta) = \sum_{i=1}^n (y_i - \beta_0 - x_{i1} \beta_1 - \cdots - x_{ip} \beta_p)^2
```

- $`\text{RSS}`$ stands for **residual sum of squares**.
- The goal: find the $`\beta`$ that makes the observed points as close as possible to the regression hyperplane.

**Geometric interpretation:**
- For two predictors, the model defines a plane in 3D space.
- Each data point is a triplet $`(x_1, x_2, y)`$.
- The residual is the vertical distance from the point to the plane.
- The best-fit plane minimizes the sum of squared residuals.

**Why squared error?**
- Squared error is mathematically convenient (leads to closed-form solution).
- It penalizes large errors more heavily than small ones.
- Other loss functions (like absolute error) are possible, but less common in basic regression.

---

### 2.1.3. Least Squares Estimate: The Normal Equation

In matrix notation, the RSS is:

```math
\text{RSS}(\boldsymbol{\beta}) = \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2
```

To find the $`\beta`$ that minimizes RSS, we take the derivative with respect to $`\boldsymbol{\beta}`$ and set it to zero:

```math
\mathbf{X}^T \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}
```

- This is called the **normal equation**.
- If $`X^T X`$ is invertible, the unique solution is:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
```

- $`\hat{\boldsymbol{\beta}}`$ is the vector of least squares estimates.
- The hat $`\hat{}`$ denotes an estimate based on the observed data.

**Note:** If $`X^T X`$ is not invertible (e.g., due to collinearity or $`p > n`$), there are infinitely many solutions. Regularization or dimensionality reduction is needed in such cases (see later sections).

**Python Example: Computing Least Squares Solution**

```python
import numpy as np

# X: n x (p+1) design matrix (with intercept column)
# y: n x 1 response vector
X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])  # Example with intercept
y = np.array([1, 2, 3])

# Compute least squares estimate
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_hat)
```

---

### 2.1.4. Least Squares Output: Fitted Values, Residuals, and Variance

Once we have $`\hat{\boldsymbol{\beta}}`$, we can compute several important quantities:

- **Fitted values:** $`\hat{\mathbf{y}} = \mathbf{X} \hat{\boldsymbol{\beta}}`$
- **Residuals:** $`\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}`$
- **Residual sum of squares (RSS):** $`\|\mathbf{r}\|^2`$
- **Estimate of error variance ($`\sigma^2`$):**

```math
\hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1}
```

  - The denominator $`n - p - 1`$ is the **degrees of freedom**: $`n`$ observations minus $`p+1`$ estimated parameters (including the intercept).
  - Each column of $`X`$ imposes a constraint, reducing the degrees of freedom.

**Python Example: Fitted Values and Residuals**

```python
# Given X, y, and beta_hat from above

y_hat = X @ beta_hat
residuals = y - y_hat
RSS = np.sum(residuals ** 2)
sigma2_hat = RSS / (X.shape[0] - X.shape[1])
print("Fitted values:", y_hat)
print("Residuals:", residuals)
print("Estimated error variance:", sigma2_hat)
```

---

**Key Points:**
- Always check the rank of $`X`$ before computing $`\hat{\boldsymbol{\beta}}`$.
- The properties and interpretation of the least squares solution depend on the structure of $`X`$ and the assumptions of the model.
- Understanding the geometry and algebra of linear regression is essential for more advanced statistical learning methods.

---

*This section provided a detailed introduction to multiple linear regression, its matrix formulation, the least squares principle, and the computation of fitted values and residuals. In the next sections, we will discuss geometric interpretations, statistical properties, and practical issues in regression modeling.*
