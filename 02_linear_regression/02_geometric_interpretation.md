# 2.2. Geometric Interpretation: The Visual Foundation of Linear Regression

The geometric interpretation of least squares provides a powerful visual and mathematical framework for understanding linear regression. Instead of focusing on the $`(p+1)`$-dimensional feature space, we work in the $`n`$-dimensional space of observations, where each data point is represented as a vector. This perspective reveals the fundamental structure of linear regression and helps us understand concepts like projection, orthogonality, and the coefficient of determination.

## 2.2.1. Vector Spaces: The Mathematical Foundation

### What is a Vector Space?

A vector space is a mathematical structure that provides the foundation for understanding linear regression geometrically. It's a collection of objects (vectors) that can be added together and multiplied by scalars while satisfying certain axioms.

**Key Properties of Vector Spaces**:
1. **Closure under addition**: Adding two vectors gives another vector
2. **Closure under scalar multiplication**: Multiplying a vector by a scalar gives another vector
3. **Associative and commutative properties**: Vector addition behaves like regular addition
4. **Distributive properties**: Scalar multiplication distributes over vector addition
5. **Identity elements**: Zero vector and scalar identity (1)

### Understanding Vectors

**Definition**: A vector is an ordered list of numbers that can be visualized as:
- A point in space
- An arrow from the origin to that point
- A directed line segment

**Notation**: We typically write vectors as column vectors:
```math
\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}
```

**Dimensions**: Vectors can be:
- **2D**: $`\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix}`$ (points in a plane)
- **3D**: $`\mathbf{v} = \begin{pmatrix} x \\ y \\ z \end{pmatrix}`$ (points in space)
- **nD**: $`\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}`$ (points in n-dimensional space)

### Vector Operations

**Vector Addition**:
```math
\mathbf{a} + \mathbf{b} = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} + \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix} = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{pmatrix}
```

**Scalar Multiplication**:
```math
c \mathbf{a} = c \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} = \begin{pmatrix} c a_1 \\ c a_2 \\ \vdots \\ c a_n \end{pmatrix}
```

**Geometric Interpretation**:
- **Addition**: Move from the tip of one vector to the tip of another
- **Scalar multiplication**: Scale the length of a vector by a factor

### Example: Vector Operations in Practice

```math
2 \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix} + 3 \begin{pmatrix} 3 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \\ 0 \end{pmatrix} + \begin{pmatrix} 9 \\ 3 \\ 3 \end{pmatrix} = \begin{pmatrix} 11 \\ 7 \\ 3 \end{pmatrix}
```

**What this means**:
- Scale the first vector by 2: $`\begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix} \rightarrow \begin{pmatrix} 2 \\ 4 \\ 0 \end{pmatrix}`$
- Scale the second vector by 3: $`\begin{pmatrix} 3 \\ 1 \\ 1 \end{pmatrix} \rightarrow \begin{pmatrix} 9 \\ 3 \\ 3 \end{pmatrix}`$
- Add the scaled vectors component-wise

### Python Implementation: Vector Operations

See the complete implementation in [`code/vector_operations.py`](code/vector_operations.py) which demonstrates basic vector operations in 3D space with visualization.

### Linear Subspaces: The Building Blocks

**Definition**: A linear subspace is a subset of a vector space that is closed under vector addition and scalar multiplication.

**Formal Definition**: A subset $`S`$ of $`\mathbb{R}^n`$ is a linear subspace if:

1. **Zero vector**: $`\mathbf{0} \in S`$ (contains the origin)
2. **Closure under addition**: If $`\mathbf{u}, \mathbf{v} \in S`$, then $`\mathbf{u} + \mathbf{v} \in S`$
3. **Closure under scalar multiplication**: If $`\mathbf{u} \in S`$ and $`c`$ is a scalar, then $`c\mathbf{u} \in S``

**Key Properties**:
- Always contains the origin (zero vector)
- Dimension is the number of linearly independent vectors needed to span it
- In $`\mathbb{R}^2`$: subspaces are lines through the origin
- In $`\mathbb{R}^3`$: subspaces can be lines or planes through the origin

### Examples of Linear Subspaces

**1D Subspace (Line)**:
```math
S = \{ c \begin{pmatrix} 1 \\ 2 \end{pmatrix} : c \in \mathbb{R} \}
```

**2D Subspace (Plane)**:
```math
S = \{ c_1 \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} + c_2 \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} : c_1, c_2 \in \mathbb{R} \}
```

![Linear Subspace Examples](img/w2_example_subspace.png)
*Figure: Examples of linear subspaces in regression geometry*

### Column Space: The Heart of Linear Regression

**Definition**: The column space of a matrix $`X`$ is the set of all possible linear combinations of its columns:

```math
C(X) = \{ \mathbf{X} \boldsymbol{\beta} : \boldsymbol{\beta} \in \mathbb{R}^{p+1} \}
```

**Interpretation in Regression**:
- Each column of $`X`$ represents a predictor variable
- The column space contains all possible predicted values
- It's a subspace of $`\mathbb{R}^n`$ (where $`n`$ is the number of observations)

**Example**: For a design matrix with 2 predictors:
```math
X = \begin{pmatrix} 1 & x_{11} & x_{12} \\ 1 & x_{21} & x_{22} \\ 1 & x_{31} & x_{32} \end{pmatrix}
```

The column space is:
```math
C(X) = \{ \beta_0 \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} + \beta_1 \begin{pmatrix} x_{11} \\ x_{21} \\ x_{31} \end{pmatrix} + \beta_2 \begin{pmatrix} x_{12} \\ x_{22} \\ x_{32} \end{pmatrix} : \beta_0, \beta_1, \beta_2 \in \mathbb{R} \}
```

### Python Implementation: Column Space

See the complete implementation in [`code/column_space_demo.py`](code/column_space_demo.py) which demonstrates the concept of column space with 3D visualization and examples of different coefficient vectors.

## 2.2.2. Projection: The Geometric Foundation of Least Squares

### The Projection Problem

The least squares optimization problem can be understood geometrically as finding the projection of the response vector $`\mathbf{y}`$ onto the column space of $`X`$.

**Mathematical Formulation**:
```math
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|^2
```

**Geometric Interpretation**:
- The column space $`C(X)`$ is a subspace of $`\mathbb{R}^n`$
- The vector $`\mathbf{y}`$ may not lie in $`C(X)`$
- The least squares solution finds the point in $`C(X)`$ closest to $`\mathbf{y}`$
- This closest point is the **orthogonal projection** of $`\mathbf{y}`$ onto $`C(X)`$

### Understanding Projection

**What is Projection?**
Projection is the process of finding the closest point in a subspace to a given vector. It's like casting a shadow of a vector onto a plane or line.

**Key Properties**:
1. **Minimal Distance**: The projected point is the closest point in the subspace to the original vector
2. **Orthogonality**: The difference between the original vector and its projection is orthogonal to the subspace
3. **Uniqueness**: The projection is unique (assuming the subspace is well-defined)

### Orthogonal Decomposition

The least squares solution decomposes $`\mathbf{y}`$ into two orthogonal components:

1. **Predicted values**: $`\hat{\mathbf{y}} = \mathbf{X} \hat{\boldsymbol{\beta}}`$ (lies in $`C(X)`$)
2. **Residual vector**: $`\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}`$ (orthogonal to $`C(X)`$)

**Mathematical Properties**:
- **Orthogonality**: $`\hat{\mathbf{y}}^T \mathbf{r} = 0`$
- **Decomposition**: $`\mathbf{y} = \hat{\mathbf{y}} + \mathbf{r}`$
- **Pythagorean Theorem**: $`\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2`$

### Visual Understanding

**2D Example**: Imagine projecting a point onto a line
- The projection is the foot of the perpendicular from the point to the line
- The residual is the perpendicular distance from the point to the line

**3D Example**: Imagine projecting a point onto a plane
- The projection is the foot of the perpendicular from the point to the plane
- The residual is the perpendicular distance from the point to the plane

### Python Implementation: Projection and Orthogonality

See the complete implementation in [`code/projection_analysis.py`](code/projection_analysis.py) which demonstrates projection and orthogonality in linear regression, including 3D visualization and analysis of projection properties.

### The Projection Matrix (Hat Matrix)

**Definition**: The projection matrix $`H`$ is defined as:
```math
H = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
```

**Properties**:
1. **Projection**: $`\hat{\mathbf{y}} = H\mathbf{y}`$
2. **Idempotent**: $`H^2 = H`$
3. **Symmetric**: $`H^T = H`$
4. **Trace**: $`\text{tr}(H) = p+1`$ (number of parameters)

**Interpretation**: The hat matrix "puts a hat" on $`\mathbf{y}`$ to get $`\hat{\mathbf{y}}`$.

### Geometric Intuition

**Why Projection Works**:
- The column space $`C(X)`$ contains all possible linear combinations of predictors
- The response vector $`\mathbf{y}`$ may not lie exactly in this space due to noise
- Projection finds the closest point in the space to $`\mathbf{y}`$
- This closest point gives us the best linear approximation

**Connection to Least Squares**:
- Minimizing $`\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2`$ is equivalent to finding the projection
- The residual vector $`\mathbf{r}`$ is perpendicular to the column space
- This perpendicularity ensures we've found the closest point

## 2.2.3. R²: The Coefficient of Determination

### What is R²?

$`R^2`$ (R-squared) is a fundamental measure of model fit that quantifies the proportion of variance in the response variable explained by the predictors. It's one of the most widely used metrics in regression analysis.

### Mathematical Definition

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

- **Total Sum of Squares (TSS)**: $`\| \mathbf{y} - \bar{\mathbf{y}} \|^2`$
- **Explained Sum of Squares (ESS)**: $`\| \hat{\mathbf{y}} - \bar{\mathbf{y}} \|^2`$
- **Residual Sum of Squares (RSS)**: $`\| \mathbf{r} \|^2`$

### Alternative Expressions

```math
R^2 = \frac{\text{ESS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}
```

### Key Properties

1. **Range**: $`0 \leq R^2 \leq 1`$
2. **Perfect Fit**: $`R^2 = 1`$ means all residuals are zero
3. **No Improvement**: $`R^2 = 0`$ means the model performs no better than predicting the mean
4. **Correlation**: In multiple regression, $`R^2`$ is the squared correlation between $`y`$ and $`\hat{y}`$
5. **Simple Regression**: In simple regression, $`R^2`$ is the squared correlation between $`y`$ and $`x`$

### Understanding R² Geometrically

**Visual Interpretation**:
- Imagine the response vector $`\mathbf{y}`$ centered at the mean
- The fitted values $`\hat{\mathbf{y}}`$ are the projection onto the column space
- $`R^2`$ measures how much of the total variation is "explained" by the projection
- It's the ratio of the squared length of the projection to the squared length of the original vector

**Example**: If $`R^2 = 0.8`$, then 80% of the variance in $`y`$ is explained by the linear model.

### Python Implementation: R² Analysis

See the complete implementation in [`code/r_squared_analysis.py`](code/r_squared_analysis.py) which provides comprehensive analysis of R-squared including geometric interpretation, variance decomposition, and visualization.

### Invariance Properties

$`R^2`$ has several important invariance properties:

1. **Location Invariance**: Adding a constant to $`y`$ does not change $`R^2`$
2. **Scale Invariance**: Multiplying $`y`$ by a constant does not change $`R^2`$
3. **Symmetry in Simple Regression**: $`R^2`$ is the same whether we predict $`Y`$ from $`X`$ or $`X`$ from $`Y``

### Interpretation and Limitations

**Interpretation**:
- **High $`R^2`$** (e.g., 0.7 or 0.8): Suggests a good fit, but doesn't guarantee model validity
- **Low $`R^2`$**: Doesn't necessarily mean the model is useless; it may still provide useful predictions
- **Context Matters**: What constitutes a "good" $`R^2`$ depends on the field and application

**Limitations**:
1. **Overfitting**: Adding more predictors (even irrelevant ones) can artificially increase $`R^2`$
2. **No Penalty**: $`R^2`$ doesn't account for the number of predictors
3. **Non-linear Relationships**: $`R^2`$ only measures linear relationships
4. **Outliers**: Can be sensitive to outliers

### Adjusted R²

To address the limitation of $`R^2`$ increasing with more predictors, we use adjusted $`R^2`$:

```math
R^2_{\text{adj}} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)} = 1 - (1 - R^2) \frac{n-1}{n-p-1}
```

**Properties**:
- Penalizes models with many predictors
- Can decrease when adding irrelevant variables
- More appropriate for model comparison
- Accounts for degrees of freedom

### Python Implementation: Adjusted R²

The adjusted R-squared computation is included in [`code/r_squared_analysis.py`](code/r_squared_analysis.py) as part of the comprehensive R-squared analysis.

This geometric understanding of $`R^2`$ provides a solid foundation for interpreting model performance and understanding the relationship between observed and predicted values in linear regression.

## 2.2.4. Linear Transformations of X: Understanding Invariance

Linear transformations of the design matrix $`X`$ have important implications for the least squares solution. Understanding these transformations helps us interpret results and handle data preprocessing.

### What are Linear Transformations?

A linear transformation of $`X`$ involves multiplying $`X`$ by a matrix $`A`$:
```math
X' = XA
```

where $`A`$ is a $`(p+1) \times (p+1)`$ transformation matrix.

### Effect on the Fit

**Key Result**: If we transform $`X`$ to $`X' = XA`$ where $`A`$ is a full-rank matrix, then:

- The column space $`C(X') = C(X)`$ remains the same
- The fitted values $`\hat{\mathbf{y}}`$ are unchanged
- The residuals $`\mathbf{r}`$ are unchanged
- $`R^2`$ is unchanged
- However, the coefficients $`\boldsymbol{\beta}`$ will change

**Mathematical Justification**:
```math
\hat{\mathbf{y}}' = X' \hat{\boldsymbol{\beta}}' = XA \hat{\boldsymbol{\beta}}' = X \hat{\boldsymbol{\beta}} = \hat{\mathbf{y}}
```

This means $`A \hat{\boldsymbol{\beta}}' = \hat{\boldsymbol{\beta}}`$, so $`\hat{\boldsymbol{\beta}}' = A^{-1} \hat{\boldsymbol{\beta}}`$.

### Common Linear Transformations

**1. Scaling Predictors**:
```math
X' = X \begin{pmatrix} 1 & 0 & 0 \\ 0 & c & 0 \\ 0 & 0 & 1 \end{pmatrix}
```

This scales the second predictor by a factor $`c`$.

**2. Centering Predictors**:
```math
X' = X \begin{pmatrix} 1 & -\bar{x}_1 & -\bar{x}_2 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
```

This centers the predictors around their means.

**3. Standardization**:
```math
X' = X \begin{pmatrix} 1 & -\bar{x}_1/s_1 & -\bar{x}_2/s_2 \\ 0 & 1/s_1 & 0 \\ 0 & 0 & 1/s_2 \end{pmatrix}
```

This standardizes the predictors to have mean 0 and standard deviation 1.

### Example: Scaling Predictors

**Original Model**:
```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2
```

**Scaled Model**:
```math
y = \beta_0' + \beta_1' (c x_1) + \beta_2' x_2
```

**Relationship**:
```math
\beta_1' = \beta_1 / c
```

### Python Implementation: Linear Transformations

See the complete implementation in [`code/linear_transformations.py`](code/linear_transformations.py) which demonstrates the effect of different linear transformations (scaling, centering, standardization) on regression coefficients while preserving fitted values.

## 2.2.5. Rank Deficiency: When Things Go Wrong

Rank deficiency occurs when the design matrix $`X`$ does not have full column rank, meaning some columns are linear combinations of others. This is a critical issue in linear regression that affects the uniqueness and interpretation of solutions.

### What is Rank Deficiency?

**Definition**: $`X`$ is rank deficient if its rank is less than $`p+1`$ (the number of columns).

**Mathematical Condition**:
```math
\text{rank}(X) < p + 1
```

This means $`X^T X`$ is not invertible, and the normal equation has infinitely many solutions.

### Common Causes of Rank Deficiency

**1. Perfect Collinearity**:
Two predictors are perfectly correlated:
```python
# Example: Temperature in Celsius and Fahrenheit
temp_c = np.array([0, 10, 20, 30])
temp_f = 9/5 * temp_c + 32  # Perfect linear relationship
X = np.column_stack([np.ones(4), temp_c, temp_f])
```

**2. Redundant Variables**:
A predictor is a linear combination of others:
```python
# Example: Sum to constant
age_young = np.array([30, 25, 40])
age_middle = np.array([45, 50, 35])
age_old = 100 - age_young - age_middle  # Perfect linear combination
X = np.column_stack([np.ones(3), age_young, age_middle, age_old])
```

**3. Categorical Variables**:
Including all levels of a categorical variable with an intercept:
```python
# Example: One-hot encoding with all levels
category_A = np.array([1, 0, 0, 1])
category_B = np.array([0, 1, 0, 0])
category_C = np.array([0, 0, 1, 0])
# category_D = 1 - category_A - category_B - category_C (perfect collinearity)
X = np.column_stack([np.ones(4), category_A, category_B, category_C])
```

### Consequences of Rank Deficiency

**1. Non-unique Solutions**:
- $`(X^T X)^{-1}`$ does not exist
- There are infinitely many $`\boldsymbol{\beta}`$ that give the same fitted values
- The normal equation has multiple solutions

**2. Software Behavior**:
Different software packages handle rank deficiency differently:
- **R's `lm()`**: Drops redundant columns and marks their coefficients as `NA`
- **Python's scikit-learn**: Returns the minimum-norm solution using the Moore-Penrose pseudoinverse
- **NumPy's `np.linalg.lstsq()`**: Also uses the pseudoinverse

**3. Interpretation Problems**:
- Individual coefficients may not be interpretable
- Standard errors may be infinite or very large
- Confidence intervals may be meaningless

### Python Implementation: Rank Deficiency Analysis

See the complete implementation in [`code/rank_deficiency.py`](code/rank_deficiency.py) which demonstrates rank deficiency detection and handling, including examples of perfect collinearity and redundant variables.

### Handling Rank Deficiency

**1. Remove Redundant Variables**:
- Identify and remove perfectly collinear predictors
- Use stepwise selection or regularization
- Consider the scientific meaning of the variables

**2. Regularization**:
- Ridge regression: $`\hat{\boldsymbol{\beta}}_{ridge} = (X^T X + \lambda I)^{-1} X^T y`$
- Lasso regression: Adds L1 penalty
- Elastic net: Combines L1 and L2 penalties

**3. Principal Component Analysis (PCA)**:
- Transform to orthogonal components
- Use only the first few principal components
- Maintains most of the variance while eliminating collinearity

**4. Data Collection**:
- Collect more diverse data
- Ensure predictors are not perfectly correlated
- Consider the experimental design

### Best Practices

**1. Always Check Rank**:
```python
rank = np.linalg.matrix_rank(X)
if rank < X.shape[1]:
    print("Warning: Rank deficiency detected")
```

**2. Monitor Condition Number**:
```python
eigenvals = np.linalg.eigvals(X.T @ X)
condition_number = np.max(eigenvals) / np.min(eigenvals[eigenvals > 1e-10])
if condition_number > 1e12:
    print("Warning: High condition number")
```

**3. Use Regularization**:
When rank deficiency is detected, consider using regularized methods that provide stable solutions.

**4. Interpret Results Carefully**:
- Individual coefficients may not be meaningful
- Focus on overall model performance
- Consider the scientific context

## 2.2.6. Advanced Geometric Concepts

### The Hat Matrix and Leverage

**Hat Matrix Properties**:
The projection matrix $`H = X(X^T X)^{-1} X^T`$ has several important properties:

1. **Projection**: $`\hat{\mathbf{y}} = H\mathbf{y}`$
2. **Idempotent**: $`H^2 = H`$
3. **Symmetric**: $`H^T = H`$
4. **Trace**: $`\text{tr}(H) = p+1`$

**Leverage**:
The diagonal elements $`h_{ii}`$ of the hat matrix are called leverage values:
```math
h_{ii} = \mathbf{x}_i^T (X^T X)^{-1} \mathbf{x}_i
```

**Interpretation**:
- $`h_{ii}`$ measures the influence of observation $`i`$ on its own fitted value
- High leverage points are potentially influential
- Rule of thumb: $`h_{ii} > 2(p+1)/n`$ indicates high leverage

### Cook's Distance

Cook's distance measures the influence of each observation on the entire regression:

```math
D_i = \frac{(\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})^T X^T X (\hat{\boldsymbol{\beta}} - \hat{\boldsymbol{\beta}}_{(i)})}{(p+1) \hat{\sigma}^2}
```

where $`\hat{\boldsymbol{\beta}}_{(i)}`$ is the estimate with observation $`i`$ removed.

### Python Implementation: Advanced Diagnostics

See the complete implementation in [`code/advanced_diagnostics.py`](code/advanced_diagnostics.py) which provides comprehensive diagnostic measures including leverage, studentized residuals, Cook's distance, and visualization plots.

## 2.2.7. Summary and Key Insights

### What We've Learned

The geometric interpretation of linear regression provides deep insights into:

1. **Vector Spaces**: The mathematical foundation for understanding regression
2. **Projection**: The geometric basis of least squares estimation
3. **R-squared**: The proportion of variance explained by the model
4. **Linear Transformations**: How data preprocessing affects results
5. **Rank Deficiency**: When and why problems occur
6. **Diagnostics**: How to assess model quality and identify influential observations

### Key Geometric Insights

**1. Projection is Optimal**:
- Least squares finds the orthogonal projection of $`\mathbf{y}`$ onto $`C(X)`$
- This projection minimizes the Euclidean distance
- The residual vector is orthogonal to the column space

**2. R-squared is Geometric**:
- $`R^2`$ measures the ratio of explained to total variation
- It's the squared cosine of the angle between centered vectors
- Perfect fit means $`\mathbf{y}`$ lies in the column space

**3. Invariance Under Transformations**:
- Linear transformations preserve the column space
- Fitted values and residuals are unchanged
- Only coefficient interpretations change

**4. Rank Deficiency is Geometric**:
- Occurs when columns are linearly dependent
- The column space has lower dimension than expected
- Solutions exist but are not unique

### Practical Applications

**1. Model Diagnostics**:
- Use leverage to identify influential points
- Use Cook's distance to assess overall influence
- Use studentized residuals to detect outliers

**2. Data Preprocessing**:
- Centering affects intercept interpretation
- Scaling affects coefficient magnitudes
- Standardization makes coefficients comparable

**3. Model Selection**:
- R-squared helps assess fit quality
- Adjusted R-squared penalizes complexity
- Cross-validation provides out-of-sample assessment

### Advanced Topics

This geometric foundation prepares us for:

1. **Generalized Linear Models**: Extending beyond normal errors
2. **Regularization**: Ridge, Lasso, and Elastic Net
3. **Non-linear Methods**: Kernel methods and splines
4. **Multivariate Analysis**: Principal components and factor analysis
5. **Time Series**: Autocorrelation and stationarity

### Code Summary

Throughout this document, we've implemented comprehensive Python code examples:

- **Vector operations** and visualization: [`code/vector_operations.py`](code/vector_operations.py)
- **Column space demonstration**: [`code/column_space_demo.py`](code/column_space_demo.py)
- **Projection analysis** with orthogonality checks: [`code/projection_analysis.py`](code/projection_analysis.py)
- **R-squared computation** and interpretation: [`code/r_squared_analysis.py`](code/r_squared_analysis.py)
- **Linear transformation** effects: [`code/linear_transformations.py`](code/linear_transformations.py)
- **Rank deficiency** detection and handling: [`code/rank_deficiency.py`](code/rank_deficiency.py)
- **Advanced diagnostics** including leverage and influence measures: [`code/advanced_diagnostics.py`](code/advanced_diagnostics.py)

This comprehensive geometric understanding provides the foundation for mastering linear regression and understanding more advanced statistical learning methods.

---

**Navigation:**
- **Next Topic:** [Practical Issues](03_practical_issues.md) - Real-world implementation considerations and best practices
- **Previous Topic:** [Multiple Linear Regression](01_mulitple_linear_regression.md) - Core concepts and mathematical foundations
