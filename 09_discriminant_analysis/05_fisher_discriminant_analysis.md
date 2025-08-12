# 9.5. Fisher Discriminant Analysis

## 9.5.0. Introduction and Motivation

Fisher Discriminant Analysis (FDA), also known as Fisher's Linear Discriminant Analysis, is a fundamental supervised dimensionality reduction technique that finds optimal projection directions to maximize class separation. Unlike unsupervised methods like PCA, FDA leverages class label information to find directions that are most discriminative for classification.

### Key Concepts

**Supervised vs Unsupervised Dimensionality Reduction**:
- **Unsupervised** (e.g., PCA): Uses only feature data $`X`$, ignores labels $`Y`$
- **Supervised** (e.g., FDA): Uses both features $`X`$ and labels $`Y`$ to find discriminative directions

### Fisher's Intuition

Fisher's key insight was to find a projection direction that:
1. **Maximizes** the separation between class means (between-class variance)
2. **Minimizes** the spread within each class (within-class variance)

This leads to the famous **Fisher criterion**:

```math
J(\mathbf{a}) = \frac{\text{Between-class variance}}{\text{Within-class variance}} = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}}
```

Where $`\mathbf{a}`$ is the projection direction we seek to find.

## 9.5.1. Mathematical Foundation

### The Fisher Criterion

Let's formalize Fisher's objective. Given a projection direction $`\mathbf{a} \in \mathbb{R}^p`$, we want to maximize:

```math
J(\mathbf{a}) = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}}
```

### Between-Class Scatter Matrix ($`\mathbf{B}`$)

The between-class scatter matrix measures how far apart the class means are:

```math
\mathbf{B} = \frac{1}{K-1} \sum_{k=1}^K n_k (\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})^T
```

Where:
- $`\boldsymbol{\mu}_k`$ is the mean of class $`k`$
- $`\bar{\boldsymbol{\mu}} = \frac{1}{n} \sum_{k=1}^K n_k \boldsymbol{\mu}_k`$ is the overall mean
- $`n_k`$ is the number of samples in class $`k`$
- $`K`$ is the number of classes

**Intuition**: $`\mathbf{B}`$ captures the variance of class centers around the overall mean.

### Within-Class Scatter Matrix ($`\mathbf{W}`$)

The within-class scatter matrix measures the spread within each class:

```math
\mathbf{W} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
```

**Intuition**: $`\mathbf{W}`$ is essentially the pooled covariance matrix, measuring how tightly points cluster around their class means.

### Geometric Interpretation

Every data point $`\mathbf{x}_i`$ can be decomposed as:

```math
\mathbf{x}_i = \underbrace{(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})}_{\text{Within-class deviation}} + \underbrace{\boldsymbol{\mu}_{y_i}}_{\text{Class center}}
```

Where:
- $`(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})`$ represents the deviation from the class mean (captured by $`\mathbf{W}`$)
- $`\boldsymbol{\mu}_{y_i}`$ represents the class center (captured by $`\mathbf{B}`$)

## 9.5.2. The Generalized Eigenvalue Problem

### Optimization Formulation

Maximizing the Fisher criterion leads to a **generalized eigenvalue problem**:

```math
\mathbf{B} \mathbf{a} = \lambda \mathbf{W} \mathbf{a}
```

This can be rewritten as:

```math
\mathbf{W}^{-1} \mathbf{B} \mathbf{a} = \lambda \mathbf{a}
```

### Solution Properties

1. **Number of Directions**: We can find at most $`K-1`$ non-zero eigenvalues because $`\text{rank}(\mathbf{B}) \leq K-1`$

2. **Eigenvalue Interpretation**: The eigenvalues $`\lambda_i`$ represent the ratio of between-class to within-class variance along each direction

3. **Optimal Directions**: The eigenvectors $`\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_{K-1}`$ are the optimal projection directions

### Mathematical Derivation

To find the maximum of $`J(\mathbf{a})`$, we set the gradient to zero:

```math
\nabla_{\mathbf{a}} J(\mathbf{a}) = \frac{2\mathbf{B}\mathbf{a}(\mathbf{a}^T\mathbf{W}\mathbf{a}) - 2\mathbf{W}\mathbf{a}(\mathbf{a}^T\mathbf{B}\mathbf{a})}{(\mathbf{a}^T\mathbf{W}\mathbf{a})^2} = 0
```

This simplifies to:

```math
\mathbf{B}\mathbf{a} = \frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} \mathbf{W}\mathbf{a}
```

Recognizing that $`\frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} = J(\mathbf{a})`$ is the eigenvalue $`\lambda`$, we get:

```math
\mathbf{B}\mathbf{a} = \lambda \mathbf{W}\mathbf{a}
```

## 9.5.3. Connection to Linear Discriminant Analysis

### Equivalence Under Normality Assumptions

When we assume:
1. Classes follow multivariate normal distributions
2. All classes share the same covariance matrix $`\boldsymbol{\Sigma}`$

Then FDA and LDA produce **equivalent subspaces**:

```math
\mathbf{W} \approx \boldsymbol{\Sigma} \quad \text{and} \quad \mathbf{B} \approx \boldsymbol{\Sigma}_B
```

Where $`\boldsymbol{\Sigma}_B`$ is the between-class covariance matrix in LDA.

### Key Differences

| Aspect | FDA | LDA |
|--------|-----|-----|
| **Assumptions** | No distributional assumptions | Multivariate normal, equal covariance |
| **Objective** | Maximize class separation | Minimize classification error |
| **Output** | Projection directions | Classification rule |
| **Flexibility** | More general | More restrictive |

### Practical Implementation

In practice, FDA directions can be extracted from LDA:

```python
# FDA directions from LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
fda_directions = lda.scalings_  # These are the FDA directions
```

## 9.5.4. Supervised Dimension Reduction

### Why Supervised?

FDA is **supervised** because it uses class labels $`Y`$ to find discriminative directions. This is fundamentally different from PCA:

- **PCA**: Directions maximize variance regardless of class labels
- **FDA**: Directions maximize class separation

### Example: Toy Data Visualization

Consider a 2D dataset with 3 classes. The complete implementation is provided in the code files:

**Python:** See `compare_pca_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `compare_pca_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

These functions demonstrate the key difference between PCA and FDA:
- **PCA** finds directions that maximize variance regardless of class labels
- **FDA** finds directions that maximize class separation by considering both between-class and within-class variance

The visualization shows how FDA achieves much better class separation than PCA when class information is available.

### Extension to Regression

FDA can be extended to regression problems by discretizing the continuous response. The implementation is provided in the code files:

**Python:** See `fda_for_regression()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

This approach discretizes the continuous response variable into bins and then applies FDA to find discriminative directions for the discretized classes.

## 9.5.5. Implementation from Scratch

The complete implementation of Fisher Discriminant Analysis from scratch is provided in the following code files:

**Python Implementation:** [`code/fda_implementation.py`](code/fda_implementation.py)

**R Implementation:** [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

These files contain:

- Complete `FisherDiscriminantAnalysis` class with parameter estimation
- Between-class and within-class scatter matrix calculations
- Generalized eigenvalue problem solution
- Regularization handling for singular matrices
- Comparison with library implementations (sklearn LDA, MASS LDA)
- Visualization functions for projections and discriminant directions
- Separation criterion calculations
- Comprehensive demonstration functions

The implementation solves the generalized eigenvalue problem $`\mathbf{B} \mathbf{a} = \lambda \mathbf{W} \mathbf{a}`$ to find optimal projection directions that maximize class separation.

## 9.5.6. Risk of Overfitting

### The Overfitting Problem

When $`p \gg n`$ (high-dimensional data with few samples), FDA can overfit severely. This happens because:

1. **Perfect Separation**: With $`p \geq n`$, we can always find directions that perfectly separate classes
2. **Random Features**: Even random noise can appear discriminative in high dimensions
3. **Limited Degrees of Freedom**: The within-class scatter matrix becomes singular

### Example: Overfitting Demonstration

The overfitting demonstration is implemented in the code files:

**Python:** See `demonstrate_overfitting()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `demonstrate_overfitting()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example shows how FDA can achieve perfect separation even with random features when the number of features greatly exceeds the number of samples, demonstrating the overfitting problem in high-dimensional settings.

### Mitigation Strategies

#### 1. Regularization

The regularized FDA implementation is provided in the code files:

**Python:** See `regularized_fda()` and `calculate_scatter_matrices()` functions in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `regularized_fda()` and `calculate_scatter_matrices()` functions in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach adds a regularization term to the within-class scatter matrix to prevent singularity and improve numerical stability.

#### 2. Feature Selection

The FDA with feature selection implementation is provided in the code files:

**Python:** See `fda_with_feature_selection()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `fda_with_feature_selection()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach uses statistical tests (F-test) to select the most discriminative features before applying FDA.

#### 3. Cross-Validation

The cross-validation implementation is provided in the code files:

**Python:** See `cross_validate_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `cross_validate_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach uses stratified k-fold cross-validation to assess the generalization performance of FDA projections.

## 9.5.7. Real-World Applications

### Example 1: Face Recognition

The face recognition example is implemented in the code files:

**Python:** See `face_recognition_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `face_recognition_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example demonstrates FDA for face recognition using the Olivetti faces dataset, showing how FDA can reduce high-dimensional face data to discriminative components while maintaining classification accuracy.

### Example 2: Gene Expression Analysis

The gene expression analysis example is implemented in the code files:

**Python:** See `gene_expression_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `gene_expression_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example shows FDA applied to high-dimensional gene expression data, demonstrating feature selection and dimensionality reduction for biological data analysis.

## 9.5.8. Summary and Best Practices

### Key Takeaways

1. **FDA Objective**: Maximize between-class variance while minimizing within-class variance
2. **Supervised Nature**: Uses class labels to find discriminative directions
3. **Dimensionality Reduction**: Naturally reduces to $`K-1`$ dimensions
4. **Connection to LDA**: Equivalent under normality assumptions

### Best Practices

1. **Data Preprocessing**:
   - Standardize features
   - Handle missing values
   - Check for multicollinearity

2. **Dimensionality Management**:
   - Use regularization when $`p \gg n`$
   - Apply feature selection
   - Cross-validate results

3. **Model Validation**:
   - Check for overfitting
   - Use cross-validation
   - Monitor separation metrics

4. **Interpretation**:
   - Examine discriminant directions
   - Analyze explained variance ratios
   - Visualize projections

### When to Use FDA

**Use FDA when**:
- You need supervised dimensionality reduction
- Classes are well-separated
- Interpretability is important
- You want to reduce to $`K-1`$ dimensions

**Consider alternatives when**:
- Classes overlap significantly (use other methods)
- You need more than $`K-1`$ dimensions
- Data is non-linear (use kernel methods)

### Limitations

1. **Linear Assumption**: Only finds linear projections
2. **Overfitting Risk**: Can overfit in high dimensions
3. **Normality Assumption**: Implicit in the formulation
4. **Limited Dimensions**: Maximum $`K-1`$ components

Fisher Discriminant Analysis remains a powerful and interpretable method for supervised dimensionality reduction, providing a solid foundation for understanding the relationship between classes in high-dimensional data.
