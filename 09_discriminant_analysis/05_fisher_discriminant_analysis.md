# 9.5. Fisher Discriminant Analysis

## 9.5.0. Introduction and Motivation

Fisher Discriminant Analysis (FDA), also known as Fisher's Linear Discriminant Analysis, is a fundamental supervised dimensionality reduction technique that finds optimal projection directions to maximize class separation. Unlike unsupervised methods like PCA, FDA leverages class label information to find directions that are most discriminative for classification.

**Intuitive Understanding**: FDA is like being a detective who wants to find the best angle to view a crime scene so that different types of evidence (classes) are most clearly separated. Imagine you're trying to organize a library where books from different genres are mixed together. Instead of just looking at the books randomly (like PCA), FDA uses the genre labels to find the best way to arrange the books so that each genre is clearly separated from the others. It's like finding the perfect viewing angle that makes different groups stand out most clearly.

### Key Concepts

**Supervised vs Unsupervised Dimensionality Reduction**:
- **Unsupervised** (e.g., PCA): Uses only feature data $`X`$, ignores labels $`Y`$ - like organizing books by size or color without knowing their genres
- **Supervised** (e.g., FDA): Uses both features $`X`$ and labels $`Y`$ to find discriminative directions - like organizing books by genre to make each genre clearly distinct

**Intuition**: The key difference is that FDA uses the "answers" (class labels) to guide the dimensionality reduction, while PCA just looks at the data structure without knowing what the classes are. This makes FDA much more effective for classification tasks.

### Fisher's Intuition

Fisher's key insight was to find a projection direction that:
1. **Maximizes** the separation between class means (between-class variance) - like making sure different genres are far apart from each other
2. **Minimizes** the spread within each class (within-class variance) - like making sure books within the same genre are close together

This leads to the famous **Fisher criterion**:

$$ J(\mathbf{a}) = \frac{\text{Between-class variance}}{\text{Within-class variance}} = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}} $$

Where $`\mathbf{a}`$ is the projection direction we seek to find.

**Intuition**: This ratio is like asking "how well separated are the different groups compared to how spread out each group is internally?" A high ratio means the groups are well-separated and tightly clustered, which is exactly what we want for classification.

## 9.5.1. Mathematical Foundation

### The Fisher Criterion

Let's formalize Fisher's objective. Given a projection direction $`\mathbf{a} \in \mathbb{R}^p`$, we want to maximize:

$$ J(\mathbf{a}) = \frac{\mathbf{a}^T \mathbf{B} \mathbf{a}}{\mathbf{a}^T \mathbf{W} \mathbf{a}} $$

**Intuition**: This formula measures how good a projection direction is for separating classes. The numerator measures how far apart the class centers are when projected, while the denominator measures how spread out each class is when projected. We want to maximize this ratio to get the best separation.

### Between-Class Scatter Matrix ($`\mathbf{B}`$)

The between-class scatter matrix measures how far apart the class means are:

$$ \mathbf{B} = \frac{1}{K-1} \sum_{k=1}^K n_k (\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_k - \bar{\boldsymbol{\mu}})^T $$

Where:
- $`\boldsymbol{\mu}_k`$ is the mean of class $`k`$ - like the typical profile for each disease
- $`\bar{\boldsymbol{\mu}} = \frac{1}{n} \sum_{k=1}^K n_k \boldsymbol{\mu}_k`$ is the overall mean - like the average profile across all diseases
- $`n_k`$ is the number of samples in class $`k`$ - like how many patients have each disease
- $`K`$ is the number of classes - like how many different diseases we're studying

**Intuition**: $`\mathbf{B}`$ captures the variance of class centers around the overall mean. It's like measuring how different the typical symptoms are for each disease compared to the average symptoms across all diseases. A large $`\mathbf{B}`$ means the diseases are quite different from each other.

### Within-Class Scatter Matrix ($`\mathbf{W}`$)

The within-class scatter matrix measures the spread within each class:

$$ \mathbf{W} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i=k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T $$

**Intuition**: $`\mathbf{W}`$ is essentially the pooled covariance matrix, measuring how tightly points cluster around their class means. It's like measuring how variable the symptoms are within each disease. A small $`\mathbf{W}`$ means patients with the same disease have similar symptoms.

### Geometric Interpretation

Every data point $`\mathbf{x}_i`$ can be decomposed as:

$$ \mathbf{x}_i = \underbrace{(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})}_{\text{Within-class deviation}} + \underbrace{\boldsymbol{\mu}_{y_i}}_{\text{Class center}} $$

Where:
- $`(\mathbf{x}_i - \boldsymbol{\mu}_{y_i})`$ represents the deviation from the class mean (captured by $`\mathbf{W}`$) - like how different this patient's symptoms are from the typical symptoms for their disease
- $`\boldsymbol{\mu}_{y_i}`$ represents the class center (captured by $`\mathbf{B}`$) - like the typical symptoms for this patient's disease

**Intuition**: This decomposition shows that every patient's symptoms can be broken down into two parts: how typical they are for their disease (the class center), and how unusual they are within their disease (the deviation). FDA tries to find directions that emphasize the differences between diseases while minimizing the variations within each disease.

## 9.5.2. The Generalized Eigenvalue Problem

### Optimization Formulation

Maximizing the Fisher criterion leads to a **generalized eigenvalue problem**:

$$ \mathbf{B} \mathbf{a} = \lambda \mathbf{W} \mathbf{a} $$

This can be rewritten as:

$$ \mathbf{W}^{-1} \mathbf{B} \mathbf{a} = \lambda \mathbf{a} $$

**Intuition**: This equation is asking "what direction $`\mathbf{a}`$ makes the between-class separation (B) large relative to the within-class spread (W)?" The eigenvalue $`\lambda`$ tells us how good this direction is - larger eigenvalues mean better separation.

### Solution Properties

1. **Number of Directions**: We can find at most $`K-1`$ non-zero eigenvalues because $`\text{rank}(\mathbf{B}) \leq K-1`$ - like only needing K-1 different angles to separate K groups

2. **Eigenvalue Interpretation**: The eigenvalues $`\lambda_i`$ represent the ratio of between-class to within-class variance along each direction - like measuring how good each viewing angle is for separating the groups

3. **Optimal Directions**: The eigenvectors $`\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_{K-1}`$ are the optimal projection directions - like the best angles to view the data for maximum separation

**Intuition**: These properties tell us that FDA naturally reduces the dimensionality to K-1 dimensions, which is exactly what we need to separate K classes. Each direction provides a different "view" of the data that helps separate the classes.

### Mathematical Derivation

To find the maximum of $`J(\mathbf{a})`$, we set the gradient to zero:

$$ \nabla_{\mathbf{a}} J(\mathbf{a}) = \frac{2\mathbf{B}\mathbf{a}(\mathbf{a}^T\mathbf{W}\mathbf{a}) - 2\mathbf{W}\mathbf{a}(\mathbf{a}^T\mathbf{B}\mathbf{a})}{(\mathbf{a}^T\mathbf{W}\mathbf{a})^2} = 0 $$

This simplifies to:

$$ \mathbf{B}\mathbf{a} = \frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} \mathbf{W}\mathbf{a} $$

Recognizing that $`\frac{\mathbf{a}^T\mathbf{B}\mathbf{a}}{\mathbf{a}^T\mathbf{W}\mathbf{a}} = J(\mathbf{a})`$ is the eigenvalue $`\lambda`$, we get:

$$ \mathbf{B}\mathbf{a} = \lambda \mathbf{W}\mathbf{a} $$

**Intuition**: This derivation shows that the optimal direction is one where the between-class separation (B) and within-class spread (W) are perfectly balanced according to the eigenvalue $`\lambda`$. The eigenvalue tells us exactly how good this balance is.

## 9.5.3. Connection to Linear Discriminant Analysis

### Equivalence Under Normality Assumptions

When we assume:
1. Classes follow multivariate normal distributions - like diseases having bell-shaped symptom distributions
2. All classes share the same covariance matrix $`\boldsymbol{\Sigma}`$ - like all diseases having similar symptom relationship patterns

Then FDA and LDA produce **equivalent subspaces**:

$$ \mathbf{W} \approx \boldsymbol{\Sigma} \quad \text{and} \quad \mathbf{B} \approx \boldsymbol{\Sigma}_B $$

Where $`\boldsymbol{\Sigma}_B`$ is the between-class covariance matrix in LDA.

**Intuition**: This means that when our data follows certain statistical assumptions (Gaussian distributions with shared covariance), FDA and LDA give us the same optimal directions. This is like saying that the best angles to view the data for separation (FDA) are the same as the best angles for classification (LDA).

### Key Differences

| Aspect | FDA | LDA |
|--------|-----|-----|
| **Assumptions** | No distributional assumptions - like working with any type of data | Multivariate normal, equal covariance - like assuming specific data patterns |
| **Objective** | Maximize class separation - like finding the best viewing angles | Minimize classification error - like finding the best decision rules |
| **Output** | Projection directions - like the best angles to view the data | Classification rule - like the decision boundary |
| **Flexibility** | More general - like working with any data type | More restrictive - like requiring specific data patterns |

**Intuition**: FDA is more flexible because it doesn't make strong assumptions about the data distribution, while LDA is more specific but provides both dimensionality reduction and classification. FDA is like finding the best viewing angles for any type of data, while LDA is like finding both the best viewing angles and the best decision rules for normally distributed data.

### Practical Implementation

In practice, FDA directions can be extracted from LDA:

```python
# FDA directions from LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
fda_directions = lda.scalings_  # These are the FDA directions
```

**Intuition**: This practical connection means that if we have normally distributed data, we can use LDA to get FDA directions. It's like using a more sophisticated tool (LDA) to get the same result as a simpler tool (FDA) when the data meets the right conditions.

## 9.5.4. Supervised Dimension Reduction

### Why Supervised?

FDA is **supervised** because it uses class labels $`Y`$ to find discriminative directions. This is fundamentally different from PCA:

- **PCA**: Directions maximize variance regardless of class labels - like organizing books by size without caring about genre
- **FDA**: Directions maximize class separation - like organizing books by genre to make each genre clearly distinct

**Intuition**: The key insight is that FDA uses the "answers" (class labels) to guide the dimensionality reduction. This is like having a teacher who tells you which books belong to which genres, allowing you to organize them much more effectively than if you just looked at the books' physical properties.

### Example: Toy Data Visualization

Consider a 2D dataset with 3 classes. The complete implementation is provided in the code files:

**Python:** See `compare_pca_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `compare_pca_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

These functions demonstrate the key difference between PCA and FDA:
- **PCA** finds directions that maximize variance regardless of class labels - like finding the direction where the data spreads out the most
- **FDA** finds directions that maximize class separation by considering both between-class and within-class variance - like finding the direction where different classes are most clearly separated

The visualization shows how FDA achieves much better class separation than PCA when class information is available.

**Intuition**: This comparison shows that using class labels (supervised learning) can dramatically improve the quality of dimensionality reduction for classification tasks. It's like the difference between organizing books randomly versus organizing them by genre.

### Extension to Regression

FDA can be extended to regression problems by discretizing the continuous response. The implementation is provided in the code files:

**Python:** See `fda_for_regression()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

This approach discretizes the continuous response variable into bins and then applies FDA to find discriminative directions for the discretized classes.

**Intuition**: This extension allows FDA to work with continuous outcomes by turning them into discrete categories. It's like converting a continuous temperature scale into discrete categories (cold, warm, hot) and then finding the best directions to separate these temperature categories.

## 9.5.5. Implementation from Scratch

The complete implementation of Fisher Discriminant Analysis from scratch is provided in the following code files:

**Python Implementation:** [`code/fda_implementation.py`](code/fda_implementation.py)

**R Implementation:** [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

These files contain:

- Complete `FisherDiscriminantAnalysis` class with parameter estimation - like building a complete FDA system from scratch
- Between-class and within-class scatter matrix calculations - like computing how well-separated and how tightly clustered the classes are
- Generalized eigenvalue problem solution - like finding the optimal viewing angles
- Regularization handling for singular matrices - like adding stability when the data is problematic
- Comparison with library implementations (sklearn LDA, MASS LDA) - like comparing our system with standard tools
- Visualization functions for projections and discriminant directions - like showing the optimal viewing angles
- Separation criterion calculations - like measuring how good the separation is
- Comprehensive demonstration functions - like showing how FDA works on real data

The implementation solves the generalized eigenvalue problem $`\mathbf{B} \mathbf{a} = \lambda \mathbf{W} \mathbf{a}`$ to find optimal projection directions that maximize class separation.

## 9.5.6. Risk of Overfitting

### The Overfitting Problem

When $`p \gg n`$ (high-dimensional data with few samples), FDA can overfit severely. This happens because:

1. **Perfect Separation**: With $`p \geq n`$, we can always find directions that perfectly separate classes - like having so many viewing angles that we can always find one that makes the groups look perfectly separated
2. **Random Features**: Even random noise can appear discriminative in high dimensions - like random features accidentally looking like they separate the classes
3. **Limited Degrees of Freedom**: The within-class scatter matrix becomes singular - like having too many features relative to the number of samples

**Intuition**: Overfitting in FDA is like having so many different ways to view the data that we can always find a viewing angle that makes the groups look perfectly separated, even if this separation doesn't generalize to new data. It's like memorizing the specific arrangement of books in a library rather than learning the general principles of genre organization.

### Example: Overfitting Demonstration

The overfitting demonstration is implemented in the code files:

**Python:** See `demonstrate_overfitting()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `demonstrate_overfitting()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example shows how FDA can achieve perfect separation even with random features when the number of features greatly exceeds the number of samples, demonstrating the overfitting problem in high-dimensional settings.

**Intuition**: This demonstration shows that when we have many more features than samples, FDA can find directions that perfectly separate the classes even when the features are completely random. This is like having so many different ways to view the data that we can always find a viewing angle that makes random noise look meaningful.

### Mitigation Strategies

#### 1. Regularization

The regularized FDA implementation is provided in the code files:

**Python:** See `regularized_fda()` and `calculate_scatter_matrices()` functions in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `regularized_fda()` and `calculate_scatter_matrices()` functions in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach adds a regularization term to the within-class scatter matrix to prevent singularity and improve numerical stability.

**Intuition**: Regularization is like adding a "safety net" to prevent the model from making extreme assumptions. It stabilizes the within-class scatter matrix, making the FDA directions more robust and less prone to overfitting.

#### 2. Feature Selection

The FDA with feature selection implementation is provided in the code files:

**Python:** See `fda_with_feature_selection()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `fda_with_feature_selection()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach uses statistical tests (F-test) to select the most discriminative features before applying FDA.

**Intuition**: Feature selection is like focusing only on the most important viewing angles. By removing irrelevant or redundant features, we reduce the complexity of the model and make it more robust to overfitting.

#### 3. Cross-Validation

The cross-validation implementation is provided in the code files:

**Python:** See `cross_validate_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `cross_validate_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This approach uses stratified k-fold cross-validation to assess the generalization performance of FDA projections.

**Intuition**: Cross-validation is like testing our viewing angles on different parts of the data to make sure they work well on new data, not just the data we used to find them.

## 9.5.7. Real-World Applications

### Example 1: Face Recognition

The face recognition example is implemented in the code files:

**Python:** See `face_recognition_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `face_recognition_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example demonstrates FDA for face recognition using the Olivetti faces dataset, showing how FDA can reduce high-dimensional face data to discriminative components while maintaining classification accuracy.

**Intuition**: Face recognition is like trying to find the best way to view faces so that different people are most clearly distinguished. FDA finds the optimal "viewing angles" that make each person's face most distinct from others, reducing the high-dimensional pixel data to a few key discriminative features.

### Example 2: Gene Expression Analysis

The gene expression analysis example is implemented in the code files:

**Python:** See `gene_expression_fda()` function in [`code/fda_implementation.py`](code/fda_implementation.py)

**R:** See `gene_expression_fda()` function in [`code/r_fda_implementation.R`](code/r_fda_implementation.R)

This example shows FDA applied to high-dimensional gene expression data, demonstrating feature selection and dimensionality reduction for biological data analysis.

**Intuition**: Gene expression analysis is like trying to find the most important genes that distinguish between different biological conditions (like healthy vs diseased cells). FDA finds the optimal combination of genes that best separates these conditions, reducing thousands of gene measurements to a few key discriminative features.

## 9.5.8. Summary and Best Practices

### Key Takeaways

1. **FDA Objective**: Maximize between-class variance while minimizing within-class variance - like finding the best viewing angles for maximum separation
2. **Supervised Nature**: Uses class labels to find discriminative directions - like using genre labels to organize books effectively
3. **Dimensionality Reduction**: Naturally reduces to $`K-1`$ dimensions - like only needing K-1 viewing angles to separate K groups
4. **Connection to LDA**: Equivalent under normality assumptions - like FDA and LDA giving the same results when data follows certain patterns

**Intuition**: FDA is a powerful tool for supervised dimensionality reduction that finds the optimal ways to view data for maximum class separation. It's like having a smart organizer who knows exactly how to arrange things to make different groups most clearly distinct.

### Best Practices

1. **Data Preprocessing**:
   - Standardize features - like putting all measurements on the same scale
   - Handle missing values - like dealing with incomplete data
   - Check for multicollinearity - like making sure features aren't too similar to each other

2. **Dimensionality Management**:
   - Use regularization when $`p \gg n`$ - like adding stability when you have many features but few samples
   - Apply feature selection - like focusing on the most important features
   - Cross-validate results - like testing on different parts of the data

3. **Model Validation**:
   - Check for overfitting - like making sure the separation generalizes to new data
   - Use cross-validation - like testing on different data subsets
   - Monitor separation metrics - like measuring how good the separation is

4. **Interpretation**:
   - Examine discriminant directions - like understanding what each viewing angle represents
   - Analyze explained variance ratios - like understanding how much each direction contributes
   - Visualize projections - like seeing how the data looks from the optimal angles

**Intuition**: These best practices help us build a robust and interpretable FDA system. They ensure that our dimensionality reduction works well in practice and provides insights that are useful for understanding the underlying patterns in our data.

### When to Use FDA

**Use FDA when**:
- You need supervised dimensionality reduction - like when you want to reduce dimensions while preserving class separation
- Classes are well-separated - like when different groups are clearly distinct
- Interpretability is important - like when you need to understand what the reduced dimensions represent
- You want to reduce to $`K-1`$ dimensions - like when you want the minimum number of dimensions needed to separate K classes

**Consider alternatives when**:
- Classes overlap significantly (use other methods) - like when groups are too similar to separate well
- You need more than $`K-1`$ dimensions - like when you need more detailed information
- Data is non-linear (use kernel methods) - like when the relationships are too complex for linear projections

**Intuition**: FDA is most useful when you have clearly separated classes and want to reduce dimensionality while preserving this separation. It's like having well-organized groups that you want to view from the optimal angles.

### Limitations

1. **Linear Assumption**: Only finds linear projections - like only being able to view data from straight-on angles
2. **Overfitting Risk**: Can overfit in high dimensions - like memorizing specific arrangements rather than learning general principles
3. **Normality Assumption**: Implicit in the formulation - like assuming certain data patterns
4. **Limited Dimensions**: Maximum $`K-1`$ components - like being limited to K-1 viewing angles for K groups

**Intuition**: These limitations remind us that FDA is a powerful but specialized tool. It works best when the data has clear linear separations and when we don't need too many dimensions. It's like having a very effective but somewhat limited viewing system.

Fisher Discriminant Analysis remains a powerful and interpretable method for supervised dimensionality reduction, providing a solid foundation for understanding the relationship between classes in high-dimensional data.

---

**Navigation:**
- **Next Topic:** [Naive Bayes Classifiers](06_naive_bayes_classifiers.md) - Conditional independence assumption and probabilistic classification
- **Previous Topic:** [Linear Discriminant Analysis](04_linear_discriminant_analysis.md) - Shared covariance assumption and linear decision boundaries
