# 9.2. Discriminant Analysis

## 9.2.1. Introduction to Discriminant Analysis

Discriminant analysis is a family of classification methods that model the distribution of features within each class and use Bayes' theorem to make predictions. Unlike discriminative methods that directly model $`P(Y|X)`$, discriminant analysis is a **generative approach** that models the joint distribution $`P(X, Y)`$ by decomposing it into class-conditional densities and class priors.

### Generative vs. Discriminative Approaches

| Approach | Models | Example Methods |
|----------|--------|-----------------|
| **Generative** | $`P(X, Y) = P(Y) \cdot P(X \mid Y)`$ | LDA, QDA, Naive Bayes |
| **Discriminative** | $`P(Y \mid X)`$ directly | Logistic Regression, SVM |

### Mathematical Foundation

The key insight of discriminant analysis is to decompose the joint distribution:

```math
p(x, y) = p(y) \cdot p(x \mid y)
```

where:
- $`p(y)`$ is the **class prior** (marginal distribution of classes)
- $`p(x \mid y)`$ is the **class-conditional density** (distribution of features given class)

This decomposition allows us to:
1. Estimate class priors from class frequencies in the data
2. Model class-conditional densities using parametric or non-parametric methods
3. Apply Bayes' theorem to compute posterior probabilities

### Types of Discriminant Analysis

We will explore three main approaches:

1. **Quadratic Discriminant Analysis (QDA)**: Assumes different covariance matrices for each class
2. **Linear Discriminant Analysis (LDA)**: Assumes shared covariance matrix across classes
3. **Naive Bayes**: Assumes conditional independence of features given class

## 9.2.2. Bayes' Theorem and Optimal Classification

### Derivation of Bayes' Theorem

The optimal classifier maximizes the posterior probability $`P(Y=k \mid X=x)`$. Using Bayes' theorem:

```math
P(Y = k \mid X=x) = \frac{P(X=x, Y=k)}{P(X=x)} = \frac{P(X=x \mid Y=k) \cdot P(Y=k)}{P(X=x)}
```

Let's define:
- $`f_k(x) = p(x \mid Y=k)`$: class-conditional density function
- $`\pi_k = P(Y=k)`$: class prior probability

Then:

```math
P(Y = k \mid X=x) = \frac{\pi_k f_k(x)}{P(X=x)} \propto \pi_k f_k(x)
```

Since $`P(X=x)`$ is constant across all classes, the optimal classifier is:

```math
\hat{y} = \arg\max_k P(Y=k \mid X=x) = \arg\max_k \pi_k f_k(x)
```

### Log-Likelihood Formulation

For numerical stability and computational efficiency, we often work with log-likelihoods:

```math
\hat{y} = \arg\max_k \log(\pi_k f_k(x)) = \arg\max_k [\log \pi_k + \log f_k(x)]
```

Or equivalently, minimizing the negative log-likelihood:

```math
\hat{y} = \arg\min_k [-\log \pi_k - \log f_k(x)]
```

The Bayes Classifier framework provides the foundation for all discriminant analysis methods. The `BayesClassifier` base class implements the core functionality for estimating class priors and computing posterior probabilities using Bayes' theorem.

**Key Functions:**
- `BayesClassifier.__init__()`: Initialize the base classifier
- `BayesClassifier.fit()`: Fit the classifier by estimating class priors and conditional densities
- `BayesClassifier.predict_proba()`: Compute posterior probabilities using log-likelihoods
- `BayesClassifier.predict()`: Predict class labels using maximum posterior probability
- `BayesClassifier.score()`: Compute accuracy score
- `create_gaussian_mixture_data()`: Create synthetic Gaussian mixture data for demonstrations
- `demonstrate_bayes_classifier()`: Complete demonstration with data creation and splitting

The framework uses numerical stability techniques (log-likelihoods and softmax) to handle the computational challenges of Bayes' theorem in high-dimensional spaces.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete Bayes Classifier framework.

The R implementation provides equivalent functionality using the `MASS`, `ggplot2`, and `caret` packages. The implementation demonstrates data creation, model fitting, and result visualization in R.

**Key Functions:**
- `create_gaussian_mixture_data()`: Create synthetic Gaussian mixture data
- `demonstrate_bayes_classifier()`: Complete demonstration with data creation and splitting
- `QDA()`, `LDA()`, `GaussianNaiveBayes()`: Model fitting functions
- `plot_decision_boundaries()`: Visualization of decision boundaries
- `compare_models()`: Comprehensive model comparison

The R implementation leverages established packages for robust and efficient discriminant analysis, providing a clean interface for both basic and advanced usage.

See the implementation in `code/r_discriminant_analysis_implementation.R` for the complete R-based discriminant analysis workflow.

## 9.2.3. Quadratic Discriminant Analysis (QDA)

### Mathematical Formulation

QDA assumes that each class follows a multivariate Gaussian distribution with its own mean and covariance matrix:

```math
f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)
```

The decision function becomes:

```math
\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k
```

### Parameter Estimation

For each class $`k`$:

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$ where $`n_k`$ is the number of samples in class $`k`$
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$
3. **Class covariance**: $`\hat{\Sigma}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$

The QDA implementation extends the Bayes Classifier framework to handle class-specific covariance matrices. The `QuadraticDiscriminantAnalysis` class implements the complete QDA algorithm with quadratic decision boundaries.

**Key Functions:**
- `QuadraticDiscriminantAnalysis._fit_conditional_densities()`: Fit Gaussian densities with class-specific covariances
- `QuadraticDiscriminantAnalysis.decision_function()`: Compute quadratic discriminant function values
- `plot_decision_boundaries_qda()`: Visualize QDA decision boundaries
- `demonstrate_qda()`: Complete demonstration with model fitting and evaluation

QDA is particularly effective when classes have different covariance structures, allowing for more flexible decision boundaries compared to LDA.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete QDA workflow.

## 9.2.4. Linear Discriminant Analysis (LDA)

### Mathematical Formulation

LDA assumes that all classes share the same covariance matrix $`\Sigma`$:

```math
f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right)
```

The decision function becomes linear:

```math
\delta_k(x) = \mu_k^T \Sigma^{-1} x - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k
```

### Parameter Estimation

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$
3. **Shared covariance**: $`\hat{\Sigma} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$

The LDA implementation extends the Bayes Classifier framework to use a shared covariance matrix across all classes. The `LinearDiscriminantAnalysis` class implements the complete LDA algorithm with linear decision boundaries.

**Key Functions:**
- `LinearDiscriminantAnalysis._fit_conditional_densities()`: Fit Gaussian densities with shared covariance
- `LinearDiscriminantAnalysis.decision_function()`: Compute linear discriminant function values
- `compare_qda_lda()`: Compare QDA and LDA performance with visualization
- `demonstrate_lda()`: Complete demonstration with model fitting and evaluation

LDA is particularly effective when classes have similar covariance structures, providing linear decision boundaries that are often more robust in high-dimensional spaces.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete LDA workflow.

## 9.2.5. Naive Bayes

### Mathematical Formulation

Naive Bayes assumes conditional independence of features given the class:

```math
f_k(x) = \prod_{j=1}^p f_{kj}(x_j)
```

where $`f_{kj}(x_j)`$ is the marginal density of feature $`j`$ in class $`k`$.

The decision function becomes:

```math
\delta_k(x) = \log \pi_k + \sum_{j=1}^p \log f_{kj}(x_j)
```

The Gaussian Naive Bayes implementation extends the Bayes Classifier framework to assume conditional independence of features given the class. The `GaussianNaiveBayes` class implements the complete naive Bayes algorithm with independent Gaussian distributions.

**Key Functions:**
- `GaussianNaiveBayes._fit_conditional_densities()`: Fit independent Gaussian densities for each feature
- `GaussianNaiveBayes.decision_function()`: Compute naive Bayes decision function values
- `demonstrate_naive_bayes()`: Complete demonstration with model fitting and evaluation

Naive Bayes is particularly effective when features are approximately independent given the class, providing fast and often surprisingly accurate predictions even with the independence assumption.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete Gaussian Naive Bayes workflow.

## 9.2.6. Fisher's Discriminant Analysis (FDA)

### Mathematical Foundation

FDA finds a linear projection that maximizes the ratio of between-class variance to within-class variance:

```math
J(w) = \frac{w^T S_B w}{w^T S_W w}
```

where:
- $`S_B = \sum_{k=1}^K n_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^T`$ is the between-class scatter matrix
- $`S_W = \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^T`$ is the within-class scatter matrix
- $`\bar{\mu} = \frac{1}{n} \sum_{i=1}^n x_i`$ is the overall mean

### Solution

The optimal projection vector is the eigenvector corresponding to the largest eigenvalue of $`S_W^{-1} S_B`$:

```math
S_W^{-1} S_B w = \lambda w
```

The Fisher's Discriminant Analysis implementation provides dimensionality reduction by finding optimal linear projections that maximize between-class variance while minimizing within-class variance. The `FishersDiscriminantAnalysis` class implements the complete FDA algorithm.

**Key Functions:**
- `FishersDiscriminantAnalysis.__init__()`: Initialize FDA with number of components
- `FishersDiscriminantAnalysis.fit()`: Fit FDA by computing scatter matrices and solving eigenvalue problem
- `FishersDiscriminantAnalysis.transform()`: Transform data using FDA projection
- `FishersDiscriminantAnalysis.fit_transform()`: Fit FDA and transform data in one step
- `demonstrate_fda()`: Complete demonstration with visualization and LDA application

FDA is particularly useful for dimensionality reduction in classification problems, providing optimal projections that preserve class separability.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete FDA workflow.

## 9.2.7. Model Comparison and Selection

### Theoretical Comparison

| Method | Assumptions | Decision Boundary | Complexity |
|--------|-------------|-------------------|------------|
| **QDA** | Different covariances | Quadratic | $`O(p^2)`$ |
| **LDA** | Shared covariance | Linear | $`O(p^2)`$ |
| **Naive Bayes** | Feature independence | Piecewise linear | $`O(p)`$ |

The comprehensive model comparison implementation provides systematic evaluation of different discriminant analysis methods, including both custom implementations and scikit-learn equivalents. The comparison framework evaluates multiple performance metrics and computational efficiency.

**Key Functions:**
- `comprehensive_model_comparison()`: Compare multiple discriminant analysis methods
- `demonstrate_model_comparison()`: Complete demonstration with visualization
- Evaluates accuracy, precision, recall, F1-score, fit time, and prediction time
- Compares custom implementations with scikit-learn equivalents

The comparison provides insights into the relative performance of different discriminant analysis approaches and helps guide model selection decisions.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete model comparison workflow.

## 9.2.8. Practical Considerations

### Model Selection Guidelines

1. **Use LDA when**:
   - Classes have similar covariance structures
   - Sample size is small relative to number of features
   - Linear decision boundaries are appropriate

2. **Use QDA when**:
   - Classes have different covariance structures
   - Sufficient data to estimate class-specific covariances
   - Non-linear decision boundaries are needed

3. **Use Naive Bayes when**:
   - Features are approximately independent given class
   - High-dimensional data with limited samples
   - Fast prediction is required

### Regularization and Robustness

The regularization implementation provides techniques to improve LDA performance in high-dimensional settings where the covariance matrix may be ill-conditioned. Regularization helps stabilize parameter estimation and improve generalization.

**Key Functions:**
- `regularized_lda()`: Implement LDA with shrinkage regularization
- `demonstrate_regularization()`: Complete demonstration with different regularization levels
- Uses scikit-learn's LDA with shrinkage parameter for robust estimation

Regularization is particularly important when the number of features is large relative to the sample size, helping to prevent overfitting and improve model stability.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete regularization workflow.

---

## Code Files Summary

The discriminant analysis concepts have been implemented in the following code files:

### Python Implementation (`code/discriminant_analysis_implementation.py`)
- **Bayes Classifier Framework**: `BayesClassifier` base class with core functionality
- **Quadratic Discriminant Analysis**: `QuadraticDiscriminantAnalysis` class with class-specific covariances
- **Linear Discriminant Analysis**: `LinearDiscriminantAnalysis` class with shared covariance
- **Gaussian Naive Bayes**: `GaussianNaiveBayes` class with feature independence assumption
- **Fisher's Discriminant Analysis**: `FishersDiscriminantAnalysis` class for dimensionality reduction
- **Model Comparison**: Comprehensive evaluation framework with multiple metrics
- **Regularization**: LDA with shrinkage for robust estimation
- **Demonstration Functions**: Complete workflows for each method and comparison

### R Implementation (`code/r_discriminant_analysis_implementation.R`)
- **Bayes Classifier Framework**: Conceptual framework for R implementations
- **QDA, LDA, Naive Bayes**: Model fitting functions using established R packages
- **Fisher's Discriminant Analysis**: FDA implementation using MASS package
- **Visualization**: Decision boundary plotting and FDA projection visualization
- **Model Comparison**: Comprehensive comparison framework in R
- **Regularization**: Regularized LDA with shrinkage parameter
- **Demonstration Functions**: Complete workflows for each method

Both implementations provide comprehensive coverage of discriminant analysis concepts with practical examples and demonstrate the relationship between theoretical foundations and practical applications in classification problems.
