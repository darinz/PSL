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

### Practical Comparison

```python
def comprehensive_model_comparison(X_train, y_train, X_test, y_test):
    """Comprehensive comparison of discriminant analysis methods"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import time
    
    models = {
        'Our QDA': QuadraticDiscriminantAnalysis(),
        'Our LDA': LinearDiscriminantAnalysis(),
        'Our Naive Bayes': GaussianNaiveBayes(),
        'Sklearn LDA': SklearnLDA(),
        'Sklearn QDA': SklearnQDA(),
        'Sklearn Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        # Time the fitting
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fit_time': fit_time,
            'predict_time': predict_time
        }
    
    # Create comparison table
    df_results = pd.DataFrame(results).T
    print("Model Comparison Results:")
    print(df_results.round(4))
    
    return results, df_results

# Run comprehensive comparison
results, df_results = comprehensive_model_comparison(X_train, y_train, X_test, y_test)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
metrics = ['accuracy', 'precision', 'recall', 'f1']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = [results[name][metric] for name in results.keys()]
    names = list(results.keys())
    
    bars = ax.bar(range(len(values)), values, alpha=0.8)
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

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

```python
def regularized_lda(X_train, y_train, X_test, y_test, alpha=0.1):
    """Regularized LDA with shrinkage"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    # Regularized LDA with shrinkage
    lda_reg = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=alpha)
    lda_reg.fit(X_train, y_train)
    
    accuracy = lda_reg.score(X_test, y_test)
    print(f"Regularized LDA (α={alpha}) accuracy: {accuracy:.3f}")
    
    return lda_reg

# Test regularization
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.1)
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.5)
regularized_lda(X_train, y_train, X_test, y_test, alpha=0.9)
```

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of discriminant analysis methods. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and comparison of different approaches.
