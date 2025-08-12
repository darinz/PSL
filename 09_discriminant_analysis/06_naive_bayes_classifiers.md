# 9.6. Naive Bayes Classifiers

## 9.6.0. Introduction and Motivation

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with a strong (naive) assumption of conditional independence between features. Despite its simplicity, Naive Bayes often performs surprisingly well and is widely used in text classification, spam filtering, medical diagnosis, and many other applications.

### The "Naive" Assumption

The "Naive" part comes from the **conditional independence assumption**: given the class label, all features are assumed to be independent of each other. This is often violated in real-world data, but the method still works well in practice.

### Why Naive Bayes Works

1. **Computational Efficiency**: Independence assumption dramatically reduces parameter count
2. **Robustness**: Works well even when independence assumption is violated
3. **Interpretability**: Easy to understand and explain
4. **Small Sample Performance**: Works well with limited training data

## 9.6.1. Mathematical Foundation

### Bayes' Theorem

The foundation of Naive Bayes is Bayes' theorem:

```math
P(Y=k | X=x) = \frac{P(X=x | Y=k) \cdot P(Y=k)}{P(X=x)}
```

Where:
- $`P(Y=k | X=x)`$ is the **posterior probability** of class $`k`$ given features $`x`$
- $`P(X=x | Y=k)`$ is the **likelihood** of features $`x`$ given class $`k`$
- $`P(Y=k)`$ is the **prior probability** of class $`k`$
- $`P(X=x)`$ is the **evidence** (normalizing constant)

### The Decision Function

For classification, we want to find the class that maximizes the posterior probability:

```math
\hat{y} = \arg\max_k P(Y=k | X=x)
```

Since $`P(X=x)`$ is the same for all classes, we can ignore it and maximize:

```math
\hat{y} = \arg\max_k P(X=x | Y=k) \cdot P(Y=k)
```

Or equivalently, using logarithms to avoid numerical underflow:

```math
\hat{y} = \arg\max_k \log P(X=x | Y=k) + \log P(Y=k)
```

### The Naive Independence Assumption

The key assumption is that features are conditionally independent given the class:

```math
P(X=x | Y=k) = P(X_1=x_1 | Y=k) \cdot P(X_2=x_2 | Y=k) \cdots P(X_p=x_p | Y=k)
```

This allows us to factorize the joint likelihood into a product of individual feature likelihoods:

```math
f_k(x) = f_{k1}(x_1) \times f_{k2}(x_2) \times \cdots \times f_{kp}(x_p)
```

Where $`f_{kj}(x_j)`$ is the probability density (or mass) function for feature $`j`$ in class $`k`$.

## 9.6.2. Parameter Estimation

### Prior Probabilities

The prior probability of class $`k`$ is estimated as:

```math
\hat{\pi}_k = P(Y=k) = \frac{n_k}{n}
```

Where $`n_k`$ is the number of samples in class $`k`$ and $`n`$ is the total number of samples.

### Likelihood Estimation

The estimation of $`f_{kj}(x_j)`$ depends on the type of features:

#### 1. Discrete Features (Categorical)

For discrete features, we use empirical probabilities:

```math
\hat{f}_{kj}(x_j) = P(X_j = x_j | Y = k) = \frac{\text{count}(X_j = x_j, Y = k)}{\text{count}(Y = k)}
```

#### 2. Continuous Features (Numerical)

For continuous features, we have two options:

**Parametric Approach (Gaussian Naive Bayes)**:
```math
f_{kj}(x_j) = \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)
```

Where:
- $`\mu_{kj} = \frac{1}{n_k} \sum_{i: y_i=k} x_{ij}`$ (mean of feature $`j`$ in class $`k`$)
- $`\sigma_{kj}^2 = \frac{1}{n_k-1} \sum_{i: y_i=k} (x_{ij} - \mu_{kj})^2`$ (variance of feature $`j`$ in class $`k`$)

**Non-parametric Approach (Kernel Density Estimation)**:
```math
f_{kj}(x_j) = \frac{1}{n_k h} \sum_{i: y_i=k} K\left(\frac{x_j - x_{ij}}{h}\right)
```

Where $`K`$ is a kernel function (e.g., Gaussian) and $`h`$ is the bandwidth.

### Parameter Count

For **parametric Naive Bayes** with $`p`$ features and $`K`$ classes:
- **Means**: $`K \times p`$ parameters
- **Variances**: $`K \times p`$ parameters  
- **Priors**: $`K`$ parameters
- **Total**: $`2Kp + K`$ parameters

This is much smaller than the $`K \times 2^p`$ parameters needed without the independence assumption.

## 9.6.3. Classification Decision Function

### Log-Likelihood Formulation

To avoid numerical underflow, we work with logarithms. The decision function becomes:

```math
d_k(x) = \log P(Y=k) + \sum_{j=1}^p \log f_{kj}(x_j)
```

### Gaussian Naive Bayes Decision Function

For Gaussian Naive Bayes, the decision function is:

```math
\begin{split}
d_k(x) &= \log \pi_k + \sum_{j=1}^p \log f_{kj}(x_j) \\
&= \log \pi_k + \sum_{j=1}^p \log \left(\frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)\right) \\
&= \log \pi_k + \sum_{j=1}^p \left(-\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma_{kj}^2) - \frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right) \\
&= \log \pi_k - \frac{p}{2}\log(2\pi) - \frac{1}{2}\sum_{j=1}^p \log(\sigma_{kj}^2) - \frac{1}{2}\sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
\end{split}
```

### Numerical Stability Issues

The key insight is that we can drop constant terms that don't depend on the class:

```math
d_k(x) = \log \pi_k - \frac{1}{2}\sum_{j=1}^p \log(\sigma_{kj}^2) - \frac{1}{2}\sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}
```

**Critical Issue**: When $`x_j`$ is far from $`\mu_{kj}`$, the exponential term becomes very small, leading to numerical underflow. Some implementations truncate these values, which can lead to incorrect predictions.

## 9.6.4. Implementation from Scratch

The complete implementation of Naive Bayes Classifier from scratch is provided in the following code files:

**Python Implementation:** [`code/naive_bayes_implementation.py`](code/naive_bayes_implementation.py)

**R Implementation:** [`code/r_naive_bayes_implementation.R`](code/r_naive_bayes_implementation.R)

These files contain:

- Complete `NaiveBayesClassifier` class with parameter estimation
- Prior probability and likelihood estimation for each class
- Log-probability based prediction for numerical stability
- Comparison with library implementations (sklearn GaussianNB, e1071 naiveBayes)
- Visualization functions for decision boundaries and feature importance
- Comprehensive demonstration functions with synthetic data
- Feature importance analysis based on variance ratios

The implementation follows the mathematical formulation using log-probabilities to avoid numerical underflow issues, and includes regularization to prevent zero variances.

## 9.6.5. Numerical Stability Issues

### The Problem

When computing probabilities for points far from the class means, the Gaussian PDF becomes extremely small:

```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
```

For large $`|x - \mu|`$, this approaches zero, causing numerical underflow.

### Demonstration of the Issue

The numerical stability demonstration is implemented in the code files:

**Python:** See `demonstrate_numerical_issues()` function in [`code/naive_bayes_implementation.py`](code/naive_bayes_implementation.py)

**R:** See `demonstrate_numerical_issues_r()` function in [`code/r_naive_bayes_implementation.R`](code/r_naive_bayes_implementation.R)

This demonstration shows how Gaussian PDF values become extremely small for points far from the mean, causing numerical underflow, while log-PDF values remain numerically stable.

### Solutions

#### 1. Use Log-Probabilities (Recommended)

Always work with log-probabilities to avoid underflow:

```python
def safe_naive_bayes_predict(X, model):
    """
    Safe Naive Bayes prediction using log-probabilities
    """
    log_proba = model.predict_log_proba(X)
    return model.classes_[np.argmax(log_proba, axis=1)]
```

#### 2. Add Regularization

Add small constants to prevent zero variances:

```python
def regularized_naive_bayes(X, y, epsilon=1e-9):
    """
    Naive Bayes with regularization
    """
    # ... existing code ...
    
    # Regularize variances
    self.variances_ = np.maximum(self.variances_, epsilon)
    
    return self
```

#### 3. Truncation (Not Recommended)

Some packages truncate very small probabilities, but this can lead to incorrect predictions:

```python
def truncated_naive_bayes(X, model, threshold=1e-10):
    """
    Naive Bayes with truncation (not recommended)
    """
    proba = model.predict_proba(X)
    proba = np.maximum(proba, threshold)  # Truncate small values
    return model.classes_[np.argmax(proba, axis=1)]
```

## 9.6.6. Variants of Naive Bayes

### 1. Gaussian Naive Bayes

For continuous features, assumes Gaussian distribution:

```python
class GaussianNaiveBayes(NaiveBayesClassifier):
    def __init__(self):
        super().__init__(feature_type='gaussian')
```

### 2. Multinomial Naive Bayes

For discrete count data (e.g., text classification):

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        
    def fit(self, X, y):
        # Count features for each class
        # Apply Laplace smoothing
        # Estimate class-conditional probabilities
        pass
```

### 3. Bernoulli Naive Bayes

For binary features:

```python
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of feature being 1 for each class
        # Apply Laplace smoothing
        pass
```

### 4. Categorical Naive Bayes

For categorical features:

```python
class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of each category for each class
        # Apply Laplace smoothing
        pass
```

## 9.6.7. Real-World Applications

### Example 1: Text Classification

```python
def text_classification_example():
    """
    Naive Bayes for text classification
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Sample text data
    texts = [
        "great movie amazing acting",
        "terrible film waste of time", 
        "excellent performance brilliant",
        "boring plot disappointing",
        "fantastic story wonderful",
        "awful acting bad script",
        "outstanding film superb",
        "poor quality terrible",
        "incredible movie perfect",
        "horrible waste bad"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fit Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    
    print("Text Classification Results:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, 
                               target_names=['Negative', 'Positive']))
    
    # Feature importance
    feature_names = vectorizer.get_feature_names_out()
    log_probs = nb.feature_log_prob_
    
    # Show most discriminative words
    positive_words = log_probs[1] - log_probs[0]
    negative_words = log_probs[0] - log_probs[1]
    
    print("\nMost Positive Words:")
    pos_indices = np.argsort(positive_words)[-5:]
    for idx in pos_indices:
        print(f"  {feature_names[idx]}: {positive_words[idx]:.3f}")
    
    print("\nMost Negative Words:")
    neg_indices = np.argsort(negative_words)[-5:]
    for idx in neg_indices:
        print(f"  {feature_names[idx]}: {negative_words[idx]:.3f}")
    
    return nb, vectorizer
```

### Example 2: Medical Diagnosis

```python
def medical_diagnosis_example():
    """
    Naive Bayes for medical diagnosis
    """
    # Simulate medical data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, blood_pressure, cholesterol, glucose
    age = np.random.normal(50, 15, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    glucose = np.random.normal(100, 20, n_samples)
    
    X = np.column_stack([age, blood_pressure, cholesterol, glucose])
    
    # Disease risk based on features
    risk_score = (age * 0.1 + (blood_pressure - 120) * 0.05 + 
                  (cholesterol - 200) * 0.02 + (glucose - 100) * 0.03 +
                  np.random.normal(0, 0.1, n_samples))
    
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit Naive Bayes
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Medical Diagnosis Results:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 'Glucose']
    feature_importance = np.zeros(4)
    
    for j in range(4):
        overall_mean = np.mean(X[:, j])
        between_var = np.sum([np.sum(y == c) * (np.mean(X[y == c, j]) - overall_mean)**2 
                             for c in np.unique(y)])
        within_var = np.sum([np.sum((X[y == c, j] - np.mean(X[y == c, j]))**2) 
                            for c in np.unique(y)])
        feature_importance[j] = between_var / within_var if within_var > 0 else 0
    
    # Plot feature importance
    plt.figure(figsize=(10, 4))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance in Medical Diagnosis')
    plt.ylabel('Between/Within Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return nb, feature_importance
```

## 9.6.8. Advantages and Limitations

### Advantages

1. **Simplicity**: Easy to understand and implement
2. **Speed**: Fast training and prediction
3. **Small Sample Performance**: Works well with limited data
4. **Interpretability**: Clear probabilistic interpretation
5. **Handles Missing Data**: Can handle missing features gracefully

### Limitations

1. **Independence Assumption**: Often violated in real data
2. **Feature Scaling**: Sensitive to feature scaling
3. **Zero Frequency Problem**: Can't handle unseen feature values
4. **Continuous Features**: Assumes specific distributions
5. **Correlated Features**: Performance degrades with correlated features

### When to Use Naive Bayes

**Use Naive Bayes when**:
- You have limited training data
- Features are approximately independent
- You need fast training and prediction
- Interpretability is important
- You're doing text classification

**Consider alternatives when**:
- Features are highly correlated
- You have complex feature interactions
- You need high accuracy (consider ensemble methods)
- You have large amounts of training data

## 9.6.9. Summary and Best Practices

### Key Takeaways

1. **Independence Assumption**: The core assumption that makes Naive Bayes "naive"
2. **Log-Probabilities**: Always use log-probabilities for numerical stability
3. **Parameter Count**: Only $`2Kp + K`$ parameters needed
4. **Variants**: Choose the right variant for your data type

### Best Practices

1. **Data Preprocessing**:
   - Handle missing values appropriately
   - Scale features if using Gaussian Naive Bayes
   - Apply Laplace smoothing for discrete features

2. **Model Selection**:
   - Use Gaussian NB for continuous features
   - Use Multinomial NB for count data
   - Use Bernoulli NB for binary features

3. **Numerical Stability**:
   - Always work with log-probabilities
   - Add small constants to prevent zero variances
   - Avoid truncation of small probabilities

4. **Evaluation**:
   - Use cross-validation for small datasets
   - Check for feature independence violations
   - Monitor for numerical issues

### Implementation Checklist

- [ ] Choose appropriate Naive Bayes variant
- [ ] Handle missing values
- [ ] Apply feature scaling if needed
- [ ] Use log-probabilities for numerical stability
- [ ] Add regularization to prevent zero variances
- [ ] Validate independence assumption
- [ ] Cross-validate model performance

Naive Bayes remains a powerful and interpretable classification method that provides an excellent baseline for many machine learning problems, especially when computational efficiency and interpretability are important.
