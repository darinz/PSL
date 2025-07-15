# 12.4. AdaBoosting

Many of you are already familiar with how decision trees work, the underlying concepts, and the criteria used to split a tree. You can grow a decision tree, let it reach a certain size, and then apply pruning techniques to obtain your final classification model.

However, we know that a single tree often doesn't perform very well on its own. That's where ensemble methods come into play, such as Random Forests, which build on the principles we've learned about regression trees. Another powerful ensemble method is boosting, and in this discussion, we'll delve into the concept of boosting.

Boosting, specifically AdaBoost, was introduced in the context of classification. We'll explore what AdaBoost does and what we can infer about the final classifier from this boosting algorithm. It's worth noting that AdaBoost is essentially a **gradient-based** algorithm, aiming to fit the model using an **exponential loss** function.

## 12.4.1. Introduction to Boosting

### What is Boosting?

Boosting is an ensemble learning technique that combines multiple weak learners to create a strong learner. Unlike bagging (used in Random Forests), which builds independent models in parallel, boosting builds models sequentially, where each new model focuses on the mistakes of the previous ones.

### Key Principles of Boosting

1. **Sequential Learning**: Models are built one after another, each learning from the errors of its predecessors
2. **Weighted Data**: Training instances are weighted, with misclassified instances getting higher weights
3. **Weak Learners**: Each base model is intentionally kept simple (weak) but better than random guessing
4. **Weighted Combination**: Final prediction is a weighted vote of all weak learners

### Why AdaBoost?

AdaBoost (Adaptive Boosting) was one of the first practical boosting algorithms, introduced by Freund and Schapire in 1995. It's particularly effective because:

- It automatically adapts to the errors of previous classifiers
- It can handle both binary and multi-class problems
- It's resistant to overfitting in many cases
- It provides a theoretical guarantee of performance improvement

## 12.4.2. Mathematical Foundation

### Problem Setup

Consider a binary classification problem with:
- Training data: $\{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$
- Labels: $y_i \in \{-1, +1\}$ (note: AdaBoost uses ±1 instead of 0/1)
- Weak learners: $g_t(x) \in \{-1, +1\}$ for iteration $t$
- Final classifier: $G(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t g_t(x)\right)$

### Exponential Loss Function

AdaBoost minimizes the exponential loss function:

```math
L(y, f(x)) = \exp(-y \cdot f(x))
```

where $f(x) = \sum_{t=1}^T \alpha_t g_t(x)$ is the weighted combination of weak learners.

**Why exponential loss?**
- It heavily penalizes misclassifications
- It's differentiable and convex
- It leads to a simple update rule for weights

### Weight Update Mechanism

The key insight of AdaBoost is how it updates instance weights:

```math
w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i g_t(x_i))
```

This means:
- Correctly classified instances: weight decreases
- Misclassified instances: weight increases

## 12.4.3. The AdaBoost Algorithm

### Algorithm Steps

**Input**: Training data $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, number of iterations $T$

**Initialize**: $w_i^{(1)} = \frac{1}{n}$ for all $i = 1, \ldots, n$

**For** $t = 1, 2, \ldots, T$:

1. **Train weak learner** $g_t(x)$ on weighted data
2. **Compute weighted error**:
   ```math
   \epsilon_t = \sum_{i=1}^n w_i^{(t)} \cdot I(y_i \neq g_t(x_i))
   ```
3. **Compute classifier weight**:
   ```math
   \alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
   ```
4. **Update instance weights**:
   ```math
   w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i g_t(x_i))
   ```
5. **Normalize weights**:
   ```math
   w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^n w_j^{(t+1)}}
   ```

**Output**: Final classifier $G(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t g_t(x)\right)$

### Key Insights

1. **Classifier Weight $\alpha_t$**: 
   - If $\epsilon_t < 0.5$ (better than random), then $\alpha_t > 0$
   - If $\epsilon_t > 0.5$ (worse than random), then $\alpha_t < 0$ (effectively flips the classifier)
   - If $\epsilon_t = 0.5$ (random), then $\alpha_t = 0$ (classifier is ignored)

2. **Weight Update**:
   - Correctly classified: $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t}$ (weight decreases)
   - Misclassified: $w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t}$ (weight increases)

## 12.4.4. Implementation

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

class AdaBoost:
    def __init__(self, n_estimators=50, max_depth=1):
        """
        AdaBoost classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of weak learners
        max_depth : int
            Maximum depth of decision tree weak learners
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
    def fit(self, X, y):
        """
        Fit AdaBoost classifier
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (should be {-1, 1})
        """
        n_samples = X.shape[0]
        
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples
        
        # Convert labels to {-1, 1} if needed
        y = np.array(y)
        if set(y) == {0, 1}:
            y = 2 * y - 1
        
        for t in range(self.n_estimators):
            # Train weak learner
            estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y
            error = np.average(incorrect, weights=sample_weights)
            
            # Handle case where error is 0 or >= 0.5
            if error <= 0:
                error = 1e-10
            elif error >= 0.5:
                error = 0.5 - 1e-10
                
            # Calculate estimator weight
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(alpha * incorrect * ((predictions != y) * 2 - 1))
            sample_weights /= np.sum(sample_weights)
            
            # Store results
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)
            
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            
        return np.sign(predictions)
    
    def staged_predict(self, X):
        """
        Return staged predictions for X
        """
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            yield np.sign(predictions)
    
    def get_feature_importance(self, X):
        """
        Get feature importance based on weighted average of weak learners
        """
        importance = np.zeros(X.shape[1])
        total_weight = sum(self.estimator_weights)
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            if hasattr(estimator, 'feature_importances_'):
                importance += (alpha / total_weight) * estimator.feature_importances_
                
        return importance

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                         n_informative=2, n_clusters_per_class=1, 
                         random_state=42, class_sep=1.5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert labels to {-1, 1}
y_train_boost = 2 * y_train - 1
y_test_boost = 2 * y_test - 1

# Train AdaBoost
ada = AdaBoost(n_estimators=50, max_depth=1)
ada.fit(X_train, y_train_boost)

# Make predictions
y_pred = ada.predict(X_test)

# Evaluate
print("AdaBoost Performance:")
print(f"Accuracy: {accuracy_score(y_test_boost, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_boost, y_pred))

# Visualize training progress
plt.figure(figsize=(15, 5))

# Plot 1: Error rates of weak learners
plt.subplot(1, 3, 1)
plt.plot(ada.estimator_errors, 'b-', label='Weak Learner Error')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing')
plt.xlabel('Iteration')
plt.ylabel('Error Rate')
plt.title('Weak Learner Error Rates')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Estimator weights
plt.subplot(1, 3, 2)
plt.plot(ada.estimator_weights, 'g-', label='Estimator Weight')
plt.xlabel('Iteration')
plt.ylabel('Weight (α)')
plt.title('Estimator Weights')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Cumulative training accuracy
plt.subplot(1, 3, 3)
train_accuracies = []
for pred in ada.staged_predict(X_train):
    train_accuracies.append(accuracy_score(y_train_boost, pred))

plt.plot(train_accuracies, 'r-', label='Training Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Decision boundary visualization
def plot_decision_boundary(X, y, model, title):
    """Plot decision boundary for 2D data"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()

plt.figure(figsize=(12, 4))

# Single decision tree
single_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
single_tree.fit(X_train, y_train_boost)

plt.subplot(1, 3, 1)
plot_decision_boundary(X_test, y_test_boost, single_tree, 'Single Decision Tree')

# AdaBoost with few iterations
ada_few = AdaBoost(n_estimators=5, max_depth=1)
ada_few.fit(X_train, y_train_boost)

plt.subplot(1, 3, 2)
plot_decision_boundary(X_test, y_test_boost, ada_few, 'AdaBoost (5 iterations)')

# AdaBoost with many iterations
plt.subplot(1, 3, 3)
plot_decision_boundary(X_test, y_test_boost, ada, 'AdaBoost (50 iterations)')

plt.tight_layout()
plt.show()
```

### R Implementation

```r
# AdaBoost implementation in R
library(rpart)
library(ggplot2)

ada_boost <- function(X, y, n_estimators = 50, max_depth = 1) {
  n_samples <- nrow(X)
  
  # Initialize weights
  sample_weights <- rep(1/n_samples, n_samples)
  
  # Convert labels to {-1, 1} if needed
  if (all(y %in% c(0, 1))) {
    y <- 2 * y - 1
  }
  
  estimators <- list()
  estimator_weights <- numeric(n_estimators)
  estimator_errors <- numeric(n_estimators)
  
  for (t in 1:n_estimators) {
    # Train weak learner (decision stump)
    formula <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
    estimator <- rpart(formula, data = data.frame(X, y), 
                      weights = sample_weights, 
                      control = rpart.control(maxdepth = max_depth))
    
    # Make predictions
    predictions <- predict(estimator, data.frame(X), type = "class")
    predictions <- as.numeric(as.character(predictions))
    
    # Calculate weighted error
    incorrect <- predictions != y
    error <- weighted.mean(incorrect, sample_weights)
    
    # Handle edge cases
    if (error <= 0) error <- 1e-10
    if (error >= 0.5) error <- 0.5 - 1e-10
    
    # Calculate estimator weight
    alpha <- 0.5 * log((1 - error) / error)
    
    # Update sample weights
    sample_weights <- sample_weights * exp(alpha * incorrect * (2 * (predictions != y) - 1))
    sample_weights <- sample_weights / sum(sample_weights)
    
    # Store results
    estimators[[t]] <- estimator
    estimator_weights[t] <- alpha
    estimator_errors[t] <- error
  }
  
  return(list(estimators = estimators, 
              estimator_weights = estimator_weights,
              estimator_errors = estimator_errors))
}

predict_ada_boost <- function(model, X) {
  predictions <- rep(0, nrow(X))
  
  for (i in seq_along(model$estimators)) {
    pred <- predict(model$estimators[[i]], data.frame(X), type = "class")
    pred <- as.numeric(as.character(pred))
    predictions <- predictions + model$estimator_weights[i] * pred
  }
  
  return(sign(predictions))
}

# Generate synthetic data
set.seed(42)
n_samples <- 1000
X <- data.frame(
  x1 = rnorm(n_samples),
  x2 = rnorm(n_samples)
)
y <- ifelse(X$x1 + X$x2 > 0, 1, 0)

# Train AdaBoost
ada_model <- ada_boost(X, y, n_estimators = 50, max_depth = 1)

# Visualize results
results_df <- data.frame(
  iteration = 1:50,
  error = ada_model$estimator_errors,
  weight = ada_model$estimator_weights
)

# Plot error rates
p1 <- ggplot(results_df, aes(x = iteration, y = error)) +
  geom_line(color = "blue") +
  geom_hline(yintercept = 0.5, color = "red", linestyle = "dashed") +
  labs(title = "Weak Learner Error Rates",
       x = "Iteration", y = "Error Rate") +
  theme_minimal()

# Plot estimator weights
p2 <- ggplot(results_df, aes(x = iteration, y = weight)) +
  geom_line(color = "green") +
  labs(title = "Estimator Weights",
       x = "Iteration", y = "Weight (α)") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)
```

## 12.4.5. Theoretical Analysis

### Training Error Bound

The key theoretical result of AdaBoost is that the training error can be bounded by:

```math
\text{Training-Err}(G_T) \leq \prod_{t=1}^T Z_t
```

where $Z_t$ is the normalization factor at iteration $t$.

### Proof of Error Bound

Let's prove this step by step:

1. **Express training error in terms of exponential loss**:
   ```math
   \text{Training-Err}(G_T) = \sum_{i=1}^n \frac{1}{n} I\left(y_i \neq \text{sign}\left(\sum_{t=1}^T \alpha_t g_t(x_i)\right)\right)
   ```

2. **Use the indicator function bound**:
   ```math
   I(z < 0) \leq e^{-z} \quad \text{for all } z \in \mathbb{R}
   ```

3. **Apply the bound**:
   ```math
   \text{Training-Err}(G_T) \leq \sum_{i=1}^n \frac{1}{n} \exp\left(-\sum_{t=1}^T \alpha_t y_i g_t(x_i)\right)
   ```

4. **Factor the exponential**:
   ```math
   \sum_{i=1}^n \frac{1}{n} \exp\left(-\sum_{t=1}^T \alpha_t y_i g_t(x_i)\right) = \sum_{i=1}^n \frac{1}{n} \prod_{t=1}^T \exp\left(-\alpha_t y_i g_t(x_i)\right)
   ```

5. **Use weight update relationship**:
   ```math
   \exp\left(-\alpha_t y_i g_t(x_i)\right) = \frac{w_i^{(t+1)}}{w_i^{(t)}} Z_t
   ```

6. **Telescope the product**:
   ```math
   \sum_{i=1}^n w_i^{(1)} \frac{w_i^{(2)}}{w_i^{(1)}} \cdots \frac{w_i^{(T+1)}}{w_i^{(T)}} \prod_{t=1}^T Z_t = \prod_{t=1}^T Z_t
   ```

### Analysis of $Z_t$

The normalization factor $Z_t$ can be expressed as:

```math
Z_t = (1 - \epsilon_t) \exp(-\alpha_t) + \epsilon_t \exp(\alpha_t)
```

Substituting $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$:

```math
Z_t = 2\sqrt{\epsilon_t(1 - \epsilon_t)}
```

**Key observations**:
- $Z_t < 1$ when $\epsilon_t \neq 0.5$
- $Z_t = 1$ when $\epsilon_t = 0.5$ (random guessing)
- The product $\prod_{t=1}^T Z_t$ decreases exponentially with $T$

## 12.4.6. Practical Considerations

### Choosing Weak Learners

1. **Decision Stumps** (depth=1): Most common choice
   - Fast to train
   - Simple interpretability
   - Often sufficient for good performance

2. **Deeper Trees**: Can capture more complex patterns
   - Risk of overfitting
   - Slower training
   - May not improve performance significantly

3. **Other Weak Learners**:
   - Linear classifiers
   - Neural networks with few hidden units
   - Any classifier that performs better than random

### Number of Iterations

1. **Too Few**: May not capture complex patterns
2. **Too Many**: Risk of overfitting
3. **Cross-Validation**: Use to find optimal number
4. **Early Stopping**: Monitor validation error

### Regularization

1. **Shrinkage**: Multiply $\alpha_t$ by a learning rate $\eta < 1$
2. **Subsampling**: Use only a fraction of data at each iteration
3. **Feature Subsampling**: Use only a subset of features

### Advantages and Disadvantages

**Advantages**:
- Simple to implement
- Few hyperparameters to tune
- Resistant to overfitting in many cases
- Can handle different types of weak learners
- Provides feature importance

**Disadvantages**:
- Sequential training (not parallelizable)
- Sensitive to noisy data
- Can overfit with too many iterations
- Computationally expensive for large datasets

## 12.4.7. Advanced Topics

### Multi-class AdaBoost

AdaBoost can be extended to multi-class problems:

1. **One-vs-All**: Train binary classifiers for each class
2. **SAMME**: Multi-class extension of AdaBoost
3. **SAMME.R**: Real-valued version of SAMME

### AdaBoost for Regression

AdaBoost can be adapted for regression:

1. **AdaBoost.R2**: Uses squared error loss
2. **AdaBoost.R**: Uses absolute error loss
3. **Gradient Boosting**: More general framework

### Connection to Other Methods

1. **Gradient Boosting**: AdaBoost is a special case with exponential loss
2. **LogitBoost**: Uses logistic loss instead of exponential loss
3. **BrownBoost**: Adaptive version that handles noisy data better

## 12.4.8. Real-World Applications

### Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load text data
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)
y = 2 * (newsgroups.target == 1) - 1  # Convert to {-1, 1}

# Train AdaBoost
ada_text = AdaBoost(n_estimators=100, max_depth=1)
ada_text.fit(X, y)

# Feature importance
feature_importance = ada_text.get_feature_importance(X)
top_features = np.argsort(feature_importance)[-10:]

print("Top 10 most important features:")
for i, idx in enumerate(reversed(top_features)):
    feature_name = vectorizer.get_feature_names_out()[idx]
    importance = feature_importance[idx]
    print(f"{i+1}. {feature_name}: {importance:.4f}")
```

### Medical Diagnosis

```python
# Example: Breast cancer diagnosis
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = 2 * cancer.target - 1  # Convert to {-1, 1}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train AdaBoost
ada_medical = AdaBoost(n_estimators=50, max_depth=1)
ada_medical.fit(X_train, y_train)

# Evaluate
y_pred = ada_medical.predict(X_test)
print("Medical Diagnosis Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Sensitivity: {accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]):.4f}")
print(f"Specificity: {accuracy_score(y_test[y_test == -1], y_pred[y_test == -1]):.4f}")

# Feature importance for medical interpretation
feature_importance = ada_medical.get_feature_importance(X_train)
top_medical_features = np.argsort(feature_importance)[-5:]

print("\nTop 5 most important medical features:")
for i, idx in enumerate(reversed(top_medical_features)):
    feature_name = cancer.feature_names[idx]
    importance = feature_importance[idx]
    print(f"{i+1}. {feature_name}: {importance:.4f}")
```

## 12.4.9. Summary

AdaBoost is a powerful and elegant boosting algorithm that:

1. **Sequentially combines weak learners** to create a strong classifier
2. **Adapts to errors** by updating instance weights
3. **Provides theoretical guarantees** on training error reduction
4. **Is simple to implement** and has few hyperparameters
5. **Works well in practice** for many classification problems

The key insights are:
- **Weight updates** focus attention on difficult examples
- **Classifier weights** $\alpha_t$ determine the contribution of each weak learner
- **Exponential loss** provides a natural way to combine predictions
- **Theoretical bounds** guarantee performance improvement under certain conditions

While AdaBoost has been largely superseded by more sophisticated methods like Gradient Boosting and XGBoost, it remains an important algorithm for understanding the principles of boosting and ensemble learning.

The algorithm's simplicity and theoretical elegance make it an excellent starting point for learning about boosting methods, and it continues to be effective for many practical applications where interpretability and ease of implementation are important considerations.
