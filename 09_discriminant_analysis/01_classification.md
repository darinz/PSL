# 9.1. Introduction to Classification

## 9.1.1. What is Classification?

Classification is a fundamental supervised learning problem where we predict categorical labels (classes) for new observations based on patterns learned from labeled training data. Unlike regression, which predicts continuous numerical values, classification deals with discrete outcomes.

### Problem Formulation

Consider a dataset with $`n`$ observations, each consisting of $`p`$ features or measurements. These observations belong to different distinct classes. In **binary classification**, we have exactly two classes, typically labeled as 0 and 1.

**Mathematical Setup**:
- **Features**: $`X \in \mathbb{R}^p`$ (p-dimensional feature vector)
- **Target**: $`Y \in \{0, 1\}`$ (binary class label)
- **Training Data**: $`\{(x_i, y_i)\}_{i=1}^n`$ where $`x_i \in \mathbb{R}^p`$ and $`y_i \in \{0, 1\}`$

### Real-World Examples

1. **Credit Risk Assessment**: Predict whether a loan applicant will default (Y=1) or repay (Y=0) based on features like income, credit score, employment history, etc.

2. **Medical Diagnosis**: Classify patients as having a disease (Y=1) or being healthy (Y=0) based on symptoms, test results, and medical history.

3. **Spam Detection**: Determine if an email is spam (Y=1) or legitimate (Y=0) using features like sender information, content analysis, and metadata.

4. **Sentiment Analysis**: Classify text as positive (Y=1) or negative (Y=0) sentiment based on word frequencies and linguistic features.

### Classification vs. Regression

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Output** | Discrete classes | Continuous values |
| **Goal** | Predict class labels | Predict numerical values |
| **Evaluation** | Accuracy, precision, recall | MSE, MAE, R² |
| **Algorithms** | Logistic regression, SVM, Random Forest | Linear regression, Ridge, Lasso |

## 9.1.2. The Classification Framework

### Step-by-Step Process

#### 1. Data Collection and Preprocessing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Example: Credit Risk Dataset
def create_credit_dataset(n_samples=1000, random_state=42):
    """Create synthetic credit risk dataset"""
    np.random.seed(random_state)
    
    # Generate features
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    debt_ratio = np.random.beta(2, 5, n_samples) * 2  # 0 to 2
    employment_years = np.random.exponential(5, n_samples)
    
    # Create feature matrix
    X = np.column_stack([income, credit_score, debt_ratio, employment_years])
    
    # Generate target based on features (with some noise)
    risk_score = (0.3 * (income - 50000) / 20000 + 
                  0.4 * (credit_score - 700) / 100 + 
                  0.2 * (debt_ratio - 1) + 
                  0.1 * (employment_years - 5) / 5)
    
    # Add noise and threshold
    risk_score += np.random.normal(0, 0.2, n_samples)
    y = (risk_score > 0).astype(int)
    
    return X, y

# Create dataset
X, y = create_credit_dataset()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dataset shape:", X.shape)
print("Class distribution:")
print(pd.Series(y).value_counts(normalize=True))
```

```r
# R implementation
library(ggplot2)
library(caret)
library(dplyr)

# Create synthetic credit dataset
create_credit_dataset <- function(n_samples = 1000, random_state = 42) {
  set.seed(random_state)
  
  # Generate features
  income <- rnorm(n_samples, mean = 50000, sd = 20000)
  credit_score <- rnorm(n_samples, mean = 700, sd = 100)
  debt_ratio <- rbeta(n_samples, 2, 5) * 2
  employment_years <- rexp(n_samples, rate = 1/5)
  
  # Create feature matrix
  X <- data.frame(
    income = income,
    credit_score = credit_score,
    debt_ratio = debt_ratio,
    employment_years = employment_years
  )
  
  # Generate target
  risk_score <- 0.3 * (income - 50000) / 20000 + 
                0.4 * (credit_score - 700) / 100 + 
                0.2 * (debt_ratio - 1) + 
                0.1 * (employment_years - 5) / 5
  
  risk_score <- risk_score + rnorm(n_samples, 0, 0.2)
  y <- as.factor(ifelse(risk_score > 0, 1, 0))
  
  return(list(X = X, y = y))
}

# Create dataset
data <- create_credit_dataset()
X <- data$X
y <- data$y

# Split data
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Standardize features
preprocess_params <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preprocess_params, X_train)
X_test_scaled <- predict(preprocess_params, X_test)

cat("Dataset shape:", nrow(X), "x", ncol(X), "\n")
cat("Class distribution:\n")
print(table(y) / length(y))
```

#### 2. Function Selection: Classification Models

A classification function $`f: \mathbb{R}^p \rightarrow \{0, 1\}`$ maps feature vectors to class labels. Different algorithms provide different functional forms:

```python
# Example classification functions
class ClassificationModels:
    def __init__(self):
        pass
    
    def linear_classifier(self, X, w, b):
        """Linear classifier: f(x) = sign(w^T x + b)"""
        scores = np.dot(X, w) + b
        return (scores > 0).astype(int)
    
    def logistic_classifier(self, X, w, b):
        """Logistic classifier: f(x) = 1 if P(Y=1|X) > 0.5"""
        scores = np.dot(X, w) + b
        probabilities = 1 / (1 + np.exp(-scores))
        return (probabilities > 0.5).astype(int)
    
    def nearest_neighbor_classifier(self, X_train, y_train, X_test, k=1):
        """k-NN classifier"""
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        return knn.predict(X_test)
    
    def decision_tree_classifier(self, X_train, y_train, X_test, max_depth=3):
        """Decision tree classifier"""
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X_train, y_train)
        return tree.predict(X_test)

# Example usage
models = ClassificationModels()

# Linear classifier
w = np.array([0.1, -0.2, 0.3, -0.1])
b = 0.5
linear_predictions = models.linear_classifier(X_test_scaled, w, b)

# k-NN classifier
knn_predictions = models.nearest_neighbor_classifier(X_train_scaled, y_train, X_test_scaled, k=5)

# Decision tree classifier
tree_predictions = models.decision_tree_classifier(X_train_scaled, y_train, X_test_scaled)
```

#### 3. Loss Functions for Classification

The loss function $`L(f(x), y)`$ quantifies the cost of prediction errors:

```python
class ClassificationLoss:
    def __init__(self):
        pass
    
    def zero_one_loss(self, y_pred, y_true):
        """0-1 Loss: L(f(x), y) = 0 if y = f(x), 1 otherwise"""
        return np.mean(y_pred != y_true)
    
    def hinge_loss(self, scores, y_true):
        """Hinge loss for SVM: L = max(0, 1 - y * score)"""
        y_true_binary = 2 * y_true - 1  # Convert to {-1, 1}
        return np.mean(np.maximum(0, 1 - y_true_binary * scores))
    
    def logistic_loss(self, scores, y_true):
        """Logistic loss: L = -log(P(Y=y|X))"""
        probabilities = 1 / (1 + np.exp(-scores))
        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(probabilities) + 
                       (1 - y_true) * np.log(1 - probabilities))
    
    def cross_entropy_loss(self, probabilities, y_true):
        """Cross-entropy loss for multi-class"""
        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(probabilities), axis=1))

# Example usage
loss_functions = ClassificationLoss()

# Calculate different losses
scores = np.dot(X_test_scaled, w) + b
probabilities = 1 / (1 + np.exp(-scores))

print("0-1 Loss:", loss_functions.zero_one_loss(linear_predictions, y_test))
print("Hinge Loss:", loss_functions.hinge_loss(scores, y_test))
print("Logistic Loss:", loss_functions.logistic_loss(scores, y_test))
```

#### 4. Optimization: Finding the Best Classifier

The goal is to minimize the empirical risk:

```math
\min_f \frac{1}{n} \sum_{i=1}^n L(f(x_i), y_i)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ClassificationOptimization:
    def __init__(self):
        pass
    
    def optimize_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Optimize logistic regression using sklearn"""
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
        
        return {
            'model': lr,
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def optimize_svm(self, X_train, y_train, X_test, y_test):
        """Optimize SVM classifier"""
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        
        y_pred = svm.predict(X_test)
        y_prob = svm.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
        
        return {
            'model': svm,
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def optimize_random_forest(self, X_train, y_train, X_test, y_test):
        """Optimize random forest classifier"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
        
        return {
            'model': rf,
            'predictions': y_pred,
            'probabilities': y_prob,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

# Optimize different classifiers
optimizer = ClassificationOptimization()

lr_results = optimizer.optimize_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
svm_results = optimizer.optimize_svm(X_train_scaled, y_train, X_test_scaled, y_test)
rf_results = optimizer.optimize_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

# Compare results
print("Model Comparison:")
print(f"Logistic Regression: {lr_results['accuracy']:.3f} (CV: {lr_results['cv_mean']:.3f} ± {lr_results['cv_std']:.3f})")
print(f"SVM: {svm_results['accuracy']:.3f} (CV: {svm_results['cv_mean']:.3f} ± {svm_results['cv_std']:.3f})")
print(f"Random Forest: {rf_results['accuracy']:.3f} (CV: {rf_results['cv_mean']:.3f} ± {rf_results['cv_std']:.3f})")
```

## 9.1.3. The Bayes Optimal Classifier

### Theoretical Foundation

In the ideal scenario with infinite data, we can derive the optimal classifier that minimizes the expected loss (risk).

#### Risk Function

The risk function is the expected loss over the true data distribution:

```math
\text{Risk}[f] = \mathbb{E}_{X, Y} L(f(X), Y) = \int_{\mathcal{X}} \int_{\mathcal{Y}} L(y, f(x)) p(x, y) dy dx
```

#### Factorization and Optimization

Using the law of total probability, we can factorize the joint distribution:

```math
p(x, y) = p(y \mid x) p(x)
```

This allows us to rewrite the risk function as:

```math
\text{Risk}[f] = \int_{\mathcal{X}} \left[ \int_{\mathcal{Y}} L(y, f(x)) p(y \mid x) dy \right] p(x) dx
```

The key insight is that we can minimize the risk by minimizing the conditional expected loss at each point $`x`$.

#### Binary Classification with 0-1 Loss

For binary classification with 0-1 loss, the conditional expected loss becomes:

```math
\mathbb{E}_{Y \mid X=x} L(y, f(x)) = L(1, f(x)) \cdot P(Y=1 \mid x) + L(0, f(x)) \cdot P(Y=0 \mid x)
```

Let $`\eta(x) = P(Y=1 \mid x)`$ be the conditional probability of class 1. Then:

```math
\mathbb{E}_{Y \mid X=x} L(y, f(x)) = \begin{cases}
\eta(x), & \text{if } f(x) = 0 \\
1 - \eta(x), & \text{if } f(x) = 1
\end{cases}
```

#### Bayes Optimal Rule

The optimal classifier minimizes this conditional expected loss:

```math
f^*(x) = \arg\min_{f(x)} \mathbb{E}_{Y \mid X=x} L(y, f(x)) = \begin{cases}
1, & \text{if } \eta(x) \geq 0.5 \\
0, & \text{if } \eta(x) < 0.5
\end{cases}
```

This is the **Bayes optimal classifier** or **Bayes rule**.

### Implementation: Bayes Optimal Classifier

```python
class BayesOptimalClassifier:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        """Estimate P(Y=1|X) using kernel density estimation"""
        from sklearn.neighbors import KernelDensity
        
        # Separate data by class
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]
        
        # Estimate class-conditional densities
        self.kde_0 = KernelDensity(bandwidth=0.5, kernel='gaussian')
        self.kde_1 = KernelDensity(bandwidth=0.5, kernel='gaussian')
        
        self.kde_0.fit(X_class_0)
        self.kde_1.fit(X_class_1)
        
        # Estimate class priors
        self.prior_0 = len(X_class_0) / len(X)
        self.prior_1 = len(X_class_1) / len(X)
        
        return self
    
    def predict_proba(self, X):
        """Estimate P(Y=1|X) using Bayes rule"""
        # Get log densities
        log_dens_0 = self.kde_0.score_samples(X)
        log_dens_1 = self.kde_1.score_samples(X)
        
        # Convert to densities
        dens_0 = np.exp(log_dens_0)
        dens_1 = np.exp(log_dens_1)
        
        # Apply Bayes rule
        numerator = dens_1 * self.prior_1
        denominator = dens_0 * self.prior_0 + dens_1 * self.prior_1
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)
        
        return numerator / denominator
    
    def predict(self, X):
        """Predict class labels using Bayes optimal rule"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

# Example usage
bayes_classifier = BayesOptimalClassifier()
bayes_classifier.fit(X_train_scaled, y_train)

bayes_predictions = bayes_classifier.predict(X_test_scaled)
bayes_probabilities = bayes_classifier.predict_proba(X_test_scaled)

print("Bayes Optimal Classifier Accuracy:", accuracy_score(y_test, bayes_predictions))
```

### Multi-Class Extension

For $`K`$ classes, the Bayes optimal classifier predicts:

```math
f^*(x) = \arg\max_{k \in \{1, \ldots, K\}} P(Y=k \mid X=x)
```

```python
def multi_class_bayes_optimal(X, y, X_test):
    """Multi-class Bayes optimal classifier"""
    from sklearn.naive_bayes import GaussianNB
    
    gnb = GaussianNB()
    gnb.fit(X, y)
    
    predictions = gnb.predict(X_test)
    probabilities = gnb.predict_proba(X_test)
    
    return predictions, probabilities

# Example with 3 classes
def create_multi_class_dataset(n_samples=1000, n_classes=3):
    """Create synthetic multi-class dataset"""
    np.random.seed(42)
    
    # Generate features from different Gaussian distributions
    X = np.vstack([
        np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//3),
        np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], n_samples//3),
        np.random.multivariate_normal([-2, 2], [[0.5, 0], [0, 0.5]], n_samples//3)
    ])
    
    y = np.repeat([0, 1, 2], n_samples//3)
    
    return X, y

# Multi-class example
X_multi, y_multi = create_multi_class_dataset()
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

multi_predictions, multi_probabilities = multi_class_bayes_optimal(
    X_train_multi, y_train_multi, X_test_multi
)

print("Multi-class Bayes Optimal Accuracy:", accuracy_score(y_test_multi, multi_predictions))
```

## 9.1.4. Decision Boundaries and Visualization

### Understanding Decision Boundaries

A decision boundary is the set of points where the classifier is indifferent between classes. For the Bayes optimal classifier, the decision boundary is where $`\eta(x) = 0.5`$.

```python
def plot_decision_boundaries(X, y, classifiers, titles):
    """Plot decision boundaries for different classifiers"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    for i, (classifier, title) in enumerate(zip(classifiers, titles)):
        # Fit classifier
        classifier.fit(X, y)
        
        # Predict on mesh grid
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[i].contourf(xx, yy, Z, alpha=0.4)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k')
        axes[i].set_title(title)
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

# Create 2D dataset for visualization
def create_2d_dataset(n_samples=300):
    """Create 2D dataset for visualization"""
    np.random.seed(42)
    
    # Generate two classes with different distributions
    X_class_0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
    X_class_1 = np.random.multivariate_normal([2, 2], [[1, -0.5], [-0.5, 1]], n_samples//2)
    
    X = np.vstack([X_class_0, X_class_1])
    y = np.repeat([0, 1], n_samples//2)
    
    return X, y

# Create dataset and classifiers
X_2d, y_2d = create_2d_dataset()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    LogisticRegression(random_state=42),
    SVC(kernel='rbf', random_state=42),
    DecisionTreeClassifier(max_depth=3, random_state=42),
    BayesOptimalClassifier()
]

titles = [
    'Logistic Regression (Linear)',
    'SVM with RBF Kernel (Non-linear)',
    'Decision Tree (Piecewise Linear)',
    'Bayes Optimal Classifier'
]

# Plot decision boundaries
plot_decision_boundaries(X_2d, y_2d, classifiers, titles)
```

### Linear vs. Non-linear Decision Boundaries

```python
def compare_linear_nonlinear():
    """Compare linear and non-linear classifiers"""
    # Create non-linearly separable dataset
    np.random.seed(42)
    n_samples = 200
    
    # Generate circular dataset
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r_inner = np.random.normal(2, 0.3, n_samples//2)
    r_outer = np.random.normal(4, 0.3, n_samples//2)
    
    X_inner = np.column_stack([r_inner * np.cos(theta[:n_samples//2]), 
                              r_inner * np.sin(theta[:n_samples//2])])
    X_outer = np.column_stack([r_outer * np.cos(theta[n_samples//2:]), 
                              r_outer * np.sin(theta[n_samples//2:])])
    
    X_circular = np.vstack([X_inner, X_outer])
    y_circular = np.repeat([0, 1], n_samples//2)
    
    # Compare classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    lr = LogisticRegression(random_state=42)
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_rbf = SVC(kernel='rbf', random_state=42)
    
    classifiers = [lr, svm_linear, svm_rbf]
    titles = ['Logistic Regression', 'SVM (Linear)', 'SVM (RBF)']
    
    plot_decision_boundaries(X_circular, y_circular, classifiers, titles)
    
    # Print accuracies
    for classifier, title in zip(classifiers, titles):
        classifier.fit(X_circular, y_circular)
        accuracy = classifier.score(X_circular, y_circular)
        print(f"{title} Accuracy: {accuracy:.3f}")

# Run comparison
compare_linear_nonlinear()
```

## 9.1.5. Evaluation Metrics

### Classification Performance Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)

class ClassificationEvaluator:
    def __init__(self):
        pass
    
    def evaluate_classifier(self, y_true, y_pred, y_prob=None):
        """Comprehensive evaluation of a classifier"""
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred)
        results['recall'] = recall_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        if y_prob is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_prob, title="Precision-Recall Curve"):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Evaluate our classifiers
evaluator = ClassificationEvaluator()

# Evaluate logistic regression
lr_eval = evaluator.evaluate_classifier(y_test, lr_results['predictions'], lr_results['probabilities'])
print("Logistic Regression Results:")
for metric, value in lr_eval.items():
    if metric != 'confusion_matrix':
        print(f"{metric}: {value:.3f}")

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_test, lr_results['predictions'], "Logistic Regression")

# Plot ROC curve
evaluator.plot_roc_curve(y_test, lr_results['probabilities'], "Logistic Regression ROC")

# Plot precision-recall curve
evaluator.plot_precision_recall_curve(y_test, lr_results['probabilities'], "Logistic Regression PR")
```

## 9.1.6. Practical Considerations

### Class Imbalance

```python
def handle_class_imbalance():
    """Demonstrate handling of class imbalance"""
    # Create imbalanced dataset
    np.random.seed(42)
    n_samples = 1000
    
    # 90% class 0, 10% class 1
    X_imb = np.random.randn(n_samples, 2)
    y_imb = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Add some signal
    X_imb[y_imb == 1] += np.array([1, 1])
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
    )
    
    # Standard classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    lr_imb = LogisticRegression(random_state=42)
    rf_imb = RandomForestClassifier(random_state=42)
    
    lr_imb.fit(X_train_imb, y_train_imb)
    rf_imb.fit(X_train_imb, y_train_imb)
    
    lr_pred = lr_imb.predict(X_test_imb)
    rf_pred = rf_imb.predict(X_test_imb)
    
    print("Imbalanced Dataset Results:")
    print(f"Class distribution: {np.bincount(y_test_imb)}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test_imb, lr_pred):.3f}")
    print(f"Random Forest Accuracy: {accuracy_score(y_test_imb, rf_pred):.3f}")
    
    # Handle imbalance with class weights
    lr_weighted = LogisticRegression(class_weight='balanced', random_state=42)
    rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    lr_weighted.fit(X_train_imb, y_train_imb)
    rf_weighted.fit(X_train_imb, y_train_imb)
    
    lr_w_pred = lr_weighted.predict(X_test_imb)
    rf_w_pred = rf_weighted.predict(X_test_imb)
    
    print("\nWith Class Weights:")
    print(f"Logistic Regression F1: {f1_score(y_test_imb, lr_w_pred):.3f}")
    print(f"Random Forest F1: {f1_score(y_test_imb, rf_w_pred):.3f}")

# Run imbalance handling example
handle_class_imbalance()
```

### Feature Importance and Interpretability

```python
def analyze_feature_importance():
    """Analyze feature importance in classification"""
    # Use our credit dataset
    feature_names = ['Income', 'Credit_Score', 'Debt_Ratio', 'Employment_Years']
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    # Logistic regression coefficients
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    plt.figure(figsize=(10, 6))
    coef = lr.coef_[0]
    indices = np.argsort(np.abs(coef))[::-1]
    
    plt.bar(range(len(coef)), coef[indices])
    plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Coefficients (Logistic Regression)')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.show()

# Analyze feature importance
analyze_feature_importance()
```

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of classification concepts. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and handling of common challenges in classification problems.
