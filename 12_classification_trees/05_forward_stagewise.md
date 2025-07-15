# 12.5. Forward Stagewise Additive Modeling

Boosting algorithms, particularly the AdaBoost algorithm, might appear mysterious due to their complex nature. To leverage the concept of boosting in various applications, it's important to understand the mathematical foundations of boosting, which is fundamentally a form of a greedy algorithm.

In the context of boosting, we're essentially looking to combine multiple functions into a stronger model. Consider an **additive model**:

```math
f(x) = \alpha_1 g_1(x) + \alpha_2 g_2(x) + \cdots + \alpha_T g_T(x)
```

where $`g_t(x)`$ is a classifier or a regression function.

It is challenging to optimize this function, since we have to consider not only the alpha values but also optimize the functions g themselves. The approach often used here is **Forward Stagewise Optimization**, which begins with a baseline of no functions, then incrementally adds to the model by optimizing one weight and one function at a time, keeping previously selected elements fixed.

## 12.5.1. Introduction to Forward Stagewise Additive Modeling

### What is Forward Stagewise Additive Modeling?

Forward Stagewise Additive Modeling (FSAM) is a general framework for building complex models by sequentially adding simple base learners. It's the mathematical foundation underlying many boosting algorithms, including AdaBoost, Gradient Boosting, and XGBoost.

### Key Principles

1. **Sequential Learning**: Models are built one at a time, each focusing on the residuals of previous models
2. **Additive Structure**: Final model is a weighted sum of base learners
3. **Greedy Optimization**: At each step, optimize only the current base learner and its weight
4. **Residual Fitting**: Each new base learner is trained to predict the residuals from previous models

### Mathematical Framework

The general form of an additive model is:

```math
f(x) = \sum_{t=1}^T \alpha_t g_t(x)
```

where:
- $`f(x)`$ is the final prediction
- $`\alpha_t`$ is the weight for the $`t`$-th base learner
- $`g_t(x)`$ is the $`t`$-th base learner (e.g., decision tree, linear model)

## 12.5.2. Forward Stagewise Optimization Algorithm

### Algorithm Overview

**Input**: Training data $`\{(x_1, y_1), \ldots, (x_n, y_n)\}`$, loss function $`L(y, f(x))`$, base learner family $`\mathcal{G}`$, number of iterations $`T`$

**Initialize**: $`f_0(x) = 0`$

**For** $`t = 1, 2, \ldots, T`$:

1. **Compute residuals**: $`r_{it} = -\frac{\partial L(y_i, f_{t-1}(x_i))}{\partial f_{t-1}(x_i)}`$
2. **Fit base learner**: $`g_t = \arg\min_{g \in \mathcal{G}} \sum_{i=1}^n (r_{it} - g(x_i))^2`$
3. **Find optimal weight**: $`\alpha_t = \arg\min_{\alpha} \sum_{i=1}^n L(y_i, f_{t-1}(x_i) + \alpha g_t(x_i))`$
4. **Update model**: $`f_t(x) = f_{t-1}(x) + \alpha_t g_t(x)`$

**Output**: Final model $`f_T(x)`$

### Why Forward Stagewise?

The key insight is that optimizing all parameters simultaneously is computationally intractable. Instead, we:

1. **Fix previous models**: Keep $`f_{t-1}(x)`$ unchanged
2. **Optimize current step**: Find best $`\alpha_t`$ and $`g_t`$ given previous models
3. **Greedy approach**: This may not be globally optimal but is computationally feasible

## 12.5.3. Connection to AdaBoost

### AdaBoost as Forward Stagewise

AdaBoost is a special case of forward stagewise additive modeling with:

1. **Exponential Loss**: $`L(y, f(x)) = \exp(-y \cdot f(x))`$
2. **Binary Classification**: $`y \in \{-1, +1\}`$
3. **Base Learners**: Weak classifiers $`g_t(x) \in \{-1, +1\}`$

### Mathematical Derivation

At iteration $`t`$, we want to minimize:

```math
\sum_{i=1}^n \exp(-y_i \cdot (f_{t-1}(x_i) + \alpha g_t(x_i)))
```

This can be rewritten as:

```math
\sum_{i=1}^n w_i^{(t)} \exp(-\alpha y_i g_t(x_i))
```

where $`w_i^{(t)} = \exp(-y_i \cdot f_{t-1}(x_i))`$ are the instance weights.

### Optimal Weight Derivation

The optimal $`\alpha_t`$ can be found in closed form:

```math
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
```

where $`\epsilon_t`$ is the weighted error rate:

```math
\epsilon_t = \sum_{i=1}^n w_i^{(t)} \cdot I(y_i \neq g_t(x_i))
```

**Proof**:
Let's minimize the exponential loss with respect to $`\alpha`$:

```math
\frac{\partial}{\partial \alpha} \sum_{i=1}^n w_i^{(t)} \exp(-\alpha y_i g_t(x_i)) = 0
```

This gives:

```math
\sum_{i=1}^n w_i^{(t)} (-y_i g_t(x_i)) \exp(-\alpha y_i g_t(x_i)) = 0
```

Splitting into correctly and incorrectly classified instances:

```math
(1 - \epsilon_t) \exp(-\alpha) - \epsilon_t \exp(\alpha) = 0
```

Solving for $`\alpha`$:

```math
\alpha = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
```

## 12.5.4. Implementation

### Python Implementation of Forward Stagewise

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns

class ForwardStagewiseAdditiveModel:
    def __init__(self, base_learner, loss_function, n_estimators=100, learning_rate=1.0):
        """
        Forward Stagewise Additive Model
        
        Parameters:
        -----------
        base_learner : estimator
            Base learner (e.g., DecisionTreeRegressor)
        loss_function : str
            Loss function ('squared_error', 'exponential', 'logistic')
        n_estimators : int
            Number of base learners
        learning_rate : float
            Learning rate (shrinkage)
        """
        self.base_learner = base_learner
        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.training_losses = []
        
    def _compute_residuals(self, y, predictions):
        """Compute residuals based on loss function"""
        if self.loss_function == 'squared_error':
            return y - predictions
        elif self.loss_function == 'exponential':
            # For exponential loss, residuals are weighted
            return -y * np.exp(-y * predictions)
        elif self.loss_function == 'logistic':
            # For logistic loss
            prob = 1 / (1 + np.exp(-predictions))
            return y - prob
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    def _find_optimal_weight(self, y, current_predictions, base_predictions):
        """Find optimal weight for the current base learner"""
        if self.loss_function == 'squared_error':
            # Closed form solution for squared error
            numerator = np.sum(base_predictions * (y - current_predictions))
            denominator = np.sum(base_predictions ** 2)
            return numerator / denominator if denominator > 0 else 0
        else:
            # Line search for other loss functions
            best_alpha = 0
            best_loss = float('inf')
            
            for alpha in np.linspace(-2, 2, 100):
                new_predictions = current_predictions + alpha * base_predictions
                if self.loss_function == 'exponential':
                    loss = np.mean(np.exp(-y * new_predictions))
                elif self.loss_function == 'logistic':
                    loss = np.mean(np.log(1 + np.exp(-y * new_predictions)))
                
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
            
            return best_alpha
    
    def fit(self, X, y):
        """Fit the forward stagewise additive model"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for t in range(self.n_estimators):
            # Compute residuals
            residuals = self._compute_residuals(y, predictions)
            
            # Fit base learner to residuals
            estimator = clone(self.base_learner)
            estimator.fit(X, residuals)
            base_predictions = estimator.predict(X)
            
            # Find optimal weight
            alpha = self._find_optimal_weight(y, predictions, base_predictions)
            alpha *= self.learning_rate  # Apply shrinkage
            
            # Update predictions
            predictions += alpha * base_predictions
            
            # Store results
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            
            # Compute training loss
            if self.loss_function == 'squared_error':
                loss = mean_squared_error(y, predictions)
            elif self.loss_function == 'exponential':
                loss = np.mean(np.exp(-y * predictions))
            elif self.loss_function == 'logistic':
                loss = np.mean(np.log(1 + np.exp(-y * predictions)))
            
            self.training_losses.append(loss)
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            
        return predictions
    
    def staged_predict(self, X):
        """Return staged predictions"""
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            yield predictions.copy()

# Example 1: Regression with Squared Error Loss
print("=== Forward Stagewise Regression ===")

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train forward stagewise model
base_learner_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
fsam_reg = ForwardStagewiseAdditiveModel(
    base_learner=base_learner_reg,
    loss_function='squared_error',
    n_estimators=50,
    learning_rate=0.1
)

fsam_reg.fit(X_train_reg, y_train_reg)

# Evaluate
y_pred_reg = fsam_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Test MSE: {mse:.4f}")

# Example 2: Classification with Exponential Loss (AdaBoost-like)
print("\n=== Forward Stagewise Classification ===")

# Generate classification data
X_clf, y_clf = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1,
                                 random_state=42, class_sep=1.5)

# Convert to {-1, 1}
y_clf = 2 * y_clf - 1

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Train forward stagewise model
base_learner_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
fsam_clf = ForwardStagewiseAdditiveModel(
    base_learner=base_learner_clf,
    loss_function='exponential',
    n_estimators=50,
    learning_rate=1.0
)

fsam_clf.fit(X_train_clf, y_train_clf)

# Evaluate
y_pred_clf = np.sign(fsam_clf.predict(X_test_clf))
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Test Accuracy: {accuracy:.4f}")

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Training loss progression
plt.subplot(1, 3, 1)
plt.plot(fsam_reg.training_losses, 'b-', label='Regression (Squared Error)')
plt.plot(fsam_clf.training_losses, 'r-', label='Classification (Exponential)')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Estimator weights
plt.subplot(1, 3, 2)
plt.plot(fsam_reg.estimator_weights, 'b-', label='Regression Weights')
plt.plot(fsam_clf.estimator_weights, 'r-', label='Classification Weights')
plt.xlabel('Iteration')
plt.ylabel('Weight (α)')
plt.title('Estimator Weights')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Cumulative predictions
plt.subplot(1, 3, 3)
reg_predictions = list(fsam_reg.staged_predict(X_test_reg))
clf_predictions = list(fsam_clf.staged_predict(X_test_clf))

plt.plot([mean_squared_error(y_test_reg, pred) for pred in reg_predictions], 
         'b-', label='Regression MSE')
plt.plot([accuracy_score(y_test_clf, np.sign(pred)) for pred in clf_predictions], 
         'r-', label='Classification Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Performance')
plt.title('Performance vs Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### R Implementation

```r
# Forward Stagewise Additive Modeling in R
library(rpart)
library(ggplot2)
library(gridExtra)

forward_stagewise_additive <- function(X, y, base_learner = "tree", 
                                      loss_function = "squared_error",
                                      n_estimators = 100, learning_rate = 1.0) {
  n_samples <- nrow(X)
  
  # Initialize
  predictions <- rep(0, n_samples)
  estimators <- list()
  estimator_weights <- numeric(n_estimators)
  training_losses <- numeric(n_estimators)
  
  for (t in 1:n_estimators) {
    # Compute residuals
    if (loss_function == "squared_error") {
      residuals <- y - predictions
    } else if (loss_function == "exponential") {
      residuals <- -y * exp(-y * predictions)
    } else if (loss_function == "logistic") {
      prob <- 1 / (1 + exp(-predictions))
      residuals <- y - prob
    }
    
    # Fit base learner
    if (base_learner == "tree") {
      formula <- as.formula(paste("residuals ~", paste(colnames(X), collapse = " + ")))
      estimator <- rpart(formula, data = data.frame(X, residuals), 
                        control = rpart.control(maxdepth = 3))
      base_predictions <- predict(estimator, data.frame(X))
    }
    
    # Find optimal weight
    if (loss_function == "squared_error") {
      numerator <- sum(base_predictions * (y - predictions))
      denominator <- sum(base_predictions^2)
      alpha <- ifelse(denominator > 0, numerator / denominator, 0)
    } else {
      # Line search for other loss functions
      best_alpha <- 0
      best_loss <- Inf
      
      for (alpha_candidate in seq(-2, 2, length.out = 100)) {
        new_predictions <- predictions + alpha_candidate * base_predictions
        
        if (loss_function == "exponential") {
          loss <- mean(exp(-y * new_predictions))
        } else if (loss_function == "logistic") {
          loss <- mean(log(1 + exp(-y * new_predictions)))
        }
        
        if (loss < best_loss) {
          best_loss <- loss
          best_alpha <- alpha_candidate
        }
      }
      alpha <- best_alpha
    }
    
    # Apply learning rate
    alpha <- alpha * learning_rate
    
    # Update predictions
    predictions <- predictions + alpha * base_predictions
    
    # Store results
    estimators[[t]] <- estimator
    estimator_weights[t] <- alpha
    
    # Compute training loss
    if (loss_function == "squared_error") {
      training_losses[t] <- mean((y - predictions)^2)
    } else if (loss_function == "exponential") {
      training_losses[t] <- mean(exp(-y * predictions))
    } else if (loss_function == "logistic") {
      training_losses[t] <- mean(log(1 + exp(-y * predictions)))
    }
  }
  
  return(list(estimators = estimators,
              estimator_weights = estimator_weights,
              training_losses = training_losses,
              final_predictions = predictions))
}

predict_fsam <- function(model, X) {
  predictions <- rep(0, nrow(X))
  
  for (i in seq_along(model$estimators)) {
    pred <- predict(model$estimators[[i]], data.frame(X))
    predictions <- predictions + model$estimator_weights[i] * pred
  }
  
  return(predictions)
}

# Generate synthetic data
set.seed(42)
n_samples <- 1000

# Regression data
X_reg <- data.frame(
  x1 = rnorm(n_samples),
  x2 = rnorm(n_samples)
)
y_reg <- 2 * X_reg$x1 + 3 * X_reg$x2 + rnorm(n_samples, 0, 0.1)

# Classification data
X_clf <- data.frame(
  x1 = rnorm(n_samples),
  x2 = rnorm(n_samples)
)
y_clf <- ifelse(X_clf$x1 + X_clf$x2 > 0, 1, -1)

# Train models
fsam_reg <- forward_stagewise_additive(X_reg, y_reg, "tree", "squared_error", 50, 0.1)
fsam_clf <- forward_stagewise_additive(X_clf, y_clf, "tree", "exponential", 50, 1.0)

# Create visualization
results_df <- data.frame(
  iteration = rep(1:50, 2),
  loss = c(fsam_reg$training_losses, fsam_clf$training_losses),
  weight = c(fsam_reg$estimator_weights, fsam_clf$estimator_weights),
  type = rep(c("Regression", "Classification"), each = 50)
)

# Plot training losses
p1 <- ggplot(results_df, aes(x = iteration, y = loss, color = type)) +
  geom_line() +
  labs(title = "Training Loss vs Iterations",
       x = "Iteration", y = "Training Loss") +
  theme_minimal()

# Plot estimator weights
p2 <- ggplot(results_df, aes(x = iteration, y = weight, color = type)) +
  geom_line() +
  labs(title = "Estimator Weights",
       x = "Iteration", y = "Weight (α)") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)
```

## 12.5.5. Mathematical Analysis

### Loss Functions and Their Properties

#### 1. Squared Error Loss

```math
L(y, f(x)) = \frac{1}{2}(y - f(x))^2
```

**Properties**:
- Convex and differentiable
- Sensitive to outliers
- Closed-form solution for optimal weight
- Residuals: $`r_i = y_i - f(x_i)`$

#### 2. Exponential Loss

```math
L(y, f(x)) = \exp(-y \cdot f(x))
```

**Properties**:
- Heavily penalizes misclassifications
- Used in AdaBoost
- Can lead to overfitting
- Residuals: $`r_i = -y_i \exp(-y_i \cdot f(x_i))`$

#### 3. Logistic Loss

```math
L(y, f(x)) = \log(1 + \exp(-y \cdot f(x)))
```

**Properties**:
- More robust than exponential loss
- Used in LogitBoost
- Better theoretical properties
- Residuals: $`r_i = y_i - \frac{1}{1 + \exp(-f(x_i))}`$

### Convergence Analysis

#### Training Loss Convergence

Under certain conditions, the training loss converges to a local minimum:

```math
\lim_{T \to \infty} \frac{1}{n} \sum_{i=1}^n L(y_i, f_T(x_i)) = L^*
```

where $`L^*`$ is the minimum achievable loss.

#### Rate of Convergence

The convergence rate depends on the loss function and base learner:

1. **Squared Error**: Linear convergence under strong convexity
2. **Exponential**: Exponential convergence but risk of overfitting
3. **Logistic**: Linear convergence with better generalization

### Regularization

#### Learning Rate (Shrinkage)

Multiply the optimal weight by a learning rate $`\eta < 1`$:

```math
\alpha_t = \eta \cdot \arg\min_{\alpha} \sum_{i=1}^n L(y_i, f_{t-1}(x_i) + \alpha g_t(x_i))
```

**Benefits**:
- Slower convergence but better generalization
- Reduces overfitting
- More stable training

#### Subsampling

Use only a fraction of data at each iteration:

```math
\mathcal{S}_t \subset \{1, 2, \ldots, n\}, \quad |\mathcal{S}_t| = \lfloor \rho n \rfloor
```

where $`\rho \in (0, 1]`$ is the subsampling ratio.

## 12.5.6. Comparison with Other Methods

### Forward Stagewise vs. Backward Elimination

| Aspect | Forward Stagewise | Backward Elimination |
|--------|-------------------|---------------------|
| **Direction** | Add variables one by one | Remove variables one by one |
| **Computational Cost** | $`O(T \cdot \text{cost}(g))`$ | $`O(p \cdot \text{cost}(g))`$ |
| **Optimality** | Greedy, not globally optimal | Greedy, not globally optimal |
| **Interpretability** | Natural ordering of importance | Natural ordering of importance |

### Forward Stagewise vs. Gradient Boosting

| Aspect | Forward Stagewise | Gradient Boosting |
|--------|-------------------|-------------------|
| **Optimization** | Line search for $`\alpha_t`$ | Gradient descent |
| **Flexibility** | Any loss function | Any differentiable loss |
| **Computational Cost** | Higher (line search) | Lower (gradient computation) |
| **Theoretical Guarantees** | Limited | Strong convergence results |

### Forward Stagewise vs. AdaBoost

| Aspect | Forward Stagewise | AdaBoost |
|--------|-------------------|----------|
| **Loss Function** | Any loss function | Exponential loss only |
| **Base Learners** | Any learner | Weak classifiers |
| **Weight Update** | Line search | Closed form |
| **Application** | Regression and classification | Classification only |

## 12.5.7. Advanced Topics

### Multi-class Extension

For $`K`$ classes, extend to:

```math
f_k(x) = \sum_{t=1}^T \alpha_t g_{tk}(x), \quad k = 1, 2, \ldots, K
```

where $`g_{tk}(x)`$ predicts the $`k`$-th class.

### Robust Loss Functions

#### Huber Loss

```math
L(y, f(x)) = \begin{cases}
\frac{1}{2}(y - f(x))^2 & \text{if } |y - f(x)| \leq \delta \\
\delta|y - f(x)| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
```

#### Quantile Loss

```math
L(y, f(x)) = \rho_\tau(y - f(x))
```

where $`\rho_\tau(u) = u(\tau - I(u < 0))`$ for quantile $`\tau`$.

### Feature Importance

Compute feature importance as weighted average:

```math
\text{Importance}(j) = \sum_{t=1}^T |\alpha_t| \cdot \text{Importance}_t(j)
```

where $`\text{Importance}_t(j)`$ is the importance of feature $`j`$ in base learner $`t`$.

## 12.5.8. Practical Considerations

### Hyperparameter Tuning

1. **Number of Iterations** ($`T`$):
   - Too few: Underfitting
   - Too many: Overfitting
   - Use cross-validation

2. **Learning Rate** ($`\eta`$):
   - Smaller values: Better generalization, slower convergence
   - Larger values: Faster convergence, risk of overfitting
   - Typical range: $`[0.01, 0.3]`$

3. **Base Learner Complexity**:
   - Simpler learners: More iterations needed, better generalization
   - Complex learners: Fewer iterations, risk of overfitting

### Computational Efficiency

1. **Early Stopping**: Monitor validation loss
2. **Subsampling**: Use fraction of data per iteration
3. **Parallelization**: Train base learners in parallel
4. **Memory Management**: Store only necessary information

### Model Interpretation

1. **Feature Importance**: Weighted average across base learners
2. **Partial Dependencies**: Effect of individual features
3. **Interaction Effects**: Captured by tree-based base learners
4. **Model Complexity**: Number of base learners and their complexity

## 12.5.9. Real-World Applications

### Financial Risk Modeling

```python
# Example: Credit risk prediction
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Simulate financial data
np.random.seed(42)
n_samples = 10000

# Features: income, age, credit_score, debt_ratio, payment_history
X_fin = pd.DataFrame({
    'income': np.random.lognormal(10, 0.5, n_samples),
    'age': np.random.normal(45, 15, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'payment_history': np.random.poisson(2, n_samples)
})

# Target: default (1) or not (0)
y_fin = (X_fin['debt_ratio'] > 0.4) | (X_fin['credit_score'] < 600)
y_fin = 2 * y_fin.astype(int) - 1  # Convert to {-1, 1}

# Train forward stagewise model
base_learner_fin = DecisionTreeClassifier(max_depth=4, random_state=42)
fsam_fin = ForwardStagewiseAdditiveModel(
    base_learner=base_learner_fin,
    loss_function='exponential',
    n_estimators=100,
    learning_rate=0.1
)

fsam_fin.fit(X_fin, y_fin)

# Feature importance analysis
feature_importance = np.zeros(X_fin.shape[1])
total_weight = sum(fsam_fin.estimator_weights)

for alpha, estimator in zip(fsam_fin.estimator_weights, fsam_fin.estimators):
    if hasattr(estimator, 'feature_importances_'):
        feature_importance += (alpha / total_weight) * estimator.feature_importances_

# Display feature importance
feature_names = X_fin.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Feature Importance for Credit Risk:")
print(importance_df)
```

### Medical Diagnosis

```python
# Example: Disease prediction
from sklearn.datasets import load_breast_cancer

# Load medical data
cancer = load_breast_cancer()
X_med = cancer.data
y_med = 2 * cancer.target - 1  # Convert to {-1, 1}

# Train forward stagewise model
base_learner_med = DecisionTreeClassifier(max_depth=3, random_state=42)
fsam_med = ForwardStagewiseAdditiveModel(
    base_learner=base_learner_med,
    loss_function='logistic',  # More robust than exponential
    n_estimators=50,
    learning_rate=0.1
)

fsam_med.fit(X_med, y_med)

# Model evaluation
y_pred_med = np.sign(fsam_med.predict(X_med))
accuracy = accuracy_score(y_med, y_pred_med)

print(f"Medical Diagnosis Accuracy: {accuracy:.4f}")

# Analyze model stability
staged_predictions = list(fsam_med.staged_predict(X_med))
staged_accuracies = [accuracy_score(y_med, np.sign(pred)) for pred in staged_predictions]

plt.figure(figsize=(10, 6))
plt.plot(staged_accuracies, 'b-', linewidth=2)
plt.axhline(y=accuracy, color='r', linestyle='--', label='Final Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Model Convergence in Medical Diagnosis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 12.5.10. Summary

Forward Stagewise Additive Modeling is a powerful and flexible framework that:

1. **Provides a unified view** of many boosting algorithms
2. **Offers mathematical foundation** for understanding boosting
3. **Enables flexible loss functions** beyond exponential loss
4. **Supports various base learners** (trees, linear models, etc.)
5. **Provides interpretable models** with feature importance

### Key Insights

- **Sequential optimization** makes complex problems tractable
- **Residual fitting** focuses each base learner on current errors
- **Weight optimization** ensures optimal contribution of each base learner
- **Regularization** (learning rate, subsampling) improves generalization

### When to Use Forward Stagewise

**Advantages**:
- Flexible loss functions
- Interpretable models
- Theoretical foundation
- Good performance on many problems

**Disadvantages**:
- Computationally expensive (line search)
- Sequential training (not parallelizable)
- May require more tuning than specialized algorithms

### Modern Context

While forward stagewise additive modeling provides the theoretical foundation, modern implementations often use:

1. **Gradient Boosting**: More efficient optimization
2. **XGBoost**: Advanced regularization and optimization
3. **LightGBM**: Gradient-based with efficient tree building
4. **CatBoost**: Specialized for categorical features

However, understanding forward stagewise additive modeling remains crucial for:
- **Algorithm design**: Developing new boosting methods
- **Model interpretation**: Understanding how boosting works
- **Hyperparameter tuning**: Making informed choices
- **Troubleshooting**: Diagnosing model issues

The framework continues to be relevant for both theoretical understanding and practical applications in machine learning.
