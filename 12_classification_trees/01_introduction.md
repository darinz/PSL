# 12.1. Introduction

Classification trees are a fundamental machine learning technique that extends the concept of decision trees from regression to classification problems. Just as in our previous discussion about regression trees, when it comes to classification trees, we must also focus on three essential aspects:

## 12.1.1. The Three Key Components

### 1. Where to Split

This involves deciding on the variable (denoted as $`j`$) and the split value ($`s`$) that divides our data into two parts, based on whether $`X_j < s`$ or not.

**Mathematical Formulation**: For a feature $`j`$ and split point $`s`$, we create two regions:
```math
R_1(j, s) = \{X | X_j \leq s\} \quad \text{and} \quad R_2(j, s) = \{X | X_j > s\}
```

**Key Considerations**:
- **Feature Selection**: Which variable provides the best split?
- **Split Point**: What threshold value maximizes separation?
- **Binary Splits**: Each split creates exactly two child nodes

### 2. When to Stop

As previously discussed, the general strategy is to initially construct a large tree and then employ a pruning process based on a loss plus penalty criteria. This strategy helps prevent overfitting.

**Stopping Criteria**:
- **Minimum node size**: Stop when node contains fewer than $`n_{\min}`$ samples
- **Maximum depth**: Stop when tree reaches maximum depth $`d_{\max}`$
- **Pure nodes**: Stop when all samples in node belong to same class
- **Minimum improvement**: Stop when split improvement is below threshold

**Pruning Strategy**:
```math
\text{Cost}(T) = \text{Loss}(T) + \alpha \cdot \text{Complexity}(T)
```

where $`\alpha`$ is the regularization parameter controlling tree size.

### 3. How to Predict at Each Leaf Node

Depending on whether we are dealing with regression or classification, we adopt different approaches for making predictions at leaf nodes.

#### Regression Trees
For regression, at each leaf node, we calculate the average Y value based on the training samples within that node:
```math
\hat{y}_{\text{leaf}} = \frac{1}{n_{\text{leaf}}} \sum_{i \in \text{leaf}} y_i
```

#### Classification Trees
For classification, we apply a similar concept. When a leaf node contains observations from $`K`$ classes, we can either:

**Majority Voting**:
```math
\hat{y}_{\text{leaf}} = \arg\max_{k} n_k
```

where $`n_k`$ is the number of samples of class $`k`$ in the leaf.

**Class Probabilities**:
```math
P(y = k | \text{leaf}) = \frac{n_k}{n_{\text{leaf}}}
```

where $`n_{\text{leaf}} = \sum_{k=1}^K n_k`$ is the total number of samples in the leaf.

## 12.1.2. Goodness-of-Split Criterion

### Regression vs Classification

In the context of regression, this often involves calculating the reduction in residual sum of squares. Specifically, we consider a node $`T`$:

**Regression Split Criterion**:
```math
\Delta \text{RSS} = \text{RSS}(T) - \left[\text{RSS}(T_L) + \text{RSS}(T_R)\right]
```

where:
- $`\text{RSS}(T) = \sum_{i \in T} (y_i - \bar{y}_T)^2`$
- $`\text{RSS}(T_L) = \sum_{i \in T_L} (y_i - \bar{y}_{T_L})^2`$
- $`\text{RSS}(T_R) = \sum_{i \in T_R} (y_i - \bar{y}_{T_R})^2`$

**Classification Split Criterion**:
For classification, we use impurity measures instead of RSS:

```math
\Delta I = I(T) - \left[\frac{n_L}{n_T} I(T_L) + \frac{n_R}{n_T} I(T_R)\right]
```

where $`I(T)`$ is the impurity measure for node $`T`$.

### Common Impurity Measures

#### 1. Gini Impurity
```math
I_{\text{Gini}}(T) = 1 - \sum_{k=1}^K p_k^2
```

where $`p_k = \frac{n_k}{n_T}`$ is the proportion of class $`k`$ in node $`T`$.

#### 2. Entropy
```math
I_{\text{Entropy}}(T) = -\sum_{k=1}^K p_k \log_2(p_k)
```

#### 3. Misclassification Error
```math
I_{\text{Error}}(T) = 1 - \max_k p_k
```

### Properties of Impurity Measures

1. **Range**: All measures are in $`[0, 1]`$ for binary classification
2. **Minimum**: Achieved when node is pure (all samples same class)
3. **Maximum**: Achieved when classes are equally distributed
4. **Differentiability**: Gini and Entropy are differentiable, Error is not

## 12.1.3. The Greedy Algorithm

The process of searching for the best split follows a basic greedy algorithm:

### Algorithm Steps

1. **Start at root node** with all training data
2. **For each feature** $`j = 1, 2, \ldots, p`$:
   - Sort unique values of feature $`j`$
   - **For each split point** $`s`$ (midpoint between consecutive values):
     - Split data: $`X_j \leq s`$ vs $`X_j > s``
     - Calculate impurity reduction $`\Delta I`$
3. **Select best split**: Choose $`(j^*, s^*)`$ that maximizes $`\Delta I`$
4. **Create child nodes**: Split data according to best split
5. **Recurse**: Apply algorithm to each child node

### Computational Complexity

- **Time**: $`O(p \cdot n \log n)`$ per node (sorting dominates)
- **Space**: $`O(n)`$ for storing node data
- **Total**: $`O(p \cdot n \log n \cdot \text{number of nodes})`$

## 12.1.4. Implementation and Examples

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ClassificationTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.tree = None
        
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def entropy(self, y):
        """Calculate entropy"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def misclassification_error(self, y):
        """Calculate misclassification error"""
        classes, counts = np.unique(y, return_counts=True)
        return 1 - np.max(counts) / len(y)
    
    def calculate_impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.criterion == 'gini':
            return self.gini_impurity(y)
        elif self.criterion == 'entropy':
            return self.entropy(y)
        elif self.criterion == 'error':
            return self.misclassification_error(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def find_best_split(self, X, y):
        """Find the best split for the data"""
        n_samples, n_features = X.shape
        best_impurity_reduction = 0
        best_feature = None
        best_threshold = None
        
        # Calculate parent impurity
        parent_impurity = self.calculate_impurity(y)
        
        for feature in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Create split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split doesn't meet minimum requirements
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                # Calculate impurity for children
                left_impurity = self.calculate_impurity(y[left_mask])
                right_impurity = self.calculate_impurity(y[right_mask])
                
                # Calculate weighted impurity
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                # Calculate impurity reduction
                impurity_reduction = parent_impurity - weighted_impurity
                
                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_impurity_reduction
    
    def create_leaf(self, y):
        """Create a leaf node"""
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        probabilities = counts / len(y)
        return {
            'type': 'leaf',
            'prediction': majority_class,
            'probabilities': dict(zip(classes, probabilities)),
            'n_samples': len(y)
        }
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return self.create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, impurity_reduction = self.find_best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None or impurity_reduction <= 0:
            return self.create_leaf(y)
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Create internal node
        node = {
            'type': 'internal',
            'feature': best_feature,
            'threshold': best_threshold,
            'impurity_reduction': impurity_reduction,
            'n_samples': n_samples
        }
        
        # Recursively build children
        node['left'] = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Fit the classification tree"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, node):
        """Predict for a single sample"""
        if node['type'] == 'leaf':
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_single(x, node['left'])
        else:
            return self.predict_single(x, node['right'])
    
    def predict(self, X):
        """Predict for multiple samples"""
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x, self.tree))
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = []
        for x in X:
            proba = self.predict_proba_single(x, self.tree)
            probabilities.append(proba)
        return np.array(probabilities)
    
    def predict_proba_single(self, x, node):
        """Predict probabilities for a single sample"""
        if node['type'] == 'leaf':
            return list(node['probabilities'].values())
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_proba_single(x, node['left'])
        else:
            return self.predict_proba_single(x, node['right'])

# Generate classification data
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, 
                          n_clusters_per_class=1, n_redundant=0, 
                          random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)

# Compare different impurity measures
criteria = ['gini', 'entropy']
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, criterion in enumerate(criteria):
    # Fit custom tree
    custom_tree = ClassificationTree(max_depth=3, criterion=criterion, 
                                    random_state=42)
    custom_tree.fit(X_train, y_train)
    
    # Fit sklearn tree for comparison
    sklearn_tree = DecisionTreeClassifier(max_depth=3, criterion=criterion, 
                                         random_state=42)
    sklearn_tree.fit(X_train, y_train)
    
    # Plotting
    ax = axes[i]
    
    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Make predictions
    Z = custom_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Classification Tree ({criterion.upper()})')
    
    # Print accuracy
    train_acc = accuracy_score(y_train, custom_tree.predict(X_train))
    test_acc = accuracy_score(y_test, custom_tree.predict(X_test))
    ax.text(0.02, 0.98, f'Train Acc: {train_acc:.3f}\nTest Acc: {test_acc:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Compare impurity measures
print("Comparison of Impurity Measures:")
for criterion in criteria:
    tree = ClassificationTree(max_depth=3, criterion=criterion, random_state=42)
    tree.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    
    print(f"{criterion.upper()}:")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    print()
```

### R Implementation

```r
library(rpart)
library(rpart.plot)
library(ggplot2)

# Generate classification data
set.seed(42)
n <- 200
X <- matrix(rnorm(2*n), ncol=2)
y <- ifelse(X[,1] + X[,2] > 0, 1, 0)

# Create data frame
data <- data.frame(X1 = X[,1], X2 = X[,2], y = factor(y))

# Fit classification tree
tree_model <- rpart(y ~ X1 + X2, data = data, method = "class", 
                   control = rpart.control(maxdepth = 3))

# Plot tree
rpart.plot(tree_model, box.palette = "RdBu", shadow.col = "gray", 
           nn = TRUE, main = "Classification Tree")

# Make predictions
data$pred <- predict(tree_model, data, type = "class")

# Plot decision boundary
ggplot(data, aes(x = X1, y = X2, color = y)) +
  geom_point(alpha = 0.7) +
  geom_point(data = data[data$y != data$pred, ], 
             aes(x = X1, y = X2), shape = 21, size = 3, 
             fill = "transparent", color = "red") +
  labs(title = "Classification Tree Decision Boundary",
       subtitle = "Red circles indicate misclassifications") +
  theme_minimal()

# Print tree summary
print(tree_model)
printcp(tree_model)
```

## 12.1.5. Advantages and Limitations

### Advantages

1. **Interpretability**: Easy to understand and visualize
2. **No Assumptions**: No assumptions about data distribution
3. **Handles Mixed Data**: Can handle both numerical and categorical features
4. **Feature Importance**: Natural feature selection through splits
5. **Robust**: Insensitive to monotone transformations

### Limitations

1. **Instability**: Small changes in data can lead to very different trees
2. **Overfitting**: Tendency to overfit without proper regularization
3. **Axis-Aligned**: Can only create axis-aligned decision boundaries
4. **Greedy**: Local optimization may miss global optimum
5. **High Variance**: Individual trees have high variance

## 12.1.6. Summary

Classification trees extend regression trees to classification problems by:

1. **Impurity Measures**: Using Gini, entropy, or misclassification error instead of RSS
2. **Prediction Methods**: Majority voting or class probabilities at leaf nodes
3. **Split Criteria**: Maximizing impurity reduction
4. **Greedy Algorithm**: Same recursive splitting approach

Key insights:
- **Impurity measures** control split quality
- **Stopping criteria** prevent overfitting
- **Greedy approach** is computationally efficient
- **Tree structure** provides interpretability

This foundation sets the stage for more advanced tree-based methods like random forests and gradient boosting, which address many of the limitations of single classification trees.
