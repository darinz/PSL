# 12.2. Impurity Measures

In the context of classification trees, the selection of a suitable goodness-of-split criterion is a critical consideration. Typically, we rely on a concept known as the "gain" of an impurity measure. But what exactly is this impurity measure?

## 12.2.1. Impurity Measures

### Definition and Properties

The impurity measure is a function $`I(p_1, \dots, p_K)`$ defined over a probability distribution representing $`K`$ classes. For instance, if $`K`$ equals three, we work with a probability vector $`(p_1, p_2, p_3)`$. These values represent the probabilities of occurrence for each of the three classes.

**Mathematical Definition**: An impurity measure $`I(p_1, \dots, p_K)`$ satisfies:
1. **Non-negativity**: $`I(p_1, \dots, p_K) \geq 0`$
2. **Symmetry**: $`I(p_1, \dots, p_K) = I(p_{\sigma(1)}, \dots, p_{\sigma(K)})`$ for any permutation $`\sigma`$
3. **Minimum at pure nodes**: $`I(1, 0, \dots, 0) = I(0, 1, 0, \dots, 0) = \dots = I(0, \dots, 0, 1) = 0`$
4. **Maximum at uniform distribution**: $`I(1/K, 1/K, \dots, 1/K)`$ is maximum

### Intuitive Understanding

The impurity measure quantifies the "impurity" or randomness of the distribution. It reaches its maximum value when all classes are equally likely and its minimum when only one class is certain (i.e., $`p_j`$ equals one for one class). Importantly, the impurity measure is always symmetric because it operates on probabilities, making it independent of class labels' order.

**Key Properties**:
- **Maximum** occurs at $`(1/K, \dots, 1/K)`$ (the most impure node)
- **Minimum** occurs at $`p_j = 1`$ (the purest node)
- **Symmetric function** of $`p_1, \dots, p_K`$, i.e., permutation of $`p_j`$ does not affect $`I(\cdot)`$

### Visual Representation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_impurity_measures():
    """Visualize different impurity measures for binary classification"""
    p1 = np.linspace(0, 1, 100)
    p2 = 1 - p1
    
    # Calculate impurity measures
    gini = 1 - (p1**2 + p2**2)
    entropy = -p1 * np.log2(p1 + 1e-10) - p2 * np.log2(p2 + 1e-10)
    misclassification = 1 - np.maximum(p1, p2)
    
    plt.figure(figsize=(12, 8))
    
    # Plot impurity measures
    plt.subplot(2, 2, 1)
    plt.plot(p1, gini, 'b-', linewidth=2, label='Gini')
    plt.plot(p1, entropy, 'r-', linewidth=2, label='Entropy')
    plt.plot(p1, misclassification, 'g-', linewidth=2, label='Misclassification')
    plt.xlabel('Probability of Class 1 (p₁)')
    plt.ylabel('Impurity')
    plt.title('Impurity Measures for Binary Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3D visualization for ternary classification
    plt.subplot(2, 2, 2)
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create triangular grid
    n_points = 50
    p1_3d = np.linspace(0, 1, n_points)
    p2_3d = np.linspace(0, 1, n_points)
    P1, P2 = np.meshgrid(p1_3d, p2_3d)
    P3 = 1 - P1 - P2
    
    # Only keep valid probability combinations
    valid_mask = (P3 >= 0) & (P3 <= 1)
    P1_valid = P1[valid_mask]
    P2_valid = P2[valid_mask]
    P3_valid = P3[valid_mask]
    
    # Calculate Gini for valid points
    gini_3d = 1 - (P1_valid**2 + P2_valid**2 + P3_valid**2)
    
    ax = plt.subplot(2, 2, 2, projection='3d')
    scatter = ax.scatter(P1_valid, P2_valid, gini_3d, c=gini_3d, cmap='viridis')
    ax.set_xlabel('p₁')
    ax.set_ylabel('p₂')
    ax.set_zlabel('Gini Impurity')
    ax.set_title('Gini Impurity for Ternary Classification')
    plt.colorbar(scatter)
    
    # Plot contour for ternary classification
    plt.subplot(2, 2, 3)
    # Create triangular contour
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    
    # Calculate Gini
    gini_contour = 1 - (X**2 + Y**2 + Z**2)
    gini_contour[Z < 0] = np.nan  # Mask invalid regions
    
    contour = plt.contourf(X, Y, gini_contour, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('p₁')
    plt.ylabel('p₂')
    plt.title('Gini Impurity Contour (Ternary)')
    
    # Add triangle boundary
    triangle = Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]], 
                      facecolor='none', edgecolor='black', linewidth=2)
    plt.gca().add_patch(triangle)
    
    # Compare impurity measures at different distributions
    plt.subplot(2, 2, 4)
    distributions = [
        [1.0, 0.0, 0.0],  # Pure class 1
        [0.8, 0.1, 0.1],  # Mostly class 1
        [0.5, 0.3, 0.2],  # Mixed
        [0.33, 0.33, 0.34],  # Nearly uniform
        [0.33, 0.33, 0.33]   # Uniform
    ]
    
    labels = ['Pure', 'Mostly 1', 'Mixed', 'Near Uniform', 'Uniform']
    x_pos = np.arange(len(distributions))
    
    gini_values = []
    entropy_values = []
    misclass_values = []
    
    for dist in distributions:
        p1, p2, p3 = dist
        gini_values.append(1 - (p1**2 + p2**2 + p3**2))
        entropy_values.append(-p1*np.log2(p1+1e-10) - p2*np.log2(p2+1e-10) - p3*np.log2(p3+1e-10))
        misclass_values.append(1 - max(p1, p2, p3))
    
    width = 0.25
    plt.bar(x_pos - width, gini_values, width, label='Gini', alpha=0.8)
    plt.bar(x_pos, entropy_values, width, label='Entropy', alpha=0.8)
    plt.bar(x_pos + width, misclass_values, width, label='Misclassification', alpha=0.8)
    
    plt.xlabel('Distribution Type')
    plt.ylabel('Impurity Value')
    plt.title('Comparison of Impurity Measures')
    plt.xticks(x_pos, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.show()

plot_impurity_measures()
```

## 12.2.2. Goodness-of-Split Criterion

### Mathematical Formulation

Once we have defined the impurity measure, we can derive the goodness-of-split criterion, denoted as:

```math
\Phi(j,s) = i(t) - \left[p_R \cdot i(t_R) + p_L \cdot i(t_L)\right]
```

where:

```math
\begin{aligned}
i(t) &= I(p_t(1), \dots, p_t(K)) \\
p_t(j) &= \text{frequency of class } j \text{ at node } t
\end{aligned}
```

### Interpretation

When we split a node into left and right nodes, we evaluate the impurity measure at the parent node (original node $`t`$) based on the empirical distribution of frequencies across the $`K`$ classes. We also calculate the impurity measure at the left and right nodes if no split is applied.

However, unlike the residual sum of squares, the impurity measure is not cumulative; it represents a quantity at the distribution level. Therefore, we must compute a **weighted sum** to determine $`\Phi`$, where $`p_R`$ represents the proportion of samples in the right node and $`p_L`$ represents the proportion in the left node.

### Implementation

```python
def calculate_split_gain(X, y, feature, threshold, impurity_func):
    """Calculate the gain of a split"""
    # Split the data
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    # Calculate class frequencies for parent and children
    parent_classes, parent_counts = np.unique(y, return_counts=True)
    parent_probs = parent_counts / len(y)
    
    left_classes, left_counts = np.unique(y[left_mask], return_counts=True)
    left_probs = left_counts / np.sum(left_mask)
    
    right_classes, right_counts = np.unique(y[right_mask], return_counts=True)
    right_probs = right_counts / np.sum(right_mask)
    
    # Calculate impurity
    parent_impurity = impurity_func(parent_probs)
    left_impurity = impurity_func(left_probs)
    right_impurity = impurity_func(right_probs)
    
    # Calculate proportions
    p_left = np.sum(left_mask) / len(y)
    p_right = np.sum(right_mask) / len(y)
    
    # Calculate gain
    gain = parent_impurity - (p_left * left_impurity + p_right * right_impurity)
    
    return gain

def find_best_split(X, y, impurity_func):
    """Find the best split using impurity gain"""
    n_samples, n_features = X.shape
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    for feature in range(n_features):
        # Get unique values for this feature
        thresholds = np.unique(X[:, feature])
        
        for threshold in thresholds:
            gain = calculate_split_gain(X, y, feature, threshold, impurity_func)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain
```

## 12.2.3. Choice of Impurity Measures

### Three Common Impurity Measures

The choice of impurity measure for classification trees includes:

```math
\begin{aligned}
\text{Misclassification Rate} &: 1 - \max_j p_j \\
\text{Entropy (Deviance)} &: -\sum_{j=1}^K p_j \log p_j \\
\text{Gini Index} &: \sum_{j=1}^K p_j(1-p_j) = 1 - \sum_j p_j^2
\end{aligned}
```

### 1. Misclassification Rate

**Formula**: $`I_{\text{Error}}(p_1, \dots, p_K) = 1 - \max_j p_j`$

**Properties**:
- **Range**: $`[0, 1-1/K]`$
- **Maximum**: Achieved when all classes are equally likely
- **Minimum**: Achieved when one class has probability 1
- **Differentiability**: Not differentiable at points where maximum probability changes

**Intuition**: In this measure, majority voting is used, and the class corresponding to the maximum $`p_j`$ is considered correct. The misclassification rate is computed as 1 minus the maximum $`p_j`$. This measure is symmetric and attains its maximum with equally likely classes and its minimum when only one class exists.

### 2. Entropy

**Formula**: $`I_{\text{Entropy}}(p_1, \dots, p_K) = -\sum_{j=1}^K p_j \log p_j`$

**Properties**:
- **Range**: $`[0, \log_2(K)]`$
- **Maximum**: Achieved when all classes are equally likely
- **Minimum**: Achieved when one class has probability 1
- **Differentiability**: Differentiable everywhere except at boundaries

**Intuition**: Entropy is a popular impurity measure that quantifies the randomness of a distribution. It is commonly used in various fields such as coding theory, communication, and physics to describe the uncertainty or randomness in a discrete distribution over $`K`$ classes. Like misclassification rate, entropy also reaches its maximum at a uniform distribution and its minimum at a deterministic distribution.

### 3. Gini Index

**Formula**: $`I_{\text{Gini}}(p_1, \dots, p_K) = \sum_{j=1}^K p_j(1-p_j) = 1 - \sum_j p_j^2`$

**Properties**:
- **Range**: $`[0, 1-1/K]`$
- **Maximum**: Achieved when all classes are equally likely
- **Minimum**: Achieved when one class has probability 1
- **Differentiability**: Differentiable everywhere

**Intuition**: The Gini index is another widely used impurity measure. It shares similarities with entropy in terms of performance. The choice between Gini index and entropy often depends on the specific application and preference. In practice, entropy is commonly used due to its connection with likelihood for a multinomial distribution.

### Comparison and Analysis

```python
def compare_impurity_measures():
    """Compare different impurity measures"""
    # Test different probability distributions
    test_distributions = [
        [1.0, 0.0, 0.0],      # Pure
        [0.9, 0.05, 0.05],    # Nearly pure
        [0.7, 0.2, 0.1],      # Mixed
        [0.5, 0.3, 0.2],      # More mixed
        [0.4, 0.3, 0.3],      # Nearly uniform
        [1/3, 1/3, 1/3]       # Uniform
    ]
    
    labels = ['Pure', 'Nearly Pure', 'Mixed', 'More Mixed', 'Near Uniform', 'Uniform']
    
    def gini_impurity(p):
        return 1 - np.sum(np.array(p)**2)
    
    def entropy_impurity(p):
        return -np.sum(np.array(p) * np.log2(np.array(p) + 1e-10))
    
    def misclassification_impurity(p):
        return 1 - np.max(p)
    
    results = []
    for dist in test_distributions:
        gini = gini_impurity(dist)
        entropy = entropy_impurity(dist)
        misclass = misclassification_impurity(dist)
        results.append([gini, entropy, misclass])
    
    results = np.array(results)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot comparison
    x = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x - width, results[:, 0], width, label='Gini', alpha=0.8)
    ax1.bar(x, results[:, 1], width, label='Entropy', alpha=0.8)
    ax1.bar(x + width, results[:, 2], width, label='Misclassification', alpha=0.8)
    
    ax1.set_xlabel('Distribution Type')
    ax1.set_ylabel('Impurity Value')
    ax1.set_title('Comparison of Impurity Measures')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Line plot showing behavior
    p1_values = np.linspace(0, 1, 100)
    p2_values = 0.5 * (1 - p1_values)
    p3_values = 0.5 * (1 - p1_values)
    
    gini_line = []
    entropy_line = []
    misclass_line = []
    
    for p1 in p1_values:
        p2 = p2_values[int(p1 * 99)]
        p3 = p3_values[int(p1 * 99)]
        
        if p1 + p2 + p3 <= 1:  # Valid probability distribution
            gini_line.append(gini_impurity([p1, p2, p3]))
            entropy_line.append(entropy_impurity([p1, p2, p3]))
            misclass_line.append(misclassification_impurity([p1, p2, p3]))
        else:
            gini_line.append(np.nan)
            entropy_line.append(np.nan)
            misclass_line.append(np.nan)
    
    ax2.plot(p1_values, gini_line, 'b-', linewidth=2, label='Gini')
    ax2.plot(p1_values, entropy_line, 'r-', linewidth=2, label='Entropy')
    ax2.plot(p1_values, misclass_line, 'g-', linewidth=2, label='Misclassification')
    ax2.set_xlabel('Probability of Class 1')
    ax2.set_ylabel('Impurity Value')
    ax2.set_title('Impurity Measures vs Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical comparison
    print("Numerical Comparison of Impurity Measures:")
    print(f"{'Distribution':<15} {'Gini':<8} {'Entropy':<8} {'Misclass':<8}")
    print("-" * 45)
    for i, label in enumerate(labels):
        print(f"{label:<15} {results[i, 0]:<8.3f} {results[i, 1]:<8.3f} {results[i, 2]:<8.3f}")

compare_impurity_measures()
```

### Practical Considerations

It's important to note that entropy is a strictly concave function, which means it strongly favors splits leading to pure nodes. This characteristic makes entropy a suitable choice during the initial tree construction phase, where achieving purity is desirable. Subsequently, when pruning the tree, one may switch to using either the misclassification rate or entropy, depending on the ultimate classification goal.

### Theoretical Properties

```python
def analyze_impurity_properties():
    """Analyze theoretical properties of impurity measures"""
    
    def gini_impurity(p):
        return 1 - np.sum(np.array(p)**2)
    
    def entropy_impurity(p):
        return -np.sum(np.array(p) * np.log2(np.array(p) + 1e-10))
    
    def misclassification_impurity(p):
        return 1 - np.max(p)
    
    # Test symmetry property
    p1 = [0.3, 0.5, 0.2]
    p2 = [0.5, 0.2, 0.3]  # Permutation of p1
    
    print("Symmetry Property Test:")
    print(f"Original distribution: {p1}")
    print(f"Permuted distribution: {p2}")
    print(f"Gini - Original: {gini_impurity(p1):.4f}, Permuted: {gini_impurity(p2):.4f}")
    print(f"Entropy - Original: {entropy_impurity(p1):.4f}, Permuted: {entropy_impurity(p2):.4f}")
    print(f"Misclass - Original: {misclassification_impurity(p1):.4f}, Permuted: {misclassification_impurity(p2):.4f}")
    print()
    
    # Test concavity
    print("Concavity Analysis:")
    print("Entropy is strictly concave, encouraging pure splits")
    print("Gini is also concave but less strict than entropy")
    print("Misclassification error is not differentiable at all points")
    print()
    
    # Test sensitivity to small changes
    p_base = [0.5, 0.3, 0.2]
    p_perturbed = [0.51, 0.29, 0.2]
    
    print("Sensitivity to Small Changes:")
    print(f"Base distribution: {p_base}")
    print(f"Perturbed distribution: {p_perturbed}")
    print(f"Gini change: {abs(gini_impurity(p_base) - gini_impurity(p_perturbed)):.6f}")
    print(f"Entropy change: {abs(entropy_impurity(p_base) - entropy_impurity(p_perturbed)):.6f}")
    print(f"Misclass change: {abs(misclassification_impurity(p_base) - misclassification_impurity(p_perturbed)):.6f}")

analyze_impurity_properties()
```

## 12.2.4. Summary

The choice of impurity measure significantly affects the behavior of classification trees:

1. **Gini Index**: Most commonly used, differentiable, good balance
2. **Entropy**: Strongly encourages pure splits, differentiable
3. **Misclassification Error**: Direct interpretation, not differentiable

**Key insights**:
- **Entropy** is preferred during tree growing due to its concavity
- **Gini** is often used in practice due to computational efficiency
- **Misclassification error** is useful for final evaluation
- All measures are **symmetric** and **bounded**
- **Differentiability** affects optimization behavior

The choice between these measures often depends on the specific application, computational considerations, and the desired balance between interpretability and performance.
