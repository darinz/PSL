# 12.3. Misclassification Rate vs. Entropy

Now, let's delve into the mathematical distinctions between the misclassification rate and entropy, two commonly used impurity measures in classification trees. Understanding these differences is crucial for choosing the right impurity measure for different stages of tree construction.

## 12.3.1. Mathematical Framework

### Binary Classification Setting

To illustrate these differences, let's consider a hypothetical scenario where we partition a set of $`n`$ observations within a node $`t`$ into two child nodes: left and right, containing $`n_L`$ and $`n_R`$ observations, respectively.

For simplicity, we'll assume there are only two classes. In the case of two classes, the impurity measure can be expressed as a function of the percentage of samples in one class. Let's denote the percentage of class zero as $`p_t`$ at a node $`t`$. This impurity function is essentially a function of $`p_t`$.

### Split Gain Formulation

The gain of the impurity measure is then determined by the difference in impurity at node $`t`$ (without a split) and the weighted sum of impurities in the left and right nodes. The weights are proportional to the sample sizes in the respective nodes.

```math
\begin{split}
\Phi(j,s) &= i(t) - \left[p_R \cdot i(t_R) + p_L \cdot i(t_L)\right] \\
&= f(p_t) - \left[\frac{n_R}{n_R + n_L} \cdot f(p_{t_R}) + \frac{n_L}{n_R + n_L} \cdot f(p_{t_L})\right]
\end{split}
```

### Weighted Average Property

Importantly, we observe that the percentage of class zero at the parent node $`t`$ is equal to the weighted sum of the percentage of class one in the two child nodes. This equality arises due to the weights being determined by the sample sizes in each node.

```math
\begin{split}
p_t &= \frac{n_R \cdot p_{t_R} + n_L \cdot p_{t_L}}{n_R + n_L} \\
&= \left(\frac{n_R}{n_R + n_L}\right) \cdot p_{t_R} + \left(\frac{n_L}{n_R + n_L}\right) \cdot p_{t_L}
\end{split}
```

### Concavity and Split Gain

The goodness of split $`\Phi(j,s)`$ can be represented as the discrepancy between the function evaluated at a weighted sum of $`p_{t_R}`$ and $`p_{t_L}`$, and the weighted sum of the function evaluated at $`p_{t_R}`$ and $`p_{t_L}`$.

Crucially, the gain in the impurity measure will be positive if the function $`f`$ is strictly concave. This property ensures that the difference between impurity measures at the parent node and the child nodes is always positive when evaluating the gain.

## 12.3.2. Mathematical Properties

### Concavity Analysis

**Definition**: A function $`f`$ is **concave** if for any $`x_1, x_2`$ and $`\lambda \in [0, 1]`$:
```math
f(\lambda x_1 + (1-\lambda) x_2) \geq \lambda f(x_1) + (1-\lambda) f(x_2)
```

**Strictly concave** if the inequality is strict for $`\lambda \in (0, 1)`$.

### Jensen's Inequality

For a concave function $`f`$ and weights $`w_1, w_2`$ with $`w_1 + w_2 = 1`$:
```math
f(w_1 x_1 + w_2 x_2) \geq w_1 f(x_1) + w_2 f(x_2)
```

This is exactly what we have in our split gain formula!

## 12.3.3. Visual Comparison

Let's visualize the key differences between misclassification rate and entropy:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def plot_impurity_comparison():
    """Visualize misclassification rate vs entropy"""
    p = np.linspace(0, 1, 1000)
    
    # Calculate impurity measures
    misclassification = 1 - np.maximum(p, 1-p)  # min(p, 1-p)
    entropy = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
    
    # Scale entropy to match misclassification at p=0.5
    entropy_scaled = entropy / entropy[500] * misclassification[500]
    
    plt.figure(figsize=(15, 10))
    
    # Main comparison plot
    plt.subplot(2, 2, 1)
    plt.plot(p, misclassification, 'b-', linewidth=3, label='Misclassification Rate')
    plt.plot(p, entropy_scaled, 'r-', linewidth=3, label='Entropy (Scaled)')
    plt.xlabel('Probability of Class 0 (p)')
    plt.ylabel('Impurity')
    plt.title('Misclassification Rate vs Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight key points
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add annotations
    plt.annotate('p = 0.5', xy=(0.5, 0.5), xytext=(0.6, 0.6),
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Concavity demonstration
    plt.subplot(2, 2, 2)
    p1, p2 = 0.3, 0.7
    p_weighted = 0.5  # (p1 + p2) / 2
    
    # Plot points
    plt.plot([p1, p2], [misclassification[int(p1*1000)], misclassification[int(p2*1000)]], 
             'bo-', linewidth=2, label='Linear interpolation')
    plt.plot([p1, p2], [entropy_scaled[int(p1*1000)], entropy_scaled[int(p2*1000)]], 
             'ro-', linewidth=2, label='Entropy values')
    
    # Plot weighted average point
    plt.plot(p_weighted, misclassification[int(p_weighted*1000)], 'bs', markersize=10, 
             label='Misclassification at weighted avg')
    plt.plot(p_weighted, entropy_scaled[int(p_weighted*1000)], 'rs', markersize=10, 
             label='Entropy at weighted avg')
    
    # Plot function values
    plt.plot(p, misclassification, 'b-', alpha=0.3)
    plt.plot(p, entropy_scaled, 'r-', alpha=0.3)
    
    plt.xlabel('Probability of Class 0 (p)')
    plt.ylabel('Impurity')
    plt.title('Concavity Demonstration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Split gain analysis
    plt.subplot(2, 2, 3)
    
    # Example: parent node with p=0.5, split into p1=0.3, p2=0.7
    p_parent = 0.5
    p_left = 0.3
    p_right = 0.7
    w_left = 0.5
    w_right = 0.5
    
    # Calculate gains
    misclass_parent = 1 - max(p_parent, 1-p_parent)
    misclass_left = 1 - max(p_left, 1-p_left)
    misclass_right = 1 - max(p_right, 1-p_right)
    misclass_gain = misclass_parent - (w_left * misclass_left + w_right * misclass_right)
    
    entropy_parent = -p_parent * np.log2(p_parent) - (1-p_parent) * np.log2(1-p_parent)
    entropy_left = -p_left * np.log2(p_left) - (1-p_left) * np.log2(1-p_left)
    entropy_right = -p_right * np.log2(p_right) - (1-p_right) * np.log2(1-p_right)
    entropy_gain = entropy_parent - (w_left * entropy_left + w_right * entropy_right)
    
    # Plot the split scenario
    plt.plot([p_left, p_right], [misclass_left, misclass_right], 'bo-', linewidth=2, 
             label=f'Misclassification Gain: {misclass_gain:.3f}')
    plt.plot([p_left, p_right], [entropy_left, entropy_right], 'ro-', linewidth=2, 
             label=f'Entropy Gain: {entropy_gain:.3f}')
    
    plt.axhline(y=misclass_parent, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(y=entropy_parent, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Probability of Class 0 (p)')
    plt.ylabel('Impurity')
    plt.title('Split Gain Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zero gain scenario
    plt.subplot(2, 2, 4)
    
    # Example: both child nodes on same side of 0.5
    p_parent = 0.6
    p_left = 0.55
    p_right = 0.65
    
    misclass_parent = 1 - max(p_parent, 1-p_parent)
    misclass_left = 1 - max(p_left, 1-p_left)
    misclass_right = 1 - max(p_right, 1-p_right)
    misclass_gain_zero = misclass_parent - (w_left * misclass_left + w_right * misclass_right)
    
    entropy_parent = -p_parent * np.log2(p_parent) - (1-p_parent) * np.log2(1-p_parent)
    entropy_left = -p_left * np.log2(p_left) - (1-p_left) * np.log2(1-p_left)
    entropy_right = -p_right * np.log2(p_right) - (1-p_right) * np.log2(1-p_right)
    entropy_gain_zero = entropy_parent - (w_left * entropy_left + w_right * entropy_right)
    
    plt.plot([p_left, p_right], [misclass_left, misclass_right], 'bo-', linewidth=2, 
             label=f'Misclassification Gain: {misclass_gain_zero:.3f}')
    plt.plot([p_left, p_right], [entropy_left, entropy_right], 'ro-', linewidth=2, 
             label=f'Entropy Gain: {entropy_gain_zero:.3f}')
    
    plt.axhline(y=misclass_parent, color='blue', linestyle='--', alpha=0.7)
    plt.axhline(y=entropy_parent, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Probability of Class 0 (p)')
    plt.ylabel('Impurity')
    plt.title('Zero Gain Scenario (Same Side of 0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical results
    print("Split Gain Analysis:")
    print(f"Scenario 1 - Different sides of 0.5:")
    print(f"  Parent: p={p_parent}, Misclassification gain: {misclass_gain:.4f}, Entropy gain: {entropy_gain:.4f}")
    print(f"Scenario 2 - Same side of 0.5:")
    print(f"  Parent: p={p_parent}, Misclassification gain: {misclass_gain_zero:.4f}, Entropy gain: {entropy_gain_zero:.4f}")

plot_impurity_comparison()
```

## 12.3.4. Mathematical Analysis

### Misclassification Rate Properties

**Formula**: $`f_{\text{misclass}}(p) = \min(p, 1-p)`$

**Properties**:
- **Piecewise linear**: Linear on $`[0, 0.5]`$ and $`[0.5, 1]`$
- **Not strictly concave**: Linear segments violate strict concavity
- **Zero gain scenarios**: When both child nodes are on the same side of $`0.5`$

**Mathematical Analysis**:
```math
f_{\text{misclass}}(p) = \begin{cases}
p & \text{if } p \leq 0.5 \\
1-p & \text{if } p > 0.5
\end{cases}
```

### Entropy Properties

**Formula**: $`f_{\text{entropy}}(p) = -p \log_2(p) - (1-p) \log_2(1-p)`$

**Properties**:
- **Strictly concave**: Second derivative is negative everywhere
- **Always positive gain**: Jensen's inequality guarantees positive split gain
- **Smooth**: Differentiable everywhere except at boundaries

**Mathematical Analysis**:
```math
f''_{\text{entropy}}(p) = -\frac{1}{p(1-p)\ln(2)} < 0 \quad \text{for } p \in (0, 1)
```

## 12.3.5. Practical Implications

### Tree Construction Phase

During tree construction, we want to encourage splits that lead to purer nodes. Entropy and Gini index are preferred because:

1. **Strictly concave**: Always provide positive split gain
2. **Encourage purity**: Strongly favor splits that create pure nodes
3. **Smooth optimization**: Differentiable functions work better with optimization algorithms

### Tree Pruning Phase

During pruning, we may want to use misclassification rate because:

1. **Direct interpretation**: Directly measures classification error
2. **Final goal alignment**: Matches the ultimate objective of minimizing misclassification
3. **Computational efficiency**: Simpler to compute

### Implementation Example

```python
def compare_split_gains():
    """Compare split gains for different scenarios"""
    
    def misclassification_impurity(p):
        return min(p, 1-p)
    
    def entropy_impurity(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, impurity_func):
        """Calculate split gain for given impurity function"""
        parent_impurity = impurity_func(p_parent)
        left_impurity = impurity_func(p_left)
        right_impurity = impurity_func(p_right)
        
        weighted_child_impurity = w_left * left_impurity + w_right * right_impurity
        gain = parent_impurity - weighted_child_impurity
        
        return gain
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Different sides of 0.5',
            'p_parent': 0.5,
            'p_left': 0.3,
            'p_right': 0.7,
            'w_left': 0.5,
            'w_right': 0.5
        },
        {
            'name': 'Same side of 0.5 (left)',
            'p_parent': 0.6,
            'p_left': 0.55,
            'p_right': 0.65,
            'w_left': 0.5,
            'w_right': 0.5
        },
        {
            'name': 'Same side of 0.5 (right)',
            'p_parent': 0.4,
            'p_left': 0.35,
            'p_right': 0.45,
            'w_left': 0.5,
            'w_right': 0.5
        },
        {
            'name': 'Extreme split',
            'p_parent': 0.5,
            'p_left': 0.1,
            'p_right': 0.9,
            'w_left': 0.5,
            'w_right': 0.5
        }
    ]
    
    print("Split Gain Comparison:")
    print("-" * 80)
    print(f"{'Scenario':<25} {'Misclass Gain':<15} {'Entropy Gain':<15}")
    print("-" * 80)
    
    for scenario in scenarios:
        misclass_gain = calculate_split_gain(
            scenario['p_parent'], scenario['p_left'], scenario['p_right'],
            scenario['w_left'], scenario['w_right'], misclassification_impurity
        )
        
        entropy_gain = calculate_split_gain(
            scenario['p_parent'], scenario['p_left'], scenario['p_right'],
            scenario['w_left'], scenario['w_right'], entropy_impurity
        )
        
        print(f"{scenario['name']:<25} {misclass_gain:<15.4f} {entropy_gain:<15.4f}")
    
    print("\nKey Observations:")
    print("1. Entropy always provides positive gain (strictly concave)")
    print("2. Misclassification can give zero gain when both children are on same side of 0.5")
    print("3. Entropy encourages more aggressive splitting")

compare_split_gains()
```

## 12.3.6. Theoretical Analysis

### Jensen's Inequality Application

For our split gain formula:
```math
\Phi(j,s) = f(p_t) - \left[w_L \cdot f(p_{t_L}) + w_R \cdot f(p_{t_R})\right]
```

where $`p_t = w_L \cdot p_{t_L} + w_R \cdot p_{t_R}`$ and $`w_L + w_R = 1`$.

**For concave functions**: $`f(p_t) \geq w_L \cdot f(p_{t_L}) + w_R \cdot f(p_{t_R})`$

**For strictly concave functions**: $`f(p_t) > w_L \cdot f(p_{t_L}) + w_R \cdot f(p_{t_R})`$ when $`p_{t_L} \neq p_{t_R}`$

### Zero Gain Scenarios

**Misclassification Rate**: Zero gain occurs when both $`p_{t_L}`$ and $`p_{t_R}`$ are on the same side of $`0.5`$.

**Entropy**: Never gives zero gain for non-trivial splits due to strict concavity.

## 12.3.7. Summary

The key differences between misclassification rate and entropy are:

### Mathematical Properties
1. **Misclassification Rate**: Piecewise linear, not strictly concave
2. **Entropy**: Strictly concave, smooth function

### Practical Behavior
1. **Misclassification Rate**: Can give zero gain for certain splits
2. **Entropy**: Always gives positive gain for non-trivial splits

### Recommendations
1. **Tree Construction**: Use entropy or Gini index (strictly concave)
2. **Tree Pruning**: Use misclassification rate (direct interpretation)

**Key insights**:
- **Concavity** determines whether splits always provide positive gain
- **Entropy** encourages more aggressive splitting during tree growth
- **Misclassification rate** aligns with final classification objective
- **Jensen's inequality** explains why concave functions work well for splits

This understanding helps in choosing the right impurity measure for different stages of decision tree construction and optimization.
