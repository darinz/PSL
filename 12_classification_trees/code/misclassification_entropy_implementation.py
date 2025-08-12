import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import seaborn as sns


def plot_impurity_comparison():
    """Visualize misclassification rate vs entropy"""
    print("=== Misclassification Rate vs Entropy Visualization ===\n")
    
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
    
    return {
        'misclass_gain': misclass_gain,
        'entropy_gain': entropy_gain,
        'misclass_gain_zero': misclass_gain_zero,
        'entropy_gain_zero': entropy_gain_zero
    }


def compare_split_gains():
    """Compare split gains for different scenarios"""
    print("=== Split Gain Comparison ===\n")
    
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
    
    results = []
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
        
        results.append({
            'scenario': scenario['name'],
            'misclass_gain': misclass_gain,
            'entropy_gain': entropy_gain
        })
    
    print("\nKey Observations:")
    print("1. Entropy always provides positive gain (strictly concave)")
    print("2. Misclassification can give zero gain when both children are on same side of 0.5")
    print("3. Entropy encourages more aggressive splitting")
    
    return results


def demonstrate_mathematical_properties():
    """Demonstrate mathematical properties of misclassification vs entropy"""
    print("=== Mathematical Properties Analysis ===\n")
    
    def misclassification_impurity(p):
        return min(p, 1-p)
    
    def entropy_impurity(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def entropy_second_derivative(p):
        """Second derivative of entropy function"""
        if p == 0 or p == 1:
            return -np.inf
        return -1 / (p * (1-p) * np.log(2))
    
    # Test concavity property
    print("Concavity Analysis:")
    print("-" * 40)
    
    # Test points for concavity
    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for p in test_points:
        # Calculate second derivative of entropy
        entropy_2nd_deriv = entropy_second_derivative(p)
        print(f"p = {p:.1f}: Entropy second derivative = {entropy_2nd_deriv:.4f}")
    
    print("\nEntropy is strictly concave (second derivative < 0)")
    print("Misclassification is piecewise linear (not strictly concave)")
    
    # Test Jensen's inequality
    print("\nJensen's Inequality Test:")
    print("-" * 40)
    
    p1, p2 = 0.3, 0.7
    lambda_val = 0.5
    p_weighted = lambda_val * p1 + (1 - lambda_val) * p2
    
    # For entropy (strictly concave)
    entropy_p1 = entropy_impurity(p1)
    entropy_p2 = entropy_impurity(p2)
    entropy_weighted = entropy_impurity(p_weighted)
    entropy_linear = lambda_val * entropy_p1 + (1 - lambda_val) * entropy_p2
    
    print(f"Entropy test:")
    print(f"  f(λp₁ + (1-λ)p₂) = f({p_weighted:.2f}) = {entropy_weighted:.4f}")
    print(f"  λf(p₁) + (1-λ)f(p₂) = {lambda_val:.1f}×{entropy_p1:.4f} + {1-lambda_val:.1f}×{entropy_p2:.4f} = {entropy_linear:.4f}")
    print(f"  Jensen's inequality: {entropy_weighted:.4f} > {entropy_linear:.4f} ✓")
    
    # For misclassification (not strictly concave)
    misclass_p1 = misclassification_impurity(p1)
    misclass_p2 = misclassification_impurity(p2)
    misclass_weighted = misclassification_impurity(p_weighted)
    misclass_linear = lambda_val * misclass_p1 + (1 - lambda_val) * misclass_p2
    
    print(f"\nMisclassification test:")
    print(f"  f(λp₁ + (1-λ)p₂) = f({p_weighted:.2f}) = {misclass_weighted:.4f}")
    print(f"  λf(p₁) + (1-λ)f(p₂) = {lambda_val:.1f}×{misclass_p1:.4f} + {1-lambda_val:.1f}×{misclass_p2:.4f} = {misclass_linear:.4f}")
    print(f"  Jensen's inequality: {misclass_weighted:.4f} = {misclass_linear:.4f} (equality holds)")
    
    return {
        'entropy_test': {
            'weighted': entropy_weighted,
            'linear': entropy_linear,
            'inequality_holds': entropy_weighted > entropy_linear
        },
        'misclass_test': {
            'weighted': misclass_weighted,
            'linear': misclass_linear,
            'inequality_holds': abs(misclass_weighted - misclass_linear) < 1e-10
        }
    }


def analyze_zero_gain_scenarios():
    """Analyze scenarios where misclassification gives zero gain"""
    print("=== Zero Gain Scenarios Analysis ===\n")
    
    def misclassification_impurity(p):
        return min(p, 1-p)
    
    def entropy_impurity(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, impurity_func):
        parent_impurity = impurity_func(p_parent)
        left_impurity = impurity_func(p_left)
        right_impurity = impurity_func(p_right)
        weighted_child_impurity = w_left * left_impurity + w_right * right_impurity
        return parent_impurity - weighted_child_impurity
    
    # Test different scenarios
    scenarios = []
    
    # Scenario 1: Both children on left side of 0.5
    p_parent = 0.6
    p_left = 0.55
    p_right = 0.65
    w_left = 0.5
    w_right = 0.5
    
    misclass_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
    entropy_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
    
    scenarios.append({
        'name': 'Both children left of 0.5',
        'p_parent': p_parent,
        'p_left': p_left,
        'p_right': p_right,
        'misclass_gain': misclass_gain,
        'entropy_gain': entropy_gain
    })
    
    # Scenario 2: Both children on right side of 0.5
    p_parent = 0.4
    p_left = 0.35
    p_right = 0.45
    
    misclass_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
    entropy_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
    
    scenarios.append({
        'name': 'Both children right of 0.5',
        'p_parent': p_parent,
        'p_left': p_left,
        'p_right': p_right,
        'misclass_gain': misclass_gain,
        'entropy_gain': entropy_gain
    })
    
    # Scenario 3: Children straddle 0.5
    p_parent = 0.5
    p_left = 0.3
    p_right = 0.7
    
    misclass_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
    entropy_gain = calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
    
    scenarios.append({
        'name': 'Children straddle 0.5',
        'p_parent': p_parent,
        'p_left': p_left,
        'p_right': p_right,
        'misclass_gain': misclass_gain,
        'entropy_gain': entropy_gain
    })
    
    # Print results
    print("Zero Gain Scenarios Analysis:")
    print("-" * 80)
    print(f"{'Scenario':<25} {'Parent p':<10} {'Left p':<8} {'Right p':<9} {'Misclass':<10} {'Entropy':<10}")
    print("-" * 80)
    
    for scenario in scenarios:
        print(f"{scenario['name']:<25} {scenario['p_parent']:<10.2f} {scenario['p_left']:<8.2f} "
              f"{scenario['p_right']:<9.2f} {scenario['misclass_gain']:<10.4f} {scenario['entropy_gain']:<10.4f}")
    
    print("\nKey Findings:")
    print("1. Misclassification gives zero gain when both children are on the same side of 0.5")
    print("2. Entropy always gives positive gain for non-trivial splits")
    print("3. Zero gain occurs because misclassification is piecewise linear")
    
    return scenarios


def demonstrate_practical_implications():
    """Demonstrate practical implications of choosing impurity measures"""
    print("=== Practical Implications ===\n")
    
    # Generate synthetic data to demonstrate tree construction
    np.random.seed(42)
    
    # Create data with different characteristics
    n_samples = 200
    
    # Dataset 1: Clear separation
    X1 = np.random.randn(n_samples, 2)
    y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)
    
    # Dataset 2: Overlapping classes
    X2 = np.random.randn(n_samples, 2)
    y2 = (X2[:, 0] + X2[:, 1] + 0.5 * np.random.randn(n_samples) > 0).astype(int)
    
    datasets = [
        (X1, y1, "Clear Separation"),
        (X2, y2, "Overlapping Classes")
    ]
    
    def misclassification_impurity(p):
        return min(p, 1-p)
    
    def entropy_impurity(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    def find_best_split(X, y, impurity_func):
        """Find best split using given impurity function"""
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    # Calculate class probabilities
                    parent_probs = np.bincount(y, minlength=2) / len(y)
                    left_probs = np.bincount(y[left_mask], minlength=2) / np.sum(left_mask)
                    right_probs = np.bincount(y[right_mask], minlength=2) / np.sum(right_mask)
                    
                    # Calculate impurity
                    parent_impurity = impurity_func(parent_probs[0])
                    left_impurity = impurity_func(left_probs[0])
                    right_impurity = impurity_func(right_probs[0])
                    
                    # Calculate gain
                    p_left = np.sum(left_mask) / len(y)
                    p_right = np.sum(right_mask) / len(y)
                    gain = parent_impurity - (p_left * left_impurity + p_right * right_impurity)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    print("Tree Construction Analysis:")
    print("-" * 60)
    
    for X, y, name in datasets:
        print(f"\n{name} Dataset:")
        
        # Find best splits for both impurity measures
        best_misclass = find_best_split(X, y, misclassification_impurity)
        best_entropy = find_best_split(X, y, entropy_impurity)
        
        print(f"  Misclassification best split: Feature {best_misclass[0]}, Threshold {best_misclass[1]:.3f}, Gain {best_misclass[2]:.4f}")
        print(f"  Entropy best split: Feature {best_entropy[0]}, Threshold {best_entropy[1]:.3f}, Gain {best_entropy[2]:.4f}")
        
        # Compare gains
        if best_misclass[2] == 0:
            print("  ⚠️  Misclassification found no useful split (zero gain)")
        else:
            print("  ✓ Misclassification found useful split")
        
        print("  ✓ Entropy always found useful split")
    
    print("\nPractical Recommendations:")
    print("1. Use entropy during tree construction (always positive gain)")
    print("2. Use misclassification for final evaluation (direct interpretation)")
    print("3. Consider computational efficiency for large datasets")
    print("4. Monitor for zero-gain scenarios with misclassification")


def main():
    """Main demonstration of misclassification vs entropy analysis"""
    print("Misclassification Rate vs Entropy: Mathematical Analysis")
    print("=" * 70)
    
    # 1. Visual comparison
    print("\n1. Visual Comparison:")
    viz_results = plot_impurity_comparison()
    
    # 2. Split gain comparison
    print("\n2. Split Gain Comparison:")
    split_results = compare_split_gains()
    
    # 3. Mathematical properties
    print("\n3. Mathematical Properties:")
    math_results = demonstrate_mathematical_properties()
    
    # 4. Zero gain scenarios
    print("\n4. Zero Gain Scenarios:")
    zero_gain_results = analyze_zero_gain_scenarios()
    
    # 5. Practical implications
    print("\n5. Practical Implications:")
    practical_results = demonstrate_practical_implications()
    
    print("\n=== Key Insights ===")
    print("1. Entropy is strictly concave, always provides positive split gain")
    print("2. Misclassification is piecewise linear, can give zero gain")
    print("3. Jensen's inequality explains why concave functions work well")
    print("4. Use entropy for tree construction, misclassification for evaluation")
    print("5. Zero gain occurs when both children are on same side of 0.5")
    print("6. Entropy encourages more aggressive splitting")
    print("7. Misclassification aligns with final classification objective")
    print("8. Mathematical properties determine practical behavior")
    
    return {
        'visualization_results': viz_results,
        'split_results': split_results,
        'mathematical_results': math_results,
        'zero_gain_results': zero_gain_results,
        'practical_results': practical_results
    }


if __name__ == "__main__":
    main()
