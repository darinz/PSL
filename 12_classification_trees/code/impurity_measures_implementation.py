import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


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


def demonstrate_split_gain():
    """Demonstrate split gain calculation"""
    print("=== Split Gain Demonstration ===\n")
    
    # Create simple example data
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 2], [5, 3], [6, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    print("Data:")
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"  Sample {i+1}: X={x}, y={label}")
    
    # Define impurity functions
    def gini_impurity(p):
        return 1 - np.sum(p**2)
    
    def entropy_impurity(p):
        return -np.sum(p * np.log2(p + 1e-10))
    
    def misclassification_impurity(p):
        return 1 - np.max(p)
    
    # Test different splits
    print("\nTesting different splits:")
    print("Feature | Threshold | Gain (Gini) | Gain (Entropy) | Gain (Misclass)")
    print("-" * 70)
    
    for feature in range(2):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain_gini = calculate_split_gain(X, y, feature, threshold, gini_impurity)
            gain_entropy = calculate_split_gain(X, y, feature, threshold, entropy_impurity)
            gain_misclass = calculate_split_gain(X, y, feature, threshold, misclassification_impurity)
            
            print(f"   {feature}    |    {threshold:.1f}    |    {gain_gini:.3f}     |     {gain_entropy:.3f}     |     {gain_misclass:.3f}")
    
    # Find best split for each impurity measure
    print("\nBest splits for each impurity measure:")
    best_gini = find_best_split(X, y, gini_impurity)
    best_entropy = find_best_split(X, y, entropy_impurity)
    best_misclass = find_best_split(X, y, misclassification_impurity)
    
    print(f"Gini: Feature {best_gini[0]} <= {best_gini[1]:.1f}, Gain = {best_gini[2]:.3f}")
    print(f"Entropy: Feature {best_entropy[0]} <= {best_entropy[1]:.1f}, Gain = {best_entropy[2]:.3f}")
    print(f"Misclassification: Feature {best_misclass[0]} <= {best_misclass[1]:.1f}, Gain = {best_misclass[2]:.3f}")
    
    return X, y, (best_gini, best_entropy, best_misclass)


def compare_impurity_measures():
    """Compare different impurity measures"""
    print("=== Impurity Measures Comparison ===\n")
    
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
    
    return results, labels


def analyze_impurity_properties():
    """Analyze theoretical properties of impurity measures"""
    print("=== Impurity Measures Properties Analysis ===\n")
    
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
    
    # Visualize concavity
    plt.figure(figsize=(12, 8))
    
    # Test concavity with interpolation
    p1_range = np.linspace(0, 1, 100)
    p2_range = 0.5 * (1 - p1_range)
    p3_range = 0.5 * (1 - p1_range)
    
    gini_values = []
    entropy_values = []
    misclass_values = []
    
    for p1, p2, p3 in zip(p1_range, p2_range, p3_range):
        if p1 + p2 + p3 <= 1:
            gini_values.append(gini_impurity([p1, p2, p3]))
            entropy_values.append(entropy_impurity([p1, p2, p3]))
            misclass_values.append(misclassification_impurity([p1, p2, p3]))
        else:
            gini_values.append(np.nan)
            entropy_values.append(np.nan)
            misclass_values.append(np.nan)
    
    plt.subplot(2, 2, 1)
    plt.plot(p1_range, gini_values, 'b-', linewidth=2, label='Gini')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Gini Impurity')
    plt.title('Gini Impurity - Concave Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(p1_range, entropy_values, 'r-', linewidth=2, label='Entropy')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Entropy')
    plt.title('Entropy - Strictly Concave Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(p1_range, misclass_values, 'g-', linewidth=2, label='Misclassification')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error - Non-differentiable')
    plt.grid(True, alpha=0.3)
    
    # Compare all three
    plt.subplot(2, 2, 4)
    plt.plot(p1_range, gini_values, 'b-', linewidth=2, label='Gini')
    plt.plot(p1_range, entropy_values, 'r-', linewidth=2, label='Entropy')
    plt.plot(p1_range, misclass_values, 'g-', linewidth=2, label='Misclassification')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Impurity Value')
    plt.title('Comparison of Impurity Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_practical_considerations():
    """Demonstrate practical considerations in impurity measure choice"""
    print("=== Practical Considerations ===\n")
    
    # Generate data with different characteristics
    np.random.seed(42)
    
    # Create data with clear separation
    X_clear = np.random.randn(200, 2)
    y_clear = (X_clear[:, 0] + X_clear[:, 1] > 0).astype(int)
    
    # Create data with overlap
    X_overlap = np.random.randn(200, 2)
    y_overlap = (X_overlap[:, 0] + X_overlap[:, 1] + 0.5 * np.random.randn(200) > 0).astype(int)
    
    datasets = [
        (X_clear, y_clear, "Clear Separation"),
        (X_overlap, y_overlap, "Overlapping Classes")
    ]
    
    def gini_impurity(p):
        return 1 - np.sum(p**2)
    
    def entropy_impurity(p):
        return -np.sum(p * np.log2(p + 1e-10))
    
    def misclassification_impurity(p):
        return 1 - np.max(p)
    
    impurity_funcs = [
        ("Gini", gini_impurity),
        ("Entropy", entropy_impurity),
        ("Misclassification", misclassification_impurity)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (X, y, title) in enumerate(datasets):
        for j, (name, impurity_func) in enumerate(impurity_funcs):
            ax = axes[i, j]
            
            # Find best split
            best_feature, best_threshold, best_gain = find_best_split(X, y, impurity_func)
            
            # Create decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                 np.arange(y_min, y_max, 0.01))
            
            # Make predictions based on best split
            Z = (xx <= best_threshold) if best_feature == 0 else (yy <= best_threshold)
            Z = Z.astype(int)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'{title} - {name}\nGain: {best_gain:.3f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("Analysis of Impurity Measures in Practice:")
    print("1. Gini: Good balance between computational efficiency and performance")
    print("2. Entropy: Strongly encourages pure splits, may lead to overfitting")
    print("3. Misclassification: Direct interpretation but less smooth optimization")
    print("4. Choice depends on data characteristics and computational constraints")


def main():
    """Main demonstration of impurity measures"""
    print("Impurity Measures: Implementation and Analysis")
    print("=" * 60)
    
    # 1. Visualize impurity measures
    print("\n1. Impurity Measures Visualization:")
    plot_impurity_measures()
    
    # 2. Demonstrate split gain calculation
    print("\n2. Split Gain Demonstration:")
    split_results = demonstrate_split_gain()
    
    # 3. Compare impurity measures
    print("\n3. Impurity Measures Comparison:")
    comparison_results = compare_impurity_measures()
    
    # 4. Analyze theoretical properties
    print("\n4. Theoretical Properties Analysis:")
    analyze_impurity_properties()
    
    # 5. Practical considerations
    print("\n5. Practical Considerations:")
    demonstrate_practical_considerations()
    
    print("\n=== Key Insights ===")
    print("1. Gini Index: Most commonly used, differentiable, good balance")
    print("2. Entropy: Strongly encourages pure splits, differentiable")
    print("3. Misclassification Error: Direct interpretation, not differentiable")
    print("4. All measures are symmetric and bounded")
    print("5. Choice depends on application and computational considerations")
    print("6. Entropy is preferred during tree growing due to concavity")
    print("7. Gini is often used in practice due to efficiency")
    print("8. Misclassification error is useful for final evaluation")
    
    return {
        'split_results': split_results,
        'comparison_results': comparison_results
    }


if __name__ == "__main__":
    main()
