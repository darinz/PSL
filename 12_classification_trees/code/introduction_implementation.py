import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns


class ClassificationTree:
    """Custom implementation of classification tree"""
    
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


def demonstrate_impurity_measures():
    """Demonstrate different impurity measures"""
    print("=== Impurity Measures Demonstration ===\n")
    
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
    
    results = {}
    
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
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8, cmap='RdYlBu')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Classification Tree ({criterion.upper()})')
        
        # Print accuracy
        train_acc = accuracy_score(y_train, custom_tree.predict(X_train))
        test_acc = accuracy_score(y_test, custom_tree.predict(X_test))
        ax.text(0.02, 0.98, f'Train Acc: {train_acc:.3f}\nTest Acc: {test_acc:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        results[criterion] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'tree': custom_tree
        }
    
    plt.tight_layout()
    plt.show()
    
    # Compare impurity measures
    print("Comparison of Impurity Measures:")
    for criterion in criteria:
        print(f"{criterion.upper()}:")
        print(f"  Train Accuracy: {results[criterion]['train_acc']:.3f}")
        print(f"  Test Accuracy: {results[criterion]['test_acc']:.3f}")
        print()
    
    return results


def demonstrate_tree_structure():
    """Demonstrate tree structure and node properties"""
    print("\n=== Tree Structure Demonstration ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, 
                              random_state=42)
    
    # Fit tree with different depths
    depths = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, depth in enumerate(depths):
        tree = ClassificationTree(max_depth=depth, criterion='gini', random_state=42)
        tree.fit(X, y)
        
        ax = axes[i // 2, i % 2]
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        # Make predictions
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'Tree Depth: {depth}')
        
        # Calculate accuracy
        acc = accuracy_score(y, tree.predict(X))
        ax.text(0.02, 0.98, f'Accuracy: {acc:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return depths


def demonstrate_stopping_criteria():
    """Demonstrate different stopping criteria"""
    print("\n=== Stopping Criteria Demonstration ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2, 
                              n_clusters_per_class=1, n_redundant=0, 
                              random_state=42)
    
    # Test different stopping criteria
    criteria_configs = [
        {'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Max Depth = 2'},
        {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'name': 'Max Depth = 5'},
        {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'name': 'Min Split = 10'},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 20, 'name': 'Min Leaf = 20'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, config in enumerate(criteria_configs):
        tree = ClassificationTree(
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            criterion='gini',
            random_state=42
        )
        tree.fit(X, y)
        
        ax = axes[i // 2, i % 2]
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        # Make predictions
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(config['name'])
        
        # Calculate accuracy
        acc = accuracy_score(y, tree.predict(X))
        ax.text(0.02, 0.98, f'Accuracy: {acc:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return criteria_configs


def demonstrate_greedy_algorithm():
    """Demonstrate the greedy algorithm step by step"""
    print("\n=== Greedy Algorithm Demonstration ===\n")
    
    # Generate simple data for demonstration
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 2], [5, 3], [6, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    print("Data:")
    for i, (x, label) in enumerate(zip(X, y)):
        print(f"  Sample {i+1}: X={x}, y={label}")
    
    # Calculate initial impurity
    tree = ClassificationTree(criterion='gini')
    initial_impurity = tree.calculate_impurity(y)
    print(f"\nInitial Gini impurity: {initial_impurity:.3f}")
    
    # Test all possible splits
    print("\nTesting all possible splits:")
    print("Feature | Threshold | Left Impurity | Right Impurity | Reduction")
    print("-" * 65)
    
    best_reduction = 0
    best_split = None
    
    for feature in range(2):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                left_impurity = tree.calculate_impurity(y[left_mask])
                right_impurity = tree.calculate_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / len(y)
                reduction = initial_impurity - weighted_impurity
                
                print(f"   {feature}    |    {threshold:.1f}    |     {left_impurity:.3f}     |     {right_impurity:.3f}     |   {reduction:.3f}")
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_split = (feature, threshold)
    
    print(f"\nBest split: Feature {best_split[0]} <= {best_split[1]:.1f}")
    print(f"Impurity reduction: {best_reduction:.3f}")
    
    return X, y, best_split


def demonstrate_advantages_limitations():
    """Demonstrate advantages and limitations of classification trees"""
    print("\n=== Advantages and Limitations ===\n")
    
    # Generate data with different characteristics
    np.random.seed(42)
    
    # 1. Axis-aligned boundaries (limitation)
    X1 = np.random.randn(200, 2)
    y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)
    
    # 2. Non-axis-aligned boundaries (challenging for trees)
    X2 = np.random.randn(200, 2)
    y2 = (X2[:, 0]**2 + X2[:, 1]**2 < 1).astype(int)
    
    # 3. XOR-like pattern (very challenging for trees)
    X3 = np.random.randn(200, 2)
    y3 = ((X3[:, 0] > 0) & (X3[:, 1] > 0) | (X3[:, 0] < 0) & (X3[:, 1] < 0)).astype(int)
    
    datasets = [
        (X1, y1, "Linear Separable"),
        (X2, y2, "Circular Boundary"),
        (X3, y3, "XOR Pattern")
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    for i, (X, y, title) in enumerate(datasets):
        # Fit tree
        tree = ClassificationTree(max_depth=5, criterion='gini', random_state=42)
        tree.fit(X, y)
        
        # Plot original data
        ax1 = axes[i, 0]
        ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title(f'{title} - Original Data')
        
        # Plot decision boundary
        ax2 = axes[i, 1]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax2.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='RdYlBu')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        acc = accuracy_score(y, tree.predict(X))
        ax2.set_title(f'{title} - Decision Boundary (Acc: {acc:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis:")
    print("1. Linear Separable: Trees work well with axis-aligned boundaries")
    print("2. Circular Boundary: Trees struggle with curved boundaries")
    print("3. XOR Pattern: Trees have difficulty with non-linear patterns")
    print("\nThis demonstrates the axis-aligned limitation of decision trees.")
    
    return datasets


def main():
    """Main demonstration of classification tree concepts"""
    print("Classification Trees: Introduction Implementation")
    print("=" * 60)
    
    # 1. Impurity measures demonstration
    print("\n1. Impurity Measures Demonstration:")
    impurity_results = demonstrate_impurity_measures()
    
    # 2. Tree structure demonstration
    print("\n2. Tree Structure Demonstration:")
    structure_results = demonstrate_tree_structure()
    
    # 3. Stopping criteria demonstration
    print("\n3. Stopping Criteria Demonstration:")
    stopping_results = demonstrate_stopping_criteria()
    
    # 4. Greedy algorithm demonstration
    print("\n4. Greedy Algorithm Demonstration:")
    greedy_results = demonstrate_greedy_algorithm()
    
    # 5. Advantages and limitations
    print("\n5. Advantages and Limitations:")
    limitations_results = demonstrate_advantages_limitations()
    
    print("\n=== Key Insights ===")
    print("1. Impurity measures (Gini, Entropy) control split quality")
    print("2. Stopping criteria prevent overfitting")
    print("3. Greedy algorithm is computationally efficient")
    print("4. Trees create axis-aligned decision boundaries")
    print("5. Tree structure provides interpretability")
    print("6. Trees can struggle with non-linear patterns")
    print("7. Different impurity measures may lead to different trees")
    print("8. Tree depth controls model complexity")
    
    return {
        'impurity_results': impurity_results,
        'structure_results': structure_results,
        'stopping_results': stopping_results,
        'greedy_results': greedy_results,
        'limitations_results': limitations_results
    }


if __name__ == "__main__":
    main()
