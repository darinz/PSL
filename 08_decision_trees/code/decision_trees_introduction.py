"""
Decision Trees Introduction - Python Implementation

This module demonstrates the fundamental concepts of decision trees including:
- Recursive splitting and region partitioning
- Entropy loss and Gini impurity calculations
- Regression trees with least-squares loss
- Regularization techniques
- Runtime complexity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time

class DecisionTreeDemo:
    """Demonstration class for decision tree concepts"""
    
    def __init__(self):
        self.rng = np.random.RandomState(42)
    
    def create_2d_dataset(self, n_samples=200, noise=0.3):
        """Create a 2D dataset for visualization"""
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1, 
            random_state=42,
            noise=noise
        )
        return X, y
    
    def entropy_loss(self, y):
        """Calculate entropy loss for a region"""
        if len(y) == 0:
            return 0
        
        # Calculate class probabilities
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy (avoid log(0))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def gini_impurity(self, y):
        """Calculate Gini impurity for a region"""
        if len(y) == 0:
            return 0
        
        # Calculate class probabilities
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate Gini impurity
        gini = 1 - np.sum(probabilities**2)
        return gini
    
    def information_gain(self, y_parent, y_left, y_right):
        """Calculate information gain for a split"""
        # Parent entropy
        parent_entropy = self.entropy_loss(y_parent)
        
        # Weighted average of children entropy
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0
        
        left_entropy = self.entropy_loss(y_left)
        right_entropy = self.entropy_loss(y_right)
        
        weighted_entropy = (n_left * left_entropy + n_right * right_entropy) / n_total
        
        return parent_entropy - weighted_entropy
    
    def find_best_split(self, X, y):
        """Find the best split for a region (greedy approach)"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Create split
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self.information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def demonstrate_recursive_splitting(self):
        """Demonstrate recursive splitting process"""
        print("=== Recursive Splitting Demonstration ===")
        
        # Create dataset
        X, y = self.create_2d_dataset(n_samples=100, noise=0.2)
        
        # Initial region (all data)
        print(f"Initial region: {len(y)} samples, entropy: {self.entropy_loss(y):.3f}")
        
        # First split
        feature, threshold, gain = self.find_best_split(X, y)
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        print(f"Best split: Feature {feature} < {threshold:.3f}")
        print(f"Information gain: {gain:.3f}")
        print(f"Left region: {len(y_left)} samples, entropy: {self.entropy_loss(y_left):.3f}")
        print(f"Right region: {len(y_right)} samples, entropy: {self.entropy_loss(y_right):.3f}")
        
        return X, y, feature, threshold
    
    def compare_entropy_gini(self):
        """Compare entropy loss vs Gini impurity"""
        print("\n=== Entropy vs Gini Impurity Comparison ===")
        
        # Create different class distributions
        distributions = [
            ([1, 1, 1, 1, 1], "Pure class 1"),
            ([1, 1, 1, 0, 0], "80% class 1"),
            ([1, 1, 0, 0, 0], "60% class 1"),
            ([1, 0, 0, 0, 0], "20% class 1"),
            ([1, 1, 1, 1, 0], "80% class 1"),
        ]
        
        print(f"{'Distribution':<15} {'Entropy':<10} {'Gini':<10}")
        print("-" * 40)
        
        for y, desc in distributions:
            entropy = self.entropy_loss(y)
            gini = self.gini_impurity(y)
            print(f"{desc:<15} {entropy:<10.3f} {gini:<10.3f}")
    
    def regression_tree_demo(self):
        """Demonstrate regression trees"""
        print("\n=== Regression Tree Demonstration ===")
        
        # Create regression dataset
        X, y = make_regression(n_samples=200, n_features=2, noise=0.5, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train regression tree
        reg_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        reg_tree.fit(X_train, y_train)
        
        # Predictions
        y_pred = reg_tree.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Regression Tree MSE: {mse:.3f}")
        print(f"Tree depth: {reg_tree.get_depth()}")
        print(f"Number of leaves: {reg_tree.get_n_leaves()}")
        
        return reg_tree, X_test, y_test, y_pred
    
    def regularization_demo(self):
        """Demonstrate regularization techniques"""
        print("\n=== Regularization Demonstration ===")
        
        # Create dataset prone to overfitting
        X, y = self.create_2d_dataset(n_samples=50, noise=0.1)
        
        # Different regularization parameters
        configs = [
            {"max_depth": 10, "min_samples_leaf": 1, "name": "No regularization"},
            {"max_depth": 3, "min_samples_leaf": 1, "name": "Max depth = 3"},
            {"max_depth": 10, "min_samples_leaf": 5, "name": "Min leaf size = 5"},
            {"max_depth": 5, "min_samples_leaf": 3, "name": "Both constraints"},
        ]
        
        results = []
        
        for config in configs:
            tree = DecisionTreeClassifier(
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                random_state=42
            )
            
            # Time the training
            start_time = time.time()
            tree.fit(X, y)
            train_time = time.time() - start_time
            
            # Predictions
            y_pred = tree.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            results.append({
                "name": config["name"],
                "depth": tree.get_depth(),
                "leaves": tree.get_n_leaves(),
                "accuracy": accuracy,
                "train_time": train_time
            })
        
        # Display results
        print(f"{'Configuration':<20} {'Depth':<6} {'Leaves':<7} {'Accuracy':<10} {'Time (ms)':<10}")
        print("-" * 60)
        for result in results:
            print(f"{result['name']:<20} {result['depth']:<6} {result['leaves']:<7} "
                  f"{result['accuracy']:<10.3f} {result['train_time']*1000:<10.1f}")
    
    def runtime_complexity_demo(self):
        """Demonstrate runtime complexity"""
        print("\n=== Runtime Complexity Analysis ===")
        
        # Test different dataset sizes
        sizes = [100, 500, 1000, 2000, 5000]
        train_times = []
        test_times = []
        
        for size in sizes:
            # Create dataset
            X, y = make_classification(n_samples=size, n_features=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train tree
            tree = DecisionTreeClassifier(random_state=42)
            
            start_time = time.time()
            tree.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            start_time = time.time()
            tree.predict(X_test)
            test_time = time.time() - start_time
            
            train_times.append(train_time)
            test_times.append(test_time)
        
        # Display results
        print(f"{'Size':<8} {'Train (ms)':<12} {'Test (ms)':<12} {'Depth':<8}")
        print("-" * 45)
        for i, size in enumerate(sizes):
            tree = DecisionTreeClassifier(random_state=42)
            X, y = make_classification(n_samples=size, n_features=10, random_state=42)
            tree.fit(X, y)
            print(f"{size:<8} {train_times[i]*1000:<12.1f} {test_times[i]*1000:<12.1f} {tree.get_depth():<8}")
    
    def visualize_decision_boundaries(self):
        """Visualize decision boundaries for different depths"""
        print("\n=== Decision Boundary Visualization ===")
        
        # Create dataset
        X, y = self.create_2d_dataset(n_samples=200, noise=0.3)
        
        # Create subplots for different depths
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        depths = [1, 2, 3, 5, 8, 10]
        
        for i, depth in enumerate(depths):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Train tree
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(X, y)
            
            # Create mesh for decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            # Predict on mesh
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.4)
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black')
            ax.set_title(f'Depth = {depth}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Decision boundary visualization saved as 'decision_boundaries.png'")

def main():
    """Main demonstration function"""
    demo = DecisionTreeDemo()
    
    # Run all demonstrations
    demo.demonstrate_recursive_splitting()
    demo.compare_entropy_gini()
    demo.regression_tree_demo()
    demo.regularization_demo()
    demo.runtime_complexity_demo()
    demo.visualize_decision_boundaries()

if __name__ == "__main__":
    main()
