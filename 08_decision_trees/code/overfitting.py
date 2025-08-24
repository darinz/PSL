"""
Overfitting in Decision Trees - Python Implementation

This module demonstrates overfitting in decision trees and various techniques
to control it, including early stopping, pruning, and regularization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

class OverfittingDemo:
    """Demonstration class for overfitting in decision trees"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def create_overfitting_dataset(self, n_samples=200, noise=0.3):
        """Create a dataset prone to overfitting"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=self.random_state,
            noise=noise
        )
        return X, y
    
    def demonstrate_depth_vs_performance(self):
        """Demonstrate how tree depth affects performance"""
        print("=== Tree Depth vs Performance Analysis ===")
        
        # Create dataset
        X, y = self.create_overfitting_dataset(n_samples=200, noise=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different depths
        depths = [1, 2, 3, 5, 8, 10, 15, 20]
        train_scores = []
        test_scores = []
        tree_depths = []
        num_leaves = []
        
        for depth in depths:
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )
            tree.fit(X_train, y_train)
            
            # Calculate scores
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            tree_depths.append(tree.get_depth())
            num_leaves.append(tree.get_n_leaves())
            
            print(f"Depth {depth:2d}: Train={train_score:.3f}, Test={test_score:.3f}, "
                  f"Actual Depth={tree.get_depth()}, Leaves={tree.get_n_leaves()}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Training vs Test accuracy
        plt.subplot(1, 3, 1)
        plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
        plt.plot(depths, test_scores, 's-', label='Test Accuracy', linewidth=2)
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Tree Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Number of leaves
        plt.subplot(1, 3, 2)
        plt.plot(depths, num_leaves, 'o-', color='green', linewidth=2)
        plt.xlabel('Max Depth')
        plt.ylabel('Number of Leaves')
        plt.title('Tree Complexity vs Depth')
        plt.grid(True, alpha=0.3)
        
        # Overfitting gap
        plt.subplot(1, 3, 3)
        overfitting_gap = np.array(train_scores) - np.array(test_scores)
        plt.plot(depths, overfitting_gap, 'o-', color='red', linewidth=2)
        plt.xlabel('Max Depth')
        plt.ylabel('Overfitting Gap')
        plt.title('Training-Test Gap vs Depth')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('depth_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Depth vs performance analysis saved as 'depth_vs_performance.png'")
        
        return depths, train_scores, test_scores, num_leaves
    
    def early_stopping_demo(self):
        """Demonstrate early stopping techniques"""
        print("\n=== Early Stopping Demonstration ===")
        
        # Create dataset
        X, y = self.create_overfitting_dataset(n_samples=150, noise=0.25)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Different early stopping conditions
        conditions = [
            {"max_depth": 10, "min_samples_split": 1, "min_samples_leaf": 1, "name": "No constraints"},
            {"max_depth": 3, "min_samples_split": 1, "min_samples_leaf": 1, "name": "Max depth = 3"},
            {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 1, "name": "Min split = 5"},
            {"max_depth": 10, "min_samples_split": 1, "min_samples_leaf": 5, "name": "Min leaf = 5"},
            {"max_depth": 5, "min_samples_split": 3, "min_samples_leaf": 3, "name": "All constraints"},
        ]
        
        results = []
        
        for condition in conditions:
            tree = DecisionTreeClassifier(
                max_depth=condition["max_depth"],
                min_samples_split=condition["min_samples_split"],
                min_samples_leaf=condition["min_samples_leaf"],
                random_state=self.random_state
            )
            
            tree.fit(X_train, y_train)
            
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            results.append({
                "name": condition["name"],
                "depth": tree.get_depth(),
                "leaves": tree.get_n_leaves(),
                "train_score": train_score,
                "test_score": test_score,
                "overfitting_gap": train_score - test_score
            })
        
        # Display results
        print(f"{'Condition':<20} {'Depth':<6} {'Leaves':<7} {'Train':<8} {'Test':<8} {'Gap':<8}")
        print("-" * 65)
        for result in results:
            print(f"{result['name']:<20} {result['depth']:<6} {result['leaves']:<7} "
                  f"{result['train_score']:<8.3f} {result['test_score']:<8.3f} "
                  f"{result['overfitting_gap']:<8.3f}")
        
        return results
    
    def pruning_demo(self):
        """Demonstrate pruning techniques"""
        print("\n=== Pruning Demonstration ===")
        
        # Create dataset
        X, y = self.create_overfitting_dataset(n_samples=200, noise=0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Train a complex tree first
        complex_tree = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=1,
            min_samples_leaf=1,
            random_state=self.random_state
        )
        complex_tree.fit(X_train, y_train)
        
        print(f"Complex tree - Depth: {complex_tree.get_depth()}, "
              f"Leaves: {complex_tree.get_n_leaves()}")
        print(f"Training accuracy: {complex_tree.score(X_train, y_train):.3f}")
        print(f"Test accuracy: {complex_tree.score(X_test, y_test):.3f}")
        
        # Cost complexity pruning
        path = complex_tree.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        
        pruned_trees = []
        for alpha in ccp_alphas:
            pruned_tree = DecisionTreeClassifier(
                random_state=self.random_state,
                ccp_alpha=alpha
            )
            pruned_tree.fit(X_train, y_train)
            pruned_trees.append(pruned_tree)
        
        # Evaluate pruned trees
        train_scores = [tree.score(X_train, y_train) for tree in pruned_trees]
        test_scores = [tree.score(X_test, y_test) for tree in pruned_trees]
        depths = [tree.get_depth() for tree in pruned_trees]
        leaves = [tree.get_n_leaves() for tree in pruned_trees]
        
        # Plot pruning results
        plt.figure(figsize=(15, 5))
        
        # Accuracy vs alpha
        plt.subplot(1, 3, 1)
        plt.plot(ccp_alphas, train_scores, 'o-', label='Training', linewidth=2)
        plt.plot(ccp_alphas, test_scores, 's-', label='Test', linewidth=2)
        plt.xlabel('CCP Alpha')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Pruning Parameter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tree depth vs alpha
        plt.subplot(1, 3, 2)
        plt.plot(ccp_alphas, depths, 'o-', color='green', linewidth=2)
        plt.xlabel('CCP Alpha')
        plt.ylabel('Tree Depth')
        plt.title('Tree Depth vs Pruning Parameter')
        plt.grid(True, alpha=0.3)
        
        # Number of leaves vs alpha
        plt.subplot(1, 3, 3)
        plt.plot(ccp_alphas, leaves, 'o-', color='red', linewidth=2)
        plt.xlabel('CCP Alpha')
        plt.ylabel('Number of Leaves')
        plt.title('Tree Complexity vs Pruning Parameter')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pruning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Pruning analysis saved as 'pruning_analysis.png'")
        
        # Find optimal alpha
        best_idx = np.argmax(test_scores)
        optimal_alpha = ccp_alphas[best_idx]
        optimal_tree = pruned_trees[best_idx]
        
        print(f"\nOptimal pruning parameter (alpha): {optimal_alpha:.4f}")
        print(f"Optimal tree - Depth: {optimal_tree.get_depth()}, "
              f"Leaves: {optimal_tree.get_n_leaves()}")
        print(f"Optimal test accuracy: {test_scores[best_idx]:.3f}")
        
        return pruned_trees, ccp_alphas, test_scores
    
    def regularization_comparison(self):
        """Compare different regularization techniques"""
        print("\n=== Regularization Techniques Comparison ===")
        
        # Create dataset
        X, y = self.create_overfitting_dataset(n_samples=300, noise=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Different regularization approaches
        approaches = [
            {"name": "No regularization", "params": {}},
            {"name": "Max depth only", "params": {"max_depth": 4}},
            {"name": "Min samples only", "params": {"min_samples_split": 10, "min_samples_leaf": 5}},
            {"name": "Max features only", "params": {"max_features": 1}},
            {"name": "All constraints", "params": {
                "max_depth": 4,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": 1
            }}
        ]
        
        results = []
        
        for approach in approaches:
            tree = DecisionTreeClassifier(
                random_state=self.random_state,
                **approach["params"]
            )
            
            # Cross-validation
            cv_scores = cross_val_score(tree, X_train, y_train, cv=5)
            
            # Train on full training set
            tree.fit(X_train, y_train)
            
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            results.append({
                "name": approach["name"],
                "depth": tree.get_depth(),
                "leaves": tree.get_n_leaves(),
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "overfitting_gap": train_score - test_score
            })
        
        # Display results
        print(f"{'Approach':<20} {'Depth':<6} {'Leaves':<7} {'Train':<8} {'Test':<8} {'CV':<8} {'Gap':<8}")
        print("-" * 75)
        for result in results:
            print(f"{result['name']:<20} {result['depth']:<6} {result['leaves']:<7} "
                  f"{result['train_score']:<8.3f} {result['test_score']:<8.3f} "
                  f"{result['cv_mean']:<8.3f} {result['overfitting_gap']:<8.3f}")
        
        return results
    
    def bias_variance_analysis(self):
        """Analyze bias-variance trade-off"""
        print("\n=== Bias-Variance Trade-off Analysis ===")
        
        # Create multiple datasets
        n_datasets = 20
        n_samples = 100
        
        all_train_scores = []
        all_test_scores = []
        depths = [1, 2, 3, 5, 8, 10]
        
        for depth in depths:
            train_scores = []
            test_scores = []
            
            for _ in range(n_datasets):
                # Create dataset with different random seeds
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=2,
                    n_redundant=0,
                    n_informative=2,
                    n_clusters_per_class=1,
                    random_state=self.rng.randint(0, 1000),
                    noise=0.3
                )
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=self.random_state
                )
                
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=self.random_state
                )
                tree.fit(X_train, y_train)
                
                train_scores.append(tree.score(X_train, y_train))
                test_scores.append(tree.score(X_test, y_test))
            
            all_train_scores.append(train_scores)
            all_test_scores.append(test_scores)
        
        # Calculate bias and variance
        train_means = [np.mean(scores) for scores in all_train_scores]
        train_stds = [np.std(scores) for scores in all_train_scores]
        test_means = [np.mean(scores) for scores in all_test_scores]
        test_stds = [np.std(scores) for scores in all_test_scores]
        
        # Plot bias-variance trade-off
        plt.figure(figsize=(15, 5))
        
        # Training performance
        plt.subplot(1, 3, 1)
        plt.errorbar(depths, train_means, yerr=train_stds, 
                    fmt='o-', label='Training', capsize=5, linewidth=2)
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Training Performance (Bias)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Test performance
        plt.subplot(1, 3, 2)
        plt.errorbar(depths, test_means, yerr=test_stds, 
                    fmt='s-', label='Test', capsize=5, linewidth=2)
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Test Performance (Variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Variance (std) vs depth
        plt.subplot(1, 3, 3)
        plt.plot(depths, train_stds, 'o-', label='Training Variance', linewidth=2)
        plt.plot(depths, test_stds, 's-', label='Test Variance', linewidth=2)
        plt.xlabel('Tree Depth')
        plt.ylabel('Standard Deviation')
        plt.title('Variance vs Tree Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bias_variance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Bias-variance analysis saved as 'bias_variance_analysis.png'")
        
        return depths, train_means, test_means, train_stds, test_stds
    
    def visualize_overfitting(self):
        """Visualize overfitting with decision boundaries"""
        print("\n=== Overfitting Visualization ===")
        
        # Create dataset
        X, y = self.create_overfitting_dataset(n_samples=150, noise=0.2)
        
        # Create subplots for different depths
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        depths = [1, 2, 3, 5, 8, 10, 15, 20]
        
        for i, depth in enumerate(depths):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state
            )
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
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black', s=30)
            ax.set_title(f'Depth = {depth}\nLeaves = {tree.get_n_leaves()}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('overfitting_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Overfitting visualization saved as 'overfitting_visualization.png'")
    
    def run_complete_analysis(self):
        """Run complete overfitting analysis"""
        print("=== Complete Overfitting Analysis ===")
        
        # Run all demonstrations
        self.demonstrate_depth_vs_performance()
        self.early_stopping_demo()
        self.pruning_demo()
        self.regularization_comparison()
        self.bias_variance_analysis()
        self.visualize_overfitting()
        
        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("- depth_vs_performance.png: Depth vs performance analysis")
        print("- pruning_analysis.png: Pruning parameter analysis")
        print("- bias_variance_analysis.png: Bias-variance trade-off")
        print("- overfitting_visualization.png: Decision boundary visualization")

def main():
    """Main function to run the overfitting analysis"""
    demo = OverfittingDemo(random_state=42)
    demo.run_complete_analysis()

if __name__ == "__main__":
    main()
