"""
Boosting - Python Implementation

This module demonstrates boosting algorithms including AdaBoost,
weak learners, and ensemble methods for improving classification performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_circles
import warnings
warnings.filterwarnings('ignore')

class BoostingDemo:
    """Demonstration class for boosting algorithms"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def create_boosting_dataset(self, n_samples=200, noise=0.3):
        """Create a dataset suitable for boosting demonstration"""
        # Create a dataset that's difficult for simple classifiers
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=self.random_state)
        return X, y
    
    def create_weak_learners_demo(self):
        """Demonstrate weak learners and their limitations"""
        print("=== Weak Learners Demonstration ===")
        
        # Create dataset
        X, y = self.create_boosting_dataset(n_samples=300, noise=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different weak learners
        weak_learners = [
            {"name": "Decision Stump (depth=1)", "params": {"max_depth": 1}},
            {"name": "Shallow Tree (depth=2)", "params": {"max_depth": 2}},
            {"name": "Very Shallow Tree (depth=1, min_samples=20)", 
             "params": {"max_depth": 1, "min_samples_split": 20}},
            {"name": "Linear Separator (depth=1, max_features=1)", 
             "params": {"max_depth": 1, "max_features": 1}}
        ]
        
        results = []
        
        for learner in weak_learners:
            tree = DecisionTreeClassifier(
                random_state=self.random_state,
                **learner["params"]
            )
            
            # Train and evaluate
            tree.fit(X_train, y_train)
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            results.append({
                "name": learner["name"],
                "train_score": train_score,
                "test_score": test_score,
                "tree": tree
            })
            
            print(f"{learner['name']:<40} Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return results, X_train, y_train, X_test, y_test
    
    def visualize_weak_learners(self, results, X, y):
        """Visualize weak learners and their decision boundaries"""
        print("\n=== Weak Learners Visualization ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, result in enumerate(results):
            tree = result["tree"]
            ax = axes[i]
            
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
            ax.set_title(f"{result['name']}\nTrain: {result['train_score']:.3f}, Test: {result['test_score']:.3f}")
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('weak_learners.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Weak learners visualization saved as 'weak_learners.png'")
    
    def adaboost_demo(self):
        """Demonstrate AdaBoost algorithm"""
        print("\n=== AdaBoost Demonstration ===")
        
        # Create dataset
        X, y = self.create_boosting_dataset(n_samples=400, noise=0.25)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different numbers of estimators
        n_estimators_list = [1, 5, 10, 20, 50, 100]
        results = []
        
        for n_estimators in n_estimators_list:
            # Create AdaBoost classifier
            ada = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=1, random_state=self.random_state),
                n_estimators=n_estimators,
                random_state=self.random_state
            )
            
            # Train and evaluate
            ada.fit(X_train, y_train)
            train_score = ada.score(X_train, y_train)
            test_score = ada.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(ada, X_train, y_train, cv=5)
            
            results.append({
                "n_estimators": n_estimators,
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classifier": ada
            })
            
            print(f"Estimators: {n_estimators:3d}, Train: {train_score:.3f}, "
                  f"Test: {test_score:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results, X_train, y_train, X_test, y_test
    
    def visualize_adaboost_progression(self, results, X, y):
        """Visualize AdaBoost progression with different numbers of estimators"""
        print("\n=== AdaBoost Progression Visualization ===")
        
        # Select specific numbers of estimators to visualize
        viz_estimators = [1, 5, 10, 50]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, n_est in enumerate(viz_estimators):
            # Find the corresponding result
            result = next(r for r in results if r["n_estimators"] == n_est)
            ada = result["classifier"]
            ax = axes[i]
            
            # Create mesh for decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            # Predict on mesh
            Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.4)
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black', s=30)
            ax.set_title(f"AdaBoost with {n_est} estimators\n"
                        f"Train: {result['train_score']:.3f}, Test: {result['test_score']:.3f}")
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('adaboost_progression.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("AdaBoost progression visualization saved as 'adaboost_progression.png'")
    
    def analyze_boosting_performance(self, results):
        """Analyze boosting performance vs number of estimators"""
        print("\n=== Boosting Performance Analysis ===")
        
        # Extract data for plotting
        n_estimators = [r["n_estimators"] for r in results]
        train_scores = [r["train_score"] for r in results]
        test_scores = [r["test_score"] for r in results]
        cv_means = [r["cv_mean"] for r in results]
        cv_stds = [r["cv_std"] for r in results]
        
        # Create plots
        plt.figure(figsize=(15, 5))
        
        # Training vs Test accuracy
        plt.subplot(1, 3, 1)
        plt.plot(n_estimators, train_scores, 'o-', label='Training', linewidth=2)
        plt.plot(n_estimators, test_scores, 's-', label='Test', linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Number of Estimators')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cross-validation scores
        plt.subplot(1, 3, 2)
        plt.errorbar(n_estimators, cv_means, yerr=cv_stds, 
                    fmt='o-', capsize=5, linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Cross-validation Accuracy')
        plt.title('CV Accuracy vs Number of Estimators')
        plt.grid(True, alpha=0.3)
        
        # Overfitting gap
        plt.subplot(1, 3, 3)
        overfitting_gap = np.array(train_scores) - np.array(test_scores)
        plt.plot(n_estimators, overfitting_gap, 'o-', color='red', linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Overfitting Gap')
        plt.title('Training-Test Gap vs Number of Estimators')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('boosting_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Boosting performance analysis saved as 'boosting_performance.png'")
        
        return n_estimators, train_scores, test_scores, cv_means
    
    def analyze_estimator_weights(self, ada_classifier, X_train, y_train):
        """Analyze the weights of individual estimators in AdaBoost"""
        print("\n=== Estimator Weights Analysis ===")
        
        # Get estimator weights
        weights = ada_classifier.estimator_weights_
        n_estimators = len(weights)
        
        # Plot weights
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, n_estimators + 1), weights, 'o-', linewidth=2)
        plt.xlabel('Estimator Index')
        plt.ylabel('Weight')
        plt.title('Estimator Weights in AdaBoost')
        plt.grid(True, alpha=0.3)
        
        # Cumulative weights
        plt.subplot(1, 2, 2)
        cumulative_weights = np.cumsum(weights)
        plt.plot(range(1, n_estimators + 1), cumulative_weights, 's-', linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Cumulative Weight')
        plt.title('Cumulative Estimator Weights')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('estimator_weights.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Estimator weights analysis saved as 'estimator_weights.png'")
        
        # Print statistics
        print(f"Number of estimators: {n_estimators}")
        print(f"Average weight: {np.mean(weights):.4f}")
        print(f"Weight standard deviation: {np.std(weights):.4f}")
        print(f"Min weight: {np.min(weights):.4f}")
        print(f"Max weight: {np.max(weights):.4f}")
        
        return weights
    
    def compare_with_single_tree(self, ada_results, X_train, y_train, X_test, y_test):
        """Compare AdaBoost with a single deep decision tree"""
        print("\n=== AdaBoost vs Single Deep Tree Comparison ===")
        
        # Train a single deep tree
        deep_tree = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state
        )
        deep_tree.fit(X_train, y_train)
        
        # Evaluate deep tree
        deep_tree_train = deep_tree.score(X_train, y_train)
        deep_tree_test = deep_tree.score(X_test, y_test)
        deep_tree_cv = cross_val_score(deep_tree, X_train, y_train, cv=5)
        
        print(f"Single Deep Tree:")
        print(f"  Train accuracy: {deep_tree_train:.3f}")
        print(f"  Test accuracy: {deep_tree_test:.3f}")
        print(f"  CV accuracy: {deep_tree_cv.mean():.3f} ± {deep_tree_cv.std():.3f}")
        print(f"  Tree depth: {deep_tree.get_depth()}")
        print(f"  Number of leaves: {deep_tree.get_n_leaves()}")
        
        # Compare with best AdaBoost result
        best_ada = max(ada_results, key=lambda x: x["test_score"])
        print(f"\nBest AdaBoost ({best_ada['n_estimators']} estimators):")
        print(f"  Train accuracy: {best_ada['train_score']:.3f}")
        print(f"  Test accuracy: {best_ada['test_score']:.3f}")
        print(f"  CV accuracy: {best_ada['cv_mean']:.3f} ± {best_ada['cv_std']:.3f}")
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        
        comparison_data = {
            'Single Deep Tree': [deep_tree_train, deep_tree_test, deep_tree_cv.mean()],
            'AdaBoost': [best_ada['train_score'], best_ada['test_score'], best_ada['cv_mean']]
        }
        
        x = np.arange(3)
        width = 0.35
        
        plt.bar(x - width/2, comparison_data['Single Deep Tree'], width, 
               label='Single Deep Tree', alpha=0.8)
        plt.bar(x + width/2, comparison_data['AdaBoost'], width, 
               label='AdaBoost', alpha=0.8)
        
        plt.xlabel('Metric')
        plt.ylabel('Accuracy')
        plt.title('Single Deep Tree vs AdaBoost Comparison')
        plt.xticks(x, ['Training', 'Test', 'CV'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('adaboost_vs_deep_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Comparison visualization saved as 'adaboost_vs_deep_tree.png'")
        
        return deep_tree, best_ada
    
    def demonstrate_boosting_robustness(self):
        """Demonstrate robustness of boosting to noise"""
        print("\n=== Boosting Robustness to Noise ===")
        
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for noise in noise_levels:
            # Create dataset with specific noise level
            X, y = self.create_boosting_dataset(n_samples=300, noise=noise)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            # Train AdaBoost
            ada = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=1, random_state=self.random_state),
                n_estimators=50,
                random_state=self.random_state
            )
            ada.fit(X_train, y_train)
            
            # Train single deep tree
            deep_tree = DecisionTreeClassifier(
                max_depth=10,
                random_state=self.random_state
            )
            deep_tree.fit(X_train, y_train)
            
            # Evaluate
            ada_test = ada.score(X_test, y_test)
            tree_test = deep_tree.score(X_test, y_test)
            
            results.append({
                "noise": noise,
                "adaboost": ada_test,
                "deep_tree": tree_test
            })
            
            print(f"Noise: {noise:.1f}, AdaBoost: {ada_test:.3f}, Deep Tree: {tree_test:.3f}")
        
        # Plot robustness comparison
        plt.figure(figsize=(10, 6))
        
        noise_levels = [r["noise"] for r in results]
        ada_scores = [r["adaboost"] for r in results]
        tree_scores = [r["deep_tree"] for r in results]
        
        plt.plot(noise_levels, ada_scores, 'o-', label='AdaBoost', linewidth=2)
        plt.plot(noise_levels, tree_scores, 's-', label='Deep Tree', linewidth=2)
        plt.xlabel('Noise Level')
        plt.ylabel('Test Accuracy')
        plt.title('Robustness to Noise: AdaBoost vs Deep Tree')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('boosting_robustness.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Robustness analysis saved as 'boosting_robustness.png'")
        
        return results
    
    def run_complete_analysis(self):
        """Run complete boosting analysis"""
        print("=== Complete Boosting Analysis ===")
        
        # 1. Weak learners demonstration
        weak_results, X_train, y_train, X_test, y_test = self.create_weak_learners_demo()
        self.visualize_weak_learners(weak_results, X_train, y_train)
        
        # 2. AdaBoost demonstration
        ada_results, X_train, y_train, X_test, y_test = self.adaboost_demo()
        self.visualize_adaboost_progression(ada_results, X_train, y_train)
        self.analyze_boosting_performance(ada_results)
        
        # 3. Estimator weights analysis
        best_ada = max(ada_results, key=lambda x: x["test_score"])
        self.analyze_estimator_weights(best_ada["classifier"], X_train, y_train)
        
        # 4. Comparison with single deep tree
        self.compare_with_single_tree(ada_results, X_train, y_train, X_test, y_test)
        
        # 5. Robustness demonstration
        self.demonstrate_boosting_robustness()
        
        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("- weak_learners.png: Weak learners visualization")
        print("- adaboost_progression.png: AdaBoost progression")
        print("- boosting_performance.png: Performance analysis")
        print("- estimator_weights.png: Estimator weights analysis")
        print("- adaboost_vs_deep_tree.png: Comparison with deep tree")
        print("- boosting_robustness.png: Robustness to noise")

def main():
    """Main function to run the boosting analysis"""
    demo = BoostingDemo(random_state=42)
    demo.run_complete_analysis()

if __name__ == "__main__":
    main()
