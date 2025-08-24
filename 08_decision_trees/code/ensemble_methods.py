"""
Ensemble Methods - Python Implementation

This module demonstrates ensemble methods including bagging, random forests,
and model averaging techniques for improving decision tree performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, make_circles
import warnings
warnings.filterwarnings('ignore')

class EnsembleMethodsDemo:
    """Demonstration class for ensemble methods"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def create_ensemble_dataset(self, n_samples=300, noise=0.3):
        """Create a dataset suitable for ensemble methods demonstration"""
        # Create a complex dataset that benefits from ensemble methods
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=self.random_state)
        return X, y
    
    def demonstrate_single_tree_limitations(self):
        """Demonstrate limitations of single decision trees"""
        print("=== Single Decision Tree Limitations ===")
        
        # Create dataset
        X, y = self.create_ensemble_dataset(n_samples=400, noise=0.25)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different single tree configurations
        tree_configs = [
            {"name": "Shallow Tree (depth=3)", "params": {"max_depth": 3}},
            {"name": "Medium Tree (depth=5)", "params": {"max_depth": 5}},
            {"name": "Deep Tree (depth=10)", "params": {"max_depth": 10}},
            {"name": "Unlimited Tree", "params": {}}
        ]
        
        results = []
        
        for config in tree_configs:
            tree = DecisionTreeClassifier(
                random_state=self.random_state,
                **config["params"]
            )
            
            # Train and evaluate
            tree.fit(X_train, y_train)
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(tree, X_train, y_train, cv=5)
            
            results.append({
                "name": config["name"],
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "tree": tree
            })
            
            print(f"{config['name']:<25} Train: {train_score:.3f}, "
                  f"Test: {test_score:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results, X_train, y_train, X_test, y_test
    
    def visualize_single_trees(self, results, X, y):
        """Visualize single decision trees and their limitations"""
        print("\n=== Single Trees Visualization ===")
        
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
            ax.set_title(f"{result['name']}\n"
                        f"Train: {result['train_score']:.3f}, Test: {result['test_score']:.3f}")
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('single_trees_limitations.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Single trees visualization saved as 'single_trees_limitations.png'")
    
    def bagging_demo(self):
        """Demonstrate bagging (Bootstrap Aggregating)"""
        print("\n=== Bagging Demonstration ===")
        
        # Create dataset
        X, y = self.create_ensemble_dataset(n_samples=500, noise=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different numbers of estimators
        n_estimators_list = [1, 5, 10, 20, 50, 100]
        results = []
        
        for n_estimators in n_estimators_list:
            # Create bagging classifier
            bagging = BaggingClassifier(
                DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                n_estimators=n_estimators,
                random_state=self.random_state
            )
            
            # Train and evaluate
            bagging.fit(X_train, y_train)
            train_score = bagging.score(X_train, y_train)
            test_score = bagging.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(bagging, X_train, y_train, cv=5)
            
            results.append({
                "n_estimators": n_estimators,
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classifier": bagging
            })
            
            print(f"Estimators: {n_estimators:3d}, Train: {train_score:.3f}, "
                  f"Test: {test_score:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results, X_train, y_train, X_test, y_test
    
    def random_forest_demo(self):
        """Demonstrate Random Forest"""
        print("\n=== Random Forest Demonstration ===")
        
        # Create dataset
        X, y = self.create_ensemble_dataset(n_samples=500, noise=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Test different Random Forest configurations
        rf_configs = [
            {"name": "Small RF (10 trees)", "params": {"n_estimators": 10}},
            {"name": "Medium RF (50 trees)", "params": {"n_estimators": 50}},
            {"name": "Large RF (100 trees)", "params": {"n_estimators": 100}},
            {"name": "RF with max_features=1", "params": {"n_estimators": 50, "max_features": 1}},
            {"name": "RF with max_depth=3", "params": {"n_estimators": 50, "max_depth": 3}},
            {"name": "RF with bootstrap=False", "params": {"n_estimators": 50, "bootstrap": False}}
        ]
        
        results = []
        
        for config in rf_configs:
            rf = RandomForestClassifier(
                random_state=self.random_state,
                **config["params"]
            )
            
            # Train and evaluate
            rf.fit(X_train, y_train)
            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
            
            results.append({
                "name": config["name"],
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classifier": rf
            })
            
            print(f"{config['name']:<25} Train: {train_score:.3f}, "
                  f"Test: {test_score:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results, X_train, y_train, X_test, y_test
    
    def analyze_ensemble_performance(self, bagging_results, rf_results):
        """Analyze performance of different ensemble methods"""
        print("\n=== Ensemble Performance Analysis ===")
        
        # Create comparison plots
        plt.figure(figsize=(15, 5))
        
        # Bagging performance
        plt.subplot(1, 3, 1)
        n_estimators = [r["n_estimators"] for r in bagging_results]
        bagging_train = [r["train_score"] for r in bagging_results]
        bagging_test = [r["test_score"] for r in bagging_results]
        
        plt.plot(n_estimators, bagging_train, 'o-', label='Training', linewidth=2)
        plt.plot(n_estimators, bagging_test, 's-', label='Test', linewidth=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('Bagging Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Random Forest comparison
        plt.subplot(1, 3, 2)
        rf_names = [r["name"] for r in rf_results]
        rf_test = [r["test_score"] for r in rf_results]
        
        x_pos = np.arange(len(rf_names))
        plt.bar(x_pos, rf_test, alpha=0.8)
        plt.xlabel('Random Forest Configuration')
        plt.ylabel('Test Accuracy')
        plt.title('Random Forest Configurations')
        plt.xticks(x_pos, [name.split('(')[0].strip() for name in rf_names], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Overfitting comparison
        plt.subplot(1, 3, 3)
        bagging_gap = np.array(bagging_train) - np.array(bagging_test)
        rf_gaps = [r["train_score"] - r["test_score"] for r in rf_results]
        
        plt.plot(n_estimators, bagging_gap, 'o-', label='Bagging', linewidth=2)
        plt.axhline(y=np.mean(rf_gaps), color='red', linestyle='--', label='RF Average')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Overfitting Gap')
        plt.title('Overfitting Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Ensemble performance analysis saved as 'ensemble_performance.png'")
        
        return bagging_results, rf_results
    
    def visualize_ensemble_decision_boundaries(self, bagging_results, rf_results, X, y):
        """Visualize decision boundaries of ensemble methods"""
        print("\n=== Ensemble Decision Boundaries Visualization ===")
        
        # Select specific configurations to visualize
        viz_configs = [
            ("Bagging (10 estimators)", bagging_results[2]["classifier"]),  # 10 estimators
            ("Bagging (50 estimators)", bagging_results[4]["classifier"]),  # 50 estimators
            ("Random Forest (50 trees)", rf_results[1]["classifier"]),      # Medium RF
            ("Random Forest (max_features=1)", rf_results[3]["classifier"]) # RF with max_features=1
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (name, classifier) in enumerate(viz_configs):
            ax = axes[i]
            
            # Create mesh for decision boundary
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                np.arange(y_min, y_max, 0.02))
            
            # Predict on mesh
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.4)
            ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black', s=30)
            ax.set_title(name)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.savefig('ensemble_decision_boundaries.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Ensemble decision boundaries saved as 'ensemble_decision_boundaries.png'")
    
    def analyze_feature_importance(self, rf_results, X_train):
        """Analyze feature importance in Random Forest"""
        print("\n=== Feature Importance Analysis ===")
        
        # Get the best Random Forest classifier
        best_rf = max(rf_results, key=lambda x: x["test_score"])
        rf = best_rf["classifier"]
        
        # Get feature importance
        feature_importance = rf.feature_importances_
        feature_names = [f"Feature {i+1}" for i in range(len(feature_importance))]
        
        # Create importance plot
        plt.figure(figsize=(10, 6))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Feature importance analysis saved as 'feature_importance.png'")
        
        # Print importance values
        print("Feature Importance:")
        for feature, importance in zip(feature_names, feature_importance):
            print(f"  {feature}: {importance:.4f}")
        
        return feature_importance
    
    def compare_ensemble_methods(self, single_tree_results, bagging_results, rf_results):
        """Compare all ensemble methods"""
        print("\n=== Ensemble Methods Comparison ===")
        
        # Get best results from each method
        best_single = max(single_tree_results, key=lambda x: x["test_score"])
        best_bagging = max(bagging_results, key=lambda x: x["test_score"])
        best_rf = max(rf_results, key=lambda x: x["test_score"])
        
        # Create comparison table
        comparison_data = {
            'Method': ['Single Tree', 'Bagging', 'Random Forest'],
            'Configuration': [
                best_single["name"],
                f"{best_bagging['n_estimators']} estimators",
                best_rf["name"]
            ],
            'Train Score': [best_single["train_score"], best_bagging["train_score"], best_rf["train_score"]],
            'Test Score': [best_single["test_score"], best_bagging["test_score"], best_rf["test_score"]],
            'CV Score': [best_single["cv_mean"], best_bagging["cv_mean"], best_rf["cv_mean"]],
            'Overfitting Gap': [
                best_single["train_score"] - best_single["test_score"],
                best_bagging["train_score"] - best_bagging["test_score"],
                best_rf["train_score"] - best_rf["test_score"]
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nEnsemble Methods Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Create comparison plot
        plt.figure(figsize=(12, 5))
        
        # Performance comparison
        plt.subplot(1, 2, 1)
        methods = comparison_df['Method']
        test_scores = comparison_df['Test Score']
        
        bars = plt.bar(methods, test_scores, alpha=0.8, color=['blue', 'green', 'red'])
        plt.ylabel('Test Accuracy')
        plt.title('Test Performance Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Overfitting comparison
        plt.subplot(1, 2, 2)
        overfitting_gaps = comparison_df['Overfitting Gap']
        
        bars = plt.bar(methods, overfitting_gaps, alpha=0.8, color=['blue', 'green', 'red'])
        plt.ylabel('Overfitting Gap')
        plt.title('Overfitting Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, gap in zip(bars, overfitting_gaps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{gap:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Ensemble comparison saved as 'ensemble_comparison.png'")
        
        return comparison_df
    
    def demonstrate_ensemble_robustness(self):
        """Demonstrate robustness of ensemble methods"""
        print("\n=== Ensemble Methods Robustness ===")
        
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = []
        
        for noise in noise_levels:
            # Create dataset with specific noise level
            X, y = self.create_ensemble_dataset(n_samples=400, noise=noise)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            # Train different methods
            single_tree = DecisionTreeClassifier(max_depth=5, random_state=self.random_state)
            bagging = BaggingClassifier(
                DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
                n_estimators=50, random_state=self.random_state
            )
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            
            # Train and evaluate
            single_tree.fit(X_train, y_train)
            bagging.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            
            single_score = single_tree.score(X_test, y_test)
            bagging_score = bagging.score(X_test, y_test)
            rf_score = rf.score(X_test, y_test)
            
            results.append({
                "noise": noise,
                "single_tree": single_score,
                "bagging": bagging_score,
                "random_forest": rf_score
            })
            
            print(f"Noise: {noise:.1f}, Single: {single_score:.3f}, "
                  f"Bagging: {bagging_score:.3f}, RF: {rf_score:.3f}")
        
        # Plot robustness comparison
        plt.figure(figsize=(10, 6))
        
        noise_levels = [r["noise"] for r in results]
        single_scores = [r["single_tree"] for r in results]
        bagging_scores = [r["bagging"] for r in results]
        rf_scores = [r["random_forest"] for r in results]
        
        plt.plot(noise_levels, single_scores, 'o-', label='Single Tree', linewidth=2)
        plt.plot(noise_levels, bagging_scores, 's-', label='Bagging', linewidth=2)
        plt.plot(noise_levels, rf_scores, '^-', label='Random Forest', linewidth=2)
        plt.xlabel('Noise Level')
        plt.ylabel('Test Accuracy')
        plt.title('Robustness to Noise: Ensemble Methods Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_robustness.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Ensemble robustness analysis saved as 'ensemble_robustness.png'")
        
        return results
    
    def run_complete_analysis(self):
        """Run complete ensemble methods analysis"""
        print("=== Complete Ensemble Methods Analysis ===")
        
        # 1. Single tree limitations
        single_results, X_train, y_train, X_test, y_test = self.demonstrate_single_tree_limitations()
        self.visualize_single_trees(single_results, X_train, y_train)
        
        # 2. Bagging demonstration
        bagging_results, X_train, y_train, X_test, y_test = self.bagging_demo()
        
        # 3. Random Forest demonstration
        rf_results, X_train, y_train, X_test, y_test = self.random_forest_demo()
        
        # 4. Performance analysis
        self.analyze_ensemble_performance(bagging_results, rf_results)
        
        # 5. Decision boundaries visualization
        self.visualize_ensemble_decision_boundaries(bagging_results, rf_results, X_train, y_train)
        
        # 6. Feature importance analysis
        self.analyze_feature_importance(rf_results, X_train)
        
        # 7. Methods comparison
        self.compare_ensemble_methods(single_results, bagging_results, rf_results)
        
        # 8. Robustness demonstration
        self.demonstrate_ensemble_robustness()
        
        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("- single_trees_limitations.png: Single trees visualization")
        print("- ensemble_performance.png: Performance analysis")
        print("- ensemble_decision_boundaries.png: Decision boundaries")
        print("- feature_importance.png: Feature importance analysis")
        print("- ensemble_comparison.png: Methods comparison")
        print("- ensemble_robustness.png: Robustness analysis")

def main():
    """Main function to run the ensemble methods analysis"""
    demo = EnsembleMethodsDemo(random_state=42)
    demo.run_complete_analysis()

if __name__ == "__main__":
    main()
