import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import seaborn as sns


class AdaBoost:
    def __init__(self, n_estimators=50, max_depth=1):
        """
        AdaBoost classifier
        
        Parameters:
        -----------
        n_estimators : int
            Number of weak learners
        max_depth : int
            Maximum depth of decision tree weak learners
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        
    def fit(self, X, y):
        """
        Fit AdaBoost classifier
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (should be {-1, 1})
        """
        n_samples = X.shape[0]
        
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples
        
        # Convert labels to {-1, 1} if needed
        y = np.array(y)
        if set(y) == {0, 1}:
            y = 2 * y - 1
        
        for t in range(self.n_estimators):
            # Train weak learner
            estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y
            error = np.average(incorrect, weights=sample_weights)
            
            # Handle case where error is 0 or >= 0.5
            if error <= 0:
                error = 1e-10
            elif error >= 0.5:
                error = 0.5 - 1e-10
                
            # Calculate estimator weight
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(alpha * incorrect * ((predictions != y) * 2 - 1))
            sample_weights /= np.sum(sample_weights)
            
            # Store results
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(error)
            
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            
        return np.sign(predictions)
    
    def staged_predict(self, X):
        """
        Return staged predictions for X
        """
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            yield np.sign(predictions)
    
    def get_feature_importance(self, X):
        """
        Get feature importance based on weighted average of weak learners
        """
        importance = np.zeros(X.shape[1])
        total_weight = sum(self.estimator_weights)
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            if hasattr(estimator, 'feature_importances_'):
                importance += (alpha / total_weight) * estimator.feature_importances_
                
        return importance


def demonstrate_basic_adaboost():
    """Demonstrate basic AdaBoost functionality"""
    print("=== Basic AdaBoost Demonstration ===\n")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42, class_sep=1.5)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert labels to {-1, 1}
    y_train_boost = 2 * y_train - 1
    y_test_boost = 2 * y_test - 1

    # Train AdaBoost
    ada = AdaBoost(n_estimators=50, max_depth=1)
    ada.fit(X_train, y_train_boost)

    # Make predictions
    y_pred = ada.predict(X_test)

    # Evaluate
    print("AdaBoost Performance:")
    print(f"Accuracy: {accuracy_score(y_test_boost, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_boost, y_pred))
    
    return ada, X_train, X_test, y_train_boost, y_test_boost


def visualize_training_progress(ada, X_train, y_train_boost):
    """Visualize AdaBoost training progress"""
    print("=== Training Progress Visualization ===\n")
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Error rates of weak learners
    plt.subplot(1, 3, 1)
    plt.plot(ada.estimator_errors, 'b-', label='Weak Learner Error')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing')
    plt.xlabel('Iteration')
    plt.ylabel('Error Rate')
    plt.title('Weak Learner Error Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Estimator weights
    plt.subplot(1, 3, 2)
    plt.plot(ada.estimator_weights, 'g-', label='Estimator Weight')
    plt.xlabel('Iteration')
    plt.ylabel('Weight (α)')
    plt.title('Estimator Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Cumulative training accuracy
    plt.subplot(1, 3, 3)
    train_accuracies = []
    for pred in ada.staged_predict(X_train):
        train_accuracies.append(accuracy_score(y_train_boost, pred))

    plt.plot(train_accuracies, 'r-', label='Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return train_accuracies


def plot_decision_boundary(X, y, model, title):
    """Plot decision boundary for 2D data"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar()


def demonstrate_decision_boundaries(X_train, X_test, y_train_boost, y_test_boost):
    """Demonstrate decision boundaries for different models"""
    print("=== Decision Boundary Comparison ===\n")
    
    plt.figure(figsize=(12, 4))

    # Single decision tree
    single_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    single_tree.fit(X_train, y_train_boost)

    plt.subplot(1, 3, 1)
    plot_decision_boundary(X_test, y_test_boost, single_tree, 'Single Decision Tree')

    # AdaBoost with few iterations
    ada_few = AdaBoost(n_estimators=5, max_depth=1)
    ada_few.fit(X_train, y_train_boost)

    plt.subplot(1, 3, 2)
    plot_decision_boundary(X_test, y_test_boost, ada_few, 'AdaBoost (5 iterations)')

    # AdaBoost with many iterations
    ada_many = AdaBoost(n_estimators=50, max_depth=1)
    ada_many.fit(X_train, y_train_boost)
    
    plt.subplot(1, 3, 3)
    plot_decision_boundary(X_test, y_test_boost, ada_many, 'AdaBoost (50 iterations)')

    plt.tight_layout()
    plt.show()


def demonstrate_text_classification():
    """Demonstrate AdaBoost for text classification"""
    print("=== Text Classification with AdaBoost ===\n")
    
    try:
        # Load text data
        categories = ['alt.atheism', 'soc.religion.christian']
        newsgroups = fetch_20newsgroups(subset='train', categories=categories)

        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(newsgroups.data)
        y = 2 * (newsgroups.target == 1) - 1  # Convert to {-1, 1}

        # Train AdaBoost
        ada_text = AdaBoost(n_estimators=100, max_depth=1)
        ada_text.fit(X, y)

        # Feature importance
        feature_importance = ada_text.get_feature_importance(X)
        top_features = np.argsort(feature_importance)[-10:]

        print("Top 10 most important features:")
        for i, idx in enumerate(reversed(top_features)):
            feature_name = vectorizer.get_feature_names_out()[idx]
            importance = feature_importance[idx]
            print(f"{i+1}. {feature_name}: {importance:.4f}")
            
        return ada_text, vectorizer, feature_importance
        
    except Exception as e:
        print(f"Text classification demonstration failed: {e}")
        print("This might be due to network issues or missing data.")
        return None, None, None


def demonstrate_medical_diagnosis():
    """Demonstrate AdaBoost for medical diagnosis"""
    print("=== Medical Diagnosis with AdaBoost ===\n")
    
    # Load data
    cancer = load_breast_cancer()
    X = cancer.data
    y = 2 * cancer.target - 1  # Convert to {-1, 1}

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train AdaBoost
    ada_medical = AdaBoost(n_estimators=50, max_depth=1)
    ada_medical.fit(X_train, y_train)

    # Evaluate
    y_pred = ada_medical.predict(X_test)
    print("Medical Diagnosis Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Sensitivity: {accuracy_score(y_test[y_test == 1], y_pred[y_test == 1]):.4f}")
    print(f"Specificity: {accuracy_score(y_test[y_test == -1], y_pred[y_test == -1]):.4f}")

    # Feature importance for medical interpretation
    feature_importance = ada_medical.get_feature_importance(X_train)
    top_medical_features = np.argsort(feature_importance)[-5:]

    print("\nTop 5 most important medical features:")
    for i, idx in enumerate(reversed(top_medical_features)):
        feature_name = cancer.feature_names[idx]
        importance = feature_importance[idx]
        print(f"{i+1}. {feature_name}: {importance:.4f}")
        
    return ada_medical, cancer, feature_importance


def analyze_theoretical_properties():
    """Analyze theoretical properties of AdaBoost"""
    print("=== Theoretical Properties Analysis ===\n")
    
    # Generate data for analysis
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42, class_sep=1.0)
    
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Train AdaBoost with different numbers of iterations
    iterations = [1, 5, 10, 20, 50, 100]
    training_errors = []
    test_errors = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for n_iter in iterations:
        ada = AdaBoost(n_estimators=n_iter, max_depth=1)
        ada.fit(X_train, y_train)
        
        # Calculate training error
        train_pred = ada.predict(X_train)
        train_error = 1 - accuracy_score(y_train, train_pred)
        training_errors.append(train_error)
        
        # Calculate test error
        test_pred = ada.predict(X_test)
        test_error = 1 - accuracy_score(y_test, test_pred)
        test_errors.append(test_error)
    
    # Plot error analysis
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, training_errors, 'b-o', label='Training Error')
    plt.plot(iterations, test_errors, 'r-o', label='Test Error')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Number of Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Analyze Z_t values
    ada_full = AdaBoost(n_estimators=50, max_depth=1)
    ada_full.fit(X_train, y_train)
    
    Z_t_values = []
    for error in ada_full.estimator_errors:
        Z_t = 2 * np.sqrt(error * (1 - error))
        Z_t_values.append(Z_t)
    
    plt.subplot(1, 2, 2)
    plt.plot(Z_t_values, 'g-o', label='Z_t values')
    plt.axhline(y=1, color='r', linestyle='--', label='Z_t = 1 (random)')
    plt.xlabel('Iteration')
    plt.ylabel('Z_t')
    plt.title('Normalization Factor Z_t')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print theoretical analysis
    print("Theoretical Analysis:")
    print(f"Final training error: {training_errors[-1]:.4f}")
    print(f"Product of Z_t values: {np.prod(Z_t_values):.4f}")
    print(f"Error bound satisfied: {training_errors[-1] <= np.prod(Z_t_values)}")
    
    return training_errors, test_errors, Z_t_values


def demonstrate_practical_considerations():
    """Demonstrate practical considerations for AdaBoost"""
    print("=== Practical Considerations ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=10, n_redundant=5, 
                             n_informative=5, n_clusters_per_class=1, 
                             random_state=42, class_sep=1.0)
    
    y = 2 * y - 1  # Convert to {-1, 1}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different weak learner depths
    depths = [1, 2, 3, 5]
    depth_results = []
    
    for depth in depths:
        ada = AdaBoost(n_estimators=50, max_depth=depth)
        ada.fit(X_train, y_train)
        
        train_pred = ada.predict(X_train)
        test_pred = ada.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        depth_results.append({
            'depth': depth,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting': train_acc - test_acc
        })
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    depths = [r['depth'] for r in depth_results]
    train_accs = [r['train_acc'] for r in depth_results]
    test_accs = [r['test_acc'] for r in depth_results]
    
    plt.plot(depths, train_accs, 'b-o', label='Training Accuracy')
    plt.plot(depths, test_accs, 'r-o', label='Test Accuracy')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Weak Learner Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    overfitting = [r['overfitting'] for r in depth_results]
    plt.plot(depths, overfitting, 'g-o', label='Overfitting Gap')
    plt.xlabel('Tree Depth')
    plt.ylabel('Overfitting Gap')
    plt.title('Overfitting vs Weak Learner Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print("Practical Recommendations:")
    print("1. Use shallow trees (depth=1) for weak learners")
    print("2. Monitor overfitting with deeper trees")
    print("3. Use cross-validation to find optimal number of iterations")
    print("4. Consider early stopping for large datasets")
    
    return depth_results


def main():
    """Main demonstration of AdaBoost"""
    print("AdaBoost: Implementation and Analysis")
    print("=" * 60)
    
    # 1. Basic AdaBoost demonstration
    print("\n1. Basic AdaBoost Demonstration:")
    ada, X_train, X_test, y_train_boost, y_test_boost = demonstrate_basic_adaboost()
    
    # 2. Training progress visualization
    print("\n2. Training Progress Visualization:")
    train_accuracies = visualize_training_progress(ada, X_train, y_train_boost)
    
    # 3. Decision boundary comparison
    print("\n3. Decision Boundary Comparison:")
    demonstrate_decision_boundaries(X_train, X_test, y_train_boost, y_test_boost)
    
    # 4. Text classification
    print("\n4. Text Classification Application:")
    ada_text, vectorizer, text_importance = demonstrate_text_classification()
    
    # 5. Medical diagnosis
    print("\n5. Medical Diagnosis Application:")
    ada_medical, cancer, medical_importance = demonstrate_medical_diagnosis()
    
    # 6. Theoretical analysis
    print("\n6. Theoretical Properties Analysis:")
    training_errors, test_errors, Z_t_values = analyze_theoretical_properties()
    
    # 7. Practical considerations
    print("\n7. Practical Considerations:")
    depth_results = demonstrate_practical_considerations()
    
    print("\n=== Key Insights ===")
    print("1. AdaBoost sequentially combines weak learners")
    print("2. Weight updates focus on difficult examples")
    print("3. Exponential loss provides natural combination")
    print("4. Theoretical bounds guarantee improvement")
    print("5. Shallow trees work best as weak learners")
    print("6. Monitor overfitting with validation data")
    print("7. Feature importance available through weighted average")
    print("8. Effective for both binary and multi-class problems")
    
    return {
        'ada': ada,
        'train_accuracies': train_accuracies,
        'ada_text': ada_text,
        'ada_medical': ada_medical,
        'training_errors': training_errors,
        'test_errors': test_errors,
        'Z_t_values': Z_t_values,
        'depth_results': depth_results
    }


if __name__ == "__main__":
    main()
