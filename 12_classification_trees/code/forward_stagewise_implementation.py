import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_regression, make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import clone
import seaborn as sns
import pandas as pd


class ForwardStagewiseAdditiveModel:
    def __init__(self, base_learner, loss_function, n_estimators=100, learning_rate=1.0):
        """
        Forward Stagewise Additive Model
        
        Parameters:
        -----------
        base_learner : estimator
            Base learner (e.g., DecisionTreeRegressor)
        loss_function : str
            Loss function ('squared_error', 'exponential', 'logistic')
        n_estimators : int
            Number of base learners
        learning_rate : float
            Learning rate (shrinkage)
        """
        self.base_learner = base_learner
        self.loss_function = loss_function
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []
        self.training_losses = []
        
    def _compute_residuals(self, y, predictions):
        """Compute residuals based on loss function"""
        if self.loss_function == 'squared_error':
            return y - predictions
        elif self.loss_function == 'exponential':
            # For exponential loss, residuals are weighted
            return -y * np.exp(-y * predictions)
        elif self.loss_function == 'logistic':
            # For logistic loss
            prob = 1 / (1 + np.exp(-predictions))
            return y - prob
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    def _find_optimal_weight(self, y, current_predictions, base_predictions):
        """Find optimal weight for the current base learner"""
        if self.loss_function == 'squared_error':
            # Closed form solution for squared error
            numerator = np.sum(base_predictions * (y - current_predictions))
            denominator = np.sum(base_predictions ** 2)
            return numerator / denominator if denominator > 0 else 0
        else:
            # Line search for other loss functions
            best_alpha = 0
            best_loss = float('inf')
            
            for alpha in np.linspace(-2, 2, 100):
                new_predictions = current_predictions + alpha * base_predictions
                if self.loss_function == 'exponential':
                    loss = np.mean(np.exp(-y * new_predictions))
                elif self.loss_function == 'logistic':
                    loss = np.mean(np.log(1 + np.exp(-y * new_predictions)))
                
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
            
            return best_alpha
    
    def fit(self, X, y):
        """Fit the forward stagewise additive model"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for t in range(self.n_estimators):
            # Compute residuals
            residuals = self._compute_residuals(y, predictions)
            
            # Fit base learner to residuals
            estimator = clone(self.base_learner)
            estimator.fit(X, residuals)
            base_predictions = estimator.predict(X)
            
            # Find optimal weight
            alpha = self._find_optimal_weight(y, predictions, base_predictions)
            alpha *= self.learning_rate  # Apply shrinkage
            
            # Update predictions
            predictions += alpha * base_predictions
            
            # Store results
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
            
            # Compute training loss
            if self.loss_function == 'squared_error':
                loss = mean_squared_error(y, predictions)
            elif self.loss_function == 'exponential':
                loss = np.mean(np.exp(-y * predictions))
            elif self.loss_function == 'logistic':
                loss = np.mean(np.log(1 + np.exp(-y * predictions)))
            
            self.training_losses.append(loss)
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            
        return predictions
    
    def staged_predict(self, X):
        """Return staged predictions"""
        predictions = np.zeros(X.shape[0])
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            predictions += alpha * estimator.predict(X)
            yield predictions.copy()
    
    def get_feature_importance(self, X):
        """Get feature importance based on weighted average of base learners"""
        importance = np.zeros(X.shape[1])
        total_weight = sum(abs(w) for w in self.estimator_weights)
        
        for alpha, estimator in zip(self.estimator_weights, self.estimators):
            if hasattr(estimator, 'feature_importances_'):
                importance += (abs(alpha) / total_weight) * estimator.feature_importances_
                
        return importance


def demonstrate_basic_forward_stagewise():
    """Demonstrate basic Forward Stagewise Additive Modeling"""
    print("=== Forward Stagewise Additive Modeling Demonstration ===\n")
    
    # Example 1: Regression with Squared Error Loss
    print("1. Regression with Squared Error Loss:")
    
    # Generate regression data
    X_reg, y_reg = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    # Train forward stagewise model
    base_learner_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
    fsam_reg = ForwardStagewiseAdditiveModel(
        base_learner=base_learner_reg,
        loss_function='squared_error',
        n_estimators=50,
        learning_rate=0.1
    )

    fsam_reg.fit(X_train_reg, y_train_reg)

    # Evaluate
    y_pred_reg = fsam_reg.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    print(f"   Test MSE: {mse:.4f}")
    
    # Example 2: Classification with Exponential Loss (AdaBoost-like)
    print("\n2. Classification with Exponential Loss:")
    
    # Generate classification data
    X_clf, y_clf = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                                     n_informative=2, n_clusters_per_class=1,
                                     random_state=42, class_sep=1.5)

    # Convert to {-1, 1}
    y_clf = 2 * y_clf - 1

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )

    # Train forward stagewise model
    base_learner_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    fsam_clf = ForwardStagewiseAdditiveModel(
        base_learner=base_learner_clf,
        loss_function='exponential',
        n_estimators=50,
        learning_rate=1.0
    )

    fsam_clf.fit(X_train_clf, y_train_clf)

    # Evaluate
    y_pred_clf = np.sign(fsam_clf.predict(X_test_clf))
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    return fsam_reg, fsam_clf, (X_train_reg, X_test_reg, y_train_reg, y_test_reg), (X_train_clf, X_test_clf, y_train_clf, y_test_clf)


def visualize_training_progress(fsam_reg, fsam_clf):
    """Visualize training progress for Forward Stagewise models"""
    print("=== Training Progress Visualization ===\n")
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Training loss progression
    plt.subplot(1, 3, 1)
    plt.plot(fsam_reg.training_losses, 'b-', label='Regression (Squared Error)', linewidth=2)
    plt.plot(fsam_clf.training_losses, 'r-', label='Classification (Exponential)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Estimator weights
    plt.subplot(1, 3, 2)
    plt.plot(fsam_reg.estimator_weights, 'b-', label='Regression Weights', linewidth=2)
    plt.plot(fsam_clf.estimator_weights, 'r-', label='Classification Weights', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Weight (α)')
    plt.title('Estimator Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Cumulative predictions
    plt.subplot(1, 3, 3)
    reg_predictions = list(fsam_reg.staged_predict(X_test_reg))
    clf_predictions = list(fsam_clf.staged_predict(X_test_clf))

    plt.plot([mean_squared_error(y_test_reg, pred) for pred in reg_predictions], 
             'b-', label='Regression MSE', linewidth=2)
    plt.plot([accuracy_score(y_test_clf, np.sign(pred)) for pred in clf_predictions], 
             'r-', label='Classification Accuracy', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('Performance vs Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_loss_functions():
    """Demonstrate different loss functions in Forward Stagewise"""
    print("=== Loss Functions Comparison ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42, class_sep=1.0)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models with different loss functions
    base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    loss_functions = ['exponential', 'logistic']
    models = {}
    
    for loss_func in loss_functions:
        model = ForwardStagewiseAdditiveModel(
            base_learner=base_learner,
            loss_function=loss_func,
            n_estimators=50,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        models[loss_func] = model
    
    # Compare training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for loss_func, model in models.items():
        plt.plot(model.training_losses, label=f'{loss_func.capitalize()} Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare test performance
    plt.subplot(1, 2, 2)
    for loss_func, model in models.items():
        staged_preds = list(model.staged_predict(X_test))
        accuracies = [accuracy_score(y_test, np.sign(pred)) for pred in staged_preds]
        plt.plot(accuracies, label=f'{loss_func.capitalize()} Loss', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Test Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Loss Function Analysis:")
    print("1. Exponential Loss: Heavily penalizes misclassifications, can overfit")
    print("2. Logistic Loss: More robust, better theoretical properties")
    print("3. Both show similar convergence patterns but different generalization")


def demonstrate_learning_rate_effects():
    """Demonstrate the effects of learning rate"""
    print("=== Learning Rate Effects ===\n")
    
    # Generate data
    X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different learning rates
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    models = {}
    
    for lr in learning_rates:
        base_learner = DecisionTreeRegressor(max_depth=3, random_state=42)
        model = ForwardStagewiseAdditiveModel(
            base_learner=base_learner,
            loss_function='squared_error',
            n_estimators=50,
            learning_rate=lr
        )
        model.fit(X_train, y_train)
        models[lr] = model
    
    # Visualize effects
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for lr, model in models.items():
        plt.plot(model.training_losses, label=f'LR = {lr}', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for lr, model in models.items():
        staged_preds = list(model.staged_predict(X_test))
        mses = [mean_squared_error(y_test, pred) for pred in staged_preds]
        plt.plot(mses, label=f'LR = {lr}', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Test MSE')
    plt.title('Test Performance vs Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Learning Rate Analysis:")
    print("1. Smaller learning rates: Slower convergence, better generalization")
    print("2. Larger learning rates: Faster convergence, risk of overfitting")
    print("3. Optimal learning rate balances convergence speed and generalization")


def demonstrate_financial_risk_modeling():
    """Demonstrate Forward Stagewise for financial risk modeling"""
    print("=== Financial Risk Modeling ===\n")
    
    # Simulate financial data
    np.random.seed(42)
    n_samples = 10000

    # Features: income, age, credit_score, debt_ratio, payment_history
    X_fin = pd.DataFrame({
        'income': np.random.lognormal(10, 0.5, n_samples),
        'age': np.random.normal(45, 15, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        'payment_history': np.random.poisson(2, n_samples)
    })

    # Target: default (1) or not (0)
    y_fin = (X_fin['debt_ratio'] > 0.4) | (X_fin['credit_score'] < 600)
    y_fin = 2 * y_fin.astype(int) - 1  # Convert to {-1, 1}

    # Train forward stagewise model
    base_learner_fin = DecisionTreeClassifier(max_depth=4, random_state=42)
    fsam_fin = ForwardStagewiseAdditiveModel(
        base_learner=base_learner_fin,
        loss_function='exponential',
        n_estimators=100,
        learning_rate=0.1
    )

    fsam_fin.fit(X_fin, y_fin)

    # Feature importance analysis
    feature_importance = fsam_fin.get_feature_importance(X_fin)

    # Display feature importance
    feature_names = X_fin.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("Feature Importance for Credit Risk:")
    print(importance_df)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importance_df['importance'])
    plt.yticks(range(len(feature_names)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Credit Risk Modeling')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fsam_fin, importance_df


def demonstrate_medical_diagnosis():
    """Demonstrate Forward Stagewise for medical diagnosis"""
    print("=== Medical Diagnosis ===\n")
    
    # Load medical data
    cancer = load_breast_cancer()
    X_med = cancer.data
    y_med = 2 * cancer.target - 1  # Convert to {-1, 1}

    # Train forward stagewise model
    base_learner_med = DecisionTreeClassifier(max_depth=3, random_state=42)
    fsam_med = ForwardStagewiseAdditiveModel(
        base_learner=base_learner_med,
        loss_function='logistic',  # More robust than exponential
        n_estimators=50,
        learning_rate=0.1
    )

    fsam_med.fit(X_med, y_med)

    # Model evaluation
    y_pred_med = np.sign(fsam_med.predict(X_med))
    accuracy = accuracy_score(y_med, y_pred_med)

    print(f"Medical Diagnosis Accuracy: {accuracy:.4f}")

    # Analyze model stability
    staged_predictions = list(fsam_med.staged_predict(X_med))
    staged_accuracies = [accuracy_score(y_med, np.sign(pred)) for pred in staged_predictions]

    plt.figure(figsize=(10, 6))
    plt.plot(staged_accuracies, 'b-', linewidth=2)
    plt.axhline(y=accuracy, color='r', linestyle='--', label='Final Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Convergence in Medical Diagnosis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return fsam_med, accuracy, staged_accuracies


def analyze_theoretical_properties():
    """Analyze theoretical properties of Forward Stagewise"""
    print("=== Theoretical Properties Analysis ===\n")
    
    # Generate data for analysis
    X, y = make_regression(n_samples=500, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test different numbers of iterations
    iterations = [1, 5, 10, 20, 50, 100]
    training_losses = []
    test_losses = []
    
    for n_iter in iterations:
        base_learner = DecisionTreeRegressor(max_depth=3, random_state=42)
        model = ForwardStagewiseAdditiveModel(
            base_learner=base_learner,
            loss_function='squared_error',
            n_estimators=n_iter,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
        
        # Training loss
        training_losses.append(model.training_losses[-1])
        
        # Test loss
        y_pred = model.predict(X_test)
        test_losses.append(mean_squared_error(y_test, y_pred))
    
    # Plot convergence analysis
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, training_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(iterations, test_losses, 'r-o', label='Test Loss', linewidth=2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Number of Iterations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Analyze convergence rate
    plt.subplot(1, 2, 2)
    plt.plot(iterations, np.array(training_losses) - min(training_losses), 'b-o', 
             label='Training Loss Gap', linewidth=2)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss Gap')
    plt.title('Convergence Rate Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("Theoretical Analysis:")
    print(f"1. Final training loss: {training_losses[-1]:.6f}")
    print(f"2. Final test loss: {test_losses[-1]:.6f}")
    print(f"3. Overfitting gap: {training_losses[-1] - test_losses[-1]:.6f}")
    print("4. Model shows good convergence properties")
    
    return training_losses, test_losses


def main():
    """Main demonstration of Forward Stagewise Additive Modeling"""
    print("Forward Stagewise Additive Modeling: Implementation and Analysis")
    print("=" * 70)
    
    # 1. Basic demonstration
    print("\n1. Basic Forward Stagewise Demonstration:")
    fsam_reg, fsam_clf, reg_data, clf_data = demonstrate_basic_forward_stagewise()
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = reg_data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = clf_data
    
    # 2. Training progress visualization
    print("\n2. Training Progress Visualization:")
    visualize_training_progress(fsam_reg, fsam_clf)
    
    # 3. Loss functions comparison
    print("\n3. Loss Functions Comparison:")
    demonstrate_loss_functions()
    
    # 4. Learning rate effects
    print("\n4. Learning Rate Effects:")
    demonstrate_learning_rate_effects()
    
    # 5. Financial risk modeling
    print("\n5. Financial Risk Modeling Application:")
    fsam_fin, fin_importance = demonstrate_financial_risk_modeling()
    
    # 6. Medical diagnosis
    print("\n6. Medical Diagnosis Application:")
    fsam_med, med_accuracy, med_staged = demonstrate_medical_diagnosis()
    
    # 7. Theoretical analysis
    print("\n7. Theoretical Properties Analysis:")
    training_losses, test_losses = analyze_theoretical_properties()
    
    print("\n=== Key Insights ===")
    print("1. Forward Stagewise provides a unified framework for boosting")
    print("2. Different loss functions have different convergence properties")
    print("3. Learning rate controls the trade-off between speed and generalization")
    print("4. Sequential optimization makes complex problems tractable")
    print("5. Residual fitting focuses each base learner on current errors")
    print("6. Weight optimization ensures optimal contribution of each base learner")
    print("7. Regularization improves generalization performance")
    print("8. Feature importance provides interpretability")
    
    return {
        'fsam_reg': fsam_reg,
        'fsam_clf': fsam_clf,
        'fsam_fin': fsam_fin,
        'fsam_med': fsam_med,
        'fin_importance': fin_importance,
        'med_accuracy': med_accuracy,
        'training_losses': training_losses,
        'test_losses': test_losses
    }


if __name__ == "__main__":
    main()
