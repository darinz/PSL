import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from scipy import stats


class LinearDiscriminantAnalysisFromScratch:
    """
    Linear Discriminant Analysis implementation from scratch
    """
    
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        """
        Fit LDA model
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get unique classes and their counts
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = np.sum(y == c) / n_samples
            
        # Calculate class means
        self.means_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            self.means_[i] = np.mean(X[y == c], axis=0)
            
        # Calculate pooled covariance matrix
        self.covariance_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_samples = X[y == c]
            class_mean = self.means_[i]
            diff = class_samples - class_mean
            self.covariance_ += diff.T @ diff
            
        self.covariance_ /= (n_samples - n_classes)
        
        # Add regularization for numerical stability
        self.covariance_ += self.regularization * np.eye(n_features)
        
        # Calculate coefficients and intercepts
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        
        cov_inv = np.linalg.inv(self.covariance_)
        for i in range(n_classes):
            self.coef_[i] = -2 * cov_inv @ self.means_[i]
            self.intercept_[i] = (self.means_[i] @ cov_inv @ self.means_[i] + 
                                 np.log(np.linalg.det(self.covariance_)) - 
                                 2 * np.log(self.priors_[i]))
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        discriminant_scores = self.decision_function(X)
        return self.classes_[np.argmax(discriminant_scores, axis=1)]
    
    def decision_function(self, X):
        """
        Compute discriminant scores
        """
        X = np.asarray(X)
        return X @ self.coef_.T + self.intercept_
    
    def transform(self, X, n_components=None):
        """
        Transform data using LDA projection
        """
        if n_components is None:
            n_components = len(self.classes_) - 1
            
        # Calculate between-class scatter matrix
        overall_mean = np.average(self.means_, weights=self.priors_, axis=0)
        between_scatter = np.zeros((X.shape[1], X.shape[1]))
        
        for i, c in enumerate(self.classes_):
            diff = self.means_[i] - overall_mean
            between_scatter += self.priors_[i] * np.outer(diff, diff)
            
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eigh(
            np.linalg.inv(self.covariance_) @ between_scatter
        )
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top components
        projection_matrix = eigenvecs[:, :n_components]
        
        return X @ projection_matrix
    
    def score(self, X, y):
        """
        Compute accuracy score
        """
        return accuracy_score(y, self.predict(X))


class RegularizedLDA:
    """
    Regularized LDA with shrinkage parameter
    """
    
    def __init__(self, alpha=0.1, regularization=1e-6):
        self.alpha = alpha
        self.regularization = regularization
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        """
        Fit regularized LDA model
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get unique classes and their counts
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = np.sum(y == c) / n_samples
            
        # Calculate class means
        self.means_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            self.means_[i] = np.mean(X[y == c], axis=0)
            
        # Calculate pooled covariance matrix
        self.covariance_ = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_samples = X[y == c]
            class_mean = self.means_[i]
            diff = class_samples - class_mean
            self.covariance_ += diff.T @ diff
            
        self.covariance_ /= (n_samples - n_classes)
        
        # Apply regularization: convex combination with identity matrix
        identity = np.eye(n_features)
        self.covariance_ = (1 - self.alpha) * self.covariance_ + self.alpha * identity
        
        # Add small regularization for numerical stability
        self.covariance_ += self.regularization * np.eye(n_features)
        
        # Calculate coefficients and intercepts
        self.coef_ = np.zeros((n_classes, n_features))
        self.intercept_ = np.zeros(n_classes)
        
        cov_inv = np.linalg.inv(self.covariance_)
        for i in range(n_classes):
            self.coef_[i] = -2 * cov_inv @ self.means_[i]
            self.intercept_[i] = (self.means_[i] @ cov_inv @ self.means_[i] + 
                                 np.log(np.linalg.det(self.covariance_)) - 
                                 2 * np.log(self.priors_[i]))
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        discriminant_scores = self.decision_function(X)
        return self.classes_[np.argmax(discriminant_scores, axis=1)]
    
    def decision_function(self, X):
        """
        Compute discriminant scores
        """
        X = np.asarray(X)
        return X @ self.coef_.T + self.intercept_
    
    def score(self, X, y):
        """
        Compute accuracy score
        """
        return accuracy_score(y, self.predict(X))


def generate_lda_data(n_samples=1000, n_features=2, n_classes=3, random_state=42):
    """
    Generate synthetic data suitable for LDA
    """
    np.random.seed(random_state)
    
    # Generate class means
    means = np.random.randn(n_classes, n_features) * 2
    
    # Generate shared covariance matrix
    A = np.random.randn(n_features, n_features)
    covariance = A @ A.T + np.eye(n_features)
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        class_samples = np.random.multivariate_normal(
            means[i], covariance, samples_per_class
        )
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    return np.vstack(X), np.array(y)


def demonstrate_lda():
    """
    Demonstrate LDA with synthetic data
    """
    # Generate data
    X, y = generate_lda_data(n_samples=900, n_features=2, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit our implementation
    lda_scratch = LinearDiscriminantAnalysisFromScratch()
    lda_scratch.fit(X_train, y_train)
    
    # Fit sklearn implementation
    lda_sklearn = LinearDiscriminantAnalysis()
    lda_sklearn.fit(X_train, y_train)
    
    # Compare predictions
    y_pred_scratch = lda_scratch.predict(X_test)
    y_pred_sklearn = lda_sklearn.predict(X_test)
    
    print("Accuracy Comparison:")
    print(f"Our Implementation: {accuracy_score(y_test, y_pred_scratch):.4f}")
    print(f"Sklearn Implementation: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[0].set_title('Original Data')
    axes[0].legend()
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = lda_scratch.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    for i in range(3):
        mask = y == i
        axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[1].set_title('Decision Boundaries')
    axes[1].legend()
    
    # LDA projection
    X_transformed = lda_scratch.transform(X)
    for i in range(3):
        mask = y == i
        axes[2].scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                       alpha=0.6, label=f'Class {i}')
    axes[2].set_title('LDA Projection (2D → 2D)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return lda_scratch, lda_sklearn


def regularized_lda(X, y, alpha=0.1):
    """
    Regularized LDA with shrinkage parameter alpha
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Calculate pooled covariance
    covariance = np.zeros((n_features, n_features))
    for i, c in enumerate(np.unique(y)):
        class_samples = X[y == c]
        class_mean = np.mean(class_samples, axis=0)
        diff = class_samples - class_mean
        covariance += diff.T @ diff
    
    covariance /= (n_samples - n_classes)
    
    # Regularization: convex combination with identity matrix
    identity = np.eye(n_features)
    regularized_cov = (1 - alpha) * covariance + alpha * identity
    
    return regularized_cov


def kernel_lda(X, y, kernel='rbf', gamma=1.0):
    """
    Kernel LDA implementation
    """
    # Compute kernel matrix
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    
    # Apply LDA in kernel space
    # (Implementation details omitted for brevity)
    pass


def multiclass_lda(X, y):
    """
    Multi-class LDA with dimensionality reduction
    """
    n_classes = len(np.unique(y))
    n_components = min(n_classes - 1, X.shape[1])
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_transformed = lda.fit_transform(X, y)
    
    return X_transformed, lda


def evaluate_lda_model(X_train, X_test, y_train, y_test):
    """
    Comprehensive LDA model evaluation
    """
    # Fit model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    
    # Predictions
    y_pred = lda.predict(X_test)
    y_pred_proba = lda.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC (for binary classification)
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def lda_diagnostics(X, y, lda_model):
    """
    Diagnostic plots for LDA
    """
    # 1. Check normality assumption
    residuals = []
    for i, class_label in enumerate(lda_model.classes_):
        class_mask = y == class_label
        class_residuals = X[class_mask] - lda_model.means_[i]
        residuals.extend(class_residuals.flatten())
    
    # Q-Q plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot for Normality Check")
    
    # 2. Check homoscedasticity
    plt.subplot(1, 3, 2)
    X_transformed = lda_model.transform(X)
    plt.scatter(X_transformed[:, 0], residuals[:len(X_transformed)], alpha=0.5)
    plt.xlabel("First LDA Component")
    plt.ylabel("Residuals")
    plt.title("Homoscedasticity Check")
    
    # 3. Feature importance
    plt.subplot(1, 3, 3)
    feature_importance = np.abs(lda_model.coef_[0])
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel("Feature Index")
    plt.ylabel("|Coefficient|")
    plt.title("Feature Importance")
    
    plt.tight_layout()
    plt.show()


def iris_lda_example():
    """
    LDA on the Iris dataset
    """
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # LDA with cross-validation
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, X, y, cv=5)
    
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Dimensionality reduction
    X_transformed = lda.fit_transform(X, y)
    print(f"Original dimensions: {X.shape[1]}")
    print(f"LDA dimensions: {X_transformed.shape[1]}")
    
    return lda, X_transformed, scores


def credit_risk_lda():
    """
    LDA for credit risk assessment
    """
    # Simulate credit data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: income, debt, credit_score, age
    income = np.random.lognormal(10, 0.5, n_samples)
    debt = np.random.lognormal(8, 0.3, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    age = np.random.normal(35, 10, n_samples)
    
    X = np.column_stack([income, debt, credit_score, age])
    
    # Risk classification (0: low risk, 1: high risk)
    risk_score = (income * 0.3 + debt * (-0.4) + credit_score * 0.2 + age * 0.1 + 
                  np.random.normal(0, 0.1, n_samples))
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Credit Risk Classification Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Income', 'Debt', 'Credit Score', 'Age']
    importance = np.abs(lda.coef_[0])
    
    plt.figure(figsize=(10, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importance in Credit Risk LDA")
    plt.ylabel("|Coefficient|")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return lda, accuracy


def regularized_lda_cv(X, y, alphas=np.logspace(-4, 1, 20)):
    """
    Cross-validated regularized LDA
    """
    # Create custom LDA with regularization
    class RegularizedLDA:
        def __init__(self, alpha=0.1):
            self.alpha = alpha
            
        def fit(self, X, y):
            # Implementation with regularization
            pass
    
    # Grid search
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(RegularizedLDA(), param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_


def lda_with_feature_selection(X, y, n_features=10):
    """
    LDA with feature selection
    """
    # Select top features
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_selected, y)
    
    return lda, selector


def robust_lda_evaluation(X, y, n_splits=5):
    """
    Robust LDA evaluation with cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        score = lda.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def plot_lda_decision_boundaries(X, y, lda_model, title="LDA Decision Boundaries"):
    """
    Plot LDA decision boundaries
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Predict on mesh grid
    Z = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.figure(figsize=(12, 5))
    
    # Decision boundaries
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    for i in range(len(np.unique(y))):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.8, edgecolors='k', label=f'Class {i}')
    plt.title(f'{title} - Decision Boundaries')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Posterior probabilities
    plt.subplot(1, 2, 2)
    Z_proba = lda_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)
    plt.contourf(xx, yy, Z_proba, alpha=0.4, cmap='RdBu_r')
    for i in range(len(np.unique(y))):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.8, edgecolors='k', label=f'Class {i}')
    plt.title(f'{title} - Posterior Probabilities')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate LDA implementation
    """
    print("Linear Discriminant Analysis Demonstration")
    print("=" * 50)
    
    # Demonstrate basic LDA
    print("\n1. Basic LDA Demonstration:")
    lda_scratch, lda_sklearn = demonstrate_lda()
    
    # Iris dataset example
    print("\n2. Iris Dataset Example:")
    iris_lda, iris_transformed, iris_scores = iris_lda_example()
    
    # Credit risk example
    print("\n3. Credit Risk Assessment Example:")
    credit_lda, credit_accuracy = credit_risk_lda()
    
    # Generate data for diagnostics
    X, y = generate_lda_data(n_samples=900, n_features=2, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Model evaluation
    print("\n4. Model Evaluation:")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    results = evaluate_lda_model(X_train, X_test, y_train, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    
    # Diagnostics
    print("\n5. Model Diagnostics:")
    lda_diagnostics(X_train, y_train, lda)
    
    # Robust evaluation
    print("\n6. Robust Evaluation:")
    mean_score, std_score = robust_lda_evaluation(X, y, n_splits=5)
    print(f"Cross-validation accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return lda_scratch, lda_sklearn, iris_lda, credit_lda


if __name__ == "__main__":
    main()
