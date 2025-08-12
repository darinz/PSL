import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implementation from scratch
    """
    
    def __init__(self, feature_type='gaussian'):
        self.feature_type = feature_type
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.variances_ = None
        
    def fit(self, X, y):
        """
        Fit Naive Bayes classifier
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        # Estimate parameters for each class
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            n_class = np.sum(class_mask)
            
            # Prior probability
            self.priors_[i] = n_class / n_samples
            
            # Mean and variance for each feature
            self.means_[i] = np.mean(class_data, axis=0)
            self.variances_[i] = np.var(class_data, axis=0, ddof=1)
            
            # Add small constant to avoid zero variance
            self.variances_[i] = np.maximum(self.variances_[i], 1e-9)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        """
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        log_proba = self.predict_log_proba(X)
        # Convert log probabilities to probabilities
        proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
        return proba / np.sum(proba, axis=1, keepdims=True)
    
    def predict_log_proba(self, X):
        """
        Predict log class probabilities (numerically stable)
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # Log prior
            log_proba[:, i] = np.log(self.priors_[i])
            
            # Log likelihood for each feature
            for j in range(n_features):
                mu = self.means_[i, j]
                sigma2 = self.variances_[i, j]
                
                # Gaussian log-likelihood
                log_likelihood = -0.5 * np.log(2 * np.pi * sigma2) - \
                                0.5 * (X[:, j] - mu)**2 / sigma2
                
                log_proba[:, i] += log_likelihood
        
        return log_proba
    
    def score(self, X, y):
        """
        Return accuracy score
        """
        return accuracy_score(y, self.predict(X))


def demonstrate_naive_bayes():
    """
    Demonstrate Naive Bayes with synthetic data
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0: centered at (0, 0, 0, 0)
    X0 = np.random.multivariate_normal([0, 0, 0, 0], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    # Class 1: centered at (2, 2, 0, 0)
    X1 = np.random.multivariate_normal([2, 2, 0, 0], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    # Class 2: centered at (0, 0, 2, 2)
    X2 = np.random.multivariate_normal([0, 0, 2, 2], 
                                     [[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit our implementation
    nb_scratch = NaiveBayesClassifier()
    nb_scratch.fit(X_train, y_train)
    
    # Fit sklearn implementation
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    
    # Compare predictions
    y_pred_scratch = nb_scratch.predict(X_test)
    y_pred_sklearn = nb_sklearn.predict(X_test)
    
    print("Naive Bayes Results:")
    print("-" * 40)
    print(f"Our Implementation Accuracy: {nb_scratch.score(X_test, y_test):.4f}")
    print(f"Sklearn Implementation Accuracy: {nb_sklearn.score(X_test, y_test):.4f}")
    
    # Compare parameters
    print(f"\nParameter Comparison:")
    print(f"Our means shape: {nb_scratch.means_.shape}")
    print(f"Sklearn means shape: {nb_sklearn.theta_.shape}")
    print(f"Our variances shape: {nb_scratch.variances_.shape}")
    print(f"Sklearn variances shape: {nb_sklearn.var_.shape}")
    
    # Visualize decision boundaries (first 2 features)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[0].set_title('Original Data (Features 0 & 1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Create test points (use mean values for other features)
    test_points = np.zeros((10000, 4))
    test_points[:, 0] = xx.ravel()
    test_points[:, 1] = yy.ravel()
    test_points[:, 2] = np.mean(X[:, 2])  # Use mean of feature 2
    test_points[:, 3] = np.mean(X[:, 3])  # Use mean of feature 3
    
    Z = nb_scratch.predict(test_points)
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    for i in range(3):
        mask = y == i
        axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[1].set_title('Decision Boundaries (Our Implementation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Feature importance (based on variance ratios)
    feature_importance = np.zeros(n_features)
    for j in range(n_features):
        # Calculate ratio of between-class to within-class variance
        overall_mean = np.mean(X[:, j])
        between_var = np.sum([np.sum(y == c) * (np.mean(X[y == c, j]) - overall_mean)**2 
                             for c in np.unique(y)])
        within_var = np.sum([np.sum((X[y == c, j] - np.mean(X[y == c, j]))**2) 
                            for c in np.unique(y)])
        feature_importance[j] = between_var / within_var if within_var > 0 else 0
    
    axes[2].bar(range(n_features), feature_importance)
    axes[2].set_title('Feature Importance (Variance Ratio)')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Between/Within Variance Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return nb_scratch, nb_sklearn


def demonstrate_numerical_issues():
    """
    Demonstrate numerical stability issues in Naive Bayes
    """
    # Generate data with extreme values
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.normal(0, 1, 100)
    
    # Extreme data
    X_extreme = np.random.normal(10, 1, 100)
    
    # Compute Gaussian PDF
    def gaussian_pdf(x, mu, sigma):
        return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    def gaussian_log_pdf(x, mu, sigma):
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2
    
    # Test points
    test_points = np.linspace(-5, 15, 1000)
    
    # Compute probabilities
    pdf_normal = gaussian_pdf(test_points, 0, 1)
    pdf_extreme = gaussian_pdf(test_points, 10, 1)
    
    log_pdf_normal = gaussian_log_pdf(test_points, 0, 1)
    log_pdf_extreme = gaussian_log_pdf(test_points, 10, 1)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # PDF for normal data
    axes[0, 0].plot(test_points, pdf_normal)
    axes[0, 0].set_title('Gaussian PDF (μ=0, σ=1)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # PDF for extreme data
    axes[0, 1].plot(test_points, pdf_extreme)
    axes[0, 1].set_title('Gaussian PDF (μ=10, σ=1)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log PDF for normal data
    axes[1, 0].plot(test_points, log_pdf_normal)
    axes[1, 0].set_title('Gaussian Log-PDF (μ=0, σ=1)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('log f(x)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log PDF for extreme data
    axes[1, 1].plot(test_points, log_pdf_extreme)
    axes[1, 1].set_title('Gaussian Log-PDF (μ=10, σ=1)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('log f(x)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate numerical issues
    print("Numerical Stability Analysis:")
    print("-" * 40)
    
    # Test with extreme point
    extreme_point = 20
    mu = 0
    sigma = 1
    
    pdf_value = gaussian_pdf(extreme_point, mu, sigma)
    log_pdf_value = gaussian_log_pdf(extreme_point, mu, sigma)
    
    print(f"Point: {extreme_point}")
    print(f"Mean: {mu}, Std: {sigma}")
    print(f"PDF value: {pdf_value:.2e}")
    print(f"Log-PDF value: {log_pdf_value:.4f}")
    print(f"Recovered PDF: {np.exp(log_pdf_value):.2e}")
    
    return pdf_value, log_pdf_value


def safe_naive_bayes_predict(X, model):
    """
    Safe Naive Bayes prediction using log-probabilities
    """
    log_proba = model.predict_log_proba(X)
    return model.classes_[np.argmax(log_proba, axis=1)]


def regularized_naive_bayes(X, y, epsilon=1e-9):
    """
    Naive Bayes with regularization
    """
    nb = NaiveBayesClassifier()
    nb.fit(X, y)
    
    # Regularize variances
    nb.variances_ = np.maximum(nb.variances_, epsilon)
    
    return nb


def truncated_naive_bayes(X, model, threshold=1e-10):
    """
    Naive Bayes with truncation (not recommended)
    """
    proba = model.predict_proba(X)
    proba = np.maximum(proba, threshold)  # Truncate small values
    return model.classes_[np.argmax(proba, axis=1)]


class GaussianNaiveBayes(NaiveBayesClassifier):
    def __init__(self):
        super().__init__(feature_type='gaussian')


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        
    def fit(self, X, y):
        # Count features for each class
        # Apply Laplace smoothing
        # Estimate class-conditional probabilities
        pass


class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of feature being 1 for each class
        # Apply Laplace smoothing
        pass


class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        # Estimate probability of each category for each class
        # Apply Laplace smoothing
        pass


def text_classification_example():
    """
    Naive Bayes for text classification
    """
    # Sample text data
    texts = [
        "great movie amazing acting",
        "terrible film waste of time", 
        "excellent performance brilliant",
        "boring plot disappointing",
        "fantastic story wonderful",
        "awful acting bad script",
        "outstanding film superb",
        "poor quality terrible",
        "incredible movie perfect",
        "horrible waste bad"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fit Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    
    print("Text Classification Results:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, 
                               target_names=['Negative', 'Positive']))
    
    # Feature importance
    feature_names = vectorizer.get_feature_names_out()
    log_probs = nb.feature_log_prob_
    
    # Show most discriminative words
    positive_words = log_probs[1] - log_probs[0]
    negative_words = log_probs[0] - log_probs[1]
    
    print("\nMost Positive Words:")
    pos_indices = np.argsort(positive_words)[-5:]
    for idx in pos_indices:
        print(f"  {feature_names[idx]}: {positive_words[idx]:.3f}")
    
    print("\nMost Negative Words:")
    neg_indices = np.argsort(negative_words)[-5:]
    for idx in neg_indices:
        print(f"  {feature_names[idx]}: {negative_words[idx]:.3f}")
    
    return nb, vectorizer


def medical_diagnosis_example():
    """
    Naive Bayes for medical diagnosis
    """
    # Simulate medical data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, blood_pressure, cholesterol, glucose
    age = np.random.normal(50, 15, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    glucose = np.random.normal(100, 20, n_samples)
    
    X = np.column_stack([age, blood_pressure, cholesterol, glucose])
    
    # Disease risk based on features
    risk_score = (age * 0.1 + (blood_pressure - 120) * 0.05 + 
                  (cholesterol - 200) * 0.02 + (glucose - 100) * 0.03 +
                  np.random.normal(0, 0.1, n_samples))
    
    y = (risk_score > np.median(risk_score)).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit Naive Bayes
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    
    # Predictions
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Medical Diagnosis Results:")
    print("-" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 'Glucose']
    feature_importance = np.zeros(4)
    
    for j in range(4):
        overall_mean = np.mean(X[:, j])
        between_var = np.sum([np.sum(y == c) * (np.mean(X[y == c, j]) - overall_mean)**2 
                             for c in np.unique(y)])
        within_var = np.sum([np.sum((X[y == c, j] - np.mean(X[y == c, j]))**2) 
                            for c in np.unique(y)])
        feature_importance[j] = between_var / within_var if within_var > 0 else 0
    
    # Plot feature importance
    plt.figure(figsize=(10, 4))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance in Medical Diagnosis')
    plt.ylabel('Between/Within Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return nb, feature_importance


def plot_naive_bayes_decision_boundaries(X, y, model, title="Naive Bayes Decision Boundaries"):
    """
    Plot Naive Bayes decision boundaries
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Create test points (use mean values for other features)
    test_points = np.zeros((10000, X.shape[1]))
    test_points[:, 0] = xx.ravel()
    test_points[:, 1] = yy.ravel()
    
    # Use mean values for remaining features
    for j in range(2, X.shape[1]):
        test_points[:, j] = np.mean(X[:, j])
    
    # Predict
    Z = model.predict(test_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    for i in range(len(np.unique(y))):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_feature_independence(X, y):
    """
    Analyze feature independence assumption
    """
    n_features = X.shape[1]
    correlations = np.zeros((n_features, n_features))
    
    # Calculate correlations
    for i in range(n_features):
        for j in range(n_features):
            correlations[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='RdBu_r', center=0,
                xticklabels=[f'F{i}' for i in range(n_features)],
                yticklabels=[f'F{i}' for i in range(n_features)])
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # Calculate average absolute correlation (excluding diagonal)
    avg_corr = np.mean(np.abs(correlations[np.triu_indices(n_features, k=1)]))
    print(f"Average absolute correlation: {avg_corr:.4f}")
    
    return correlations, avg_corr


def main():
    """
    Main function to demonstrate Naive Bayes implementation
    """
    print("Naive Bayes Classifier Demonstration")
    print("=" * 50)
    
    # Basic demonstration
    print("\n1. Basic Naive Bayes Demonstration:")
    nb_scratch, nb_sklearn = demonstrate_naive_bayes()
    
    # Numerical stability demonstration
    print("\n2. Numerical Stability Issues:")
    pdf_val, log_pdf_val = demonstrate_numerical_issues()
    
    # Text classification example
    print("\n3. Text Classification Example:")
    nb_text, vectorizer = text_classification_example()
    
    # Medical diagnosis example
    print("\n4. Medical Diagnosis Example:")
    nb_medical, feature_importance = medical_diagnosis_example()
    
    # Generate data for additional analysis
    np.random.seed(42)
    X, y = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000), np.random.randint(0, 2, 1000)
    
    # Feature independence analysis
    print("\n5. Feature Independence Analysis:")
    correlations, avg_corr = analyze_feature_independence(X, y)
    
    # Decision boundaries
    print("\n6. Decision Boundaries:")
    nb_model = NaiveBayesClassifier()
    nb_model.fit(X, y)
    plot_naive_bayes_decision_boundaries(X, y, nb_model)
    
    return nb_scratch, nb_sklearn, nb_text, nb_medical


if __name__ == "__main__":
    main()
