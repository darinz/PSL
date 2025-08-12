import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import make_classification, load_iris
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier implementation from scratch
    """
    
    def __init__(self, variant='gaussian', alpha=1e-10):
        self.variant = variant
        self.alpha = alpha  # Regularization parameter
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.variances_ = None
        self.feature_probs_ = None
        
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
        
        # Calculate prior probabilities
        self.priors_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.priors_[i] = np.sum(y == c) / n_samples
        
        if self.variant == 'gaussian':
            self._fit_gaussian(X, y)
        elif self.variant == 'multinomial':
            self._fit_multinomial(X, y)
        elif self.variant == 'bernoulli':
            self._fit_bernoulli(X, y)
        elif self.variant == 'categorical':
            self._fit_categorical(X, y)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        
        return self
    
    def _fit_gaussian(self, X, y):
        """Fit Gaussian Naive Bayes"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            
            # Calculate means and variances
            self.means_[i] = np.mean(class_data, axis=0)
            self.variances_[i] = np.var(class_data, axis=0, ddof=1)
            
            # Add regularization to prevent zero variance
            self.variances_[i] = np.maximum(self.variances_[i], self.alpha)
    
    def _fit_multinomial(self, X, y):
        """Fit Multinomial Naive Bayes"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.feature_probs_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            
            # Calculate feature probabilities with Laplace smoothing
            feature_counts = np.sum(class_data, axis=0)
            total_count = np.sum(feature_counts)
            
            self.feature_probs_[i] = (feature_counts + self.alpha) / (total_count + n_features * self.alpha)
    
    def _fit_bernoulli(self, X, y):
        """Fit Bernoulli Naive Bayes"""
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.feature_probs_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            
            # Calculate probability of feature being present
            feature_present = np.sum(class_data > 0, axis=0)
            n_class_samples = np.sum(class_mask)
            
            self.feature_probs_[i] = (feature_present + self.alpha) / (n_class_samples + 2 * self.alpha)
    
    def _fit_categorical(self, X, y):
        """Fit Categorical Naive Bayes"""
        # This is a simplified version for demonstration
        # In practice, you'd need to handle different categorical encodings
        self._fit_gaussian(X, y)  # Use Gaussian as approximation
    
    def predict(self, X):
        """Predict class labels"""
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        log_probs = self.predict_log_proba(X)
        # Convert log probabilities back to probabilities
        probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def predict_log_proba(self, X):
        """Predict log class probabilities"""
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probs = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # Add log prior
            log_probs[:, i] = np.log(self.priors_[i])
            
            if self.variant == 'gaussian':
                log_probs[:, i] += self._gaussian_log_likelihood(X, i)
            elif self.variant == 'multinomial':
                log_probs[:, i] += self._multinomial_log_likelihood(X, i)
            elif self.variant == 'bernoulli':
                log_probs[:, i] += self._bernoulli_log_likelihood(X, i)
            elif self.variant == 'categorical':
                log_probs[:, i] += self._gaussian_log_likelihood(X, i)
        
        return log_probs
    
    def _gaussian_log_likelihood(self, X, class_idx):
        """Calculate Gaussian log-likelihood"""
        means = self.means_[class_idx]
        variances = self.variances_[class_idx]
        
        # Gaussian log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * variances) + 
            (X - means) ** 2 / variances, axis=1
        )
        
        return log_likelihood
    
    def _multinomial_log_likelihood(self, X, class_idx):
        """Calculate Multinomial log-likelihood"""
        feature_probs = self.feature_probs_[class_idx]
        
        # Multinomial log-likelihood
        log_likelihood = np.sum(X * np.log(feature_probs + 1e-10), axis=1)
        
        return log_likelihood
    
    def _bernoulli_log_likelihood(self, X, class_idx):
        """Calculate Bernoulli log-likelihood"""
        feature_probs = self.feature_probs_[class_idx]
        
        # Bernoulli log-likelihood
        X_binary = (X > 0).astype(float)
        log_likelihood = np.sum(
            X_binary * np.log(feature_probs + 1e-10) + 
            (1 - X_binary) * np.log(1 - feature_probs + 1e-10), axis=1
        )
        
        return log_likelihood
    
    def score(self, X, y):
        """Compute accuracy score"""
        return accuracy_score(y, self.predict(X))


class GaussianNaiveBayes(NaiveBayesClassifier):
    """Gaussian Naive Bayes for continuous features"""
    def __init__(self, alpha=1e-10):
        super().__init__(variant='gaussian', alpha=alpha)


class MultinomialNaiveBayes(NaiveBayesClassifier):
    """Multinomial Naive Bayes for count data"""
    def __init__(self, alpha=1.0):
        super().__init__(variant='multinomial', alpha=alpha)


class BernoulliNaiveBayes(NaiveBayesClassifier):
    """Bernoulli Naive Bayes for binary features"""
    def __init__(self, alpha=1.0):
        super().__init__(variant='bernoulli', alpha=alpha)


class CategoricalNaiveBayes(NaiveBayesClassifier):
    """Categorical Naive Bayes for categorical features"""
    def __init__(self, alpha=1.0):
        super().__init__(variant='categorical', alpha=alpha)


def generate_synthetic_data(n_samples=1000, n_features=2, n_classes=3, random_state=42):
    """
    Generate synthetic data for Naive Bayes demonstration
    """
    np.random.seed(random_state)
    
    # Generate class means
    means = np.random.randn(n_classes, n_features) * 2
    
    # Generate diagonal covariance matrices (independent features)
    covariances = []
    for i in range(n_classes):
        # Create diagonal covariance matrix
        cov = np.eye(n_features) * np.random.uniform(0.5, 2.0, n_features)
        covariances.append(cov)
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        class_samples = np.random.multivariate_normal(
            means[i], covariances[i], samples_per_class
        )
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    return np.vstack(X), np.array(y)


def demonstrate_naive_bayes():
    """
    Demonstrate Naive Bayes with synthetic data
    """
    # Generate data
    X, y = generate_synthetic_data(n_samples=900, n_features=2, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit our implementation
    nb_scratch = GaussianNaiveBayes()
    nb_scratch.fit(X_train, y_train)
    
    # Fit sklearn implementation
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    
    # Compare predictions
    y_pred_scratch = nb_scratch.predict(X_test)
    y_pred_sklearn = nb_sklearn.predict(X_test)
    
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
    axes[0].grid(True, alpha=0.3)
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = nb_scratch.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z, alpha=0.3)
    for i in range(3):
        mask = y == i
        axes[1].scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f'Class {i}')
    axes[1].set_title('Decision Boundaries')
    axes[1].legend()
    
    # Feature importance
    feature_importance = np.var(nb_scratch.means_, axis=0) / np.mean(nb_scratch.variances_, axis=0)
    axes[2].bar(range(len(feature_importance)), feature_importance)
    axes[2].set_title('Feature Importance (Variance Ratio)')
    axes[2].set_xlabel('Feature')
    axes[2].set_ylabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    return nb_scratch, nb_sklearn


def demonstrate_numerical_issues():
    """
    Demonstrate numerical stability issues in Naive Bayes
    """
    # Generate data with one class far from others
    np.random.seed(42)
    
    # Class 0: centered at (0, 0)
    X0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    
    # Class 1: centered at (10, 10) - far from class 0
    X1 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], 100)
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(100), np.ones(100)])
    
    # Test point far from both classes
    test_point = np.array([[20, 20]])
    
    # Fit Naive Bayes
    nb = GaussianNaiveBayes()
    nb.fit(X, y)
    
    # Calculate probabilities using different methods
    print("Numerical Stability Demonstration:")
    print("-" * 50)
    
    # Method 1: Direct probability calculation (problematic)
    means = nb.means_
    variances = nb.variances_
    
    # Calculate Gaussian PDF directly
    pdf_values = []
    for i in range(len(nb.classes_)):
        pdf = 1.0
        for j in range(test_point.shape[1]):
            pdf *= norm.pdf(test_point[0, j], means[i, j], np.sqrt(variances[i, j]))
        pdf_values.append(pdf)
    
    print(f"Direct PDF values: {pdf_values}")
    print(f"Direct probabilities: {np.array(pdf_values) / np.sum(pdf_values)}")
    
    # Method 2: Log-probability calculation (stable)
    log_probs = nb.predict_log_proba(test_point)
    probs = np.exp(log_probs - np.max(log_probs))
    probs = probs / np.sum(probs)
    
    print(f"Log-probability approach: {probs.flatten()}")
    
    # Visualize the issue
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original data
    for i in range(2):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0].scatter(test_point[0, 0], test_point[0, 1], c='red', s=100, marker='x', label='Test Point')
    axes[0].set_title('Data and Test Point')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PDF values vs distance
    distances = np.linspace(0, 30, 100)
    pdf_at_distance = norm.pdf(distances, 0, 1)
    log_pdf_at_distance = norm.logpdf(distances, 0, 1)
    
    axes[1].plot(distances, pdf_at_distance, label='PDF', alpha=0.7)
    axes[1].set_xlabel('Distance from Mean')
    axes[1].set_ylabel('PDF Value')
    axes[1].set_title('PDF vs Distance (Numerical Underflow)')
    axes[1].grid(True, alpha=0.3)
    
    # Add log-PDF on secondary axis
    ax2 = axes[1].twinx()
    ax2.plot(distances, log_pdf_at_distance, 'r--', label='Log-PDF', alpha=0.7)
    ax2.set_ylabel('Log-PDF Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    plt.show()
    
    return nb, pdf_values, probs


def safe_naive_bayes_predict(X, nb_model):
    """
    Safe prediction using log-probabilities
    """
    log_probs = nb_model.predict_log_proba(X)
    return nb_model.classes_[np.argmax(log_probs, axis=1)]


def regularized_naive_bayes(X, y, alpha=1e-10):
    """
    Regularized Naive Bayes to prevent zero variances
    """
    nb = GaussianNaiveBayes(alpha=alpha)
    nb.fit(X, y)
    return nb


def truncated_naive_bayes(X, y, threshold=1e-10):
    """
    Naive Bayes with probability truncation (not recommended)
    """
    nb = GaussianNaiveBayes()
    nb.fit(X, y)
    
    # Override predict method to truncate probabilities
    def predict_truncated(X):
        probs = nb.predict_proba(X)
        # Truncate very small probabilities
        probs = np.maximum(probs, threshold)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return nb.classes_[np.argmax(probs, axis=1)]
    
    nb.predict = predict_truncated
    return nb


def text_classification_example():
    """
    Naive Bayes for text classification (sentiment analysis)
    """
    # Sample text data
    texts = [
        "I love this movie, it's amazing!",
        "This is the worst film I've ever seen",
        "Great acting and wonderful story",
        "Terrible plot, boring characters",
        "Fantastic cinematography and direction",
        "Awful script and poor acting",
        "Beautiful and inspiring movie",
        "Disappointing and waste of time",
        "Excellent performance by all actors",
        "Horrible waste of money"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    
    # Vectorize text
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Fit Multinomial Naive Bayes
    nb = MultinomialNaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)
    
    # Predict
    y_pred = nb.predict(X_test)
    
    print("Text Classification Results:")
    print("-" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Feature importance (most discriminative words)
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = np.abs(nb.feature_probs_[1] - nb.feature_probs_[0])
    
    # Get top discriminative words
    top_indices = np.argsort(feature_importance)[-10:]
    
    print("\nTop Discriminative Words:")
    for idx in reversed(top_indices):
        word = feature_names[idx]
        importance = feature_importance[idx]
        pos_prob = nb.feature_probs_[1, idx]
        neg_prob = nb.feature_probs_[0, idx]
        print(f"{word}: {importance:.4f} (pos: {pos_prob:.4f}, neg: {neg_prob:.4f})")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_indices)), feature_importance[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Discriminative Power')
    plt.title('Most Discriminative Words for Sentiment Analysis')
    plt.tight_layout()
    plt.show()
    
    return nb, vectorizer


def medical_diagnosis_example():
    """
    Naive Bayes for medical diagnosis
    """
    # Simulate medical data
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic medical features
    # Feature 0: Age (normalized)
    age = np.random.normal(0, 1, n_samples)
    
    # Feature 1: Blood pressure (normalized)
    bp = np.random.normal(0, 1, n_samples)
    
    # Feature 2: Cholesterol level (normalized)
    cholesterol = np.random.normal(0, 1, n_samples)
    
    # Feature 3: BMI (normalized)
    bmi = np.random.normal(0, 1, n_samples)
    
    X = np.column_stack([age, bp, cholesterol, bmi])
    
    # Generate disease labels based on features
    # Higher values of features increase disease probability
    disease_prob = 1 / (1 + np.exp(-(0.3*age + 0.5*bp + 0.4*cholesterol + 0.2*bmi)))
    y = np.random.binomial(1, disease_prob)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Fit Gaussian Naive Bayes
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    
    # Predict
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)
    
    print("Medical Diagnosis Results:")
    print("-" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Disease']))
    
    # Feature importance analysis
    feature_names = ['Age', 'Blood Pressure', 'Cholesterol', 'BMI']
    feature_importance = np.var(nb.means_, axis=0) / np.mean(nb.variances_, axis=0)
    
    print("\nFeature Importance (Variance Ratio):")
    for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
        print(f"{name}: {importance:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Feature distributions by class
    for i, feature_name in enumerate(feature_names):
        row, col = i // 2, i % 2
        
        for j, class_label in enumerate([0, 1]):
            mask = y == class_label
            axes[row, col].hist(X[mask, i], alpha=0.7, label=f'Class {class_label}', bins=20)
        
        axes[row, col].set_title(f'{feature_name} Distribution')
        axes[row, col].set_xlabel(feature_name)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ROC curve for probability predictions
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Medical Diagnosis')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return nb, feature_importance


def compare_naive_bayes_variants():
    """
    Compare different Naive Bayes variants
    """
    # Generate different types of data
    np.random.seed(42)
    
    # 1. Continuous data (Gaussian)
    X_gaussian, y_gaussian = generate_synthetic_data(n_samples=300, n_features=2, n_classes=2)
    
    # 2. Count data (Multinomial)
    X_multinomial = np.random.poisson(5, (300, 10))
    y_multinomial = np.random.randint(0, 2, 300)
    
    # 3. Binary data (Bernoulli)
    X_bernoulli = np.random.binomial(1, 0.3, (300, 10))
    y_bernoulli = np.random.randint(0, 2, 300)
    
    # Test different variants
    variants = {
        'Gaussian': GaussianNaiveBayes(),
        'Multinomial': MultinomialNaiveBayes(),
        'Bernoulli': BernoulliNaiveBayes()
    }
    
    datasets = {
        'Gaussian': (X_gaussian, y_gaussian),
        'Multinomial': (X_multinomial, y_multinomial),
        'Bernoulli': (X_bernoulli, y_bernoulli)
    }
    
    results = {}
    
    for variant_name, model in variants.items():
        for data_name, (X, y) in datasets.items():
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            key = f"{variant_name} on {data_name}"
            results[key] = accuracy
    
    # Display results
    print("Naive Bayes Variants Comparison:")
    print("-" * 50)
    for key, accuracy in results.items():
        print(f"{key}: {accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    variants_list = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(range(len(variants_list)), accuracies)
    plt.xlabel('Model-Dataset Combination')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Variants Performance Comparison')
    plt.xticks(range(len(variants_list)), variants_list, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_feature_independence(X, y):
    """
    Analyze feature independence assumption
    """
    n_features = X.shape[1]
    correlations = np.zeros((n_features, n_features))
    
    # Calculate correlations for each class
    classes = np.unique(y)
    
    for c in classes:
        class_mask = y == c
        class_data = X[class_mask]
        class_corr = np.corrcoef(class_data.T)
        correlations += class_corr
    
    correlations /= len(classes)
    
    # Visualize correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix (Averaged over Classes)')
    plt.tight_layout()
    plt.show()
    
    # Calculate average absolute correlation (excluding diagonal)
    mask = ~np.eye(n_features, dtype=bool)
    avg_correlation = np.mean(np.abs(correlations[mask]))
    
    print(f"Average absolute correlation: {avg_correlation:.4f}")
    print("Correlation interpretation:")
    if avg_correlation < 0.1:
        print("Features are approximately independent (good for Naive Bayes)")
    elif avg_correlation < 0.3:
        print("Features have moderate correlation (acceptable for Naive Bayes)")
    else:
        print("Features are highly correlated (may affect Naive Bayes performance)")
    
    return correlations, avg_correlation


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
    nb_numerical, pdf_vals, probs = demonstrate_numerical_issues()
    
    # Text classification example
    print("\n3. Text Classification Example:")
    nb_text, vectorizer = text_classification_example()
    
    # Medical diagnosis example
    print("\n4. Medical Diagnosis Example:")
    nb_medical, feature_importance = medical_diagnosis_example()
    
    # Compare variants
    print("\n5. Naive Bayes Variants Comparison:")
    results = compare_naive_bayes_variants()
    
    # Analyze feature independence
    print("\n6. Feature Independence Analysis:")
    X, y = generate_synthetic_data(n_samples=500, n_features=4, n_classes=3)
    correlations, avg_corr = analyze_feature_independence(X, y)
    
    # Cross-validation comparison
    print("\n7. Cross-Validation Comparison:")
    X, y = generate_synthetic_data(n_samples=300, n_features=2, n_classes=2)
    
    nb_scratch = GaussianNaiveBayes()
    nb_sklearn = GaussianNB()
    
    cv_scratch = cross_val_score(nb_scratch, X, y, cv=5)
    cv_sklearn = cross_val_score(nb_sklearn, X, y, cv=5)
    
    print(f"Our Implementation CV Score: {cv_scratch.mean():.4f} (+/- {cv_scratch.std() * 2:.4f})")
    print(f"Sklearn Implementation CV Score: {cv_sklearn.mean():.4f} (+/- {cv_sklearn.std() * 2:.4f})")
    
    return (nb_scratch, nb_sklearn, nb_text, nb_medical, 
            results, correlations, cv_scratch, cv_sklearn)


if __name__ == "__main__":
    main()
