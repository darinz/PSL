import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_olivetti_faces
import seaborn as sns


class FisherDiscriminantAnalysis:
    """
    Fisher Discriminant Analysis implementation from scratch
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.explained_variance_ratio_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit FDA model
        
        Parameters:
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Set number of components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # Calculate class means and overall mean
        class_means = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_means[i] = np.mean(X[class_mask], axis=0)
            class_counts[i] = np.sum(class_mask)
        
        overall_mean = np.average(class_means, weights=class_counts, axis=0)
        
        # Calculate between-class scatter matrix
        B = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            diff = class_means[i] - overall_mean
            B += class_counts[i] * np.outer(diff, diff)
        B /= (n_classes - 1)
        
        # Calculate within-class scatter matrix
        W = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            class_mask = y == c
            class_data = X[class_mask]
            diff = class_data - class_means[i]
            W += diff.T @ diff
        W /= (n_samples - n_classes)
        
        # Solve generalized eigenvalue problem: B * a = λ * W * a
        # This is equivalent to: W^(-1) * B * a = λ * a
        try:
            W_inv = np.linalg.inv(W)
            eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
            
            # Sort eigenvalues in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select top components
            self.scalings_ = eigenvecs[:, :self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components]
            
        except np.linalg.LinAlgError:
            # Handle singular W matrix
            print("Warning: Singular within-class scatter matrix. Using regularization.")
            W_reg = W + 1e-6 * np.eye(n_features)
            W_inv = np.linalg.inv(W_reg)
            eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
            
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            self.scalings_ = eigenvecs[:, :self.n_components]
            self.explained_variance_ratio_ = eigenvals[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform data using FDA projection
        """
        if self.scalings_ is None:
            raise ValueError("Model must be fitted before transform")
        
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """
        Fit FDA and transform data
        """
        return self.fit(X, y).transform(X)
    
    def get_discriminant_directions(self):
        """
        Return the discriminant directions (eigenvectors)
        """
        return self.scalings_


def generate_toy_data(n_samples=300, random_state=42):
    """
    Generate toy data for FDA demonstration
    """
    np.random.seed(random_state)
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0: centered at (0, 0)
    X0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_per_class)
    
    # Class 1: centered at (3, 2)
    X1 = np.random.multivariate_normal([3, 2], [[1, 0.5], [0.5, 1]], n_per_class)
    
    # Class 2: centered at (1, 4)
    X2 = np.random.multivariate_normal([1, 4], [[1, 0.5], [0.5, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    return X, y


def compare_pca_fda():
    """
    Compare PCA and FDA on toy data
    """
    X, y = generate_toy_data()
    
    # Apply PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)
    
    # Apply FDA (via LDA)
    lda = LinearDiscriminantAnalysis()
    X_fda = lda.fit_transform(X, y)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    for i in range(3):
        mask = y == i
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PCA projection
    for i in range(3):
        mask = y == i
        axes[0, 1].scatter(X_pca[mask], np.zeros_like(X_pca[mask]), alpha=0.7, label=f'Class {i}')
    axes[0, 1].set_title('PCA Projection (1D)')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-0.1, 0.1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # FDA projection
    for i in range(3):
        mask = y == i
        axes[1, 0].scatter(X_fda[mask], np.zeros_like(X_fda[mask]), alpha=0.7, label=f'Class {i}')
    axes[1, 0].set_title('FDA Projection (1D)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-0.1, 0.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Projection directions
    pca_direction = pca.components_[0]
    fda_direction = lda.scalings_[:, 0]
    
    # Normalize for visualization
    pca_direction = pca_direction / np.linalg.norm(pca_direction)
    fda_direction = fda_direction / np.linalg.norm(fda_direction)
    
    for i in range(3):
        mask = y == i
        axes[1, 1].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    
    # Plot projection directions
    origin = np.array([0, 0])
    axes[1, 1].quiver(origin[0], origin[1], pca_direction[0], pca_direction[1], 
                     color='red', scale=5, label='PCA Direction', linewidth=3)
    axes[1, 1].quiver(origin[0], origin[1], fda_direction[0], fda_direction[1], 
                     color='green', scale=5, label='FDA Direction', linewidth=3)
    
    axes[1, 1].set_title('Projection Directions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print separation metrics
    print("Class Separation Analysis:")
    print("-" * 40)
    
    # Calculate separation for PCA
    pca_separation = calculate_separation(X_pca, y)
    print(f"PCA Separation: {pca_separation:.4f}")
    
    # Calculate separation for FDA
    fda_separation = calculate_separation(X_fda, y)
    print(f"FDA Separation: {fda_separation:.4f}")
    
    return pca, lda


def calculate_separation(X_proj, y):
    """
    Calculate Fisher's separation criterion for projected data
    """
    classes = np.unique(y)
    overall_mean = np.mean(X_proj)
    
    # Between-class variance
    between_var = 0
    for c in classes:
        class_mean = np.mean(X_proj[y == c])
        n_class = np.sum(y == c)
        between_var += n_class * (class_mean - overall_mean) ** 2
    
    # Within-class variance
    within_var = 0
    for c in classes:
        class_data = X_proj[y == c]
        class_mean = np.mean(class_data)
        within_var += np.sum((class_data - class_mean) ** 2)
    
    return between_var / within_var if within_var > 0 else 0


def fda_for_regression(X, y, n_bins=10):
    """
    Apply FDA to regression by discretizing the response
    """
    # Discretize y into bins
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    
    # Apply FDA
    lda = LinearDiscriminantAnalysis()
    X_fda = lda.fit_transform(X, y_binned)
    
    return X_fda, lda


def demonstrate_fda_scratch():
    """
    Demonstrate FDA implementation from scratch
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    
    # Generate 3 classes with different means
    n_per_class = n_samples // 3
    
    # Class 0
    X0 = np.random.multivariate_normal([0, 0, 0, 0], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    # Class 1
    X1 = np.random.multivariate_normal([3, 2, 1, 0], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    # Class 2
    X2 = np.random.multivariate_normal([1, 4, 2, 3], 
                                     [[1, 0.5, 0.3, 0.2],
                                      [0.5, 1, 0.4, 0.3],
                                      [0.3, 0.4, 1, 0.5],
                                      [0.2, 0.3, 0.5, 1]], n_per_class)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), 2 * np.ones(n_per_class)])
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=2)
    X_fda = fda.fit_transform(X, y)
    
    # Compare with sklearn LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (first 2 dimensions)
    for i in range(3):
        mask = y == i
        axes[0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0].set_title('Original Data (First 2 Dimensions)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Our FDA projection
    for i in range(3):
        mask = y == i
        axes[1].scatter(X_fda[mask, 0], X_fda[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[1].set_title('Our FDA Projection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sklearn LDA projection
    for i in range(3):
        mask = y == i
        axes[2].scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[2].set_title('Sklearn LDA Projection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("FDA Results:")
    print("-" * 30)
    print(f"Number of components: {fda.n_components}")
    print(f"Explained variance ratios: {fda.explained_variance_ratio_}")
    print(f"Discriminant directions shape: {fda.scalings_.shape}")
    
    # Calculate separation
    separation_our = calculate_separation(X_fda, y)
    separation_sklearn = calculate_separation(X_lda, y)
    
    print(f"\nSeparation Analysis:")
    print(f"Our FDA: {separation_our:.4f}")
    print(f"Sklearn LDA: {separation_sklearn:.4f}")
    
    return fda, lda


def demonstrate_overfitting():
    """
    Demonstrate FDA overfitting in high dimensions
    """
    np.random.seed(42)
    
    # Generate high-dimensional data with random features
    n_samples = 20
    n_features = 50  # Much larger than n_samples
    
    # Random features
    X = np.random.randn(n_samples, n_features)
    
    # Binary labels
    y = np.random.randint(0, 2, n_samples)
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=1)
    X_fda = fda.fit_transform(X, y)
    
    # Calculate separation
    separation = calculate_separation(X_fda, y)
    
    print(f"High-dimensional FDA Results:")
    print(f"n_samples: {n_samples}")
    print(f"n_features: {n_features}")
    print(f"Separation: {separation:.4f}")
    print(f"Perfect separation achieved: {separation > 100}")
    
    # Visualize projection
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda[mask], np.zeros_like(X_fda[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.title('FDA Projection (Random Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with low-dimensional case
    X_low = X[:, :5]  # Use only first 5 features
    fda_low = FisherDiscriminantAnalysis(n_components=1)
    X_fda_low = fda_low.fit_transform(X_low, y)
    separation_low = calculate_separation(X_fda_low, y)
    
    plt.subplot(1, 2, 2)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda_low[mask], np.zeros_like(X_fda_low[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.title(f'FDA Projection (5 Features)\nSeparation: {separation_low:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return separation, separation_low


def regularized_fda(X, y, alpha=0.1, n_components=None):
    """
    Regularized FDA with shrinkage
    """
    n_classes = len(np.unique(y))
    n_samples, n_features = X.shape
    
    if n_components is None:
        n_components = min(n_classes - 1, n_features)
    
    # Calculate scatter matrices
    B, W = calculate_scatter_matrices(X, y)
    
    # Regularize W
    W_reg = W + alpha * np.eye(n_features)
    
    # Solve eigenvalue problem
    W_inv = np.linalg.inv(W_reg)
    eigenvals, eigenvecs = np.linalg.eigh(W_inv @ B)
    
    # Sort and select
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    scalings = eigenvecs[:, :n_components]
    
    return scalings, eigenvals[:n_components]


def calculate_scatter_matrices(X, y):
    """
    Calculate between-class and within-class scatter matrices
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape
    
    # Class means and counts
    class_means = np.zeros((n_classes, n_features))
    class_counts = np.zeros(n_classes)
    
    for i, c in enumerate(classes):
        class_mask = y == c
        class_means[i] = np.mean(X[class_mask], axis=0)
        class_counts[i] = np.sum(class_mask)
    
    overall_mean = np.average(class_means, weights=class_counts, axis=0)
    
    # Between-class scatter
    B = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        diff = class_means[i] - overall_mean
        B += class_counts[i] * np.outer(diff, diff)
    B /= (n_classes - 1)
    
    # Within-class scatter
    W = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        class_mask = y == c
        class_data = X[class_mask]
        diff = class_data - class_means[i]
        W += diff.T @ diff
    W /= (n_samples - n_classes)
    
    return B, W


def fda_with_feature_selection(X, y, n_features=10, n_components=None):
    """
    FDA with feature selection to reduce dimensionality
    """
    # Select most discriminative features
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=n_components)
    X_fda = fda.fit_transform(X_selected, y)
    
    return X_fda, fda, selector


def cross_validate_fda(X, y, n_splits=5):
    """
    Cross-validate FDA to assess generalization
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    separations = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit FDA on training data
        fda = FisherDiscriminantAnalysis()
        fda.fit(X_train, y_train)
        
        # Transform test data
        X_test_fda = fda.transform(X_test)
        
        # Calculate separation on test data
        separation = calculate_separation(X_test_fda, y_test)
        separations.append(separation)
    
    return np.mean(separations), np.std(separations)


def face_recognition_fda():
    """
    FDA for face recognition (simplified example)
    """
    # Load face dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply FDA
    fda = FisherDiscriminantAnalysis(n_components=39)  # 40 classes - 1
    X_train_fda = fda.fit_transform(X_train, y_train)
    X_test_fda = fda.transform(X_test)
    
    # Classify using k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_fda, y_train)
    y_pred = knn.predict(X_test_fda)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Face Recognition Accuracy: {accuracy:.4f}")
    
    # Visualize first few discriminant directions
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        direction = fda.scalings_[:, i].reshape(64, 64)
        axes[row, col].imshow(direction, cmap='RdBu_r')
        axes[row, col].set_title(f'Direction {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fda, accuracy


def gene_expression_fda():
    """
    FDA for gene expression classification
    """
    # Simulate gene expression data
    np.random.seed(42)
    n_samples = 100
    n_genes = 1000
    
    # Generate data with some discriminative genes
    X = np.random.randn(n_samples, n_genes)
    
    # Add discriminative signal to first 50 genes
    X[:50, :50] += 2  # Class 0
    X[50:, :50] -= 2  # Class 1
    
    y = np.hstack([np.zeros(50), np.ones(50)])
    
    # Apply FDA with feature selection
    X_fda, fda, selector = fda_with_feature_selection(X, y, n_features=100, n_components=1)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Original data (first 2 genes)
    plt.subplot(1, 3, 1)
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    plt.xlabel('Gene 1')
    plt.ylabel('Gene 2')
    plt.title('Original Data (First 2 Genes)')
    plt.legend()
    
    # Selected features
    plt.subplot(1, 3, 2)
    selected_features = selector.get_support()
    plt.bar(range(100), selected_features[:100])
    plt.xlabel('Gene Index')
    plt.ylabel('Selected')
    plt.title('Feature Selection')
    
    # FDA projection
    plt.subplot(1, 3, 3)
    for i in range(2):
        mask = y == i
        plt.scatter(X_fda[mask], np.zeros_like(X_fda[mask]), 
                   alpha=0.7, label=f'Class {i}')
    plt.xlabel('FDA Component')
    plt.ylabel('')
    plt.title('FDA Projection')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fda, selector


def plot_fda_directions(X, y, fda_model, title="FDA Discriminant Directions"):
    """
    Plot FDA discriminant directions and their importance
    """
    # Get discriminant directions
    directions = fda_model.get_discriminant_directions()
    explained_var = fda_model.explained_variance_ratio_
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    for i in range(len(np.unique(y))):
        mask = y == i
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], alpha=0.7, label=f'Class {i}')
    axes[0, 0].set_title('Original Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Explained variance ratio
    axes[0, 1].bar(range(len(explained_var)), explained_var)
    axes[0, 1].set_title('Explained Variance Ratio')
    axes[0, 1].set_xlabel('Component')
    axes[0, 1].set_ylabel('Variance Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # First discriminant direction
    if directions.shape[1] >= 1:
        direction1 = directions[:, 0]
        axes[1, 0].bar(range(len(direction1)), direction1)
        axes[1, 0].set_title('First Discriminant Direction')
        axes[1, 0].set_xlabel('Feature')
        axes[1, 0].set_ylabel('Coefficient')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Second discriminant direction (if available)
    if directions.shape[1] >= 2:
        direction2 = directions[:, 1]
        axes[1, 1].bar(range(len(direction2)), direction2)
        axes[1, 1].set_title('Second Discriminant Direction')
        axes[1, 1].set_xlabel('Feature')
        axes[1, 1].set_ylabel('Coefficient')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate FDA implementation
    """
    print("Fisher Discriminant Analysis Demonstration")
    print("=" * 50)
    
    # Compare PCA vs FDA
    print("\n1. PCA vs FDA Comparison:")
    pca, lda = compare_pca_fda()
    
    # Demonstrate FDA from scratch
    print("\n2. FDA Implementation from Scratch:")
    fda, lda_sklearn = demonstrate_fda_scratch()
    
    # Demonstrate overfitting
    print("\n3. Overfitting Demonstration:")
    high_sep, low_sep = demonstrate_overfitting()
    
    # Face recognition example
    print("\n4. Face Recognition Example:")
    face_fda, face_accuracy = face_recognition_fda()
    
    # Gene expression example
    print("\n5. Gene Expression Analysis:")
    gene_fda, gene_selector = gene_expression_fda()
    
    # Generate data for cross-validation
    X, y = generate_toy_data(n_samples=300)
    
    # Cross-validation
    print("\n6. Cross-Validation:")
    mean_sep, std_sep = cross_validate_fda(X, y, n_splits=5)
    print(f"Cross-validation separation: {mean_sep:.4f} (+/- {std_sep:.4f})")
    
    # Plot FDA directions
    print("\n7. FDA Directions Analysis:")
    plot_fda_directions(X, y, fda)
    
    return fda, lda_sklearn, face_fda, gene_fda


if __name__ == "__main__":
    main()
