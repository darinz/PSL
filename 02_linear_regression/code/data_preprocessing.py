import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(X_raw):
    """
    Preprocess data for linear regression
    
    Parameters:
    X_raw: raw predictor matrix (without intercept)
    
    Returns:
    X_processed: processed design matrix
    """
    # Center predictors
    X_centered = X_raw - np.mean(X_raw, axis=0)
    
    # Scale predictors (optional)
    X_scaled = X_centered / np.std(X_centered, axis=0)
    
    # Add intercept column
    X_processed = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    
    return X_processed

# Example with preprocessing
np.random.seed(42)
X_raw = np.random.randn(100, 3)
X_processed = preprocess_data(X_raw)

print("=== Data Preprocessing ===")
print("Original data shape:", X_raw.shape)
print("Processed data shape:", X_processed.shape)
print("Original data mean:", np.mean(X_raw, axis=0))
print("Original data std:", np.std(X_raw, axis=0))
print("Processed data mean (excluding intercept):", np.mean(X_processed[:, 1:], axis=0))
print("Processed data std (excluding intercept):", np.std(X_processed[:, 1:], axis=0))
