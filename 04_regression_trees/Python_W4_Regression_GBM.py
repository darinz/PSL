"""
Gradient Boosting Machines (GBM) for Regression
===============================================

This script demonstrates how to use Gradient Boosting Machines for regression
on the Boston Housing dataset. It includes:
- Data loading and preprocessing
- Model fitting with different hyperparameters
- Performance evaluation with MSE plots
- Feature importance analysis

Author: Statistical Learning Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
random.seed(124)

def load_housing_data():
    """
    Load the Boston Housing dataset from URL.
    
    Returns:
        pandas.DataFrame: Housing dataset
    """
    url = "https://liangfgithub.github.io/Data/HousingData.csv"
    housing = pd.read_csv(url)
    return housing

def prepare_data(housing):
    """
    Prepare features and target variables, split into train/test sets.
    
    Args:
        housing (pandas.DataFrame): Housing dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = housing.drop("Y", axis=1)
    y = housing["Y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=124
    )
    
    return X_train, X_test, y_train, y_test

def fit_gbm_model(X_train, y_train, learning_rate=1.0, n_estimators=100, subsample=1.0):
    """
    Fit a Gradient Boosting Regressor model.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        learning_rate (float): Learning rate (shrinkage factor)
        n_estimators (int): Number of trees
        subsample (float): Fraction of training data used for learning
        
    Returns:
        GradientBoostingRegressor: Fitted model
    """
    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model_performance(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance by calculating MSE for each boosting iteration.
    
    Args:
        model (GradientBoostingRegressor): Fitted GBM model
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        tuple: (train_mse, test_mse, best_iter)
    """
    n_estimators = model.n_estimators
    
    # Calculate training MSE for each iteration
    train_mse = np.zeros(n_estimators, dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(X_train)):
        train_mse[i] = np.mean((y_train - y_pred) ** 2.0)
    
    # Calculate test MSE for each iteration
    test_mse = np.zeros(n_estimators, dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_mse[i] = np.mean((y_test - y_pred) ** 2.0)
    
    # Find best iteration (minimum test MSE)
    best_iter = np.argmin(test_mse)
    
    return train_mse, test_mse, best_iter

def plot_performance(train_mse, test_mse, best_iter, title="GBM Performance"):
    """
    Plot training and test MSE over boosting iterations.
    
    Args:
        train_mse (numpy.ndarray): Training MSE for each iteration
        test_mse (numpy.ndarray): Test MSE for each iteration
        best_iter (int): Best iteration (minimum test MSE)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training and test MSE
    iterations = np.arange(1, len(train_mse) + 1)
    plt.plot(iterations, train_mse, "-", label="Training MSE")
    plt.plot(iterations, test_mse, "-", label="Test MSE")
    
    # Mark best iteration
    plt.axvline(x=best_iter + 1, color='k', ls=":", label=f"Best iteration: {best_iter + 1}")
    
    plt.xlabel("Boosting Iterations")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title="Feature Importances"):
    """
    Plot feature importances from the GBM model.
    
    Args:
        model (GradientBoostingRegressor): Fitted GBM model
        feature_names (list): Names of features
        title (str): Plot title
    """
    # Get feature importances and sort them
    importances = pd.Series(
        model.feature_importances_, 
        index=feature_names
    ).sort_values()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Mean decrease in impurity")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the complete GBM regression example.
    """
    print("Loading Boston Housing dataset...")
    housing = load_housing_data()
    print(f"Dataset shape: {housing.shape}")
    print(f"Features: {list(housing.columns)}")
    print()
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(housing)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print()
    
    # Fit first GBM model (high learning rate, few trees)
    print("Fitting GBM Model 1 (learning_rate=1.0, n_estimators=100)...")
    model1 = fit_gbm_model(X_train, y_train, learning_rate=1.0, n_estimators=100)
    
    # Evaluate first model
    print("Evaluating Model 1 performance...")
    train_mse1, test_mse1, best_iter1 = evaluate_model_performance(
        model1, X_train, y_train, X_test, y_test
    )
    
    print(f"Best iteration: {best_iter1 + 1}")
    print(f"Best test MSE: {test_mse1[best_iter1]:.4f}")
    print()
    
    # Plot performance for first model
    plot_performance(train_mse1, test_mse1, best_iter1, "GBM Model 1 Performance")
    
    # Fit second GBM model (low learning rate, many trees, subsampling)
    print("Fitting GBM Model 2 (learning_rate=0.02, n_estimators=1000, subsample=0.5)...")
    model2 = fit_gbm_model(X_train, y_train, learning_rate=0.02, n_estimators=1000, subsample=0.5)
    
    # Evaluate second model
    print("Evaluating Model 2 performance...")
    train_mse2, test_mse2, best_iter2 = evaluate_model_performance(
        model2, X_train, y_train, X_test, y_test
    )
    
    print(f"Best iteration: {best_iter2 + 1}")
    print(f"Best test MSE: {test_mse2[best_iter2]:.4f}")
    print()
    
    # Plot performance for second model
    plot_performance(train_mse2, test_mse2, best_iter2, "GBM Model 2 Performance")
    
    # Plot feature importance from first model
    print("Plotting feature importances...")
    plot_feature_importance(model1, X_train.columns, "Feature Importances (Model 1)")
    
    # Compare final performance
    print("Final Performance Comparison:")
    print(f"Model 1 (high learning rate): Test MSE = {test_mse1[-1]:.4f}")
    print(f"Model 2 (low learning rate):  Test MSE = {test_mse2[-1]:.4f}")
    
    if test_mse1[-1] < test_mse2[-1]:
        print("Model 1 performed better on test set.")
    else:
        print("Model 2 performed better on test set.")

if __name__ == "__main__":
    main() 