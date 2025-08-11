"""
Linear Regression Analysis in Python
====================================

This script demonstrates linear regression analysis using the Prostate cancer dataset.
It covers data loading, exploration, model fitting using multiple approaches, and prediction.

Key Learning Objectives:
- Data preprocessing and exploration
- Multiple approaches to linear regression (sklearn, statsmodels, numpy)
- Model interpretation and evaluation
- Prediction with new data
- Understanding the impact of irrelevant variables

"""

# =============================================================================
# Import Libraries
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# Data Loading and Understanding
# =============================================================================

def load_prostate_data():
    """
    Load the Prostate cancer dataset from the Elements of Statistical Learning website.
    
    Dataset Description:
    - Purpose: Examine correlation between prostate-specific antigen levels and clinical measures
    - Population: Men about to receive radical prostatectomy
    - Variables: 8 predictors + 1 response variable
    
    Returns:
        pandas.DataFrame: Cleaned dataset ready for analysis
    """
    
    # Load data from URL
    url = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
    data = pd.read_table(url)
    
    print("="*60)
    print("PROSTATE CANCER DATASET LOADED")
    print("="*60)
    print(f"Original data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Remove the index column (first column) which is redundant
    data = data.drop(data.columns[0], axis=1)
    
    print(f"Cleaned data shape: {data.shape}")
    print("="*60)
    
    return data

def explore_data(data):
    """
    Comprehensive data exploration to understand the dataset structure and quality.
    
    Args:
        data (pandas.DataFrame): The dataset to explore
    """
    
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found!")
    else:
        print("⚠ Missing values detected - consider imputation")
    
    # Display data types and basic info
    print(f"\nData types:")
    print(data.dtypes)
    
    # Descriptive statistics
    print(f"\nDescriptive statistics:")
    print(data.describe().round(3))
    
    # Variable descriptions for reference
    variable_descriptions = {
        'lcavol': 'Log cancer volume',
        'lweight': 'Log prostate weight', 
        'age': 'Age in years',
        'lbph': 'Log of benign prostatic hyperplasia amount',
        'svi': 'Seminal vesicle invasion (binary)',
        'lcp': 'Log of capsular penetration',
        'gleason': 'Gleason score (numeric)',
        'pgg45': 'Percent of Gleason score 4 or 5',
        'lpsa': 'Log prostate-specific antigen (response)'
    }
    
    print(f"\nVariable descriptions:")
    for var, desc in variable_descriptions.items():
        print(f"  {var}: {desc}")

def visualize_data(data):
    """
    Create comprehensive visualizations to understand data relationships.
    
    Args:
        data (pandas.DataFrame): The dataset to visualize
    """
    
    print("\n" + "="*50)
    print("DATA VISUALIZATION")
    print("="*50)
    
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of All Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Create pairplot for key variables (subset to avoid overcrowding)
    key_vars = ['lcavol', 'lweight', 'age', 'lpsa']
    print(f"\nCreating pairplot for key variables: {key_vars}")
    sns.pairplot(data[key_vars], diag_kind='kde')
    plt.suptitle('Pairwise Relationships of Key Variables', y=1.02, fontsize=14)
    plt.show()
    
    # Box plot for response variable
    plt.figure(figsize=(8, 6))
    plt.boxplot(data['lpsa'])
    plt.title('Distribution of Log Prostate-Specific Antigen (Response)', fontsize=12)
    plt.ylabel('lpsa')
    plt.show()

# =============================================================================
# Linear Regression Implementation Methods
# =============================================================================

def sklearn_linear_regression(X, y, X_new=None):
    """
    Implement linear regression using scikit-learn.
    
    Advantages:
    - Simple and intuitive API
    - Fast implementation
    - Good for production use
    
    Args:
        X (pandas.DataFrame): Predictor variables
        y (pandas.Series): Response variable
        X_new (pandas.DataFrame, optional): New data for prediction
    
    Returns:
        dict: Model results including coefficients and predictions
    """
    
    print("\n" + "="*50)
    print("SCIKIT-LEARN LINEAR REGRESSION")
    print("="*50)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Extract results
    intercept = model.intercept_
    coefficients = model.coef_
    
    print(f"Intercept: {intercept:.6f}")
    print("\nCoefficients:")
    for i, col in enumerate(X.columns):
        print(f"  {col:>10}: {coefficients[i]:>10.6f}")
    
    # Make predictions if new data provided
    predictions = None
    if X_new is not None:
        predictions = model.predict(X_new)
        print(f"\nPredictions for new data:")
        for i, pred in enumerate(predictions):
            print(f"  Observation {i+1}: {pred:.6f}")
    
    return {
        'intercept': intercept,
        'coefficients': coefficients,
        'predictions': predictions,
        'model': model
    }

def statsmodels_formula_regression(data, X_new=None):
    """
    Implement linear regression using statsmodels formula API (R-like syntax).
    
    Advantages:
    - R-like formula syntax
    - Comprehensive statistical output
    - Built-in hypothesis testing
    
    Args:
        data (pandas.DataFrame): Complete dataset
        X_new (pandas.DataFrame, optional): New data for prediction
    
    Returns:
        dict: Model results including statistical summaries
    """
    
    print("\n" + "="*50)
    print("STATSMODELS FORMULA API REGRESSION")
    print("="*50)
    
    # Create formula string (R-like syntax)
    predictors = ' + '.join(data.columns[:-1])  # All columns except response
    formula = f"lpsa ~ {predictors}"
    print(f"Formula: {formula}")
    
    # Fit the model
    model = smf.ols(formula, data=data).fit()
    
    # Display summary statistics
    print(f"\nModel Summary:")
    print(f"R-squared: {model.rsquared:.6f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.6f}")
    print(f"AIC: {model.aic:.6f}")
    print(f"BIC: {model.bic:.6f}")
    
    # Display coefficients with statistical significance
    print(f"\nCoefficients with statistical significance:")
    print(model.summary().tables[1])
    
    # Make predictions if new data provided
    predictions = None
    if X_new is not None:
        predictions = model.predict(X_new)
        print(f"\nPredictions for new data:")
        for i, pred in enumerate(predictions):
            print(f"  Observation {i+1}: {pred:.6f}")
    
    return {
        'model': model,
        'predictions': predictions,
        'rsquared': model.rsquared,
        'adj_rsquared': model.rsquared_adj
    }

def numpy_manual_regression(X, y, X_new=None):
    """
    Implement linear regression manually using NumPy (normal equations).
    
    Educational Purpose:
    - Shows the mathematical foundation
    - Demonstrates matrix operations
    - Helps understand the underlying algorithm
    
    Args:
        X (pandas.DataFrame): Predictor variables
        y (pandas.Series): Response variable
        X_new (pandas.DataFrame, optional): New data for prediction
    
    Returns:
        dict: Model results including coefficients and predictions
    """
    
    print("\n" + "="*50)
    print("NUMPY MANUAL REGRESSION (NORMAL EQUATIONS)")
    print("="*50)
    
    # Convert to numpy arrays
    X_np = X.values
    y_np = y.values
    
    # Add intercept column (column of ones)
    X_with_intercept = np.column_stack([np.ones(len(X_np)), X_np])
    
    # Solve normal equations: β = (X'X)^(-1) X'y
    print("Solving normal equations: β = (X'X)^(-1) X'y")
    
    # Calculate (X'X)^(-1)
    XtX = X_with_intercept.T @ X_with_intercept
    XtX_inv = np.linalg.inv(XtX)
    
    # Calculate X'y
    Xty = X_with_intercept.T @ y_np
    
    # Calculate coefficients
    coefficients = XtX_inv @ Xty
    
    intercept = coefficients[0]
    coefs = coefficients[1:]
    
    print(f"Intercept: {intercept:.6f}")
    print("\nCoefficients:")
    for i, col in enumerate(X.columns):
        print(f"  {col:>10}: {coefs[i]:>10.6f}")
    
    # Make predictions if new data provided
    predictions = None
    if X_new is not None:
        X_new_np = X_new.values
        X_new_with_intercept = np.column_stack([np.ones(len(X_new_np)), X_new_np])
        predictions = X_new_with_intercept @ coefficients
        
        print(f"\nPredictions for new data:")
        for i, pred in enumerate(predictions):
            print(f"  Observation {i+1}: {pred:.6f}")
    
    return {
        'intercept': intercept,
        'coefficients': coefs,
        'predictions': predictions
    }

def calculate_regression_metrics(y_true, y_pred, n_predictors):
    """
    Calculate key regression metrics manually for educational purposes.
    
    Args:
        y_true (array-like): True response values
        y_pred (array-like): Predicted response values
        n_predictors (int): Number of predictor variables
    
    Returns:
        dict: Dictionary containing various regression metrics
    """
    
    print("\n" + "="*50)
    print("REGRESSION METRICS CALCULATION")
    print("="*50)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Sample size
    n = len(y_true)
    p = n_predictors
    
    print(f"Sample size (n): {n}")
    print(f"Number of predictors (p): {p}")
    print(f"Degrees of freedom: {n - p - 1}")
    
    # Residual Standard Error (σ̂)
    # Formula: σ̂ = √(Σ(r_i²) / (n - p - 1))
    residual_ss = np.sum(residuals**2)
    residual_se = np.sqrt(residual_ss / (n - p - 1))
    print(f"\nResidual Standard Error (σ̂): {residual_se:.6f}")
    print(f"  Formula: √(Σ(r_i²) / (n - p - 1))")
    print(f"  Calculation: √({residual_ss:.6f} / {n - p - 1}) = {residual_se:.6f}")
    
    # R-squared
    # Formula: R² = 1 - (SS_res / SS_tot)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (residual_ss / ss_total)
    print(f"\nR-squared (R²): {r_squared:.6f}")
    print(f"  Formula: 1 - (SS_res / SS_tot)")
    print(f"  Calculation: 1 - ({residual_ss:.6f} / {ss_total:.6f}) = {r_squared:.6f}")
    
    # Adjusted R-squared
    # Formula: R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    print(f"\nAdjusted R-squared: {adj_r_squared:.6f}")
    print(f"  Formula: 1 - (1 - R²) * (n - 1) / (n - p - 1)")
    
    # Log-likelihood (assuming normal distribution)
    log_likelihood = -(n/2) * np.log(2*np.pi) - (n/2) * np.log(residual_se**2) - (1/(2*residual_se**2)) * residual_ss
    print(f"\nLog-likelihood: {log_likelihood:.6f}")
    
    return {
        'residual_se': residual_se,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'log_likelihood': log_likelihood,
        'residuals': residuals
    }

def demonstrate_irrelevant_variables(data, X_new):
    """
    Demonstrate the impact of adding irrelevant variables to the model.
    
    Educational Purpose:
    - Shows that adding irrelevant variables increases R² but not necessarily predictive power
    - Demonstrates the importance of model parsimony
    - Illustrates overfitting concept
    
    Args:
        data (pandas.DataFrame): Original dataset
        X_new (pandas.DataFrame): New data for prediction
    """
    
    print("\n" + "="*50)
    print("IMPACT OF IRRELEVANT VARIABLES")
    print("="*50)
    
    # Create a copy of the data
    data_with_junk = data.copy()
    
    # Add irrelevant variables
    np.random.seed(42)  # For reproducibility
    data_with_junk['junk1'] = np.random.normal(0, 1, len(data))
    data_with_junk['junk2'] = np.random.uniform(0, 1, len(data))
    data_with_junk['junk3'] = np.random.poisson(5, len(data))
    
    # Add same variables to new data
    X_new_with_junk = X_new.copy()
    X_new_with_junk['junk1'] = np.random.normal(0, 1, len(X_new))
    X_new_with_junk['junk2'] = np.random.uniform(0, 1, len(X_new))
    X_new_with_junk['junk3'] = np.random.poisson(5, len(X_new))
    
    # Fit models
    print("Fitting original model (without junk variables)...")
    original_model = smf.ols('lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45', 
                            data=data).fit()
    
    print("Fitting model with junk variables...")
    junk_model = smf.ols('lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45 + junk1 + junk2 + junk3', 
                        data=data_with_junk).fit()
    
    # Compare models
    print(f"\nModel Comparison:")
    print(f"{'Metric':<20} {'Original':<15} {'With Junk':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'R-squared':<20} {original_model.rsquared:<15.6f} {junk_model.rsquared:<15.6f} {junk_model.rsquared - original_model.rsquared:<15.6f}")
    print(f"{'Adj R-squared':<20} {original_model.rsquared_adj:<15.6f} {junk_model.rsquared_adj:<15.6f} {junk_model.rsquared_adj - original_model.rsquared_adj:<15.6f}")
    print(f"{'AIC':<20} {original_model.aic:<15.6f} {junk_model.aic:<15.6f} {junk_model.aic - original_model.aic:<15.6f}")
    print(f"{'BIC':<20} {original_model.bic:<15.6f} {junk_model.bic:<15.6f} {junk_model.bic - original_model.bic:<15.6f}")
    
    # Compare predictions
    original_pred = original_model.predict(X_new)
    junk_pred = junk_model.predict(X_new_with_junk)
    
    print(f"\nPrediction Comparison:")
    print(f"{'Observation':<12} {'Original':<15} {'With Junk':<15} {'Difference':<15}")
    print("-" * 57)
    for i in range(len(original_pred)):
        diff = junk_pred[i] - original_pred[i]
        print(f"{i+1:<12} {original_pred[i]:<15.6f} {junk_pred[i]:<15.6f} {diff:<15.6f}")
    
    print(f"\nKey Insights:")
    print(f"• R-squared increased by {junk_model.rsquared - original_model.rsquared:.6f}")
    print(f"• Adjusted R-squared decreased by {original_model.rsquared_adj - junk_model.rsquared_adj:.6f}")
    print(f"• AIC and BIC increased, indicating worse model fit")
    print(f"• Predictions changed, showing potential overfitting")

# =============================================================================
# Main Analysis Function
# =============================================================================

def main():
    """
    Main function that orchestrates the entire linear regression analysis.
    """
    
    print("LINEAR REGRESSION ANALYSIS IN PYTHON")
    print("="*60)
    print("Educational demonstration of multiple approaches to linear regression")
    print("="*60)
    
    # Step 1: Load and explore data
    data = load_prostate_data()
    explore_data(data)
    visualize_data(data)
    
    # Step 2: Prepare data for modeling
    X = data.iloc[:, :-1]  # All columns except the last one (predictors)
    y = data.iloc[:, -1]   # Last column (response)
    
    print(f"\nPredictor variables: {list(X.columns)}")
    print(f"Response variable: {y.name}")
    
    # Step 3: Create new data for prediction
    print("\n" + "="*50)
    print("CREATING NEW DATA FOR PREDICTION")
    print("="*50)
    
    # Create realistic new observations
    X_new = pd.DataFrame({
        'lcavol': [1.5, 1.8, 2.1],
        'lweight': [3.5, 3.8, 4.0],
        'age': [65, 70, 55],
        'lbph': [-0.5, -0.3, 0.1],
        'svi': [0, 1, 0],
        'lcp': [-0.5, -0.3, 0.2],
        'gleason': [7, 8, 6],
        'pgg45': [15, 20, 10]
    })
    
    print("New observations for prediction:")
    print(X_new)
    
    # Step 4: Implement different regression approaches
    sklearn_results = sklearn_linear_regression(X, y, X_new)
    statsmodels_results = statsmodels_formula_regression(data, X_new)
    numpy_results = numpy_manual_regression(X, y, X_new)
    
    # Step 5: Calculate and compare metrics
    y_pred_sklearn = sklearn_results['model'].predict(X)
    metrics = calculate_regression_metrics(y, y_pred_sklearn, len(X.columns))
    
    # Step 6: Demonstrate impact of irrelevant variables
    demonstrate_irrelevant_variables(data, X_new)
    
    # Step 7: Summary and conclusions
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY AND CONCLUSIONS")
    print("="*60)
    
    print(f"Dataset: Prostate cancer data")
    print(f"Sample size: {len(data)}")
    print(f"Number of predictors: {len(X.columns)}")
    print(f"Response variable: {y.name}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
    print(f"Adjusted R-squared: {metrics['adj_r_squared']:.4f}")
    print(f"Residual Standard Error: {metrics['residual_se']:.4f}")
    
    print(f"\nKey Learning Points:")
    print(f"1. All three methods (sklearn, statsmodels, numpy) produce identical results")
    print(f"2. Adding irrelevant variables increases R² but decreases adjusted R²")
    print(f"3. Model parsimony is important for generalization")
    print(f"4. Understanding the mathematical foundation helps interpret results")
    
    print(f"\nNext Steps:")
    print(f"• Consider cross-validation for model validation")
    print(f"• Explore regularization techniques (Ridge, Lasso)")
    print(f"• Investigate model diagnostics and assumptions")
    print(f"• Consider feature selection methods")
    
    print("="*60)

# =============================================================================
# Execute the analysis
# =============================================================================

if __name__ == "__main__":
    main() 