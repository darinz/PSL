"""
Advanced Linear Regression Analysis in Python (Part II)
=======================================================

This script demonstrates advanced concepts in linear regression analysis including:
- Training vs Test Error Analysis
- Coefficient Interpretation and Multicollinearity
- Partial Regression Coefficients (Frisch-Waugh-Lovell Theorem)
- Hypothesis Testing (F-tests and t-tests)
- Collinearity Detection and Analysis

Key Learning Objectives:
- Understand overfitting and the bias-variance tradeoff
- Interpret coefficients in simple vs multiple regression
- Apply the Frisch-Waugh-Lovell theorem for partial effects
- Perform hypothesis testing manually and with statistical packages
- Detect and handle multicollinearity

"""

# =============================================================================
# Import Libraries
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import scipy.stats as stats

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_prepare_data():
    """
    Load and prepare the prostate cancer dataset for analysis.
    
    Returns:
        pandas.DataFrame: Cleaned dataset ready for analysis
    """
    
    print("="*60)
    print("LOADING PROSTATE CANCER DATASET")
    print("="*60)
    
    # Load data from URL
    url = "https://hastie.su.domains/ElemStatLearn/datasets/prostate.data"
    data = pd.read_table(url)
    
    print(f"Original data shape: {data.shape}")
    print(f"Original columns: {list(data.columns)}")
    
    # Data cleaning steps
    data.drop(data.columns[0], axis=1, inplace=True)  # Drop index column
    data.drop('train', axis=1, inplace=True)          # Drop train/test indicator
    data.columns.values[8] = 'Y'                      # Rename response variable
    
    print(f"Cleaned data shape: {data.shape}")
    print(f"Final columns: {list(data.columns)}")
    print("="*60)
    
    return data

# =============================================================================
# Training vs Test Error Analysis
# =============================================================================

def analyze_training_vs_test_error(data, n_simulations=5):
    """
    Demonstrate the bias-variance tradeoff through training vs test error analysis.
    
    Educational Purpose:
    - Shows how training error decreases monotonically with more predictors
    - Demonstrates that test error may not follow the same pattern
    - Illustrates the concept of overfitting
    - Provides visual evidence of the bias-variance tradeoff
    
    Args:
        data (pandas.DataFrame): Complete dataset
        n_simulations (int): Number of random splits to demonstrate variability
    """
    
    print("\n" + "="*60)
    print("TRAINING VS TEST ERROR ANALYSIS")
    print("="*60)
    print("Demonstrating the Bias-Variance Tradeoff")
    print("="*60)
    
    n, p = data.shape
    p_predictors = p - 1  # Number of predictors (excluding response)
    
    print(f"Sample size: {n}")
    print(f"Number of predictors: {p_predictors}")
    print(f"Training proportion: 60%")
    print(f"Test proportion: 40%")
    
    # Create figure for multiple simulations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for sim in range(min(n_simulations, 6)):
        # Set different random seeds for each simulation
        np.random.seed(20 + sim)
        
        # Split data
        X = data.iloc[:, :-1]  # Predictors
        y = data.iloc[:, -1]   # Response
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.6, random_state=20 + sim
        )
        
        # Arrays to store errors
        train_MSE = []
        test_MSE = []
        predictor_counts = []
        
        # Fit models with progressively more predictors
        for i in range(1, p_predictors + 1):
            # Fit model with i predictors
            model = LinearRegression()
            model.fit(X_train.iloc[:, :i], y_train)
            
            # Training predictions and MSE
            train_pred = model.predict(X_train.iloc[:, :i])
            train_mse = np.mean((y_train - train_pred)**2)
            train_MSE.append(train_mse)
            
            # Test predictions and MSE
            test_pred = model.predict(X_test.iloc[:, :i])
            test_mse = np.mean((y_test - test_pred)**2)
            test_MSE.append(test_mse)
            
            predictor_counts.append(i)
        
        # Plot results
        ax = axes[sim]
        ax.plot(predictor_counts, train_MSE, 'b-o', label='Training Error', linewidth=2, markersize=6)
        ax.plot(predictor_counts, test_MSE, 'r-s', label='Test Error', linewidth=2, markersize=6)
        ax.set_xlabel('Number of Predictors')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title(f'Simulation {sim + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some insights
        if sim == 0:
            print(f"\nSimulation {sim + 1} Results:")
            print(f"  Training error always decreases: {all(np.diff(train_MSE) <= 0)}")
            print(f"  Test error sometimes increases: {any(np.diff(test_MSE) > 0)}")
            print(f"  Minimum test error at {predictor_counts[np.argmin(test_MSE)]} predictors")
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis of one simulation
    print(f"\n" + "="*50)
    print("DETAILED ANALYSIS OF SIMULATION 1")
    print("="*50)
    
    np.random.seed(20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=20
    )
    
    train_MSE = []
    test_MSE = []
    
    for i in range(1, p_predictors + 1):
        model = LinearRegression()
        model.fit(X_train.iloc[:, :i], y_train)
        
        train_pred = model.predict(X_train.iloc[:, :i])
        train_mse = np.mean((y_train - train_pred)**2)
        train_MSE.append(train_mse)
        
        test_pred = model.predict(X_test.iloc[:, :i])
        test_mse = np.mean((y_test - test_pred)**2)
        test_MSE.append(test_mse)
    
    # Create comparison table
    print(f"{'Predictors':<12} {'Train MSE':<15} {'Test MSE':<15} {'Difference':<15}")
    print("-" * 60)
    for i, (train_mse, test_mse) in enumerate(zip(train_MSE, test_MSE)):
        diff = test_mse - train_mse
        print(f"{i+1:<12} {train_mse:<15.6f} {test_mse:<15.6f} {diff:<15.6f}")
    
    # Analyze differences between adjacent terms
    train_diff = np.diff(train_MSE)
    test_diff = np.diff(test_MSE)
    
    print(f"\nAnalysis of Changes:")
    print(f"Training error changes (all should be negative):")
    print(f"  All negative: {all(train_diff <= 0)}")
    print(f"  Mean change: {np.mean(train_diff):.6f}")
    
    print(f"\nTest error changes (may be positive):")
    print(f"  Any positive: {any(test_diff > 0)}")
    print(f"  Mean change: {np.mean(test_diff):.6f}")
    print(f"  Number of increases: {np.sum(test_diff > 0)}")
    
    print(f"\nKey Insights:")
    print(f"• Training error always decreases (monotonic)")
    print(f"• Test error may increase, indicating overfitting")
    print(f"• Optimal model complexity balances bias and variance")
    print(f"• Cross-validation helps identify optimal complexity")

# =============================================================================
# Coefficient Interpretation Analysis
# =============================================================================

def analyze_coefficient_interpretation(data):
    """
    Demonstrate how coefficient interpretation changes between simple and multiple regression.
    
    Educational Purpose:
    - Shows the difference between simple and multiple linear regression
    - Demonstrates the impact of multicollinearity on coefficient signs
    - Illustrates why coefficients can change sign when adding predictors
    - Provides correlation analysis to understand relationships
    
    Args:
        data (pandas.DataFrame): Complete dataset
    """
    
    print("\n" + "="*60)
    print("COEFFICIENT INTERPRETATION ANALYSIS")
    print("="*60)
    print("Simple vs Multiple Linear Regression")
    print("="*60)
    
    # Simple Linear Regression with age only
    print("1. SIMPLE LINEAR REGRESSION (Age only)")
    print("-" * 40)
    
    simple_model = smf.ols("Y ~ age", data=data).fit()
    print(f"Age coefficient: {simple_model.params['age']:.6f}")
    print(f"P-value: {simple_model.pvalues['age']:.6f}")
    print(f"R-squared: {simple_model.rsquared:.6f}")
    
    # Multiple Linear Regression with all predictors
    print(f"\n2. MULTIPLE LINEAR REGRESSION (All predictors)")
    print("-" * 40)
    
    predictors = list(data.columns[:-1])  # All except response
    formula = "Y ~ " + " + ".join(predictors)
    multiple_model = smf.ols(formula, data=data).fit()
    
    print(f"Age coefficient: {multiple_model.params['age']:.6f}")
    print(f"P-value: {multiple_model.pvalues['age']:.6f}")
    print(f"R-squared: {multiple_model.rsquared:.6f}")
    
    # Compare coefficients
    print(f"\n3. COEFFICIENT COMPARISON")
    print("-" * 40)
    print(f"Simple regression age coefficient:  {simple_model.params['age']:.6f}")
    print(f"Multiple regression age coefficient: {multiple_model.params['age']:.6f}")
    print(f"Difference: {multiple_model.params['age'] - simple_model.params['age']:.6f}")
    
    # Correlation analysis
    print(f"\n4. CORRELATION ANALYSIS")
    print("-" * 40)
    correlation_matrix = data.corr()
    age_correlations = correlation_matrix['age'].sort_values(ascending=False)
    
    print("Correlations with age:")
    for var, corr in age_correlations.items():
        if var != 'age':
            print(f"  {var:>10}: {corr:>8.3f}")
    
    # Visualize correlations
    plt.figure(figsize=(12, 5))
    
    # Correlation heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Age vs Y scatter plots
    plt.subplot(1, 2, 2)
    plt.scatter(data['age'], data['Y'], alpha=0.6, s=50)
    
    # Add regression lines
    x_range = np.linspace(data['age'].min(), data['age'].max(), 100)
    
    # Simple regression line
    simple_line = simple_model.params['Intercept'] + simple_model.params['age'] * x_range
    plt.plot(x_range, simple_line, 'r-', linewidth=2, label='Simple Regression')
    
    # Multiple regression line (holding other variables at mean)
    other_means = data[predictors].mean()
    multiple_line = (multiple_model.params['Intercept'] + 
                    multiple_model.params['age'] * x_range +
                    sum(multiple_model.params[var] * other_means[var] 
                        for var in predictors if var != 'age'))
    plt.plot(x_range, multiple_line, 'b--', linewidth=2, label='Multiple Regression')
    
    plt.xlabel('Age')
    plt.ylabel('Y (lpsa)')
    plt.title('Age vs Y with Regression Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n5. INTERPRETATION")
    print("-" * 40)
    print(f"• Simple regression: Age has a {simple_model.params['age']:.3f} effect on Y")
    print(f"• Multiple regression: Age has a {multiple_model.params['age']:.3f} effect on Y")
    print(f"• The difference occurs because:")
    print(f"  - In simple regression, age captures both direct and indirect effects")
    print(f"  - In multiple regression, age captures only the direct effect")
    print(f"  - Other variables may be correlated with age and Y")
    print(f"• This is why multiple regression coefficients are called 'partial effects'")

# =============================================================================
# Partial Regression Coefficient (Frisch-Waugh-Lovell Theorem)
# =============================================================================

def demonstrate_partial_regression(data):
    """
    Demonstrate the Frisch-Waugh-Lovell theorem for partial regression coefficients.
    
    Educational Purpose:
    - Shows how to isolate the effect of one predictor
    - Demonstrates the mathematical foundation of partial effects
    - Illustrates why multiple regression coefficients are called "partial"
    - Provides step-by-step implementation of the theorem
    
    Args:
        data (pandas.DataFrame): Complete dataset
    """
    
    print("\n" + "="*60)
    print("PARTIAL REGRESSION COEFFICIENT ANALYSIS")
    print("="*60)
    print("Frisch-Waugh-Lovell Theorem Implementation")
    print("="*60)
    
    # Target variable and predictor of interest
    target_var = 'Y'
    focus_var = 'age'
    
    print(f"Demonstrating partial effect of '{focus_var}' on '{target_var}'")
    print(f"Using the Frisch-Waugh-Lovell theorem")
    
    # Step 1: Regress Y on all predictors except the focus variable
    print(f"\nStep 1: Regress {target_var} on all predictors except {focus_var}")
    print("-" * 60)
    
    other_predictors = [col for col in data.columns if col not in [target_var, focus_var]]
    X_step1 = data[other_predictors]
    y_step1 = data[target_var]
    
    model_step1 = LinearRegression()
    model_step1.fit(X_step1, y_step1)
    y_pred_step1 = model_step1.predict(X_step1)
    
    # Calculate residuals (y*)
    y_star = y_step1 - y_pred_step1
    
    print(f"Residuals from Step 1 (y*):")
    print(f"  Mean: {np.mean(y_star):.6f}")
    print(f"  Std: {np.std(y_star):.6f}")
    print(f"  Range: [{np.min(y_star):.6f}, {np.max(y_star):.6f}]")
    
    # Step 2: Regress the focus variable on all other predictors
    print(f"\nStep 2: Regress {focus_var} on all other predictors")
    print("-" * 60)
    
    X_step2 = data[other_predictors]
    y_step2 = data[focus_var]
    
    model_step2 = LinearRegression()
    model_step2.fit(X_step2, y_step2)
    y_pred_step2 = model_step2.predict(X_step2)
    
    # Calculate residuals (x*)
    x_star = y_step2 - y_pred_step2
    
    print(f"Residuals from Step 2 (x*):")
    print(f"  Mean: {np.mean(x_star):.6f}")
    print(f"  Std: {np.std(x_star):.6f}")
    print(f"  Range: [{np.min(x_star):.6f}, {np.max(x_star):.6f}]")
    
    # Step 3: Regress y* on x*
    print(f"\nStep 3: Regress y* on x*")
    print("-" * 60)
    
    model_step3 = LinearRegression()
    model_step3.fit(x_star.values.reshape(-1, 1), y_star)
    partial_coefficient = model_step3.coef_[0]
    
    print(f"Partial regression coefficient: {partial_coefficient:.8f}")
    
    # Compare with full model
    print(f"\nComparison with Full Model")
    print("-" * 60)
    
    # Fit full model using statsmodels for easy coefficient extraction
    formula = f"{target_var} ~ " + " + ".join([focus_var] + other_predictors)
    full_model = smf.ols(formula, data=data).fit()
    full_coefficient = full_model.params[focus_var]
    
    print(f"Partial regression coefficient: {partial_coefficient:.8f}")
    print(f"Full model coefficient:        {full_coefficient:.8f}")
    print(f"Difference:                    {abs(partial_coefficient - full_coefficient):.8f}")
    print(f"Are they equal?                {np.isclose(partial_coefficient, full_coefficient)}")
    
    # Verify residuals are the same
    print(f"\nResidual Comparison")
    print("-" * 60)
    
    full_residuals = data[target_var] - full_model.predict(data)
    partial_residuals = y_star - model_step3.predict(x_star.values.reshape(-1, 1))
    
    print(f"Full model residuals sum of squares: {np.sum(full_residuals**2):.8f}")
    print(f"Partial model residuals sum of squares: {np.sum(partial_residuals**2):.8f}")
    print(f"Difference: {abs(np.sum(full_residuals**2) - np.sum(partial_residuals**2)):.8f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original relationship
    plt.subplot(1, 3, 1)
    plt.scatter(data[focus_var], data[target_var], alpha=0.6)
    plt.xlabel(focus_var)
    plt.ylabel(target_var)
    plt.title(f'Original: {focus_var} vs {target_var}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals from step 1 vs original focus variable
    plt.subplot(1, 3, 2)
    plt.scatter(data[focus_var], y_star, alpha=0.6)
    plt.xlabel(focus_var)
    plt.ylabel(f'{target_var} residuals (y*)')
    plt.title(f'Step 1: {focus_var} vs {target_var} residuals')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Partial regression (y* vs x*)
    plt.subplot(1, 3, 3)
    plt.scatter(x_star, y_star, alpha=0.6)
    
    # Add regression line
    x_range = np.linspace(x_star.min(), x_star.max(), 100)
    y_pred_partial = partial_coefficient * x_range
    plt.plot(x_range, y_pred_partial, 'r-', linewidth=2)
    
    plt.xlabel(f'{focus_var} residuals (x*)')
    plt.ylabel(f'{target_var} residuals (y*)')
    plt.title(f'Partial Regression: y* vs x*')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nInterpretation:")
    print(f"• The partial coefficient isolates the effect of {focus_var}")
    print(f"• It removes the influence of other variables")
    print(f"• This is why multiple regression coefficients are 'partial effects'")
    print(f"• The theorem shows that partial and full model coefficients are identical")

# =============================================================================
# Hypothesis Testing (F-test and t-test)
# =============================================================================

def perform_hypothesis_testing(data):
    """
    Perform hypothesis testing for individual predictors using F-tests and t-tests.
    
    Educational Purpose:
    - Shows manual calculation of F-statistics and t-statistics
    - Demonstrates the relationship between F-test and t-test
    - Compares manual calculations with statistical package results
    - Illustrates hypothesis testing concepts
    
    Args:
        data (pandas.DataFrame): Complete dataset
    """
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING ANALYSIS")
    print("="*60)
    print("F-test and t-test for Individual Predictors")
    print("="*60)
    
    # Focus on testing the 'age' variable
    test_var = 'age'
    print(f"Testing the significance of '{test_var}' predictor")
    
    # Prepare data
    X = data.iloc[:, :-1]  # All predictors
    y = data.iloc[:, -1]   # Response
    X_without_age = X.drop(test_var, axis=1)  # Predictors without age
    
    # Fit models
    print(f"\n1. MODEL FITTING")
    print("-" * 40)
    
    # Full model (with age)
    full_model = LinearRegression()
    full_model.fit(X, y)
    full_pred = full_model.predict(X)
    
    # Reduced model (without age)
    reduced_model = LinearRegression()
    reduced_model.fit(X_without_age, y)
    reduced_pred = reduced_model.predict(X_without_age)
    
    print(f"Full model R²: {full_model.score(X, y):.6f}")
    print(f"Reduced model R²: {reduced_model.score(X_without_age, y):.6f}")
    
    # Calculate test statistics manually
    print(f"\n2. MANUAL CALCULATION")
    print("-" * 40)
    
    n = len(y)
    p_full = X.shape[1]      # Number of predictors in full model
    p_reduced = X_without_age.shape[1]  # Number of predictors in reduced model
    
    # Calculate sums of squares
    SS_res_full = np.sum((y - full_pred)**2)
    SS_res_reduced = np.sum((y - reduced_pred)**2)
    SS_reg = SS_res_reduced - SS_res_full  # Extra sum of squares due to age
    
    # Calculate F-statistic
    df_reg = p_full - p_reduced  # Degrees of freedom for regression (should be 1)
    df_res = n - p_full - 1      # Degrees of freedom for residuals
    
    MS_reg = SS_reg / df_reg
    MS_res = SS_res_full / df_res
    F_stat = MS_reg / MS_res
    
    print(f"Extra SS due to {test_var}: {SS_reg:.6f}")
    print(f"Residual SS (full model): {SS_res_full:.6f}")
    print(f"Degrees of freedom (regression): {df_reg}")
    print(f"Degrees of freedom (residuals): {df_res}")
    print(f"F-statistic: {F_stat:.6f}")
    
    # Calculate p-value
    p_value_f = 1 - stats.f.cdf(F_stat, df_reg, df_res)
    print(f"P-value (F-test): {p_value_f:.6f}")
    
    # Calculate t-statistic (for single variable, F = t²)
    t_stat = np.sqrt(F_stat)
    print(f"t-statistic: {t_stat:.6f}")
    
    # Two-sided t-test p-value
    p_value_t = 2 * (1 - stats.t.cdf(abs(t_stat), df_res))
    print(f"P-value (t-test): {p_value_t:.6f}")
    
    # Compare with statsmodels results
    print(f"\n3. STATSMODELS COMPARISON")
    print("-" * 40)
    
    formula = f"Y ~ " + " + ".join(X.columns)
    statsmodels_model = smf.ols(formula, data=data).fit()
    
    # Extract results for the test variable
    age_results = statsmodels_model.summary().tables[1]
    age_row = age_results.data[1]  # Assuming age is the first predictor
    
    print(f"Statsmodels results for {test_var}:")
    print(f"  Coefficient: {statsmodels_model.params[test_var]:.6f}")
    print(f"  t-statistic: {statsmodels_model.tvalues[test_var]:.6f}")
    print(f"  P-value: {statsmodels_model.pvalues[test_var]:.6f}")
    
    # Compare manual vs statsmodels
    print(f"\n4. COMPARISON")
    print("-" * 40)
    print(f"{'Statistic':<15} {'Manual':<15} {'Statsmodels':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'t-statistic':<15} {t_stat:<15.6f} {statsmodels_model.tvalues[test_var]:<15.6f} {abs(t_stat - statsmodels_model.tvalues[test_var]):<15.6f}")
    print(f"{'P-value':<15} {p_value_t:<15.6f} {statsmodels_model.pvalues[test_var]:<15.6f} {abs(p_value_t - statsmodels_model.pvalues[test_var]):<15.6f}")
    
    # Test multiple variables
    print(f"\n5. TESTING MULTIPLE VARIABLES")
    print("-" * 40)
    
    # Test removing first 3 variables
    vars_to_remove = X.columns[:3].tolist()
    X_reduced = X.drop(vars_to_remove, axis=1)
    
    reduced_model_multi = LinearRegression()
    reduced_model_multi.fit(X_reduced, y)
    reduced_pred_multi = reduced_model_multi.predict(X_reduced)
    
    SS_res_reduced_multi = np.sum((y - reduced_pred_multi)**2)
    SS_reg_multi = SS_res_reduced_multi - SS_res_full
    
    df_reg_multi = len(vars_to_remove)
    MS_reg_multi = SS_reg_multi / df_reg_multi
    F_stat_multi = MS_reg_multi / MS_res
    
    p_value_f_multi = 1 - stats.f.cdf(F_stat_multi, df_reg_multi, df_res)
    
    print(f"Testing removal of variables: {vars_to_remove}")
    print(f"F-statistic: {F_stat_multi:.6f}")
    print(f"P-value: {p_value_f_multi:.6f}")
    print(f"Degrees of freedom: ({df_reg_multi}, {df_res})")
    
    print(f"\nInterpretation:")
    print(f"• F-test and t-test give identical results for single variables")
    print(f"• F = t² for single variable tests")
    print(f"• Manual calculations match statistical package results")
    print(f"• Multiple variable tests use F-distribution with multiple df")

# =============================================================================
# Collinearity Analysis
# =============================================================================

def analyze_collinearity():
    """
    Analyze collinearity using the Car Seat Position dataset.
    
    Educational Purpose:
    - Demonstrates the effects of high correlations among predictors
    - Shows how collinearity affects coefficient significance
    - Illustrates the difference between high R² and significant coefficients
    - Provides methods to detect and handle collinearity
    
    Returns:
        pandas.DataFrame: Seat position dataset
    """
    
    print("\n" + "="*60)
    print("COLLINEARITY ANALYSIS")
    print("="*60)
    print("Car Seat Position Dataset Analysis")
    print("="*60)
    
    # Load seat position data
    url = "https://liangfgithub.github.io/Data/SeatPos.csv"
    seatpos = pd.read_csv(url)
    
    print(f"Dataset shape: {seatpos.shape}")
    print(f"Variables: {list(seatpos.columns)}")
    
    # Variable descriptions
    var_descriptions = {
        'Age': 'Driver age in years',
        'Weight': 'Driver weight in kg',
        'HtShoes': 'Height with shoes in cm',
        'Ht': 'Height without shoes in cm',
        'Seated': 'Seated height in cm',
        'Arm': 'Lower arm length in cm',
        'Thigh': 'Thigh length in cm',
        'Leg': 'Lower leg length in cm',
        'hipcenter': 'Horizontal distance of hip center from fixed location (mm)'
    }
    
    print(f"\nVariable descriptions:")
    for var, desc in var_descriptions.items():
        print(f"  {var}: {desc}")
    
    # Data exploration
    print(f"\n1. DATA EXPLORATION")
    print("-" * 40)
    print(f"Summary statistics:")
    print(seatpos.describe().round(2))
    
    # Correlation analysis
    print(f"\n2. CORRELATION ANALYSIS")
    print("-" * 40)
    correlation_matrix = seatpos.corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
    
    print(f"High correlations (|r| > 0.8):")
    for var1, var2, corr in high_corr_pairs:
        print(f"  {var1} - {var2}: {corr:.3f}")
    
    # Visualize correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix - Car Seat Position Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Fit full model
    print(f"\n3. FULL MODEL ANALYSIS")
    print("-" * 40)
    
    predictors = [col for col in seatpos.columns if col != 'hipcenter']
    formula = "hipcenter ~ " + " + ".join(predictors)
    full_model = smf.ols(formula, data=seatpos).fit()
    
    print(f"Full model results:")
    print(f"  R-squared: {full_model.rsquared:.6f}")
    print(f"  Adjusted R-squared: {full_model.rsquared_adj:.6f}")
    print(f"  F-statistic: {full_model.fvalue:.6f}")
    print(f"  F-test p-value: {full_model.f_pvalue:.6f}")
    
    # Check individual coefficient significance
    print(f"\nIndividual coefficient significance:")
    print(f"{'Variable':<12} {'Coefficient':<15} {'t-statistic':<15} {'P-value':<15}")
    print("-" * 60)
    
    significant_vars = []
    for var in predictors:
        coef = full_model.params[var]
        t_stat = full_model.tvalues[var]
        p_val = full_model.pvalues[var]
        significance = "Significant" if p_val < 0.05 else "Not significant"
        
        print(f"{var:<12} {coef:<15.6f} {t_stat:<15.6f} {p_val:<15.6f}")
        
        if p_val < 0.05:
            significant_vars.append(var)
    
    print(f"\nSignificant variables: {significant_vars}")
    print(f"Number of significant variables: {len(significant_vars)} out of {len(predictors)}")
    
    # Try reduced models
    print(f"\n4. REDUCED MODEL ANALYSIS")
    print("-" * 40)
    
    # Model with only height
    height_model = smf.ols("hipcenter ~ Ht", data=seatpos).fit()
    
    # Model with height and weight
    height_weight_model = smf.ols("hipcenter ~ Ht + Weight", data=seatpos).fit()
    
    # Model with height, weight, and age
    height_weight_age_model = smf.ols("hipcenter ~ Ht + Weight + Age", data=seatpos).fit()
    
    # Compare models
    models = {
        'Height only': height_model,
        'Height + Weight': height_weight_model,
        'Height + Weight + Age': height_weight_age_model,
        'All variables': full_model
    }
    
    print(f"{'Model':<25} {'R²':<10} {'Adj R²':<10} {'F-stat':<10} {'P-value':<10}")
    print("-" * 70)
    
    for name, model in models.items():
        print(f"{name:<25} {model.rsquared:<10.4f} {model.rsquared_adj:<10.4f} "
              f"{model.fvalue:<10.4f} {model.f_pvalue:<10.4f}")
    
    # Condition number analysis
    print(f"\n5. CONDITION NUMBER ANALYSIS")
    print("-" * 40)
    
    X = seatpos[predictors]
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Calculate condition number
    eigenvals = np.linalg.eigvals(X_with_intercept.T @ X_with_intercept)
    condition_number = np.sqrt(np.max(eigenvals) / np.min(eigenvals))
    
    print(f"Condition number: {condition_number:.2f}")
    print(f"Log10(condition number): {np.log10(condition_number):.2f}")
    
    if condition_number > 30:
        print(f"⚠️  High condition number indicates multicollinearity")
    else:
        print(f"✓ Condition number is acceptable")
    
    # VIF calculation (simplified)
    print(f"\n6. VARIANCE INFLATION FACTORS (VIF)")
    print("-" * 40)
    
    vif_values = {}
    for var in predictors:
        # Regress this variable on all others
        other_vars = [v for v in predictors if v != var]
        if other_vars:
            vif_model = smf.ols(f"{var} ~ " + " + ".join(other_vars), data=seatpos).fit()
            vif = 1 / (1 - vif_model.rsquared)
            vif_values[var] = vif
    
    print(f"{'Variable':<12} {'VIF':<10}")
    print("-" * 25)
    for var, vif in vif_values.items():
        status = "High" if vif > 10 else "Acceptable"
        print(f"{var:<12} {vif:<10.2f} ({status})")
    
    print(f"\nKey Insights:")
    print(f"• High R² but few significant coefficients indicates multicollinearity")
    print(f"• Condition number > 30 suggests numerical problems")
    print(f"• VIF > 10 indicates problematic multicollinearity")
    print(f"• Removing correlated variables often improves interpretability")
    
    return seatpos

# =============================================================================
# Main Analysis Function
# =============================================================================

def main():
    """
    Main function that orchestrates the entire advanced linear regression analysis.
    """
    
    print("ADVANCED LINEAR REGRESSION ANALYSIS IN PYTHON (PART II)")
    print("="*70)
    print("Demonstrating Advanced Concepts in Linear Regression")
    print("="*70)
    
    # Step 1: Load and prepare data
    data = load_and_prepare_data()
    
    # Step 2: Training vs Test Error Analysis
    analyze_training_vs_test_error(data, n_simulations=6)
    
    # Step 3: Coefficient Interpretation
    analyze_coefficient_interpretation(data)
    
    # Step 4: Partial Regression Analysis
    demonstrate_partial_regression(data)
    
    # Step 5: Hypothesis Testing
    perform_hypothesis_testing(data)
    
    # Step 6: Collinearity Analysis
    seatpos_data = analyze_collinearity()
    
    # Step 7: Summary and Conclusions
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY AND CONCLUSIONS")
    print("="*70)
    
    print(f"Key Concepts Demonstrated:")
    print(f"1. Bias-Variance Tradeoff")
    print(f"   • Training error decreases with more predictors")
    print(f"   • Test error may increase, indicating overfitting")
    print(f"   • Optimal model complexity balances bias and variance")
    print()
    print(f"2. Coefficient Interpretation")
    print(f"   • Simple vs Multiple Linear Regression differences")
    print(f"   • Partial effects vs total effects")
    print(f"   • Impact of multicollinearity on coefficient signs")
    print()
    print(f"3. Partial Regression Coefficients")
    print(f"   • Frisch-Waugh-Lovell theorem implementation")
    print(f"   • Isolating effects of individual predictors")
    print(f"   • Mathematical foundation of partial effects")
    print()
    print(f"4. Hypothesis Testing")
    print(f"   • F-test and t-test for individual predictors")
    print(f"   • Manual calculations vs statistical packages")
    print(f"   • Testing multiple variables simultaneously")
    print()
    print(f"5. Multicollinearity")
    print(f"   • Detection methods (correlation, VIF, condition number)")
    print(f"   • Effects on coefficient interpretation")
    print(f"   • Strategies for handling collinearity")
    print()
    print(f"Practical Applications:")
    print(f"• Use cross-validation to find optimal model complexity")
    print(f"• Interpret coefficients as partial effects")
    print(f"• Check for multicollinearity before interpreting coefficients")
    print(f"• Consider variable selection or regularization techniques")
    print("="*70)

# =============================================================================
# Execute the analysis
# =============================================================================

if __name__ == "__main__":
    main() 