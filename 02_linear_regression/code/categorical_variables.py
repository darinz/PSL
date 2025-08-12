"""
Categorical Variables in Linear Regression
=========================================

This module demonstrates how to handle categorical variables in linear regression,
including one-hot encoding, interaction terms, and different encoding strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats

def demonstrate_categorical_variables():
    """Demonstrate categorical variables handling in linear regression"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data with categorical variable
    n = 300
    sizes = np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.4, 0.35, 0.25])
    x_continuous = np.random.normal(0, 1, n)
    
    # True model with different effects by size
    # Small: baseline effect
    # Medium: +2 units higher intercept, +0.5 additional slope
    # Large: +4 units higher intercept, +1.0 additional slope
    
    y = np.zeros(n)
    for i, size in enumerate(sizes):
        if size == 'Small':
            y[i] = 5 + 1.5 * x_continuous[i] + np.random.normal(0, 0.5)
        elif size == 'Medium':
            y[i] = 7 + 2.0 * x_continuous[i] + np.random.normal(0, 0.5)
        else:  # Large
            y[i] = 9 + 2.5 * x_continuous[i] + np.random.normal(0, 0.5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Size': sizes,
        'X': x_continuous,
        'Y': y
    })
    
    print("=== TRUE MODEL ===")
    print("Small:  Y = 5 + 1.5*X + ε")
    print("Medium: Y = 7 + 2.0*X + ε")
    print("Large:  Y = 9 + 2.5*X + ε")
    
    return df

def manual_one_hot_encoding(df):
    """Demonstrate manual one-hot encoding"""
    
    print("\n=== METHOD 1: MANUAL ONE-HOT ENCODING ===")
    
    # Create dummy variables manually
    df['Size_Medium'] = (df['Size'] == 'Medium').astype(int)
    df['Size_Large'] = (df['Size'] == 'Large').astype(int)
    
    # Fit model with main effects only
    X_main = df[['X', 'Size_Medium', 'Size_Large']].values
    model_main = LinearRegression()
    model_main.fit(X_main, df['Y'])
    
    print("Main effects model:")
    print(f"Intercept: {model_main.intercept_:.3f}")
    print(f"X coefficient: {model_main.coef_[0]:.3f}")
    print(f"Medium effect: {model_main.coef_[1]:.3f}")
    print(f"Large effect: {model_main.coef_[2]:.3f}")
    print(f"R²: {r2_score(df['Y'], model_main.predict(X_main)):.3f}")
    
    return X_main, model_main

def sklearn_one_hot_encoding(df):
    """Demonstrate sklearn OneHotEncoder"""
    
    print("\n=== METHOD 2: SKLEARN ONEHOTENCODER ===")
    
    encoder = OneHotEncoder(drop='first', sparse=False)
    size_encoded = encoder.fit_transform(df[['Size']])
    feature_names = encoder.get_feature_names_out(['Size'])
    
    X_sklearn = np.column_stack([df['X'].values, size_encoded])
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_sklearn, df['Y'])
    
    print("sklearn encoding results:")
    print(f"Intercept: {model_sklearn.intercept_:.3f}")
    print(f"X coefficient: {model_sklearn.coef_[0]:.3f}")
    for i, name in enumerate(feature_names):
        print(f"{name}: {model_sklearn.coef_[i+1]:.3f}")
    print(f"R²: {r2_score(df['Y'], model_sklearn.predict(X_sklearn)):.3f}")
    
    return X_sklearn, model_sklearn, feature_names

def interaction_model(df):
    """Demonstrate interaction model with categorical variables"""
    
    print("\n=== METHOD 3: INTERACTION MODEL ===")
    
    # Create interaction terms
    df['X_Medium'] = df['X'] * df['Size_Medium']
    df['X_Large'] = df['X'] * df['Size_Large']
    
    X_interaction = df[['X', 'Size_Medium', 'Size_Large', 'X_Medium', 'X_Large']].values
    model_interaction = LinearRegression()
    model_interaction.fit(X_interaction, df['Y'])
    
    print("Interaction model:")
    print(f"Intercept: {model_interaction.intercept_:.3f}")
    print(f"X coefficient (baseline): {model_interaction.coef_[0]:.3f}")
    print(f"Medium intercept effect: {model_interaction.coef_[1]:.3f}")
    print(f"Large intercept effect: {model_interaction.coef_[2]:.3f}")
    print(f"Medium slope effect: {model_interaction.coef_[3]:.3f}")
    print(f"Large slope effect: {model_interaction.coef_[4]:.3f}")
    print(f"R²: {r2_score(df['Y'], model_interaction.predict(X_interaction)):.3f}")
    
    return X_interaction, model_interaction

def compare_models(df, X_main, model_main, X_sklearn, model_sklearn, X_interaction, model_interaction):
    """Compare different models"""
    
    print("\n=== MODEL COMPARISON ===")
    models = {
        'Main Effects': model_main,
        'sklearn Encoding': model_sklearn,
        'Interaction': model_interaction
    }
    
    X_matrices = {
        'Main Effects': X_main,
        'sklearn Encoding': X_sklearn,
        'Interaction': X_interaction
    }
    
    comparison_df = pd.DataFrame({
        'Model': list(models.keys()),
        'R²': [r2_score(df['Y'], models[name].predict(X_matrices[name])) for name in models.keys()],
        'Parameters': [len(models[name].coef_) + 1 for name in models.keys()]
    })
    
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def visualize_categorical_models(df, X_main, model_main, X_interaction, model_interaction):
    """Visualize categorical variable models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data by size
    colors = {'Small': 'blue', 'Medium': 'green', 'Large': 'red'}
    for size in ['Small', 'Medium', 'Large']:
        mask = df['Size'] == size
        axes[0, 0].scatter(df[mask]['X'], df[mask]['Y'], 
                          c=colors[size], label=size, alpha=0.6)
    
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Original Data by Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Main effects model predictions
    for size in ['Small', 'Medium', 'Large']:
        mask = df['Size'] == size
        x_vals = df[mask]['X'].values
        if size == 'Small':
            y_pred = model_main.intercept_ + model_main.coef_[0] * x_vals
        elif size == 'Medium':
            y_pred = model_main.intercept_ + model_main.coef_[0] * x_vals + model_main.coef_[1]
        else:  # Large
            y_pred = model_main.intercept_ + model_main.coef_[0] * x_vals + model_main.coef_[2]
        
        axes[0, 1].plot(x_vals, y_pred, c=colors[size], linewidth=2, label=f'{size} (Main)')
    
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Predicted Y')
    axes[0, 1].set_title('Main Effects Model')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Interaction model predictions
    for size in ['Small', 'Medium', 'Large']:
        mask = df['Size'] == size
        x_vals = df[mask]['X'].values
        if size == 'Small':
            y_pred = model_interaction.intercept_ + model_interaction.coef_[0] * x_vals
        elif size == 'Medium':
            y_pred = (model_interaction.intercept_ + model_interaction.coef_[1] + 
                     (model_interaction.coef_[0] + model_interaction.coef_[3]) * x_vals)
        else:  # Large
            y_pred = (model_interaction.intercept_ + model_interaction.coef_[2] + 
                     (model_interaction.coef_[0] + model_interaction.coef_[4]) * x_vals)
        
        axes[1, 0].plot(x_vals, y_pred, c=colors[size], linewidth=2, label=f'{size} (Interaction)')
    
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Predicted Y')
    axes[1, 0].set_title('Interaction Model')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals comparison
    residuals_main = df['Y'] - model_main.predict(X_main)
    residuals_interaction = df['Y'] - model_interaction.predict(X_interaction)
    
    axes[1, 1].scatter(model_main.predict(X_main), residuals_main, 
                      alpha=0.6, label='Main Effects', color='blue')
    axes[1, 1].scatter(model_interaction.predict(X_interaction), residuals_interaction, 
                      alpha=0.6, label='Interaction', color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_interaction_significance(X_main, model_main, X_interaction, model_interaction, df):
    """Test significance of interaction terms"""
    
    print("\n=== INTERACTION SIGNIFICANCE TEST ===")
    
    # F-test for interaction terms
    residuals_main = df['Y'] - model_main.predict(X_main)
    residuals_interaction = df['Y'] - model_interaction.predict(X_interaction)
    
    rss_main = np.sum(residuals_main**2)
    rss_interaction = np.sum(residuals_interaction**2)
    f_stat = ((rss_main - rss_interaction) / 2) / (rss_interaction / (len(df) - 6))
    f_p_value = 1 - stats.f.cdf(f_stat, 2, len(df) - 6)
    
    print(f"F-statistic for interaction: {f_stat:.3f}")
    print(f"p-value: {f_p_value:.3f}")
    print(f"Interaction significant: {'Yes' if f_p_value < 0.05 else 'No'}")

def alternative_encoding_methods(df):
    """Demonstrate alternative encoding methods"""
    
    print("\n=== ALTERNATIVE ENCODING METHODS ===")
    
    # 1. Ordinal encoding
    print("1. Ordinal Encoding:")
    size_ordinal = {'Small': 1, 'Medium': 2, 'Large': 3}
    df['Size_Ordinal'] = df['Size'].map(size_ordinal)
    
    X_ordinal = df[['X', 'Size_Ordinal']].values
    model_ordinal = LinearRegression()
    model_ordinal.fit(X_ordinal, df['Y'])
    
    print(f"  Intercept: {model_ordinal.intercept_:.3f}")
    print(f"  X coefficient: {model_ordinal.coef_[0]:.3f}")
    print(f"  Size coefficient: {model_ordinal.coef_[1]:.3f}")
    print(f"  R²: {r2_score(df['Y'], model_ordinal.predict(X_ordinal)):.3f}")
    
    # 2. Frequency encoding
    print("\n2. Frequency Encoding:")
    size_freq = df['Size'].value_counts(normalize=True)
    df['Size_Freq'] = df['Size'].map(size_freq)
    
    X_freq = df[['X', 'Size_Freq']].values
    model_freq = LinearRegression()
    model_freq.fit(X_freq, df['Y'])
    
    print(f"  Intercept: {model_freq.intercept_:.3f}")
    print(f"  X coefficient: {model_freq.coef_[0]:.3f}")
    print(f"  Size frequency coefficient: {model_freq.coef_[1]:.3f}")
    print(f"  R²: {r2_score(df['Y'], model_freq.predict(X_freq)):.3f}")
    
    return X_ordinal, model_ordinal, X_freq, model_freq

if __name__ == "__main__":
    # Generate data
    df = demonstrate_categorical_variables()
    
    # Manual one-hot encoding
    X_main, model_main = manual_one_hot_encoding(df)
    
    # Sklearn one-hot encoding
    X_sklearn, model_sklearn, feature_names = sklearn_one_hot_encoding(df)
    
    # Interaction model
    X_interaction, model_interaction = interaction_model(df)
    
    # Compare models
    comparison_df = compare_models(df, X_main, model_main, X_sklearn, model_sklearn, X_interaction, model_interaction)
    
    # Visualize results
    visualize_categorical_models(df, X_main, model_main, X_interaction, model_interaction)
    
    # Test interaction significance
    test_interaction_significance(X_main, model_main, X_interaction, model_interaction, df)
    
    # Alternative encoding methods
    X_ordinal, model_ordinal, X_freq, model_freq = alternative_encoding_methods(df)
