import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import seaborn as sns


class RetrospectiveSamplingDemo:
    def __init__(self):
        self.population_data = None
        self.retrospective_data = None
        self.population_model = None
        self.retrospective_model = None
        
    def generate_population_data(self, n_population=10000, prevalence=0.01):
        """Generate population data with specified prevalence"""
        np.random.seed(42)
        
        # Generate features
        X = np.random.randn(n_population, 3)
        X[:, 0] = 1  # Add intercept
        
        # True population parameters
        self.true_alpha = -4.6  # Logit of 0.01 prevalence
        self.true_beta = np.array([0.5, -0.3, 0.8])
        
        # Generate probabilities and outcomes
        z = X @ np.concatenate([[self.true_alpha], self.true_beta])
        p = 1 / (1 + np.exp(-z))
        y = np.random.binomial(1, p)
        
        self.population_data = pd.DataFrame({
            'X1': X[:, 1],
            'X2': X[:, 2],
            'X3': X[:, 3],
            'y': y,
            'probability': p
        })
        
        print(f"Population Summary:")
        print(f"Total samples: {n_population}")
        print(f"Cases: {y.sum()} ({y.sum()/n_population:.3f})")
        print(f"Controls: {n_population - y.sum()} ({(n_population - y.sum())/n_population:.3f})")
        
        return self.population_data
    
    def create_retrospective_sample(self, n_cases=100, n_controls=100):
        """Create retrospective sample with specified case/control counts"""
        cases = self.population_data[self.population_data['y'] == 1]
        controls = self.population_data[self.population_data['y'] == 0]
        
        # Sample cases and controls
        sampled_cases = cases.sample(n=min(n_cases, len(cases)), random_state=42)
        sampled_controls = controls.sample(n=min(n_controls, len(controls)), random_state=42)
        
        self.retrospective_data = pd.concat([sampled_cases, sampled_controls])
        
        print(f"\nRetrospective Sample Summary:")
        print(f"Cases: {len(sampled_cases)}")
        print(f"Controls: {len(sampled_controls)}")
        print(f"Total: {len(self.retrospective_data)}")
        
        return self.retrospective_data
    
    def fit_models(self):
        """Fit logistic regression models to both datasets"""
        # Population model (if we had random sampling)
        X_pop = self.population_data[['X1', 'X2', 'X3']].values
        y_pop = self.population_data['y'].values
        
        self.population_model = LogisticRegression(fit_intercept=True, random_state=42)
        self.population_model.fit(X_pop, y_pop)
        
        # Retrospective model
        X_retro = self.retrospective_data[['X1', 'X2', 'X3']].values
        y_retro = self.retrospective_data['y'].values
        
        self.retrospective_model = LogisticRegression(fit_intercept=True, random_state=42)
        self.retrospective_model.fit(X_retro, y_retro)
        
        return self.population_model, self.retrospective_model
    
    def compare_models(self):
        """Compare population and retrospective models"""
        print("\n=== Model Comparison ===")
        
        # Extract coefficients
        pop_intercept = self.population_model.intercept_[0]
        pop_coef = self.population_model.coef_[0]
        
        retro_intercept = self.retrospective_model.intercept_[0]
        retro_coef = self.retrospective_model.coef_[0]
        
        # Create comparison table
        comparison = pd.DataFrame({
            'True': np.concatenate([[self.true_alpha], self.true_beta]),
            'Population': np.concatenate([[pop_intercept], pop_coef]),
            'Retrospective': np.concatenate([[retro_intercept], retro_coef]),
            'Difference': np.concatenate([[retro_intercept - pop_intercept], retro_coef - pop_coef])
        }, index=['Intercept', 'X1', 'X2', 'X3'])
        
        print(comparison.round(4))
        
        # Theoretical intercept adjustment
        n_cases = (self.retrospective_data['y'] == 1).sum()
        n_controls = (self.retrospective_data['y'] == 0).sum()
        n_population = len(self.population_data)
        n_cases_pop = self.population_data['y'].sum()
        n_controls_pop = len(self.population_data) - self.population_data['y'].sum()
        
        pi_1 = n_cases / n_cases_pop  # Sampling probability for cases
        pi_0 = n_controls / n_controls_pop  # Sampling probability for controls
        
        theoretical_adjustment = np.log(pi_1 / pi_0)
        actual_adjustment = retro_intercept - pop_intercept
        
        print(f"\nSampling Probabilities:")
        print(f"π₁ (cases): {pi_1:.6f}")
        print(f"π₀ (controls): {pi_0:.6f}")
        print(f"log(π₁/π₀): {theoretical_adjustment:.4f}")
        print(f"Actual intercept difference: {actual_adjustment:.4f}")
        print(f"Difference: {abs(theoretical_adjustment - actual_adjustment):.4f}")
        
        return comparison
    
    def evaluate_predictions(self):
        """Evaluate model performance on population data"""
        X_pop = self.population_data[['X1', 'X2', 'X3']].values
        y_pop = self.population_data['y'].values
        
        # Predictions from both models
        pop_pred_proba = self.population_model.predict_proba(X_pop)[:, 1]
        retro_pred_proba = self.retrospective_model.predict_proba(X_pop)[:, 1]
        
        # Adjust retrospective predictions
        adjusted_pred_proba = self.adjust_probabilities(retro_pred_proba)
        
        # Calculate metrics
        results = {}
        
        for name, pred_proba in [('Population', pop_pred_proba), 
                                ('Retrospective', retro_pred_proba),
                                ('Adjusted', adjusted_pred_proba)]:
            pred_class = (pred_proba >= 0.5).astype(int)
            accuracy = accuracy_score(y_pop, pred_class)
            auc = roc_auc_score(y_pop, pred_proba)
            
            results[name] = {
                'Accuracy': accuracy,
                'AUC': auc,
                'Mean Probability': pred_proba.mean()
            }
        
        print("\n=== Prediction Performance ===")
        results_df = pd.DataFrame(results).T
        print(results_df.round(4))
        
        return results
    
    def adjust_probabilities(self, retro_probs):
        """Adjust retrospective probabilities to population scale"""
        # Calculate sampling probabilities
        n_cases = (self.retrospective_data['y'] == 1).sum()
        n_controls = (self.retrospective_data['y'] == 0).sum()
        n_population = len(self.population_data)
        n_cases_pop = self.population_data['y'].sum()
        n_controls_pop = len(self.population_data) - self.population_data['y'].sum()
        
        pi_1 = n_cases / n_cases_pop
        pi_0 = n_controls / n_controls_pop
        
        # Adjust probabilities
        adjusted_probs = retro_probs / (retro_probs + (1 - retro_probs) * pi_0 / pi_1)
        
        return adjusted_probs
    
    def visualize_comparison(self):
        """Visualize the comparison between models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coefficient comparison
        comparison = self.compare_models()
        
        x = np.arange(len(comparison.index))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, comparison['True'], width, label='True', alpha=0.8)
        axes[0, 0].bar(x + width/2, comparison['Retrospective'], width, label='Retrospective', alpha=0.8)
        axes[0, 0].set_xlabel('Parameters')
        axes[0, 0].set_ylabel('Coefficient Value')
        axes[0, 0].set_title('Coefficient Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(comparison.index)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Probability distributions
        X_pop = self.population_data[['X1', 'X2', 'X3']].values
        pop_pred_proba = self.population_model.predict_proba(X_pop)[:, 1]
        retro_pred_proba = self.retrospective_model.predict_proba(X_pop)[:, 1]
        adjusted_pred_proba = self.adjust_probabilities(retro_pred_proba)
        
        axes[0, 1].hist(pop_pred_proba, bins=50, alpha=0.7, label='Population Model', density=True)
        axes[0, 1].hist(retro_pred_proba, bins=50, alpha=0.7, label='Retrospective Model', density=True)
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Probability Distributions')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Adjusted vs Population probabilities
        axes[1, 0].scatter(pop_pred_proba, adjusted_pred_proba, alpha=0.5)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Population Model Probability')
        axes[1, 0].set_ylabel('Adjusted Retrospective Probability')
        axes[1, 0].set_title('Adjusted vs Population Probabilities')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ROC curves
        y_pop = self.population_data['y'].values
        
        for name, pred_proba in [('Population', pop_pred_proba), 
                                ('Retrospective', retro_pred_proba),
                                ('Adjusted', adjusted_pred_proba)]:
            fpr, tpr, _ = roc_curve(y_pop, pred_proba)
            auc = roc_auc_score(y_pop, pred_proba)
            axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demonstrate_retrospective_sampling():
    """Main demonstration of retrospective sampling"""
    # Create demonstration
    demo = RetrospectiveSamplingDemo()
    
    # Generate data
    population_data = demo.generate_population_data(n_population=10000, prevalence=0.01)
    retrospective_data = demo.create_retrospective_sample(n_cases=100, n_controls=100)
    
    # Fit models
    pop_model, retro_model = demo.fit_models()
    
    # Compare models
    comparison = demo.compare_models()
    
    # Evaluate predictions
    results = demo.evaluate_predictions()
    
    # Visualize results
    demo.visualize_comparison()
    
    return demo, comparison, results


def demonstrate_sampling_ratios(demo):
    """Demonstrate different sampling ratios"""
    print("\n=== Different Sampling Ratios ===")
    sampling_ratios = [(50, 50), (100, 100), (200, 100), (100, 200)]
    
    results = []
    
    for n_cases, n_controls in sampling_ratios:
        print(f"\nSampling {n_cases} cases and {n_controls} controls:")
        
        # Create new retrospective sample
        retro_data = demo.create_retrospective_sample(n_cases=n_cases, n_controls=n_controls)
        
        # Fit model
        retro_model = LogisticRegression(fit_intercept=True, random_state=42)
        retro_model.fit(retro_data[['X1', 'X2', 'X3']].values, retro_data['y'].values)
        
        # Compare coefficients
        pop_coef = demo.population_model.coef_[0]
        retro_coef = retro_model.coef_[0]
        
        coef_diff = np.linalg.norm(pop_coef - retro_coef)
        intercept_diff = retro_model.intercept_[0] - demo.population_model.intercept_[0]
        
        print(f"  Coefficient difference: {coef_diff:.4f}")
        print(f"  Intercept difference: {intercept_diff:.4f}")
        
        results.append({
            'n_cases': n_cases,
            'n_controls': n_controls,
            'coef_diff': coef_diff,
            'intercept_diff': intercept_diff
        })
    
    return pd.DataFrame(results)


def demonstrate_prevalence_effects():
    """Demonstrate effects of different population prevalences"""
    print("\n=== Prevalence Effects ===")
    prevalences = [0.001, 0.01, 0.05, 0.1]
    
    results = []
    
    for prevalence in prevalences:
        print(f"\nPopulation prevalence: {prevalence}")
        
        # Generate new population data
        demo = RetrospectiveSamplingDemo()
        population_data = demo.generate_population_data(n_population=10000, prevalence=prevalence)
        retrospective_data = demo.create_retrospective_sample(n_cases=100, n_controls=100)
        
        # Fit models
        pop_model, retro_model = demo.fit_models()
        
        # Compare coefficients
        pop_coef = pop_model.coef_[0]
        retro_coef = retro_model.coef_[0]
        
        coef_diff = np.linalg.norm(pop_coef - retro_coef)
        intercept_diff = retro_model.intercept_[0] - pop_model.intercept_[0]
        
        print(f"  Coefficient difference: {coef_diff:.4f}")
        print(f"  Intercept difference: {intercept_diff:.4f}")
        
        results.append({
            'prevalence': prevalence,
            'coef_diff': coef_diff,
            'intercept_diff': intercept_diff
        })
    
    return pd.DataFrame(results)


def demonstrate_probability_calibration():
    """Demonstrate probability calibration methods"""
    print("\n=== Probability Calibration ===")
    
    # Create demonstration
    demo = RetrospectiveSamplingDemo()
    population_data = demo.generate_population_data(n_population=10000, prevalence=0.01)
    retrospective_data = demo.create_retrospective_sample(n_cases=100, n_controls=100)
    
    # Fit models
    pop_model, retro_model = demo.fit_models()
    
    # Get predictions
    X_pop = population_data[['X1', 'X2', 'X3']].values
    y_pop = population_data['y'].values
    
    pop_pred_proba = pop_model.predict_proba(X_pop)[:, 1]
    retro_pred_proba = retro_model.predict_proba(X_pop)[:, 1]
    adjusted_pred_proba = demo.adjust_probabilities(retro_pred_proba)
    
    # Compare probability distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Population probabilities
    axes[0].hist(pop_pred_proba, bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Population Model Probabilities')
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Retrospective probabilities
    axes[1].hist(retro_pred_proba, bins=50, alpha=0.7, color='red')
    axes[1].set_title('Retrospective Model Probabilities')
    axes[1].set_xlabel('Probability')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)
    
    # Adjusted probabilities
    axes[2].hist(adjusted_pred_proba, bins=50, alpha=0.7, color='green')
    axes[2].set_title('Adjusted Probabilities')
    axes[2].set_xlabel('Probability')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nProbability Summary Statistics:")
    print(f"Population model mean: {pop_pred_proba.mean():.4f}")
    print(f"Retrospective model mean: {retro_pred_proba.mean():.4f}")
    print(f"Adjusted model mean: {adjusted_pred_proba.mean():.4f}")
    
    return {
        'pop_pred_proba': pop_pred_proba,
        'retro_pred_proba': retro_pred_proba,
        'adjusted_pred_proba': adjusted_pred_proba
    }


def demonstrate_theoretical_derivation():
    """Demonstrate the theoretical derivation with numerical examples"""
    print("\n=== Theoretical Derivation Demonstration ===")
    
    # Create simple example
    demo = RetrospectiveSamplingDemo()
    population_data = demo.generate_population_data(n_population=10000, prevalence=0.01)
    retrospective_data = demo.create_retrospective_sample(n_cases=100, n_controls=100)
    
    # Calculate sampling probabilities
    n_cases = (retrospective_data['y'] == 1).sum()
    n_controls = (retrospective_data['y'] == 0).sum()
    n_cases_pop = population_data['y'].sum()
    n_controls_pop = len(population_data) - population_data['y'].sum()
    
    pi_1 = n_cases / n_cases_pop
    pi_0 = n_controls / n_controls_pop
    
    print(f"Sampling probabilities:")
    print(f"π₁ (cases): {pi_1:.6f}")
    print(f"π₀ (controls): {pi_0:.6f}")
    print(f"log(π₁/π₀): {np.log(pi_1/pi_0):.4f}")
    
    # Fit models
    pop_model, retro_model = demo.fit_models()
    
    # Compare intercepts
    pop_intercept = pop_model.intercept_[0]
    retro_intercept = retro_model.intercept_[0]
    
    print(f"\nIntercept comparison:")
    print(f"Population intercept: {pop_intercept:.4f}")
    print(f"Retrospective intercept: {retro_intercept:.4f}")
    print(f"Difference: {retro_intercept - pop_intercept:.4f}")
    print(f"Theoretical adjustment: {np.log(pi_1/pi_0):.4f}")
    
    # Demonstrate probability adjustment
    print(f"\nProbability adjustment example:")
    retro_prob = 0.7  # Example retrospective probability
    adjusted_prob = retro_prob / (retro_prob + (1 - retro_prob) * pi_0 / pi_1)
    print(f"Retrospective probability: {retro_prob:.3f}")
    print(f"Adjusted probability: {adjusted_prob:.3f}")
    
    return {
        'pi_1': pi_1,
        'pi_0': pi_0,
        'log_ratio': np.log(pi_1/pi_0),
        'pop_intercept': pop_intercept,
        'retro_intercept': retro_intercept
    }


def demonstrate_practical_applications():
    """Demonstrate practical applications of retrospective sampling"""
    print("\n=== Practical Applications ===")
    
    # Medical research example
    print("\n1. Medical Research - Rare Disease Study:")
    print("   - Disease prevalence: 0.5%")
    print("   - Random sampling: Need ~20,000 people for 100 cases")
    print("   - Retrospective sampling: Directly sample 100 cases + 100 controls")
    print("   - Efficiency gain: 100x more efficient")
    
    # Fraud detection example
    print("\n2. Fraud Detection - Credit Card Fraud:")
    print("   - Fraud rate: 0.1%")
    print("   - Random sampling: Need ~100,000 transactions for 100 fraud cases")
    print("   - Retrospective sampling: Directly sample 100 fraud + 100 legitimate")
    print("   - Efficiency gain: 500x more efficient")
    
    # Quality control example
    print("\n3. Quality Control - Manufacturing Defects:")
    print("   - Defect rate: 0.01%")
    print("   - Random sampling: Need ~1,000,000 items for 100 defects")
    print("   - Retrospective sampling: Directly sample 100 defects + 100 good items")
    print("   - Efficiency gain: 5000x more efficient")
    
    # Demonstrate with actual data
    demo = RetrospectiveSamplingDemo()
    
    # Simulate rare disease data
    population_data = demo.generate_population_data(n_population=100000, prevalence=0.005)
    retrospective_data = demo.create_retrospective_sample(n_cases=100, n_controls=100)
    
    # Fit models
    pop_model, retro_model = demo.fit_models()
    
    # Compare performance
    X_pop = population_data[['X1', 'X2', 'X3']].values
    y_pop = population_data['y'].values
    
    pop_pred_proba = pop_model.predict_proba(X_pop)[:, 1]
    retro_pred_proba = retro_model.predict_proba(X_pop)[:, 1]
    adjusted_pred_proba = demo.adjust_probabilities(retro_pred_proba)
    
    print(f"\n4. Performance Comparison:")
    print(f"   Population model AUC: {roc_auc_score(y_pop, pop_pred_proba):.3f}")
    print(f"   Retrospective model AUC: {roc_auc_score(y_pop, retro_pred_proba):.3f}")
    print(f"   Adjusted model AUC: {roc_auc_score(y_pop, adjusted_pred_proba):.3f}")
    
    return {
        'population_data': population_data,
        'retrospective_data': retrospective_data,
        'pop_model': pop_model,
        'retro_model': retro_model
    }


def demonstrate_limitations_and_cautions():
    """Demonstrate limitations and cautions of retrospective sampling"""
    print("\n=== Limitations and Cautions ===")
    
    # 1. Selection bias
    print("\n1. Selection Bias:")
    print("   - Cases and controls must be representative of their populations")
    print("   - Hospital-based studies may not represent community cases")
    print("   - Control selection must be appropriate for the research question")
    
    # 2. Information bias
    print("\n2. Information Bias:")
    print("   - Recall bias: Cases may remember exposures differently")
    print("   - Interviewer bias: Knowledge of case/control status may affect data collection")
    print("   - Measurement bias: Different data collection methods for cases vs controls")
    
    # 3. Confounding
    print("\n3. Confounding:")
    print("   - Retrospective sampling doesn't eliminate confounding")
    print("   - Must still control for relevant confounders")
    print("   - Stratification or regression adjustment still needed")
    
    # 4. Generalizability
    print("\n4. Generalizability:")
    print("   - Results may not generalize to different populations")
    print("   - Sampling frame must be clearly defined")
    print("   - External validity depends on study design")
    
    # Demonstrate with simulation
    print("\n5. Simulation Example - Selection Bias:")
    
    # Create biased sampling
    demo = RetrospectiveSamplingDemo()
    population_data = demo.generate_population_data(n_population=10000, prevalence=0.01)
    
    # Introduce selection bias: cases with higher X1 values are more likely to be sampled
    cases = population_data[population_data['y'] == 1]
    controls = population_data[population_data['y'] == 0]
    
    # Biased sampling: prefer cases with higher X1
    case_weights = np.exp(cases['X1'])  # Higher X1 = higher sampling probability
    case_weights = case_weights / case_weights.sum()
    
    biased_cases = cases.sample(n=100, weights=case_weights, random_state=42)
    unbiased_controls = controls.sample(n=100, random_state=42)
    
    biased_retrospective_data = pd.concat([biased_cases, unbiased_controls])
    
    # Fit models
    X_pop = population_data[['X1', 'X2', 'X3']].values
    y_pop = population_data['y'].values
    
    pop_model = LogisticRegression(fit_intercept=True, random_state=42)
    pop_model.fit(X_pop, y_pop)
    
    biased_model = LogisticRegression(fit_intercept=True, random_state=42)
    biased_model.fit(biased_retrospective_data[['X1', 'X2', 'X3']].values, 
                     biased_retrospective_data['y'].values)
    
    # Compare coefficients
    print(f"   Population X1 coefficient: {pop_model.coef_[0][0]:.4f}")
    print(f"   Biased sample X1 coefficient: {biased_model.coef_[0][0]:.4f}")
    print(f"   Bias in X1 coefficient: {biased_model.coef_[0][0] - pop_model.coef_[0][0]:.4f}")
    
    return {
        'population_data': population_data,
        'biased_retrospective_data': biased_retrospective_data,
        'pop_model': pop_model,
        'biased_model': biased_model
    }


def main():
    """Main function to demonstrate retrospective sampling"""
    print("Retrospective Sampling in Logistic Regression")
    print("=" * 60)
    
    # 1. Basic demonstration
    print("\n1. Basic Demonstration:")
    demo, comparison, results = demonstrate_retrospective_sampling()
    
    # 2. Different sampling ratios
    print("\n2. Sampling Ratio Analysis:")
    sampling_results = demonstrate_sampling_ratios(demo)
    
    # 3. Prevalence effects
    print("\n3. Prevalence Effects:")
    prevalence_results = demonstrate_prevalence_effects()
    
    # 4. Probability calibration
    print("\n4. Probability Calibration:")
    calibration_results = demonstrate_probability_calibration()
    
    # 5. Theoretical derivation
    print("\n5. Theoretical Derivation:")
    theory_results = demonstrate_theoretical_derivation()
    
    # 6. Practical applications
    print("\n6. Practical Applications:")
    applications_results = demonstrate_practical_applications()
    
    # 7. Limitations and cautions
    print("\n7. Limitations and Cautions:")
    limitations_results = demonstrate_limitations_and_cautions()
    
    print("\n=== Key Insights ===")
    print("1. Coefficients remain unbiased in retrospective sampling")
    print("2. Only intercept needs adjustment")
    print("3. Probabilities can be calibrated for population inference")
    print("4. Model performance is maintained")
    print("5. Selection bias is a major concern")
    print("6. Retrospective sampling is highly efficient for rare outcomes")
    
    return {
        'demo': demo,
        'comparison': comparison,
        'results': results,
        'sampling_results': sampling_results,
        'prevalence_results': prevalence_results,
        'calibration_results': calibration_results,
        'theory_results': theory_results,
        'applications_results': applications_results,
        'limitations_results': limitations_results
    }


if __name__ == "__main__":
    main()
