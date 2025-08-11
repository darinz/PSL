import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import time

class VariableSelectionComparison:
    """Comprehensive comparison of variable selection and regularization methods"""
    
    def __init__(self, n_samples=200, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_design_matrices(self):
        """Generate three different design matrices"""
        
        # Base features (5 features)
        n_base = 5
        X_base = np.random.randn(self.n_samples, n_base)
        
        # Scenario 1: Curated features (X1)
        self.X1 = X_base.copy()
        
        # Scenario 2: Extended features with interactions (X2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X2_extended = poly.fit_transform(X_base)
        # Remove the constant term and keep only meaningful interactions
        self.X2 = X2_extended[:, 1:]  # Remove intercept, keep all other terms
        
        # Scenario 3: High-dimensional with noise (X3)
        n_noise = 500
        noise_features = np.zeros((self.n_samples, n_noise))
        
        # Generate noise features by shuffling true features
        for i in range(n_noise):
            # Randomly select a true feature and shuffle its values
            true_feature_idx = np.random.randint(0, self.X2.shape[1])
            noise_features[:, i] = np.random.permutation(self.X2[:, true_feature_idx])
        
        self.X3 = np.hstack([self.X2, noise_features])
        
        return self.X1, self.X2, self.X3
    
    def generate_response(self, X, sparsity_level=0.3):
        """Generate response variable with specified sparsity"""
        n_features = X.shape[1]
        n_active = max(1, int(n_features * sparsity_level))
        
        # True coefficients (sparse)
        true_beta = np.zeros(n_features)
        active_indices = np.random.choice(n_features, n_active, replace=False)
        true_beta[active_indices] = np.random.randn(n_active) * 2
        
        # Generate response
        y = X @ true_beta + 0.5 * np.random.randn(self.n_samples)
        
        return y, true_beta
    
    def implement_pcr(self, X, y, n_components=None):
        """Implement Principal Components Regression"""
        if n_components is None:
            n_components = min(X.shape[1], X.shape[0] - 1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Fit linear regression on principal components
        pcr_model = LinearRegression()
        pcr_model.fit(X_pca, y)
        
        # Transform coefficients back to original space
        beta_pcr = pca.components_.T @ pcr_model.coef_
        
        return beta_pcr, pca, scaler
    
    def implement_subset_selection(self, X, y, max_features=None):
        """Implement forward stepwise selection"""
        if max_features is None:
            max_features = min(X.shape[1], X.shape[0] - 1)
        
        n_features = X.shape[1]
        selected_features = []
        remaining_features = list(range(n_features))
        
        for step in range(max_features):
            best_score = float('inf')
            best_feature = None
            
            for feature in remaining_features:
                # Add feature to current selection
                current_features = selected_features + [feature]
                X_subset = X[:, current_features]
                
                # Fit model and compute cross-validation score
                model = LinearRegression()
                scores = cross_val_score(model, X_subset, y, cv=5, scoring='neg_mean_squared_error')
                score = -scores.mean()
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
        
        # Fit final model
        X_final = X[:, selected_features]
        model = LinearRegression()
        model.fit(X_final, y)
        
        # Create full coefficient vector
        beta_subset = np.zeros(n_features)
        beta_subset[selected_features] = model.coef_
        
        return beta_subset, selected_features
    
    def compare_methods(self, X, y, true_beta=None):
        """Compare all methods on given data"""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Standardize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Ordinary Least Squares
        start_time = time.time()
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        ols_time = time.time() - start_time
        
        results['OLS'] = {
            'coefficients': ols.coef_,
            'test_mse': mean_squared_error(y_test, ols.predict(X_test_scaled)),
            'test_r2': r2_score(y_test, ols.predict(X_test_scaled)),
            'n_nonzero': np.sum(ols.coef_ != 0),
            'training_time': ols_time
        }
        
        # 2. Ridge Regression
        start_time = time.time()
        ridge_cv = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 20)}, cv=5)
        ridge_cv.fit(X_train_scaled, y_train)
        ridge_time = time.time() - start_time
        
        results['Ridge'] = {
            'coefficients': ridge_cv.best_estimator_.coef_,
            'test_mse': mean_squared_error(y_test, ridge_cv.predict(X_test_scaled)),
            'test_r2': r2_score(y_test, ridge_cv.predict(X_test_scaled)),
            'n_nonzero': np.sum(ridge_cv.best_estimator_.coef_ != 0),
            'training_time': ridge_time,
            'best_alpha': ridge_cv.best_params_['alpha']
        }
        
        # 3. Lasso Regression
        start_time = time.time()
        lasso_cv = GridSearchCV(Lasso(max_iter=2000), {'alpha': np.logspace(-3, 1, 20)}, cv=5)
        lasso_cv.fit(X_train_scaled, y_train)
        lasso_time = time.time() - start_time
        
        results['Lasso'] = {
            'coefficients': lasso_cv.best_estimator_.coef_,
            'test_mse': mean_squared_error(y_test, lasso_cv.predict(X_test_scaled)),
            'test_r2': r2_score(y_test, lasso_cv.predict(X_test_scaled)),
            'n_nonzero': np.sum(lasso_cv.best_estimator_.coef_ != 0),
            'training_time': lasso_time,
            'best_alpha': lasso_cv.best_params_['alpha']
        }
        
        # 4. Elastic Net
        start_time = time.time()
        elastic_cv = GridSearchCV(
            ElasticNet(max_iter=2000), 
            {'alpha': np.logspace(-3, 1, 10), 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}, 
            cv=5
        )
        elastic_cv.fit(X_train_scaled, y_train)
        elastic_time = time.time() - start_time
        
        results['ElasticNet'] = {
            'coefficients': elastic_cv.best_estimator_.coef_,
            'test_mse': mean_squared_error(y_test, elastic_cv.predict(X_test_scaled)),
            'test_r2': r2_score(y_test, elastic_cv.predict(X_test_scaled)),
            'n_nonzero': np.sum(elastic_cv.best_estimator_.coef_ != 0),
            'training_time': elastic_time,
            'best_params': elastic_cv.best_params_
        }
        
        # 5. Principal Components Regression
        start_time = time.time()
        n_components = min(20, X_train_scaled.shape[1])  # Limit components for computational efficiency
        beta_pcr, pca, pca_scaler = self.implement_pcr(X_train_scaled, y_train, n_components)
        pcr_time = time.time() - start_time
        
        # Transform test data and make predictions
        X_test_pca = pca.transform(X_test_scaled)
        pcr_pred = X_test_pca @ pca.components_[:n_components] @ beta_pcr[:n_components]
        
        results['PCR'] = {
            'coefficients': beta_pcr,
            'test_mse': mean_squared_error(y_test, pcr_pred),
            'test_r2': r2_score(y_test, pcr_pred),
            'n_nonzero': np.sum(beta_pcr != 0),
            'training_time': pcr_time,
            'n_components': n_components
        }
        
        # 6. Subset Selection
        start_time = time.time()
        beta_subset, selected_features = self.implement_subset_selection(X_train_scaled, y_train)
        subset_time = time.time() - start_time
        
        subset_pred = X_test_scaled @ beta_subset
        
        results['SubsetSelection'] = {
            'coefficients': beta_subset,
            'test_mse': mean_squared_error(y_test, subset_pred),
            'test_r2': r2_score(y_test, subset_pred),
            'n_nonzero': len(selected_features),
            'training_time': subset_time,
            'selected_features': selected_features
        }
        
        # Add variable selection accuracy if true coefficients are known
        if true_beta is not None:
            for method in results:
                if method != 'OLS':
                    # Calculate precision and recall for variable selection
                    true_nonzero = true_beta != 0
                    pred_nonzero = results[method]['coefficients'] != 0
                    
                    tp = np.sum(true_nonzero & pred_nonzero)
                    fp = np.sum(~true_nonzero & pred_nonzero)
                    fn = np.sum(true_nonzero & ~pred_nonzero)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    results[method]['precision'] = precision
                    results[method]['recall'] = recall
                    results[method]['f1_score'] = f1
        
        return results
    
    def run_comprehensive_study(self):
        """Run comprehensive comparison study"""
        print("Generating design matrices...")
        X1, X2, X3 = self.generate_design_matrices()
        
        print("Generating response variables...")
        y1, beta1 = self.generate_response(X1, sparsity_level=0.8)  # Most features active
        y2, beta2 = self.generate_response(X2, sparsity_level=0.3)  # Some features active
        y3, beta3 = self.generate_response(X3, sparsity_level=0.05)  # Very sparse
        
        scenarios = {
            'X1 (Curated Features)': (X1, y1, beta1),
            'X2 (Extended Features)': (X2, y2, beta2),
            'X3 (High-Dimensional + Noise)': (X3, y3, beta3)
        }
        
        all_results = {}
        
        for scenario_name, (X, y, beta) in scenarios.items():
            print(f"\nAnalyzing {scenario_name}...")
            print(f"Data shape: {X.shape}")
            print(f"True non-zero coefficients: {np.sum(beta != 0)}")
            
            results = self.compare_methods(X, y, beta)
            all_results[scenario_name] = results
            
            # Print summary
            print(f"\nResults for {scenario_name}:")
            print("-" * 80)
            print(f"{'Method':<15} {'Test MSE':<12} {'Test R²':<10} {'Non-zero':<10} {'Time (s)':<10}")
            print("-" * 80)
            
            for method, result in results.items():
                print(f"{method:<15} {result['test_mse']:<12.4f} {result['test_r2']:<10.4f} "
                      f"{result['n_nonzero']:<10} {result['training_time']:<10.4f}")
        
        return all_results, scenarios
    
    def visualize_results(self, all_results):
        """Create comprehensive visualizations"""
        scenarios = list(all_results.keys())
        methods = list(all_results[scenarios[0]].keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Test MSE comparison
        for i, scenario in enumerate(scenarios):
            mses = [all_results[scenario][method]['test_mse'] for method in methods]
            axes[0, 0].bar(np.arange(len(methods)) + i*0.15, mses, width=0.15, 
                          label=scenario, alpha=0.8)
        axes[0, 0].set_title('Test MSE Comparison')
        axes[0, 0].set_ylabel('Mean Squared Error')
        axes[0, 0].set_xticks(np.arange(len(methods)) + 0.15)
        axes[0, 0].set_xticklabels(methods, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Test R² comparison
        for i, scenario in enumerate(scenarios):
            r2s = [all_results[scenario][method]['test_r2'] for method in methods]
            axes[0, 1].bar(np.arange(len(methods)) + i*0.15, r2s, width=0.15, 
                          label=scenario, alpha=0.8)
        axes[0, 1].set_title('Test R² Comparison')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_xticks(np.arange(len(methods)) + 0.15)
        axes[0, 1].set_xticklabels(methods, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Number of non-zero coefficients
        for i, scenario in enumerate(scenarios):
            n_zeros = [all_results[scenario][method]['n_nonzero'] for method in methods]
            axes[0, 2].bar(np.arange(len(methods)) + i*0.15, n_zeros, width=0.15, 
                          label=scenario, alpha=0.8)
        axes[0, 2].set_title('Number of Non-zero Coefficients')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_xticks(np.arange(len(methods)) + 0.15)
        axes[0, 2].set_xticklabels(methods, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training time comparison
        for i, scenario in enumerate(scenarios):
            times = [all_results[scenario][method]['training_time'] for method in methods]
            axes[1, 0].bar(np.arange(len(methods)) + i*0.15, times, width=0.15, 
                          label=scenario, alpha=0.8)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticks(np.arange(len(methods)) + 0.15)
        axes[1, 0].set_xticklabels(methods, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Variable selection precision (if available)
        if 'precision' in all_results[scenarios[0]][methods[1]]:
            for i, scenario in enumerate(scenarios):
                precisions = [all_results[scenario][method].get('precision', 0) for method in methods]
                axes[1, 1].bar(np.arange(len(methods)) + i*0.15, precisions, width=0.15, 
                              label=scenario, alpha=0.8)
            axes[1, 1].set_title('Variable Selection Precision')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_xticks(np.arange(len(methods)) + 0.15)
            axes[1, 1].set_xticklabels(methods, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Variable selection recall (if available)
        if 'recall' in all_results[scenarios[0]][methods[1]]:
            for i, scenario in enumerate(scenarios):
                recalls = [all_results[scenario][method].get('recall', 0) for method in methods]
                axes[1, 2].bar(np.arange(len(methods)) + i*0.15, recalls, width=0.15, 
                              label=scenario, alpha=0.8)
            axes[1, 2].set_title('Variable Selection Recall')
            axes[1, 2].set_ylabel('Recall')
            axes[1, 2].set_xticks(np.arange(len(methods)) + 0.15)
            axes[1, 2].set_xticklabels(methods, rotation=45)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def select_method(X, y, problem_context):
    """
    Decision tree for selecting variable selection/regularization method
    
    Parameters:
    - X: Design matrix
    - y: Response variable
    - problem_context: Dictionary with problem characteristics
    """
    n, p = X.shape
    
    # Check dimensionality
    if p < 10:
        if problem_context.get('expert_knowledge', False):
            return "OLS or Ridge"
        else:
            return "Ridge or Subset Selection"
    
    elif p < 50:
        if problem_context.get('multicollinearity', False):
            return "Ridge or Elastic Net"
        else:
            return "Lasso or Elastic Net"
    
    else:  # p >= 50
        if problem_context.get('sparse_signal', True):
            return "Lasso or Elastic Net"
        else:
            return "Ridge or PCR"

# Run the comprehensive study
if __name__ == "__main__":
    print("Starting Comprehensive Variable Selection Study")
    print("=" * 60)
    
    study = VariableSelectionComparison(n_samples=200, random_state=42)
    all_results, scenarios = study.run_comprehensive_study()
    
    # Create visualizations
    study.visualize_results(all_results)
    
    print("\nStudy completed!")
