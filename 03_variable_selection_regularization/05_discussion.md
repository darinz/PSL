# 3.5. Discussion: Comparing Variable Selection and Regularization Methods

## Introduction

Having explored various techniques for variable selection and regularization—including subset selection, ridge regression, lasso regression, and principal components regression—we now address the critical question: **Which method is most appropriate for a given situation?** This discussion provides a comprehensive framework for understanding the strengths, limitations, and optimal use cases for each method.

## 3.5.1 Theoretical Framework for Method Comparison

### The Bias-Variance Tradeoff Revisited

![Bias-Variance Trade-off and Model Complexity](../_images/w3_fig_3_11.png)

*Figure: The relationship between model complexity, training error, and test error. Illustrates the bias-variance trade-off central to variable selection and regularization.*

All variable selection and regularization methods can be understood through the bias-variance decomposition of prediction error:

```math
\text{MSE}(\hat{f}) = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2
```

where:
- $\text{Bias}^2(\hat{f})$ is the squared bias of the estimator
- $\text{Var}(\hat{f})$ is the variance of the estimator
- $\sigma^2$ is the irreducible error

Different methods achieve different points on the bias-variance tradeoff curve:

1. **Subset Selection**: Low bias, high variance
2. **Ridge Regression**: Moderate bias, low variance
3. **Lasso Regression**: Moderate bias, low variance, with sparsity
4. **Principal Components Regression**: High bias, very low variance

### Mathematical Characterization of Methods

Let's characterize each method mathematically:

**Subset Selection:**
```math
\hat{\boldsymbol{\beta}}_{\text{subset}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_0
```

**Ridge Regression:**
```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|^2_2
```

**Lasso Regression:**
```math
\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1
```

**Principal Components Regression:**
```math
\hat{\boldsymbol{\beta}}_{\text{PCR}} = \mathbf{V}_k(\mathbf{V}_k^T\mathbf{X}^T\mathbf{X}\mathbf{V}_k)^{-1}\mathbf{V}_k^T\mathbf{X}^T\mathbf{y}
```

where $\mathbf{V}_k$ contains the first $k$ principal component directions.

## 3.5.2 Simulation Study Framework

### Design Matrix Specifications

We examine three distinct scenarios that represent common real-world situations:

#### Scenario 1: Curated Feature Set (X1)
- **Structure**: Small set of carefully selected features
- **Characteristics**: Low dimensionality, high signal-to-noise ratio
- **Expected Performance**: Full model often sufficient

#### Scenario 2: Extended Feature Set with Correlations (X2)
- **Structure**: Original features plus quadratic and interaction terms
- **Characteristics**: Moderate dimensionality, correlated features
- **Expected Performance**: Shrinkage methods beneficial

#### Scenario 3: High-Dimensional with Noise (X3)
- **Structure**: Extended features plus 500 noise features
- **Characteristics**: High dimensionality, low signal-to-noise ratio
- **Expected Performance**: Variable selection crucial

### Performance Metrics

We evaluate methods using multiple criteria:

1. **Prediction Accuracy**: Mean squared error on test set
2. **Model Complexity**: Number of non-zero coefficients
3. **Variable Selection Accuracy**: Precision and recall for true variables
4. **Computational Efficiency**: Training time
5. **Stability**: Consistency across different random seeds

## 3.5.3 Comprehensive Implementation

### Python Implementation

```python
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

# Run the comprehensive study
if __name__ == "__main__":
    print("Starting Comprehensive Variable Selection Study")
    print("=" * 60)
    
    study = VariableSelectionComparison(n_samples=200, random_state=42)
    all_results, scenarios = study.run_comprehensive_study()
    
    # Create visualizations
    study.visualize_results(all_results)
    
    print("\nStudy completed!")
```

### R Implementation

```r
# Load libraries
library(glmnet)
library(pls)
library(leaps)
library(ggplot2)
library(dplyr)
library(tidyr)

# Comprehensive comparison function
compare_variable_selection_methods <- function(X, y, true_beta = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Split data
  set.seed(42)
  train_idx <- sample(1:n, 0.7 * n)
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Standardize data
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                         scale = attr(X_train_scaled, "scaled:scale"))
  
  results <- list()
  
  # 1. Ordinary Least Squares
  start_time <- Sys.time()
  ols_model <- lm(y_train ~ X_train_scaled - 1)
  ols_time <- difftime(Sys.time(), start_time, units = "secs")
  
  ols_pred <- X_test_scaled %*% coef(ols_model)
  ols_mse <- mean((y_test - ols_pred)^2)
  ols_r2 <- 1 - sum((y_test - ols_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$OLS <- list(
    coefficients = coef(ols_model),
    test_mse = ols_mse,
    test_r2 = ols_r2,
    n_nonzero = sum(coef(ols_model) != 0),
    training_time = as.numeric(ols_time)
  )
  
  # 2. Ridge Regression
  start_time <- Sys.time()
  ridge_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 0, standardize = FALSE)
  ridge_model <- glmnet(X_train_scaled, y_train, alpha = 0, lambda = ridge_cv$lambda.min)
  ridge_time <- difftime(Sys.time(), start_time, units = "secs")
  
  ridge_pred <- predict(ridge_model, newx = X_test_scaled)
  ridge_mse <- mean((y_test - ridge_pred)^2)
  ridge_r2 <- 1 - sum((y_test - ridge_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$Ridge <- list(
    coefficients = as.vector(coef(ridge_model))[-1],  # Remove intercept
    test_mse = ridge_mse,
    test_r2 = ridge_r2,
    n_nonzero = sum(coef(ridge_model)[-1] != 0),
    training_time = as.numeric(ridge_time),
    best_alpha = ridge_cv$lambda.min
  )
  
  # 3. Lasso Regression
  start_time <- Sys.time()
  lasso_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 1, standardize = FALSE)
  lasso_model <- glmnet(X_train_scaled, y_train, alpha = 1, lambda = lasso_cv$lambda.min)
  lasso_time <- difftime(Sys.time(), start_time, units = "secs")
  
  lasso_pred <- predict(lasso_model, newx = X_test_scaled)
  lasso_mse <- mean((y_test - lasso_pred)^2)
  lasso_r2 <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$Lasso <- list(
    coefficients = as.vector(coef(lasso_model))[-1],  # Remove intercept
    test_mse = lasso_mse,
    test_r2 = lasso_r2,
    n_nonzero = sum(coef(lasso_model)[-1] != 0),
    training_time = as.numeric(lasso_time),
    best_alpha = lasso_cv$lambda.min
  )
  
  # 4. Elastic Net
  start_time <- Sys.time()
  elastic_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 0.5, standardize = FALSE)
  elastic_model <- glmnet(X_train_scaled, y_train, alpha = 0.5, lambda = elastic_cv$lambda.min)
  elastic_time <- difftime(Sys.time(), start_time, units = "secs")
  
  elastic_pred <- predict(elastic_model, newx = X_test_scaled)
  elastic_mse <- mean((y_test - elastic_pred)^2)
  elastic_r2 <- 1 - sum((y_test - elastic_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$ElasticNet <- list(
    coefficients = as.vector(coef(elastic_model))[-1],  # Remove intercept
    test_mse = elastic_mse,
    test_r2 = elastic_r2,
    n_nonzero = sum(coef(elastic_model)[-1] != 0),
    training_time = as.numeric(elastic_time),
    best_alpha = elastic_cv$lambda.min
  )
  
  # 5. Principal Components Regression
  start_time <- Sys.time()
  n_components <- min(20, p)  # Limit components
  pcr_model <- pcr(y_train ~ X_train_scaled, ncomp = n_components, validation = "CV")
  pcr_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Find optimal number of components
  opt_comp <- which.min(pcr_model$validation$PRESS)
  pcr_pred <- predict(pcr_model, newdata = data.frame(X_train_scaled = X_test_scaled), ncomp = opt_comp)
  pcr_mse <- mean((y_test - pcr_pred)^2)
  pcr_r2 <- 1 - sum((y_test - pcr_pred)^2) / sum((y_test - mean(y_test))^2)
  
  # Transform coefficients back to original space
  pcr_coef <- coef(pcr_model, ncomp = opt_comp)
  
  results$PCR <- list(
    coefficients = as.vector(pcr_coef),
    test_mse = pcr_mse,
    test_r2 = pcr_r2,
    n_nonzero = sum(pcr_coef != 0),
    training_time = as.numeric(pcr_time),
    n_components = opt_comp
  )
  
  # 6. Subset Selection (Forward Stepwise)
  start_time <- Sys.time()
  max_vars <- min(20, p)  # Limit for computational efficiency
  subset_model <- regsubsets(y_train ~ X_train_scaled, data = data.frame(X_train_scaled, y_train), 
                            nvmax = max_vars, method = "forward")
  subset_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Find optimal subset size using BIC
  opt_size <- which.min(summary(subset_model)$bic)
  subset_coef <- coef(subset_model, opt_size)
  
  # Create full coefficient vector
  full_coef <- rep(0, p)
  var_names <- names(subset_coef)[-1]  # Remove intercept
  var_indices <- as.numeric(substr(var_names, 15, nchar(var_names)))  # Extract indices
  full_coef[var_indices] <- subset_coef[-1]
  
  subset_pred <- X_test_scaled %*% full_coef
  subset_mse <- mean((y_test - subset_pred)^2)
  subset_r2 <- 1 - sum((y_test - subset_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$SubsetSelection <- list(
    coefficients = full_coef,
    test_mse = subset_mse,
    test_r2 = subset_r2,
    n_nonzero = opt_size,
    training_time = as.numeric(subset_time),
    selected_size = opt_size
  )
  
  # Add variable selection metrics if true coefficients are known
  if (!is.null(true_beta)) {
    for (method in names(results)) {
      if (method != "OLS") {
        true_nonzero <- true_beta != 0
        pred_nonzero <- results[[method]]$coefficients != 0
        
        tp <- sum(true_nonzero & pred_nonzero)
        fp <- sum(!true_nonzero & pred_nonzero)
        fn <- sum(true_nonzero & !pred_nonzero)
        
        precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
        f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        
        results[[method]]$precision <- precision
        results[[method]]$recall <- recall
        results[[method]]$f1_score <- f1
      }
    }
  }
  
  return(results)
}

# Generate design matrices
generate_design_matrices <- function(n_samples = 200, seed = 42) {
  set.seed(seed)
  
  # Base features
  n_base <- 5
  X_base <- matrix(rnorm(n_samples * n_base), n_samples, n_base)
  
  # Scenario 1: Curated features
  X1 <- X_base
  
  # Scenario 2: Extended features with interactions
  X2_extended <- model.matrix(~ .^2, data = data.frame(X_base))[, -1]  # Remove intercept
  X2 <- X2_extended
  
  # Scenario 3: High-dimensional with noise
  n_noise <- 500
  noise_features <- matrix(0, n_samples, n_noise)
  
  for (i in 1:n_noise) {
    true_feature_idx <- sample(1:ncol(X2), 1)
    noise_features[, i] <- sample(X2[, true_feature_idx])
  }
  
  X3 <- cbind(X2, noise_features)
  
  return(list(X1 = X1, X2 = X2, X3 = X3))
}

# Generate response variables
generate_response <- function(X, sparsity_level = 0.3, seed = 42) {
  set.seed(seed)
  
  n_features <- ncol(X)
  n_active <- max(1, round(n_features * sparsity_level))
  
  # True coefficients (sparse)
  true_beta <- rep(0, n_features)
  active_indices <- sample(1:n_features, n_active)
  true_beta[active_indices] <- rnorm(n_active) * 2
  
  # Generate response
  y <- X %*% true_beta + 0.5 * rnorm(nrow(X))
  
  return(list(y = y, true_beta = true_beta))
}

# Run comprehensive study
run_comprehensive_study <- function() {
  cat("Generating design matrices...\n")
  design_matrices <- generate_design_matrices(n_samples = 200)
  
  cat("Generating response variables...\n")
  response1 <- generate_response(design_matrices$X1, sparsity_level = 0.8)
  response2 <- generate_response(design_matrices$X2, sparsity_level = 0.3)
  response3 <- generate_response(design_matrices$X3, sparsity_level = 0.05)
  
  scenarios <- list(
    "X1 (Curated Features)" = list(X = design_matrices$X1, y = response1$y, beta = response1$true_beta),
    "X2 (Extended Features)" = list(X = design_matrices$X2, y = response2$y, beta = response2$true_beta),
    "X3 (High-Dimensional + Noise)" = list(X = design_matrices$X3, y = response3$y, beta = response3$true_beta)
  )
  
  all_results <- list()
  
  for (scenario_name in names(scenarios)) {
    cat(sprintf("\nAnalyzing %s...\n", scenario_name))
    scenario <- scenarios[[scenario_name]]
    
    cat(sprintf("Data shape: %d x %d\n", nrow(scenario$X), ncol(scenario$X)))
    cat(sprintf("True non-zero coefficients: %d\n", sum(scenario$beta != 0)))
    
    results <- compare_variable_selection_methods(scenario$X, scenario$y, scenario$beta)
    all_results[[scenario_name]] <- results
    
    # Print summary
    cat(sprintf("\nResults for %s:\n", scenario_name))
    cat(paste(rep("-", 80), collapse = ""), "\n")
    cat(sprintf("%-15s %-12s %-10s %-10s %-10s\n", "Method", "Test MSE", "Test R²", "Non-zero", "Time (s)"))
    cat(paste(rep("-", 80), collapse = ""), "\n")
    
    for (method in names(results)) {
      result <- results[[method]]
      cat(sprintf("%-15s %-12.4f %-10.4f %-10d %-10.4f\n", 
                  method, result$test_mse, result$test_r2, result$n_nonzero, result$training_time))
    }
  }
  
  return(all_results)
}

# Run the study
cat("Starting Comprehensive Variable Selection Study\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

all_results <- run_comprehensive_study()

cat("\nStudy completed!\n")
```

## 3.5.4 Key Insights and Recommendations

### Scenario-Specific Recommendations

#### Scenario 1: Curated Features (X1)
**Characteristics:**
- Low dimensionality (5 features)
- High signal-to-noise ratio
- Expert-selected features

**Best Methods:**
1. **Ordinary Least Squares**: Often sufficient due to low dimensionality
2. **Ridge Regression**: Provides slight regularization benefit
3. **Subset Selection**: May help identify most important features

**Why These Work:**
- Low-dimensional problems rarely require aggressive regularization
- Expert knowledge reduces the need for automatic variable selection
- Simple methods avoid overfitting

#### Scenario 2: Extended Features with Correlations (X2)
**Characteristics:**
- Moderate dimensionality (15-20 features)
- Correlated features (quadratic and interaction terms)
- Mixed signal strength

**Best Methods:**
1. **Ridge Regression**: Handles multicollinearity effectively
2. **Elastic Net**: Combines benefits of ridge and lasso
3. **Principal Components Regression**: Reduces dimensionality while preserving variance

**Why These Work:**
- Ridge regression stabilizes coefficient estimates under multicollinearity
- Elastic net provides both shrinkage and variable selection
- PCR reduces dimensionality while maintaining predictive power

#### Scenario 3: High-Dimensional with Noise (X3)
**Characteristics:**
- High dimensionality (500+ features)
- Low signal-to-noise ratio
- Many irrelevant features

**Best Methods:**
1. **Lasso Regression**: Automatic variable selection crucial
2. **Elastic Net**: Handles correlated features while selecting variables
3. **Subset Selection**: Can identify truly important features

**Why These Work:**
- Lasso's sparsity is essential for high-dimensional problems
- Variable selection removes noise features
- Regularization prevents overfitting

### Method Selection Decision Tree

```python
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
```

### Performance Trade-offs

| Method | Prediction Accuracy | Interpretability | Computational Cost | Variable Selection |
|--------|-------------------|------------------|-------------------|-------------------|
| OLS | High (low-dim) | High | Low | None |
| Ridge | High | Medium | Low | None |
| Lasso | High | High | Medium | Automatic |
| Elastic Net | High | High | Medium | Automatic |
| PCR | Medium | Low | Medium | Manual |
| Subset Selection | High | High | High | Manual |

## 3.5.5 Practical Guidelines

### When to Use Each Method

**Use Ordinary Least Squares when:**
- Number of predictors is small (< 10)
- Predictors are uncorrelated
- Sample size is large relative to number of predictors
- Primary goal is interpretation

**Use Ridge Regression when:**
- Predictors are highly correlated
- You want to keep all variables
- Primary goal is prediction accuracy
- Sample size is small relative to number of predictors

**Use Lasso Regression when:**
- You want automatic variable selection
- The true model is sparse
- Interpretability is important
- You have many irrelevant predictors

**Use Elastic Net when:**
- Predictors are correlated but you want variable selection
- You want a compromise between ridge and lasso
- The true model has grouped variables

**Use Principal Components Regression when:**
- Predictors are highly correlated
- You want to reduce dimensionality
- The first few principal components capture most variance
- Prediction is more important than interpretation

**Use Subset Selection when:**
- You want explicit control over variable selection
- Computational cost is not a concern
- You have domain knowledge about variable importance
- You want to understand the selection process

### Best Practices

1. **Always standardize predictors** before applying regularization methods
2. **Use cross-validation** to select tuning parameters
3. **Validate on a holdout set** to assess generalization performance
4. **Consider the problem context** when choosing methods
5. **Check for multicollinearity** and choose methods accordingly
6. **Assess variable selection stability** for high-dimensional problems
7. **Consider computational constraints** for large datasets

### Common Pitfalls

1. **Not standardizing data**: Can lead to inconsistent results
2. **Ignoring multicollinearity**: Can affect method performance
3. **Over-regularization**: Can remove important variables
4. **Under-regularization**: May not address overfitting
5. **Not validating assumptions**: Can lead to poor performance
6. **Ignoring computational cost**: May not be practical for large datasets

## Summary

The choice of variable selection and regularization method depends critically on the problem characteristics:

1. **Dimensionality**: Low-dimensional problems favor simpler methods
2. **Correlation structure**: Correlated predictors benefit from ridge or elastic net
3. **Sparsity**: Sparse signals benefit from lasso or subset selection
4. **Computational constraints**: Large datasets may require efficient methods
5. **Interpretability requirements**: Some methods provide better interpretability

The simulation study framework provides a systematic way to compare methods across different scenarios, helping practitioners make informed decisions based on their specific problem characteristics and constraints.
