# 10.4. Retrospective Sampling in Logistic Regression

## Introduction

Retrospective sampling (also known as case-control sampling) is a common data collection strategy in medical research, epidemiology, and other fields where the outcome of interest is rare. This sampling method creates unique challenges for logistic regression that every practitioner should understand.

## What is Retrospective Sampling?

### Definition
**Retrospective sampling** is a sampling strategy where we sample based on the outcome variable rather than randomly from the population. Specifically:
- We sample a fixed number of cases (individuals with the outcome of interest, e.g., cancer patients)
- We sample a fixed number of controls (individuals without the outcome, e.g., healthy individuals)
- The sampling is independent of the predictor variables

### Motivation
Consider a rare disease that affects only 1% of the population. To study this disease:
- **Random sampling**: We would need to sample ~10,000 people to get ~100 cases
- **Retrospective sampling**: We can directly sample 100 cases and 100 controls

This makes retrospective sampling much more efficient for rare outcomes.

## Mathematical Foundation

### Population Model
In the population, we assume the true relationship follows:
```math
P(Y=1|X=x) = \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)}
```

Where:
- $Y$ is the binary outcome (1 = case, 0 = control)
- $X$ are the predictor variables
- $\alpha$ is the intercept
- $\beta$ are the coefficients

### Sampling Process
Let $Z$ be an indicator variable for whether an individual is sampled:
```math
Z = \begin{cases}
1 & \text{if individual is sampled} \\
0 & \text{if individual is not sampled}
\end{cases}
```

In retrospective sampling:
- $P(Z=1|Y=1) = \pi_1$ (sampling probability for cases)
- $P(Z=1|Y=0) = \pi_0$ (sampling probability for controls)

### The Problem
We want to estimate $P(Y=1|X=x)$, but our data gives us $P(Y=1|Z=1, X=x)$.

## Mathematical Derivation

### Bayes' Theorem Application
Using Bayes' theorem:
```math
P(Y=1|Z=1, X=x) = \frac{P(Z=1|Y=1, X=x) P(Y=1|X=x)}{P(Z=1|X=x)}
```

Since sampling is independent of $X$ given $Y$:
```math
P(Z=1|Y=1, X=x) = P(Z=1|Y=1) = \pi_1
```

And:
```math
P(Z=1|X=x) = P(Z=1|Y=1, X=x) P(Y=1|X=x) + P(Z=1|Y=0, X=x) P(Y=0|X=x)
```

Substituting:
```math
P(Z=1|X=x) = \pi_1 P(Y=1|X=x) + \pi_0 P(Y=0|X=x)
```

### Retrospective Probability
Therefore:
```math
\begin{split}
P(Y=1|Z=1, X=x) &= \frac{\pi_1 P(Y=1|X=x)}{\pi_1 P(Y=1|X=x) + \pi_0 P(Y=0|X=x)} \\
&= \frac{\pi_1 P(Y=1|X=x)}{\pi_1 P(Y=1|X=x) + \pi_0 (1 - P(Y=1|X=x))}
\end{split}
```

### Key Result
Substituting the logistic model:
```math
\begin{split}
P(Y=1|Z=1, X=x) &= \frac{\pi_1 \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)}}{\pi_1 \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)} + \pi_0 \frac{1}{1 + \exp(\alpha + x^T \beta)}} \\
&= \frac{\pi_1 \exp(\alpha + x^T \beta)}{\pi_1 \exp(\alpha + x^T \beta) + \pi_0} \\
&= \frac{\exp(\alpha + x^T \beta)}{\exp(\alpha + x^T \beta) + \frac{\pi_0}{\pi_1}} \\
&= \frac{\exp(\alpha + x^T \beta)}{\exp(\alpha + x^T \beta) + \exp(\log \frac{\pi_0}{\pi_1})} \\
&= \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta - \log \frac{\pi_0}{\pi_1})} \\
&= \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta - \log \frac{\pi_0}{\pi_1})}
\end{split}
```

This can be rewritten as:
```math
P(Y=1|Z=1, X=x) = \frac{\exp(\alpha^* + x^T \beta)}{1 + \exp(\alpha^* + x^T \beta)}
```

Where:
```math
\alpha^* = \alpha + \log \frac{\pi_1}{\pi_0}
```

### The Key Insight
**The coefficients $\beta$ remain the same!** Only the intercept changes by $\log \frac{\pi_1}{\pi_0}$.

## Practical Implications

### 1. **Coefficient Interpretation**
- The $\beta$ coefficients have the same interpretation as in random sampling
- They represent the log-odds ratio for a unit change in the predictor
- This is why logistic regression is robust to retrospective sampling

### 2. **Intercept Adjustment**
To get the population intercept $\alpha$ from the retrospective sample intercept $\alpha^*$:
```math
\alpha = \alpha^* - \log \frac{\pi_1}{\pi_0}
```

### 3. **Probability Estimation**
To estimate population probabilities:
```math
P(Y=1|X=x) = \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)}
```

Where $\alpha$ is the adjusted intercept.

## Implementation and Demonstration

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
        from sklearn.metrics import roc_curve
        
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

# Run demonstration
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

# Demonstrate different sampling ratios
print("\n=== Different Sampling Ratios ===")
sampling_ratios = [(50, 50), (100, 100), (200, 100), (100, 200)]

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
```

### R Implementation

```r
# Retrospective Sampling in Logistic Regression

# Load required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)

# Generate population data
generate_population_data <- function(n_population = 10000, prevalence = 0.01) {
  set.seed(42)
  
  # Generate features
  X <- matrix(rnorm(n_population * 3), n_population, 3)
  
  # True population parameters
  true_alpha <- -4.6  # Logit of 0.01 prevalence
  true_beta <- c(0.5, -0.3, 0.8)
  
  # Generate probabilities and outcomes
  z <- true_alpha + X %*% true_beta
  p <- 1 / (1 + exp(-z))
  y <- rbinom(n_population, 1, p)
  
  population_data <- data.frame(
    X1 = X[, 1],
    X2 = X[, 2],
    X3 = X[, 3],
    y = y,
    probability = p
  )
  
  cat("Population Summary:\n")
  cat("Total samples:", n_population, "\n")
  cat("Cases:", sum(y), "(", sum(y)/n_population, ")\n")
  cat("Controls:", n_population - sum(y), "(", (n_population - sum(y))/n_population, ")\n")
  
  return(list(data = population_data, true_alpha = true_alpha, true_beta = true_beta))
}

# Create retrospective sample
create_retrospective_sample <- function(population_data, n_cases = 100, n_controls = 100) {
  cases <- population_data[population_data$y == 1, ]
  controls <- population_data[population_data$y == 0, ]
  
  # Sample cases and controls
  sampled_cases <- cases[sample(nrow(cases), min(n_cases, nrow(cases))), ]
  sampled_controls <- controls[sample(nrow(controls), min(n_controls, nrow(controls))), ]
  
  retrospective_data <- rbind(sampled_cases, sampled_controls)
  
  cat("\nRetrospective Sample Summary:\n")
  cat("Cases:", nrow(sampled_cases), "\n")
  cat("Controls:", nrow(sampled_controls), "\n")
  cat("Total:", nrow(retrospective_data), "\n")
  
  return(retrospective_data)
}

# Fit models and compare
compare_models <- function(population_data, retrospective_data, true_alpha, true_beta) {
  # Population model
  pop_model <- glm(y ~ X1 + X2 + X3, data = population_data, family = binomial)
  
  # Retrospective model
  retro_model <- glm(y ~ X1 + X2 + X3, data = retrospective_data, family = binomial)
  
  # Extract coefficients
  pop_coef <- coef(pop_model)
  retro_coef <- coef(retro_model)
  
  # Create comparison table
  comparison <- data.frame(
    True = c(true_alpha, true_beta),
    Population = pop_coef,
    Retrospective = retro_coef,
    Difference = retro_coef - pop_coef
  )
  rownames(comparison) <- c("Intercept", "X1", "X2", "X3")
  
  print("=== Model Comparison ===")
  print(round(comparison, 4))
  
  # Theoretical intercept adjustment
  n_cases <- sum(retrospective_data$y == 1)
  n_controls <- sum(retrospective_data$y == 0)
  n_population <- nrow(population_data)
  n_cases_pop <- sum(population_data$y == 1)
  n_controls_pop <- nrow(population_data) - sum(population_data$y == 1)
  
  pi_1 <- n_cases / n_cases_pop  # Sampling probability for cases
  pi_0 <- n_controls / n_controls_pop  # Sampling probability for controls
  
  theoretical_adjustment <- log(pi_1 / pi_0)
  actual_adjustment <- retro_coef[1] - pop_coef[1]
  
  cat("\nSampling Probabilities:\n")
  cat("π₁ (cases):", pi_1, "\n")
  cat("π₀ (controls):", pi_0, "\n")
  cat("log(π₁/π₀):", theoretical_adjustment, "\n")
  cat("Actual intercept difference:", actual_adjustment, "\n")
  cat("Difference:", abs(theoretical_adjustment - actual_adjustment), "\n")
  
  return(list(comparison = comparison, 
              pop_model = pop_model, 
              retro_model = retro_model,
              theoretical_adjustment = theoretical_adjustment))
}

# Adjust probabilities
adjust_probabilities <- function(retro_probs, population_data, retrospective_data) {
  # Calculate sampling probabilities
  n_cases <- sum(retrospective_data$y == 1)
  n_controls <- sum(retrospective_data$y == 0)
  n_cases_pop <- sum(population_data$y == 1)
  n_controls_pop <- nrow(population_data) - sum(population_data$y == 1)
  
  pi_1 <- n_cases / n_cases_pop
  pi_0 <- n_controls / n_controls_pop
  
  # Adjust probabilities
  adjusted_probs <- retro_probs / (retro_probs + (1 - retro_probs) * pi_0 / pi_1)
  
  return(adjusted_probs)
}

# Evaluate predictions
evaluate_predictions <- function(population_data, pop_model, retro_model) {
  # Predictions from both models
  pop_pred_proba <- predict(pop_model, newdata = population_data, type = "response")
  retro_pred_proba <- predict(retro_model, newdata = population_data, type = "response")
  
  # Adjust retrospective predictions
  adjusted_pred_proba <- adjust_probabilities(retro_pred_proba, population_data, 
                                             data.frame(y = c(rep(1, 100), rep(0, 100))))
  
  # Calculate metrics
  results <- list()
  
  for (name in c("Population", "Retrospective", "Adjusted")) {
    if (name == "Population") {
      pred_proba <- pop_pred_proba
    } else if (name == "Retrospective") {
      pred_proba <- retro_pred_proba
    } else {
      pred_proba <- adjusted_pred_proba
    }
    
    pred_class <- ifelse(pred_proba >= 0.5, 1, 0)
    accuracy <- mean(pred_class == population_data$y)
    
    # Calculate AUC
    library(pROC)
    auc <- auc(population_data$y, pred_proba)
    
    results[[name]] <- list(
      Accuracy = accuracy,
      AUC = auc,
      Mean_Probability = mean(pred_proba)
    )
  }
  
  print("\n=== Prediction Performance ===")
  results_df <- do.call(rbind, lapply(results, function(x) {
    data.frame(Accuracy = x$Accuracy, AUC = x$AUC, Mean_Probability = x$Mean_Probability)
  }))
  rownames(results_df) <- names(results)
  print(round(results_df, 4))
  
  return(list(results = results, 
              pop_pred_proba = pop_pred_proba,
              retro_pred_proba = retro_pred_proba,
              adjusted_pred_proba = adjusted_pred_proba))
}

# Visualize results
visualize_comparison <- function(population_data, comparison, pop_pred_proba, 
                                retro_pred_proba, adjusted_pred_proba) {
  # 1. Coefficient comparison
  p1 <- ggplot(comparison, aes(x = rownames(comparison))) +
    geom_bar(aes(y = True, fill = "True"), stat = "identity", alpha = 0.8, position = position_dodge(0.8)) +
    geom_bar(aes(y = Retrospective, fill = "Retrospective"), stat = "identity", alpha = 0.8, position = position_dodge(0.8)) +
    scale_fill_manual(values = c("True" = "blue", "Retrospective" = "red")) +
    labs(title = "Coefficient Comparison", x = "Parameters", y = "Coefficient Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 2. Probability distributions
  prob_data <- data.frame(
    Probability = c(pop_pred_proba, retro_pred_proba),
    Model = rep(c("Population", "Retrospective"), each = length(pop_pred_proba))
  )
  
  p2 <- ggplot(prob_data, aes(x = Probability, fill = Model)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 50) +
    labs(title = "Probability Distributions", x = "Predicted Probability", y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 3. Adjusted vs Population probabilities
  p3 <- ggplot(data.frame(Population = pop_pred_proba, Adjusted = adjusted_pred_proba), 
               aes(x = Population, y = Adjusted)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Adjusted vs Population Probabilities") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # 4. ROC curves
  library(pROC)
  
  roc_pop <- roc(population_data$y, pop_pred_proba)
  roc_retro <- roc(population_data$y, retro_pred_proba)
  roc_adj <- roc(population_data$y, adjusted_pred_proba)
  
  roc_data <- data.frame(
    FPR = c(roc_pop$specificities, roc_retro$specificities, roc_adj$specificities),
    TPR = c(roc_pop$sensitivities, roc_retro$sensitivities, roc_adj$sensitivities),
    Model = rep(c("Population", "Retrospective", "Adjusted"), 
                c(length(roc_pop$specificities), length(roc_retro$specificities), length(roc_adj$specificities)))
  )
  
  p4 <- ggplot(roc_data, aes(x = 1 - FPR, y = TPR, color = Model)) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", alpha = 0.5) +
    labs(title = "ROC Curves", x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
}

# Run demonstration
result <- generate_population_data(n_population = 10000, prevalence = 0.01)
population_data <- result$data
true_alpha <- result$true_alpha
true_beta <- result$true_beta

retrospective_data <- create_retrospective_sample(population_data, n_cases = 100, n_controls = 100)

model_comparison <- compare_models(population_data, retrospective_data, true_alpha, true_beta)

evaluation <- evaluate_predictions(population_data, model_comparison$pop_model, model_comparison$retro_model)

visualize_comparison(population_data, model_comparison$comparison, 
                    evaluation$pop_pred_proba, evaluation$retro_pred_proba, evaluation$adjusted_pred_proba)

# Demonstrate different sampling ratios
cat("\n=== Different Sampling Ratios ===\n")
sampling_ratios <- list(c(50, 50), c(100, 100), c(200, 100), c(100, 200))

for (ratio in sampling_ratios) {
  n_cases <- ratio[1]
  n_controls <- ratio[2]
  
  cat("\nSampling", n_cases, "cases and", n_controls, "controls:\n")
  
  # Create new retrospective sample
  retro_data <- create_retrospective_sample(population_data, n_cases = n_cases, n_controls = n_controls)
  
  # Fit model
  retro_model <- glm(y ~ X1 + X2 + X3, data = retro_data, family = binomial)
  
  # Compare coefficients
  pop_coef <- coef(model_comparison$pop_model)
  retro_coef <- coef(retro_model)
  
  coef_diff <- sqrt(sum((pop_coef[-1] - retro_coef[-1])^2))  # Exclude intercept
  intercept_diff <- retro_coef[1] - pop_coef[1]
  
  cat("  Coefficient difference:", round(coef_diff, 4), "\n")
  cat("  Intercept difference:", round(intercept_diff, 4), "\n")
}
```

## Key Insights

### 1. **Coefficient Invariance**
The most important result is that the $\beta$ coefficients are invariant to retrospective sampling. This means:
- The relationship between predictors and outcome is preserved
- We can interpret coefficients the same way as in random sampling
- The model's discriminative ability is maintained

### 2. **Intercept Adjustment**
Only the intercept needs adjustment:
```math
\alpha_{\text{population}} = \alpha_{\text{retrospective}} - \log \frac{\pi_1}{\pi_0}
```

### 3. **Probability Calibration**
To get population probabilities from retrospective probabilities:
```math
P(Y=1|X=x) = \frac{P(Y=1|Z=1, X=x)}{P(Y=1|Z=1, X=x) + (1 - P(Y=1|Z=1, X=x)) \frac{\pi_0}{\pi_1}}
```

### 4. **Practical Considerations**
- **Sample Size**: Retrospective sampling allows efficient study of rare outcomes
- **Bias**: No bias in coefficients, only in intercept and probabilities
- **Calibration**: Probabilities need adjustment for population inference

## Applications

### 1. **Medical Research**
- Case-control studies for rare diseases
- Drug safety studies
- Epidemiological research

### 2. **Fraud Detection**
- Studying rare fraud cases
- Credit card fraud detection
- Insurance fraud analysis

### 3. **Quality Control**
- Defect detection in manufacturing
- Anomaly detection in systems
- Rare event prediction

## Summary

Retrospective sampling is a powerful tool for studying rare outcomes, and logistic regression handles it elegantly:

1. **Coefficients remain unbiased** - the core relationships are preserved
2. **Only intercept needs adjustment** - simple correction formula
3. **Probabilities can be calibrated** - for population inference
4. **Model performance is maintained** - discriminative ability preserved

This makes logistic regression particularly robust for retrospective studies, which are common in medical research and other fields where outcomes are rare.
