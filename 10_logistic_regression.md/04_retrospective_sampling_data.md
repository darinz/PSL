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

The complete implementation and demonstration of retrospective sampling in logistic regression is provided in the code files:

**Python Implementation:** See `RetrospectiveSamplingDemo` class and comprehensive demonstrations in [`code/retrospective_sampling_implementation.py`](code/retrospective_sampling_implementation.py)

**R Implementation:** See analysis functions and demonstrations in [`code/r_retrospective_sampling_implementation.R`](code/r_retrospective_sampling_implementation.R)

These implementations include:

- **RetrospectiveSamplingDemo Class**: Complete implementation for analyzing retrospective sampling
- **Population Data Generation**: Generation of population data with specified prevalence
- **Retrospective Sampling**: Creation of case-control samples with specified ratios
- **Model Comparison**: Systematic comparison between population and retrospective models
- **Coefficient Analysis**: Analysis of coefficient invariance and intercept adjustment
- **Probability Calibration**: Methods for adjusting retrospective probabilities to population scale
- **Performance Evaluation**: Comprehensive evaluation of model performance
- **Visualization Tools**: Data visualization and model comparison plotting
- **Comprehensive Demonstrations**: 
  - Basic retrospective sampling analysis
  - Different sampling ratio effects
  - Prevalence effects on model performance
  - Probability calibration methods
  - Theoretical derivation verification
  - Practical applications demonstration
  - Limitations and cautions analysis

The implementations provide hands-on experience with retrospective sampling, demonstrating both the mathematical foundations and practical computational aspects of this important sampling strategy.




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
