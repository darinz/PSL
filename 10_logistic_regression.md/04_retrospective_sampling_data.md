# 10.4. Retrospective Sampling in Logistic Regression

## Introduction

Retrospective sampling (also known as case-control sampling) is a common data collection strategy in medical research, epidemiology, and other fields where the outcome of interest is rare. This sampling method creates unique challenges for logistic regression that every practitioner should understand.

**Intuitive Understanding**: Retrospective sampling is like being a detective who studies crime scenes by focusing on solved cases rather than randomly investigating the entire city. Instead of randomly sampling from the population (which would be like randomly knocking on doors), we deliberately collect a specific number of cases (people with the disease) and controls (people without the disease). This is much more efficient when the disease is rare - like studying a rare cancer that affects only 1% of people. We can get 100 cancer patients and 100 healthy people much faster than randomly sampling 10,000 people to find 100 cancer cases.

### Why This Matters

**Intuition**: This sampling strategy is crucial because many important outcomes in real life are rare. Think about rare diseases, fraud cases, or manufacturing defects - they might only occur in 1% or less of the population. If we had to randomly sample to study these rare events, we'd need enormous sample sizes. Retrospective sampling lets us study rare outcomes efficiently, but it creates mathematical challenges that we need to understand and handle properly.

## What is Retrospective Sampling?

### Definition
**Retrospective sampling** is a sampling strategy where we sample based on the outcome variable rather than randomly from the population. Specifically:
- We sample a fixed number of cases (individuals with the outcome of interest, e.g., cancer patients)
- We sample a fixed number of controls (individuals without the outcome, e.g., healthy individuals)
- The sampling is independent of the predictor variables

**Intuition**: This is like having two separate recruitment processes. First, we go to hospitals and recruit 100 cancer patients (cases). Then, we go to the general population and recruit 100 healthy people (controls). The key insight is that we're not sampling randomly from the whole population - we're deliberately choosing equal numbers of cases and controls, regardless of how rare the disease is in the real population.

### Motivation
Consider a rare disease that affects only 1% of the population. To study this disease:
- **Random sampling**: We would need to sample ~10,000 people to get ~100 cases
- **Retrospective sampling**: We can directly sample 100 cases and 100 controls

This makes retrospective sampling much more efficient for rare outcomes.

**Intuition**: This efficiency gain is enormous! Instead of spending months and huge resources to find 100 rare disease cases through random sampling, we can get them directly from hospitals in a few weeks. This is why case-control studies are so popular in medical research - they make studying rare diseases practical and affordable.

## Mathematical Foundation

### Population Model
In the population, we assume the true relationship follows:
$$ P(Y=1|X=x) = \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)} $$

Where:
- $`Y`$ is the binary outcome (1 = case, 0 = control) - like having the disease
- $`X`$ are the predictor variables - like age, smoking status, family history
- $`\alpha`$ is the intercept - like the baseline risk of disease in the population
- $`\beta`$ are the coefficients - like how much each risk factor increases disease probability

**Intuition**: This is our "true" model of how disease probability relates to risk factors in the entire population. The intercept α represents the baseline disease risk, and the coefficients β tell us how much each risk factor (like smoking, age, etc.) increases or decreases the disease probability.

### Sampling Process
Let $`Z`$ be an indicator variable for whether an individual is sampled:
$$ Z = \begin{cases}
1 & \text{if individual is sampled} \\
0 & \text{if individual is not sampled}
\end{cases} $$

In retrospective sampling:
- $`P(Z=1|Y=1) = \pi_1`$ (sampling probability for cases) - like the probability of recruiting a cancer patient
- $`P(Z=1|Y=0) = \pi_0`$ (sampling probability for controls) - like the probability of recruiting a healthy person

**Intuition**: These sampling probabilities represent how we recruit our study participants. If we want equal numbers of cases and controls, we might set π₁ = π₀ = 0.5, meaning we have a 50% chance of recruiting any given cancer patient and a 50% chance of recruiting any given healthy person. The key insight is that these sampling probabilities are typically much higher than the disease prevalence in the population.

### The Problem
We want to estimate $`P(Y=1|X=x)`$, but our data gives us $`P(Y=1|Z=1, X=x)`$.

**Intuition**: This is the core problem! We want to know the true disease probability in the population given certain risk factors, but our data only tells us the disease probability among the people we happened to sample. Since we deliberately sampled more cases than would occur naturally, our sample is biased toward higher disease prevalence.

## Mathematical Derivation

### Bayes' Theorem Application
Using Bayes' theorem:
$$ P(Y=1|Z=1, X=x) = \frac{P(Z=1|Y=1, X=x) P(Y=1|X=x)}{P(Z=1|X=x)} $$

Since sampling is independent of $`X`$ given $`Y`$:
$$ P(Z=1|Y=1, X=x) = P(Z=1|Y=1) = \pi_1 $$

And:
$$ P(Z=1|X=x) = P(Z=1|Y=1, X=x) P(Y=1|X=x) + P(Z=1|Y=0, X=x) P(Y=0|X=x) $$

Substituting:
$$ P(Z=1|X=x) = \pi_1 P(Y=1|X=x) + \pi_0 P(Y=0|X=x) $$

**Intuition**: This step uses Bayes' theorem to relate what we observe (disease probability in our sample) to what we want to know (disease probability in the population). The key assumption is that our sampling doesn't depend on the risk factors - we're equally likely to sample a smoker or non-smoker with cancer, and equally likely to sample a smoker or non-smoker without cancer.

### Retrospective Probability
Therefore:
$$ \begin{split}
P(Y=1|Z=1, X=x) &= \frac{\pi_1 P(Y=1|X=x)}{\pi_1 P(Y=1|X=x) + \pi_0 P(Y=0|X=x)} \\
&= \frac{\pi_1 P(Y=1|X=x)}{\pi_1 P(Y=1|X=x) + \pi_0 (1 - P(Y=1|X=x))}
\end{split} $$

**Intuition**: This formula shows how the disease probability in our sample relates to the disease probability in the population. The numerator represents the probability of sampling a diseased person with these risk factors, and the denominator represents the total probability of sampling anyone with these risk factors.

### Key Result
Substituting the logistic model:
$$ \begin{split}
P(Y=1|Z=1, X=x) &= \frac{\pi_1 \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)}}{\pi_1 \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)} + \pi_0 \frac{1}{1 + \exp(\alpha + x^T \beta)}} \\
&= \frac{\pi_1 \exp(\alpha + x^T \beta)}{\pi_1 \exp(\alpha + x^T \beta) + \pi_0} \\
&= \frac{\exp(\alpha + x^T \beta)}{\exp(\alpha + x^T \beta) + \frac{\pi_0}{\pi_1}} \\
&= \frac{\exp(\alpha + x^T \beta)}{\exp(\alpha + x^T \beta) + \exp(\log \frac{\pi_0}{\pi_1})} \\
&= \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta - \log \frac{\pi_0}{\pi_1})} \\
&= \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta - \log \frac{\pi_0}{\pi_1})}
\end{split} $$

This can be rewritten as:
$$ P(Y=1|Z=1, X=x) = \frac{\exp(\alpha^* + x^T \beta)}{1 + \exp(\alpha^* + x^T \beta)} $$

Where:
$$ \alpha^* = \alpha + \log \frac{\pi_1}{\pi_0} $$

### The Key Insight
**The coefficients $`\beta`$ remain the same!** Only the intercept changes by $`\log \frac{\pi_1}{\pi_0}`$.

**Intuition**: This is the beautiful result! Even though we sampled cases and controls in a biased way, the relationship between risk factors and disease (the β coefficients) stays exactly the same. Only the baseline risk (the intercept) changes. This means we can still interpret how much smoking increases cancer risk, even though our sample has artificially high cancer prevalence.

## Practical Implications

### 1. **Coefficient Interpretation**
- The $`\beta`$ coefficients have the same interpretation as in random sampling
- They represent the log-odds ratio for a unit change in the predictor
- This is why logistic regression is robust to retrospective sampling

**Intuition**: This is why logistic regression is so powerful for case-control studies! We can still say "smoking increases cancer risk by X times" even though our sample has artificially high cancer rates. The relationship between smoking and cancer doesn't change just because we sampled more cancer patients.

### 2. **Intercept Adjustment**
To get the population intercept $`\alpha`$ from the retrospective sample intercept $`\alpha^*`$:
$$ \alpha = \alpha^* - \log \frac{\pi_1}{\pi_0} $$

**Intuition**: If we want to estimate the true baseline disease risk in the population, we need to adjust for our sampling bias. If we sampled equal numbers of cases and controls (π₁ = π₀), then log(π₁/π₀) = 0, and no adjustment is needed. But if we sampled more cases than controls, we need to subtract this bias from our estimated intercept.

### 3. **Probability Estimation**
To estimate population probabilities:
$$ P(Y=1|X=x) = \frac{\exp(\alpha + x^T \beta)}{1 + \exp(\alpha + x^T \beta)} $$

Where $`\alpha`$ is the adjusted intercept.

**Intuition**: Once we've adjusted the intercept, we can estimate true disease probabilities in the population. This is crucial for public health applications where we want to know the actual risk for different groups of people.

## Implementation and Demonstration

The complete implementation and demonstration of retrospective sampling in logistic regression is provided in the code files:

**Python Implementation:** See `RetrospectiveSamplingDemo` class and comprehensive demonstrations in [`code/retrospective_sampling_implementation.py`](code/retrospective_sampling_implementation.py)

**R Implementation:** See analysis functions and demonstrations in [`code/r_retrospective_sampling_implementation.R`](code/r_retrospective_sampling_implementation.R)

These implementations include:

- **RetrospectiveSamplingDemo Class**: Complete implementation for analyzing retrospective sampling - like having a complete system for understanding case-control studies
- **Population Data Generation**: Generation of population data with specified prevalence - like creating realistic disease scenarios
- **Retrospective Sampling**: Creation of case-control samples with specified ratios - like simulating the actual sampling process
- **Model Comparison**: Systematic comparison between population and retrospective models - like comparing what we get vs what we want
- **Coefficient Analysis**: Analysis of coefficient invariance and intercept adjustment - like verifying that relationships stay the same
- **Probability Calibration**: Methods for adjusting retrospective probabilities to population scale - like converting sample probabilities to population probabilities
- **Performance Evaluation**: Comprehensive evaluation of model performance - like testing how well our adjustments work
- **Visualization Tools**: Data visualization and model comparison plotting - like seeing the differences between population and sample models
- **Comprehensive Demonstrations**: 
  - Basic retrospective sampling analysis - like understanding the fundamental problem
  - Different sampling ratio effects - like seeing how different sampling strategies affect results
  - Prevalence effects on model performance - like understanding how disease rarity affects our studies
  - Probability calibration methods - like learning how to adjust for sampling bias
  - Theoretical derivation verification - like confirming our mathematical results
  - Practical applications demonstration - like seeing real-world examples
  - Limitations and cautions analysis - like understanding when this approach might fail

The implementations provide hands-on experience with retrospective sampling, demonstrating both the mathematical foundations and practical computational aspects of this important sampling strategy.

## Key Insights

### 1. **Coefficient Invariance**
The most important result is that the $`\beta`$ coefficients are invariant to retrospective sampling. This means:
- The relationship between predictors and outcome is preserved
- We can interpret coefficients the same way as in random sampling
- The model's discriminative ability is maintained

**Intuition**: This invariance is like discovering that the relationship between smoking and cancer doesn't change whether we study 100 cancer patients or 10,000 random people. The core biological relationship is preserved, even though our sampling method is biased.

### 2. **Intercept Adjustment**
Only the intercept needs adjustment:
$$ \alpha_{\text{population}} = \alpha_{\text{retrospective}} - \log \frac{\pi_1}{\pi_0} $$

**Intuition**: This simple formula is the key to correcting our sampling bias. If we sampled equal numbers of cases and controls, no adjustment is needed. But if we sampled more cases than controls, we need to reduce our estimated baseline risk accordingly.

### 3. **Probability Calibration**
To get population probabilities from retrospective probabilities:
$$ P(Y=1|X=x) = \frac{P(Y=1|Z=1, X=x)}{P(Y=1|Z=1, X=x) + (1 - P(Y=1|Z=1, X=x)) \frac{\pi_0}{\pi_1}} $$

**Intuition**: This formula lets us convert our biased sample probabilities back to true population probabilities. It's like having a "correction factor" that undoes our sampling bias.

### 4. **Practical Considerations**
- **Sample Size**: Retrospective sampling allows efficient study of rare outcomes
- **Bias**: No bias in coefficients, only in intercept and probabilities
- **Calibration**: Probabilities need adjustment for population inference

**Intuition**: These practical considerations help us understand when and how to use retrospective sampling. It's perfect for studying rare diseases, but we need to be careful about interpreting probabilities and always adjust for our sampling bias.

## Applications

### 1. **Medical Research**
- Case-control studies for rare diseases
- Drug safety studies
- Epidemiological research

**Intuition**: Medical research is the classic application. When studying rare cancers, genetic disorders, or adverse drug reactions, retrospective sampling is often the only practical approach. We can't wait to randomly sample enough cases - we need to go directly to hospitals and recruit patients.

### 2. **Fraud Detection**
- Studying rare fraud cases
- Credit card fraud detection
- Insurance fraud analysis

**Intuition**: Fraud is typically rare (maybe 1% of transactions), so studying it through random sampling would require enormous datasets. Instead, we can deliberately collect fraud cases and compare them to legitimate transactions.

### 3. **Quality Control**
- Defect detection in manufacturing
- Anomaly detection in systems
- Rare event prediction

**Intuition**: Manufacturing defects, system failures, and other rare events are perfect candidates for retrospective sampling. We can study the few defective products or failed systems and compare them to the many successful ones.

## Summary

Retrospective sampling is a powerful tool for studying rare outcomes, and logistic regression handles it elegantly:

1. **Coefficients remain unbiased** - the core relationships are preserved
2. **Only intercept needs adjustment** - simple correction formula
3. **Probabilities can be calibrated** - for population inference
4. **Model performance is maintained** - discriminative ability preserved

This makes logistic regression particularly robust for retrospective studies, which are common in medical research and other fields where outcomes are rare.

**Intuition**: The beauty of logistic regression is that it's robust to this type of sampling bias. Unlike many other statistical methods, we can still get valid estimates of risk factor effects even when our sampling is deliberately biased. This makes it the go-to method for case-control studies and other retrospective designs.

This understanding is crucial for practitioners working with case-control studies, as it ensures that logistic regression results are properly interpreted and that the odds ratio estimates are valid despite the sampling design.

---

**Navigation:**
- **Next Topic:** *This is the last topic in the logistic regression section*
- **Previous Topic:** [Separable Data](03_seperable_data.md) - Handling perfectly separable data and convergence issues
