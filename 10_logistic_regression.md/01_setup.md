# 10.1. Setup

## 10.1.0. Introduction

Logistic Regression is one of the most fundamental and widely-used classification methods in machine learning. Unlike Discriminant Analysis, which follows a generative approach by modeling class-conditional distributions, Logistic Regression takes a **discriminative approach** by directly modeling the posterior probability $`P(Y=1 | X=x)`$.

### Key Concepts

- **Discriminative vs Generative**: Logistic Regression directly models $`P(Y=1 | X=x)`$ without modeling the joint distribution
- **Link Function**: Transforms the constrained probability to an unconstrained space
- **Maximum Likelihood**: Uses log-likelihood as the objective function
- **Linear Decision Boundary**: Results in linear decision boundaries in the feature space

## 10.1.1. The Binary Classification Problem

### Problem Formulation

In binary classification, we have:
- **Features**: $`X \in \mathbb{R}^p`$ (p-dimensional feature vector)
- **Target**: $`Y \in \{0, 1\}`$ (binary outcome)
- **Goal**: Learn a function that predicts $`P(Y=1 | X=x)`$

### Optimal Classifier

From our previous discussions, we know that the **Bayes optimal classifier** for binary classification is:

```math
\hat{y} = \begin{cases} 
1 & \text{if } P(Y=1 | X=x) > 0.5 \\
0 & \text{otherwise}
\end{cases}
```

This means the optimal classifier depends entirely on the **posterior probability**:

```math
\eta(x) = P(Y=1 | X=x)
```

## 10.1.2. Direct Modeling Approach

### The Challenge

We want to directly model $`\eta(x)`$, but there's a fundamental challenge:

**Problem**: $`\eta(x)`$ is constrained to $`[0, 1]`$ (it's a probability), but linear models $`x^T \beta`$ are unconstrained and can output any real value.

**Solution**: Use a **link function** to transform the constrained probability to an unconstrained space.

### Link Function Framework

We model the transformation of $`\eta(x)`$ with a linear function:

```math
g(\eta(x)) = x^T \beta
```

Where:
- $`g(\cdot)`$ is the **link function** (transformation)
- $`x^T \beta`$ is the **linear predictor**
- $`\beta`$ includes the intercept (we assume $`x_0 = 1`$ for the intercept)

### The Inverse Transformation

To get back to probabilities, we apply the inverse link function:

```math
\eta(x) = g^{-1}(x^T \beta)
```

## 10.1.3. The Logit Link Function

### Definition

In Logistic Regression, we use the **logit** (log-odds) link function:

```math
g(\eta(x)) = \text{logit}(\eta(x)) = \log \frac{\eta(x)}{1 - \eta(x)}
```

### Properties of the Logit Function

The logit function has several important properties:

1. **Domain**: $`\eta(x) \in (0, 1)`$ → $`\text{logit}(\eta(x)) \in (-\infty, +\infty)`$
2. **Monotonicity**: Strictly increasing function
3. **Symmetry**: $`\text{logit}(p) = -\text{logit}(1-p)`$

### Key Values

Let's examine the behavior at key probability values:

```math
\begin{align}
\text{When } \eta(x) = 0.5 &: \text{logit}(0.5) = \log \frac{0.5}{0.5} = \log(1) = 0 \\
\text{When } \eta(x) > 0.5 &: \text{logit}(\eta(x)) > 0 \text{ (positive values)} \\
\text{When } \eta(x) < 0.5 &: \text{logit}(\eta(x)) < 0 \text{ (negative values)}
\end{align}
```

### Visualization of the Logit Function

The logit function visualization and its properties are implemented in the code files:

**Python Implementation:** See `visualize_logit_function()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `visualize_logit_function_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions create comprehensive visualizations showing:
- The logit function mapping probabilities to unconstrained values
- The sigmoid (inverse logit) function mapping linear predictors to probabilities
- The symmetry property of the logit function
- Decision boundary visualization for logistic regression

The visualizations demonstrate how the logit function transforms constrained probabilities (0,1) to unconstrained values (-∞, +∞), enabling the use of linear models for probability estimation.

## 10.1.4. The Sigmoid Function

### Inverse of the Logit

The inverse of the logit function is the **sigmoid** (logistic) function:

```math
\eta(x) = g^{-1}(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}} = \sigma(x^T \beta)
```

Where $`\sigma(z) = \frac{1}{1 + e^{-z}}`$ is the sigmoid function.

### Properties of the Sigmoid Function

1. **Range**: $`\sigma(z) \in (0, 1)`$ for all $`z \in \mathbb{R}`$
2. **Monotonicity**: Strictly increasing
3. **Symmetry**: $`\sigma(-z) = 1 - \sigma(z)`$
4. **Derivative**: $`\sigma'(z) = \sigma(z)(1 - \sigma(z))`$

### Mathematical Relationship

The complete Logistic Regression model is:

```math
P(Y=1 | X=x) = \eta(x) = \sigma(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}}
```

## 10.1.5. The Data and Parameters

### Data Structure

For each observation $`i = 1, 2, \ldots, n`$, we have:

- **Feature vector**: $`x_i \in \mathbb{R}^p`$ (including intercept $`x_{i0} = 1`$)
- **Binary outcome**: $`y_i \in \{0, 1\}`$
- **True probability**: $`\eta(x_i) = P(Y_i=1 | X_i=x_i)`$

### Unknown Parameters

The unknown parameter vector $`\beta \in \mathbb{R}^p`$ includes:
- $`\beta_0`$: Intercept term
- $`\beta_1, \beta_2, \ldots, \beta_{p-1}`$: Feature coefficients

### The Estimation Problem

Our goal is to estimate $`\beta`$ from the observed data $`\{(x_i, y_i)\}_{i=1}^n`$.

## 10.1.6. Loss Function Selection

### Why Not L2 Loss?

One might consider using the squared error loss:

```math
L_{\text{MSE}}(\beta) = \sum_{i=1}^n (y_i - \eta(x_i))^2
```

However, this has several limitations:

1. **Small Gradients**: Since $`|y_i - \eta(x_i)| \leq 1`$, squaring makes gradients very small
2. **Training Difficulties**: Small gradients make optimization slow and can lead to getting stuck
3. **Non-convexity**: The squared error loss is not convex for logistic regression

### The Log-Likelihood Approach

Instead, we use the **negative log-likelihood** as our loss function:

```math
L(\beta) = -\sum_{i=1}^n \log P(Y_i = y_i | X_i = x_i)
```

### Likelihood Function

For binary outcomes, the likelihood is:

```math
P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i}
```

This can be written more compactly as:

```math
P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i}
```

### Log-Likelihood

Taking the logarithm:

```math
\begin{split}
\log P(Y_i = y_i | X_i = x_i) &= y_i \log \eta(x_i) + (1 - y_i) \log(1 - \eta(x_i)) \\
&= y_i \log \frac{\eta(x_i)}{1 - \eta(x_i)} + \log(1 - \eta(x_i)) \\
&= y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta})
\end{split}
```

### Final Loss Function

The negative log-likelihood loss function is:

```math
L(\beta) = -\sum_{i=1}^n \left[ y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta}) \right]
```

## 10.1.7. Comparison of Loss Functions

### Visual Comparison

The comparison of MSE and log-likelihood loss functions is implemented in the code files:

**Python Implementation:** See `compare_loss_functions()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `compare_loss_functions_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions demonstrate the key differences between MSE and log-likelihood loss functions for logistic regression:

- **MSE Loss**: Shows flat gradients and poor optimization properties
- **Log-Likelihood Loss**: Provides meaningful gradients and better optimization characteristics
- **Gradient Comparison**: Quantifies the difference in gradient magnitudes
- **Visual Analysis**: Side-by-side plots showing loss landscapes

The comparison reveals why log-likelihood is preferred over MSE for logistic regression, as it provides better optimization properties and statistical foundations.

## 10.1.8. Advantages of Log-Likelihood

### Why Log-Likelihood is Better

1. **Convexity**: The negative log-likelihood is convex, ensuring global optimality
2. **Proper Gradients**: Provides meaningful gradients for optimization
3. **Statistical Foundation**: Based on maximum likelihood estimation
4. **Interpretability**: Directly related to probability modeling

### Mathematical Properties

The log-likelihood function has several desirable properties:

1. **Convexity**: The Hessian matrix is positive semi-definite
2. **Uniqueness**: Under mild conditions, the maximum likelihood estimator is unique
3. **Asymptotic Properties**: MLE is consistent and asymptotically normal

## 10.1.9. Summary and Next Steps

### What We've Established

1. **Problem Setup**: Binary classification with direct probability modeling
2. **Link Function**: Logit transformation to handle probability constraints
3. **Model Form**: $`P(Y=1 | X=x) = \sigma(x^T \beta)`$
4. **Loss Function**: Negative log-likelihood for optimization

### Key Insights

- **Discriminative Approach**: Direct modeling of $`P(Y=1 | X=x)`$
- **Link Function**: Transforms constrained probabilities to unconstrained space
- **Loss Selection**: Log-likelihood provides better optimization properties than MSE

### Next Steps

In the following sections, we will:
1. **Parameter Estimation**: Derive the maximum likelihood estimator
2. **Optimization**: Implement gradient-based optimization algorithms
3. **Model Evaluation**: Assess model performance and interpretability
4. **Extensions**: Handle multi-class classification and regularization

### Implementation Preview

The complete logistic regression setup demonstration is implemented in the code files:

**Python Implementation:** See `logistic_regression_setup_demo()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `logistic_regression_setup_demo_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions provide a comprehensive demonstration of the logistic regression setup:

- **Data Generation**: Synthetic binary classification data with known parameters
- **Visualization**: Scatter plots showing class separation and probability distributions
- **Parameter Analysis**: Examination of true parameters and class balance
- **Setup Summary**: Complete overview of the problem setup

The demonstration shows how logistic regression transforms linear predictors into probabilities through the sigmoid function, creating a complete framework for binary classification.

This setup provides the foundation for understanding logistic regression as both a probabilistic model and an optimization problem, setting the stage for maximum likelihood estimation and practical applications.

---

**Navigation:**
- **Next Topic:** [Maximum Likelihood Estimation](02_mle.md) - Likelihood function, optimization, and parameter estimation
- **Previous Topic:** [Logistic Regression Overview](README.md) - Overview of logistic regression concepts and applications
