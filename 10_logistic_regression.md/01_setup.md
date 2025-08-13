# 10.1. Setup

## 10.1.0. Introduction

Logistic Regression is one of the most fundamental and widely-used classification methods in machine learning. Unlike Discriminant Analysis, which follows a generative approach by modeling class-conditional distributions, Logistic Regression takes a **discriminative approach** by directly modeling the posterior probability $`P(Y=1 | X=x)`$.

**Intuitive Understanding**: Logistic Regression is like being a doctor who learns to directly estimate the probability of a disease given a patient's symptoms, without needing to understand the complete underlying disease mechanisms. Imagine you're trying to predict whether a patient has diabetes based on their age, weight, and blood sugar levels. Instead of modeling how these factors relate to the disease process (generative approach), you directly learn the probability of diabetes given these symptoms (discriminative approach). It's like having a smart calculator that takes symptoms as input and outputs a probability of disease.

### Key Concepts

- **Discriminative vs Generative**: Logistic Regression directly models $`P(Y=1 | X=x)`$ without modeling the joint distribution - like directly learning disease probability from symptoms rather than understanding the disease process
- **Link Function**: Transforms the constrained probability to an unconstrained space - like converting a probability (0-1) into any number (-∞ to +∞) so we can use linear models
- **Maximum Likelihood**: Uses log-likelihood as the objective function - like finding the most likely explanation for the data we observed
- **Linear Decision Boundary**: Results in linear decision boundaries in the feature space - like drawing straight lines to separate healthy from diseased patients

**Intuition**: These concepts work together to create a powerful but simple classification system. The discriminative approach focuses on what we really care about (disease probability), the link function lets us use simple linear models, maximum likelihood finds the best explanation for our data, and linear boundaries make the system easy to understand and use.

## 10.1.1. The Binary Classification Problem

### Problem Formulation

In binary classification, we have:
- **Features**: $`X \in \mathbb{R}^p`$ (p-dimensional feature vector) - like a patient's symptoms and test results
- **Target**: $`Y \in \{0, 1\}`$ (binary outcome) - like whether the patient has the disease (1) or not (0)
- **Goal**: Learn a function that predicts $`P(Y=1 | X=x)`$ - like learning to estimate disease probability from symptoms

**Intuition**: This is like learning to be a diagnostic expert who can look at a patient's symptoms and give a probability estimate of whether they have a particular disease. The features are all the information we have about the patient, and the target is the true disease status.

### Optimal Classifier

From our previous discussions, we know that the **Bayes optimal classifier** for binary classification is:

$$ \hat{y} = \begin{cases} 
1 & \text{if } P(Y=1 | X=x) > 0.5 \\
0 & \text{otherwise}
\end{cases} $$

This means the optimal classifier depends entirely on the **posterior probability**:

$$ \eta(x) = P(Y=1 | X=x) $$

**Intuition**: The optimal decision rule is simple: if the probability of disease is greater than 50%, we predict disease; otherwise, we predict no disease. This makes perfect sense - we choose the more likely outcome. The key insight is that we only need to know the probability of the positive class (disease) to make optimal decisions.

## 10.1.2. Direct Modeling Approach

### The Challenge

We want to directly model $`\eta(x)`$, but there's a fundamental challenge:

**Problem**: $`\eta(x)`$ is constrained to $`[0, 1]`$ (it's a probability), but linear models $`x^T \beta`$ are unconstrained and can output any real value.

**Intuition**: This is like trying to use a thermometer (which can measure any temperature) to measure something that can only be between 0 and 1 (like a probability). We need a way to transform the constrained probability into an unconstrained space where linear models can work.

**Solution**: Use a **link function** to transform the constrained probability to an unconstrained space.

### Link Function Framework

We model the transformation of $`\eta(x)`$ with a linear function:

$$ g(\eta(x)) = x^T \beta $$

Where:
- $`g(\cdot)`$ is the **link function** (transformation) - like a converter that transforms probabilities to any number
- $`x^T \beta`$ is the **linear predictor** - like a simple weighted sum of symptoms
- $`\beta`$ includes the intercept (we assume $`x_0 = 1`$ for the intercept) - like the weights for each symptom plus a baseline

**Intuition**: This framework says that we can transform the disease probability using some function, and this transformed value should be a simple linear combination of the patient's symptoms. The link function is like a translator that converts between the language of probabilities and the language of linear combinations.

### The Inverse Transformation

To get back to probabilities, we apply the inverse link function:

$$ \eta(x) = g^{-1}(x^T \beta) $$

**Intuition**: Once we have the linear combination of symptoms, we need to transform it back to a probability. This is like having a two-way translator - we can go from probabilities to linear combinations and back again.

## 10.1.3. The Logit Link Function

### Definition

In Logistic Regression, we use the **logit** (log-odds) link function:

$$ g(\eta(x)) = \text{logit}(\eta(x)) = \log \frac{\eta(x)}{1 - \eta(x)} $$

**Intuition**: The logit function transforms a probability into "log-odds." If the probability of disease is 0.8, the odds are 4:1 (0.8/0.2), and the log-odds are log(4) ≈ 1.39. This transformation takes probabilities between 0 and 1 and maps them to any real number.

### Properties of the Logit Function

The logit function has several important properties:

1. **Domain**: $`\eta(x) \in (0, 1)`$ → $`\text{logit}(\eta(x)) \in (-\infty, +\infty)`$ - like transforming a constrained probability into any possible number
2. **Monotonicity**: Strictly increasing function - like higher probabilities always giving higher log-odds
3. **Symmetry**: $`\text{logit}(p) = -\text{logit}(1-p)`$ - like the log-odds for disease being the negative of the log-odds for no disease

**Intuition**: These properties make the logit function perfect for our needs. It transforms probabilities to any real number (solving our constraint problem), preserves the ordering (higher probabilities give higher log-odds), and has nice symmetry properties.

### Key Values

Let's examine the behavior at key probability values:

$$ \begin{align}
\text{When } \eta(x) = 0.5 &: \text{logit}(0.5) = \log \frac{0.5}{0.5} = \log(1) = 0 \\
\text{When } \eta(x) > 0.5 &: \text{logit}(\eta(x)) > 0 \text{ (positive values)} \\
\text{When } \eta(x) < 0.5 &: \text{logit}(\eta(x)) < 0 \text{ (negative values)}
\end{align} $$

**Intuition**: This shows that the logit function has a natural interpretation. When the probability is 50-50, the log-odds are zero. When the probability is higher than 50%, the log-odds are positive (favoring disease). When the probability is lower than 50%, the log-odds are negative (favoring no disease).

### Visualization of the Logit Function

The logit function visualization and its properties are implemented in the code files:

**Python Implementation:** See `visualize_logit_function()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `visualize_logit_function_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions create comprehensive visualizations showing:
- The logit function mapping probabilities to unconstrained values - like seeing how probabilities transform to log-odds
- The sigmoid (inverse logit) function mapping linear predictors to probabilities - like seeing how symptom combinations transform back to probabilities
- The symmetry property of the logit function - like seeing the mirror-image relationship
- Decision boundary visualization for logistic regression - like seeing how the model separates patients

The visualizations demonstrate how the logit function transforms constrained probabilities (0,1) to unconstrained values (-∞, +∞), enabling the use of linear models for probability estimation.

## 10.1.4. The Sigmoid Function

### Inverse of the Logit

The inverse of the logit function is the **sigmoid** (logistic) function:

$$ \eta(x) = g^{-1}(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}} = \sigma(x^T \beta) $$

Where $`\sigma(z) = \frac{1}{1 + e^{-z}}`$ is the sigmoid function.

**Intuition**: The sigmoid function is like a "probability converter" that takes any number (the linear combination of symptoms) and converts it to a probability between 0 and 1. It's the perfect inverse of the logit function - it undoes the transformation we did earlier.

### Properties of the Sigmoid Function

1. **Range**: $`\sigma(z) \in (0, 1)`$ for all $`z \in \mathbb{R}`$ - like always outputting a valid probability
2. **Monotonicity**: Strictly increasing - like higher symptom scores always giving higher disease probabilities
3. **Symmetry**: $`\sigma(-z) = 1 - \sigma(z)`$ - like the probability of disease being the complement of the probability of no disease
4. **Derivative**: $`\sigma'(z) = \sigma(z)(1 - \sigma(z))`$ - like having a simple formula for how the probability changes with symptoms

**Intuition**: These properties make the sigmoid function ideal for probability modeling. It always gives valid probabilities, preserves ordering, has nice symmetry, and has a simple derivative that makes optimization easy.

### Mathematical Relationship

The complete Logistic Regression model is:

$$ P(Y=1 | X=x) = \eta(x) = \sigma(x^T \beta) = \frac{1}{1 + e^{-x^T \beta}} $$

**Intuition**: This is the complete recipe for logistic regression. We take the patient's symptoms, compute a weighted sum, apply the sigmoid function, and get the probability of disease. It's like having a mathematical formula that converts symptoms directly into disease probability.

## 10.1.5. The Data and Parameters

### Data Structure

For each observation $`i = 1, 2, \ldots, n`$, we have:

- **Feature vector**: $`x_i \in \mathbb{R}^p`$ (including intercept $`x_{i0} = 1`$) - like each patient's symptoms and test results
- **Binary outcome**: $`y_i \in \{0, 1\}`$ - like whether each patient actually has the disease
- **True probability**: $`\eta(x_i) = P(Y_i=1 | X_i=x_i)`$ - like the true probability of disease for each patient

**Intuition**: This is like having a dataset of patients where we know their symptoms and whether they have the disease. We want to learn from this data to predict disease probability for new patients.

### Unknown Parameters

The unknown parameter vector $`\beta \in \mathbb{R}^p`$ includes:
- $`\beta_0`$: Intercept term - like a baseline disease probability
- $`\beta_1, \beta_2, \ldots, \beta_{p-1}`$: Feature coefficients - like the importance weight for each symptom

**Intuition**: These parameters are what we need to learn. The intercept is like the baseline risk of disease in the population, and the feature coefficients tell us how much each symptom increases or decreases the disease probability.

### The Estimation Problem

Our goal is to estimate $`\beta`$ from the observed data $`\{(x_i, y_i)\}_{i=1}^n`$.

**Intuition**: This is like learning the weights for each symptom by looking at many patients and their outcomes. We want to find the weights that best explain the observed disease patterns.

## 10.1.6. Loss Function Selection

### Why Not L2 Loss?

One might consider using the squared error loss:

$$ L_{\text{MSE}}(\beta) = \sum_{i=1}^n (y_i - \eta(x_i))^2 $$

However, this has several limitations:

1. **Small Gradients**: Since $`|y_i - \eta(x_i)| \leq 1`$, squaring makes gradients very small - like having very weak signals for learning
2. **Training Difficulties**: Small gradients make optimization slow and can lead to getting stuck - like trying to climb a very gentle slope
3. **Non-convexity**: The squared error loss is not convex for logistic regression - like having multiple valleys that can trap the optimization

**Intuition**: Using squared error for logistic regression is like trying to use a hammer to drive a screw - it's the wrong tool for the job. The squared error loss doesn't work well with probability outputs and can make learning very difficult.

### The Log-Likelihood Approach

Instead, we use the **negative log-likelihood** as our loss function:

$$ L(\beta) = -\sum_{i=1}^n \log P(Y_i = y_i | X_i = x_i) $$

**Intuition**: This approach asks "how likely is our model to produce the data we actually observed?" We want to find the parameters that make our observed data as likely as possible. It's like asking "what disease probabilities would best explain the disease outcomes we saw in our patients?"

### Likelihood Function

For binary outcomes, the likelihood is:

$$ P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i} $$

This can be written more compactly as:

$$ P(Y_i = y_i | X_i = x_i) = \eta(x_i)^{y_i} \cdot (1 - \eta(x_i))^{1 - y_i} $$

**Intuition**: This formula says that if a patient has the disease (y_i = 1), we want our model to give a high probability of disease. If a patient doesn't have the disease (y_i = 0), we want our model to give a low probability of disease. The likelihood is high when our predictions match the reality.

### Log-Likelihood

Taking the logarithm:

$$ \begin{split}
\log P(Y_i = y_i | X_i = x_i) &= y_i \log \eta(x_i) + (1 - y_i) \log(1 - \eta(x_i)) \\
&= y_i \log \frac{\eta(x_i)}{1 - \eta(x_i)} + \log(1 - \eta(x_i)) \\
&= y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta})
\end{split} $$

**Intuition**: Taking the log makes the math easier and avoids numerical problems with very small probabilities. The log-likelihood has a nice form that's easy to work with and optimize.

### Final Loss Function

The negative log-likelihood loss function is:

$$ L(\beta) = -\sum_{i=1}^n \left[ y_i \cdot x_i^T \beta - \log(1 + e^{x_i^T \beta}) \right] $$

**Intuition**: This is our final objective function. We want to minimize this loss, which means maximizing the likelihood of our observed data. It's like finding the disease probability model that best explains what we actually saw in our patients.

## 10.1.7. Comparison of Loss Functions

### Visual Comparison

The comparison of MSE and log-likelihood loss functions is implemented in the code files:

**Python Implementation:** See `compare_loss_functions()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `compare_loss_functions_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions demonstrate the key differences between MSE and log-likelihood loss functions for logistic regression:

- **MSE Loss**: Shows flat gradients and poor optimization properties - like trying to learn on a very flat surface
- **Log-Likelihood Loss**: Provides meaningful gradients and better optimization characteristics - like having clear directions for improvement
- **Gradient Comparison**: Quantifies the difference in gradient magnitudes - like measuring how strong the learning signals are
- **Visual Analysis**: Side-by-side plots showing loss landscapes - like comparing the terrain of two different optimization problems

The comparison reveals why log-likelihood is preferred over MSE for logistic regression, as it provides better optimization properties and statistical foundations.

**Intuition**: This comparison shows why log-likelihood is the right choice. It's like comparing two different fitness landscapes - one is flat and hard to navigate, while the other has clear paths to the optimum.

## 10.1.8. Advantages of Log-Likelihood

### Why Log-Likelihood is Better

1. **Convexity**: The negative log-likelihood is convex, ensuring global optimality - like having only one valley to find
2. **Proper Gradients**: Provides meaningful gradients for optimization - like having clear directions for improvement
3. **Statistical Foundation**: Based on maximum likelihood estimation - like using well-established statistical principles
4. **Interpretability**: Directly related to probability modeling - like having a clear connection between the loss and what we're trying to predict

**Intuition**: These advantages make log-likelihood the natural choice for logistic regression. It's like having the right tool for the job - designed specifically for probability estimation with good optimization properties.

### Mathematical Properties

The log-likelihood function has several desirable properties:

1. **Convexity**: The Hessian matrix is positive semi-definite - like having a bowl-shaped loss surface
2. **Uniqueness**: Under mild conditions, the maximum likelihood estimator is unique - like having only one best solution
3. **Asymptotic Properties**: MLE is consistent and asymptotically normal - like getting better estimates with more data

**Intuition**: These mathematical properties ensure that our optimization problem is well-behaved and that our estimates have good statistical properties. It's like having guarantees that our learning process will work well.

## 10.1.9. Summary and Next Steps

### What We've Established

1. **Problem Setup**: Binary classification with direct probability modeling - like learning to predict disease probability from symptoms
2. **Link Function**: Logit transformation to handle probability constraints - like converting probabilities to any number for linear modeling
3. **Model Form**: $`P(Y=1 | X=x) = \sigma(x^T \beta)`$ - like the complete formula for disease probability
4. **Loss Function**: Negative log-likelihood for optimization - like the right objective function for learning

### Key Insights

- **Discriminative Approach**: Direct modeling of $`P(Y=1 | X=x)`$ - like focusing on what we really care about
- **Link Function**: Transforms constrained probabilities to unconstrained space - like solving the constraint problem
- **Loss Selection**: Log-likelihood provides better optimization properties than MSE - like choosing the right tool for the job

**Intuition**: These insights show that logistic regression is a well-designed system for probability estimation. It directly models what we care about (disease probability), uses the right transformations to handle constraints, and employs the right loss function for learning.

### Next Steps

In the following sections, we will:
1. **Parameter Estimation**: Derive the maximum likelihood estimator - like learning the best weights from data
2. **Optimization**: Implement gradient-based optimization algorithms - like finding the best parameters efficiently
3. **Model Evaluation**: Assess model performance and interpretability - like testing how well our diagnostic system works
4. **Extensions**: Handle multi-class classification and regularization - like expanding to more complex problems

### Implementation Preview

The complete logistic regression setup demonstration is implemented in the code files:

**Python Implementation:** See `logistic_regression_setup_demo()` in [`code/setup_implementation.py`](code/setup_implementation.py)

**R Implementation:** See `logistic_regression_setup_demo_r()` in [`code/r_setup_implementation.R`](code/r_setup_implementation.R)

These functions provide a comprehensive demonstration of the logistic regression setup:

- **Data Generation**: Synthetic binary classification data with known parameters - like creating example patients with known disease probabilities
- **Visualization**: Scatter plots showing class separation and probability distributions - like seeing how symptoms relate to disease probability
- **Parameter Analysis**: Examination of true parameters and class balance - like understanding the underlying disease model
- **Setup Summary**: Complete overview of the problem setup - like having a complete picture of what we're trying to learn

The demonstration shows how logistic regression transforms linear predictors into probabilities through the sigmoid function, creating a complete framework for binary classification.

**Intuition**: This demonstration brings all the concepts together, showing how the mathematical framework translates into a practical system for probability estimation. It's like seeing the complete diagnostic system in action.

This setup provides the foundation for understanding logistic regression as both a probabilistic model and an optimization problem, setting the stage for maximum likelihood estimation and practical applications.

---

**Navigation:**
- **Next Topic:** [Maximum Likelihood Estimation](02_mle.md) - Likelihood function, optimization, and parameter estimation
- **Previous Topic:** [Logistic Regression Overview](README.md) - Overview of logistic regression concepts and applications
