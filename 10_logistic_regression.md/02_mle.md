# 10.2. Maximum Likelihood Estimation (MLE)

## Introduction

Maximum Likelihood Estimation (MLE) is the cornerstone of parameter estimation in logistic regression. Unlike linear regression where we can derive closed-form solutions, logistic regression requires iterative optimization due to the nonlinear nature of the sigmoid function. In this section, we'll derive the MLE step-by-step and implement the optimization algorithms.

**Intuitive Understanding**: Maximum Likelihood Estimation is like being a detective trying to find the most likely explanation for the evidence you've observed. Imagine you're a medical researcher who has collected data on many patients - their symptoms and whether they have a disease. MLE asks: "What disease probability model would make the observed patient outcomes most likely?" It's like finding the best explanation for why some patients got sick and others didn't, given their symptoms. We're essentially asking "What weights for each symptom would best explain the disease patterns we actually saw?"

### Why MLE for Logistic Regression?

**Intuition**: Unlike linear regression where we can solve for the best parameters directly (like solving a simple equation), logistic regression is more complex because we're working with probabilities that are constrained between 0 and 1. The sigmoid function makes the relationship between symptoms and disease probability nonlinear, so we need an iterative approach - like gradually refining our detective's theory based on the evidence.

## Mathematical Foundation

### Step 1: From Logit to Probability

We start with the logit transformation that connects our linear predictor to the probability:

$$ \log \frac{\eta(x)}{1-\eta(x)} = x^T \beta $$

This equation states that the log-odds of the positive class is a linear function of our features. To work with probabilities directly, we need to solve for $`\eta(x)`$:

**Intuition**: This is like saying "the log-odds of disease is a weighted sum of symptoms." The logit function transforms our constrained probability into an unconstrained space where we can use linear models. It's like converting a probability (which must be between 0 and 1) into any number (which can be any real value).

$$ \begin{split}
\log \frac{\eta(x)}{1-\eta(x)} &= x^T \beta \\
\frac{\eta(x)}{1-\eta(x)} &= \exp(x^T \beta) \\
\eta(x) &= \exp(x^T \beta) \cdot (1-\eta(x)) \\
\eta(x) &= \exp(x^T \beta) - \exp(x^T \beta) \cdot \eta(x) \\
\eta(x) + \exp(x^T \beta) \cdot \eta(x) &= \exp(x^T \beta) \\
\eta(x) \cdot (1 + \exp(x^T \beta)) &= \exp(x^T \beta) \\
\eta(x) &= \frac{\exp(x^T \beta)}{1 + \exp(x^T \beta)}
\end{split} $$

**Intuition**: This algebraic manipulation is like solving a puzzle. We start with the log-odds equation and systematically rearrange it to isolate the probability on one side. Each step is like moving pieces around until we get the probability expressed purely in terms of our symptom weights and the sigmoid function.

### Step 2: Unified Probability Expression

We can express both $`P(Y=1|X=x)`$ and $`P(Y=0|X=x)`$ in a unified form using the sigmoid function:

$$ \begin{split}
P(Y=1|X=x) &= \eta(x) = \frac{\exp(x^T \beta)}{1 + \exp(x^T \beta)} = \sigma(x^T \beta) \\
P(Y=0|X=x) &= 1 - \eta(x) = \frac{1}{1 + \exp(x^T \beta)} = 1 - \sigma(x^T \beta)
\end{split} $$

Where $`\sigma(z) = \frac{e^z}{1 + e^z}`$ is the sigmoid function.

**Intuition**: This unified expression is like having a single formula that works for both disease and no-disease cases. If a patient has the disease (Y=1), we use the sigmoid function directly. If they don't have the disease (Y=0), we use one minus the sigmoid function. This elegant formulation lets us handle both cases with a single mathematical expression.

### Step 3: Likelihood Function

For a dataset with $`n`$ independent observations $`(x_i, y_i)`$, the likelihood function is:

$$ L(\beta) = \prod_{i=1}^n P(Y_i = y_i | X_i = x_i) $$

Using our unified probability expression:

$$ L(\beta) = \prod_{i=1}^n \sigma(x_i^T \beta)^{y_i} (1 - \sigma(x_i^T \beta))^{1-y_i} $$

**Intuition**: The likelihood function asks "how likely is our model to produce the exact outcomes we observed?" For each patient, if they have the disease (y_i = 1), we want our model to give a high probability of disease. If they don't have the disease (y_i = 0), we want our model to give a low probability of disease. We multiply all these probabilities together because we assume the patients are independent - like saying "what's the probability that our model would correctly predict all these patient outcomes?"

### Step 4: Log-Likelihood Function

Taking the natural logarithm (which preserves the maximum and simplifies calculations):

$$ \begin{split}
\ell(\beta) &= \log L(\beta) \\
&= \sum_{i=1}^n \log \left[ \sigma(x_i^T \beta)^{y_i} (1 - \sigma(x_i^T \beta))^{1-y_i} \right] \\
&= \sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \beta) + (1-y_i) \log (1 - \sigma(x_i^T \beta)) \right]
\end{split} $$

This is the **log-likelihood function** that we want to maximize.

**Intuition**: Taking the log transforms multiplication into addition, which is much easier to work with mathematically. It's like converting a complex multiplication problem into a simpler addition problem. The log-likelihood is like a "score" that measures how well our model explains the data - higher scores mean better explanations. We want to find the symptom weights that give us the highest possible score.

## Gradient and Hessian Derivation

### First Derivative (Gradient)

To find the maximum, we set the gradient to zero:

$$ \frac{\partial \ell(\beta)}{\partial \beta} = 0 $$

Let's compute this step by step:

**Intuition**: The gradient tells us the direction of steepest increase in our log-likelihood score. Setting it to zero is like finding the point where we can't improve our score anymore - the peak of the hill. It's like a detective saying "I've found the best explanation, I can't make it any better."

$$ \begin{split}
\frac{\partial \ell(\beta)}{\partial \beta} &= \sum_{i=1}^n \frac{\partial}{\partial \beta} \left[ y_i \log \sigma(x_i^T \beta) + (1-y_i) \log (1 - \sigma(x_i^T \beta)) \right] \\
&= \sum_{i=1}^n \left[ y_i \frac{1}{\sigma(x_i^T \beta)} \frac{\partial \sigma(x_i^T \beta)}{\partial \beta} + (1-y_i) \frac{1}{1-\sigma(x_i^T \beta)} \frac{\partial (1-\sigma(x_i^T \beta))}{\partial \beta} \right]
\end{split} $$

Using the chain rule and the fact that $`\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z))`$:

**Intuition**: The chain rule is like following a chain of cause and effect. We want to know how changing the symptom weights affects the log-likelihood. This happens through several steps: weights affect the linear predictor, which affects the sigmoid function, which affects the probability, which affects the log-likelihood. The chain rule helps us track this entire chain.

$$ \begin{split}
\frac{\partial \sigma(x_i^T \beta)}{\partial \beta} &= \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) \cdot x_i \\
\frac{\partial (1-\sigma(x_i^T \beta))}{\partial \beta} &= -\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) \cdot x_i
\end{split} $$

**Intuition**: The derivative of the sigmoid function has a beautiful form: $`\sigma(z)(1-\sigma(z))`$. This is like saying "the rate of change is highest when the probability is around 50%" - when we're most uncertain. When we're very confident (probability close to 0 or 1), the rate of change is small.

Substituting back:

$$ \begin{split}
\frac{\partial \ell(\beta)}{\partial \beta} &= \sum_{i=1}^n \left[ y_i \frac{1}{\sigma(x_i^T \beta)} \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i + (1-y_i) \frac{1}{1-\sigma(x_i^T \beta)} (-\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta))) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i (1-\sigma(x_i^T \beta)) x_i - (1-y_i) \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i x_i - y_i \sigma(x_i^T \beta) x_i - \sigma(x_i^T \beta) x_i + y_i \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n \left[ y_i x_i - \sigma(x_i^T \beta) x_i \right] \\
&= \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta))
\end{split} $$

Therefore:

$$ \frac{\partial \ell(\beta)}{\partial \beta} = \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta)) = X^T(y - \hat{y}) $$

Where $`X`$ is the design matrix, $`y`$ is the vector of observed outcomes, and $`\hat{y}`$ is the vector of predicted probabilities.

**Intuition**: This elegant result says that the gradient is the sum of prediction errors weighted by the symptoms. For each patient, we take the difference between what actually happened (y_i) and what our model predicted (σ(x_i^T β)), and multiply by their symptoms. This makes perfect sense - if our model underpredicted disease for a patient with certain symptoms, we should increase the weights for those symptoms.

### Second Derivative (Hessian)

The Hessian matrix is:

$$ H(\beta) = \frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T} $$

Computing this:

**Intuition**: The Hessian tells us about the curvature of our log-likelihood surface. It's like knowing not just which direction to go (gradient), but also how steep the hill is and how it curves. This information helps us take bigger steps when the surface is flat and smaller steps when it's steep.

$$ \begin{split}
\frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T} &= \frac{\partial}{\partial \beta^T} \left[ \sum_{i=1}^n x_i (y_i - \sigma(x_i^T \beta)) \right] \\
&= \sum_{i=1}^n x_i \frac{\partial}{\partial \beta^T} (y_i - \sigma(x_i^T \beta)) \\
&= \sum_{i=1}^n x_i \left[ -\frac{\partial \sigma(x_i^T \beta)}{\partial \beta^T} \right] \\
&= \sum_{i=1}^n x_i \left[ -\sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i^T \right] \\
&= -\sum_{i=1}^n \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) x_i x_i^T
\end{split} $$

In matrix form:

$$ H(\beta) = -X^T W X $$

Where $`W`$ is a diagonal matrix with $`W_{ii} = \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta))`$.

**Intuition**: The Hessian has a beautiful structure. The negative sign indicates that our log-likelihood is concave (like an upside-down bowl), which means any local maximum is also the global maximum. The W matrix contains weights that depend on our current predictions - patients with uncertain predictions (probabilities around 50%) get higher weights because their predictions are most sensitive to parameter changes.

## Newton-Raphson Algorithm

Since the gradient equation $`\frac{\partial \ell(\beta)}{\partial \beta} = 0`$ has no closed-form solution, we use the Newton-Raphson iterative algorithm:

$$ \beta^{(t+1)} = \beta^{(t)} - H(\beta^{(t)})^{-1} \nabla \ell(\beta^{(t)}) $$

Substituting our expressions:

$$ \beta^{(t+1)} = \beta^{(t)} + (X^T W^{(t)} X)^{-1} X^T(y - \hat{y}^{(t)}) $$

This is equivalent to solving a weighted least squares problem at each iteration.

**Intuition**: Newton-Raphson is like a smart hill-climbing algorithm. Instead of just following the gradient (which might take many small steps), it uses information about the curvature to take bigger, more intelligent steps. It's like a hiker who not only knows which direction is uphill but also knows how steep the hill is, so they can take appropriately sized steps.

The algorithm works by:
1. **Current Position**: Start with some guess for the symptom weights
2. **Direction**: Calculate the gradient (which direction to go)
3. **Step Size**: Use the Hessian to determine how big a step to take
4. **Update**: Move to the new position
5. **Repeat**: Until we can't improve anymore

## Reweighted Least Squares (IRLS) Algorithm

The Newton-Raphson method can be reformulated as an **Iteratively Reweighted Least Squares (IRLS)** algorithm:

**Intuition**: IRLS is like solving a series of weighted linear regression problems. At each step, we pretend our logistic regression problem is actually a linear regression problem, but we give different weights to different patients based on how uncertain our current predictions are. Patients with uncertain predictions get more influence on the next update.

### Algorithm Steps:

1. **Initialize**: $`\beta^{(0)} = 0`$ or use a reasonable starting point
2. **For iteration $`t = 0, 1, 2, \ldots`$**:
   - Compute predicted probabilities: $`\hat{y}_i^{(t)} = \sigma(x_i^T \beta^{(t)})`$
   - Compute working response: $`z_i^{(t)} = x_i^T \beta^{(t)} + \frac{y_i - \hat{y}_i^{(t)}}{\hat{y}_i^{(t)}(1-\hat{y}_i^{(t)})}`$
   - Compute weights: $`w_i^{(t)} = \hat{y}_i^{(t)}(1-\hat{y}_i^{(t)})`$
   - Update parameters: $`\beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} z^{(t)}`$
3. **Convergence**: Stop when $`||\beta^{(t+1)} - \beta^{(t)}|| < \epsilon`$

**Intuition**: Each step of IRLS is like solving a weighted linear regression problem where:
- **Working Response**: We create a "fake" continuous outcome that, when we solve the linear regression, gives us the right update for our logistic regression parameters
- **Weights**: Patients with uncertain predictions (probabilities around 50%) get higher weights because their predictions are most informative for learning
- **Update**: We solve a weighted least squares problem to find the best linear fit to our working response

## Implementation

The complete MLE implementation for logistic regression is provided in the code files:

**Python Implementation:** See `LogisticRegressionMLE` class and comprehensive demonstrations in [`code/mle_implementation.py`](code/mle_implementation.py)

**R Implementation:** See optimization functions and demonstrations in [`code/r_mle_implementation.R`](code/r_mle_implementation.R)

These implementations include:

- **LogisticRegressionMLE Class**: Complete implementation with Newton-Raphson and IRLS methods - like having a complete diagnostic system that learns from patient data
- **Numerical Stability**: Proper handling of overflow and underflow issues - like preventing computational errors when dealing with very small or large numbers
- **Convergence Tracking**: History tracking for log-likelihood and parameter norms - like monitoring how the learning process progresses
- **Comprehensive Demonstrations**: 
  - Method comparison (Newton-Raphson vs IRLS) - like comparing different learning strategies
  - Convergence visualization - like seeing how quickly the algorithm finds the best parameters
  - Parameter comparison with sklearn/glm - like checking our results against standard tools
  - Decision boundary visualization - like seeing how the learned model separates patients
  - Gradient and Hessian analysis - like understanding the mathematical properties
  - Numerical stability testing - like ensuring our algorithm works reliably
  - Optimization method comparison - like choosing the best learning approach

The implementations demonstrate the mathematical foundations while providing practical, robust optimization algorithms for logistic regression parameter estimation.

## Key Insights

### 1. **Concavity of Log-Likelihood**
The Hessian matrix $`H(\beta) = -X^T W X`$ is negative semi-definite because:
- $`W`$ is diagonal with positive entries $`w_i = \sigma(x_i^T \beta)(1-\sigma(x_i^T \beta)) > 0`$ - like having positive weights for all patients
- $`X^T W X`$ is positive semi-definite - like having a positive definite quadratic form
- Therefore, $`-X^T W X`$ is negative semi-definite - like having a concave function

This guarantees that any local maximum is also the global maximum.

**Intuition**: This is like having a landscape with only one peak - no matter where you start climbing, you'll always reach the same highest point. This is crucial because it means our optimization algorithm won't get stuck in local optima - there's only one best solution.

### 2. **Connection to Linear Regression**
The gradient equation $`X^T(y - \hat{y}) = 0`$ is similar to the normal equations in linear regression, but with predicted probabilities instead of linear predictions.

**Intuition**: This connection shows that logistic regression is like linear regression but with a "probability wrapper." The core idea is the same - find parameters that minimize prediction errors - but we're predicting probabilities instead of continuous values, and we use a different loss function.

### 3. **Numerical Stability**
- Use `np.clip()` to prevent overflow in sigmoid function - like preventing the exponential function from producing numbers too large for computers to handle
- Add small epsilon to prevent `log(0)` in likelihood computation - like avoiding taking the log of zero
- Use pseudo-inverse when Hessian is singular - like handling cases where the matrix can't be inverted

**Intuition**: Numerical stability is like making sure our calculations don't break down due to computer limitations. Just like we need to handle edge cases in medical diagnosis (like patients with unusual symptoms), we need to handle edge cases in our mathematical computations.

### 4. **Convergence Properties**
- Newton-Raphson typically converges in 5-10 iterations - like finding the best diagnosis in just a few rounds of testing
- IRLS is more numerically stable but may require more iterations - like a more careful but slower diagnostic process
- Both methods achieve the same optimal solution - like different diagnostic approaches leading to the same conclusion

**Intuition**: These convergence properties help us choose the right algorithm for our needs. Newton-Raphson is like a fast but potentially risky approach, while IRLS is like a slower but more reliable approach. Both get us to the same destination.

### 5. **Computational Complexity**
- Each iteration: $`O(np^2 + p^3)`$ where $`n`$ is sample size, $`p`$ is number of features - like the computational cost growing with both the number of patients and the number of symptoms
- Matrix inversion dominates for large $`p`$ - like the most expensive step being solving a system of equations
- Sparse matrix techniques can improve efficiency - like using shortcuts when many symptoms are irrelevant

**Intuition**: Understanding computational complexity helps us scale our diagnostic system. Just like a doctor needs to be efficient when seeing many patients, our algorithm needs to be efficient when dealing with large datasets.

## Applications and Extensions

### 1. **Regularized Logistic Regression**
Add L1/L2 penalties to the log-likelihood:

$$ \ell_{\text{penalized}}(\beta) = \ell(\beta) - \lambda \sum_{j=1}^p |\beta_j| \quad \text{(L1)} $$

**Intuition**: Regularization is like adding "common sense" constraints to our diagnostic system. L1 regularization (Lasso) encourages using fewer symptoms by setting some weights to exactly zero - like a doctor who focuses on the most important symptoms. L2 regularization (Ridge) prevents any single symptom from having too much influence - like a doctor who considers all symptoms but doesn't overemphasize any one.

### 2. **Multinomial Logistic Regression**
Extend to $`K > 2`$ classes using softmax function:

$$ P(Y=k|X=x) = \frac{\exp(x^T \beta_k)}{\sum_{j=1}^K \exp(x^T \beta_j)} $$

**Intuition**: Multinomial logistic regression is like extending our diagnostic system to handle multiple diseases instead of just one. Instead of predicting "disease or no disease," we predict which of several possible diseases the patient has. The softmax function ensures that the probabilities for all diseases sum to 1.

### 3. **Bayesian Logistic Regression**
Use MCMC or variational inference to obtain posterior distributions of parameters.

**Intuition**: Bayesian logistic regression is like being a doctor who acknowledges uncertainty in their diagnostic system. Instead of giving a single best estimate for each symptom's importance, we give a range of possible values with probabilities. This is like saying "this symptom is probably important, but I'm not completely certain."

The MLE approach provides a solid foundation for understanding and implementing logistic regression, with clear connections to both linear regression and modern machine learning techniques.

**Intuition**: This foundation shows how logistic regression connects to fundamental statistical principles while providing practical tools for real-world applications. It's like having both the theoretical understanding and the practical skills needed to build effective diagnostic systems.

---

**Navigation:**
- **Next Topic:** [Separable Data](03_seperable_data.md) - Handling perfectly separable data and convergence issues
- **Previous Topic:** [Setup and Introduction](01_setup.md) - Mathematical foundations and problem formulation
