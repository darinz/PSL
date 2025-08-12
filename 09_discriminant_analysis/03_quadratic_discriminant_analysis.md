# 9.3. Quadratic Discriminant Analysis

## 9.3.1. Introduction to QDA

Quadratic Discriminant Analysis (QDA) is a powerful classification method that models each class as a multivariate Gaussian distribution with its own mean vector and covariance matrix. Unlike Linear Discriminant Analysis (LDA), which assumes all classes share the same covariance structure, QDA allows for class-specific covariance matrices, making it more flexible for capturing complex decision boundaries.

**Intuitive Understanding**: QDA is like understanding that each disease has its own unique "fingerprint" in terms of how symptoms relate to each other. Imagine you're a doctor trying to diagnose patients based on multiple symptoms. With QDA, you learn that the flu might have symptoms that are highly correlated (fever and chills often go together), while diabetes might have more independent symptoms (high blood sugar doesn't necessarily predict vision problems). Each disease has its own pattern of symptom relationships, and QDA captures these unique patterns to make more accurate diagnoses.

### Key Characteristics of QDA

1. **Class-Specific Covariances**: Each class has its own covariance matrix $`\Sigma_k`$ - like each disease having its own unique pattern of symptom relationships
2. **Quadratic Decision Boundaries**: The decision function is quadratic in the feature vector - like drawing curved lines to separate different diseases instead of straight lines
3. **Generative Model**: Models the joint distribution $`P(X, Y)`$ through class-conditional densities - like understanding the complete "recipe" for how each disease generates symptoms
4. **Bayes Optimal**: Under Gaussian assumptions, QDA provides the Bayes optimal classifier - like having the theoretically best possible diagnostic system

**Intuition**: These characteristics make QDA a very flexible and powerful classifier. The class-specific covariances allow it to capture the unique ways that features relate to each other within each class, while the quadratic decision boundaries allow it to create complex separation rules that can handle non-linear patterns in the data.

### When to Use QDA

- Classes have different covariance structures - like when different diseases have very different symptom patterns
- Sufficient data to estimate class-specific covariances reliably - like having enough patients to learn unique disease profiles
- Non-linear decision boundaries are needed - like when diseases require complex separation rules
- High-dimensional data with enough samples per class - like having many symptoms but enough patients to learn the patterns

**Intuition**: QDA is most useful when your classes are truly different in how their features relate to each other. If all classes follow similar patterns, LDA might be simpler and more robust. But if each class has its own unique "personality" in terms of feature relationships, QDA can capture this complexity and provide better classification.

## 9.3.2. Mathematical Foundation

### Multivariate Gaussian Distribution

For each class $`k`$, we assume the feature vector $`X`$ follows a multivariate normal distribution:

$$ X \mid Y = k \sim \mathcal{N}(\mu_k, \Sigma_k) $$

where:
- $`\mu_k \in \mathbb{R}^p`$ is the mean vector for class $`k`$ - like the typical symptom profile for disease k
- $`\Sigma_k \in \mathbb{R}^{p \times p}`$ is the covariance matrix for class $`k`$ - like the pattern of how symptoms relate to each other for disease k

**Intuition**: The multivariate Gaussian assumption means that for each disease, the symptoms follow a bell-shaped distribution around the typical symptom profile, with the covariance matrix describing how symptoms vary and relate to each other. This is like saying "patients with the flu typically have these symptoms, but the exact combination varies in a predictable way."

### Parameter Notation

Let's define the precision matrix (inverse covariance) as $`\Theta_k = \Sigma_k^{-1}`$:

$$ \mu_k = \begin{pmatrix} 
\mu_{k,1} \\ 
\mu_{k,2} \\ 
\vdots \\ 
\mu_{k,p} 
\end{pmatrix}_{p \times 1}, \quad
\Theta_k = \Sigma_k^{-1} = \begin{pmatrix} 
\theta_{k,11} & \cdots & \theta_{k,1p} \\ 
\vdots & \ddots & \vdots \\ 
\theta_{k,p1} & \cdots & \theta_{k,pp} 
\end{pmatrix}_{p \times p} $$

**Intuition**: The precision matrix $`\Theta_k`$ is like the "inverse" of the covariance matrix. While the covariance matrix tells us how symptoms relate to each other, the precision matrix tells us how to "undo" these relationships. It's like having a recipe that tells you how to adjust for the fact that certain symptoms tend to go together.

### Class-Conditional Density Function

The probability density function for class $`k`$ is:

$$ f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right) $$

**Intuition**: This formula gives us the probability of seeing a particular set of symptoms if the patient has disease k. The exponential term measures how "unusual" the symptom combination is for this disease, while the denominator normalizes the probability so it sums to 1.

The quadratic term in the exponent can be expanded as:

$$ (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) = \sum_{j=1}^p \sum_{l=1}^p \theta_{k,jl} (x_j - \mu_{k,j}) (x_l - \mu_{k,l}) $$

**Intuition**: This expansion shows that the "unusualness" score is computed by looking at all pairs of symptoms and how they differ from the typical values. The coefficients $`\theta_{k,jl}`$ tell us how important each symptom pair is for determining whether the patient has disease k.

### Bayes Decision Rule

Using Bayes' theorem, the posterior probability is:

$$ P(Y = k \mid X = x) \propto \pi_k f_k(x) \propto e^{-d_k(x)/2} $$

where $`d_k(x)`$ is the **quadratic discriminant function**:

$$ \begin{split}
d_k(x) &= 2[-\log f_k(x) - \log \pi_k] - \text{Constant} \\
&= (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log|\Sigma_k| - 2\log \pi_k
\end{split} $$

**Intuition**: The discriminant function $`d_k(x)`$ gives us a "score" for how likely the patient has disease k. Lower scores mean higher probability. The function combines three pieces of information: how unusual the symptoms are for this disease, how spread out this disease typically is, and how common this disease is in the population.

### Components of the Discriminant Function

The function $`d_k(x)`$ consists of three terms:

1. **Mahalanobis Distance**: $`(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)`$ - measures the distance from $`x`$ to class center $`\mu_k`$ in the metric defined by $`\Sigma_k^{-1}`$ - like measuring how far the patient's symptoms are from the typical symptoms for this disease, taking into account how symptoms relate to each other

2. **Log Determinant**: $`\log|\Sigma_k|`$ - penalizes classes with larger covariance matrices (more spread out) - like penalizing diseases that have very variable symptoms, since they're harder to diagnose confidently

3. **Prior Term**: $`-2\log \pi_k`$ - incorporates class prior probabilities - like giving a bonus to more common diseases, since they're more likely to be the correct diagnosis

**Intuition**: These three terms work together to give us a comprehensive score for each disease. The Mahalanobis distance tells us how well the symptoms fit this disease, the log determinant penalizes diseases with very variable symptoms, and the prior term gives a bonus to more common diseases.

### Decision Rule

The optimal classification rule is:

$$ \hat{y} = \arg\min_k d_k(x) $$

**Intuition**: We simply choose the disease that gives us the lowest score (highest probability). This is like saying "choose the disease that best explains the symptoms, taking into account how common the disease is and how variable its symptoms are."

## 9.3.3. Parameter Estimation

### Maximum Likelihood Estimation

Given training data $`\{(x_i, y_i)\}_{i=1}^n`$, we estimate parameters using maximum likelihood:

#### Class Priors
$$ \hat{\pi}_k = \frac{n_k}{n} $$
where $`n_k = \sum_{i=1}^n \mathbb{I}(y_i = k)`$ is the number of samples in class $`k`$.

**Intuition**: We estimate how common each disease is by simply counting how many patients in our training data have each disease. This gives us our baseline expectation about disease prevalence.

#### Class Means
$$ \hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i $$

**Intuition**: We estimate the typical symptom profile for each disease by averaging the symptoms of all patients with that disease. This gives us the "center" of each disease's symptom distribution.

#### Class Covariances
$$ \hat{\Sigma}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T $$

**Intuition**: We estimate how symptoms relate to each other for each disease by looking at how the symptoms of patients with that disease vary around the typical profile. This tells us the unique "fingerprint" of each disease.

### Numerical Stability

When $`\Sigma_k`$ is singular or near-singular (common in high dimensions), we use regularization:

$$ \hat{\Sigma}_k^{reg} = \hat{\Sigma}_k + \epsilon I_p $$

where $`\epsilon > 0`$ is a small constant (e.g., $`10^{-6}`$).

**Intuition**: When we have many symptoms but few patients, the covariance matrix can become unstable (like trying to estimate a complex pattern with very little data). Regularization adds a small amount of "noise" to make the matrix more stable, like adding a safety net to prevent the model from making extreme assumptions.

## 9.3.4. Implementation: QDA from Scratch

The complete implementation of Quadratic Discriminant Analysis is provided in the following code files:

**Python Implementation:** [`code/qda_implementation.py`](code/qda_implementation.py)

**R Implementation:** [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These files contain:

- Complete QDA class implementation with parameter estimation - like building a complete diagnostic system from scratch
- Regularized QDA for high-dimensional data - like adding stability to the diagnostic system when we have many symptoms but few patients
- Decision boundary visualization functions - like showing how the diagnostic rules separate different diseases
- Model comparison utilities (QDA vs LDA) - like comparing flexible vs simple diagnostic approaches
- Parameter analysis and diagnostics - like understanding what the diagnostic system learned about each disease
- Cross-validation and model selection - like testing the diagnostic system to make sure it works well
- Real-world examples (Iris dataset, credit risk assessment) - like applying the diagnostic system to real problems

The Python implementation includes a custom `QuadraticDiscriminantAnalysis` class that mirrors the scikit-learn API, while the R implementation provides both MASS package integration and custom functions for educational purposes.

## 9.3.5. Decision Boundaries and Visualization

### Understanding QDA Decision Boundaries

QDA produces quadratic decision boundaries because the discriminant function $`d_k(x)`$ is quadratic in $`x`$. For two classes, the decision boundary is where $`d_1(x) = d_2(x)`$.

**Intuition**: The decision boundary is like a fence that separates different diseases in the symptom space. Because QDA allows each disease to have its own unique pattern of symptom relationships, this fence can be curved (quadratic) rather than straight (linear). This allows QDA to capture complex patterns that linear methods would miss.

The visualization functions for decision boundaries are implemented in the code files:

**Python:** See `plot_qda_decision_boundaries()` and `compare_qda_lda_boundaries()` functions in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `plot_qda_decision_boundaries()` and `compare_qda_lda_boundaries()` functions in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions create mesh grids over the feature space, compute predictions for each grid point, and visualize both decision boundaries and posterior probabilities. The comparison function demonstrates the key differences between QDA's quadratic boundaries and LDA's linear boundaries.

## 9.3.6. Model Analysis and Diagnostics

### Parameter Analysis

The parameter analysis functions are implemented in the code files:

**Python:** See `analyze_qda_parameters()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `analyze_qda_parameters()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions provide comprehensive analysis of QDA model parameters including:
- Class prior probabilities - like understanding how common each disease is
- Class mean vectors - like understanding the typical symptom profile for each disease
- Covariance matrices with heatmap visualizations - like seeing the "fingerprint" of each disease's symptom relationships
- Log determinants for each class - like understanding how variable each disease's symptoms are

**Intuition**: Parameter analysis helps us understand what the QDA model learned about each class. We can see which diseases are more common, what the typical symptoms are for each disease, and how symptoms relate to each other within each disease. This gives us insights into the underlying patterns in our data.

### Mahalanobis Distance Analysis

The Mahalanobis distance analysis functions are implemented in the code files:

**Python:** See `analyze_mahalanobis_distances()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `analyze_mahalanobis_distances()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions compute and visualize the distribution of Mahalanobis distances for each class, comparing them with the theoretical chi-squared distribution to assess model fit and identify potential outliers.

**Intuition**: Mahalanobis distance analysis helps us understand how well our model fits the data. If the distances follow the expected chi-squared distribution, our Gaussian assumption is reasonable. If not, we might need to consider different distributional assumptions or identify unusual cases that don't fit our model well.

## 9.3.7. High-Dimensional QDA

### Challenges in High Dimensions

When the number of features $`p`$ is large relative to the sample size, QDA faces several challenges:

1. **Curse of Dimensionality**: Need $`O(p^2)`$ parameters per class - like needing many more patients to learn complex symptom relationships when we have many symptoms
2. **Singular Covariance**: Covariance matrices become singular - like the symptom patterns becoming unstable when we have too many symptoms relative to patients
3. **Overfitting**: Model complexity increases with $`p^2`$ - like the diagnostic system becoming too complex and fitting noise in the data

**Intuition**: High-dimensional QDA is like trying to learn very complex disease profiles when we have many symptoms but few patients. The model becomes unstable and prone to overfitting, like a doctor trying to learn complex diagnostic rules from very few cases.

### Regularization Techniques

The regularized QDA implementation and high-dimensional testing functions are provided in the code files:

**Python:** See `RegularizedQDA` class and `test_high_dimensional_qda()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `test_high_dimensional_qda()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

The `RegularizedQDA` class implements three regularization strategies:
- **Diagonal regularization**: Forces covariance matrices to be diagonal - like assuming symptoms are independent within each disease
- **Shrinkage regularization**: Shrinks covariance matrices toward a target (scaled identity) - like pulling the disease profiles toward simpler patterns
- **Ridge regularization**: Adds a small constant to the diagonal for numerical stability - like adding a safety net to prevent instability

**Intuition**: Regularization is like adding constraints to prevent the model from becoming too complex. We trade some flexibility for stability, making the model more robust when we have limited data.

The high-dimensional testing function generates synthetic data with sparse covariance structures and compares the performance of different regularization approaches.

## 9.3.8. Model Selection and Validation

### Cross-Validation for QDA

The cross-validation and model selection functions are implemented in the code files:

**Python:** See `qda_cross_validation()` and `qda_grid_search()` functions in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_cross_validation()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

These functions provide:
- **Cross-validation**: Evaluates QDA performance using k-fold cross-validation - like testing the diagnostic system on different groups of patients to make sure it works well
- **Grid search**: Finds optimal regularization parameters using cross-validation - like finding the best balance between flexibility and stability
- **Model selection**: Compares different QDA configurations systematically - like comparing different diagnostic approaches to find the best one

**Intuition**: Model selection and validation are like thoroughly testing a diagnostic system before using it in practice. We want to make sure it works well on new patients, not just the ones we used to train it.

## 9.3.9. Real-World Applications

### Example: Iris Dataset

The Iris dataset example is implemented in the code files:

**Python:** See `qda_iris_example()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_iris_example()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

This example demonstrates QDA on the classic Iris dataset, using only two classes for binary classification. It includes data preprocessing, model fitting, evaluation with confusion matrices, and visualization of results.

**Intuition**: The Iris dataset is like a simple diagnostic problem where we're trying to distinguish between two types of flowers based on their measurements. QDA helps us understand how the measurements relate to each other for each flower type and creates a decision rule to classify new flowers.

### Example: Credit Risk Assessment

The credit risk assessment example is implemented in the code files:

**Python:** See `qda_credit_risk_example()` function in [`code/qda_implementation.py`](code/qda_implementation.py)

**R:** See `qda_credit_risk_example()` function in [`code/r_qda_implementation.R`](code/r_qda_implementation.R)

This example creates synthetic credit data with features like income, credit score, debt ratio, and employment years. It demonstrates how QDA can be used for risk assessment, including feature analysis and visualization of class-specific parameter differences.

**Intuition**: Credit risk assessment is like diagnosing whether a loan applicant is likely to default. QDA helps us understand how different financial characteristics relate to each other for risky vs safe borrowers and creates a decision rule to classify new applicants.

This comprehensive expansion provides detailed mathematical foundations, practical implementations, and clear explanations of Quadratic Discriminant Analysis. The code examples demonstrate both theoretical concepts and their practical application, including visualization, evaluation, and handling of common challenges in high-dimensional settings.

---

**Navigation:**
- **Next Topic:** [Linear Discriminant Analysis](04_linear_discriminant_analysis.md) - Shared covariance assumption and linear decision boundaries
- **Previous Topic:** [Discriminant Analysis](02_discriminant_analysis.md) - Bayes' theorem application and joint distribution factorization
