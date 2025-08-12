# 9.2. Discriminant Analysis

## 9.2.1. Introduction to Discriminant Analysis

Discriminant analysis is a family of classification methods that model the distribution of features within each class and use Bayes' theorem to make predictions. Unlike discriminative methods that directly model $`P(Y|X)`$, discriminant analysis is a **generative approach** that models the joint distribution $`P(X, Y)`$ by decomposing it into class-conditional densities and class priors.

**Intuitive Understanding**: Discriminant analysis is like understanding how different groups of people behave and then using that knowledge to classify new people. Imagine you're a doctor trying to diagnose patients based on their symptoms. Instead of just learning which symptoms predict which disease, you learn the complete "profile" of each disease - what symptoms are common, how severe they are, how they vary, and how often each disease occurs. Then when a new patient comes in, you compare their symptoms to each disease profile and choose the most likely diagnosis. This approach is called "generative" because you're modeling how each class (disease) generates its features (symptoms).

### Generative vs. Discriminative Approaches

| Approach | Models | Example Methods |
|----------|--------|-----------------|
| **Generative** | $`P(X, Y) = P(Y) \cdot P(X \mid Y)`$ | LDA, QDA, Naive Bayes |
| **Discriminative** | $`P(Y \mid X)`$ directly | Logistic Regression, SVM |

**Intuition**: The key difference is what you're trying to learn. Generative methods are like learning the complete "recipe" for each class - you understand how each disease typically presents, including the full range of possible symptoms and their variations. Discriminative methods are more like learning a simple rule - "if you see these symptoms, predict this disease" without understanding the full disease profile. Generative methods give you a richer understanding of the data, while discriminative methods focus on the decision boundary.

### Mathematical Foundation

The key insight of discriminant analysis is to decompose the joint distribution:

$$ p(x, y) = p(y) \cdot p(x \mid y) $$

where:
- $`p(y)`$ is the **class prior** (marginal distribution of classes) - like how common each disease is in the population
- $`p(x \mid y)`$ is the **class-conditional density** (distribution of features given class) - like the typical symptom profile for each disease

**Intuition**: This decomposition is like breaking down the problem into two parts: (1) how common is each disease in the population? and (2) what do patients with each disease typically look like? This makes the problem much easier to handle because we can estimate each part separately. The class prior tells us our baseline expectations, while the class-conditional density tells us how features vary within each class.

This decomposition allows us to:
1. Estimate class priors from class frequencies in the data - like counting how many patients have each disease
2. Model class-conditional densities using parametric or non-parametric methods - like understanding the typical symptom patterns for each disease
3. Apply Bayes' theorem to compute posterior probabilities - like updating our beliefs about the diagnosis based on the patient's symptoms

### Types of Discriminant Analysis

We will explore three main approaches:

1. **Quadratic Discriminant Analysis (QDA)**: Assumes different covariance matrices for each class - like understanding that different diseases have different patterns of symptom variation
2. **Linear Discriminant Analysis (LDA)**: Assumes shared covariance matrix across classes - like assuming all diseases have similar patterns of symptom variation
3. **Naive Bayes**: Assumes conditional independence of features given class - like assuming that knowing one symptom doesn't tell you much about other symptoms for the same disease

**Intuition**: These three approaches represent different levels of complexity in modeling the relationships between features. QDA is the most flexible, allowing each class to have its own unique pattern of feature relationships. LDA is more restrictive, assuming all classes follow similar patterns. Naive Bayes is the simplest, assuming features don't influence each other within each class.

## 9.2.2. Bayes' Theorem and Optimal Classification

### Derivation of Bayes' Theorem

The optimal classifier maximizes the posterior probability $`P(Y=k \mid X=x)`$. Using Bayes' theorem:

$$ P(Y = k \mid X=x) = \frac{P(X=x, Y=k)}{P(X=x)} = \frac{P(X=x \mid Y=k) \cdot P(Y=k)}{P(X=x)} $$

**Intuition**: Bayes' theorem is like updating our beliefs based on new evidence. We start with our prior belief about how common each disease is, then we update this belief based on the patient's symptoms. The posterior probability tells us how likely each disease is given what we observe.

Let's define:
- $`f_k(x) = p(x \mid Y=k)`$: class-conditional density function - like the probability of seeing these symptoms if the patient has disease k
- $`\pi_k = P(Y=k)`$: class prior probability - like how common disease k is in the population

Then:

$$ P(Y = k \mid X=x) = \frac{\pi_k f_k(x)}{P(X=x)} \propto \pi_k f_k(x) $$

**Intuition**: The posterior probability is proportional to the product of the prior probability and the likelihood. This makes sense - we combine our baseline expectation (prior) with how well the evidence fits each hypothesis (likelihood). The denominator $`P(X=x)`$ is just a normalizing constant that ensures the probabilities sum to 1.

Since $`P(X=x)`$ is constant across all classes, the optimal classifier is:

$$ \hat{y} = \arg\max_k P(Y=k \mid X=x) = \arg\max_k \pi_k f_k(x) $$

**Intuition**: The optimal decision rule is beautifully simple: choose the class that maximizes the product of the prior probability and the likelihood. This is like saying "choose the disease that is both common and explains the symptoms well."

### Log-Likelihood Formulation

For numerical stability and computational efficiency, we often work with log-likelihoods:

$$ \hat{y} = \arg\max_k \log(\pi_k f_k(x)) = \arg\max_k [\log \pi_k + \log f_k(x)] $$

Or equivalently, minimizing the negative log-likelihood:

$$ \hat{y} = \arg\min_k [-\log \pi_k - \log f_k(x)] $$

**Intuition**: Working with logarithms has several advantages. First, it converts multiplication to addition, which is computationally more stable. Second, it prevents numerical underflow when dealing with very small probabilities. Third, it makes the optimization problem more numerically stable. The log-likelihood formulation is like working with "scores" instead of probabilities - we add up the log-prior and log-likelihood to get a total score for each class.

The Bayes Classifier framework provides the foundation for all discriminant analysis methods. The `BayesClassifier` base class implements the core functionality for estimating class priors and computing posterior probabilities using Bayes' theorem.

**Key Functions:**
- `BayesClassifier.__init__()`: Initialize the base classifier - like setting up the diagnostic system
- `BayesClassifier.fit()`: Fit the classifier by estimating class priors and conditional densities - like learning disease profiles from patient data
- `BayesClassifier.predict_proba()`: Compute posterior probabilities using log-likelihoods - like calculating the probability of each disease given the symptoms
- `BayesClassifier.predict()`: Predict class labels using maximum posterior probability - like making the most likely diagnosis
- `BayesClassifier.score()`: Compute accuracy score - like measuring how often the diagnosis is correct
- `create_gaussian_mixture_data()`: Create synthetic Gaussian mixture data for demonstrations - like creating artificial patient data for testing
- `demonstrate_bayes_classifier()`: Complete demonstration with data creation and splitting - like showing the complete diagnostic workflow

The framework uses numerical stability techniques (log-likelihoods and softmax) to handle the computational challenges of Bayes' theorem in high-dimensional spaces.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete Bayes Classifier framework.

The R implementation provides equivalent functionality using the `MASS`, `ggplot2`, and `caret` packages. The implementation demonstrates data creation, model fitting, and result visualization in R.

**Key Functions:**
- `create_gaussian_mixture_data()`: Create synthetic Gaussian mixture data - like the R version of artificial patient data
- `demonstrate_bayes_classifier()`: Complete demonstration with data creation and splitting - like the R diagnostic workflow
- `QDA()`, `LDA()`, `GaussianNaiveBayes()`: Model fitting functions - like R disease profile learning
- `plot_decision_boundaries()`: Visualization of decision boundaries - like showing how the diagnostic rules separate different diseases
- `compare_models()`: Comprehensive model comparison - like comparing different diagnostic approaches

The R implementation leverages established packages for robust and efficient discriminant analysis, providing a clean interface for both basic and advanced usage.

See the implementation in `code/r_discriminant_analysis_implementation.R` for the complete R-based discriminant analysis workflow.

## 9.2.3. Quadratic Discriminant Analysis (QDA)

### Mathematical Formulation

QDA assumes that each class follows a multivariate Gaussian distribution with its own mean and covariance matrix:

$$ f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right) $$

**Intuition**: QDA is like understanding that each disease has its own unique "fingerprint" in terms of symptoms. The mean $`\mu_k`$ represents the typical symptom profile for disease k, while the covariance matrix $`\Sigma_k`$ captures how symptoms vary and relate to each other for that disease. Different diseases might have very different patterns - one disease might have symptoms that are highly correlated (like fever and chills), while another might have more independent symptoms.

The decision function becomes:

$$ \delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k $$

**Intuition**: The decision function computes a "score" for each class. The first term $`-\frac{1}{2} \log|\Sigma_k|`$ is like a penalty for how spread out the class is - more spread out classes get a higher penalty. The second term $`-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)`$ is like measuring how far the observation is from the class center, weighted by the class's shape. The third term $`\log \pi_k`$ is like a bonus for more common classes.

### Parameter Estimation

For each class $`k`$:

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$ where $`n_k`$ is the number of samples in class $`k`$ - like estimating disease prevalence from the data
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$ - like computing the average symptom profile for each disease
3. **Class covariance**: $`\hat{\Sigma}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$ - like understanding how symptoms vary and relate to each other for each disease

**Intuition**: Parameter estimation is like learning the disease profiles from the data. We estimate how common each disease is, what the typical symptoms are for each disease, and how symptoms vary and relate to each other within each disease. This gives us a complete understanding of each disease's "signature."

The QDA implementation extends the Bayes Classifier framework to handle class-specific covariance matrices. The `QuadraticDiscriminantAnalysis` class implements the complete QDA algorithm with quadratic decision boundaries.

**Key Functions:**
- `QuadraticDiscriminantAnalysis._fit_conditional_densities()`: Fit Gaussian densities with class-specific covariances - like learning unique disease profiles
- `QuadraticDiscriminantAnalysis.decision_function()`: Compute quadratic discriminant function values - like calculating disease scores
- `plot_decision_boundaries_qda()`: Visualize QDA decision boundaries - like showing how QDA separates different diseases
- `demonstrate_qda()`: Complete demonstration with model fitting and evaluation - like showing the complete QDA diagnostic workflow

QDA is particularly effective when classes have different covariance structures, allowing for more flexible decision boundaries compared to LDA.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete QDA workflow.

## 9.2.4. Linear Discriminant Analysis (LDA)

### Mathematical Formulation

LDA assumes that all classes share the same covariance matrix $`\Sigma`$:

$$ f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right) $$

**Intuition**: LDA is like assuming that all diseases have the same "pattern" of symptom variation, but different typical symptom profiles. This is like saying that while different diseases might have different typical symptoms, the way symptoms relate to each other (correlations, variances) is the same across all diseases. This is a more restrictive assumption than QDA, but it can be more robust when you have limited data.

The decision function becomes linear:

$$ \delta_k(x) = \mu_k^T \Sigma^{-1} x - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k $$

**Intuition**: The linear decision function is much simpler than the quadratic one. It's like having a simple scoring system where you multiply each symptom by a weight and add up the scores. The weights $`\Sigma^{-1} \mu_k`$ are like the "importance" of each symptom for disease k, taking into account how symptoms relate to each other.

### Parameter Estimation

1. **Class prior**: $`\hat{\pi}_k = \frac{n_k}{n}`$ - like estimating disease prevalence
2. **Class mean**: $`\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i`$ - like computing average symptom profiles
3. **Shared covariance**: $`\hat{\Sigma} = \frac{1}{n-K} \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T`$ - like understanding the common pattern of symptom variation across all diseases

**Intuition**: The shared covariance assumption means we pool all the data to understand how symptoms relate to each other, rather than learning separate patterns for each disease. This is like saying "the way symptoms vary is similar across all diseases, so we can learn this pattern from all the data together."

The LDA implementation extends the Bayes Classifier framework to use a shared covariance matrix across all classes. The `LinearDiscriminantAnalysis` class implements the complete LDA algorithm with linear decision boundaries.

**Key Functions:**
- `LinearDiscriminantAnalysis._fit_conditional_densities()`: Fit Gaussian densities with shared covariance - like learning disease profiles with common symptom patterns
- `LinearDiscriminantAnalysis.decision_function()`: Compute linear discriminant function values - like calculating linear disease scores
- `compare_qda_lda()`: Compare QDA and LDA performance with visualization - like comparing flexible vs. simple diagnostic approaches
- `demonstrate_lda()`: Complete demonstration with model fitting and evaluation - like showing the complete LDA diagnostic workflow

LDA is particularly effective when classes have similar covariance structures, providing linear decision boundaries that are often more robust in high-dimensional spaces.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete LDA workflow.

## 9.2.5. Naive Bayes

### Mathematical Formulation

Naive Bayes assumes conditional independence of features given the class:

$$ f_k(x) = \prod_{j=1}^p f_{kj}(x_j) $$

where $`f_{kj}(x_j)`$ is the marginal density of feature $`j`$ in class $`k`$.

**Intuition**: Naive Bayes is like assuming that once you know the disease, the symptoms are independent of each other. This is like saying "if I know the patient has the flu, knowing they have a fever doesn't tell me anything about whether they have a cough - these symptoms are independent given the disease." This is often a strong assumption, but it can work surprisingly well and is computationally very efficient.

The decision function becomes:

$$ \delta_k(x) = \log \pi_k + \sum_{j=1}^p \log f_{kj}(x_j) $$

**Intuition**: The decision function is very simple - we just add up the log-probabilities of each symptom given the disease, plus the log-prior probability of the disease. This is like having a simple checklist: "How likely is fever given this disease? How likely is cough given this disease? etc." and then adding up all the evidence.

The Gaussian Naive Bayes implementation extends the Bayes Classifier framework to assume conditional independence of features given the class. The `GaussianNaiveBayes` class implements the complete naive Bayes algorithm with independent Gaussian distributions.

**Key Functions:**
- `GaussianNaiveBayes._fit_conditional_densities()`: Fit independent Gaussian densities for each feature - like learning independent symptom probabilities for each disease
- `GaussianNaiveBayes.decision_function()`: Compute naive Bayes decision function values - like calculating simple disease scores
- `demonstrate_naive_bayes()`: Complete demonstration with model fitting and evaluation - like showing the complete naive Bayes diagnostic workflow

Naive Bayes is particularly effective when features are approximately independent given the class, providing fast and often surprisingly accurate predictions even with the independence assumption.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete Gaussian Naive Bayes workflow.

## 9.2.6. Fisher's Discriminant Analysis (FDA)

### Mathematical Foundation

FDA finds a linear projection that maximizes the ratio of between-class variance to within-class variance:

$$ J(w) = \frac{w^T S_B w}{w^T S_W w} $$

where:
- $`S_B = \sum_{k=1}^K n_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^T`$ is the between-class scatter matrix - like measuring how different the disease centers are from each other
- $`S_W = \sum_{k=1}^K \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^T`$ is the within-class scatter matrix - like measuring how spread out each disease is
- $`\bar{\mu} = \frac{1}{n} \sum_{i=1}^n x_i`$ is the overall mean - like the average symptom profile across all diseases

**Intuition**: FDA is like finding the best "angle" to view the data from. We want a projection that makes the different diseases look as different as possible from each other (maximize between-class variance) while making each disease look as compact as possible (minimize within-class variance). This is like finding the best way to arrange patients in a room so that patients with the same disease are close together, but patients with different diseases are far apart.

### Solution

The optimal projection vector is the eigenvector corresponding to the largest eigenvalue of $`S_W^{-1} S_B`$:

$$ S_W^{-1} S_B w = \lambda w $$

**Intuition**: The solution involves solving an eigenvalue problem. The matrix $`S_W^{-1} S_B`$ is like a "discrimination matrix" that captures the ratio of between-class to within-class variation. The eigenvector with the largest eigenvalue gives us the direction that maximizes this ratio - the best direction for separating the classes.

The Fisher's Discriminant Analysis implementation provides dimensionality reduction by finding optimal linear projections that maximize between-class variance while minimizing within-class variance. The `FishersDiscriminantAnalysis` class implements the complete FDA algorithm.

**Key Functions:**
- `FishersDiscriminantAnalysis.__init__()`: Initialize FDA with number of components - like setting up the projection system
- `FishersDiscriminantAnalysis.fit()`: Fit FDA by computing scatter matrices and solving eigenvalue problem - like finding the best projection direction
- `FishersDiscriminantAnalysis.transform()`: Transform data using FDA projection - like projecting patients onto the best viewing angle
- `FishersDiscriminantAnalysis.fit_transform()`: Fit FDA and transform data in one step - like finding the best view and then looking at the data
- `demonstrate_fda()`: Complete demonstration with visualization and LDA application - like showing how FDA helps with classification

FDA is particularly useful for dimensionality reduction in classification problems, providing optimal projections that preserve class separability.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete FDA workflow.

## 9.2.7. Model Comparison and Selection

### Theoretical Comparison

| Method | Assumptions | Decision Boundary | Complexity |
|--------|-------------|-------------------|------------|
| **QDA** | Different covariances | Quadratic | $`O(p^2)`$ |
| **LDA** | Shared covariance | Linear | $`O(p^2)`$ |
| **Naive Bayes** | Feature independence | Piecewise linear | $`O(p)`$ |

**Intuition**: This table shows the trade-offs between different methods. QDA is the most flexible but requires the most parameters and can be prone to overfitting. LDA is more restrictive but more robust with limited data. Naive Bayes is the simplest and fastest but makes the strongest independence assumption. The choice depends on your data characteristics and computational constraints.

The comprehensive model comparison implementation provides systematic evaluation of different discriminant analysis methods, including both custom implementations and scikit-learn equivalents. The comparison framework evaluates multiple performance metrics and computational efficiency.

**Key Functions:**
- `comprehensive_model_comparison()`: Compare multiple discriminant analysis methods - like systematically testing different diagnostic approaches
- `demonstrate_model_comparison()`: Complete demonstration with visualization - like showing how different methods perform
- Evaluates accuracy, precision, recall, F1-score, fit time, and prediction time - like measuring both accuracy and efficiency
- Compares custom implementations with scikit-learn equivalents - like validating our implementations against established tools

The comparison provides insights into the relative performance of different discriminant analysis approaches and helps guide model selection decisions.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete model comparison workflow.

## 9.2.8. Practical Considerations

### Model Selection Guidelines

1. **Use LDA when**:
   - Classes have similar covariance structures - like when different diseases have similar symptom patterns
   - Sample size is small relative to number of features - like when you have few patients but many symptoms
   - Linear decision boundaries are appropriate - like when diseases can be separated by simple rules

2. **Use QDA when**:
   - Classes have different covariance structures - like when different diseases have very different symptom patterns
   - Sufficient data to estimate class-specific covariances - like when you have enough patients to learn unique disease profiles
   - Non-linear decision boundaries are needed - like when diseases require complex separation rules

3. **Use Naive Bayes when**:
   - Features are approximately independent given class - like when symptoms don't strongly influence each other
   - High-dimensional data with limited samples - like when you have many symptoms but few patients
   - Fast prediction is required - like when you need quick diagnoses

**Intuition**: These guidelines help you choose the right tool for the job. The key is to match the complexity of your model to the complexity of your data. If your data is simple, use a simple model. If your data is complex, use a more flexible model, but make sure you have enough data to support it.

### Regularization and Robustness

The regularization implementation provides techniques to improve LDA performance in high-dimensional settings where the covariance matrix may be ill-conditioned. Regularization helps stabilize parameter estimation and improve generalization.

**Intuition**: Regularization is like adding a "safety net" to prevent the model from making extreme assumptions. When you have many features but few samples, the covariance matrix can become unstable, leading to poor performance. Regularization adds a small amount of bias to reduce variance, making the model more robust.

**Key Functions:**
- `regularized_lda()`: Implement LDA with shrinkage regularization - like adding stability to the diagnostic system
- `demonstrate_regularization()`: Complete demonstration with different regularization levels - like showing how regularization improves performance
- Uses scikit-learn's LDA with shrinkage parameter for robust estimation - like using proven regularization techniques

Regularization is particularly important when the number of features is large relative to the sample size, helping to prevent overfitting and improve model stability.

See the implementation in `code/discriminant_analysis_implementation.py` for the complete regularization workflow.

---

## Code Files Summary

The discriminant analysis concepts have been implemented in the following code files:

### Python Implementation (`code/discriminant_analysis_implementation.py`)
- **Bayes Classifier Framework**: `BayesClassifier` base class with core functionality - like the foundation for all diagnostic methods
- **Quadratic Discriminant Analysis**: `QuadraticDiscriminantAnalysis` class with class-specific covariances - like flexible disease profiling
- **Linear Discriminant Analysis**: `LinearDiscriminantAnalysis` class with shared covariance - like simple disease profiling
- **Gaussian Naive Bayes**: `GaussianNaiveBayes` class with feature independence assumption - like simple symptom-based diagnosis
- **Fisher's Discriminant Analysis**: `FishersDiscriminantAnalysis` class for dimensionality reduction - like finding the best viewing angle for diagnosis
- **Model Comparison**: Comprehensive evaluation framework with multiple metrics - like systematic testing of diagnostic methods
- **Regularization**: LDA with shrinkage for robust estimation - like adding stability to diagnostic systems
- **Demonstration Functions**: Complete workflows for each method and comparison - like complete diagnostic workflows

### R Implementation (`code/r_discriminant_analysis_implementation.R`)
- **Bayes Classifier Framework**: Conceptual framework for R implementations - like the R foundation for diagnostic methods
- **QDA, LDA, Naive Bayes**: Model fitting functions using established R packages - like R disease profiling tools
- **Fisher's Discriminant Analysis**: FDA implementation using MASS package - like R dimensionality reduction for diagnosis
- **Visualization**: Decision boundary plotting and FDA projection visualization - like R diagnostic visualization tools
- **Model Comparison**: Comprehensive comparison framework in R - like R systematic testing of diagnostic methods
- **Regularization**: Regularized LDA with shrinkage parameter - like R stable diagnostic systems
- **Demonstration Functions**: Complete workflows for each method - like R complete diagnostic workflows

Both implementations provide comprehensive coverage of discriminant analysis concepts with practical examples and demonstrate the relationship between theoretical foundations and practical applications in classification problems.

---

**Navigation:**
- **Next Topic:** [Quadratic Discriminant Analysis](03_quadratic_discriminant_analysis.md) - Multivariate normal distribution assumption and quadratic decision boundaries
- **Previous Topic:** [Introduction to Classification](01_classification.md) - Definition of classification problems and optimal classifier derivation
