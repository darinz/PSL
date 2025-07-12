# Introduction to Statistical Learning

Statistical learning is a fundamental framework for understanding how to extract meaningful patterns from data and make predictions about future observations. This field combines principles from statistics, computer science, and mathematics to develop algorithms that can learn from data without being explicitly programmed for every possible scenario.

## 1.1.1. Types of Statistical Learning Problems

Statistical learning problems can be broadly categorized based on the nature of the data and the learning objectives. Understanding these categories is crucial for selecting appropriate methods and interpreting results correctly.

### Supervised Learning: Predicting Numerical Values (Regression)

Supervised learning addresses problems where we have a target variable, denoted as $`Y`$, and a set of features or covariates, represented as $`X`$, which is typically a multidimensional vector. Our goal is to build a predictive model $`f: \mathcal{X} \rightarrow \mathcal{Y}`$ that maps input features to target values.

**Mathematical Framework:**
Given a training dataset $`\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}`$ where:
- $`x_i \in \mathbb{R}^p`$ represents the feature vector for the $`i`$-th observation
- $`y_i \in \mathbb{R}`$ represents the target value for the $`i`$-th observation
- $`n`$ is the number of training examples
- $`p`$ is the number of features

We seek to find a function $`f`$ that minimizes the expected prediction error:

```math
f^* = \arg\min_{f \in \mathcal{F}} \mathbb{E}_{(X,Y)}[L(Y, f(X))]
```

where $`L`$ is a loss function and $`\mathcal{F}`$ is the class of functions we consider.

**Real-World Examples:**
- **Project 1**: Predicting house sale prices in Ames, Iowa based on features like square footage, number of bedrooms, location, etc.
- **Project 2**: Forecasting Walmart store sales at a department level using historical sales data, promotional events, and seasonal patterns.

In these examples, the target variable $`Y`$ takes continuous numerical values, making them **regression** problems. The goal is to predict a continuous outcome rather than discrete categories.

### Supervised Learning: Classifying Categorical Data (Classification)

Classification problems involve predicting categorical outcomes where $`Y`$ takes discrete values from a finite set of classes. The mathematical framework is similar, but the target space is discrete.

**Mathematical Framework:**
For classification with $`K`$ classes, we have:
- $`y_i \in \{1, 2, \ldots, K\}`$ for the $`i`$-th observation
- The model $`f: \mathbb{R}^p \rightarrow \{1, 2, \ldots, K\}`$ maps features to class labels
- Often, we work with probability estimates $`P(Y = k | X = x)`$ for each class $`k`$

The optimal classifier (Bayes classifier) is given by:

```math
f^*(x) = \arg\max_{k \in \{1,\ldots,K\}} P(Y = k | X = x)
```

**Real-World Examples:**
- **Project 3**: Determining whether movie reviews are positive or negative based on text content
- **Credit Risk Assessment**: Predicting whether a borrower will default on a loan based on financial history, income, and other characteristics

These are **classification** problems where the goal is to assign observations to predefined categories.

### Unsupervised Learning: Discovering Hidden Patterns

Unsupervised learning operates without target variables. Instead, we seek to discover intrinsic structure, patterns, or relationships within the data itself.

**Mathematical Framework:**
Given only feature data $`\mathcal{D} = \{x_1, x_2, \ldots, x_n\}`$ where $`x_i \in \mathbb{R}^p`$, we aim to:
- Find clusters or groups in the data
- Discover latent variables or factors
- Identify associations and dependencies
- Reduce dimensionality while preserving important structure

**Key Techniques:**
1. **Clustering**: Partition data into groups based on similarity
2. **Dimensionality Reduction**: Find lower-dimensional representations
3. **Association Rules**: Discover relationships between variables
4. **Density Estimation**: Model the underlying data distribution

**Real-World Examples:**
- **Market Segmentation**: Identifying distinct customer groups based on purchasing behavior
- **Recommendation Systems**: Finding associations between products purchased together
- **Anomaly Detection**: Identifying unusual patterns in network traffic or financial transactions

### Summary of Statistical Learning Problem Types

| Learning Type | Target Variable | Goal | Example |
|---------------|-----------------|------|---------|
| **Supervised - Regression** | Continuous $`Y \in \mathbb{R}`$ | Predict numerical values | House price prediction |
| **Supervised - Classification** | Categorical $`Y \in \{1,\ldots,K\}`$ | Assign to categories | Spam detection |
| **Unsupervised** | None | Discover patterns | Customer segmentation |

### Beyond the Basics: Advanced Learning Paradigms

Real-world problems often don't fit neatly into these categories, leading to hybrid approaches:

**Semi-Supervised Learning:**
When labeled data is scarce or expensive to obtain, we can leverage both labeled and unlabeled data:

```math
\mathcal{D}_{\text{labeled}} = \{(x_1, y_1), \ldots, (x_l, y_l)\}
```
```math
\mathcal{D}_{\text{unlabeled}} = \{x_{l+1}, \ldots, x_n\}
```

The goal is to use the unlabeled data to improve the model learned from the limited labeled data.

**Active Learning:**
Instead of passively receiving labeled data, the algorithm actively selects which examples to label, maximizing information gain.

**Transfer Learning:**
Leverage knowledge learned from one task to improve performance on a related task, even when the data distributions differ.

## 1.1.2. The Challenge of Supervised Learning

Supervised learning appears deceptively simple: collect data, build a model, and make predictions. However, the fundamental challenge lies in the tension between fitting the training data well and generalizing to unseen data.

### The Learning Process

<img src="./img/supervised_learning.png" width="500px"/>

**Step 1: Data Collection**
We start with a collection of $`n`$ training examples $`\mathcal{D} = \{(x_1, y_1), \ldots, (x_n, y_n)\}`$. Each $`x_i \in \mathbb{R}^p`$ is a feature vector, and $`y_i`$ is the corresponding target value.

**Step 2: Model Specification**
We choose a family of functions $`\mathcal{F} = \{f(x; \theta) : \theta \in \Theta\}`$ parameterized by $`\theta`$. Common choices include:
- Linear models: $`f(x; w, b) = w^T x + b`$
- Neural networks: $`f(x; W, b) = \sigma(W^T x + b)`$
- Decision trees: Piecewise constant functions

**Step 3: Loss Function Definition**
We define a loss function $`L(y, \hat{y})`$ that measures the cost of predicting $`\hat{y}`$ when the true value is $`y`$:

- **Regression**: Mean Squared Error (MSE)
```math
L(y, \hat{y}) = (y - \hat{y})^2
```

- **Classification**: Cross-entropy loss
```math
L(y, \hat{y}) = -\sum_{k=1}^K y_k \log(\hat{y}_k)
```

**Step 4: Empirical Risk Minimization**
We minimize the empirical risk (average loss on training data):

```math
\hat{\theta} = \arg\min_{\theta \in \Theta} \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i; \theta))
```

### The Fundamental Challenge: Generalization

The core challenge in supervised learning is that minimizing training error doesn't guarantee good performance on new data. This is formalized through the concept of **generalization error**:

```math
R(f) = \mathbb{E}_{(X,Y)}[L(Y, f(X))]
```

The generalization error measures the expected loss on unseen data drawn from the same distribution as the training data.

**The Bias-Variance Tradeoff:**
The generalization error can be decomposed into three components:

```math
\mathbb{E}[(Y - f(X))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
```

where:
- **Bias**: $`\text{Bias}^2 = (\mathbb{E}[f(X)] - f^*(X))^2`$ - how far our predictions are from the true function on average
- **Variance**: $`\text{Var}(f(X))`$ - how much our predictions vary across different training sets
- **Irreducible Error**: $`\text{Var}(\epsilon)`$ - noise in the data that cannot be predicted

### Overfitting: The Central Problem

**Overfitting** occurs when a model learns the training data too well, including noise and idiosyncrasies, leading to poor generalization. This happens when:

1. The model is too complex relative to the amount of training data
2. The number of parameters $`p`$ is large compared to the sample size $`n`$
3. The training data contains noise or outliers

**Mathematical Characterization:**
Let $`\hat{f}_n`$ be the model learned from $`n`$ training examples. Overfitting occurs when:

```math
\text{Training Error}(\hat{f}_n) \ll \text{Test Error}(\hat{f}_n)
```

**Prevention Strategies:**
1. **Regularization**: Add penalty terms to the loss function
```math
\hat{\theta} = \arg\min_{\theta} \left\{ \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i; \theta)) + \lambda \Omega(\theta) \right\}
```

2. **Cross-Validation**: Use held-out data to estimate generalization error
3. **Early Stopping**: Stop training before the model overfits
4. **Model Selection**: Choose simpler models when data is limited

### Learning vs. Optimization: A Critical Distinction

While optimization is essential for learning, it's crucial to understand that learning is not just optimization. The key insight is that we optimize an **empirical risk** $`R_n(f)`$ but care about the **true risk** $`R(f)`$:

```math
R_n(f) = \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i))
```
```math
R(f) = \mathbb{E}_{(X,Y)}[L(Y, f(X))]
```

**The Learning Guarantee:**
Under certain conditions, we can bound the difference between empirical and true risk:

```math
P(|R_n(f) - R(f)| > \epsilon) \leq 2\exp(-2n\epsilon^2)
```

This bound shows that as $`n \rightarrow \infty`$, $`R_n(f) \rightarrow R(f)`$ with high probability.

**Practical Implications:**
- Sometimes suboptimal solutions (e.g., from gradient descent) generalize better than exact optima
- The goal is good generalization, not perfect optimization
- Regularization often improves generalization even when it increases training error

## 1.1.3. The Curse of Dimensionality

The curse of dimensionality refers to the phenomenon where the performance of many algorithms deteriorates as the number of features (dimensions) increases, even when the additional features contain no useful information.

### Mathematical Intuition

Consider a unit hypercube in $`p`$ dimensions. The volume of a sphere inscribed in this cube decreases exponentially with dimension:

```math
V_{\text{sphere}} = \frac{\pi^{p/2}}{\Gamma(p/2 + 1)} \left(\frac{1}{2}\right)^p
```

As $`p \rightarrow \infty`$, $`V_{\text{sphere}} \rightarrow 0`$, meaning most of the volume is concentrated in the corners.

**Implications for Learning:**
- Data becomes increasingly sparse in high dimensions
- Distance metrics become less meaningful
- The "neighborhood" of any point becomes empty

### Impact on k-Nearest Neighbors (kNN)

The kNN algorithm is particularly susceptible to the curse of dimensionality. Consider the 1-NN classifier:

**Algorithm:**
1. For a new point $`x`$, find the nearest neighbor $`x_i`$ in the training set
2. Predict $`y_i`$ as the class for $`x`$

**Mathematical Analysis:**
Let $`d(x, x_i)`$ be the Euclidean distance between points. In high dimensions, all points become approximately equidistant:

```math
\lim_{p \rightarrow \infty} \frac{\max_{i,j} d(x_i, x_j) - \min_{i,j} d(x_i, x_j)}{\min_{i,j} d(x_i, x_j)} = 0
```

This means kNN loses its discriminative power in high dimensions.

### Impact on Linear Classifiers

Linear classifiers face a different but related challenge. Consider the linear model:

```math
f(x; w) = w^T x + b
```

**The Interpolation Problem:**
When $`p \geq n`$ (more features than samples), we can often find a perfect fit to the training data:

```math
\exists w \text{ such that } w^T x_i + b = y_i \text{ for all } i = 1, \ldots, n
```

This perfect fit on training data typically corresponds to poor generalization.

**Mathematical Illustration:**
For $`p = n`$, the system of equations $`Xw = y`$ has a unique solution when $`X`$ is full rank. This solution achieves zero training error but may have high test error.

<img src="./img/linear_function.png" width="500"/>

### Strategies for Combating the Curse of Dimensionality

**1. Feature Selection:**
Choose a subset of relevant features:
```math
\mathcal{S} \subset \{1, 2, \ldots, p\}, \quad |\mathcal{S}| \ll p
```

**2. Dimensionality Reduction:**
Project data to a lower-dimensional space:
```math
z = W^T x, \quad W \in \mathbb{R}^{p \times k}, \quad k \ll p
```

**3. Regularization:**
Add constraints to prevent overfitting:
```math
\|w\|_1 \leq t \quad \text{(Lasso)} \quad \text{or} \quad \|w\|_2 \leq t \quad \text{(Ridge)}
```

**4. Kernel Methods:**
Work in high-dimensional feature spaces implicitly through kernels:
```math
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
```

### Practical Guidelines

1. **Collect more data** when possible (increase $`n`$)
2. **Use domain knowledge** to select relevant features
3. **Apply regularization** to prevent overfitting
4. **Consider simpler models** when data is limited
5. **Use cross-validation** to estimate generalization error

The curse of dimensionality is a fundamental challenge in statistical learning that requires careful consideration of the trade-off between model complexity and available data. Understanding this phenomenon is crucial for developing effective learning algorithms and interpreting their performance.

