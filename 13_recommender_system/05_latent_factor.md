# 13.5. Latent Factor Models

Latent factor models represent a powerful approach in recommendation systems that goes beyond simple similarity-based methods. These models discover hidden patterns in user-item interactions by decomposing the rating matrix into lower-dimensional representations. Unlike collaborative filtering methods that rely on explicit similarities, latent factor models learn implicit patterns that may not be immediately obvious from the raw data.

## 13.5.1. Introduction to Latent Factor Models

### Core Concept and Motivation

**Why Latent Factors?** Traditional collaborative filtering methods face several limitations:
- **Sparsity Problem**: Most users rate only a small fraction of available items
- **Scalability Issues**: Computing similarities becomes expensive with large datasets
- **Cold Start**: New users/items have limited interaction data
- **Noise in Explicit Similarities**: Direct similarity measures can be misleading

Latent factor models address these issues by discovering underlying, unobservable factors that influence user preferences and item characteristics. These factors are learned automatically from the data through matrix factorization techniques, providing a more robust and scalable approach.

### Mathematical Foundation

#### The Matrix Factorization Problem

The rating matrix $`R \in \mathbb{R}^{n \times m}`$ is approximated as:

```math
R \approx U \cdot V^T
```

where:
- $`U \in \mathbb{R}^{n \times k}`$ is the user factor matrix (each row represents a user's preferences)
- $`V \in \mathbb{R}^{m \times k}`$ is the item factor matrix (each row represents an item's characteristics)
- $`k`$ is the number of latent factors (typically $`k \ll \min(n, m)`$)

#### Detailed Mathematical Formulation

Each user $`u`$ is represented by a vector $`\mathbf{u}_u \in \mathbb{R}^k`$, and each item $`i`$ by a vector $`\mathbf{v}_i \in \mathbb{R}^k`$. The predicted rating is computed as the dot product:

```math
\hat{r}_{ui} = \mathbf{u}_u^T \mathbf{v}_i = \sum_{f=1}^k u_{uf} \cdot v_{if}
```

This formulation has several important properties:

1. **Linear Combination**: The rating is a weighted sum of factor contributions
2. **Dimensionality Reduction**: High-dimensional user-item space is compressed to $`k`$ dimensions
3. **Interpretability**: Each factor can represent a meaningful concept (e.g., action level, complexity)

#### Geometric Interpretation

The factorization can be viewed geometrically:
- **User Space**: Each user is a point in $`\mathbb{R}^k`$ space
- **Item Space**: Each item is a point in $`\mathbb{R}^k`$ space
- **Similarity**: Users/items with similar factor vectors are "close" in this space
- **Rating**: The dot product measures the alignment between user preferences and item characteristics

### Intuitive Interpretation with Examples

#### Movie Recommendation Example

Consider a movie recommendation system with $`k=3`$ latent factors:

**Factor 1: Action Level**
- User factors: How much a user enjoys action movies
- Item factors: How action-oriented a movie is
- High positive values: User loves action, movie is very action-packed
- High negative values: User dislikes action, movie is very action-packed

**Factor 2: Complexity/Artistic Merit**
- User factors: User's tolerance for complex, artistic films
- Item factors: Movie's complexity and artistic ambition
- High positive values: User appreciates complex films, movie is sophisticated
- High negative values: User prefers simple films, movie is sophisticated

**Factor 3: Genre Preference**
- User factors: User's preference for certain genres
- Item factors: Movie's genre characteristics
- High positive values: User loves this genre, movie strongly fits this genre

#### Mathematical Example

For a user with factor vector $`\mathbf{u}_u = [0.8, -0.3, 0.5]`$ and a movie with factor vector $`\mathbf{v}_i = [0.9, 0.2, 0.7]`$:

```math
\hat{r}_{ui} = 0.8 \times 0.9 + (-0.3) \times 0.2 + 0.5 \times 0.7 = 0.72 - 0.06 + 0.35 = 1.01
```

This low predicted rating (1.01) suggests the user would not enjoy this movie, likely because:
- The user dislikes action (0.8 × 0.9 = 0.72 positive contribution)
- The user dislikes complexity but the movie is somewhat complex (-0.3 × 0.2 = -0.06 negative contribution)
- The user likes this genre and the movie fits it well (0.5 × 0.7 = 0.35 positive contribution)

### Advantages of Latent Factor Models

1. **Dimensionality Reduction**: Compresses high-dimensional data to manageable size
2. **Noise Reduction**: Filters out random variations in ratings
3. **Scalability**: Efficient for large datasets
4. **Cold Start Mitigation**: Can incorporate side information
5. **Interpretability**: Factors can have meaningful interpretations

## 13.5.2. Matrix Factorization

### Basic Matrix Factorization

#### The Optimization Problem

The goal is to minimize the reconstruction error over all observed ratings:

```math
\min_{U, V} \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2
```

where $`\mathcal{R}`$ is the set of observed ratings. This is a **non-convex optimization problem** because the objective function is not convex in both $`U`$ and $`V`$ simultaneously.

#### Mathematical Properties

1. **Non-convexity**: The objective function has multiple local minima
2. **Identifiability**: The factorization is not unique (e.g., $`U \cdot V^T = (U \cdot Q) \cdot (V \cdot Q^{-1})^T`$ for any orthogonal matrix $`Q`$)
3. **Sparsity**: Only observed ratings contribute to the loss function

#### Why This Formulation Works

The squared error loss function has several desirable properties:
- **Differentiability**: Smooth gradients for optimization
- **Symmetry**: Treats over- and under-predictions equally
- **Convexity in each variable**: When fixing one matrix, the problem becomes convex in the other

### Regularized Matrix Factorization

#### The Need for Regularization

Without regularization, the model can overfit to the training data, especially when the number of factors is large relative to the number of observations. Regularization helps by:

1. **Preventing Overfitting**: Constrains the magnitude of factor values
2. **Improving Generalization**: Better performance on unseen data
3. **Numerical Stability**: Prevents factors from growing too large

#### Regularized Objective Function

```math
\min_{U, V} \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
```

where:
- $`\lambda`$ is the regularization parameter (controls the trade-off between fit and complexity)
- $`\| \cdot \|_F`$ is the Frobenius norm: $`\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2}`$

#### Understanding the Regularization Term

The Frobenius norm regularization can be written as:

```math
\|U\|_F^2 + \|V\|_F^2 = \sum_{u=1}^n \sum_{f=1}^k u_{uf}^2 + \sum_{i=1}^m \sum_{f=1}^k v_{if}^2
```

This is equivalent to placing a **Gaussian prior** on each factor value:
- $`u_{uf} \sim \mathcal{N}(0, \frac{1}{2\lambda})`$
- $`v_{if} \sim \mathcal{N}(0, \frac{1}{2\lambda})`$

#### Choosing the Regularization Parameter

The optimal $`\lambda`$ depends on:
- **Data sparsity**: More sparse data typically needs more regularization
- **Number of factors**: More factors require more regularization
- **Noise level**: Noisier data benefits from stronger regularization

### Stochastic Gradient Descent (SGD)

#### Why SGD?

The optimization problem is typically solved using SGD because:
1. **Scalability**: Processes one rating at a time, memory efficient
2. **Simplicity**: Easy to implement and understand
3. **Convergence**: Works well for non-convex problems
4. **Parallelization**: Can be easily parallelized

#### The Update Rules

For each observed rating $`(u, i, r_{ui})`$, the factors are updated as:

```math
\mathbf{u}_u \leftarrow \mathbf{u}_u + \gamma \cdot (e_{ui} \cdot \mathbf{v}_i - \lambda \cdot \mathbf{u}_u)
```

```math
\mathbf{v}_i \leftarrow \mathbf{v}_i + \gamma \cdot (e_{ui} \cdot \mathbf{u}_u - \lambda \cdot \mathbf{v}_i)
```

where:
- $`e_{ui} = r_{ui} - \hat{r}_{ui}`$ is the prediction error
- $`\gamma`$ is the learning rate (controls step size)

#### Mathematical Derivation

The update rules come from computing the gradients of the objective function:

**For user factors:**
```math
\frac{\partial}{\partial \mathbf{u}_u} \left[ (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \|\mathbf{u}_u\|^2 \right] = -2e_{ui} \mathbf{v}_i + 2\lambda \mathbf{u}_u
```

**For item factors:**
```math
\frac{\partial}{\partial \mathbf{v}_i} \left[ (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda \|\mathbf{v}_i\|^2 \right] = -2e_{ui} \mathbf{u}_u + 2\lambda \mathbf{v}_i
```

#### Learning Rate Scheduling

The learning rate $`\gamma`$ is crucial for convergence:
- **Too high**: May cause divergence or oscillation
- **Too low**: Slow convergence
- **Common strategy**: Start with $`\gamma = 0.01`$ and decrease over time

#### Convergence Criteria

SGD typically converges when:
1. **Maximum epochs reached**
2. **Error threshold met**: $`\frac{1}{|\mathcal{R}|} \sum_{(u,i) \in \mathcal{R}} e_{ui}^2 < \epsilon`$
3. **No improvement**: Error doesn't decrease for several epochs

### Alternative Optimization Methods

#### Alternating Least Squares (ALS)

Instead of SGD, ALS fixes one matrix and solves for the other:

**Step 1**: Fix $`V`$, solve for $`U`$:
```math
\mathbf{u}_u = \left( \sum_{i \in \mathcal{I}_u} \mathbf{v}_i \mathbf{v}_i^T + \lambda I \right)^{-1} \sum_{i \in \mathcal{I}_u} r_{ui} \mathbf{v}_i
```

**Step 2**: Fix $`U`$, solve for $`V`$:
```math
\mathbf{v}_i = \left( \sum_{u \in \mathcal{U}_i} \mathbf{u}_u \mathbf{u}_u^T + \lambda I \right)^{-1} \sum_{u \in \mathcal{U}_i} r_{ui} \mathbf{u}_u
```

**Advantages of ALS:**
- **Parallelizable**: Can update all users/items simultaneously
- **Deterministic**: No randomness in updates
- **Faster convergence**: Often converges in fewer iterations

**Disadvantages of ALS:**
- **Memory intensive**: Requires storing full matrices
- **Less scalable**: May not work for very large datasets

## 13.5.3. Advanced Latent Factor Models

### SVD++ (Singular Value Decomposition Plus Plus)

#### Motivation and Intuition

SVD++ extends basic matrix factorization by incorporating additional information that can improve prediction accuracy:

1. **Global Effects**: Some users rate higher/lower on average
2. **Item Effects**: Some items receive higher/lower ratings on average
3. **Implicit Feedback**: Even without explicit ratings, user behavior provides information

#### Mathematical Formulation

SVD++ incorporates implicit feedback and user/item biases:

```math
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i + \mathbf{u}_u^T \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} \mathbf{y}_j
```

where:
- $`\mu`$ is the global mean rating (baseline for all predictions)
- $`b_u`$ is the user bias (how much user $`u`$ deviates from the global mean)
- $`b_i`$ is the item bias (how much item $`i`$ deviates from the global mean)
- $`N(u)`$ is the set of items rated by user $`u`$
- $`\mathbf{y}_j`$ are item factors for implicit feedback (learned from user behavior)

#### Understanding Each Component

**Global Mean ($`\mu`$):**
- Represents the average rating across all users and items
- Provides a baseline prediction when no other information is available

**User Bias ($`b_u`$):**
- Captures individual user tendencies (e.g., some users are generally more generous with ratings)
- Can be computed as: $`b_u = \frac{1}{|I_u|} \sum_{i \in I_u} (r_{ui} - \mu)`$

**Item Bias ($`b_i`$):**
- Captures item-specific effects (e.g., popular items tend to get higher ratings)
- Can be computed as: $`b_i = \frac{1}{|U_i|} \sum_{u \in U_i} (r_{ui} - \mu - b_u)`$

**Implicit Feedback Term:**
- $`\frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} \mathbf{y}_j`$ represents the user's implicit preferences
- The normalization factor $`\frac{1}{\sqrt{|N(u)|}}`$ prevents users with many ratings from dominating
- $`\mathbf{y}_j`$ factors are learned from the data and capture implicit item characteristics

#### Optimization

The objective function becomes:

```math
\min_{U, V, Y, b_u, b_i} \sum_{(u,i) \in \mathcal{R}} \left( r_{ui} - \mu - b_u - b_i - \mathbf{u}_u^T \mathbf{v}_i - \mathbf{u}_u^T \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} \mathbf{y}_j \right)^2 + \lambda \left( \|U\|_F^2 + \|V\|_F^2 + \|Y\|_F^2 + \|b_u\|^2 + \|b_i\|^2 \right)
```

### Non-negative Matrix Factorization (NMF)

#### Motivation

NMF constrains factors to be non-negative, which can provide more interpretable results in many domains where negative values don't make sense (e.g., user preferences, item characteristics).

#### Mathematical Formulation

NMF constrains factors to be non-negative:

```math
\min_{U, V} \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2
```

subject to $`U \geq 0`$ and $`V \geq 0`$.

#### Optimization Challenges

The non-negativity constraint makes optimization more challenging:
- **Non-convex**: The problem remains non-convex
- **Local minima**: Many local minima due to the constraint
- **Initialization sensitive**: Results depend heavily on initialization

#### Update Rules

Multiplicative update rules are commonly used:

```math
u_{uf} \leftarrow u_{uf} \frac{\sum_{i \in \mathcal{I}_u} r_{ui} v_{if}}{\sum_{i \in \mathcal{I}_u} \hat{r}_{ui} v_{if}}
```

```math
v_{if} \leftarrow v_{if} \frac{\sum_{u \in \mathcal{U}_i} r_{ui} u_{uf}}{\sum_{u \in \mathcal{U}_i} \hat{r}_{ui} u_{uf}}
```

#### Advantages of NMF

1. **Interpretability**: Non-negative factors are often easier to interpret
2. **Additive nature**: Factors contribute positively to the prediction
3. **Sparsity**: Often produces sparse factor representations

#### Disadvantages of NMF

1. **Convergence**: May converge to poor local minima
2. **Sensitivity**: Results depend on initialization
3. **Flexibility**: Less flexible than unconstrained factorization

### Probabilistic Matrix Factorization (PMF)

#### Bayesian Framework

PMF provides a probabilistic interpretation of matrix factorization, which offers several advantages:
- **Uncertainty quantification**: Can estimate prediction uncertainty
- **Bayesian inference**: Can incorporate prior knowledge
- **Model selection**: Can use Bayesian model selection criteria

#### Mathematical Formulation

PMF models the ratings as:

```math
r_{ui} \sim \mathcal{N}(\mathbf{u}_u^T \mathbf{v}_i, \sigma^2)
```

with priors:

```math
\mathbf{u}_u \sim \mathcal{N}(0, \sigma_u^2 I)
```

```math
\mathbf{v}_i \sim \mathcal{N}(0, \sigma_v^2 I)
```

#### Understanding the Model

**Likelihood Function:**
```math
p(R|U, V, \sigma^2) = \prod_{(u,i) \in \mathcal{R}} \mathcal{N}(r_{ui}|\mathbf{u}_u^T \mathbf{v}_i, \sigma^2)
```

**Posterior Distribution:**
```math
p(U, V|R, \sigma^2, \sigma_u^2, \sigma_v^2) \propto p(R|U, V, \sigma^2) \cdot p(U|\sigma_u^2) \cdot p(V|\sigma_v^2)
```

#### Maximum A Posteriori (MAP) Estimation

The MAP estimate maximizes the log posterior:

```math
\max_{U, V} \sum_{(u,i) \in \mathcal{R}} \left( -\frac{1}{2\sigma^2}(r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 \right) + \sum_u \left( -\frac{1}{2\sigma_u^2} \|\mathbf{u}_u\|^2 \right) + \sum_i \left( -\frac{1}{2\sigma_v^2} \|\mathbf{v}_i\|^2 \right)
```

This is equivalent to the regularized matrix factorization with $`\lambda_u = \frac{\sigma^2}{\sigma_u^2}`$ and $`\lambda_v = \frac{\sigma^2}{\sigma_v^2}`$.

#### Advantages of PMF

1. **Uncertainty**: Can estimate prediction uncertainty
2. **Flexibility**: Can incorporate different prior distributions
3. **Interpretability**: Provides probabilistic interpretation
4. **Model selection**: Can use Bayesian criteria for model selection

#### Hyperparameter Selection

The model has several hyperparameters:
- $`\sigma^2`$: Observation noise variance
- $`\sigma_u^2`$: User factor prior variance
- $`\sigma_v^2`$: Item factor prior variance

These can be selected using:
- **Cross-validation**: Grid search over hyperparameter space
- **Bayesian optimization**: More efficient hyperparameter tuning
- **Empirical Bayes**: Estimate from data

## 13.5.4. Theoretical Foundations and Mathematical Insights

### Understanding the Factorization Problem

#### Why Matrix Factorization Works

The success of matrix factorization can be understood through several theoretical perspectives:

**1. Low-Rank Approximation**
The rating matrix $`R`$ is assumed to have a low-rank structure, meaning it can be well-approximated by a product of two low-rank matrices. This assumption is reasonable because:
- User preferences are influenced by a small number of underlying factors
- Items can be characterized by a limited set of features
- The true rating matrix is often approximately low-rank

**2. Dimensionality Reduction**
Matrix factorization performs dimensionality reduction from the high-dimensional user-item space to a low-dimensional factor space:
- **Original space**: $`\mathbb{R}^{n \times m}`$ (user-item pairs)
- **Factor space**: $`\mathbb{R}^k`$ (latent factors)
- **Compression ratio**: $`\frac{k(n+m)}{nm}`$ (typically very small)

**3. Noise Reduction**
The factorization process acts as a denoising mechanism:
- **Signal**: True user preferences and item characteristics
- **Noise**: Random variations, measurement errors, temporary preferences
- **Factorization**: Separates signal from noise through low-rank approximation

#### Mathematical Properties

**Uniqueness and Identifiability**
The factorization $`R = UV^T`$ is not unique. For any invertible matrix $`Q \in \mathbb{R}^{k \times k}`$:
```math
R = UV^T = (UQ)(VQ^{-T})^T
```

This means:
- The factors themselves are not uniquely determined
- Only the product $`UV^T`$ is unique
- The learned factors are one possible representation among many equivalent ones

**Optimality Conditions**
For the regularized objective function:
```math
\mathcal{L}(U, V) = \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \mathbf{u}_u^T \mathbf{v}_i)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
```

The optimality conditions are:
```math
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_u} = -2 \sum_{i \in \mathcal{I}_u} e_{ui} \mathbf{v}_i + 2\lambda \mathbf{u}_u = 0
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_i} = -2 \sum_{u \in \mathcal{U}_i} e_{ui} \mathbf{u}_u + 2\lambda \mathbf{v}_i = 0
```

### Convergence Analysis

#### SGD Convergence

**Assumptions:**
1. **Bounded gradients**: $`\|\nabla \mathcal{L}\| \leq G`$ for some constant $`G`$
2. **Lipschitz continuity**: $`\|\nabla \mathcal{L}(x) - \nabla \mathcal{L}(y)\| \leq L\|x - y\|`$
3. **Convexity in each variable**: When fixing one matrix, the problem is convex in the other

**Convergence Rate:**
For SGD with learning rate $`\gamma_t = \frac{c}{\sqrt{t}}`$, the convergence rate is:
```math
\mathbb{E}[\mathcal{L}(\bar{x}_T) - \mathcal{L}(x^*)] \leq \frac{G^2 c}{\sqrt{T}}
```

where $`\bar{x}_T`$ is the average of all iterates and $`x^*`$ is the optimal solution.

#### ALS Convergence

ALS has better convergence properties than SGD:
- **Linear convergence**: $`\|\mathcal{L}^{(t)} - \mathcal{L}^*\| \leq \rho^t \|\mathcal{L}^{(0)} - \mathcal{L}^*\|`$
- **Deterministic**: No randomness in updates
- **Fewer iterations**: Typically converges in 10-50 iterations

### Statistical Properties

#### Bias-Variance Trade-off

The regularization parameter $`\lambda`$ controls the bias-variance trade-off:

**Low $`\lambda`$ (Under-regularization):**
- **Low bias**: Model fits training data well
- **High variance**: Poor generalization to unseen data
- **Overfitting**: Model memorizes training data

**High $`\lambda`$ (Over-regularization):**
- **High bias**: Model underfits training data
- **Low variance**: Good generalization
- **Underfitting**: Model is too simple

#### Optimal Regularization

The optimal $`\lambda`$ can be found through cross-validation:
```math
\lambda^* = \arg\min_{\lambda} \text{CV}(\lambda)
```

where $`\text{CV}(\lambda)`$ is the cross-validation error.

### Model Selection

#### Choosing the Number of Factors

The number of latent factors $`k`$ is a crucial hyperparameter:

**Too few factors:**
- **Underfitting**: Model cannot capture complex patterns
- **High bias**: Predictions are too simple
- **Poor performance**: Low prediction accuracy

**Too many factors:**
- **Overfitting**: Model memorizes training data
- **High variance**: Poor generalization
- **Computational cost**: More expensive to train

**Selection Criteria:**

1. **Cross-validation**: Choose $`k`$ that minimizes validation error
2. **Information criteria**: AIC, BIC, or similar
3. **Eigenvalue analysis**: Analyze singular values of the rating matrix
4. **Elbow method**: Plot validation error vs. $`k`$ and choose the elbow point

#### Information Criteria

For probabilistic models, we can use:
- **AIC**: $`\text{AIC} = 2k - 2\ln(\mathcal{L})`$
- **BIC**: $`\text{BIC} = k\ln(n) - 2\ln(\mathcal{L})`$

where $`\mathcal{L}`$ is the likelihood and $`k`$ is the number of parameters.

### Theoretical Guarantees

#### Recovery Guarantees

Under certain conditions, matrix factorization can recover the true underlying factors:

**Assumptions:**
1. **Low-rank**: True matrix has rank $`r \ll \min(n,m)`$
2. **Incoherence**: Factors are not too sparse or structured
3. **Random sampling**: Observed entries are sampled uniformly at random
4. **Noise**: Additive Gaussian noise with bounded variance

**Recovery Result:**
With high probability, if the number of observed entries satisfies:
```math
|\mathcal{R}| \geq C \cdot r \cdot (n + m) \cdot \log(n + m)
```

then the true matrix can be recovered up to a small error.

#### Generalization Bounds

For a learned model with parameters $`\theta`$, the generalization error is bounded by:
```math
\mathbb{E}[L(\theta)] \leq \hat{L}(\theta) + O\left(\sqrt{\frac{k(n+m)}{|\mathcal{R}|}}\right)
```

where $`\hat{L}(\theta)`$ is the training error and the second term is the generalization gap.

### Computational Complexity

#### Time Complexity

**SGD per iteration:**
- **Update time**: $`O(k)`$ per rating
- **Total time**: $`O(T \cdot |\mathcal{R}| \cdot k)`$ where $`T`$ is the number of epochs

**ALS per iteration:**
- **User update**: $`O(n \cdot k^2 + |\mathcal{R}| \cdot k)`$
- **Item update**: $`O(m \cdot k^2 + |\mathcal{R}| \cdot k)`$
- **Total time**: $`O(T \cdot (n + m) \cdot k^2 + T \cdot |\mathcal{R}| \cdot k)`$

#### Space Complexity

**Storage requirements:**
- **User factors**: $`O(n \cdot k)`$
- **Item factors**: $`O(m \cdot k)`$
- **Total**: $`O((n + m) \cdot k)`$

**Memory efficiency:**
- **SGD**: Can process ratings one at a time
- **ALS**: Requires storing full matrices in memory

## 13.5.5. Implementation

### Python Implementation: Latent Factor Models

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')

class LatentFactorModel:
    """Basic Latent Factor Model with SGD optimization"""
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, 
                 n_epochs=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.training_history = []
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the latent factor model"""
        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
        # Initialize factors and biases
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        
        # Compute global mean
        self.global_mean = ratings_df[rating_col].mean()
        
        # Convert to numpy arrays for faster computation
        user_indices = np.array([self.user_mapping[user] for user in ratings_df[user_col]])
        item_indices = np.array([self.item_mapping[item] for item in ratings_df[item_col]])
        ratings = np.array(ratings_df[rating_col])
        
        # SGD training
        for epoch in range(self.n_epochs):
            total_error = 0
            
            # Shuffle the data
            indices = np.random.permutation(len(ratings))
            
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating
                pred = self._predict_single(u, i)
                
                # Compute error
                error = r - pred
                total_error += error ** 2
                
                # Update factors and biases
                self._update_factors(u, i, error)
            
            # Store training history
            avg_error = total_error / len(ratings)
            self.training_history.append(avg_error)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Average Error = {avg_error:.4f}")
        
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        return (self.global_mean + 
                self.user_biases[user_idx] + 
                self.item_biases[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
    
    def _update_factors(self, user_idx, item_idx, error):
        """Update factors and biases using SGD"""
        # Update user factors
        self.user_factors[user_idx] += (self.learning_rate * 
                                       (error * self.item_factors[item_idx] - 
                                        self.regularization * self.user_factors[user_idx]))
        
        # Update item factors
        self.item_factors[item_idx] += (self.learning_rate * 
                                       (error * self.user_factors[user_idx] - 
                                        self.regularization * self.item_factors[item_idx]))
        
        # Update biases
        self.user_biases[user_idx] += self.learning_rate * (error - self.regularization * self.user_biases[user_idx])
        self.item_biases[item_idx] += self.learning_rate * (error - self.regularization * self.item_biases[item_idx])
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self._predict_single(user_idx, item_idx)
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """Generate top-n recommendations for a user"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_factor = self.user_factors[user_idx]
        
        # Predict ratings for all items
        predictions = []
        for item_id, item_idx in self.item_mapping.items():
            if exclude_rated:
                # Skip if user has rated this item (would need to track rated items)
                pass
            
            pred_rating = self._predict_single(user_idx, item_idx)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_items(self, item_id, n_similar=5):
        """Find items similar to the given item based on latent factors"""
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        item_factor = self.item_factors[item_idx]
        
        # Compute similarities with all other items
        similarities = []
        for other_item_id, other_item_idx in self.item_mapping.items():
            if other_item_id != item_id:
                other_factor = self.item_factors[other_item_idx]
                similarity = np.dot(item_factor, other_factor) / (
                    np.linalg.norm(item_factor) * np.linalg.norm(other_factor)
                )
                similarities.append((other_item_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

class SVDppModel:
    """SVD++ Model with implicit feedback"""
    
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.1, 
                 n_epochs=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.implicit_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_items = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the SVD++ model"""
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(ratings_df[user_col].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(ratings_df[item_col].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
        # Initialize factors
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.implicit_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        
        # Compute global mean
        self.global_mean = ratings_df[rating_col].mean()
        
        # Create user-item mapping for implicit feedback
        self.user_items = {}
        for user_id in ratings_df[user_col].unique():
            user_idx = self.user_mapping[user_id]
            user_ratings = ratings_df[ratings_df[user_col] == user_id]
            self.user_items[user_idx] = [self.item_mapping[item] for item in user_ratings[item_col]]
        
        # Convert to numpy arrays
        user_indices = np.array([self.user_mapping[user] for user in ratings_df[user_col]])
        item_indices = np.array([self.item_mapping[item] for item in ratings_df[item_col]])
        ratings = np.array(ratings_df[rating_col])
        
        # SGD training
        for epoch in range(self.n_epochs):
            total_error = 0
            
            # Shuffle the data
            indices = np.random.permutation(len(ratings))
            
            for idx in indices:
                u = user_indices[idx]
                i = item_indices[idx]
                r = ratings[idx]
                
                # Predict rating
                pred = self._predict_single(u, i)
                
                # Compute error
                error = r - pred
                total_error += error ** 2
                
                # Update factors
                self._update_factors(u, i, error)
            
            if epoch % 20 == 0:
                avg_error = total_error / len(ratings)
                print(f"Epoch {epoch}: Average Error = {avg_error:.4f}")
        
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """Predict rating for a single user-item pair"""
        # Basic prediction
        pred = (self.global_mean + 
                self.user_biases[user_idx] + 
                self.item_biases[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        # Add implicit feedback term
        if user_idx in self.user_items:
            user_rated_items = self.user_items[user_idx]
            if len(user_rated_items) > 0:
                implicit_sum = np.sum(self.implicit_factors[user_rated_items], axis=0)
                pred += np.dot(self.user_factors[user_idx], implicit_sum) / np.sqrt(len(user_rated_items))
        
        return pred
    
    def _update_factors(self, user_idx, item_idx, error):
        """Update factors using SGD"""
        # Update user factors
        self.user_factors[user_idx] += (self.learning_rate * 
                                       (error * self.item_factors[item_idx] - 
                                        self.regularization * self.user_factors[user_idx]))
        
        # Update item factors
        self.item_factors[item_idx] += (self.learning_rate * 
                                       (error * self.user_factors[user_idx] - 
                                        self.regularization * self.item_factors[item_idx]))
        
        # Update biases
        self.user_biases[user_idx] += self.learning_rate * (error - self.regularization * self.user_biases[user_idx])
        self.item_biases[item_idx] += self.learning_rate * (error - self.regularization * self.item_biases[item_idx])
        
        # Update implicit factors
        if user_idx in self.user_items:
            user_rated_items = self.user_items[user_idx]
            if len(user_rated_items) > 0:
                factor_update = (error * self.user_factors[user_idx] / np.sqrt(len(user_rated_items)) - 
                               self.regularization * self.implicit_factors[item_idx])
                self.implicit_factors[item_idx] += self.learning_rate * factor_update
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self._predict_single(user_idx, item_idx)

# Generate synthetic data with latent structure
np.random.seed(42)
n_users = 300
n_items = 200
n_ratings = 3000

# Create synthetic ratings with latent factors
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(8, 25)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create latent factor structure
        # Factor 1: Action vs Drama preference
        # Factor 2: Complexity preference
        # Factor 3: Genre preference
        
        user_action_pref = np.random.normal(0, 1)  # User's action preference
        user_complexity_pref = np.random.normal(0, 1)  # User's complexity preference
        user_genre_pref = np.random.normal(0, 1)  # User's genre preference
        
        item_action_level = np.random.normal(0, 1)  # Item's action level
        item_complexity = np.random.normal(0, 1)  # Item's complexity
        item_genre = np.random.normal(0, 1)  # Item's genre
        
        # Compute rating based on latent factors
        latent_score = (user_action_pref * item_action_level + 
                       user_complexity_pref * item_complexity + 
                       user_genre_pref * item_genre)
        
        # Add noise and convert to 1-5 scale
        rating = max(1, min(5, 3 + latent_score + np.random.normal(0, 0.5)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with Latent Structure:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.3f}")

# Split data for evaluation
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Train different latent factor models
print("\n=== Training Latent Factor Models ===")

# Basic Latent Factor Model
lf_model = LatentFactorModel(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=100)
lf_model.fit(train_df)

# SVD++ Model
svdpp_model = SVDppModel(n_factors=10, learning_rate=0.01, regularization=0.1, n_epochs=100)
svdpp_model.fit(train_df)

# NMF Model (using sklearn)
train_matrix = train_df.pivot_table(
    index='user_id', columns='item_id', values='rating', fill_value=0
)
nmf_model = NMF(n_components=10, random_state=42, max_iter=100)
nmf_user_factors = nmf_model.fit_transform(train_matrix)
nmf_item_factors = nmf_model.components_.T

# Evaluate models
def evaluate_model(model, test_df, model_type='custom'):
    """Evaluate model on test set"""
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        if model_type == 'nmf':
            # For NMF, need to handle missing users/items
            if user_id in train_matrix.index and item_id in train_matrix.columns:
                user_idx = train_matrix.index.get_loc(user_id)
                item_idx = train_matrix.columns.get_loc(item_id)
                pred_rating = np.dot(nmf_user_factors[user_idx], nmf_item_factors[item_idx])
            else:
                pred_rating = np.nan
        else:
            pred_rating = model.predict(user_id, item_id)
        
        if not np.isnan(pred_rating):
            predictions.append(pred_rating)
            actuals.append(actual_rating)
    
    if len(predictions) == 0:
        return {'mae': np.inf, 'rmse': np.inf, 'coverage': 0}
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    coverage = len(predictions) / len(test_df)
    
    return {'mae': mae, 'rmse': rmse, 'coverage': coverage}

# Evaluate all models
models = {
    'Latent Factor': lf_model,
    'SVD++': svdpp_model,
    'NMF': None
}

results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    model_type = 'nmf' if name == 'NMF' else 'custom'
    results[name] = evaluate_model(model, test_df, model_type)

# Display results
print("\n=== Evaluation Results ===")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print()

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Training history
plt.subplot(3, 4, 1)
plt.plot(lf_model.training_history, label='Latent Factor')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Average Error')
plt.legend()

# Plot 2: User factors visualization (first 2 dimensions)
plt.subplot(3, 4, 2)
user_factors_2d = lf_model.user_factors[:, :2]
plt.scatter(user_factors_2d[:, 0], user_factors_2d[:, 1], alpha=0.6)
plt.title('User Factors (First 2 Dimensions)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# Plot 3: Item factors visualization (first 2 dimensions)
plt.subplot(3, 4, 3)
item_factors_2d = lf_model.item_factors[:, :2]
plt.scatter(item_factors_2d[:, 0], item_factors_2d[:, 1], alpha=0.6)
plt.title('Item Factors (First 2 Dimensions)')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# Plot 4: Factor importance
plt.subplot(3, 4, 4)
factor_importance = np.var(lf_model.user_factors, axis=0)
plt.bar(range(len(factor_importance)), factor_importance)
plt.title('Factor Importance (Variance)')
plt.xlabel('Factor')
plt.ylabel('Variance')

# Plot 5: MAE comparison
plt.subplot(3, 4, 5)
mae_values = [results[name]['mae'] for name in results.keys()]
plt.bar(results.keys(), mae_values, color=['blue', 'red', 'green'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)

# Plot 6: RMSE comparison
plt.subplot(3, 4, 6)
rmse_values = [results[name]['rmse'] for name in results.keys()]
plt.bar(results.keys(), rmse_values, color=['blue', 'red', 'green'])
plt.title('RMSE Comparison')
plt.ylabel('Root Mean Square Error')
plt.xticks(rotation=45)

# Plot 7: Coverage comparison
plt.subplot(3, 4, 7)
coverage_values = [results[name]['coverage'] for name in results.keys()]
plt.bar(results.keys(), coverage_values, color=['blue', 'red', 'green'])
plt.title('Coverage Comparison')
plt.ylabel('Coverage')
plt.xticks(rotation=45)

# Plot 8: Prediction vs Actual (Latent Factor)
plt.subplot(3, 4, 8)
lf_predictions = []
lf_actuals = []
for _, row in test_df.head(100).iterrows():
    pred = lf_model.predict(row['user_id'], row['item_id'])
    if not np.isnan(pred):
        lf_predictions.append(pred)
        lf_actuals.append(row['rating'])

plt.scatter(lf_actuals, lf_predictions, alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
plt.title('Latent Factor: Predicted vs Actual')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')

# Plot 9: User bias distribution
plt.subplot(3, 4, 9)
plt.hist(lf_model.user_biases, bins=30, alpha=0.7, edgecolor='black')
plt.title('User Bias Distribution')
plt.xlabel('Bias')
plt.ylabel('Frequency')

# Plot 10: Item bias distribution
plt.subplot(3, 4, 10)
plt.hist(lf_model.item_biases, bins=30, alpha=0.7, edgecolor='black')
plt.title('Item Bias Distribution')
plt.xlabel('Bias')
plt.ylabel('Frequency')

# Plot 11: Factor correlation matrix
plt.subplot(3, 4, 11)
factor_corr = np.corrcoef(lf_model.user_factors.T)
sns.heatmap(factor_corr, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
plt.title('Factor Correlation Matrix')

# Plot 12: Model comparison summary
plt.subplot(3, 4, 12)
comparison_metrics = ['MAE', 'RMSE', 'Coverage']
comparison_values = [
    [results[name]['mae'] for name in results.keys()],
    [results[name]['rmse'] for name in results.keys()],
    [results[name]['coverage'] for name in results.keys()]
]

x = np.arange(len(comparison_metrics))
width = 0.25

for i, (name, values) in enumerate(zip(results.keys(), np.array(comparison_values).T)):
    plt.bar(x + i*width, values, width, label=name)

plt.title('Model Comparison')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks(x + width, comparison_metrics)
plt.legend()

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Analysis ===")

# Compare factor interpretations
print("Factor Analysis:")
for i in range(min(5, lf_model.n_factors)):
    user_factor_std = np.std(lf_model.user_factors[:, i])
    item_factor_std = np.std(lf_model.item_factors[:, i])
    print(f"Factor {i+1}: User std = {user_factor_std:.3f}, Item std = {item_factor_std:.3f}")

# Compare prediction patterns
test_sample = test_df.head(50)
lf_preds = []
svdpp_preds = []
actuals = []

for _, row in test_sample.iterrows():
    lf_pred = lf_model.predict(row['user_id'], row['item_id'])
    svdpp_pred = svdpp_model.predict(row['user_id'], row['item_id'])
    
    if not (np.isnan(lf_pred) or np.isnan(svdpp_pred)):
        lf_preds.append(lf_pred)
        svdpp_preds.append(svdpp_pred)
        actuals.append(row['rating'])

print(f"\nPrediction Statistics:")
print(f"Latent Factor:")
print(f"  Mean: {np.mean(lf_preds):.3f}")
print(f"  Std: {np.std(lf_preds):.3f}")
print(f"  Range: [{np.min(lf_preds):.3f}, {np.max(lf_preds):.3f}]")

print(f"\nSVD++:")
print(f"  Mean: {np.mean(svdpp_preds):.3f}")
print(f"  Std: {np.std(svdpp_preds):.3f}")
print(f"  Range: [{np.min(svdpp_preds):.3f}, {np.max(svdpp_preds):.3f}]")

# Test recommendations
test_user = 0
print(f"\nTop 5 recommendations for User {test_user}:")
recommendations = lf_model.recommend(test_user, n_recommendations=5)
for i, (item_id, pred_rating) in enumerate(recommendations):
    print(f"  {i+1}. Item {item_id}: Predicted rating = {pred_rating:.3f}")

# Test similar items
test_item = 0
print(f"\nTop 5 similar items to Item {test_item}:")
similar_items = lf_model.get_similar_items(test_item, n_similar=5)
for i, (item_id, similarity) in enumerate(similar_items):
    print(f"  {i+1}. Item {item_id}: Similarity = {similarity:.3f}")
```

### R Implementation

```r
# Latent Factor Models in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(NMF)

# Generate synthetic data with latent structure
set.seed(42)
n_users <- 300
n_items <- 200
n_ratings <- 3000

# Create synthetic ratings with latent factors
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(8:25, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create latent factor structure
    user_action_pref <- rnorm(1, 0, 1)
    user_complexity_pref <- rnorm(1, 0, 1)
    user_genre_pref <- rnorm(1, 0, 1)
    
    item_action_level <- rnorm(1, 0, 1)
    item_complexity <- rnorm(1, 0, 1)
    item_genre <- rnorm(1, 0, 1)
    
    # Compute rating based on latent factors
    latent_score <- (user_action_pref * item_action_level + 
                     user_complexity_pref * item_complexity + 
                     user_genre_pref * item_genre)
    
    # Add noise and convert to 1-5 scale
    rating <- max(1, min(5, 3 + latent_score + rnorm(1, 0, 0.5)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Create rating matrix
rating_matrix <- ratings_df %>%
  spread(item_id, rating, fill = NA) %>%
  select(-user_id) %>%
  as.matrix()

# Convert to realRatingMatrix
rating_matrix_real <- as(rating_matrix, "realRatingMatrix")

# Split data for evaluation
set.seed(42)
train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
train_df <- ratings_df[train_indices, ]
test_df <- ratings_df[-train_indices, ]

# Create training matrix
train_matrix <- train_df %>%
  spread(item_id, rating, fill = 0) %>%
  select(-user_id) %>%
  as.matrix()

# Test different latent factor methods
methods <- c("SVD", "NMF")

results <- list()

for (method in methods) {
  cat("Testing", method, "\n")
  
  if (method == "SVD") {
    # SVD-based recommendation
    model <- Recommender(train_matrix_real, method = "SVD")
  } else if (method == "NMF") {
    # Non-negative Matrix Factorization
    nmf_result <- nmf(train_matrix, 10, method = "brunet", nrun = 1)
    # For simplicity, we'll use a basic approach
    model <- list(type = "NMF", factors = nmf_result)
  }
  
  # Generate predictions
  predictions <- predict(model, train_matrix_real[1:min(10, nrow(train_matrix_real))], n = 5)
  
  # Store results
  results[[method]] <- list(
    model = model,
    predictions = predictions
  )
}

# Visualization
# Rating distribution
p1 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# User-item matrix heatmap (sample)
sample_matrix <- rating_matrix[1:20, 1:20]
sample_df <- expand.grid(
  user_id = 1:20,
  item_id = 1:20
)
sample_df$rating <- as.vector(sample_matrix)

p2 <- ggplot(sample_df, aes(x = item_id, y = user_id, fill = rating)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Rating Matrix (Sample)",
       x = "Item ID", y = "User ID") +
  theme_minimal()

# Method comparison
method_names <- names(results)
comparison_df <- data.frame(
  method = method_names,
  mae = c(0.5, 0.6),  # Placeholder values
  rmse = c(0.7, 0.8)  # Placeholder values
)

p3 <- ggplot(comparison_df, aes(x = method, y = mae)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "MAE Comparison",
       x = "Method", y = "Mean Absolute Error") +
  theme_minimal()

p4 <- ggplot(comparison_df, aes(x = method, y = rmse)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(title = "RMSE Comparison",
       x = "Method", y = "Root Mean Square Error") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, p3, p4, ncol = 2)
```

## 13.5.5. Advanced Topics

### Deep Learning Approaches

#### Neural Collaborative Filtering (NCF)
```math
\hat{r}_{ui} = f(\mathbf{u}_u, \mathbf{v}_i) = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [\mathbf{u}_u; \mathbf{v}_i] + \mathbf{b}_1) + \mathbf{b}_2)
```

#### Autoencoder-based CF
```math
\text{Encoder}: h = \sigma(W_e \cdot r_u + b_e)
```

```math
\text{Decoder}: \hat{r}_u = \sigma(W_d \cdot h + b_d)
```

### Temporal Dynamics

#### Time-aware Matrix Factorization
```math
\hat{r}_{ui}(t) = \mu + b_u(t) + b_i(t) + \mathbf{u}_u^T \mathbf{v}_i + \mathbf{u}_u^T \mathbf{v}_i(t)
```

where $`b_u(t)`$ and $`b_i(t)`$ are time-dependent biases.

### Context-aware Models

#### Factorization Machines
```math
\hat{r}_{ui} = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j
```

## 13.5.6. Evaluation and Validation

### Cross-validation Strategies

#### Leave-One-Out Cross-validation
```math
\text{CV Score} = \frac{1}{|\mathcal{R}|} \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \hat{r}_{ui}^{(-u,i)})^2
```

where $`\hat{r}_{ui}^{(-u,i)}`$ is the prediction without the $(u,i)$ pair.

#### Time-based Split
- Train on historical data
- Test on recent data
- More realistic for production systems

### Hyperparameter Tuning

#### Grid Search for Latent Factors
```python
best_score = float('inf')
best_k = None

for k in [5, 10, 15, 20, 25]:
    model = LatentFactorModel(n_factors=k)
    score = cross_validate(model, data)
    if score < best_score:
        best_score = score
        best_k = k
```

## 13.5.7. Production Considerations

### Scalability

#### Stochastic Gradient Descent
- Process one rating at a time
- Memory efficient
- Can be parallelized

#### Alternating Least Squares (ALS)
```math
\mathbf{u}_u = (\sum_{i \in \mathcal{I}_u} \mathbf{v}_i \mathbf{v}_i^T + \lambda I)^{-1} \sum_{i \in \mathcal{I}_u} r_{ui} \mathbf{v}_i
```

### Online Learning

#### Incremental Updates
```math
\mathbf{u}_u^{(t+1)} = \mathbf{u}_u^{(t)} + \gamma \cdot \nabla_{\mathbf{u}_u} \mathcal{L}
```

### Cold Start Handling

#### Content-based Initialization
```math
\mathbf{u}_u = \frac{1}{|\mathcal{I}_u|} \sum_{i \in \mathcal{I}_u} \mathbf{v}_i + \text{content\_features}_u
```

## 13.5.8. Practical Considerations and Best Practices

### Data Preprocessing

#### Handling Missing Values

**Imputation Strategies:**
1. **Mean imputation**: Replace missing values with user/item means
2. **Median imputation**: More robust to outliers
3. **Zero imputation**: Simple but may introduce bias
4. **Matrix completion**: Use low-rank approximation to fill missing values

**Example:**
```python
# User mean imputation
user_means = ratings_df.groupby('user_id')['rating'].mean()
ratings_matrix = ratings_df.pivot_table(
    index='user_id', columns='item_id', values='rating'
).fillna(user_means)
```

#### Normalization and Scaling

**Rating normalization:**
```math
r_{ui}^{norm} = \frac{r_{ui} - \mu_u}{\sigma_u}
```

where $`\mu_u`$ and $`\sigma_u`$ are the mean and standard deviation of user $`u`$'s ratings.

**Benefits:**
- **Reduces user bias**: Accounts for different rating scales
- **Improves convergence**: Normalized data often converges faster
- **Better generalization**: More robust to outliers

#### Outlier Detection and Handling

**Methods:**
1. **Z-score**: Flag ratings with $`|z| > 3`$
2. **IQR method**: Flag ratings outside $`Q1 - 1.5 \times IQR`$ to $`Q3 + 1.5 \times IQR`$
3. **Robust statistics**: Use median and MAD instead of mean and std

### Hyperparameter Tuning

#### Systematic Approach

**1. Define Search Space:**
```python
param_grid = {
    'n_factors': [5, 10, 15, 20, 25],
    'learning_rate': [0.001, 0.01, 0.1],
    'regularization': [0.01, 0.1, 1.0],
    'n_epochs': [50, 100, 200]
}
```

**2. Cross-validation Strategy:**
- **Time-based split**: Train on historical data, validate on recent data
- **User-based split**: Some users in train, others in validation
- **Rating-based split**: Random split of ratings (less realistic)

**3. Evaluation Metrics:**
```python
def evaluate_model(model, test_data):
    predictions = []
    actuals = []
    
    for user_id, item_id, rating in test_data:
        pred = model.predict(user_id, item_id)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(rating)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return {'mae': mae, 'rmse': rmse}
```

#### Bayesian Optimization

For more efficient hyperparameter tuning:

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def objective(params):
    n_factors, learning_rate, regularization = params
    model = LatentFactorModel(
        n_factors=int(n_factors),
        learning_rate=learning_rate,
        regularization=regularization
    )
    model.fit(train_data)
    return evaluate_model(model, val_data)['rmse']

# Define search space
space = [
    Integer(5, 30, name='n_factors'),
    Real(1e-4, 1e-1, prior='log-uniform', name='learning_rate'),
    Real(1e-3, 1e0, prior='log-uniform', name='regularization')
]

# Optimize
result = gp_minimize(objective, space, n_calls=50)
```

### Model Interpretability

#### Factor Analysis

**Visualizing factors:**
```python
def analyze_factors(model, item_names=None):
    """Analyze and visualize learned factors"""
    
    # Factor importance
    factor_importance = np.var(model.item_factors, axis=0)
    
    # Factor correlation
    factor_corr = np.corrcoef(model.item_factors.T)
    
    # Top items per factor
    top_items_per_factor = []
    for f in range(model.n_factors):
        factor_scores = model.item_factors[:, f]
        top_indices = np.argsort(factor_scores)[-10:]  # Top 10
        top_items_per_factor.append(top_indices)
    
    return {
        'importance': factor_importance,
        'correlation': factor_corr,
        'top_items': top_items_per_factor
    }
```

#### Understanding Predictions

**Decomposing predictions:**
```python
def explain_prediction(model, user_id, item_id):
    """Explain why a user might like/dislike an item"""
    
    user_idx = model.user_mapping[user_id]
    item_idx = model.item_mapping[item_id]
    
    user_factor = model.user_factors[user_idx]
    item_factor = model.item_factors[item_idx]
    
    # Factor-wise contributions
    contributions = user_factor * item_factor
    
    # Overall prediction
    prediction = model.predict(user_id, item_id)
    
    return {
        'prediction': prediction,
        'contributions': contributions,
        'user_factor': user_factor,
        'item_factor': item_factor
    }
```

### Production Deployment

#### Model Serving

**API Design:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data['user_id']
    item_id = data['item_id']
    
    prediction = model.predict(user_id, item_id)
    
    return jsonify({
        'user_id': user_id,
        'item_id': item_id,
        'prediction': prediction,
        'confidence': calculate_confidence(prediction)
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['user_id']
    n_recommendations = data.get('n_recommendations', 10)
    
    recommendations = model.recommend(user_id, n_recommendations)
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations
    })
```

#### Model Updates

**Incremental Learning:**
```python
def update_model_incrementally(model, new_ratings):
    """Update model with new ratings without retraining"""
    
    for user_id, item_id, rating in new_ratings:
        if user_id in model.user_mapping and item_id in model.item_mapping:
            user_idx = model.user_mapping[user_id]
            item_idx = model.item_mapping[item_id]
            
            # Single SGD update
            pred = model._predict_single(user_idx, item_idx)
            error = rating - pred
            
            model._update_factors(user_idx, item_idx, error)
    
    return model
```

#### Monitoring and A/B Testing

**Performance Monitoring:**
```python
def monitor_model_performance(model, recent_ratings):
    """Monitor model performance on recent data"""
    
    predictions = []
    actuals = []
    
    for user_id, item_id, rating in recent_ratings:
        pred = model.predict(user_id, item_id)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(rating)
    
    metrics = {
        'mae': mean_absolute_error(actuals, predictions),
        'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
        'coverage': len(predictions) / len(recent_ratings)
    }
    
    return metrics
```

### Common Pitfalls and Solutions

#### Overfitting

**Symptoms:**
- Training error much lower than validation error
- Poor performance on new users/items
- Factors become very large in magnitude

**Solutions:**
1. **Increase regularization**: Higher $`\lambda`$ values
2. **Reduce factors**: Fewer latent factors
3. **Early stopping**: Stop training when validation error increases
4. **Cross-validation**: Use proper validation strategy

#### Cold Start Problem

**For New Users:**
1. **Content-based initialization**: Use user features to initialize factors
2. **Popular item recommendations**: Recommend popular items initially
3. **Hybrid approaches**: Combine with content-based methods

**For New Items:**
1. **Item similarity**: Use similar items' factors
2. **Content features**: Initialize with item features
3. **Temporal decay**: Give more weight to recent interactions

#### Data Sparsity

**Handling sparse data:**
1. **Implicit feedback**: Use click/view data in addition to ratings
2. **Side information**: Incorporate user/item features
3. **Regularization**: Stronger regularization for sparse data
4. **Sampling**: Use negative sampling for implicit feedback

### Performance Optimization

#### Computational Efficiency

**Vectorized operations:**
```python
def vectorized_predict(model, user_ids, item_ids):
    """Vectorized prediction for multiple user-item pairs"""
    
    user_indices = [model.user_mapping.get(uid, -1) for uid in user_ids]
    item_indices = [model.item_mapping.get(iid, -1) for iid in item_ids]
    
    # Filter valid pairs
    valid_mask = [(u >= 0) and (i >= 0) for u, i in zip(user_indices, item_indices)]
    
    if not any(valid_mask):
        return np.full(len(user_ids), np.nan)
    
    valid_user_factors = model.user_factors[[u for u, valid in zip(user_indices, valid_mask) if valid]]
    valid_item_factors = model.item_factors[[i for i, valid in zip(item_indices, valid_mask) if valid]]
    
    # Vectorized dot product
    predictions = np.sum(valid_user_factors * valid_item_factors, axis=1)
    
    # Fill results
    result = np.full(len(user_ids), np.nan)
    result[valid_mask] = predictions
    
    return result
```

#### Memory Optimization

**Sparse storage:**
```python
from scipy.sparse import csr_matrix

def create_sparse_matrix(ratings_df):
    """Create sparse rating matrix"""
    
    user_ids = ratings_df['user_id'].unique()
    item_ids = ratings_df['item_id'].unique()
    
    user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
    item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
    
    rows = [user_mapping[uid] for uid in ratings_df['user_id']]
    cols = [item_mapping[iid] for iid in ratings_df['item_id']]
    data = ratings_df['rating'].values
    
    return csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
```

## 13.5.9. Summary

### Key Advantages

1. **Captures Complex Patterns**: Discovers hidden relationships that explicit similarity measures miss
2. **Scalable**: Works efficiently with large datasets through dimensionality reduction
3. **Flexible**: Can incorporate various side information (user features, item features, temporal dynamics)
4. **Interpretable**: Factors can have meaningful interpretations in many domains
5. **Robust**: Handles noise and missing data better than memory-based methods

### Key Limitations

1. **Cold Start**: Problems with new users/items that have limited interaction data
2. **Black Box**: Factors may not always be interpretable or meaningful
3. **Overfitting**: Requires careful regularization and hyperparameter tuning
4. **Computational Cost**: Training can be expensive for very large datasets
5. **Non-uniqueness**: Factor representations are not unique, making interpretation challenging

### Best Practices

1. **Choose Appropriate Factors**: Balance model complexity with performance using cross-validation
2. **Regularize Properly**: Use appropriate regularization to prevent overfitting
3. **Handle Missing Data**: Use appropriate imputation strategies or implicit feedback
4. **Validate Thoroughly**: Use multiple evaluation metrics and proper validation strategies
5. **Monitor Performance**: Track model drift and performance degradation over time
6. **Preprocess Data**: Normalize ratings and handle outliers appropriately
7. **Tune Hyperparameters**: Use systematic approaches like grid search or Bayesian optimization
8. **Consider Production Needs**: Design for scalability, interpretability, and maintainability

### When to Use Latent Factor Models

**Use when:**
- You have sufficient user-item interaction data
- You want to discover implicit patterns in the data
- You need scalable solutions for large datasets
- You want to incorporate side information
- You need interpretable factor representations

**Consider alternatives when:**
- You have very sparse data with limited interactions
- You need highly interpretable recommendations
- You have rich content features but limited interaction data
- You need real-time personalization for new users

Latent factor models represent a powerful and flexible approach to recommendation systems, capable of discovering complex patterns in user-item interactions. When properly implemented and tuned, they can provide excellent recommendation quality while maintaining reasonable computational efficiency. The key to success lies in understanding the mathematical foundations, choosing appropriate hyperparameters, and implementing robust evaluation and monitoring strategies.

---

**Next**: [Challenges in Recommender Systems](06_challenges.md) - Explore the key challenges and limitations faced by modern recommendation systems.
