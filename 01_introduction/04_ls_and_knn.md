# 1.2.1. Introduction to LS and kNN

In this section, we examine two fundamental supervised learning approaches: k-Nearest Neighbor (kNN) and linear regression. These algorithms represent different paradigms in machine learning - kNN is a non-parametric, instance-based method, while linear regression is a parametric, model-based approach. Understanding their strengths, weaknesses, and mathematical foundations is crucial for building intuition about the bias-variance tradeoff.

**Think of kNN and linear regression as two different approaches to solving problems, like two different ways to find your way in a new city.** Linear regression is like using a map with clear directions - it gives you a simple rule to follow. kNN is like asking locals for directions - it uses specific examples to guide you. Both can get you where you need to go, but they work very differently and are better suited for different situations.

## k-Nearest Neighbor (kNN)

The k-Nearest Neighbor algorithm is one of the simplest yet most powerful non-parametric learning methods. It operates on a fundamental principle: similar inputs should have similar outputs. This "local averaging" approach makes no assumptions about the underlying data distribution.

**Intuitive Understanding**: kNN is like asking your neighbors for advice. If you want to know what restaurant to try, you ask the people who live near you what they like. The more neighbors you ask (larger k), the more reliable your recommendation, but also the less specific to your exact situation.

### Mathematical Formulation

For a test point $`x`$, kNN identifies the $`k`$ training samples closest to $`x`$ and uses their target values to make a prediction.

**Distance Metric**: Typically uses Euclidean distance in $`\mathbb{R}^p`$:
$$ d(x_i, x_j) = \sqrt{\sum_{l=1}^p (x_{il} - x_{jl})^2} $$

**Understanding Distance Metrics:**

1. **Euclidean Distance**: Most common for continuous features
   - Measures "straight-line" distance between points - like measuring the direct distance between two houses
   - Sensitive to scale of features - like caring equally about differences in square footage and number of bedrooms
   - Assumes all features are equally important - like treating location and price as equally important when choosing a house

2. **Manhattan Distance (L1 norm)**:
$$ d(x_i, x_j) = \sum_{l=1}^p |x_{il} - x_{jl}| $$
   - Measures "city block" distance - like walking along streets in a grid
   - Less sensitive to outliers than Euclidean - like being less affected by one very unusual house
   - Useful when features have different scales - like when square footage and number of bedrooms are on very different scales

3. **Weighted Euclidean Distance**:
$$ d(x_i, x_j) = \sqrt{\sum_{l=1}^p w_l (x_{il} - x_{jl})^2} $$
   - Allows different importance for different features - like caring more about location than square footage
   - Weights $`w_l`$ can be learned or set based on domain knowledge - like knowing that location is twice as important as size

**Intuition**: Distance metrics are like different ways to measure how similar two things are. Euclidean distance is like measuring the direct distance between two points, while Manhattan distance is like measuring how far you'd have to walk along city streets.

**Regression**: Output the average of the $`Y`$ values of the $`k`$ nearest neighbors:
$$ \hat{f}(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i $$
where $`\mathcal{N}_k(x)`$ is the set of indices of the $`k`$ nearest neighbors of $`x`$.

**Classification**: Return the majority vote or probability based on class frequency:
$$ \hat{f}(x) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}[y_i = c] $$

**Probability Estimation**:
$$ P(Y = c | X = x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{I}[y_i = c] $$

**Understanding the kNN Algorithm:**

1. **Neighborhood Definition**: $`\mathcal{N}_k(x)`$ contains indices of $`k`$ closest training points - like finding the k closest houses to yours
2. **Local Averaging**: Prediction is average of neighbors' values - like averaging the prices of nearby houses
3. **No Training**: Algorithm simply stores training data - like just remembering all the houses you've seen
4. **Lazy Learning**: Computation happens only at prediction time - like only looking up directions when you need them

**Intuition**: kNN is like a "memory-based" approach. Instead of learning a rule, it just remembers all the examples and uses the most similar ones to make predictions.

### Example Walkthrough

Consider $`k=5`$ in a binary classification problem. If among the five nearest neighbors:
- 3 have $`Y=1`$
- 2 have $`Y=0`$

Then:
- **Majority vote**: Predict $`Y=1`$ - like 3 out of 5 neighbors recommending the same restaurant
- **Probability estimates**: $`P(Y=1) = 3/5 = 0.6`$, $`P(Y=0) = 2/5 = 0.4`$ - like 60% of neighbors liking the restaurant

**Detailed Example: House Price Prediction**

Suppose we have training data with house features (square footage, bedrooms) and prices:
- House 1: (1500 sq ft, 3 beds) → $300,000
- House 2: (1600 sq ft, 3 beds) → $320,000
- House 3: (1400 sq ft, 2 beds) → $280,000
- House 4: (1700 sq ft, 4 beds) → $350,000
- House 5: (1550 sq ft, 3 beds) → $310,000

For a new house: (1520 sq ft, 3 beds)

**Step 1: Calculate distances**
- Distance to House 1: $`\sqrt{(1520-1500)^2 + (3-3)^2} = 20`$ - like House 1 is very similar
- Distance to House 2: $`\sqrt{(1520-1600)^2 + (3-3)^2} = 80`$ - like House 2 is somewhat similar
- Distance to House 3: $`\sqrt{(1520-1400)^2 + (3-2)^2} = 120.04`$ - like House 3 is less similar
- Distance to House 4: $`\sqrt{(1520-1700)^2 + (3-4)^2} = 180.01`$ - like House 4 is quite different
- Distance to House 5: $`\sqrt{(1520-1550)^2 + (3-3)^2} = 30`$ - like House 5 is very similar

**Step 2: Find k=3 nearest neighbors**
- House 1 (distance 20) - most similar
- House 5 (distance 30) - second most similar
- House 2 (distance 80) - third most similar

**Step 3: Predict price**
- $`\hat{y} = \frac{300,000 + 310,000 + 320,000}{3} = 310,000`$ - like averaging the prices of the three most similar houses

**Intuition**: This is like saying "houses similar to mine sold for around $310,000, so mine is probably worth about that much."

### Algorithm Properties

**No Training Phase**: kNN is a "lazy learner" - it simply stores the training data and performs computation only at prediction time.

**Intuition**: kNN is like having a photographic memory - you don't process or summarize the information, you just remember everything and look it up when needed.

**Local Approximation**: kNN approximates the true function $`f^*(x)`$ locally around each test point:
$$ \hat{f}(x) \approx \mathbb{E}[Y | X \in \mathcal{B}_k(x)] $$
where $`\mathcal{B}_k(x)`$ is the neighborhood defined by the $`k`$ nearest neighbors.

**Understanding Local Approximation:**

1. **Neighborhood**: $`\mathcal{B}_k(x)`$ is the region containing the $`k`$ nearest neighbors - like the area around your house
2. **Local Expectation**: We estimate the expected value of $`Y`$ in this neighborhood - like the average house price in your neighborhood
3. **Assumption**: Points in the neighborhood have similar $`Y`$ values - like houses in the same area having similar prices

**Intuition**: kNN assumes that nearby points are similar. This is like assuming that houses in the same neighborhood have similar prices, or that people with similar characteristics have similar preferences.

**Mathematical Properties:**

1. **Consistency**: Under certain conditions, kNN converges to the true function - like asking more and more neighbors gives you better estimates
2. **No Parametric Assumptions**: Works with any data distribution - like not assuming house prices follow any particular pattern
3. **Adaptive Bandwidth**: Neighborhood size adapts to local density - like asking more neighbors in areas where houses are spread out

## Parameters and Tuning

### Neighborhood Size ($`k`$)

The choice of $`k`$ fundamentally affects the bias-variance tradeoff:

**$`k=1`$ (1NN)**:
- Uses only the nearest training sample - like asking only your closest neighbor
- Training error is zero (perfect interpolation) - like always getting the exact same answer as your neighbor
- Highest variance, lowest bias - like being very sensitive to your neighbor's specific situation
- Complexity approximately $`n`$ parameters - like having as many parameters as training examples

**$`k=n`$ (global average)**:
- Uses all training samples equally - like asking everyone in the city
- Prediction is constant for all $`x`$ (like fitting only an intercept) - like everyone giving the same average answer
- Lowest variance, highest bias - like getting a very stable but generic answer
- Complexity approximately 1 parameter - like having just one parameter (the average)

**General Case**: Complexity is approximately $`n/k`$ parameters.

**Optimal $`k`$**: Typically chosen via cross-validation to balance bias and variance.

**Intuition**: Choosing k is like deciding how many neighbors to ask for advice. Too few and you might get unreliable advice. Too many and you get generic advice that doesn't apply to your specific situation.

**Mathematical Analysis of k Selection:**

The optimal $`k`$ depends on:
1. **Sample size $`n`$**: Larger $`n`$ allows larger $`k`$ - like having more people to ask in a bigger city
2. **Data dimensionality $`p`$**: Higher $`p`$ requires smaller $`k`$ (curse of dimensionality) - like needing more specific advice when considering many factors
3. **Noise level**: More noise requires larger $`k`$ for smoothing - like asking more people when opinions are very varied
4. **Local structure**: Complex local patterns require smaller $`k`$ - like needing specific advice in a complex situation

**Example: k Selection via Cross-Validation**

See the complete implementation in [`code/knn_k_selection.py`](code/knn_k_selection.py) which demonstrates how to select the optimal k value for kNN using cross-validation.

### Distance Metrics

**Euclidean Distance** (default for continuous features):
$$ d(x_i, x_j) = \sqrt{\sum_{l=1}^p (x_{il} - x_{jl})^2} $$

**Intuition**: Euclidean distance is like measuring the straight-line distance between two points. It's the most natural way to measure distance in most situations.

**Manhattan Distance** (L1 norm):
$$ d(x_i, x_j) = \sum_{l=1}^p |x_{il} - x_{jl}| $$

**Intuition**: Manhattan distance is like measuring how far you'd have to walk along city streets to get from one point to another. It's less sensitive to outliers because it doesn't "cut corners."

**Weighted Distances**:
$$ d(x_i, x_j) = \sqrt{\sum_{l=1}^p w_l (x_{il} - x_{jl})^2} $$

**Intuition**: Weighted distances are like caring more about some differences than others. For example, you might care more about location than square footage when choosing a house.

**Domain-Specific Metrics**: For images (pixel similarity), text (cosine similarity), or user preferences.

**Cosine Similarity** (for text data):
$$ \text{cosine}(x_i, x_j) = \frac{x_i^T x_j}{\|x_i\| \|x_j\|} $$

**Intuition**: Cosine similarity measures the angle between two vectors, ignoring their magnitude. It's like comparing the direction of two arrows rather than their length.

**Mahalanobis Distance** (accounts for feature correlations):
$$ d(x_i, x_j) = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)} $$
where $`\Sigma`$ is the covariance matrix.

**Intuition**: Mahalanobis distance accounts for how features are related to each other. It's like knowing that when square footage increases, price usually increases too, so you adjust your distance measure accordingly.

## Linear Regression

Linear regression is a parametric method that assumes a linear relationship between features and target. It's computationally efficient and provides interpretable results.

**Intuitive Understanding**: Linear regression is like finding a simple rule that explains how one thing relates to another. It's like saying "for every extra bedroom, a house costs $50,000 more" - a simple, interpretable relationship.

### Mathematical Formulation

**Model**: Assume a linear relationship between $`X`$ and $`Y`$:
$$ Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p + \epsilon = X^T \beta + \epsilon $$

where:
- $`\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T`$ is the parameter vector
- $`X = (1, X_1, \ldots, X_p)^T`$ includes the intercept term
- $`\epsilon \sim N(0, \sigma^2)`$ is the error term

**Understanding the Linear Model:**

1. **Additive Structure**: Each feature contributes linearly to the prediction - like each bedroom adding a fixed amount to the price
2. **Intercept**: $`\beta_0`$ represents the baseline prediction when all features are zero - like the base price of a house with no bedrooms
3. **Slope Coefficients**: $`\beta_j`$ represents the change in $`Y`$ per unit change in $`X_j`$ - like how much price increases for each additional square foot
4. **Error Term**: $`\epsilon`$ captures unmodeled variation and measurement error - like random factors that affect house prices

**Intuition**: Linear regression assumes that the relationship between features and target is like a straight line. Each feature has a fixed effect on the outcome, and these effects add up.

**Estimation**: Minimize the sum of squared residuals:
$$ \hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - x_i^T \beta)^2 $$

**Understanding Least Squares:**

The objective function is:
$$ L(\beta) = \sum_{i=1}^n (y_i - x_i^T \beta)^2 = \|y - X\beta\|^2 $$

Taking the gradient and setting to zero:
$$ \nabla L(\beta) = -2X^T(y - X\beta) = 0 $$

Solving for $`\beta`$:
$$ X^T X \beta = X^T y $$

**Intuition**: Least squares finds the line that minimizes the total squared distance from all data points to the line. It's like finding the best-fitting line through a scatter plot.

**Closed-Form Solution**: For $`p`$-dimensional $`X`$, we estimate $`p+1`$ parameters:
$$ \hat{\beta} = (X^T X)^{-1} X^T y $$

**Prediction**: $`\hat{y} = x^T \hat{\beta}`$

**Understanding the Solution:**

1. **Normal Equations**: $`X^T X \beta = X^T y`$ are the normal equations - like the mathematical conditions for the best-fitting line
2. **Matrix Inversion**: $`(X^T X)^{-1}`$ exists if $`X`$ has full column rank - like being able to solve the system of equations
3. **Projection**: $`\hat{y}`$ is the projection of $`y`$ onto the column space of $`X`$ - like finding the closest point on the line to each data point

**Intuition**: The solution finds the best linear combination of features that predicts the target. It's like finding the optimal weights for each feature.

**Example: Simple Linear Regression**

Consider predicting house price ($`Y`$) from square footage ($`X`$):

Training data:
- House 1: 1500 sq ft → $300,000
- House 2: 2000 sq ft → $400,000
- House 3: 2500 sq ft → $500,000

**Step 1: Set up matrices**
$$ X = \begin{bmatrix} 1 & 1500 \\ 1 & 2000 \\ 1 & 2500 \end{bmatrix}, \quad y = \begin{bmatrix} 300000 \\ 400000 \\ 500000 \end{bmatrix} $$

**Step 2: Compute normal equations**
$$ X^T X = \begin{bmatrix} 3 & 6000 \\ 6000 & 12500000 \end{bmatrix} $$
$$ X^T y = \begin{bmatrix} 1200000 \\ 2450000000 \end{bmatrix} $$

**Step 3: Solve for $`\beta`$**
$$ \hat{\beta} = (X^T X)^{-1} X^T y = \begin{bmatrix} 100000 \\ 160 \end{bmatrix} $$

**Step 4: Prediction equation**
$$ \hat{y} = 100000 + 160x $$

**Intuition**: This tells us that houses have a base price of $100,000, and each additional square foot adds $160 to the price. This is a simple, interpretable rule.

### Linear Regression for Classification

**Binary Classification**: Code $`Y`$ as 0 or 1 and use linear regression:
$$ \hat{P}(Y = 1 | X = x) = x^T \hat{\beta} $$

**Decision Rule**: Predict class 1 if $`\hat{P}(Y = 1 | X = x) > 0.5`$:
$$ \hat{f}(x) = \mathbb{I}[x^T \hat{\beta} > 0.5] $$

**Tunable Threshold**: The 0.5 threshold can be adjusted based on class imbalance or cost considerations.

**Understanding Linear Classification:**

1. **Linear Decision Boundary**: $`x^T \beta = 0.5`$ defines the decision boundary - like a line that separates the two classes
2. **Probability Interpretation**: $`x^T \beta`$ can be interpreted as log-odds - like how much more likely one class is than the other
3. **Limitations**: Predictions can be outside $`[0,1]`$ interval - like predicting probabilities greater than 1 or less than 0

**Intuition**: Linear classification finds a line (or hyperplane) that best separates the two classes. It's like drawing a line through a scatter plot to separate two groups of points.

**Example: Linear Classification**

Consider classifying emails as spam (1) or not spam (0) based on word frequencies:

Training data:
- Email 1: (0.1, 0.2) → 0 (not spam)
- Email 2: (0.8, 0.1) → 1 (spam)
- Email 3: (0.9, 0.3) → 1 (spam)

**Step 1: Fit linear regression**
$$ \hat{\beta} = \begin{bmatrix} -0.5 \\ 1.5 \\ 0.5 \end{bmatrix} $$

**Step 2: Decision boundary**
$$ -0.5 + 1.5x_1 + 0.5x_2 = 0.5 $$
$$ 1.5x_1 + 0.5x_2 = 1.0 $$

**Step 3: Classification rule**
- Predict spam if $`1.5x_1 + 0.5x_2 > 1.0`$ - like high frequency of certain words indicates spam
- Predict not spam otherwise

**Intuition**: This creates a simple rule: if the weighted sum of word frequencies is above a threshold, it's spam. It's like having a checklist for identifying spam emails.

## Pros and Cons Analysis

### Linear Regression Advantages

1. **Computational Efficiency**: $`O(np^2 + p^3)`$ for training, $`O(p)`$ for prediction
2. **Interpretability**: Coefficients have clear meaning
3. **Statistical Inference**: Confidence intervals, hypothesis tests available
4. **Scalability**: Works well with large datasets
5. **Theoretical Foundation**: Well-understood properties

**Understanding Computational Complexity:**

1. **Training**: $`O(np^2)`$ for matrix multiplication + $`O(p^3)`$ for matrix inversion - like solving a system of equations
2. **Prediction**: $`O(p)`$ for single prediction (just matrix-vector multiplication) - like plugging numbers into a formula
3. **Memory**: $`O(p^2)`$ to store $`(X^T X)^{-1}`$ - like storing the solution to the equations

**Intuition**: Linear regression is computationally efficient because it finds a simple mathematical solution. It's like solving an equation once and then just plugging in new values.

**Statistical Inference:**

Confidence intervals for coefficients:
$$ \hat{\beta}_j \pm t_{n-p-1, \alpha/2} \cdot \text{SE}(\hat{\beta}_j) $$

where $`\text{SE}(\hat{\beta}_j) = \sqrt{\sigma^2 (X^T X)^{-1}_{jj}}`$

**Intuition**: Statistical inference tells us how confident we can be in our estimates. It's like knowing not just that each bedroom adds $50,000 to price, but also how certain we are about that number.

### Linear Regression Drawbacks

1. **Linear Assumption**: May miss non-linear relationships - like assuming house prices always increase linearly with size
2. **Invalid Probabilities**: Predictions can be outside $`[0, 1]`$ interval - like predicting a 120% chance of something happening
3. **Squared Loss**: Not optimal for classification performance - like using the wrong measure of success
4. **Feature Interactions**: Cannot capture complex interactions without manual feature engineering - like not automatically discovering that location and size interact

**Intuition**: Linear regression is limited by its assumption of linearity. It's like trying to fit a straight line to data that follows a curve - it will miss the true pattern.

**Example: Linear vs. Non-linear Relationship**

See the complete implementation in [`code/linear_vs_polynomial_regression.py`](code/linear_vs_polynomial_regression.py) which demonstrates how linear regression fails on non-linear data while polynomial regression can capture the relationship.

### kNN Advantages

1. **No Assumptions**: Works with any data distribution - like not assuming any particular pattern
2. **Non-linear**: Can capture complex decision boundaries - like finding complex patterns in the data
3. **Local Adaptation**: Automatically adapts to local structure - like adjusting to different neighborhoods
4. **Conceptual Simplicity**: Easy to understand and implement - like the idea of asking neighbors

**Intuition**: kNN is very flexible because it doesn't make strong assumptions about the data. It's like being open-minded and not assuming you know the pattern beforehand.

**Example: kNN Capturing Non-linear Boundaries**

See the complete implementation in [`code/knn_nonlinear_boundary.py`](code/knn_nonlinear_boundary.py) which demonstrates how kNN can capture complex non-linear decision boundaries.

### kNN Drawbacks

1. **Computational Cost**: $`O(n)`$ for each prediction - like having to check every house in the database
2. **Curse of Dimensionality**: Performance degrades in high dimensions - like having trouble finding similar houses when considering too many features
3. **No Interpretability**: Black-box predictions - like not knowing why a particular prediction was made
4. **Sensitive to Irrelevant Features**: All features weighted equally - like treating square footage and paint color as equally important

**Understanding the Curse of Dimensionality:**

In high dimensions, all points become approximately equidistant:
$$ \lim_{p \rightarrow \infty} \frac{\max_{i,j} d(x_i, x_j) - \min_{i,j} d(x_i, x_j)}{\min_{i,j} d(x_i, x_j)} = 0 $$

**Intuition**: The curse of dimensionality is like trying to find similar houses when you're considering too many features. Eventually, every house becomes equally different from every other house, making the concept of "similarity" meaningless.

**Example: Curse of Dimensionality**

See the complete implementation in [`code/curse_of_dimensionality_demo.py`](code/curse_of_dimensionality_demo.py) which demonstrates how distances become less meaningful as dimensionality increases.

## Model Complexity Analysis

### Degrees of Freedom (DF)

**Definition**: DF measures the effective number of parameters or the flexibility of a model.

**Linear Regression**: 
- **Model DF**: $`p+1`$ (number of coefficients) - like having one parameter for each feature plus an intercept
- **Residual DF**: $`n-(p+1)`$ (for statistical inference) - like having leftover degrees of freedom for error estimation

**Mathematical Interpretation**: The least squares prediction $`\hat{y}`$ lies in a $`(p+1)`$-dimensional subspace, while the residual vector $`(y-\hat{y})`$ lies in a $`(n-p-1)`$-dimensional subspace.

**Intuition**: Degrees of freedom measure how flexible a model is. Linear regression has limited flexibility (one parameter per feature), while kNN can be very flexible or very rigid depending on k.

**kNN**:
- **Approximate DF**: $`n/k`$ - like having n/k effective parameters
- **$`k=1`$**: DF ≈ $`n`$ (highest complexity) - like having as many parameters as data points
- **$`k=n`$**: DF ≈ $`1`$ (lowest complexity) - like having just one parameter (the average)

**Understanding Degrees of Freedom:**

1. **Linear Regression**: Each parameter contributes one degree of freedom - like each feature having its own effect
2. **kNN**: Complexity decreases as $`k`$ increases - like asking more neighbors making the answer more stable
3. **Effective Parameters**: kNN has $`n/k`$ effective parameters because each neighbor contributes $`1/k`$ to the prediction - like each neighbor having a small influence

**Intuition**: Degrees of freedom measure how much the model can "wiggle" to fit the data. More degrees of freedom means more flexibility, but also more potential for overfitting.

### Complexity Comparison

| Algorithm | Parameters | Training Time | Prediction Time | Flexibility |
|-----------|------------|---------------|-----------------|-------------|
| Linear Regression | $`p+1`$ | $`O(np^2 + p^3)`$ | $`O(p)`$ | Low |
| kNN | $`n/k`$ | $`O(1)`$ | $`O(n)`$ | High |

**Understanding the Trade-offs:**

1. **Training Time**: kNN has no training, linear regression requires matrix operations - like kNN just remembering data vs. linear regression solving equations
2. **Prediction Time**: Linear regression is fast, kNN requires distance computations - like linear regression plugging into a formula vs. kNN searching through all data
3. **Memory**: Linear regression stores $`p+1`$ parameters, kNN stores all training data - like linear regression storing a formula vs. kNN storing a database
4. **Flexibility**: kNN can capture complex patterns, linear regression is limited to linear relationships - like kNN being adaptable vs. linear regression being rigid

**Intuition**: These trade-offs are fundamental. You can't have everything - fast training, fast prediction, low memory usage, and high flexibility. You have to choose based on your specific needs.

## Theoretical Foundations

### kNN Consistency

Under certain conditions, kNN is consistent (converges to the Bayes classifier):

**Theorem**: If $`k \rightarrow \infty`$ and $`k/n \rightarrow 0`$ as $`n \rightarrow \infty`$, then:
$$ \lim_{n \rightarrow \infty} \mathbb{E}[(\hat{f}_n(X) - f^*(X))^2] = 0 $$

**Understanding the Conditions:**

1. **$`k \rightarrow \infty`$**: Neighborhood size grows to include more points - like asking more and more neighbors
2. **$`k/n \rightarrow 0`$**: Neighborhood becomes smaller relative to sample size - like the neighborhood becoming more localized
3. **Result**: Local approximation becomes more accurate - like getting better and better local estimates

**Proof Sketch:**

The kNN estimator can be written as:
$$ \hat{f}_n(x) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i $$

As $`n \rightarrow \infty`$ and $`k/n \rightarrow 0`$:
1. The neighborhood $`\mathcal{N}_k(x)`$ becomes smaller - like focusing on a smaller and smaller area
2. Points in the neighborhood become closer to $`x`$ - like neighbors becoming more similar
3. $`\hat{f}_n(x) \rightarrow \mathbb{E}[Y|X=x] = f^*(x)`$ - like the local average approaching the true value

**Intuition**: As you get more data and ask more neighbors (but keep the neighborhood small relative to the total data), your local estimates become more and more accurate.

### Linear Regression Optimality

**Gauss-Markov Theorem**: Under the linear model assumptions, the least squares estimator is the Best Linear Unbiased Estimator (BLUE).

**Assumptions for Gauss-Markov:**
1. **Linearity**: $`Y = X^T \beta + \epsilon`$ - like the true relationship being linear
2. **Random Sampling**: Data is randomly sampled - like not having systematic bias in data collection
3. **No Perfect Multicollinearity**: $`X`$ has full column rank - like features not being perfectly correlated
4. **Homoscedasticity**: $`\text{Var}(\epsilon_i) = \sigma^2`$ for all $`i`$ - like error variance being constant
5. **No Autocorrelation**: $`\text{Cov}(\epsilon_i, \epsilon_j) = 0`$ for $`i \neq j`$ - like errors being independent

**Maximum Likelihood**: Under normality assumption, least squares is equivalent to maximum likelihood estimation.

**Understanding MLE for Linear Regression:**

If $`\epsilon \sim N(0, \sigma^2)`$, then $`Y \sim N(X^T \beta, \sigma^2)`$.

The likelihood function is:
$$ L(\beta, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^T \beta)^2}{2\sigma^2}\right) $$

The log-likelihood is:
$$ \ell(\beta, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - x_i^T \beta)^2 $$

Maximizing with respect to $`\beta`$ is equivalent to minimizing the sum of squared residuals.

**Intuition**: Maximum likelihood estimation finds the parameters that make the observed data most likely. Under normal errors, this is equivalent to minimizing squared errors.

# 1.2.2. Simulation Study

We now conduct a comprehensive simulation study to compare kNN and linear regression on two carefully designed examples. This study will illustrate the bias-variance tradeoff and help us understand when each method performs well.

**Intuition**: This simulation study is like running experiments to see which method works better in different situations. It's like testing two different approaches to solving the same problem and seeing which one works better.

## Data Generation Process

### Example 1: Simple Gaussian Classes

**Data Structure**: Binary classification with two classes (0 and 1) in two-dimensional feature space.

**Class 1 Data Generation**:
$$ X | Y = 1 \sim N(\mu_1, \sigma^2 I_2) $$
where $`\mu_1 = (1, 1)^T`$ and $`\sigma^2 = 1`$.

**Class 0 Data Generation**:
$$ X | Y = 0 \sim N(\mu_0, \sigma^2 I_2) $$
where $`\mu_0 = (-1, -1)^T`$ and $`\sigma^2 = 1`$.

**Class Prior**: $`P(Y = 1) = P(Y = 0) = 0.5`$

**Sample Sizes**: 
- Training: 200 samples (100 per class)
- Test: 10,000 samples (5,000 per class)

**Key Characteristics**:
- Linear decision boundary exists
- Equal class priors
- Homoscedastic Gaussian noise
- Well-separated class means

**Understanding the Data Generation:**

1. **Class Separation**: $`\|\mu_1 - \mu_0\| = \sqrt{8} \approx 2.83`$ (well-separated) - like the two classes being clearly different
2. **Linear Boundary**: The optimal decision boundary is linear - like being able to draw a straight line to separate the classes
3. **Equal Variance**: Both classes have the same covariance structure - like both classes having the same amount of variability
4. **Balanced Classes**: Equal prior probabilities - like both classes being equally common

**Intuition**: This is like having two groups of people who are clearly different from each other, and you can draw a straight line to separate them. It's a simple, clean problem where linear methods should work well.

### Example 2: Complex Mixture Distribution

**Data Structure**: Binary classification with mixture distributions for each class.

**Class 1 Data Generation**:
$$ X | Y = 1 \sim \sum_{j=1}^{10} w_j N(\mu_{1j}, \sigma^2 I_2) $$
where $`w_j = 1/10`$ for all $`j`$ and $`\mu_{1j}`$ are 10 different centers.

**Class 0 Data Generation**:
$$ X | Y = 0 \sim \sum_{j=1}^{10} w_j N(\mu_{0j}, \sigma^2 I_2) $$
where $`w_j = 1/10`$ for all $`j`$ and $`\mu_{0j}`$ are 10 different centers.

**Key Characteristics**:
- Non-linear decision boundary
- Complex class-conditional distributions
- Multiple modes per class
- Challenging for linear methods

**Understanding Mixture Distributions:**

1. **Multiple Modes**: Each class has 10 different centers - like each class having multiple subgroups
2. **Non-linear Boundary**: The optimal decision boundary is highly non-linear - like needing a complex curve to separate the classes
3. **Complex Structure**: Data cannot be separated by a single hyperplane - like the groups being intermingled in a complex way
4. **Local Patterns**: Different regions have different class distributions - like different neighborhoods having different characteristics

**Intuition**: This is like having two groups of people who are mixed together in a complex way, with multiple subgroups within each group. You can't separate them with a simple straight line - you need a more complex approach.

## Mixture Distribution Theory

### Mathematical Definition

A **mixture distribution** is a probabilistic model representing various subgroups within a larger population. The probability density function is:

$$ f(x) = \sum_{j=1}^k w_j f_j(x) $$

where:
- $`w_j`$ are mixing weights with $`\sum_{j=1}^k w_j = 1`$
- $`f_j(x)`$ are component densities (e.g., Gaussian PDFs)

**Understanding Mixture Models:**

1. **Component Densities**: Each $`f_j(x)`$ represents a subpopulation - like different types of houses
2. **Mixing Weights**: $`w_j`$ represents the proportion of the population in component $`j`$ - like what fraction of houses are each type
3. **Flexibility**: Can model complex, multi-modal distributions - like capturing the fact that house prices cluster around different values
4. **Interpretability**: Each component can have meaningful interpretation - like each cluster representing a different neighborhood

**Intuition**: Mixture models are like saying "the population is made up of several different groups, each with their own characteristics." It's like recognizing that house prices cluster around different values because houses are in different neighborhoods.

### Sampling from Mixture Distributions

**Two-Step Process**:

1. **Component Selection**: Draw $`Z \sim \text{Categorical}(w_1, \ldots, w_k)`$ - like randomly choosing which neighborhood a house is in
2. **Data Generation**: Draw $`X | Z = j \sim f_j(x)`$ - like generating house characteristics based on the chosen neighborhood

**Mathematical Justification**: This treats the mixture as the marginal of a joint distribution:
$$ f(x, z) = f(z) f(x | z) = w_z f_z(x) $$

**Marginal Distribution**: $`f(x) = \sum_{z=1}^k f(x, z) = \sum_{j=1}^k w_j f_j(x)`$

**Intuition**: Sampling from a mixture is like first deciding which group something belongs to, then generating its characteristics based on that group. It's like first deciding which neighborhood to build a house in, then determining its price based on that neighborhood.

**Example: Gaussian Mixture Model**

See the complete implementation in [`code/gaussian_mixture_model.py`](code/gaussian_mixture_model.py) which demonstrates fitting a Gaussian Mixture Model to multi-modal data.

## Implementation Strategy

### kNN Implementation

1. **Parameter Grid**: Define a set of $`k`$ values (e.g., $`k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201\}`$)

2. **Error Storage**: Initialize vectors to store training and test errors for each $`k`$

3. **Distance Computation**: For each test point, compute distances to all training points

4. **Prediction**: For each $`k`$:
   - Find $`k`$ nearest neighbors
   - Compute majority vote or average
   - Calculate classification error

**Intuition**: This is like systematically testing different numbers of neighbors to see which gives the best results. It's like trying different group sizes when asking for advice.

**Example: kNN Implementation**

See the complete implementation in [`code/knn_implementation.py`](code/knn_implementation.py) which provides a function to evaluate kNN for different k values.

### Linear Regression Implementation

1. **Data Preparation**: Convert categorical labels (0, 1) to numerical values

2. **Model Fitting**: Fit linear regression using least squares:
$$ \hat{\beta} = (X^T X)^{-1} X^T y $$

3. **Prediction**: Compute $`\hat{P}(Y = 1 | X = x) = x^T \hat{\beta}`$

4. **Classification**: Apply threshold at 0.5:
$$ \hat{f}(x) = \mathbb{I}[x^T \hat{\beta} > 0.5] $$

5. **Error Calculation**: Compute training and test classification errors

**Intuition**: This is like finding the best straight line to separate the two classes, then using that line to make predictions. It's like drawing a line through a scatter plot and using it to classify new points.

**Example: Linear Regression Implementation**

See the complete implementation in [`code/linear_regression_implementation.py`](code/linear_regression_implementation.py) which provides a function to evaluate linear regression for classification tasks.

### Performance Evaluation

**Metrics**:
- Training Error: $`\frac{1}{n} \sum_{i=1}^n \mathbb{I}[y_i \neq \hat{f}(x_i)]`$ - like how well the model does on the data it was trained on
- Test Error: $`\frac{1}{N} \sum_{j=1}^N \mathbb{I}[y_j^* \neq \hat{f}(x_j^*)]`$ - like how well the model does on new data

**Visualization**: Plot error curves vs. model complexity (k for kNN, fixed complexity for linear regression)

**Intuition**: Performance evaluation is like testing how well each method works. Training error tells us how well the model fits the data it learned from, while test error tells us how well it generalizes to new data.

**Example: Performance Visualization**

See the complete implementation in [`code/performance_comparison.py`](code/performance_comparison.py) which provides functions to visualize and compare the performance of kNN and linear regression.

## Expected Results

### Example 1 Predictions

**Linear Regression**: Should perform well due to:
- Linear decision boundary exists - like the data being separable by a straight line
- Gaussian class-conditional distributions - like the data following simple patterns
- Well-separated class means - like the classes being clearly different

**kNN**: Should also perform well, with optimal $`k`$ likely in the middle range (5-15)

**Intuition**: Both methods should work well on this simple, linear problem. It's like both a simple rule and asking neighbors should give good advice when the situation is straightforward.

### Example 2 Predictions

**Linear Regression**: Should perform poorly due to:
- Non-linear decision boundary - like needing a complex curve to separate the classes
- Complex mixture distributions - like the data having multiple patterns
- Multiple modes per class - like each class having multiple subgroups

**kNN**: Should perform better than linear regression, with optimal $`k`$ likely smaller than in Example 1

**Intuition**: Linear regression will struggle with this complex, non-linear problem, while kNN should be able to adapt to the local patterns. It's like a simple rule failing in a complex situation, but asking neighbors still working because they can adapt to local conditions.

### Bias-Variance Analysis

**Example 1**: Both methods should have low bias, with kNN showing higher variance for small $`k`$

**Example 2**: Linear regression will have high bias, while kNN can achieve lower bias at the cost of higher variance

**Understanding the Results:**

1. **Example 1 (Linear Data)**:
   - Linear regression: Low bias, low variance - like a simple rule working well in a simple situation
   - kNN: Low bias, moderate variance (depends on k) - like asking neighbors working well but with some variability
   - Both methods should achieve similar performance

2. **Example 2 (Non-linear Data)**:
   - Linear regression: High bias (cannot capture non-linear patterns) - like a simple rule failing in a complex situation
   - kNN: Low bias, higher variance - like asking neighbors working well but being more variable
   - kNN should outperform linear regression

**Intuition**: This simulation study will show us when each method works best. It's like running experiments to see which approach works better in different situations, helping us understand the trade-offs between simplicity and flexibility.

This simulation study will provide concrete evidence of the bias-variance tradeoff and help us understand the strengths and limitations of each method. The results will demonstrate when each algorithm is most appropriate and how to choose optimal parameters for different types of data.

---

**Navigation:**
- **Next Topic:** [Bayes Rule](05_bayes_rule.md) - Understanding Bayesian decision theory and optimal classification
- **Previous Topic:** [Bias and Variance Tradeoff](03_bias_variance.md) - Understanding the fundamental tradeoff between model complexity and generalization
