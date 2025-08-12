# 5.1. Polynomial Regression

## 5.1.1. Introduction to Polynomial Regression

Polynomial regression is a form of nonlinear regression that models the relationship between a dependent variable and one or more independent variables as an $`n`$-th degree polynomial. While the relationship between variables is nonlinear, the model remains linear in the parameters, making it a special case of multiple linear regression.

**Intuitive Understanding**: Polynomial regression is like using a flexible curve to fit data points instead of a straight line. Imagine you're trying to draw a smooth curve through a set of points on a piece of paper. A straight line might miss the pattern, but a curved line (like a parabola or cubic curve) can follow the natural shape of the data. Polynomial regression gives us mathematical tools to find the best curve that fits our data, whether it's a simple curve (quadratic) or a more complex one (cubic, quartic, etc.). The key insight is that we're still using linear regression techniques, but we're applying them to transformed features (powers of the original variable) rather than the original features themselves.

### Mathematical Framework

Consider a polynomial regression model of degree $`d`$ with a single predictor variable:

$$ y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \epsilon_i $$

where:
- $`y_i`$ is the response variable for observation $`i`$ - like the house price we want to predict
- $`x_i`$ is the predictor variable - like the house size
- $`\beta_0, \beta_1, \ldots, \beta_d`$ are the polynomial coefficients - like the weights for each power of house size
- $`\epsilon_i`$ is the error term, typically assumed to be $`\epsilon_i \sim N(0, \sigma^2)`$ - like the random variation in house prices

**Intuition**: This formula says that we can predict the response by adding up different powers of the predictor variable, each multiplied by its own coefficient. For example, if we're predicting house prices based on size, we might use: price = β₀ + β₁×size + β₂×size² + β₃×size³. This allows us to capture relationships that aren't straight lines, like when house prices increase faster than linearly with size (maybe due to luxury features in larger houses).

### Matrix Formulation

The polynomial regression model can be expressed in matrix form as:

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon} $$

where:

$$ \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad
\mathbf{X} = \begin{pmatrix} 
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{pmatrix}, \quad
\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d \end{pmatrix}, \quad
\boldsymbol{\epsilon} = \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{pmatrix} $$

**Intuition**: This matrix formulation shows that polynomial regression is really just multiple linear regression in disguise! Instead of having different predictor variables (like size, age, location), we have different powers of the same variable (size, size², size³, etc.). The design matrix X contains all the powers of our predictor variable, and we're still solving the same linear regression problem we learned earlier.

### Parameter Estimation

The least squares estimator for the polynomial coefficients is:

$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} $$

The fitted values are:

$$ \hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y} $$

where $`\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T`$ is the hat matrix.

**Intuition**: This is exactly the same formula we used for multiple linear regression! The beauty of polynomial regression is that we can use all our existing linear regression tools and theory. We're just transforming our single predictor variable into multiple features (its powers) and then applying standard linear regression techniques.

### Degrees of Freedom

For a polynomial of degree $`d`$, the model has $`d + 1`$ parameters (including the intercept). The residual degrees of freedom are:

$$ \text{df}_{\text{residual}} = n - (d + 1) $$

![Degrees of Freedom and Model Flexibility](../_images/w5_ss_DR.png)

*Figure: Illustration of how degrees of freedom affect the flexibility of a polynomial regression model.*

**Intuition**: Degrees of freedom represent how much flexibility our model has to fit the data. A higher-degree polynomial has more parameters and thus more flexibility to follow the data points. However, this flexibility comes at a cost - we need more data to reliably estimate more parameters, and we risk overfitting (memorizing the training data instead of learning the true pattern).

## 5.1.2. Basis Functions and Orthogonal Polynomials

### Standard Polynomial Basis

The standard polynomial basis functions are:

$$ \phi_0(x) = 1, \quad \phi_1(x) = x, \quad \phi_2(x) = x^2, \quad \ldots, \quad \phi_d(x) = x^d $$

**Intuition**: These basis functions are like the building blocks of our polynomial curve. Each function represents a different power of x, and we combine them with different weights (coefficients) to create our final curve. Think of it like mixing different ingredients in a recipe - we can add more or less of each power to get the flavor (shape) we want.

### Orthogonal Polynomials

To avoid multicollinearity issues, orthogonal polynomials are often used. The Gram-Schmidt orthogonalization process creates orthogonal basis functions:

$$ p_0(x) = 1 $$

$$ p_1(x) = x - \frac{\sum_{i=1}^n x_i}{n} $$

$$ p_j(x) = (x - \alpha_j)p_{j-1}(x) - \beta_j p_{j-2}(x) $$

where $`\alpha_j`$ and $`\beta_j`$ are chosen to ensure orthogonality.

**Intuition**: Orthogonal polynomials are like creating a set of building blocks that don't interfere with each other. In standard polynomials, the powers of x (x, x², x³, etc.) are highly correlated - if you know x, you can predict x² pretty well. This correlation can cause numerical problems and make the coefficients hard to interpret. Orthogonal polynomials are designed so that each new term is independent of the previous ones, making the model more stable and the coefficients more meaningful.

### Implementation of Orthogonal Polynomials

**Python Implementation:** [polynomial_utilities.py](code/polynomial_utilities.py) - `create_orthogonal_polynomials()` and `create_standard_polynomials()` functions

The orthogonal polynomials implementation includes:

- **Legendre Polynomials**: Uses scipy's Legendre polynomials for numerical stability - like using proven, stable building blocks
- **Normalization**: Scales input to [-1, 1] interval for optimal orthogonality - like standardizing our measurements to a common scale
- **Standard Polynomials**: Alternative implementation using sklearn's PolynomialFeatures - like having both simple and sophisticated tools
- **Feature Creation**: Efficient creation of polynomial basis functions - like quickly building our curve components

Key features:
- Numerical stability through proper normalization - like ensuring our calculations don't break down
- Support for both orthogonal and standard polynomial bases - like having multiple approaches for different situations
- Integration with scipy and sklearn libraries - like using proven, well-tested tools

## 5.1.3. Model Selection and Degree Selection

### Information Criteria

#### Akaike Information Criterion (AIC)

$$ \text{AIC} = 2k - 2\ln(L) $$

where $`k = d + 1`$ is the number of parameters and $`L`$ is the likelihood.

For normal errors, AIC becomes:

$$ \text{AIC} = n\ln(\text{RSS}/n) + 2(d + 1) $$

**Intuition**: AIC is like a sophisticated scoring system that balances how well our curve fits the data against how complex it is. The first term (n×ln(RSS/n)) measures the fit quality - lower is better. The second term (2(d+1)) penalizes complexity - more parameters mean a higher penalty. AIC helps us find the sweet spot where we have enough flexibility to capture the true pattern but not so much that we're just memorizing the data.

#### Bayesian Information Criterion (BIC)

$$ \text{BIC} = \ln(n)k - 2\ln(L) $$

For normal errors:

$$ \text{BIC} = n\ln(\text{RSS}/n) + (d + 1)\ln(n) $$

**Intuition**: BIC is similar to AIC but penalizes complexity more heavily, especially when we have lots of data. The penalty term (d+1)×ln(n) grows faster with sample size than AIC's penalty. This makes BIC prefer simpler models when we have large datasets, which is often a good thing because we have enough data to reliably estimate simpler patterns.

### Cross-Validation

The $`k`$-fold cross-validation score is:

$$ \text{CV}(d) = \frac{1}{k}\sum_{i=1}^k \frac{1}{n_i}\sum_{j \in \text{fold}_i} (y_j - \hat{y}_j^{(-i)})^2 $$

where $`\hat{y}_j^{(-i)}`$ is the prediction for observation $`j`$ using the model trained on all folds except fold $`i`$.

**Intuition**: Cross-validation is like testing our curve on data it hasn't seen before. We split our data into k parts, train our polynomial on k-1 parts, and test it on the remaining part. We repeat this k times, each time using a different part for testing. This gives us a realistic estimate of how well our curve will perform on new data, which is the ultimate test of whether we've found the right complexity level.

### Forward and Backward Selection

#### Forward and Backward Selection Algorithms

**Python Implementation:** [polynomial_utilities.py](code/polynomial_utilities.py) - `forward_polynomial_selection()` and `backward_polynomial_selection()` functions

The model selection algorithms include:

- **Forward Selection**: Incrementally adds polynomial terms from degree 1 to max_degree - like building a curve step by step, starting simple and adding complexity only when it helps
- **Backward Selection**: Starts with max_degree and removes terms sequentially - like starting with a complex curve and simplifying it by removing unnecessary parts
- **Multiple Criteria**: Support for AIC, BIC, and cross-validation based selection - like having multiple judges to evaluate our curve
- **Comprehensive Evaluation**: Calculates all relevant metrics for each degree - like getting a complete report card for each complexity level

Key features:
- Systematic exploration of polynomial degrees - like methodically testing different curve complexities
- Multiple information criteria for model selection - like having different perspectives on what makes a good model
- Cross-validation integration for robust evaluation - like ensuring our choice works well on new data
- Visualization of selection results - like seeing how different complexity levels perform

## 5.1.4. Complete Polynomial Regression Implementation

### Python Implementation

**Complete Implementation:** [polynomial_regression.py](code/polynomial_regression.py)

The Python implementation includes:

- **PolynomialRegression Class**: Complete implementation with training, prediction, and evaluation - like a complete toolkit for building polynomial curves
- **Orthogonal Polynomials**: Support for both standard and orthogonal polynomial bases - like having both simple and sophisticated curve-building tools
- **Comprehensive Metrics**: MSE, RMSE, R², Adjusted R², AIC, and BIC calculations - like having a complete evaluation system for our curves
- **Model Demonstration**: Complete example with synthetic data and visualization - like worked examples showing how to build and evaluate curves
- **Equation Generation**: Automatic generation of polynomial equations - like automatically writing down the mathematical formula for our curve

Key features:
- Configurable polynomial degree and basis type - like adjustable settings for curve complexity and type
- Built-in performance metrics calculation - like automatic evaluation of curve quality
- Comprehensive visualization tools - like tools to see how well our curve fits the data
- Integration with sklearn for robust implementation - like using proven, reliable tools

### R Implementation

**Complete R Implementation:** [r_polynomial_regression.R](code/r_polynomial_regression.R)

The R implementation provides:

- **Polynomial Feature Creation**: Efficient creation of polynomial basis functions - like quickly building the components of our curve
- **Model Fitting**: Complete polynomial regression implementation using lm() - like using R's proven linear regression tools for polynomial fitting
- **Comprehensive Metrics**: MSE, RMSE, R², Adjusted R², AIC, and BIC calculations - like complete evaluation of curve performance
- **Visualization**: Advanced plotting using ggplot2 for model comparison - like publication-quality graphs showing curve fits
- **Cross-Validation**: Built-in cross-validation for degree selection - like automatic testing of different curve complexities
- **Residual Analysis**: Complete diagnostic tools for model validation - like tools to check if our curve assumptions are reasonable

Key features:
- Uses base R lm() function for robust fitting - like using R's battle-tested linear regression engine
- Integration with ggplot2 for publication-quality plots - like professional-looking visualizations
- Comprehensive model evaluation and comparison - like complete analysis of curve performance
- Modular function design for easy customization - like flexible tools that can be adapted to different needs

## 5.1.5. Model Diagnostics and Validation

### Residual Analysis

**Python Implementation:** [polynomial_regression.py](code/polynomial_regression.py) - `analyze_polynomial_residuals()` function

The residual analysis implementation includes:

- **Diagnostic Plots**: Comprehensive 2x2 grid of residual plots - like a complete health check for our curve
- **Normality Tests**: Shapiro-Wilk and Jarque-Bera tests for residual normality - like checking if our errors follow the expected pattern
- **Visualization**: Residuals vs fitted, Q-Q plots, residuals vs predictor, and histogram - like multiple views of how well our curve performs
- **Statistical Validation**: Formal hypothesis tests for model assumptions - like rigorous testing of whether our curve is appropriate

Key features:
- Complete diagnostic suite for polynomial regression - like a comprehensive health check for our curve
- Integration with scipy for statistical testing - like using proven statistical tools
- Publication-quality visualization - like professional-looking diagnostic plots
- Comprehensive model validation tools - like thorough testing of curve assumptions

**Intuition**: Residual analysis is like checking the quality of our curve fit. We look at the differences between our predictions and the actual data points to see if our curve is working well. Good residuals should be random (no patterns), normally distributed, and have constant variance. If we see patterns in the residuals, it might mean our curve isn't complex enough or we're missing important features.

### Cross-Validation for Model Selection

**Python Implementation:** [polynomial_regression.py](code/polynomial_regression.py) - `cross_validate_polynomial_degree()` function

The cross-validation implementation includes:

- **K-Fold Cross-Validation**: Systematic evaluation of polynomial degrees - like testing different curve complexities on multiple datasets
- **Optimal Degree Selection**: Automatic identification of best polynomial degree - like automatically finding the right curve complexity
- **Visualization**: Plot of CV scores vs polynomial degree - like seeing how different complexities perform
- **Robust Evaluation**: Multiple fold evaluation for reliable model selection - like ensuring our choice is stable and reliable

Key features:
- Integration with sklearn's KFold for robust validation - like using proven cross-validation tools
- Automatic optimal degree identification - like having an expert automatically choose the best curve complexity
- Comprehensive visualization of selection process - like seeing the evidence behind our choice
- Reliable model selection methodology - like using a proven method for choosing curve complexity

**Intuition**: Cross-validation for model selection is like testing different curve complexities on data we haven't seen before. We try different polynomial degrees (linear, quadratic, cubic, etc.) and see which one performs best on held-out data. This helps us avoid overfitting - choosing a curve that's too complex and just memorizes the training data.

## 5.1.6. Limitations and Alternatives

### Limitations of Polynomial Regression

1. **Overfitting**: High-degree polynomials can fit noise in the data - like memorizing random fluctuations instead of learning the true pattern
2. **Extrapolation Issues**: Polynomials behave poorly outside the training range - like our curve going crazy when we try to predict beyond our data
3. **Global Nature**: Assumes the same relationship holds across the entire domain - like assuming the same curve shape applies everywhere
4. **Interpretability**: High-degree terms are difficult to interpret - like having a complex formula that's hard to understand

**Intuition**: These limitations are like the trade-offs we face when choosing any modeling approach. Polynomial regression is powerful but has specific weaknesses. It's like having a very flexible tool that can fit many shapes, but that flexibility comes with risks - it might fit the wrong things or behave badly in new situations.

### Mathematical Analysis of Limitations

#### Extrapolation Problem

For a polynomial $`f(x) = \sum_{i=0}^d \beta_i x^i`$, the behavior as $`x \to \infty`$ is dominated by the highest degree term:

$$ \lim_{x \to \infty} f(x) = \lim_{x \to \infty} \beta_d x^d $$

This leads to explosive growth or decay outside the training range.

**Intuition**: This mathematical result shows why polynomials are terrible at extrapolation. When we go far outside our training range, the highest power term dominates, and polynomials either shoot up to infinity or crash down to negative infinity. It's like having a curve that works well in our neighborhood but goes crazy when we try to use it in a different city.

#### Runge's Phenomenon

For equally spaced interpolation points, high-degree polynomials can exhibit oscillatory behavior:

$$ f(x) = \frac{1}{1 + 25x^2} $$

The interpolating polynomial of degree $`n`$ at $`n+1`$ equally spaced points can have maximum error growing exponentially with $`n`$.

**Intuition**: Runge's phenomenon is a famous example showing that more complex polynomials aren't always better. Even when we have a simple, smooth function, a high-degree polynomial trying to fit it perfectly at many points can create wild oscillations between the points. It's like trying to thread a needle with a rope - the rope is too flexible and goes all over the place instead of following a smooth path.

### Alternatives to Polynomial Regression

1. **Spline Regression**: Piecewise polynomials with continuity constraints - like using different curves for different regions, but making sure they connect smoothly
2. **Local Polynomial Regression**: Fitting polynomials in local neighborhoods - like using simple curves in small areas instead of one complex curve everywhere
3. **Kernel Regression**: Non-parametric smoothing methods - like letting the data itself determine the curve shape
4. **Basis Expansion**: Using other basis functions (Fourier, wavelet, etc.) - like using different building blocks (sine waves, wavelets) instead of polynomial powers

**Intuition**: These alternatives are like having different tools in our toolbox. Each has its own strengths and is better suited for different situations. Splines are great when we want smooth curves that can change behavior in different regions. Local polynomials are good when the relationship changes across the domain. Kernel regression is flexible and data-driven. Basis expansions give us different mathematical building blocks to work with.

## Summary

Polynomial regression provides a flexible approach to modeling nonlinear relationships while maintaining linearity in parameters. Key concepts include:

1. **Mathematical Foundation**: Linear in parameters, nonlinear in predictors - like using linear regression tools to build curved relationships
2. **Model Selection**: AIC, BIC, and cross-validation for degree selection - like having multiple ways to choose the right curve complexity
3. **Orthogonal Polynomials**: Avoiding multicollinearity issues - like using independent building blocks for more stable curves
4. **Diagnostics**: Residual analysis and model validation - like checking the health of our curve fit
5. **Limitations**: Overfitting, extrapolation issues, and global assumptions - like understanding when polynomial regression might not be the best choice

The method serves as a foundation for more advanced nonlinear regression techniques like splines and local polynomial methods.

**Intuition**: Polynomial regression is like having a mathematical Swiss Army knife for curve fitting. It can capture many different types of nonlinear relationships by combining different powers of our predictor variables. The key insight is that we're still using linear regression techniques, but we're applying them to transformed features. This gives us the flexibility to model curved relationships while keeping the mathematical framework simple and well-understood.

However, this flexibility comes with trade-offs. We need to be careful about choosing the right complexity level to avoid overfitting. We also need to be aware that polynomials can behave badly outside our training range and that they assume the same relationship holds everywhere. For these reasons, polynomial regression is often best used as a starting point or as part of more sophisticated methods like splines.

## Code Files Summary

The following code files provide complete implementations of the concepts discussed in this chapter:

### Python Implementation
- **[polynomial_utilities.py](code/polynomial_utilities.py)**: Utility functions for orthogonal polynomials, feature creation, and model selection algorithms (forward/backward selection) - like tools for building and selecting polynomial curves
- **[polynomial_regression.py](code/polynomial_regression.py)**: Complete polynomial regression implementation including the PolynomialRegression class, demonstration functions, residual analysis, and cross-validation - like a complete toolkit for polynomial curve fitting

### R Implementation
- **[r_polynomial_regression.R](code/r_polynomial_regression.R)**: Complete R implementation using base R functions with comprehensive model fitting, evaluation, visualization, and diagnostic tools - like a complete R toolkit for polynomial regression

Each file includes comprehensive examples, demonstrations, and analysis tools to help understand and apply polynomial regression concepts in practice.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning. Springer.
- Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to linear regression analysis. John Wiley & Sons.

Polynomial regression provides a natural extension of linear regression for modeling nonlinear relationships, but requires careful consideration of degree selection, overfitting, and interpretability.

---

**Navigation:**
- **Next Topic:** [Cubic Splines](02_cubic_splines.md) - Piecewise polynomial functions with continuity constraints
- **Previous Topic:** [Nonlinear Regression Overview](README.md) - Overview of nonlinear regression methods and applications
