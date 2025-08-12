# 2. Linear Regression: Foundation of Statistical Learning

## 2.1. Multiple Linear Regression: A Comprehensive Guide

Multiple linear regression (MLR) is the cornerstone of statistical learning and predictive modeling. It provides a powerful yet interpretable framework for understanding relationships between variables and making predictions. This section provides a deep dive into the theory, implementation, and practical considerations of MLR.

**Think of multiple linear regression as the "Swiss Army knife" of statistical modeling.** Just as a Swiss Army knife has multiple tools that work together to solve various problems, multiple linear regression combines multiple factors to predict an outcome. It's like having a recipe where each ingredient contributes to the final taste, and you can adjust the amounts to get exactly the flavor you want.

### What is Multiple Linear Regression?

Multiple linear regression models the relationship between a single response variable and multiple predictor variables using a linear function. The model assumes that the response variable can be expressed as a weighted sum of the predictors plus some random error.

**Intuitive Understanding**: Multiple linear regression is like creating a recipe for predicting house prices. You know that house prices depend on square footage, number of bedrooms, location, and other factors. Each factor has its own "weight" or importance, and together they determine the final price. The model learns these weights from data.

**Key Insight**: Despite its name, "linear" refers to linearity in the parameters, not necessarily in the predictor variables themselves. This allows for polynomial terms, interactions, and transformations while maintaining the linear framework.

**Intuition**: This means you can include squared terms (like square footage squared) or interaction terms (like square footage × number of bedrooms) and still use linear regression. It's like being able to add spices and seasonings to your recipe while still following the basic cooking method.

### Why Study Multiple Linear Regression?

1. **Foundation**: Many advanced methods build upon linear regression concepts - like learning basic cooking techniques before mastering complex recipes
2. **Interpretability**: Coefficients have clear, meaningful interpretations - like knowing exactly how much each ingredient affects the final taste
3. **Computational Efficiency**: Fast to fit and make predictions - like having a quick, reliable cooking method
4. **Statistical Theory**: Well-understood properties and inference methods - like understanding the science behind why recipes work
5. **Benchmark**: Often serves as a baseline for comparing more complex models - like comparing new recipes to classic ones

**Intuition**: Multiple linear regression is like learning to cook with a recipe book. Once you understand the basics, you can create more complex dishes, adapt recipes to your taste, and even invent new ones.

## 2.1.1. Mathematical Foundation

### The Linear Model

The general form of the multiple linear regression model is:

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon $$

where:
- $`y`$ is the response (dependent) variable
- $`x_1, x_2, \ldots, x_p`$ are the predictor (independent) variables
- $`\beta_0`$ is the intercept (baseline value when all predictors are zero)
- $`\beta_1, \beta_2, \ldots, \beta_p`$ are the regression coefficients (slopes)
- $`\varepsilon`$ is the error term, representing unmodeled variation

**Intuitive Understanding**: This equation is like a recipe where $`y`$ is the final dish, each $`x_j`$ is an ingredient, each $`\beta_j`$ is how much of that ingredient to use, $`\beta_0`$ is the base flavor, and $`\varepsilon`$ is the random variation that makes each dish slightly different.

### Understanding the Components

**Intercept ($`\beta_0`$)**:
- Represents the expected value of $`y`$ when all predictors are zero - like the base price of a house with zero square footage and zero bedrooms
- May or may not have practical interpretation depending on the data - like a base flavor that might not make sense in isolation
- Can be eliminated by centering predictors: $`x_i' = x_i - \bar{x}_i`$ - like adjusting all ingredients relative to their average amounts

**Intuition**: The intercept is like the starting point or baseline. In house pricing, it might represent the minimum cost of land or basic construction. In cooking, it might be the base flavor before adding any specific ingredients.

**Regression Coefficients ($`\beta_j`$)**:
- $`\beta_j`$ represents the expected change in $`y`$ for a one-unit increase in $`x_j`$, holding all other predictors constant - like how much the house price increases for each additional square foot
- This is the **partial effect** of $`x_j`$ on $`y`$ - like the isolated effect of adding more salt to a dish
- Units: change in $`y`$ per unit change in $`x_j`$ - like dollars per square foot or taste units per gram of salt

**Intuition**: Each coefficient tells you the isolated effect of one variable. It's like knowing that adding one more bedroom increases house price by $50,000, regardless of the house's size or location.

**Error Term ($`\varepsilon`$)**:
- Captures all variation in $`y`$ not explained by the linear combination of predictors - like the random factors that make each house unique
- Assumed to be random with mean zero and constant variance - like random cooking variations that average out
- Represents measurement error, omitted variables, and model misspecification - like forgetting to measure some ingredients or using slightly different cooking techniques

**Intuition**: The error term is like the "unpredictable" part of the outcome. In house pricing, it might include factors like the seller's motivation, market timing, or unique features. In cooking, it might be slight variations in ingredient quality or cooking conditions.

### Assumptions of Linear Regression

For valid inference and optimal properties, we typically assume:

1. **Linearity**: $`E[y \mid x_1, \ldots, x_p] = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p`$ - like the relationship between ingredients and final taste being predictable
2. **Independence**: Observations are independent of each other - like each house sale being independent of others
3. **Homoscedasticity**: $`\text{Var}(\varepsilon_i) = \sigma^2`$ for all $`i`$ - like the random variation being consistent across all observations
4. **Normality**: $`\varepsilon_i \sim N(0, \sigma^2)`$ (for inference) - like random errors following a bell curve
5. **No Multicollinearity**: Predictors are not perfectly correlated - like not having two ingredients that are essentially the same thing

**Intuition**: These assumptions are like the rules for a reliable recipe. If you follow them, you get consistent, predictable results. If you violate them, your predictions become unreliable.

## 2.1.2. Matrix Representation: The Power of Linear Algebra

### Why Use Matrix Notation?

Matrix notation provides several advantages:
- **Compactness**: Express complex operations in simple formulas - like writing a recipe in shorthand
- **Computational Efficiency**: Leverage optimized linear algebra libraries - like using a food processor instead of chopping by hand
- **Theoretical Clarity**: Reveal geometric and algebraic insights - like understanding the structure of the recipe
- **Generalization**: Extends naturally to more complex models - like adapting a basic recipe to create variations

**Intuition**: Matrix notation is like having a standardized recipe format. Once you learn the format, you can quickly understand any recipe and easily modify it for different situations.

### The Matrix Formulation

For $`n`$ observations and $`p`$ predictors, we can write the model as:

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

where:
- $`\mathbf{y}`$ is an $`n \times 1`$ vector of observed responses
- $`\mathbf{X}`$ is the $`n \times (p+1)`$ design matrix (including intercept column)
- $`\boldsymbol{\beta}`$ is a $`(p+1) \times 1`$ vector of coefficients
- $`\boldsymbol{\varepsilon}`$ is an $`n \times 1`$ vector of errors

**Intuitive Understanding**: This is like having a recipe book where each row is a different dish, each column is a different ingredient, and the coefficients tell you how much of each ingredient to use.

### Detailed Matrix Structure

**Response Vector**:
$$ \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix} $$

**Intuition**: This is like a list of all the final dish prices or tastes - one for each observation.

**Design Matrix**:
$$ \mathbf{X} = \begin{pmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix} $$

**Intuition**: This is like a recipe book where each row is a different house (or dish), and each column is a different feature (or ingredient). The first column of ones represents the base flavor or starting point.

**Coefficient Vector**:
$$ \boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix} $$

**Intuition**: This is like the list of ingredient amounts or feature weights that the model learns.

**Error Vector**:
$$ \boldsymbol{\varepsilon} = \begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{pmatrix} $$

**Intuition**: This is like the random variations that make each dish or house unique.

### Complete Matrix Equation

$$ \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix} = 
\begin{pmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1p} \\
1 & x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
\begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix} +
\begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{pmatrix} $$

**Intuition**: This equation says that each observed outcome equals the weighted sum of features plus some random variation. It's like saying each house price equals the base price plus the contribution from each feature plus some random factors.

### Understanding the Design Matrix

**Column Structure**:
- **First column**: All ones (for the intercept term) - like the base ingredient that goes into every dish
- **Remaining columns**: Observed values of each predictor - like the amounts of each specific ingredient
- **Row $`i`$**: Values for observation $`i`$ across all predictors - like the recipe for dish number $`i`$

**Matrix Dimensions**:
- $`n`$ rows (one per observation) - like one row per house or dish
- $`p+1`$ columns ($`p`$ predictors plus intercept) - like one column per feature plus the base column
- Total elements: $`n \times (p+1)`$ - like the total number of measurements in your dataset

**Intuition**: The design matrix is like a spreadsheet where each row is a different observation and each column is a different variable. It's the organized data that feeds into your model.

### Classical vs. Modern Settings

**Classical Setting ($`n \gg p`$):**
- More observations than predictors - like having many more dishes than ingredients
- Design matrix is "tall and skinny" - like a recipe book with many recipes but few ingredients
- $`\mathbf{X}^T\mathbf{X}`$ is typically invertible - like having enough information to determine the recipe
- Unique solution exists - like having one best way to combine the ingredients
- Well-understood statistical properties - like reliable cooking methods

![Classical Setting: Large n, Small p](img/w2_large_n_small_p.png)

*Figure: Classical setting with many observations and few predictors*

**Intuition**: This is like having a large cookbook with many recipes but only a few basic ingredients. You have plenty of examples to learn from, and the relationships are clear.

**Modern Setting ($`p \gg n`$):**
- More predictors than observations - like having more ingredients than dishes
- Design matrix is "short and fat" - like a recipe book with few recipes but many ingredients
- $`\mathbf{X}^T\mathbf{X}`$ is not invertible - like not having enough information to determine the recipe
- Infinitely many solutions exist - like having many possible ways to combine the ingredients
- Requires regularization or feature selection - like needing to simplify the recipe or choose which ingredients to use

![Modern Setting: Large p, Small n](img/w2_large_p_small_n.png)

*Figure: Modern setting with many predictors and few observations*

**Intuition**: This is like having a small cookbook but access to many ingredients. You don't have enough examples to learn how to use all the ingredients, so you need to be selective or use special techniques.

**Example**: In genomics, we might have 100 patients ($`n=100`$) but 20,000 gene expressions ($`p=20,000`$).

**Intuition**: This is like trying to predict a disease from gene data. You have many more genes than patients, so you need special methods to handle this "high-dimensional" problem.

## 2.1.3. Least Squares Estimation: The Foundation

### The Least Squares Principle

The most common method for estimating regression coefficients is **least squares**. The idea is to find the coefficient values that minimize the sum of squared differences between observed and predicted values.

**Intuitive Understanding**: Least squares is like adjusting a recipe to minimize the total "distance" between your predictions and the actual results. It's like tasting each dish and adjusting the ingredient amounts to make the next batch better.

**Objective Function**:
$$ \text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2 $$

where:
- $`\text{RSS}`$ = Residual Sum of Squares
- $`y_i`$ = observed value for observation $`i`$
- $`\hat{y}_i`$ = predicted value for observation $`i`$

**Intuition**: RSS measures the total squared "mistakes" your model makes. It's like adding up all the squared differences between what you predicted and what actually happened.

### Matrix Form of RSS

In matrix notation, the RSS becomes:
$$ \text{RSS}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$

**Intuition**: This is the same idea but written in matrix form. It's like writing the recipe adjustment in a more compact mathematical language.

### Geometric Interpretation

**Vector Space View**:
- $`\mathbf{y}`$ is a point in $`\mathbb{R}^n`$ - like the actual outcomes plotted in high-dimensional space
- $`\mathbf{X}\boldsymbol{\beta}`$ lies in the column space of $`\mathbf{X}`$ - like all possible predictions your model can make
- The residual $`\mathbf{y} - \mathbf{X}\boldsymbol{\beta}`$ is the vector from the prediction to the observed value - like the "error vector" pointing from prediction to reality
- Least squares finds the point in the column space closest to $`\mathbf{y}`$ - like finding the best possible prediction within your model's capabilities

**2D Example**: For simple linear regression, we find the line that minimizes the sum of squared vertical distances from points to the line.

![Least Squares Visualization](img/w2_LS.png)

*Figure: Least Squares Principle—minimizing the sum of squared vertical distances from points to the regression line*

**Intuition**: This is like finding the best-fitting line through a scatter plot. The line minimizes the total squared distance from all points to the line.

### Why Squared Error?

**Mathematical Advantages**:
1. **Differentiability**: Smooth function, easy to optimize - like having a smooth surface to find the minimum
2. **Closed-form solution**: Leads to analytical solution - like having a formula for the answer
3. **Statistical properties**: Optimal under normality assumption - like being the best method when errors are normally distributed
4. **Computational efficiency**: Fast algorithms available - like having efficient tools to find the solution

**Alternative Loss Functions**:
- **Absolute error**: $`\sum |y_i - \hat{y}_i|`$ (robust to outliers) - like being less sensitive to extreme mistakes
- **Huber loss**: Combines squared and absolute error - like using squared error for small mistakes and absolute error for large ones
- **Quantile loss**: For quantile regression - like predicting the median instead of the mean

**Intuition**: Squared error is like penalizing mistakes more heavily as they get bigger. A prediction that's off by 2 units is penalized 4 times as much as a prediction that's off by 1 unit.

## 2.1.4. The Normal Equation: Analytical Solution

### Derivation of the Normal Equation

To find the minimum of RSS, we take the derivative with respect to $`\boldsymbol{\beta}`$ and set it to zero:

**Step 1**: Expand the RSS
$$ \text{RSS}(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} $$

**Intuition**: This is like expanding the recipe to see all the terms that contribute to the total error.

**Step 2**: Take the derivative
$$ \frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} $$

**Intuition**: This is like finding the direction of steepest descent - where to adjust the recipe to reduce errors.

**Step 3**: Set to zero and solve
$$ -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0} $$

**Intuition**: This is like finding the point where no further adjustment will improve the recipe.

**Step 4**: Rearrange to get the normal equation
$$ \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} $$

**Intuition**: This is the key equation that gives us the optimal recipe - the best combination of ingredients.

### The Least Squares Solution

If $`\mathbf{X}^T\mathbf{X}`$ is invertible, the unique solution is:
$$ \hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} $$

**Components**:
- $`\mathbf{X}^T\mathbf{X}`$: Gram matrix (contains inner products of predictor columns) - like a matrix of how much each ingredient correlates with each other
- $`\mathbf{X}^T\mathbf{y}`$: Cross-product of predictors with response - like how much each ingredient correlates with the outcome
- $`(\mathbf{X}^T\mathbf{X})^{-1}`$: Inverse of Gram matrix - like "undoing" the correlations to get clean effects

**Intuition**: This formula tells us exactly how much of each ingredient to use to get the best possible recipe.

### Understanding the Solution

**Geometric Interpretation**:
- $`\mathbf{X}^T\mathbf{X}`$ measures the "spread" of the predictors - like how much the ingredients vary
- $`\mathbf{X}^T\mathbf{y}`$ measures the "alignment" between predictors and response - like how well each ingredient predicts the outcome
- The solution finds the optimal linear combination of predictors - like finding the perfect blend of ingredients

**Statistical Interpretation**:
- $`\hat{\boldsymbol{\beta}}`$ is the best linear unbiased estimator (BLUE) under Gauss-Markov assumptions - like having the most reliable recipe possible
- The solution minimizes both bias and variance among linear estimators - like balancing accuracy and consistency

**Intuition**: This solution is like finding the recipe that gives you the most reliable results - not too sensitive to small changes in ingredients, but still accurate.

### Computational Implementation

See the complete implementation in [`code/least_squares_estimation.py`](code/least_squares_estimation.py) which demonstrates least squares estimation using the normal equation.

### Numerical Stability Considerations

**Potential Issues**:
1. **Near-singular $`\mathbf{X}^T\mathbf{X}`$**: Can cause numerical instability - like having ingredients that are almost the same
2. **Large condition number**: Small changes in data cause large changes in estimates - like a recipe that's very sensitive to small changes
3. **Computational complexity**: $`O(p^3)`$ for matrix inversion - like having to do a lot of calculations

**Solutions**:
1. **QR decomposition**: More numerically stable - like using a more reliable cooking method
2. **Singular value decomposition (SVD)**: Handles rank-deficient cases - like handling recipes with missing ingredients
3. **Regularization**: Adds stability (ridge regression) - like adding constraints to make the recipe more reliable

**Intuition**: These are like different cooking techniques that handle difficult situations more gracefully.

## 2.1.5. Fitted Values, Residuals, and Model Diagnostics

### Fitted Values

Once we have $`\hat{\boldsymbol{\beta}}`$, we can compute fitted values:
$$ \hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} $$

**Properties**:
- $`\hat{\mathbf{y}}`$ lies in the column space of $`\mathbf{X}`$ - like all predictions being possible combinations of the features
- $`\hat{\mathbf{y}}`$ is the orthogonal projection of $`\mathbf{y}`$ onto the column space - like the best possible prediction within the model's capabilities
- $`\hat{\mathbf{y}}`$ minimizes the distance from $`\mathbf{y}`$ to the column space - like finding the closest possible prediction to reality

**Intuition**: Fitted values are like the predictions your model makes for each observation. They're the "best guess" your model can make given the available information.

### Residuals

Residuals are the differences between observed and fitted values:
$$ \mathbf{r} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}} $$

**Properties**:
- $`\mathbf{r}`$ is orthogonal to the column space of $`\mathbf{X}`$ - like the errors being independent of the predictions
- $`\sum_{i=1}^n r_i = 0`$ (if intercept is included) - like the errors averaging to zero
- $`\sum_{i=1}^n r_i x_{ij} = 0`$ for all $`j`$ (orthogonality conditions) - like the errors being uncorrelated with each predictor

**Intuition**: Residuals are like the "mistakes" your model makes. They show you where your predictions were too high or too low, and they should be random if your model is good.

### Residual Sum of Squares (RSS)

$$ \text{RSS} = \|\mathbf{r}\|^2 = \mathbf{r}^T\mathbf{r} = \sum_{i=1}^n r_i^2 $$

**Degrees of Freedom**:
- Total degrees of freedom: $`n`$ - like the total number of observations
- Model degrees of freedom: $`p+1`$ (number of parameters) - like the number of ingredients in your recipe
- Residual degrees of freedom: $`n - p - 1`$ - like the remaining "freedom" after fitting the model

**Intuition**: Degrees of freedom are like the "flexibility" in your model. You start with $`n`$ pieces of information, use $`p+1`$ to fit the model, and have $`n-p-1`$ left to estimate how much random variation there is.

### Error Variance Estimation

The error variance $`\sigma^2`$ is estimated by:
$$ \hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1} = \frac{\|\mathbf{r}\|^2}{n - p - 1} $$

**Why $`n - p - 1`$?**
- Each parameter estimated reduces degrees of freedom - like each ingredient you learn about reduces the uncertainty
- We need at least $`p+1`$ observations to estimate $`p+1`$ parameters - like needing at least as many dishes as ingredients
- The denominator ensures unbiased estimation - like getting an accurate estimate of the random variation

**Intuition**: This is like estimating how much random variation there is in your cooking. You divide the total squared mistakes by the remaining degrees of freedom to get an unbiased estimate.

### Comprehensive Implementation

See the complete implementation in [`code/linear_regression_analysis.py`](code/linear_regression_analysis.py) which demonstrates complete linear regression analysis including coefficient estimation, standard errors, and goodness-of-fit measures.

### Model Diagnostics

**Residual Analysis**:

See the complete implementation in [`code/diagnostic_plots.py`](code/diagnostic_plots.py) which creates diagnostic plots for linear regression including residuals vs fitted, Q-Q plots, scale-location plots, and residuals vs leverage.

**Intuition**: Model diagnostics are like quality control for your recipe. They help you check if your assumptions are reasonable and if your model is working well.

## 2.1.6. Statistical Inference and Hypothesis Testing

### Coefficient Inference

Under the normality assumption, the least squares estimator follows:
$$ \hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\beta}, \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}) $$

**Intuition**: This tells us that our estimated coefficients are normally distributed around the true values, with a variance that depends on the data structure.

**Individual Coefficient Tests**:
For testing $`H_0: \beta_j = 0`$ vs $`H_1: \beta_j \neq 0`$:
$$ t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p-1} $$

where:
$$ \text{SE}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 [(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}} $$

**Intuition**: This test asks "Is this ingredient really important?" It compares the estimated effect to its standard error to see if the effect is significantly different from zero.

### Confidence Intervals

A $`100(1-\alpha)\%`$ confidence interval for $`\beta_j`$ is:
$$ \hat{\beta}_j \pm t_{\alpha/2, n-p-1} \cdot \text{SE}(\hat{\beta}_j) $$

**Intuition**: This gives you a range where you're confident the true effect lies. It's like saying "I'm 95% confident that adding one more bedroom increases house price by between $40,000 and $60,000."

### F-Test for Overall Model

Test $`H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0`$ vs $`H_1: \text{at least one } \beta_j \neq 0`$:
$$ F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)} \sim F_{p, n-p-1} $$

**Intuition**: This test asks "Does my recipe work at all?" It compares how much the model explains to how much it doesn't explain, accounting for the number of ingredients used.

### Implementation of Statistical Tests

See the complete implementation in [`code/statistical_inference.py`](code/statistical_inference.py) which performs statistical inference including confidence intervals, hypothesis tests, and F-tests for linear regression.

## 2.1.7. Model Assessment and Validation

### Goodness of Fit Measures

**R-squared ($`R^2`$)**:
$$ R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{SSR}}{\text{TSS}} $$

where:
- $`\text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2`$ (Total Sum of Squares)
- $`\text{SSR} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2`$ (Sum of Squares Regression)

**Interpretation**: $`R^2`$ is the proportion of variance in $`y`$ explained by the model.

**Intuition**: R-squared is like the "success rate" of your recipe. If $`R^2 = 0.8`$, your model explains 80% of the variation in the outcome. It's like saying your recipe works 80% of the time.

**Adjusted R-squared**:
$$ R^2_{adj} = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)} = 1 - (1-R^2)\frac{n-1}{n-p-1} $$

**Interpretation**: Penalizes for model complexity, more appropriate for model comparison.

**Intuition**: Adjusted R-squared is like R-squared but with a penalty for using too many ingredients. It helps you avoid overfitting by preferring simpler recipes.

### Cross-Validation

See the complete implementation in [`code/cross_validation_assessment.py`](code/cross_validation_assessment.py) which demonstrates cross-validation assessment for linear regression using scikit-learn.

**Intuition**: Cross-validation is like testing your recipe on different sets of ingredients to make sure it works reliably. It helps you estimate how well your model will perform on new data.

## 2.1.8. Practical Considerations and Best Practices

### Data Preprocessing

**Centering and Scaling**:

See the complete implementation in [`code/data_preprocessing.py`](code/data_preprocessing.py) which shows data preprocessing techniques including centering and scaling for linear regression.

**Intuition**: Data preprocessing is like preparing your ingredients before cooking. Centering and scaling make the variables more comparable and the model more stable.

### Handling Multicollinearity

**Variance Inflation Factor (VIF)**:

See the complete implementation in [`code/multicollinearity_check.py`](code/multicollinearity_check.py) which computes Variance Inflation Factors (VIF) to detect multicollinearity.

**Intuition**: Multicollinearity is like having ingredients that are very similar to each other. It makes it hard to tell which ingredient is really important and can make your recipe unstable.

### Model Selection

**Stepwise Selection**:

See the complete implementation in [`code/forward_selection.py`](code/forward_selection.py) which implements forward stepwise selection for variable selection in linear regression.

**Intuition**: Model selection is like choosing which ingredients to include in your recipe. You want enough ingredients to make it tasty, but not so many that it becomes complicated and unreliable.

## 2.1.9. Advanced Topics

### Ridge Regression (L2 Regularization)

When $`\mathbf{X}^T\mathbf{X}`$ is near-singular, we can add regularization:
$$ \hat{\boldsymbol{\beta}}_{ridge} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right\} $$

**Solution**:
$$ \hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} $$

**Intuition**: Ridge regression is like adding a constraint that prevents any ingredient from being used in extreme amounts. It makes the recipe more stable and reliable.

### Lasso Regression (L1 Regularization)

For feature selection:
$$ \hat{\boldsymbol{\beta}}_{lasso} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\} $$

**Intuition**: Lasso regression is like forcing some ingredients to be used in zero amounts. It automatically selects which ingredients are most important and sets the others to zero.

### Polynomial Regression

Extend linear regression to capture non-linear relationships:

See the complete implementation in [`code/polynomial_regression.py`](code/polynomial_regression.py) which demonstrates polynomial regression to capture non-linear relationships.

**Intuition**: Polynomial regression is like adding more complex cooking techniques. Instead of just adding ingredients linearly, you can add squared terms, cubed terms, and interactions to capture more complex relationships.

## 2.1.10. Summary and Key Takeaways

### What We've Learned

1. **Mathematical Foundation**: Linear regression models relationships using linear combinations of predictors - like creating recipes with weighted ingredients
2. **Matrix Formulation**: Compact representation enabling efficient computation and theoretical insights - like having a standardized recipe format
3. **Least Squares**: Optimal estimation method under standard assumptions - like finding the best possible recipe
4. **Statistical Inference**: Hypothesis testing and confidence intervals for coefficients - like testing whether ingredients really matter
5. **Model Assessment**: R-squared, cross-validation, and diagnostic plots - like evaluating how well your recipe works
6. **Practical Considerations**: Data preprocessing, multicollinearity, and model selection - like handling real-world cooking challenges

### Key Properties

**Optimality**: Under Gauss-Markov assumptions, least squares estimators are BLUE (Best Linear Unbiased Estimators) - like having the most reliable recipe possible

**Interpretability**: Coefficients represent partial effects, holding other variables constant - like knowing exactly how each ingredient affects the final taste

**Flexibility**: Can handle polynomial terms, interactions, and transformations while maintaining linear framework - like being able to add spices and seasonings to your basic recipe

**Computational Efficiency**: Fast to fit and make predictions, even with large datasets - like having a quick, reliable cooking method

### When to Use Linear Regression

**Appropriate When**:
- Relationship between predictors and response is approximately linear - like when ingredients combine in predictable ways
- Predictors are not highly correlated - like when ingredients are distinct and independent
- Sample size is sufficient relative to number of predictors - like having enough dishes to learn from
- Interpretability is important - like when you need to understand what each ingredient contributes

**Consider Alternatives When**:
- Strong non-linear relationships exist - like when ingredients interact in complex ways
- High-dimensional data with many predictors - like having too many ingredients to manage
- Complex interaction patterns - like when the effect of one ingredient depends heavily on others
- Non-Gaussian error distributions - like when the random variation doesn't follow a bell curve

### Next Steps

This foundation in linear regression prepares us for:
1. **Generalized Linear Models**: Extending to non-Gaussian responses - like adapting recipes for different types of dishes
2. **Regularization Methods**: Ridge, Lasso, and Elastic Net - like adding constraints to make recipes more reliable
3. **Non-linear Methods**: Polynomial regression, splines, and kernel methods - like using more sophisticated cooking techniques
4. **Advanced Topics**: Mixed models, time series, and causal inference - like mastering advanced culinary arts

**Intuition**: Linear regression is like learning to cook with basic recipes. Once you master these fundamentals, you can create more complex dishes, adapt to different situations, and even invent new culinary techniques.

Linear regression remains one of the most important tools in statistical learning, providing both practical utility and theoretical insights that extend to more complex modeling approaches.

---

**Navigation:**
- **Next Topic:** [Geometric Interpretation](02_geometric_interpretation.md) - Visual and mathematical foundation of linear regression through vector spaces and projection
- **Previous Topic:** [Linear Regression Overview](README.md) - Overview of linear regression materials and learning objectives
