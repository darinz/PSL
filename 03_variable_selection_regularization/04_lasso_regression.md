# 3.4. Lasso Regression

## Introduction

Lasso (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani in 1996, is a powerful regularization technique that combines variable selection with coefficient shrinkage. Unlike ridge regression, lasso can produce exactly zero coefficients, making it particularly valuable for sparse modeling and automatic feature selection.

**Intuitive Understanding**: Lasso regression is like a "selective chef" who not only moderates ingredient amounts but can completely eliminate ingredients that aren't essential. Think of it as a cooking technique that encourages both moderation and selection - you can use ingredients in smaller amounts, but you can also choose to use no amount at all of certain ingredients. This creates recipes that are both simple and effective, using only the most important ingredients.

![Lasso Regression Solution Paths](../_images/w4_solution_path.png)

*Figure: Solution paths for lasso regression. Shows how coefficients change as the regularization parameter varies.*

**Intuition**: This graph shows how lasso regression "shrinks and eliminates" ingredient effects as the regularization parameter increases. It's like gradually reducing ingredient amounts, but with the key difference that some ingredients get completely eliminated (set to zero) while others are just reduced. This creates sparse recipes that use only essential ingredients.

## 3.4.1 Mathematical Foundation

### The Lasso Objective Function

Lasso regression modifies the standard least squares objective by adding an L1 penalty on the coefficient vector:

$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1 $$

where:
- $`\mathbf{y} \in \mathbb{R}^n`$ is the response vector - like the taste ratings of different dishes
- $`\mathbf{X} \in \mathbb{R}^{n \times p}`$ is the design matrix - like the recipe book with ingredient amounts
- $`\boldsymbol{\beta} \in \mathbb{R}^p`$ is the coefficient vector - like the importance of each ingredient
- $`\lambda \geq 0`$ is the regularization parameter - like how strict you are about ingredient selection and moderation
- $`\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|`$ is the L1 norm - like the total "absolute importance" of all ingredients

**Intuition**: This objective function is like trying to create the best-tasting dish (minimize prediction error) while also keeping the total absolute importance of all ingredients small (penalize the sum of absolute values). The key difference from ridge regression is that lasso can completely eliminate ingredients by setting their coefficients to exactly zero.

### Key Properties of the L1 Penalty

1. **Non-differentiability**: The L1 penalty is not differentiable at zero - like having a sharp corner in the penalty function
2. **Sparsity**: Can produce exactly zero coefficients - like completely eliminating certain ingredients
3. **Convexity**: The L1 penalty is convex, making optimization tractable - like having a well-behaved optimization landscape
4. **Scale sensitivity**: Unlike L2 penalty, L1 is sensitive to predictor scaling - like having different rules for different ingredient measurement units

**Intuition**: The L1 penalty is like a "sharp knife" that can completely cut out ingredients, while the L2 penalty is like a "gentle hand" that only reduces them. The sharp corner at zero is what allows lasso to set coefficients exactly to zero, creating truly sparse models.

### Orthogonal Design Matrix Case

When the design matrix $`\mathbf{X}`$ is orthogonal (i.e., $`\mathbf{X}^T\mathbf{X} = \mathbf{I}_p`$), the lasso problem can be decomposed into $`p`$ independent one-dimensional problems.

**Intuition**: This is like having ingredients that are completely independent of each other - using more salt doesn't affect how much pepper you need, and vice versa. In this special case, we can solve for each ingredient's optimal amount separately, making the problem much simpler.

First, let's decompose the residual sum of squares:

$$ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} + \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2 $$

Using the Pythagorean theorem and orthogonality:

$$ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}}\|^2_2 + \|\mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2 $$

The cross-product term vanishes because the residual vector $`\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}}`$ is orthogonal to the column space of $`\mathbf{X}`$.

**Intuition**: This decomposition is like separating the "irreducible error" (how well the best possible recipe fits the data) from the "estimation error" (how far our recipe is from the best possible recipe). The orthogonality means these errors don't interfere with each other.

Therefore, the lasso objective becomes:

$$ \begin{align*}
\hat{\boldsymbol{\beta}}_{\text{lasso}} &= \arg\min_{\boldsymbol{\beta}} \left[\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[\|\mathbf{X}\hat{\boldsymbol{\beta}}^{\text{OLS}} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta})^T\mathbf{X}^T\mathbf{X}(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \left[(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}^{\text{OLS}} - \boldsymbol{\beta}) + \lambda \|\boldsymbol{\beta}\|_1\right] \\
&= \arg\min_{\boldsymbol{\beta}} \sum_{j=1}^p \left[(\beta_j - \hat{\beta}_j^{\text{OLS}})^2 + \lambda|\beta_j|\right]
\end{align*} $$

This decomposition allows us to solve for each $`\beta_j`$ independently.

**Intuition**: This shows that when ingredients are independent, we can optimize each ingredient separately. For each ingredient, we're balancing how close we want to be to the OLS estimate versus how much we want to penalize using that ingredient at all.

## 3.4.2 The Soft-Thresholding Operator

### One-Dimensional Lasso Problem

For each component, we need to solve:

$$ \min_{x} (x - a)^2 + \lambda|x| $$

where $`a = \hat{\beta}_j^{\text{OLS}}`$ and $`x = \beta_j`$.

**Intuition**: This is like deciding how much of a single ingredient to use. We want to be close to the "best" amount (OLS estimate) but also want to avoid using too much of any ingredient. The penalty encourages us to use less, and if the ingredient isn't very important, we might not use it at all.

### Subgradient Analysis

Since the absolute value function is not differentiable at zero, we use subgradients. The subgradient of $`|x|`$ at $`x = 0`$ is any value in $`[-1, 1]`$.

**Intuition**: The absolute value function has a sharp corner at zero, like a V-shape. At this corner, there's no single "slope" - instead, there's a range of possible slopes. This is what allows lasso to set coefficients exactly to zero.

The optimality condition is:

$$ 2(x^* - a) + \lambda z^* = 0 $$

where $`z^*`$ is the subgradient of $`|x|`$ at $`x^*`$.

**Intuition**: This condition says that the "pull toward the OLS estimate" (first term) must balance the "penalty for using the ingredient" (second term). When the ingredient is set to zero, the penalty can take any value in the range $`[-1, 1]`$, allowing this balance to be achieved.

### Solution: Soft-Thresholding

The solution is given by the soft-thresholding operator:

$$ x^* = S_{\lambda/2}(a) = \text{sign}(a)(|a| - \lambda/2)_+ = \begin{cases}
a - \lambda/2, & \text{if } a > \lambda/2 \\
0, & \text{if } |a| \leq \lambda/2 \\
a + \lambda/2, & \text{if } a < -\lambda/2
\end{cases} $$

where $`(x)_+ = \max(x, 0)`$ is the positive part function.

**Intuition**: This soft-thresholding operator is like a "smart ingredient selector":
- If an ingredient is very important (large OLS coefficient), we use it but reduce the amount by $`\lambda/2`$
- If an ingredient is moderately important (medium OLS coefficient), we might eliminate it entirely (set to zero)
- If an ingredient is unimportant (small OLS coefficient), we definitely eliminate it
- The threshold $`\lambda/2`$ determines which ingredients are important enough to keep

### Component-Wise Lasso Solution

For orthogonal design matrices, the lasso solution is:

$$ \hat{\beta}_j^{\text{lasso}} = \begin{cases}
\text{sign}(\hat{\beta}_j^{\text{OLS}})(|\hat{\beta}_j^{\text{OLS}}| - \lambda/2), & \text{if } |\hat{\beta}_j^{\text{OLS}}| > \lambda/2 \\
0, & \text{if } |\hat{\beta}_j^{\text{OLS}}| \leq \lambda/2
\end{cases} $$

**Intuition**: This formula shows that lasso either:
1. **Keeps and shrinks** important ingredients (those with OLS coefficients larger than $`\lambda/2`$)
2. **Eliminates** unimportant ingredients (those with OLS coefficients smaller than $`\lambda/2`$)

The larger $`\lambda`$ is, the higher the threshold for keeping an ingredient, leading to sparser models.

### Geometric Interpretation

The soft-thresholding operator can be understood geometrically:

1. **Shrinkage**: Coefficients are shrunk toward zero by $`\lambda/2`$ - like reducing ingredient amounts
2. **Thresholding**: Coefficients smaller than $`\lambda/2`$ in magnitude are set to zero - like eliminating unimportant ingredients
3. **Sign preservation**: The sign of non-zero coefficients is preserved - like keeping the direction of the effect (positive or negative)

![Lasso Duality and Geometry](../_images/w3_lasso_duality.png)

*Figure: Geometric interpretation of the lasso constraint and solution. The diamond-shaped constraint region leads to sparse solutions.*

**Intuition**: The diamond-shaped constraint region is like a "budget" for total ingredient importance. The solution occurs where the "taste improvement" contours touch the diamond boundary. The sharp corners of the diamond make it likely that some coefficients will be exactly zero, creating sparse solutions.

## 3.4.3 Lasso vs Ridge: Geometric Comparison

### Constrained Optimization Formulation

Both lasso and ridge can be formulated as constrained optimization problems:

**Lasso:**
$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_1 \leq t $$

**Ridge:**
$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_2^2 \leq t $$

**Intuition**: Both methods limit the total "importance" of ingredients, but they measure this importance differently:
- **Lasso**: Uses the sum of absolute values (L1 norm) - like counting the total "absolute importance"
- **Ridge**: Uses the sum of squares (L2 norm) - like measuring the total "squared importance"

### Geometric Interpretation

The constraint regions have different shapes:

1. **L1 ball (Lasso)**: Diamond-shaped in 2D, octahedron in 3D - like a diamond with sharp corners
2. **L2 ball (Ridge)**: Circular in 2D, spherical in 3D - like a smooth circle

The solution occurs where the contours of the RSS (ellipsoids) touch the constraint boundary.

**Intuition**: The sharp corners of the diamond make it likely that the solution will occur at a corner, where some coefficients are exactly zero. The smooth circle of ridge regression makes corner solutions unlikely, so all coefficients are typically non-zero.

### Key Differences

| Property | Lasso | Ridge |
|----------|-------|-------|
| Constraint shape | Diamond (L1 ball) | Circle (L2 ball) |
| Corner solutions | Yes (sparse) | No (dense) |
| Variable selection | Automatic | Manual |
| Coefficient shrinkage | Non-linear | Linear |
| Multicollinearity | Groups one variable | Groups all variables |

**Intuition**: This table shows that lasso is like a "selective chef" who can eliminate ingredients entirely, while ridge is like a "moderate chef" who reduces all ingredients but keeps them all. When ingredients are similar (multicollinearity), lasso picks one while ridge reduces all of them.

### Thresholding Mechanisms

1. **Hard thresholding (Subset selection)**: Coefficients are either kept at OLS value or set to zero - like a strict chef who either uses the full amount or nothing
2. **Soft thresholding (Lasso)**: Coefficients are shrunk toward zero, with some set to exactly zero - like a smart chef who reduces amounts and eliminates unimportant ingredients
3. **Linear shrinkage (Ridge)**: All coefficients are shrunk proportionally, rarely reaching zero - like a moderate chef who reduces all ingredients by the same proportion

**Intuition**: Lasso provides a middle ground between the extremes of subset selection (all or nothing) and ridge regression (proportional reduction). It's like having a sophisticated cooking technique that can both reduce and eliminate ingredients as needed.

## 3.4.4 Coordinate Descent Algorithm

### Algorithm Overview

For general design matrices, lasso doesn't have a closed-form solution. The coordinate descent algorithm updates one coefficient at a time while keeping others fixed.

**Intuition**: This is like adjusting one ingredient at a time while keeping all other ingredients fixed. You cycle through all ingredients, optimizing each one in turn, until the recipe is as good as it can be.

### Algorithm Steps

1. **Initialize**: $`\boldsymbol{\beta}^{(0)} = \mathbf{0}`$ - like starting with no ingredients
2. **For iteration $`k = 1, 2, \ldots`$**:
   - For $`j = 1, 2, \ldots, p`$:
     - Compute partial residual: $`r_j = \mathbf{y} - \sum_{l \neq j} \mathbf{x}_l \beta_l^{(k)}`$ - like calculating how the dish tastes without ingredient j
     - Compute univariate OLS: $`\tilde{\beta}_j = \mathbf{x}_j^T r_j / \|\mathbf{x}_j\|^2`$ - like finding the best amount of ingredient j given the current taste
     - Apply soft-thresholding: $`\beta_j^{(k+1)} = S_{\lambda/(2\|\mathbf{x}_j\|^2)}(\tilde{\beta}_j)`$ - like deciding whether to use ingredient j and how much
3. **Convergence**: Stop when coefficients change by less than tolerance - like stopping when further adjustments don't improve the recipe

**Intuition**: This algorithm is like an iterative cooking process where you adjust one ingredient at a time, each time making the recipe a little better, until you can't improve it anymore.

### Mathematical Derivation

For coordinate $`j`$, the objective function becomes:

$$ \min_{\beta_j} \|\mathbf{r}_j - \mathbf{x}_j\beta_j\|^2_2 + \lambda|\beta_j| $$

where $`\mathbf{r}_j = \mathbf{y} - \sum_{l \neq j} \mathbf{x}_l\beta_l`$ is the partial residual.

The solution is:

$$ \beta_j = S_{\lambda/(2\|\mathbf{x}_j\|^2)}\left(\frac{\mathbf{x}_j^T\mathbf{r}_j}{\|\mathbf{x}_j\|^2}\right) $$

**Intuition**: For each ingredient, we're solving a one-dimensional lasso problem. The partial residual represents how the dish tastes without this ingredient, and we're deciding whether and how much of this ingredient to add back.

### Convergence Properties

1. **Monotonicity**: The objective function decreases at each iteration - like the recipe getting better with each adjustment
2. **Convergence**: The algorithm converges to a global minimum - like eventually finding the best possible recipe
3. **Finite convergence**: For some problems, convergence occurs in finitely many steps - like sometimes finding the optimal recipe in a finite number of adjustments

**Intuition**: These properties ensure that the algorithm will eventually find the best lasso solution, just like an iterative cooking process will eventually produce the best recipe.

## 3.4.5 Uniqueness and Solution Properties

### Uniqueness Conditions

The lasso solution is unique when:

1. **Full-rank design matrix**: $`\text{rank}(\mathbf{X}) = p`$ - like having independent ingredients
2. **Sufficient observations**: $`n \geq p`$ - like having enough taste tests
3. **Strict convexity**: The objective function is strictly convex - like having a well-behaved optimization landscape

**Intuition**: These conditions ensure that there's only one best lasso recipe. When ingredients are independent and we have enough data, the optimization problem has a unique solution.

### Non-uniqueness Scenarios

When $`p > n`$ or $`\mathbf{X}`$ is not full-rank:

1. **Multiple solutions**: Different coefficient vectors may give the same fitted values - like different recipes producing the same taste
2. **Unique fitted values**: The predicted values $`\hat{\mathbf{y}}`$ are always unique - like the predicted taste being unique even if the recipe isn't
3. **Unique L1 norm**: The L1 norm of the solution is always unique - like the total "importance" of ingredients being unique

**Intuition**: When we have more ingredients than taste tests, or when ingredients are similar, there might be multiple ways to achieve the same result. However, the predicted taste and the total ingredient importance are always unique.

### Solution Characterization

For any lasso solution $`\hat{\boldsymbol{\beta}}`$:

1. **Optimality conditions**: Must satisfy the subgradient conditions - like satisfying the balance between taste and penalty
2. **Support recovery**: The set of non-zero coefficients is well-defined - like knowing which ingredients are actually used
3. **Sign consistency**: The signs of non-zero coefficients are consistent across solutions - like knowing whether each ingredient has a positive or negative effect

**Intuition**: Even when there are multiple solutions, they all share important properties: they use the same set of ingredients and those ingredients have consistent effects on taste.

## 3.4.6 Practical Implementation

### Python Implementation

See the complete Python implementation in [`code/lasso_regression_detailed.py`](code/lasso_regression_detailed.py) which demonstrates comprehensive lasso regression with coordinate descent, soft thresholding, and variable selection analysis.

### R Implementation

See the complete R implementation in [`code/lasso_regression_detailed.R`](code/lasso_regression_detailed.R) which demonstrates comprehensive lasso regression with coordinate descent and variable selection using the glmnet package.

## 3.4.7 Advanced Topics

### Elastic Net

Elastic net combines L1 and L2 penalties:

$$ \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2_2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|^2_2 $$

This provides a compromise between lasso and ridge, offering both variable selection and group selection.

**Intuition**: Elastic net is like a "balanced chef" who can both eliminate ingredients (like lasso) and moderate ingredient amounts (like ridge). This is particularly useful when you have similar ingredients - it can eliminate some while keeping others in moderate amounts.

### Group Lasso

For grouped variables, group lasso uses:

$$ P(\boldsymbol{\beta}) = \sum_{g=1}^G \|\boldsymbol{\beta}_g\|_2 $$

where $`\boldsymbol{\beta}_g`$ represents coefficients for group $`g`$.

**Intuition**: Group lasso is like a chef who works with ingredient categories. Instead of deciding on individual ingredients, you decide on entire categories (like "spices" or "vegetables"). You either use all ingredients in a category or none of them.

### Adaptive Lasso

Adaptive lasso uses weighted L1 penalty:

$$ P(\boldsymbol{\beta}) = \sum_{j=1}^p w_j |\beta_j| $$

where weights $`w_j`$ are typically based on initial OLS estimates.

**Intuition**: Adaptive lasso is like a smart chef who gives different penalties to different ingredients based on their initial importance. Important ingredients get lighter penalties, while unimportant ingredients get heavier penalties, making the selection more intelligent.

### Lasso for Classification

Lasso can be extended to classification using logistic regression with L1 penalty:

$$ \min_{\boldsymbol{\beta}} \sum_{i=1}^n \log(1 + e^{-y_i\mathbf{x}_i^T\boldsymbol{\beta}}) + \lambda\|\boldsymbol{\beta}\|_1 $$

**Intuition**: This is like using the same "selective ingredient" principle for classification problems. Instead of predicting continuous taste scores, you're predicting categories (like "spicy" vs "mild"), but you still want to use only the most important ingredients.

## 3.4.8 Model Selection and Validation

### Choosing the Regularization Parameter

1. **Cross-validation**: Most common approach - like testing the recipe in different kitchens
2. **Information criteria**: AIC, BIC with effective degrees of freedom - like using recipe scoring systems that account for complexity
3. **Stability selection**: Assess variable selection stability - like checking if the same ingredients are selected across different taste tests
4. **Bayesian methods**: Empirical Bayes, hierarchical models - like using sophisticated statistical methods

**Intuition**: Choosing the right regularization parameter is like deciding how strict to be about ingredient selection. Too strict and you might eliminate important ingredients; too lenient and you might keep unnecessary ones.

### Variable Selection Stability

Lasso's variable selection can be unstable. Stability selection addresses this by:

1. Running lasso on multiple subsamples - like testing the recipe with different sets of taste tests
2. Computing selection frequencies - like counting how often each ingredient is selected
3. Selecting variables with high selection probability - like keeping only ingredients that are consistently important

**Intuition**: This is like a chef who tests their recipe multiple times to see which ingredients are consistently important. Ingredients that are selected across many different tests are more likely to be truly important.

### Model Diagnostics

1. **Residual analysis**: Check for model adequacy - like tasting the dish to see if the recipe worked
2. **Influence diagnostics**: Identify influential observations - like finding which taste tests had unusual results
3. **Variable importance**: Assess coefficient stability - like checking if ingredient importance changes with different data
4. **Prediction intervals**: Quantify uncertainty - like giving a range of possible taste scores

**Intuition**: These diagnostics help you understand whether your lasso regression recipe is working well and where it might be failing. They're like quality control checks for your cooking.

## 3.4.9 Practical Guidelines

### When to Use Lasso

**Use lasso when:**
- You want automatic variable selection - like wanting the recipe to automatically choose which ingredients to use
- The true model is sparse - like knowing that only a few ingredients really matter
- Interpretability is important - like needing to understand exactly which ingredients contribute to the taste
- You have many predictors relative to sample size - like having many ingredients but few taste tests
- You want a sparse model - like wanting a simple recipe with few ingredients

**Consider alternatives when:**
- Predictors are highly correlated (use elastic net) - like having very similar ingredients that are hard to distinguish
- You want to keep all variables (use ridge) - like wanting to use all available ingredients
- You have grouped variables (use group lasso) - like having ingredient categories that should be treated together
- The true model is dense - like knowing that many ingredients are important

**Intuition**: Lasso is like a "selective chef" who prefers simple recipes with only essential ingredients. It's good when you want interpretability and believe that only a few ingredients really matter.

### Best Practices

1. **Always standardize predictors** before applying lasso - like measuring all ingredients on the same scale
2. **Use cross-validation** to select the regularization parameter - like testing the recipe in different kitchens
3. **Check variable selection stability** across different samples - like ensuring the same ingredients are selected consistently
4. **Validate on a holdout set** to assess generalization - like testing the recipe on completely new taste tests
5. **Consider the bias-variance tradeoff** when interpreting results - like understanding the trade-off between simplicity and accuracy

**Intuition**: These practices ensure that your lasso regression approach is robust and reliable, just like following proven cooking techniques.

### Common Pitfalls

1. **Not standardizing data**: Can lead to inconsistent results - like having unfair rules for different ingredients
2. **Over-regularization**: Choosing λ too large can remove important variables - like being too strict and eliminating essential ingredients
3. **Under-regularization**: Choosing λ too small may not address overfitting - like not being strict enough
4. **Ignoring multicollinearity**: Can affect variable selection - like not accounting for similar ingredients
5. **Not validating variable selection**: Can lead to spurious findings - like not checking if selected ingredients are truly important

**Intuition**: These pitfalls are like common cooking mistakes that can ruin an otherwise good recipe. Being aware of them helps you avoid them.

## Summary

Lasso regression is a powerful regularization technique that combines variable selection with coefficient shrinkage. Its key features are:

1. **Sparsity**: Can produce exactly zero coefficients through soft thresholding - like completely eliminating unimportant ingredients
2. **Variable selection**: Automatic feature selection - like automatically choosing which ingredients to use
3. **Convex optimization**: Computationally tractable - like having a reliable cooking method
4. **Geometric interpretation**: L1 constraint leads to corner solutions - like the diamond-shaped constraint creating sparse solutions
5. **Coordinate descent**: Efficient algorithm for general design matrices - like an iterative cooking process

Lasso is particularly valuable in high-dimensional settings where sparsity is expected, providing both prediction accuracy and interpretability through automatic variable selection.

**Intuition**: Lasso regression is like having a "selective chef" who can both reduce ingredient amounts and completely eliminate ingredients that aren't essential. It's particularly useful when you have many ingredients but believe only a few are truly important. While it might not create the most complex recipe, it creates one that's simple, interpretable, and often more reliable across different kitchens.

---

**Navigation:**
- **Next Topic:** [Discussion and Comparison](05_discussion.md) - Comparing variable selection and regularization methods
- **Previous Topic:** [Ridge Regression](03_ridge_regression.md) - L2 regularization and coefficient shrinkage
