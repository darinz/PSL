# 3.1. Subset Selection

Variable selection, also known as feature selection or subset selection, is a fundamental technique in statistical modeling that addresses the challenge of identifying the most relevant predictors from a potentially large set of candidate variables. This process is crucial for building interpretable, efficient, and generalizable models.

**Intuitive Understanding**: Subset selection is like being a chef who has access to a huge pantry full of ingredients but needs to choose only the most essential ones for a particular dish. You want to create a recipe that's not too simple (missing important flavors) but not too complex (overwhelming and hard to follow). The goal is to find the perfect subset of ingredients that makes the dish taste great without unnecessary complexity.

## 3.1.1. Why Subset Selection: The Curse of Dimensionality and Model Complexity

In modern statistical applications, there is often a vast array of potential predictors. Sometimes, the number of predictors $`p`$ can exceed the sample size $`n`$, leading to what is known as the **curse of dimensionality**. In certain applications, the primary objective is to pinpoint a subset of these predictors that have the most significant relevance to the response variable. For such tasks, variable selection becomes indispensable.

**Intuitive Understanding**: The curse of dimensionality is like trying to find a specific spice in a massive warehouse. As the warehouse gets bigger (more dimensions), it becomes exponentially harder to find what you're looking for. In cooking, this is like having so many ingredients that you can't possibly test all combinations to find the best recipe.

### The Fundamental Question: More Variables = Better Predictions?

However, if our sole aim is to achieve accurate predictions without being concerned about the relevance of predictors in our regression model to $`Y`$, do we still need variable selection? Can adding more variables always lead to better predictions?

**Intuitive Understanding**: This is like asking "If I have access to every possible ingredient, should I use them all in my recipe?" The answer isn't always yes - sometimes more ingredients make a dish worse, not better. Think of it like adding too many spices to a dish - it can become overwhelming and lose its character.

This question touches on the fundamental trade-off between **model complexity** and **generalization ability**. To understand this trade-off, we need to explore the theoretical foundations of training and test errors in linear regression.

**Intuition**: Model complexity is like recipe complexity - a simple recipe might be too basic, but a very complex recipe might be hard to follow and might not work well in different kitchens. We need to find the right balance.

### The Bias-Variance Trade-off and Model Complexity

![Bias-Variance Trade-off and Model Complexity](../_images/w3_fig_3_11.png)

*Figure: The relationship between model complexity, training error, and test error. Illustrates the bias-variance trade-off central to variable selection.*

**Intuition**: This graph shows the classic "Goldilocks problem" - too few ingredients (high bias, underfitting) makes the dish too simple, while too many ingredients (high variance, overfitting) makes it too complex and unreliable. The sweet spot is in the middle.

### Mathematical Foundation: Training vs. Test Error

Let's embark on a theoretical exploration of the training and test errors in a linear regression model. Consider a training dataset $`\{(\mathbf{x}_i, y_i)\}_{i=1}^n`$ of size $`n`$. Using this data, we can fit a linear regression model, yielding a least squares estimate $`\hat{\boldsymbol{\beta}}`$.

**Intuition**: Think of the training data as your "cooking practice sessions" where you try different recipes and see how they taste. The test data is like "real cooking" where you serve the dish to actual diners.

**Training Error Definition:**
$$ \text{Train Err} = \|\mathbf{y} - \mathbf{X} \hat{\boldsymbol{\beta}}\|^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2 $$

where $`\hat{\boldsymbol{\beta}} \in \mathbb{R}^p`$ is the least squares estimate of the regression parameter.

**Intuition**: Training error is like measuring how well your recipe works in your own kitchen with your own ingredients. It's the "practice score" - how close your predictions are to what actually happened during practice.

**Test Error Definition:**
Now, consider a separate test dataset $`\{(\mathbf{x}_i, y_i^*)\}_{i=1}^n`$ collected at the same locations $`\mathbf{x}_i`$'s. The test error is:

$$ \text{Test Err} = \|\mathbf{y}^* - \mathbf{X} \hat{\boldsymbol{\beta}}\|^2 = \sum_{i=1}^n (y_i^* - \mathbf{x}_i^T \hat{\boldsymbol{\beta}})^2 $$

**Intuition**: Test error is like measuring how well your recipe works when someone else cooks it in a different kitchen. It's the "real-world score" - how well your recipe generalizes to new situations.

### Key Assumptions and Relationships

It is crucial to note that while both $`y_i`$ and $`y_i^*`$ are random and independent of each other, they are assumed to follow the same distribution with:
- **Mean**: $`f(\mathbf{x}_i)`$ (the true underlying function) - like the "true taste" that a perfect recipe would achieve
- **Variance**: $`\sigma^2`$ (constant error variance) - like the random variations in cooking that we can't control

The estimate $`\hat{\boldsymbol{\beta}}`$ is also random, with its randomness originating from the training data $`\mathbf{y}`$. This means that:
- $`\mathbf{y}`$ and $`\hat{\boldsymbol{\beta}}`$ are **correlated** (both depend on the same training data) - like your recipe being based on your practice sessions
- $`\mathbf{y}^*`$ and $`\hat{\boldsymbol{\beta}}`$ are **independent** (test data is independent of training data) - like the real cooking being independent of your practice

**Intuition**: This independence is crucial - it's like saying that how well your recipe works in a real restaurant doesn't depend on how well it worked in your practice kitchen, but both are trying to achieve the same "true taste."

### Expected Error Decomposition

If we break down the expectations of both errors, they can be segmented into three fundamental components:

$$ \begin{aligned}
\mathbb{E}[\text{Train Err}] &= \text{(Unavoidable Error)} - p\sigma^2 + \text{Bias}^2 \\
\mathbb{E}[\text{Test Err}] &= \text{(Unavoidable Error)} + p\sigma^2 + \text{Bias}^2
\end{aligned} $$

**Intuition**: This decomposition shows that both training and test errors have the same unavoidable error and bias, but they differ in how they handle the "dimensional error" - the error that comes from having to estimate the recipe from limited practice sessions.

#### Component Analysis:

**1. Unavoidable Error:**
$$ \text{Unavoidable Error} = n\sigma^2 $$

This error persists even if we knew the true function $`f`$. It represents the irreducible error due to the inherent noise in the data.

**Intuition**: This is like the random variations in cooking that you can't control - the quality of ingredients varies, the stove temperature fluctuates, etc. Even a perfect recipe would have some variation in results.

**2. Bias Term:**
$$ \text{Bias}^2 = \sum_{i=1}^n [f(\mathbf{x}_i) - \mathbf{x}_i^T \boldsymbol{\beta}]^2 $$

This emerges if the true function $`f`$ deviates from linearity or if we include an incomplete set of predictors in our model.

**Intuition**: Bias is like the systematic difference between your simplified recipe and the true perfect recipe. It's the error that comes from making simplifying assumptions - like assuming a linear relationship when the true relationship is more complex.

**3. Dimensional Error Term:**
$$ \text{Dimensional Error} = p\sigma^2 $$

This is where things get intriguing. Its sign changes between training and test errors:
- **Positive sign in test error**: Arises because we rely on the estimated $`\hat{\boldsymbol{\beta}}`$ instead of the true $`\boldsymbol{\beta}`$ - like using your estimated recipe instead of the perfect recipe
- **Negative sign in training error**: Can be attributed to the positive correlation between $`\hat{\boldsymbol{\beta}}`$ and $`\mathbf{y}`$ - like your recipe being optimized for your practice data

**Intuition**: The dimensional error is like the "optimization bonus" you get in practice (negative sign) versus the "estimation penalty" you pay in real cooking (positive sign). More ingredients (higher p) means bigger effects in both directions.

### Mathematical Derivation of Error Decomposition

Let's derive this decomposition step by step:

**For Training Error:**
$$ \begin{aligned}
\mathbb{E}[\text{Train Err}] &= \mathbb{E}[\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{H}\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y} - \mathbf{H}\mathbf{y}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}\|^2 - 2\mathbf{y}^T\mathbf{H}\mathbf{y} + \mathbf{y}^T\mathbf{H}^T\mathbf{H}\mathbf{y}] \\
&= \mathbb{E}[\|\mathbf{y}\|^2 - \mathbf{y}^T\mathbf{H}\mathbf{y}] \\
&= n\sigma^2 + \|f(\mathbf{X})\|^2 - (n\sigma^2 + \|f(\mathbf{X})\|^2 - p\sigma^2) \\
&= n\sigma^2 - p\sigma^2 + \text{Bias}^2
\end{aligned} $$

where $`\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T`$ is the hat matrix.

**Intuition**: This derivation shows that in practice, you get a "bonus" of $`p\sigma^2`$ because your recipe is optimized for your practice data. The hat matrix $`\mathbf{H}`$ is like the "recipe optimization machine" that finds the best fit for your practice sessions.

**For Test Error:**
$$ \begin{aligned}
\mathbb{E}[\text{Test Err}] &= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\boldsymbol{\beta} + \mathbf{X}\boldsymbol{\beta} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2] \\
&= \mathbb{E}[\|\mathbf{y}^* - \mathbf{X}\boldsymbol{\beta}\|^2] + \mathbb{E}[\|\mathbf{X}(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})\|^2] \\
&= n\sigma^2 + p\sigma^2 + \text{Bias}^2
\end{aligned} $$

**Intuition**: In real cooking, you pay a penalty of $`p\sigma^2`$ because you're using your estimated recipe instead of the perfect recipe. The more ingredients you have (higher p), the bigger this penalty becomes.

### Practical Implications

This theoretical framework reveals several important insights:

1. **The Bias-Variance Trade-off**: Adding more predictors can reduce bias but increases variance (dimensional error) - like adding more ingredients can make the recipe more accurate but also more sensitive to variations

2. **Overfitting Risk**: The training error systematically underestimates the test error by $`2p\sigma^2`$ - like your practice scores being overly optimistic about how well you'll do in real cooking

3. **Optimal Model Complexity**: There exists an optimal number of predictors that minimizes test error - like finding the perfect number of ingredients for a dish

4. **Variable Selection Necessity**: Even for pure prediction, variable selection is crucial to avoid overfitting - like choosing only the most important ingredients even if you have access to everything

**Intuition**: These insights tell us that variable selection isn't just about interpretability - it's about finding the right level of complexity that works well in the real world, not just in practice.

### Python Example: Demonstrating the Error Decomposition

See the complete implementation in [`code/error_decomposition.py`](code/error_decomposition.py) which demonstrates error decomposition in variable selection with comprehensive visualization and analysis.

It is crucial to note that whole both y_i and y-star_i are random and independent of each other. They are assumed to follow the same distribution which has a mean of f(x_i) and variance sigma-square. Another random term is beta-hat, whose randomness originates from the data y. This means that y and beta-hat are correlated (therefore, both colored in blue) but y-star (colored in red) and beta-hat are independent.

If we break down the expectations of both errors, they can be segmented into three parts ([see the derivation here](https://liangfgithub.github.io/Notes/lec_W3_VariableSelection_appendix.pdf)):

$$ \begin{split}\mathbb{E} [ \text{Train Err}  ] & = \text{(Unavoidable Err)} - p \sigma^2  + \text{Bias} \\
\mathbb{E} [ \text{Test Err}  ] &= \text{(Unavoidable Err)} + p \sigma^2  + \text{Bias}\end{split} $$

1. **Unavoidable Error**: This error persists even if we knew true function f. When the error terms are assumed to be independent with mean zero and variance sigma-square, the unavoidable error is equal to n times sigma-square.

**Intuition**: This is like the random cooking variations that even a perfect recipe can't eliminate - ingredient quality varies, cooking conditions change, etc.

2. **Bias**: This emerges if the true function f deviates from linearity or if, for instance, it involves three predictors, but we include only two in our model.

**Intuition**: This is like the systematic error that comes from using a simplified recipe when the true relationship is more complex - like assuming a linear relationship between salt and taste when it's actually quadratic.

3. **Dimensional Error term**, $`p \sigma^2`$: This is where things get intriguing. Its sign changes between training and test errors. The positive sign in the test error arises because of our reliance on the estimated beta instead of the true beta. The negative sign in the training error can be attributed to the positive correlation between beta-hat and y.

**Intuition**: This dimensional error is the key insight - in practice you get a bonus (negative sign) because your recipe is optimized for your practice data, but in real cooking you pay a penalty (positive sign) because you're using an estimated recipe instead of the perfect one.

In conclusion, whether our primary objective lies in identifying a subset of relevant predictors or merely in enhancing prediction accuracy, it becomes evident that variable selection plays a crucial role.

**Intuition**: Variable selection is essential not just for interpretability, but for creating recipes that work well in the real world, not just in practice. It's about finding the right balance between simplicity and accuracy.

## 3.1.2. Selection Criteria: Balancing Fit and Complexity

How do we determine which variables to retain and which to discard? This is one of the most fundamental challenges in statistical modeling, requiring a careful balance between model fit and complexity.

**Intuitive Understanding**: This is like deciding which ingredients to keep in your recipe. You want enough ingredients to make the dish taste good, but not so many that it becomes overwhelming or hard to follow. The challenge is finding the right scoring system to balance taste (fit) with simplicity (complexity).

### The P-Value Pitfall

One might initially think of using p-values obtained from a linear regression model that includes all variables. In the resulting summary table, each variable is assigned a p-value. A common practice might be to use these p-values, setting a threshold (like 5%), and dropping variables with values exceeding this. But is this approach optimal?

**Intuition**: This is like looking at each ingredient individually and asking "Does this ingredient matter?" But the problem is that the importance of one ingredient can change depending on what other ingredients you're using. Salt might seem unimportant if you're already using soy sauce, but very important if you're not.

**The Fundamental Problem:**
The crux of the issue is that a variable's p-value is contingent upon the other variables included in the model. Recall that the p-value for a variable assesses its **conditional** contribution in the presence of other variables in the model. If we remove any variable, the entire set of p-values could shift dramatically.

**Intuition**: This is like the "ingredient interaction problem" - the importance of salt depends on whether you're using soy sauce, the importance of pepper depends on whether you're using chili, etc. You can't just test ingredients in isolation.

**Mathematical Illustration:**
Consider a model with two correlated predictors $`X_1`$ and $`X_2`$:
$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon $$

The p-value for $`\beta_1`$ depends on whether $`X_2`$ is in the model:
- **With $`X_2`$**: Tests $`H_0: \beta_1 = 0`$ conditional on $`X_2`$ - like asking "Does salt matter when I'm already using soy sauce?"
- **Without $`X_2`$**: Tests $`H_0: \beta_1 = 0`$ in the presence of omitted variable bias - like asking "Does salt matter?" when salt and soy sauce are always used together

Thus, simply using a snapshot of p-values from a full model is not recommended.

**Intuition**: This shows why stepwise selection based on p-values can be misleading - the p-values keep changing as you add or remove variables, like a moving target.

### Model Scoring Approach

Instead of using p-values, we can assign a score to each model and then utilize an algorithm to determine the best one. Here, 'model' refers to a linear regression model containing a specific subset of variables.

**Intuition**: This is like giving each possible recipe a score that balances taste with simplicity, then systematically searching for the recipe with the best score.

**The Combinatorial Challenge:**
Imagine we have 10 non-intercept predictors. Excluding the intercept, which is always present, our subset of variables will be a combination of these 10 predictors. The potential models can be indexed using binary vectors, with a '1' indicating the presence of a variable and '0' its absence.

**Number of Possible Models:**
$$ \text{Number of models} = 2^p $$

For $`p = 10`$, this gives $`2^{10} = 1024`$ possible models. Even for just 10 predictors, the model possibilities exceed a thousand, underscoring the significance of efficient search algorithms.

**Intuition**: This is like having 10 ingredients and needing to test every possible combination of them. With 10 ingredients, you have over 1000 possible recipes to evaluate. This is why we need smart search strategies rather than trying everything.

### Mathematical Framework for Model Scoring

The score for model selection typically comprises two components:

$$ \text{Model Score} = \text{Goodness of Fit} + \text{Complexity Penalty} $$

**Intuition**: This is like a recipe scoring system that balances taste (goodness of fit) with simplicity (complexity penalty). A good recipe should taste good but not be too complicated.

**1. Goodness of Fit Measure:**
Often an increasing function of the residual sum of squares (RSS):
$$ \text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2 $$

**Intuition**: RSS measures how well the recipe predicts the actual taste. Lower RSS means better predictions - like having smaller differences between predicted and actual taste scores.

**2. Complexity Penalty Term:**
Often an increasing function of $`p`$, the number of non-intercept variables. This penalty discourages overfitting by penalizing model complexity.

**Intuition**: The complexity penalty is like a "simplicity bonus" - simpler recipes get rewarded, more complex recipes get penalized. This prevents overfitting by encouraging parsimony.

### Popular Model Selection Criteria

#### 1. Mallow's $`C_p`$ Statistic

**Definition:**
$$ C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2_{\text{full}}} - n + 2p $$

**Alternative Form:**
$$ C_p = \text{RSS}_p + 2\hat{\sigma}^2_{\text{full}} \times p $$

**Interpretation:**
- $`\text{RSS}_p`$ = Residual sum of squares for the model with $`p`$ predictors - like how well the recipe predicts taste
- $`\hat{\sigma}^2_{\text{full}}`$ = Estimated error variance from the full model - like the baseline noise level in cooking
- The model with $`C_p \approx p`$ is considered optimal - like finding the recipe where the score equals the number of ingredients

**Theoretical Foundation:**
Mallow's $`C_p`$ estimates the expected prediction error:
$$ \mathbb{E}[C_p] \approx \mathbb{E}[\text{Test Error}] $$

**Intuition**: Mallow's Cp is like a "prediction accuracy estimator" - it tries to estimate how well your recipe will work in real cooking, not just in practice.

#### 2. Akaike Information Criterion (AIC)

**Definition:**
$$ \text{AIC} = -2\log L(\hat{\boldsymbol{\beta}}) + 2p $$

**For Linear Regression with Normal Errors:**
$$ \text{AIC} = n\log(\text{RSS}/n) + 2p + \text{constant} $$

**Interpretation:**
- Balances model fit (log-likelihood) with complexity (number of parameters) - like balancing taste with recipe simplicity
- Penalty of 2 per additional parameter - like a fixed penalty for each additional ingredient
- AIC estimates the relative Kullback-Leibler divergence - like measuring how different your recipe is from the true perfect recipe

**Intuition**: AIC is like a "lenient recipe scorer" - it's willing to add ingredients if they improve the taste, with a moderate penalty for complexity.

#### 3. Bayesian Information Criterion (BIC)

**Definition:**
$$ \text{BIC} = -2\log L(\hat{\boldsymbol{\beta}}) + (\log n)p $$

**For Linear Regression with Normal Errors:**
$$ \text{BIC} = n\log(\text{RSS}/n) + (\log n)p + \text{constant} $$

**Interpretation:**
- Similar to AIC but with stronger penalty for complexity - like a stricter recipe scorer
- Penalty of $`\log n`$ per additional parameter - like a penalty that grows with the number of taste tests
- BIC estimates the posterior probability of the model - like the probability that this is the true recipe given the data

**Intuition**: BIC is like a "strict recipe scorer" - it's more conservative about adding ingredients and prefers simpler recipes, especially when you have lots of data.

### Mathematical Comparison of Criteria

**Penalty Comparison:**
$$ \begin{aligned}
\text{Mallow's } C_p &: \text{Constant penalty} \\
\text{AIC} &: 2p \text{ penalty} \\
\text{BIC} &: (\log n)p \text{ penalty}
\end{aligned} $$

**Asymptotic Behavior:**
- **AIC**: Penalty remains constant as $`n \to \infty`$ - like the penalty staying the same regardless of how many taste tests you do
- **BIC**: Penalty grows with $`\log n`$ as $`n \to \infty`$ - like the penalty getting stricter as you get more data
- **Mallow's $`C_p`$**: Constant penalty, closely related to AIC - like having a fixed penalty regardless of data size

**Intuition**: This shows the different philosophies - AIC is more lenient and focuses on prediction, while BIC is stricter and focuses on finding the true model. Mallow's Cp is similar to AIC but more directly focused on prediction accuracy.

### Comprehensive Python Example: Model Selection Criteria

See the complete implementation in [`code/model_selection_criteria.py`](code/model_selection_criteria.py) which demonstrates model selection criteria (AIC, BIC, Mallow's Cp) with comprehensive analysis and visualization.

AIC and BIC are versatile and can be applied to any statistical model. Although one might default to AIC and BIC, it's useful to consider Mallow's Cp, which aligns well with our theoretical understanding of training and test errors. The intent is always to minimize the test error, making Mallow's Cp a robust choice for model selection, especially in the context of linear regression.

**Intuition**: While AIC and BIC are general-purpose recipe scoring systems, Mallow's Cp is specifically designed for linear regression and directly estimates prediction accuracy. It's like having a specialized scoring system for a particular type of cooking.

## 3.1.3. AIC vs BIC: Philosophical and Practical Differences

AIC and BIC are the two most widely used model selection criteria, but they embody different philosophical approaches and practical considerations. Understanding their differences is crucial for making informed decisions in variable selection.

**Intuitive Understanding**: AIC and BIC are like two different restaurant critics with different philosophies. AIC is like a critic who focuses on how well the dish tastes and is willing to forgive complexity if it improves the flavor. BIC is like a critic who values simplicity and elegance, preferring dishes that achieve great taste with minimal ingredients.

### Mathematical Comparison

**Key Differences**: Both AIC and BIC serve as model selection criteria, with the primary difference being in their penalty terms. The coefficients, specifically the '2' in AIC and $`\log(n)`$ in BIC, can be thought of as the "cost" associated with adding an additional predictor to the model.

**Mathematical Formulation:**
$$ \begin{aligned}
\text{AIC} &= -2\log L(\hat{\boldsymbol{\beta}}) + 2p \\
\text{BIC} &= -2\log L(\hat{\boldsymbol{\beta}}) + \log(n)p
\end{aligned} $$

**Penalty Comparison:**
- **AIC Penalty**: $`2p`$ (constant per parameter) - like a fixed cost for each additional ingredient
- **BIC Penalty**: $`\log(n)p`$ (grows with sample size) - like a cost that increases as you get more data

**Intuition**: AIC charges a flat rate for each ingredient, while BIC charges more for each ingredient when you have more data to work with.

### Asymptotic Behavior Analysis

**Sample Size Dependence:**
As the sample size $`n`$ grows:
- **AIC**: The cost incurred remains constant at 2 per parameter - like the ingredient cost staying the same regardless of how many taste tests you do
- **BIC**: The cost increases with $`\log(n)`$ per parameter - like the ingredient cost going up as you get more experience

**Mathematical Illustration:**
$$ \begin{aligned}
\text{AIC penalty ratio} &= \frac{2(p+1)}{2p} = 1 + \frac{1}{p} \\
\text{BIC penalty ratio} &= \frac{\log(n)(p+1)}{\log(n)p} = 1 + \frac{1}{p}
\end{aligned} $$

While the ratios are mathematically identical, the absolute penalties differ dramatically:
- For $`n = 100`$: BIC penalty = $`\log(100) \approx 4.6`$ per parameter - like a moderate penalty
- For $`n = 1000`$: BIC penalty = $`\log(1000) \approx 6.9`$ per parameter - like a higher penalty
- For $`n = 10000`$: BIC penalty = $`\log(10000) \approx 9.2`$ per parameter - like a much higher penalty

**Intuition**: This shows that BIC becomes increasingly strict about adding ingredients as you get more data, while AIC maintains the same level of strictness.

### Practical Implications

**Model Selection Behavior:**
Given the distinct penalties, it's common to see AIC and BIC favor different models when applied to the same dataset. Generally:
- **AIC tends to select larger models** compared to BIC - like preferring recipes with more ingredients
- **BIC tends to select more parsimonious models** with fewer predictors - like preferring simpler recipes

**Mallow's Cp Relationship:**
Mallow's $`C_p`$ aligns closely with AIC because its penalties are constant and don't hinge on $`n`$. However, for many practical purposes, focusing on AIC and BIC might suffice.

**Intuition**: Mallow's Cp is like AIC's close cousin - they both have similar philosophies about recipe complexity.

### Underlying Philosophies

#### AIC: Prediction-Oriented Approach

**Philosophy**: AIC aims to minimize the predictive error. It prioritizes accurate predictions even if it means including variables that might not necessarily be crucial.

**Theoretical Foundation:**
AIC estimates the relative Kullback-Leibler divergence between the true model and the candidate model:
$$ \text{AIC} \approx 2 \times \text{KL divergence} + \text{constant} $$

**Key Characteristics:**
- **Prediction-focused**: Optimizes for out-of-sample prediction accuracy - like focusing on how well the recipe works in different kitchens
- **Less conservative**: Willing to include potentially irrelevant variables - like being willing to add extra spices if they might help
- **Sample size invariant**: Penalty doesn't change with sample size - like maintaining the same ingredient cost regardless of experience

**Intuition**: AIC is like a chef who says "If it might make the dish taste better, let's try it. We can always adjust later."

#### BIC: Model Identification Approach

**Philosophy**: BIC focuses on model parsimony and identifying truly relevant variables. It's more conservative and emphasizes the exclusion of unnecessary predictors.

**Theoretical Foundation:**
BIC approximates the posterior probability of the model (under certain assumptions):
$$ P(\text{Model} | \text{Data}) \propto \exp\left(-\frac{1}{2}\text{BIC}\right) $$

**Key Characteristics:**
- **Model identification-focused**: Aims to find the "true" model - like trying to discover the authentic recipe
- **More conservative**: Stronger penalty for complexity - like being very careful about adding ingredients
- **Sample size dependent**: Penalty increases with sample size - like becoming more strict as you learn more

**Intuition**: BIC is like a chef who says "Let's keep it simple and authentic. Only add ingredients that we're really sure about."

### Error Types in Variable Selection

Two primary errors can arise during variable selection:

1. **Excluding Signals (Type I Error)**: Leaving out variables crucial to $`y`$ - like forgetting to add salt to a dish
2. **Including Noise (Type II Error)**: Incorporating variables that don't significantly impact $`y`$ - like adding unnecessary spices

**Impact Analysis:**
While BIC considers both errors equally significant, their impacts on predictions differ:

**Including Irrelevant Variables:**
- An irrelevant variable included in the model will have its influence diminish as the sample size increases - like an unnecessary spice becoming less noticeable as you cook more
- The estimated coefficient will eventually move towards zero - like the spice amount getting smaller and smaller
- **Impact**: Minimal long-term harm to prediction accuracy - like the dish still tasting good even with the extra spice

**Excluding Relevant Variables:**
- Excluding a relevant variable introduces a bias that persists regardless of sample size - like always forgetting salt, which ruins every dish
- This bias cannot be overcome by increasing sample size - like no amount of practice will fix a recipe that's missing a key ingredient
- **Impact**: Persistent degradation of prediction accuracy - like consistently disappointing results

**Result**: AIC has a lighter penalty for adding new predictors because the cost of including noise is lower than the cost of excluding signals.

**Intuition**: This is like the cooking principle that it's better to have a slightly over-seasoned dish than a bland one. You can always adjust seasoning, but you can't add flavor that was never there.

### Comprehensive Python Example: AIC vs BIC Comparison

See the complete implementation in [`code/aic_bic_comparison.py`](code/aic_bic_comparison.py) which demonstrates AIC vs BIC comparison across different sample sizes with comprehensive analysis and visualization.

### Decision Guidelines

**When to Use AIC:**
- Primary goal is **prediction accuracy** - like focusing on taste over recipe simplicity
- Sample size is **small to moderate** - like when you're still learning to cook
- You're willing to include potentially irrelevant variables - like being experimental with ingredients
- Focus is on **out-of-sample performance** - like how well the recipe works in different kitchens

**When to Use BIC:**
- Primary goal is **model identification** - like trying to find the authentic recipe
- Sample size is **large** - like when you have lots of cooking experience
- You want a **parsimonious model** - like preferring simple, elegant recipes
- Focus is on **understanding variable importance** - like understanding which ingredients really matter

**In Conclusion**: If your primary goal is prediction, lean towards AIC. But if you're keen on selecting a model with only truly relevant features, BIC is your go-to. The choice should align with your research objectives and the nature of your data.

**Intuition**: Choose AIC if you want the best-tasting dish possible, even if it's complex. Choose BIC if you want to understand the essential ingredients and create a simple, authentic recipe.

## 3.1.4. Search Algorithms: Finding the Optimal Model

Once you've chosen your model selection criteria, the next step is to employ a search algorithm to pinpoint the model with the smallest score. This is a critical step that balances computational efficiency with finding the optimal solution.

**Intuitive Understanding**: This is like having a systematic way to search through all possible recipe combinations to find the best one. You need a strategy that's efficient (doesn't take forever) but thorough (doesn't miss the best recipe). Different search strategies have different trade-offs between speed and thoroughness.

### The Computational Challenge

**Problem Complexity:**
For $`p`$ predictors, there are $`2^p`$ possible models to evaluate:
$$ \text{Number of models} = \sum_{k=0}^p \binom{p}{k} = 2^p $$

**Computational Growth:**
- $`p = 10`$: 1,024 models - like having 10 ingredients and testing every possible combination
- $`p = 20`$: 1,048,576 models - like having 20 ingredients and testing over a million combinations
- $`p = 30`$: 1,073,741,824 models - like having 30 ingredients and testing over a billion combinations

This exponential growth makes exhaustive search computationally infeasible for large $`p`$.

**Intuition**: This is like the difference between testing a few ingredient combinations versus trying every possible combination. With many ingredients, exhaustive search becomes impossible.

### Level-wise Search Algorithm (Best Subset Selection)

A popular method is the 'level-wise search algorithm', which works as follows:

**Intuition**: This is like organizing your recipe search by complexity - first try all single-ingredient recipes, then all two-ingredient recipes, and so on. This way you can see how adding ingredients affects the taste.

#### Algorithm Steps:

**1. Grouping Models by Size:**
Imagine there are $`p`$ potential predictors. Models can then be grouped into $`p`$ groups:
- **Group 1**: Models with 1 predictor ($`\binom{p}{1}`$ models) - like all single-ingredient recipes
- **Group 2**: Models with 2 predictors ($`\binom{p}{2}`$ models) - like all two-ingredient recipes
- ...
- **Group p**: Model with all $`p`$ predictors (1 model) - like the recipe with all ingredients

**Intuition**: This is like organizing your cookbook by complexity - simple recipes first, then more complex ones.

**2. Identifying Optimal Models Within Groups:**
Given that models within a group share the same size, their penalties are identical. Therefore, within each group, the model with the smallest residual sum of squares is considered optimal for that group.

**Mathematical Formulation:**
For group $`k`$ (models with $`k`$ predictors):
$$ \text{Best model in group } k = \arg\min_{\substack{S \subseteq \{1,\ldots,p\} \\ |S| = k}} \text{RSS}(S) $$

**Intuition**: Within each complexity level, find the recipe that tastes best (lowest RSS). Since all recipes in a group have the same complexity penalty, you just need to find the tastiest one.

**3. Evaluating Model Scores:**
Next, evaluate the score (residual sum of squares plus the penalty) of these $`p`$ models and select the one with the lowest score:
$$ \text{Optimal model} = \arg\min_{k \in \{1,\ldots,p\}} \text{Score}(\text{Best model in group } k) $$

**Intuition**: Now compare the best recipe from each complexity level, taking into account both taste and simplicity. Choose the one with the best overall score.

#### Computational Considerations:

The computational demands at step 2 can be immense, especially when $`p`$ is large. Typically, this algorithm may not be advisable when $`p > 40`$.

**Complexity Analysis:**
- **Time Complexity**: $`O(2^p)`$ in worst case - like exponential growth in cooking time
- **Space Complexity**: $`O(2^p)`$ for storing all models - like needing exponential storage space
- **Practical Limit**: $`p \leq 40`$ for reasonable computation time - like practical limits on recipe complexity

**Intuition**: This is like having a limit on how many ingredients you can reasonably test - beyond a certain point, it becomes impractical.

### Greedy Algorithms: Efficient Alternatives

For significantly large $`p`$ values, employing greedy algorithms is beneficial. These algorithms search for the optimal model following a specific path, sacrificing global optimality for computational efficiency.

**Intuition**: Greedy algorithms are like cooking strategies that make local decisions - they're not guaranteed to find the absolute best recipe, but they're much faster and often find very good recipes.

#### 1. Forward Selection

**Algorithm:**
1. Start with the null model (only intercept) - like starting with just the basic dish
2. At each step, add the predictor that most improves the model score - like adding the ingredient that improves the taste the most
3. Continue until no further improvement is possible - like stopping when adding more ingredients doesn't help

**Mathematical Formulation:**
$$ \begin{aligned}
S_0 &= \emptyset \\
S_{t+1} &= S_t \cup \{j^*\} \\
\text{where } j^* &= \arg\min_{j \notin S_t} \text{Score}(S_t \cup \{j\})
\end{aligned} $$

**Advantages:**
- Computationally efficient: $`O(p^2)`$ complexity - like quadratic growth instead of exponential
- Works well when true model is sparse - like when only a few ingredients really matter
- Can handle $`p > n`$ scenarios - like when you have more ingredients than taste tests

**Disadvantages:**
- Cannot remove variables once added - like not being able to remove an ingredient once you've added it
- May get stuck in local optima - like finding a good recipe but missing an even better one
- Sensitive to the order of variable addition - like the order of adding ingredients affecting the final result

**Intuition**: Forward selection is like building a recipe one ingredient at a time, always adding what seems best at the moment. It's fast but might miss better combinations.

#### 2. Backward Elimination

**Algorithm:**
1. Start with the full model (all predictors) - like starting with all possible ingredients
2. At each step, remove the predictor whose removal most improves the model score - like removing the ingredient that hurts the taste the least
3. Continue until no further improvement is possible - like stopping when removing more ingredients doesn't help

**Mathematical Formulation:**
$$ \begin{aligned}
S_0 &= \{1, 2, \ldots, p\} \\
S_{t+1} &= S_t \setminus \{j^*\} \\
\text{where } j^* &= \arg\min_{j \in S_t} \text{Score}(S_t \setminus \{j\})
\end{aligned} $$

**Advantages:**
- Computationally efficient: $`O(p^2)`$ complexity - like quadratic growth instead of exponential
- Works well when most variables are relevant - like when most ingredients contribute to the taste
- Can handle multicollinearity better than forward selection - like dealing better with similar ingredients

**Disadvantages:**
- Cannot add variables once removed - like not being able to add back an ingredient once you've removed it
- May get stuck in local optima - like finding a good recipe but missing an even better one
- Requires $`p < n`$ to start - like needing fewer ingredients than taste tests to begin

**Intuition**: Backward elimination is like starting with a complex recipe and simplifying it by removing unnecessary ingredients. It's good when you suspect most ingredients matter.

#### 3. Stepwise Algorithm (Forward-Backward)

**Algorithm:**
This is a blend of backward and forward methods:
1. Start with the full model and move backward - like starting with all ingredients and removing some
2. At each stage, in addition to removing predictors, consider reintroducing ones previously removed - like being able to add back ingredients you removed earlier
3. The process halts when adding or removing predictors no longer improves the score - like stopping when no changes help

**Mathematical Formulation:**
$$ \begin{aligned}
\text{At step } t: \\
\text{Backward step: } S_{t+1} &= S_t \setminus \{j^*\} \text{ if } \text{Score}(S_t \setminus \{j^*\}) < \text{Score}(S_t) \\
\text{Forward step: } S_{t+1} &= S_t \cup \{j^*\} \text{ if } \text{Score}(S_t \cup \{j^*\}) < \text{Score}(S_t)
\end{aligned} $$

**Advantages:**
- More flexible than pure forward or backward - like being able to both add and remove ingredients
- Can escape local optima - like being able to try different combinations
- Often finds better solutions than pure greedy methods - like finding better recipes through more flexible searching

**Disadvantages:**
- More computationally intensive - like taking more time to search
- Still not guaranteed to find global optimum - like still not guaranteed to find the absolute best recipe
- May oscillate between similar models - like switching between very similar recipes

**Intuition**: Stepwise selection is like having a flexible cooking strategy where you can both add and remove ingredients as needed. It's more thorough but takes more time.

### Local vs Global Optimality

The nature of greedy algorithms, given their specific path of search, means they may stop at a locally optimal solution rather than a globally optimal one. However, they're faster and often yield solutions that are practically sufficient.

**Mathematical Illustration:**
Consider a landscape of model scores:
$$ \text{Global optimum} = \min_{S \subseteq \{1,\ldots,p\}} \text{Score}(S) $$

Greedy algorithms find:
$$ \text{Local optimum} = \text{Score}(S_{\text{greedy}}) \geq \text{Score}(S_{\text{global}}) $$

**Intuition**: This is like the difference between finding the best recipe in your neighborhood (local optimum) versus finding the best recipe in the world (global optimum). The local one might be very good and much easier to find.

### Comprehensive Python Example: Search Algorithms

See the complete implementation in [`code/search_algorithms.py`](code/search_algorithms.py) which demonstrates different search algorithms for variable selection (exhaustive, forward, backward, stepwise) with comprehensive analysis and visualization.

## 3.1.5. R/Python Code for Subset Selection

- Rcode: [R_W3_VarSel_SubsetSelection](./R_W3_VarSel_SubsetSelection.R)
- Python: [Python_W3_VarSel_SubsetSelection](./Python_W3_VarSel_SubsetSelection.py)

## 3.1.6. Variable Screening: Handling High-Dimensional Data

Among the three model selection procedures — complete, forward, and backward — stepwise is the most computationally intensive. However, compared to forward and backward methods, stepwise is less prone to prematurely settling on a local optimum. The forward method doesn't allow for the removal of variables once they've been added, even if they become less relevant as other predictors are included. Conversely, the backward method can't reintroduce a predictor that might seem unimportant in the presence of other variables but could be beneficial if certain variables are removed.

**Intuitive Understanding**: This is like the difference between cooking strategies. Forward selection is like adding ingredients one by one and never removing them - even if they become less important as you add other ingredients. Backward elimination is like removing ingredients one by one and never adding them back - even if they would be useful once you remove other ingredients. Stepwise is like being flexible - you can both add and remove ingredients as needed.

### Algorithm Comparison Summary

**Computational Complexity:**
$$ \begin{aligned}
\text{Exhaustive Search} &: O(2^p) \\
\text{Forward Selection} &: O(p^2) \\
\text{Backward Elimination} &: O(p^2) \\
\text{Stepwise Selection} &: O(p^2) \text{ (but more iterations)}
\end{aligned} $$

**Recommendation**: If computational resources allow, we recommend the stepwise approach, beginning with the full model, as it provides the best balance between finding good solutions and avoiding local optima.

**Intuition**: Stepwise selection is like having the most flexible cooking strategy - you can adjust your recipe as you go, adding or removing ingredients based on what works best. It takes more time but gives you the best chance of finding a great recipe.

### The High-Dimensional Challenge: When $`p > n`$

However, what if $`p`$ (number of predictors) exceeds $`n`$ (sample size)? This scenario, known as the **high-dimensional problem**, presents unique challenges.

**Intuitive Understanding**: This is like having more ingredients than taste tests. You can't possibly test all combinations, and some combinations might give you perfect results just by chance (like overfitting to your limited taste tests). You need special strategies to handle this situation.

#### Mathematical Foundation of the Problem

**Perfect Fit Issue:**
For any linear regression model with sample size $`n`$, adding more than $`n-1`$ non-intercept predictors will result in a residual sum of squares of zero. This occurs because:

$$ \text{rank}(\mathbf{X}) \leq \min(n, p) $$

When $`p > n`$, the design matrix $`\mathbf{X}`$ has more columns than rows, leading to:
$$ \text{RSS} = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\|^2 = 0 $$

**Intuition**: This is like having more ingredients than taste tests - you can always find a combination that perfectly matches your limited taste tests, but this doesn't mean it's a good recipe. It's like memorizing the answers to a small test instead of learning the material.

**Criterion Breakdown:**
Consequently, both AIC and BIC metrics become undefined, as the first term of AIC and BIC — being the logarithm of the residual sum of squares — equals negative infinity:

$$ \begin{aligned}
\text{AIC} &= n\log(0/n) + 2p = -\infty + 2p \\
\text{BIC} &= n\log(0/n) + \log(n)p = -\infty + \log(n)p
\end{aligned} $$

**Intuition**: This is like trying to score a recipe when it gets a perfect score on your limited taste tests - the scoring system breaks down because it can't distinguish between truly good recipes and ones that just happened to work on your small sample.

#### Solutions for High-Dimensional Data

**1. Model Size Capping:**
Despite the criterion breakdown, search algorithms can still be utilized by setting a cap on the model size. Under the stepwise procedure, for example, when your model reaches this threshold, you can only remove predictors, not add them.

**Mathematical Formulation:**
$$ \text{Maximum model size} = \min(p, n-1) $$

**Intuition**: This is like limiting your recipe to use fewer ingredients than you have taste tests. You can still create good recipes, but you're forced to be selective about which ingredients to use.

**2. Variable Screening:**
When $`p > n`$, directly using the full model as a starting point isn't feasible. It's recommended to use screening procedures to identify a starting model for the stepwise process.

**Intuition**: This is like doing a quick preliminary taste test to identify which ingredients seem promising before doing a more thorough evaluation.

### Screening Methods

#### 1. Correlation-Based Screening

**Algorithm:**
A simple screening approach is to rank predictors based on their correlation magnitude with the outcome variable $`Y`$ and retain only the top $`K`$ predictors (e.g., $`K = n/3`$).

**Mathematical Formulation:**
$$ \text{Correlation with } Y: \rho_j = \frac{\text{Cov}(X_j, Y)}{\sqrt{\text{Var}(X_j)\text{Var}(Y)}} $$

**Screening Rule:**
$$ S_{\text{screen}} = \{j : |\rho_j| \geq \text{threshold}\} $$

where the threshold is chosen to select approximately $`K`$ variables.

**Intuition**: This is like doing a quick taste test of each ingredient individually to see which ones seem to affect the taste the most. You keep the ingredients that show the strongest individual effects.

#### 2. Univariate Regression Screening

**Algorithm:**
This method mirrors the process of executing individual simple linear regressions for $`Y`$ against each predictor and ranking them based on p-values.

**Mathematical Formulation:**
For each predictor $`X_j`$:
$$ Y = \beta_0 + \beta_j X_j + \epsilon $$

**Screening Rule:**
$$ S_{\text{screen}} = \{j : p\text{-value}_j \leq \alpha_{\text{screen}}\} $$

where $`\alpha_{\text{screen}}`$ is chosen to control the number of selected variables.

**Intuition**: This is like doing a more thorough individual test of each ingredient - not just tasting it, but doing a proper cooking test to see if it really makes a difference.

#### 3. Mutual Information Screening

**Algorithm:**
For non-linear relationships, mutual information can be used as a screening criterion.

**Mathematical Formulation:**
$$ I(X_j; Y) = \int\int p(x_j, y) \log\left(\frac{p(x_j, y)}{p(x_j)p(y)}\right) dx_j dy $$

**Screening Rule:**
$$ S_{\text{screen}} = \{j : I(X_j; Y) \geq \text{threshold}\} $$

**Intuition**: This is like testing for more complex relationships between ingredients and taste - not just linear effects, but any kind of relationship that might exist.

### Comprehensive Python Example: Variable Screening

See the complete implementation in [`code/screening_stepwise.py`](code/screening_stepwise.py) which demonstrates screening and stepwise selection for high-dimensional variable selection with comprehensive analysis and visualization.

### Key Insights and Recommendations

**1. Screening Benefits:**
- Reduces computational complexity from $`O(2^p)`$ to $`O(2^K)`$ - like reducing the number of recipes to test from billions to thousands
- Makes high-dimensional variable selection feasible - like making it possible to find good recipes even with many ingredients
- Provides starting point for more sophisticated methods - like doing a quick preliminary test before detailed evaluation

**2. Screening Limitations:**
- May miss important variables with weak marginal effects - like missing ingredients that only work well in combination
- Sensitive to correlation structure among predictors - like being fooled by similar ingredients
- Different screening methods may select different variables - like different preliminary tests giving different results

**3. Best Practices:**
- Use multiple screening methods and combine results - like doing several different preliminary tests
- Apply stepwise selection on screened variables - like doing detailed testing on the promising ingredients
- Validate final model with cross-validation - like testing the final recipe in different kitchens
- Consider domain knowledge in screening decisions - like using cooking experience to guide ingredient selection

**4. When to Use Screening:**
- $`p > n`$ scenarios - like having more ingredients than taste tests
- Computational constraints - like limited time for recipe testing
- Initial variable selection in large datasets - like preliminary ingredient selection for large cookbooks
- Preprocessing step for more sophisticated methods - like preparation for detailed recipe development

Although this elementary procedure might overlook crucial variables, the subsequent stepwise process can potentially reincorporate them into the model, making screening a valuable tool for high-dimensional variable selection.

**Intuition**: Screening is like doing a quick preliminary taste test to narrow down your options before doing detailed recipe development. It's not perfect, but it makes the impossible possible and the impractical practical.

---

**Navigation:**
- **Next Topic:** [Regularization Framework](02_regularization.md) - Mathematical foundations and unified approach to regularization
- **Previous Topic:** [Variable Selection and Regularization Overview](README.md) - Overview of variable selection and regularization techniques
