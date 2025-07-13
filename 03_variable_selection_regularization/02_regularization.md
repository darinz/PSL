# 3.2. Regularization

In this lecture, we’ll delve into variable selection using the regularization framework. Previously, we discussed using AIC or BIC for variable subset selection. This process can be framed as the pursuit of minimizing a specific objective function.

Assuming we know or can reliably estimate the error variance, $\sigma^2$, the objective function becomes:

$$
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \mathbf{X}  \boldsymbol{\beta}\|^2 + \lambda \| \boldsymbol{\beta} \|_0,
$$

where $\| \boldsymbol{\beta} \|_0 = \sum_{j=1}^p \mathbf{1}_{\{ \beta_j \ne 0 \}}$ counts the number of predictors in our model. Both AIC and BIC essentially correspond to different choices of $\lambda$. Yet, framing AIC and BIC under this unified objective function doesn’t simplify matters, because it poses the same computational challenges as AIC or BIC. In the worst-case scenario, we’d need to explore all possible subsets of $\beta$.

Next, we’ll introduce two alternative regularization or penalty terms for $\beta$. Ridge regression employs the L2 penalty on $\beta$, while Lasso regression uses the L1 penalty.

$$
\begin{align*}
\text{Ridge} & : \min \| \mathbf{y} - \mathbf{X}  \boldsymbol{\beta}\|^2 + \lambda \| \boldsymbol{\beta} \|^2 \\
\text{Lasso} & : \min \| \mathbf{y} - \mathbf{X}  \boldsymbol{\beta}\|^2 + \lambda \| \boldsymbol{\beta} \|_1
\end{align*}
$$

The $L_0$ norm’s focus is on the presence or absence of predictors, without regard to their magnitude. However, this isn’t true for the $L_1$ or $L_2$ norms. This can create complications.

For instance, a predictor $X_j$ that measures price in dollars can be transformed to measure in thousands of dollars, which, while only a scale change, can drastically impact the Ridge or Lasso $\beta$ estimates. Furthermore, a simple location shift of predictors can influence the intercept. If we penalize the intercept, any minor location change in $X$ or $Y$ will yield different Ridge or Lasso intercepts.

These inconsistencies are problematic. Ideally, the algorithm should yield consistent results regardless of predictor scaling or location changes.

Due to this, it’s customary to preprocess the data. This involves centering and scaling each column of the design matrix to achieve a mean of zero and unit sample variance. We also center $Y$. This ensures the intercept, when regressing $Y$ against $X$, is consistently zero. In effect, the design matrix $X$ only contains columns without the intercept.

This preprocessing might seem intricate, but many software packages handle this automatically. They apply transformations before executing their algorithms and then revert the coefficients to their original scale, reintroducing the intercept.

**Send standardized data to the algorithm, and obtain:**

$$
\left( \frac{Y - m_y}{se_y} \right) = \hat{\beta}_1 \cdot \left( \frac{X_1 - m_{x,1}}{se_{x,1}} \right) + \cdots + \hat{\beta}_p \cdot \left( \frac{X_p - m_{x,p}}{se_{x,p}} \right)
$$

**Scale back:**

$$
Y = \hat{\beta}_0 + \hat{\beta}_1 \frac{se_y}{se_{x,1}} X_1 + \cdots + \hat{\beta}_p \frac{se_y}{se_{x,p}} X_p
$$

This intercept $\beta_0$ can be computed using the following formula:

$$
\hat{\beta}_0 = m_y - \sum_{j=1}^p \hat{\beta}_j \cdot m_{x,j} \frac{se_y}{se_{x,j}}.
$$
