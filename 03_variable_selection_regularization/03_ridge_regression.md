# 3.3. Ridge Regression

## 3.3.1. Introduction to Ridge

In ridge regression, the objective function to be minimized is a smooth quadratic function of $\beta$. This function consists of two terms: the residual sum of squares and the L2 norm of $\beta$.

$$
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \mathbf{X}  \boldsymbol{\beta}\|^2 + \lambda \| \boldsymbol{\beta} \|^2
$$

To find the minimizer of this objective function, multiple approaches can be employed. For instance, taking the derivative with respect to $\beta$ and setting it to zero leads to a solvable form for the minimizer. Alternatively, one can express the objective function as the residual sum of squares for an augmented linear regression model.

In this augmented model, we have $n+p$ observations. The response vector is formed by stacking $p$ zero responses onto the original $y$ vector. The design matrix for the newly added $p$ responses is simply an identity matrix. Defining this augmented response vector as $\tilde{y}$ and the new design matrix as $\tilde{\mathbf{X}}$, one finds that their residual sum of squares is identical to the ridge regression objective function.

$$
\begin{pmatrix} \mathbf{y}_{n \times 1} \\ \mathbf{0}_{p \times 1} \end{pmatrix} = \begin{pmatrix} \mathbf{X}_{n \times p} \\ \sqrt{\lambda} \mathbf{I}_p \end{pmatrix}_{(n+p) \times p} \boldsymbol{\beta}_{p \times 1} + \text{ error}
$$

We can then find the ridge regression solution using this new design matrix and new response vector.

$$
\begin{align*}
\tilde{\mathbf{X}}^t \tilde{\mathbf{X}}  & =  \left (\mathbf{X}^t \, \sqrt{\lambda} \mathbf{I}_p \right ) \begin{pmatrix} \mathbf{X}_{n \times p} \\ \sqrt{\lambda} \mathbf{I}_p \end{pmatrix} \\
& = (\mathbf{X}^t \mathbf{X} + \lambda \mathbf{I}) \\
\tilde{\mathbf{X}}^t \tilde{\mathbf{y}} & = \left (\mathbf{X}^t \, \sqrt{\lambda} \mathbf{I}_p \right ) \begin{pmatrix} \mathbf{y}_{n \times 1} \\ \mathbf{0}_{p \times 1} \end{pmatrix} = \mathbf{X}^t \mathbf{y} \\
\hat{\boldsymbol{\beta}}^{\text{ridge}} & = (\tilde{\mathbf{X}}^t \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^t \tilde{\mathbf{y}} = (\mathbf{X}^t \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^t \mathbf{y}.
\end{align*}
$$

A key benefit of ridge regression is its applicability to design matrices that are not of full rank. This is made possible by adding $\lambda \mathbf{I}$ to $\mathbf{X}^t \mathbf{X}$, which allows us to invert the resulting matrix. This feature sets ridge regression apart from ordinary least squares (OLS) methods, making it a versatile tool for tackling a wide range of problems.

## 3.3.2. The Shrinkage Effect

When discussing ridge regression, it’s crucial to consider its nature as a ‘shrinkage’ method. This becomes evident when we look at the special case where the design matrix $X$ is orthogonal.

Suppose $X$ is an $n \times p$ matrix where the columns are orthonormal (i.e., unit length and mutually orthogonal). Then $\mathbf{X}^t \mathbf{X} = \mathbf{I}$.


$$
\begin{align*}
\hat{\boldsymbol{\beta}}^{\mathrm{LS}} \quad &= \; (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} = \mathbf{X}^T \mathbf{y} \\
\hat{\boldsymbol{\beta}}^{\mathrm{ridge}} \quad &= \; (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y} \\
&= \frac{1}{1 + \lambda} \mathbf{X}^T \mathbf{y} = \frac{1}{1 + \lambda} \hat{\boldsymbol{\beta}}^{\mathrm{LS}} \\
\\
\hat{\mathbf{y}}_{\mathrm{LS}} \quad &= \; \mathbf{X} \hat{\boldsymbol{\beta}}^{\mathrm{LS}} \\
\hat{\mathbf{y}}_{\mathrm{ridge}} \quad &= \; \mathbf{X} \hat{\boldsymbol{\beta}}^{\mathrm{ridge}} = \frac{1}{1 + \lambda} \mathbf{y}_{\mathrm{LS}}
\end{align*}
$$

It becomes clear that the ridge estimate is a shrunk version of the OLS estimate since $\frac{1}{1 + \lambda} < 1$ for any $\lambda > 0$.

In situations where the columns of $X$ are not orthogonal, we can reformulate the regression problem using an orthogonal version of $X$, achieved through techniques such as Principal Component Analysis (PCA) or Singular Value Decomposition (SVD). In this transformed space, it becomes evident that the ridge estimates and predictions serve as shrunken versions of their least squares (LS) counterparts.

Consider a singular value decomposition (SVD) of $\mathbf{X}$:

$$
\mathbf{X}_{n \times p} = \mathbf{U}_{n \times p} \mathbf{D}_{p \times p} \mathbf{V}^T_{p \times p},
$$

where

- $\mathbf{U}_{n \times p}$: columns $\mathbf{u}_j$ form an orthonormal (ON) basis for $C(\mathbf{X})$, $\mathbf{U}^T \mathbf{U} = I_p$.
- $\mathbf{V}_{p \times p}$: columns $\mathbf{v}_j$ form an ON basis for $\mathbb{R}^p$ with $\mathbf{V}^T \mathbf{V} = I_p$.
- $\mathbf{D}_{p \times p}$: diagonal matrix with diagonal entries $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$ being the singular values of $\mathbf{X}$.

<span style="color:teal">For ease of exposition we assume $n > p$ and $\operatorname{rank}(\mathbf{X}) = p$. Therefore $d_p > 0$.</span>

- PCA: write $\mathbf{X} = \mathbf{F} \mathbf{V}^T$ where each column of $\mathbf{F}_{n \times p} = \mathbf{U} \mathbf{D}$ is the so-called principal components and each column of $\mathbf{V}$ is the principal component directions of $\mathbf{X}$.

Next, we can rewrite the problem in terms of a transformed design matrix $F$ and new coefficient vector $\alpha$, which is a rotation transformation of the original $\beta$.


Write

$$
\mathbf{y} - \mathbf{X}\boldsymbol{\beta} = \mathbf{y} - \mathbf{U}\mathbf{D}\mathbf{V}^T\boldsymbol{\beta} = \mathbf{y} - \mathbf{F}\boldsymbol{\alpha}.
$$

There is a one-to-one correspondence between $\boldsymbol{\beta}_{p \times 1}$ and $\boldsymbol{\alpha}_{p \times 1}$ and $\|\boldsymbol{\beta}\|^2 = \|\boldsymbol{\alpha}\|^2$. So

$$
\min_{\boldsymbol{\beta} \in \mathbb{R}^p} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \iff \min_{\boldsymbol{\alpha} \in \mathbb{R}^p} \|\mathbf{y} - \mathbf{F}\boldsymbol{\alpha}\|^2 + \lambda \|\boldsymbol{\alpha}\|^2.
$$

$$
\begin{align*}
\hat{\boldsymbol{\alpha}}^{\mathrm{LS}} &= \mathbf{D}^{-1} \mathbf{U}^T \mathbf{y}, \qquad \hat{\alpha}^{\mathrm{LS}}_j = \frac{1}{d_j} \mathbf{u}_j^T \mathbf{y} \\
\hat{\boldsymbol{\alpha}}^{\mathrm{ridge}} &= \operatorname{diag}\left( \frac{d_j}{d_j^2 + \lambda} \right) \mathbf{U}^T \mathbf{y}, \qquad \hat{\alpha}^{\mathrm{ridge}}_j = \frac{d_j^2}{d_j^2 + \lambda} \hat{\alpha}^{\mathrm{LS}}_j
\end{align*}
$$

So the ridge estimate $\hat{\boldsymbol{\alpha}}^{\mathrm{ridge}}$ shrinks the LS estimate $\hat{\boldsymbol{\alpha}}^{\mathrm{LS}}$ by the factor $\frac{d_j^2}{d_j^2 + \lambda}$: directions with smaller eigenvalues get more shrinkage.

### Summary

In ridge regression, the coefficients and predictions are shrinkage versions of those in OLS. The degree of shrinkage varies with the magnitude of singular values, shrinking less for larger singular values and more for smaller ones. This accounts for the regularization effect that makes ridge regression robust, especially when $X$ has multicollinearity or is not full-rank.

## 3.3.3. Why Shrinkage

A natural question that arises is why one might want to shrink the least squares estimate. One might argue that the least squares estimate is unbiased, so applying shrinkage, which would introduce bias, seems counterintuitive. After all, isn’t unbiasedness a desirable property?

To explore this, let’s consider a simple one-dimensional estimation problem and examine the mean squared error (MSE) of two estimators. MSE, defined as the expected squared difference between the estimator and the true value, is equal to Bias-square plus Variance.

Consider a simple estimation problem: $Z_1, \ldots, Z_n$ iid $\sim N(\theta, \sigma^2)$. What’s the MSE of $\bar{Z}$ and what’s the MSE of $\frac{1}{2} \bar{Z}$?

$$
\begin{align*}
\mathrm{MSE}(\bar{Z}) \quad &= \; \mathbb{E}(\bar{Z} - \theta)^2 = \frac{\sigma^2}{n} \\
\mathrm{MSE}\left( \frac{1}{2} \bar{Z} \right) \quad &= \; \mathbb{E}(\bar{Z} - \theta)^2 = \frac{\theta^2}{4} + \frac{1}{4} \frac{\sigma^2}{n}
\end{align*}
$$

When comparing the MSE of these two estimators, it’s not straightforward to determine which estimator performs better. The effectiveness of the shrinkage depends on the magnitude of $\theta^2$.

In summary, while shrinkage introduces bias, it also reduces variance. This trade-off may result in an overall lower MSE, making shrinkage a worthwhile consideration in certain situations.

## 3.3.4. Degree-of-Freedom of Ridge Regression

When discussing linear regression with variable selection, a key consideration is the model’s dimensionality, which refers to the number of parameters or predictors utilized.

A pertinent question in this context is: What is the effective degree of freedom for a ridge regression model? Despite using a $p$-dimensional coefficient vector, the model’s effective dimensionality may be less due to the shrinkage effect introduced by the regularization term.

In ridge regression, the regularization strength is controlled by $\lambda$. This parameter directly impacts the shrinkage factor, usually represented as $a/(a + \lambda)$, where $a$ and $\lambda$ are both positive.

- As $\lambda$ approaches infinity, the shrinkage factor nears zero, essentially reducing the dimension of the model to zero.
- On the other hand, as $\lambda$ goes to zero, the ridge regression model becomes a standard $p$-dimensional least squares regression model.

A commonly-adopted metric for the degree of freedom involves the sum of the normalized variance between the observed $y_i$ and its corresponding predicted value $\hat{y}_i$, across all $n$ samples. In matrix notation, if a method returns $\hat{\mathbf{y}} = \mathbf{A} \mathbf{y}$, the degree of freedom can be computed as the trace of matrix $A$.

One way to measure the degree of freedom (df) of a method is

$$
df = \frac{1}{\sigma^2} \sum_{i=1}^n \operatorname{Cov}(y_i, \hat{y}_i).
$$

Suppose a method returns the $n$ fitted values as $\hat{\mathbf{y}} = \mathbf{A}_{n \times n} \mathbf{y}$ where $\mathbf{A}$ is an $n$-by-$n$ matrix not depending on $\mathbf{y}$ (of course, it depends on the $\mathbf{x}_i$'s). Then

$$
df = \frac{1}{\sigma^2} \sum_{i=1}^n \operatorname{Cov}(y_i, \hat{y}_i) = \sum_{i=1}^n A_{ii} = \operatorname{tr}(\mathbf{A}).
$$

For example, for a linear regression model with $p$ coefficients, we all agree that the degree of freedom is $p$. If using the formula above we have

$$
df = \operatorname{tr}(\mathbf{H}) = p, \qquad \hat{\mathbf{y}}_{\mathrm{LS}} = \mathbf{H} \mathbf{y}
$$

which also gives us $df = p$.

Similarly, we can compute the effective degree of freedom in ridge regression as follows.

For ridge regression, we have $\hat{\mathbf{y}}_{\mathrm{ridge}} = \mathbf{S}_\lambda \mathbf{y}$, where

$$
\mathbf{S}_\lambda = \mathbf{X}(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} \mathbf{u}_j \mathbf{u}_j^T.
$$

We can define the **effective df** of ridge regression to be

$$
df(\lambda) = \operatorname{tr}(\mathbf{S}_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}.
$$

When the tuning parameter $\lambda = 0$ (i.e., no regularization), $df(\lambda) = p$; when $\lambda$ goes to $\infty$, $df(\lambda)$ goes to $0$.

Distinct from other variable selection methods, ridge regression allows for a fractional degree of freedom. This value can range continuously between zero and $p$, contingent upon the regularization parameter $\lambda$.