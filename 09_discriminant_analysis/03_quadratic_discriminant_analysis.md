# 9.3. Quadratic Discriminant Analysis

In our previous discussion, we explored the process of constructing a classification rule. To do this effectively, we need to estimate two key components:

- $f_k(x)$: conditional density function of $X|Y=k$
- $\pi_k$: the marginal probability or prior probability for class 'k'.

Estimating $\pi_k$ is straightforward, as it involves calculating the frequency of occurrence for a discrete random variable Y.

On the other hand, estimating $f_k(x)$, which represents the distribution of the p-dimensional feature vector x, requires a bit more consideration. When X is continuous, a natural choice is to assume a normal distribution. So, let's assume that given Y = k, we model the p-dimensional feature vector x for this particular class as a multivariate normal distribution.

This distribution has two crucial parameters: $\mu_k$, a p-dimensional mean vector, and $\Sigma_k$, a p-by-p covariance matrix. It's worth noting that we'll represent the inverse of the covariance matrix as $\Theta$. This inverse covariance matrix is often referred to as the precision matrix.

$$\begin{split}
\mu_k = \left ( \begin{array}{c} \mu_{k,1} \\ \mu_{k,2} \\ \vdots \\ \mu_{k,p} \end{array} \right )_{p \times 1}, \quad
\Theta = \Sigma_k^{-1} = \left ( \begin{array}{ccc} \theta_{k,11} & \cdots & \theta_{k, 1p} \\ & & \\ \theta_{k, p1} & \cdots & \theta_{k, pp} \end{array} \right)_{p \times p}.
\end{split}$$

For a multivariate normal distribution with these parameters, the density function takes the following form:

$$f_k(x) = \frac{1}{(\sqrt{2 \pi})^p} \frac{1}{| \Sigma_k|^{1/2}} \exp \Big [ - \frac{1}{2} (x- \mu_k)^t \Sigma_k^{-1} (x - \mu_k) \Big ]$$

where

$$(x- \mu_k)^t \Sigma_k^{-1} (x - \mu_k) = \sum_{j=1}^p \sum_{l=1}^p \theta_{k, jl} (x_{j}- \mu_{k,j}) (x_{l} - \mu_{k,l}).$$

Now, move on to computing the Bayes rule for our model. If we assume that $f_k$ follows a normal distribution, we have

$$P(Y = k \mid x) \propto \pi_k f_k(x) \propto e^{-d_k(x)/2}$$

where

$$\begin{split}
d_k(x) &= 2 \Big [ - \log f_k(x) - \log \pi_k \Big ] - \text{Constant} \\
&= (x-\mu_k)^t \Sigma_k^{-1} (x-\mu_k) + \log |\Sigma_k| - 2 \log \pi_k,
\end{split}$$

Since our objective is to maximize $P(Y = k \mid x)$, we can equivalently minimize the quantity $d_k$. Breaking down $d_k$, it consists of three terms:

- **The 1st term** is the quadratic term mentioned earlier, representing the **Mahalanobis distance** between data point $x$ and the center $\mu_k$;
- **The 2nd term** is the log of the determinant of the covariance matrix, and
- **The 3rd term** involves the frequency of the class.

This classification rule, described here, is known as **quadratic discriminant analysis (QDA)**. The key characteristic of QDA is that the function used to make decisions is quadratic in terms of the feature vector $x$.

For example, if we have two classes (K = 2), the decision boundary would be where $d_k(x) = 1/2$, indicating equal likelihood for both classes. Due to the quadratic nature of $d_k(x)$, the decision boundary will also be quadratic in terms of $x$.

In practice, estimating the three sets of parameters $(\pi_k, \mu_k, \Sigma_k)$ involves straightforward calculations. $\pi_k$ can be computed as the frequency of class k in the training data; $\mu_k$ and $\Sigma_k$ can be estimated by the sample mean and sample covariance matrix of the data points belonging to each class separately. Once these parameters are estimated, we can calculate $d_k(x)$ for every class k and select the class that minimizes $d_k$ as our prediction.

It's important to note that in these calculations, the focus is not on $\Sigma_k$ itself but on its inverse. In cases where $\Sigma_k$ is not invertible, typically due to a large number of dimensions (p), a simple solution is to add a small constant ($\epsilon$) times the identity matrix to $\Sigma_k$ and then take its inverse $(\Sigma_k + \epsilon \mathbf{I}_p)^{-1},$ ensuring invertibility.
