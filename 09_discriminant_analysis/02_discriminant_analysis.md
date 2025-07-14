# 9.2. Discriminant Analysis

In our previous discussions, we explored the essentials of crafting an optimal classifier. To achieve this, our primary focus lies in understanding the conditional probability $P(Y = k|X=x)$.

A logical starting point would involve estimating the joint distribution of (X, Y) based on the available data. Subsequently, we can calculate the conditional probability of Y given X=x.

We aim to learn the joint distribution $p(x,y)$ by breaking it down into two parts:

$$p(x,y) = p(y) \cdot p(x | y)$$

Estimating $p(y)$, the marginal distribution of Y, is relatively straightforward since Y is a discrete random variable. Essentially, we determine the frequency of each class in the dataset. Then proceed to estimate $p(x|y)$, the distribution of the feature vector x in each class.

Once these two components are acquired, we can reconstruct the joint distribution and subsequently compute the conditional probabilities of Y give x (aka, Bayes's Theorem). This approach, which involves learning the classifications in this manner, is commonly referred to as **discriminant analysis**.

Specifically, we will discuss three classification rules falling under this discriminant analysis umbrella:

- **Quadratic Discriminant Analysis (QDA)**
- **Linear Discriminant Analysis (LDA)**
- **Naive Bayes**

Additionally, we will explore the relationship between LDA and Fisher's Discriminant Analysis (FDA).

## 9.2.1. Bayes' Theorem

First, let's refresh our memory regarding the derivation of the optimal classifier using Bayes' Theorem. Our primary objective is to calculate the posterior probability $P(Y=k | X=x)$.

$$\begin{split}
P(Y = k | X=x) &= \frac{P(X=x, Y=k)}{P(X=x)} \\
&= \frac{P(X=x | Y=k) \cdot P(Y =k)}{P(X=x)} \\
&= \frac{\pi_k f_k(x)}{P(X=x)} \propto \pi_k f_k(x)
\end{split}$$

where

- $f_k(x) = p(x | Y=k)$: conditional density function of $X|Y=k$
- $\pi_k = P(Y=k)$: the marginal probability or prior probability for class 'k'.

As mentioned earlier, our goal is to identify 'k' that maximizes the conditional probability. This is equivalent to finding 'k' that maximizes the numerator $\pi_k f_k(x)$, as the denominator remains constant. When maximizing a product, we can achieve the same result by taking the logarithm of the product. Therefore, we can write the decision function as

$$\arg\max_k \pi_k f_k(x) = \arg\max_k \Big [ \log \pi_k + \log f_k(x) \Big ],$$

or equivalently

$$\arg\min_k = - \Big [ \log \pi_k + \log f_k(x) \Big ].$$

In constructing the optimal classifier, our sole requirements are estimating $\pi_k$ and $f_k(x)$.
