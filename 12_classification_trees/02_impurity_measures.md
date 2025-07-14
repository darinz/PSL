# 12.2. Impurity Measures

In the context of classification trees, the selection of a suitable goodness-of-split criterion is a critical consideration. Typically, we rely on a concept known as the "gain" of an impurity measure. But what exactly is this impurity measure?

## 12.2.1. Impurity Measures

The impurity measure is a function $I(p_1, \dots, p_K)$ defined over a probability distribution representing K classes. For instance, if K equals three, we work with a probability vector $(p_1, p_2, p_3).$ These values represent the probabilities of occurrence for each of the three classes.

The impurity measure quantifies the "impurity" or randomness of the distribution. It reaches its maximum value when all classes are equally likely and its minimum when only one class is certain (i.e., $p_j$ equals one for one class). Importantly, the impurity measure is always symmetric because it operates on probabilities, making it independent of class labels' order.

- maximum occurs at $(1/K, \dots, 1/K)$ (the most impure node);
- minimum occurs at $p_j = 1$ (the purest node)
- symmetric function of $p_1, \dots, p_K$, i.e., permutation of $p_j$ does not affect $I(\cdot).$

Ideally, in classification, we aim for nodes to be as pure as possible, which corresponds to a small impurity measure. In this regard, the impurity measure serves a purpose akin to the residual sum of squares in regression.

## 12.2.2. Goodness-of-split Criterion

Once we have defined the impurity measure, we can derive the goodness-of-split criterion, denoted as

$$\Phi(j,s) = i(t) - \big [ p_R \cdot i(t_R)  + p_L \cdot i(t_L)  \big ]$$

where

$$\begin{split}i(t) &=  I(p_t(1), \dots, p_t(K)) \\
p_t(j) &= \text{ frequency of class $j$ at node $t$}\end{split}$$

When we split a node into left and right nodes, we evaluate the impurity measure at the parent node (original node t) based on the empirical distribution of frequencies across the K classes. We also calculate the impurity measure at the left and right nodes if no split is applied.

However, unlike the residual sum of squares, the impurity measure is not cumulative; it represents a quantity at the distribution level. Therefore, we must compute a **weighted sum** to determine $\Phi$, where $p_R$ represents the proportion of samples in the right node and $p_L$ represents the proportion in the left node.

The criteria $\Phi,$ representing the gain of the impurity measure, is computed as the difference between i) the impurity measure when no split is applied and ii) the weighted sum of impurity measures in the left and right nodes after a split.

## 12.2.3. Choice of Impurity Measures

The choice of impurity measure for classification trees includes:

$$\begin{split}\text{Misclassification Rate } & :  1- \max_j p_j \\
\text{ Entropy (Deviance) } & : - \sum_{j=1}^K p_j \log p_j \\
\text{Gini index } & : \sum_{j=1}^K p_j(1-p_j) = 1- \sum_j p_j^2\end{split}$$

1. **Misclassification Rate:**
   In this measure, majority voting is used, and the class corresponding to the maximum $p_j$ is considered correct. The misclassification rate is computed as 1 minus the maximum $p_j$. This measure is symmetric and attains its maximum with equally likely classes and its minimum when only one class exists.

2. **Entropy:**
   Entropy is a popular impurity measure that quantifies the randomness of a distribution. It is commonly used in various fields such as coding theory, communication, and physics to describe the uncertainty or randomness in a discrete distribution over K classes. Like misclassification rate, entropy also reaches its maximum at a uniform distribution and its minimum at a deterministic distribution. Entropy is often favored when growing the tree, as it encourages the creation of pure nodes, which facilitates pruning in later stages.

3. **Gini Index:**
   The Gini index is another widely used impurity measure. It shares similarities with entropy in terms of performance. The choice between Gini index and entropy often depends on the specific application and preference. In practice, entropy is commonly used due to its connection with likelihood for a multinomial distribution.

It's important to note that entropy is a strictly concave function, which means it strongly favors splits leading to pure nodes. This characteristic makes entropy a suitable choice during the initial tree construction phase, where achieving purity is desirable. Subsequently, when pruning the tree, one may switch to using either the misclassification rate or entropy, depending on the ultimate classification goal.
