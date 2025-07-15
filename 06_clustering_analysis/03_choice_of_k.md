# 6.3. Choice of K

## 6.3.1. Introduction

In supervised learning, the goal is clear: make accurate predictions for the target variable, Y. But unsupervised learning, such as clustering, doesn't have a Y variable, making it challenging to evaluate its accuracy or effectiveness. This lack of a clear target introduces complexities when determining the optimal number of clusters, K.

In supervised scenarios like regression, the go-to method for tuning parameters is cross-validation. However, applying cross-validation directly to clustering isn't straightforward. Despite these challenges, several techniques aid in determining the optimal K. Key among them are gap statistics, silhouette statistics, and prediction strength.

## 6.3.2. Gap Statistics

When examining clustering effectiveness, many measures gauge the compactness or tightness of clusters. A common metric is the within cluster sum of squares, which, when based on the L2 distance, matches the objective function of K-means.

$$SS(K) = \sum_{k=1}^K \sum_{z_i=k} \| x_i - m_k\|^2.$$

It's natural to aim for a smaller SS, indicating tighter clusters. However, as the number of clusters (K) increases, the SS inherently decreases for the same dataset. Thus, relying solely on SS can be misleading when selecting the optimal K.

To determine the optimal K, researchers often use the "elbow method." Here, the sum of squares is plotted against K. If a curve is observed with a distinct "elbow" point, that point often signifies the best K value. However, in real-world data, identifying the precise elbow can be challenging due to noise and complexity.

The **gap statistic** (Tibshirani, Walther and Hastie, 2001) compares the clustering of actual data against a random clustering from a reference distribution. It's calculated by measuring the SS from the observed data against the expected log sum of squares from a reference set. This reference set is derived from a distribution that has no intrinsic clustering, meaning an ideal number of clusters would be one.

```math
G(K) = \mathbb{E}_0 \Big [ \log SS^*(K) \Big ]- \log SS_{\text{obs}}(K) \approx \frac{1}{B} \sum_{b=1}^B \log SS^*_b(K) - \log SS_{\text{obs}}(K)
```

To estimate the gap statistic, multiple samples from the reference distribution are taken, and the average over these samples provides an expectation. As K grows, even though the sum of squares shrinks, the difference (or gap) may not always decrease. A high gap statistic suggests that the SS for the observed data at a particular K is notably smaller than its reference counterpart, indicating good clustering.

### Generating Data from the Reference Distribution

There are two proposed methods:

1. **Uniform Sampling**: Here, the reference data is uniformly sampled over the range of the observed data. This method may not be effective if the observed data has distinct shapes.

2. **Principal Component Based Sampling**: This method samples over the range of the principal components of the observed data, ensuring better alignment with the data's structure.

### Determining Optimal K with Gap Statistic

Plot the gap statistic values for different K.

The optimal K is determined either by identifying the highest gap statistic or, in a sequential approach, by selecting the first K where its gap statistic exceeds that of K+1.

Since the gap statistic is based on random sampling, there's inherent variability. One-standard-error principle is used to account for this uncertainty. We compare the gap statistic at K to the lower bound of the gap statistic for K+1 (subtracting one standard error). If the former is greater, we consider that K as optimal.

$$K_{\text{opt}} = \arg\min_K \{K : G(K) \ge G(K+1) - s_{K+1} \}$$

where $s_K = \text{sd}_0(\log SS(K)) \sqrt{1+1/B}$.

## 6.3.3. Silhouette Statistics (Expanded)

The **Silhouette statistic** (Rousseeuw, 1987) provides an interpretable measure of how well each observation lies within its cluster, balancing cohesion (how close it is to its own cluster) and separation (how far it is from the next closest cluster).

### Definition

For each observation $`i`$:
- $`a_i`$ = average distance from $`i`$ to all other points in its own cluster (cohesion)
- $`b_i`$ = minimum average distance from $`i`$ to all points in any other cluster (separation)

The silhouette value for observation $`i`$ is:

```math
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
```

- $`s_i \approx 1`$: well-clustered, far from other clusters
- $`s_i \approx 0`$: on the border between clusters
- $`s_i < 0`$: possibly misclassified

#### Visual Explanation (from image)
- $`a_i`$: mean intra-cluster distance (to own cluster)
- $`b_i`$: mean nearest-cluster distance (to next closest cluster)
- $`s_i`$ is high when $`a_i`$ is much less than $`b_i`$

### Silhouette Coefficient

The **Silhouette Coefficient** (SC) for the clustering is the average $`s_i`$ over all samples:

```math
SC = \frac{1}{n} \sum_{i=1}^n s_i
```

#### Interpretation Benchmarks
- $`SC > 0.70`$: Strong structure
- $`SC > 0.50`$: Reasonable structure
- $`SC > 0.26`$: Weak structure, may be artificial
- $`SC < 0.26`$: No substantial structure

### Choosing K with Silhouette

Compute $`SC`$ for a range of $`K`$ and select the $`K`$ with the highest $`SC`$ or above a threshold.

### Python Example

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

X = ... # your data
K_range = range(2, 10)
sc_scores = []
for K in K_range:
    kmeans = KMeans(n_clusters=K, n_init=10).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    sc_scores.append(score)

import matplotlib.pyplot as plt
plt.plot(K_range, sc_scores, marker='o')
plt.xlabel('Number of clusters K')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Analysis for Optimal K')
plt.show()
```

### R Example

```r
library(cluster)
K_range <- 2:10
sc_scores <- numeric(length(K_range))
for (i in seq_along(K_range)) {
  km <- kmeans(X, centers = K_range[i], nstart = 10)
  sc_scores[i] <- mean(silhouette(km$cluster, dist(X))[, 3])
}
plot(K_range, sc_scores, type = 'b', xlab = 'Number of clusters K', ylab = 'Silhouette Coefficient')
```

---

## 6.3.4. Prediction Strength (Expanded)

**Prediction Strength** (Tibshirani & Walther, 2005) is a stability-based method for choosing $`K`$ by measuring how reproducible the clustering is under data splitting.

### Algorithm Steps

1. **Split the data** into two sets: A (training) and B (test)
2. **Cluster B** into $`K`$ clusters: $`C_1, \ldots, C_K`$
3. **Cluster A** into $`K`$ clusters, then assign B to clusters using A's centroids (predict cluster labels for B)
4. **Compare**: For each cluster $`C_j`$ in B, for every pair of points in $`C_j`$, check if they are also together in the predicted clustering
5. **Prediction strength** for $`K`$ is the minimum proportion of pairs in any cluster that are together in both clusterings

### Mathematical Definition

Let $`M`$ be the co-membership matrix for B in the true clustering, and $`M'`$ for the predicted clustering. For each cluster $`C_j`$:

```math
PS_j = \frac{1}{\binom{m_j}{2}} \sum_{i < l, i, l \in C_j} \mathbb{I}\{M_{il} = M'_{il} = 1\}
```

where $`m_j`$ is the size of cluster $`C_j`$.

The **prediction strength** for $`K`$ is:

```math
PS(K) = \min_j PS_j
```

### Choosing K

Select the largest $`K`$ such that $`PS(K)`$ exceeds a threshold (e.g., 0.8).

### Python Example

```python
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

def prediction_strength(X, K, n_splits=5, threshold=0.8):
    n = X.shape[0]
    ps_scores = []
    for split in range(n_splits):
        idx = np.random.permutation(n)
        A, B = X[idx[:n//2]], X[idx[n//2:]]
        # Cluster B
        kmeans_B = KMeans(n_clusters=K, n_init=10).fit(B)
        labels_B = kmeans_B.labels_
        # Cluster A, assign B
        kmeans_A = KMeans(n_clusters=K, n_init=10).fit(A)
        labels_B_pred = kmeans_A.predict(B)
        # For each cluster in B, compute PS_j
        ps_j = []
        for j in range(K):
            members = np.where(labels_B == j)[0]
            if len(members) < 2:
                continue
            pairs = [(i, l) for idx, i in enumerate(members) for l in members[idx+1:]]
            agree = sum(labels_B_pred[i] == labels_B_pred[l] for i, l in pairs)
            ps_j.append(agree / len(pairs))
        if ps_j:
            ps_scores.append(min(ps_j))
    return np.mean(ps_scores)

# Example usage:
K_range = range(2, 10)
ps_results = [prediction_strength(X, K) for K in K_range]
plt.plot(K_range, ps_results, marker='o')
plt.axhline(0.8, color='red', linestyle='--', label='Threshold')
plt.xlabel('Number of clusters K')
plt.ylabel('Prediction Strength')
plt.legend()
plt.show()
```

### R Example

```r
library(fpc)
ps <- prediction.strength(X, Gmin=2, Gmax=10, M=10, clustermethod=kmeansCBI)
plot(2:10, ps$mean.pred, type='b', xlab='Number of clusters K', ylab='Prediction Strength')
abline(h=0.8, col='red', lty=2)
```

---

## 6.3.5. Summary and Best Practices

- **Gap statistic**: Compares clustering to a null reference; robust but computationally intensive
- **Silhouette**: Measures cohesion/separation; easy to interpret and compute
- **Prediction strength**: Measures stability; good for practical validation
- **No single method is perfect**; use multiple criteria and domain knowledge
- **Visualize**: Always inspect cluster assignments and validation plots
