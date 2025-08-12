# 6.3. Choice of K

## 6.3.1. Introduction

In supervised learning, the goal is clear: make accurate predictions for the target variable, Y. But unsupervised learning, such as clustering, doesn't have a Y variable, making it challenging to evaluate its accuracy or effectiveness. This lack of a clear target introduces complexities when determining the optimal number of clusters, K.

**Intuitive Understanding**: Choosing the right number of clusters is like deciding how many different categories to use when organizing a library. If you have too few categories (like just "books" and "not books"), everything gets mixed together and it's hard to find what you need. If you have too many categories (like separate categories for each author, year, and genre), you end up with many tiny groups that don't make sense. The challenge is that there's no "correct answer" written on the wall - you have to figure out the best number of categories by looking at how well the organization works. This is exactly what we face in clustering: we need to determine the optimal number of groups (K) without having a "right answer" to compare against.

In supervised scenarios like regression, the go-to method for tuning parameters is cross-validation. However, applying cross-validation directly to clustering isn't straightforward. Despite these challenges, several techniques aid in determining the optimal K. Key among them are gap statistics, silhouette statistics, and prediction strength.

**Intuition**: In supervised learning, it's like having a teacher who tells you if your predictions are right or wrong. You can easily test different approaches and see which one works best. But in clustering, there's no teacher - you're trying to find natural groups in the data without knowing what those groups should be. It's like trying to organize a collection of objects without anyone telling you how many categories there should be or what they should be called.

## 6.3.2. Gap Statistics

When examining clustering effectiveness, many measures gauge the compactness or tightness of clusters. A common metric is the within cluster sum of squares, which, when based on the L2 distance, matches the objective function of K-means.

$$ SS(K) = \sum_{k=1}^K \sum_{z_i=k} \| x_i - m_k\|^2 $$

**Intuition**: The within-cluster sum of squares (SS) measures how "tight" or "compact" our groups are. Think of it like measuring how close all the items in each group are to their group center. If all items in a group are very close to the center, the group is tight and well-organized. If items are scattered far from the center, the group is loose and messy. We want to minimize this measure to create tight, well-organized groups.

It's natural to aim for a smaller SS, indicating tighter clusters. However, as the number of clusters (K) increases, the SS inherently decreases for the same dataset. Thus, relying solely on SS can be misleading when selecting the optimal K.

**Intuition**: This is like saying that if you keep dividing your library into more and more categories, each category will naturally become smaller and more specific. A category with just one book will have zero "messiness" because there's only one item in it. But having a separate category for each book doesn't make sense - it's not a useful organization. The challenge is finding the sweet spot where you have enough categories to be organized but not so many that the organization becomes meaningless.

To determine the optimal K, researchers often use the "elbow method." Here, the sum of squares is plotted against K. If a curve is observed with a distinct "elbow" point, that point often signifies the best K value. However, in real-world data, identifying the precise elbow can be challenging due to noise and complexity.

**Intuition**: The elbow method is like looking at a graph of how much your organization improves as you add more categories. At first, adding categories dramatically reduces the messiness (the curve drops steeply). But eventually, adding more categories doesn't help much (the curve flattens out). The "elbow" is the point where the curve stops dropping steeply and starts to flatten. This is often the optimal number of categories - you've captured most of the useful organization without going overboard.

The **gap statistic** (Tibshirani, Walther and Hastie, 2001) compares the clustering of actual data against a random clustering from a reference distribution. It's calculated by measuring the SS from the observed data against the expected log sum of squares from a reference set. This reference set is derived from a distribution that has no intrinsic clustering, meaning an ideal number of clusters would be one.

$$ G(K) = \mathbb{E}_0 \Big [ \log SS^*(K) \Big ]- \log SS_{\text{obs}}(K) \approx \frac{1}{B} \sum_{b=1}^B \log SS^*_b(K) - \log SS_{\text{obs}}(K) $$

**Intuition**: The gap statistic is like comparing how well you can organize your actual data versus how well you could organize completely random data. If your data has natural groups, you should be able to organize it much better than random data. The gap statistic measures this difference. A large gap means your organization is much better than what you'd expect from random data, suggesting you've found real structure in your data.

To estimate the gap statistic, multiple samples from the reference distribution are taken, and the average over these samples provides an expectation. As K grows, even though the sum of squares shrinks, the difference (or gap) may not always decrease. A high gap statistic suggests that the SS for the observed data at a particular K is notably smaller than its reference counterpart, indicating good clustering.

**Intuition**: This is like testing your organization skills on many different random collections of items. If you can consistently organize your actual data much better than these random collections, it suggests your data has real structure. The gap statistic tells you how much better your organization is compared to what you'd expect from random data.

### Generating Data from the Reference Distribution

There are two proposed methods:

1. **Uniform Sampling**: Here, the reference data is uniformly sampled over the range of the observed data. This method may not be effective if the observed data has distinct shapes.

**Intuition**: Uniform sampling is like creating random data by randomly placing items across the entire range of your actual data. It's like randomly scattering books across all possible locations in your library. This works well if your data is roughly uniform, but if your data has specific patterns or shapes, this method might not capture the right comparison.

2. **Principal Component Based Sampling**: This method samples over the range of the principal components of the observed data, ensuring better alignment with the data's structure.

**Intuition**: Principal component based sampling is like creating random data that follows the same overall shape and structure as your actual data, but without the natural groupings. It's like creating a version of your library where books are randomly placed but still follow the overall layout and structure. This gives you a more realistic comparison because the random data has the same basic characteristics as your real data.

### Determining Optimal K with Gap Statistic

Plot the gap statistic values for different K.

The optimal K is determined either by identifying the highest gap statistic or, in a sequential approach, by selecting the first K where its gap statistic exceeds that of K+1.

Since the gap statistic is based on random sampling, there's inherent variability. One-standard-error principle is used to account for this uncertainty. We compare the gap statistic at K to the lower bound of the gap statistic for K+1 (subtracting one standard error). If the former is greater, we consider that K as optimal.

$$ K_{\text{opt}} = \arg\min_K \{K : G(K) \ge G(K+1) - s_{K+1} \} $$

where $`s_K = \text{sd}_0(\log SS(K)) \sqrt{1+1/B}`$.

**Intuition**: The one-standard-error rule is like being conservative in your choice. Instead of picking the K with the absolute highest gap statistic (which might be due to random chance), you pick the smallest K where the gap statistic is still "significantly" better than the next K. It's like saying "I want the simplest organization that still works well" rather than "I want the organization that looks best, even if the improvement is tiny."

## 6.3.3. Silhouette Statistics (Expanded)

The **Silhouette statistic** (Rousseeuw, 1987) provides an interpretable measure of how well each observation lies within its cluster, balancing cohesion (how close it is to its own cluster) and separation (how far it is from the next closest cluster).

**Intuition**: The silhouette statistic is like measuring how well each item "fits" in its group. For each item, we ask: "How close is this item to other items in its own group?" and "How far is this item from items in the nearest other group?" If an item is very close to its own group and far from other groups, it's well-placed. If it's equally close to its own group and other groups, it's on the border. If it's closer to other groups than its own, it might be in the wrong group.

### Definition

For each observation $`i`$:
- $`a_i`$ = average distance from $`i`$ to all other points in its own cluster (cohesion) - like how close this item is to other items in its group
- $`b_i`$ = minimum average distance from $`i`$ to all points in any other cluster (separation) - like how far this item is from the nearest other group

The silhouette value for observation $`i`$ is:

$$ s_i = \frac{b_i - a_i}{\max(a_i, b_i)} $$

- $`s_i \approx 1`$: well-clustered, far from other clusters - like an item that's very close to its own group and far from other groups
- $`s_i \approx 0`$: on the border between clusters - like an item that's equally close to its own group and other groups
- $`s_i < 0`$: possibly misclassified - like an item that's actually closer to other groups than its own group

**Intuition**: The silhouette formula creates a score between -1 and 1. If b_i (distance to other groups) is much larger than a_i (distance to own group), the item is well-placed and gets a high score. If a_i and b_i are similar, the item is on the border and gets a score near zero. If a_i is larger than b_i, the item might be in the wrong group and gets a negative score.

#### Visual Explanation (from image)
- $`a_i`$: mean intra-cluster distance (to own cluster) - like the average distance from this item to other items in its group
- $`b_i`$: mean nearest-cluster distance (to next closest cluster) - like the average distance from this item to items in the nearest other group
- $`s_i`$ is high when $`a_i`$ is much less than $`b_i`$ - like when an item is much closer to its own group than to other groups

### Silhouette Coefficient

The **Silhouette Coefficient** (SC) for the clustering is the average $`s_i`$ over all samples:

$$ SC = \frac{1}{n} \sum_{i=1}^n s_i $$

**Intuition**: The silhouette coefficient is like the average "fit score" for all items in your organization. It tells you how well, on average, all items fit into their assigned groups. A high score means most items are well-placed, while a low score means many items are poorly placed or on borders between groups.

#### Interpretation Benchmarks
- $`SC > 0.70`$: Strong structure - like having a very clear, well-defined organization
- $`SC > 0.50`$: Reasonable structure - like having a decent organization with some clear groups
- $`SC > 0.26`$: Weak structure, may be artificial - like having groups that are somewhat arbitrary
- $`SC < 0.26`$: No substantial structure - like having an organization that doesn't really make sense

**Intuition**: These benchmarks help you interpret how good your clustering is. A score above 0.70 means you've found very clear, natural groups in your data. A score below 0.26 suggests that your groups might not be meaningful - you might be forcing an organization where none exists.

### Choosing K with Silhouette

Compute $`SC`$ for a range of $`K`$ and select the $`K`$ with the highest $`SC`$ or above a threshold.

**Intuition**: This is like trying different numbers of categories and seeing which one gives the best average "fit" for all items. You might try organizing your library into 3 categories, then 4, then 5, and so on, and pick the number that gives the highest silhouette coefficient. This tells you which organization makes the most sense for your data.

### Python Example

**Implementation:** See `silhouette_analysis()` function in [choice_of_k_implementation.py](code/choice_of_k_implementation.py)

The function computes silhouette scores for a range of K values and provides both average silhouette scores and individual sample scores for detailed analysis.

### R Example

**Implementation:** See `silhouette_analysis()` function in [r_choice_of_k_implementation.R](code/r_choice_of_k_implementation.R)

The function computes silhouette scores for a range of K values using R's cluster package and provides comprehensive analysis capabilities.

---

## 6.3.4. Prediction Strength (Expanded)

**Prediction Strength** (Tibshirani & Walther, 2005) is a stability-based method for choosing $`K`$ by measuring how reproducible the clustering is under data splitting.

**Intuition**: Prediction strength is like testing how stable your organization is. If you split your data in half and organize each half separately, do you get similar groups? If you do, it suggests your organization is stable and meaningful. If you get very different groups, it suggests your organization might be arbitrary or unstable.

### Algorithm Steps

1. **Split the data** into two sets: A (training) and B (test) - like dividing your library into two halves
2. **Cluster B** into $`K`$ clusters: $`C_1, \ldots, C_K`$ - like organizing the first half of your library into K categories
3. **Cluster A** into $`K`$ clusters, then assign B to clusters using A's centroids (predict cluster labels for B) - like organizing the second half using the same categories you found in the first half
4. **Compare**: For each cluster $`C_j`$ in B, for every pair of points in $`C_j`$, check if they are also together in the predicted clustering - like checking if items that were grouped together in the first organization are still grouped together in the second organization
5. **Prediction strength** for $`K`$ is the minimum proportion of pairs in any cluster that are together in both clusterings - like finding the worst-performing group and using that as your measure

**Intuition**: This process is like having two librarians organize the same collection independently. If both librarians come up with similar organizations, it suggests there are natural, stable groups in the data. If they come up with very different organizations, it suggests the groups might be arbitrary.

### Mathematical Definition

Let $`M`$ be the co-membership matrix for B in the true clustering, and $`M'`$ for the predicted clustering. For each cluster $`C_j`$:

$$ PS_j = \frac{1}{\binom{m_j}{2}} \sum_{i < l, i, l \in C_j} \mathbb{I}\{M_{il} = M'_{il} = 1\} $$

where $`m_j`$ is the size of cluster $`C_j`$.

The **prediction strength** for $`K`$ is:

$$ PS(K) = \min_j PS_j $$

**Intuition**: For each group, we measure what fraction of item pairs that were together in the first organization are still together in the second organization. The prediction strength is the worst-performing group - the group where the most pairs got separated. This is conservative: if even one group is unstable, the whole organization is considered unstable.

### Choosing K

Select the largest $`K`$ such that $`PS(K)`$ exceeds a threshold (e.g., 0.8).

**Intuition**: This is like saying "I want the most detailed organization possible, but only if it's stable." You start with a small number of groups and keep adding more until the organization becomes unstable. The largest stable organization is your optimal K.

### Python Example

**Implementation:** See `prediction_strength()` and `compute_prediction_strength_range()` functions in [choice_of_k_implementation.py](code/choice_of_k_implementation.py)

The functions implement the complete prediction strength algorithm with data splitting, clustering, and pair-wise agreement computation for determining optimal K.

### R Example

**Implementation:** See `prediction_strength()` and `compute_prediction_strength_range()` functions in [r_choice_of_k_implementation.R](code/r_choice_of_k_implementation.R)

The functions implement the complete prediction strength algorithm with data splitting, clustering, and pair-wise agreement computation for determining optimal K using R's native functions.

---

## 6.3.5. Summary and Best Practices

- **Gap statistic**: Compares clustering to a null reference; robust but computationally intensive - like comparing your organization to random organization
- **Silhouette**: Measures cohesion/separation; easy to interpret and compute - like measuring how well each item fits in its group
- **Prediction strength**: Measures stability; good for practical validation - like testing if your organization is stable and reproducible
- **No single method is perfect**; use multiple criteria and domain knowledge - like using multiple tests to evaluate your organization
- **Visualize**: Always inspect cluster assignments and validation plots - like looking at your organization to make sure it makes sense

**Intuition**: Each method has its strengths and weaknesses. The gap statistic is like comparing your organization to random organization - it's robust but takes more work. The silhouette is like measuring how well each item fits - it's easy to understand and compute. Prediction strength is like testing if your organization is stable - it's good for practical validation. The best approach is to use multiple methods and combine them with your knowledge of the data.

## Code Files Summary

The following code files contain the complete implementations for choosing the optimal number of clusters K:

### Python Files
- **[choice_of_k_implementation.py](code/choice_of_k_implementation.py)**: Main implementation with gap statistics, silhouette analysis, prediction strength, and comprehensive K selection methods - like a complete toolkit for evaluating different numbers of groups

### R Files
- **[r_choice_of_k_implementation.R](code/r_choice_of_k_implementation.R)**: Complete R implementation with gap statistics, silhouette analysis, prediction strength, and ggplot2 visualizations - like a complete R toolkit for evaluating different numbers of groups

### Key Features Implemented
- **Gap Statistic**: Complete implementation with uniform and PCA-based reference data generation, one-standard-error rule for optimal K selection - like tools for comparing your organization to random organization
- **Silhouette Analysis**: Comprehensive silhouette computation with individual sample scores and visualization capabilities - like tools for measuring how well each item fits in its group
- **Prediction Strength**: Full implementation with data splitting, clustering stability assessment, and threshold-based K selection - like tools for testing if your organization is stable
- **Comprehensive K Selection**: Multi-method approach combining all techniques for robust K selection - like using multiple tests to find the best number of groups
- **Visualization Tools**: Publication-quality plots for gap statistics, silhouette analysis, and prediction strength using matplotlib/seaborn and ggplot2 - like visual tools for understanding how well different numbers of groups work
- **Method Comparison**: Systematic comparison of different K selection methods on various data types - like comparing different evaluation methods to see which works best
- **Robust Implementation**: Error handling, reproducibility controls, and comprehensive documentation - like reliable tools that work consistently
- **Demonstration Functions**: Complete examples with synthetic data and real-world application scenarios - like worked examples showing how to choose the right number of groups

## References

- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(2), 411-423.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.
- Tibshirani, R., & Walther, G. (2005). Cluster validation by prediction strength. Journal of Computational and Graphical Statistics, 14(3), 511-528.
