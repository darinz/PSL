# Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

## Overview

This module covers fundamental and modernized clustering analysis techniques and algorithms used in unsupervised learning.

## Topics Covered

### 1. Distance Measures
- **Euclidean Distance**: Standard L2 distance for numerical data

  ![Euclidean distance between two points in 2D space.](../_images/w6_euclidean_dist.png)
  *Figure: Euclidean distance between two points in 2D space.*

- **L-infinity Distance**: Maximum absolute difference across dimensions
- **Jaccard Distance**: For set-based similarity and text data
- **Hamming Distance**: For strings of equal length
- **Edit Distance**: For string transformations
- **Cosine Distance**: Angle-based similarity between vectors
- **Multidimensional Scaling (MDS)**: Converting distance matrices to coordinate representations

  ![Multidimensional scaling (MDS) double centering.](../_images/w6_mds_double_centering.png)
  *Figure: Multidimensional scaling (MDS) double centering.*

- **Expanded mathematical derivations and LaTeX formatting**

### 2. K-means and K-medoids
- **K-means Algorithm**: Iterative partitioning with mean-based centers

  ![K-means clustering process.](../_images/w6_kmeans.png)
  *Figure: K-means clustering process.*

- **Objective Function**: Within-cluster sum of squares optimization
- **Convergence Properties**: Local minima and initialization sensitivity

  ![K-means can converge to local minima depending on initialization.](../_images/w6_kmeans_local_minimal.png)
  *Figure: K-means can converge to local minima depending on initialization.*

- **Dimension Reduction**: PCA and Random Projection techniques
- **K-medoids (PAM)**: Medoid-based clustering using actual data points
- **Alternative Distance Measures**: Extending beyond Euclidean distance
- **Expanded code and math explanations**

### 3. Choice of K
- **Gap Statistics**: Comparing observed vs. reference clustering
- **Silhouette Statistics**: Measuring cluster cohesion and separation

  ![Silhouette definition for cluster validation.](../_images/w6_sil_def.png)
  *Figure: Silhouette definition for cluster validation.*

- **Prediction Strength**: Cross-validation approach for cluster validation
- **Elbow Method**: Visual inspection of within-cluster variation
- **Enhanced visual and LaTeX math explanations**

### 4. Hierarchical Clustering
- **Agglomerative Approach**: Bottom-up cluster construction
- **Linkage Methods**:
  - Single-linkage: Minimum distance between clusters
  - Complete-linkage: Maximum distance between clusters
  - Average-linkage: Mean distance between clusters

  ![Single linkage and complete linkage in hierarchical clustering.](../_images/w6_hist_single.png)
  *Figure: Single linkage in hierarchical clustering.*

  ![Complete linkage in hierarchical clustering.](../_images/w6_hist_complete.png)
  *Figure: Complete linkage in hierarchical clustering.*

- **Dendrogram Visualization**: Tree representation of clustering hierarchy
- **Flexible K Selection**: Post-clustering choice of number of clusters
- **Expanded code and math explanations**

  ![Dendrogram example for hierarchical clustering.](../_images/w6_kmeans.png)
  *Figure: Dendrogram example for hierarchical clustering.*

## Files

### Documentation
- `01_distance_measures.md` – Comprehensive coverage of distance metrics and MDS
- `02_k-means.md` – K-means algorithm, variants, and optimization
- `03_choice_of_k.md` – Methods for determining optimal number of clusters
- `04_hierarchical_clustering.md` – Hierarchical clustering approaches

### Code Examples
- `Python_W6_Cluster.html` – Complete Python implementation with examples
- `Rcode_W6_Cluster.html` – Complete R implementation with examples

## Key Concepts

### Distance Measures
Distance functions must satisfy:
1. **Non-negativity**: $`d(x,z) \geq 0`$ and $`d(x,z) = 0`$ iff $`x = z`$
2. **Symmetry**: $`d(x,z) = d(z,x)`$
3. **Triangle inequality**: $`d(x,y) \leq d(x,z) + d(z,y)`$

### K-means Algorithm
1. **Initialize**: Choose $`K`$ cluster centers
2. **Assign**: Assign each point to nearest center
3. **Update**: Recalculate centers as means of assigned points
4. **Repeat**: Until convergence

### Hierarchical Clustering
- Starts with $`n`$ singleton clusters
- Iteratively merges closest clusters
- Produces dendrogram for flexible $`K`$ selection
- Different linkage methods produce varying cluster shapes

## Applications

Clustering analysis is widely used in:
- Customer segmentation
- Image segmentation
- Document clustering
- Bioinformatics
- Market research
- Anomaly detection

## Prerequisites

- Basic understanding of linear algebra
- Familiarity with Python or R
- Knowledge of statistical concepts
- Understanding of optimization principles

## References

- Elements of Statistical Learning (ESL)
- Introduction to Statistical Learning (ISL)
- Tibshirani, Walther, and Hastie (2001) – Gap Statistics
- Rousseeuw (1987) – Silhouette Statistics
- Tibshirani and Walther (2005) – Prediction Strength

## Related Modules

- [Linear Regression](../02_linear_regression/) – Supervised learning foundation
- [Variable Selection](../03_variable_selection_regularization/) – Feature selection techniques
- [Nonlinear Regression](../05_nonlinear_regression/) – Advanced regression methods 