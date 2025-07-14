# Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## Overview

This module covers fundamental clustering analysis techniques and algorithms used in unsupervised learning. Clustering aims to group similar objects together while keeping dissimilar objects in different groups, without prior knowledge of the true cluster assignments.

## Topics Covered

### 1. Distance Measures
- **Euclidean Distance**: Standard L2 distance for numerical data
- **L-infinity Distance**: Maximum absolute difference across dimensions
- **Jaccard Distance**: For set-based similarity and text data
- **Hamming Distance**: For strings of equal length
- **Edit Distance**: For string transformations
- **Cosine Distance**: Angle-based similarity between vectors
- **Multidimensional Scaling (MDS)**: Converting distance matrices to coordinate representations

### 2. K-means and K-medoids
- **K-means Algorithm**: Iterative partitioning with mean-based centers
- **Objective Function**: Within-cluster sum of squares optimization
- **Convergence Properties**: Local minima and initialization sensitivity
- **Dimension Reduction**: PCA and Random Projection techniques
- **K-medoids (PAM)**: Medoid-based clustering using actual data points
- **Alternative Distance Measures**: Extending beyond Euclidean distance

### 3. Choice of K
- **Gap Statistics**: Comparing observed vs. reference clustering
- **Silhouette Statistics**: Measuring cluster cohesion and separation
- **Prediction Strength**: Cross-validation approach for cluster validation
- **Elbow Method**: Visual inspection of within-cluster variation

### 4. Hierarchical Clustering
- **Agglomerative Approach**: Bottom-up cluster construction
- **Linkage Methods**:
  - Single-linkage: Minimum distance between clusters
  - Complete-linkage: Maximum distance between clusters
  - Average-linkage: Mean distance between clusters
- **Dendrogram Visualization**: Tree representation of clustering hierarchy
- **Flexible K Selection**: Post-clustering choice of number of clusters

## Files

### Documentation
- `01_distance_measures.md` - Comprehensive coverage of distance metrics and MDS
- `02_k-means.md` - K-means algorithm, variants, and optimization
- `03_choice_of_k.md` - Methods for determining optimal number of clusters
- `04_hierarchical_clustering.md` - Hierarchical clustering approaches

### Code Examples
- `Python_W6_Cluster.html` - Complete Python implementation with examples
- `Rcode_W6_Cluster.html` - Complete R implementation with examples

## Key Concepts

### Distance Measures
Distance functions must satisfy:
1. Non-negativity: d(x,z) ≥ 0 and d(x,z) = 0 iff x = z
2. Symmetry: d(x,z) = d(z,x)
3. Triangle inequality: d(x,y) ≤ d(x,z) + d(z,y)

### K-means Algorithm
1. **Initialize**: Choose K cluster centers
2. **Assign**: Assign each point to nearest center
3. **Update**: Recalculate centers as means of assigned points
4. **Repeat**: Until convergence

### Hierarchical Clustering
- Starts with n singleton clusters
- Iteratively merges closest clusters
- Produces dendrogram for flexible K selection
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
- Tibshirani, Walther, and Hastie (2001) - Gap Statistics
- Rousseeuw (1987) - Silhouette Statistics
- Tibshirani and Walther (2005) - Prediction Strength

## Related Modules

- [Linear Regression](../02_linear_regression/) - Supervised learning foundation
- [Variable Selection](../03_variable_selection_regularization/) - Feature selection techniques
- [Nonlinear Regression](../05_nonlinear_regression/) - Advanced regression methods 