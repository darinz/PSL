# Latent Structure Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)
[![Topic](https://img.shields.io/badge/Topic-Unsupervised%20Learning-purple.svg)]()

## Overview

This module explores advanced, modernized unsupervised learning techniques that model latent (hidden) structures in data. The content has been expanded and clarified for accessibility, with detailed mathematical derivations, code explanations, and improved formatting using inline ($`...`$) and display math (```math) LaTeX. Where possible, image-based equations and text have been converted to selectable, copyable LaTeX in the markdown files for clarity and accessibility.

## Topics Covered

### 1. Model-based Clustering
- **Mixture Model Framework**: Clustering as fitting mixture distributions
- **Gaussian Mixtures**: Normal component distributions for continuous data
- **Model Selection**: Using AIC/BIC for optimal number of components
- **Parameter Estimation**: Maximum Likelihood Estimation challenges
- **Expanded mathematical derivations and LaTeX formatting**

### 2. Mixture Models
- **Mathematical Foundation**: Weighted sum of probability density functions
- **Data Generation Process**: Two-stage latent variable framework
- **Two-Component Gaussian**: Detailed analysis of simple case
- **Kullback-Leibler Divergence**: Distance measure between distributions
- **Iterative Estimation**: Expectation-Maximization approach
- **Expanded code and math explanations**

### 3. EM Algorithm
- **Expectation-Maximization**: Iterative optimization for latent variables
- **E-step**: Computing expected values of latent variables
- **M-step**: Maximizing likelihood with respect to parameters
- **Convergence Properties**: Monotonic improvement of marginal likelihood
- **Connection to K-means**: Special case with zero variance
- **Variational EM**: Approximate inference for complex models
- **Expanded code and LaTeX math explanations**

### 4. Latent Dirichlet Allocation (LDA)
- **Document Modeling**: Topic-based representation of text collections
- **Bag of Words**: Frequency-based document representation
- **Topic Distributions**: Probability distributions over vocabulary
- **Document-Topic Mixtures**: Flexible topic assignments per document
- **Word-Level Latent Variables**: Individual word topic assignments
- **Dimensionality Reduction**: Compact document representations
- **Expanded code and math explanations**

### 5. Hidden Markov Models (HMM)
- **Sequential Data**: Modeling temporal dependencies
- **Hidden States**: Latent variables following Markov process
- **Emission Probabilities**: Observable variable distributions
- **Transition Matrix**: State-to-state transition probabilities
- **Forward-Backward Algorithm**: Computing marginal probabilities
- **Viterbi Algorithm**: Finding most likely state sequence
- **Baum-Welch Algorithm**: Parameter estimation via EM
- **Expanded code and LaTeX math explanations**

## Recent Enhancements

- **Expanded Explanations:** All modules now feature clearer, more detailed explanations of mathematical concepts and algorithms.
- **LaTeX Math Formatting:** All math is now formatted using inline ($`...`$) and display (```math) LaTeX for readability and copy-paste support.
- **Code Examples:** Python and R code snippets are provided and explained for all major algorithms.
- **Image-to-Text Conversion:** PNG images containing math or text have been transcribed into markdown with LaTeX where possible, improving accessibility.
- **Visual Aids:** Diagrams and figures are referenced and described in context to support conceptual understanding.

## Key Concepts

### Mixture Model Structure
$`f(x) = \sum_{k=1}^K \pi_k f_k(x \mid \theta_k)`$

### EM Algorithm Steps
1. **E-step**: Compute $`Q(\theta|\theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]`$
2. **M-step**: Maximize $`\theta^{(t+1)} = \arg\max_{\theta} Q(\theta|\theta^{(t)})`$

### LDA Model Components
- **Topic Matrix**: $`V \times K`$ matrix of word distributions per topic
- **Document-Topic Matrix**: $`K \times n`$ matrix of topic weights per document
- **Word-Topic Assignments**: Individual word-level latent variables

### HMM Parameters
- **Initial Distribution**: $`w_{m_z \times 1}`$ for $`Z_1`$
- **Transition Matrix**: $`A_{m_z \times m_z}`$ for $`Z_t \rightarrow Z_{t+1}`$
- **Emission Matrix**: $`B_{m_z \times m_x}`$ for $`Z_t \rightarrow X_t`$

## Files

### Documentation
- `01_model-based_clustering.md` – Introduction to mixture model clustering
- `02_mixture_models.md` – Mathematical foundations and estimation
- `03_em_algorithm.md` – Expectation-Maximization algorithm details
- `04_latent_dirichlet_allocation_model.md` – LDA for document modeling
- `05_hidden_markov_models.md` – HMM for sequential data analysis

## Applications

### Model-based Clustering
- Customer segmentation with uncertainty
- Image segmentation with probabilistic assignments
- Anomaly detection in complex systems

### Mixture Models
- Financial data modeling
- Biological data analysis
- Sensor data clustering

### LDA Applications
- Document classification and clustering
- Topic modeling in social media
- Information retrieval systems
- Content recommendation

### HMM Applications
- Speech recognition and synthesis
- Bioinformatics (gene finding, protein structure)
- Financial time series analysis
- Natural language processing
- Computer vision (gesture recognition)

## Algorithmic Complexity

### EM Algorithm
- **Time Complexity**: $`O(n \times K \times d \times \text{iterations})`$ for $`n`$ samples, $`K`$ components, $`d`$ dimensions
- **Space Complexity**: $`O(n \times K)`$ for storing responsibilities
- **Convergence**: Guaranteed to improve likelihood monotonically

### LDA
- **Time Complexity**: $`O(n \times K \times V \times \text{iterations})`$ for $`n`$ documents, $`K`$ topics, $`V`$ vocabulary
- **Space Complexity**: $`O(K \times V + n \times K)`$ for topic and document matrices
- **Variational Inference**: Approximate but scalable

### HMM
- **Forward-Backward**: $`O(n \times m^2)`$ for $`n`$ observations, $`m`$ states
- **Viterbi**: $`O(n \times m^2)`$ for most likely path
- **Baum-Welch**: $`O(n \times m^2 \times \text{iterations})`$ for parameter estimation

## Prerequisites

- Understanding of probability theory and statistics
- Familiarity with clustering algorithms (K-means)
- Knowledge of optimization techniques
- Basic linear algebra and calculus
- Experience with Python or R programming

## Related Modules

- [Clustering Analysis](../06_clustering_analysis/) – Foundation clustering methods
- [Variable Selection](../03_variable_selection_regularization/) – Feature selection techniques
- [Nonlinear Regression](../05_nonlinear_regression/) – Advanced modeling approaches

## References

- Dempster, Laird, and Rubin (1977) – EM Algorithm
- Blei, Ng, and Jordan (2003) – Latent Dirichlet Allocation
- Rabiner (1989) – Hidden Markov Models Tutorial
- McLachlan and Peel (2000) – Finite Mixture Models
- Bishop (2006) – Pattern Recognition and Machine Learning

> **Note:** Where images previously contained mathematical expressions or text, these have been transcribed into markdown with LaTeX for clarity and accessibility. Visuals are now referenced in context to support the expanded explanations. 