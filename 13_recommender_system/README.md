# 13. Recommender Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/darinz/Statistical-Learning)

This section covers modern, expanded recommender systems, from fundamental approaches to cutting-edge deep learning methods.

## Example Implementations

- [Movie-Rec-Sys (Streamlit Movie Recommendation App)](https://github.com/darinz/Movie-Rec-Sys): A Streamlit-based movie recommendation system with interactive UI and content-based filtering.
- [Movie-Recommender (Python MovieLens Recommender)](https://github.com/darinz/Movie-Recommender): A Python implementation of a recommender system using the MovieLens dataset and collaborative filtering.

## Contents

### 13.1. Introduction
- **File**: `01_introduction.md`
- **Topics**:
  - Definition and applications of recommender systems
  - Real-world examples (Amazon, Netflix, YouTube, Spotify, etc.)
  - Evolution from non-personalized to personalized systems
  - Three main approaches: content-based, collaborative filtering, latent factor models
  - Modern developments with deep learning
  - Expanded mathematical derivations and LaTeX formatting

### 13.2. Content-Based Methods
- **File**: `02_content-based.md`
- **Topics**:
  - Item and user profile construction
  - Feature space representation
  - Proximity-based recommendations
  - Advantages: immediate recommendations, good for new items, transparency
  - Limitations: predictable recommendations, feature engineering challenges
  - Enhanced code and math explanations

### 13.3. Collaborative Filtering
- **File**: `03_collaborative_filtering.md`
- **Topics**:
  - Rating matrix and interaction data
  - Explicit vs. implicit feedback
  - Matrix completion problem
  - Similarity metrics: Jaccard, Cosine, Centered Cosine (Pearson)
  - Key issues with missing data and similarity calculations
  - Expanded code and math explanations

### 13.4. UBCF and IBCF
- **File**: `04_ubcf-ibcf.md`
- **Topics**:
  - **User-Based CF (UBCF)**: Similar users like similar items
  - **Item-Based CF (IBCF)**: Users like items similar to their preferences
  - Mathematical formulations and algorithms
  - Pros and cons of collaborative filtering
  - Computational considerations and real-world applications
  - Enhanced code and math explanations

### 13.5. Latent Factor Model
- **File**: `05_latent_factor.md`
- **Topics**:
  - SVD-based matrix decomposition
  - User and item embedding in shared space
  - Handling missing entries with regularization
  - Baseline models for bias correction
  - Gradient descent optimization
  - Expanded code and math explanations

### 13.6. Challenges and Strategies
- **File**: `06_challenges.md`
- **Topics**:
  - Computational efficiency through pre-clustering
  - Combining multiple recommender systems
  - Contextual information (location, time, device)
  - Evaluation metrics (RMSE, Precision@K, diversity, serendipity)
  - User feedback incorporation
  - Cold start and scalability challenges
  - Enhanced code and math explanations

### 13.7. Deep Recommender Systems
- **File**: `07_deep_recommender_systems.md`
- **Topics**:
  - Deep learning for latent feature creation
  - Autoencoders for nonlinear embedding
  - Two-stage recommender systems (filtering + ranking)
  - Google's Deep and Wide Model
  - Real-world applications (Spotify, modern platforms)
  - Expanded code and math explanations

## Key Concepts

### Fundamental Approaches
- **Content-Based**: Uses item features and user preferences
- **Collaborative Filtering**: Leverages user-item interactions
- **Latent Factor Models**: Matrix decomposition approaches

### Similarity Metrics
- **Jaccard Similarity**: $`\frac{|A \cap B|}{|A \cup B|}`$
- **Cosine Similarity**: $`\frac{u^t v}{\| u\| \cdot \| v\|}`$
- **Centered Cosine**: $`\frac{(u - \bar{u})^t (v - \bar{v})}{\| u - \bar{u} \| \cdot \| v- \bar{v} \|}`$

### Matrix Decomposition
$`R_{m \times n} \approx U_{m \times d} V^t_{d \times n}`$

### Loss Function with Regularization
$`\sum_{R_{ij} \ne ?} (R_{ij} - u_i^t v_j)^2 + \lambda_1 \text{Pen}(U) + \lambda_2 \text{Pen}(V)`$

## Recent Enhancements

- **Expanded Explanations:** All modules now feature clearer, more detailed explanations of mathematical concepts and algorithms.
- **LaTeX Math Formatting:** All math is now formatted using inline ($`...`$) and display (```math) LaTeX for readability and copy-paste support.
- **Code Examples:** Python and R code snippets are provided and explained for all major algorithms.
- **Image-to-Text Conversion:** PNG images containing math or text have been transcribed into markdown with LaTeX where possible, improving accessibility.
- **Visual Aids:** Diagrams and figures are referenced and described in context to support conceptual understanding.

## Practical Considerations

### When to Use Different Methods
- **Content-Based**: When item features are well-defined and available
- **Collaborative Filtering**: When user interaction data is abundant
- **Latent Factor Models**: For large-scale systems with sparse data
- **Deep Learning**: For complex, high-dimensional data

### Evaluation Metrics
- **RMSE**: Traditional prediction accuracy
- **Precision@K**: Top-K recommendation accuracy
- **Diversity**: Variety in recommendations
- **Serendipity**: Discovery of unexpected items

### Challenges
- **Cold Start**: New users/items without interaction data
- **Scalability**: Handling large datasets efficiently
- **Sparsity**: Most user-item pairs have no interaction
- **Bias**: Popularity bias and filter bubbles

## Modern Developments

### Deep Learning Approaches
- **Neural Collaborative Filtering**: Deep learning for CF
- **Autoencoders**: Nonlinear dimensionality reduction
- **Two-Stage Systems**: Filtering + ranking architecture
- **Contextual Models**: Incorporating additional features

### Industry Applications
- **Netflix**: Movie and TV show recommendations
- **Spotify**: Music recommendation with audio features
- **Amazon**: Product recommendations
- **YouTube**: Video recommendations

## Related Sections
- **Week 6**: Clustering Analysis (for user/item clustering)
- **Week 9**: Discriminant Analysis (classification approaches)
- **Week 10**: Logistic Regression (ranking models)

## Code Resources
- Python implementations for collaborative filtering
- R packages for recommender systems
- Deep learning frameworks (TensorFlow, PyTorch)
- Real-world datasets and examples

## Additional Resources

- [Dive into Deep Learning: Recommender Systems](https://d2l.ai/chapter_recommender-systems/index.html)
- [Wikipedia: Recommender system](https://en.wikipedia.org/wiki/Recommender_system)
- [NVIDIA Glossary: Recommendation System](https://www.nvidia.com/en-us/glossary/recommendation-system/)
- [Google Developers: Types of Recommendation Systems](https://developers.google.com/machine-learning/recommendation/overview/types)
