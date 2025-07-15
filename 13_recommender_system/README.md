# 13. Recommender Systems

This section covers modern recommender systems, from fundamental approaches to cutting-edge deep learning methods. Recommender systems are essential tools for filtering information and providing personalized recommendations across various platforms.

## Contents

### 13.1. Introduction
- **File**: `01_introduction.md`
- **Topics**:
  - Definition and applications of recommender systems
  - Real-world examples (Amazon, Netflix, YouTube, Spotify, etc.)
  - Evolution from non-personalized to personalized systems
  - Three main approaches: content-based, collaborative filtering, latent factor models
  - Modern developments with deep learning

### 13.2. Content-Based Methods
- **File**: `02_content-based.md`
- **Topics**:
  - Item and user profile construction
  - Feature space representation
  - Proximity-based recommendations
  - Advantages: immediate recommendations, good for new items, transparency
  - Limitations: predictable recommendations, feature engineering challenges

### 13.3. Collaborative Filtering
- **File**: `03_collaborative_filtering.md`
- **Topics**:
  - Rating matrix and interaction data
  - Explicit vs. implicit feedback
  - Matrix completion problem
  - Similarity metrics: Jaccard, Cosine, Centered Cosine (Pearson)
  - Key issues with missing data and similarity calculations

### 13.4. UBCF and IBCF
- **File**: `04_ubcf-ibcf.md`
- **Topics**:
  - **User-Based CF (UBCF)**: Similar users like similar items
  - **Item-Based CF (IBCF)**: Users like items similar to their preferences
  - Mathematical formulations and algorithms
  - Pros and cons of collaborative filtering
  - Computational considerations and real-world applications

### 13.5. Latent Factor Model
- **File**: `05_latent_factor.md`
- **Topics**:
  - SVD-based matrix decomposition
  - User and item embedding in shared space
  - Handling missing entries with regularization
  - Baseline models for bias correction
  - Gradient descent optimization

### 13.6. Challenges and Strategies
- **File**: `06_challenges.md`
- **Topics**:
  - Computational efficiency through pre-clustering
  - Combining multiple recommender systems
  - Contextual information (location, time, device)
  - Evaluation metrics (RMSE, Precision@K, diversity, serendipity)
  - User feedback incorporation
  - Cold start and scalability challenges

### 13.7. Deep Recommender Systems
- **File**: `07_deep_recommender_systems.md`
- **Topics**:
  - Deep learning for latent feature creation
  - Autoencoders for nonlinear embedding
  - Two-stage recommender systems (filtering + ranking)
  - Google's Deep and Wide Model
  - Real-world applications (Spotify, modern platforms)

## Key Concepts

### Fundamental Approaches
- **Content-Based**: Uses item features and user preferences
- **Collaborative Filtering**: Leverages user-item interactions
- **Latent Factor Models**: Matrix decomposition approaches

### Similarity Metrics
- **Jaccard Similarity**: $\frac{|A \cap B|}{|A \cup B|}$
- **Cosine Similarity**: $\frac{u^t v}{\| u\| \cdot \| v\|}$
- **Centered Cosine**: $\frac{(u - \bar{u})^t (v - \bar{v})}{\| u - \bar{u} \| \cdot \| v- \bar{v} \|}$

### Matrix Decomposition
$$R_{m \times n} \approx U_{m \times d} V^t_{d \times n}$$

### Loss Function with Regularization
$$\sum_{R_{ij} \ne ?} (R_{ij} - u_i^t v_j)^2 + \lambda_1 \text{Pen}(U) + \lambda_2 \text{Pen}(V)$$

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
