# 13.1. Introduction

In this section, we're going to delve into the world of recommender systems. Let's start with its definition, which you might find on Wikipedia. Whether you call it a recommender system, a recommendation system, a recommender engine, or a recommendation platform, the core function remains consistent: to filter vast amounts of information and present users with options that align with their preferences.

## 13.1.1. What is a Recommender System?

A **recommender system** is an information filtering system that seeks to predict the "rating" or "preference" that a user would give to an item. The goal is to provide personalized recommendations that help users discover items they might be interested in but haven't encountered yet.

### Formal Definition

Mathematically, a recommender system can be formalized as follows:

```math
f: \mathcal{U} \times \mathcal{I} \rightarrow \mathcal{R}
```

where:
- $`\mathcal{U}`$ is the set of users
- $`\mathcal{I}`$ is the set of items
- $`\mathcal{R}`$ is the set of possible ratings/preferences
- $`f(u, i)`$ predicts the rating that user $`u`$ would give to item $`i`$

### Core Components

1. **Users** ($`u \in \mathcal{U}`$): The individuals receiving recommendations
2. **Items** ($`i \in \mathcal{I}`$): The objects being recommended (products, movies, songs, etc.)
3. **Ratings** ($`r_{ui} \in \mathcal{R}`$): Explicit or implicit feedback from users
4. **Prediction Function** ($`f(u, i)`$): The algorithm that generates recommendations

## 13.1.2. The Recommender System Landscape

We live in an age where recommender systems are woven into the fabric of our digital experiences. Visit any e-commerce site—Amazon, Wayfair, Walmart—and you'll encounter product suggestions tailored to your interests. This extends to entertainment and social platforms as well: Netflix curates our watchlists, YouTube and Google News personalize our feeds, Pinterest enhances our visual discoveries, Spotify selects music for our tastes, Facebook suggests friends, LinkedIn connects us with professional contacts. And let's not overlook the world of online dating, with platforms like OkCupid, which leverages these systems to suggest potential romantic matches.

### Real-World Applications

#### E-commerce Platforms
- **Amazon**: Product recommendations based on purchase history, browsing behavior, and similar users
- **Netflix**: Movie and TV show recommendations using collaborative filtering and content-based methods
- **Spotify**: Music recommendations using audio features and listening patterns
- **YouTube**: Video recommendations based on watch history and user engagement

#### Social Media
- **Facebook**: Friend suggestions, content recommendations
- **LinkedIn**: Professional connections, job recommendations
- **Instagram**: Content and user recommendations
- **Twitter**: Tweet and user recommendations

#### Specialized Platforms
- **Dating Apps**: Partner matching using preference learning
- **News Aggregators**: Article recommendations based on reading history
- **Travel Platforms**: Destination and accommodation recommendations

## 13.1.3. Historical Evolution

### Non-Personalized Systems

Historically, we've seen non-personalized recommender systems, such as generic top-five lists—for instance, "Top Five Winter Boots for Women" or "Best Cyber Monday Deals." These recommendations were grounded in either expert knowledge or simple aggregated statistics like best-selling books.

**Mathematical Formulation**:
```math
\text{Recommendation}(i) = \text{Popularity}(i) = \frac{\sum_{u \in \mathcal{U}} I(r_{ui} > 0)}{|\mathcal{U}|}
```

where $`I(\cdot)`$ is the indicator function.

**Limitations**:
- No personalization
- Popularity bias
- Cold start problem for new items
- Ignores individual preferences

### The Rise of Personalization

The goal, then, is to develop personalized recommender systems. We will explore fundamental techniques such as:

- **Content-based methods**: Analyze item attributes and user preferences
- **Collaborative filtering**: Leverage user-item interaction patterns
- **Latent factor models**: Discover hidden patterns in the data

## 13.1.4. Core Recommendation Paradigms

### 1. Content-Based Filtering

Content-based methods focus on item attributes and user preferences:

```math
\text{Similarity}(i, j) = \text{sim}(\text{features}(i), \text{features}(j))
```

**Key Components**:
- **Item Profiles**: Feature vectors describing item characteristics
- **User Profiles**: Feature vectors representing user preferences
- **Similarity Metrics**: Cosine similarity, Euclidean distance, etc.

**Advantages**:
- No cold start for new users
- Interpretable recommendations
- Can handle new items with known features

**Disadvantages**:
- Requires rich item metadata
- Limited to item features
- Overspecialization (filter bubble)

### 2. Collaborative Filtering

Collaborative filtering analyzes user-item interaction patterns:

#### User-Based CF
```math
\text{Prediction}(u, i) = \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
```

#### Item-Based CF
```math
\text{Prediction}(u, i) = \frac{\sum_{j \in N(i)} \text{sim}(i, j) \cdot r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i, j)|}
```

where $`N(u)`$ and $`N(i)`$ are neighborhoods of similar users and items, respectively.

**Advantages**:
- No need for item metadata
- Discovers serendipitous recommendations
- Leverages collective intelligence

**Disadvantages**:
- Cold start problem
- Sparsity issues
- Scalability challenges

### 3. Latent Factor Models

Latent factor models discover hidden patterns in the data:

```math
r_{ui} \approx \mathbf{u}_u^T \mathbf{v}_i + b_u + b_i + \mu
```

where:
- $`\mathbf{u}_u`$ is the user latent vector
- $`\mathbf{v}_i`$ is the item latent vector
- $`b_u, b_i`$ are user and item biases
- $`\mu`$ is the global mean rating

**Advantages**:
- Handles sparsity well
- Captures complex patterns
- Scalable to large datasets

**Disadvantages**:
- Less interpretable
- Requires sufficient data
- Sensitive to hyperparameters

## 13.1.5. Implementation Examples

### Python Implementation: Basic Recommender System Framework

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RecommenderSystem:
    def __init__(self, method='collaborative'):
        """
        Basic Recommender System
        
        Parameters:
        -----------
        method : str
            Recommendation method ('collaborative', 'content', 'latent')
        """
        self.method = method
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, ratings_df):
        """
        Fit the recommender system
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        if self.method == 'collaborative':
            self._fit_collaborative()
        elif self.method == 'latent':
            self._fit_latent()
        elif self.method == 'content':
            self._fit_content(ratings_df)
            
    def _fit_collaborative(self):
        """Fit collaborative filtering model"""
        # Compute user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Compute item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
    def _fit_latent(self, n_factors=10):
        """Fit latent factor model using NMF"""
        # Apply NMF for matrix factorization
        nmf = NMF(n_components=n_factors, random_state=42)
        self.user_factors = nmf.fit_transform(self.user_item_matrix)
        self.item_factors = nmf.components_.T
        
    def _fit_content(self, ratings_df):
        """Fit content-based model (simplified)"""
        # For simplicity, we'll use item popularity as content features
        item_popularity = ratings_df.groupby('item_id')['rating'].mean()
        self.item_features = item_popularity.to_dict()
        
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if self.method == 'collaborative':
            return self._predict_collaborative(user_id, item_id)
        elif self.method == 'latent':
            return self._predict_latent(user_id, item_id)
        elif self.method == 'content':
            return self._predict_content(user_id, item_id)
            
    def _predict_collaborative(self, user_id, item_id):
        """User-based collaborative filtering prediction"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.values.mean()
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Find similar users who rated this item
        user_ratings = self.user_item_matrix.iloc[:, item_idx]
        similar_users = self.user_similarity[user_idx]
        
        # Weighted average of similar users' ratings
        valid_ratings = user_ratings[user_ratings > 0]
        if len(valid_ratings) == 0:
            return self.user_item_matrix.values.mean()
            
        user_indices = valid_ratings.index
        similarities = [similar_users[self.user_item_matrix.index.get_loc(uid)] for uid in user_indices]
        
        weighted_sum = sum(sim * rating for sim, rating in zip(similarities, valid_ratings))
        total_similarity = sum(abs(sim) for sim in similarities)
        
        return weighted_sum / total_similarity if total_similarity > 0 else valid_ratings.mean()
        
    def _predict_latent(self, user_id, item_id):
        """Latent factor model prediction"""
        if user_id not in self.user_item_matrix.index or item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.values.mean()
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
    def _predict_content(self, user_id, item_id):
        """Content-based prediction (simplified)"""
        if item_id in self.item_features:
            return self.item_features[item_id]
        return self.user_item_matrix.values.mean()
        
    def recommend(self, user_id, n_recommendations=5):
        """Generate top-n recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find items the user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
            
        # Sort by predicted rating and return top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# Generate synthetic data
np.random.seed(42)
n_users = 100
n_items = 50
n_ratings = 1000

# Create synthetic ratings
user_ids = np.random.randint(0, n_users, n_ratings)
item_ids = np.random.randint(0, n_items, n_ratings)
ratings = np.random.randint(1, 6, n_ratings)  # 1-5 scale

# Create DataFrame
ratings_df = pd.DataFrame({
    'user_id': user_ids,
    'item_id': item_ids,
    'rating': ratings
})

# Remove duplicates
ratings_df = ratings_df.drop_duplicates(['user_id', 'item_id'])

print("Synthetic Ratings Dataset:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.3f}")

# Test different recommendation methods
methods = ['collaborative', 'latent', 'content']
results = {}

for method in methods:
    print(f"\n=== Testing {method.upper()} Filtering ===")
    
    # Initialize and fit model
    recommender = RecommenderSystem(method=method)
    recommender.fit(ratings_df)
    
    # Test predictions for a sample user
    test_user = 0
    recommendations = recommender.recommend(test_user, n_recommendations=5)
    
    print(f"Top 5 recommendations for user {test_user}:")
    for item_id, pred_rating in recommendations:
        print(f"  Item {item_id}: Predicted rating = {pred_rating:.3f}")
    
    # Evaluate on a few test cases
    test_cases = [
        (0, 10), (0, 20), (1, 15), (1, 25), (2, 30)
    ]
    
    predictions = []
    for user_id, item_id in test_cases:
        pred = recommender.predict(user_id, item_id)
        predictions.append(pred)
        print(f"  User {user_id}, Item {item_id}: Predicted = {pred:.3f}")
    
    results[method] = predictions

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Rating distribution
plt.subplot(1, 3, 1)
ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

# Plot 2: User-item matrix heatmap (sample)
plt.subplot(1, 3, 2)
sample_matrix = ratings_df.pivot_table(
    index='user_id', columns='item_id', values='rating', fill_value=0
).iloc[:20, :20]
sns.heatmap(sample_matrix, cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('User-Item Matrix (Sample)')
plt.xlabel('Item ID')
plt.ylabel('User ID')

# Plot 3: Method comparison
plt.subplot(1, 3, 3)
methods_list = list(results.keys())
predictions_matrix = np.array(list(results.values())).T

x = np.arange(len(test_cases))
width = 0.25

for i, method in enumerate(methods_list):
    plt.bar(x + i*width, predictions_matrix[:, i], width, label=method.capitalize())

plt.xlabel('Test Cases')
plt.ylabel('Predicted Rating')
plt.title('Method Comparison')
plt.xticks(x + width, [f'({u},{i})' for u, i in test_cases])
plt.legend()

plt.tight_layout()
plt.show()
```

### R Implementation

```r
# Recommender System Implementation in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)

# Generate synthetic data
set.seed(42)
n_users <- 100
n_items <- 50
n_ratings <- 1000

# Create synthetic ratings
user_ids <- sample(1:n_users, n_ratings, replace = TRUE)
item_ids <- sample(1:n_items, n_ratings, replace = TRUE)
ratings <- sample(1:5, n_ratings, replace = TRUE)

# Create data frame
ratings_df <- data.frame(
  user_id = user_ids,
  item_id = item_ids,
  rating = ratings
)

# Remove duplicates
ratings_df <- ratings_df[!duplicated(ratings_df[, c("user_id", "item_id")]), ]

# Create rating matrix
rating_matrix <- ratings_df %>%
  spread(item_id, rating, fill = 0) %>%
  select(-user_id) %>%
  as.matrix()

# Convert to realRatingMatrix for recommenderlab
rating_matrix_real <- as(rating_matrix, "realRatingMatrix")

# Test different recommendation methods
methods <- c("UBCF", "IBCF", "POPULAR")

results <- list()

for (method in methods) {
  cat("=== Testing", method, "===\n")
  
  # Train model
  model <- Recommender(rating_matrix_real, method = method)
  
  # Generate recommendations
  recommendations <- predict(model, rating_matrix_real[1:5], n = 5)
  
  # Display recommendations
  for (i in 1:5) {
    cat("User", i, "recommendations:", as(recommendations[i], "list")[[1]], "\n")
  }
  
  # Store results
  results[[method]] <- model
}

# Visualization
# Rating distribution
p1 <- ggplot(ratings_df, aes(x = factor(rating))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Rating Distribution",
       x = "Rating", y = "Count") +
  theme_minimal()

# User-item matrix heatmap (sample)
sample_matrix <- rating_matrix[1:20, 1:20]
sample_df <- expand.grid(
  user_id = 1:20,
  item_id = 1:20
)
sample_df$rating <- as.vector(sample_matrix)

p2 <- ggplot(sample_df, aes(x = item_id, y = user_id, fill = rating)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "User-Item Matrix (Sample)",
       x = "Item ID", y = "User ID") +
  theme_minimal()

# Combine plots
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)
```

## 13.1.6. Evaluation Metrics

### Rating Prediction Metrics

#### Mean Absolute Error (MAE)
```math
\text{MAE} = \frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} |r_{ui} - \hat{r}_{ui}|
```

#### Root Mean Square Error (RMSE)
```math
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2}
```

### Ranking Metrics

#### Precision@k
```math
\text{Precision@k} = \frac{|\text{Recommended items} \cap \text{Relevant items}|}{k}
```

#### Recall@k
```math
\text{Recall@k} = \frac{|\text{Recommended items} \cap \text{Relevant items}|}{|\text{Relevant items}|}
```

#### Normalized Discounted Cumulative Gain (NDCG)
```math
\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
```

where:
```math
\text{DCG@k} = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}
```

## 13.1.7. Challenges and Limitations

### 1. Cold Start Problem

**New User Problem**: How to recommend items to users with no interaction history?
```math
\text{Challenge}: \text{Recommend}(u_{\text{new}}, i) = ?
```

**New Item Problem**: How to recommend items that have no ratings?
```math
\text{Challenge}: \text{Recommend}(u, i_{\text{new}}) = ?
```

### 2. Sparsity

Most user-item matrices are extremely sparse:
```math
\text{Sparsity} = 1 - \frac{|\{(u,i): r_{ui} \text{ exists}\}|}{|\mathcal{U}| \times |\mathcal{I}|}
```

### 3. Scalability

As the number of users and items grows:
- Memory requirements increase quadratically
- Computational complexity becomes prohibitive
- Real-time recommendations become challenging

### 4. Bias and Fairness

- **Popularity Bias**: Popular items get recommended more often
- **Filter Bubble**: Users see only similar content
- **Demographic Bias**: Recommendations may favor certain groups

## 13.1.8. Modern Developments

### Deep Learning Approaches

#### Neural Collaborative Filtering (NCF)
```math
\hat{r}_{ui} = f(\mathbf{u}_u, \mathbf{v}_i) = \sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 [\mathbf{u}_u; \mathbf{v}_i] + \mathbf{b}_1) + \mathbf{b}_2)
```

#### Autoencoders
```math
\text{Encoder}: h = f(x) \\
\text{Decoder}: \hat{x} = g(h) \\
\text{Loss}: L = \|x - \hat{x}\|^2
```

### Context-Aware Recommendations

Incorporating contextual information:
```math
\hat{r}_{ui} = f(\mathbf{u}_u, \mathbf{v}_i, \mathbf{c}_t)
```

where $`\mathbf{c}_t`$ represents contextual features (time, location, mood, etc.).

### Multi-Objective Optimization

Balancing multiple objectives:
```math
L = \alpha \cdot L_{\text{accuracy}} + \beta \cdot L_{\text{diversity}} + \gamma \cdot L_{\text{fairness}}
```

## 13.1.9. The Netflix Prize Legacy

The Netflix Prize competition (2006-2009) was a landmark event that significantly advanced the field of recommender systems. The goal was to improve Netflix's recommendation algorithm by 10% in terms of RMSE.

**Key Contributions**:
1. **Ensemble Methods**: Combining multiple algorithms
2. **Matrix Factorization**: SVD and its variants
3. **Temporal Dynamics**: Modeling time-evolving preferences
4. **Neighborhood Methods**: User-based and item-based collaborative filtering

**Mathematical Impact**:
```math
\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2} < 0.8572
```

The winning team achieved a 10.06% improvement over Netflix's existing system.

## 13.1.10. Future Directions

### 1. Explainable AI

Making recommendations interpretable:
- **Attention Mechanisms**: Highlighting important features
- **Rule-Based Systems**: Generating human-readable explanations
- **Counterfactual Explanations**: "If you liked X, you might like Y because..."

### 2. Multi-Modal Recommendations

Incorporating multiple data types:
```math
\hat{r}_{ui} = f(\text{text}(i), \text{image}(i), \text{audio}(i), \mathbf{u}_u)
```

### 3. Reinforcement Learning

Learning optimal recommendation policies:
```math
\pi^*(s) = \arg\max_a Q^*(s, a)
```

where $`s`$ is the user state and $`a`$ is the recommended item.

### 4. Federated Learning

Privacy-preserving recommendations:
```math
\mathbf{w}_{\text{global}} = \frac{1}{N} \sum_{i=1}^N \mathbf{w}_i
```

where each client $`i`$ trains locally and only shares model updates.

## 13.1.11. Summary

Recommender systems have evolved from simple popularity-based methods to sophisticated AI-powered systems. The field continues to grow with:

1. **Diverse Applications**: From e-commerce to healthcare
2. **Advanced Algorithms**: Deep learning, reinforcement learning
3. **Ethical Considerations**: Fairness, transparency, privacy
4. **Real-World Impact**: Billions of recommendations served daily

### Key Takeaways

- **Personalization is Key**: Modern systems must provide tailored recommendations
- **Data is Crucial**: Quality and quantity of data determine system performance
- **Evaluation Matters**: Multiple metrics needed for comprehensive assessment
- **Scalability is Essential**: Systems must handle millions of users and items
- **Ethics is Important**: Consider bias, fairness, and user privacy

### Next Steps

In the following lectures, we will dive deeper into:
- Content-based filtering techniques
- Collaborative filtering algorithms
- Matrix factorization methods
- Advanced deep learning approaches
- Evaluation and deployment strategies

The journey into recommender systems is just beginning, and the opportunities for innovation are endless.
