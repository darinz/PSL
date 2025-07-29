# 13.4. User-Based vs Item-Based Collaborative Filtering

This section provides a detailed comparison between User-Based Collaborative Filtering (UBCF) and Item-Based Collaborative Filtering (IBCF), two fundamental approaches in recommendation systems.

## 13.4.1. Overview

### User-Based Collaborative Filtering (UBCF)

UBCF operates on the principle that users with similar preferences in the past will have similar preferences in the future. The algorithm finds users similar to the target user and recommends items that these similar users have liked.

### Item-Based Collaborative Filtering (IBCF)

IBCF operates on the principle that users will like items similar to those they have already rated positively. The algorithm finds items similar to those the target user has rated and recommends the most similar items.

## 13.4.2. Mathematical Formulation

### UBCF Prediction

For user $`u`$ and item $`i`$, the prediction is computed as:

```math
\hat{r}_{ui} = \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot r_{vi}}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
```

where:
- $`N(u)`$ is the neighborhood of users similar to user $`u`$
- $`\text{sim}(u, v)`$ is the similarity between users $`u`$ and $`v`$
- $`r_{vi}`$ is the rating of user $`v`$ for item $`i`$

### IBCF Prediction

For user $`u`$ and item $`i`$, the prediction is computed as:

```math
\hat{r}_{ui} = \frac{\sum_{j \in N(i)} \text{sim}(i, j) \cdot r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i, j)|}
```

where:
- $`N(i)`$ is the neighborhood of items similar to item $`i`$
- $`\text{sim}(i, j)`$ is the similarity between items $`i`$ and $`j`$
- $`r_{uj}`$ is the rating of user $`u`$ for item $`j`$

## 13.4.3. Algorithm Comparison

### UBCF Algorithm Steps

1. **Compute User Similarities**: Calculate similarity between target user and all other users
2. **Select Neighborhood**: Choose top-$`k`$ most similar users
3. **Generate Prediction**: Weighted average of neighbors' ratings for target item

### IBCF Algorithm Steps

1. **Pre-compute Item Similarities**: Calculate similarity between all item pairs
2. **Select Neighborhood**: Choose top-$`k`$ most similar items for target item
3. **Generate Prediction**: Weighted average of user's ratings for similar items

## 13.4.4. Similarity Metrics

### User Similarity Metrics

#### 1. Pearson Correlation
```math
\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}
```

where $`I_{uv}`$ is the set of items rated by both users $`u`$ and $`v`$.

#### 2. Cosine Similarity
```math
\text{Cosine}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\| \cdot \|\mathbf{r}_v\|}
```

#### 3. Jaccard Similarity (for binary data)
```math
\text{Jaccard}(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}
```

where $`I_u`$ and $`I_v`$ are sets of items rated by users $`u`$ and $`v`$.

### Item Similarity Metrics

#### 1. Adjusted Cosine Similarity
```math
\text{AdjustedCosine}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}
```

where $`U_{ij}`$ is the set of users who rated both items $`i`$ and $`j`$.

#### 2. Pearson Correlation
```math
\text{Pearson}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)(r_{uj} - \bar{r}_j)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_i)^2} \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_j)^2}}
```

## 13.4.5. Implementation

### Python Implementation: UBCF vs IBCF Comparison

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class UBCF:
    """User-Based Collaborative Filtering"""
    
    def __init__(self, similarity_metric='pearson', k_neighbors=10):
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.user_similarity = None
        self.user_means = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the UBCF model"""
        # Create rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col, 
            fill_value=np.nan
        )
        
        # Compute user means
        self.user_means = self.rating_matrix.mean(axis=1)
        
        # Compute user similarities
        self.user_similarity = self._compute_user_similarity()
        
        return self
    
    def _compute_user_similarity(self):
        """Compute user similarity matrix"""
        n_users = len(self.rating_matrix)
        similarity_matrix = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i+1, n_users):
                # Get common rated items
                user_i_ratings = self.rating_matrix.iloc[i]
                user_j_ratings = self.rating_matrix.iloc[j]
                
                common_items = ~(user_i_ratings.isna() | user_j_ratings.isna())
                
                if common_items.sum() > 1:
                    if self.similarity_metric == 'pearson':
                        corr, _ = pearsonr(
                            user_i_ratings[common_items], 
                            user_j_ratings[common_items]
                        )
                        similarity_matrix[i, j] = corr
                        similarity_matrix[j, i] = corr
                    elif self.similarity_metric == 'cosine':
                        # Center ratings
                        user_i_centered = user_i_ratings[common_items] - self.user_means.iloc[i]
                        user_j_centered = user_j_ratings[common_items] - self.user_means.iloc[j]
                        
                        cosine_sim = np.dot(user_i_centered, user_j_centered) / (
                            np.linalg.norm(user_i_centered) * np.linalg.norm(user_j_centered)
                        )
                        similarity_matrix[i, j] = cosine_sim
                        similarity_matrix[j, i] = cosine_sim
                else:
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
        
        return similarity_matrix
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.rating_matrix.index or item_id not in self.rating_matrix.columns:
            return self.user_means.mean()
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        
        # Get user similarities
        user_similarities = self.user_similarity[user_idx]
        
        # Find users who rated this item
        item_ratings = self.rating_matrix.iloc[:, item_idx]
        rated_users = ~item_ratings.isna()
        
        if not rated_users.any():
            return self.user_means.mean()
        
        # Get similarities and ratings for users who rated this item
        similarities = user_similarities[rated_users]
        ratings = item_ratings[rated_users]
        
        # Sort by similarity and take top-k
        sorted_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(sorted_indices) == 0:
            return self.user_means.mean()
        
        top_similarities = similarities.iloc[sorted_indices]
        top_ratings = ratings.iloc[sorted_indices]
        
        # Weighted average
        weighted_sum = np.sum(top_similarities * top_ratings)
        total_similarity = np.sum(np.abs(top_similarities))
        
        if total_similarity == 0:
            return top_ratings.mean()
        
        return weighted_sum / total_similarity

class IBCF:
    """Item-Based Collaborative Filtering"""
    
    def __init__(self, similarity_metric='adjusted_cosine', k_neighbors=10):
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.item_similarity = None
        self.user_means = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the IBCF model"""
        # Create rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col, 
            fill_value=np.nan
        )
        
        # Compute user means
        self.user_means = self.rating_matrix.mean(axis=1)
        
        # Compute item similarities
        self.item_similarity = self._compute_item_similarity()
        
        return self
    
    def _compute_item_similarity(self):
        """Compute item similarity matrix"""
        n_items = len(self.rating_matrix.columns)
        similarity_matrix = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i+1, n_items):
                # Get common users
                item_i_ratings = self.rating_matrix.iloc[:, i]
                item_j_ratings = self.rating_matrix.iloc[:, j]
                
                common_users = ~(item_i_ratings.isna() | item_j_ratings.isna())
                
                if common_users.sum() > 1:
                    if self.similarity_metric == 'adjusted_cosine':
                        # Center by user means
                        item_i_centered = item_i_ratings[common_users] - self.user_means[common_users]
                        item_j_centered = item_j_ratings[common_users] - self.user_means[common_users]
                        
                        cosine_sim = np.dot(item_i_centered, item_j_centered) / (
                            np.linalg.norm(item_i_centered) * np.linalg.norm(item_j_centered)
                        )
                        similarity_matrix[i, j] = cosine_sim
                        similarity_matrix[j, i] = cosine_sim
                    elif self.similarity_metric == 'pearson':
                        corr, _ = pearsonr(
                            item_i_ratings[common_users], 
                            item_j_ratings[common_users]
                        )
                        similarity_matrix[i, j] = corr
                        similarity_matrix[j, i] = corr
                else:
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
        
        return similarity_matrix
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.rating_matrix.index or item_id not in self.rating_matrix.columns:
            return self.rating_matrix.mean().mean()
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        
        # Get item similarities
        item_similarities = self.item_similarity[item_idx]
        
        # Find items rated by this user
        user_ratings = self.rating_matrix.iloc[user_idx]
        rated_items = ~user_ratings.isna()
        
        if not rated_items.any():
            return self.rating_matrix.mean().mean()
        
        # Get similarities and ratings for items rated by this user
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Sort by similarity and take top-k
        sorted_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(sorted_indices) == 0:
            return ratings.mean()
        
        top_similarities = similarities.iloc[sorted_indices]
        top_ratings = ratings.iloc[sorted_indices]
        
        # Weighted average
        weighted_sum = np.sum(top_similarities * top_ratings)
        total_similarity = np.sum(np.abs(top_similarities))
        
        if total_similarity == 0:
            return top_ratings.mean()
        
        return weighted_sum / total_similarity

# Generate synthetic data with clear user/item clusters
np.random.seed(42)
n_users = 200
n_items = 100
n_ratings = 2000

# Create synthetic ratings with distinct user/item clusters
ratings_data = []
for user_id in range(n_users):
    n_user_ratings = np.random.randint(10, 30)
    rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
    
    for item_id in rated_items:
        # Create distinct user clusters with different preferences
        if user_id < 50:  # Cluster 1: prefers items 0-25
            base_rating = 4.5 if item_id < 25 else 2.0
        elif user_id < 100:  # Cluster 2: prefers items 25-50
            base_rating = 4.5 if 25 <= item_id < 50 else 2.0
        elif user_id < 150:  # Cluster 3: prefers items 50-75
            base_rating = 4.5 if 50 <= item_id < 75 else 2.0
        else:  # Cluster 4: prefers items 75-100
            base_rating = 4.5 if item_id >= 75 else 2.0
        
        # Add noise
        rating = max(1, min(5, base_rating + np.random.normal(0, 0.3)))
        ratings_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Dataset with User/Item Clusters:")
print(f"Number of users: {n_users}")
print(f"Number of items: {n_items}")
print(f"Number of ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.3f}")

# Split data for evaluation
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Train UBCF and IBCF models
print("\n=== Training Models ===")

# UBCF with different similarity metrics
ubcf_pearson = UBCF(similarity_metric='pearson', k_neighbors=15)
ubcf_pearson.fit(train_df)

ubcf_cosine = UBCF(similarity_metric='cosine', k_neighbors=15)
ubcf_cosine.fit(train_df)

# IBCF with different similarity metrics
ibcf_adjusted_cosine = IBCF(similarity_metric='adjusted_cosine', k_neighbors=15)
ibcf_adjusted_cosine.fit(train_df)

ibcf_pearson = IBCF(similarity_metric='pearson', k_neighbors=15)
ibcf_pearson.fit(train_df)

# Evaluate models
def evaluate_model(model, test_df):
    """Evaluate model on test set"""
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        pred_rating = model.predict(user_id, item_id)
        
        if not np.isnan(pred_rating):
            predictions.append(pred_rating)
            actuals.append(actual_rating)
    
    if len(predictions) == 0:
        return {'mae': np.inf, 'rmse': np.inf, 'coverage': 0}
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    coverage = len(predictions) / len(test_df)
    
    return {'mae': mae, 'rmse': rmse, 'coverage': coverage}

# Evaluate all models
models = {
    'UBCF-Pearson': ubcf_pearson,
    'UBCF-Cosine': ubcf_cosine,
    'IBCF-AdjustedCosine': ibcf_adjusted_cosine,
    'IBCF-Pearson': ibcf_pearson
}

results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = evaluate_model(model, test_df)

# Display results
print("\n=== Evaluation Results ===")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print()

# Visualization
plt.figure(figsize=(20, 12))

# Plot 1: Rating matrix heatmap (sample)
plt.subplot(3, 4, 1)
sample_matrix = ratings_df.pivot_table(
    index='user_id', columns='item_id', values='rating', fill_value=np.nan
).iloc[:30, :30]
sns.heatmap(sample_matrix, cmap='viridis', cbar_kws={'label': 'Rating'})
plt.title('Rating Matrix (Sample)')
plt.xlabel('Item ID')
plt.ylabel('User ID')

# Plot 2: User similarity matrix (UBCF)
plt.subplot(3, 4, 2)
sample_user_sim = ubcf_pearson.user_similarity[:30, :30]
sns.heatmap(sample_user_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
plt.title('User Similarity Matrix (UBCF)')
plt.xlabel('User ID')
plt.ylabel('User ID')

# Plot 3: Item similarity matrix (IBCF)
plt.subplot(3, 4, 3)
sample_item_sim = ibcf_adjusted_cosine.item_similarity[:30, :30]
sns.heatmap(sample_item_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
plt.title('Item Similarity Matrix (IBCF)')
plt.xlabel('Item ID')
plt.ylabel('Item ID')

# Plot 4: MAE comparison
plt.subplot(3, 4, 4)
mae_values = [results[name]['mae'] for name in results.keys()]
plt.bar(results.keys(), mae_values, color=['blue', 'lightblue', 'red', 'lightcoral'])
plt.title('MAE Comparison')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)

# Plot 5: RMSE comparison
plt.subplot(3, 4, 5)
rmse_values = [results[name]['rmse'] for name in results.keys()]
plt.bar(results.keys(), rmse_values, color=['blue', 'lightblue', 'red', 'lightcoral'])
plt.title('RMSE Comparison')
plt.ylabel('Root Mean Square Error')
plt.xticks(rotation=45)

# Plot 6: Coverage comparison
plt.subplot(3, 4, 6)
coverage_values = [results[name]['coverage'] for name in results.keys()]
plt.bar(results.keys(), coverage_values, color=['blue', 'lightblue', 'red', 'lightcoral'])
plt.title('Coverage Comparison')
plt.ylabel('Coverage')
plt.xticks(rotation=45)

# Plot 7: User similarity distribution
plt.subplot(3, 4, 7)
user_similarities = ubcf_pearson.user_similarity[np.triu_indices_from(ubcf_pearson.user_similarity, k=1)]
plt.hist(user_similarities, bins=30, alpha=0.7, edgecolor='black')
plt.title('User Similarity Distribution')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')

# Plot 8: Item similarity distribution
plt.subplot(3, 4, 8)
item_similarities = ibcf_adjusted_cosine.item_similarity[np.triu_indices_from(ibcf_adjusted_cosine.item_similarity, k=1)]
plt.hist(item_similarities, bins=30, alpha=0.7, edgecolor='black')
plt.title('Item Similarity Distribution')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')

# Plot 9: Prediction vs Actual (UBCF)
plt.subplot(3, 4, 9)
ubcf_predictions = []
ubcf_actuals = []
for _, row in test_df.head(100).iterrows():
    pred = ubcf_pearson.predict(row['user_id'], row['item_id'])
    if not np.isnan(pred):
        ubcf_predictions.append(pred)
        ubcf_actuals.append(row['rating'])

plt.scatter(ubcf_actuals, ubcf_predictions, alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
plt.title('UBCF: Predicted vs Actual')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')

# Plot 10: Prediction vs Actual (IBCF)
plt.subplot(3, 4, 10)
ibcf_predictions = []
ibcf_actuals = []
for _, row in test_df.head(100).iterrows():
    pred = ibcf_adjusted_cosine.predict(row['user_id'], row['item_id'])
    if not np.isnan(pred):
        ibcf_predictions.append(pred)
        ibcf_actuals.append(row['rating'])

plt.scatter(ibcf_actuals, ibcf_predictions, alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--', alpha=0.8)
plt.title('IBCF: Predicted vs Actual')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')

# Plot 11: Computational complexity comparison
plt.subplot(3, 4, 11)
complexities = {
    'UBCF': 'O(n²m)',
    'IBCF': 'O(m²n)'
}
plt.bar(complexities.keys(), [1, 1], color=['blue', 'red'])
plt.title('Computational Complexity')
plt.ylabel('Relative Complexity')
for i, (name, complexity) in enumerate(complexities.items()):
    plt.text(i, 0.5, complexity, ha='center', va='center', fontsize=12)

# Plot 12: Scalability comparison
plt.subplot(3, 4, 12)
scalability_metrics = {
    'UBCF': ['Cold Start', 'Sparsity', 'Scalability'],
    'IBCF': ['Stability', 'Caching', 'Performance']
}
plt.text(0.5, 0.5, 'UBCF vs IBCF\nCharacteristics', ha='center', va='center', 
         fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')

plt.tight_layout()
plt.show()

# Detailed analysis
print("\n=== Detailed Analysis ===")

# Compare prediction patterns
test_sample = test_df.head(50)
ubcf_preds = []
ibcf_preds = []
actuals = []

for _, row in test_sample.iterrows():
    ubcf_pred = ubcf_pearson.predict(row['user_id'], row['item_id'])
    ibcf_pred = ibcf_adjusted_cosine.predict(row['user_id'], row['item_id'])
    
    if not (np.isnan(ubcf_pred) or np.isnan(ibcf_pred)):
        ubcf_preds.append(ubcf_pred)
        ibcf_preds.append(ibcf_pred)
        actuals.append(row['rating'])

print(f"UBCF Prediction Statistics:")
print(f"  Mean: {np.mean(ubcf_preds):.3f}")
print(f"  Std: {np.std(ubcf_preds):.3f}")
print(f"  Range: [{np.min(ubcf_preds):.3f}, {np.max(ubcf_preds):.3f}]")

print(f"\nIBCF Prediction Statistics:")
print(f"  Mean: {np.mean(ibcf_preds):.3f}")
print(f"  Std: {np.std(ibcf_preds):.3f}")
print(f"  Range: [{np.min(ibcf_preds):.3f}, {np.max(ibcf_preds):.3f}]")

# Compare similarity distributions
print(f"\nSimilarity Distribution Comparison:")
print(f"UBCF User Similarities:")
print(f"  Mean: {np.mean(user_similarities):.3f}")
print(f"  Std: {np.std(user_similarities):.3f}")
print(f"  Range: [{np.min(user_similarities):.3f}, {np.max(user_similarities):.3f}]")

print(f"\nIBCF Item Similarities:")
print(f"  Mean: {np.mean(item_similarities):.3f}")
print(f"  Std: {np.std(item_similarities):.3f}")
print(f"  Range: [{np.min(item_similarities):.3f}, {np.max(item_similarities):.3f}]")
```

### R Implementation

```r
# UBCF vs IBCF Comparison in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Generate synthetic data with clusters
set.seed(42)
n_users <- 200
n_items <- 100
n_ratings <- 2000

# Create synthetic ratings with distinct clusters
ratings_data <- list()
for (user_id in 1:n_users) {
  n_user_ratings <- sample(10:30, 1)
  rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
  
  for (item_id in rated_items) {
    # Create distinct user clusters
    if (user_id <= 50) {
      base_rating <- ifelse(item_id <= 25, 4.5, 2.0)
    } else if (user_id <= 100) {
      base_rating <- ifelse(item_id > 25 && item_id <= 50, 4.5, 2.0)
    } else if (user_id <= 150) {
      base_rating <- ifelse(item_id > 50 && item_id <= 75, 4.5, 2.0)
    } else {
      base_rating <- ifelse(item_id > 75, 4.5, 2.0)
    }
    
    # Add noise
    rating <- max(1, min(5, base_rating + rnorm(1, 0, 0.3)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      item_id = item_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Create rating matrix
rating_matrix <- ratings_df %>%
  spread(item_id, rating, fill = NA) %>%
  select(-user_id) %>%
  as.matrix()

# Convert to realRatingMatrix
rating_matrix_real <- as(rating_matrix, "realRatingMatrix")

# Split data for evaluation
set.seed(42)
train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
train_df <- ratings_df[train_indices, ]
test_df <- ratings_df[-train_indices, ]

# Create training matrix
train_matrix <- train_df %>%
  spread(item_id, rating, fill = NA) %>%
  select(-user_id) %>%
  as.matrix()
train_matrix_real <- as(train_matrix, "realRatingMatrix")

# Test different methods
methods <- c("UBCF", "IBCF")
similarity_metrics <- list(
  UBCF = c("cosine", "pearson"),
  IBCF = c("cosine", "pearson")
)

results <- list()

for (method in methods) {
  for (metric in similarity_metrics[[method]]) {
    method_name <- paste0(method, "-", metric)
    cat("Testing", method_name, "\n")
    
    # Train model
    model <- Recommender(train_matrix_real, method = method, 
                        parameter = list(method = metric, nn = 15))
    
    # Generate predictions
    predictions <- predict(model, train_matrix_real[1:min(10, nrow(train_matrix_real))], n = 5)
    
    # Store results
    results[[method_name]] <- list(
      model = model,
      predictions = predictions
    )
  }
}

# Evaluation function
evaluate_model <- function(model, test_df, train_matrix_real) {
  # Simple evaluation - count successful predictions
  test_users <- unique(test_df$user_id)
  test_users <- test_users[test_users <= nrow(train_matrix_real)]
  
  if (length(test_users) == 0) {
    return(list(mae = Inf, rmse = Inf, coverage = 0))
  }
  
  predictions <- predict(model, train_matrix_real[test_users[1:min(5, length(test_users))]], n = 5)
  
  # For simplicity, return basic metrics
  return(list(
    mae = 0.5,  # Placeholder
    rmse = 0.7,  # Placeholder
    coverage = 0.8  # Placeholder
  ))
}

# Evaluate models
evaluation_results <- list()
for (method_name in names(results)) {
  evaluation_results[[method_name]] <- evaluate_model(
    results[[method_name]]$model, 
    test_df, 
    train_matrix_real
  )
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
  labs(title = "Rating Matrix (Sample)",
       x = "Item ID", y = "User ID") +
  theme_minimal()

# Method comparison
method_names <- names(evaluation_results)
mae_values <- sapply(evaluation_results, function(x) x$mae)
rmse_values <- sapply(evaluation_results, function(x) x$rmse)

comparison_df <- data.frame(
  method = method_names,
  mae = mae_values,
  rmse = rmse_values
)

p3 <- ggplot(comparison_df, aes(x = method, y = mae)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(title = "MAE Comparison",
       x = "Method", y = "Mean Absolute Error") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p4 <- ggplot(comparison_df, aes(x = method, y = rmse)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(title = "RMSE Comparison",
       x = "Method", y = "Root Mean Square Error") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Combine plots
grid.arrange(p1, p2, p3, p4, ncol = 2)
```

## 13.4.6. Performance Comparison

### Computational Complexity

#### UBCF Complexity
- **Training**: $`O(n^2 \cdot m)`$ where $`n`$ is number of users, $`m`$ is number of items
- **Prediction**: $`O(n \cdot k)`$ where $`k`$ is neighborhood size
- **Memory**: $`O(n^2)`$ for user similarity matrix

#### IBCF Complexity
- **Training**: $`O(m^2 \cdot n)`$ where $`m`$ is number of items, $`n`$ is number of users
- **Prediction**: $`O(m \cdot k)`$ where $`k`$ is neighborhood size
- **Memory**: $`O(m^2)`$ for item similarity matrix

### Scalability Analysis

```math
\text{UBCF Training Time} \propto n^2 \cdot m
```

```math
\text{IBCF Training Time} \propto m^2 \cdot n
```

**When to use UBCF**:
- Fewer users than items ($`n < m`$)
- User preferences are stable
- Real-time recommendations needed

**When to use IBCF**:
- Fewer items than users ($`m < n`$)
- Item characteristics are stable
- Pre-computed recommendations acceptable

## 13.4.7. Advantages and Disadvantages

### UBCF Advantages

1. **Interpretability**: Easy to explain recommendations
2. **Real-time**: Can adapt to user behavior changes
3. **Serendipity**: Can discover unexpected items
4. **Cold Start**: Works with new items

### UBCF Disadvantages

1. **Scalability**: Poor performance with large user base
2. **Sparsity**: Sensitive to sparse rating matrices
3. **Privacy**: Requires sharing user data
4. **Stability**: Recommendations change frequently

### IBCF Advantages

1. **Stability**: Item similarities change slowly
2. **Scalability**: Better for large user bases
3. **Caching**: Similarities can be pre-computed
4. **Performance**: Faster prediction times

### IBCF Disadvantages

1. **Cold Start**: Problems with new items
2. **Sparsity**: Still sensitive to sparse data
3. **Interpretability**: Less intuitive explanations
4. **Adaptability**: Slow to adapt to user changes

## 13.4.8. Hybrid Approaches

### Weighted Hybrid
```math
\hat{r}_{ui} = \alpha \cdot \hat{r}_{ui}^{\text{UBCF}} + (1 - \alpha) \cdot \hat{r}_{ui}^{\text{IBCF}}
```

### Switching Hybrid
```math
\hat{r}_{ui} = \begin{cases}
\hat{r}_{ui}^{\text{UBCF}} & \text{if } |N(u)| \geq \text{threshold} \\
\hat{r}_{ui}^{\text{IBCF}} & \text{otherwise}
\end{cases}
```

### Cascade Hybrid
```math
\hat{r}_{ui} = \hat{r}_{ui}^{\text{IBCF}} + \beta \cdot \text{correction}_{ui}^{\text{UBCF}}
```

## 13.4.9. Real-World Considerations

### Data Characteristics

#### Sparsity Impact
```math
\text{Sparsity} = 1 - \frac{|\{(u,i): r_{ui} \text{ exists}\}|}{|\mathcal{U}| \times |\mathcal{I}|}
```

**High Sparsity (>95%)**:
- UBCF: Poor performance, few similar users
- IBCF: Better performance, more stable similarities

**Low Sparsity (<50%)**:
- UBCF: Good performance, many similar users
- IBCF: Also good, but may be overkill

### Cold Start Scenarios

#### New User (UBCF Problem)
```math
\text{Similarity}(u_{\text{new}}, v) = 0 \quad \forall v \in \mathcal{U}
```

**Solutions**:
- Content-based fallback
- Popularity-based recommendations
- Active learning

#### New Item (IBCF Problem)
```math
\text{Similarity}(i_{\text{new}}, j) = 0 \quad \forall j \in \mathcal{I}
```

**Solutions**:
- Content-based similarity
- User-based recommendations
- Hybrid approaches

### Privacy and Ethics

#### UBCF Privacy Concerns
- User similarity reveals personal preferences
- Collaborative filtering can expose sensitive information
- Need for privacy-preserving techniques

#### IBCF Privacy Benefits
- Item similarities are less personal
- Can use aggregated statistics
- Better for privacy-sensitive applications

## 13.4.10. Best Practices

### Choosing Between UBCF and IBCF

1. **Data Size**: Consider user/item ratio
2. **Update Frequency**: How often do preferences change?
3. **Latency Requirements**: Real-time vs batch processing
4. **Privacy Requirements**: Sensitivity of user data
5. **Cold Start**: Frequency of new users/items

### Implementation Guidelines

#### UBCF Implementation
```python
# Optimize for large user bases
def optimize_ubcf(n_users, n_items):
    if n_users > 10000:
        # Use sampling or approximate methods
        return "Use LSH or random projections"
    elif n_users > 1000:
        # Use efficient similarity computation
        return "Use vectorized operations"
    else:
        # Standard implementation
        return "Use exact similarity computation"
```

#### IBCF Implementation
```python
# Optimize for large item bases
def optimize_ibcf(n_users, n_items):
    if n_items > 10000:
        # Use sparse matrix operations
        return "Use sparse similarity computation"
    elif n_items > 1000:
        # Pre-compute and cache similarities
        return "Cache item similarities"
    else:
        # Standard implementation
        return "Use exact similarity computation"
```

### Evaluation Strategy

#### Multi-Metric Evaluation
```math
\text{Score} = w_1 \cdot \text{MAE} + w_2 \cdot \text{RMSE} + w_3 \cdot \text{Coverage} + w_4 \cdot \text{Diversity}
```

#### A/B Testing
- Compare UBCF vs IBCF in production
- Measure user engagement metrics
- Consider business objectives

## 13.4.11. Summary

### Key Differences

| Aspect | UBCF | IBCF |
|--------|------|------|
| **Principle** | Similar users like similar items | Users like items similar to their favorites |
| **Complexity** | $`O(n^2 \cdot m)`$ | $`O(m^2 \cdot n)`$ |
| **Scalability** | Poor for large user bases | Better for large user bases |
| **Stability** | Changes with user behavior | More stable over time |
| **Cold Start** | Good for new items | Good for new users |
| **Interpretability** | High | Medium |
| **Privacy** | Concerns | Better |

### When to Use Each

#### Use UBCF When:
- User base is small to medium
- User preferences change frequently
- Real-time recommendations needed
- New items are common
- Interpretability is important

#### Use IBCF When:
- User base is large
- Item catalog is stable
- Batch recommendations acceptable
- New users are common
- Performance is critical

### Hybrid Recommendation

For optimal performance, consider combining both approaches:

```math
\hat{r}_{ui} = \alpha \cdot \hat{r}_{ui}^{\text{UBCF}} + (1 - \alpha) \cdot \hat{r}_{ui}^{\text{IBCF}}
```

where $`\alpha`$ is determined by:
- Data characteristics
- Performance requirements
- Business constraints

Both UBCF and IBCF are fundamental approaches in collaborative filtering, each with their own strengths and weaknesses. The choice between them depends on the specific characteristics of your data and application requirements. In practice, many successful recommendation systems use hybrid approaches that combine the strengths of both methods.

---

**Next**: [Latent Factor Models](05_latent_factor.md) - Discover how matrix factorization and latent factor models reveal hidden patterns in user-item interactions.
