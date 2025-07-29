# 13.2. Content-Based Methods

Let's start with content-based methods. These are intuitive by nature.

As a company, we construct a comprehensive catalog of our products, ensuring we know how to accurately represent each item. Consider movies or articles, for example; each can be depicted by a multi-dimensional feature vector.

We might describe a movie using features such as its release date, genre, director, and lead actors, and so on. In this way, we construct an **item profile**, visualizing each item as a distinct point within a multi-dimensional feature space, represented here by blue dots.

## 13.2.1. Introduction to Content-Based Filtering

Content-based filtering is a recommendation approach that analyzes the characteristics of items and user preferences to make recommendations. Unlike collaborative filtering, which relies on user-item interaction patterns, content-based methods focus on the intrinsic properties of items and how they align with user preferences.

### Core Principle

The fundamental idea is simple: **"If a user liked an item, they will likely enjoy similar items."** This similarity is computed based on the features or attributes of the items.

### Mathematical Foundation

Content-based filtering can be formalized as follows:

```math
\text{Recommendation}(u, i) = \text{Similarity}(\text{UserProfile}(u), \text{ItemProfile}(i))
```

where:
- $`\text{UserProfile}(u)`$ represents user $`u`$'s preference vector
- $`\text{ItemProfile}(i)`$ represents item $`i`$'s feature vector
- $`\text{Similarity}(\cdot, \cdot)`$ is a similarity function

## 13.2.2. Item Profiling

### Feature Vector Construction

Each item is represented as a feature vector in a multi-dimensional space:

```math
\text{ItemProfile}(i) = \mathbf{f}_i = [f_{i1}, f_{i2}, \ldots, f_{id}]^T
```

where $`f_{ij}`$ represents the $`j`$-th feature of item $`i`$.

### Feature Types

#### 1. Categorical Features
For discrete categories like genre, director, or actor:

```math
f_{ij} = \begin{cases}
1 & \text{if item } i \text{ has feature } j \\
0 & \text{otherwise}
\end{cases}
```

#### 2. Numerical Features
For continuous values like release year, rating, or price:

```math
f_{ij} = \text{normalized\_value}(i, j)
```

#### 3. Text Features
For textual content like descriptions or reviews:

```math
f_{ij} = \text{TF-IDF}(i, j) = \text{TF}(i, j) \times \text{IDF}(j)
```

where:
- $`\text{TF}(i, j)`$ is the term frequency of word $`j`$ in item $`i`$
- $`\text{IDF}(j)`$ is the inverse document frequency of word $`j`$

### Example: Movie Profiling

Consider a movie with the following features:

```math
\text{MovieProfile} = \begin{bmatrix}
\text{Action} & 1 \\
\text{Drama} & 0 \\
\text{Comedy} & 0 \\
\text{Thriller} & 1 \\
\text{Year} & 0.8 \\
\text{Budget} & 0.6 \\
\text{Director\_Spielberg} & 1 \\
\text{Actor\_Cruise} & 1 \\
\text{Length} & 0.7
\end{bmatrix}
```

## 13.2.3. User Profiling

### Profile Construction Methods

#### 1. Explicit Profiling
Users directly specify their preferences:

```math
\text{UserProfile}(u) = \mathbf{p}_u = [p_{u1}, p_{u2}, \ldots, p_{ud}]^T
```

where $`p_{uj}`$ represents user $`u`$'s preference for feature $`j`$.

#### 2. Implicit Profiling
Preferences are inferred from user behavior:

```math
\mathbf{p}_u = \frac{\sum_{i \in \mathcal{I}_u} w_{ui} \cdot \mathbf{f}_i}{\sum_{i \in \mathcal{I}_u} w_{ui}}
```

where:
- $`\mathcal{I}_u`$ is the set of items rated by user $`u`$
- $`w_{ui}`$ is the weight of item $`i`$ for user $`u`$ (e.g., rating, recency)

#### 3. Time-Weighted Profiling
Recent interactions are weighted more heavily:

```math
w_{ui} = \exp\left(-\lambda \cdot (t_{\text{current}} - t_{ui})\right)
```

where $`t_{ui}`$ is the time when user $`u`$ interacted with item $`i`$.

### Example: User Profile Construction

For a user who rated several movies:

```math
\text{UserProfile} = \begin{bmatrix}
\text{Action} & 0.8 \\
\text{Drama} & 0.3 \\
\text{Comedy} & 0.6 \\
\text{Thriller} & 0.9 \\
\text{Year} & 0.7 \\
\text{Budget} & 0.5 \\
\text{Director\_Spielberg} & 0.9 \\
\text{Actor\_Cruise} & 0.8 \\
\text{Length} & 0.6
\end{bmatrix}
```

## 13.2.4. Similarity Computation

### Similarity Metrics

#### 1. Cosine Similarity
Most commonly used for content-based filtering:

```math
\text{Similarity}(\mathbf{p}_u, \mathbf{f}_i) = \cos(\theta) = \frac{\mathbf{p}_u \cdot \mathbf{f}_i}{\|\mathbf{p}_u\| \cdot \|\mathbf{f}_i\|}
```

#### 2. Euclidean Distance
```math
\text{Similarity}(\mathbf{p}_u, \mathbf{f}_i) = \frac{1}{1 + \|\mathbf{p}_u - \mathbf{f}_i\|}
```

#### 3. Pearson Correlation
```math
\text{Similarity}(\mathbf{p}_u, \mathbf{f}_i) = \frac{\sum_{j=1}^d (p_{uj} - \bar{p}_u)(f_{ij} - \bar{f}_i)}{\sqrt{\sum_{j=1}^d (p_{uj} - \bar{p}_u)^2} \sqrt{\sum_{j=1}^d (f_{ij} - \bar{f}_i)^2}}
```

#### 4. Jaccard Similarity
For binary features:

```math
\text{Similarity}(\mathbf{p}_u, \mathbf{f}_i) = \frac{|\mathbf{p}_u \cap \mathbf{f}_i|}{|\mathbf{p}_u \cup \mathbf{f}_i|}
```

### Recommendation Score

The final recommendation score is computed as:

```math
\text{Score}(u, i) = \text{Similarity}(\mathbf{p}_u, \mathbf{f}_i) \times \text{Popularity}(i) \times \text{Novelty}(i)
```

where:
- $`\text{Popularity}(i)`$ accounts for item popularity
- $`\text{Novelty}(i)`$ promotes diversity in recommendations

## 13.2.5. Implementation

### Python Implementation: Content-Based Recommender

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, similarity_metric='cosine'):
        """
        Content-Based Recommender System
        
        Parameters:
        -----------
        similarity_metric : str
            Similarity metric ('cosine', 'euclidean', 'pearson')
        """
        self.similarity_metric = similarity_metric
        self.item_profiles = None
        self.user_profiles = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def _compute_similarity(self, profile1, profile2):
        """Compute similarity between two profiles"""
        if self.similarity_metric == 'cosine':
            return cosine_similarity([profile1], [profile2])[0][0]
        elif self.similarity_metric == 'euclidean':
            distance = np.linalg.norm(profile1 - profile2)
            return 1 / (1 + distance)
        elif self.similarity_metric == 'pearson':
            return np.corrcoef(profile1, profile2)[0, 1]
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def create_item_profiles(self, items_df, feature_columns, text_columns=None):
        """
        Create item profiles from item features
        
        Parameters:
        -----------
        items_df : pandas.DataFrame
            DataFrame containing item features
        feature_columns : list
            List of feature column names
        text_columns : list, optional
            List of text column names for TF-IDF
        """
        profiles = []
        feature_names = []
        
        # Handle categorical features
        for col in feature_columns:
            if items_df[col].dtype == 'object':
                # Encode categorical features
                le = LabelEncoder()
                encoded_values = le.fit_transform(items_df[col])
                profiles.append(encoded_values)
                feature_names.extend([f"{col}_{val}" for val in le.classes_])
                self.label_encoders[col] = le
            else:
                # Numerical features
                profiles.append(items_df[col].values)
                feature_names.append(col)
        
        # Handle text features
        if text_columns:
            for col in text_columns:
                tfidf = TfidfVectorizer(max_features=50, stop_words='english')
                text_features = tfidf.fit_transform(items_df[col].fillna(''))
                profiles.append(text_features.toarray())
                feature_names.extend([f"{col}_{word}" for word in tfidf.get_feature_names_out()])
        
        # Combine all features
        self.item_profiles = np.hstack(profiles)
        self.feature_names = feature_names
        
        # Normalize features
        self.item_profiles = self.scaler.fit_transform(self.item_profiles)
        
        return self.item_profiles
    
    def create_user_profiles(self, ratings_df, items_df, user_id_col='user_id', 
                           item_id_col='item_id', rating_col='rating'):
        """
        Create user profiles from ratings and item features
        
        Parameters:
        -----------
        ratings_df : pandas.DataFrame
            DataFrame containing user ratings
        items_df : pandas.DataFrame
            DataFrame containing item features
        """
        if self.item_profiles is None:
            raise ValueError("Item profiles must be created first")
        
        user_profiles = {}
        
        for user_id in ratings_df[user_id_col].unique():
            user_ratings = ratings_df[ratings_df[user_id_col] == user_id]
            
            # Get items rated by this user
            rated_items = user_ratings[item_id_col].values
            ratings = user_ratings[rating_col].values
            
            # Find corresponding item profiles
            item_indices = [items_df.index.get_loc(item_id) for item_id in rated_items]
            item_profiles = self.item_profiles[item_indices]
            
            # Compute weighted average (weighted by ratings)
            weights = ratings / ratings.sum()
            user_profile = np.average(item_profiles, weights=weights, axis=0)
            
            user_profiles[user_id] = user_profile
        
        self.user_profiles = user_profiles
        return user_profiles
    
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        """
        Generate recommendations for a user
        
        Parameters:
        -----------
        user_id : int
            User ID to generate recommendations for
        n_recommendations : int
            Number of recommendations to generate
        exclude_rated : bool
            Whether to exclude items the user has already rated
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Compute similarities with all items
        similarities = []
        for i, item_profile in enumerate(self.item_profiles):
            similarity = self._compute_similarity(user_profile, item_profile)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top recommendations
        return similarities[:n_recommendations]
    
    def get_feature_importance(self, user_id, top_features=10):
        """Get most important features for a user"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Get feature importance (absolute values)
        feature_importance = [(name, abs(value)) for name, value in zip(self.feature_names, user_profile)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_features]
    
    def visualize_profiles(self, user_ids=None, n_items=20):
        """Visualize user and item profiles using PCA"""
        if user_ids is None:
            user_ids = list(self.user_profiles.keys())[:5]
        
        # Combine user and item profiles
        all_profiles = []
        profile_labels = []
        profile_types = []
        
        # Add user profiles
        for user_id in user_ids:
            all_profiles.append(self.user_profiles[user_id])
            profile_labels.append(f"User {user_id}")
            profile_types.append("User")
        
        # Add item profiles (sample)
        item_indices = np.random.choice(len(self.item_profiles), n_items, replace=False)
        for idx in item_indices:
            all_profiles.append(self.item_profiles[idx])
            profile_labels.append(f"Item {idx}")
            profile_types.append("Item")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        profiles_2d = pca.fit_transform(all_profiles)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot users and items
        for i, (profile, label, profile_type) in enumerate(zip(profiles_2d, profile_labels, profile_types)):
            if profile_type == "User":
                plt.scatter(profile[0], profile[1], c='red', s=100, marker='s', label=label if i < len(user_ids) else "")
            else:
                plt.scatter(profile[0], profile[1], c='blue', s=50, alpha=0.6)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('User and Item Profiles in 2D Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Generate synthetic movie data
np.random.seed(42)
n_movies = 100
n_users = 50

# Create movie features
movies_df = pd.DataFrame({
    'movie_id': range(n_movies),
    'title': [f'Movie_{i}' for i in range(n_movies)],
    'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Thriller', 'Romance'], n_movies),
    'year': np.random.randint(1990, 2024, n_movies),
    'rating': np.random.uniform(1, 10, n_movies),
    'budget': np.random.uniform(1, 100, n_movies),
    'director': np.random.choice(['Spielberg', 'Nolan', 'Tarantino', 'Scorsese', 'Cameron'], n_movies),
    'description': [f'Description for movie {i}' for i in range(n_movies)]
})

# Create synthetic ratings
ratings_data = []
for user_id in range(n_users):
    n_ratings = np.random.randint(5, 20)
    rated_movies = np.random.choice(n_movies, n_ratings, replace=False)
    
    for movie_id in rated_movies:
        # Simulate user preferences based on movie features
        movie = movies_df.iloc[movie_id]
        base_rating = 5
        
        # Genre preferences (simulate user taste)
        if movie['genre'] in ['Action', 'Thriller']:
            base_rating += np.random.normal(1, 1)
        elif movie['genre'] in ['Drama', 'Romance']:
            base_rating += np.random.normal(-1, 1)
        
        # Year preference (prefer newer movies)
        year_factor = (movie['year'] - 1990) / (2024 - 1990)
        base_rating += year_factor * 2
        
        # Add noise
        rating = max(1, min(10, base_rating + np.random.normal(0, 1)))
        ratings_data.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating
        })

ratings_df = pd.DataFrame(ratings_data)

print("Synthetic Movie Dataset:")
print(f"Number of movies: {n_movies}")
print(f"Number of users: {n_users}")
print(f"Number of ratings: {len(ratings_df)}")

# Initialize and train content-based recommender
recommender = ContentBasedRecommender(similarity_metric='cosine')

# Create item profiles
feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
text_columns = ['description']
item_profiles = recommender.create_item_profiles(movies_df, feature_columns, text_columns)

print(f"\nItem profiles shape: {item_profiles.shape}")
print(f"Number of features: {len(recommender.feature_names)}")

# Create user profiles
user_profiles = recommender.create_user_profiles(ratings_df, movies_df)

print(f"Number of user profiles: {len(user_profiles)}")

# Generate recommendations for a sample user
test_user = 0
recommendations = recommender.recommend(test_user, n_recommendations=10)

print(f"\nTop 10 recommendations for User {test_user}:")
for i, (item_idx, similarity) in enumerate(recommendations):
    movie = movies_df.iloc[item_idx]
    print(f"{i+1}. {movie['title']} ({movie['genre']}, {movie['year']}) - Similarity: {similarity:.3f}")

# Get feature importance for the user
feature_importance = recommender.get_feature_importance(test_user, top_features=10)

print(f"\nTop 10 most important features for User {test_user}:")
for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.3f}")

# Visualize profiles
recommender.visualize_profiles(user_ids=[0, 1, 2], n_items=30)

# Compare different similarity metrics
similarity_metrics = ['cosine', 'euclidean', 'pearson']
results = {}

for metric in similarity_metrics:
    print(f"\n=== Testing {metric.upper()} Similarity ===")
    
    recommender_metric = ContentBasedRecommender(similarity_metric=metric)
    recommender_metric.create_item_profiles(movies_df, feature_columns, text_columns)
    recommender_metric.create_user_profiles(ratings_df, movies_df)
    
    recommendations = recommender_metric.recommend(test_user, n_recommendations=5)
    results[metric] = recommendations
    
    print(f"Top 5 recommendations:")
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"  {i+1}. {movie['title']} - Similarity: {similarity:.3f}")

# Visualization of similarity distributions
plt.figure(figsize=(15, 5))

for i, (metric, recommendations) in enumerate(results.items()):
    similarities = [sim for _, sim in recommendations]
    
    plt.subplot(1, 3, i+1)
    plt.hist(similarities, bins=10, alpha=0.7, edgecolor='black')
    plt.title(f'{metric.capitalize()} Similarity Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### R Implementation

```r
# Content-Based Recommender System in R
library(tm)
library(proxy)
library(ggplot2)
library(dplyr)
library(tidyr)

content_based_recommender <- function(similarity_metric = "cosine") {
  list(
    similarity_metric = similarity_metric,
    item_profiles = NULL,
    user_profiles = NULL,
    feature_names = NULL
  )
}

compute_similarity <- function(profile1, profile2, metric = "cosine") {
  if (metric == "cosine") {
    return(sum(profile1 * profile2) / (sqrt(sum(profile1^2)) * sqrt(sum(profile2^2))))
  } else if (metric == "euclidean") {
    distance <- sqrt(sum((profile1 - profile2)^2))
    return(1 / (1 + distance))
  } else if (metric == "pearson") {
    return(cor(profile1, profile2, method = "pearson"))
  }
}

create_item_profiles <- function(recommender, items_df, feature_columns, text_columns = NULL) {
  profiles <- list()
  feature_names <- c()
  
  # Handle categorical features
  for (col in feature_columns) {
    if (is.character(items_df[[col]])) {
      # Encode categorical features
      unique_values <- unique(items_df[[col]])
      encoded_matrix <- matrix(0, nrow = nrow(items_df), ncol = length(unique_values))
      
      for (i in 1:length(unique_values)) {
        encoded_matrix[items_df[[col]] == unique_values[i], i] <- 1
      }
      
      profiles[[length(profiles) + 1]] <- encoded_matrix
      feature_names <- c(feature_names, paste0(col, "_", unique_values))
    } else {
      # Numerical features
      profiles[[length(profiles) + 1]] <- matrix(items_df[[col]], ncol = 1)
      feature_names <- c(feature_names, col)
    }
  }
  
  # Handle text features
  if (!is.null(text_columns)) {
    for (col in text_columns) {
      # Create corpus
      corpus <- Corpus(VectorSource(items_df[[col]]))
      
      # Create document-term matrix
      dtm <- DocumentTermMatrix(corpus, control = list(
        removePunctuation = TRUE,
        removeNumbers = TRUE,
        stopwords = TRUE,
        weighting = weightTfIdf
      ))
      
      # Convert to matrix
      text_matrix <- as.matrix(dtm)
      
      # Limit features
      if (ncol(text_matrix) > 50) {
        text_matrix <- text_matrix[, 1:50]
      }
      
      profiles[[length(profiles) + 1]] <- text_matrix
      feature_names <- c(feature_names, paste0(col, "_", colnames(text_matrix)))
    }
  }
  
  # Combine all features
  recommender$item_profiles <- do.call(cbind, profiles)
  recommender$feature_names <- feature_names
  
  # Normalize features
  recommender$item_profiles <- scale(recommender$item_profiles)
  
  return(recommender)
}

create_user_profiles <- function(recommender, ratings_df, items_df) {
  user_profiles <- list()
  
  for (user_id in unique(ratings_df$user_id)) {
    user_ratings <- ratings_df[ratings_df$user_id == user_id, ]
    
    # Get items rated by this user
    rated_items <- user_ratings$item_id
    ratings <- user_ratings$rating
    
    # Find corresponding item profiles
    item_indices <- match(rated_items, items_df$movie_id)
    item_profiles <- recommender$item_profiles[item_indices, ]
    
    # Compute weighted average (weighted by ratings)
    weights <- ratings / sum(ratings)
    user_profile <- colSums(t(item_profiles) * weights)
    
    user_profiles[[as.character(user_id)]] <- user_profile
  }
  
  recommender$user_profiles <- user_profiles
  return(recommender)
}

recommend <- function(recommender, user_id, n_recommendations = 5) {
  if (!(as.character(user_id) %in% names(recommender$user_profiles))) {
    return(list())
  }
  
  user_profile <- recommender$user_profiles[[as.character(user_id)]]
  
  # Compute similarities with all items
  similarities <- sapply(1:nrow(recommender$item_profiles), function(i) {
    compute_similarity(user_profile, recommender$item_profiles[i, ], recommender$similarity_metric)
  })
  
  # Sort by similarity
  sorted_indices <- order(similarities, decreasing = TRUE)
  
  # Return top recommendations
  result <- list()
  for (i in 1:n_recommendations) {
    result[[i]] <- list(
      item_index = sorted_indices[i],
      similarity = similarities[sorted_indices[i]]
    )
  }
  
  return(result)
}

# Generate synthetic data
set.seed(42)
n_movies <- 100
n_users <- 50

# Create movie features
movies_df <- data.frame(
  movie_id = 1:n_movies,
  title = paste0("Movie_", 1:n_movies),
  genre = sample(c("Action", "Drama", "Comedy", "Thriller", "Romance"), n_movies, replace = TRUE),
  year = sample(1990:2023, n_movies, replace = TRUE),
  rating = runif(n_movies, 1, 10),
  budget = runif(n_movies, 1, 100),
  director = sample(c("Spielberg", "Nolan", "Tarantino", "Scorsese", "Cameron"), n_movies, replace = TRUE),
  description = paste0("Description for movie ", 1:n_movies)
)

# Create synthetic ratings
ratings_data <- list()
for (user_id in 1:n_users) {
  n_ratings <- sample(5:20, 1)
  rated_movies <- sample(1:n_movies, n_ratings, replace = FALSE)
  
  for (movie_id in rated_movies) {
    movie <- movies_df[movie_id, ]
    base_rating <- 5
    
    # Genre preferences
    if (movie$genre %in% c("Action", "Thriller")) {
      base_rating <- base_rating + rnorm(1, 1, 1)
    } else if (movie$genre %in% c("Drama", "Romance")) {
      base_rating <- base_rating + rnorm(1, -1, 1)
    }
    
    # Year preference
    year_factor <- (movie$year - 1990) / (2023 - 1990)
    base_rating <- base_rating + year_factor * 2
    
    # Add noise
    rating <- max(1, min(10, base_rating + rnorm(1, 0, 1)))
    
    ratings_data[[length(ratings_data) + 1]] <- list(
      user_id = user_id,
      movie_id = movie_id,
      rating = rating
    )
  }
}

ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))

# Initialize and train recommender
recommender <- content_based_recommender("cosine")
feature_columns <- c("genre", "year", "rating", "budget", "director")
text_columns <- c("description")

recommender <- create_item_profiles(recommender, movies_df, feature_columns, text_columns)
recommender <- create_user_profiles(recommender, ratings_df, movies_df)

# Generate recommendations
test_user <- 1
recommendations <- recommend(recommender, test_user, 10)

cat("Top 10 recommendations for User", test_user, ":\n")
for (i in 1:length(recommendations)) {
  item_idx <- recommendations[[i]]$item_index
  similarity <- recommendations[[i]]$similarity
  movie <- movies_df[item_idx, ]
  cat(sprintf("%d. %s (%s, %d) - Similarity: %.3f\n", 
              i, movie$title, movie$genre, movie$year, similarity))
}
```

## 13.2.6. Advanced Content-Based Techniques

### TF-IDF for Text Features

For textual content like movie descriptions or reviews:

```math
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
```

where:
- $`\text{TF}(t, d)`$ is the term frequency of term $`t`$ in document $`d`$
- $`\text{IDF}(t) = \log\left(\frac{N}{df(t)}\right)`$ is the inverse document frequency

### Feature Engineering

#### 1. Feature Selection
```math
\text{Information Gain}(F) = H(Y) - H(Y|F)
```

where $`H(Y)`$ is the entropy of the target variable.

#### 2. Feature Scaling
```math
f'_{ij} = \frac{f_{ij} - \mu_j}{\sigma_j}
```

where $`\mu_j`$ and $`\sigma_j`$ are the mean and standard deviation of feature $`j`$.

#### 3. Feature Weighting
```math
w_j = \frac{\text{Importance}(j)}{\sum_{k=1}^d \text{Importance}(k)}
```

### Hybrid Approaches

#### Content + Collaborative
```math
\text{Score}(u, i) = \alpha \cdot \text{ContentScore}(u, i) + (1-\alpha) \cdot \text{CollaborativeScore}(u, i)
```

#### Content + Popularity
```math
\text{Score}(u, i) = \text{Similarity}(u, i) \times \text{Popularity}(i)^{\beta}
```

## 13.2.7. Evaluation and Metrics

### Content-Based Specific Metrics

#### 1. Feature Coverage
```math
\text{Coverage} = \frac{|\{i: \text{has\_features}(i)\}|}{|\mathcal{I}|}
```

#### 2. Diversity
```math
\text{Diversity} = \frac{1}{|\mathcal{R}|} \sum_{i,j \in \mathcal{R}} (1 - \text{Similarity}(i, j))
```

#### 3. Novelty
```math
\text{Novelty} = \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} \log_2(\text{Popularity}(i))
```

### A/B Testing Framework

```python
def evaluate_content_based(recommender, test_users, test_items, ground_truth):
    """Evaluate content-based recommender"""
    precision_scores = []
    recall_scores = []
    
    for user_id in test_users:
        recommendations = recommender.recommend(user_id, n_recommendations=10)
        recommended_items = [item_idx for item_idx, _ in recommendations]
        
        # Get ground truth for this user
        true_items = ground_truth.get(user_id, [])
        
        # Compute precision and recall
        if len(recommended_items) > 0:
            precision = len(set(recommended_items) & set(true_items)) / len(recommended_items)
            precision_scores.append(precision)
        
        if len(true_items) > 0:
            recall = len(set(recommended_items) & set(true_items)) / len(true_items)
            recall_scores.append(recall)
    
    return {
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1_score': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                   (np.mean(precision_scores) + np.mean(recall_scores))
    }
```

## 13.2.8. Real-World Applications

### Movie Recommendation System

```python
# Example: MovieLens dataset
from sklearn.datasets import fetch_openml

# Load MovieLens dataset
movies = fetch_openml(name='movielens-100k', as_frame=True)
movies_df = movies.frame

# Feature engineering
movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
movies_df['title_clean'] = movies_df['title'].str.replace(r'\(\d{4}\)', '').str.strip()

# Create content-based recommender
movie_recommender = ContentBasedRecommender()
feature_columns = ['year', 'rating']
text_columns = ['title_clean']

item_profiles = movie_recommender.create_item_profiles(movies_df, feature_columns, text_columns)
user_profiles = movie_recommender.create_user_profiles(ratings_df, movies_df)

# Generate recommendations
recommendations = movie_recommender.recommend(user_id=1, n_recommendations=10)
```

### Music Recommendation System

```python
# Example: Music features
music_features = {
    'tempo': [120, 140, 90, 160],  # BPM
    'energy': [0.8, 0.6, 0.9, 0.4],  # Energy level
    'valence': [0.7, 0.3, 0.8, 0.2],  # Positivity
    'danceability': [0.9, 0.5, 0.7, 0.3],  # Danceability
    'genre': ['pop', 'rock', 'electronic', 'jazz']
}

# Create music recommender
music_recommender = ContentBasedRecommender()
# ... implementation similar to movie recommender
```

## 13.2.9. Challenges and Solutions

### 1. Feature Engineering Challenges

**Challenge**: Extracting meaningful features from unstructured data
**Solution**: 
- Use pre-trained models for feature extraction
- Apply domain-specific feature engineering
- Leverage transfer learning

### 2. Cold Start Problem

**Challenge**: New items with no interaction history
**Solution**:
- Use item metadata for initial recommendations
- Implement hybrid approaches
- Leverage content similarity

### 3. Scalability Issues

**Challenge**: Computing similarities for large item catalogs
**Solution**:
- Use approximate nearest neighbor search
- Implement locality-sensitive hashing
- Apply dimensionality reduction

### 4. Overspecialization

**Challenge**: Recommendations become too narrow
**Solution**:
- Introduce randomness in recommendations
- Use diversity metrics
- Implement serendipity measures

## 13.2.10. Summary

Content-based filtering is a powerful and intuitive approach to recommendation systems that:

1. **Leverages Item Features**: Uses intrinsic properties of items
2. **Provides Transparency**: Clear reasoning for recommendations
3. **Handles Cold Start**: Works with new items and users
4. **Enables Personalization**: Tailored to individual preferences

### Key Advantages

- **No Cold Start**: Can recommend new items immediately
- **Interpretability**: Clear feature-based explanations
- **Independence**: Doesn't require other users' data
- **Flexibility**: Works with any type of item features

### Key Limitations

- **Feature Dependency**: Requires rich item metadata
- **Overspecialization**: May create filter bubbles
- **Feature Engineering**: Requires domain expertise
- **Limited Discovery**: Focuses on similar items

### Best Practices

1. **Feature Engineering**: Invest in high-quality feature extraction
2. **Similarity Metrics**: Choose appropriate similarity functions
3. **Hybrid Approaches**: Combine with other methods
4. **Evaluation**: Use multiple metrics for comprehensive assessment
5. **Diversity**: Promote variety in recommendations

Content-based filtering remains a fundamental approach in recommendation systems, particularly valuable for domains with rich item metadata and when interpretability is important. When combined with other techniques, it can create powerful hybrid recommendation systems that leverage the strengths of multiple approaches.

---

**Next**: [Collaborative Filtering](03_collaborative_filtering.md) - Discover how user-item interaction patterns drive recommendations through collective intelligence.
