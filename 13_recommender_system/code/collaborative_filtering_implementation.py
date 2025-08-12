import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFiltering:
    def __init__(self, method='user', similarity_metric='cosine', k_neighbors=10):
        """
        Collaborative Filtering Recommender
        
        Parameters:
        -----------
        method : str
            'user' for user-based CF, 'item' for item-based CF
        similarity_metric : str
            'cosine', 'pearson', 'jaccard', 'adjusted_cosine'
        k_neighbors : int
            Number of neighbors to consider
        """
        self.method = method
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_means = None
        self.item_means = None
        
    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Fit the collaborative filtering model"""
        # Create rating matrix
        self.rating_matrix = ratings_df.pivot_table(
            index=user_col, 
            columns=item_col, 
            values=rating_col, 
            fill_value=np.nan
        )
        
        # Compute means
        self.user_means = self.rating_matrix.mean(axis=1)
        self.item_means = self.rating_matrix.mean(axis=0)
        
        # Compute similarities
        if self.method == 'user':
            self.user_similarity = self._compute_user_similarity()
        else:
            self.item_similarity = self._compute_item_similarity()
            
        return self
    
    def _compute_user_similarity(self):
        """Compute user similarity matrix"""
        if self.similarity_metric == 'cosine':
            # Fill NaN with 0 for cosine similarity
            matrix_filled = self.rating_matrix.fillna(0)
            return cosine_similarity(matrix_filled)
        
        elif self.similarity_metric == 'pearson':
            # Compute Pearson correlation for each user pair
            n_users = len(self.rating_matrix)
            similarity_matrix = np.zeros((n_users, n_users))
            
            for i in range(n_users):
                for j in range(i+1, n_users):
                    # Get common rated items
                    user_i_ratings = self.rating_matrix.iloc[i]
                    user_j_ratings = self.rating_matrix.iloc[j]
                    
                    common_items = ~(user_i_ratings.isna() | user_j_ratings.isna())
                    
                    if common_items.sum() > 1:
                        corr, _ = pearsonr(
                            user_i_ratings[common_items], 
                            user_j_ratings[common_items]
                        )
                        similarity_matrix[i, j] = corr
                        similarity_matrix[j, i] = corr
                    else:
                        similarity_matrix[i, j] = 0
                        similarity_matrix[j, i] = 0
            
            return similarity_matrix
        
        elif self.similarity_metric == 'jaccard':
            # Convert to binary (rated/not rated)
            binary_matrix = ~self.rating_matrix.isna()
            return cosine_similarity(binary_matrix)
    
    def _compute_item_similarity(self):
        """Compute item similarity matrix"""
        if self.similarity_metric == 'cosine':
            # Fill NaN with 0 for cosine similarity
            matrix_filled = self.rating_matrix.fillna(0)
            return cosine_similarity(matrix_filled.T)
        
        elif self.similarity_metric == 'adjusted_cosine':
            # Center by user means
            centered_matrix = self.rating_matrix.sub(self.user_means, axis=0)
            # Fill NaN with 0
            centered_matrix = centered_matrix.fillna(0)
            return cosine_similarity(centered_matrix.T)
        
        elif self.similarity_metric == 'pearson':
            # Compute Pearson correlation for each item pair
            n_items = len(self.rating_matrix.columns)
            similarity_matrix = np.zeros((n_items, n_items))
            
            for i in range(n_items):
                for j in range(i+1, n_items):
                    # Get common users
                    item_i_ratings = self.rating_matrix.iloc[:, i]
                    item_j_ratings = self.rating_matrix.iloc[:, j]
                    
                    common_users = ~(item_i_ratings.isna() | item_j_ratings.isna())
                    
                    if common_users.sum() > 1:
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
        if self.method == 'user':
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)
    
    def _predict_user_based(self, user_id, item_id):
        """User-based prediction"""
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
    
    def _predict_item_based(self, user_id, item_id):
        """Item-based prediction"""
        if user_id not in self.rating_matrix.index or item_id not in self.rating_matrix.columns:
            return self.item_means.mean()
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        
        # Get item similarities
        item_similarities = self.item_similarity[item_idx]
        
        # Find items rated by this user
        user_ratings = self.rating_matrix.iloc[user_idx]
        rated_items = ~user_ratings.isna()
        
        if not rated_items.any():
            return self.item_means.mean()
        
        # Get similarities and ratings for items rated by this user
        similarities = item_similarities[rated_items]
        ratings = user_ratings[rated_items]
        
        # Sort by similarity and take top-k
        sorted_indices = np.argsort(similarities)[::-1][:self.k_neighbors]
        
        if len(sorted_indices) == 0:
            return self.item_means.mean()
        
        top_similarities = similarities.iloc[sorted_indices]
        top_ratings = ratings.iloc[sorted_indices]
        
        # Weighted average
        weighted_sum = np.sum(top_similarities * top_ratings)
        total_similarity = np.sum(np.abs(top_similarities))
        
        if total_similarity == 0:
            return top_ratings.mean()
        
        return weighted_sum / total_similarity
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate top-n recommendations for a user"""
        if user_id not in self.rating_matrix.index:
            return []
        
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_items = user_ratings.isna()
        
        if not unrated_items.any():
            return []
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in user_ratings[unrated_items].index:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id, n_similar=5):
        """Get most similar users"""
        if user_id not in self.rating_matrix.index:
            return []
        
        user_idx = self.rating_matrix.index.get_loc(user_id)
        similarities = self.user_similarity[user_idx]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
        similar_users = []
        
        for idx in sorted_indices:
            user_id_similar = self.rating_matrix.index[idx]
            similarity = similarities[idx]
            similar_users.append((user_id_similar, similarity))
        
        return similar_users
    
    def get_similar_items(self, item_id, n_similar=5):
        """Get most similar items"""
        if item_id not in self.rating_matrix.columns:
            return []
        
        item_idx = self.rating_matrix.columns.get_loc(item_id)
        similarities = self.item_similarity[item_idx]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1][1:n_similar+1]  # Exclude self
        similar_items = []
        
        for idx in sorted_indices:
            item_id_similar = self.rating_matrix.columns[idx]
            similarity = similarities[idx]
            similar_items.append((item_id_similar, similarity))
        
        return similar_items


def generate_synthetic_ratings_data(n_users=100, n_items=50, n_ratings=1000, random_state=42):
    """Generate synthetic ratings data with some structure"""
    np.random.seed(random_state)
    
    # Create synthetic ratings with some structure
    ratings_data = []
    for user_id in range(n_users):
        n_user_ratings = np.random.randint(5, 20)
        rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
        
        for item_id in rated_items:
            # Simulate user preferences (some users prefer certain item ranges)
            if user_id < 30:  # First group prefers items 0-15
                base_rating = 4 if item_id < 15 else 2
            elif user_id < 60:  # Second group prefers items 15-30
                base_rating = 4 if 15 <= item_id < 30 else 2
            else:  # Third group prefers items 30+
                base_rating = 4 if item_id >= 30 else 2
            
            # Add noise
            rating = max(1, min(5, base_rating + np.random.normal(0, 0.5)))
            ratings_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
    
    ratings_df = pd.DataFrame(ratings_data)
    return ratings_df


def demonstrate_basic_collaborative_filtering():
    """Demonstrate basic collaborative filtering functionality"""
    print("=== Basic Collaborative Filtering ===\n")
    
    # Generate synthetic data
    ratings_df = generate_synthetic_ratings_data()
    
    print("Synthetic Ratings Dataset:")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
    print(f"Number of items: {ratings_df['item_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Sparsity: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()):.3f}")
    
    # Test different collaborative filtering approaches
    methods = ['user', 'item']
    similarity_metrics = ['cosine', 'pearson']
    results = {}
    
    for method in methods:
        for metric in similarity_metrics:
            print(f"\n=== Testing {method.upper()}-based CF with {metric.upper()} similarity ===")
            
            # Initialize and fit model
            cf_model = CollaborativeFiltering(method=method, similarity_metric=metric, k_neighbors=10)
            cf_model.fit(ratings_df)
            
            # Test predictions for a sample user
            test_user = 0
            recommendations = cf_model.recommend(test_user, n_recommendations=5)
            
            print(f"Top 5 recommendations for User {test_user}:")
            for i, (item_id, pred_rating) in enumerate(recommendations):
                print(f"  {i+1}. Item {item_id}: Predicted rating = {pred_rating:.3f}")
            
            # Get similar users/items
            if method == 'user':
                similar_entities = cf_model.get_similar_users(test_user, n_similar=3)
                print(f"Most similar users to User {test_user}:")
            else:
                test_item = 0
                similar_entities = cf_model.get_similar_items(test_item, n_similar=3)
                print(f"Most similar items to Item {test_item}:")
            
            for entity_id, similarity in similar_entities:
                print(f"  {entity_id}: Similarity = {similarity:.3f}")
            
            # Store results for comparison
            results[f"{method}_{metric}"] = {
                'recommendations': recommendations,
                'similar_entities': similar_entities
            }
    
    return ratings_df, results


def demonstrate_similarity_metrics():
    """Demonstrate different similarity metrics"""
    print("=== Similarity Metrics Comparison ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_ratings_data(n_users=50, n_items=30)
    
    # Test different similarity metrics
    metrics = ['cosine', 'pearson', 'jaccard']
    results = {}
    
    for metric in metrics:
        print(f"Testing {metric.upper()} similarity...")
        
        # User-based CF
        cf_user = CollaborativeFiltering(method='user', similarity_metric=metric, k_neighbors=5)
        cf_user.fit(ratings_df)
        
        # Item-based CF
        cf_item = CollaborativeFiltering(method='item', similarity_metric=metric, k_neighbors=5)
        cf_item.fit(ratings_df)
        
        # Test predictions
        test_user = 0
        test_item = 0
        
        user_recommendations = cf_user.recommend(test_user, n_recommendations=3)
        item_recommendations = cf_item.recommend(test_user, n_recommendations=3)
        
        similar_users = cf_user.get_similar_users(test_user, n_similar=3)
        similar_items = cf_item.get_similar_items(test_item, n_similar=3)
        
        results[metric] = {
            'user_recommendations': user_recommendations,
            'item_recommendations': item_recommendations,
            'similar_users': similar_users,
            'similar_items': similar_items
        }
        
        print(f"  User-based recommendations: {user_recommendations}")
        print(f"  Item-based recommendations: {item_recommendations}")
        print(f"  Similar users: {similar_users}")
        print(f"  Similar items: {similar_items}")
    
    return results


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics for collaborative filtering"""
    print("=== Evaluation Metrics ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_ratings_data()
    
    # Split data for evaluation
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Test different methods
    methods = ['user', 'item']
    metrics = ['cosine', 'pearson']
    evaluation_results = {}
    
    for method in methods:
        for metric in metrics:
            print(f"Evaluating {method.upper()}-{metric.upper()}...")
            
            # Train model
            cf_model = CollaborativeFiltering(method=method, similarity_metric=metric, k_neighbors=10)
            cf_model.fit(train_df)
            
            # Make predictions on test set
            predictions = []
            actuals = []
            
            for _, row in test_df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                actual_rating = row['rating']
                
                # Only predict if user and item exist in training data
                if user_id in cf_model.rating_matrix.index and item_id in cf_model.rating_matrix.columns:
                    pred_rating = cf_model.predict(user_id, item_id)
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)
            
            if len(predictions) > 0:
                # Calculate metrics
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                
                evaluation_results[f"{method}_{metric}"] = {
                    'mae': mae,
                    'rmse': rmse,
                    'n_predictions': len(predictions)
                }
                
                print(f"  MAE: {mae:.3f}")
                print(f"  RMSE: {rmse:.3f}")
                print(f"  Number of predictions: {len(predictions)}")
    
    return evaluation_results


def demonstrate_visualization():
    """Demonstrate visualizations for collaborative filtering"""
    print("=== Collaborative Filtering Visualizations ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_ratings_data()
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Rating matrix heatmap (sample)
    plt.subplot(2, 3, 1)
    sample_matrix = ratings_df.pivot_table(
        index='user_id', columns='item_id', values='rating', fill_value=np.nan
    ).iloc[:20, :20]
    sns.heatmap(sample_matrix, cmap='viridis', cbar_kws={'label': 'Rating'})
    plt.title('Rating Matrix (Sample)')
    plt.xlabel('Item ID')
    plt.ylabel('User ID')
    
    # Plot 2: User similarity matrix (sample)
    plt.subplot(2, 3, 2)
    user_cf = CollaborativeFiltering(method='user', similarity_metric='cosine')
    user_cf.fit(ratings_df)
    sample_user_sim = user_cf.user_similarity[:20, :20]
    sns.heatmap(sample_user_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
    plt.title('User Similarity Matrix (Sample)')
    plt.xlabel('User ID')
    plt.ylabel('User ID')
    
    # Plot 3: Item similarity matrix (sample)
    plt.subplot(2, 3, 3)
    item_cf = CollaborativeFiltering(method='item', similarity_metric='cosine')
    item_cf.fit(ratings_df)
    sample_item_sim = item_cf.item_similarity[:20, :20]
    sns.heatmap(sample_item_sim, cmap='coolwarm', center=0, cbar_kws={'label': 'Similarity'})
    plt.title('Item Similarity Matrix (Sample)')
    plt.xlabel('Item ID')
    plt.ylabel('Item ID')
    
    # Plot 4: Rating distribution
    plt.subplot(2, 3, 4)
    ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Plot 5: Method comparison - predicted ratings
    plt.subplot(2, 3, 5)
    methods_comparison = []
    
    # Test different methods
    for method in ['user', 'item']:
        for metric in ['cosine', 'pearson']:
            cf_model = CollaborativeFiltering(method=method, similarity_metric=metric, k_neighbors=10)
            cf_model.fit(ratings_df)
            
            # Get recommendations for a test user
            test_user = 0
            recommendations = cf_model.recommend(test_user, n_recommendations=10)
            pred_ratings = [rating for _, rating in recommendations]
            
            methods_comparison.append({
                'method': f"{method.upper()}-{metric.upper()}",
                'ratings': pred_ratings
            })
    
    # Create box plot
    for i, method_data in enumerate(methods_comparison):
        plt.boxplot(method_data['ratings'], positions=[i], labels=[method_data['method']])
    
    plt.title('Predicted Ratings by Method')
    plt.ylabel('Predicted Rating')
    plt.xticks(rotation=45)
    
    # Plot 6: Similarity distribution
    plt.subplot(2, 3, 6)
    similarities = []
    
    # Collect similarities from different methods
    for method in ['user', 'item']:
        cf_model = CollaborativeFiltering(method=method, similarity_metric='cosine')
        cf_model.fit(ratings_df)
        
        if method == 'user':
            sim_matrix = cf_model.user_similarity
        else:
            sim_matrix = cf_model.item_similarity
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        similarities.extend(upper_triangle)
    
    plt.hist(similarities, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Similarity Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def demonstrate_cold_start():
    """Demonstrate cold start handling"""
    print("=== Cold Start Handling ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_ratings_data()
    
    # Simulate cold start scenarios
    print("1. New User Problem:")
    print("   - User with no ratings")
    print("   - Using popularity-based fallback")
    
    # Calculate item popularity
    item_popularity = ratings_df.groupby('item_id')['rating'].agg(['count', 'mean']).sort_values('count', ascending=False)
    
    print(f"   Most popular items:")
    for i, (item_id, (count, mean_rating)) in enumerate(item_popularity.head(5).iterrows()):
        print(f"     Item {item_id}: {count} ratings, avg rating {mean_rating:.2f}")
    
    print("\n2. New Item Problem:")
    print("   - Item with no ratings")
    print("   - Using user average ratings")
    
    # Calculate user average ratings
    user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean().sort_values(ascending=False)
    
    print(f"   Users with highest average ratings:")
    for user_id, avg_rating in user_avg_ratings.head(5).items():
        print(f"     User {user_id}: avg rating {avg_rating:.2f}")
    
    print("\n3. Hybrid Approach:")
    print("   - Combining collaborative filtering with content-based methods")
    print("   - Using weighted combination of predictions")
    
    # Simulate hybrid prediction
    cf_prediction = 3.5  # Collaborative filtering prediction
    cb_prediction = 4.2  # Content-based prediction
    alpha = 0.7  # Weight for CF
    
    hybrid_prediction = alpha * cf_prediction + (1 - alpha) * cb_prediction
    print(f"   Hybrid prediction: {alpha} * {cf_prediction} + {1-alpha} * {cb_prediction} = {hybrid_prediction:.2f}")
    
    return {
        'item_popularity': item_popularity,
        'user_avg_ratings': user_avg_ratings,
        'hybrid_example': hybrid_prediction
    }


def demonstrate_scalability():
    """Demonstrate scalability considerations"""
    print("=== Scalability Analysis ===\n")
    
    # Test with different dataset sizes
    dataset_sizes = [50, 100, 200, 500]
    training_times = []
    prediction_times = []
    
    for size in dataset_sizes:
        print(f"Testing with {size} users...")
        
        # Generate data
        ratings_df = generate_synthetic_ratings_data(n_users=size, n_items=size//2)
        
        # Time training
        import time
        start_time = time.time()
        
        cf_model = CollaborativeFiltering(method='user', similarity_metric='cosine', k_neighbors=10)
        cf_model.fit(ratings_df)
        
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        # Time predictions
        start_time = time.time()
        for user_id in range(min(5, size)):
            cf_model.recommend(user_id, n_recommendations=5)
        
        prediction_time = time.time() - start_time
        prediction_times.append(prediction_time)
        
        print(f"  Training time: {training_time:.3f}s")
        print(f"  Prediction time (5 users): {prediction_time:.3f}s")
    
    # Visualize scalability
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dataset_sizes, training_times, 'o-', label='Training Time')
    plt.xlabel('Dataset Size (users)')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time vs Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dataset_sizes, prediction_times, 'o-', label='Prediction Time')
    plt.xlabel('Dataset Size (users)')
    plt.ylabel('Time (seconds)')
    plt.title('Prediction Time vs Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'dataset_sizes': dataset_sizes,
        'training_times': training_times,
        'prediction_times': prediction_times
    }


def demonstrate_advanced_techniques():
    """Demonstrate advanced collaborative filtering techniques"""
    print("=== Advanced Techniques ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_ratings_data()
    
    print("1. Constrained Similarity:")
    print("   - Adding minimum overlap threshold")
    print("   - Filtering out users/items with insufficient common ratings")
    
    # Implement constrained similarity
    def constrained_similarity(similarity_matrix, rating_matrix, min_overlap=3):
        """Apply minimum overlap constraint to similarity matrix"""
        constrained_sim = similarity_matrix.copy()
        
        if similarity_matrix.shape[0] == len(rating_matrix):  # User similarity
            for i in range(len(rating_matrix)):
                for j in range(i+1, len(rating_matrix)):
                    # Count common rated items
                    user_i_ratings = rating_matrix.iloc[i]
                    user_j_ratings = rating_matrix.iloc[j]
                    common_items = ~(user_i_ratings.isna() | user_j_ratings.isna())
                    
                    if common_items.sum() < min_overlap:
                        constrained_sim[i, j] = 0
                        constrained_sim[j, i] = 0
        
        return constrained_sim
    
    # Test constrained similarity
    cf_model = CollaborativeFiltering(method='user', similarity_metric='cosine')
    cf_model.fit(ratings_df)
    
    original_similarities = cf_model.user_similarity[0, 1:6]  # First user with next 5 users
    constrained_sim = constrained_similarity(cf_model.user_similarity, cf_model.rating_matrix, min_overlap=5)
    constrained_similarities = constrained_sim[0, 1:6]
    
    print(f"   Original similarities: {original_similarities}")
    print(f"   Constrained similarities: {constrained_similarities}")
    
    print("\n2. Time-Aware Similarity:")
    print("   - Incorporating temporal information")
    print("   - Decaying similarity based on time difference")
    
    # Simulate time-aware similarity
    def time_aware_similarity(base_similarity, time_diff, decay_rate=0.1):
        """Apply temporal decay to similarity"""
        return base_similarity * np.exp(-decay_rate * time_diff)
    
    base_similarities = np.array([0.8, 0.6, 0.9, 0.4, 0.7])
    time_diffs = np.array([1, 3, 0, 5, 2])  # Time differences in months
    
    time_aware_similarities = time_aware_similarity(base_similarities, time_diffs)
    
    print(f"   Base similarities: {base_similarities}")
    print(f"   Time differences: {time_diffs}")
    print(f"   Time-aware similarities: {time_aware_similarities}")
    
    print("\n3. Category-Aware Similarity:")
    print("   - Weighting similarities by item categories")
    print("   - Different weights for different categories")
    
    # Simulate category-aware similarity
    def category_aware_similarity(similarities, categories, weights):
        """Apply category weights to similarities"""
        weighted_similarities = np.zeros_like(similarities)
        
        for i, (sim, cat) in enumerate(zip(similarities, categories)):
            weight = weights.get(cat, 1.0)  # Default weight 1.0
            weighted_similarities[i] = sim * weight
        
        return weighted_similarities
    
    similarities = np.array([0.8, 0.6, 0.9, 0.4, 0.7])
    categories = ['action', 'drama', 'action', 'comedy', 'drama']
    category_weights = {'action': 1.2, 'drama': 1.0, 'comedy': 0.8}
    
    category_aware_similarities = category_aware_similarity(similarities, categories, category_weights)
    
    print(f"   Base similarities: {similarities}")
    print(f"   Categories: {categories}")
    print(f"   Category weights: {category_weights}")
    print(f"   Category-aware similarities: {category_aware_similarities}")
    
    return {
        'constrained_example': (original_similarities, constrained_similarities),
        'time_aware_example': (base_similarities, time_aware_similarities),
        'category_aware_example': (similarities, category_aware_similarities)
    }


def main():
    """Main demonstration of collaborative filtering"""
    print("Collaborative Filtering: Implementation and Analysis")
    print("=" * 60)
    
    # 1. Basic collaborative filtering demonstration
    print("\n1. Basic Collaborative Filtering:")
    ratings_df, basic_results = demonstrate_basic_collaborative_filtering()
    
    # 2. Similarity metrics comparison
    print("\n2. Similarity Metrics Comparison:")
    similarity_results = demonstrate_similarity_metrics()
    
    # 3. Evaluation metrics
    print("\n3. Evaluation Metrics:")
    evaluation_results = demonstrate_evaluation_metrics()
    
    # 4. Visualizations
    print("\n4. Visualizations:")
    demonstrate_visualization()
    
    # 5. Cold start handling
    print("\n5. Cold Start Handling:")
    cold_start_results = demonstrate_cold_start()
    
    # 6. Scalability analysis
    print("\n6. Scalability Analysis:")
    scalability_results = demonstrate_scalability()
    
    # 7. Advanced techniques
    print("\n7. Advanced Techniques:")
    advanced_results = demonstrate_advanced_techniques()
    
    print("\n=== Key Insights ===")
    print("1. Collaborative filtering leverages user-item interaction patterns")
    print("2. Different similarity metrics produce different results")
    print("3. User-based and item-based approaches have different strengths")
    print("4. Evaluation requires multiple metrics for comprehensive assessment")
    print("5. Cold start can be handled with various strategies")
    print("6. Scalability becomes important with large datasets")
    print("7. Advanced techniques can improve performance and interpretability")
    
    return {
        'ratings_df': ratings_df,
        'basic_results': basic_results,
        'similarity_results': similarity_results,
        'evaluation_results': evaluation_results,
        'cold_start_results': cold_start_results,
        'scalability_results': scalability_results,
        'advanced_results': advanced_results
    }


if __name__ == "__main__":
    main()
