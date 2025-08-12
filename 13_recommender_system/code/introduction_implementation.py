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


def generate_synthetic_data(n_users=100, n_items=50, n_ratings=1000, random_state=42):
    """Generate synthetic ratings data for testing"""
    np.random.seed(random_state)
    
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
    
    return ratings_df


def demonstrate_basic_recommender_system():
    """Demonstrate basic recommender system functionality"""
    print("=== Basic Recommender System Demonstration ===\n")
    
    # Generate synthetic data
    ratings_df = generate_synthetic_data()
    
    print("Synthetic Ratings Dataset:")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
    print(f"Number of items: {ratings_df['item_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Sparsity: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()):.3f}")
    
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
    
    return ratings_df, results


def visualize_recommender_system(ratings_df, results):
    """Create comprehensive visualizations for recommender system analysis"""
    print("=== Recommender System Visualizations ===\n")
    
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
    
    test_cases = [(0, 10), (0, 20), (1, 15), (1, 25), (2, 30)]
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


def demonstrate_collaborative_filtering():
    """Demonstrate collaborative filtering in detail"""
    print("=== Collaborative Filtering Demonstration ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_data(n_users=50, n_items=30, n_ratings=500)
    
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='user_id', columns='item_id', values='rating', fill_value=0
    )
    
    # Compute similarities
    user_similarity = cosine_similarity(user_item_matrix)
    item_similarity = cosine_similarity(user_item_matrix.T)
    
    print("Similarity Analysis:")
    print(f"User similarity matrix shape: {user_similarity.shape}")
    print(f"Item similarity matrix shape: {item_similarity.shape}")
    print(f"Average user similarity: {user_similarity.mean():.3f}")
    print(f"Average item similarity: {item_similarity.mean():.3f}")
    
    # Find most similar users
    user_id = 0
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = user_similarity[user_idx]
    most_similar = np.argsort(similar_users)[-6:-1][::-1]  # Top 5 (excluding self)
    
    print(f"\nMost similar users to user {user_id}:")
    for i, sim_idx in enumerate(most_similar):
        sim_user_id = user_item_matrix.index[sim_idx]
        similarity = similar_users[sim_idx]
        print(f"  User {sim_user_id}: Similarity = {similarity:.3f}")
    
    # Find most similar items
    item_id = 0
    item_idx = user_item_matrix.columns.get_loc(item_id)
    similar_items = item_similarity[item_idx]
    most_similar_items = np.argsort(similar_items)[-6:-1][::-1]  # Top 5 (excluding self)
    
    print(f"\nMost similar items to item {item_id}:")
    for i, sim_idx in enumerate(most_similar_items):
        sim_item_id = user_item_matrix.columns[sim_idx]
        similarity = similar_items[sim_idx]
        print(f"  Item {sim_item_id}: Similarity = {similarity:.3f}")
    
    return user_item_matrix, user_similarity, item_similarity


def demonstrate_latent_factor_models():
    """Demonstrate latent factor models"""
    print("=== Latent Factor Models Demonstration ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_data(n_users=100, n_items=50, n_ratings=800)
    
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='user_id', columns='item_id', values='rating', fill_value=0
    )
    
    # Apply NMF for matrix factorization
    n_factors = 10
    nmf = NMF(n_components=n_factors, random_state=42)
    user_factors = nmf.fit_transform(user_item_matrix)
    item_factors = nmf.components_.T
    
    print("Latent Factor Analysis:")
    print(f"User factors shape: {user_factors.shape}")
    print(f"Item factors shape: {item_factors.shape}")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.3f}")
    
    # Analyze factor importance
    factor_importance = np.sum(np.abs(item_factors), axis=0)
    print(f"\nFactor importance (sum of absolute values):")
    for i, importance in enumerate(factor_importance):
        print(f"  Factor {i}: {importance:.3f}")
    
    # Visualize latent factors
    plt.figure(figsize=(12, 4))
    
    # Plot 1: User factors heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(user_factors[:20, :], cmap='viridis', cbar_kws={'label': 'Factor Value'})
    plt.title('User Factors (First 20 Users)')
    plt.xlabel('Latent Factor')
    plt.ylabel('User ID')
    
    # Plot 2: Item factors heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(item_factors[:20, :].T, cmap='viridis', cbar_kws={'label': 'Factor Value'})
    plt.title('Item Factors (First 20 Items)')
    plt.xlabel('Item ID')
    plt.ylabel('Latent Factor')
    
    # Plot 3: Factor importance
    plt.subplot(1, 3, 3)
    plt.bar(range(n_factors), factor_importance)
    plt.title('Factor Importance')
    plt.xlabel('Latent Factor')
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    return user_factors, item_factors, nmf


def demonstrate_content_based_filtering():
    """Demonstrate content-based filtering"""
    print("=== Content-Based Filtering Demonstration ===\n")
    
    # Generate data with item features
    np.random.seed(42)
    n_items = 50
    n_features = 5
    
    # Create item features (e.g., movie genres, book categories)
    item_features = np.random.rand(n_items, n_features)
    
    # Create user preferences
    n_users = 30
    user_preferences = np.random.rand(n_users, n_features)
    
    # Generate ratings based on user-item similarity
    ratings = []
    for user_id in range(n_users):
        for item_id in range(n_items):
            # Calculate similarity between user preferences and item features
            similarity = np.dot(user_preferences[user_id], item_features[item_id])
            # Add some noise
            rating = max(1, min(5, int(similarity * 2 + np.random.normal(0, 0.5))))
            ratings.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
    
    ratings_df = pd.DataFrame(ratings)
    
    print("Content-Based Filtering Analysis:")
    print(f"Number of items: {n_items}")
    print(f"Number of features: {n_features}")
    print(f"Number of users: {n_users}")
    print(f"Number of ratings: {len(ratings_df)}")
    
    # Calculate item-item similarity based on features
    item_similarity = cosine_similarity(item_features)
    
    # Find similar items
    target_item = 0
    similar_items = item_similarity[target_item]
    most_similar = np.argsort(similar_items)[-6:-1][::-1]
    
    print(f"\nMost similar items to item {target_item} (based on features):")
    for i, sim_idx in enumerate(most_similar):
        similarity = similar_items[sim_idx]
        print(f"  Item {sim_idx}: Similarity = {similarity:.3f}")
    
    # Visualize item features
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Item features heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(item_features[:20, :], cmap='viridis', cbar_kws={'label': 'Feature Value'})
    plt.title('Item Features (First 20 Items)')
    plt.xlabel('Feature')
    plt.ylabel('Item ID')
    
    # Plot 2: User preferences heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(user_preferences[:20, :], cmap='viridis', cbar_kws={'label': 'Preference Value'})
    plt.title('User Preferences (First 20 Users)')
    plt.xlabel('Feature')
    plt.ylabel('User ID')
    
    # Plot 3: Item similarity distribution
    plt.subplot(1, 3, 3)
    plt.hist(item_similarity.flatten(), bins=30, alpha=0.7)
    plt.title('Item Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return ratings_df, item_features, user_preferences, item_similarity


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics for recommender systems"""
    print("=== Evaluation Metrics Demonstration ===\n")
    
    # Generate synthetic data
    ratings_df = generate_synthetic_data(n_users=100, n_items=50, n_ratings=1000)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train different models
    methods = ['collaborative', 'latent', 'content']
    results = {}
    
    for method in methods:
        print(f"Training {method} model...")
        
        # Train model
        recommender = RecommenderSystem(method=method)
        recommender.fit(train_df)
        
        # Make predictions on test set
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            pred = recommender.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        # Calculate metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        
        results[method] = {
            'mae': mae,
            'rmse': rmse,
            'predictions': predictions,
            'actuals': actuals
        }
        
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: MAE comparison
    plt.subplot(1, 3, 1)
    methods_list = list(results.keys())
    mae_values = [results[method]['mae'] for method in methods_list]
    plt.bar(methods_list, mae_values)
    plt.title('Mean Absolute Error Comparison')
    plt.ylabel('MAE')
    
    # Plot 2: RMSE comparison
    plt.subplot(1, 3, 2)
    rmse_values = [results[method]['rmse'] for method in methods_list]
    plt.bar(methods_list, rmse_values)
    plt.title('Root Mean Square Error Comparison')
    plt.ylabel('RMSE')
    
    # Plot 3: Prediction vs Actual scatter plot (for best method)
    plt.subplot(1, 3, 3)
    best_method = min(results.keys(), key=lambda x: results[x]['rmse'])
    predictions = results[best_method]['predictions']
    actuals = results[best_method]['actuals']
    
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([1, 5], [1, 5], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f'Predictions vs Actual ({best_method})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_challenges():
    """Demonstrate common challenges in recommender systems"""
    print("=== Recommender System Challenges Demonstration ===\n")
    
    # 1. Sparsity Challenge
    print("1. Sparsity Challenge:")
    sparsity_levels = [0.95, 0.98, 0.99, 0.995]
    
    for sparsity in sparsity_levels:
        n_users, n_items = 100, 50
        n_ratings = int((1 - sparsity) * n_users * n_items)
        
        ratings_df = generate_synthetic_data(n_users, n_items, n_ratings)
        actual_sparsity = 1 - len(ratings_df) / (n_users * n_items)
        
        print(f"  Target sparsity: {sparsity:.3f}, Actual sparsity: {actual_sparsity:.3f}")
        print(f"  Number of ratings: {len(ratings_df)}")
    
    # 2. Cold Start Challenge
    print("\n2. Cold Start Challenge:")
    
    # Generate data with some new users and items
    ratings_df = generate_synthetic_data(n_users=100, n_items=50, n_ratings=800)
    
    # Add new users and items
    new_users = [100, 101, 102]  # Users with no ratings
    new_items = [50, 51, 52]     # Items with no ratings
    
    print("  New users (no ratings):", new_users)
    print("  New items (no ratings):", new_items)
    
    # Test prediction for new users/items
    recommender = RecommenderSystem(method='collaborative')
    recommender.fit(ratings_df)
    
    print("  Predictions for new user 100, item 10:", recommender.predict(100, 10))
    print("  Predictions for user 0, new item 50:", recommender.predict(0, 50))
    
    # 3. Popularity Bias
    print("\n3. Popularity Bias:")
    
    # Analyze rating distribution
    rating_counts = ratings_df['item_id'].value_counts()
    print(f"  Most popular item: {rating_counts.index[0]} ({rating_counts.iloc[0]} ratings)")
    print(f"  Least popular item: {rating_counts.index[-1]} ({rating_counts.iloc[-1]} ratings)")
    print(f"  Popularity ratio: {rating_counts.iloc[0] / rating_counts.iloc[-1]:.1f}:1")
    
    # Visualize popularity distribution
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Item popularity distribution
    plt.subplot(1, 3, 1)
    plt.hist(rating_counts.values, bins=20, alpha=0.7)
    plt.title('Item Popularity Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Items')
    
    # Plot 2: Rating distribution by popularity
    plt.subplot(1, 3, 2)
    popular_items = rating_counts.head(10).index
    unpopular_items = rating_counts.tail(10).index
    
    popular_ratings = ratings_df[ratings_df['item_id'].isin(popular_items)]['rating']
    unpopular_ratings = ratings_df[ratings_df['item_id'].isin(unpopular_items)]['rating']
    
    plt.hist(popular_ratings, alpha=0.7, label='Popular Items', bins=10)
    plt.hist(unpopular_ratings, alpha=0.7, label='Unpopular Items', bins=10)
    plt.title('Rating Distribution by Popularity')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 3: Sparsity vs performance
    plt.subplot(1, 3, 3)
    sparsity_levels = [0.95, 0.98, 0.99, 0.995]
    rmse_values = []
    
    for sparsity in sparsity_levels:
        n_users, n_items = 50, 25
        n_ratings = int((1 - sparsity) * n_users * n_items)
        
        if n_ratings > 0:
            ratings_df = generate_synthetic_data(n_users, n_items, n_ratings)
            train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
            
            recommender = RecommenderSystem(method='collaborative')
            recommender.fit(train_df)
            
            predictions = []
            actuals = []
            for _, row in test_df.iterrows():
                pred = recommender.predict(row['user_id'], row['item_id'])
                predictions.append(pred)
                actuals.append(row['rating'])
            
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            rmse_values.append(rmse)
        else:
            rmse_values.append(np.nan)
    
    plt.plot(sparsity_levels, rmse_values, 'o-')
    plt.title('Performance vs Sparsity')
    plt.xlabel('Sparsity')
    plt.ylabel('RMSE')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration of recommender system introduction"""
    print("Recommender System Introduction: Implementation and Analysis")
    print("=" * 70)
    
    # 1. Basic recommender system demonstration
    print("\n1. Basic Recommender System Demonstration:")
    ratings_df, results = demonstrate_basic_recommender_system()
    
    # 2. Visualization
    print("\n2. Recommender System Visualizations:")
    visualize_recommender_system(ratings_df, results)
    
    # 3. Collaborative filtering demonstration
    print("\n3. Collaborative Filtering Demonstration:")
    user_item_matrix, user_similarity, item_similarity = demonstrate_collaborative_filtering()
    
    # 4. Latent factor models demonstration
    print("\n4. Latent Factor Models Demonstration:")
    user_factors, item_factors, nmf = demonstrate_latent_factor_models()
    
    # 5. Content-based filtering demonstration
    print("\n5. Content-Based Filtering Demonstration:")
    content_ratings, item_features, user_preferences, item_similarity = demonstrate_content_based_filtering()
    
    # 6. Evaluation metrics demonstration
    print("\n6. Evaluation Metrics Demonstration:")
    evaluation_results = demonstrate_evaluation_metrics()
    
    # 7. Challenges demonstration
    print("\n7. Recommender System Challenges Demonstration:")
    demonstrate_challenges()
    
    print("\n=== Key Insights ===")
    print("1. Collaborative filtering leverages user-item interaction patterns")
    print("2. Latent factor models discover hidden patterns in the data")
    print("3. Content-based filtering uses item attributes and user preferences")
    print("4. Sparsity is a major challenge in real-world recommender systems")
    print("5. Cold start problem affects new users and items")
    print("6. Popularity bias can lead to filter bubbles")
    print("7. Multiple evaluation metrics are needed for comprehensive assessment")
    print("8. Different methods have different strengths and limitations")
    
    return {
        'ratings_df': ratings_df,
        'results': results,
        'user_item_matrix': user_item_matrix,
        'user_similarity': user_similarity,
        'item_similarity': item_similarity,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'content_ratings': content_ratings,
        'item_features': item_features,
        'user_preferences': user_preferences,
        'evaluation_results': evaluation_results
    }


if __name__ == "__main__":
    main()
