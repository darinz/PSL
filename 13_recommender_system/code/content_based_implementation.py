import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


def generate_synthetic_movie_data(n_movies=100, n_users=50, random_state=42):
    """Generate synthetic movie data for testing"""
    np.random.seed(random_state)
    
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
    
    return movies_df, ratings_df


def demonstrate_basic_content_based():
    """Demonstrate basic content-based recommender system"""
    print("=== Basic Content-Based Recommender System ===\n")
    
    # Generate synthetic data
    movies_df, ratings_df = generate_synthetic_movie_data()
    
    print("Synthetic Movie Dataset:")
    print(f"Number of movies: {len(movies_df)}")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
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
    
    return movies_df, ratings_df, recommender


def demonstrate_feature_importance():
    """Demonstrate feature importance analysis"""
    print("=== Feature Importance Analysis ===\n")
    
    # Generate data and train recommender
    movies_df, ratings_df, recommender = demonstrate_basic_content_based()
    
    # Get feature importance for multiple users
    test_users = [0, 1, 2]
    
    for user_id in test_users:
        feature_importance = recommender.get_feature_importance(user_id, top_features=10)
        
        print(f"\nTop 10 most important features for User {user_id}:")
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.3f}")
    
    # Visualize feature importance
    plt.figure(figsize=(15, 5))
    
    for i, user_id in enumerate(test_users):
        feature_importance = recommender.get_feature_importance(user_id, top_features=10)
        features, importances = zip(*feature_importance)
        
        plt.subplot(1, 3, i+1)
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - User {user_id}')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def demonstrate_similarity_metrics():
    """Demonstrate different similarity metrics"""
    print("=== Similarity Metrics Comparison ===\n")
    
    # Generate data
    movies_df, ratings_df = generate_synthetic_movie_data()
    
    # Compare different similarity metrics
    similarity_metrics = ['cosine', 'euclidean', 'pearson']
    results = {}
    
    for metric in similarity_metrics:
        print(f"Testing {metric.upper()} similarity...")
        
        recommender = ContentBasedRecommender(similarity_metric=metric)
        
        # Create profiles
        feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
        text_columns = ['description']
        recommender.create_item_profiles(movies_df, feature_columns, text_columns)
        recommender.create_user_profiles(ratings_df, movies_df)
        
        # Generate recommendations
        test_user = 0
        recommendations = recommender.recommend(test_user, n_recommendations=5)
        results[metric] = recommendations
        
        print(f"Top 5 recommendations:")
        for i, (item_idx, similarity) in enumerate(recommendations):
            movie = movies_df.iloc[item_idx]
            print(f"  {i+1}. {movie['title']} - Similarity: {similarity:.3f}")
    
    # Visualize similarity distributions
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
    
    return results


def demonstrate_profile_visualization():
    """Demonstrate profile visualization using PCA"""
    print("=== Profile Visualization ===\n")
    
    # Generate data and train recommender
    movies_df, ratings_df, recommender = demonstrate_basic_content_based()
    
    # Visualize profiles
    recommender.visualize_profiles(user_ids=[0, 1, 2], n_items=30)
    
    # Additional analysis: Profile clustering
    print("\nAnalyzing profile clusters...")
    
    # Get all user profiles
    user_profiles_array = np.array(list(recommender.user_profiles.values()))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(user_profiles_array)
    
    # Visualize clusters
    pca = PCA(n_components=2)
    profiles_2d = pca.fit_transform(user_profiles_array)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(profiles_2d[:, 0], profiles_2d[:, 1], c=clusters, cmap='viridis', s=100)
    plt.colorbar(scatter)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('User Profile Clusters')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze cluster characteristics
    for cluster_id in range(3):
        cluster_users = [user_id for user_id, cluster in zip(recommender.user_profiles.keys(), clusters) if cluster == cluster_id]
        print(f"\nCluster {cluster_id} has {len(cluster_users)} users")
        
        # Get average feature importance for this cluster
        cluster_profiles = [recommender.user_profiles[user_id] for user_id in cluster_users]
        avg_profile = np.mean(cluster_profiles, axis=0)
        
        # Get top features for this cluster
        feature_importance = [(name, abs(value)) for name, value in zip(recommender.feature_names, avg_profile)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 5 features for Cluster {cluster_id}:")
        for feature, importance in feature_importance[:5]:
            print(f"  {feature}: {importance:.3f}")


def demonstrate_advanced_features():
    """Demonstrate advanced content-based features"""
    print("=== Advanced Content-Based Features ===\n")
    
    # Generate data
    movies_df, ratings_df = generate_synthetic_movie_data()
    
    # Add more complex features
    movies_df['decade'] = (movies_df['year'] // 10) * 10
    movies_df['budget_category'] = pd.cut(movies_df['budget'], bins=3, labels=['Low', 'Medium', 'High'])
    movies_df['rating_category'] = pd.cut(movies_df['rating'], bins=3, labels=['Poor', 'Average', 'Good'])
    
    # Create recommender with advanced features
    recommender = ContentBasedRecommender(similarity_metric='cosine')
    
    # Use more comprehensive feature set
    feature_columns = ['genre', 'year', 'rating', 'budget', 'director', 'decade', 'budget_category', 'rating_category']
    text_columns = ['description']
    
    item_profiles = recommender.create_item_profiles(movies_df, feature_columns, text_columns)
    user_profiles = recommender.create_user_profiles(ratings_df, movies_df)
    
    print(f"Advanced feature set:")
    print(f"Number of features: {len(recommender.feature_names)}")
    print(f"Feature names: {recommender.feature_names[:10]}...")  # Show first 10
    
    # Generate recommendations
    test_user = 0
    recommendations = recommender.recommend(test_user, n_recommendations=10)
    
    print(f"\nTop 10 recommendations with advanced features:")
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"{i+1}. {movie['title']} ({movie['genre']}, {movie['year']}, {movie['budget_category']}) - Similarity: {similarity:.3f}")
    
    # Analyze feature diversity
    print(f"\nFeature Analysis:")
    print(f"Total features: {len(recommender.feature_names)}")
    
    # Count feature types
    categorical_features = [f for f in recommender.feature_names if '_' in f and not f.startswith('description_')]
    numerical_features = [f for f in recommender.feature_names if f not in categorical_features and not f.startswith('description_')]
    text_features = [f for f in recommender.feature_names if f.startswith('description_')]
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Text features: {len(text_features)}")
    
    return movies_df, ratings_df, recommender


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics for content-based systems"""
    print("=== Evaluation Metrics ===\n")
    
    # Generate data
    movies_df, ratings_df = generate_synthetic_movie_data()
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train recommender on training data
    recommender = ContentBasedRecommender(similarity_metric='cosine')
    
    feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
    text_columns = ['description']
    
    recommender.create_item_profiles(movies_df, feature_columns, text_columns)
    recommender.create_user_profiles(train_ratings, movies_df)
    
    # Evaluate on test data
    precision_scores = []
    recall_scores = []
    
    test_users = test_ratings['user_id'].unique()[:10]  # Test on first 10 users
    
    for user_id in test_users:
        # Get recommendations
        recommendations = recommender.recommend(user_id, n_recommendations=10)
        recommended_items = [item_idx for item_idx, _ in recommendations]
        
        # Get ground truth (items rated 4+ in test set)
        user_test_ratings = test_ratings[test_ratings['user_id'] == user_id]
        true_items = user_test_ratings[user_test_ratings['rating'] >= 4]['movie_id'].values
        
        # Convert to item indices
        true_indices = [movies_df[movies_df['movie_id'] == item_id].index[0] for item_id in true_items]
        
        # Compute precision and recall
        if len(recommended_items) > 0:
            precision = len(set(recommended_items) & set(true_indices)) / len(recommended_items)
            precision_scores.append(precision)
        
        if len(true_indices) > 0:
            recall = len(set(recommended_items) & set(true_indices)) / len(true_indices)
            recall_scores.append(recall)
    
    # Calculate average metrics
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    print(f"Evaluation Results:")
    print(f"Average Precision@10: {avg_precision:.3f}")
    print(f"Average Recall@10: {avg_recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(precision_scores, bins=10, alpha=0.7, edgecolor='black')
    plt.title('Precision Distribution')
    plt.xlabel('Precision@10')
    plt.ylabel('Frequency')
    plt.axvline(avg_precision, color='red', linestyle='--', label=f'Mean: {avg_precision:.3f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(recall_scores, bins=10, alpha=0.7, edgecolor='black')
    plt.title('Recall Distribution')
    plt.xlabel('Recall@10')
    plt.ylabel('Frequency')
    plt.axvline(avg_recall, color='red', linestyle='--', label=f'Mean: {avg_recall:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': f1_score,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores
    }


def demonstrate_cold_start():
    """Demonstrate cold start handling"""
    print("=== Cold Start Handling ===\n")
    
    # Generate data
    movies_df, ratings_df = generate_synthetic_movie_data()
    
    # Create recommender
    recommender = ContentBasedRecommender(similarity_metric='cosine')
    
    feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
    text_columns = ['description']
    
    recommender.create_item_profiles(movies_df, feature_columns, text_columns)
    recommender.create_user_profiles(ratings_df, movies_df)
    
    # Simulate new user (no ratings)
    new_user_id = 999
    
    # Strategy 1: Use popular items as initial profile
    popular_items = ratings_df.groupby('movie_id')['rating'].mean().sort_values(ascending=False).head(10)
    popular_indices = [movies_df[movies_df['movie_id'] == item_id].index[0] for item_id in popular_items.index]
    popular_profiles = recommender.item_profiles[popular_indices]
    new_user_profile = np.mean(popular_profiles, axis=0)
    
    # Add to recommender
    recommender.user_profiles[new_user_id] = new_user_profile
    
    # Generate recommendations for new user
    recommendations = recommender.recommend(new_user_id, n_recommendations=10)
    
    print(f"Recommendations for new user (popular items strategy):")
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"{i+1}. {movie['title']} ({movie['genre']}, {movie['year']}) - Similarity: {similarity:.3f}")
    
    # Strategy 2: Use genre-based profile
    genre_preferences = {'Action': 0.8, 'Drama': 0.3, 'Comedy': 0.6, 'Thriller': 0.9, 'Romance': 0.2}
    
    # Create profile based on genre preferences
    genre_profile = np.zeros(len(recommender.feature_names))
    for i, feature in enumerate(recommender.feature_names):
        if feature.startswith('genre_'):
            genre = feature.replace('genre_', '')
            if genre in genre_preferences:
                genre_profile[i] = genre_preferences[genre]
    
    # Normalize profile
    genre_profile = genre_profile / np.linalg.norm(genre_profile)
    
    # Add to recommender
    recommender.user_profiles[new_user_id] = genre_profile
    
    # Generate recommendations
    recommendations = recommender.recommend(new_user_id, n_recommendations=10)
    
    print(f"\nRecommendations for new user (genre preferences strategy):")
    for i, (item_idx, similarity) in enumerate(recommendations):
        movie = movies_df.iloc[item_idx]
        print(f"{i+1}. {movie['title']} ({movie['genre']}, {movie['year']}) - Similarity: {similarity:.3f}")
    
    return recommender


def demonstrate_scalability():
    """Demonstrate scalability considerations"""
    print("=== Scalability Analysis ===\n")
    
    # Test with different dataset sizes
    dataset_sizes = [50, 100, 200, 500]
    training_times = []
    recommendation_times = []
    
    for size in dataset_sizes:
        print(f"Testing with {size} movies...")
        
        # Generate data
        movies_df, ratings_df = generate_synthetic_movie_data(n_movies=size, n_users=size//2)
        
        # Time training
        import time
        start_time = time.time()
        
        recommender = ContentBasedRecommender(similarity_metric='cosine')
        feature_columns = ['genre', 'year', 'rating', 'budget', 'director']
        text_columns = ['description']
        
        recommender.create_item_profiles(movies_df, feature_columns, text_columns)
        recommender.create_user_profiles(ratings_df, movies_df)
        
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        # Time recommendations
        start_time = time.time()
        for user_id in list(recommender.user_profiles.keys())[:5]:
            recommender.recommend(user_id, n_recommendations=10)
        
        recommendation_time = time.time() - start_time
        recommendation_times.append(recommendation_time)
        
        print(f"  Training time: {training_time:.3f}s")
        print(f"  Recommendation time (5 users): {recommendation_time:.3f}s")
    
    # Visualize scalability
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dataset_sizes, training_times, 'o-', label='Training Time')
    plt.xlabel('Dataset Size (movies)')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time vs Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(dataset_sizes, recommendation_times, 'o-', label='Recommendation Time')
    plt.xlabel('Dataset Size (movies)')
    plt.ylabel('Time (seconds)')
    plt.title('Recommendation Time vs Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'dataset_sizes': dataset_sizes,
        'training_times': training_times,
        'recommendation_times': recommendation_times
    }


def main():
    """Main demonstration of content-based recommender system"""
    print("Content-Based Recommender System: Implementation and Analysis")
    print("=" * 70)
    
    # 1. Basic content-based demonstration
    print("\n1. Basic Content-Based Recommender System:")
    movies_df, ratings_df, recommender = demonstrate_basic_content_based()
    
    # 2. Feature importance analysis
    print("\n2. Feature Importance Analysis:")
    demonstrate_feature_importance()
    
    # 3. Similarity metrics comparison
    print("\n3. Similarity Metrics Comparison:")
    similarity_results = demonstrate_similarity_metrics()
    
    # 4. Profile visualization
    print("\n4. Profile Visualization:")
    demonstrate_profile_visualization()
    
    # 5. Advanced features
    print("\n5. Advanced Features:")
    advanced_movies, advanced_ratings, advanced_recommender = demonstrate_advanced_features()
    
    # 6. Evaluation metrics
    print("\n6. Evaluation Metrics:")
    evaluation_results = demonstrate_evaluation_metrics()
    
    # 7. Cold start handling
    print("\n7. Cold Start Handling:")
    cold_start_recommender = demonstrate_cold_start()
    
    # 8. Scalability analysis
    print("\n8. Scalability Analysis:")
    scalability_results = demonstrate_scalability()
    
    print("\n=== Key Insights ===")
    print("1. Content-based filtering leverages item features for recommendations")
    print("2. Feature engineering is crucial for system performance")
    print("3. Different similarity metrics can produce different results")
    print("4. User profiles can be clustered to understand user segments")
    print("5. Advanced features improve recommendation quality")
    print("6. Evaluation requires multiple metrics for comprehensive assessment")
    print("7. Cold start can be handled with various strategies")
    print("8. Scalability becomes important with large datasets")
    
    return {
        'movies_df': movies_df,
        'ratings_df': ratings_df,
        'recommender': recommender,
        'similarity_results': similarity_results,
        'evaluation_results': evaluation_results,
        'scalability_results': scalability_results
    }


if __name__ == "__main__":
    main()
