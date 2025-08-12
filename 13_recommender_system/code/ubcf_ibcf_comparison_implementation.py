import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate recommendations for a user"""
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
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate recommendations for a user"""
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


class HybridRecommender:
    """Hybrid recommendation system combining UBCF and IBCF"""
    
    def __init__(self, ubcf_model, ibcf_model, alpha=0.5):
        self.ubcf_model = ubcf_model
        self.ibcf_model = ibcf_model
        self.alpha = alpha  # Weight for UBCF, (1-alpha) for IBCF
        
    def predict(self, user_id, item_id):
        """Predict rating using hybrid approach"""
        ubcf_pred = self.ubcf_model.predict(user_id, item_id)
        ibcf_pred = self.ibcf_model.predict(user_id, item_id)
        
        # Handle NaN predictions
        if np.isnan(ubcf_pred) and np.isnan(ibcf_pred):
            return np.nan
        elif np.isnan(ubcf_pred):
            return ibcf_pred
        elif np.isnan(ibcf_pred):
            return ubcf_pred
        
        return self.alpha * ubcf_pred + (1 - self.alpha) * ibcf_pred
    
    def set_alpha(self, alpha):
        """Set the weight parameter"""
        self.alpha = alpha
    
    def optimize_alpha(self, validation_df):
        """Optimize alpha using validation data"""
        best_alpha = 0.5
        best_mae = float('inf')
        
        for alpha in np.arange(0, 1.1, 0.1):
            self.alpha = alpha
            mae = self.evaluate(validation_df)['mae']
            
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        
        self.alpha = best_alpha
        return best_alpha
    
    def evaluate(self, test_df):
        """Evaluate hybrid model"""
        return evaluate_model(self, test_df)


def generate_synthetic_data_with_clusters(n_users=200, n_items=100, n_ratings=2000, random_state=42):
    """Generate synthetic data with clear user/item clusters"""
    np.random.seed(random_state)
    
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
    return ratings_df


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


def demonstrate_basic_comparison():
    """Demonstrate basic UBCF vs IBCF comparison"""
    print("=== Basic UBCF vs IBCF Comparison ===\n")
    
    # Generate synthetic data
    ratings_df = generate_synthetic_data_with_clusters()
    
    print("Synthetic Dataset with User/Item Clusters:")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
    print(f"Number of items: {ratings_df['item_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Sparsity: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['item_id'].nunique()):.3f}")
    
    # Split data for evaluation
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
    
    return ratings_df, results, models


def demonstrate_similarity_analysis():
    """Demonstrate similarity analysis between UBCF and IBCF"""
    print("=== Similarity Analysis ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_data_with_clusters()
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train models
    ubcf = UBCF(similarity_metric='pearson', k_neighbors=15)
    ubcf.fit(train_df)
    
    ibcf = IBCF(similarity_metric='adjusted_cosine', k_neighbors=15)
    ibcf.fit(train_df)
    
    # Analyze similarity distributions
    user_similarities = ubcf.user_similarity[np.triu_indices_from(ubcf.user_similarity, k=1)]
    item_similarities = ibcf.item_similarity[np.triu_indices_from(ibcf.item_similarity, k=1)]
    
    print("Similarity Distribution Comparison:")
    print(f"UBCF User Similarities:")
    print(f"  Mean: {np.mean(user_similarities):.3f}")
    print(f"  Std: {np.std(user_similarities):.3f}")
    print(f"  Range: [{np.min(user_similarities):.3f}, {np.max(user_similarities):.3f}]")
    
    print(f"\nIBCF Item Similarities:")
    print(f"  Mean: {np.mean(item_similarities):.3f}")
    print(f"  Std: {np.std(item_similarities):.3f}")
    print(f"  Range: [{np.min(item_similarities):.3f}, {np.max(item_similarities):.3f}]")
    
    return ubcf, ibcf, user_similarities, item_similarities


def demonstrate_hybrid_approaches():
    """Demonstrate hybrid recommendation approaches"""
    print("=== Hybrid Recommendation Approaches ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_data_with_clusters()
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train base models
    ubcf = UBCF(similarity_metric='pearson', k_neighbors=15)
    ubcf.fit(train_df)
    
    ibcf = IBCF(similarity_metric='adjusted_cosine', k_neighbors=15)
    ibcf.fit(train_df)
    
    # Test different hybrid weights
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    hybrid_results = {}
    
    for alpha in weights:
        hybrid = HybridRecommender(ubcf, ibcf, alpha=alpha)
        results = evaluate_model(hybrid, test_df)
        hybrid_results[f"Hybrid-{alpha:.1f}"] = results
    
    # Display results
    print("Hybrid Approach Results:")
    for name, metrics in hybrid_results.items():
        print(f"{name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Coverage: {metrics['coverage']:.4f}")
        print()
    
    # Find optimal weight
    best_alpha = min(hybrid_results.keys(), key=lambda x: hybrid_results[x]['mae'])
    print(f"Best performing hybrid: {best_alpha}")
    
    return hybrid_results


def demonstrate_scalability_analysis():
    """Demonstrate scalability analysis"""
    print("=== Scalability Analysis ===\n")
    
    # Test with different dataset sizes
    dataset_sizes = [50, 100, 200, 500]
    training_times = {'UBCF': [], 'IBCF': []}
    
    for size in dataset_sizes:
        print(f"Testing with {size} users and {size//2} items...")
        
        # Generate data
        ratings_df = generate_synthetic_data_with_clusters(n_users=size, n_items=size//2)
        
        # Time UBCF training
        import time
        start_time = time.time()
        
        ubcf = UBCF(similarity_metric='pearson', k_neighbors=10)
        ubcf.fit(ratings_df)
        
        ubcf_time = time.time() - start_time
        training_times['UBCF'].append(ubcf_time)
        
        # Time IBCF training
        start_time = time.time()
        
        ibcf = IBCF(similarity_metric='adjusted_cosine', k_neighbors=10)
        ibcf.fit(ratings_df)
        
        ibcf_time = time.time() - start_time
        training_times['IBCF'].append(ibcf_time)
        
        print(f"  UBCF training time: {ubcf_time:.3f}s")
        print(f"  IBCF training time: {ibcf_time:.3f}s")
        print(f"  Ratio (UBCF/IBCF): {ubcf_time/ibcf_time:.2f}")
    
    return dataset_sizes, training_times


def demonstrate_visualization():
    """Demonstrate comprehensive visualizations"""
    print("=== UBCF vs IBCF Visualizations ===\n")
    
    # Generate data and train models
    ratings_df = generate_synthetic_data_with_clusters()
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train models
    ubcf_pearson = UBCF(similarity_metric='pearson', k_neighbors=15)
    ubcf_pearson.fit(train_df)
    
    ibcf_adjusted_cosine = IBCF(similarity_metric='adjusted_cosine', k_neighbors=15)
    ibcf_adjusted_cosine.fit(train_df)
    
    # Evaluate models
    models = {
        'UBCF-Pearson': ubcf_pearson,
        'IBCF-AdjustedCosine': ibcf_adjusted_cosine
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, test_df)
    
    # Create comprehensive visualizations
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
    plt.bar(results.keys(), mae_values, color=['blue', 'red'])
    plt.title('MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45)
    
    # Plot 5: RMSE comparison
    plt.subplot(3, 4, 5)
    rmse_values = [results[name]['rmse'] for name in results.keys()]
    plt.bar(results.keys(), rmse_values, color=['blue', 'red'])
    plt.title('RMSE Comparison')
    plt.ylabel('Root Mean Square Error')
    plt.xticks(rotation=45)
    
    # Plot 6: Coverage comparison
    plt.subplot(3, 4, 6)
    coverage_values = [results[name]['coverage'] for name in results.keys()]
    plt.bar(results.keys(), coverage_values, color=['blue', 'red'])
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
    
    return results


def demonstrate_detailed_analysis():
    """Demonstrate detailed analysis of UBCF vs IBCF"""
    print("=== Detailed Analysis ===\n")
    
    # Generate data
    ratings_df = generate_synthetic_data_with_clusters()
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    # Train models
    ubcf = UBCF(similarity_metric='pearson', k_neighbors=15)
    ubcf.fit(train_df)
    
    ibcf = IBCF(similarity_metric='adjusted_cosine', k_neighbors=15)
    ibcf.fit(train_df)
    
    # Compare prediction patterns
    test_sample = test_df.head(50)
    ubcf_preds = []
    ibcf_preds = []
    actuals = []
    
    for _, row in test_sample.iterrows():
        ubcf_pred = ubcf.predict(row['user_id'], row['item_id'])
        ibcf_pred = ibcf.predict(row['user_id'], row['item_id'])
        
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
    user_similarities = ubcf.user_similarity[np.triu_indices_from(ubcf.user_similarity, k=1)]
    item_similarities = ibcf.item_similarity[np.triu_indices_from(ibcf.item_similarity, k=1)]
    
    print(f"\nSimilarity Distribution Comparison:")
    print(f"UBCF User Similarities:")
    print(f"  Mean: {np.mean(user_similarities):.3f}")
    print(f"  Std: {np.std(user_similarities):.3f}")
    print(f"  Range: [{np.min(user_similarities):.3f}, {np.max(user_similarities):.3f}]")
    
    print(f"\nIBCF Item Similarities:")
    print(f"  Mean: {np.mean(item_similarities):.3f}")
    print(f"  Std: {np.std(item_similarities):.3f}")
    print(f"  Range: [{np.min(item_similarities):.3f}, {np.max(item_similarities):.3f}]")
    
    # Analyze cold start scenarios
    print(f"\nCold Start Analysis:")
    
    # New user scenario
    new_user_id = max(ratings_df['user_id']) + 1
    new_item_id = max(ratings_df['item_id']) + 1
    
    ubcf_new_user_pred = ubcf.predict(new_user_id, 0)  # Try to predict for new user
    ibcf_new_user_pred = ibcf.predict(new_user_id, 0)
    
    print(f"  New User Prediction (UBCF): {ubcf_new_user_pred}")
    print(f"  New User Prediction (IBCF): {ibcf_new_user_pred}")
    
    # New item scenario
    ubcf_new_item_pred = ubcf.predict(0, new_item_id)  # Try to predict for new item
    ibcf_new_item_pred = ibcf.predict(0, new_item_id)
    
    print(f"  New Item Prediction (UBCF): {ubcf_new_item_pred}")
    print(f"  New Item Prediction (IBCF): {ibcf_new_item_pred}")
    
    return {
        'ubcf_preds': ubcf_preds,
        'ibcf_preds': ibcf_preds,
        'actuals': actuals,
        'user_similarities': user_similarities,
        'item_similarities': item_similarities
    }


def main():
    """Main demonstration of UBCF vs IBCF comparison"""
    print("UBCF vs IBCF: Comprehensive Comparison and Analysis")
    print("=" * 60)
    
    # 1. Basic comparison demonstration
    print("\n1. Basic UBCF vs IBCF Comparison:")
    ratings_df, results, models = demonstrate_basic_comparison()
    
    # 2. Similarity analysis
    print("\n2. Similarity Analysis:")
    ubcf, ibcf, user_similarities, item_similarities = demonstrate_similarity_analysis()
    
    # 3. Hybrid approaches
    print("\n3. Hybrid Recommendation Approaches:")
    hybrid_results = demonstrate_hybrid_approaches()
    
    # 4. Scalability analysis
    print("\n4. Scalability Analysis:")
    dataset_sizes, training_times = demonstrate_scalability_analysis()
    
    # 5. Visualizations
    print("\n5. Comprehensive Visualizations:")
    viz_results = demonstrate_visualization()
    
    # 6. Detailed analysis
    print("\n6. Detailed Analysis:")
    detailed_results = demonstrate_detailed_analysis()
    
    print("\n=== Key Insights ===")
    print("1. UBCF excels in interpretability and handling new items")
    print("2. IBCF excels in scalability and stability")
    print("3. Hybrid approaches often provide the best performance")
    print("4. Choice depends on data characteristics and application requirements")
    print("5. Both methods have different strengths and use cases")
    print("6. Scalability considerations are crucial for large datasets")
    
    return {
        'ratings_df': ratings_df,
        'results': results,
        'models': models,
        'hybrid_results': hybrid_results,
        'training_times': training_times,
        'detailed_results': detailed_results
    }


if __name__ == "__main__":
    main()
