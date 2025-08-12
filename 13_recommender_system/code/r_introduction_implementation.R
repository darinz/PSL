# Recommender System Implementation in R

# Load required libraries
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Generate synthetic data
generate_synthetic_data <- function(n_users = 100, n_items = 50, n_ratings = 1000, seed = 42) {
  set.seed(seed)
  
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
  
  return(ratings_df)
}

# Create rating matrix from data frame
create_rating_matrix <- function(ratings_df) {
  rating_matrix <- ratings_df %>%
    spread(item_id, rating, fill = 0) %>%
    select(-user_id) %>%
    as.matrix()
  
  return(rating_matrix)
}

# Demonstrate basic recommender system
demonstrate_basic_recommender_system <- function() {
  cat("=== Basic Recommender System Demonstration ===\n\n")
  
  # Generate synthetic data
  ratings_df <- generate_synthetic_data()
  
  cat("Synthetic Ratings Dataset:\n")
  cat("Number of users:", length(unique(ratings_df$user_id)), "\n")
  cat("Number of items:", length(unique(ratings_df$item_id)), "\n")
  cat("Number of ratings:", nrow(ratings_df), "\n")
  cat("Sparsity:", 1 - nrow(ratings_df) / (length(unique(ratings_df$user_id)) * length(unique(ratings_df$item_id))), "\n")
  
  # Create rating matrix
  rating_matrix <- create_rating_matrix(ratings_df)
  
  # Convert to realRatingMatrix for recommenderlab
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  
  # Test different recommendation methods
  methods <- c("UBCF", "IBCF", "POPULAR")
  results <- list()
  
  for (method in methods) {
    cat("\n=== Testing", method, "===\n")
    
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
  
  return(list(ratings_df = ratings_df, rating_matrix = rating_matrix, results = results))
}

# Visualize recommender system
visualize_recommender_system <- function(ratings_df, rating_matrix) {
  cat("=== Recommender System Visualizations ===\n\n")
  
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
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(p1 = p1, p2 = p2))
}

# Demonstrate collaborative filtering
demonstrate_collaborative_filtering <- function() {
  cat("=== Collaborative Filtering Demonstration ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_data(n_users = 50, n_items = 30, n_ratings = 500)
  rating_matrix <- create_rating_matrix(ratings_df)
  
  # Convert to realRatingMatrix
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  
  # User-based collaborative filtering
  ubcf_model <- Recommender(rating_matrix_real, method = "UBCF")
  
  # Item-based collaborative filtering
  ibcf_model <- Recommender(rating_matrix_real, method = "IBCF")
  
  # Generate recommendations
  ubcf_recommendations <- predict(ubcf_model, rating_matrix_real[1:5], n = 5)
  ibcf_recommendations <- predict(ibcf_model, rating_matrix_real[1:5], n = 5)
  
  cat("User-Based Collaborative Filtering:\n")
  for (i in 1:5) {
    cat("User", i, "recommendations:", as(ubcf_recommendations[i], "list")[[1]], "\n")
  }
  
  cat("\nItem-Based Collaborative Filtering:\n")
  for (i in 1:5) {
    cat("User", i, "recommendations:", as(ibcf_recommendations[i], "list")[[1]], "\n")
  }
  
  # Analyze similarity matrices
  ubcf_similarity <- similarity(rating_matrix_real[1:10], method = "cosine")
  ibcf_similarity <- similarity(rating_matrix_real[, 1:10], method = "cosine")
  
  cat("\nSimilarity Analysis:\n")
  cat("User similarity matrix dimensions:", dim(ubcf_similarity), "\n")
  cat("Item similarity matrix dimensions:", dim(ibcf_similarity), "\n")
  cat("Average user similarity:", mean(as.matrix(ubcf_similarity), na.rm = TRUE), "\n")
  cat("Average item similarity:", mean(as.matrix(ibcf_similarity), na.rm = TRUE), "\n")
  
  return(list(
    ubcf_model = ubcf_model,
    ibcf_model = ibcf_model,
    ubcf_recommendations = ubcf_recommendations,
    ibcf_recommendations = ibcf_recommendations,
    ubcf_similarity = ubcf_similarity,
    ibcf_similarity = ibcf_similarity
  ))
}

# Demonstrate latent factor models
demonstrate_latent_factor_models <- function() {
  cat("=== Latent Factor Models Demonstration ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_data(n_users = 100, n_items = 50, n_ratings = 800)
  rating_matrix <- create_rating_matrix(ratings_df)
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  
  # SVD-based latent factor model
  svd_model <- Recommender(rating_matrix_real, method = "SVD")
  
  # Generate recommendations
  svd_recommendations <- predict(svd_model, rating_matrix_real[1:5], n = 5)
  
  cat("SVD-Based Latent Factor Model:\n")
  for (i in 1:5) {
    cat("User", i, "recommendations:", as(svd_recommendations[i], "list")[[1]], "\n")
  }
  
  # Analyze model parameters
  model_info <- getModel(svd_model)
  cat("\nModel Analysis:\n")
  cat("Number of factors:", ncol(model_info$svd$v), "\n")
  cat("Reconstruction error:", model_info$svd$d[1], "\n")
  
  # Visualize latent factors
  user_factors <- model_info$svd$u
  item_factors <- model_info$svd$v
  
  # Create factor importance plot
  factor_importance <- colSums(abs(item_factors))
  factor_df <- data.frame(
    factor = 1:length(factor_importance),
    importance = factor_importance
  )
  
  p1 <- ggplot(factor_df, aes(x = factor, y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Latent Factor Importance",
         x = "Latent Factor", y = "Importance") +
    theme_minimal()
  
  # User factors heatmap (first 20 users, first 10 factors)
  user_factors_df <- expand.grid(
    user_id = 1:20,
    factor = 1:10
  )
  user_factors_df$value <- as.vector(user_factors[1:20, 1:10])
  
  p2 <- ggplot(user_factors_df, aes(x = factor, y = user_id, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "User Factors (First 20 Users, 10 Factors)",
         x = "Latent Factor", y = "User ID") +
    theme_minimal()
  
  # Item factors heatmap (first 20 items, first 10 factors)
  item_factors_df <- expand.grid(
    item_id = 1:20,
    factor = 1:10
  )
  item_factors_df$value <- as.vector(item_factors[1:20, 1:10])
  
  p3 <- ggplot(item_factors_df, aes(x = factor, y = item_id, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "Item Factors (First 20 Items, 10 Factors)",
         x = "Latent Factor", y = "Item ID") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(
    svd_model = svd_model,
    svd_recommendations = svd_recommendations,
    user_factors = user_factors,
    item_factors = item_factors,
    factor_importance = factor_importance
  ))
}

# Demonstrate content-based filtering
demonstrate_content_based_filtering <- function() {
  cat("=== Content-Based Filtering Demonstration ===\n\n")
  
  # Generate data with item features
  set.seed(42)
  n_items <- 50
  n_features <- 5
  
  # Create item features (e.g., movie genres, book categories)
  item_features <- matrix(runif(n_items * n_features), nrow = n_items, ncol = n_features)
  
  # Create user preferences
  n_users <- 30
  user_preferences <- matrix(runif(n_users * n_features), nrow = n_users, ncol = n_features)
  
  # Generate ratings based on user-item similarity
  ratings <- data.frame()
  for (user_id in 1:n_users) {
    for (item_id in 1:n_items) {
      # Calculate similarity between user preferences and item features
      similarity <- sum(user_preferences[user_id, ] * item_features[item_id, ])
      # Add some noise
      rating <- max(1, min(5, round(similarity * 2 + rnorm(1, 0, 0.5))))
      ratings <- rbind(ratings, data.frame(
        user_id = user_id,
        item_id = item_id,
        rating = rating
      ))
    }
  }
  
  cat("Content-Based Filtering Analysis:\n")
  cat("Number of items:", n_items, "\n")
  cat("Number of features:", n_features, "\n")
  cat("Number of users:", n_users, "\n")
  cat("Number of ratings:", nrow(ratings), "\n")
  
  # Calculate item-item similarity based on features
  item_similarity <- cor(t(item_features))
  
  # Find similar items
  target_item <- 1
  similar_items <- item_similarity[target_item, ]
  most_similar <- order(similar_items, decreasing = TRUE)[2:6]  # Top 5 (excluding self)
  
  cat("\nMost similar items to item", target_item, "(based on features):\n")
  for (i in 1:5) {
    sim_idx <- most_similar[i]
    similarity <- similar_items[sim_idx]
    cat("  Item", sim_idx, ": Similarity =", round(similarity, 3), "\n")
  }
  
  # Visualize item features
  # Item features heatmap (first 20 items)
  item_features_df <- expand.grid(
    item_id = 1:20,
    feature = 1:n_features
  )
  item_features_df$value <- as.vector(item_features[1:20, ])
  
  p1 <- ggplot(item_features_df, aes(x = feature, y = item_id, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "Item Features (First 20 Items)",
         x = "Feature", y = "Item ID") +
    theme_minimal()
  
  # User preferences heatmap (first 20 users)
  user_preferences_df <- expand.grid(
    user_id = 1:20,
    feature = 1:n_features
  )
  user_preferences_df$value <- as.vector(user_preferences[1:20, ])
  
  p2 <- ggplot(user_preferences_df, aes(x = feature, y = user_id, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "User Preferences (First 20 Users)",
         x = "Feature", y = "User ID") +
    theme_minimal()
  
  # Item similarity distribution
  similarity_values <- as.vector(item_similarity[upper.tri(item_similarity)])
  similarity_df <- data.frame(similarity = similarity_values)
  
  p3 <- ggplot(similarity_df, aes(x = similarity)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
    labs(title = "Item Similarity Distribution",
         x = "Similarity", y = "Frequency") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(
    ratings = ratings,
    item_features = item_features,
    user_preferences = user_preferences,
    item_similarity = item_similarity
  ))
}

# Demonstrate evaluation metrics
demonstrate_evaluation_metrics <- function() {
  cat("=== Evaluation Metrics Demonstration ===\n\n")
  
  # Generate synthetic data
  ratings_df <- generate_synthetic_data(n_users = 100, n_items = 50, n_ratings = 1000)
  rating_matrix <- create_rating_matrix(ratings_df)
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  
  # Split data (simplified - in practice, use proper cross-validation)
  set.seed(42)
  train_indices <- sample(1:nrow(rating_matrix_real), 0.8 * nrow(rating_matrix_real))
  train_data <- rating_matrix_real[train_indices]
  test_data <- rating_matrix_real[-train_indices]
  
  # Train different models
  methods <- c("UBCF", "IBCF", "POPULAR")
  results <- list()
  
  for (method in methods) {
    cat("Training", method, "model...\n")
    
    # Train model
    model <- Recommender(train_data, method = method)
    
    # Generate predictions
    predictions <- predict(model, test_data, type = "ratings")
    
    # Calculate evaluation metrics
    evaluation <- calcPredictionAccuracy(predictions, test_data)
    
    results[[method]] <- list(
      model = model,
      predictions = predictions,
      evaluation = evaluation
    )
    
    cat("  RMSE:", round(evaluation["RMSE"], 3), "\n")
    cat("  MAE:", round(evaluation["MAE"], 3), "\n")
  }
  
  # Visualize results
  # Create comparison data frame
  comparison_df <- data.frame(
    method = rep(methods, each = 2),
    metric = rep(c("RMSE", "MAE"), times = length(methods)),
    value = c(
      results$UBCF$evaluation["RMSE"], results$UBCF$evaluation["MAE"],
      results$IBCF$evaluation["RMSE"], results$IBCF$evaluation["MAE"],
      results$POPULAR$evaluation["RMSE"], results$POPULAR$evaluation["MAE"]
    )
  )
  
  p1 <- ggplot(comparison_df, aes(x = method, y = value, fill = metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Evaluation Metrics Comparison",
         x = "Method", y = "Error", fill = "Metric") +
    theme_minimal()
  
  # Prediction vs Actual scatter plot (for best method)
  best_method <- names(which.min(sapply(results, function(x) x$evaluation["RMSE"])))
  predictions <- results[[best_method]]$predictions
  actuals <- getRatings(test_data)
  
  # Flatten predictions and actuals for plotting
  pred_values <- as.vector(as(predictions, "matrix"))
  actual_values <- as.vector(actuals)
  
  # Remove NA values
  valid_indices <- !is.na(pred_values) & !is.na(actual_values)
  pred_values <- pred_values[valid_indices]
  actual_values <- actual_values[valid_indices]
  
  # Sample for plotting (if too many points)
  if (length(pred_values) > 1000) {
    sample_indices <- sample(1:length(pred_values), 1000)
    pred_values <- pred_values[sample_indices]
    actual_values <- actual_values[sample_indices]
  }
  
  scatter_df <- data.frame(
    actual = actual_values,
    predicted = pred_values
  )
  
  p2 <- ggplot(scatter_df, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = paste("Predictions vs Actual (", best_method, ")", sep = ""),
         x = "Actual Rating", y = "Predicted Rating") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(results)
}

# Demonstrate challenges
demonstrate_challenges <- function() {
  cat("=== Recommender System Challenges Demonstration ===\n\n")
  
  # 1. Sparsity Challenge
  cat("1. Sparsity Challenge:\n")
  sparsity_levels <- c(0.95, 0.98, 0.99, 0.995)
  
  for (sparsity in sparsity_levels) {
    n_users <- 100
    n_items <- 50
    n_ratings <- round((1 - sparsity) * n_users * n_items)
    
    if (n_ratings > 0) {
      ratings_df <- generate_synthetic_data(n_users, n_items, n_ratings)
      actual_sparsity <- 1 - nrow(ratings_df) / (n_users * n_items)
      
      cat("  Target sparsity:", sparsity, "Actual sparsity:", round(actual_sparsity, 3), "\n")
      cat("  Number of ratings:", nrow(ratings_df), "\n")
    }
  }
  
  # 2. Cold Start Challenge
  cat("\n2. Cold Start Challenge:\n")
  
  # Generate data with some new users and items
  ratings_df <- generate_synthetic_data(n_users = 100, n_items = 50, n_ratings = 800)
  
  # Add new users and items
  new_users <- c(101, 102, 103)  # Users with no ratings
  new_items <- c(51, 52, 53)     # Items with no ratings
  
  cat("  New users (no ratings):", paste(new_users, collapse = ", "), "\n")
  cat("  New items (no ratings):", paste(new_items, collapse = ", "), "\n")
  
  # Test prediction for new users/items
  rating_matrix <- create_rating_matrix(ratings_df)
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  
  model <- Recommender(rating_matrix_real, method = "UBCF")
  
  # Note: In practice, you would need to handle cold start properly
  cat("  Note: Cold start requires special handling in real implementations\n")
  
  # 3. Popularity Bias
  cat("\n3. Popularity Bias:\n")
  
  # Analyze rating distribution
  rating_counts <- table(ratings_df$item_id)
  most_popular <- names(which.max(rating_counts))
  least_popular <- names(which.min(rating_counts))
  
  cat("  Most popular item:", most_popular, "(", max(rating_counts), "ratings)\n")
  cat("  Least popular item:", least_popular, "(", min(rating_counts), "ratings)\n")
  cat("  Popularity ratio:", round(max(rating_counts) / min(rating_counts), 1), ":1\n")
  
  # Visualize popularity distribution
  popularity_df <- data.frame(
    item_id = names(rating_counts),
    count = as.numeric(rating_counts)
  )
  
  p1 <- ggplot(popularity_df, aes(x = count)) +
    geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) +
    labs(title = "Item Popularity Distribution",
         x = "Number of Ratings", y = "Number of Items") +
    theme_minimal()
  
  # Rating distribution by popularity
  popular_items <- names(sort(rating_counts, decreasing = TRUE)[1:10])
  unpopular_items <- names(sort(rating_counts)[1:10])
  
  popular_ratings <- ratings_df[ratings_df$item_id %in% popular_items, "rating"]
  unpopular_ratings <- ratings_df[ratings_df$item_id %in% unpopular_items, "rating"]
  
  rating_comparison_df <- data.frame(
    rating = c(popular_ratings, unpopular_ratings),
    type = c(rep("Popular Items", length(popular_ratings)),
             rep("Unpopular Items", length(unpopular_ratings)))
  )
  
  p2 <- ggplot(rating_comparison_df, aes(x = rating, fill = type)) +
    geom_histogram(position = "dodge", bins = 10, alpha = 0.7) +
    labs(title = "Rating Distribution by Popularity",
         x = "Rating", y = "Frequency", fill = "Item Type") +
    theme_minimal()
  
  # Sparsity vs performance
  sparsity_levels <- c(0.95, 0.98, 0.99, 0.995)
  rmse_values <- c()
  
  for (sparsity in sparsity_levels) {
    n_users <- 50
    n_items <- 25
    n_ratings <- round((1 - sparsity) * n_users * n_items)
    
    if (n_ratings > 0) {
      ratings_df <- generate_synthetic_data(n_users, n_items, n_ratings)
      rating_matrix <- create_rating_matrix(ratings_df)
      rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
      
      # Simple evaluation (in practice, use proper cross-validation)
      tryCatch({
        model <- Recommender(rating_matrix_real, method = "UBCF")
        predictions <- predict(model, rating_matrix_real, type = "ratings")
        evaluation <- calcPredictionAccuracy(predictions, rating_matrix_real)
        rmse_values <- c(rmse_values, evaluation["RMSE"])
      }, error = function(e) {
        rmse_values <<- c(rmse_values, NA)
      })
    } else {
      rmse_values <- c(rmse_values, NA)
    }
  }
  
  performance_df <- data.frame(
    sparsity = sparsity_levels,
    rmse = rmse_values
  )
  
  p3 <- ggplot(performance_df, aes(x = sparsity, y = rmse)) +
    geom_line() +
    geom_point() +
    labs(title = "Performance vs Sparsity",
         x = "Sparsity", y = "RMSE") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, ncol = 3)
}

# Main demonstration function
main_r <- function() {
  cat("Recommender System Introduction: Implementation and Analysis\n")
  cat("=" * 70, "\n")
  
  # 1. Basic recommender system demonstration
  cat("\n1. Basic Recommender System Demonstration:\n")
  basic_results <- demonstrate_basic_recommender_system()
  
  # 2. Visualization
  cat("\n2. Recommender System Visualizations:\n")
  viz_results <- visualize_recommender_system(basic_results$ratings_df, basic_results$rating_matrix)
  
  # 3. Collaborative filtering demonstration
  cat("\n3. Collaborative Filtering Demonstration:\n")
  cf_results <- demonstrate_collaborative_filtering()
  
  # 4. Latent factor models demonstration
  cat("\n4. Latent Factor Models Demonstration:\n")
  lf_results <- demonstrate_latent_factor_models()
  
  # 5. Content-based filtering demonstration
  cat("\n5. Content-Based Filtering Demonstration:\n")
  cb_results <- demonstrate_content_based_filtering()
  
  # 6. Evaluation metrics demonstration
  cat("\n6. Evaluation Metrics Demonstration:\n")
  eval_results <- demonstrate_evaluation_metrics()
  
  # 7. Challenges demonstration
  cat("\n7. Recommender System Challenges Demonstration:\n")
  demonstrate_challenges()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Collaborative filtering leverages user-item interaction patterns\n")
  cat("2. Latent factor models discover hidden patterns in the data\n")
  cat("3. Content-based filtering uses item attributes and user preferences\n")
  cat("4. Sparsity is a major challenge in real-world recommender systems\n")
  cat("5. Cold start problem affects new users and items\n")
  cat("6. Popularity bias can lead to filter bubbles\n")
  cat("7. Multiple evaluation metrics are needed for comprehensive assessment\n")
  cat("8. Different methods have different strengths and limitations\n")
  
  return(list(
    basic_results = basic_results,
    viz_results = viz_results,
    cf_results = cf_results,
    lf_results = lf_results,
    cb_results = cb_results,
    eval_results = eval_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
