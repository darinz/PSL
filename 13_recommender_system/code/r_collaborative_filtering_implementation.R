# Collaborative Filtering in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(proxy)
library(cluster)
library(factoextra)

# Generate synthetic data
generate_synthetic_ratings_data <- function(n_users = 100, n_items = 50, seed = 42) {
  set.seed(seed)
  
  # Create synthetic ratings with structure
  ratings_data <- list()
  for (user_id in 1:n_users) {
    n_user_ratings <- sample(5:20, 1)
    rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
    
    for (item_id in rated_items) {
      # Simulate user preferences
      if (user_id <= 30) {
        base_rating <- ifelse(item_id <= 15, 4, 2)
      } else if (user_id <= 60) {
        base_rating <- ifelse(item_id > 15 && item_id <= 30, 4, 2)
      } else {
        base_rating <- ifelse(item_id > 30, 4, 2)
      }
      
      # Add noise
      rating <- max(1, min(5, base_rating + rnorm(1, 0, 0.5)))
      
      ratings_data[[length(ratings_data) + 1]] <- list(
        user_id = user_id,
        item_id = item_id,
        rating = rating
      )
    }
  }
  
  ratings_df <- do.call(rbind, lapply(ratings_data, as.data.frame))
  return(ratings_df)
}

# Collaborative Filtering class
CollaborativeFiltering <- function(method = "user", similarity_metric = "cosine", k_neighbors = 10) {
  list(
    method = method,
    similarity_metric = similarity_metric,
    k_neighbors = k_neighbors,
    rating_matrix = NULL,
    user_similarity = NULL,
    item_similarity = NULL,
    user_means = NULL,
    item_means = NULL
  )
}

# Fit the collaborative filtering model
fit_cf <- function(cf_model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create rating matrix
  cf_model$rating_matrix <- ratings_df %>%
    spread(!!sym(item_col), !!sym(rating_col), fill = NA) %>%
    select(-!!sym(user_col)) %>%
    as.matrix()
  
  # Compute means
  cf_model$user_means <- rowMeans(cf_model$rating_matrix, na.rm = TRUE)
  cf_model$item_means <- colMeans(cf_model$rating_matrix, na.rm = TRUE)
  
  # Compute similarities
  if (cf_model$method == "user") {
    cf_model$user_similarity <- compute_user_similarity(cf_model)
  } else {
    cf_model$item_similarity <- compute_item_similarity(cf_model)
  }
  
  return(cf_model)
}

# Compute user similarity matrix
compute_user_similarity <- function(cf_model) {
  if (cf_model$similarity_metric == "cosine") {
    # Fill NaN with 0 for cosine similarity
    matrix_filled <- cf_model$rating_matrix
    matrix_filled[is.na(matrix_filled)] <- 0
    return(as.matrix(cosine(t(matrix_filled))))
  } else if (cf_model$similarity_metric == "pearson") {
    # Compute Pearson correlation
    return(cor(t(cf_model$rating_matrix), use = "pairwise.complete.obs"))
  } else if (cf_model$similarity_metric == "jaccard") {
    # Convert to binary (rated/not rated)
    binary_matrix <- !is.na(cf_model$rating_matrix)
    return(as.matrix(cosine(t(binary_matrix))))
  }
}

# Compute item similarity matrix
compute_item_similarity <- function(cf_model) {
  if (cf_model$similarity_metric == "cosine") {
    # Fill NaN with 0 for cosine similarity
    matrix_filled <- cf_model$rating_matrix
    matrix_filled[is.na(matrix_filled)] <- 0
    return(as.matrix(cosine(matrix_filled)))
  } else if (cf_model$similarity_metric == "adjusted_cosine") {
    # Center by user means
    centered_matrix <- cf_model$rating_matrix - cf_model$user_means
    centered_matrix[is.na(centered_matrix)] <- 0
    return(as.matrix(cosine(centered_matrix)))
  } else if (cf_model$similarity_metric == "pearson") {
    # Compute Pearson correlation
    return(cor(cf_model$rating_matrix, use = "pairwise.complete.obs"))
  }
}

# Predict rating for user-item pair
predict_cf <- function(cf_model, user_id, item_id) {
  if (cf_model$method == "user") {
    return(predict_user_based(cf_model, user_id, item_id))
  } else {
    return(predict_item_based(cf_model, user_id, item_id))
  }
}

# User-based prediction
predict_user_based <- function(cf_model, user_id, item_id) {
  # Find user and item indices
  user_idx <- which(rownames(cf_model$rating_matrix) == user_id)
  item_idx <- which(colnames(cf_model$rating_matrix) == item_id)
  
  if (length(user_idx) == 0 || length(item_idx) == 0) {
    return(mean(cf_model$user_means, na.rm = TRUE))
  }
  
  # Get user similarities
  user_similarities <- cf_model$user_similarity[user_idx, ]
  
  # Find users who rated this item
  item_ratings <- cf_model$rating_matrix[, item_idx]
  rated_users <- !is.na(item_ratings)
  
  if (!any(rated_users)) {
    return(mean(cf_model$user_means, na.rm = TRUE))
  }
  
  # Get similarities and ratings for users who rated this item
  similarities <- user_similarities[rated_users]
  ratings <- item_ratings[rated_users]
  
  # Sort by similarity and take top-k
  sorted_indices <- order(similarities, decreasing = TRUE)[1:cf_model$k_neighbors]
  
  if (length(sorted_indices) == 0) {
    return(mean(cf_model$user_means, na.rm = TRUE))
  }
  
  top_similarities <- similarities[sorted_indices]
  top_ratings <- ratings[sorted_indices]
  
  # Weighted average
  weighted_sum <- sum(top_similarities * top_ratings)
  total_similarity <- sum(abs(top_similarities))
  
  if (total_similarity == 0) {
    return(mean(top_ratings, na.rm = TRUE))
  }
  
  return(weighted_sum / total_similarity)
}

# Item-based prediction
predict_item_based <- function(cf_model, user_id, item_id) {
  # Find user and item indices
  user_idx <- which(rownames(cf_model$rating_matrix) == user_id)
  item_idx <- which(colnames(cf_model$rating_matrix) == item_id)
  
  if (length(user_idx) == 0 || length(item_idx) == 0) {
    return(mean(cf_model$item_means, na.rm = TRUE))
  }
  
  # Get item similarities
  item_similarities <- cf_model$item_similarity[item_idx, ]
  
  # Find items rated by this user
  user_ratings <- cf_model$rating_matrix[user_idx, ]
  rated_items <- !is.na(user_ratings)
  
  if (!any(rated_items)) {
    return(mean(cf_model$item_means, na.rm = TRUE))
  }
  
  # Get similarities and ratings for items rated by this user
  similarities <- item_similarities[rated_items]
  ratings <- user_ratings[rated_items]
  
  # Sort by similarity and take top-k
  sorted_indices <- order(similarities, decreasing = TRUE)[1:cf_model$k_neighbors]
  
  if (length(sorted_indices) == 0) {
    return(mean(cf_model$item_means, na.rm = TRUE))
  }
  
  top_similarities <- similarities[sorted_indices]
  top_ratings <- ratings[sorted_indices]
  
  # Weighted average
  weighted_sum <- sum(top_similarities * top_ratings)
  total_similarity <- sum(abs(top_similarities))
  
  if (total_similarity == 0) {
    return(mean(top_ratings, na.rm = TRUE))
  }
  
  return(weighted_sum / total_similarity)
}

# Generate recommendations for a user
recommend_cf <- function(cf_model, user_id, n_recommendations = 5) {
  user_idx <- which(rownames(cf_model$rating_matrix) == user_id)
  
  if (length(user_idx) == 0) {
    return(list())
  }
  
  user_ratings <- cf_model$rating_matrix[user_idx, ]
  unrated_items <- is.na(user_ratings)
  
  if (!any(unrated_items)) {
    return(list())
  }
  
  # Predict ratings for unrated items
  predictions <- list()
  for (item_id in names(user_ratings[unrated_items])) {
    pred_rating <- predict_cf(cf_model, user_id, as.numeric(item_id))
    predictions[[length(predictions) + 1]] <- list(
      item_id = as.numeric(item_id),
      rating = pred_rating
    )
  }
  
  # Sort by predicted rating
  predictions <- predictions[order(sapply(predictions, function(x) x$rating), decreasing = TRUE)]
  return(predictions[1:n_recommendations])
}

# Get similar users
get_similar_users <- function(cf_model, user_id, n_similar = 5) {
  user_idx <- which(rownames(cf_model$rating_matrix) == user_id)
  
  if (length(user_idx) == 0) {
    return(list())
  }
  
  similarities <- cf_model$user_similarity[user_idx, ]
  
  # Sort by similarity (exclude self)
  sorted_indices <- order(similarities, decreasing = TRUE)[2:(n_similar + 1)]
  similar_users <- list()
  
  for (idx in sorted_indices) {
    user_id_similar <- as.numeric(rownames(cf_model$rating_matrix)[idx])
    similarity <- similarities[idx]
    similar_users[[length(similar_users) + 1]] <- list(
      user_id = user_id_similar,
      similarity = similarity
    )
  }
  
  return(similar_users)
}

# Get similar items
get_similar_items <- function(cf_model, item_id, n_similar = 5) {
  item_idx <- which(colnames(cf_model$rating_matrix) == item_id)
  
  if (length(item_idx) == 0) {
    return(list())
  }
  
  similarities <- cf_model$item_similarity[item_idx, ]
  
  # Sort by similarity (exclude self)
  sorted_indices <- order(similarities, decreasing = TRUE)[2:(n_similar + 1)]
  similar_items <- list()
  
  for (idx in sorted_indices) {
    item_id_similar <- as.numeric(colnames(cf_model$rating_matrix)[idx])
    similarity <- similarities[idx]
    similar_items[[length(similar_items) + 1]] <- list(
      item_id = item_id_similar,
      similarity = similarity
    )
  }
  
  return(similar_items)
}

# Demonstrate basic collaborative filtering
demonstrate_basic_collaborative_filtering <- function() {
  cat("=== Basic Collaborative Filtering ===\n\n")
  
  # Generate synthetic data
  ratings_df <- generate_synthetic_ratings_data()
  
  cat("Synthetic Ratings Dataset:\n")
  cat("Number of users:", length(unique(ratings_df$user_id)), "\n")
  cat("Number of items:", length(unique(ratings_df$item_id)), "\n")
  cat("Number of ratings:", nrow(ratings_df), "\n")
  cat("Sparsity:", 1 - nrow(ratings_df) / (length(unique(ratings_df$user_id)) * length(unique(ratings_df$item_id))), "\n")
  
  # Test different collaborative filtering approaches
  methods <- c("user", "item")
  similarity_metrics <- c("cosine", "pearson")
  results <- list()
  
  for (method in methods) {
    for (metric in similarity_metrics) {
      cat("\n=== Testing", toupper(method), "-based CF with", toupper(metric), "similarity ===\n")
      
      # Initialize and fit model
      cf_model <- CollaborativeFiltering(method = method, similarity_metric = metric, k_neighbors = 10)
      cf_model <- fit_cf(cf_model, ratings_df)
      
      # Test predictions for a sample user
      test_user <- 1
      recommendations <- recommend_cf(cf_model, test_user, n_recommendations = 5)
      
      cat("Top 5 recommendations for User", test_user, ":\n")
      for (i in seq_along(recommendations)) {
        rec <- recommendations[[i]]
        cat("  ", i, ". Item", rec$item_id, ": Predicted rating =", round(rec$rating, 3), "\n")
      }
      
      # Get similar users/items
      if (method == "user") {
        similar_entities <- get_similar_users(cf_model, test_user, n_similar = 3)
        cat("Most similar users to User", test_user, ":\n")
      } else {
        test_item <- 1
        similar_entities <- get_similar_items(cf_model, test_item, n_similar = 3)
        cat("Most similar items to Item", test_item, ":\n")
      }
      
      for (entity in similar_entities) {
        cat("  ", entity$user_id, ": Similarity =", round(entity$similarity, 3), "\n")
      }
      
      # Store results for comparison
      results[[paste0(method, "_", metric)]] <- list(
        recommendations = recommendations,
        similar_entities = similar_entities
      )
    }
  }
  
  return(list(ratings_df = ratings_df, results = results))
}

# Demonstrate similarity metrics
demonstrate_similarity_metrics <- function() {
  cat("=== Similarity Metrics Comparison ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_ratings_data(n_users = 50, n_items = 30)
  
  # Test different similarity metrics
  metrics <- c("cosine", "pearson", "jaccard")
  results <- list()
  
  for (metric in metrics) {
    cat("Testing", toupper(metric), "similarity...\n")
    
    # User-based CF
    cf_user <- CollaborativeFiltering(method = "user", similarity_metric = metric, k_neighbors = 5)
    cf_user <- fit_cf(cf_user, ratings_df)
    
    # Item-based CF
    cf_item <- CollaborativeFiltering(method = "item", similarity_metric = metric, k_neighbors = 5)
    cf_item <- fit_cf(cf_item, ratings_df)
    
    # Test predictions
    test_user <- 1
    test_item <- 1
    
    user_recommendations <- recommend_cf(cf_user, test_user, n_recommendations = 3)
    item_recommendations <- recommend_cf(cf_item, test_user, n_recommendations = 3)
    
    similar_users <- get_similar_users(cf_user, test_user, n_similar = 3)
    similar_items <- get_similar_items(cf_item, test_item, n_similar = 3)
    
    results[[metric]] <- list(
      user_recommendations = user_recommendations,
      item_recommendations = item_recommendations,
      similar_users = similar_users,
      similar_items = similar_items
    )
    
    cat("  User-based recommendations:", sapply(user_recommendations, function(x) x$item_id), "\n")
    cat("  Item-based recommendations:", sapply(item_recommendations, function(x) x$item_id), "\n")
    cat("  Similar users:", sapply(similar_users, function(x) x$user_id), "\n")
    cat("  Similar items:", sapply(similar_items, function(x) x$item_id), "\n")
  }
  
  return(results)
}

# Demonstrate evaluation metrics
demonstrate_evaluation_metrics <- function() {
  cat("=== Evaluation Metrics ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_ratings_data()
  
  # Split data for evaluation (simple random split)
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), size = 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Test different methods
  methods <- c("user", "item")
  metrics <- c("cosine", "pearson")
  evaluation_results <- list()
  
  for (method in methods) {
    for (metric in metrics) {
      cat("Evaluating", toupper(method), "-", toupper(metric), "...\n")
      
      # Train model
      cf_model <- CollaborativeFiltering(method = method, similarity_metric = metric, k_neighbors = 10)
      cf_model <- fit_cf(cf_model, train_df)
      
      # Make predictions on test set
      predictions <- numeric()
      actuals <- numeric()
      
      for (i in 1:nrow(test_df)) {
        user_id <- test_df$user_id[i]
        item_id <- test_df$item_id[i]
        actual_rating <- test_df$rating[i]
        
        # Only predict if user and item exist in training data
        if (user_id %in% rownames(cf_model$rating_matrix) && 
            item_id %in% colnames(cf_model$rating_matrix)) {
          pred_rating <- predict_cf(cf_model, user_id, item_id)
          predictions <- c(predictions, pred_rating)
          actuals <- c(actuals, actual_rating)
        }
      }
      
      if (length(predictions) > 0) {
        # Calculate metrics
        mae <- mean(abs(actuals - predictions))
        rmse <- sqrt(mean((actuals - predictions)^2))
        
        evaluation_results[[paste0(method, "_", metric)]] <- list(
          mae = mae,
          rmse = rmse,
          n_predictions = length(predictions)
        )
        
        cat("  MAE:", round(mae, 3), "\n")
        cat("  RMSE:", round(rmse, 3), "\n")
        cat("  Number of predictions:", length(predictions), "\n")
      }
    }
  }
  
  return(evaluation_results)
}

# Demonstrate visualizations
demonstrate_visualization <- function() {
  cat("=== Collaborative Filtering Visualizations ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_ratings_data()
  
  # Create rating matrix
  rating_matrix <- ratings_df %>%
    spread(item_id, rating, fill = NA) %>%
    select(-user_id) %>%
    as.matrix()
  
  # Create visualizations
  plots <- list()
  
  # Plot 1: Rating distribution
  p1 <- ggplot(ratings_df, aes(x = factor(rating))) +
    geom_bar(fill = "steelblue") +
    labs(title = "Rating Distribution",
         x = "Rating", y = "Count") +
    theme_minimal()
  
  # Plot 2: User-item matrix heatmap (sample)
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
  
  # Plot 3: User similarity matrix (sample)
  cf_user <- CollaborativeFiltering(method = "user", similarity_metric = "cosine")
  cf_user <- fit_cf(cf_user, ratings_df)
  sample_user_sim <- cf_user$user_similarity[1:20, 1:20]
  
  user_sim_df <- expand.grid(
    user1 = 1:20,
    user2 = 1:20
  )
  user_sim_df$similarity <- as.vector(sample_user_sim)
  
  p3 <- ggplot(user_sim_df, aes(x = user2, y = user1, fill = similarity)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = "User Similarity Matrix (Sample)",
         x = "User ID", y = "User ID") +
    theme_minimal()
  
  # Plot 4: Item similarity matrix (sample)
  cf_item <- CollaborativeFiltering(method = "item", similarity_metric = "cosine")
  cf_item <- fit_cf(cf_item, ratings_df)
  sample_item_sim <- cf_item$item_similarity[1:20, 1:20]
  
  item_sim_df <- expand.grid(
    item1 = 1:20,
    item2 = 1:20
  )
  item_sim_df$similarity <- as.vector(sample_item_sim)
  
  p4 <- ggplot(item_sim_df, aes(x = item2, y = item1, fill = similarity)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = "Item Similarity Matrix (Sample)",
         x = "Item ID", y = "Item ID") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  return(list(
    rating_distribution = p1,
    rating_matrix = p2,
    user_similarity = p3,
    item_similarity = p4
  ))
}

# Demonstrate cold start handling
demonstrate_cold_start <- function() {
  cat("=== Cold Start Handling ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_ratings_data()
  
  cat("1. New User Problem:\n")
  cat("   - User with no ratings\n")
  cat("   - Using popularity-based fallback\n")
  
  # Calculate item popularity
  item_popularity <- ratings_df %>%
    group_by(item_id) %>%
    summarise(
      count = n(),
      mean_rating = mean(rating, na.rm = TRUE)
    ) %>%
    arrange(desc(count))
  
  cat("   Most popular items:\n")
  for (i in 1:5) {
    item <- item_popularity[i, ]
    cat("     Item", item$item_id, ":", item$count, "ratings, avg rating", round(item$mean_rating, 2), "\n")
  }
  
  cat("\n2. New Item Problem:\n")
  cat("   - Item with no ratings\n")
  cat("   - Using user average ratings\n")
  
  # Calculate user average ratings
  user_avg_ratings <- ratings_df %>%
    group_by(user_id) %>%
    summarise(avg_rating = mean(rating, na.rm = TRUE)) %>%
    arrange(desc(avg_rating))
  
  cat("   Users with highest average ratings:\n")
  for (i in 1:5) {
    user <- user_avg_ratings[i, ]
    cat("     User", user$user_id, ": avg rating", round(user$avg_rating, 2), "\n")
  }
  
  cat("\n3. Hybrid Approach:\n")
  cat("   - Combining collaborative filtering with content-based methods\n")
  cat("   - Using weighted combination of predictions\n")
  
  # Simulate hybrid prediction
  cf_prediction <- 3.5  # Collaborative filtering prediction
  cb_prediction <- 4.2  # Content-based prediction
  alpha <- 0.7  # Weight for CF
  
  hybrid_prediction <- alpha * cf_prediction + (1 - alpha) * cb_prediction
  cat("   Hybrid prediction:", alpha, "*", cf_prediction, "+", 1-alpha, "*", cb_prediction, "=", round(hybrid_prediction, 2), "\n")
  
  return(list(
    item_popularity = item_popularity,
    user_avg_ratings = user_avg_ratings,
    hybrid_example = hybrid_prediction
  ))
}

# Demonstrate scalability
demonstrate_scalability <- function() {
  cat("=== Scalability Analysis ===\n\n")
  
  # Test with different dataset sizes
  dataset_sizes <- c(50, 100, 200, 500)
  training_times <- numeric()
  prediction_times <- numeric()
  
  for (size in dataset_sizes) {
    cat("Testing with", size, "users...\n")
    
    # Generate data
    ratings_df <- generate_synthetic_ratings_data(n_users = size, n_items = size %/% 2)
    
    # Time training
    start_time <- Sys.time()
    
    cf_model <- CollaborativeFiltering(method = "user", similarity_metric = "cosine", k_neighbors = 10)
    cf_model <- fit_cf(cf_model, ratings_df)
    
    training_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    training_times <- c(training_times, training_time)
    
    # Time predictions
    start_time <- Sys.time()
    for (user_id in 1:min(5, size)) {
      recommend_cf(cf_model, user_id, n_recommendations = 5)
    }
    
    prediction_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    prediction_times <- c(prediction_times, prediction_time)
    
    cat("  Training time:", round(training_time, 3), "s\n")
    cat("  Prediction time (5 users):", round(prediction_time, 3), "s\n")
  }
  
  # Create scalability plot
  scalability_df <- data.frame(
    dataset_size = dataset_sizes,
    training_time = training_times,
    prediction_time = prediction_times
  )
  
  p1 <- ggplot(scalability_df, aes(x = dataset_size, y = training_time)) +
    geom_line() + geom_point() +
    labs(title = "Training Time vs Dataset Size",
         x = "Dataset Size (users)", y = "Time (seconds)") +
    theme_minimal()
  
  p2 <- ggplot(scalability_df, aes(x = dataset_size, y = prediction_time)) +
    geom_line() + geom_point() +
    labs(title = "Prediction Time vs Dataset Size",
         x = "Dataset Size (users)", y = "Time (seconds)") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(
    dataset_sizes = dataset_sizes,
    training_times = training_times,
    prediction_times = prediction_times,
    plots = list(training = p1, prediction = p2)
  ))
}

# Demonstrate advanced techniques
demonstrate_advanced_techniques <- function() {
  cat("=== Advanced Techniques ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_ratings_data()
  
  cat("1. Constrained Similarity:\n")
  cat("   - Adding minimum overlap threshold\n")
  cat("   - Filtering out users/items with insufficient common ratings\n")
  
  # Implement constrained similarity
  constrained_similarity <- function(similarity_matrix, rating_matrix, min_overlap = 3) {
    constrained_sim <- similarity_matrix
    
    if (nrow(similarity_matrix) == nrow(rating_matrix)) {  # User similarity
      for (i in 1:nrow(rating_matrix)) {
        for (j in (i+1):nrow(rating_matrix)) {
          # Count common rated items
          user_i_ratings <- rating_matrix[i, ]
          user_j_ratings <- rating_matrix[j, ]
          common_items <- !is.na(user_i_ratings) & !is.na(user_j_ratings)
          
          if (sum(common_items) < min_overlap) {
            constrained_sim[i, j] <- 0
            constrained_sim[j, i] <- 0
          }
        }
      }
    }
    
    return(constrained_sim)
  }
  
  # Test constrained similarity
  cf_model <- CollaborativeFiltering(method = "user", similarity_metric = "cosine")
  cf_model <- fit_cf(cf_model, ratings_df)
  
  original_similarities <- cf_model$user_similarity[1, 2:6]  # First user with next 5 users
  constrained_sim <- constrained_similarity(cf_model$user_similarity, cf_model$rating_matrix, min_overlap = 5)
  constrained_similarities <- constrained_sim[1, 2:6]
  
  cat("   Original similarities:", round(original_similarities, 3), "\n")
  cat("   Constrained similarities:", round(constrained_similarities, 3), "\n")
  
  cat("\n2. Time-Aware Similarity:\n")
  cat("   - Incorporating temporal information\n")
  cat("   - Decaying similarity based on time difference\n")
  
  # Simulate time-aware similarity
  time_aware_similarity <- function(base_similarity, time_diff, decay_rate = 0.1) {
    return(base_similarity * exp(-decay_rate * time_diff))
  }
  
  base_similarities <- c(0.8, 0.6, 0.9, 0.4, 0.7)
  time_diffs <- c(1, 3, 0, 5, 2)  # Time differences in months
  
  time_aware_similarities <- time_aware_similarity(base_similarities, time_diffs)
  
  cat("   Base similarities:", base_similarities, "\n")
  cat("   Time differences:", time_diffs, "\n")
  cat("   Time-aware similarities:", round(time_aware_similarities, 3), "\n")
  
  cat("\n3. Category-Aware Similarity:\n")
  cat("   - Weighting similarities by item categories\n")
  cat("   - Different weights for different categories\n")
  
  # Simulate category-aware similarity
  category_aware_similarity <- function(similarities, categories, weights) {
    weighted_similarities <- numeric(length(similarities))
    
    for (i in seq_along(similarities)) {
      weight <- weights[[categories[i]]]
      if (is.null(weight)) weight <- 1.0  # Default weight
      weighted_similarities[i] <- similarities[i] * weight
    }
    
    return(weighted_similarities)
  }
  
  similarities <- c(0.8, 0.6, 0.9, 0.4, 0.7)
  categories <- c("action", "drama", "action", "comedy", "drama")
  category_weights <- list(action = 1.2, drama = 1.0, comedy = 0.8)
  
  category_aware_similarities <- category_aware_similarity(similarities, categories, category_weights)
  
  cat("   Base similarities:", similarities, "\n")
  cat("   Categories:", categories, "\n")
  cat("   Category weights:", category_weights, "\n")
  cat("   Category-aware similarities:", round(category_aware_similarities, 3), "\n")
  
  return(list(
    constrained_example = list(original = original_similarities, constrained = constrained_similarities),
    time_aware_example = list(base = base_similarities, time_aware = time_aware_similarities),
    category_aware_example = list(base = similarities, category_aware = category_aware_similarities)
  ))
}

# Main demonstration function
main_r <- function() {
  cat("Collaborative Filtering: Implementation and Analysis\n")
  cat("=" %R% 60, "\n")
  
  # 1. Basic collaborative filtering demonstration
  cat("\n1. Basic Collaborative Filtering:\n")
  basic_results <- demonstrate_basic_collaborative_filtering()
  
  # 2. Similarity metrics comparison
  cat("\n2. Similarity Metrics Comparison:\n")
  similarity_results <- demonstrate_similarity_metrics()
  
  # 3. Evaluation metrics
  cat("\n3. Evaluation Metrics:\n")
  evaluation_results <- demonstrate_evaluation_metrics()
  
  # 4. Visualizations
  cat("\n4. Visualizations:\n")
  visualization_results <- demonstrate_visualization()
  
  # 5. Cold start handling
  cat("\n5. Cold Start Handling:\n")
  cold_start_results <- demonstrate_cold_start()
  
  # 6. Scalability analysis
  cat("\n6. Scalability Analysis:\n")
  scalability_results <- demonstrate_scalability()
  
  # 7. Advanced techniques
  cat("\n7. Advanced Techniques:\n")
  advanced_results <- demonstrate_advanced_techniques()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Collaborative filtering leverages user-item interaction patterns\n")
  cat("2. Different similarity metrics produce different results\n")
  cat("3. User-based and item-based approaches have different strengths\n")
  cat("4. Evaluation requires multiple metrics for comprehensive assessment\n")
  cat("5. Cold start can be handled with various strategies\n")
  cat("6. Scalability becomes important with large datasets\n")
  cat("7. Advanced techniques can improve performance and interpretability\n")
  
  return(list(
    basic_results = basic_results,
    similarity_results = similarity_results,
    evaluation_results = evaluation_results,
    visualization_results = visualization_results,
    cold_start_results = cold_start_results,
    scalability_results = scalability_results,
    advanced_results = advanced_results
  ))
}

# Run main function if not interactive
if (!interactive()) {
  main_r()
}
