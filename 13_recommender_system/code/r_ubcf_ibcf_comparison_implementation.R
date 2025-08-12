# UBCF vs IBCF Comparison in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(proxy)
library(cluster)
library(factoextra)

# Generate synthetic data with clusters
generate_synthetic_data_with_clusters <- function(n_users = 200, n_items = 100, seed = 42) {
  set.seed(seed)
  
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
  return(ratings_df)
}

# UBCF Implementation
UBCF <- function(similarity_metric = "pearson", k_neighbors = 10) {
  list(
    similarity_metric = similarity_metric,
    k_neighbors = k_neighbors,
    rating_matrix = NULL,
    user_similarity = NULL,
    user_means = NULL
  )
}

fit_ubcf <- function(ubcf_model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create rating matrix
  rating_matrix <- ratings_df %>%
    spread(!!sym(item_col), !!sym(rating_col), fill = NA) %>%
    select(-!!sym(user_col)) %>%
    as.matrix()
  
  # Compute user means
  user_means <- rowMeans(rating_matrix, na.rm = TRUE)
  
  # Compute user similarities
  user_similarity <- compute_user_similarity(rating_matrix, user_means, ubcf_model$similarity_metric)
  
  # Update model
  ubcf_model$rating_matrix <- rating_matrix
  ubcf_model$user_similarity <- user_similarity
  ubcf_model$user_means <- user_means
  
  return(ubcf_model)
}

compute_user_similarity <- function(rating_matrix, user_means, similarity_metric) {
  n_users <- nrow(rating_matrix)
  similarity_matrix <- matrix(0, n_users, n_users)
  
  for (i in 1:n_users) {
    for (j in (i+1):n_users) {
      if (j <= n_users) {
        # Get common rated items
        user_i_ratings <- rating_matrix[i, ]
        user_j_ratings <- rating_matrix[j, ]
        
        common_items <- !is.na(user_i_ratings) & !is.na(user_j_ratings)
        
        if (sum(common_items) > 1) {
          if (similarity_metric == "pearson") {
            corr <- cor(user_i_ratings[common_items], user_j_ratings[common_items], 
                       method = "pearson", use = "complete.obs")
            similarity_matrix[i, j] <- corr
            similarity_matrix[j, i] <- corr
          } else if (similarity_metric == "cosine") {
            # Center ratings
            user_i_centered <- user_i_ratings[common_items] - user_means[i]
            user_j_centered <- user_j_ratings[common_items] - user_means[j]
            
            cosine_sim <- sum(user_i_centered * user_j_centered) / 
                         (sqrt(sum(user_i_centered^2)) * sqrt(sum(user_j_centered^2)))
            similarity_matrix[i, j] <- cosine_sim
            similarity_matrix[j, i] <- cosine_sim
          }
        }
      }
    }
  }
  
  return(similarity_matrix)
}

predict_ubcf <- function(ubcf_model, user_id, item_id) {
  if (user_id > nrow(ubcf_model$rating_matrix) || item_id > ncol(ubcf_model$rating_matrix)) {
    return(mean(ubcf_model$user_means, na.rm = TRUE))
  }
  
  # Get user similarities
  user_similarities <- ubcf_model$user_similarity[user_id, ]
  
  # Find users who rated this item
  item_ratings <- ubcf_model$rating_matrix[, item_id]
  rated_users <- !is.na(item_ratings)
  
  if (!any(rated_users)) {
    return(mean(ubcf_model$user_means, na.rm = TRUE))
  }
  
  # Get similarities and ratings for users who rated this item
  similarities <- user_similarities[rated_users]
  ratings <- item_ratings[rated_users]
  
  # Sort by similarity and take top-k
  sorted_indices <- order(similarities, decreasing = TRUE)[1:ubcf_model$k_neighbors]
  sorted_indices <- sorted_indices[!is.na(sorted_indices)]
  
  if (length(sorted_indices) == 0) {
    return(mean(ubcf_model$user_means, na.rm = TRUE))
  }
  
  top_similarities <- similarities[sorted_indices]
  top_ratings <- ratings[sorted_indices]
  
  # Weighted average
  weighted_sum <- sum(top_similarities * top_ratings)
  total_similarity <- sum(abs(top_similarities))
  
  if (total_similarity == 0) {
    return(mean(top_ratings))
  }
  
  return(weighted_sum / total_similarity)
}

# IBCF Implementation
IBCF <- function(similarity_metric = "adjusted_cosine", k_neighbors = 10) {
  list(
    similarity_metric = similarity_metric,
    k_neighbors = k_neighbors,
    rating_matrix = NULL,
    item_similarity = NULL,
    user_means = NULL
  )
}

fit_ibcf <- function(ibcf_model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create rating matrix
  rating_matrix <- ratings_df %>%
    spread(!!sym(item_col), !!sym(rating_col), fill = NA) %>%
    select(-!!sym(user_col)) %>%
    as.matrix()
  
  # Compute user means
  user_means <- rowMeans(rating_matrix, na.rm = TRUE)
  
  # Compute item similarities
  item_similarity <- compute_item_similarity(rating_matrix, user_means, ibcf_model$similarity_metric)
  
  # Update model
  ibcf_model$rating_matrix <- rating_matrix
  ibcf_model$item_similarity <- item_similarity
  ibcf_model$user_means <- user_means
  
  return(ibcf_model)
}

compute_item_similarity <- function(rating_matrix, user_means, similarity_metric) {
  n_items <- ncol(rating_matrix)
  similarity_matrix <- matrix(0, n_items, n_items)
  
  for (i in 1:n_items) {
    for (j in (i+1):n_items) {
      if (j <= n_items) {
        # Get common users
        item_i_ratings <- rating_matrix[, i]
        item_j_ratings <- rating_matrix[, j]
        
        common_users <- !is.na(item_i_ratings) & !is.na(item_j_ratings)
        
        if (sum(common_users) > 1) {
          if (similarity_metric == "adjusted_cosine") {
            # Center by user means
            item_i_centered <- item_i_ratings[common_users] - user_means[common_users]
            item_j_centered <- item_j_ratings[common_users] - user_means[common_users]
            
            cosine_sim <- sum(item_i_centered * item_j_centered) / 
                         (sqrt(sum(item_i_centered^2)) * sqrt(sum(item_j_centered^2)))
            similarity_matrix[i, j] <- cosine_sim
            similarity_matrix[j, i] <- cosine_sim
          } else if (similarity_metric == "pearson") {
            corr <- cor(item_i_ratings[common_users], item_j_ratings[common_users], 
                       method = "pearson", use = "complete.obs")
            similarity_matrix[i, j] <- corr
            similarity_matrix[j, i] <- corr
          }
        }
      }
    }
  }
  
  return(similarity_matrix)
}

predict_ibcf <- function(ibcf_model, user_id, item_id) {
  if (user_id > nrow(ibcf_model$rating_matrix) || item_id > ncol(ibcf_model$rating_matrix)) {
    return(mean(ibcf_model$rating_matrix, na.rm = TRUE))
  }
  
  # Get item similarities
  item_similarities <- ibcf_model$item_similarity[item_id, ]
  
  # Find items rated by this user
  user_ratings <- ibcf_model$rating_matrix[user_id, ]
  rated_items <- !is.na(user_ratings)
  
  if (!any(rated_items)) {
    return(mean(ibcf_model$rating_matrix, na.rm = TRUE))
  }
  
  # Get similarities and ratings for items rated by this user
  similarities <- item_similarities[rated_items]
  ratings <- user_ratings[rated_items]
  
  # Sort by similarity and take top-k
  sorted_indices <- order(similarities, decreasing = TRUE)[1:ibcf_model$k_neighbors]
  sorted_indices <- sorted_indices[!is.na(sorted_indices)]
  
  if (length(sorted_indices) == 0) {
    return(mean(ratings))
  }
  
  top_similarities <- similarities[sorted_indices]
  top_ratings <- ratings[sorted_indices]
  
  # Weighted average
  weighted_sum <- sum(top_similarities * top_ratings)
  total_similarity <- sum(abs(top_similarities))
  
  if (total_similarity == 0) {
    return(mean(top_ratings))
  }
  
  return(weighted_sum / total_similarity)
}

# Hybrid Recommender
HybridRecommender <- function(ubcf_model, ibcf_model, alpha = 0.5) {
  list(
    ubcf_model = ubcf_model,
    ibcf_model = ibcf_model,
    alpha = alpha
  )
}

predict_hybrid <- function(hybrid_model, user_id, item_id) {
  ubcf_pred <- predict_ubcf(hybrid_model$ubcf_model, user_id, item_id)
  ibcf_pred <- predict_ibcf(hybrid_model$ibcf_model, user_id, item_id)
  
  # Handle NaN predictions
  if (is.na(ubcf_pred) && is.na(ibcf_pred)) {
    return(NA)
  } else if (is.na(ubcf_pred)) {
    return(ibcf_pred)
  } else if (is.na(ibcf_pred)) {
    return(ubcf_pred)
  }
  
  return(hybrid_model$alpha * ubcf_pred + (1 - hybrid_model$alpha) * ibcf_pred)
}

# Evaluation function
evaluate_model <- function(model, test_df, predict_func) {
  predictions <- numeric(nrow(test_df))
  actuals <- numeric(nrow(test_df))
  valid_predictions <- 0
  
  for (i in 1:nrow(test_df)) {
    user_id <- test_df$user_id[i]
    item_id <- test_df$item_id[i]
    actual_rating <- test_df$rating[i]
    
    pred_rating <- predict_func(model, user_id, item_id)
    
    if (!is.na(pred_rating)) {
      valid_predictions <- valid_predictions + 1
      predictions[valid_predictions] <- pred_rating
      actuals[valid_predictions] <- actual_rating
    }
  }
  
  if (valid_predictions == 0) {
    return(list(mae = Inf, rmse = Inf, coverage = 0))
  }
  
  predictions <- predictions[1:valid_predictions]
  actuals <- actuals[1:valid_predictions]
  
  mae <- mean(abs(predictions - actuals))
  rmse <- sqrt(mean((predictions - actuals)^2))
  coverage <- valid_predictions / nrow(test_df)
  
  return(list(mae = mae, rmse = rmse, coverage = coverage))
}

# Demonstration functions
demonstrate_basic_comparison <- function() {
  cat("=== Basic UBCF vs IBCF Comparison ===\n\n")
  
  # Generate synthetic data
  ratings_df <- generate_synthetic_data_with_clusters()
  
  cat("Synthetic Dataset with User/Item Clusters:\n")
  cat("Number of users:", length(unique(ratings_df$user_id)), "\n")
  cat("Number of items:", length(unique(ratings_df$item_id)), "\n")
  cat("Number of ratings:", nrow(ratings_df), "\n")
  sparsity <- 1 - nrow(ratings_df) / (length(unique(ratings_df$user_id)) * length(unique(ratings_df$item_id)))
  cat("Sparsity:", round(sparsity, 3), "\n")
  
  # Split data for evaluation
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train UBCF and IBCF models
  cat("\n=== Training Models ===\n")
  
  # UBCF with different similarity metrics
  ubcf_pearson <- UBCF(similarity_metric = "pearson", k_neighbors = 15)
  ubcf_pearson <- fit_ubcf(ubcf_pearson, train_df)
  
  ubcf_cosine <- UBCF(similarity_metric = "cosine", k_neighbors = 15)
  ubcf_cosine <- fit_ubcf(ubcf_cosine, train_df)
  
  # IBCF with different similarity metrics
  ibcf_adjusted_cosine <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 15)
  ibcf_adjusted_cosine <- fit_ibcf(ibcf_adjusted_cosine, train_df)
  
  ibcf_pearson <- IBCF(similarity_metric = "pearson", k_neighbors = 15)
  ibcf_pearson <- fit_ibcf(ibcf_pearson, train_df)
  
  # Evaluate models
  models <- list(
    "UBCF-Pearson" = list(model = ubcf_pearson, predict_func = predict_ubcf),
    "UBCF-Cosine" = list(model = ubcf_cosine, predict_func = predict_ubcf),
    "IBCF-AdjustedCosine" = list(model = ibcf_adjusted_cosine, predict_func = predict_ibcf),
    "IBCF-Pearson" = list(model = ibcf_pearson, predict_func = predict_ibcf)
  )
  
  results <- list()
  for (name in names(models)) {
    cat("Evaluating", name, "...\n")
    results[[name]] <- evaluate_model(models[[name]]$model, test_df, models[[name]]$predict_func)
  }
  
  # Display results
  cat("\n=== Evaluation Results ===\n")
  for (name in names(results)) {
    cat(name, ":\n")
    cat("  MAE:", round(results[[name]]$mae, 4), "\n")
    cat("  RMSE:", round(results[[name]]$rmse, 4), "\n")
    cat("  Coverage:", round(results[[name]]$coverage, 4), "\n")
    cat("\n")
  }
  
  return(list(ratings_df = ratings_df, results = results, models = models))
}

demonstrate_similarity_analysis <- function() {
  cat("=== Similarity Analysis ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_data_with_clusters()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  
  # Train models
  ubcf <- UBCF(similarity_metric = "pearson", k_neighbors = 15)
  ubcf <- fit_ubcf(ubcf, train_df)
  
  ibcf <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 15)
  ibcf <- fit_ibcf(ibcf, train_df)
  
  # Analyze similarity distributions
  user_similarities <- ubcf$user_similarity[upper.tri(ubcf$user_similarity)]
  item_similarities <- ibcf$item_similarity[upper.tri(ibcf$item_similarity)]
  
  cat("Similarity Distribution Comparison:\n")
  cat("UBCF User Similarities:\n")
  cat("  Mean:", round(mean(user_similarities, na.rm = TRUE), 3), "\n")
  cat("  Std:", round(sd(user_similarities, na.rm = TRUE), 3), "\n")
  cat("  Range: [", round(min(user_similarities, na.rm = TRUE), 3), ", ", 
      round(max(user_similarities, na.rm = TRUE), 3), "]\n", sep = "")
  
  cat("\nIBCF Item Similarities:\n")
  cat("  Mean:", round(mean(item_similarities, na.rm = TRUE), 3), "\n")
  cat("  Std:", round(sd(item_similarities, na.rm = TRUE), 3), "\n")
  cat("  Range: [", round(min(item_similarities, na.rm = TRUE), 3), ", ", 
      round(max(item_similarities, na.rm = TRUE), 3), "]\n", sep = "")
  
  return(list(ubcf = ubcf, ibcf = ibcf, user_similarities = user_similarities, item_similarities = item_similarities))
}

demonstrate_hybrid_approaches <- function() {
  cat("=== Hybrid Recommendation Approaches ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_data_with_clusters()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train base models
  ubcf <- UBCF(similarity_metric = "pearson", k_neighbors = 15)
  ubcf <- fit_ubcf(ubcf, train_df)
  
  ibcf <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 15)
  ibcf <- fit_ibcf(ibcf, train_df)
  
  # Test different hybrid weights
  weights <- c(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
  hybrid_results <- list()
  
  for (alpha in weights) {
    hybrid <- HybridRecommender(ubcf, ibcf, alpha = alpha)
    results <- evaluate_model(hybrid, test_df, predict_hybrid)
    hybrid_results[[paste0("Hybrid-", alpha)]] <- results
  }
  
  # Display results
  cat("Hybrid Approach Results:\n")
  for (name in names(hybrid_results)) {
    cat(name, ":\n")
    cat("  MAE:", round(hybrid_results[[name]]$mae, 4), "\n")
    cat("  RMSE:", round(hybrid_results[[name]]$rmse, 4), "\n")
    cat("  Coverage:", round(hybrid_results[[name]]$coverage, 4), "\n")
    cat("\n")
  }
  
  # Find optimal weight
  best_alpha <- names(hybrid_results)[which.min(sapply(hybrid_results, function(x) x$mae))]
  cat("Best performing hybrid:", best_alpha, "\n")
  
  return(hybrid_results)
}

demonstrate_scalability_analysis <- function() {
  cat("=== Scalability Analysis ===\n\n")
  
  # Test with different dataset sizes
  dataset_sizes <- c(50, 100, 200, 500)
  training_times <- list(UBCF = numeric(length(dataset_sizes)), IBCF = numeric(length(dataset_sizes)))
  
  for (i in seq_along(dataset_sizes)) {
    size <- dataset_sizes[i]
    cat("Testing with", size, "users and", size %/% 2, "items...\n")
    
    # Generate data
    ratings_df <- generate_synthetic_data_with_clusters(n_users = size, n_items = size %/% 2)
    
    # Time UBCF training
    start_time <- Sys.time()
    
    ubcf <- UBCF(similarity_metric = "pearson", k_neighbors = 10)
    ubcf <- fit_ubcf(ubcf, ratings_df)
    
    ubcf_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    training_times$UBCF[i] <- ubcf_time
    
    # Time IBCF training
    start_time <- Sys.time()
    
    ibcf <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 10)
    ibcf <- fit_ibcf(ibcf, ratings_df)
    
    ibcf_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    training_times$IBCF[i] <- ibcf_time
    
    cat("  UBCF training time:", round(ubcf_time, 3), "s\n")
    cat("  IBCF training time:", round(ibcf_time, 3), "s\n")
    cat("  Ratio (UBCF/IBCF):", round(ubcf_time/ibcf_time, 2), "\n")
  }
  
  return(list(dataset_sizes = dataset_sizes, training_times = training_times))
}

demonstrate_visualization <- function() {
  cat("=== UBCF vs IBCF Visualizations ===\n\n")
  
  # Generate data and train models
  ratings_df <- generate_synthetic_data_with_clusters()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train models
  ubcf_pearson <- UBCF(similarity_metric = "pearson", k_neighbors = 15)
  ubcf_pearson <- fit_ubcf(ubcf_pearson, train_df)
  
  ibcf_adjusted_cosine <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 15)
  ibcf_adjusted_cosine <- fit_ibcf(ibcf_adjusted_cosine, train_df)
  
  # Evaluate models
  models <- list(
    "UBCF-Pearson" = list(model = ubcf_pearson, predict_func = predict_ubcf),
    "IBCF-AdjustedCosine" = list(model = ibcf_adjusted_cosine, predict_func = predict_ibcf)
  )
  
  results <- list()
  for (name in names(models)) {
    results[[name]] <- evaluate_model(models[[name]]$model, test_df, models[[name]]$predict_func)
  }
  
  # Create visualizations
  # Rating distribution
  p1 <- ggplot(ratings_df, aes(x = factor(rating))) +
    geom_bar(fill = "steelblue") +
    labs(title = "Rating Distribution",
         x = "Rating", y = "Count") +
    theme_minimal()
  
  # User-item matrix heatmap (sample)
  sample_matrix <- ratings_df %>%
    spread(item_id, rating, fill = NA) %>%
    select(-user_id) %>%
    as.matrix()
  
  sample_matrix <- sample_matrix[1:min(20, nrow(sample_matrix)), 1:min(20, ncol(sample_matrix))]
  sample_df <- expand.grid(
    user_id = 1:nrow(sample_matrix),
    item_id = 1:ncol(sample_matrix)
  )
  sample_df$rating <- as.vector(sample_matrix)
  
  p2 <- ggplot(sample_df, aes(x = item_id, y = user_id, fill = rating)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "Rating Matrix (Sample)",
         x = "Item ID", y = "User ID") +
    theme_minimal()
  
  # Method comparison
  method_names <- names(results)
  mae_values <- sapply(results, function(x) x$mae)
  rmse_values <- sapply(results, function(x) x$rmse)
  
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
  
  # Similarity distributions
  user_similarities <- ubcf_pearson$user_similarity[upper.tri(ubcf_pearson$user_similarity)]
  item_similarities <- ibcf_adjusted_cosine$item_similarity[upper.tri(ibcf_adjusted_cosine$item_similarity)]
  
  p5 <- ggplot(data.frame(similarity = user_similarities), aes(x = similarity)) +
    geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
    labs(title = "User Similarity Distribution (UBCF)",
         x = "Similarity Score", y = "Frequency") +
    theme_minimal()
  
  p6 <- ggplot(data.frame(similarity = item_similarities), aes(x = similarity)) +
    geom_histogram(bins = 30, fill = "red", alpha = 0.7) +
    labs(title = "Item Similarity Distribution (IBCF)",
         x = "Similarity Score", y = "Frequency") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2)
  
  return(results)
}

demonstrate_detailed_analysis <- function() {
  cat("=== Detailed Analysis ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_data_with_clusters()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train models
  ubcf <- UBCF(similarity_metric = "pearson", k_neighbors = 15)
  ubcf <- fit_ubcf(ubcf, train_df)
  
  ibcf <- IBCF(similarity_metric = "adjusted_cosine", k_neighbors = 15)
  ibcf <- fit_ibcf(ibcf, train_df)
  
  # Compare prediction patterns
  test_sample <- test_df[1:min(50, nrow(test_df)), ]
  ubcf_preds <- numeric(nrow(test_sample))
  ibcf_preds <- numeric(nrow(test_sample))
  actuals <- numeric(nrow(test_sample))
  valid_count <- 0
  
  for (i in 1:nrow(test_sample)) {
    ubcf_pred <- predict_ubcf(ubcf, test_sample$user_id[i], test_sample$item_id[i])
    ibcf_pred <- predict_ibcf(ibcf, test_sample$user_id[i], test_sample$item_id[i])
    
    if (!is.na(ubcf_pred) && !is.na(ibcf_pred)) {
      valid_count <- valid_count + 1
      ubcf_preds[valid_count] <- ubcf_pred
      ibcf_preds[valid_count] <- ibcf_pred
      actuals[valid_count] <- test_sample$rating[i]
    }
  }
  
  ubcf_preds <- ubcf_preds[1:valid_count]
  ibcf_preds <- ibcf_preds[1:valid_count]
  actuals <- actuals[1:valid_count]
  
  cat("UBCF Prediction Statistics:\n")
  cat("  Mean:", round(mean(ubcf_preds), 3), "\n")
  cat("  Std:", round(sd(ubcf_preds), 3), "\n")
  cat("  Range: [", round(min(ubcf_preds), 3), ", ", round(max(ubcf_preds), 3), "]\n", sep = "")
  
  cat("\nIBCF Prediction Statistics:\n")
  cat("  Mean:", round(mean(ibcf_preds), 3), "\n")
  cat("  Std:", round(sd(ibcf_preds), 3), "\n")
  cat("  Range: [", round(min(ibcf_preds), 3), ", ", round(max(ibcf_preds), 3), "]\n", sep = "")
  
  # Compare similarity distributions
  user_similarities <- ubcf$user_similarity[upper.tri(ubcf$user_similarity)]
  item_similarities <- ibcf$item_similarity[upper.tri(ibcf$item_similarity)]
  
  cat("\nSimilarity Distribution Comparison:\n")
  cat("UBCF User Similarities:\n")
  cat("  Mean:", round(mean(user_similarities, na.rm = TRUE), 3), "\n")
  cat("  Std:", round(sd(user_similarities, na.rm = TRUE), 3), "\n")
  cat("  Range: [", round(min(user_similarities, na.rm = TRUE), 3), ", ", 
      round(max(user_similarities, na.rm = TRUE), 3), "]\n", sep = "")
  
  cat("\nIBCF Item Similarities:\n")
  cat("  Mean:", round(mean(item_similarities, na.rm = TRUE), 3), "\n")
  cat("  Std:", round(sd(item_similarities, na.rm = TRUE), 3), "\n")
  cat("  Range: [", round(min(item_similarities, na.rm = TRUE), 3), ", ", 
      round(max(item_similarities, na.rm = TRUE), 3), "]\n", sep = "")
  
  # Analyze cold start scenarios
  cat("\nCold Start Analysis:\n")
  
  # New user scenario
  new_user_id <- max(ratings_df$user_id) + 1
  new_item_id <- max(ratings_df$item_id) + 1
  
  ubcf_new_user_pred <- predict_ubcf(ubcf, new_user_id, 1)  # Try to predict for new user
  ibcf_new_user_pred <- predict_ibcf(ibcf, new_user_id, 1)
  
  cat("  New User Prediction (UBCF):", ubcf_new_user_pred, "\n")
  cat("  New User Prediction (IBCF):", ibcf_new_user_pred, "\n")
  
  # New item scenario
  ubcf_new_item_pred <- predict_ubcf(ubcf, 1, new_item_id)  # Try to predict for new item
  ibcf_new_item_pred <- predict_ibcf(ibcf, 1, new_item_id)
  
  cat("  New Item Prediction (UBCF):", ubcf_new_item_pred, "\n")
  cat("  New Item Prediction (IBCF):", ibcf_new_item_pred, "\n")
  
  return(list(
    ubcf_preds = ubcf_preds,
    ibcf_preds = ibcf_preds,
    actuals = actuals,
    user_similarities = user_similarities,
    item_similarities = item_similarities
  ))
}

main_r <- function() {
  cat("UBCF vs IBCF: Comprehensive Comparison and Analysis\n")
  cat("=" %R% 60, "\n")
  
  # 1. Basic comparison demonstration
  cat("\n1. Basic UBCF vs IBCF Comparison:\n")
  basic_results <- demonstrate_basic_comparison()
  
  # 2. Similarity analysis
  cat("\n2. Similarity Analysis:\n")
  similarity_results <- demonstrate_similarity_analysis()
  
  # 3. Hybrid approaches
  cat("\n3. Hybrid Recommendation Approaches:\n")
  hybrid_results <- demonstrate_hybrid_approaches()
  
  # 4. Scalability analysis
  cat("\n4. Scalability Analysis:\n")
  scalability_results <- demonstrate_scalability_analysis()
  
  # 5. Visualizations
  cat("\n5. Comprehensive Visualizations:\n")
  viz_results <- demonstrate_visualization()
  
  # 6. Detailed analysis
  cat("\n6. Detailed Analysis:\n")
  detailed_results <- demonstrate_detailed_analysis()
  
  cat("\n=== Key Insights ===\n")
  cat("1. UBCF excels in interpretability and handling new items\n")
  cat("2. IBCF excels in scalability and stability\n")
  cat("3. Hybrid approaches often provide the best performance\n")
  cat("4. Choice depends on data characteristics and application requirements\n")
  cat("5. Both methods have different strengths and use cases\n")
  cat("6. Scalability considerations are crucial for large datasets\n")
  
  return(list(
    basic_results = basic_results,
    similarity_results = similarity_results,
    hybrid_results = hybrid_results,
    scalability_results = scalability_results,
    viz_results = viz_results,
    detailed_results = detailed_results
  ))
}

# Run main function if not interactive
if (!interactive()) {
  main_r()
}
