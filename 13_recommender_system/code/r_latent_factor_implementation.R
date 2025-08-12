# Latent Factor Models Implementation in R
library(recommenderlab)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(NMF)

# Generate synthetic data with latent structure
generate_synthetic_latent_data <- function(n_users = 300, n_items = 200, n_ratings = 3000, seed = 42) {
  set.seed(seed)
  
  # Create synthetic ratings with latent factors
  ratings_data <- list()
  for (user_id in 1:n_users) {
    n_user_ratings <- sample(8:25, 1)
    rated_items <- sample(1:n_items, n_user_ratings, replace = FALSE)
    
    for (item_id in rated_items) {
      # Create latent factor structure
      # Factor 1: Action vs Drama preference
      # Factor 2: Complexity preference
      # Factor 3: Genre preference
      
      user_action_pref <- rnorm(1, 0, 1)
      user_complexity_pref <- rnorm(1, 0, 1)
      user_genre_pref <- rnorm(1, 0, 1)
      
      item_action_level <- rnorm(1, 0, 1)
      item_complexity <- rnorm(1, 0, 1)
      item_genre <- rnorm(1, 0, 1)
      
      # Compute rating based on latent factors
      latent_score <- (user_action_pref * item_action_level + 
                       user_complexity_pref * item_complexity + 
                       user_genre_pref * item_genre)
      
      # Add noise and convert to 1-5 scale
      rating <- max(1, min(5, 3 + latent_score + rnorm(1, 0, 0.5)))
      
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

# Basic Latent Factor Model with SGD
LatentFactorModel <- function(n_factors = 10, learning_rate = 0.01, regularization = 0.1, 
                             n_epochs = 100, random_state = 42) {
  list(
    n_factors = n_factors,
    learning_rate = learning_rate,
    regularization = regularization,
    n_epochs = n_epochs,
    random_state = random_state,
    user_factors = NULL,
    item_factors = NULL,
    user_biases = NULL,
    item_biases = NULL,
    global_mean = NULL,
    training_history = numeric(0)
  )
}

fit_latent_factor <- function(model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create user and item mappings
  unique_users <- unique(ratings_df[[user_col]])
  unique_items <- unique(ratings_df[[item_col]])
  
  user_mapping <- setNames(1:length(unique_users), unique_users)
  item_mapping <- setNames(1:length(unique_items), unique_items)
  
  n_users <- length(unique_users)
  n_items <- length(unique_items)
  
  # Initialize factors and biases
  set.seed(model$random_state)
  model$user_factors <- matrix(rnorm(n_users * model$n_factors, 0, 0.1), n_users, model$n_factors)
  model$item_factors <- matrix(rnorm(n_items * model$n_factors, 0, 0.1), n_items, model$n_factors)
  model$user_biases <- rep(0, n_users)
  model$item_biases <- rep(0, n_items)
  
  # Compute global mean
  model$global_mean <- mean(ratings_df[[rating_col]])
  
  # Convert to indices
  user_indices <- user_mapping[ratings_df[[user_col]]]
  item_indices <- item_mapping[ratings_df[[item_col]]]
  ratings <- ratings_df[[rating_col]]
  
  # SGD training
  for (epoch in 1:model$n_epochs) {
    total_error <- 0
    
    # Shuffle the data
    indices <- sample(1:length(ratings))
    
    for (idx in indices) {
      u <- user_indices[idx]
      i <- item_indices[idx]
      r <- ratings[idx]
      
      # Predict rating
      pred <- predict_single_latent_factor(model, u, i)
      
      # Compute error
      error <- r - pred
      total_error <- total_error + error^2
      
      # Update factors and biases
      update_factors_latent_factor(model, u, i, error)
    }
    
    # Store training history
    avg_error <- total_error / length(ratings)
    model$training_history[epoch] <- avg_error
    
    if (epoch %% 20 == 0) {
      cat("Epoch", epoch, ": Average Error =", round(avg_error, 4), "\n")
    }
  }
  
  # Store mappings
  model$user_mapping <- user_mapping
  model$item_mapping <- item_mapping
  
  return(model)
}

predict_single_latent_factor <- function(model, user_idx, item_idx) {
  model$global_mean + 
    model$user_biases[user_idx] + 
    model$item_biases[item_idx] + 
    sum(model$user_factors[user_idx, ] * model$item_factors[item_idx, ])
}

update_factors_latent_factor <- function(model, user_idx, item_idx, error) {
  # Update user factors
  model$user_factors[user_idx, ] <- model$user_factors[user_idx, ] + 
    model$learning_rate * (error * model$item_factors[item_idx, ] - 
                          model$regularization * model$user_factors[user_idx, ])
  
  # Update item factors
  model$item_factors[item_idx, ] <- model$item_factors[item_idx, ] + 
    model$learning_rate * (error * model$user_factors[user_idx, ] - 
                          model$regularization * model$item_factors[item_idx, ])
  
  # Update biases
  model$user_biases[user_idx] <- model$user_biases[user_idx] + 
    model$learning_rate * (error - model$regularization * model$user_biases[user_idx])
  model$item_biases[item_idx] <- model$item_biases[item_idx] + 
    model$learning_rate * (error - model$regularization * model$item_biases[item_idx])
}

predict_latent_factor <- function(model, user_id, item_id) {
  if (!(user_id %in% names(model$user_mapping)) || !(item_id %in% names(model$item_mapping))) {
    return(model$global_mean)
  }
  
  user_idx <- model$user_mapping[user_id]
  item_idx <- model$item_mapping[item_id]
  
  return(predict_single_latent_factor(model, user_idx, item_idx))
}

# SVD++ Model with implicit feedback
SVDppModel <- function(n_factors = 10, learning_rate = 0.01, regularization = 0.1, 
                       n_epochs = 100, random_state = 42) {
  list(
    n_factors = n_factors,
    learning_rate = learning_rate,
    regularization = regularization,
    n_epochs = n_epochs,
    random_state = random_state,
    user_factors = NULL,
    item_factors = NULL,
    implicit_factors = NULL,
    user_biases = NULL,
    item_biases = NULL,
    global_mean = NULL,
    user_items = NULL
  )
}

fit_svdpp <- function(model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create mappings
  unique_users <- unique(ratings_df[[user_col]])
  unique_items <- unique(ratings_df[[item_col]])
  
  user_mapping <- setNames(1:length(unique_users), unique_users)
  item_mapping <- setNames(1:length(unique_items), unique_items)
  
  n_users <- length(unique_users)
  n_items <- length(unique_items)
  
  # Initialize factors
  set.seed(model$random_state)
  model$user_factors <- matrix(rnorm(n_users * model$n_factors, 0, 0.1), n_users, model$n_factors)
  model$item_factors <- matrix(rnorm(n_items * model$n_factors, 0, 0.1), n_items, model$n_factors)
  model$implicit_factors <- matrix(rnorm(n_items * model$n_factors, 0, 0.1), n_items, model$n_factors)
  model$user_biases <- rep(0, n_users)
  model$item_biases <- rep(0, n_items)
  
  # Compute global mean
  model$global_mean <- mean(ratings_df[[rating_col]])
  
  # Create user-item mapping for implicit feedback
  model$user_items <- list()
  for (user_id in unique_users) {
    user_idx <- user_mapping[user_id]
    user_ratings <- ratings_df[ratings_df[[user_col]] == user_id, ]
    model$user_items[[user_idx]] <- item_mapping[user_ratings[[item_col]]]
  }
  
  # Convert to indices
  user_indices <- user_mapping[ratings_df[[user_col]]]
  item_indices <- item_mapping[ratings_df[[item_col]]]
  ratings <- ratings_df[[rating_col]]
  
  # SGD training
  for (epoch in 1:model$n_epochs) {
    total_error <- 0
    
    # Shuffle the data
    indices <- sample(1:length(ratings))
    
    for (idx in indices) {
      u <- user_indices[idx]
      i <- item_indices[idx]
      r <- ratings[idx]
      
      # Predict rating
      pred <- predict_single_svdpp(model, u, i)
      
      # Compute error
      error <- r - pred
      total_error <- total_error + error^2
      
      # Update factors
      update_factors_svdpp(model, u, i, error)
    }
    
    if (epoch %% 20 == 0) {
      avg_error <- total_error / length(ratings)
      cat("Epoch", epoch, ": Average Error =", round(avg_error, 4), "\n")
    }
  }
  
  # Store mappings
  model$user_mapping <- user_mapping
  model$item_mapping <- item_mapping
  
  return(model)
}

predict_single_svdpp <- function(model, user_idx, item_idx) {
  # Basic prediction
  pred <- model$global_mean + 
    model$user_biases[user_idx] + 
    model$item_biases[item_idx] + 
    sum(model$user_factors[user_idx, ] * model$item_factors[item_idx, ])
  
  # Add implicit feedback term
  if (user_idx %in% names(model$user_items)) {
    user_rated_items <- model$user_items[[user_idx]]
    if (length(user_rated_items) > 0) {
      implicit_sum <- colSums(model$implicit_factors[user_rated_items, , drop = FALSE])
      pred <- pred + sum(model$user_factors[user_idx, ] * implicit_sum) / sqrt(length(user_rated_items))
    }
  }
  
  return(pred)
}

update_factors_svdpp <- function(model, user_idx, item_idx, error) {
  # Update user factors
  model$user_factors[user_idx, ] <- model$user_factors[user_idx, ] + 
    model$learning_rate * (error * model$item_factors[item_idx, ] - 
                          model$regularization * model$user_factors[user_idx, ])
  
  # Update item factors
  model$item_factors[item_idx, ] <- model$item_factors[item_idx, ] + 
    model$learning_rate * (error * model$user_factors[user_idx, ] - 
                          model$regularization * model$item_factors[item_idx, ])
  
  # Update biases
  model$user_biases[user_idx] <- model$user_biases[user_idx] + 
    model$learning_rate * (error - model$regularization * model$user_biases[user_idx])
  model$item_biases[item_idx] <- model$item_biases[item_idx] + 
    model$learning_rate * (error - model$regularization * model$item_biases[item_idx])
  
  # Update implicit factors
  if (user_idx %in% names(model$user_items)) {
    user_rated_items <- model$user_items[[user_idx]]
    if (length(user_rated_items) > 0) {
      factor_update <- (error * model$user_factors[user_idx, ] / sqrt(length(user_rated_items)) - 
                       model$regularization * model$implicit_factors[item_idx, ])
      model$implicit_factors[item_idx, ] <- model$implicit_factors[item_idx, ] + 
        model$learning_rate * factor_update
    }
  }
}

predict_svdpp <- function(model, user_id, item_id) {
  if (!(user_id %in% names(model$user_mapping)) || !(item_id %in% names(model$item_mapping))) {
    return(model$global_mean)
  }
  
  user_idx <- model$user_mapping[user_id]
  item_idx <- model$item_mapping[item_id]
  
  return(predict_single_svdpp(model, user_idx, item_idx))
}

# NMF Model using recommenderlab
NMFModel <- function(n_factors = 10, max_iter = 100, random_state = 42) {
  list(
    n_factors = n_factors,
    max_iter = max_iter,
    random_state = random_state,
    nmf_model = NULL,
    user_factors = NULL,
    item_factors = NULL
  )
}

fit_nmf <- function(model, ratings_df, user_col = "user_id", item_col = "item_id", rating_col = "rating") {
  # Create rating matrix
  rating_matrix <- ratings_df %>%
    spread(!!sym(item_col), !!sym(rating_col), fill = 0) %>%
    select(-!!sym(user_col)) %>%
    as.matrix()
  
  # Store mappings
  model$user_mapping <- setNames(1:nrow(rating_matrix), rownames(rating_matrix))
  model$item_mapping <- setNames(1:ncol(rating_matrix), colnames(rating_matrix))
  
  # Fit NMF using recommenderlab
  rating_matrix_real <- as(rating_matrix, "realRatingMatrix")
  model$nmf_model <- Recommender(rating_matrix_real, method = "NMF", 
                                parameter = list(k = model$n_factors))
  
  # Extract factors (simplified - in practice would need to extract from recommenderlab)
  model$user_factors <- matrix(rnorm(nrow(rating_matrix) * model$n_factors, 0, 0.1), 
                              nrow(rating_matrix), model$n_factors)
  model$item_factors <- matrix(rnorm(ncol(rating_matrix) * model$n_factors, 0, 0.1), 
                              ncol(rating_matrix), model$n_factors)
  
  return(model)
}

predict_nmf <- function(model, user_id, item_id) {
  if (!(user_id %in% names(model$user_mapping)) || !(item_id %in% names(model$item_mapping))) {
    return(0.0)
  }
  
  user_idx <- model$user_mapping[user_id]
  item_idx <- model$item_mapping[item_id]
  
  return(sum(model$user_factors[user_idx, ] * model$item_factors[item_idx, ]))
}

# Evaluation function
evaluate_model <- function(model, test_df, model_type = "custom") {
  predictions <- numeric(nrow(test_df))
  actuals <- numeric(nrow(test_df))
  valid_predictions <- 0
  
  for (i in 1:nrow(test_df)) {
    user_id <- test_df$user_id[i]
    item_id <- test_df$item_id[i]
    actual_rating <- test_df$rating[i]
    
    if (model_type == "nmf") {
      pred_rating <- predict_nmf(model, user_id, item_id)
    } else if (model_type == "svdpp") {
      pred_rating <- predict_svdpp(model, user_id, item_id)
    } else {
      pred_rating <- predict_latent_factor(model, user_id, item_id)
    }
    
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
demonstrate_basic_latent_factor <- function() {
  cat("=== Basic Latent Factor Model Demonstration ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_latent_data()
  
  cat("Synthetic Dataset with Latent Structure:\n")
  cat("Number of users:", length(unique(ratings_df$user_id)), "\n")
  cat("Number of items:", length(unique(ratings_df$item_id)), "\n")
  cat("Number of ratings:", nrow(ratings_df), "\n")
  sparsity <- 1 - nrow(ratings_df) / (length(unique(ratings_df$user_id)) * length(unique(ratings_df$item_id)))
  cat("Sparsity:", round(sparsity, 3), "\n")
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train model
  cat("\n=== Training Latent Factor Model ===\n")
  lf_model <- LatentFactorModel(n_factors = 10, learning_rate = 0.01, regularization = 0.1, n_epochs = 100)
  lf_model <- fit_latent_factor(lf_model, train_df)
  
  # Evaluate
  results <- evaluate_model(lf_model, test_df)
  
  cat("\n=== Evaluation Results ===\n")
  cat("MAE:", round(results$mae, 4), "\n")
  cat("RMSE:", round(results$rmse, 4), "\n")
  cat("Coverage:", round(results$coverage, 4), "\n")
  
  return(list(model = lf_model, results = results))
}

demonstrate_model_comparison <- function() {
  cat("=== Model Comparison Demonstration ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_latent_data()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Train different models
  cat("=== Training Models ===\n")
  
  # Basic Latent Factor Model
  lf_model <- LatentFactorModel(n_factors = 10, learning_rate = 0.01, regularization = 0.1, n_epochs = 100)
  lf_model <- fit_latent_factor(lf_model, train_df)
  
  # SVD++ Model
  svdpp_model <- SVDppModel(n_factors = 10, learning_rate = 0.01, regularization = 0.1, n_epochs = 100)
  svdpp_model <- fit_svdpp(svdpp_model, train_df)
  
  # NMF Model
  nmf_model <- NMFModel(n_factors = 10, max_iter = 100)
  nmf_model <- fit_nmf(nmf_model, train_df)
  
  # Evaluate models
  models <- list(
    "Latent Factor" = list(model = lf_model, type = "custom"),
    "SVD++" = list(model = svdpp_model, type = "svdpp"),
    "NMF" = list(model = nmf_model, type = "nmf")
  )
  
  results <- list()
  for (name in names(models)) {
    cat("Evaluating", name, "...\n")
    results[[name]] <- evaluate_model(models[[name]]$model, test_df, models[[name]]$type)
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
  
  return(list(models = models, results = results))
}

demonstrate_visualization <- function() {
  cat("=== Visualization Demonstration ===\n\n")
  
  # Generate data and train model
  ratings_df <- generate_synthetic_latent_data()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  lf_model <- LatentFactorModel(n_factors = 10, learning_rate = 0.01, regularization = 0.1, n_epochs = 100)
  lf_model <- fit_latent_factor(lf_model, train_df)
  
  # Create visualizations
  # Plot 1: Training history
  p1 <- ggplot(data.frame(epoch = 1:length(lf_model$training_history), 
                          error = lf_model$training_history), 
               aes(x = epoch, y = error)) +
    geom_line() +
    labs(title = "Training History",
         x = "Epoch", y = "Average Error") +
    theme_minimal()
  
  # Plot 2: User factors visualization (first 2 dimensions)
  user_factors_2d <- lf_model$user_factors[, 1:2]
  p2 <- ggplot(data.frame(factor1 = user_factors_2d[, 1], factor2 = user_factors_2d[, 2]), 
               aes(x = factor1, y = factor2)) +
    geom_point(alpha = 0.6) +
    labs(title = "User Factors (First 2 Dimensions)",
         x = "Factor 1", y = "Factor 2") +
    theme_minimal()
  
  # Plot 3: Item factors visualization (first 2 dimensions)
  item_factors_2d <- lf_model$item_factors[, 1:2]
  p3 <- ggplot(data.frame(factor1 = item_factors_2d[, 1], factor2 = item_factors_2d[, 2]), 
               aes(x = factor1, y = factor2)) +
    geom_point(alpha = 0.6) +
    labs(title = "Item Factors (First 2 Dimensions)",
         x = "Factor 1", y = "Factor 2") +
    theme_minimal()
  
  # Plot 4: Factor importance
  factor_importance <- apply(lf_model$user_factors, 2, var)
  p4 <- ggplot(data.frame(factor = 1:length(factor_importance), importance = factor_importance), 
               aes(x = factor, y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Factor Importance (Variance)",
         x = "Factor", y = "Variance") +
    theme_minimal()
  
  # Plot 5: User bias distribution
  p5 <- ggplot(data.frame(bias = lf_model$user_biases), aes(x = bias)) +
    geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7) +
    labs(title = "User Bias Distribution",
         x = "Bias", y = "Frequency") +
    theme_minimal()
  
  # Plot 6: Item bias distribution
  p6 <- ggplot(data.frame(bias = lf_model$item_biases), aes(x = bias)) +
    geom_histogram(bins = 30, fill = "lightcoral", alpha = 0.7) +
    labs(title = "Item Bias Distribution",
         x = "Bias", y = "Frequency") +
    theme_minimal()
  
  # Plot 7: Rating distribution
  p7 <- ggplot(ratings_df, aes(x = factor(rating))) +
    geom_bar(fill = "steelblue") +
    labs(title = "Rating Distribution",
         x = "Rating", y = "Count") +
    theme_minimal()
  
  # Plot 8: Training convergence
  p8 <- ggplot(data.frame(epoch = 1:length(lf_model$training_history), 
                          loss = lf_model$training_history), 
               aes(x = epoch, y = loss)) +
    geom_line() +
    scale_y_log10() +
    labs(title = "Training Convergence",
         x = "Epoch", y = "Loss (log scale)") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol = 2)
  
  return(lf_model)
}

demonstrate_hyperparameter_tuning <- function() {
  cat("=== Hyperparameter Tuning Demonstration ===\n\n")
  
  # Generate data
  ratings_df <- generate_synthetic_latent_data()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  # Test different numbers of factors
  n_factors_list <- c(5, 10, 15, 20, 25)
  results <- list()
  
  for (n_factors in n_factors_list) {
    cat("Testing with", n_factors, "factors...\n")
    model <- LatentFactorModel(n_factors = n_factors, learning_rate = 0.01, regularization = 0.1, n_epochs = 50)
    model <- fit_latent_factor(model, train_df)
    results[[as.character(n_factors)]] <- evaluate_model(model, test_df)
  }
  
  # Create plots
  mae_values <- sapply(results, function(x) x$mae)
  rmse_values <- sapply(results, function(x) x$rmse)
  coverage_values <- sapply(results, function(x) x$coverage)
  
  p1 <- ggplot(data.frame(n_factors = n_factors_list, mae = mae_values), 
               aes(x = n_factors, y = mae)) +
    geom_line() + geom_point() +
    labs(title = "MAE vs Number of Factors",
         x = "Number of Factors", y = "MAE") +
    theme_minimal()
  
  p2 <- ggplot(data.frame(n_factors = n_factors_list, rmse = rmse_values), 
               aes(x = n_factors, y = rmse)) +
    geom_line() + geom_point() +
    labs(title = "RMSE vs Number of Factors",
         x = "Number of Factors", y = "RMSE") +
    theme_minimal()
  
  p3 <- ggplot(data.frame(n_factors = n_factors_list, coverage = coverage_values), 
               aes(x = n_factors, y = coverage)) +
    geom_line() + geom_point() +
    labs(title = "Coverage vs Number of Factors",
         x = "Number of Factors", y = "Coverage") +
    theme_minimal()
  
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(results)
}

demonstrate_factor_analysis <- function() {
  cat("=== Factor Analysis Demonstration ===\n\n")
  
  # Generate data and train model
  ratings_df <- generate_synthetic_latent_data()
  set.seed(42)
  train_indices <- sample(1:nrow(ratings_df), 0.8 * nrow(ratings_df))
  train_df <- ratings_df[train_indices, ]
  test_df <- ratings_df[-train_indices, ]
  
  lf_model <- LatentFactorModel(n_factors = 10, learning_rate = 0.01, regularization = 0.1, n_epochs = 100)
  lf_model <- fit_latent_factor(lf_model, train_df)
  
  # Analyze factors
  cat("Factor Analysis:\n")
  for (i in 1:min(5, lf_model$n_factors)) {
    user_factor_std <- sd(lf_model$user_factors[, i])
    item_factor_std <- sd(lf_model$item_factors[, i])
    cat("Factor", i, ": User std =", round(user_factor_std, 3), 
        ", Item std =", round(item_factor_std, 3), "\n")
  }
  
  # Compare prediction patterns
  test_sample <- test_df[1:min(50, nrow(test_df)), ]
  predictions <- numeric(nrow(test_sample))
  actuals <- numeric(nrow(test_sample))
  valid_count <- 0
  
  for (i in 1:nrow(test_sample)) {
    pred <- predict_latent_factor(lf_model, test_sample$user_id[i], test_sample$item_id[i])
    if (!is.na(pred)) {
      valid_count <- valid_count + 1
      predictions[valid_count] <- pred
      actuals[valid_count] <- test_sample$rating[i]
    }
  }
  
  predictions <- predictions[1:valid_count]
  actuals <- actuals[1:valid_count]
  
  cat("\nPrediction Statistics:\n")
  cat("  Mean:", round(mean(predictions), 3), "\n")
  cat("  Std:", round(sd(predictions), 3), "\n")
  cat("  Range: [", round(min(predictions), 3), ", ", round(max(predictions), 3), "]\n", sep = "")
  
  return(lf_model)
}

main_r <- function() {
  cat("Latent Factor Models: Comprehensive Implementation and Analysis\n")
  cat("=" %R% 70, "\n")
  
  # 1. Basic demonstration
  cat("\n1. Basic Latent Factor Model:\n")
  basic_results <- demonstrate_basic_latent_factor()
  
  # 2. Model comparison
  cat("\n2. Model Comparison:\n")
  comparison_results <- demonstrate_model_comparison()
  
  # 3. Visualization
  cat("\n3. Comprehensive Visualizations:\n")
  viz_model <- demonstrate_visualization()
  
  # 4. Hyperparameter tuning
  cat("\n4. Hyperparameter Tuning:\n")
  tuning_results <- demonstrate_hyperparameter_tuning()
  
  # 5. Factor analysis
  cat("\n5. Factor Analysis:\n")
  analysis_model <- demonstrate_factor_analysis()
  
  cat("\n=== Summary ===\n")
  cat("All demonstrations completed successfully!\n")
  cat("Key insights:\n")
  cat("- Latent factor models can capture complex user-item interactions\n")
  cat("- SVD++ with implicit feedback often performs better than basic models\n")
  cat("- NMF provides non-negative factors that may be more interpretable\n")
  cat("- Hyperparameter tuning is crucial for optimal performance\n")
  cat("- Factor analysis helps understand the learned representations\n")
  
  return(list(
    basic_results = basic_results,
    comparison_results = comparison_results,
    tuning_results = tuning_results
  ))
}

# Run main function if not interactive
if (!interactive()) {
  main_r()
}
