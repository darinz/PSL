# Gradient Boosting Machines (GBM) Implementation in R
# ===================================================
#
# This script demonstrates GBM implementation using R,
# including training, evaluation, hyperparameter tuning, and comparison with Random Forest.

library(gbm)
library(ggplot2)
library(dplyr)
library(MASS)
library(randomForest)

# Function to demonstrate GBM
demonstrate_gbm_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Prepare data
  X <- Boston[, -ncol(Boston)]
  y <- Boston$medv
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Train GBM
  gbm_model <- gbm(
    medv ~ .,
    data = data.frame(X_train, medv = y_train),
    distribution = "gaussian",
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = FALSE
  )
  
  # Find optimal number of trees
  best_iter <- gbm.perf(gbm_model, method = "cv")
  
  # Make predictions
  predictions <- predict(gbm_model, X_test, n.trees = best_iter)
  
  # Calculate metrics
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Test MSE:", round(mse, 4), "\n")
  cat("Test R²:", round(r2, 4), "\n")
  cat("Optimal trees:", best_iter, "\n")
  
  # Variable importance
  importance_df <- summary(gbm_model, n.trees = best_iter, plotit = FALSE)
  
  print("Variable Importance:")
  print(importance_df)
  
  # Visualize results
  par(mfrow = c(2, 2))
  
  # CV error vs number of trees
  plot(gbm_model$cv.error, type = "l", xlab = "Number of Trees",
       ylab = "CV Error", main = "Cross-Validation Error")
  abline(v = best_iter, col = "red", lty = 2)
  
  # Predictions vs actual
  plot(y_test, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "GBM Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  # Variable importance plot
  barplot(importance_df$rel.inf, names.arg = importance_df$var,
          main = "Variable Importance", las = 2)
  
  # Residuals
  residuals <- y_test - predictions
  plot(predictions, residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residual Plot")
  abline(h = 0, col = "red", lty = 2)
  
  return(gbm_model)
}

# Function to tune hyperparameters
tune_gbm_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Define parameter grid
  param_grid <- expand.grid(
    n.trees = c(50, 100, 200),
    interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1, 0.2),
    bag.fraction = c(0.5, 0.8, 1.0)
  )
  
  # Train models
  results <- list()
  for (i in 1:nrow(param_grid)) {
    cat("Training model", i, "of", nrow(param_grid), "\n")
    
    gbm_model <- gbm(
      medv ~ .,
      data = Boston,
      distribution = "gaussian",
      n.trees = param_grid$n.trees[i],
      interaction.depth = param_grid$interaction.depth[i],
      shrinkage = param_grid$shrinkage[i],
      bag.fraction = param_grid$bag.fraction[i],
      cv.folds = 5,
      verbose = FALSE
    )
    
    # Get CV error
    best_iter <- gbm.perf(gbm_model, method = "cv", plotit = FALSE)
    cv_error <- gbm_model$cv.error[best_iter]
    
    results[[i]] <- cv_error
  }
  
  # Find best parameters
  best_idx <- which.min(unlist(results))
  best_params <- param_grid[best_idx, ]
  
  cat("Best parameters:\n")
  cat("n.trees:", best_params$n.trees, "\n")
  cat("interaction.depth:", best_params$interaction.depth, "\n")
  cat("shrinkage:", best_params$shrinkage, "\n")
  cat("bag.fraction:", best_params$bag.fraction, "\n")
  cat("Best CV MSE:", results[[best_idx]], "\n")
  
  return(best_params)
}

# Function to compare GBM with Random Forest
compare_gbm_rf_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  train_data <- Boston[train_indices, ]
  test_data <- Boston[-train_indices, ]
  
  # Train GBM
  gbm_model <- gbm(
    medv ~ .,
    data = train_data,
    distribution = "gaussian",
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = FALSE
  )
  
  # Find optimal number of trees
  best_iter <- gbm.perf(gbm_model, method = "cv", plotit = FALSE)
  
  # Train Random Forest
  rf_model <- randomForest(
    medv ~ .,
    data = train_data,
    ntree = 100,
    mtry = sqrt(ncol(train_data) - 1),
    importance = TRUE
  )
  
  # Make predictions
  gbm_pred <- predict(gbm_model, test_data, n.trees = best_iter)
  rf_pred <- predict(rf_model, test_data)
  
  # Calculate metrics
  gbm_mse <- mean((test_data$medv - gbm_pred)^2)
  gbm_r2 <- 1 - sum((test_data$medv - gbm_pred)^2) / sum((test_data$medv - mean(test_data$medv))^2)
  
  rf_mse <- mean((test_data$medv - rf_pred)^2)
  rf_r2 <- 1 - sum((test_data$medv - rf_pred)^2) / sum((test_data$medv - mean(test_data$medv))^2)
  
  # Results
  results <- data.frame(
    Model = c("Gradient Boosting", "Random Forest"),
    MSE = c(gbm_mse, rf_mse),
    R2 = c(gbm_r2, rf_r2)
  )
  
  print("Performance Comparison:")
  print(results)
  
  # Visualize comparison
  par(mfrow = c(1, 2))
  
  # Predictions comparison
  plot(test_data$medv, gbm_pred, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Predictions Comparison")
  points(test_data$medv, rf_pred, pch = 19, col = "red", alpha = 0.6)
  abline(0, 1, col = "black", lty = 2)
  legend("topleft", legend = c("GBM", "RF"), col = c("blue", "red"), pch = 19)
  
  # Residuals comparison
  gbm_residuals <- test_data$medv - gbm_pred
  rf_residuals <- test_data$medv - rf_pred
  
  plot(gbm_pred, gbm_residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residuals Comparison")
  points(rf_pred, rf_residuals, pch = 19, col = "red", alpha = 0.6)
  abline(h = 0, col = "black", lty = 2)
  legend("topright", legend = c("GBM", "RF"), col = c("blue", "red"), pch = 19)
  
  return(list(gbm_model = gbm_model, rf_model = rf_model, results = results))
}

# Function to demonstrate early stopping
demonstrate_early_stopping_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  train_data <- Boston[train_indices, ]
  test_data <- Boston[-train_indices, ]
  
  # Train GBM with many trees
  gbm_model <- gbm(
    medv ~ .,
    data = train_data,
    distribution = "gaussian",
    n.trees = 1000,  # Large number
    interaction.depth = 3,
    shrinkage = 0.1,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = FALSE
  )
  
  # Find optimal number of trees
  best_iter <- gbm.perf(gbm_model, method = "cv")
  
  cat("Optimal number of trees:", best_iter, "\n")
  cat("CV error at optimal:", gbm_model$cv.error[best_iter], "\n")
  
  # Plot CV error
  plot(gbm_model$cv.error, type = "l", xlab = "Number of Trees",
       ylab = "CV Error", main = "Cross-Validation Error with Early Stopping")
  abline(v = best_iter, col = "red", lty = 2)
  
  return(list(gbm_model = gbm_model, best_iter = best_iter))
}

# Function to analyze feature importance
analyze_feature_importance_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Train GBM
  gbm_model <- gbm(
    medv ~ .,
    data = Boston,
    distribution = "gaussian",
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    verbose = FALSE
  )
  
  # Get feature importance
  importance_df <- summary(gbm_model, plotit = FALSE)
  
  # Plot feature importance
  par(mfrow = c(1, 1))
  barplot(importance_df$rel.inf, names.arg = importance_df$var,
          main = "GBM Feature Importance", las = 2, col = "steelblue")
  
  print("Feature Importance:")
  print(importance_df)
  
  return(importance_df)
}

# Function to demonstrate learning rate effect
demonstrate_learning_rate_effect_r <- function() {
  # Load data
  data(Boston, package = "MASS")
  
  # Different learning rates
  learning_rates <- c(0.01, 0.1, 0.2)
  results <- list()
  
  for (lr in learning_rates) {
    cat("Training with learning rate:", lr, "\n")
    
    gbm_model <- gbm(
      medv ~ .,
      data = Boston,
      distribution = "gaussian",
      n.trees = 100,
      interaction.depth = 3,
      shrinkage = lr,
      cv.folds = 5,
      verbose = FALSE
    )
    
    best_iter <- gbm.perf(gbm_model, method = "cv", plotit = FALSE)
    cv_error <- gbm_model$cv.error[best_iter]
    
    results[[as.character(lr)]] <- list(
      learning_rate = lr,
      best_iter = best_iter,
      cv_error = cv_error
    )
  }
  
  # Plot results
  lr_values <- sapply(results, function(x) x$learning_rate)
  cv_errors <- sapply(results, function(x) x$cv_error)
  
  plot(lr_values, cv_errors, type = "b", pch = 19, col = "blue",
       xlab = "Learning Rate", ylab = "CV Error",
       main = "Effect of Learning Rate on CV Error")
  
  print("Learning Rate Effect:")
  for (lr in names(results)) {
    cat("Learning rate:", lr, "- CV Error:", results[[lr]]$cv_error, "\n")
  }
  
  return(results)
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate GBM
  cat("=== GBM DEMONSTRATION ===\n")
  gbm_model_r <- demonstrate_gbm_r()
  
  # Tune hyperparameters
  cat("\n=== HYPERPARAMETER TUNING ===\n")
  best_params_r <- tune_gbm_r()
  
  # Compare with Random Forest
  cat("\n=== COMPARISON WITH RANDOM FOREST ===\n")
  comparison_results <- compare_gbm_rf_r()
  
  # Demonstrate early stopping
  cat("\n=== EARLY STOPPING DEMONSTRATION ===\n")
  early_stopping_results <- demonstrate_early_stopping_r()
  
  # Analyze feature importance
  cat("\n=== FEATURE IMPORTANCE ANALYSIS ===\n")
  importance_analysis <- analyze_feature_importance_r()
  
  # Demonstrate learning rate effect
  cat("\n=== LEARNING RATE EFFECT ===\n")
  lr_effect <- demonstrate_learning_rate_effect_r()
}
