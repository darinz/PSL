# Random Forest Implementation in R
# =================================
#
# This script demonstrates Random Forest implementation using R,
# including training, evaluation, variable importance, and hyperparameter tuning.

library(randomForest)
library(ggplot2)
library(dplyr)
library(MASS)

# Function to demonstrate Random Forest
demonstrate_random_forest_r <- function() {
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
  
  # Train Random Forest
  rf_model <- randomForest(
    medv ~ ., 
    data = data.frame(X_train, medv = y_train),
    ntree = 100,
    mtry = sqrt(ncol(X_train)),  # sqrt(p) for regression
    importance = TRUE,
    keep.forest = TRUE
  )
  
  # Make predictions
  predictions <- predict(rf_model, X_test)
  
  # Calculate metrics
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Test MSE:", round(mse, 4), "\n")
  cat("Test R²:", round(r2, 4), "\n")
  cat("OOB MSE:", round(rf_model$mse[length(rf_model$mse)], 4), "\n")
  
  # Variable importance
  importance_df <- data.frame(
    feature = rownames(importance(rf_model)),
    importance = importance(rf_model)[, "%IncMSE"]
  ) %>%
    arrange(desc(importance))
  
  print("Variable Importance (Permutation):")
  print(importance_df)
  
  # Visualize results
  par(mfrow = c(2, 2))
  
  # Predictions vs actual
  plot(y_test, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Random Forest Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  # Variable importance plot
  varImpPlot(rf_model, main = "Variable Importance")
  
  # OOB error vs number of trees
  plot(rf_model$mse, type = "l", xlab = "Number of Trees",
       ylab = "OOB MSE", main = "OOB Error vs Number of Trees")
  
  # Residuals
  residuals <- y_test - predictions
  plot(predictions, residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residual Plot")
  abline(h = 0, col = "red", lty = 2)
  
  return(rf_model)
}

# Function to tune hyperparameters
tune_random_forest_r <- function() {
  library(caret)
  
  # Load data
  data(Boston, package = "MASS")
  
  # Define parameter grid
  param_grid <- expand.grid(
    mtry = c(2, 4, 6, 8),
    ntree = c(50, 100, 200)
  )
  
  # Control for cross-validation
  control <- trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  )
  
  # Train models
  results <- list()
  for (i in 1:nrow(param_grid)) {
    cat("Training model", i, "of", nrow(param_grid), "\n")
    
    rf_model <- randomForest(
      medv ~ .,
      data = Boston,
      mtry = param_grid$mtry[i],
      ntree = param_grid$ntree[i]
    )
    
    # Cross-validation score
    cv_scores <- numeric(5)
    for (fold in 1:5) {
      # Simple CV implementation
      test_indices <- sample(1:nrow(Boston), size = nrow(Boston) %/% 5)
      train_data <- Boston[-test_indices, ]
      test_data <- Boston[test_indices, ]
      
      fold_model <- randomForest(
        medv ~ .,
        data = train_data,
        mtry = param_grid$mtry[i],
        ntree = param_grid$ntree[i]
      )
      
      predictions <- predict(fold_model, test_data)
      cv_scores[fold] <- mean((test_data$medv - predictions)^2)
    }
    
    results[[i]] <- mean(cv_scores)
  }
  
  # Find best parameters
  best_idx <- which.min(unlist(results))
  best_params <- param_grid[best_idx, ]
  
  cat("Best parameters:\n")
  cat("mtry:", best_params$mtry, "\n")
  cat("ntree:", best_params$ntree, "\n")
  cat("Best CV MSE:", results[[best_idx]], "\n")
  
  return(best_params)
}

# Function to demonstrate bootstrap sampling
demonstrate_bootstrap_sampling_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 100
  X1 <- rnorm(n_samples)
  X2 <- rnorm(n_samples)
  X3 <- rnorm(n_samples)
  y <- 2 * X1 + 1.5 * X2 - 0.8 * X3 + rnorm(n_samples, 0, 0.5)
  
  cat("=== BOOTSTRAP SAMPLING DEMONSTRATION ===\n")
  cat("Original dataset:", n_samples, "samples\n")
  
  # Create multiple bootstrap samples
  n_bootstrap <- 10
  unique_samples_list <- numeric(n_bootstrap)
  
  for (i in 1:n_bootstrap) {
    # Create bootstrap sample
    indices <- sample(1:n_samples, size = n_samples, replace = TRUE)
    unique_samples <- length(unique(indices))
    unique_samples_list[i] <- unique_samples
    
    cat("Bootstrap", i, ":", unique_samples, "unique samples (", 
        round(unique_samples/n_samples*100, 1), "%)\n")
  }
  
  # Theoretical expectation
  theoretical_unique <- n_samples * (1 - exp(-1))
  cat("\nTheoretical expectation:", round(theoretical_unique, 1), 
      "unique samples (", round(theoretical_unique/n_samples*100, 1), "%)\n")
  cat("Average observed:", round(mean(unique_samples_list), 1), 
      "unique samples (", round(mean(unique_samples_list)/n_samples*100, 1), "%)\n")
  
  return(list(X = data.frame(X1, X2, X3), y = y))
}

# Function to demonstrate bagging
demonstrate_bagging_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 500
  X1 <- rnorm(n_samples)
  X2 <- rnorm(n_samples)
  X3 <- rnorm(n_samples)
  X4 <- rnorm(n_samples)
  y <- 2 * X1 + 1.5 * X2 - 0.8 * X3 + 0.5 * X4 + rnorm(n_samples, 0, 0.5)
  
  data <- data.frame(X1, X2, X3, X4, y = y)
  
  # Split data
  train_indices <- sample(1:n_samples, size = 0.8 * n_samples)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  cat("=== BAGGING DEMONSTRATION ===\n")
  cat("Training set:", nrow(train_data), "samples\n")
  cat("Test set:", nrow(test_data), "samples\n")
  
  # Train single tree
  single_tree <- rpart(y ~ ., data = train_data, control = rpart.control(maxdepth = 10))
  single_pred <- predict(single_tree, test_data)
  single_mse <- mean((test_data$y - single_pred)^2)
  single_r2 <- 1 - sum((test_data$y - single_pred)^2) / sum((test_data$y - mean(test_data$y))^2)
  
  cat("\nSingle Tree:\n")
  cat("  MSE:", round(single_mse, 4), "\n")
  cat("  R²:", round(single_r2, 4), "\n")
  
  # Train bagging ensemble
  bagging_model <- randomForest(
    y ~ .,
    data = train_data,
    ntree = 50,
    mtry = ncol(train_data) - 1,  # Use all features for bagging
    importance = FALSE
  )
  
  bagging_pred <- predict(bagging_model, test_data)
  bagging_mse <- mean((test_data$y - bagging_pred)^2)
  bagging_r2 <- 1 - sum((test_data$y - bagging_pred)^2) / sum((test_data$y - mean(test_data$y))^2)
  
  cat("\nBagging Ensemble (50 trees):\n")
  cat("  MSE:", round(bagging_mse, 4), "\n")
  cat("  R²:", round(bagging_r2, 4), "\n")
  cat("  MSE Improvement:", round((single_mse - bagging_mse) / single_mse * 100, 1), "%\n")
  
  return(list(single_tree = single_tree, bagging_model = bagging_model))
}

# Function to analyze ensemble size effect
analyze_ensemble_size_effect_r <- function() {
  # Load Boston data
  data(Boston, package = "MASS")
  
  # Split data
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  train_data <- Boston[train_indices, ]
  test_data <- Boston[-train_indices, ]
  
  ensemble_sizes <- c(1, 5, 10, 25, 50, 100)
  mse_scores <- numeric(length(ensemble_sizes))
  r2_scores <- numeric(length(ensemble_sizes))
  
  cat("=== ENSEMBLE SIZE ANALYSIS ===\n")
  
  for (i, n_trees in enumerate(ensemble_sizes)) {
    if (n_trees == 1) {
      # Single tree
      model <- rpart(medv ~ ., data = train_data, control = rpart.control(maxdepth = 10))
    } else {
      # Random Forest
      model <- randomForest(
        medv ~ .,
        data = train_data,
        ntree = n_trees,
        mtry = sqrt(ncol(train_data) - 1)
      )
    }
    
    predictions <- predict(model, test_data)
    mse <- mean((test_data$medv - predictions)^2)
    r2 <- 1 - sum((test_data$medv - predictions)^2) / sum((test_data$medv - mean(test_data$medv))^2)
    
    mse_scores[i] <- mse
    r2_scores[i] <- r2
    
    cat(sprintf("%3d trees: MSE = %.4f, R² = %.4f\n", n_trees, mse, r2))
  }
  
  # Visualize results
  par(mfrow = c(1, 2))
  
  plot(ensemble_sizes, mse_scores, type = "b", pch = 19, col = "blue",
       xlab = "Number of Trees", ylab = "Mean Squared Error",
       main = "MSE vs Ensemble Size")
  grid()
  
  plot(ensemble_sizes, r2_scores, type = "b", pch = 19, col = "red",
       xlab = "Number of Trees", ylab = "R² Score",
       main = "R² vs Ensemble Size")
  grid()
  
  return(list(ensemble_sizes = ensemble_sizes, mse_scores = mse_scores, r2_scores = r2_scores))
}

# Function to create partial dependence plots
partial_dependence_plot_r <- function(rf_model, data, feature_name) {
  # Generate feature values
  feature_values <- seq(min(data[[feature_name]]), max(data[[feature_name]]), length.out = 50)
  
  # Calculate partial dependence
  pd_values <- numeric(length(feature_values))
  
  for (i, val in enumerate(feature_values)) {
    # Create modified dataset
    data_temp <- data
    data_temp[[feature_name]] <- val
    
    # Make predictions
    predictions <- predict(rf_model, data_temp)
    pd_values[i] <- mean(predictions)
  }
  
  # Plot
  plot(feature_values, pd_values, type = "l", lwd = 2, col = "blue",
       xlab = feature_name, ylab = "Partial Dependence",
       main = paste("Partial Dependence Plot for", feature_name))
  grid()
  
  return(list(feature_values = feature_values, pd_values = pd_values))
}

# Function to predict with confidence intervals
predict_with_intervals_r <- function(rf_model, newdata, confidence = 0.95) {
  # Get predictions from all trees
  predictions <- predict(rf_model, newdata, predict.all = TRUE)
  
  # Calculate quantiles
  alpha <- 1 - confidence
  lower_quantile <- alpha / 2
  upper_quantile <- 1 - alpha / 2
  
  mean_pred <- rowMeans(predictions$individual)
  lower_bound <- apply(predictions$individual, 1, quantile, probs = lower_quantile)
  upper_bound <- apply(predictions$individual, 1, quantile, probs = upper_quantile)
  
  return(list(mean = mean_pred, lower = lower_bound, upper = upper_bound))
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate Random Forest
  cat("=== RANDOM FOREST DEMONSTRATION ===\n")
  rf_model_r <- demonstrate_random_forest_r()
  
  # Demonstrate bootstrap sampling
  cat("\n=== BOOTSTRAP SAMPLING DEMONSTRATION ===\n")
  bootstrap_data <- demonstrate_bootstrap_sampling_r()
  
  # Demonstrate bagging
  cat("\n=== BAGGING DEMONSTRATION ===\n")
  bagging_models <- demonstrate_bagging_r()
  
  # Analyze ensemble size effect
  cat("\n=== ENSEMBLE SIZE ANALYSIS ===\n")
  ensemble_analysis <- analyze_ensemble_size_effect_r()
  
  # Tune hyperparameters
  cat("\n=== HYPERPARAMETER TUNING ===\n")
  best_params_r <- tune_random_forest_r()
  
  # Create partial dependence plot
  data(Boston, package = "MASS")
  partial_dependence_plot_r(rf_model_r, Boston, "lstat")
}
