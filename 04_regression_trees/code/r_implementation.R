# Regression Tree Implementation in R
# ===================================
#
# This script demonstrates regression tree implementation using R,
# including building, pruning, and evaluation with comprehensive examples.

library(rpart)
library(rpart.plot)
library(ggplot2)
library(MASS)

# Function to build regression tree
build_regression_tree <- function(X, y, max_depth = NULL, min_samples_split = 2) {
  # Create data frame
  data <- data.frame(X, y = y)
  
  # Control parameters
  control <- rpart.control(
    maxdepth = max_depth,
    minsplit = min_samples_split,
    cp = 0  # No pruning during building
  )
  
  # Build tree
  tree <- rpart(y ~ ., data = data, control = control)
  
  return(tree)
}

# Function to find optimal alpha using cross-validation
find_optimal_alpha <- function(tree, X, y, cv_folds = 5) {
  # Get complexity parameter sequence
  cp_table <- tree$cptable
  
  # Cross-validation
  cv_errors <- numeric(nrow(cp_table))
  
  for (i in 1:nrow(cp_table)) {
    cp_val <- cp_table[i, "CP"]
    
    # Prune tree
    pruned_tree <- prune(tree, cp = cp_val)
    
    # Cross-validation error (simplified)
    cv_error <- 0
    for (fold in 1:cv_folds) {
      # Split data (simplified - in practice use proper CV)
      n <- nrow(X)
      test_indices <- sample(1:n, size = n %/% cv_folds)
      train_indices <- setdiff(1:n, test_indices)
      
      # Train on subset
      train_data <- data.frame(X[train_indices, ], y = y[train_indices])
      test_data <- data.frame(X[test_indices, ], y = y[test_indices])
      
      # Fit tree
      fold_tree <- rpart(y ~ ., data = train_data, control = rpart.control(cp = cp_val))
      
      # Predict and calculate error
      predictions <- predict(fold_tree, test_data)
      cv_error <- cv_error + mean((test_data$y - predictions)^2)
    }
    
    cv_errors[i] <- cv_error / cv_folds
  }
  
  # Find optimal CP
  optimal_idx <- which.min(cv_errors)
  optimal_cp <- cp_table[optimal_idx, "CP"]
  
  return(list(cp = optimal_cp, cv_errors = cv_errors, cp_table = cp_table))
}

# Example with Boston housing data
demonstrate_regression_tree_r <- function() {
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
  
  # Build full tree
  full_tree <- build_regression_tree(X_train, y_train, max_depth = 10)
  
  # Find optimal alpha
  optimal_result <- find_optimal_alpha(full_tree, X_train, y_train)
  
  # Prune tree
  optimal_tree <- prune(full_tree, cp = optimal_result$cp)
  
  # Make predictions
  predictions <- predict(optimal_tree, X_test)
  
  # Calculate metrics
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Test MSE:", round(mse, 4), "\n")
  cat("Test R²:", round(r2, 4), "\n")
  cat("Optimal CP:", round(optimal_result$cp, 6), "\n")
  
  # Visualize tree
  par(mfrow = c(1, 2))
  
  # Plot tree structure
  rpart.plot(optimal_tree, main = "Regression Tree Structure")
  
  # Plot predictions vs actual
  plot(y_test, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Regression Tree Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  # Plot cross-validation error
  plot(optimal_result$cp_table[, "CP"], optimal_result$cv_errors,
       type = "b", log = "x", xlab = "Complexity Parameter (CP)",
       ylab = "Cross-Validation Error",
       main = "Cross-Validation Error vs CP")
  abline(v = optimal_result$cp, col = "red", lty = 2)
  
  return(optimal_tree)
}

# Function to analyze tree performance
analyze_tree_performance_r <- function(tree, X, y) {
  predictions <- predict(tree, data.frame(X))
  residuals <- y - predictions
  
  # Performance metrics
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(residuals))
  r2 <- 1 - sum(residuals^2) / sum((y - mean(y))^2)
  
  cat("Performance Metrics:\n")
  cat("  MSE:", round(mse, 4), "\n")
  cat("  RMSE:", round(rmse, 4), "\n")
  cat("  MAE:", round(mae, 4), "\n")
  cat("  R²:", round(r2, 4), "\n")
  
  # Residual analysis plots
  par(mfrow = c(2, 2))
  
  # Residuals vs fitted
  plot(predictions, residuals, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Predicted Values", ylab = "Residuals",
       main = "Residuals vs Predicted")
  abline(h = 0, col = "red", lty = 2)
  
  # Residual histogram
  hist(residuals, main = "Residual Distribution", xlab = "Residuals",
       col = "lightblue", border = "black")
  
  # Q-Q plot
  qqnorm(residuals, main = "Q-Q Plot of Residuals")
  qqline(residuals, col = "red")
  
  # Actual vs Predicted
  plot(y, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Actual vs Predicted")
  abline(0, 1, col = "red", lty = 2)
  
  return(list(mse = mse, rmse = rmse, mae = mae, r2 = r2))
}

# Function to compare with linear model
compare_with_linear_model_r <- function(X_train, X_test, y_train, y_test) {
  # Fit linear model
  train_data <- data.frame(X_train, y = y_train)
  test_data <- data.frame(X_test, y = y_test)
  
  linear_model <- lm(y ~ ., data = train_data)
  y_pred_linear <- predict(linear_model, test_data)
  
  # Fit tree model
  tree_model <- rpart(y ~ ., data = train_data, control = rpart.control(maxdepth = 5, minsplit = 10))
  y_pred_tree <- predict(tree_model, test_data)
  
  # Compare performance
  mse_linear <- mean((y_test - y_pred_linear)^2)
  r2_linear <- 1 - sum((y_test - y_pred_linear)^2) / sum((y_test - mean(y_test))^2)
  
  mse_tree <- mean((y_test - y_pred_tree)^2)
  r2_tree <- 1 - sum((y_test - y_pred_tree)^2) / sum((y_test - mean(y_test))^2)
  
  cat("Model Comparison:\n")
  cat(sprintf("%-15s %-10s %-8s\n", "Model", "MSE", "R²"))
  cat(paste(rep("-", 35), collapse = ""), "\n")
  cat(sprintf("%-15s %-10.4f %-8.4f\n", "Linear", mse_linear, r2_linear))
  cat(sprintf("%-15s %-10.4f %-8.4f\n", "Tree", mse_tree, r2_tree))
  
  # Visualize comparison
  par(mfrow = c(1, 2))
  
  # Predictions comparison
  plot(y_test, y_pred_linear, pch = 19, col = "red", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Predictions Comparison")
  points(y_test, y_pred_tree, pch = 19, col = "blue", alpha = 0.6)
  abline(0, 1, col = "black", lty = 2)
  legend("topleft", legend = c("Linear", "Tree"), 
         col = c("red", "blue"), pch = 19)
  
  # Residuals comparison
  plot(y_test, y_test - y_pred_linear, pch = 19, col = "red", alpha = 0.6,
       xlab = "Actual Values", ylab = "Residuals",
       main = "Residuals Comparison")
  points(y_test, y_test - y_pred_tree, pch = 19, col = "blue", alpha = 0.6)
  abline(h = 0, col = "black", lty = 2)
  legend("topright", legend = c("Linear", "Tree"), 
         col = c("red", "blue"), pch = 19)
  
  return(list(linear_model = linear_model, tree_model = tree_model))
}

# Function to demonstrate tree building with synthetic data
demonstrate_synthetic_data <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 100
  X1 <- rnorm(n_samples)
  X2 <- rnorm(n_samples)
  X3 <- rnorm(n_samples)
  
  # Create response with some structure
  y <- 2 * X1 + 1.5 * X2 - 0.8 * X3 + rnorm(n_samples, 0, 0.5)
  
  # Create data frame
  data <- data.frame(X1 = X1, X2 = X2, X3 = X3, y = y)
  
  cat("=== SYNTHETIC DATA DEMONSTRATION ===\n")
  cat("Dataset:", n_samples, "samples, 3 features\n")
  cat("Target range:", round(range(y), 2), "\n")
  
  # Build tree
  tree <- rpart(y ~ ., data = data, control = rpart.control(maxdepth = 3, minsplit = 5))
  
  # Tree statistics
  cat("\nTree Statistics:\n")
  cat("  Number of nodes:", length(unique(tree$where)), "\n")
  cat("  Maximum depth:", max(rpart:::tree.depth(tree$frame$var)), "\n")
  
  # Print tree structure
  cat("\nTree Structure:\n")
  print(tree)
  
  # Make predictions
  predictions <- predict(tree, data)
  mse <- mean((y - predictions)^2)
  r2 <- 1 - sum((y - predictions)^2) / sum((y - mean(y))^2)
  
  cat("\nModel Performance:\n")
  cat("  MSE:", round(mse, 4), "\n")
  cat("  R²:", round(r2, 4), "\n")
  
  # Visualize tree
  par(mfrow = c(1, 2))
  rpart.plot(tree, main = "Synthetic Data Tree")
  
  # Predictions vs actual
  plot(y, predictions, pch = 19, col = "blue", alpha = 0.6,
       xlab = "Actual Values", ylab = "Predicted Values",
       main = "Synthetic Data Predictions")
  abline(0, 1, col = "red", lty = 2)
  
  return(tree)
}

# Function to demonstrate pruning sequence
demonstrate_pruning_sequence <- function() {
  # Load Boston data
  data(Boston, package = "MASS")
  
  # Prepare data
  X <- Boston[, -ncol(Boston)]
  y <- Boston$medv
  
  # Build full tree
  full_tree <- rpart(medv ~ ., data = Boston, control = rpart.control(maxdepth = 8, minsplit = 5))
  
  cat("=== PRUNING SEQUENCE DEMONSTRATION ===\n")
  cat("Full tree nodes:", length(unique(full_tree$where)), "\n")
  
  # Get complexity parameter table
  cp_table <- full_tree$cptable
  
  cat("\nPruning sequence:\n")
  cat("  Number of trees:", nrow(cp_table), "\n")
  cat("  CP range:", round(range(cp_table[, "CP"]), 6), "\n")
  
  # Evaluate each tree
  cat("\nTree evaluation:\n")
  cat(sprintf("%-12s %-8s %-10s %-8s\n", "CP", "Nodes", "MSE", "R²"))
  cat(paste(rep("-", 40), collapse = ""), "\n")
  
  for (i in 1:nrow(cp_table)) {
    cp_val <- cp_table[i, "CP"]
    pruned_tree <- prune(full_tree, cp = cp_val)
    predictions <- predict(pruned_tree, Boston)
    mse <- mean((y - predictions)^2)
    r2 <- 1 - sum((y - predictions)^2) / sum((y - mean(y))^2)
    n_nodes <- length(unique(pruned_tree$where))
    
    cat(sprintf("%-12.6f %-8d %-10.4f %-8.4f\n", cp_val, n_nodes, mse, r2))
  }
  
  # Visualize pruning sequence
  par(mfrow = c(2, 2))
  
  # CP vs nodes
  plot(cp_table[, "CP"], cp_table[, "nsplit"] + 1, type = "b", log = "x",
       xlab = "Complexity Parameter (CP)", ylab = "Number of Nodes",
       main = "Tree Size vs CP")
  grid()
  
  # CP vs relative error
  plot(cp_table[, "CP"], cp_table[, "rel error"], type = "b", log = "x",
       xlab = "Complexity Parameter (CP)", ylab = "Relative Error",
       main = "Relative Error vs CP")
  grid()
  
  # CP vs cross-validation error
  plot(cp_table[, "CP"], cp_table[, "xerror"], type = "b", log = "x",
       xlab = "Complexity Parameter (CP)", ylab = "Cross-Validation Error",
       main = "CV Error vs CP")
  grid()
  
  # Nodes vs R²
  nodes <- cp_table[, "nsplit"] + 1
  r2_values <- 1 - cp_table[, "rel error"]
  plot(nodes, r2_values, type = "b",
       xlab = "Number of Nodes", ylab = "R²",
       main = "R² vs Tree Size")
  grid()
  
  return(full_tree)
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate with Boston housing data
  cat("=== BOSTON HOUSING DEMONSTRATION ===\n")
  tree_boston <- demonstrate_regression_tree_r()
  
  # Analyze performance
  data(Boston, package = "MASS")
  X <- Boston[, -ncol(Boston)]
  y <- Boston$medv
  performance <- analyze_tree_performance_r(tree_boston, X, y)
  
  # Compare with linear model
  set.seed(42)
  train_indices <- sample(1:nrow(Boston), size = 0.8 * nrow(Boston))
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  comparison <- compare_with_linear_model_r(X_train, X_test, y_train, y_test)
  
  # Demonstrate with synthetic data
  cat("\n=== SYNTHETIC DATA DEMONSTRATION ===\n")
  tree_synthetic <- demonstrate_synthetic_data()
  
  # Demonstrate pruning sequence
  cat("\n=== PRUNING SEQUENCE DEMONSTRATION ===\n")
  tree_pruning <- demonstrate_pruning_sequence()
}
