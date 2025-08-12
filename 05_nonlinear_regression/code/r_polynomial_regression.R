# Polynomial Regression Implementation in R
# ========================================
#
# This script provides a complete implementation of polynomial regression
# including model fitting, evaluation, and visualization.

library(ggplot2)
library(dplyr)

# Function to create polynomial features
create_polynomial_features <- function(X, degree) {
  X_poly <- matrix(1, nrow = length(X), ncol = degree + 1)
  for (d in 1:degree) {
    X_poly[, d + 1] <- X^d
  }
  return(X_poly)
}

# Function to fit polynomial regression
fit_polynomial_regression <- function(X, y, degree) {
  X_poly <- create_polynomial_features(X, degree)
  
  # Fit linear regression
  model <- lm(y ~ X_poly - 1)  # -1 removes intercept since we include it in X_poly
  
  return(list(
    model = model,
    coefficients = coef(model),
    fitted_values = fitted(model),
    residuals = residuals(model)
  ))
}

# Function to calculate model metrics
calculate_polynomial_metrics <- function(model, X, y, degree) {
  y_pred <- fitted(model)
  
  # Basic metrics
  mse <- mean((y - y_pred)^2)
  rmse <- sqrt(mse)
  r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
  
  # Adjusted R-squared
  n <- length(y)
  p <- degree + 1
  adj_r2 <- 1 - (1 - r2) * (n - 1) / (n - p - 1)
  
  # AIC and BIC
  rss <- sum((y - y_pred)^2)
  aic <- n * log(rss/n) + 2 * p
  bic <- n * log(rss/n) + p * log(n)
  
  return(list(
    MSE = mse,
    RMSE = rmse,
    R2 = r2,
    Adjusted_R2 = adj_r2,
    AIC = aic,
    BIC = bic
  ))
}

# Function to demonstrate polynomial regression
demonstrate_polynomial_regression_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(-3, 3, length.out = 100)
  y_true <- 2 + 3*X - 0.5*X^2 + 0.1*X^3
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Test different polynomial degrees
  degrees <- 1:6
  models <- list()
  metrics <- list()
  
  for (degree in degrees) {
    # Fit model
    model_result <- fit_polynomial_regression(X, y, degree)
    models[[degree]] <- model_result
    
    # Calculate metrics
    metrics[[degree]] <- calculate_polynomial_metrics(
      model_result$model, X, y, degree
    )
    
    cat("Degree", degree, ":\n")
    cat("  R²:", round(metrics[[degree]]$R2, 4), "\n")
    cat("  AIC:", round(metrics[[degree]]$AIC, 4), "\n")
    cat("  BIC:", round(metrics[[degree]]$BIC, 4), "\n\n")
  }
  
  # Create visualization
  X_plot <- seq(-3, 3, length.out = 200)
  
  # Data frame for plotting
  plot_data <- data.frame(
    X = rep(X_plot, length(degrees)),
    Degree = rep(degrees, each = length(X_plot)),
    Y = NA
  )
  
  # Calculate predictions for each degree
  for (degree in degrees) {
    X_poly_plot <- create_polynomial_features(X_plot, degree)
    y_plot <- X_poly_plot %*% models[[degree]]$coefficients
    plot_data$Y[plot_data$Degree == degree] <- y_plot
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = plot_data[plot_data$Degree %in% c(1, 2, 3), ], 
              aes(X, Y, color = factor(Degree))) +
    labs(title = "Polynomial Fits", x = "X", y = "Y", color = "Degree") +
    theme_minimal()
  
  # Metrics plots
  metrics_df <- data.frame(
    Degree = degrees,
    R2 = sapply(metrics, function(m) m$R2),
    AIC = sapply(metrics, function(m) m$AIC),
    BIC = sapply(metrics, function(m) m$BIC)
  )
  
  p2 <- ggplot(metrics_df, aes(Degree, R2)) +
    geom_line() + geom_point() +
    labs(title = "R² vs Degree") +
    theme_minimal()
  
  p3 <- ggplot(metrics_df, aes(Degree, AIC)) +
    geom_line() + geom_point() +
    labs(title = "AIC vs Degree") +
    theme_minimal()
  
  p4 <- ggplot(metrics_df, aes(Degree, BIC)) +
    geom_line() + geom_point() +
    labs(title = "BIC vs Degree") +
    theme_minimal()
  
  # Residuals plot
  residuals_df <- data.frame(
    Predicted = models[[3]]$fitted_values,
    Residuals = models[[3]]$residuals
  )
  
  p5 <- ggplot(residuals_df, aes(Predicted, Residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", lty = 2) +
    labs(title = "Residuals (Degree 3)") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  print(p5)
  
  return(list(models = models, metrics = metrics))
}

# Function to perform cross-validation for degree selection
cross_validate_polynomial_degree_r <- function(X, y, max_degree = 8, cv_folds = 5) {
  n <- length(X)
  fold_size <- floor(n / cv_folds)
  cv_scores <- numeric(max_degree)
  
  for (degree in 1:max_degree) {
    fold_scores <- numeric(cv_folds)
    
    for (fold in 1:cv_folds) {
      # Create fold indices
      start_idx <- (fold - 1) * fold_size + 1
      end_idx <- ifelse(fold == cv_folds, n, fold * fold_size)
      test_indices <- start_idx:end_idx
      train_indices <- setdiff(1:n, test_indices)
      
      # Split data
      X_train <- X[train_indices]
      y_train <- y[train_indices]
      X_test <- X[test_indices]
      y_test <- y[test_indices]
      
      # Fit model
      model_result <- fit_polynomial_regression(X_train, y_train, degree)
      
      # Make predictions
      X_test_poly <- create_polynomial_features(X_test, degree)
      y_pred <- X_test_poly %*% model_result$coefficients
      
      # Calculate MSE
      fold_scores[fold] <- mean((y_test - y_pred)^2)
    }
    
    cv_scores[degree] <- mean(fold_scores)
  }
  
  # Find optimal degree
  optimal_degree <- which.min(cv_scores)
  
  # Plot CV scores
  cv_df <- data.frame(Degree = 1:max_degree, CV_MSE = cv_scores)
  p_cv <- ggplot(cv_df, aes(Degree, CV_MSE)) +
    geom_line() + geom_point() +
    geom_vline(xintercept = optimal_degree, color = "red", lty = 2) +
    labs(title = paste("Cross-Validation MSE (Optimal degree:", optimal_degree, ")"),
         x = "Polynomial Degree", y = "CV MSE") +
    theme_minimal()
  
  print(p_cv)
  
  return(list(optimal_degree = optimal_degree, cv_scores = cv_scores))
}

# Function to analyze residuals
analyze_polynomial_residuals_r <- function(model, X, y) {
  residuals <- model$residuals
  fitted_values <- model$fitted_values
  
  # Create diagnostic plots
  par(mfrow = c(2, 2))
  
  # Residuals vs Fitted
  plot(fitted_values, residuals, pch = 19, alpha = 0.6,
       xlab = "Fitted Values", ylab = "Residuals",
       main = "Residuals vs Fitted")
  abline(h = 0, col = "red", lty = 2)
  
  # Q-Q Plot
  qqnorm(residuals, main = "Q-Q Plot of Residuals")
  qqline(residuals, col = "red")
  
  # Residuals vs Predictor
  plot(X, residuals, pch = 19, alpha = 0.6,
       xlab = "X", ylab = "Residuals",
       main = "Residuals vs X")
  abline(h = 0, col = "red", lty = 2)
  
  # Histogram of residuals
  hist(residuals, main = "Histogram of Residuals",
       xlab = "Residuals", ylab = "Frequency")
  
  # Statistical tests
  cat("Shapiro-Wilk test for normality:\n")
  shapiro_test <- shapiro.test(residuals)
  print(shapiro_test)
  
  return(residuals)
}

# Function to demonstrate forward selection
forward_polynomial_selection_r <- function(X, y, max_degree = 8, criterion = "aic") {
  scores <- numeric(max_degree)
  
  for (degree in 1:max_degree) {
    model_result <- fit_polynomial_regression(X, y, degree)
    metrics <- calculate_polynomial_metrics(model_result$model, X, y, degree)
    
    if (criterion == "aic") {
      scores[degree] <- metrics$AIC
    } else if (criterion == "bic") {
      scores[degree] <- metrics$BIC
    }
  }
  
  best_degree <- which.min(scores)
  
  # Plot scores
  scores_df <- data.frame(Degree = 1:max_degree, Score = scores)
  p_scores <- ggplot(scores_df, aes(Degree, Score)) +
    geom_line() + geom_point() +
    geom_vline(xintercept = best_degree, color = "red", lty = 2) +
    labs(title = paste("Forward Selection (", toupper(criterion), ") - Best degree:", best_degree),
         x = "Polynomial Degree", y = toupper(criterion)) +
    theme_minimal()
  
  print(p_scores)
  
  return(list(best_degree = best_degree, scores = scores))
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate polynomial regression
  cat("=== POLYNOMIAL REGRESSION DEMONSTRATION ===\n")
  results_r <- demonstrate_polynomial_regression_r()
  
  # Demonstrate cross-validation
  cat("\n=== CROSS-VALIDATION FOR DEGREE SELECTION ===\n")
  set.seed(42)
  X <- seq(-3, 3, length.out = 100)
  y_true <- 2 + 3*X - 0.5*X^2 + 0.1*X^3
  y <- y_true + rnorm(100, 0, 0.5)
  
  cv_results <- cross_validate_polynomial_degree_r(X, y, max_degree = 8)
  cat("Optimal degree from CV:", cv_results$optimal_degree, "\n")
  
  # Demonstrate forward selection
  cat("\n=== FORWARD SELECTION ===\n")
  forward_aic <- forward_polynomial_selection_r(X, y, max_degree = 8, criterion = "aic")
  cat("Best degree (AIC):", forward_aic$best_degree, "\n")
  
  forward_bic <- forward_polynomial_selection_r(X, y, max_degree = 8, criterion = "bic")
  cat("Best degree (BIC):", forward_bic$best_degree, "\n")
  
  # Demonstrate residual analysis
  cat("\n=== RESIDUAL ANALYSIS ===\n")
  best_model <- fit_polynomial_regression(X, y, cv_results$optimal_degree)
  residuals <- analyze_polynomial_residuals_r(best_model, X, y)
}
