# Cubic Spline Implementation in R
# ================================
#
# This script provides a complete implementation of cubic spline regression
# including regular and natural cubic splines, basis functions, and comprehensive demonstrations.

library(splines)
library(ggplot2)
library(dplyr)

# Function to create truncated power basis
create_truncated_power_basis <- function(X, knots) {
  n_samples <- length(X)
  n_knots <- length(knots)
  
  # Basis matrix: [1, x, x^2, x^3, (x-xi_1)_+^3, ..., (x-xi_m)_+^3]
  basis_matrix <- matrix(0, nrow = n_samples, ncol = n_knots + 4)
  
  # Polynomial terms
  basis_matrix[, 1] <- 1
  basis_matrix[, 2] <- X
  basis_matrix[, 3] <- X^2
  basis_matrix[, 4] <- X^3
  
  # Truncated power terms
  for (i in 1:n_knots) {
    basis_matrix[, i + 4] <- pmax(0, X - knots[i])^3
  }
  
  return(basis_matrix)
}

# Function to fit cubic spline regression
fit_cubic_spline <- function(X, y, knots, natural = FALSE) {
  if (natural) {
    # Use natural cubic splines
    spline_model <- lm(y ~ ns(X, knots = knots))
  } else {
    # Use regular cubic splines
    spline_model <- lm(y ~ bs(X, knots = knots, degree = 3))
  }
  
  return(spline_model)
}

# Function to demonstrate cubic splines
demonstrate_cubic_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Define knots
  knots <- c(2, 4, 6, 8)
  
  # Fit different types of splines
  splines <- list()
  
  # Regular cubic spline using B-splines
  splines$Regular <- lm(y ~ bs(X, knots = knots, degree = 3))
  
  # Natural cubic spline
  splines$Natural <- lm(y ~ ns(X, knots = knots))
  
  # Smoothing spline
  splines$Smoothing <- smooth.spline(X, y, cv = TRUE)
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    geom_vline(xintercept = knots, linetype = "dotted", color = "gray", alpha = 0.7) +
    labs(title = "Cubic Spline Fits", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X_plot)$y
    } else {
      y_pred <- predict(splines[[name]], data.frame(X = X_plot))
    }
    
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Basis functions plot
  basis_matrix <- create_truncated_power_basis(X_plot, knots)
  basis_df <- data.frame(
    X = rep(X_plot, ncol(basis_matrix)),
    Basis = rep(paste("Basis", 1:ncol(basis_matrix)), each = length(X_plot)),
    Value = as.vector(basis_matrix)
  )
  
  p2 <- ggplot(basis_df, aes(X, Value, color = Basis)) +
    geom_line() +
    labs(title = "Truncated Power Basis Functions", x = "X", y = "Basis Function Value") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Residuals plot
  residuals_df <- data.frame(
    Predicted = numeric(0),
    Residuals = numeric(0),
    Type = character(0)
  )
  
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X)$y
    } else {
      y_pred <- predict(splines[[name]])
    }
    
    residuals_df <- rbind(residuals_df, data.frame(
      Predicted = y_pred,
      Residuals = y - y_pred,
      Type = name
    ))
  }
  
  p3 <- ggplot(residuals_df, aes(Predicted, Residuals, color = Type)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals", x = "Predicted Values", y = "Residuals") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  # Model comparison
  cat("Model Comparison:\n")
  for (name in names(splines)) {
    if (name == "Smoothing") {
      y_pred <- predict(splines[[name]], X)$y
    } else {
      y_pred <- predict(splines[[name]])
    }
    
    mse <- mean((y - y_pred)^2)
    r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
    
    cat(sprintf("%s Spline: MSE = %.4f, R² = %.4f\n", name, mse, r2))
  }
  
  return(splines)
}

# Function to demonstrate advanced spline features
demonstrate_advanced_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Different knot configurations
  knot_configs <- list(
    "Few knots" = c(3, 7),
    "Many knots" = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
    "Optimal knots" = c(2, 4, 6, 8)
  )
  
  # Compare different spline types
  spline_comparison <- list()
  
  # Regular cubic splines with different knots
  for (name in names(knot_configs)) {
    spline_comparison[[name]] <- lm(y ~ bs(X, knots = knot_configs[[name]], degree = 3))
  }
  
  # Natural cubic splines
  spline_comparison$"Natural" <- lm(y ~ ns(X, knots = c(2, 4, 6, 8)))
  
  # Smoothing spline
  spline_comparison$"Smoothing" <- smooth.spline(X, y, cv = TRUE)
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create comparison plot
  p_comparison <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Spline Type Comparison", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for each spline type
  for (name in names(spline_comparison)) {
    if (name == "Smoothing") {
      y_pred <- predict(spline_comparison[[name]], X_plot)$y
    } else {
      y_pred <- predict(spline_comparison[[name]], data.frame(X = X_plot))
    }
    
    p_comparison <- p_comparison + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                                            aes(X, y), color = name, size = 1)
  }
  
  print(p_comparison)
  
  # Cross-validation for knot selection
  n_knots_range <- 2:10
  cv_scores <- numeric(length(n_knots_range))
  
  for (i, n_knots in enumerate(n_knots_range)) {
    # Create knots
    knots <- quantile(X, probs = seq(0, 1, length.out = n_knots + 2))[2:(n_knots + 1)]
    
    # Create basis matrix
    basis_matrix <- create_truncated_power_basis(X, knots)
    
    # Simple cross-validation (5-fold)
    cv_scores_temp <- numeric(5)
    fold_size <- length(X) %/% 5
    
    for (fold in 1:5) {
      # Create fold indices
      start_idx <- (fold - 1) * fold_size + 1
      end_idx <- ifelse(fold == 5, length(X), fold * fold_size)
      test_indices <- start_idx:end_idx
      train_indices <- setdiff(1:length(X), test_indices)
      
      # Split data
      X_train <- X[train_indices]
      y_train <- y[train_indices]
      X_test <- X[test_indices]
      y_test <- y[test_indices]
      
      # Fit model
      basis_train <- create_truncated_power_basis(X_train, knots)
      model <- lm(y_train ~ basis_train - 1)
      
      # Predict
      basis_test <- create_truncated_power_basis(X_test, knots)
      y_pred <- predict(model, data.frame(basis_test))
      
      # Calculate MSE
      cv_scores_temp[fold] <- mean((y_test - y_pred)^2)
    }
    
    cv_scores[i] <- mean(cv_scores_temp)
  }
  
  # Plot CV scores
  cv_df <- data.frame(NumKnots = n_knots_range, CV_MSE = cv_scores)
  p_cv <- ggplot(cv_df, aes(NumKnots, CV_MSE)) +
    geom_line() + geom_point() +
    labs(title = "Cross-Validation for Knot Selection", 
         x = "Number of Knots", y = "CV MSE") +
    theme_minimal()
  
  print(p_cv)
  
  # Find optimal number of knots
  optimal_n_knots <- n_knots_range[which.min(cv_scores)]
  cat("Optimal number of knots:", optimal_n_knots, "\n")
  
  return(list(
    spline_comparison = spline_comparison,
    cv_scores = cv_scores,
    optimal_n_knots = optimal_n_knots
  ))
}

# Function to analyze spline diagnostics
analyze_spline_diagnostics_r <- function(spline_model, X, y) {
  # Get predictions
  if (inherits(spline_model, "smooth.spline")) {
    y_pred <- predict(spline_model, X)$y
  } else {
    y_pred <- predict(spline_model)
  }
  
  residuals <- y - y_pred
  
  # Create diagnostic plots
  par(mfrow = c(2, 2))
  
  # Residuals vs Fitted
  plot(y_pred, residuals, pch = 19, alpha = 0.6,
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

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic cubic splines
  cat("=== BASIC CUBIC SPLINES DEMONSTRATION ===\n")
  spline_models_r <- demonstrate_cubic_splines_r()
  
  # Demonstrate advanced features
  cat("\n=== ADVANCED SPLINE FEATURES ===\n")
  advanced_results <- demonstrate_advanced_splines_r()
  
  # Demonstrate diagnostics
  cat("\n=== SPLINE DIAGNOSTICS ===\n")
  # Use the natural spline for diagnostics
  natural_spline <- lm(y ~ ns(X, knots = c(2, 4, 6, 8)))
  residuals <- analyze_spline_diagnostics_r(natural_spline, X, y)
}
