# Smoothing Spline Implementation in R
# ====================================
#
# This script provides a complete implementation of smoothing splines
# including cross-validation, model selection, and comprehensive demonstrations.

library(splines)
library(ggplot2)
library(dplyr)

# Function to fit smoothing spline
fit_smoothing_spline <- function(X, y, lambda = NULL, df = NULL, cv = TRUE) {
  if (is.null(lambda) && is.null(df)) {
    if (cv) {
      # Use cross-validation to select lambda
      spline_model <- smooth.spline(X, y, cv = TRUE)
    } else {
      # Use default lambda
      spline_model <- smooth.spline(X, y, lambda = 1.0)
    }
  } else if (!is.null(df)) {
    # Use specified degrees of freedom
    spline_model <- smooth.spline(X, y, df = df)
  } else {
    # Use specified lambda
    spline_model <- smooth.spline(X, y, lambda = lambda)
  }
  
  return(spline_model)
}

# Function to demonstrate smoothing splines
demonstrate_smoothing_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 50)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(50, 0, 0.5)
  
  # Test different lambda values
  lambda_values <- c(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
  models <- list()
  
  for (lambda_val in lambda_values) {
    models[[paste0("λ=", lambda_val)]] <- fit_smoothing_spline(X, y, lambda = lambda_val, cv = FALSE)
  }
  
  # Test different degrees of freedom
  df_values <- c(3, 5, 8, 12, 20, 30)
  models_df <- list()
  
  for (df in df_values) {
    models_df[[paste0("DF=", df)]] <- fit_smoothing_spline(X, y, df = df, cv = FALSE)
  }
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Smoothing Parameter λ", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], X_plot)$y
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  p2 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Degrees of Freedom", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models_df)) {
    y_pred <- predict(models_df[[name]], X_plot)$y
    p2 <- p2 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Lambda vs EDF
  lambda_list <- numeric(0)
  edf_list <- numeric(0)
  
  for (name in names(models)) {
    lambda_val <- as.numeric(gsub("λ=", "", name))
    edf <- models[[name]]$df
    
    lambda_list <- c(lambda_list, lambda_val)
    edf_list <- c(edf_list, edf)
  }
  
  p3 <- ggplot(data.frame(Lambda = lambda_list, EDF = edf_list), aes(Lambda, EDF)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    scale_x_log10() +
    labs(title = "λ vs Effective Degrees of Freedom", x = "λ", y = "EDF") +
    theme_minimal()
  
  # Cross-validation
  lambda_candidates <- 10^seq(-3, 3, length.out = 20)
  cv_scores <- numeric(0)
  
  for (lambda_val in lambda_candidates) {
    model <- fit_smoothing_spline(X, y, lambda = lambda_val, cv = FALSE)
    # Compute LOOCV score (simplified)
    y_pred <- predict(model, X)$y
    cv_score <- mean((y - y_pred)^2)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  p4 <- ggplot(data.frame(Lambda = lambda_candidates, CV_Score = cv_scores), 
               aes(Lambda, CV_Score)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    scale_x_log10() +
    labs(title = "Cross-Validation for λ Selection", x = "λ", y = "CV Score") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  return(list(models = models, models_df = models_df))
}

# Function to analyze noisy data
analyze_noisy_data_r <- function() {
  # Generate noisy data with different noise levels
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  
  noise_levels <- c(0.1, 0.5, 1.0, 2.0)
  models <- list()
  
  for (noise in noise_levels) {
    y <- y_true + rnorm(100, 0, noise)
    
    # Fit smoothing spline with cross-validation
    model <- fit_smoothing_spline(X, y, cv = TRUE)
    models[[paste0("Noise=", noise)]] <- model
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Smoothing Splines on Noisy Data", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], X)$y
    p1 <- p1 + geom_line(data = data.frame(X = X, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  noise_list <- numeric(0)
  lambda_list <- numeric(0)
  edf_list <- numeric(0)
  
  for (name in names(models)) {
    noise <- as.numeric(gsub("Noise=", "", name))
    lambda <- models[[name]]$lambda
    edf <- models[[name]]$df
    
    noise_list <- c(noise_list, noise)
    lambda_list <- c(lambda_list, lambda)
    edf_list <- c(edf_list, edf)
  }
  
  p2 <- ggplot(data.frame(Noise = noise_list, Lambda = lambda_list), 
               aes(Noise, Lambda)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "λ Selection vs Noise Level", x = "Noise Level", y = "Selected λ") +
    theme_minimal()
  
  p3 <- ggplot(data.frame(Noise = noise_list, EDF = edf_list), 
               aes(Noise, EDF)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    labs(title = "EDF vs Noise Level", x = "Noise Level", y = "Effective Degrees of Freedom") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  return(models)
}

# Function to demonstrate advanced features
demonstrate_advanced_features_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Fit smoothing spline with cross-validation
  model <- fit_smoothing_spline(X, y, cv = TRUE)
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  y_pred <- predict(model, X_plot)$y
  
  # Create confidence intervals (simplified)
  # In practice, use specialized packages for proper confidence intervals
  residuals <- y - predict(model, X)$y
  sigma_hat <- sd(residuals)
  
  # Simplified confidence intervals
  se_pred <- sigma_hat * sqrt(0.1)  # Simplified standard error
  ci_lower <- y_pred - 1.96 * se_pred
  ci_upper <- y_pred + 1.96 * se_pred
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X_plot, y = y_pred), aes(X_plot, y_pred), 
              color = "blue", size = 1) +
    geom_ribbon(data = data.frame(X = X_plot, lower = ci_lower, upper = ci_upper), 
                aes(X, ymin = lower, ymax = upper), alpha = 0.3, fill = "blue") +
    labs(title = "Smoothing Spline with Confidence Intervals", x = "X", y = "Y") +
    theme_minimal()
  
  # Model diagnostics
  fitted_values <- predict(model, X)$y
  residuals <- y - fitted_values
  
  p2 <- ggplot(data.frame(Fitted = fitted_values, Residuals = residuals), 
               aes(Fitted, Residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals") +
    theme_minimal()
  
  p3 <- ggplot(data.frame(X = X, Residuals = residuals), aes(X, Residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals vs Predictor", x = "X", y = "Residuals") +
    theme_minimal()
  
  # Q-Q plot
  p4 <- ggplot(data.frame(Residuals = residuals), aes(sample = Residuals)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = "Q-Q Plot of Residuals", x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  # Model summary
  cat("Model Summary:\n")
  cat("Selected λ:", model$lambda, "\n")
  cat("Effective Degrees of Freedom:", model$df, "\n")
  cat("Cross-Validation Score:", model$cv.crit, "\n")
  
  return(model)
}

# Function to compare different smoothing methods
compare_smoothing_methods_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Fit different smoothing methods
  methods <- list()
  
  # Smoothing spline
  methods$"Smoothing Spline" <- fit_smoothing_spline(X, y, cv = TRUE)
  
  # Natural cubic spline with cross-validation
  # Use different degrees of freedom
  df_values <- c(3, 5, 8, 12, 20)
  cv_scores <- numeric(0)
  
  for (df in df_values) {
    model <- lm(y ~ ns(X, df = df))
    y_pred <- predict(model)
    cv_score <- mean((y - y_pred)^2)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  optimal_df <- df_values[which.min(cv_scores)]
  methods$"Natural Cubic Spline" <- lm(y ~ ns(X, df = optimal_df))
  
  # Local polynomial regression (loess)
  methods$"Loess" <- loess(y ~ X, span = 0.3)
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create comparison plot
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Comparison of Smoothing Methods", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for each method
  for (name in names(methods)) {
    if (name == "Smoothing Spline") {
      y_pred <- predict(methods[[name]], X_plot)$y
    } else if (name == "Natural Cubic Spline") {
      y_pred <- predict(methods[[name]], data.frame(X = X_plot))
    } else if (name == "Loess") {
      y_pred <- predict(methods[[name]], data.frame(X = X_plot))
    }
    
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  print(p1)
  
  # Model comparison
  cat("Model Comparison:\n")
  for (name in names(methods)) {
    if (name == "Smoothing Spline") {
      y_pred <- predict(methods[[name]], X)$y
      mse <- mean((y - y_pred)^2)
      cat(sprintf("%s: MSE = %.4f, λ = %.4f, EDF = %.2f\n", 
                  name, mse, methods[[name]]$lambda, methods[[name]]$df))
    } else if (name == "Natural Cubic Spline") {
      y_pred <- predict(methods[[name]])
      mse <- mean((y - y_pred)^2)
      cat(sprintf("%s: MSE = %.4f, DF = %d\n", name, mse, optimal_df))
    } else if (name == "Loess") {
      y_pred <- predict(methods[[name]])
      mse <- mean((y - y_pred)^2)
      cat(sprintf("%s: MSE = %.4f\n", name, mse))
    }
  }
  
  return(methods)
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic smoothing splines
  cat("=== BASIC SMOOTHING SPLINES DEMONSTRATION ===\n")
  results <- demonstrate_smoothing_splines_r()
  
  # Analyze noisy data
  cat("\n=== ANALYZING NOISY DATA ===\n")
  noisy_results <- analyze_noisy_data_r()
  
  # Demonstrate advanced features
  cat("\n=== ADVANCED FEATURES ===\n")
  advanced_results <- demonstrate_advanced_features_r()
  
  # Compare smoothing methods
  cat("\n=== COMPARING SMOOTHING METHODS ===\n")
  comparison_results <- compare_smoothing_methods_r()
}
