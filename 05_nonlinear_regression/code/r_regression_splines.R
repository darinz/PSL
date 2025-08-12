# Regression Spline Implementation in R
# =====================================
#
# This script provides a complete implementation of regression splines
# including cross-validation, model selection, and comprehensive demonstrations.

library(splines)
library(ggplot2)
library(dplyr)
library(caret)

# Function to fit regression spline with specified degrees of freedom
fit_regression_spline <- function(X, y, df, spline_type = "cubic") {
  if (spline_type == "cubic") {
    # Use B-splines for cubic splines
    model <- lm(y ~ bs(X, df = df))
  } else {
    # Use natural cubic splines
    model <- lm(y ~ ns(X, df = df))
  }
  return(model)
}

# Function to demonstrate regression splines
demonstrate_regression_splines_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Test different degrees of freedom
  df_values <- c(4, 6, 8, 10, 12, 15)
  models <- list()
  
  for (df in df_values) {
    models[[paste0("DF=", df)]] <- fit_regression_spline(X, y, df, "cubic")
  }
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Regression Spline Fits", x = "X", y = "Y") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], data.frame(X = X_plot))
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  df_list <- numeric(0)
  mse_list <- numeric(0)
  r2_list <- numeric(0)
  
  for (name in names(models)) {
    y_pred <- predict(models[[name]])
    mse <- mean((y - y_pred)^2)
    r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
    
    df_val <- as.numeric(gsub("DF=", "", name))
    df_list <- c(df_list, df_val)
    mse_list <- c(mse_list, mse)
    r2_list <- c(r2_list, r2)
  }
  
  p2 <- ggplot(data.frame(DF = df_list, MSE = mse_list), aes(DF, MSE)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "Model Performance vs DF", x = "Degrees of Freedom", y = "Mean Squared Error") +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
  
  # Cross-validation
  cv_scores <- numeric(0)
  for (df in df_values) {
    cv_score <- mean(cv.glm(data.frame(X = X, y = y), 
                           glm(y ~ bs(X, df = df)), K = 5)$delta)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  p3 <- ggplot(data.frame(DF = df_values, CV_MSE = cv_scores), aes(DF, CV_MSE)) +
    geom_line(color = "red") +
    geom_point(color = "red") +
    labs(title = "Cross-Validation Performance", x = "Degrees of Freedom", y = "CV MSE") +
    theme_minimal() +
    theme(panel.grid.minor = element_blank())
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  # Model comparison table
  cat("Model Comparison:\n")
  comparison_df <- data.frame(
    DF = df_list,
    MSE = mse_list,
    R2 = r2_list,
    CV_MSE = cv_scores
  )
  print(comparison_df)
  
  return(models)
}

# Function to analyze birthrate data
analyze_birthrate_data_r <- function() {
  # Generate birthrate-like data (simulated)
  set.seed(42)
  years <- 1960:2019
  birthrate <- 20 - 0.1*(years - 1960) + 2*sin(2*pi*(years - 1960)/20) + rnorm(length(years), 0, 0.5)
  
  # Test different degrees of freedom
  df_values <- c(3, 5, 7, 10, 15, 20)
  models <- list()
  
  for (df in df_values) {
    models[[paste0("DF=", df)]] <- fit_regression_spline(years, birthrate, df, "natural")
  }
  
  # Cross-validation to select optimal df
  cv_scores <- numeric(0)
  for (df in df_values) {
    cv_score <- mean(cv.glm(data.frame(years = years, birthrate = birthrate), 
                           glm(birthrate ~ ns(years, df = df)), K = 5)$delta)
    cv_scores <- c(cv_scores, cv_score)
  }
  
  optimal_df <- df_values[which.min(cv_scores)]
  cat(sprintf("Optimal degrees of freedom: %d\n", optimal_df))
  
  # Create plots
  years_plot <- seq(1960, 2020, length.out = 200)
  
  p1 <- ggplot() +
    geom_point(data = data.frame(years = years, birthrate = birthrate), 
               aes(years, birthrate), alpha = 0.7) +
    labs(title = "Birthrate Data: Spline Fits", x = "Year", y = "Birthrate") +
    theme_minimal()
  
  # Add spline predictions
  for (name in names(models)) {
    y_pred <- predict(models[[name]], data.frame(years = years_plot))
    p1 <- p1 + geom_line(data = data.frame(years = years_plot, y = y_pred), 
                         aes(years, y), color = name, size = 1)
  }
  
  p2 <- ggplot(data.frame(DF = df_values, CV_MSE = cv_scores), aes(DF, CV_MSE)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    geom_vline(xintercept = optimal_df, color = "red", linestyle = "dashed") +
    labs(title = "Model Selection", x = "Degrees of Freedom", y = "Cross-Validation MSE") +
    theme_minimal()
  
  # Residuals analysis
  best_model <- models[[paste0("DF=", optimal_df)]]
  y_pred <- predict(best_model)
  residuals <- birthrate - y_pred
  
  p3 <- ggplot(data.frame(years = years, residuals = residuals), aes(years, residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
    labs(title = "Residuals", x = "Year", y = "Residuals") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  
  return(list(models = models, optimal_df = optimal_df))
}

# Function to demonstrate advanced features
demonstrate_advanced_features_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Test different spline types
  df <- 8
  
  # Cubic spline
  cubic_model <- fit_regression_spline(X, y, df, "cubic")
  
  # Natural spline
  natural_model <- fit_regression_spline(X, y, df, "natural")
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create comparison plot
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Spline Type Comparison", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions
  y_pred_cubic <- predict(cubic_model, data.frame(X = X_plot))
  y_pred_natural <- predict(natural_model, data.frame(X = X_plot))
  
  p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred_cubic), 
                       aes(X, y), color = "blue", size = 1, label = "Cubic") +
    geom_line(data = data.frame(X = X_plot, y = y_pred_natural), 
              aes(X, y), color = "red", size = 1, label = "Natural")
  
  # Model comparison
  mse_cubic <- mean((y - predict(cubic_model))^2)
  mse_natural <- mean((y - predict(natural_model))^2)
  
  cat("Model Comparison:\n")
  cat(sprintf("Cubic Spline MSE: %.4f\n", mse_cubic))
  cat(sprintf("Natural Spline MSE: %.4f\n", mse_natural))
  
  # Print plot
  print(p1)
  
  return(list(cubic_model = cubic_model, natural_model = natural_model))
}

# Function to compare with other methods
compare_with_other_methods_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.3)
  
  # Fit different methods
  methods <- list()
  
  # Linear regression
  methods$"Linear" <- lm(y ~ X)
  
  # Polynomial regression (degree 3)
  methods$"Polynomial" <- lm(y ~ poly(X, 3))
  
  # Regression spline
  methods$"Regression Spline" <- fit_regression_spline(X, y, 8, "cubic")
  
  # Natural spline
  methods$"Natural Spline" <- fit_regression_spline(X, y, 8, "natural")
  
  # Create prediction data
  X_plot <- seq(0, 10, length.out = 200)
  
  # Create comparison plot
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Method Comparison", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for each method
  for (name in names(methods)) {
    if (name == "Linear") {
      y_pred <- predict(methods[[name]], data.frame(X = X_plot))
    } else if (name == "Polynomial") {
      y_pred <- predict(methods[[name]], data.frame(X = X_plot))
    } else {
      y_pred <- predict(methods[[name]], data.frame(X = X_plot))
    }
    
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_pred), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  cat("Method Comparison:\n")
  for (name in names(methods)) {
    y_pred <- predict(methods[[name]])
    mse <- mean((y - y_pred)^2)
    r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
    cat(sprintf("%s: MSE = %.4f, R² = %.4f\n", name, mse, r2))
  }
  
  # Print plot
  print(p1)
  
  return(methods)
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic regression splines
  cat("=== BASIC REGRESSION SPLINES DEMONSTRATION ===\n")
  results <- demonstrate_regression_splines_r()
  
  # Analyze birthrate data
  cat("\n=== ANALYZING BIRTHRATE DATA ===\n")
  birthrate_results <- analyze_birthrate_data_r()
  
  # Demonstrate advanced features
  cat("\n=== ADVANCED FEATURES ===\n")
  advanced_results <- demonstrate_advanced_features_r()
  
  # Compare with other methods
  cat("\n=== COMPARING WITH OTHER METHODS ===\n")
  comparison_results <- compare_with_other_methods_r()
}
