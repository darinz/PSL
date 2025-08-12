# Local Regression Implementation in R
# ====================================
#
# This script provides a complete implementation of local regression
# including cross-validation, model selection, and comprehensive demonstrations.

library(ggplot2)
library(dplyr)

# Function to compute kernel weights
kernel_weights <- function(u, kernel = "tricube") {
  if (kernel == "tricube") {
    return(ifelse(abs(u) < 1, (1 - abs(u)^3)^3, 0))
  } else if (kernel == "gaussian") {
    return(exp(-u^2 / 2))
  } else if (kernel == "epanechnikov") {
    return(ifelse(abs(u) < 1, 0.75 * (1 - u^2), 0))
  } else {
    stop("Unknown kernel")
  }
}

# Function to fit local regression
fit_local_regression <- function(X, y, x0, bandwidth, degree = 1, kernel = "tricube") {
  # Compute distances and weights
  distances <- abs(X - x0)
  u <- distances / bandwidth
  weights <- kernel_weights(u, kernel)
  
  # Remove points with zero weight
  mask <- weights > 0
  if (sum(mask) < degree + 1) {
    return(NA)
  }
  
  X_local <- X[mask]
  y_local <- y[mask]
  weights_local <- weights[mask]
  
  # Create polynomial basis
  X_poly <- matrix(1, nrow = length(X_local), ncol = degree + 1)
  for (d in 1:degree) {
    X_poly[, d + 1] <- (X_local - x0)^d
  }
  
  # Weighted least squares
  W <- diag(weights_local)
  XWX <- t(X_poly) %*% W %*% X_poly
  XWy <- t(X_poly) %*% W %*% y_local
  
  tryCatch({
    beta <- solve(XWX, XWy)
    return(beta[1])  # Return intercept
  }, error = function(e) {
    return(NA)
  })
}

# Function to compute bandwidth
compute_bandwidth <- function(X, x0, nn_frac = 0.3) {
  n_neighbors <- max(1, round(nn_frac * length(X)))
  distances <- abs(X - x0)
  sorted_distances <- sort(distances)
  return(sorted_distances[n_neighbors])
}

# Function to predict using local regression
predict_local_regression <- function(X, y, X_new, bandwidth = NULL, 
                                   nn_frac = 0.3, degree = 1, kernel = "tricube") {
  predictions <- numeric(length(X_new))
  
  for (i in seq_along(X_new)) {
    x0 <- X_new[i]
    
    if (is.null(bandwidth)) {
      h <- compute_bandwidth(X, x0, nn_frac)
    } else {
      h <- bandwidth
    }
    
    pred <- fit_local_regression(X, y, x0, h, degree, kernel)
    predictions[i] <- pred
  }
  
  return(predictions)
}

# Function to demonstrate local regression
demonstrate_local_regression_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Test different parameters
  X_plot <- seq(0, 10, length.out = 200)
  
  # Different bandwidths
  nn_fractions <- c(0.1, 0.3, 0.5)
  predictions_nn <- list()
  
  for (nn_frac in nn_fractions) {
    pred <- predict_local_regression(X, y, X_plot, nn_frac = nn_frac, degree = 1)
    predictions_nn[[paste0("NN=", nn_frac)]] <- pred
  }
  
  # Different degrees
  degrees <- c(0, 1, 2)
  predictions_degree <- list()
  
  for (degree in degrees) {
    pred <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = degree)
    predictions_degree[[paste0("Degree=", degree)]] <- pred
  }
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Bandwidth", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for different bandwidths
  for (name in names(predictions_nn)) {
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = predictions_nn[[name]]), 
                         aes(X, y), color = name, size = 1)
  }
  
  p2 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Effect of Polynomial Degree", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for different degrees
  for (name in names(predictions_degree)) {
    p2 <- p2 + geom_line(data = data.frame(X = X_plot, y = predictions_degree[[name]]), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Cross-validation for bandwidth selection
  nn_fractions_cv <- seq(0.05, 0.8, length.out = 20)
  cv_scores <- numeric(length(nn_fractions_cv))
  
  for (i in seq_along(nn_fractions_cv)) {
    nn_frac <- nn_fractions_cv[i]
    
    # Leave-one-out cross-validation
    cv_preds <- numeric(length(X))
    for (j in seq_along(X)) {
      X_cv <- X[-j]
      y_cv <- y[-j]
      pred <- predict_local_regression(X_cv, y_cv, X[j], nn_frac = nn_frac, degree = 1)
      cv_preds[j] <- pred
    }
    
    cv_scores[i] <- mean((y - cv_preds)^2, na.rm = TRUE)
  }
  
  p3 <- ggplot(data.frame(NN_Fraction = nn_fractions_cv, CV_Score = cv_scores), 
               aes(NN_Fraction, CV_Score)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +
    labs(title = "Bandwidth Selection via Cross-Validation", 
         x = "Nearest Neighbor Fraction", y = "Cross-Validation MSE") +
    theme_minimal()
  
  # Kernel functions
  u <- seq(-2, 2, length.out = 100)
  kernels <- c("tricube", "gaussian", "epanechnikov")
  kernel_data <- data.frame(
    u = rep(u, length(kernels)),
    weight = c(kernel_weights(u, "tricube"), 
               kernel_weights(u, "gaussian"), 
               kernel_weights(u, "epanechnikov")),
    kernel = rep(kernels, each = length(u))
  )
  
  p4 <- ggplot(kernel_data, aes(u, weight, color = kernel)) +
    geom_line(size = 1) +
    labs(title = "Kernel Functions", x = "u", y = "K(u)") +
    theme_minimal()
  
  # Print plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  return(list(predictions_nn = predictions_nn, 
              predictions_degree = predictions_degree,
              cv_scores = cv_scores))
}

# Function to analyze outliers
analyze_outliers_r <- function() {
  # Generate data with outliers
  set.seed(42)
  X <- seq(0, 10, length.out = 80)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(80, 0, 0.3)
  
  # Add outliers
  outlier_indices <- c(20, 40, 60)
  y[outlier_indices] <- y[outlier_indices] + 3 * rnorm(length(outlier_indices))
  
  # Fit models
  X_plot <- seq(0, 10, length.out = 200)
  
  # Standard local regression
  y_pred_std <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = 1)
  
  # For robust fitting, we would need to implement LOWESS
  # For now, use standard local regression
  y_pred_rob <- y_pred_std  # Placeholder
  
  # Create plots
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_point(data = data.frame(X = X[outlier_indices], y = y[outlier_indices]), 
               aes(X, y), color = "red", size = 3) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    geom_line(data = data.frame(X = X_plot, y = y_pred_std), aes(X, y), 
              color = "blue", size = 1) +
    labs(title = "Local Regression with Outliers", x = "X", y = "Y") +
    theme_minimal()
  
  print(p1)
  
  return(list(y_pred_std = y_pred_std, y_pred_rob = y_pred_rob))
}

# Function to demonstrate advanced features
demonstrate_advanced_features_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Test different kernel functions
  X_plot <- seq(0, 10, length.out = 200)
  kernels <- c("tricube", "gaussian", "epanechnikov")
  predictions_kernel <- list()
  
  for (kernel in kernels) {
    pred <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = 1, kernel = kernel)
    predictions_kernel[[kernel]] <- pred
  }
  
  # Create comparison plot
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Kernel Function Comparison", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for different kernels
  for (name in names(predictions_kernel)) {
    p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = predictions_kernel[[name]]), 
                         aes(X, y), color = name, size = 1)
  }
  
  # Model comparison
  cat("Model Comparison:\n")
  for (name in names(predictions_kernel)) {
    y_pred <- predict_local_regression(X, y, X, nn_frac = 0.3, degree = 1, kernel = name)
    mse <- mean((y - y_pred)^2, na.rm = TRUE)
    cat(sprintf("%s Kernel: MSE = %.4f\n", name, mse))
  }
  
  # Print plot
  print(p1)
  
  return(predictions_kernel)
}

# Function to compare with other methods
compare_with_other_methods_r <- function() {
  # Generate synthetic data
  set.seed(42)
  X <- seq(0, 10, length.out = 100)
  y_true <- 2 + 3*sin(X) + 0.5*X
  y <- y_true + rnorm(100, 0, 0.5)
  
  # Fit different methods
  X_plot <- seq(0, 10, length.out = 200)
  
  # Linear regression
  lm_model <- lm(y ~ X)
  y_lm <- predict(lm_model, data.frame(X = X_plot))
  
  # Polynomial regression (degree 3)
  poly_model <- lm(y ~ poly(X, 3))
  y_poly <- predict(poly_model, data.frame(X = X_plot))
  
  # Local regression
  y_local <- predict_local_regression(X, y, X_plot, nn_frac = 0.3, degree = 1)
  
  # Create comparison plot
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(X, y), alpha = 0.6) +
    geom_line(data = data.frame(X = X, y = y_true), aes(X, y), 
              linetype = "dashed", color = "black", size = 1) +
    labs(title = "Method Comparison", x = "X", y = "Y") +
    theme_minimal()
  
  # Add predictions for each method
  p1 <- p1 + geom_line(data = data.frame(X = X_plot, y = y_lm), 
                       aes(X, y), color = "blue", size = 1, label = "Linear") +
    geom_line(data = data.frame(X = X_plot, y = y_poly), 
              aes(X, y), color = "green", size = 1, label = "Polynomial") +
    geom_line(data = data.frame(X = X_plot, y = y_local), 
              aes(X, y), color = "red", size = 1, label = "Local Regression")
  
  # Model comparison
  cat("Method Comparison:\n")
  methods <- list(
    "Linear" = y_lm,
    "Polynomial" = y_poly,
    "Local Regression" = y_local
  )
  
  for (name in names(methods)) {
    y_pred <- methods[[name]]
    mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
    cat(sprintf("%s: MSE = %.4f\n", name, mse))
  }
  
  # Print plot
  print(p1)
  
  return(methods)
}

# Main execution
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic local regression
  cat("=== BASIC LOCAL REGRESSION DEMONSTRATION ===\n")
  results <- demonstrate_local_regression_r()
  
  # Analyze outliers
  cat("\n=== ANALYZING OUTLIERS ===\n")
  outlier_results <- analyze_outliers_r()
  
  # Demonstrate advanced features
  cat("\n=== ADVANCED FEATURES ===\n")
  advanced_results <- demonstrate_advanced_features_r()
  
  # Compare with other methods
  cat("\n=== COMPARING WITH OTHER METHODS ===\n")
  comparison_results <- compare_with_other_methods_r()
}
