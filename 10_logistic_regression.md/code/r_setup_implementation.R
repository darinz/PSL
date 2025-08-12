# Logistic Regression Setup Implementation in R
library(ggplot2)
library(gridExtra)
library(MASS)
library(caret)
library(pROC)

# Visualize the logit function and its properties
visualize_logit_function_r <- function() {
  # Generate probability values
  p <- seq(0.01, 0.99, length.out = 1000)
  
  # Compute logit values
  logit_p <- log(p / (1 - p))
  
  # Create data frames for plotting
  logit_df <- data.frame(probability = p, logit = logit_p)
  
  # Inverse logit (sigmoid) function
  x <- seq(-6, 6, length.out = 1000)
  sigmoid_x <- 1 / (1 + exp(-x))
  sigmoid_df <- data.frame(linear_predictor = x, probability = sigmoid_x)
  
  # Symmetry property
  p_sym <- seq(0.1, 0.9, length.out = 100)
  logit_p_sym <- log(p_sym / (1 - p_sym))
  logit_1_minus_p <- log((1 - p_sym) / p_sym)
  symmetry_df <- data.frame(
    probability = p_sym,
    logit_p = logit_p_sym,
    logit_1_minus_p = logit_1_minus_p
  )
  
  # Create plots
  p1 <- ggplot(logit_df, aes(x = probability, y = logit)) +
    geom_line(color = "blue", size = 1) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = 0.5, color = "red", linetype = "dashed", alpha = 0.7) +
    labs(title = "Logit Function",
         x = "Probability η(x)",
         y = "Logit g(η(x))") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlim(0, 1) +
    grid()
  
  p2 <- ggplot(sigmoid_df, aes(x = linear_predictor, y = probability)) +
    geom_line(color = "green", size = 1) +
    geom_hline(yintercept = 0.5, color = "red", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = 0, color = "red", linetype = "dashed", alpha = 0.7) +
    labs(title = "Sigmoid Function (Inverse Logit)",
         x = "Linear Predictor x^T β",
         y = "Probability η(x)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p3 <- ggplot(symmetry_df) +
    geom_line(aes(x = probability, y = logit_p, color = "logit(p)"), size = 1) +
    geom_line(aes(x = probability, y = logit_1_minus_p, color = "logit(1-p)"), 
              linetype = "dashed", size = 1) +
    labs(title = "Symmetry: logit(p) = -logit(1-p)",
         x = "Probability p",
         y = "Logit Value",
         color = "Function") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  # Decision boundary visualization
  x1 <- seq(-3, 3, length.out = 100)
  x2 <- seq(-3, 3, length.out = 100)
  grid_points <- expand.grid(x1 = x1, x2 = x2)
  
  # Example: β = [-0.5, 1, 1] (intercept, x1, x2)
  beta <- c(-0.5, 1, 1)
  grid_points$z <- 1 / (1 + exp(-(beta[1] + beta[2] * grid_points$x1 + beta[3] * grid_points$x2)))
  
  p4 <- ggplot(grid_points, aes(x = x1, y = x2, z = z)) +
    geom_contour_filled(bins = 20) +
    geom_contour(breaks = 0.5, color = "black", size = 1) +
    labs(title = "Logistic Regression Decision Boundary",
         x = "Feature 1",
         y = "Feature 2",
         fill = "Probability") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5)
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  return(list(p = p, logit_p = logit_p, x = x, sigmoid_x = sigmoid_x))
}

# Compare MSE and log-likelihood loss functions
compare_loss_functions_r <- function() {
  # Generate sample data
  set.seed(42)
  n_samples <- 1000
  
  # True parameters
  beta_true <- c(-1.5, 2.0, -0.8)
  
  # Generate features
  X <- matrix(rnorm(n_samples * 2), nrow = n_samples, ncol = 2)
  X_with_intercept <- cbind(1, X)
  
  # Generate true probabilities
  logits <- X_with_intercept %*% beta_true
  true_probs <- 1 / (1 + exp(-logits))
  
  # Generate binary outcomes
  y <- rbinom(n_samples, 1, true_probs)
  
  # Define loss functions
  mse_loss <- function(beta, X, y) {
    probs <- 1 / (1 + exp(-X %*% beta))
    return(mean((y - probs)^2))
  }
  
  log_likelihood_loss <- function(beta, X, y) {
    logits <- X %*% beta
    return(-mean(y * logits - log(1 + exp(logits))))
  }
  
  # Test different beta values
  beta_range <- seq(-3, 3, length.out = 100)
  mse_losses <- numeric(length(beta_range))
  ll_losses <- numeric(length(beta_range))
  
  for (i in 1:length(beta_range)) {
    beta_test <- c(beta_range[i], 2.0, -0.8)
    mse_losses[i] <- mse_loss(beta_test, X_with_intercept, y)
    ll_losses[i] <- log_likelihood_loss(beta_test, X_with_intercept, y)
  }
  
  # Create data frames for plotting
  loss_df <- data.frame(
    beta = rep(beta_range, 2),
    loss = c(mse_losses, ll_losses),
    type = rep(c("MSE", "Log-Likelihood"), each = length(beta_range))
  )
  
  # Visualization
  p1 <- ggplot(subset(loss_df, type == "MSE"), aes(x = beta, y = loss)) +
    geom_line(color = "blue", size = 1) +
    geom_vline(xintercept = beta_true[1], color = "red", linetype = "dashed", 
               aes(label = "True β₀")) +
    labs(title = "Mean Squared Error Loss",
         x = "β₀ (Intercept)",
         y = "MSE Loss") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p2 <- ggplot(subset(loss_df, type == "Log-Likelihood"), aes(x = beta, y = loss)) +
    geom_line(color = "green", size = 1) +
    geom_vline(xintercept = beta_true[1], color = "red", linetype = "dashed", 
               aes(label = "True β₀")) +
    labs(title = "Negative Log-Likelihood Loss",
         x = "β₀ (Intercept)",
         y = "Negative Log-Likelihood") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  # Print comparison
  cat("Loss Function Comparison:\n")
  cat("-" * 40, "\n")
  cat(sprintf("MSE Loss at true β₀: %.6f\n", mse_losses[50]))
  cat(sprintf("Log-Likelihood Loss at true β₀: %.6f\n", ll_losses[50]))
  cat(sprintf("MSE Loss gradient (approximate): %.6f\n", abs(mse_losses[51] - mse_losses[49])))
  cat(sprintf("Log-Likelihood gradient (approximate): %.6f\n", abs(ll_losses[51] - ll_losses[49])))
  
  return(list(mse_losses = mse_losses, ll_losses = ll_losses))
}

# Demonstrate the complete setup of logistic regression
logistic_regression_setup_demo_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 500
  n_features <- 2
  
  # True parameters
  beta_true <- c(-1.0, 2.0, -1.5)
  
  # Generate features
  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
  X_with_intercept <- cbind(1, X)
  
  # Generate probabilities
  logits <- X_with_intercept %*% beta_true
  probabilities <- 1 / (1 + exp(-logits))
  
  # Generate outcomes
  y <- rbinom(n_samples, 1, probabilities)
  
  # Create data frame for plotting
  plot_df <- data.frame(
    x1 = X[, 1],
    x2 = X[, 2],
    class = factor(y),
    probability = probabilities
  )
  
  # Visualize the data
  p1 <- ggplot(plot_df, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Binary Classification Data",
         x = "Feature 1",
         y = "Feature 2",
         color = "Class") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p2 <- ggplot(plot_df, aes(x = probability)) +
    geom_histogram(bins = 30, alpha = 0.7, fill = "steelblue", color = "black") +
    labs(title = "Distribution of True Probabilities",
         x = "True Probability P(Y=1|X)",
         y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  cat("Logistic Regression Setup Summary:\n")
  cat("-" * 40, "\n")
  cat(sprintf("Number of samples: %d\n", n_samples))
  cat(sprintf("Number of features: %d\n", n_features))
  cat(sprintf("True parameters: [%.1f, %.1f, %.1f]\n", beta_true[1], beta_true[2], beta_true[3]))
  cat(sprintf("Class balance: %.3f (proportion of class 1)\n", mean(y)))
  
  return(list(X = X, y = y, beta_true = beta_true))
}

# Demonstrate key properties of the logit link function
demonstrate_link_function_properties_r <- function() {
  # Generate test probabilities
  p_values <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  
  # Calculate logit values
  logit_values <- log(p_values / (1 - p_values))
  
  # Calculate inverse (sigmoid)
  sigmoid_values <- 1 / (1 + exp(-logit_values))
  
  cat("Link Function Properties Demonstration:\n")
  cat("-" * 50, "\n")
  cat(sprintf("%12s %10s %10s\n", "Probability", "Logit", "Sigmoid"))
  cat("-" * 50, "\n")
  
  for (i in 1:length(p_values)) {
    cat(sprintf("%12.2f %10.3f %10.3f\n", p_values[i], logit_values[i], sigmoid_values[i]))
  }
  
  # Demonstrate symmetry
  cat("\nSymmetry Property:\n")
  cat("-" * 30, "\n")
  for (p in c(0.1, 0.25, 0.5)) {
    logit_p <- log(p / (1 - p))
    logit_1_minus_p <- log((1 - p) / p)
    cat(sprintf("logit(%.2f) = %.3f, logit(%.2f) = %.3f\n", p, logit_p, 1-p, logit_1_minus_p))
    cat(sprintf("Sum: %.3f (should be 0)\n", logit_p + logit_1_minus_p))
  }
  
  return(list(p_values = p_values, logit_values = logit_values, sigmoid_values = sigmoid_values))
}

# Analyze the decision boundary of logistic regression
analyze_decision_boundary_r <- function() {
  # Generate grid of points
  x1 <- seq(-4, 4, length.out = 100)
  x2 <- seq(-4, 4, length.out = 100)
  grid_points <- expand.grid(x1 = x1, x2 = x2)
  
  # Different parameter sets
  beta_sets <- list(
    Linear = c(0, 1, 0),      # x1 = 0
    Diagonal = c(0, 1, 1),    # x1 + x2 = 0
    Offset = c(-1, 1, 1),     # x1 + x2 = 1
    Complex = c(-0.5, 2, -1)  # 2x1 - x2 = 0.5
  )
  
  plots <- list()
  
  for (i in 1:length(beta_sets)) {
    name <- names(beta_sets)[i]
    beta <- beta_sets[[i]]
    
    # Calculate probabilities
    grid_points$z <- 1 / (1 + exp(-(beta[1] + beta[2] * grid_points$x1 + beta[3] * grid_points$x2)))
    
    # Add decision boundary equation
    if (beta[2] != 0 && beta[3] != 0) {
      eq <- sprintf("%.1fx₁ + %.1fx₂ = %.1f", beta[2], beta[3], -beta[1])
    } else if (beta[2] != 0) {
      eq <- sprintf("x₁ = %.1f", -beta[1]/beta[2])
    } else if (beta[3] != 0) {
      eq <- sprintf("x₂ = %.1f", -beta[1]/beta[3])
    } else {
      eq <- "No boundary"
    }
    
    p <- ggplot(grid_points, aes(x = x1, y = x2, z = z)) +
      geom_contour_filled(bins = 20) +
      geom_contour(breaks = 0.5, color = "black", size = 1) +
      labs(title = paste(name, ":", eq),
           x = "Feature 1 (x₁)",
           y = "Feature 2 (x₂)",
           fill = "Probability") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5)) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5)
    
    plots[[i]] <- p
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  return(beta_sets)
}

# Compare logistic regression with linear regression for classification
compare_with_linear_regression_r <- function() {
  # Generate data
  set.seed(42)
  n_samples <- 200
  
  # True parameters for logistic regression
  beta_true <- c(-1.0, 2.0, -1.5)
  
  # Generate features
  X <- matrix(rnorm(n_samples * 2), nrow = n_samples, ncol = 2)
  X_with_intercept <- cbind(1, X)
  
  # Generate probabilities and outcomes
  logits <- X_with_intercept %*% beta_true
  probabilities <- 1 / (1 + exp(-logits))
  y <- rbinom(n_samples, 1, probabilities)
  
  # Fit logistic regression
  lr_model <- glm(y ~ X, family = binomial(link = "logit"))
  
  # Fit linear regression (treating binary as continuous)
  linear_model <- lm(y ~ X)
  
  # Predictions
  lr_probs <- predict(lr_model, type = "response")
  linear_preds <- predict(linear_model)
  
  # Create data frame for plotting
  plot_df <- data.frame(
    x1 = X[, 1],
    x2 = X[, 2],
    class = factor(y),
    true_prob = probabilities,
    lr_prob = lr_probs,
    linear_pred = linear_preds
  )
  
  # Create plots
  p1 <- ggplot(plot_df, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Original Data",
         x = "Feature 1",
         y = "Feature 2",
         color = "Class") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p2 <- ggplot(plot_df, aes(x = x1, y = x2, color = true_prob)) +
    geom_point(alpha = 0.7) +
    labs(title = "True Probabilities",
         x = "Feature 1",
         y = "Feature 2",
         color = "Probability") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
    grid()
  
  p3 <- ggplot(plot_df, aes(x = x1, y = x2, color = lr_prob)) +
    geom_point(alpha = 0.7) +
    labs(title = "Logistic Regression Predictions",
         x = "Feature 1",
         y = "Feature 2",
         color = "Probability") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
    grid()
  
  p4 <- ggplot(plot_df, aes(x = x1, y = x2, color = linear_pred)) +
    geom_point(alpha = 0.7) +
    labs(title = "Linear Regression Predictions",
         x = "Feature 1",
         y = "Feature 2",
         color = "Prediction") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0.5) +
    grid()
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  # Print comparison
  cat("Logistic vs Linear Regression Comparison:\n")
  cat("-" * 50, "\n")
  cat(sprintf("Logistic Regression - Predictions in [0,1]: %.3f to %.3f\n", 
              min(lr_probs), max(lr_probs)))
  cat(sprintf("Linear Regression - Predictions in [0,1]: %.3f to %.3f\n", 
              min(linear_preds), max(linear_preds)))
  cat(sprintf("Linear Regression - Predictions outside [0,1]: %d\n", 
              sum(linear_preds < 0 | linear_preds > 1)))
  
  return(list(lr_model = lr_model, linear_model = linear_model, 
              lr_probs = lr_probs, linear_preds = linear_preds))
}

# Demonstrate properties of different loss functions
demonstrate_loss_function_properties_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 100
  
  # True parameters
  beta_true <- c(-0.5, 1.0)
  
  # Generate single feature
  X <- matrix(rnorm(n_samples), ncol = 1)
  X_with_intercept <- cbind(1, X)
  
  # Generate true probabilities and outcomes
  logits <- X_with_intercept %*% beta_true
  true_probs <- 1 / (1 + exp(-logits))
  y <- rbinom(n_samples, 1, true_probs)
  
  # Define loss functions
  mse_loss <- function(beta) {
    probs <- 1 / (1 + exp(-X_with_intercept %*% beta))
    return(mean((y - probs)^2))
  }
  
  log_likelihood_loss <- function(beta) {
    logits <- X_with_intercept %*% beta
    return(-mean(y * logits - log(1 + exp(logits))))
  }
  
  hinge_loss <- function(beta) {
    # Simplified hinge loss for demonstration
    scores <- X_with_intercept %*% beta
    return(mean(pmax(0, 1 - (2*y - 1) * scores)))
  }
  
  # Test different beta values
  beta_range <- seq(-2, 2, length.out = 50)
  mse_losses <- numeric(length(beta_range))
  ll_losses <- numeric(length(beta_range))
  hinge_losses <- numeric(length(beta_range))
  
  for (i in 1:length(beta_range)) {
    beta_test <- c(beta_range[i], 1.0)
    mse_losses[i] <- mse_loss(beta_test)
    ll_losses[i] <- log_likelihood_loss(beta_test)
    hinge_losses[i] <- hinge_loss(beta_test)
  }
  
  # Create data frame for plotting
  loss_df <- data.frame(
    beta = rep(beta_range, 3),
    loss = c(mse_losses, ll_losses, hinge_losses),
    type = rep(c("MSE", "Log-Likelihood", "Hinge"), each = length(beta_range))
  )
  
  # Create plots
  p1 <- ggplot(subset(loss_df, type == "MSE"), aes(x = beta, y = loss)) +
    geom_line(color = "blue", size = 1) +
    geom_vline(xintercept = beta_true[1], color = "red", linetype = "dashed", 
               aes(label = "True β₀")) +
    labs(title = "Mean Squared Error Loss",
         x = "β₀ (Intercept)",
         y = "MSE Loss") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p2 <- ggplot(subset(loss_df, type == "Log-Likelihood"), aes(x = beta, y = loss)) +
    geom_line(color = "green", size = 1) +
    geom_vline(xintercept = beta_true[1], color = "red", linetype = "dashed", 
               aes(label = "True β₀")) +
    labs(title = "Negative Log-Likelihood Loss",
         x = "β₀ (Intercept)",
         y = "Negative Log-Likelihood") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  p3 <- ggplot(subset(loss_df, type == "Hinge"), aes(x = beta, y = loss)) +
    geom_line(color = "magenta", size = 1) +
    geom_vline(xintercept = beta_true[1], color = "red", linetype = "dashed", 
               aes(label = "True β₀")) +
    labs(title = "Hinge Loss",
         x = "β₀ (Intercept)",
         y = "Hinge Loss") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    grid()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  # Find minima
  mse_min_idx <- which.min(mse_losses)
  ll_min_idx <- which.min(ll_losses)
  hinge_min_idx <- which.min(hinge_losses)
  
  cat("Loss Function Properties:\n")
  cat("-" * 40, "\n")
  cat(sprintf("MSE Loss minimum at β₀ = %.3f\n", beta_range[mse_min_idx]))
  cat(sprintf("Log-Likelihood minimum at β₀ = %.3f\n", beta_range[ll_min_idx]))
  cat(sprintf("Hinge Loss minimum at β₀ = %.3f\n", beta_range[hinge_min_idx]))
  cat(sprintf("True β₀ = %.3f\n", beta_true[1]))
  
  return(list(mse_losses = mse_losses, ll_losses = ll_losses, hinge_losses = hinge_losses))
}

# Main function to demonstrate logistic regression setup
main_r <- function() {
  cat("Logistic Regression Setup Demonstration\n")
  cat("=" * 50, "\n")
  
  # 1. Visualize logit function
  cat("\n1. Logit Function Visualization:\n")
  logit_data <- visualize_logit_function_r()
  
  # 2. Compare loss functions
  cat("\n2. Loss Function Comparison:\n")
  loss_comparison <- compare_loss_functions_r()
  
  # 3. Setup demonstration
  cat("\n3. Complete Setup Demonstration:\n")
  setup_data <- logistic_regression_setup_demo_r()
  
  # 4. Link function properties
  cat("\n4. Link Function Properties:\n")
  link_properties <- demonstrate_link_function_properties_r()
  
  # 5. Decision boundary analysis
  cat("\n5. Decision Boundary Analysis:\n")
  decision_boundaries <- analyze_decision_boundary_r()
  
  # 6. Compare with linear regression
  cat("\n6. Comparison with Linear Regression:\n")
  regression_comparison <- compare_with_linear_regression_r()
  
  # 7. Loss function properties
  cat("\n7. Loss Function Properties:\n")
  loss_properties <- demonstrate_loss_function_properties_r()
  
  return(list(
    logit_data = logit_data,
    loss_comparison = loss_comparison,
    setup_data = setup_data,
    link_properties = link_properties,
    decision_boundaries = decision_boundaries,
    regression_comparison = regression_comparison,
    loss_properties = loss_properties
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
