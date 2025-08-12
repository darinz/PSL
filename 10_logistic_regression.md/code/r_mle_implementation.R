# Maximum Likelihood Estimation for Logistic Regression in R

# Load required libraries
library(ggplot2)
library(dplyr)
library(gridExtra)
library(MASS)

# Sigmoid function with numerical stability
sigmoid <- function(z) {
  z <- pmin(pmax(z, -500), 500)  # Prevent overflow
  return(1 / (1 + exp(-z)))
}

# Log-likelihood function
log_likelihood <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  p <- pmin(pmax(p, 1e-15), 1-1e-15)  # Prevent log(0)
  return(sum(y * log(p) + (1-y) * log(1-p)))
}

# Gradient function
gradient <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  return(t(X) %*% (y - p))
}

# Hessian function
hessian <- function(beta, X, y) {
  z <- X %*% beta
  p <- sigmoid(z)
  W <- diag(as.vector(p * (1-p)))
  return(-t(X) %*% W %*% X)
}

# Newton-Raphson optimization
newton_raphson <- function(X, y, max_iter = 100, tol = 1e-6) {
  n_features <- ncol(X)
  beta <- rep(0, n_features)
  history <- list(log_likelihood = numeric(max_iter), 
                  beta_norm = numeric(max_iter))
  
  for (iteration in 1:max_iter) {
    # Compute current predictions
    z <- X %*% beta
    p <- sigmoid(z)
    
    # Store history
    history$log_likelihood[iteration] <- log_likelihood(beta, X, y)
    history$beta_norm[iteration] <- norm(beta, "2")
    
    # Compute gradient and Hessian
    grad <- gradient(beta, X, y)
    H <- hessian(beta, X, y)
    
    # Newton-Raphson update
    tryCatch({
      delta <- solve(H, grad)
      beta_new <- beta - delta
      
      # Check convergence
      if (norm(beta_new - beta, "2") < tol) {
        cat("Newton-Raphson converged after", iteration, "iterations\n")
        break
      }
      
      beta <- beta_new
    }, error = function(e) {
      cat("Hessian is singular, using pseudo-inverse\n")
      delta <- MASS::ginv(H) %*% grad
      beta <<- beta - delta
    })
  }
  
  return(list(beta = beta, history = history))
}

# Iteratively Reweighted Least Squares
irls <- function(X, y, max_iter = 100, tol = 1e-6) {
  n_features <- ncol(X)
  beta <- rep(0, n_features)
  history <- list(log_likelihood = numeric(max_iter), 
                  beta_norm = numeric(max_iter))
  
  for (iteration in 1:max_iter) {
    # Compute current predictions
    z <- X %*% beta
    p <- sigmoid(z)
    
    # Store history
    history$log_likelihood[iteration] <- log_likelihood(beta, X, y)
    history$beta_norm[iteration] <- norm(beta, "2")
    
    # Compute working response and weights
    working_response <- z + (y - p) / (p * (1-p) + 1e-15)
    weights <- p * (1-p)
    
    # Weighted least squares update
    W <- diag(as.vector(weights))
    tryCatch({
      beta_new <- solve(t(X) %*% W %*% X, t(X) %*% W %*% working_response)
      
      # Check convergence
      if (norm(beta_new - beta, "2") < tol) {
        cat("IRLS converged after", iteration, "iterations\n")
        break
      }
      
      beta <- beta_new
    }, error = function(e) {
      cat("Matrix is singular, using pseudo-inverse\n")
      beta_new <- MASS::ginv(t(X) %*% W %*% X) %*% t(X) %*% W %*% working_response
      beta <<- beta_new
    })
  }
  
  return(list(beta = beta, history = history))
}

# Generate synthetic data
generate_synthetic_data <- function(n_samples = 1000, n_features = 3, random_state = 42) {
  set.seed(random_state)
  
  # True parameters
  true_beta <- c(-2.0, 1.5, -0.8)
  
  # Generate features
  X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)
  X[, 1] <- 1  # Add intercept
  
  # Generate probabilities and outcomes
  z <- X %*% true_beta
  p <- 1 / (1 + exp(-z))
  y <- rbinom(n_samples, 1, p)
  
  return(list(X = X, y = y, true_beta = true_beta))
}

# Demonstrate MLE methods
demonstrate_mle_methods_r <- function() {
  # Generate synthetic data
  data <- generate_synthetic_data()
  X <- data$X
  y <- data$y
  true_beta <- data$true_beta
  
  cat("Synthetic Data Summary:\n")
  cat("Number of samples:", nrow(X), "\n")
  cat("Number of features:", ncol(X), "\n")
  cat("True parameters:", true_beta, "\n")
  cat("Class distribution:", table(y), "\n")
  
  # Fit models using different methods
  methods <- c("newton", "irls")
  models <- list()
  
  for (method in methods) {
    cat("\n=== Fitting with", toupper(method), "method ===\n")
    
    if (method == "newton") {
      result <- newton_raphson(X, y)
    } else {
      result <- irls(X, y)
    }
    
    models[[method]] <- result
    
    cat("Estimated parameters:", result$beta, "\n")
    cat("True parameters:", true_beta, "\n")
    cat("Parameter difference:", norm(result$beta - true_beta, "2"), "\n")
  }
  
  # Compare with glm
  cat("\n=== Comparing with glm ===\n")
  glm_model <- glm(y ~ X - 1, family = binomial)
  glm_beta <- coef(glm_model)
  
  cat("GLM parameters:", glm_beta, "\n")
  cat("GLM vs Newton difference:", norm(glm_beta - models$newton$beta, "2"), "\n")
  cat("GLM vs IRLS difference:", norm(glm_beta - models$irls$beta, "2"), "\n")
  
  return(list(models = models, glm_model = glm_model, glm_beta = glm_beta, true_beta = true_beta))
}

# Visualize convergence
visualize_convergence_r <- function(models) {
  # Convergence plots
  convergence_plots <- list()
  
  for (method in names(models)) {
    history <- models[[method]]$history
    
    # Log-likelihood convergence
    p1 <- ggplot(data.frame(iteration = 1:length(history$log_likelihood), 
                            log_likelihood = history$log_likelihood)) +
      geom_line(aes(x = iteration, y = log_likelihood)) +
      labs(title = paste(toupper(method), "- Log-Likelihood Convergence"),
           x = "Iteration", y = "Log-Likelihood") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    # Parameter norm convergence
    p2 <- ggplot(data.frame(iteration = 1:length(history$beta_norm), 
                            beta_norm = history$beta_norm)) +
      geom_line(aes(x = iteration, y = beta_norm)) +
      labs(title = paste(toupper(method), "- Parameter Norm Convergence"),
           x = "Iteration", y = "||β||") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    convergence_plots[[method]] <- list(p1, p2)
  }
  
  # Display plots
  do.call(grid.arrange, c(unlist(convergence_plots, recursive = FALSE), ncol = 2))
}

# Compare parameters
compare_parameters_r <- function(models, glm_beta, true_beta) {
  # Create comparison data frame
  param_comparison <- data.frame(
    True = true_beta,
    Newton = models$newton$beta,
    IRLS = models$irls$beta,
    GLM = glm_beta
  )
  
  cat("\n=== Parameter Comparison ===\n")
  print(param_comparison)
  
  # Calculate parameter differences
  cat("\n=== Parameter Differences ===\n")
  for (method in names(models)) {
    diff <- norm(models[[method]]$beta - true_beta, "2")
    cat(toupper(method), "vs True:", diff, "\n")
  }
  
  glm_diff <- norm(glm_beta - true_beta, "2")
  cat("GLM vs True:", glm_diff, "\n")
  
  return(param_comparison)
}

# Evaluate models
evaluate_models_r <- function(models, X, y) {
  cat("\n=== Model Evaluation ===\n")
  results <- list()
  
  for (method in names(models)) {
    beta_hat <- models[[method]]$beta
    z_pred <- X %*% beta_hat
    p_pred <- sigmoid(z_pred)
    y_pred <- ifelse(p_pred >= 0.5, 1, 0)
    accuracy <- mean(y_pred == y)
    results[[method]] <- accuracy
    cat(toupper(method), "Accuracy:", accuracy, "\n")
  }
  
  return(results)
}

# Visualize decision boundaries
visualize_decision_boundaries_r <- function(models, X, y) {
  if (ncol(X) != 3) {  # Not 2D case (including intercept)
    cat("Decision boundary visualization only available for 2D case\n")
    return()
  }
  
  # Create grid
  x1_range <- range(X[, 2])
  x2_range <- range(X[, 3])
  x1_grid <- seq(x1_range[1] - 0.5, x1_range[2] + 0.5, length.out = 100)
  x2_grid <- seq(x2_range[1] - 0.5, x2_range[2] + 0.5, length.out = 100)
  grid_data <- expand.grid(x1 = x1_grid, x2 = x2_grid)
  
  # Add intercept
  X_grid <- cbind(1, grid_data$x1, grid_data$x2)
  
  # Predict probabilities for each method
  decision_plots <- list()
  
  for (method in names(models)) {
    beta_hat <- models[[method]]$beta
    z_pred <- X_grid %*% beta_hat
    p_pred <- sigmoid(z_pred)
    
    # Add predictions to grid data
    grid_data$prob <- p_pred
    
    # Create plot
    p <- ggplot() +
      geom_contour(data = grid_data, aes(x = x1, y = x2, z = prob), 
                   breaks = 0.5, color = "red", size = 1) +
      geom_point(data = data.frame(x1 = X[, 2], x2 = X[, 3], y = factor(y)), 
                 aes(x = x1, y = x2, color = y), alpha = 0.6) +
      labs(title = paste(toupper(method), "Decision Boundary"),
           x = "Feature 1", y = "Feature 2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    decision_plots[[method]] <- p
  }
  
  # Display decision boundary plots
  do.call(grid.arrange, c(decision_plots, ncol = 2))
}

# Demonstrate gradient and Hessian
demonstrate_gradient_hessian_r <- function() {
  # Generate small dataset for demonstration
  data <- generate_synthetic_data(n_samples = 100, n_features = 3)
  X <- data$X
  y <- data$y
  true_beta <- data$true_beta
  
  # Test at different parameter values
  test_betas <- list(
    rep(0, 3),
    true_beta,
    true_beta + rnorm(3) * 0.1
  )
  
  cat("=== Gradient and Hessian Demonstration ===\n")
  for (i in 1:length(test_betas)) {
    beta <- test_betas[[i]]
    cat("\nTest", i, ": β =", beta, "\n")
    
    # Compute gradient and Hessian
    grad <- gradient(beta, X, y)
    H <- hessian(beta, X, y)
    
    cat("Gradient norm:", norm(grad, "2"), "\n")
    cat("Hessian eigenvalues:", eigen(H)$values, "\n")
    cat("Hessian is negative semi-definite:", all(eigen(H)$values <= 0), "\n")
  }
  
  return(list(X = X, y = y, true_beta = true_beta))
}

# Analyze convergence properties
analyze_convergence_properties_r <- function() {
  # Test different starting points
  data <- generate_synthetic_data(n_samples = 500, n_features = 3)
  X <- data$X
  y <- data$y
  true_beta <- data$true_beta
  
  starting_points <- list(
    rep(0, 3),
    rnorm(3) * 0.1,
    rnorm(3) * 1.0,
    true_beta + rnorm(3) * 0.5
  )
  
  cat("=== Convergence Analysis ===\n")
  
  for (i in 1:length(starting_points)) {
    start_beta <- starting_points[[i]]
    cat("\nStarting point", i, ":", start_beta, "\n")
    
    # Test Newton-Raphson
    result_newton <- newton_raphson(X, y, max_iter = 20, tol = 1e-6)
    
    # Test IRLS
    result_irls <- irls(X, y, max_iter = 20, tol = 1e-6)
    
    cat("Newton iterations:", length(result_newton$history$log_likelihood), "\n")
    cat("IRLS iterations:", length(result_irls$history$log_likelihood), "\n")
    cat("Newton final log-likelihood:", result_newton$history$log_likelihood[length(result_newton$history$log_likelihood)], "\n")
    cat("IRLS final log-likelihood:", result_irls$history$log_likelihood[length(result_irls$history$log_likelihood)], "\n")
  }
  
  return(list(X = X, y = y, true_beta = true_beta))
}

# Demonstrate numerical stability
demonstrate_numerical_stability_r <- function() {
  # Generate data with potential numerical issues
  set.seed(42)
  n_samples <- 100
  n_features <- 5
  
  # Create features with high correlation (potential singularity)
  X <- matrix(rnorm(n_samples * n_features), n_samples, n_features)
  X[, 1] <- 1  # Intercept
  X[, 3] <- X[, 2] + rnorm(n_samples) * 0.01  # High correlation
  
  # True parameters with some large values
  true_beta <- c(-5.0, 10.0, -9.5, 2.0, -1.0)
  
  # Generate outcomes
  z <- X %*% true_beta
  p <- 1 / (1 + exp(-z))
  y <- rbinom(n_samples, 1, p)
  
  cat("=== Numerical Stability Demonstration ===\n")
  cat("Feature correlation:", cor(X[, 2], X[, 3]), "\n")
  cat("Logit range:", range(z), "\n")
  cat("Probability range:", range(p), "\n")
  
  # Test different methods
  methods <- c("newton", "irls")
  
  for (method in methods) {
    cat("\n", toupper(method), "method:\n")
    tryCatch({
      if (method == "newton") {
        result <- newton_raphson(X, y, max_iter = 50, tol = 1e-6)
      } else {
        result <- irls(X, y, max_iter = 50, tol = 1e-6)
      }
      cat("Converged successfully\n")
      cat("Parameter difference:", norm(result$beta - true_beta, "2"), "\n")
    }, error = function(e) {
      cat("Failed:", e$message, "\n")
    })
  }
  
  return(list(X = X, y = y, true_beta = true_beta))
}

# Compare with other optimizers
compare_with_other_optimizers_r <- function() {
  data <- generate_synthetic_data(n_samples = 500, n_features = 3)
  X <- data$X
  y <- data$y
  true_beta <- data$true_beta
  
  # Define negative log-likelihood for optim
  neg_log_likelihood <- function(beta) {
    z <- X %*% beta
    p <- 1 / (1 + exp(-z))
    p <- pmin(pmax(p, 1e-15), 1-1e-15)
    return(-sum(y * log(p) + (1-y) * log(1-p)))
  }
  
  cat("=== Optimization Method Comparison ===\n")
  
  # Test different optimization methods
  optimizers <- c("BFGS", "CG", "L-BFGS-B")
  
  results <- list()
  
  for (method in optimizers) {
    cat("\n", method, ":\n")
    tryCatch({
      result <- optim(rep(0, 3), neg_log_likelihood, method = method, 
                     control = list(maxit = 100))
      
      if (result$convergence == 0) {
        cat("Converged in", result$counts[1], "iterations\n")
        cat("Final function value:", result$value, "\n")
        cat("Parameter difference:", norm(result$par - true_beta, "2"), "\n")
        results[[method]] <- result$par
      } else {
        cat("Failed to converge\n")
      }
    }, error = function(e) {
      cat("Error:", e$message, "\n")
    })
  }
  
  return(list(results = results, true_beta = true_beta))
}

# Main function to demonstrate MLE implementation
main_r <- function() {
  cat("Maximum Likelihood Estimation for Logistic Regression\n")
  cat("=" * 60, "\n")
  
  # 1. Demonstrate MLE methods
  cat("\n1. MLE Methods Demonstration:\n")
  mle_results <- demonstrate_mle_methods_r()
  
  # 2. Visualize convergence
  cat("\n2. Convergence Visualization:\n")
  visualize_convergence_r(mle_results$models)
  
  # 3. Compare parameters
  cat("\n3. Parameter Comparison:\n")
  param_df <- compare_parameters_r(mle_results$models, mle_results$glm_beta, mle_results$true_beta)
  
  # 4. Evaluate models
  cat("\n4. Model Evaluation:\n")
  data <- generate_synthetic_data()
  results <- evaluate_models_r(mle_results$models, data$X, data$y)
  
  # 5. Visualize decision boundaries
  cat("\n5. Decision Boundary Visualization:\n")
  visualize_decision_boundaries_r(mle_results$models, data$X, data$y)
  
  # 6. Demonstrate gradient and Hessian
  cat("\n6. Gradient and Hessian Demonstration:\n")
  grad_results <- demonstrate_gradient_hessian_r()
  
  # 7. Analyze convergence properties
  cat("\n7. Convergence Analysis:\n")
  conv_results <- analyze_convergence_properties_r()
  
  # 8. Demonstrate numerical stability
  cat("\n8. Numerical Stability Demonstration:\n")
  stab_results <- demonstrate_numerical_stability_r()
  
  # 9. Compare with other optimizers
  cat("\n9. Optimization Method Comparison:\n")
  opt_results <- compare_with_other_optimizers_r()
  
  return(list(
    mle_results = mle_results,
    param_df = param_df,
    results = results,
    grad_results = grad_results,
    conv_results = conv_results,
    stab_results = stab_results,
    opt_results = opt_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
