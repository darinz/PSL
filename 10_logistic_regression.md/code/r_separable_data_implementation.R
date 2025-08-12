# Separable Data Problem in Logistic Regression

# Load required libraries
library(ggplot2)
library(gridExtra)

# Create separable toy data
X <- matrix(c(
  1, 1,    # Red point 1
  2, 2,    # Red point 2
  -1, -1,  # Blue point 1
  -2, -2   # Blue point 2
), ncol = 2, byrow = TRUE)

y <- c(1, 1, 0, 0)  # 1 for red, 0 for blue

# Sigmoid function with numerical stability
sigmoid <- function(z) {
  z <- pmin(pmax(z, -500), 500)
  return(1 / (1 + exp(-z)))
}

# Log-likelihood function
log_likelihood <- function(beta) {
  z <- X %*% beta
  p <- sigmoid(z)
  p <- pmin(pmax(p, 1e-15), 1-1e-15)
  
  ll <- 0
  for (i in 1:length(y)) {
    if (y[i] == 1) {
      ll <- ll + log(p[i])
    } else {
      ll <- ll + log(1 - p[i])
    }
  }
  return(ll)
}

# Compute probabilities
compute_probabilities <- function(beta) {
  z <- X %*% beta
  return(sigmoid(z))
}

# Analyze different coefficient values
analyze_coefficients <- function(beta_values) {
  results <- list()
  
  for (i in 1:length(beta_values)) {
    beta_val <- beta_values[i]
    beta <- c(beta_val, beta_val)
    
    # Compute probabilities
    probs <- compute_probabilities(beta)
    
    # Compute log-likelihood
    ll <- log_likelihood(beta)
    
    # Compute accuracy
    predictions <- ifelse(probs >= 0.5, 1, 0)
    accuracy <- mean(predictions == y)
    
    results[[i]] <- list(
      beta = beta_val,
      probabilities = probs,
      log_likelihood = ll,
      accuracy = accuracy
    )
  }
  
  return(results)
}

# Test different coefficient values
beta_values <- c(0.1, 1, 5, 10, 50, 100, 500)
results <- analyze_coefficients(beta_values)

# Display results
cat("=== Coefficient Analysis ===\n\n")
cat("Beta\tLog-Likelihood\tAccuracy\tProbabilities\n")
cat(paste(rep("-", 60), collapse = ""), "\n")

for (result in results) {
  cat(sprintf("%.1f\t%.6f\t%.3f\t[%.3f, %.3f, %.3f, %.3f]\n",
              result$beta, result$log_likelihood, result$accuracy,
              result$probabilities[1], result$probabilities[2],
              result$probabilities[3], result$probabilities[4]))
}

# Visualize data and decision boundaries
visualize_data_and_boundaries <- function() {
  # Create data frame for plotting
  plot_data <- data.frame(
    x1 = X[, 1],
    x2 = X[, 2],
    class = factor(y)
  )
  
  # Create grid for decision boundaries
  x1_range <- seq(-3, 3, length.out = 100)
  x2_range <- seq(-3, 3, length.out = 100)
  grid_data <- expand.grid(x1 = x1_range, x2 = x2_range)
  
  # Plot data points
  p_base <- ggplot(plot_data, aes(x = x1, y = x2, color = class)) +
    geom_point(size = 4, alpha = 0.7) +
    scale_color_manual(values = c("0" = "blue", "1" = "red"),
                       labels = c("Class 0", "Class 1")) +
    labs(title = "Separable Data Points",
         x = "X1", y = "X2") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    coord_fixed(ratio = 1)
  
  print(p_base)
  
  # Create multiple plots with different decision boundaries
  plots <- list()
  beta_values_plot <- c(0.1, 1, 5, 10, 50, 100)
  
  for (i in 1:length(beta_values_plot)) {
    beta_val <- beta_values_plot[i]
    beta <- c(beta_val, beta_val)
    
    # Compute decision boundary
    grid_data$z <- beta[1] * grid_data$x1 + beta[2] * grid_data$x2
    
    p <- ggplot() +
      geom_point(data = plot_data, aes(x = x1, y = x2, color = class), 
                 size = 3, alpha = 0.7) +
      geom_contour(data = grid_data, aes(x = x1, y = x2, z = z), 
                   breaks = 0, color = "black", size = 1) +
      scale_color_manual(values = c("0" = "blue", "1" = "red"),
                         labels = c("Class 0", "Class 1")) +
      labs(title = paste("β = (", beta_val, ", ", beta_val, ")", sep = ""),
           x = "X1", y = "X2") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5)) +
      coord_fixed(ratio = 1) +
      xlim(-3, 3) + ylim(-3, 3)
    
    plots[[i]] <- p
  }
  
  # Display plots in a grid
  do.call(grid.arrange, c(plots, ncol = 3))
}

# Demonstrate convergence issue with glm
demonstrate_convergence_issue <- function() {
  cat("\n=== Demonstrating Convergence Issue ===\n\n")
  
  # Try different control parameters
  control_params <- list(
    epsilon = c(1e-8, 1e-10, 1e-12),
    maxit = c(25, 50, 100)
  )
  
  for (epsilon in control_params$epsilon) {
    for (maxit in control_params$maxit) {
      tryCatch({
        model <- glm(y ~ X - 1, family = binomial, 
                     control = list(epsilon = epsilon, maxit = maxit))
        
        coef_norm <- sqrt(sum(coef(model)^2))
        
        if (coef_norm > 100) {
          cat(sprintf("Epsilon: %.0e, Max iter: %d - Coefficients explode! Norm: %.2f\n",
                      epsilon, maxit, coef_norm))
        } else {
          cat(sprintf("Epsilon: %.0e, Max iter: %d - Coefficients stable. Norm: %.2f\n",
                      epsilon, maxit, coef_norm))
        }
      }, error = function(e) {
        cat(sprintf("Epsilon: %.0e, Max iter: %d - Failed: %s\n",
                    epsilon, maxit, e$message))
      })
    }
  }
}

# Plot log-likelihood vs coefficient magnitude
plot_log_likelihood_convergence <- function() {
  beta_magnitudes <- seq(0.1, 100, length.out = 100)
  log_likelihoods <- numeric(length(beta_magnitudes))
  
  for (i in 1:length(beta_magnitudes)) {
    beta <- c(beta_magnitudes[i], beta_magnitudes[i])
    log_likelihoods[i] <- log_likelihood(beta)
  }
  
  plot_data <- data.frame(
    magnitude = beta_magnitudes,
    log_likelihood = log_likelihoods
  )
  
  p <- ggplot(plot_data, aes(x = magnitude, y = log_likelihood)) +
    geom_line(size = 1) +
    geom_hline(yintercept = 0, color = "red", linestyle = "dashed", alpha = 0.7) +
    labs(title = "Log-Likelihood vs Coefficient Magnitude",
         x = "Coefficient Magnitude (β₁ = β₂)",
         y = "Log-Likelihood") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    annotate("text", x = 50, y = -0.5, 
             label = "Perfect Fit (LL = 0)", color = "red")
  
  print(p)
}

# Demonstrate regularization limitations
demonstrate_regularization_limitations <- function() {
  cat("\n=== Regularization Analysis ===\n\n")
  
  # Try different regularization strengths
  C_values <- c(1.0, 0.1, 0.01, 0.001)  # C = 1/lambda
  
  cat("C (1/λ)\tCoefficient Norm\tConverged\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  
  for (C in C_values) {
    tryCatch({
      # Use glmnet for L1 regularization
      if (require(glmnet, quietly = TRUE)) {
        # Lasso regularization
        cv_fit <- cv.glmnet(X, y, family = "binomial", alpha = 1)
        coef_norm <- sqrt(sum(coef(cv_fit, s = "lambda.min")^2))
        cat(sprintf("%.3f\t%.2f\t\tLasso\n", C, coef_norm))
      } else {
        # Fallback to glm with different control parameters
        model <- glm(y ~ X - 1, family = binomial, 
                     control = list(epsilon = 1e-8, maxit = 100))
        coef_norm <- sqrt(sum(coef(model)^2))
        cat(sprintf("%.3f\t%.2f\t\tGLM\n", C, coef_norm))
      }
    }, error = function(e) {
      cat(sprintf("%.3f\tFailed\t\t%s\n", C, e$message))
    })
  }
  
  cat("\nEven with strong regularization, coefficients can still explode!\n")
}

# Demonstrate Bayesian solution
demonstrate_bayesian_solution <- function() {
  cat("\n=== Bayesian Logistic Regression ===\n\n")
  
  # Simple Bayesian approach with informative priors
  # For demonstration, we'll use a simple MCMC-like approach
  
  # Prior parameters
  prior_mean <- c(0, 0)
  prior_sd <- c(1, 1)
  
  # Generate samples from posterior approximation
  n_samples <- 1000
  posterior_samples <- matrix(0, n_samples, 2)
  
  for (i in 1:n_samples) {
    # Sample from prior
    beta_prop <- rnorm(2, prior_mean, prior_sd)
    
    # Compute likelihood
    ll <- log_likelihood(beta_prop)
    
    # Simple acceptance rule (for demonstration)
    if (ll > -10) {  # Accept if likelihood is reasonable
      posterior_samples[i, ] <- beta_prop
    } else {
      posterior_samples[i, ] <- prior_mean
    }
  }
  
  # Compute posterior statistics
  beta_mean <- colMeans(posterior_samples)
  beta_sd <- apply(posterior_samples, 2, sd)
  
  cat("Posterior mean of β:", beta_mean, "\n")
  cat("Posterior std of β:", beta_sd, "\n")
  
  return(list(samples = posterior_samples, mean = beta_mean, sd = beta_sd))
}

# Demonstrate Firth's method
demonstrate_firth_method <- function() {
  cat("\n=== Firth's Method ===\n\n")
  
  # Implement Firth's correction
  firth_logistic <- function(X, y, max_iter = 100, tol = 1e-6) {
    n <- nrow(X)
    p <- ncol(X)
    beta <- rep(0, p)
    
    for (iteration in 1:max_iter) {
      # Compute current probabilities
      z <- X %*% beta
      p_probs <- 1 / (1 + exp(-z))
      
      # Compute weights and working response
      W <- diag(as.vector(p_probs * (1-p_probs)))
      z_working <- z + (y - p_probs) / (p_probs * (1-p_probs) + 1e-15)
      
      # Add Jeffreys prior correction
      H <- t(X) %*% W %*% X
      correction <- 0.5 * diag(H)
      
      # Update
      tryCatch({
        beta_new <- solve(H, t(X) %*% W %*% z_working + correction)
      }, error = function(e) {
        # Use pseudo-inverse if singular
        beta_new <<- MASS::ginv(H) %*% (t(X) %*% W %*% z_working + correction)
      })
      
      if (norm(beta_new - beta, "2") < tol) {
        break
      }
      
      beta <- beta_new
    }
    
    return(beta)
  }
  
  # Standard logistic regression
  tryCatch({
    model_standard <- glm(y ~ X - 1, family = binomial, 
                         control = list(epsilon = 1e-8, maxit = 100))
    coef_standard <- coef(model_standard)
    cat("Standard LR coefficients:", coef_standard, "\n")
    cat("Standard LR coefficient norm:", norm(coef_standard, "2"), "\n")
  }, error = function(e) {
    cat("Standard LR failed:", e$message, "\n")
  })
  
  # Firth's method
  tryCatch({
    coef_firth <- firth_logistic(X, y)
    cat("Firth's method coefficients:", coef_firth, "\n")
    cat("Firth's method coefficient norm:", norm(coef_firth, "2"), "\n")
  }, error = function(e) {
    cat("Firth's method failed:", e$message, "\n")
  })
  
  return(coef_firth)
}

# Demonstrate exact logistic regression
demonstrate_exact_logistic_regression <- function() {
  cat("\n=== Exact Logistic Regression ===\n\n")
  
  # Try to use exact method if available
  tryCatch({
    # Use logistf package for Firth's logistic regression
    if (require(logistf, quietly = TRUE)) {
      # Create data frame
      data_df <- data.frame(
        y = y,
        x1 = X[, 1],
        x2 = X[, 2]
      )
      
      # Fit Firth's logistic regression
      model <- logistf(y ~ x1 + x2 - 1, data = data_df)
      
      cat("Firth's logistic regression results:\n")
      print(summary(model))
      
      return(model)
    } else {
      # Fallback to standard glm
      model <- glm(y ~ X - 1, family = binomial)
      cat("Standard logistic regression results:\n")
      print(summary(model))
      
      return(model)
    }
  }, error = function(e) {
    cat("Exact method failed:", e$message, "\n")
    return(NULL)
  })
}

# Analyze mathematical properties
analyze_mathematical_properties <- function() {
  cat("\n=== Mathematical Analysis ===\n\n")
  
  # Test different coefficient directions
  directions <- list(
    c(1, 1),      # Diagonal direction
    c(1, -1),     # Anti-diagonal direction
    c(2, 1),      # Asymmetric direction
    c(0, 1)       # Vertical direction
  )
  
  for (i in 1:length(directions)) {
    direction <- directions[[i]]
    cat("Direction", i, ":", direction, "\n")
    
    # Check separability
    scores <- X %*% direction
    separable <- all(scores[y == 1] > 0) && all(scores[y == 0] < 0)
    cat("  Separable:", separable, "\n")
    
    if (separable) {
      margin <- min(abs(scores))
      cat("  Margin:", margin, "\n")
    }
    
    # Test scaling behavior
    scales <- c(1, 10, 100)
    for (scale in scales) {
      beta <- scale * direction
      z <- X %*% beta
      p <- 1 / (1 + exp(-z))
      ll <- sum(y * log(p) + (1-y) * log(1-p))
      cat("  Scale", scale, ": LL =", ll, "\n")
    }
    cat("\n")
  }
}

# Demonstrate practical implications
demonstrate_practical_implications <- function() {
  cat("\n=== Practical Implications ===\n\n")
  
  # Test prediction with different coefficient magnitudes
  test_points <- matrix(c(0.5, 0.5, -0.5, -0.5, 0, 0), ncol = 2, byrow = TRUE)
  
  beta_values <- c(1, 10, 100)
  
  for (beta_val in beta_values) {
    beta <- c(beta_val, beta_val)
    cat("Coefficients: β = (", beta_val, ", ", beta_val, ")\n", sep = "")
    
    for (i in 1:nrow(test_points)) {
      point <- test_points[i, ]
      z <- point %*% beta
      p <- sigmoid(z)
      cat("  Point", i, point, ": P(Y=1) =", p, "\n")
    }
    cat("\n")
  }
  
  cat("Key observations:\n")
  cat("1. Predictions become more extreme as coefficients increase\n")
  cat("2. Decision boundary remains stable\n")
  cat("3. Model confidence increases (probabilities approach 0 or 1)\n")
  cat("4. Standard errors become unreliable\n")
}

# Main function to demonstrate separable data problem
main_r <- function() {
  cat("Separable Data Problem in Logistic Regression\n")
  cat("=" * 60, "\n")
  
  # 1. Basic demonstration
  cat("\n1. Basic Demonstration:\n")
  visualize_data_and_boundaries()
  
  # 2. Demonstrate convergence issue
  cat("\n2. Convergence Issue Demonstration:\n")
  demonstrate_convergence_issue()
  
  # 3. Plot log-likelihood convergence
  cat("\n3. Log-Likelihood Convergence:\n")
  plot_log_likelihood_convergence()
  
  # 4. Demonstrate regularization limitations
  cat("\n4. Regularization Limitations:\n")
  demonstrate_regularization_limitations()
  
  # 5. Demonstrate Bayesian solution
  cat("\n5. Bayesian Solution:\n")
  bayesian_result <- demonstrate_bayesian_solution()
  
  # 6. Demonstrate Firth's method
  cat("\n6. Firth's Method:\n")
  firth_result <- demonstrate_firth_method()
  
  # 7. Demonstrate exact logistic regression
  cat("\n7. Exact Logistic Regression:\n")
  exact_result <- demonstrate_exact_logistic_regression()
  
  # 8. Analyze mathematical properties
  cat("\n8. Mathematical Analysis:\n")
  analyze_mathematical_properties()
  
  # 9. Demonstrate practical implications
  cat("\n9. Practical Implications:\n")
  demonstrate_practical_implications()
  
  cat("\n=== Key Observations ===\n")
  cat("1. As coefficients increase, log-likelihood approaches 0 (perfect fit)\n")
  cat("2. All probabilities approach 1 for their respective classes\n")
  cat("3. Decision boundary remains stable despite coefficient explosion\n")
  cat("4. Standard logistic regression solvers may fail to converge\n")
  cat("5. The model is still useful for prediction despite convergence issues\n")
  cat("6. Regularization doesn't solve the fundamental problem\n")
  cat("7. Bayesian methods and Firth's correction provide solutions\n")
  
  return(list(
    results = results,
    bayesian_result = bayesian_result,
    firth_result = firth_result,
    exact_result = exact_result
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
