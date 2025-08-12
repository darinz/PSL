# Support Vector Machines in R

# Load required libraries
library(e1071)
library(ggplot2)
library(gridExtra)
library(kernlab)

# Generate separable data
generate_separable_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  n_class1 <- n_samples %/% 2
  n_class2 <- n_samples - n_class1
  
  # Class 1: centered at (2, 2)
  class1 <- matrix(rnorm(n_class1 * 2), n_class1, 2) + matrix(c(2, 2), n_class1, 2, byrow = TRUE)
  
  # Class 2: centered at (-2, -2)
  class2 <- matrix(rnorm(n_class2 * 2), n_class2, 2) + matrix(c(-2, -2), n_class2, 2, byrow = TRUE)
  
  X <- rbind(class1, class2)
  y <- c(rep(1, n_class1), rep(-1, n_class2))
  
  return(list(X = X, y = y))
}

# Generate non-separable data
generate_nonseparable_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate circular data
  theta <- runif(n_samples, 0, 2 * pi)
  r <- runif(n_samples, 0.5, 1.5)
  
  # Inner circle (class 1)
  n_inner <- n_samples %/% 2
  inner_theta <- theta[1:n_inner]
  inner_r <- runif(n_inner, 0.5, 1.0)
  
  # Outer circle (class -1)
  outer_theta <- theta[(n_inner + 1):n_samples]
  outer_r <- runif(n_samples - n_inner, 1.2, 1.8)
  
  X_inner <- cbind(inner_r * cos(inner_theta), inner_r * sin(inner_theta))
  X_outer <- cbind(outer_r * cos(outer_theta), outer_r * sin(outer_theta))
  
  X <- rbind(X_inner, X_outer)
  y <- c(rep(1, n_inner), rep(-1, n_samples - n_inner))
  
  return(list(X = X, y = y))
}

# Generate overlapping data
generate_overlapping_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate overlapping classes
  X <- matrix(rnorm(n_samples * 2), n_samples, 2)
  y <- ifelse(X[, 1] + X[, 2] > 0, 1, -1)
  
  # Add some noise
  y[sample(n_samples, n_samples %/% 10)] <- -y[sample(n_samples, n_samples %/% 10)]
  
  return(list(X = X, y = y))
}

# Visualize decision boundary
visualize_decision_boundary <- function(X, y, model, title = "SVM Decision Boundary") {
  # Create grid
  x_min <- min(X[, 1]) - 0.5
  x_max <- max(X[, 1]) + 0.5
  y_min <- min(X[, 2]) - 0.5
  y_max <- max(X[, 2]) + 0.5
  
  grid_x <- seq(x_min, x_max, length.out = 100)
  grid_y <- seq(y_min, y_max, length.out = 100)
  grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
  
  # Predict on grid
  grid_data$pred <- predict(model, grid_data)
  
  # Create plot
  p <- ggplot() +
    geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)), 
                 breaks = 0, color = "black", size = 1) +
    geom_point(data = data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y)), 
               aes(x = X1, y = X2, color = y), size = 3, alpha = 0.8) +
    scale_color_manual(values = c("-1" = "blue", "1" = "red")) +
    labs(title = title, x = "Feature 1", y = "Feature 2") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(p)
}

# Demonstrate separable case
demonstrate_separable_case <- function() {
  cat("=== Linear SVM: Separable Case ===\n\n")
  
  # Generate data
  data <- generate_separable_data()
  X <- data$X
  y <- data$y
  
  # Fit SVM
  model <- svm(X, y, kernel = "linear", scale = FALSE)
  
  # Evaluate
  y_pred <- predict(model, X)
  accuracy <- mean(y_pred == y)
  
  cat("Accuracy:", accuracy, "\n")
  cat("Number of support vectors:", length(model$index), "\n")
  cat("Support vector ratio:", length(model$index)/length(y), "\n\n")
  
  # Visualize
  p <- visualize_decision_boundary(X, y, model, "Linear SVM: Separable Case")
  print(p)
  
  return(model)
}

# Demonstrate non-separable case
demonstrate_nonseparable_case <- function() {
  cat("=== Nonlinear SVM: Non-Separable Case ===\n\n")
  
  # Generate data
  data <- generate_nonseparable_data()
  X <- data$X
  y <- data$y
  
  # Fit linear SVM
  linear_model <- svm(X, y, kernel = "linear", scale = FALSE)
  linear_accuracy <- mean(predict(linear_model, X) == y)
  
  # Fit RBF SVM
  rbf_model <- svm(X, y, kernel = "radial", scale = FALSE)
  rbf_accuracy <- mean(predict(rbf_model, X) == y)
  
  cat("Linear SVM Accuracy:", linear_accuracy, "\n")
  cat("RBF SVM Accuracy:", rbf_accuracy, "\n")
  cat("RBF SVM Support Vectors:", length(rbf_model$index), "\n\n")
  
  # Visualize both
  p1 <- visualize_decision_boundary(X, y, linear_model, "Linear SVM")
  p2 <- visualize_decision_boundary(X, y, rbf_model, "RBF SVM")
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(rbf_model)
}

# Demonstrate soft margin
demonstrate_soft_margin <- function() {
  cat("=== Soft Margin SVM ===\n\n")
  
  # Generate overlapping data
  data <- generate_overlapping_data()
  X <- data$X
  y <- data$y
  
  # Try different C values
  C_values <- c(0.1, 1.0, 10.0, 100.0)
  models <- list()
  plots <- list()
  
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- svm(X, y, kernel = "linear", cost = C, scale = FALSE)
    models[[i]] <- model
    
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    
    title <- paste("C =", C, "\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_decision_boundary(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print summary
  cat("Summary:\n")
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- models[[i]]
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    cat(sprintf("C = %6.1f: Accuracy = %.3f, Support Vectors = %d\n", C, accuracy, n_sv))
  }
  
  return(models)
}

# Demonstrate kernels
demonstrate_kernels <- function() {
  cat("=== Kernel Comparison ===\n\n")
  
  # Generate non-separable data
  data <- generate_nonseparable_data()
  X <- data$X
  y <- data$y
  
  # Define kernels to test
  kernels <- c("linear", "polynomial", "radial", "sigmoid")
  plots <- list()
  
  for (i in seq_along(kernels)) {
    kernel <- kernels[i]
    model <- svm(X, y, kernel = kernel, scale = FALSE)
    
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    
    title <- paste(toupper(kernel), "Kernel\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_decision_boundary(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
}

# Demonstrate hyperparameter tuning
demonstrate_hyperparameter_tuning <- function() {
  cat("=== Hyperparameter Tuning ===\n\n")
  
  # Generate data
  data <- generate_nonseparable_data(n_samples = 200)
  X <- data$X
  y <- data$y
  
  # Create data frame for tuning
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y))
  
  # Tune parameters
  tuned_model <- tune(svm, y ~ ., data = df, 
                     ranges = list(cost = c(0.1, 1, 10, 100),
                                  gamma = c(0.1, 0.5, 1, 2)),
                     kernel = "radial")
  
  cat("Best parameters:\n")
  print(tuned_model$best.parameters)
  cat("Best performance:", tuned_model$best.performance, "\n")
  
  # Visualize best model
  best_model <- tuned_model$best.model
  p <- visualize_decision_boundary(X, y, best_model, 
                                  "Best SVM (Tuned Parameters)")
  print(p)
  
  return(tuned_model)
}

# Demonstrate margin analysis
demonstrate_margin_analysis <- function() {
  cat("=== Margin Analysis ===\n\n")
  
  # Generate separable data
  data <- generate_separable_data()
  X <- data$X
  y <- data$y
  
  # Try different C values
  C_values <- c(0.1, 1.0, 10.0, 100.0)
  models <- list()
  plots <- list()
  
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- svm(X, y, kernel = "linear", cost = C, scale = FALSE)
    models[[i]] <- model
    
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    
    # Calculate margin (approximate)
    w <- model$coefs
    margin <- 2 / sqrt(sum(w^2))
    
    title <- paste("C =", C, "\nMargin:", round(margin, 3), 
                   "\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_decision_boundary(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print margin analysis
  cat("Margin Analysis:\n")
  for (i in seq_along(C_values)) {
    C <- C_values[i]
    model <- models[[i]]
    w <- model$coefs
    margin <- 2 / sqrt(sum(w^2))
    n_sv <- length(model$index)
    cat(sprintf("C = %6.1f: Margin = %.3f, Support Vectors = %d\n", C, margin, n_sv))
  }
}

# Demonstrate theoretical properties
demonstrate_theoretical_properties <- function() {
  cat("=== Theoretical Properties ===\n\n")
  
  # Generate data
  data <- generate_separable_data(n_samples = 200)
  X <- data$X
  y <- data$y
  
  # Fit SVM
  model <- svm(X, y, kernel = "linear", scale = FALSE)
  
  # Extract parameters
  w <- model$coefs
  b <- model$rho
  support_vectors <- model$SV
  support_vector_indices <- model$index
  
  cat("SVM Parameters:\n")
  cat("Weight vector w:", w, "\n")
  cat("Bias term b:", b, "\n")
  cat("Number of support vectors:", length(support_vectors), "\n")
  
  # Verify KKT conditions
  cat("\nKKT Conditions Verification:\n")
  
  # Calculate decision function values
  decision_values <- predict(model, X, decision.values = TRUE)
  decision_values <- attr(decision_values, "decision.values")
  
  # Check complementary slackness
  alpha <- abs(model$coefs)  # Dual coefficients
  margin_violations <- y * decision_values - 1
  
  cat("Complementary slackness check:\n")
  cat("  α_i * (y_i * f(x_i) - 1) should be 0 for all i\n")
  
  for (i in seq_along(support_vector_indices)) {
    sv_idx <- support_vector_indices[i]
    alpha_val <- alpha[i] if i <= length(alpha) else 0
    margin_val <- margin_violations[sv_idx]
    product <- alpha_val * margin_val
    cat(sprintf("  Support vector %d: α = %.4f, margin = %.4f, product = %.6f\n", 
                sv_idx, alpha_val, margin_val, product))
  }
  
  return(model)
}

# Demonstrate scalability analysis
demonstrate_scalability_analysis <- function() {
  cat("=== Scalability Analysis ===\n\n")
  
  # Test different dataset sizes
  sizes <- c(50, 100, 200, 500, 1000)
  results <- list()
  
  for (i in seq_along(sizes)) {
    size <- sizes[i]
    cat("Testing with", size, "samples...\n")
    
    # Generate data
    data <- generate_separable_data(n_samples = size)
    X <- data$X
    y <- data$y
    
    # Time the fitting
    start_time <- Sys.time()
    model <- svm(X, y, kernel = "linear", scale = FALSE)
    fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Time the prediction
    start_time <- Sys.time()
    y_pred <- predict(model, X)
    predict_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Calculate metrics
    accuracy <- mean(y_pred == y)
    n_sv <- length(model$index)
    sv_ratio <- n_sv / size
    
    results[[i]] <- list(
      size = size,
      fit_time = fit_time,
      predict_time = predict_time,
      accuracy = accuracy,
      n_sv = n_sv,
      sv_ratio = sv_ratio
    )
    
    cat("  Fit time:", round(fit_time, 4), "s\n")
    cat("  Predict time:", round(predict_time, 4), "s\n")
    cat("  Accuracy:", round(accuracy, 4), "\n")
    cat("  Support vectors:", n_sv, "(", round(sv_ratio, 3), ")\n\n")
  }
  
  # Create plots
  sizes <- sapply(results, function(x) x$size)
  fit_times <- sapply(results, function(x) x$fit_time)
  predict_times <- sapply(results, function(x) x$predict_time)
  accuracies <- sapply(results, function(x) x$accuracy)
  sv_ratios <- sapply(results, function(x) x$sv_ratio)
  
  # Create data frames for plotting
  fit_df <- data.frame(size = sizes, time = fit_times, type = "Fit")
  predict_df <- data.frame(size = sizes, time = predict_times, type = "Predict")
  time_df <- rbind(fit_df, predict_df)
  
  accuracy_df <- data.frame(size = sizes, accuracy = accuracies)
  sv_ratio_df <- data.frame(size = sizes, ratio = sv_ratios)
  
  # Create plots
  p1 <- ggplot(time_df, aes(x = size, y = time, color = type)) +
    geom_line() + geom_point() +
    labs(title = "Time vs Dataset Size", x = "Dataset Size", y = "Time (s)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p2 <- ggplot(accuracy_df, aes(x = size, y = accuracy)) +
    geom_line(color = "green") + geom_point(color = "green") +
    labs(title = "Accuracy vs Dataset Size", x = "Dataset Size", y = "Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p3 <- ggplot(sv_ratio_df, aes(x = size, y = ratio)) +
    geom_line(color = "purple") + geom_point(color = "purple") +
    labs(title = "Support Vector Ratio vs Dataset Size", 
         x = "Dataset Size", y = "Support Vector Ratio") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 2)
  
  return(results)
}

# Main function to demonstrate SVM introduction
main_r <- function() {
  cat("Support Vector Machines: Introduction and Implementation\n")
  cat("=" * 60, "\n")
  
  # 1. Separable case
  cat("\n1. Linear SVM: Separable Case\n")
  separable_model <- demonstrate_separable_case()
  
  # 2. Non-separable case
  cat("\n2. Nonlinear SVM: Non-Separable Case\n")
  nonseparable_model <- demonstrate_nonseparable_case()
  
  # 3. Soft margin
  cat("\n3. Soft Margin SVM\n")
  soft_margin_models <- demonstrate_soft_margin()
  
  # 4. Kernel comparison
  cat("\n4. Kernel Comparison\n")
  demonstrate_kernels()
  
  # 5. Hyperparameter tuning
  cat("\n5. Hyperparameter Tuning\n")
  tuned_model <- demonstrate_hyperparameter_tuning()
  
  # 6. Margin analysis
  cat("\n6. Margin Analysis\n")
  demonstrate_margin_analysis()
  
  # 7. Theoretical properties
  cat("\n7. Theoretical Properties\n")
  theoretical_model <- demonstrate_theoretical_properties()
  
  # 8. Scalability analysis
  cat("\n8. Scalability Analysis\n")
  scalability_results <- demonstrate_scalability_analysis()
  
  # Support vector analysis
  cat("\n=== Support Vector Analysis ===\n")
  data_types <- list(
    separable = generate_separable_data(),
    nonseparable = generate_nonseparable_data(),
    overlapping = generate_overlapping_data()
  )
  
  for (name in names(data_types)) {
    data <- data_types[[name]]
    model <- svm(data$X, data$y, kernel = "radial", scale = FALSE)
    
    n_sv <- length(model$index)
    sv_ratio <- n_sv / length(data$y)
    
    cat(name, "data:\n")
    cat("  Total samples:", length(data$y), "\n")
    cat("  Support vectors:", n_sv, "\n")
    cat("  SV ratio:", round(sv_ratio, 3), "\n\n")
  }
  
  cat("=== Key Insights ===\n")
  cat("1. SVM maximizes margin for better generalization\n")
  cat("2. Only support vectors influence the decision boundary\n")
  cat("3. Kernel trick enables nonlinear classification\n")
  cat("4. Parameter C controls margin vs error trade-off\n")
  cat("5. SVM provides sparse, robust solutions\n")
  cat("6. Theoretical foundations in structural risk minimization\n")
  
  return(list(
    separable_model = separable_model,
    nonseparable_model = nonseparable_model,
    soft_margin_models = soft_margin_models,
    tuned_model = tuned_model,
    theoretical_model = theoretical_model,
    scalability_results = scalability_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
