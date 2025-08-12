# Support Vector Machines: Separable Case Implementation

# Load required libraries
library(e1071)
library(ggplot2)
library(gridExtra)
library(kernlab)

# Generate separable data
generate_separable_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate two classes with clear separation
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

# Visualize SVM decision boundary
visualize_svm_decision_boundary <- function(X, y, model, title = "SVM Decision Boundary") {
  # Create grid
  x_min <- min(X[, 1]) - 1
  x_max <- max(X[, 1]) + 1
  y_min <- min(X[, 2]) - 1
  y_max <- max(X[, 2]) + 1
  
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
  
  # Highlight support vectors if available
  if (!is.null(model$index)) {
    support_vectors <- data.frame(X1 = X[model$index, 1], X2 = X[model$index, 2])
    p <- p + geom_point(data = support_vectors, aes(x = X1, y = X2), 
                        shape = 21, size = 4, fill = "transparent", color = "black")
  }
  
  return(p)
}

# Demonstrate separable case
demonstrate_separable_case <- function() {
  cat("=== SVM Separable Case Demonstration ===\n\n")
  
  # Generate separable data
  data <- generate_separable_data(n_samples = 100, random_state = 42)
  X <- data$X
  y <- data$y
  
  cat("Data shape:", dim(X), "\n")
  cat("Class distribution:", table(y), "\n")
  
  # Fit SVM
  svm_model <- svm(X, y, kernel = "linear", scale = FALSE)
  
  # Evaluate
  y_pred <- predict(svm_model, X)
  accuracy <- mean(y_pred == y)
  
  cat("\nSVM Results:\n")
  cat("Accuracy:", accuracy, "\n")
  cat("Number of support vectors:", length(svm_model$index), "\n")
  cat("Support vector ratio:", length(svm_model$index)/length(y), "\n")
  
  # Visualize
  p <- visualize_svm_decision_boundary(X, y, svm_model, "SVM Decision Boundary")
  print(p)
  
  return(list(model = svm_model, X = X, y = y, accuracy = accuracy))
}

# Demonstrate KKT conditions
demonstrate_kkt_conditions <- function(X, y, svm_model) {
  cat("\n=== KKT Conditions Verification ===\n\n")
  
  # Get decision function values
  decision_values <- predict(svm_model, X, decision.values = TRUE)
  decision_values <- attr(decision_values, "decision.values")
  
  # Check primal feasibility: y_i * f(x_i) >= 1
  primal_violations <- y * decision_values < 1
  cat("Primal feasibility violations:", sum(primal_violations), "\n")
  
  # Check support vector conditions
  support_vector_indices <- svm_model$index
  non_support_vector_indices <- setdiff(1:length(y), support_vector_indices)
  
  cat("\nSupport Vector Analysis:\n")
  cat("  Total points:", length(y), "\n")
  cat("  Support vectors:", length(support_vector_indices), "\n")
  cat("  Non-support vectors:", length(non_support_vector_indices), "\n")
  
  # Check that support vectors lie on margin
  sv_margin_values <- y[support_vector_indices] * decision_values[support_vector_indices]
  cat("  Support vector margin values:", round(sv_margin_values, 6), "\n")
  
  # Check that non-support vectors are beyond margin
  non_sv_margin_values <- y[non_support_vector_indices] * decision_values[non_support_vector_indices]
  cat("  Non-support vector margin values (min, mean):", 
      round(min(non_sv_margin_values), 6), ",", round(mean(non_sv_margin_values), 6), "\n")
}

# Demonstrate margin analysis
demonstrate_margin_analysis <- function() {
  cat("\n=== Margin Analysis ===\n\n")
  
  # Test different data separations
  separations <- c(1.0, 2.0, 3.0, 4.0)
  results <- list()
  
  for (i in seq_along(separations)) {
    sep <- separations[i]
    cat("Testing separation", sep, "...\n")
    
    # Generate data with different separations
    set.seed(42)
    n <- 100
    n_class1 <- n %/% 2
    n_class2 <- n - n_class1
    
    # Class 1: centered at (sep, sep)
    class1 <- matrix(rnorm(n_class1 * 2), n_class1, 2) + matrix(c(sep, sep), n_class1, 2, byrow = TRUE)
    
    # Class 2: centered at (-sep, -sep)
    class2 <- matrix(rnorm(n_class2 * 2), n_class2, 2) + matrix(c(-sep, -sep), n_class2, 2, byrow = TRUE)
    
    X <- rbind(class1, class2)
    y <- c(rep(1, n_class1), rep(-1, n_class2))
    
    # Fit SVM
    svm_model <- svm(X, y, kernel = "linear", scale = FALSE)
    
    # Calculate metrics
    accuracy <- mean(predict(svm_model, X) == y)
    n_sv <- length(svm_model$index)
    
    # Estimate margin (approximate)
    w <- svm_model$coefs
    margin <- 2 / sqrt(sum(w^2))
    
    results[[i]] <- list(
      separation = sep,
      margin = margin,
      n_support_vectors = n_sv,
      accuracy = accuracy
    )
    
    cat("  Margin:", round(margin, 4), "SVs:", n_sv, "Accuracy:", round(accuracy, 4), "\n")
  }
  
  # Create plots
  separations <- sapply(results, function(x) x$separation)
  margins <- sapply(results, function(x) x$margin)
  n_svs <- sapply(results, function(x) x$n_support_vectors)
  accuracies <- sapply(results, function(x) x$accuracy)
  
  # Create data frames for plotting
  margin_df <- data.frame(separation = separations, margin = margins)
  sv_df <- data.frame(separation = separations, n_support_vectors = n_svs)
  accuracy_df <- data.frame(separation = separations, accuracy = accuracies)
  
  # Create plots
  p1 <- ggplot(margin_df, aes(x = separation, y = margin)) +
    geom_line(color = "blue") + geom_point(color = "blue") +
    labs(title = "Margin vs Data Separation", x = "Data Separation", y = "Margin Width") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p2 <- ggplot(sv_df, aes(x = separation, y = n_support_vectors)) +
    geom_line(color = "red") + geom_point(color = "red") +
    labs(title = "Support Vectors vs Data Separation", 
         x = "Data Separation", y = "Number of Support Vectors") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p3 <- ggplot(accuracy_df, aes(x = separation, y = accuracy)) +
    geom_line(color = "green") + geom_point(color = "green") +
    labs(title = "Accuracy vs Data Separation", 
         x = "Data Separation", y = "Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(results)
}

# Demonstrate computational complexity
demonstrate_computational_complexity <- function() {
  cat("\n=== Computational Complexity Analysis ===\n\n")
  
  # Test different dataset sizes
  sizes <- c(50, 100, 200, 300, 400)
  results <- list()
  
  for (i in seq_along(sizes)) {
    size <- sizes[i]
    cat("Testing with", size, "samples...\n")
    
    # Generate data
    data <- generate_separable_data(n_samples = size, random_state = 42)
    X <- data$X
    y <- data$y
    
    # Time the fitting
    start_time <- Sys.time()
    svm_model <- svm(X, y, kernel = "linear", scale = FALSE)
    fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Time the prediction
    start_time <- Sys.time()
    y_pred <- predict(svm_model, X)
    predict_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    # Calculate metrics
    accuracy <- mean(y_pred == y)
    n_sv <- length(svm_model$index)
    
    results[[i]] <- list(
      size = size,
      fit_time = fit_time,
      predict_time = predict_time,
      n_support_vectors = n_sv,
      accuracy = accuracy
    )
    
    cat("  Fit time:", round(fit_time, 4), "s\n")
    cat("  Predict time:", round(predict_time, 4), "s\n")
    cat("  Support vectors:", n_sv, "\n")
    cat("  Accuracy:", round(accuracy, 4), "\n")
  }
  
  # Create plots
  sizes <- sapply(results, function(x) x$size)
  fit_times <- sapply(results, function(x) x$fit_time)
  predict_times <- sapply(results, function(x) x$predict_time)
  n_svs <- sapply(results, function(x) x$n_support_vectors)
  
  # Create data frames for plotting
  fit_df <- data.frame(size = sizes, time = fit_times, type = "Fit")
  predict_df <- data.frame(size = sizes, time = predict_times, type = "Predict")
  time_df <- rbind(fit_df, predict_df)
  
  sv_df <- data.frame(size = sizes, n_support_vectors = n_svs)
  
  # Create plots
  p1 <- ggplot(time_df, aes(x = size, y = time, color = type)) +
    geom_line() + geom_point() +
    labs(title = "Time vs Dataset Size", x = "Dataset Size", y = "Time (s)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p2 <- ggplot(sv_df, aes(x = size, y = n_support_vectors)) +
    geom_line(color = "green") + geom_point(color = "green") +
    labs(title = "Support Vectors vs Dataset Size", 
         x = "Dataset Size", y = "Number of Support Vectors") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(results)
}

# Demonstrate theoretical properties
demonstrate_theoretical_properties <- function() {
  cat("\n=== Theoretical Properties ===\n\n")
  
  # Generate data
  data <- generate_separable_data(n_samples = 200, random_state = 42)
  X <- data$X
  y <- data$y
  
  # Fit SVM
  svm_model <- svm(X, y, kernel = "linear", scale = FALSE)
  
  cat("1. Maximum Margin Property:\n")
  # Estimate margin
  w <- svm_model$coefs
  margin <- 2 / sqrt(sum(w^2))
  cat("   Margin width:", round(margin, 4), "\n")
  
  # Check that all points are correctly classified
  y_pred <- predict(svm_model, X)
  accuracy <- mean(y_pred == y)
  cat("   Classification accuracy:", round(accuracy, 4), "\n")
  
  cat("\n2. Support Vector Property:\n")
  cat("   Support vectors lie exactly on margin boundaries\n")
  support_vector_indices <- svm_model$index
  cat("   Number of support vectors:", length(support_vector_indices), "\n")
  
  cat("\n3. Sparsity Property:\n")
  n_sv <- length(support_vector_indices)
  n_total <- length(y)
  cat("   Total points:", n_total, "\n")
  cat("   Support vectors:", n_sv, "\n")
  cat("   Sparsity ratio:", round(n_sv/n_total, 3), "\n")
  
  cat("\n4. Dual Formulation:\n")
  cat("   Dual formulation enables kernel trick\n")
  cat("   Only inner products between data points are needed\n")
  
  return(svm_model)
}

# Demonstrate comparison with other methods
demonstrate_comparison <- function() {
  cat("\n=== Comparison with Other Methods ===\n\n")
  
  # Generate data
  data <- generate_separable_data(n_samples = 100, random_state = 42)
  X <- data$X
  y <- data$y
  
  # Fit different models
  cat("Fitting models...\n")
  
  # SVM
  svm_model <- svm(X, y, kernel = "linear", scale = FALSE)
  svm_accuracy <- mean(predict(svm_model, X) == y)
  
  # Linear discriminant analysis (if MASS is available)
  if (require(MASS, quietly = TRUE)) {
    lda_model <- lda(X, y)
    lda_pred <- predict(lda_model, X)$class
    lda_accuracy <- mean(lda_pred == y)
    cat("LDA accuracy:", round(lda_accuracy, 4), "\n")
  } else {
    lda_accuracy <- NA
    cat("LDA: MASS package not available\n")
  }
  
  # Logistic regression
  glm_model <- glm(y ~ X, family = binomial)
  glm_pred <- ifelse(predict(glm_model, X) > 0, 1, -1)
  glm_accuracy <- mean(glm_pred == y)
  
  cat("SVM accuracy:", round(svm_accuracy, 4), "\n")
  cat("Logistic regression accuracy:", round(glm_accuracy, 4), "\n")
  
  # Compare decision boundaries
  cat("\nDecision boundary comparison:\n")
  cat("SVM finds maximum margin hyperplane\n")
  cat("LDA finds optimal linear discriminant under normality assumption\n")
  cat("Logistic regression finds optimal decision boundary for classification\n")
  
  return(list(
    svm_accuracy = svm_accuracy,
    lda_accuracy = lda_accuracy,
    glm_accuracy = glm_accuracy
  ))
}

# Demonstrate advantages and limitations
demonstrate_advantages_limitations <- function() {
  cat("\n=== Advantages and Limitations ===\n\n")
  
  cat("Advantages:\n")
  cat("1. Maximum Margin: Provides good generalization\n")
  cat("2. Sparsity: Only support vectors matter\n")
  cat("3. Kernel Trick: Can handle non-linear decision boundaries\n")
  cat("4. Theoretical Guarantees: Based on solid optimization theory\n")
  cat("5. Robust: Less sensitive to small perturbations\n\n")
  
  cat("Limitations:\n")
  cat("1. Computational Cost: Scales poorly with dataset size (O(n³))\n")
  cat("2. Memory Requirements: Needs to store kernel matrix (O(n²))\n")
  cat("3. Sensitivity to Scaling: Features should be scaled\n")
  cat("4. Binary Classification: Need extensions for multi-class\n")
  cat("5. Assumes Separability: May not work well with overlapping classes\n")
  
  # Demonstrate scaling sensitivity
  cat("\nScaling Sensitivity Demonstration:\n")
  
  # Generate data
  data <- generate_separable_data(n_samples = 100, random_state = 42)
  X <- data$X
  y <- data$y
  
  # Fit SVM without scaling
  svm_unscaled <- svm(X, y, kernel = "linear", scale = FALSE)
  accuracy_unscaled <- mean(predict(svm_unscaled, X) == y)
  
  # Fit SVM with scaling
  svm_scaled <- svm(X, y, kernel = "linear", scale = TRUE)
  accuracy_scaled <- mean(predict(svm_scaled, X) == y)
  
  cat("Accuracy without scaling:", round(accuracy_unscaled, 4), "\n")
  cat("Accuracy with scaling:", round(accuracy_scaled, 4), "\n")
  
  return(list(
    accuracy_unscaled = accuracy_unscaled,
    accuracy_scaled = accuracy_scaled
  ))
}

# Main function to demonstrate SVM separable case
main_r <- function() {
  cat("Support Vector Machines: Separable Case Implementation\n")
  cat("=" * 60, "\n")
  
  # 1. Basic separable case demonstration
  cat("\n1. Basic Separable Case Demonstration:\n")
  separable_results <- demonstrate_separable_case()
  
  # 2. KKT conditions verification
  cat("\n2. KKT Conditions Verification:\n")
  demonstrate_kkt_conditions(separable_results$X, separable_results$y, separable_results$model)
  
  # 3. Margin analysis
  cat("\n3. Margin Analysis:\n")
  margin_results <- demonstrate_margin_analysis()
  
  # 4. Computational complexity
  cat("\n4. Computational Complexity Analysis:\n")
  complexity_results <- demonstrate_computational_complexity()
  
  # 5. Theoretical properties
  cat("\n5. Theoretical Properties:\n")
  theoretical_model <- demonstrate_theoretical_properties()
  
  # 6. Comparison with other methods
  cat("\n6. Comparison with Other Methods:\n")
  comparison_results <- demonstrate_comparison()
  
  # 7. Advantages and limitations
  cat("\n7. Advantages and Limitations:\n")
  advantages_results <- demonstrate_advantages_limitations()
  
  cat("\n=== Key Insights ===\n")
  cat("1. SVM finds the optimal hyperplane with maximum margin\n")
  cat("2. Only support vectors determine the decision boundary\n")
  cat("3. KKT conditions provide theoretical foundation\n")
  cat("4. Dual formulation enables kernel trick\n")
  cat("5. Computational complexity is O(n³) for training\n")
  cat("6. Prediction complexity is O(n_sv * p)\n")
  cat("7. Scaling is important for SVM performance\n")
  
  return(list(
    separable_results = separable_results,
    margin_results = margin_results,
    complexity_results = complexity_results,
    theoretical_model = theoretical_model,
    comparison_results = comparison_results,
    advantages_results = advantages_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
