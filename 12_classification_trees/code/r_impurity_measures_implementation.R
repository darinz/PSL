# Impurity Measures: Implementation and Analysis

# Load required libraries
library(ggplot2)
library(gridExtra)
library(dplyr)
library(plotly)

# Impurity measure functions
gini_impurity <- function(p) {
  return(1 - sum(p^2))
}

entropy_impurity <- function(p) {
  return(-sum(p * log2(p + 1e-10)))
}

misclassification_impurity <- function(p) {
  return(1 - max(p))
}

# Visualize impurity measures
plot_impurity_measures <- function() {
  cat("=== Impurity Measures Visualization ===\n\n")
  
  # Binary classification
  p1 <- seq(0, 1, length.out = 100)
  p2 <- 1 - p1
  
  # Calculate impurity measures
  gini <- 1 - (p1^2 + p2^2)
  entropy <- -p1 * log2(p1 + 1e-10) - p2 * log2(p2 + 1e-10)
  misclassification <- 1 - pmax(p1, p2)
  
  # Create data frame for plotting
  plot_data <- data.frame(
    p1 = p1,
    gini = gini,
    entropy = entropy,
    misclassification = misclassification
  )
  
  # Plot binary classification
  p1 <- ggplot(plot_data, aes(x = p1)) +
    geom_line(aes(y = gini, color = "Gini"), size = 1) +
    geom_line(aes(y = entropy, color = "Entropy"), size = 1) +
    geom_line(aes(y = misclassification, color = "Misclassification"), size = 1) +
    labs(title = "Impurity Measures for Binary Classification",
         x = "Probability of Class 1 (p₁)",
         y = "Impurity",
         color = "Measure") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Ternary classification comparison
  distributions <- list(
    c(1.0, 0.0, 0.0),  # Pure class 1
    c(0.8, 0.1, 0.1),  # Mostly class 1
    c(0.5, 0.3, 0.2),  # Mixed
    c(0.33, 0.33, 0.34),  # Nearly uniform
    c(0.33, 0.33, 0.33)   # Uniform
  )
  
  labels <- c('Pure', 'Mostly 1', 'Mixed', 'Near Uniform', 'Uniform')
  
  gini_values <- sapply(distributions, function(p) gini_impurity(p))
  entropy_values <- sapply(distributions, function(p) entropy_impurity(p))
  misclass_values <- sapply(distributions, function(p) misclassification_impurity(p))
  
  comparison_data <- data.frame(
    distribution = rep(labels, 3),
    impurity = c(gini_values, entropy_values, misclass_values),
    measure = rep(c("Gini", "Entropy", "Misclassification"), each = length(labels))
  )
  
  p2 <- ggplot(comparison_data, aes(x = distribution, y = impurity, fill = measure)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Comparison of Impurity Measures",
         x = "Distribution Type",
         y = "Impurity Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  cat("Impurity measures visualization completed.\n")
  cat("Key observations:\n")
  cat("1. All measures reach minimum at pure distributions\n")
  cat("2. All measures reach maximum at uniform distributions\n")
  cat("3. Entropy is more sensitive to small changes\n")
  cat("4. Misclassification error is piecewise linear\n")
  
  return(list(binary_plot = p1, comparison_plot = p2))
}

# Calculate split gain
calculate_split_gain <- function(X, y, feature, threshold, impurity_func) {
  # Split the data
  left_mask <- X[, feature] <= threshold
  right_mask <- !left_mask
  
  # Calculate class frequencies for parent and children
  parent_counts <- table(y)
  parent_probs <- parent_counts / length(y)
  
  left_counts <- table(y[left_mask])
  left_probs <- left_counts / sum(left_mask)
  
  right_counts <- table(y[right_mask])
  right_probs <- right_counts / sum(right_mask)
  
  # Calculate impurity
  parent_impurity <- impurity_func(parent_probs)
  left_impurity <- impurity_func(left_probs)
  right_impurity <- impurity_func(right_probs)
  
  # Calculate proportions
  p_left <- sum(left_mask) / length(y)
  p_right <- sum(right_mask) / length(y)
  
  # Calculate gain
  gain <- parent_impurity - (p_left * left_impurity + p_right * right_impurity)
  
  return(gain)
}

# Find best split
find_best_split <- function(X, y, impurity_func) {
  n_samples <- nrow(X)
  n_features <- ncol(X)
  best_gain <- 0
  best_feature <- NULL
  best_threshold <- NULL
  
  for (feature in 1:n_features) {
    # Get unique values for this feature
    thresholds <- unique(X[, feature])
    
    for (threshold in thresholds) {
      gain <- calculate_split_gain(X, y, feature, threshold, impurity_func)
      
      if (gain > best_gain) {
        best_gain <- gain
        best_feature <- feature
        best_threshold <- threshold
      }
    }
  }
  
  return(list(feature = best_feature, threshold = best_threshold, gain = best_gain))
}

# Demonstrate split gain calculation
demonstrate_split_gain <- function() {
  cat("=== Split Gain Demonstration ===\n\n")
  
  # Create simple example data
  X <- matrix(c(1, 2, 2, 3, 3, 1, 4, 2, 5, 3, 6, 1), ncol = 2, byrow = TRUE)
  y <- factor(c(0, 0, 0, 1, 1, 1))
  
  cat("Data:\n")
  for (i in 1:nrow(X)) {
    cat(sprintf("  Sample %d: X=(%.1f, %.1f), y=%s\n", i, X[i, 1], X[i, 2], y[i]))
  }
  
  # Test different splits
  cat("\nTesting different splits:\n")
  cat("Feature | Threshold | Gain (Gini) | Gain (Entropy) | Gain (Misclass)\n")
  cat(paste(rep("-", 70), collapse = ""), "\n")
  
  for (feature in 1:2) {
    thresholds <- unique(X[, feature])
    for (threshold in thresholds) {
      gain_gini <- calculate_split_gain(X, y, feature, threshold, gini_impurity)
      gain_entropy <- calculate_split_gain(X, y, feature, threshold, entropy_impurity)
      gain_misclass <- calculate_split_gain(X, y, feature, threshold, misclassification_impurity)
      
      cat(sprintf("   %d    |    %.1f    |    %.3f     |     %.3f     |     %.3f\n", 
                  feature, threshold, gain_gini, gain_entropy, gain_misclass))
    }
  }
  
  # Find best split for each impurity measure
  cat("\nBest splits for each impurity measure:\n")
  best_gini <- find_best_split(X, y, gini_impurity)
  best_entropy <- find_best_split(X, y, entropy_impurity)
  best_misclass <- find_best_split(X, y, misclassification_impurity)
  
  cat(sprintf("Gini: Feature %d <= %.1f, Gain = %.3f\n", 
              best_gini$feature, best_gini$threshold, best_gini$gain))
  cat(sprintf("Entropy: Feature %d <= %.1f, Gain = %.3f\n", 
              best_entropy$feature, best_entropy$threshold, best_entropy$gain))
  cat(sprintf("Misclassification: Feature %d <= %.1f, Gain = %.3f\n", 
              best_misclass$feature, best_misclass$threshold, best_misclass$gain))
  
  return(list(X = X, y = y, 
              best_gini = best_gini, 
              best_entropy = best_entropy, 
              best_misclass = best_misclass))
}

# Compare impurity measures
compare_impurity_measures <- function() {
  cat("=== Impurity Measures Comparison ===\n\n")
  
  # Test different probability distributions
  test_distributions <- list(
    c(1.0, 0.0, 0.0),      # Pure
    c(0.9, 0.05, 0.05),    # Nearly pure
    c(0.7, 0.2, 0.1),      # Mixed
    c(0.5, 0.3, 0.2),      # More mixed
    c(0.4, 0.3, 0.3),      # Nearly uniform
    c(1/3, 1/3, 1/3)       # Uniform
  )
  
  labels <- c('Pure', 'Nearly Pure', 'Mixed', 'More Mixed', 'Near Uniform', 'Uniform')
  
  results <- matrix(0, nrow = length(test_distributions), ncol = 3)
  
  for (i in seq_along(test_distributions)) {
    dist <- test_distributions[[i]]
    results[i, 1] <- gini_impurity(dist)
    results[i, 2] <- entropy_impurity(dist)
    results[i, 3] <- misclassification_impurity(dist)
  }
  
  # Create comparison plot
  comparison_data <- data.frame(
    distribution = rep(labels, 3),
    impurity = c(results[, 1], results[, 2], results[, 3]),
    measure = rep(c("Gini", "Entropy", "Misclassification"), each = length(labels))
  )
  
  p <- ggplot(comparison_data, aes(x = distribution, y = impurity, fill = measure)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Comparison of Impurity Measures",
         x = "Distribution Type",
         y = "Impurity Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
  
  # Print numerical comparison
  cat("Numerical Comparison of Impurity Measures:\n")
  cat(sprintf("%-15s %-8s %-8s %-8s\n", "Distribution", "Gini", "Entropy", "Misclass"))
  cat(paste(rep("-", 45), collapse = ""), "\n")
  for (i in seq_along(labels)) {
    cat(sprintf("%-15s %-8.3f %-8.3f %-8.3f\n", 
                labels[i], results[i, 1], results[i, 2], results[i, 3]))
  }
  
  return(list(results = results, labels = labels, plot = p))
}

# Analyze impurity properties
analyze_impurity_properties <- function() {
  cat("=== Impurity Measures Properties Analysis ===\n\n")
  
  # Test symmetry property
  p1 <- c(0.3, 0.5, 0.2)
  p2 <- c(0.5, 0.2, 0.3)  # Permutation of p1
  
  cat("Symmetry Property Test:\n")
  cat("Original distribution:", p1, "\n")
  cat("Permuted distribution:", p2, "\n")
  cat(sprintf("Gini - Original: %.4f, Permuted: %.4f\n", 
              gini_impurity(p1), gini_impurity(p2)))
  cat(sprintf("Entropy - Original: %.4f, Permuted: %.4f\n", 
              entropy_impurity(p1), entropy_impurity(p2)))
  cat(sprintf("Misclass - Original: %.4f, Permuted: %.4f\n", 
              misclassification_impurity(p1), misclassification_impurity(p2)))
  cat("\n")
  
  # Test concavity
  cat("Concavity Analysis:\n")
  cat("Entropy is strictly concave, encouraging pure splits\n")
  cat("Gini is also concave but less strict than entropy\n")
  cat("Misclassification error is not differentiable at all points\n")
  cat("\n")
  
  # Test sensitivity to small changes
  p_base <- c(0.5, 0.3, 0.2)
  p_perturbed <- c(0.51, 0.29, 0.2)
  
  cat("Sensitivity to Small Changes:\n")
  cat("Base distribution:", p_base, "\n")
  cat("Perturbed distribution:", p_perturbed, "\n")
  cat(sprintf("Gini change: %.6f\n", abs(gini_impurity(p_base) - gini_impurity(p_perturbed))))
  cat(sprintf("Entropy change: %.6f\n", abs(entropy_impurity(p_base) - entropy_impurity(p_perturbed))))
  cat(sprintf("Misclass change: %.6f\n", abs(misclassification_impurity(p_base) - misclassification_impurity(p_perturbed))))
  
  # Visualize concavity
  p1_range <- seq(0, 1, length.out = 100)
  p2_range <- 0.5 * (1 - p1_range)
  p3_range <- 0.5 * (1 - p1_range)
  
  gini_values <- numeric(length(p1_range))
  entropy_values <- numeric(length(p1_range))
  misclass_values <- numeric(length(p1_range))
  
  for (i in seq_along(p1_range)) {
    p1 <- p1_range[i]
    p2 <- p2_range[i]
    p3 <- p3_range[i]
    
    if (p1 + p2 + p3 <= 1) {
      gini_values[i] <- gini_impurity(c(p1, p2, p3))
      entropy_values[i] <- entropy_impurity(c(p1, p2, p3))
      misclass_values[i] <- misclassification_impurity(c(p1, p2, p3))
    } else {
      gini_values[i] <- NA
      entropy_values[i] <- NA
      misclass_values[i] <- NA
    }
  }
  
  concavity_data <- data.frame(
    p1 = rep(p1_range, 3),
    impurity = c(gini_values, entropy_values, misclass_values),
    measure = rep(c("Gini", "Entropy", "Misclassification"), each = length(p1_range))
  )
  
  p <- ggplot(concavity_data, aes(x = p1, y = impurity, color = measure)) +
    geom_line(size = 1) +
    labs(title = "Impurity Functions - Concavity Analysis",
         x = "Probability of Class 1",
         y = "Impurity Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  return(list(concavity_plot = p))
}

# Demonstrate practical considerations
demonstrate_practical_considerations <- function() {
  cat("=== Practical Considerations ===\n\n")
  
  # Generate data with different characteristics
  set.seed(42)
  
  # Create data with clear separation
  X_clear <- matrix(rnorm(200 * 2), ncol = 2)
  y_clear <- factor(ifelse(X_clear[, 1] + X_clear[, 2] > 0, 1, 0))
  
  # Create data with overlap
  X_overlap <- matrix(rnorm(200 * 2), ncol = 2)
  y_overlap <- factor(ifelse(X_overlap[, 1] + X_overlap[, 2] + 0.5 * rnorm(200) > 0, 1, 0))
  
  datasets <- list(
    list(X = X_clear, y = y_clear, name = "Clear Separation"),
    list(X = X_overlap, y = y_overlap, name = "Overlapping Classes")
  )
  
  impurity_funcs <- list(
    list(name = "Gini", func = gini_impurity),
    list(name = "Entropy", func = entropy_impurity),
    list(name = "Misclassification", func = misclassification_impurity)
  )
  
  plots <- list()
  
  for (i in seq_along(datasets)) {
    data <- datasets[[i]]
    X <- data$X
    y <- data$y
    name <- data$name
    
    for (j in seq_along(impurity_funcs)) {
      impurity_info <- impurity_funcs[[j]]
      
      # Find best split
      best_split <- find_best_split(X, y, impurity_info$func)
      
      # Create data frame for plotting
      df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
      
      # Create decision boundary
      x_min <- min(X[, 1]) - 1
      x_max <- max(X[, 1]) + 1
      y_min <- min(X[, 2]) - 1
      y_max <- max(X[, 2]) + 1
      
      grid_x <- seq(x_min, x_max, length.out = 50)
      grid_y <- seq(y_min, y_max, length.out = 50)
      grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
      
      # Make predictions based on best split
      if (best_split$feature == 1) {
        grid_data$pred <- ifelse(grid_data$X1 <= best_split$threshold, 0, 1)
      } else {
        grid_data$pred <- ifelse(grid_data$X2 <= best_split$threshold, 0, 1)
      }
      
      # Create plot
      p <- ggplot() +
        geom_contour(data = grid_data, aes(x = X1, y = X2, z = pred),
                     breaks = 0.5, color = "black", size = 1) +
        geom_point(data = df, aes(x = X1, y = X2, color = y), alpha = 0.7) +
        labs(title = paste(name, "-", impurity_info$name),
             subtitle = paste("Gain:", round(best_split$gain, 3))) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5))
      
      plots[[length(plots) + 1]] <- p
    }
  }
  
  # Display plots in a grid
  do.call(grid.arrange, c(plots, ncol = 3))
  
  # Print analysis
  cat("Analysis of Impurity Measures in Practice:\n")
  cat("1. Gini: Good balance between computational efficiency and performance\n")
  cat("2. Entropy: Strongly encourages pure splits, may lead to overfitting\n")
  cat("3. Misclassification: Direct interpretation but less smooth optimization\n")
  cat("4. Choice depends on data characteristics and computational constraints\n")
  
  return(list(plots = plots))
}

# Main demonstration function
main_r <- function() {
  cat("Impurity Measures: Implementation and Analysis\n")
  cat("=" * 60, "\n")
  
  # 1. Visualize impurity measures
  cat("\n1. Impurity Measures Visualization:\n")
  visualization_results <- plot_impurity_measures()
  
  # 2. Demonstrate split gain calculation
  cat("\n2. Split Gain Demonstration:\n")
  split_results <- demonstrate_split_gain()
  
  # 3. Compare impurity measures
  cat("\n3. Impurity Measures Comparison:\n")
  comparison_results <- compare_impurity_measures()
  
  # 4. Analyze theoretical properties
  cat("\n4. Theoretical Properties Analysis:\n")
  properties_results <- analyze_impurity_properties()
  
  # 5. Practical considerations
  cat("\n5. Practical Considerations:\n")
  practical_results <- demonstrate_practical_considerations()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Gini Index: Most commonly used, differentiable, good balance\n")
  cat("2. Entropy: Strongly encourages pure splits, differentiable\n")
  cat("3. Misclassification Error: Direct interpretation, not differentiable\n")
  cat("4. All measures are symmetric and bounded\n")
  cat("5. Choice depends on application and computational considerations\n")
  cat("6. Entropy is preferred during tree growing due to concavity\n")
  cat("7. Gini is often used in practice due to efficiency\n")
  cat("8. Misclassification error is useful for final evaluation\n")
  
  return(list(
    visualization_results = visualization_results,
    split_results = split_results,
    comparison_results = comparison_results,
    properties_results = properties_results,
    practical_results = practical_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
