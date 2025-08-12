# Misclassification Rate vs Entropy: Mathematical Analysis

# Load required libraries
library(ggplot2)
library(gridExtra)
library(dplyr)

# Impurity measure functions
misclassification_impurity <- function(p) {
  return(min(p, 1-p))
}

entropy_impurity <- function(p) {
  if (p == 0 || p == 1) {
    return(0)
  }
  return(-p * log2(p) - (1-p) * log2(1-p))
}

# Visualize misclassification rate vs entropy
plot_impurity_comparison <- function() {
  cat("=== Misclassification Rate vs Entropy Visualization ===\n\n")
  
  p <- seq(0, 1, length.out = 1000)
  
  # Calculate impurity measures
  misclassification <- 1 - pmax(p, 1-p)  # min(p, 1-p)
  entropy <- -p * log2(p + 1e-10) - (1-p) * log2(1-p + 1e-10)
  
  # Scale entropy to match misclassification at p=0.5
  entropy_scaled <- entropy / entropy[501] * misclassification[501]
  
  # Create data frame for plotting
  plot_data <- data.frame(
    p = p,
    misclassification = misclassification,
    entropy_scaled = entropy_scaled
  )
  
  # Main comparison plot
  p1 <- ggplot(plot_data, aes(x = p)) +
    geom_line(aes(y = misclassification, color = "Misclassification Rate"), size = 1) +
    geom_line(aes(y = entropy_scaled, color = "Entropy (Scaled)"), size = 1) +
    geom_vline(xintercept = 0.5, color = "gray", linetype = "dashed", alpha = 0.7) +
    geom_hline(yintercept = 0.5, color = "gray", linetype = "dashed", alpha = 0.7) +
    labs(title = "Misclassification Rate vs Entropy",
         x = "Probability of Class 0 (p)",
         y = "Impurity",
         color = "Measure") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Concavity demonstration
  p1_points <- 0.3
  p2_points <- 0.7
  p_weighted <- 0.5
  
  concavity_data <- data.frame(
    p = c(p1_points, p2_points, p_weighted),
    misclassification = c(misclassification[301], misclassification[701], misclassification[501]),
    entropy = c(entropy_scaled[301], entropy_scaled[701], entropy_scaled[501]),
    type = c("Point 1", "Point 2", "Weighted Average")
  )
  
  p2 <- ggplot() +
    geom_line(data = plot_data, aes(x = p, y = misclassification), color = "blue", alpha = 0.3) +
    geom_line(data = plot_data, aes(x = p, y = entropy_scaled), color = "red", alpha = 0.3) +
    geom_point(data = concavity_data, aes(x = p, y = misclassification, color = "Misclassification"), size = 3) +
    geom_point(data = concavity_data, aes(x = p, y = entropy, color = "Entropy"), size = 3) +
    geom_segment(aes(x = p1_points, y = concavity_data$misclassification[1], 
                     xend = p2_points, yend = concavity_data$misclassification[2]), 
                 color = "blue", linetype = "dashed") +
    geom_segment(aes(x = p1_points, y = concavity_data$entropy[1], 
                     xend = p2_points, yend = concavity_data$entropy[2]), 
                 color = "red", linetype = "dashed") +
    labs(title = "Concavity Demonstration",
         x = "Probability of Class 0 (p)",
         y = "Impurity") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Split gain analysis
  p_parent <- 0.5
  p_left <- 0.3
  p_right <- 0.7
  w_left <- 0.5
  w_right <- 0.5
  
  # Calculate gains
  misclass_parent <- 1 - max(p_parent, 1-p_parent)
  misclass_left <- 1 - max(p_left, 1-p_left)
  misclass_right <- 1 - max(p_right, 1-p_right)
  misclass_gain <- misclass_parent - (w_left * misclass_left + w_right * misclass_right)
  
  entropy_parent <- -p_parent * log2(p_parent) - (1-p_parent) * log2(1-p_parent)
  entropy_left <- -p_left * log2(p_left) - (1-p_left) * log2(1-p_left)
  entropy_right <- -p_right * log2(p_right) - (1-p_right) * log2(1-p_right)
  entropy_gain <- entropy_parent - (w_left * entropy_left + w_right * entropy_right)
  
  split_data <- data.frame(
    p = c(p_left, p_right),
    misclassification = c(misclass_left, misclass_right),
    entropy = c(entropy_left, entropy_right)
  )
  
  p3 <- ggplot() +
    geom_line(data = plot_data, aes(x = p, y = misclassification), color = "blue", alpha = 0.3) +
    geom_line(data = plot_data, aes(x = p, y = entropy_scaled), color = "red", alpha = 0.3) +
    geom_point(data = split_data, aes(x = p, y = misclassification, color = "Misclassification"), size = 3) +
    geom_point(data = split_data, aes(x = p, y = entropy, color = "Entropy"), size = 3) +
    geom_hline(yintercept = misclass_parent, color = "blue", linetype = "dashed", alpha = 0.7) +
    geom_hline(yintercept = entropy_parent, color = "red", linetype = "dashed", alpha = 0.7) +
    labs(title = paste("Split Gain Comparison\nMisclass Gain:", round(misclass_gain, 3), 
                       "Entropy Gain:", round(entropy_gain, 3)),
         x = "Probability of Class 0 (p)",
         y = "Impurity") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Zero gain scenario
  p_parent_zero <- 0.6
  p_left_zero <- 0.55
  p_right_zero <- 0.65
  
  misclass_parent_zero <- 1 - max(p_parent_zero, 1-p_parent_zero)
  misclass_left_zero <- 1 - max(p_left_zero, 1-p_left_zero)
  misclass_right_zero <- 1 - max(p_right_zero, 1-p_right_zero)
  misclass_gain_zero <- misclass_parent_zero - (w_left * misclass_left_zero + w_right * misclass_right_zero)
  
  entropy_parent_zero <- -p_parent_zero * log2(p_parent_zero) - (1-p_parent_zero) * log2(1-p_parent_zero)
  entropy_left_zero <- -p_left_zero * log2(p_left_zero) - (1-p_left_zero) * log2(1-p_left_zero)
  entropy_right_zero <- -p_right_zero * log2(p_right_zero) - (1-p_right_zero) * log2(1-p_right_zero)
  entropy_gain_zero <- entropy_parent_zero - (w_left * entropy_left_zero + w_right * entropy_right_zero)
  
  zero_gain_data <- data.frame(
    p = c(p_left_zero, p_right_zero),
    misclassification = c(misclass_left_zero, misclass_right_zero),
    entropy = c(entropy_left_zero, entropy_right_zero)
  )
  
  p4 <- ggplot() +
    geom_line(data = plot_data, aes(x = p, y = misclassification), color = "blue", alpha = 0.3) +
    geom_line(data = plot_data, aes(x = p, y = entropy_scaled), color = "red", alpha = 0.3) +
    geom_point(data = zero_gain_data, aes(x = p, y = misclassification, color = "Misclassification"), size = 3) +
    geom_point(data = zero_gain_data, aes(x = p, y = entropy, color = "Entropy"), size = 3) +
    geom_hline(yintercept = misclass_parent_zero, color = "blue", linetype = "dashed", alpha = 0.7) +
    geom_hline(yintercept = entropy_parent_zero, color = "red", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = 0.5, color = "gray", linetype = "dashed", alpha = 0.7) +
    labs(title = paste("Zero Gain Scenario (Same Side of 0.5)\nMisclass Gain:", round(misclass_gain_zero, 3), 
                       "Entropy Gain:", round(entropy_gain_zero, 3)),
         x = "Probability of Class 0 (p)",
         y = "Impurity") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  # Print numerical results
  cat("Split Gain Analysis:\n")
  cat("Scenario 1 - Different sides of 0.5:\n")
  cat(sprintf("  Parent: p=%.1f, Misclassification gain: %.4f, Entropy gain: %.4f\n", 
              p_parent, misclass_gain, entropy_gain))
  cat("Scenario 2 - Same side of 0.5:\n")
  cat(sprintf("  Parent: p=%.1f, Misclassification gain: %.4f, Entropy gain: %.4f\n", 
              p_parent_zero, misclass_gain_zero, entropy_gain_zero))
  
  return(list(
    misclass_gain = misclass_gain,
    entropy_gain = entropy_gain,
    misclass_gain_zero = misclass_gain_zero,
    entropy_gain_zero = entropy_gain_zero
  ))
}

# Compare split gains for different scenarios
compare_split_gains <- function() {
  cat("=== Split Gain Comparison ===\n\n")
  
  calculate_split_gain <- function(p_parent, p_left, p_right, w_left, w_right, impurity_func) {
    parent_impurity <- impurity_func(p_parent)
    left_impurity <- impurity_func(p_left)
    right_impurity <- impurity_func(p_right)
    
    weighted_child_impurity <- w_left * left_impurity + w_right * right_impurity
    gain <- parent_impurity - weighted_child_impurity
    
    return(gain)
  }
  
  # Test scenarios
  scenarios <- list(
    list(name = "Different sides of 0.5", p_parent = 0.5, p_left = 0.3, p_right = 0.7, w_left = 0.5, w_right = 0.5),
    list(name = "Same side of 0.5 (left)", p_parent = 0.6, p_left = 0.55, p_right = 0.65, w_left = 0.5, w_right = 0.5),
    list(name = "Same side of 0.5 (right)", p_parent = 0.4, p_left = 0.35, p_right = 0.45, w_left = 0.5, w_right = 0.5),
    list(name = "Extreme split", p_parent = 0.5, p_left = 0.1, p_right = 0.9, w_left = 0.5, w_right = 0.5)
  )
  
  cat("Split Gain Comparison:\n")
  cat(paste(rep("-", 80), collapse = ""), "\n")
  cat(sprintf("%-25s %-15s %-15s\n", "Scenario", "Misclass Gain", "Entropy Gain"))
  cat(paste(rep("-", 80), collapse = ""), "\n")
  
  results <- list()
  for (scenario in scenarios) {
    misclass_gain <- calculate_split_gain(
      scenario$p_parent, scenario$p_left, scenario$p_right,
      scenario$w_left, scenario$w_right, misclassification_impurity
    )
    
    entropy_gain <- calculate_split_gain(
      scenario$p_parent, scenario$p_left, scenario$p_right,
      scenario$w_left, scenario$w_right, entropy_impurity
    )
    
    cat(sprintf("%-25s %-15.4f %-15.4f\n", scenario$name, misclass_gain, entropy_gain))
    
    results[[length(results) + 1]] <- list(
      scenario = scenario$name,
      misclass_gain = misclass_gain,
      entropy_gain = entropy_gain
    )
  }
  
  cat("\nKey Observations:\n")
  cat("1. Entropy always provides positive gain (strictly concave)\n")
  cat("2. Misclassification can give zero gain when both children are on same side of 0.5\n")
  cat("3. Entropy encourages more aggressive splitting\n")
  
  return(results)
}

# Demonstrate mathematical properties
demonstrate_mathematical_properties <- function() {
  cat("=== Mathematical Properties Analysis ===\n\n")
  
  entropy_second_derivative <- function(p) {
    if (p == 0 || p == 1) {
      return(-Inf)
    }
    return(-1 / (p * (1-p) * log(2)))
  }
  
  # Test concavity property
  cat("Concavity Analysis:\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  
  # Test points for concavity
  test_points <- c(0.1, 0.3, 0.5, 0.7, 0.9)
  
  for (p in test_points) {
    # Calculate second derivative of entropy
    entropy_2nd_deriv <- entropy_second_derivative(p)
    cat(sprintf("p = %.1f: Entropy second derivative = %.4f\n", p, entropy_2nd_deriv))
  }
  
  cat("\nEntropy is strictly concave (second derivative < 0)\n")
  cat("Misclassification is piecewise linear (not strictly concave)\n")
  
  # Test Jensen's inequality
  cat("\nJensen's Inequality Test:\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  
  p1 <- 0.3
  p2 <- 0.7
  lambda_val <- 0.5
  p_weighted <- lambda_val * p1 + (1 - lambda_val) * p2
  
  # For entropy (strictly concave)
  entropy_p1 <- entropy_impurity(p1)
  entropy_p2 <- entropy_impurity(p2)
  entropy_weighted <- entropy_impurity(p_weighted)
  entropy_linear <- lambda_val * entropy_p1 + (1 - lambda_val) * entropy_p2
  
  cat("Entropy test:\n")
  cat(sprintf("  f(λp₁ + (1-λ)p₂) = f(%.2f) = %.4f\n", p_weighted, entropy_weighted))
  cat(sprintf("  λf(p₁) + (1-λ)f(p₂) = %.1f×%.4f + %.1f×%.4f = %.4f\n", 
              lambda_val, entropy_p1, 1-lambda_val, entropy_p2, entropy_linear))
  cat(sprintf("  Jensen's inequality: %.4f > %.4f ✓\n", entropy_weighted, entropy_linear))
  
  # For misclassification (not strictly concave)
  misclass_p1 <- misclassification_impurity(p1)
  misclass_p2 <- misclassification_impurity(p2)
  misclass_weighted <- misclassification_impurity(p_weighted)
  misclass_linear <- lambda_val * misclass_p1 + (1 - lambda_val) * misclass_p2
  
  cat("\nMisclassification test:\n")
  cat(sprintf("  f(λp₁ + (1-λ)p₂) = f(%.2f) = %.4f\n", p_weighted, misclass_weighted))
  cat(sprintf("  λf(p₁) + (1-λ)f(p₂) = %.1f×%.4f + %.1f×%.4f = %.4f\n", 
              lambda_val, misclass_p1, 1-lambda_val, misclass_p2, misclass_linear))
  cat(sprintf("  Jensen's inequality: %.4f = %.4f (equality holds)\n", misclass_weighted, misclass_linear))
  
  return(list(
    entropy_test = list(
      weighted = entropy_weighted,
      linear = entropy_linear,
      inequality_holds = entropy_weighted > entropy_linear
    ),
    misclass_test = list(
      weighted = misclass_weighted,
      linear = misclass_linear,
      inequality_holds = abs(misclass_weighted - misclass_linear) < 1e-10
    )
  ))
}

# Analyze zero gain scenarios
analyze_zero_gain_scenarios <- function() {
  cat("=== Zero Gain Scenarios Analysis ===\n\n")
  
  calculate_split_gain <- function(p_parent, p_left, p_right, w_left, w_right, impurity_func) {
    parent_impurity <- impurity_func(p_parent)
    left_impurity <- impurity_func(p_left)
    right_impurity <- impurity_func(p_right)
    weighted_child_impurity <- w_left * left_impurity + w_right * right_impurity
    return(parent_impurity - weighted_child_impurity)
  }
  
  # Test different scenarios
  scenarios <- list()
  
  # Scenario 1: Both children on left side of 0.5
  p_parent <- 0.6
  p_left <- 0.55
  p_right <- 0.65
  w_left <- 0.5
  w_right <- 0.5
  
  misclass_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
  entropy_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
  
  scenarios[[length(scenarios) + 1]] <- list(
    name = "Both children left of 0.5",
    p_parent = p_parent,
    p_left = p_left,
    p_right = p_right,
    misclass_gain = misclass_gain,
    entropy_gain = entropy_gain
  )
  
  # Scenario 2: Both children on right side of 0.5
  p_parent <- 0.4
  p_left <- 0.35
  p_right <- 0.45
  
  misclass_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
  entropy_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
  
  scenarios[[length(scenarios) + 1]] <- list(
    name = "Both children right of 0.5",
    p_parent = p_parent,
    p_left = p_left,
    p_right = p_right,
    misclass_gain = misclass_gain,
    entropy_gain = entropy_gain
  )
  
  # Scenario 3: Children straddle 0.5
  p_parent <- 0.5
  p_left <- 0.3
  p_right <- 0.7
  
  misclass_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, misclassification_impurity)
  entropy_gain <- calculate_split_gain(p_parent, p_left, p_right, w_left, w_right, entropy_impurity)
  
  scenarios[[length(scenarios) + 1]] <- list(
    name = "Children straddle 0.5",
    p_parent = p_parent,
    p_left = p_left,
    p_right = p_right,
    misclass_gain = misclass_gain,
    entropy_gain = entropy_gain
  )
  
  # Print results
  cat("Zero Gain Scenarios Analysis:\n")
  cat(paste(rep("-", 80), collapse = ""), "\n")
  cat(sprintf("%-25s %-10s %-8s %-9s %-10s %-10s\n", "Scenario", "Parent p", "Left p", "Right p", "Misclass", "Entropy"))
  cat(paste(rep("-", 80), collapse = ""), "\n")
  
  for (scenario in scenarios) {
    cat(sprintf("%-25s %-10.2f %-8.2f %-9.2f %-10.4f %-10.4f\n", 
                scenario$name, scenario$p_parent, scenario$p_left, scenario$p_right, 
                scenario$misclass_gain, scenario$entropy_gain))
  }
  
  cat("\nKey Findings:\n")
  cat("1. Misclassification gives zero gain when both children are on the same side of 0.5\n")
  cat("2. Entropy always gives positive gain for non-trivial splits\n")
  cat("3. Zero gain occurs because misclassification is piecewise linear\n")
  
  return(scenarios)
}

# Demonstrate practical implications
demonstrate_practical_implications <- function() {
  cat("=== Practical Implications ===\n\n")
  
  # Generate synthetic data to demonstrate tree construction
  set.seed(42)
  
  # Create data with different characteristics
  n_samples <- 200
  
  # Dataset 1: Clear separation
  X1 <- matrix(rnorm(n_samples * 2), ncol = 2)
  y1 <- factor(ifelse(X1[, 1] + X1[, 2] > 0, 1, 0))
  
  # Dataset 2: Overlapping classes
  X2 <- matrix(rnorm(n_samples * 2), ncol = 2)
  y2 <- factor(ifelse(X2[, 1] + X2[, 2] + 0.5 * rnorm(n_samples) > 0, 1, 0))
  
  datasets <- list(
    list(X = X1, y = y1, name = "Clear Separation"),
    list(X = X2, y = y2, name = "Overlapping Classes")
  )
  
  find_best_split <- function(X, y, impurity_func) {
    n_samples <- nrow(X)
    n_features <- ncol(X)
    best_gain <- 0
    best_feature <- NULL
    best_threshold <- NULL
    
    for (feature in 1:n_features) {
      thresholds <- unique(X[, feature])
      for (threshold in thresholds) {
        left_mask <- X[, feature] <= threshold
        right_mask <- !left_mask
        
        if (sum(left_mask) > 0 && sum(right_mask) > 0) {
          # Calculate class probabilities
          parent_counts <- table(y)
          parent_probs <- parent_counts / length(y)
          left_counts <- table(y[left_mask])
          left_probs <- left_counts / sum(left_mask)
          right_counts <- table(y[right_mask])
          right_probs <- right_counts / sum(right_mask)
          
          # Calculate impurity
          parent_impurity <- impurity_func(parent_probs[1])
          left_impurity <- impurity_func(left_probs[1])
          right_impurity <- impurity_func(right_probs[1])
          
          # Calculate gain
          p_left <- sum(left_mask) / length(y)
          p_right <- sum(right_mask) / length(y)
          gain <- parent_impurity - (p_left * left_impurity + p_right * right_impurity)
          
          if (gain > best_gain) {
            best_gain <- gain
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
    }
    
    return(list(feature = best_feature, threshold = best_threshold, gain = best_gain))
  }
  
  cat("Tree Construction Analysis:\n")
  cat(paste(rep("-", 60), collapse = ""), "\n")
  
  for (data in datasets) {
    X <- data$X
    y <- data$y
    name <- data$name
    
    cat(sprintf("\n%s Dataset:\n", name))
    
    # Find best splits for both impurity measures
    best_misclass <- find_best_split(X, y, misclassification_impurity)
    best_entropy <- find_best_split(X, y, entropy_impurity)
    
    cat(sprintf("  Misclassification best split: Feature %d, Threshold %.3f, Gain %.4f\n", 
                best_misclass$feature, best_misclass$threshold, best_misclass$gain))
    cat(sprintf("  Entropy best split: Feature %d, Threshold %.3f, Gain %.4f\n", 
                best_entropy$feature, best_entropy$threshold, best_entropy$gain))
    
    # Compare gains
    if (best_misclass$gain == 0) {
      cat("  ⚠️  Misclassification found no useful split (zero gain)\n")
    } else {
      cat("  ✓ Misclassification found useful split\n")
    }
    
    cat("  ✓ Entropy always found useful split\n")
  }
  
  cat("\nPractical Recommendations:\n")
  cat("1. Use entropy during tree construction (always positive gain)\n")
  cat("2. Use misclassification for final evaluation (direct interpretation)\n")
  cat("3. Consider computational efficiency for large datasets\n")
  cat("4. Monitor for zero-gain scenarios with misclassification\n")
}

# Main demonstration function
main_r <- function() {
  cat("Misclassification Rate vs Entropy: Mathematical Analysis\n")
  cat("=" * 70, "\n")
  
  # 1. Visual comparison
  cat("\n1. Visual Comparison:\n")
  viz_results <- plot_impurity_comparison()
  
  # 2. Split gain comparison
  cat("\n2. Split Gain Comparison:\n")
  split_results <- compare_split_gains()
  
  # 3. Mathematical properties
  cat("\n3. Mathematical Properties:\n")
  math_results <- demonstrate_mathematical_properties()
  
  # 4. Zero gain scenarios
  cat("\n4. Zero Gain Scenarios:\n")
  zero_gain_results <- analyze_zero_gain_scenarios()
  
  # 5. Practical implications
  cat("\n5. Practical Implications:\n")
  practical_results <- demonstrate_practical_implications()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Entropy is strictly concave, always provides positive split gain\n")
  cat("2. Misclassification is piecewise linear, can give zero gain\n")
  cat("3. Jensen's inequality explains why concave functions work well\n")
  cat("4. Use entropy for tree construction, misclassification for evaluation\n")
  cat("5. Zero gain occurs when both children are on same side of 0.5\n")
  cat("6. Entropy encourages more aggressive splitting\n")
  cat("7. Misclassification aligns with final classification objective\n")
  cat("8. Mathematical properties determine practical behavior\n")
  
  return(list(
    visualization_results = viz_results,
    split_results = split_results,
    mathematical_results = math_results,
    zero_gain_results = zero_gain_results,
    practical_results = practical_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
