# Classification Trees: Introduction Implementation

# Load required libraries
library(rpart)
library(rpart.plot)
library(ggplot2)
library(gridExtra)
library(dplyr)

# Generate classification data
generate_classification_data <- function(n_samples = 200, random_state = 42) {
  set.seed(random_state)
  X <- matrix(rnorm(2 * n_samples), ncol = 2)
  y <- ifelse(X[, 1] + X[, 2] > 0, 1, 0)
  return(list(X = X, y = factor(y)))
}

# Demonstrate basic classification tree
demonstrate_basic_tree <- function() {
  cat("=== Basic Classification Tree Demonstration ===\n\n")
  
  # Generate data
  data <- generate_classification_data(200, 42)
  X <- data$X
  y <- data$y
  
  # Create data frame
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
  
  # Fit classification tree
  tree_model <- rpart(y ~ X1 + X2, data = df, method = "class", 
                     control = rpart.control(maxdepth = 3))
  
  # Plot tree
  rpart.plot(tree_model, box.palette = "RdBu", shadow.col = "gray", 
             nn = TRUE, main = "Classification Tree")
  
  # Make predictions
  df$pred <- predict(tree_model, df, type = "class")
  
  # Plot decision boundary
  p <- ggplot(df, aes(x = X1, y = X2, color = y)) +
    geom_point(alpha = 0.7) +
    geom_point(data = df[df$y != df$pred, ], 
               aes(x = X1, y = X2), shape = 21, size = 3, 
               fill = "transparent", color = "red") +
    labs(title = "Classification Tree Decision Boundary",
         subtitle = "Red circles indicate misclassifications") +
    theme_minimal()
  
  print(p)
  
  # Print tree summary
  cat("\nTree Summary:\n")
  print(tree_model)
  cat("\nComplexity Parameter Table:\n")
  printcp(tree_model)
  
  return(tree_model)
}

# Demonstrate different impurity measures
demonstrate_impurity_measures <- function() {
  cat("\n=== Impurity Measures Demonstration ===\n\n")
  
  # Generate data
  data <- generate_classification_data(200, 42)
  X <- data$X
  y <- data$y
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
  
  # Test different impurity measures (rpart uses Gini by default)
  # We'll demonstrate by comparing different tree configurations
  
  # Fit trees with different parameters
  tree_gini <- rpart(y ~ X1 + X2, data = df, method = "class", 
                     control = rpart.control(maxdepth = 3))
  
  # For entropy-like behavior, we can adjust the split criterion
  # Note: rpart doesn't directly support entropy, but we can simulate different behaviors
  
  # Create visualization
  plots <- list()
  
  # Plot for Gini-based tree
  df_gini <- df
  df_gini$pred <- predict(tree_gini, df, type = "class")
  
  p1 <- ggplot(df_gini, aes(x = X1, y = X2, color = y)) +
    geom_point(alpha = 0.7) +
    labs(title = "Gini-based Classification Tree",
         subtitle = paste("Accuracy:", round(mean(df_gini$y == df_gini$pred), 3))) +
    theme_minimal()
  
  # Create decision boundary visualization
  x_min <- min(X[, 1]) - 1
  x_max <- max(X[, 1]) + 1
  y_min <- min(X[, 2]) - 1
  y_max <- max(X[, 2]) + 1
  
  grid_x <- seq(x_min, x_max, length.out = 100)
  grid_y <- seq(y_min, y_max, length.out = 100)
  grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
  
  grid_data$pred <- predict(tree_gini, grid_data, type = "class")
  
  p2 <- ggplot() +
    geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)),
                 breaks = 0.5, color = "black", size = 1) +
    geom_point(data = df, aes(x = X1, y = X2, color = y), alpha = 0.7) +
    labs(title = "Decision Boundary",
         subtitle = "Gini-based Tree") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  # Print comparison
  cat("Tree Performance:\n")
  cat("Gini-based tree accuracy:", round(mean(df_gini$y == df_gini$pred), 3), "\n")
  cat("Number of nodes:", length(unique(tree_gini$where)), "\n")
  
  return(tree_gini)
}

# Demonstrate tree structure with different depths
demonstrate_tree_structure <- function() {
  cat("\n=== Tree Structure Demonstration ===\n\n")
  
  # Generate data
  data <- generate_classification_data(100, 42)
  X <- data$X
  y <- data$y
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
  
  # Fit trees with different depths
  depths <- c(1, 2, 3, 4)
  trees <- list()
  plots <- list()
  
  for (i in seq_along(depths)) {
    depth <- depths[i]
    
    # Fit tree
    tree <- rpart(y ~ X1 + X2, data = df, method = "class", 
                  control = rpart.control(maxdepth = depth))
    trees[[i]] <- tree
    
    # Create visualization
    df_temp <- df
    df_temp$pred <- predict(tree, df, type = "class")
    
    # Create decision boundary
    x_min <- min(X[, 1]) - 1
    x_max <- max(X[, 1]) + 1
    y_min <- min(X[, 2]) - 1
    y_max <- max(X[, 2]) + 1
    
    grid_x <- seq(x_min, x_max, length.out = 50)
    grid_y <- seq(y_min, y_max, length.out = 50)
    grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
    
    grid_data$pred <- predict(tree, grid_data, type = "class")
    
    p <- ggplot() +
      geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)),
                   breaks = 0.5, color = "black", size = 1) +
      geom_point(data = df, aes(x = X1, y = X2, color = y), alpha = 0.7) +
      labs(title = paste("Tree Depth:", depth),
           subtitle = paste("Accuracy:", round(mean(df_temp$y == df_temp$pred), 3))) +
      theme_minimal()
    
    plots[[i]] <- p
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print summary
  cat("Tree Structure Analysis:\n")
  for (i in seq_along(depths)) {
    tree <- trees[[i]]
    df_temp <- df
    df_temp$pred <- predict(tree, df, type = "class")
    acc <- mean(df_temp$y == df_temp$pred)
    nodes <- length(unique(tree$where))
    cat(sprintf("Depth %d: Accuracy = %.3f, Nodes = %d\n", depths[i], acc, nodes))
  }
  
  return(list(trees = trees, depths = depths))
}

# Demonstrate stopping criteria
demonstrate_stopping_criteria <- function() {
  cat("\n=== Stopping Criteria Demonstration ===\n\n")
  
  # Generate data
  data <- generate_classification_data(200, 42)
  X <- data$X
  y <- data$y
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
  
  # Test different stopping criteria
  criteria_configs <- list(
    list(maxdepth = 2, minsplit = 2, minbucket = 1, name = "Max Depth = 2"),
    list(maxdepth = 5, minsplit = 2, minbucket = 1, name = "Max Depth = 5"),
    list(maxdepth = 10, minsplit = 10, minbucket = 1, name = "Min Split = 10"),
    list(maxdepth = 10, minsplit = 2, minbucket = 20, name = "Min Bucket = 20")
  )
  
  trees <- list()
  plots <- list()
  
  for (i in seq_along(criteria_configs)) {
    config <- criteria_configs[[i]]
    
    # Fit tree
    tree <- rpart(y ~ X1 + X2, data = df, method = "class", 
                  control = rpart.control(
                    maxdepth = config$maxdepth,
                    minsplit = config$minsplit,
                    minbucket = config$minbucket
                  ))
    trees[[i]] <- tree
    
    # Create visualization
    df_temp <- df
    df_temp$pred <- predict(tree, df, type = "class")
    
    # Create decision boundary
    x_min <- min(X[, 1]) - 1
    x_max <- max(X[, 1]) + 1
    y_min <- min(X[, 2]) - 1
    y_max <- max(X[, 2]) + 1
    
    grid_x <- seq(x_min, x_max, length.out = 50)
    grid_y <- seq(y_min, y_max, length.out = 50)
    grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
    
    grid_data$pred <- predict(tree, grid_data, type = "class")
    
    p <- ggplot() +
      geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)),
                   breaks = 0.5, color = "black", size = 1) +
      geom_point(data = df, aes(x = X1, y = X2, color = y), alpha = 0.7) +
      labs(title = config$name,
           subtitle = paste("Accuracy:", round(mean(df_temp$y == df_temp$pred), 3))) +
      theme_minimal()
    
    plots[[i]] <- p
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print summary
  cat("Stopping Criteria Analysis:\n")
  for (i in seq_along(criteria_configs)) {
    tree <- trees[[i]]
    df_temp <- df
    df_temp$pred <- predict(tree, df, type = "class")
    acc <- mean(df_temp$y == df_temp$pred)
    nodes <- length(unique(tree$where))
    cat(sprintf("%s: Accuracy = %.3f, Nodes = %d\n", 
                criteria_configs[[i]]$name, acc, nodes))
  }
  
  return(list(trees = trees, configs = criteria_configs))
}

# Demonstrate greedy algorithm step by step
demonstrate_greedy_algorithm <- function() {
  cat("\n=== Greedy Algorithm Demonstration ===\n\n")
  
  # Create simple data for demonstration
  X <- matrix(c(1, 2, 2, 3, 3, 1, 4, 2, 5, 3, 6, 1), ncol = 2, byrow = TRUE)
  y <- factor(c(0, 0, 0, 1, 1, 1))
  
  cat("Data:\n")
  for (i in 1:nrow(X)) {
    cat(sprintf("  Sample %d: X=(%.1f, %.1f), y=%s\n", i, X[i, 1], X[i, 2], y[i]))
  }
  
  # Calculate initial impurity (Gini)
  initial_impurity <- 1 - sum((table(y) / length(y))^2)
  cat(sprintf("\nInitial Gini impurity: %.3f\n", initial_impurity))
  
  # Test all possible splits
  cat("\nTesting all possible splits:\n")
  cat("Feature | Threshold | Left Impurity | Right Impurity | Reduction\n")
  cat(paste(rep("-", 65), collapse = ""), "\n")
  
  best_reduction <- 0
  best_split <- NULL
  
  for (feature in 1:2) {
    thresholds <- unique(X[, feature])
    for (threshold in thresholds) {
      left_mask <- X[, feature] <= threshold
      right_mask <- !left_mask
      
      if (sum(left_mask) > 0 && sum(right_mask) > 0) {
        left_impurity <- 1 - sum((table(y[left_mask]) / sum(left_mask))^2)
        right_impurity <- 1 - sum((table(y[right_mask]) / sum(right_mask))^2)
        
        n_left <- sum(left_mask)
        n_right <- sum(right_mask)
        weighted_impurity <- (n_left * left_impurity + n_right * right_impurity) / length(y)
        reduction <- initial_impurity - weighted_impurity
        
        cat(sprintf("   %d    |    %.1f    |     %.3f     |     %.3f     |   %.3f\n", 
                    feature, threshold, left_impurity, right_impurity, reduction))
        
        if (reduction > best_reduction) {
          best_reduction <- reduction
          best_split <- c(feature, threshold)
        }
      }
    }
  }
  
  cat(sprintf("\nBest split: Feature %d <= %.1f\n", best_split[1], best_split[2]))
  cat(sprintf("Impurity reduction: %.3f\n", best_reduction))
  
  return(list(X = X, y = y, best_split = best_split))
}

# Demonstrate advantages and limitations
demonstrate_advantages_limitations <- function() {
  cat("\n=== Advantages and Limitations ===\n\n")
  
  # Generate data with different characteristics
  set.seed(42)
  
  # 1. Linear separable (works well)
  X1 <- matrix(rnorm(200 * 2), ncol = 2)
  y1 <- factor(ifelse(X1[, 1] + X1[, 2] > 0, 1, 0))
  
  # 2. Circular boundary (challenging)
  X2 <- matrix(rnorm(200 * 2), ncol = 2)
  y2 <- factor(ifelse(X2[, 1]^2 + X2[, 2]^2 < 1, 1, 0))
  
  # 3. XOR pattern (very challenging)
  X3 <- matrix(rnorm(200 * 2), ncol = 2)
  y3 <- factor(ifelse((X3[, 1] > 0 & X3[, 2] > 0) | (X3[, 1] < 0 & X3[, 2] < 0), 1, 0))
  
  datasets <- list(
    list(X = X1, y = y1, name = "Linear Separable"),
    list(X = X2, y = y2, name = "Circular Boundary"),
    list(X = X3, y = y3, name = "XOR Pattern")
  )
  
  plots <- list()
  results <- list()
  
  for (i in seq_along(datasets)) {
    data <- datasets[[i]]
    X <- data$X
    y <- data$y
    name <- data$name
    
    df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = y)
    
    # Fit tree
    tree <- rpart(y ~ X1 + X2, data = df, method = "class", 
                  control = rpart.control(maxdepth = 5))
    
    # Make predictions
    df$pred <- predict(tree, df, type = "class")
    acc <- mean(df$y == df$pred)
    
    # Create decision boundary
    x_min <- min(X[, 1]) - 1
    x_max <- max(X[, 1]) + 1
    y_min <- min(X[, 2]) - 1
    y_max <- max(X[, 2]) + 1
    
    grid_x <- seq(x_min, x_max, length.out = 50)
    grid_y <- seq(y_min, y_max, length.out = 50)
    grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
    
    grid_data$pred <- predict(tree, grid_data, type = "class")
    
    # Create plot
    p <- ggplot() +
      geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)),
                   breaks = 0.5, color = "black", size = 1) +
      geom_point(data = df, aes(x = X1, y = X2, color = y), alpha = 0.7) +
      labs(title = paste(name, "- Decision Boundary"),
           subtitle = paste("Accuracy:", round(acc, 3))) +
      theme_minimal()
    
    plots[[i]] <- p
    results[[i]] <- list(name = name, accuracy = acc, tree = tree)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 1))
  
  # Print analysis
  cat("Analysis:\n")
  for (result in results) {
    cat(sprintf("%s: Accuracy = %.3f\n", result$name, result$accuracy))
  }
  cat("\nThis demonstrates the axis-aligned limitation of decision trees.\n")
  
  return(results)
}

# Main demonstration function
main_r <- function() {
  cat("Classification Trees: Introduction Implementation\n")
  cat("=" * 60, "\n")
  
  # 1. Basic tree demonstration
  cat("\n1. Basic Classification Tree:\n")
  basic_tree <- demonstrate_basic_tree()
  
  # 2. Impurity measures demonstration
  cat("\n2. Impurity Measures:\n")
  impurity_tree <- demonstrate_impurity_measures()
  
  # 3. Tree structure demonstration
  cat("\n3. Tree Structure:\n")
  structure_results <- demonstrate_tree_structure()
  
  # 4. Stopping criteria demonstration
  cat("\n4. Stopping Criteria:\n")
  stopping_results <- demonstrate_stopping_criteria()
  
  # 5. Greedy algorithm demonstration
  cat("\n5. Greedy Algorithm:\n")
  greedy_results <- demonstrate_greedy_algorithm()
  
  # 6. Advantages and limitations
  cat("\n6. Advantages and Limitations:\n")
  limitations_results <- demonstrate_advantages_limitations()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Impurity measures (Gini) control split quality\n")
  cat("2. Stopping criteria prevent overfitting\n")
  cat("3. Greedy algorithm is computationally efficient\n")
  cat("4. Trees create axis-aligned decision boundaries\n")
  cat("5. Tree structure provides interpretability\n")
  cat("6. Trees can struggle with non-linear patterns\n")
  cat("7. Tree depth controls model complexity\n")
  cat("8. rpart provides robust tree implementation\n")
  
  return(list(
    basic_tree = basic_tree,
    impurity_tree = impurity_tree,
    structure_results = structure_results,
    stopping_results = stopping_results,
    greedy_results = greedy_results,
    limitations_results = limitations_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
