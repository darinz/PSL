# Decision Trees Introduction - R Implementation
#
# This script demonstrates the fundamental concepts of decision trees including:
# - Recursive splitting and region partitioning
# - Entropy loss and Gini impurity calculations
# - Regression trees with least-squares loss
# - Regularization techniques
# - Runtime complexity analysis

library(rpart)
library(rpart.plot)
library(ggplot2)
library(gridExtra)
library(caret)
library(pROC)

# Set random seed for reproducibility
set.seed(42)

DecisionTreeDemo <- setRefClass("DecisionTreeDemo",
  fields = list(),
  methods = list(
    
    create_2d_dataset = function(n_samples = 200, noise = 0.3) {
      # Create a 2D dataset for visualization
      x1 <- rnorm(n_samples, mean = 0, sd = 1)
      x2 <- rnorm(n_samples, mean = 0, sd = 1)
      
      # Create non-linear decision boundary
      y <- ifelse(x1^2 + x2^2 < 1.5, 1, 0)
      
      # Add noise
      noise_factor <- rnorm(n_samples, mean = 0, sd = noise)
      y <- ifelse(y + noise_factor > 0.5, 1, 0)
      
      return(data.frame(x1 = x1, x2 = x2, y = factor(y)))
    },
    
    entropy_loss = function(y) {
      # Calculate entropy loss for a region
      if (length(y) == 0) return(0)
      
      # Calculate class probabilities
      class_counts <- table(y)
      probabilities <- class_counts / length(y)
      
      # Calculate entropy (avoid log(0))
      entropy <- -sum(probabilities * log2(probabilities + 1e-10))
      return(entropy)
    },
    
    gini_impurity = function(y) {
      # Calculate Gini impurity for a region
      if (length(y) == 0) return(0)
      
      # Calculate class probabilities
      class_counts <- table(y)
      probabilities <- class_counts / length(y)
      
      # Calculate Gini impurity
      gini <- 1 - sum(probabilities^2)
      return(gini)
    },
    
    information_gain = function(y_parent, y_left, y_right) {
      # Calculate information gain for a split
      # Parent entropy
      parent_entropy <- entropy_loss(y_parent)
      
      # Weighted average of children entropy
      n_left <- length(y_left)
      n_right <- length(y_right)
      n_total <- n_left + n_right
      
      if (n_total == 0) return(0)
      
      left_entropy <- entropy_loss(y_left)
      right_entropy <- entropy_loss(y_right)
      
      weighted_entropy <- (n_left * left_entropy + n_right * right_entropy) / n_total
      
      return(parent_entropy - weighted_entropy)
    },
    
    find_best_split = function(X, y) {
      # Find the best split for a region (greedy approach)
      best_gain <- -1
      best_feature <- NULL
      best_threshold <- NULL
      
      n_samples <- nrow(X)
      n_features <- ncol(X)
      
      for (feature in 1:n_features) {
        # Get unique values for this feature
        thresholds <- unique(X[, feature])
        
        for (threshold in thresholds) {
          # Create split
          left_mask <- X[, feature] < threshold
          right_mask <- !left_mask
          
          y_left <- y[left_mask]
          y_right <- y[right_mask]
          
          # Calculate information gain
          gain <- information_gain(y, y_left, y_right)
          
          if (gain > best_gain) {
            best_gain <- gain
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
      
      return(list(feature = best_feature, threshold = best_threshold, gain = best_gain))
    },
    
    demonstrate_recursive_splitting = function() {
      # Demonstrate recursive splitting process
      cat("=== Recursive Splitting Demonstration ===\n")
      
      # Create dataset
      data <- create_2d_dataset(n_samples = 100, noise = 0.2)
      X <- data[, c("x1", "x2")]
      y <- data$y
      
      # Initial region (all data)
      cat(sprintf("Initial region: %d samples, entropy: %.3f\n", 
                  length(y), entropy_loss(y)))
      
      # First split
      split_result <- find_best_split(X, y)
      feature <- split_result$feature
      threshold <- split_result$threshold
      gain <- split_result$gain
      
      left_mask <- X[, feature] < threshold
      right_mask <- !left_mask
      
      y_left <- y[left_mask]
      y_right <- y[right_mask]
      
      cat(sprintf("Best split: Feature %d < %.3f\n", feature, threshold))
      cat(sprintf("Information gain: %.3f\n", gain))
      cat(sprintf("Left region: %d samples, entropy: %.3f\n", 
                  length(y_left), entropy_loss(y_left)))
      cat(sprintf("Right region: %d samples, entropy: %.3f\n", 
                  length(y_right), entropy_loss(y_right)))
      
      return(list(X = X, y = y, feature = feature, threshold = threshold))
    },
    
    compare_entropy_gini = function() {
      # Compare entropy loss vs Gini impurity
      cat("\n=== Entropy vs Gini Impurity Comparison ===\n")
      
      # Create different class distributions
      distributions <- list(
        list(y = c(1, 1, 1, 1, 1), desc = "Pure class 1"),
        list(y = c(1, 1, 1, 0, 0), desc = "80% class 1"),
        list(y = c(1, 1, 0, 0, 0), desc = "60% class 1"),
        list(y = c(1, 0, 0, 0, 0), desc = "20% class 1"),
        list(y = c(1, 1, 1, 1, 0), desc = "80% class 1")
      )
      
      cat(sprintf("%-15s %-10s %-10s\n", "Distribution", "Entropy", "Gini"))
      cat(paste(rep("-", 40), collapse = ""), "\n")
      
      for (dist in distributions) {
        entropy <- entropy_loss(dist$y)
        gini <- gini_impurity(dist$y)
        cat(sprintf("%-15s %-10.3f %-10.3f\n", dist$desc, entropy, gini))
      }
    },
    
    regression_tree_demo = function() {
      # Demonstrate regression trees
      cat("\n=== Regression Tree Demonstration ===\n")
      
      # Create regression dataset
      n_samples <- 200
      x1 <- rnorm(n_samples, mean = 0, sd = 1)
      x2 <- rnorm(n_samples, mean = 0, sd = 1)
      y <- 2*x1 + 3*x2 + x1*x2 + rnorm(n_samples, mean = 0, sd = 0.5)
      
      data <- data.frame(x1 = x1, x2 = x2, y = y)
      
      # Split data
      train_idx <- sample(1:n_samples, 0.7 * n_samples)
      train_data <- data[train_idx, ]
      test_data <- data[-train_idx, ]
      
      # Train regression tree
      reg_tree <- rpart(y ~ x1 + x2, data = train_data, 
                       control = rpart.control(maxdepth = 5))
      
      # Predictions
      y_pred <- predict(reg_tree, test_data)
      mse <- mean((test_data$y - y_pred)^2)
      
      cat(sprintf("Regression Tree MSE: %.3f\n", mse))
      cat(sprintf("Tree depth: %d\n", max(reg_tree$frame$depth)))
      cat(sprintf("Number of leaves: %d\n", sum(reg_tree$frame$var == "<leaf>")))
      
      return(list(tree = reg_tree, test_data = test_data, y_pred = y_pred, mse = mse))
    },
    
    regularization_demo = function() {
      # Demonstrate regularization techniques
      cat("\n=== Regularization Demonstration ===\n")
      
      # Create dataset prone to overfitting
      data <- create_2d_dataset(n_samples = 50, noise = 0.1)
      
      # Different regularization parameters
      configs <- list(
        list(maxdepth = 10, minsplit = 1, name = "No regularization"),
        list(maxdepth = 3, minsplit = 1, name = "Max depth = 3"),
        list(maxdepth = 10, minsplit = 5, name = "Min split = 5"),
        list(maxdepth = 5, minsplit = 3, name = "Both constraints")
      )
      
      results <- list()
      
      for (i in seq_along(configs)) {
        config <- configs[[i]]
        
        # Time the training
        start_time <- Sys.time()
        tree <- rpart(y ~ x1 + x2, data = data,
                     control = rpart.control(maxdepth = config$maxdepth,
                                           minsplit = config$minsplit))
        train_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        
        # Predictions
        y_pred <- predict(tree, data, type = "class")
        accuracy <- mean(y_pred == data$y)
        
        results[[i]] <- list(
          name = config$name,
          depth = max(tree$frame$depth),
          leaves = sum(tree$frame$var == "<leaf>"),
          accuracy = accuracy,
          train_time = train_time
        )
      }
      
      # Display results
      cat(sprintf("%-20s %-6s %-7s %-10s %-10s\n", 
                  "Configuration", "Depth", "Leaves", "Accuracy", "Time (ms)"))
      cat(paste(rep("-", 60), collapse = ""), "\n")
      
      for (result in results) {
        cat(sprintf("%-20s %-6d %-7d %-10.3f %-10.1f\n",
                    result$name, result$depth, result$leaves,
                    result$accuracy, result$train_time * 1000))
      }
    },
    
    runtime_complexity_demo = function() {
      # Demonstrate runtime complexity
      cat("\n=== Runtime Complexity Analysis ===\n")
      
      # Test different dataset sizes
      sizes <- c(100, 500, 1000, 2000, 5000)
      train_times <- numeric(length(sizes))
      test_times <- numeric(length(sizes))
      
      for (i in seq_along(sizes)) {
        size <- sizes[i]
        
        # Create dataset
        x1 <- rnorm(size, mean = 0, sd = 1)
        x2 <- rnorm(size, mean = 0, sd = 1)
        y <- ifelse(x1^2 + x2^2 < 1.5, 1, 0)
        data <- data.frame(x1 = x1, x2 = x2, y = factor(y))
        
        # Split data
        train_idx <- sample(1:size, 0.7 * size)
        train_data <- data[train_idx, ]
        test_data <- data[-train_idx, ]
        
        # Train tree
        start_time <- Sys.time()
        tree <- rpart(y ~ x1 + x2, data = train_data)
        train_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        
        start_time <- Sys.time()
        predict(tree, test_data)
        test_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        
        train_times[i] <- train_time
        test_times[i] <- test_time
      }
      
      # Display results
      cat(sprintf("%-8s %-12s %-12s %-8s\n", "Size", "Train (ms)", "Test (ms)", "Depth"))
      cat(paste(rep("-", 45), collapse = ""), "\n")
      
      for (i in seq_along(sizes)) {
        size <- sizes[i]
        x1 <- rnorm(size, mean = 0, sd = 1)
        x2 <- rnorm(size, mean = 0, sd = 1)
        y <- ifelse(x1^2 + x2^2 < 1.5, 1, 0)
        data <- data.frame(x1 = x1, x2 = x2, y = factor(y))
        
        tree <- rpart(y ~ x1 + x2, data = data)
        depth <- max(tree$frame$depth)
        
        cat(sprintf("%-8d %-12.1f %-12.1f %-8d\n", 
                    size, train_times[i] * 1000, test_times[i] * 1000, depth))
      }
    },
    
    visualize_decision_boundaries = function() {
      # Visualize decision boundaries for different depths
      cat("\n=== Decision Boundary Visualization ===\n")
      
      # Create dataset
      data <- create_2d_dataset(n_samples = 200, noise = 0.3)
      
      # Create plots for different depths
      depths <- c(1, 2, 3, 5, 8, 10)
      plots <- list()
      
      for (i in seq_along(depths)) {
        depth <- depths[i]
        
        # Train tree
        tree <- rpart(y ~ x1 + x2, data = data,
                     control = rpart.control(maxdepth = depth))
        
        # Create grid for decision boundary
        x1_range <- seq(min(data$x1) - 0.5, max(data$x1) + 0.5, length.out = 100)
        x2_range <- seq(min(data$x2) - 0.5, max(data$x2) + 0.5, length.out = 100)
        grid <- expand.grid(x1 = x1_range, x2 = x2_range)
        
        # Predict on grid
        grid$pred <- predict(tree, grid, type = "class")
        
        # Create plot
        p <- ggplot() +
          geom_tile(data = grid, aes(x = x1, y = x2, fill = pred), alpha = 0.4) +
          geom_point(data = data, aes(x = x1, y = x2, color = y), alpha = 0.8) +
          scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
          scale_color_manual(values = c("0" = "blue", "1" = "red")) +
          labs(title = paste("Depth =", depth), x = "Feature 1", y = "Feature 2") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      combined_plot <- do.call(grid.arrange, c(plots, ncol = 3))
      
      # Save plot
      ggsave("decision_boundaries.png", combined_plot, width = 15, height = 10, dpi = 300)
      cat("Decision boundary visualization saved as 'decision_boundaries.png'\n")
      
      return(combined_plot)
    }
  )
)

# Main demonstration function
main <- function() {
  demo <- DecisionTreeDemo$new()
  
  # Run all demonstrations
  demo$demonstrate_recursive_splitting()
  demo$compare_entropy_gini()
  demo$regression_tree_demo()
  demo$regularization_demo()
  demo$runtime_complexity_demo()
  demo$visualize_decision_boundaries()
}

# Run the main function
main()
