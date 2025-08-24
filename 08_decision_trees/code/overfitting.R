# Overfitting in Decision Trees - R Implementation
#
# This script demonstrates overfitting in decision trees and various techniques
# to control it, including early stopping, pruning, and regularization.

library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(caret)
library(gridExtra)

# Set random seed for reproducibility
set.seed(42)

OverfittingDemo <- setRefClass("OverfittingDemo",
  fields = list(
    random_state = "numeric"
  ),
  methods = list(
    
    initialize = function(random_state = 42) {
      random_state <<- random_state
      set.seed(random_state)
    },
    
    create_overfitting_dataset = function(n_samples = 200, noise = 0.3) {
      # Create a dataset prone to overfitting
      x1 <- rnorm(n_samples, mean = 0, sd = 1)
      x2 <- rnorm(n_samples, mean = 0, sd = 1)
      
      # Create non-linear decision boundary
      y <- ifelse(x1^2 + x2^2 < 1.5, 1, 0)
      
      # Add noise
      noise_factor <- rnorm(n_samples, mean = 0, sd = noise)
      y <- ifelse(y + noise_factor > 0.5, 1, 0)
      
      return(data.frame(x1 = x1, x2 = x2, y = factor(y)))
    },
    
    demonstrate_depth_vs_performance = function() {
      # Demonstrate how tree depth affects performance
      cat("=== Tree Depth vs Performance Analysis ===\n")
      
      # Create dataset
      data <- create_overfitting_dataset(n_samples = 200, noise = 0.2)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different depths
      depths <- c(1, 2, 3, 5, 8, 10, 15, 20)
      train_scores <- numeric(length(depths))
      test_scores <- numeric(length(depths))
      tree_depths <- numeric(length(depths))
      num_leaves <- numeric(length(depths))
      
      for (i in seq_along(depths)) {
        depth <- depths[i]
        
        # Train tree
        tree <- rpart(y ~ x1 + x2, data = train_data,
                     control = rpart.control(maxdepth = depth))
        
        # Calculate scores
        train_pred <- predict(tree, train_data, type = "class")
        test_pred <- predict(tree, test_data, type = "class")
        
        train_scores[i] <- mean(train_pred == train_data$y)
        test_scores[i] <- mean(test_pred == test_data$y)
        tree_depths[i] <- max(tree$frame$depth)
        num_leaves[i] <- sum(tree$frame$var == "<leaf>")
        
        cat(sprintf("Depth %2d: Train=%.3f, Test=%.3f, Actual Depth=%d, Leaves=%d\n",
                    depth, train_scores[i], test_scores[i], tree_depths[i], num_leaves[i]))
      }
      
      # Create plots
      plot_data <- data.frame(
        depth = depths,
        train_score = train_scores,
        test_score = test_scores,
        num_leaves = num_leaves,
        overfitting_gap = train_scores - test_scores
      )
      
      # Training vs Test accuracy
      p1 <- ggplot(plot_data, aes(x = depth)) +
        geom_line(aes(y = train_score, color = "Training"), size = 1) +
        geom_point(aes(y = train_score, color = "Training"), size = 3) +
        geom_line(aes(y = test_score, color = "Test"), size = 1) +
        geom_point(aes(y = test_score, color = "Test"), size = 3) +
        labs(title = "Accuracy vs Tree Depth", x = "Max Depth", y = "Accuracy") +
        scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Number of leaves
      p2 <- ggplot(plot_data, aes(x = depth, y = num_leaves)) +
        geom_line(color = "green", size = 1) +
        geom_point(color = "green", size = 3) +
        labs(title = "Tree Complexity vs Depth", x = "Max Depth", y = "Number of Leaves") +
        theme_minimal()
      
      # Overfitting gap
      p3 <- ggplot(plot_data, aes(x = depth, y = overfitting_gap)) +
        geom_line(color = "red", size = 1) +
        geom_point(color = "red", size = 3) +
        labs(title = "Training-Test Gap vs Depth", x = "Max Depth", y = "Overfitting Gap") +
        theme_minimal()
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
      ggsave("depth_vs_performance.png", combined_plot, width = 15, height = 5, dpi = 300)
      
      cat("Depth vs performance analysis saved as 'depth_vs_performance.png'\n")
      
      return(list(depths = depths, train_scores = train_scores, 
                  test_scores = test_scores, num_leaves = num_leaves))
    },
    
    early_stopping_demo = function() {
      # Demonstrate early stopping techniques
      cat("\n=== Early Stopping Demonstration ===\n")
      
      # Create dataset
      data <- create_overfitting_dataset(n_samples = 150, noise = 0.25)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Different early stopping conditions
      conditions <- list(
        list(maxdepth = 10, minsplit = 1, minbucket = 1, name = "No constraints"),
        list(maxdepth = 3, minsplit = 1, minbucket = 1, name = "Max depth = 3"),
        list(maxdepth = 10, minsplit = 5, minbucket = 1, name = "Min split = 5"),
        list(maxdepth = 10, minsplit = 1, minbucket = 5, name = "Min bucket = 5"),
        list(maxdepth = 5, minsplit = 3, minbucket = 3, name = "All constraints")
      )
      
      results <- list()
      
      for (i in seq_along(conditions)) {
        condition <- conditions[[i]]
        
        tree <- rpart(y ~ x1 + x2, data = train_data,
                     control = rpart.control(maxdepth = condition$maxdepth,
                                           minsplit = condition$minsplit,
                                           minbucket = condition$minbucket))
        
        train_pred <- predict(tree, train_data, type = "class")
        test_pred <- predict(tree, test_data, type = "class")
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        results[[i]] <- list(
          name = condition$name,
          depth = max(tree$frame$depth),
          leaves = sum(tree$frame$var == "<leaf>"),
          train_score = train_score,
          test_score = test_score,
          overfitting_gap = train_score - test_score
        )
      }
      
      # Display results
      cat(sprintf("%-20s %-6s %-7s %-8s %-8s %-8s\n", 
                  "Condition", "Depth", "Leaves", "Train", "Test", "Gap"))
      cat(paste(rep("-", 65), collapse = ""), "\n")
      
      for (result in results) {
        cat(sprintf("%-20s %-6d %-7d %-8.3f %-8.3f %-8.3f\n",
                    result$name, result$depth, result$leaves,
                    result$train_score, result$test_score, result$overfitting_gap))
      }
      
      return(results)
    },
    
    pruning_demo = function() {
      # Demonstrate pruning techniques
      cat("\n=== Pruning Demonstration ===\n")
      
      # Create dataset
      data <- create_overfitting_dataset(n_samples = 200, noise = 0.3)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Train a complex tree first
      complex_tree <- rpart(y ~ x1 + x2, data = train_data,
                           control = rpart.control(maxdepth = 15, minsplit = 1, minbucket = 1))
      
      cat(sprintf("Complex tree - Depth: %d, Leaves: %d\n", 
                  max(complex_tree$frame$depth), sum(complex_tree$frame$var == "<leaf>")))
      
      train_pred <- predict(complex_tree, train_data, type = "class")
      test_pred <- predict(complex_tree, test_data, type = "class")
      
      cat(sprintf("Training accuracy: %.3f\n", mean(train_pred == train_data$y)))
      cat(sprintf("Test accuracy: %.3f\n", mean(test_pred == test_data$y)))
      
      # Cost complexity pruning
      cp_values <- complex_tree$cptable[, "CP"]
      
      pruned_trees <- list()
      train_scores <- numeric(length(cp_values))
      test_scores <- numeric(length(cp_values))
      depths <- numeric(length(cp_values))
      leaves <- numeric(length(cp_values))
      
      for (i in seq_along(cp_values)) {
        cp <- cp_values[i]
        
        pruned_tree <- prune(complex_tree, cp = cp)
        pruned_trees[[i]] <- pruned_tree
        
        train_pred <- predict(pruned_tree, train_data, type = "class")
        test_pred <- predict(pruned_tree, test_data, type = "class")
        
        train_scores[i] <- mean(train_pred == train_data$y)
        test_scores[i] <- mean(test_pred == test_data$y)
        depths[i] <- max(pruned_tree$frame$depth)
        leaves[i] <- sum(pruned_tree$frame$var == "<leaf>")
      }
      
      # Create plots
      plot_data <- data.frame(
        cp = cp_values,
        train_score = train_scores,
        test_score = test_scores,
        depth = depths,
        leaves = leaves
      )
      
      # Accuracy vs cp
      p1 <- ggplot(plot_data, aes(x = cp)) +
        geom_line(aes(y = train_score, color = "Training"), size = 1) +
        geom_point(aes(y = train_score, color = "Training"), size = 3) +
        geom_line(aes(y = test_score, color = "Test"), size = 1) +
        geom_point(aes(y = test_score, color = "Test"), size = 3) +
        labs(title = "Accuracy vs Pruning Parameter", x = "CP", y = "Accuracy") +
        scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Tree depth vs cp
      p2 <- ggplot(plot_data, aes(x = cp, y = depth)) +
        geom_line(color = "green", size = 1) +
        geom_point(color = "green", size = 3) +
        labs(title = "Tree Depth vs Pruning Parameter", x = "CP", y = "Tree Depth") +
        theme_minimal()
      
      # Number of leaves vs cp
      p3 <- ggplot(plot_data, aes(x = cp, y = leaves)) +
        geom_line(color = "red", size = 1) +
        geom_point(color = "red", size = 3) +
        labs(title = "Tree Complexity vs Pruning Parameter", x = "CP", y = "Number of Leaves") +
        theme_minimal()
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
      ggsave("pruning_analysis.png", combined_plot, width = 15, height = 5, dpi = 300)
      
      cat("Pruning analysis saved as 'pruning_analysis.png'\n")
      
      # Find optimal cp
      best_idx <- which.max(test_scores)
      optimal_cp <- cp_values[best_idx]
      optimal_tree <- pruned_trees[[best_idx]]
      
      cat(sprintf("\nOptimal pruning parameter (CP): %.4f\n", optimal_cp))
      cat(sprintf("Optimal tree - Depth: %d, Leaves: %d\n", 
                  max(optimal_tree$frame$depth), sum(optimal_tree$frame$var == "<leaf>")))
      cat(sprintf("Optimal test accuracy: %.3f\n", test_scores[best_idx]))
      
      return(list(pruned_trees = pruned_trees, cp_values = cp_values, test_scores = test_scores))
    },
    
    regularization_comparison = function() {
      # Compare different regularization techniques
      cat("\n=== Regularization Techniques Comparison ===\n")
      
      # Create dataset
      data <- create_overfitting_dataset(n_samples = 300, noise = 0.2)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Different regularization approaches
      approaches <- list(
        list(name = "No regularization", params = list()),
        list(name = "Max depth only", params = list(maxdepth = 4)),
        list(name = "Min samples only", params = list(minsplit = 10, minbucket = 5)),
        list(name = "Max features only", params = list(maxcompete = 1)),
        list(name = "All constraints", params = list(maxdepth = 4, minsplit = 10, minbucket = 5, maxcompete = 1))
      )
      
      results <- list()
      
      for (i in seq_along(approaches)) {
        approach <- approaches[[i]]
        
        # Create control parameters
        control_params <- do.call(rpart.control, approach$params)
        
        # Cross-validation
        cv_results <- train(y ~ x1 + x2, data = train_data,
                           method = "rpart",
                           trControl = trainControl(method = "cv", number = 5),
                           tuneGrid = data.frame(cp = 0.01),
                           control = control_params)
        
        # Train on full training set
        tree <- rpart(y ~ x1 + x2, data = train_data, control = control_params)
        
        train_pred <- predict(tree, train_data, type = "class")
        test_pred <- predict(tree, test_data, type = "class")
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        results[[i]] <- list(
          name = approach$name,
          depth = max(tree$frame$depth),
          leaves = sum(tree$frame$var == "<leaf>"),
          train_score = train_score,
          test_score = test_score,
          cv_mean = cv_results$results$Accuracy[1],
          cv_std = cv_results$results$AccuracySD[1],
          overfitting_gap = train_score - test_score
        )
      }
      
      # Display results
      cat(sprintf("%-20s %-6s %-7s %-8s %-8s %-8s %-8s\n", 
                  "Approach", "Depth", "Leaves", "Train", "Test", "CV", "Gap"))
      cat(paste(rep("-", 75), collapse = ""), "\n")
      
      for (result in results) {
        cat(sprintf("%-20s %-6d %-7d %-8.3f %-8.3f %-8.3f %-8.3f\n",
                    result$name, result$depth, result$leaves,
                    result$train_score, result$test_score, 
                    result$cv_mean, result$overfitting_gap))
      }
      
      return(results)
    },
    
    bias_variance_analysis = function() {
      # Analyze bias-variance trade-off
      cat("\n=== Bias-Variance Trade-off Analysis ===\n")
      
      # Create multiple datasets
      n_datasets <- 20
      n_samples <- 100
      depths <- c(1, 2, 3, 5, 8, 10)
      
      all_train_scores <- list()
      all_test_scores <- list()
      
      for (i in seq_along(depths)) {
        depth <- depths[i]
        train_scores <- numeric(n_datasets)
        test_scores <- numeric(n_datasets)
        
        for (j in 1:n_datasets) {
          # Create dataset with different random seeds
          set.seed(j)
          data <- create_overfitting_dataset(n_samples = n_samples, noise = 0.3)
          
          # Split data
          train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
          train_data <- data[train_indices, ]
          test_data <- data[-train_indices, ]
          
          # Train tree
          tree <- rpart(y ~ x1 + x2, data = train_data,
                       control = rpart.control(maxdepth = depth))
          
          # Calculate scores
          train_pred <- predict(tree, train_data, type = "class")
          test_pred <- predict(tree, test_data, type = "class")
          
          train_scores[j] <- mean(train_pred == train_data$y)
          test_scores[j] <- mean(test_pred == test_data$y)
        }
        
        all_train_scores[[i]] <- train_scores
        all_test_scores[[i]] <- test_scores
      }
      
      # Calculate bias and variance
      train_means <- sapply(all_train_scores, mean)
      train_stds <- sapply(all_train_scores, sd)
      test_means <- sapply(all_test_scores, mean)
      test_stds <- sapply(all_test_scores, sd)
      
      # Create plots
      plot_data <- data.frame(
        depth = depths,
        train_mean = train_means,
        train_std = train_stds,
        test_mean = test_means,
        test_std = test_stds
      )
      
      # Training performance
      p1 <- ggplot(plot_data, aes(x = depth, y = train_mean)) +
        geom_errorbar(aes(ymin = train_mean - train_std, ymax = train_mean + train_std), 
                     width = 0.2, color = "blue") +
        geom_line(color = "blue", size = 1) +
        geom_point(color = "blue", size = 3) +
        labs(title = "Training Performance (Bias)", x = "Tree Depth", y = "Accuracy") +
        theme_minimal()
      
      # Test performance
      p2 <- ggplot(plot_data, aes(x = depth, y = test_mean)) +
        geom_errorbar(aes(ymin = test_mean - test_std, ymax = test_mean + test_std), 
                     width = 0.2, color = "red") +
        geom_line(color = "red", size = 1) +
        geom_point(color = "red", size = 3) +
        labs(title = "Test Performance (Variance)", x = "Tree Depth", y = "Accuracy") +
        theme_minimal()
      
      # Variance (std) vs depth
      p3 <- ggplot(plot_data, aes(x = depth)) +
        geom_line(aes(y = train_std, color = "Training"), size = 1) +
        geom_point(aes(y = train_std, color = "Training"), size = 3) +
        geom_line(aes(y = test_std, color = "Test"), size = 1) +
        geom_point(aes(y = test_std, color = "Test"), size = 3) +
        labs(title = "Variance vs Tree Depth", x = "Tree Depth", y = "Standard Deviation") +
        scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
      ggsave("bias_variance_analysis.png", combined_plot, width = 15, height = 5, dpi = 300)
      
      cat("Bias-variance analysis saved as 'bias_variance_analysis.png'\n")
      
      return(list(depths = depths, train_means = train_means, test_means = test_means,
                  train_stds = train_stds, test_stds = test_stds))
    },
    
    visualize_overfitting = function() {
      # Visualize overfitting with decision boundaries
      cat("\n=== Overfitting Visualization ===\n")
      
      # Create dataset
      data <- create_overfitting_dataset(n_samples = 150, noise = 0.2)
      
      # Create plots for different depths
      depths <- c(1, 2, 3, 5, 8, 10, 15, 20)
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
          geom_point(data = data, aes(x = x1, y = x2, color = y), alpha = 0.8, size = 2) +
          scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
          scale_color_manual(values = c("0" = "blue", "1" = "red")) +
          labs(title = sprintf("Depth = %d\nLeaves = %d", depth, sum(tree$frame$var == "<leaf>")),
               x = "Feature 1", y = "Feature 2") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      combined_plot <- do.call(grid.arrange, c(plots, ncol = 4))
      ggsave("overfitting_visualization.png", combined_plot, width = 20, height = 10, dpi = 300)
      
      cat("Overfitting visualization saved as 'overfitting_visualization.png'\n")
      
      return(combined_plot)
    },
    
    run_complete_analysis = function() {
      # Run complete overfitting analysis
      cat("=== Complete Overfitting Analysis ===\n")
      
      # Run all demonstrations
      demonstrate_depth_vs_performance()
      early_stopping_demo()
      pruning_demo()
      regularization_comparison()
      bias_variance_analysis()
      visualize_overfitting()
      
      cat("\n=== Analysis Complete ===\n")
      cat("Generated files:\n")
      cat("- depth_vs_performance.png: Depth vs performance analysis\n")
      cat("- pruning_analysis.png: Pruning parameter analysis\n")
      cat("- bias_variance_analysis.png: Bias-variance trade-off\n")
      cat("- overfitting_visualization.png: Decision boundary visualization\n")
    }
  )
)

# Main function to run the overfitting analysis
main <- function() {
  demo <- OverfittingDemo$new(random_state = 42)
  demo$run_complete_analysis()
}

# Run the main function
main()
