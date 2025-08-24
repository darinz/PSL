# Boosting - R Implementation
#
# This script demonstrates boosting algorithms including AdaBoost,
# weak learners, and ensemble methods for improving classification performance.

library(adabag)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(caret)
library(gridExtra)

# Set random seed for reproducibility
set.seed(42)

BoostingDemo <- setRefClass("BoostingDemo",
  fields = list(
    random_state = "numeric"
  ),
  methods = list(
    
    initialize = function(random_state = 42) {
      random_state <<- random_state
      set.seed(random_state)
    },
    
    create_boosting_dataset = function(n_samples = 200, noise = 0.3) {
      # Create a dataset suitable for boosting demonstration
      # Create circular pattern that's difficult for simple classifiers
      theta <- runif(n_samples, 0, 2 * pi)
      r <- runif(n_samples, 0, 1)
      
      # Create two circles
      inner_circle <- r < 0.5
      outer_circle <- r >= 0.5
      
      x1 <- r * cos(theta)
      x2 <- r * sin(theta)
      
      # Add noise
      x1 <- x1 + rnorm(n_samples, 0, noise)
      x2 <- x2 + rnorm(n_samples, 0, noise)
      
      # Create labels
      y <- ifelse(inner_circle, 1, 0)
      
      return(data.frame(x1 = x1, x2 = x2, y = factor(y)))
    },
    
    create_weak_learners_demo = function() {
      # Demonstrate weak learners and their limitations
      cat("=== Weak Learners Demonstration ===\n")
      
      # Create dataset
      data <- create_boosting_dataset(n_samples = 300, noise = 0.2)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different weak learners
      weak_learners <- list(
        list(name = "Decision Stump (depth=1)", params = list(maxdepth = 1)),
        list(name = "Shallow Tree (depth=2)", params = list(maxdepth = 2)),
        list(name = "Very Shallow Tree (depth=1, minsplit=20)", 
             params = list(maxdepth = 1, minsplit = 20)),
        list(name = "Linear Separator (depth=1, maxcompete=1)", 
             params = list(maxdepth = 1, maxcompete = 1))
      )
      
      results <- list()
      
      for (i in seq_along(weak_learners)) {
        learner <- weak_learners[[i]]
        
        # Create control parameters
        control_params <- do.call(rpart.control, learner$params)
        
        tree <- rpart(y ~ x1 + x2, data = train_data, control = control_params)
        
        # Train and evaluate
        train_pred <- predict(tree, train_data, type = "class")
        test_pred <- predict(tree, test_data, type = "class")
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        results[[i]] <- list(
          name = learner$name,
          train_score = train_score,
          test_score = test_score,
          tree = tree
        )
        
        cat(sprintf("%-40s Train: %.3f, Test: %.3f\n", 
                    learner$name, train_score, test_score))
      }
      
      return(list(results = results, train_data = train_data, test_data = test_data))
    },
    
    visualize_weak_learners = function(results, train_data) {
      # Visualize weak learners and their decision boundaries
      cat("\n=== Weak Learners Visualization ===\n")
      
      plots <- list()
      for (i in seq_along(results)) {
        result <- results[[i]]
        tree <- result$tree
        
        # Create grid for decision boundary
        x1_range <- seq(min(train_data$x1) - 0.5, max(train_data$x1) + 0.5, length.out = 100)
        x2_range <- seq(min(train_data$x2) - 0.5, max(train_data$x2) + 0.5, length.out = 100)
        grid <- expand.grid(x1 = x1_range, x2 = x2_range)
        
        # Predict on grid
        grid$pred <- predict(tree, grid, type = "class")
        
        # Create plot
        p <- ggplot() +
          geom_tile(data = grid, aes(x = x1, y = x2, fill = pred), alpha = 0.4) +
          geom_point(data = train_data, aes(x = x1, y = x2, color = y), alpha = 0.8, size = 2) +
          scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
          scale_color_manual(values = c("0" = "blue", "1" = "red")) +
          labs(title = sprintf("%s\nTrain: %.3f, Test: %.3f", 
                              result$name, result$train_score, result$test_score),
               x = "Feature 1", y = "Feature 2") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      combined_plot <- do.call(grid.arrange, c(plots, ncol = 2))
      ggsave("weak_learners.png", combined_plot, width = 12, height = 10, dpi = 300)
      
      cat("Weak learners visualization saved as 'weak_learners.png'\n")
      
      return(combined_plot)
    },
    
    adaboost_demo = function() {
      # Demonstrate AdaBoost algorithm
      cat("\n=== AdaBoost Demonstration ===\n")
      
      # Create dataset
      data <- create_boosting_dataset(n_samples = 400, noise = 0.25)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different numbers of iterations
      n_iterations_list <- c(1, 5, 10, 20, 50, 100)
      results <- list()
      
      for (i in seq_along(n_iterations_list)) {
        n_iterations <- n_iterations_list[i]
        
        # Create AdaBoost classifier
        ada <- boosting(y ~ x1 + x2, data = train_data, 
                       mfinal = n_iterations, 
                       control = rpart.control(maxdepth = 1))
        
        # Train and evaluate
        train_pred <- predict(ada, train_data)
        test_pred <- predict(ada, test_data)
        
        train_score <- mean(train_pred$class == train_data$y)
        test_score <- mean(test_pred$class == test_data$y)
        
        # Cross-validation (simplified)
        cv_scores <- numeric(5)
        for (j in 1:5) {
          cv_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
          cv_train <- train_data[cv_indices, ]
          cv_test <- train_data[-cv_indices, ]
          
          cv_ada <- boosting(y ~ x1 + x2, data = cv_train, 
                           mfinal = n_iterations, 
                           control = rpart.control(maxdepth = 1))
          cv_pred <- predict(cv_ada, cv_test)
          cv_scores[j] <- mean(cv_pred$class == cv_test$y)
        }
        
        results[[i]] <- list(
          n_iterations = n_iterations,
          train_score = train_score,
          test_score = test_score,
          cv_mean = mean(cv_scores),
          cv_std = sd(cv_scores),
          classifier = ada
        )
        
        cat(sprintf("Iterations: %3d, Train: %.3f, Test: %.3f, CV: %.3f ± %.3f\n",
                    n_iterations, train_score, test_score, mean(cv_scores), sd(cv_scores)))
      }
      
      return(list(results = results, train_data = train_data, test_data = test_data))
    },
    
    visualize_adaboost_progression = function(results, train_data) {
      # Visualize AdaBoost progression with different numbers of iterations
      cat("\n=== AdaBoost Progression Visualization ===\n")
      
      # Select specific numbers of iterations to visualize
      viz_iterations <- c(1, 5, 10, 50)
      plots <- list()
      
      for (i in seq_along(viz_iterations)) {
        n_iter <- viz_iterations[i]
        
        # Find the corresponding result
        result <- results[[which(sapply(results, function(x) x$n_iterations == n_iter))]]
        ada <- result$classifier
        
        # Create grid for decision boundary
        x1_range <- seq(min(train_data$x1) - 0.5, max(train_data$x1) + 0.5, length.out = 100)
        x2_range <- seq(min(train_data$x2) - 0.5, max(train_data$x2) + 0.5, length.out = 100)
        grid <- expand.grid(x1 = x1_range, x2 = x2_range)
        
        # Predict on grid
        grid_pred <- predict(ada, grid)
        grid$pred <- grid_pred$class
        
        # Create plot
        p <- ggplot() +
          geom_tile(data = grid, aes(x = x1, y = x2, fill = pred), alpha = 0.4) +
          geom_point(data = train_data, aes(x = x1, y = x2, color = y), alpha = 0.8, size = 2) +
          scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
          scale_color_manual(values = c("0" = "blue", "1" = "red")) +
          labs(title = sprintf("AdaBoost with %d iterations\nTrain: %.3f, Test: %.3f",
                              n_iter, result$train_score, result$test_score),
               x = "Feature 1", y = "Feature 2") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      combined_plot <- do.call(grid.arrange, c(plots, ncol = 2))
      ggsave("adaboost_progression.png", combined_plot, width = 12, height = 10, dpi = 300)
      
      cat("AdaBoost progression visualization saved as 'adaboost_progression.png'\n")
      
      return(combined_plot)
    },
    
    analyze_boosting_performance = function(results) {
      # Analyze boosting performance vs number of iterations
      cat("\n=== Boosting Performance Analysis ===\n")
      
      # Extract data for plotting
      n_iterations <- sapply(results, function(x) x$n_iterations)
      train_scores <- sapply(results, function(x) x$train_score)
      test_scores <- sapply(results, function(x) x$test_score)
      cv_means <- sapply(results, function(x) x$cv_mean)
      cv_stds <- sapply(results, function(x) x$cv_std)
      
      # Create plots
      plot_data <- data.frame(
        n_iterations = n_iterations,
        train_score = train_scores,
        test_score = test_scores,
        cv_mean = cv_means,
        cv_std = cv_stds,
        overfitting_gap = train_scores - test_scores
      )
      
      # Training vs Test accuracy
      p1 <- ggplot(plot_data, aes(x = n_iterations)) +
        geom_line(aes(y = train_score, color = "Training"), size = 1) +
        geom_point(aes(y = train_score, color = "Training"), size = 3) +
        geom_line(aes(y = test_score, color = "Test"), size = 1) +
        geom_point(aes(y = test_score, color = "Test"), size = 3) +
        labs(title = "Accuracy vs Number of Iterations", x = "Number of Iterations", y = "Accuracy") +
        scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Cross-validation scores
      p2 <- ggplot(plot_data, aes(x = n_iterations, y = cv_mean)) +
        geom_errorbar(aes(ymin = cv_mean - cv_std, ymax = cv_mean + cv_std), 
                     width = 0.2, color = "green") +
        geom_line(color = "green", size = 1) +
        geom_point(color = "green", size = 3) +
        labs(title = "CV Accuracy vs Number of Iterations", x = "Number of Iterations", y = "CV Accuracy") +
        theme_minimal()
      
      # Overfitting gap
      p3 <- ggplot(plot_data, aes(x = n_iterations, y = overfitting_gap)) +
        geom_line(color = "red", size = 1) +
        geom_point(color = "red", size = 3) +
        labs(title = "Training-Test Gap vs Number of Iterations", x = "Number of Iterations", y = "Overfitting Gap") +
        theme_minimal()
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
      ggsave("boosting_performance.png", combined_plot, width = 15, height = 5, dpi = 300)
      
      cat("Boosting performance analysis saved as 'boosting_performance.png'\n")
      
      return(plot_data)
    },
    
    analyze_estimator_weights = function(ada_classifier) {
      # Analyze the weights of individual estimators in AdaBoost
      cat("\n=== Estimator Weights Analysis ===\n")
      
      # Get estimator weights
      weights <- ada_classifier$weights
      n_estimators <- length(weights)
      
      # Create plots
      plot_data <- data.frame(
        estimator = 1:n_estimators,
        weight = weights,
        cumulative_weight = cumsum(weights)
      )
      
      # Individual weights
      p1 <- ggplot(plot_data, aes(x = estimator, y = weight)) +
        geom_line(size = 1) +
        geom_point(size = 3) +
        labs(title = "Estimator Weights in AdaBoost", x = "Estimator Index", y = "Weight") +
        theme_minimal()
      
      # Cumulative weights
      p2 <- ggplot(plot_data, aes(x = estimator, y = cumulative_weight)) +
        geom_line(size = 1) +
        geom_point(size = 3) +
        labs(title = "Cumulative Estimator Weights", x = "Number of Estimators", y = "Cumulative Weight") +
        theme_minimal()
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, ncol = 2)
      ggsave("estimator_weights.png", combined_plot, width = 12, height = 5, dpi = 300)
      
      cat("Estimator weights analysis saved as 'estimator_weights.png'\n")
      
      # Print statistics
      cat(sprintf("Number of estimators: %d\n", n_estimators))
      cat(sprintf("Average weight: %.4f\n", mean(weights)))
      cat(sprintf("Weight standard deviation: %.4f\n", sd(weights)))
      cat(sprintf("Min weight: %.4f\n", min(weights)))
      cat(sprintf("Max weight: %.4f\n", max(weights)))
      
      return(weights)
    },
    
    compare_with_single_tree = function(ada_results, train_data, test_data) {
      # Compare AdaBoost with a single deep decision tree
      cat("\n=== AdaBoost vs Single Deep Tree Comparison ===\n")
      
      # Train a single deep tree
      deep_tree <- rpart(y ~ x1 + x2, data = train_data,
                        control = rpart.control(maxdepth = 10, minsplit = 5))
      
      # Evaluate deep tree
      deep_tree_train_pred <- predict(deep_tree, train_data, type = "class")
      deep_tree_test_pred <- predict(deep_tree, test_data, type = "class")
      
      deep_tree_train <- mean(deep_tree_train_pred == train_data$y)
      deep_tree_test <- mean(deep_tree_test_pred == test_data$y)
      
      # Cross-validation for deep tree
      cv_scores <- numeric(5)
      for (j in 1:5) {
        cv_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
        cv_train <- train_data[cv_indices, ]
        cv_test <- train_data[-cv_indices, ]
        
        cv_tree <- rpart(y ~ x1 + x2, data = cv_train,
                        control = rpart.control(maxdepth = 10, minsplit = 5))
        cv_pred <- predict(cv_tree, cv_test, type = "class")
        cv_scores[j] <- mean(cv_pred == cv_test$y)
      }
      
      cat(sprintf("Single Deep Tree:\n"))
      cat(sprintf("  Train accuracy: %.3f\n", deep_tree_train))
      cat(sprintf("  Test accuracy: %.3f\n", deep_tree_test))
      cat(sprintf("  CV accuracy: %.3f ± %.3f\n", mean(cv_scores), sd(cv_scores)))
      cat(sprintf("  Tree depth: %d\n", max(deep_tree$frame$depth)))
      cat(sprintf("  Number of leaves: %d\n", sum(deep_tree$frame$var == "<leaf>")))
      
      # Compare with best AdaBoost result
      best_ada <- results[[which.max(sapply(results, function(x) x$test_score))]]
      cat(sprintf("\nBest AdaBoost (%d iterations):\n", best_ada$n_iterations))
      cat(sprintf("  Train accuracy: %.3f\n", best_ada$train_score))
      cat(sprintf("  Test accuracy: %.3f\n", best_ada$test_score))
      cat(sprintf("  CV accuracy: %.3f ± %.3f\n", best_ada$cv_mean, best_ada$cv_std))
      
      # Create comparison plot
      comparison_data <- data.frame(
        Method = rep(c("Single Deep Tree", "AdaBoost"), each = 3),
        Metric = rep(c("Training", "Test", "CV"), 2),
        Accuracy = c(deep_tree_train, deep_tree_test, mean(cv_scores),
                    best_ada$train_score, best_ada$test_score, best_ada$cv_mean)
      )
      
      p <- ggplot(comparison_data, aes(x = Metric, y = Accuracy, fill = Method)) +
        geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
        labs(title = "Single Deep Tree vs AdaBoost Comparison", x = "Metric", y = "Accuracy") +
        theme_minimal()
      
      ggsave("adaboost_vs_deep_tree.png", p, width = 10, height = 6, dpi = 300)
      
      cat("Comparison visualization saved as 'adaboost_vs_deep_tree.png'\n")
      
      return(list(deep_tree = deep_tree, best_ada = best_ada))
    },
    
    demonstrate_boosting_robustness = function() {
      # Demonstrate robustness of boosting to noise
      cat("\n=== Boosting Robustness to Noise ===\n")
      
      # Test different noise levels
      noise_levels <- c(0.1, 0.2, 0.3, 0.4, 0.5)
      results <- list()
      
      for (i in seq_along(noise_levels)) {
        noise <- noise_levels[i]
        
        # Create dataset with specific noise level
        data <- create_boosting_dataset(n_samples = 300, noise = noise)
        
        # Split data
        train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
        train_data <- data[train_indices, ]
        test_data <- data[-train_indices, ]
        
        # Train AdaBoost
        ada <- boosting(y ~ x1 + x2, data = train_data, 
                       mfinal = 50, 
                       control = rpart.control(maxdepth = 1))
        
        # Train single deep tree
        deep_tree <- rpart(y ~ x1 + x2, data = train_data,
                          control = rpart.control(maxdepth = 10))
        
        # Evaluate
        ada_pred <- predict(ada, test_data)
        tree_pred <- predict(deep_tree, test_data, type = "class")
        
        ada_test <- mean(ada_pred$class == test_data$y)
        tree_test <- mean(tree_pred == test_data$y)
        
        results[[i]] <- list(
          noise = noise,
          adaboost = ada_test,
          deep_tree = tree_test
        )
        
        cat(sprintf("Noise: %.1f, AdaBoost: %.3f, Deep Tree: %.3f\n", 
                    noise, ada_test, tree_test))
      }
      
      # Plot robustness comparison
      plot_data <- data.frame(
        noise = sapply(results, function(x) x$noise),
        adaboost = sapply(results, function(x) x$adaboost),
        deep_tree = sapply(results, function(x) x$deep_tree)
      )
      
      p <- ggplot(plot_data, aes(x = noise)) +
        geom_line(aes(y = adaboost, color = "AdaBoost"), size = 1) +
        geom_point(aes(y = adaboost, color = "AdaBoost"), size = 3) +
        geom_line(aes(y = deep_tree, color = "Deep Tree"), size = 1) +
        geom_point(aes(y = deep_tree, color = "Deep Tree"), size = 3) +
        labs(title = "Robustness to Noise: AdaBoost vs Deep Tree", 
             x = "Noise Level", y = "Test Accuracy") +
        scale_color_manual(values = c("AdaBoost" = "blue", "Deep Tree" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      ggsave("boosting_robustness.png", p, width = 10, height = 6, dpi = 300)
      
      cat("Robustness analysis saved as 'boosting_robustness.png'\n")
      
      return(results)
    },
    
    run_complete_analysis = function() {
      # Run complete boosting analysis
      cat("=== Complete Boosting Analysis ===\n")
      
      # 1. Weak learners demonstration
      weak_demo <- create_weak_learners_demo()
      visualize_weak_learners(weak_demo$results, weak_demo$train_data)
      
      # 2. AdaBoost demonstration
      ada_demo <- adaboost_demo()
      visualize_adaboost_progression(ada_demo$results, ada_demo$train_data)
      analyze_boosting_performance(ada_demo$results)
      
      # 3. Estimator weights analysis
      best_ada <- ada_demo$results[[which.max(sapply(ada_demo$results, function(x) x$test_score))]]
      analyze_estimator_weights(best_ada$classifier)
      
      # 4. Comparison with single deep tree
      compare_with_single_tree(ada_demo$results, ada_demo$train_data, ada_demo$test_data)
      
      # 5. Robustness demonstration
      demonstrate_boosting_robustness()
      
      cat("\n=== Analysis Complete ===\n")
      cat("Generated files:\n")
      cat("- weak_learners.png: Weak learners visualization\n")
      cat("- adaboost_progression.png: AdaBoost progression\n")
      cat("- boosting_performance.png: Performance analysis\n")
      cat("- estimator_weights.png: Estimator weights analysis\n")
      cat("- adaboost_vs_deep_tree.png: Comparison with deep tree\n")
      cat("- boosting_robustness.png: Robustness to noise\n")
    }
  )
)

# Main function to run the boosting analysis
main <- function() {
  demo <- BoostingDemo$new(random_state = 42)
  demo$run_complete_analysis()
}

# Run the main function
main()
