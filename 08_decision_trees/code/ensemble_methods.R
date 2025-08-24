# Ensemble Methods - R Implementation
#
# This script demonstrates ensemble methods including bagging, random forests,
# and model averaging techniques for improving decision tree performance.

library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(caret)
library(gridExtra)

# Set random seed for reproducibility
set.seed(42)

EnsembleMethodsDemo <- setRefClass("EnsembleMethodsDemo",
  fields = list(
    random_state = "numeric"
  ),
  methods = list(
    
    initialize = function(random_state = 42) {
      random_state <<- random_state
      set.seed(random_state)
    },
    
    create_ensemble_dataset = function(n_samples = 300, noise = 0.3) {
      # Create a dataset suitable for ensemble methods demonstration
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
    
    demonstrate_single_tree_limitations = function() {
      # Demonstrate limitations of single decision trees
      cat("=== Single Decision Tree Limitations ===\n")
      
      # Create dataset
      data <- create_ensemble_dataset(n_samples = 400, noise = 0.25)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different single tree configurations
      tree_configs <- list(
        list(name = "Shallow Tree (depth=3)", params = list(maxdepth = 3)),
        list(name = "Medium Tree (depth=5)", params = list(maxdepth = 5)),
        list(name = "Deep Tree (depth=10)", params = list(maxdepth = 10)),
        list(name = "Unlimited Tree", params = list())
      )
      
      results <- list()
      
      for (i in seq_along(tree_configs)) {
        config <- tree_configs[[i]]
        
        # Create control parameters
        control_params <- do.call(rpart.control, config$params)
        
        tree <- rpart(y ~ x1 + x2, data = train_data, control = control_params)
        
        # Train and evaluate
        train_pred <- predict(tree, train_data, type = "class")
        test_pred <- predict(tree, test_data, type = "class")
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        # Cross-validation
        cv_scores <- numeric(5)
        for (j in 1:5) {
          cv_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
          cv_train <- train_data[cv_indices, ]
          cv_test <- train_data[-cv_indices, ]
          
          cv_tree <- rpart(y ~ x1 + x2, data = cv_train, control = control_params)
          cv_pred <- predict(cv_tree, cv_test, type = "class")
          cv_scores[j] <- mean(cv_pred == cv_test$y)
        }
        
        results[[i]] <- list(
          name = config$name,
          train_score = train_score,
          test_score = test_score,
          cv_mean = mean(cv_scores),
          cv_std = sd(cv_scores),
          tree = tree
        )
        
        cat(sprintf("%-25s Train: %.3f, Test: %.3f, CV: %.3f ± %.3f\n",
                    config$name, train_score, test_score, mean(cv_scores), sd(cv_scores)))
      }
      
      return(list(results = results, train_data = train_data, test_data = test_data))
    },
    
    visualize_single_trees = function(results, train_data) {
      # Visualize single decision trees and their limitations
      cat("\n=== Single Trees Visualization ===\n")
      
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
      ggsave("single_trees_limitations.png", combined_plot, width = 12, height = 10, dpi = 300)
      
      cat("Single trees visualization saved as 'single_trees_limitations.png'\n")
      
      return(combined_plot)
    },
    
    bagging_demo = function() {
      # Demonstrate bagging (Bootstrap Aggregating)
      cat("\n=== Bagging Demonstration ===\n")
      
      # Create dataset
      data <- create_ensemble_dataset(n_samples = 500, noise = 0.2)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different numbers of estimators
      n_estimators_list <- c(1, 5, 10, 20, 50, 100)
      results <- list()
      
      for (i in seq_along(n_estimators_list)) {
        n_estimators <- n_estimators_list[i]
        
        # Create bagging classifier using randomForest
        bagging <- randomForest(y ~ x1 + x2, data = train_data, 
                              ntree = n_estimators, 
                              replace = TRUE,
                              sampsize = nrow(train_data))
        
        # Train and evaluate
        train_pred <- predict(bagging, train_data)
        test_pred <- predict(bagging, test_data)
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        # Cross-validation
        cv_scores <- numeric(5)
        for (j in 1:5) {
          cv_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
          cv_train <- train_data[cv_indices, ]
          cv_test <- train_data[-cv_indices, ]
          
          cv_bagging <- randomForest(y ~ x1 + x2, data = cv_train, 
                                   ntree = n_estimators, 
                                   replace = TRUE,
                                   sampsize = nrow(cv_train))
          cv_pred <- predict(cv_bagging, cv_test)
          cv_scores[j] <- mean(cv_pred == cv_test$y)
        }
        
        results[[i]] <- list(
          n_estimators = n_estimators,
          train_score = train_score,
          test_score = test_score,
          cv_mean = mean(cv_scores),
          cv_std = sd(cv_scores),
          classifier = bagging
        )
        
        cat(sprintf("Estimators: %3d, Train: %.3f, Test: %.3f, CV: %.3f ± %.3f\n",
                    n_estimators, train_score, test_score, mean(cv_scores), sd(cv_scores)))
      }
      
      return(list(results = results, train_data = train_data, test_data = test_data))
    },
    
    random_forest_demo = function() {
      # Demonstrate Random Forest
      cat("\n=== Random Forest Demonstration ===\n")
      
      # Create dataset
      data <- create_ensemble_dataset(n_samples = 500, noise = 0.2)
      
      # Split data
      train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
      train_data <- data[train_indices, ]
      test_data <- data[-train_indices, ]
      
      # Test different Random Forest configurations
      rf_configs <- list(
        list(name = "Small RF (10 trees)", params = list(ntree = 10)),
        list(name = "Medium RF (50 trees)", params = list(ntree = 50)),
        list(name = "Large RF (100 trees)", params = list(ntree = 100)),
        list(name = "RF with mtry=1", params = list(ntree = 50, mtry = 1)),
        list(name = "RF with maxnodes=10", params = list(ntree = 50, maxnodes = 10)),
        list(name = "RF with replace=FALSE", params = list(ntree = 50, replace = FALSE))
      )
      
      results <- list()
      
      for (i in seq_along(rf_configs)) {
        config <- rf_configs[[i]]
        
        rf <- do.call(randomForest, c(list(formula = y ~ x1 + x2, data = train_data), config$params))
        
        # Train and evaluate
        train_pred <- predict(rf, train_data)
        test_pred <- predict(rf, test_data)
        
        train_score <- mean(train_pred == train_data$y)
        test_score <- mean(test_pred == test_data$y)
        
        # Cross-validation
        cv_scores <- numeric(5)
        for (j in 1:5) {
          cv_indices <- sample(1:nrow(train_data), 0.8 * nrow(train_data))
          cv_train <- train_data[cv_indices, ]
          cv_test <- train_data[-cv_indices, ]
          
          cv_rf <- do.call(randomForest, c(list(formula = y ~ x1 + x2, data = cv_train), config$params))
          cv_pred <- predict(cv_rf, cv_test)
          cv_scores[j] <- mean(cv_pred == cv_test$y)
        }
        
        results[[i]] <- list(
          name = config$name,
          train_score = train_score,
          test_score = test_score,
          cv_mean = mean(cv_scores),
          cv_std = sd(cv_scores),
          classifier = rf
        )
        
        cat(sprintf("%-25s Train: %.3f, Test: %.3f, CV: %.3f ± %.3f\n",
                    config$name, train_score, test_score, mean(cv_scores), sd(cv_scores)))
      }
      
      return(list(results = results, train_data = train_data, test_data = test_data))
    },
    
    analyze_ensemble_performance = function(bagging_results, rf_results) {
      # Analyze performance of different ensemble methods
      cat("\n=== Ensemble Performance Analysis ===\n")
      
      # Create plots
      plot_data_bagging <- data.frame(
        n_estimators = sapply(bagging_results, function(x) x$n_estimators),
        train_score = sapply(bagging_results, function(x) x$train_score),
        test_score = sapply(bagging_results, function(x) x$test_score)
      )
      
      plot_data_rf <- data.frame(
        name = sapply(rf_results, function(x) x$name),
        test_score = sapply(rf_results, function(x) x$test_score)
      )
      
      # Bagging performance
      p1 <- ggplot(plot_data_bagging, aes(x = n_estimators)) +
        geom_line(aes(y = train_score, color = "Training"), size = 1) +
        geom_point(aes(y = train_score, color = "Training"), size = 3) +
        geom_line(aes(y = test_score, color = "Test"), size = 1) +
        geom_point(aes(y = test_score, color = "Test"), size = 3) +
        labs(title = "Bagging Performance", x = "Number of Estimators", y = "Accuracy") +
        scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Random Forest comparison
      p2 <- ggplot(plot_data_rf, aes(x = reorder(name, test_score), y = test_score)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        labs(title = "Random Forest Configurations", x = "Configuration", y = "Test Accuracy") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      # Overfitting comparison
      overfitting_gap_bagging <- plot_data_bagging$train_score - plot_data_bagging$test_score
      overfitting_gap_rf <- sapply(rf_results, function(x) x$train_score - x$test_score)
      
      p3 <- ggplot() +
        geom_line(data = plot_data_bagging, aes(x = n_estimators, y = overfitting_gap_bagging, color = "Bagging"), size = 1) +
        geom_point(data = plot_data_bagging, aes(x = n_estimators, y = overfitting_gap_bagging, color = "Bagging"), size = 3) +
        geom_hline(yintercept = mean(overfitting_gap_rf), color = "red", linetype = "dashed", size = 1) +
        annotate("text", x = max(plot_data_bagging$n_estimators), y = mean(overfitting_gap_rf), 
                label = "RF Average", color = "red", hjust = 1) +
        labs(title = "Overfitting Comparison", x = "Number of Estimators", y = "Overfitting Gap") +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
      ggsave("ensemble_performance.png", combined_plot, width = 15, height = 5, dpi = 300)
      
      cat("Ensemble performance analysis saved as 'ensemble_performance.png'\n")
      
      return(list(bagging_results = bagging_results, rf_results = rf_results))
    },
    
    visualize_ensemble_decision_boundaries = function(bagging_results, rf_results, train_data) {
      # Visualize decision boundaries of ensemble methods
      cat("\n=== Ensemble Decision Boundaries Visualization ===\n")
      
      # Select specific configurations to visualize
      viz_configs <- list(
        list(name = "Bagging (10 estimators)", classifier = bagging_results[[3]]$classifier),
        list(name = "Bagging (50 estimators)", classifier = bagging_results[[5]]$classifier),
        list(name = "Random Forest (50 trees)", classifier = rf_results[[2]]$classifier),
        list(name = "Random Forest (mtry=1)", classifier = rf_results[[4]]$classifier)
      )
      
      plots <- list()
      for (i in seq_along(viz_configs)) {
        config <- viz_configs[[i]]
        classifier <- config$classifier
        
        # Create grid for decision boundary
        x1_range <- seq(min(train_data$x1) - 0.5, max(train_data$x1) + 0.5, length.out = 100)
        x2_range <- seq(min(train_data$x2) - 0.5, max(train_data$x2) + 0.5, length.out = 100)
        grid <- expand.grid(x1 = x1_range, x2 = x2_range)
        
        # Predict on grid
        grid$pred <- predict(classifier, grid)
        
        # Create plot
        p <- ggplot() +
          geom_tile(data = grid, aes(x = x1, y = x2, fill = pred), alpha = 0.4) +
          geom_point(data = train_data, aes(x = x1, y = x2, color = y), alpha = 0.8, size = 2) +
          scale_fill_manual(values = c("0" = "lightblue", "1" = "lightcoral")) +
          scale_color_manual(values = c("0" = "blue", "1" = "red")) +
          labs(title = config$name, x = "Feature 1", y = "Feature 2") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      combined_plot <- do.call(grid.arrange, c(plots, ncol = 2))
      ggsave("ensemble_decision_boundaries.png", combined_plot, width = 12, height = 10, dpi = 300)
      
      cat("Ensemble decision boundaries saved as 'ensemble_decision_boundaries.png'\n")
      
      return(combined_plot)
    },
    
    analyze_feature_importance = function(rf_results, train_data) {
      # Analyze feature importance in Random Forest
      cat("\n=== Feature Importance Analysis ===\n")
      
      # Get the best Random Forest classifier
      best_rf <- rf_results[[which.max(sapply(rf_results, function(x) x$test_score))]]
      rf <- best_rf$classifier
      
      # Get feature importance
      feature_importance <- importance(rf)
      feature_names <- rownames(feature_importance)
      
      # Create importance plot
      importance_df <- data.frame(
        feature = feature_names,
        importance = as.numeric(feature_importance)
      )
      
      p <- ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        labs(title = "Random Forest Feature Importance", x = "Feature", y = "Importance") +
        theme_minimal() +
        coord_flip()
      
      ggsave("feature_importance.png", p, width = 10, height = 6, dpi = 300)
      
      cat("Feature importance analysis saved as 'feature_importance.png'\n")
      
      # Print importance values
      cat("Feature Importance:\n")
      for (i in 1:nrow(importance_df)) {
        cat(sprintf("  %s: %.4f\n", importance_df$feature[i], importance_df$importance[i]))
      }
      
      return(importance_df)
    },
    
    compare_ensemble_methods = function(single_tree_results, bagging_results, rf_results) {
      # Compare all ensemble methods
      cat("\n=== Ensemble Methods Comparison ===\n")
      
      # Get best results from each method
      best_single <- single_tree_results[[which.max(sapply(single_tree_results, function(x) x$test_score))]]
      best_bagging <- bagging_results[[which.max(sapply(bagging_results, function(x) x$test_score))]]
      best_rf <- rf_results[[which.max(sapply(rf_results, function(x) x$test_score))]]
      
      # Create comparison table
      comparison_data <- data.frame(
        Method = c("Single Tree", "Bagging", "Random Forest"),
        Configuration = c(
          best_single$name,
          sprintf("%d estimators", best_bagging$n_estimators),
          best_rf$name
        ),
        Train_Score = c(best_single$train_score, best_bagging$train_score, best_rf$train_score),
        Test_Score = c(best_single$test_score, best_bagging$test_score, best_rf$test_score),
        CV_Score = c(best_single$cv_mean, best_bagging$cv_mean, best_rf$cv_mean),
        Overfitting_Gap = c(
          best_single$train_score - best_single$test_score,
          best_bagging$train_score - best_bagging$test_score,
          best_rf$train_score - best_rf$test_score
        )
      )
      
      cat("\nEnsemble Methods Comparison:\n")
      print(comparison_data)
      
      # Create comparison plot
      p1 <- ggplot(comparison_data, aes(x = Method, y = Test_Score, fill = Method)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        labs(title = "Test Performance Comparison", x = "Method", y = "Test Accuracy") +
        theme_minimal() +
        theme(legend.position = "none")
      
      p2 <- ggplot(comparison_data, aes(x = Method, y = Overfitting_Gap, fill = Method)) +
        geom_bar(stat = "identity", alpha = 0.8) +
        labs(title = "Overfitting Comparison", x = "Method", y = "Overfitting Gap") +
        theme_minimal() +
        theme(legend.position = "none")
      
      # Combine plots
      combined_plot <- grid.arrange(p1, p2, ncol = 2)
      ggsave("ensemble_comparison.png", combined_plot, width = 12, height = 5, dpi = 300)
      
      cat("Ensemble comparison saved as 'ensemble_comparison.png'\n")
      
      return(comparison_data)
    },
    
    demonstrate_ensemble_robustness = function() {
      # Demonstrate robustness of ensemble methods
      cat("\n=== Ensemble Methods Robustness ===\n")
      
      # Test different noise levels
      noise_levels <- c(0.1, 0.2, 0.3, 0.4, 0.5)
      results <- list()
      
      for (i in seq_along(noise_levels)) {
        noise <- noise_levels[i]
        
        # Create dataset with specific noise level
        data <- create_ensemble_dataset(n_samples = 400, noise = noise)
        
        # Split data
        train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
        train_data <- data[train_indices, ]
        test_data <- data[-train_indices, ]
        
        # Train different methods
        single_tree <- rpart(y ~ x1 + x2, data = train_data, 
                           control = rpart.control(maxdepth = 5))
        bagging <- randomForest(y ~ x1 + x2, data = train_data, 
                              ntree = 50, replace = TRUE, sampsize = nrow(train_data))
        rf <- randomForest(y ~ x1 + x2, data = train_data, ntree = 50)
        
        # Evaluate
        single_pred <- predict(single_tree, test_data, type = "class")
        bagging_pred <- predict(bagging, test_data)
        rf_pred <- predict(rf, test_data)
        
        single_score <- mean(single_pred == test_data$y)
        bagging_score <- mean(bagging_pred == test_data$y)
        rf_score <- mean(rf_pred == test_data$y)
        
        results[[i]] <- list(
          noise = noise,
          single_tree = single_score,
          bagging = bagging_score,
          random_forest = rf_score
        )
        
        cat(sprintf("Noise: %.1f, Single: %.3f, Bagging: %.3f, RF: %.3f\n", 
                    noise, single_score, bagging_score, rf_score))
      }
      
      # Plot robustness comparison
      plot_data <- data.frame(
        noise = sapply(results, function(x) x$noise),
        single_tree = sapply(results, function(x) x$single_tree),
        bagging = sapply(results, function(x) x$bagging),
        random_forest = sapply(results, function(x) x$random_forest)
      )
      
      p <- ggplot(plot_data, aes(x = noise)) +
        geom_line(aes(y = single_tree, color = "Single Tree"), size = 1) +
        geom_point(aes(y = single_tree, color = "Single Tree"), size = 3) +
        geom_line(aes(y = bagging, color = "Bagging"), size = 1) +
        geom_point(aes(y = bagging, color = "Bagging"), size = 3) +
        geom_line(aes(y = random_forest, color = "Random Forest"), size = 1) +
        geom_point(aes(y = random_forest, color = "Random Forest"), size = 3) +
        labs(title = "Robustness to Noise: Ensemble Methods Comparison", 
             x = "Noise Level", y = "Test Accuracy") +
        scale_color_manual(values = c("Single Tree" = "blue", "Bagging" = "green", "Random Forest" = "red")) +
        theme_minimal() +
        theme(legend.title = element_blank())
      
      ggsave("ensemble_robustness.png", p, width = 10, height = 6, dpi = 300)
      
      cat("Ensemble robustness analysis saved as 'ensemble_robustness.png'\n")
      
      return(results)
    },
    
    run_complete_analysis = function() {
      # Run complete ensemble methods analysis
      cat("=== Complete Ensemble Methods Analysis ===\n")
      
      # 1. Single tree limitations
      single_demo <- demonstrate_single_tree_limitations()
      visualize_single_trees(single_demo$results, single_demo$train_data)
      
      # 2. Bagging demonstration
      bagging_demo <- bagging_demo()
      
      # 3. Random Forest demonstration
      rf_demo <- random_forest_demo()
      
      # 4. Performance analysis
      analyze_ensemble_performance(bagging_demo$results, rf_demo$results)
      
      # 5. Decision boundaries visualization
      visualize_ensemble_decision_boundaries(bagging_demo$results, rf_demo$results, bagging_demo$train_data)
      
      # 6. Feature importance analysis
      analyze_feature_importance(rf_demo$results, rf_demo$train_data)
      
      # 7. Methods comparison
      compare_ensemble_methods(single_demo$results, bagging_demo$results, rf_demo$results)
      
      # 8. Robustness demonstration
      demonstrate_ensemble_robustness()
      
      cat("\n=== Analysis Complete ===\n")
      cat("Generated files:\n")
      cat("- single_trees_limitations.png: Single trees visualization\n")
      cat("- ensemble_performance.png: Performance analysis\n")
      cat("- ensemble_decision_boundaries.png: Decision boundaries\n")
      cat("- feature_importance.png: Feature importance analysis\n")
      cat("- ensemble_comparison.png: Methods comparison\n")
      cat("- ensemble_robustness.png: Robustness analysis\n")
    }
  )
)

# Main function to run the ensemble methods analysis
main <- function() {
  demo <- EnsembleMethodsDemo$new(random_state = 42)
  demo$run_complete_analysis()
}

# Run the main function
main()
