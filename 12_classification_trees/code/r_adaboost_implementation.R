# AdaBoost Implementation in R

# Load required libraries
library(rpart)
library(ggplot2)
library(gridExtra)
library(dplyr)

# AdaBoost implementation
ada_boost <- function(X, y, n_estimators = 50, max_depth = 1) {
  n_samples <- nrow(X)
  
  # Initialize weights
  sample_weights <- rep(1/n_samples, n_samples)
  
  # Convert labels to {-1, 1} if needed
  if (all(y %in% c(0, 1))) {
    y <- 2 * y - 1
  }
  
  estimators <- list()
  estimator_weights <- numeric(n_estimators)
  estimator_errors <- numeric(n_estimators)
  
  for (t in 1:n_estimators) {
    # Train weak learner (decision stump)
    formula <- as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
    estimator <- rpart(formula, data = data.frame(X, y), 
                      weights = sample_weights, 
                      control = rpart.control(maxdepth = max_depth))
    
    # Make predictions
    predictions <- predict(estimator, data.frame(X), type = "class")
    predictions <- as.numeric(as.character(predictions))
    
    # Calculate weighted error
    incorrect <- predictions != y
    error <- weighted.mean(incorrect, sample_weights)
    
    # Handle edge cases
    if (error <= 0) error <- 1e-10
    if (error >= 0.5) error <- 0.5 - 1e-10
    
    # Calculate estimator weight
    alpha <- 0.5 * log((1 - error) / error)
    
    # Update sample weights
    sample_weights <- sample_weights * exp(alpha * incorrect * (2 * (predictions != y) - 1))
    sample_weights <- sample_weights / sum(sample_weights)
    
    # Store results
    estimators[[t]] <- estimator
    estimator_weights[t] <- alpha
    estimator_errors[t] <- error
  }
  
  return(list(estimators = estimators, 
              estimator_weights = estimator_weights,
              estimator_errors = estimator_errors))
}

# Prediction function for AdaBoost
predict_ada_boost <- function(model, X) {
  predictions <- rep(0, nrow(X))
  
  for (i in seq_along(model$estimators)) {
    pred <- predict(model$estimators[[i]], data.frame(X), type = "class")
    pred <- as.numeric(as.character(pred))
    predictions <- predictions + model$estimator_weights[i] * pred
  }
  
  return(sign(predictions))
}

# Demonstrate basic AdaBoost
demonstrate_basic_adaboost <- function() {
  cat("=== Basic AdaBoost Demonstration ===\n\n")
  
  # Generate synthetic data
  set.seed(42)
  n_samples <- 1000
  X <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y <- ifelse(X$x1 + X$x2 > 0, 1, 0)
  
  # Train AdaBoost
  ada_model <- ada_boost(X, y, n_estimators = 50, max_depth = 1)
  
  # Make predictions
  y_pred <- predict_ada_boost(ada_model, X)
  
  # Calculate accuracy
  accuracy <- mean(y_pred == (2 * y - 1))
  cat("AdaBoost Performance:\n")
  cat(sprintf("Accuracy: %.4f\n", accuracy))
  
  return(list(ada_model = ada_model, X = X, y = y, accuracy = accuracy))
}

# Visualize training progress
visualize_training_progress <- function(ada_model) {
  cat("=== Training Progress Visualization ===\n\n")
  
  # Create results data frame
  results_df <- data.frame(
    iteration = 1:length(ada_model$estimator_errors),
    error = ada_model$estimator_errors,
    weight = ada_model$estimator_weights
  )
  
  # Plot error rates
  p1 <- ggplot(results_df, aes(x = iteration, y = error)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = 0.5, color = "red", linestyle = "dashed") +
    labs(title = "Weak Learner Error Rates",
         x = "Iteration", y = "Error Rate") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

  # Plot estimator weights
  p2 <- ggplot(results_df, aes(x = iteration, y = weight)) +
    geom_line(color = "green") +
    labs(title = "Estimator Weights",
         x = "Iteration", y = "Weight (α)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(results_df)
}

# Demonstrate decision boundaries
demonstrate_decision_boundaries <- function(X, y) {
  cat("=== Decision Boundary Comparison ===\n\n")
  
  # Convert labels for consistency
  y_boost <- 2 * y - 1
  
  # Single decision tree
  single_tree <- rpart(y_boost ~ x1 + x2, data = data.frame(X, y_boost),
                      control = rpart.control(maxdepth = 3))
  
  # AdaBoost with few iterations
  ada_few <- ada_boost(X, y, n_estimators = 5, max_depth = 1)
  
  # AdaBoost with many iterations
  ada_many <- ada_boost(X, y, n_estimators = 50, max_depth = 1)
  
  # Create visualization data
  x1_range <- seq(min(X$x1) - 1, max(X$x1) + 1, length.out = 100)
  x2_range <- seq(min(X$x2) - 1, max(X$x2) + 1, length.out = 100)
  grid_data <- expand.grid(x1 = x1_range, x2 = x2_range)
  
  # Predictions for single tree
  single_pred <- predict(single_tree, grid_data, type = "class")
  single_pred <- as.numeric(as.character(single_pred))
  
  # Predictions for AdaBoost few
  ada_few_pred <- predict_ada_boost(ada_few, grid_data)
  
  # Predictions for AdaBoost many
  ada_many_pred <- predict_ada_boost(ada_many, grid_data)
  
  # Create plots
  p1 <- ggplot() +
    geom_contour(data = data.frame(grid_data, pred = single_pred), 
                 aes(x = x1, y = x2, z = pred), breaks = 0, color = "black") +
    geom_point(data = data.frame(X, y = y_boost), aes(x = x1, y = x2, color = factor(y))) +
    labs(title = "Single Decision Tree", color = "Class") +
    theme_minimal()
  
  p2 <- ggplot() +
    geom_contour(data = data.frame(grid_data, pred = ada_few_pred), 
                 aes(x = x1, y = x2, z = pred), breaks = 0, color = "black") +
    geom_point(data = data.frame(X, y = y_boost), aes(x = x1, y = x2, color = factor(y))) +
    labs(title = "AdaBoost (5 iterations)", color = "Class") +
    theme_minimal()
  
  p3 <- ggplot() +
    geom_contour(data = data.frame(grid_data, pred = ada_many_pred), 
                 aes(x = x1, y = x2, z = pred), breaks = 0, color = "black") +
    geom_point(data = data.frame(X, y = y_boost), aes(x = x1, y = x2, color = factor(y))) +
    labs(title = "AdaBoost (50 iterations)", color = "Class") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(single_tree = single_tree, ada_few = ada_few, ada_many = ada_many))
}

# Analyze theoretical properties
analyze_theoretical_properties <- function() {
  cat("=== Theoretical Properties Analysis ===\n\n")
  
  # Generate data for analysis
  set.seed(42)
  n_samples <- 500
  X <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y <- ifelse(X$x1 + X$x2 > 0, 1, 0)
  y_boost <- 2 * y - 1
  
  # Train AdaBoost with different numbers of iterations
  iterations <- c(1, 5, 10, 20, 50, 100)
  training_errors <- numeric(length(iterations))
  test_errors <- numeric(length(iterations))
  
  # Split data
  train_idx <- sample(1:n_samples, 0.7 * n_samples)
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y_boost[train_idx]
  y_test <- y_boost[-train_idx]
  
  for (i in seq_along(iterations)) {
    ada <- ada_boost(X_train, y_train, n_estimators = iterations[i], max_depth = 1)
    
    # Calculate training error
    train_pred <- predict_ada_boost(ada, X_train)
    training_errors[i] <- 1 - mean(train_pred == y_train)
    
    # Calculate test error
    test_pred <- predict_ada_boost(ada, X_test)
    test_errors[i] <- 1 - mean(test_pred == y_test)
  }
  
  # Create error analysis plot
  error_df <- data.frame(
    iterations = iterations,
    training_error = training_errors,
    test_error = test_errors
  )
  
  p1 <- ggplot(error_df, aes(x = iterations)) +
    geom_line(aes(y = training_error, color = "Training Error")) +
    geom_line(aes(y = test_error, color = "Test Error")) +
    geom_point(aes(y = training_error, color = "Training Error")) +
    geom_point(aes(y = test_error, color = "Test Error")) +
    scale_y_log10() +
    labs(title = "Error Rate vs Number of Iterations",
         x = "Number of Iterations", y = "Error Rate (log scale)",
         color = "Error Type") +
    theme_minimal()
  
  # Analyze Z_t values
  ada_full <- ada_boost(X_train, y_train, n_estimators = 50, max_depth = 1)
  Z_t_values <- 2 * sqrt(ada_full$estimator_errors * (1 - ada_full$estimator_errors))
  
  Z_t_df <- data.frame(
    iteration = 1:length(Z_t_values),
    Z_t = Z_t_values
  )
  
  p2 <- ggplot(Z_t_df, aes(x = iteration, y = Z_t)) +
    geom_line(color = "green") +
    geom_point(color = "green") +
    geom_hline(yintercept = 1, color = "red", linestyle = "dashed") +
    labs(title = "Normalization Factor Z_t",
         x = "Iteration", y = "Z_t") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  # Print theoretical analysis
  cat("Theoretical Analysis:\n")
  cat(sprintf("Final training error: %.4f\n", training_errors[length(training_errors)]))
  cat(sprintf("Product of Z_t values: %.4f\n", prod(Z_t_values)))
  cat(sprintf("Error bound satisfied: %s\n", 
              training_errors[length(training_errors)] <= prod(Z_t_values)))
  
  return(list(training_errors = training_errors, 
              test_errors = test_errors, 
              Z_t_values = Z_t_values))
}

# Demonstrate practical considerations
demonstrate_practical_considerations <- function() {
  cat("=== Practical Considerations ===\n\n")
  
  # Generate data
  set.seed(42)
  n_samples <- 1000
  X <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples),
    x3 = rnorm(n_samples),
    x4 = rnorm(n_samples),
    x5 = rnorm(n_samples)
  )
  y <- ifelse(X$x1 + X$x2 + 0.5 * rnorm(n_samples) > 0, 1, 0)
  y_boost <- 2 * y - 1
  
  # Split data
  train_idx <- sample(1:n_samples, 0.7 * n_samples)
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y_boost[train_idx]
  y_test <- y_boost[-train_idx]
  
  # Test different weak learner depths
  depths <- c(1, 2, 3, 5)
  depth_results <- data.frame(
    depth = depths,
    train_acc = numeric(length(depths)),
    test_acc = numeric(length(depths)),
    overfitting = numeric(length(depths))
  )
  
  for (i in seq_along(depths)) {
    ada <- ada_boost(X_train, y_train, n_estimators = 50, max_depth = depths[i])
    
    train_pred <- predict_ada_boost(ada, X_train)
    test_pred <- predict_ada_boost(ada, X_test)
    
    depth_results$train_acc[i] <- mean(train_pred == y_train)
    depth_results$test_acc[i] <- mean(test_pred == y_test)
    depth_results$overfitting[i] <- depth_results$train_acc[i] - depth_results$test_acc[i]
  }
  
  # Create plots
  p1 <- ggplot(depth_results, aes(x = depth)) +
    geom_line(aes(y = train_acc, color = "Training Accuracy")) +
    geom_line(aes(y = test_acc, color = "Test Accuracy")) +
    geom_point(aes(y = train_acc, color = "Training Accuracy")) +
    geom_point(aes(y = test_acc, color = "Test Accuracy")) +
    labs(title = "Accuracy vs Weak Learner Depth",
         x = "Tree Depth", y = "Accuracy",
         color = "Accuracy Type") +
    theme_minimal()
  
  p2 <- ggplot(depth_results, aes(x = depth, y = overfitting)) +
    geom_line(color = "green") +
    geom_point(color = "green") +
    labs(title = "Overfitting vs Weak Learner Depth",
         x = "Tree Depth", y = "Overfitting Gap") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  # Print recommendations
  cat("Practical Recommendations:\n")
  cat("1. Use shallow trees (depth=1) for weak learners\n")
  cat("2. Monitor overfitting with deeper trees\n")
  cat("3. Use cross-validation to find optimal number of iterations\n")
  cat("4. Consider early stopping for large datasets\n")
  
  return(depth_results)
}

# Demonstrate real-world applications
demonstrate_real_world_applications <- function() {
  cat("=== Real-World Applications ===\n\n")
  
  # Example: Simulated medical diagnosis
  set.seed(42)
  n_patients <- 500
  
  # Simulate medical features
  age <- rnorm(n_patients, mean = 60, sd = 15)
  bmi <- rnorm(n_patients, mean = 25, sd = 5)
  blood_pressure <- rnorm(n_patients, mean = 120, sd = 20)
  cholesterol <- rnorm(n_patients, mean = 200, sd = 40)
  
  # Simulate disease probability based on features
  disease_prob <- 1 / (1 + exp(-(-2 + 0.02 * age + 0.1 * bmi + 0.01 * blood_pressure + 0.005 * cholesterol)))
  disease <- rbinom(n_patients, 1, disease_prob)
  
  # Create medical dataset
  medical_data <- data.frame(
    age = age,
    bmi = bmi,
    blood_pressure = blood_pressure,
    cholesterol = cholesterol,
    disease = disease
  )
  
  # Split data
  train_idx <- sample(1:n_patients, 0.7 * n_patients)
  train_data <- medical_data[train_idx, ]
  test_data <- medical_data[-train_idx, ]
  
  # Train AdaBoost
  ada_medical <- ada_boost(train_data[, -5], train_data$disease, n_estimators = 50, max_depth = 1)
  
  # Make predictions
  test_pred <- predict_ada_boost(ada_medical, test_data[, -5])
  test_pred_binary <- (test_pred + 1) / 2  # Convert back to 0/1
  
  # Calculate metrics
  accuracy <- mean(test_pred_binary == test_data$disease)
  sensitivity <- mean(test_pred_binary[test_data$disease == 1] == 1)
  specificity <- mean(test_pred_binary[test_data$disease == 0] == 0)
  
  cat("Medical Diagnosis Results:\n")
  cat(sprintf("Accuracy: %.4f\n", accuracy))
  cat(sprintf("Sensitivity: %.4f\n", sensitivity))
  cat(sprintf("Specificity: %.4f\n", specificity))
  
  # Feature importance analysis
  feature_importance <- numeric(4)
  total_weight <- sum(ada_medical$estimator_weights)
  
  for (i in seq_along(ada_medical$estimators)) {
    if (!is.null(ada_medical$estimators[[i]]$variable.importance)) {
      feature_importance <- feature_importance + 
        (ada_medical$estimator_weights[i] / total_weight) * 
        ada_medical$estimators[[i]]$variable.importance
    }
  }
  
  feature_names <- c("Age", "BMI", "Blood Pressure", "Cholesterol")
  importance_df <- data.frame(
    feature = feature_names,
    importance = feature_importance
  )
  
  # Plot feature importance
  p <- ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance in Medical Diagnosis",
         x = "Feature", y = "Importance") +
    theme_minimal()
  
  print(p)
  
  cat("\nTop medical features by importance:\n")
  for (i in order(feature_importance, decreasing = TRUE)) {
    cat(sprintf("%d. %s: %.4f\n", which(order(feature_importance, decreasing = TRUE) == i), 
                feature_names[i], feature_importance[i]))
  }
  
  return(list(ada_medical = ada_medical, 
              accuracy = accuracy, 
              sensitivity = sensitivity, 
              specificity = specificity,
              feature_importance = importance_df))
}

# Main demonstration function
main_r <- function() {
  cat("AdaBoost: Implementation and Analysis\n")
  cat("=" * 60, "\n")
  
  # 1. Basic AdaBoost demonstration
  cat("\n1. Basic AdaBoost Demonstration:\n")
  basic_results <- demonstrate_basic_adaboost()
  
  # 2. Training progress visualization
  cat("\n2. Training Progress Visualization:\n")
  progress_results <- visualize_training_progress(basic_results$ada_model)
  
  # 3. Decision boundary comparison
  cat("\n3. Decision Boundary Comparison:\n")
  boundary_results <- demonstrate_decision_boundaries(basic_results$X, basic_results$y)
  
  # 4. Theoretical analysis
  cat("\n4. Theoretical Properties Analysis:\n")
  theoretical_results <- analyze_theoretical_properties()
  
  # 5. Practical considerations
  cat("\n5. Practical Considerations:\n")
  practical_results <- demonstrate_practical_considerations()
  
  # 6. Real-world applications
  cat("\n6. Real-World Applications:\n")
  application_results <- demonstrate_real_world_applications()
  
  cat("\n=== Key Insights ===\n")
  cat("1. AdaBoost sequentially combines weak learners\n")
  cat("2. Weight updates focus on difficult examples\n")
  cat("3. Exponential loss provides natural combination\n")
  cat("4. Theoretical bounds guarantee improvement\n")
  cat("5. Shallow trees work best as weak learners\n")
  cat("6. Monitor overfitting with validation data\n")
  cat("7. Feature importance available through weighted average\n")
  cat("8. Effective for both binary and multi-class problems\n")
  
  return(list(
    basic_results = basic_results,
    progress_results = progress_results,
    boundary_results = boundary_results,
    theoretical_results = theoretical_results,
    practical_results = practical_results,
    application_results = application_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
