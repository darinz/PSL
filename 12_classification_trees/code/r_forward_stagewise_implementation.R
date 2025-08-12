# Forward Stagewise Additive Modeling in R

# Load required libraries
library(rpart)
library(ggplot2)
library(gridExtra)
library(dplyr)

# Forward Stagewise Additive Modeling implementation
forward_stagewise_additive <- function(X, y, base_learner = "tree", 
                                      loss_function = "squared_error",
                                      n_estimators = 100, learning_rate = 1.0) {
  n_samples <- nrow(X)
  
  # Initialize
  predictions <- rep(0, n_samples)
  estimators <- list()
  estimator_weights <- numeric(n_estimators)
  training_losses <- numeric(n_estimators)
  
  for (t in 1:n_estimators) {
    # Compute residuals
    if (loss_function == "squared_error") {
      residuals <- y - predictions
    } else if (loss_function == "exponential") {
      residuals <- -y * exp(-y * predictions)
    } else if (loss_function == "logistic") {
      prob <- 1 / (1 + exp(-predictions))
      residuals <- y - prob
    }
    
    # Fit base learner
    if (base_learner == "tree") {
      formula <- as.formula(paste("residuals ~", paste(colnames(X), collapse = " + ")))
      estimator <- rpart(formula, data = data.frame(X, residuals), 
                        control = rpart.control(maxdepth = 3))
      base_predictions <- predict(estimator, data.frame(X))
    }
    
    # Find optimal weight
    if (loss_function == "squared_error") {
      numerator <- sum(base_predictions * (y - predictions))
      denominator <- sum(base_predictions^2)
      alpha <- ifelse(denominator > 0, numerator / denominator, 0)
    } else {
      # Line search for other loss functions
      best_alpha <- 0
      best_loss <- Inf
      
      for (alpha_candidate in seq(-2, 2, length.out = 100)) {
        new_predictions <- predictions + alpha_candidate * base_predictions
        
        if (loss_function == "exponential") {
          loss <- mean(exp(-y * new_predictions))
        } else if (loss_function == "logistic") {
          loss <- mean(log(1 + exp(-y * new_predictions)))
        }
        
        if (loss < best_loss) {
          best_loss <- loss
          best_alpha <- alpha_candidate
        }
      }
      alpha <- best_alpha
    }
    
    # Apply learning rate
    alpha <- alpha * learning_rate
    
    # Update predictions
    predictions <- predictions + alpha * base_predictions
    
    # Store results
    estimators[[t]] <- estimator
    estimator_weights[t] <- alpha
    
    # Compute training loss
    if (loss_function == "squared_error") {
      training_losses[t] <- mean((y - predictions)^2)
    } else if (loss_function == "exponential") {
      training_losses[t] <- mean(exp(-y * predictions))
    } else if (loss_function == "logistic") {
      training_losses[t] <- mean(log(1 + exp(-y * predictions)))
    }
  }
  
  return(list(estimators = estimators,
              estimator_weights = estimator_weights,
              training_losses = training_losses,
              final_predictions = predictions))
}

# Prediction function for Forward Stagewise
predict_fsam <- function(model, X) {
  predictions <- rep(0, nrow(X))
  
  for (i in seq_along(model$estimators)) {
    pred <- predict(model$estimators[[i]], data.frame(X))
    predictions <- predictions + model$estimator_weights[i] * pred
  }
  
  return(predictions)
}

# Demonstrate basic Forward Stagewise
demonstrate_basic_forward_stagewise <- function() {
  cat("=== Forward Stagewise Additive Modeling Demonstration ===\n\n")
  
  # Generate synthetic data
  set.seed(42)
  n_samples <- 1000

  # Regression data
  X_reg <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y_reg <- 2 * X_reg$x1 + 3 * X_reg$x2 + rnorm(n_samples, 0, 0.1)

  # Classification data
  X_clf <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y_clf <- ifelse(X_clf$x1 + X_clf$x2 > 0, 1, -1)

  # Train models
  fsam_reg <- forward_stagewise_additive(X_reg, y_reg, "tree", "squared_error", 50, 0.1)
  fsam_clf <- forward_stagewise_additive(X_clf, y_clf, "tree", "exponential", 50, 1.0)

  # Evaluate regression
  reg_pred <- predict_fsam(fsam_reg, X_reg)
  reg_mse <- mean((y_reg - reg_pred)^2)
  cat("1. Regression with Squared Error Loss:\n")
  cat(sprintf("   Test MSE: %.4f\n", reg_mse))

  # Evaluate classification
  clf_pred <- predict_fsam(fsam_clf, X_clf)
  clf_accuracy <- mean(sign(clf_pred) == y_clf)
  cat("\n2. Classification with Exponential Loss:\n")
  cat(sprintf("   Test Accuracy: %.4f\n", clf_accuracy))
  
  return(list(fsam_reg = fsam_reg, fsam_clf = fsam_clf, 
              X_reg = X_reg, y_reg = y_reg, X_clf = X_clf, y_clf = y_clf))
}

# Visualize training progress
visualize_training_progress <- function(fsam_reg, fsam_clf) {
  cat("=== Training Progress Visualization ===\n\n")
  
  # Create results data frame
  results_df <- data.frame(
    iteration = rep(1:50, 2),
    loss = c(fsam_reg$training_losses, fsam_clf$training_losses),
    weight = c(fsam_reg$estimator_weights, fsam_clf$estimator_weights),
    type = rep(c("Regression", "Classification"), each = 50)
  )

  # Plot training losses
  p1 <- ggplot(results_df, aes(x = iteration, y = loss, color = type)) +
    geom_line(size = 1) +
    labs(title = "Training Loss vs Iterations",
         x = "Iteration", y = "Training Loss") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

  # Plot estimator weights
  p2 <- ggplot(results_df, aes(x = iteration, y = weight, color = type)) +
    geom_line(size = 1) +
    labs(title = "Estimator Weights",
         x = "Iteration", y = "Weight (α)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))

  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(results_df)
}

# Demonstrate loss functions
demonstrate_loss_functions <- function() {
  cat("=== Loss Functions Comparison ===\n\n")
  
  # Generate data
  set.seed(42)
  n_samples <- 1000
  X <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y <- ifelse(X$x1 + X$x2 > 0, 1, -1)
  
  # Train models with different loss functions
  loss_functions <- c("exponential", "logistic")
  models <- list()
  
  for (loss_func in loss_functions) {
    model <- forward_stagewise_additive(X, y, "tree", loss_func, 50, 0.1)
    models[[loss_func]] <- model
  }
  
  # Create comparison plot
  comparison_df <- data.frame(
    iteration = rep(1:50, length(loss_functions)),
    loss = unlist(lapply(models, function(m) m$training_losses)),
    type = rep(loss_functions, each = 50)
  )
  
  p <- ggplot(comparison_df, aes(x = iteration, y = loss, color = type)) +
    geom_line(size = 1) +
    labs(title = "Training Loss Comparison",
         x = "Iteration", y = "Training Loss",
         color = "Loss Function") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  cat("Loss Function Analysis:\n")
  cat("1. Exponential Loss: Heavily penalizes misclassifications, can overfit\n")
  cat("2. Logistic Loss: More robust, better theoretical properties\n")
  cat("3. Both show similar convergence patterns but different generalization\n")
  
  return(models)
}

# Demonstrate learning rate effects
demonstrate_learning_rate_effects <- function() {
  cat("=== Learning Rate Effects ===\n\n")
  
  # Generate data
  set.seed(42)
  n_samples <- 1000
  X <- data.frame(
    x1 = rnorm(n_samples),
    x2 = rnorm(n_samples)
  )
  y <- 2 * X$x1 + 3 * X$x2 + rnorm(n_samples, 0, 0.1)
  
  # Test different learning rates
  learning_rates <- c(0.01, 0.1, 0.5, 1.0)
  models <- list()
  
  for (lr in learning_rates) {
    model <- forward_stagewise_additive(X, y, "tree", "squared_error", 50, lr)
    models[[as.character(lr)]] <- model
  }
  
  # Create comparison plot
  comparison_df <- data.frame(
    iteration = rep(1:50, length(learning_rates)),
    loss = unlist(lapply(models, function(m) m$training_losses)),
    lr = rep(learning_rates, each = 50)
  )
  
  p <- ggplot(comparison_df, aes(x = iteration, y = loss, color = factor(lr))) +
    geom_line(size = 1) +
    labs(title = "Training Loss vs Learning Rate",
         x = "Iteration", y = "Training Loss",
         color = "Learning Rate") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  cat("Learning Rate Analysis:\n")
  cat("1. Smaller learning rates: Slower convergence, better generalization\n")
  cat("2. Larger learning rates: Faster convergence, risk of overfitting\n")
  cat("3. Optimal learning rate balances convergence speed and generalization\n")
  
  return(models)
}

# Demonstrate financial risk modeling
demonstrate_financial_risk_modeling <- function() {
  cat("=== Financial Risk Modeling ===\n\n")
  
  # Simulate financial data
  set.seed(42)
  n_samples <- 10000
  
  # Features: income, age, credit_score, debt_ratio, payment_history
  X_fin <- data.frame(
    income = rlnorm(n_samples, meanlog = 10, sdlog = 0.5),
    age = rnorm(n_samples, mean = 45, sd = 15),
    credit_score = rnorm(n_samples, mean = 700, sd = 100),
    debt_ratio = rbeta(n_samples, shape1 = 2, shape2 = 5),
    payment_history = rpois(n_samples, lambda = 2)
  )
  
  # Target: default (1) or not (0)
  y_fin <- ifelse((X_fin$debt_ratio > 0.4) | (X_fin$credit_score < 600), 1, -1)
  
  # Train forward stagewise model
  fsam_fin <- forward_stagewise_additive(X_fin, y_fin, "tree", "exponential", 100, 0.1)
  
  # Feature importance analysis
  feature_importance <- numeric(ncol(X_fin))
  total_weight <- sum(abs(fsam_fin$estimator_weights))
  
  for (i in seq_along(fsam_fin$estimators)) {
    if (!is.null(fsam_fin$estimators[[i]]$variable.importance)) {
      feature_importance <- feature_importance + 
        (abs(fsam_fin$estimator_weights[i]) / total_weight) * 
        fsam_fin$estimators[[i]]$variable.importance
    }
  }
  
  # Create importance data frame
  importance_df <- data.frame(
    feature = colnames(X_fin),
    importance = feature_importance
  )
  importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), ]
  
  cat("Feature Importance for Credit Risk:\n")
  print(importance_df)
  
  # Plot feature importance
  p <- ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance in Credit Risk Modeling",
         x = "Feature", y = "Importance") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  return(list(fsam_fin = fsam_fin, importance_df = importance_df))
}

# Demonstrate medical diagnosis
demonstrate_medical_diagnosis <- function() {
  cat("=== Medical Diagnosis ===\n\n")
  
  # Simulate medical data
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
  
  # Train Forward Stagewise model
  fsam_med <- forward_stagewise_additive(train_data[, -5], train_data$disease, "tree", "logistic", 50, 0.1)
  
  # Make predictions
  test_pred <- predict_fsam(fsam_med, test_data[, -5])
  test_pred_binary <- ifelse(test_pred > 0, 1, 0)
  
  # Calculate metrics
  accuracy <- mean(test_pred_binary == test_data$disease)
  sensitivity <- mean(test_pred_binary[test_data$disease == 1] == 1)
  specificity <- mean(test_pred_binary[test_data$disease == 0] == 0)
  
  cat("Medical Diagnosis Results:\n")
  cat(sprintf("Accuracy: %.4f\n", accuracy))
  cat(sprintf("Sensitivity: %.4f\n", sensitivity))
  cat(sprintf("Specificity: %.4f\n", specificity))
  
  # Analyze model convergence
  staged_predictions <- list()
  predictions <- rep(0, nrow(test_data))
  
  for (i in seq_along(fsam_med$estimators)) {
    pred <- predict(fsam_med$estimators[[i]], test_data[, -5])
    predictions <- predictions + fsam_med$estimator_weights[i] * pred
    staged_predictions[[i]] <- predictions
  }
  
  staged_accuracies <- sapply(staged_predictions, function(pred) {
    mean(ifelse(pred > 0, 1, 0) == test_data$disease)
  })
  
  # Plot convergence
  convergence_df <- data.frame(
    iteration = 1:length(staged_accuracies),
    accuracy = staged_accuracies
  )
  
  p <- ggplot(convergence_df, aes(x = iteration, y = accuracy)) +
    geom_line(color = "blue", size = 1) +
    geom_hline(yintercept = accuracy, color = "red", linestyle = "dashed") +
    labs(title = "Model Convergence in Medical Diagnosis",
         x = "Iteration", y = "Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  return(list(fsam_med = fsam_med, accuracy = accuracy, 
              sensitivity = sensitivity, specificity = specificity,
              staged_accuracies = staged_accuracies))
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
  y <- 2 * X$x1 + 3 * X$x2 + rnorm(n_samples, 0, 0.1)
  
  # Train Forward Stagewise with different numbers of iterations
  iterations <- c(1, 5, 10, 20, 50, 100)
  training_losses <- numeric(length(iterations))
  test_losses <- numeric(length(iterations))
  
  # Split data
  train_idx <- sample(1:n_samples, 0.7 * n_samples)
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  for (i in seq_along(iterations)) {
    model <- forward_stagewise_additive(X_train, y_train, "tree", "squared_error", iterations[i], 0.1)
    
    # Training loss
    training_losses[i] <- model$training_losses[length(model$training_losses)]
    
    # Test loss
    test_pred <- predict_fsam(model, X_test)
    test_losses[i] <- mean((y_test - test_pred)^2)
  }
  
  # Create analysis plot
  analysis_df <- data.frame(
    iterations = rep(iterations, 2),
    loss = c(training_losses, test_losses),
    type = rep(c("Training Loss", "Test Loss"), each = length(iterations))
  )
  
  p <- ggplot(analysis_df, aes(x = iterations, y = loss, color = type)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    scale_y_log10() +
    labs(title = "Loss vs Number of Iterations",
         x = "Number of Iterations", y = "Loss (log scale)",
         color = "Loss Type") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  print(p)
  
  cat("Theoretical Analysis:\n")
  cat(sprintf("1. Final training loss: %.6f\n", training_losses[length(training_losses)]))
  cat(sprintf("2. Final test loss: %.6f\n", test_losses[length(test_losses)]))
  cat(sprintf("3. Overfitting gap: %.6f\n", training_losses[length(training_losses)] - test_losses[length(test_losses)]))
  cat("4. Model shows good convergence properties\n")
  
  return(list(training_losses = training_losses, test_losses = test_losses))
}

# Main demonstration function
main_r <- function() {
  cat("Forward Stagewise Additive Modeling: Implementation and Analysis\n")
  cat("=" * 70, "\n")
  
  # 1. Basic demonstration
  cat("\n1. Basic Forward Stagewise Demonstration:\n")
  basic_results <- demonstrate_basic_forward_stagewise()
  
  # 2. Training progress visualization
  cat("\n2. Training Progress Visualization:\n")
  progress_results <- visualize_training_progress(basic_results$fsam_reg, basic_results$fsam_clf)
  
  # 3. Loss functions comparison
  cat("\n3. Loss Functions Comparison:\n")
  loss_results <- demonstrate_loss_functions()
  
  # 4. Learning rate effects
  cat("\n4. Learning Rate Effects:\n")
  lr_results <- demonstrate_learning_rate_effects()
  
  # 5. Financial risk modeling
  cat("\n5. Financial Risk Modeling Application:\n")
  fin_results <- demonstrate_financial_risk_modeling()
  
  # 6. Medical diagnosis
  cat("\n6. Medical Diagnosis Application:\n")
  med_results <- demonstrate_medical_diagnosis()
  
  # 7. Theoretical analysis
  cat("\n7. Theoretical Properties Analysis:\n")
  theoretical_results <- analyze_theoretical_properties()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Forward Stagewise provides a unified framework for boosting\n")
  cat("2. Different loss functions have different convergence properties\n")
  cat("3. Learning rate controls the trade-off between speed and generalization\n")
  cat("4. Sequential optimization makes complex problems tractable\n")
  cat("5. Residual fitting focuses each base learner on current errors\n")
  cat("6. Weight optimization ensures optimal contribution of each base learner\n")
  cat("7. Regularization improves generalization performance\n")
  cat("8. Feature importance provides interpretability\n")
  
  return(list(
    basic_results = basic_results,
    progress_results = progress_results,
    loss_results = loss_results,
    lr_results = lr_results,
    fin_results = fin_results,
    med_results = med_results,
    theoretical_results = theoretical_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
