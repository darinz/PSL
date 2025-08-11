# Variable Selection and Regularization Methods Comparison
# This script demonstrates comprehensive comparison of different methods

# Load required libraries
library(glmnet)
library(pls)
library(leaps)
library(ggplot2)
library(dplyr)
library(tidyr)

# Comprehensive comparison function
compare_variable_selection_methods <- function(X, y, true_beta = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Split data
  set.seed(42)
  train_idx <- sample(1:n, 0.7 * n)
  X_train <- X[train_idx, ]
  X_test <- X[-train_idx, ]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Standardize data
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                         scale = attr(X_train_scaled, "scaled:scale"))
  
  results <- list()
  
  # 1. Ordinary Least Squares
  start_time <- Sys.time()
  ols_model <- lm(y_train ~ X_train_scaled - 1)
  ols_time <- difftime(Sys.time(), start_time, units = "secs")
  
  ols_pred <- X_test_scaled %*% coef(ols_model)
  ols_mse <- mean((y_test - ols_pred)^2)
  ols_r2 <- 1 - sum((y_test - ols_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$OLS <- list(
    coefficients = coef(ols_model),
    test_mse = ols_mse,
    test_r2 = ols_r2,
    n_nonzero = sum(coef(ols_model) != 0),
    training_time = as.numeric(ols_time)
  )
  
  # 2. Ridge Regression
  start_time <- Sys.time()
  ridge_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 0, standardize = FALSE)
  ridge_model <- glmnet(X_train_scaled, y_train, alpha = 0, lambda = ridge_cv$lambda.min)
  ridge_time <- difftime(Sys.time(), start_time, units = "secs")
  
  ridge_pred <- predict(ridge_model, newx = X_test_scaled)
  ridge_mse <- mean((y_test - ridge_pred)^2)
  ridge_r2 <- 1 - sum((y_test - ridge_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$Ridge <- list(
    coefficients = as.vector(coef(ridge_model))[-1],  # Remove intercept
    test_mse = ridge_mse,
    test_r2 = ridge_r2,
    n_nonzero = sum(coef(ridge_model)[-1] != 0),
    training_time = as.numeric(ridge_time),
    best_alpha = ridge_cv$lambda.min
  )
  
  # 3. Lasso Regression
  start_time <- Sys.time()
  lasso_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 1, standardize = FALSE)
  lasso_model <- glmnet(X_train_scaled, y_train, alpha = 1, lambda = lasso_cv$lambda.min)
  lasso_time <- difftime(Sys.time(), start_time, units = "secs")
  
  lasso_pred <- predict(lasso_model, newx = X_test_scaled)
  lasso_mse <- mean((y_test - lasso_pred)^2)
  lasso_r2 <- 1 - sum((y_test - lasso_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$Lasso <- list(
    coefficients = as.vector(coef(lasso_model))[-1],  # Remove intercept
    test_mse = lasso_mse,
    test_r2 = lasso_r2,
    n_nonzero = sum(coef(lasso_model)[-1] != 0),
    training_time = as.numeric(lasso_time),
    best_alpha = lasso_cv$lambda.min
  )
  
  # 4. Elastic Net
  start_time <- Sys.time()
  elastic_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 0.5, standardize = FALSE)
  elastic_model <- glmnet(X_train_scaled, y_train, alpha = 0.5, lambda = elastic_cv$lambda.min)
  elastic_time <- difftime(Sys.time(), start_time, units = "secs")
  
  elastic_pred <- predict(elastic_model, newx = X_test_scaled)
  elastic_mse <- mean((y_test - elastic_pred)^2)
  elastic_r2 <- 1 - sum((y_test - elastic_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$ElasticNet <- list(
    coefficients = as.vector(coef(elastic_model))[-1],  # Remove intercept
    test_mse = elastic_mse,
    test_r2 = elastic_r2,
    n_nonzero = sum(coef(elastic_model)[-1] != 0),
    training_time = as.numeric(elastic_time),
    best_alpha = elastic_cv$lambda.min
  )
  
  # 5. Principal Components Regression
  start_time <- Sys.time()
  n_components <- min(20, p)  # Limit components
  pcr_model <- pcr(y_train ~ X_train_scaled, ncomp = n_components, validation = "CV")
  pcr_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Find optimal number of components
  opt_comp <- which.min(pcr_model$validation$PRESS)
  pcr_pred <- predict(pcr_model, newdata = data.frame(X_train_scaled = X_test_scaled), ncomp = opt_comp)
  pcr_mse <- mean((y_test - pcr_pred)^2)
  pcr_r2 <- 1 - sum((y_test - pcr_pred)^2) / sum((y_test - mean(y_test))^2)
  
  # Transform coefficients back to original space
  pcr_coef <- coef(pcr_model, ncomp = opt_comp)
  
  results$PCR <- list(
    coefficients = as.vector(pcr_coef),
    test_mse = pcr_mse,
    test_r2 = pcr_r2,
    n_nonzero = sum(pcr_coef != 0),
    training_time = as.numeric(pcr_time),
    n_components = opt_comp
  )
  
  # 6. Subset Selection (Forward Stepwise)
  start_time <- Sys.time()
  max_vars <- min(20, p)  # Limit for computational efficiency
  subset_model <- regsubsets(y_train ~ X_train_scaled, data = data.frame(X_train_scaled, y_train), 
                            nvmax = max_vars, method = "forward")
  subset_time <- difftime(Sys.time(), start_time, units = "secs")
  
  # Find optimal subset size using BIC
  opt_size <- which.min(summary(subset_model)$bic)
  subset_coef <- coef(subset_model, opt_size)
  
  # Create full coefficient vector
  full_coef <- rep(0, p)
  var_names <- names(subset_coef)[-1]  # Remove intercept
  var_indices <- as.numeric(substr(var_names, 15, nchar(var_names)))  # Extract indices
  full_coef[var_indices] <- subset_coef[-1]
  
  subset_pred <- X_test_scaled %*% full_coef
  subset_mse <- mean((y_test - subset_pred)^2)
  subset_r2 <- 1 - sum((y_test - subset_pred)^2) / sum((y_test - mean(y_test))^2)
  
  results$SubsetSelection <- list(
    coefficients = full_coef,
    test_mse = subset_mse,
    test_r2 = subset_r2,
    n_nonzero = opt_size,
    training_time = as.numeric(subset_time),
    selected_size = opt_size
  )
  
  # Add variable selection metrics if true coefficients are known
  if (!is.null(true_beta)) {
    for (method in names(results)) {
      if (method != "OLS") {
        true_nonzero <- true_beta != 0
        pred_nonzero <- results[[method]]$coefficients != 0
        
        tp <- sum(true_nonzero & pred_nonzero)
        fp <- sum(!true_nonzero & pred_nonzero)
        fn <- sum(true_nonzero & !pred_nonzero)
        
        precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
        f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        
        results[[method]]$precision <- precision
        results[[method]]$recall <- recall
        results[[method]]$f1_score <- f1
      }
    }
  }
  
  return(results)
}

# Generate design matrices
generate_design_matrices <- function(n_samples = 200, seed = 42) {
  set.seed(seed)
  
  # Base features
  n_base <- 5
  X_base <- matrix(rnorm(n_samples * n_base), n_samples, n_base)
  
  # Scenario 1: Curated features
  X1 <- X_base
  
  # Scenario 2: Extended features with interactions
  X2_extended <- model.matrix(~ .^2, data = data.frame(X_base))[, -1]  # Remove intercept
  X2 <- X2_extended
  
  # Scenario 3: High-dimensional with noise
  n_noise <- 500
  noise_features <- matrix(0, n_samples, n_noise)
  
  for (i in 1:n_noise) {
    true_feature_idx <- sample(1:ncol(X2), 1)
    noise_features[, i] <- sample(X2[, true_feature_idx])
  }
  
  X3 <- cbind(X2, noise_features)
  
  return(list(X1 = X1, X2 = X2, X3 = X3))
}

# Generate response variables
generate_response <- function(X, sparsity_level = 0.3, seed = 42) {
  set.seed(seed)
  
  n_features <- ncol(X)
  n_active <- max(1, round(n_features * sparsity_level))
  
  # True coefficients (sparse)
  true_beta <- rep(0, n_features)
  active_indices <- sample(1:n_features, n_active)
  true_beta[active_indices] <- rnorm(n_active) * 2
  
  # Generate response
  y <- X %*% true_beta + 0.5 * rnorm(nrow(X))
  
  return(list(y = y, true_beta = true_beta))
}

# Run comprehensive study
run_comprehensive_study <- function() {
  cat("Generating design matrices...\n")
  design_matrices <- generate_design_matrices(n_samples = 200)
  
  cat("Generating response variables...\n")
  response1 <- generate_response(design_matrices$X1, sparsity_level = 0.8)
  response2 <- generate_response(design_matrices$X2, sparsity_level = 0.3)
  response3 <- generate_response(design_matrices$X3, sparsity_level = 0.05)
  
  scenarios <- list(
    "X1 (Curated Features)" = list(X = design_matrices$X1, y = response1$y, beta = response1$true_beta),
    "X2 (Extended Features)" = list(X = design_matrices$X2, y = response2$y, beta = response2$true_beta),
    "X3 (High-Dimensional + Noise)" = list(X = design_matrices$X3, y = response3$y, beta = response3$true_beta)
  )
  
  all_results <- list()
  
  for (scenario_name in names(scenarios)) {
    cat(sprintf("\nAnalyzing %s...\n", scenario_name))
    scenario <- scenarios[[scenario_name]]
    
    cat(sprintf("Data shape: %d x %d\n", nrow(scenario$X), ncol(scenario$X)))
    cat(sprintf("True non-zero coefficients: %d\n", sum(scenario$beta != 0)))
    
    results <- compare_variable_selection_methods(scenario$X, scenario$y, scenario$beta)
    all_results[[scenario_name]] <- results
    
    # Print summary
    cat(sprintf("\nResults for %s:\n", scenario_name))
    cat(paste(rep("-", 80), collapse = ""), "\n")
    cat(sprintf("%-15s %-12s %-10s %-10s %-10s\n", "Method", "Test MSE", "Test R²", "Non-zero", "Time (s)"))
    cat(paste(rep("-", 80), collapse = ""), "\n")
    
    for (method in names(results)) {
      result <- results[[method]]
      cat(sprintf("%-15s %-12.4f %-10.4f %-10d %-10.4f\n", 
                  method, result$test_mse, result$test_r2, result$n_nonzero, result$training_time))
    }
  }
  
  return(all_results)
}

# Create visualizations
create_visualizations <- function(all_results) {
  scenarios <- names(all_results)
  methods <- names(all_results[[1]])
  
  # Prepare data for plotting
  plot_data <- data.frame()
  
  for (scenario in scenarios) {
    for (method in methods) {
      result <- all_results[[scenario]][[method]]
      
      plot_data <- rbind(plot_data, data.frame(
        Scenario = scenario,
        Method = method,
        Test_MSE = result$test_mse,
        Test_R2 = result$test_r2,
        Non_zero = result$n_nonzero,
        Training_Time = result$training_time
      ))
    }
  }
  
  # Create plots
  p1 <- ggplot(plot_data, aes(x = Method, y = Test_MSE, fill = Scenario)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(title = "Test MSE Comparison", x = "Method", y = "Mean Squared Error") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  p2 <- ggplot(plot_data, aes(x = Method, y = Test_R2, fill = Scenario)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(title = "Test R² Comparison", x = "Method", y = "R² Score") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  p3 <- ggplot(plot_data, aes(x = Method, y = Non_zero, fill = Scenario)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(title = "Number of Non-zero Coefficients", x = "Method", y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  p4 <- ggplot(plot_data, aes(x = Method, y = Training_Time, fill = Scenario)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(title = "Training Time Comparison", x = "Method", y = "Time (seconds)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Display plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
  
  return(list(p1, p2, p3, p4))
}

# Method selection decision tree
select_method <- function(X, y, problem_context = list()) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Check dimensionality
  if (p < 10) {
    if (isTRUE(problem_context$expert_knowledge)) {
      return("OLS or Ridge")
    } else {
      return("Ridge or Subset Selection")
    }
  } else if (p < 50) {
    if (isTRUE(problem_context$multicollinearity)) {
      return("Ridge or Elastic Net")
    } else {
      return("Lasso or Elastic Net")
    }
  } else {  # p >= 50
    if (isTRUE(problem_context$sparse_signal)) {
      return("Lasso or Elastic Net")
    } else {
      return("Ridge or PCR")
    }
  }
}

# Run the study
cat("Starting Comprehensive Variable Selection Study\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

all_results <- run_comprehensive_study()

# Create visualizations
cat("\nCreating visualizations...\n")
plots <- create_visualizations(all_results)

# Demonstrate method selection
cat("\n=== METHOD SELECTION EXAMPLES ===\n")

# Example 1: Low-dimensional curated features
X1 <- generate_design_matrices(n_samples = 100)$X1
cat("Low-dimensional curated features (p =", ncol(X1), "):\n")
cat("Recommended method:", select_method(X1, rep(0, nrow(X1)), list(expert_knowledge = TRUE)), "\n\n")

# Example 2: Moderate-dimensional with multicollinearity
X2 <- generate_design_matrices(n_samples = 100)$X2
cat("Moderate-dimensional with multicollinearity (p =", ncol(X2), "):\n")
cat("Recommended method:", select_method(X2, rep(0, nrow(X2)), list(multicollinearity = TRUE)), "\n\n")

# Example 3: High-dimensional sparse signal
X3 <- generate_design_matrices(n_samples = 100)$X3
cat("High-dimensional sparse signal (p =", ncol(X3), "):\n")
cat("Recommended method:", select_method(X3, rep(0, nrow(X3)), list(sparse_signal = TRUE)), "\n\n")

cat("\nStudy completed!\n")
