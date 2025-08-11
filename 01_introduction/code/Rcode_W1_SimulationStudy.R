# =============================================================================
# PSL: kNN vs Linear Regression Simulation Study
# =============================================================================
#
# This script demonstrates the bias-variance tradeoff through a comprehensive
# simulation study comparing k-Nearest Neighbors (kNN) and Linear Regression
# on two different data-generating processes.
#
# Example 1: Simple Gaussian Classes (Linear Decision Boundary)
# Example 2: Mixture of Gaussians (Non-linear Decision Boundary)
#
# Key Concepts Demonstrated:
# - Bias-Variance Tradeoff
# - Model Complexity vs. Performance
# - Cross-Validation for Model Selection
# - Bayes Optimal Classifier
# - Overfitting and Underfitting
#
# Author: Statistical Learning Course
# Date: 2024
# =============================================================================

# Load required libraries
library(class)      # For kNN implementation
library(ggplot2)    # For enhanced visualization
library(dplyr)      # For data manipulation

# Set random seed for reproducibility
set.seed(100)

# =============================================================================
# EXAMPLE 1: SIMPLE GAUSSIAN CLASSES
# =============================================================================
# Data generating process:
# - Class 1: N(μ₁=[1,0], σ²I)
# - Class 0: N(μ₀=[0,1], σ²I)
# - Linear decision boundary exists
# - Equal class priors

cat("=" %R% 60, "\n")
cat("EXAMPLE 1: SIMPLE GAUSSIAN CLASSES\n")
cat("=" %R% 60, "\n")

# =============================================================================
# Data Generation
# =============================================================================

# Set model parameters
p <- 2                    # Number of features (dimensions)
sigma <- 1               # Standard deviation for both classes
mu1 <- c(1, 0)          # Mean vector for class 1
mu0 <- c(0, 1)          # Mean vector for class 0

cat("Model Parameters:\n")
cat("  Class 1 mean:", mu1, "\n")
cat("  Class 0 mean:", mu0, "\n")
cat("  Standard deviation:", sigma, "\n")

# Generate training data
n_per_class <- 100       # Number of samples per class for training
n_total_train <- 2 * n_per_class

cat("  Training samples per class:", n_per_class, "\n")
cat("  Total training samples:", n_total_train, "\n")

# Generate training data matrix
# First generate noise: 2n × p matrix of iid N(0, σ²) samples
train_noise <- matrix(rnorm(n_total_train * p), n_total_train, p) * sigma

# Add class means: first n rows get μ₁, next n rows get μ₀
train_means <- rbind(
  matrix(rep(mu1, n_per_class), nrow = n_per_class, byrow = TRUE),
  matrix(rep(mu0, n_per_class), nrow = n_per_class, byrow = TRUE)
)

# Combine noise and means to get training data
X_train <- train_noise + train_means
y_train <- factor(c(rep(1, n_per_class), rep(0, n_per_class)))

cat("  Training data dimensions:", dim(X_train), "\n")

# Generate test data (same process, larger sample size)
n_test_per_class <- 5000
n_total_test <- 2 * n_test_per_class

cat("  Test samples per class:", n_test_per_class, "\n")
cat("  Total test samples:", n_total_test, "\n")

test_noise <- matrix(rnorm(n_total_test * p), n_total_test, p) * sigma
test_means <- rbind(
  matrix(rep(mu1, n_test_per_class), nrow = n_test_per_class, byrow = TRUE),
  matrix(rep(mu0, n_test_per_class), nrow = n_test_per_class, byrow = TRUE)
)

X_test <- test_noise + test_means
y_test <- factor(c(rep(1, n_test_per_class), rep(0, n_test_per_class)))

cat("  Test data dimensions:", dim(X_test), "\n")

# =============================================================================
# Data Visualization
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("DATA VISUALIZATION\n")
cat("-" %R% 40, "\n")

# Create data frame for ggplot2
train_df <- data.frame(
  X1 = X_train[, 1],
  X2 = X_train[, 2],
  Class = y_train
)

# Enhanced visualization using ggplot2
p1 <- ggplot(train_df, aes(x = X1, y = X2, color = Class)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_point(data = data.frame(X1 = mu1[1], X2 = mu1[2], Class = "1"), 
             shape = 3, size = 5, color = "blue") +
  geom_point(data = data.frame(X1 = mu0[1], X2 = mu0[2], Class = "0"), 
             shape = 3, size = 5, color = "red") +
  scale_color_manual(values = c("red", "blue"), 
                     labels = c("Class 0", "Class 1")) +
  labs(title = "Example 1: Simple Gaussian Classes",
       subtitle = "Training Data with True Class Means",
       x = "Feature 1", y = "Feature 2") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p1)

# =============================================================================
# k-Nearest Neighbors (kNN) Evaluation
# =============================================================================

cat("\n" %R% "=" %R% 40, "\n")
cat("kNN EVALUATION\n")
cat("=" %R% 40, "\n")

# Define k values to evaluate (from textbook)
k_values <- c(151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1)
n_k <- length(k_values)

cat("Evaluating kNN for", n_k, "different k values\n")
cat("k values:", paste(k_values, collapse = ", "), "\n\n")

# Initialize vectors to store errors
train_errors_knn <- numeric(n_k)
test_errors_knn <- numeric(n_k)

# Create results table header
cat(sprintf("%4s %12s %12s %8s\n", "k", "Train Error", "Test Error", "DF"))
cat(paste(rep("-", 40), collapse = ""), "\n")

# Evaluate kNN for each k value
for (i in 1:n_k) {
  k <- k_values[i]
  
  # Fit kNN model and get predictions
  y_train_pred <- knn(X_train, X_train, y_train, k = k)
  y_test_pred <- knn(X_train, X_test, y_train, k = k)
  
  # Calculate error rates
  train_errors_knn[i] <- sum(y_train != y_train_pred) / n_total_train
  test_errors_knn[i] <- sum(y_test != y_test_pred) / n_total_test
  
  # Approximate degrees of freedom (n/k)
  df <- round(n_total_train / k)
  
  # Print results
  cat(sprintf("%4d %12.4f %12.4f %8d\n", k, train_errors_knn[i], test_errors_knn[i], df))
}

# Display results matrix
cat("\nResults Matrix:\n")
results_matrix <- cbind(k = k_values, 
                       Train_Error = train_errors_knn, 
                       Test_Error = test_errors_knn)
print(results_matrix)

# =============================================================================
# Cross-Validation for kNN
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("5-FOLD CROSS-VALIDATION\n")
cat("-" %R% 40, "\n")

# Initialize cross-validation error vector
cv_errors <- numeric(n_k)

# Set up 5-fold cross-validation
n_folds <- 5
fold_size <- n_total_train / n_folds

# Create fold indices
fold_indices <- sample(1:n_total_train, n_total_train, replace = FALSE)
fold_boundaries <- seq(0, n_total_train, by = fold_size)

cat("Performing", n_folds, "-fold cross-validation...\n")

# Perform cross-validation
for (fold in 1:n_folds) {
  # Define test fold indices
  test_start <- fold_boundaries[fold] + 1
  test_end <- fold_boundaries[fold + 1]
  test_indices <- fold_indices[test_start:test_end]
  
  # Define training indices (all except test fold)
  train_indices <- fold_indices[-test_indices]
  
  # Split data
  X_train_cv <- X_train[train_indices, ]
  y_train_cv <- y_train[train_indices]
  X_test_cv <- X_train[test_indices, ]
  y_test_cv <- y_train[test_indices]
  
  # Evaluate each k value
  for (i in 1:n_k) {
    k <- k_values[i]
    y_pred_cv <- knn(X_train_cv, X_test_cv, y_train_cv, k = k)
    cv_errors[i] <- cv_errors[i] + sum(y_test_cv != y_pred_cv)
  }
}

# Average CV errors
cv_errors <- cv_errors / n_total_train

# Find best k
best_k_idx <- which.min(cv_errors)
best_k <- k_values[best_k_idx]

cat("\nCross-validation results:\n")
for (i in 1:n_k) {
  cat(sprintf("k = %3d: CV Error = %.4f\n", k_values[i], cv_errors[i]))
}

cat("\nBest k from CV:", best_k, "(CV Error:", cv_errors[best_k_idx], ")\n")

# Evaluate kNN with CV-chosen k
y_train_pred_cv <- knn(X_train, X_train, y_train, k = best_k)
y_test_pred_cv <- knn(X_train, X_test, y_train, k = best_k)

train_error_cv <- sum(y_train != y_train_pred_cv) / n_total_train
test_error_cv <- sum(y_test != y_test_pred_cv) / n_total_test

cat("CV kNN - Training Error:", train_error_cv, "\n")
cat("CV kNN - Test Error:", test_error_cv, "\n")

# =============================================================================
# Linear Regression Evaluation
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("LINEAR REGRESSION EVALUATION\n")
cat("-" %R% 40, "\n")

# Fit linear regression model
# Convert factor to numeric (0, 1) for regression
y_train_numeric <- as.numeric(y_train) - 1

# Fit model: Y ~ X1 + X2
lm_model <- lm(y_train_numeric ~ X_train)

# Get predictions
y_train_pred_prob <- predict(lm_model, data.frame(X_train = X_train))
y_test_pred_prob <- predict(lm_model, data.frame(X_train = X_test))

# Convert probabilities to binary predictions (threshold at 0.5)
y_train_pred_lr <- as.numeric(y_train_pred_prob > 0.5)
y_test_pred_lr <- as.numeric(y_test_pred_prob > 0.5)

# Calculate error rates
train_error_lr <- sum(y_train_numeric != y_train_pred_lr) / n_total_train
test_error_lr <- sum(as.numeric(y_test) - 1 != y_test_pred_lr) / n_total_test

cat("Linear Regression Results:\n")
cat("  Training Error:", train_error_lr, "\n")
cat("  Test Error:", test_error_lr, "\n")
cat("  Model Coefficients:\n")
cat("    Intercept:", coef(lm_model)[1], "\n")
cat("    X1 coefficient:", coef(lm_model)[2], "\n")
cat("    X2 coefficient:", coef(lm_model)[3], "\n")

# Display confusion matrices
cat("\nTraining Data Confusion Matrix:\n")
print(table(Actual = y_train, Predicted = factor(y_train_pred_lr)))

cat("\nTest Data Confusion Matrix:\n")
print(table(Actual = y_test, Predicted = factor(y_test_pred_lr)))

# =============================================================================
# Bayes Optimal Classifier
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("BAYES OPTIMAL CLASSIFIER\n")
cat("-" %R% 40, "\n")

# Bayes decision rule: predict class 1 if ||x - μ₁||² < ||x - μ₀||²
# This simplifies to: 2x^T(μ₁ - μ₀) > ||μ₁||² - ||μ₀||²

# Compute decision function components
mu_diff <- mu1 - mu0
threshold <- sum(mu1^2) - sum(mu0^2)

# Apply Bayes rule to test data
y_test_pred_bayes <- as.numeric(2 * X_test %*% matrix(mu_diff, nrow = 2) > threshold)
bayes_error <- sum(as.numeric(y_test) - 1 != y_test_pred_bayes) / n_total_test

cat("Bayes Optimal Classifier Results:\n")
cat("  Decision Rule: 2x^T(", paste(mu_diff, collapse = ", "), ") >", threshold, "\n")
cat("  Bayes Error Rate:", bayes_error, "\n")

# =============================================================================
# Performance Visualization
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("PERFORMANCE VISUALIZATION\n")
cat("-" %R% 40, "\n")

# Calculate degrees of freedom for x-axis
df_values <- round(n_total_train / k_values)

# Create performance comparison plot
plot_data <- data.frame(
  k = k_values,
  df = df_values,
  Train_Error = train_errors_knn,
  Test_Error = test_errors_knn,
  CV_Error = cv_errors
)

# Reshape data for plotting
plot_data_long <- plot_data %>%
  select(k, df, Train_Error, Test_Error, CV_Error) %>%
  tidyr::gather(key = "Error_Type", value = "Error_Rate", -k, -df)

# Create comprehensive performance plot
p2 <- ggplot(plot_data_long, aes(x = df, y = Error_Rate, color = Error_Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  
  # Add linear regression performance
  geom_point(data = data.frame(df = 3, Error_Rate = train_error_lr, 
                               Error_Type = "Linear_Train"), 
             shape = 15, size = 4, color = "blue") +
  geom_point(data = data.frame(df = 3, Error_Rate = test_error_lr, 
                               Error_Type = "Linear_Test"), 
             shape = 15, size = 4, color = "red") +
  
  # Add Bayes error line
  geom_hline(yintercept = bayes_error, linetype = "dashed", 
             color = "purple", size = 1) +
  annotate("text", x = max(df_values), y = bayes_error + 0.01, 
           label = paste("Bayes Error:", round(bayes_error, 4)), 
           color = "purple", hjust = 1) +
  
  # Add CV best k marker
  geom_point(data = data.frame(df = df_values[best_k_idx], 
                               Error_Rate = cv_errors[best_k_idx], 
                               Error_Type = "CV_Best"), 
             shape = 8, size = 5, color = "green") +
  
  scale_color_manual(values = c("blue", "red", "green", "blue", "red"),
                     labels = c("CV Error", "kNN Test Error", "CV Best k", 
                               "Linear Train", "Linear Test")) +
  
  labs(title = "Bias-Variance Tradeoff: kNN vs Linear Regression vs Bayes Optimal",
       subtitle = "Example 1: Simple Gaussian Classes",
       x = "Model Complexity (Degrees of Freedom)",
       y = "Error Rate",
       color = "Error Type") +
  
  theme_bw() +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

print(p2)

# =============================================================================
# EXAMPLE 2: MIXTURE OF GAUSSIANS
# =============================================================================

cat("\n" %R% "=" %R% 60, "\n")
cat("EXAMPLE 2: MIXTURE OF GAUSSIANS\n")
cat("=" %R% 60, "\n")

cat("Data generating process:\n")
cat("- Each class: mixture of 10 Gaussian components\n")
cat("- Non-linear decision boundary\n")
cat("- More complex class-conditional distributions\n")

# =============================================================================
# Mixture Data Generation
# =============================================================================

# Set parameters for mixture model
n_centers <- 10          # Number of mixture components per class
sigma_centers <- 1       # Standard deviation for generating centers
sigma_noise <- sqrt(1/5) # Standard deviation for generating data around centers

cat("\nMixture Model Parameters:\n")
cat("  Mixture components per class:", n_centers, "\n")
cat("  Center spread:", sigma_centers, "\n")
cat("  Noise level:", sigma_noise, "\n")

# Generate mixture centers for each class
# Class 1 centers: around (1, 0)
centers1 <- matrix(rnorm(n_centers * p), n_centers, p) * sigma_centers + 
            cbind(rep(1, n_centers), rep(0, n_centers))

# Class 0 centers: around (0, 1)
centers0 <- matrix(rnorm(n_centers * p), n_centers, p) * sigma_centers + 
            cbind(rep(0, n_centers), rep(1, n_centers))

# Generate training data from mixture
X_train_mix <- matrix(0, n_total_train, p)
y_train_mix <- factor(rep(c(1, 0), each = n_per_class))

# Generate class 1 data
for (i in 1:n_per_class) {
  # Randomly select a center
  center_idx <- sample(1:n_centers, 1)
  center <- centers1[center_idx, ]
  
  # Generate point around the center
  X_train_mix[i, ] <- rnorm(p, mean = center, sd = sigma_noise)
}

# Generate class 0 data
for (i in 1:n_per_class) {
  center_idx <- sample(1:n_centers, 1)
  center <- centers0[center_idx, ]
  X_train_mix[n_per_class + i, ] <- rnorm(p, mean = center, sd = sigma_noise)
}

cat("  Mixture training data dimensions:", dim(X_train_mix), "\n")

# =============================================================================
# Mixture Data Visualization
# =============================================================================

cat("\n" %R% "-" %R% 40, "\n")
cat("MIXTURE DATA VISUALIZATION\n")
cat("-" %R% 40, "\n")

# Create data frame for visualization
mix_df <- data.frame(
  X1 = X_train_mix[, 1],
  X2 = X_train_mix[, 2],
  Class = y_train_mix
)

# Create centers data frames
centers1_df <- data.frame(X1 = centers1[, 1], X2 = centers1[, 2], Class = "1")
centers0_df <- data.frame(X1 = centers0[, 1], X2 = centers0[, 2], Class = "0")

# Enhanced visualization
p3 <- ggplot(mix_df, aes(x = X1, y = X2, color = Class)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_point(data = centers1_df, shape = 3, size = 4, color = "blue") +
  geom_point(data = centers0_df, shape = 3, size = 4, color = "red") +
  scale_color_manual(values = c("red", "blue"), 
                     labels = c("Class 0", "Class 1")) +
  labs(title = "Example 2: Mixture of Gaussians",
       subtitle = "Training Data with Mixture Centers",
       x = "Feature 1", y = "Feature 2") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p3)

# =============================================================================
# Summary and Key Insights
# =============================================================================

cat("\n" %R% "=" %R% 60, "\n")
cat("SUMMARY AND KEY INSIGHTS\n")
cat("=" %R% 60, "\n")

cat("1. Bias-Variance Tradeoff:\n")
cat("   - Small k (high complexity): Low bias, high variance\n")
cat("   - Large k (low complexity): High bias, low variance\n")
cat("   - Optimal k balances bias and variance\n\n")

cat("2. Model Comparison:\n")
cat("   - Linear Regression: Good for Example 1 (linear boundary)\n")
cat("   - kNN: Adapts to both linear and non-linear boundaries\n")
cat("   - Bayes Rule: Theoretical optimum (when known)\n\n")

cat("3. Cross-Validation:\n")
cat("   - Provides realistic estimate of generalization error\n")
cat("   - Helps select optimal model complexity\n")
cat("   - More reliable than test set performance for model selection\n\n")

cat("4. Practical Implications:\n")
cat("   - Choose model complexity based on data characteristics\n")
cat("   - Use cross-validation for hyperparameter tuning\n")
cat("   - Consider computational cost vs. performance trade-offs\n\n")

cat("5. Key Results:\n")
cat("   - Best k from CV:", best_k, "\n")
cat("   - CV kNN Test Error:", test_error_cv, "\n")
cat("   - Linear Regression Test Error:", test_error_lr, "\n")
cat("   - Bayes Error Rate:", bayes_error, "\n")

cat("\nSimulation study completed successfully!\n") 