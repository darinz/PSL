# Linear Discriminant Analysis in R
library(MASS)
library(ggplot2)
library(gridExtra)
library(caret)
library(pROC)

# Generate synthetic data for LDA
generate_lda_data <- function(n_samples = 1000, n_features = 2, n_classes = 3, seed = 42) {
  set.seed(seed)
  
  # Generate class means
  means <- matrix(rnorm(n_classes * n_features, 0, 2), nrow = n_classes)
  
  # Generate shared covariance matrix
  A <- matrix(rnorm(n_features^2), nrow = n_features)
  covariance <- A %*% t(A) + diag(n_features)
  
  # Generate samples
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  y <- rep(0, n_samples)
  samples_per_class <- n_samples %/% n_classes
  
  for (i in 1:n_classes) {
    start_idx <- (i-1) * samples_per_class + 1
    end_idx <- i * samples_per_class
    class_samples <- MASS::mvrnorm(samples_per_class, means[i,], covariance)
    X[start_idx:end_idx,] <- class_samples
    y[start_idx:end_idx] <- i
  }
  
  return(list(X = X, y = y))
}

# Custom LDA implementation
lda_from_scratch <- function(X, y, regularization = 1e-6) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Calculate class priors
  priors <- sapply(classes, function(c) sum(y == c) / n_samples)
  
  # Calculate class means
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  for (i in 1:n_classes) {
    means[i,] <- colMeans(X[y == classes[i],, drop = FALSE])
  }
  
  # Calculate pooled covariance matrix
  covariance <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_samples <- X[y == classes[i],, drop = FALSE]
    class_mean <- means[i,]
    diff <- sweep(class_samples, 2, class_mean, "-")
    covariance <- covariance + t(diff) %*% diff
  }
  covariance <- covariance / (n_samples - n_classes)
  
  # Add regularization
  covariance <- covariance + regularization * diag(n_features)
  
  # Calculate coefficients and intercepts
  cov_inv <- solve(covariance)
  coef <- matrix(0, nrow = n_classes, ncol = n_features)
  intercept <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    coef[i,] <- -2 * cov_inv %*% means[i,]
    intercept[i] <- t(means[i,]) %*% cov_inv %*% means[i,] + 
                   log(det(covariance)) - 2 * log(priors[i])
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    covariance = covariance,
    coef = coef,
    intercept = intercept
  ))
}

# Prediction function
predict_lda <- function(model, X) {
  discriminant_scores <- X %*% t(model$coef) + matrix(model$intercept, 
                                                     nrow = nrow(X), 
                                                     ncol = length(model$intercept), 
                                                     byrow = TRUE)
  predictions <- apply(discriminant_scores, 1, which.max)
  return(model$classes[predictions])
}

# Demonstrate LDA
demonstrate_lda_r <- function() {
  # Generate data
  data <- generate_lda_data(n_samples = 900, n_features = 2, n_classes = 3)
  X <- data$X
  y <- data$y
  
  # Fit our implementation
  lda_model <- lda_from_scratch(X, y)
  
  # Fit MASS implementation
  lda_mass <- lda(X, y)
  
  # Make predictions
  y_pred_scratch <- predict_lda(lda_model, X)
  y_pred_mass <- predict(lda_mass, X)$class
  
  # Calculate accuracy
  accuracy_scratch <- mean(y_pred_scratch == y)
  accuracy_mass <- mean(y_pred_mass == y)
  
  cat("Accuracy Comparison:\n")
  cat("Our Implementation:", round(accuracy_scratch, 4), "\n")
  cat("MASS Implementation:", round(accuracy_mass, 4), "\n")
  
  # Create visualizations
  df <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y),
    pred_scratch = factor(y_pred_scratch),
    pred_mass = factor(y_pred_mass)
  )
  
  # Original data
  p1 <- ggplot(df, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = "Original Data", color = "True Class") +
    theme_minimal()
  
  # Predictions from our implementation
  p2 <- ggplot(df, aes(x = x1, y = x2, color = pred_scratch)) +
    geom_point(alpha = 0.6) +
    labs(title = "Our LDA Predictions", color = "Predicted Class") +
    theme_minimal()
  
  # Predictions from MASS
  p3 <- ggplot(df, aes(x = x1, y = x2, color = pred_mass)) +
    geom_point(alpha = 0.6) +
    labs(title = "MASS LDA Predictions", color = "Predicted Class") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(lda_model = lda_model, lda_mass = lda_mass))
}

# Regularized LDA function
regularized_lda <- function(X, y, alpha = 0.1, regularization = 1e-6) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Calculate class priors
  priors <- sapply(classes, function(c) sum(y == c) / n_samples)
  
  # Calculate class means
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  for (i in 1:n_classes) {
    means[i,] <- colMeans(X[y == classes[i],, drop = FALSE])
  }
  
  # Calculate pooled covariance matrix
  covariance <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_samples <- X[y == classes[i],, drop = FALSE]
    class_mean <- means[i,]
    diff <- sweep(class_samples, 2, class_mean, "-")
    covariance <- covariance + t(diff) %*% diff
  }
  covariance <- covariance / (n_samples - n_classes)
  
  # Apply regularization: convex combination with identity matrix
  identity <- diag(n_features)
  covariance <- (1 - alpha) * covariance + alpha * identity
  
  # Add small regularization for numerical stability
  covariance <- covariance + regularization * diag(n_features)
  
  # Calculate coefficients and intercepts
  cov_inv <- solve(covariance)
  coef <- matrix(0, nrow = n_classes, ncol = n_features)
  intercept <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    coef[i,] <- -2 * cov_inv %*% means[i,]
    intercept[i] <- t(means[i,]) %*% cov_inv %*% means[i,] + 
                   log(det(covariance)) - 2 * log(priors[i])
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    covariance = covariance,
    coef = coef,
    intercept = intercept
  ))
}

# Multi-class LDA with dimensionality reduction
multiclass_lda <- function(X, y) {
  n_classes <- length(unique(y))
  n_components <- min(n_classes - 1, ncol(X))
  
  lda_model <- lda(X, y)
  X_transformed <- predict(lda_model, X)$x[, 1:n_components, drop = FALSE]
  
  return(list(X_transformed = X_transformed, lda_model = lda_model))
}

# Model evaluation function
evaluate_lda_model <- function(X_train, X_test, y_train, y_test) {
  # Fit model
  lda_model <- lda(X_train, y_train)
  
  # Predictions
  predictions <- predict(lda_model, X_test)
  y_pred <- predictions$class
  y_pred_proba <- predictions$posterior
  
  # Calculate metrics
  accuracy <- mean(y_pred == y_test)
  
  # Confusion matrix
  cm <- table(Predicted = y_pred, Actual = y_test)
  
  # For binary classification, calculate additional metrics
  if (length(unique(y_test)) == 2) {
    # Precision, recall, F1
    tp <- sum(y_pred == 1 & y_test == 1)
    fp <- sum(y_pred == 1 & y_test == 0)
    fn <- sum(y_pred == 0 & y_test == 1)
    tn <- sum(y_pred == 0 & y_test == 0)
    
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    # AUC
    auc <- auc(roc(y_test, y_pred_proba[, 2]))
  } else {
    precision <- recall <- f1 <- auc <- NA
  }
  
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1,
    auc = auc,
    confusion_matrix = cm
  ))
}

# Diagnostic plots for LDA
lda_diagnostics <- function(X, y, lda_model) {
  # 1. Check normality assumption
  residuals <- c()
  for (i in 1:length(lda_model$classes)) {
    class_label <- lda_model$classes[i]
    class_mask <- y == class_label
    class_residuals <- X[class_mask,] - matrix(lda_model$means[i,], 
                                              nrow = sum(class_mask), 
                                              ncol = ncol(X), 
                                              byrow = TRUE)
    residuals <- c(residuals, as.vector(class_residuals))
  }
  
  # Create diagnostic plots
  par(mfrow = c(1, 3))
  
  # Q-Q plot for normality
  qqnorm(residuals, main = "Q-Q Plot for Normality Check")
  qqline(residuals)
  
  # Homoscedasticity check
  X_transformed <- predict(lda_model, X)$x
  plot(X_transformed[, 1], residuals[1:length(X_transformed[, 1])], 
       xlab = "First LDA Component", ylab = "Residuals",
       main = "Homoscedasticity Check")
  
  # Feature importance
  feature_importance <- abs(lda_model$coef[1,])
  barplot(feature_importance, names.arg = paste0("F", 1:length(feature_importance)),
          xlab = "Feature Index", ylab = "|Coefficient|",
          main = "Feature Importance")
  
  par(mfrow = c(1, 1))
}

# Iris dataset example
iris_lda_example <- function() {
  # Load iris data
  data(iris)
  
  # Prepare data
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species
  
  # LDA with cross-validation
  lda_model <- lda(X, y)
  
  # Cross-validation
  cv_results <- numeric(5)
  set.seed(42)
  folds <- createFolds(y, k = 5)
  
  for (i in 1:5) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    lda_cv <- lda(X[train_idx,], y[train_idx])
    predictions <- predict(lda_cv, X[test_idx,])
    cv_results[i] <- mean(predictions$class == y[test_idx])
  }
  
  cat("Cross-validation accuracy:", round(mean(cv_results), 4), 
      "(+/-", round(2 * sd(cv_results), 4), ")\n")
  
  # Dimensionality reduction
  X_transformed <- predict(lda_model, X)$x
  cat("Original dimensions:", ncol(X), "\n")
  cat("LDA dimensions:", ncol(X_transformed), "\n")
  
  return(list(lda_model = lda_model, X_transformed = X_transformed, cv_results = cv_results))
}

# Credit risk assessment example
credit_risk_lda <- function() {
  set.seed(42)
  n_samples <- 1000
  
  # Simulate credit data
  income <- rlnorm(n_samples, 10, 0.5)
  debt <- rlnorm(n_samples, 8, 0.3)
  credit_score <- rnorm(n_samples, 700, 100)
  age <- rnorm(n_samples, 35, 10)
  
  X <- cbind(income, debt, credit_score, age)
  
  # Risk classification
  risk_score <- income * 0.3 + debt * (-0.4) + credit_score * 0.2 + age * 0.1 + 
                rnorm(n_samples, 0, 0.1)
  y <- factor(ifelse(risk_score > median(risk_score), "High", "Low"))
  
  # Split data
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Apply LDA
  lda_model <- lda(X_train, y_train)
  predictions <- predict(lda_model, X_test)
  
  # Results
  accuracy <- mean(predictions$class == y_test)
  cat("Credit Risk Classification Accuracy:", round(accuracy, 4), "\n")
  
  # Feature importance
  feature_names <- c("Income", "Debt", "Credit Score", "Age")
  importance <- abs(lda_model$coef[1,])
  
  # Create visualization
  importance_df <- data.frame(
    feature = feature_names,
    importance = importance
  )
  
  p <- ggplot(importance_df, aes(x = feature, y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Feature Importance in Credit Risk LDA",
         x = "Features", y = "|Coefficient|") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
  
  return(list(lda_model = lda_model, accuracy = accuracy))
}

# Robust LDA evaluation with cross-validation
robust_lda_evaluation <- function(X, y, n_splits = 5) {
  set.seed(42)
  folds <- createFolds(y, k = n_splits)
  scores <- numeric(n_splits)
  
  for (i in 1:n_splits) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    lda_model <- lda(X[train_idx,], y[train_idx])
    predictions <- predict(lda_model, X[test_idx,])
    scores[i] <- mean(predictions$class == y[test_idx])
  }
  
  return(list(mean_score = mean(scores), std_score = sd(scores), scores = scores))
}

# Plot LDA decision boundaries
plot_lda_decision_boundaries <- function(X, y, lda_model, title = "LDA Decision Boundaries") {
  # Create mesh grid
  x_range <- range(X[, 1])
  y_range <- range(X[, 2])
  x_grid <- seq(x_range[1] - 1, x_range[2] + 1, by = 0.02)
  y_grid <- seq(y_range[1] - 1, y_range[2] + 1, by = 0.02)
  grid_points <- expand.grid(x = x_grid, y = y_grid)
  
  # Predict on grid
  predictions <- predict(lda_model, as.matrix(grid_points))
  Z <- as.numeric(predictions$class)
  Z_proba <- predictions$posterior[, 2]  # Probability of second class
  
  # Create data frames for plotting
  grid_df <- data.frame(
    x = grid_points$x,
    y = grid_points$y,
    class = factor(Z),
    proba = Z_proba
  )
  
  data_df <- data.frame(
    x = X[, 1],
    y = X[, 2],
    class = factor(y)
  )
  
  # Plot decision boundaries
  p1 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = as.numeric(class)), 
                        alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(title = paste(title, "- Decision Boundaries"),
         x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  # Plot posterior probabilities
  p2 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = proba), 
                        alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.5) +
    labs(title = paste(title, "- Posterior Probabilities"),
         x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
}

# Main function to demonstrate LDA implementation
main <- function() {
  cat("Linear Discriminant Analysis Demonstration\n")
  cat("=", 50, "\n")
  
  # Demonstrate basic LDA
  cat("\n1. Basic LDA Demonstration:\n")
  results <- demonstrate_lda_r()
  
  # Iris dataset example
  cat("\n2. Iris Dataset Example:\n")
  iris_result <- iris_lda_example()
  
  # Credit risk example
  cat("\n3. Credit Risk Assessment Example:\n")
  credit_result <- credit_risk_lda()
  
  # Generate data for diagnostics
  data <- generate_lda_data(n_samples = 900, n_features = 2, n_classes = 3)
  X <- data$X
  y <- data$y
  
  # Split data
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Model evaluation
  cat("\n4. Model Evaluation:\n")
  lda_model <- lda(X_train, y_train)
  eval_results <- evaluate_lda_model(X_train, X_test, y_train, y_test)
  cat("Accuracy:", round(eval_results$accuracy, 4), "\n")
  if (!is.na(eval_results$precision)) {
    cat("Precision:", round(eval_results$precision, 4), "\n")
    cat("Recall:", round(eval_results$recall, 4), "\n")
    cat("F1 Score:", round(eval_results$f1_score, 4), "\n")
    cat("AUC:", round(eval_results$auc, 4), "\n")
  }
  
  # Diagnostics
  cat("\n5. Model Diagnostics:\n")
  lda_diagnostics(X_train, y_train, lda_model)
  
  # Robust evaluation
  cat("\n6. Robust Evaluation:\n")
  robust_results <- robust_lda_evaluation(X, y, n_splits = 5)
  cat("Cross-validation accuracy:", round(robust_results$mean_score, 4), 
      "(+/-", round(robust_results$std_score, 4), ")\n")
  
  return(list(
    lda_model = results$lda_model,
    lda_mass = results$lda_mass,
    iris_result = iris_result,
    credit_result = credit_result
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
