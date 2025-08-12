# Naive Bayes Classifier in R
library(e1071)
library(ggplot2)
library(gridExtra)
library(caret)

# Custom Naive Bayes implementation
naive_bayes_scratch <- function(X, y) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Initialize parameters
  priors <- rep(0, n_classes)
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  variances <- matrix(0, nrow = n_classes, ncol = n_features)
  
  # Estimate parameters for each class
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    n_class <- sum(class_mask)
    
    # Prior probability
    priors[i] <- n_class / n_samples
    
    # Mean and variance for each feature
    means[i,] <- colMeans(class_data)
    variances[i,] <- apply(class_data, 2, var)
    
    # Add small constant to avoid zero variance
    variances[i,] <- pmax(variances[i,], 1e-9)
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    variances = variances
  ))
}

# Prediction function
predict_naive_bayes <- function(model, X) {
  X <- as.matrix(X)
  n_samples <- nrow(X)
  n_classes <- length(model$classes)
  
  log_proba <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Log prior
    log_proba[, i] <- log(model$priors[i])
    
    # Log likelihood for each feature
    for (j in 1:ncol(X)) {
      mu <- model$means[i, j]
      sigma2 <- model$variances[i, j]
      
      # Gaussian log-likelihood
      log_likelihood <- -0.5 * log(2 * pi * sigma2) - 
                       0.5 * (X[, j] - mu)^2 / sigma2
      
      log_proba[, i] <- log_proba[, i] + log_likelihood
    }
  }
  
  # Return predicted classes
  predictions <- model$classes[apply(log_proba, 1, which.max)]
  return(predictions)
}

# Demonstrate Naive Bayes
demonstrate_naive_bayes_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 1000
  n_features <- 4
  
  # Generate 3 classes
  n_per_class <- n_samples %/% 3
  
  # Class 0
  X0 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 0, 0), 
                      Sigma = diag(4))
  
  # Class 1
  X1 <- MASS::mvrnorm(n_per_class, mu = c(2, 2, 0, 0), 
                      Sigma = diag(4))
  
  # Class 2
  X2 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 2, 2), 
                      Sigma = diag(4))
  
  X <- rbind(X0, X1, X2)
  y <- rep(c(0, 1, 2), each = n_per_class)
  
  # Split data
  train_idx <- sample(1:nrow(X), 0.7 * nrow(X))
  X_train <- X[train_idx,]
  y_train <- y[train_idx]
  X_test <- X[-train_idx,]
  y_test <- y[-train_idx]
  
  # Fit our implementation
  nb_model <- naive_bayes_scratch(X_train, y_train)
  y_pred_scratch <- predict_naive_bayes(nb_model, X_test)
  
  # Fit e1071 implementation
  nb_e1071 <- naiveBayes(X_train, y_train)
  y_pred_e1071 <- predict(nb_e1071, X_test)
  
  # Calculate accuracy
  accuracy_scratch <- mean(y_pred_scratch == y_test)
  accuracy_e1071 <- mean(y_pred_e1071 == y_test)
  
  cat("Naive Bayes Results:\n")
  cat("-" * 40, "\n")
  cat("Our Implementation Accuracy:", round(accuracy_scratch, 4), "\n")
  cat("e1071 Implementation Accuracy:", round(accuracy_e1071, 4), "\n")
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  # Plot original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = "Original Data (Features 1 & 2)", color = "Class") +
    theme_minimal()
  
  # Feature importance
  feature_importance <- rep(0, n_features)
  for (j in 1:n_features) {
    overall_mean <- mean(X[, j])
    between_var <- sum(sapply(unique(y), function(c) {
      sum(y == c) * (mean(X[y == c, j]) - overall_mean)^2
    }))
    within_var <- sum(sapply(unique(y), function(c) {
      sum((X[y == c, j] - mean(X[y == c, j]))^2)
    }))
    feature_importance[j] <- between_var / within_var
  }
  
  df_importance <- data.frame(
    feature = 1:n_features,
    importance = feature_importance
  )
  
  p2 <- ggplot(df_importance, aes(x = feature, y = importance)) +
    geom_bar(stat = "identity") +
    labs(title = "Feature Importance (Variance Ratio)", 
         x = "Feature Index", y = "Between/Within Variance Ratio") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(nb_model = nb_model, nb_e1071 = nb_e1071))
}

# Numerical stability demonstration
demonstrate_numerical_issues_r <- function() {
  # Generate data with extreme values
  set.seed(42)
  
  # Normal data
  X_normal <- rnorm(100, 0, 1)
  
  # Extreme data
  X_extreme <- rnorm(100, 10, 1)
  
  # Compute Gaussian PDF
  gaussian_pdf <- function(x, mu, sigma) {
    return((1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((x - mu) / sigma)^2))
  }
  
  gaussian_log_pdf <- function(x, mu, sigma) {
    return(-0.5 * log(2 * pi * sigma^2) - 0.5 * ((x - mu) / sigma)^2)
  }
  
  # Test points
  test_points <- seq(-5, 15, length.out = 1000)
  
  # Compute probabilities
  pdf_normal <- sapply(test_points, function(x) gaussian_pdf(x, 0, 1))
  pdf_extreme <- sapply(test_points, function(x) gaussian_pdf(x, 10, 1))
  
  log_pdf_normal <- sapply(test_points, function(x) gaussian_log_pdf(x, 0, 1))
  log_pdf_extreme <- sapply(test_points, function(x) gaussian_log_pdf(x, 10, 1))
  
  # Create visualizations
  par(mfrow = c(2, 2))
  
  # PDF for normal data
  plot(test_points, pdf_normal, type = "l", main = "Gaussian PDF (μ=0, σ=1)",
       xlab = "x", ylab = "f(x)")
  grid()
  
  # PDF for extreme data
  plot(test_points, pdf_extreme, type = "l", main = "Gaussian PDF (μ=10, σ=1)",
       xlab = "x", ylab = "f(x)")
  grid()
  
  # Log PDF for normal data
  plot(test_points, log_pdf_normal, type = "l", main = "Gaussian Log-PDF (μ=0, σ=1)",
       xlab = "x", ylab = "log f(x)")
  grid()
  
  # Log PDF for extreme data
  plot(test_points, log_pdf_extreme, type = "l", main = "Gaussian Log-PDF (μ=10, σ=1)",
       xlab = "x", ylab = "log f(x)")
  grid()
  
  par(mfrow = c(1, 1))
  
  # Demonstrate numerical issues
  cat("Numerical Stability Analysis:\n")
  cat("-" * 40, "\n")
  
  # Test with extreme point
  extreme_point <- 20
  mu <- 0
  sigma <- 1
  
  pdf_value <- gaussian_pdf(extreme_point, mu, sigma)
  log_pdf_value <- gaussian_log_pdf(extreme_point, mu, sigma)
  
  cat("Point:", extreme_point, "\n")
  cat("Mean:", mu, "Std:", sigma, "\n")
  cat("PDF value:", format(pdf_value, scientific = TRUE), "\n")
  cat("Log-PDF value:", round(log_pdf_value, 4), "\n")
  cat("Recovered PDF:", format(exp(log_pdf_value), scientific = TRUE), "\n")
  
  return(list(pdf_value = pdf_value, log_pdf_value = log_pdf_value))
}

# Safe Naive Bayes prediction
safe_naive_bayes_predict <- function(X, model) {
  # This would implement log-probability based prediction
  # For simplicity, we'll use the regular prediction function
  return(predict_naive_bayes(model, X))
}

# Regularized Naive Bayes
regularized_naive_bayes <- function(X, y, epsilon = 1e-9) {
  nb_model <- naive_bayes_scratch(X, y)
  
  # Regularize variances
  nb_model$variances <- pmax(nb_model$variances, epsilon)
  
  return(nb_model)
}

# Text classification example
text_classification_example_r <- function() {
  # Sample text data
  texts <- c(
    "great movie amazing acting",
    "terrible film waste of time", 
    "excellent performance brilliant",
    "boring plot disappointing",
    "fantastic story wonderful",
    "awful acting bad script",
    "outstanding film superb",
    "poor quality terrible",
    "incredible movie perfect",
    "horrible waste bad"
  )
  
  labels <- c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0)  # 1=positive, 0=negative
  
  # Simple word-based features (for demonstration)
  # In practice, you'd use proper text vectorization
  words <- unique(unlist(strsplit(paste(texts, collapse = " "), " ")))
  
  # Create feature matrix (word presence)
  X <- matrix(0, nrow = length(texts), ncol = length(words))
  for (i in 1:length(texts)) {
    text_words <- strsplit(texts[i], " ")[[1]]
    for (j in 1:length(words)) {
      X[i, j] <- as.numeric(words[j] %in% text_words)
    }
  }
  
  # Split data
  train_idx <- sample(1:nrow(X), 0.7 * nrow(X))
  X_train <- X[train_idx,]
  y_train <- labels[train_idx]
  X_test <- X[-train_idx,]
  y_test <- labels[-train_idx]
  
  # Fit Naive Bayes
  nb_model <- naive_bayes_scratch(X_train, y_train)
  y_pred <- predict_naive_bayes(nb_model, X_test)
  
  # Calculate accuracy
  accuracy <- mean(y_pred == y_test)
  
  cat("Text Classification Results:\n")
  cat("-" * 40, "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
  
  # Show most discriminative words
  cat("\nMost discriminative words:\n")
  feature_importance <- rep(0, ncol(X))
  for (j in 1:ncol(X)) {
    overall_mean <- mean(X[, j])
    between_var <- sum(sapply(unique(labels), function(c) {
      sum(labels == c) * (mean(X[labels == c, j]) - overall_mean)^2
    }))
    within_var <- sum(sapply(unique(labels), function(c) {
      sum((X[labels == c, j] - mean(X[labels == c, j]))^2)
    }))
    feature_importance[j] <- between_var / within_var
  }
  
  # Show top words
  top_indices <- order(feature_importance, decreasing = TRUE)[1:5]
  for (i in top_indices) {
    cat("  ", words[i], ":", round(feature_importance[i], 3), "\n")
  }
  
  return(list(nb_model = nb_model, words = words, feature_importance = feature_importance))
}

# Medical diagnosis example
medical_diagnosis_example_r <- function() {
  # Simulate medical data
  set.seed(42)
  n_samples <- 1000
  
  # Features: age, blood_pressure, cholesterol, glucose
  age <- rnorm(n_samples, 50, 15)
  blood_pressure <- rnorm(n_samples, 120, 20)
  cholesterol <- rnorm(n_samples, 200, 40)
  glucose <- rnorm(n_samples, 100, 20)
  
  X <- cbind(age, blood_pressure, cholesterol, glucose)
  
  # Disease risk based on features
  risk_score <- age * 0.1 + (blood_pressure - 120) * 0.05 + 
                (cholesterol - 200) * 0.02 + (glucose - 100) * 0.03 +
                rnorm(n_samples, 0, 0.1)
  
  y <- as.numeric(risk_score > median(risk_score))
  
  # Split data
  train_idx <- sample(1:nrow(X), 0.7 * nrow(X))
  X_train <- X[train_idx,]
  y_train <- y[train_idx]
  X_test <- X[-train_idx,]
  y_test <- y[-train_idx]
  
  # Fit Naive Bayes
  nb_model <- naive_bayes_scratch(X_train, y_train)
  y_pred <- predict_naive_bayes(nb_model, X_test)
  
  # Calculate accuracy
  accuracy <- mean(y_pred == y_test)
  
  cat("Medical Diagnosis Results:\n")
  cat("-" * 40, "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
  
  # Feature importance
  feature_names <- c('Age', 'Blood Pressure', 'Cholesterol', 'Glucose')
  feature_importance <- rep(0, 4)
  
  for (j in 1:4) {
    overall_mean <- mean(X[, j])
    between_var <- sum(sapply(unique(y), function(c) {
      sum(y == c) * (mean(X[y == c, j]) - overall_mean)^2
    }))
    within_var <- sum(sapply(unique(y), function(c) {
      sum((X[y == c, j] - mean(X[y == c, j]))^2)
    }))
    feature_importance[j] <- between_var / within_var
  }
  
  # Plot feature importance
  df_importance <- data.frame(
    feature = feature_names,
    importance = feature_importance
  )
  
  p <- ggplot(df_importance, aes(x = feature, y = importance)) +
    geom_bar(stat = "identity") +
    labs(title = "Feature Importance in Medical Diagnosis",
         y = "Between/Within Variance Ratio") +
    theme_minimal()
  
  print(p)
  
  return(list(nb_model = nb_model, feature_importance = feature_importance))
}

# Analyze feature independence
analyze_feature_independence_r <- function(X, y) {
  n_features <- ncol(X)
  correlations <- matrix(0, nrow = n_features, ncol = n_features)
  
  # Calculate correlations
  for (i in 1:n_features) {
    for (j in 1:n_features) {
      correlations[i, j] <- cor(X[, i], X[, j])
    }
  }
  
  # Plot correlation matrix
  df_corr <- data.frame(
    Var1 = rep(1:n_features, each = n_features),
    Var2 = rep(1:n_features, times = n_features),
    Correlation = as.vector(correlations)
  )
  
  p <- ggplot(df_corr, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                        midpoint = 0, limit = c(-1, 1)) +
    labs(title = "Feature Correlation Matrix",
         x = "Feature", y = "Feature") +
    theme_minimal()
  
  print(p)
  
  # Calculate average absolute correlation (excluding diagonal)
  avg_corr <- mean(abs(correlations[upper.tri(correlations)]))
  cat("Average absolute correlation:", round(avg_corr, 4), "\n")
  
  return(list(correlations = correlations, avg_corr = avg_corr))
}

# Plot decision boundaries
plot_naive_bayes_decision_boundaries_r <- function(X, y, model, title = "Naive Bayes Decision Boundaries") {
  # Create mesh grid
  x_min <- min(X[, 1]) - 1
  x_max <- max(X[, 1]) + 1
  y_min <- min(X[, 2]) - 1
  y_max <- max(X[, 2]) + 1
  
  xx <- seq(x_min, x_max, length.out = 100)
  yy <- seq(y_min, y_max, length.out = 100)
  
  # Create test points (use mean values for other features)
  test_points <- matrix(0, nrow = 10000, ncol = ncol(X))
  test_points[, 1] <- rep(xx, each = 100)
  test_points[, 2] <- rep(yy, times = 100)
  
  # Use mean values for remaining features
  for (j in 3:ncol(X)) {
    test_points[, j] <- mean(X[, j])
  }
  
  # Predict
  Z <- predict_naive_bayes(model, test_points)
  Z <- matrix(Z, nrow = 100, ncol = 100)
  
  # Create plot
  df_original <- data.frame(
    x1 = X[, 1],
    x2 = X[, 2],
    class = factor(y)
  )
  
  p <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = title, color = "Class") +
    theme_minimal()
  
  print(p)
}

# Main function to demonstrate Naive Bayes implementation
main <- function() {
  cat("Naive Bayes Classifier Demonstration\n")
  cat("=", 50, "\n")
  
  # Basic demonstration
  cat("\n1. Basic Naive Bayes Demonstration:\n")
  nb_results <- demonstrate_naive_bayes_r()
  
  # Numerical stability demonstration
  cat("\n2. Numerical Stability Issues:\n")
  numerical_results <- demonstrate_numerical_issues_r()
  
  # Text classification example
  cat("\n3. Text Classification Example:\n")
  text_results <- text_classification_example_r()
  
  # Medical diagnosis example
  cat("\n4. Medical Diagnosis Example:\n")
  medical_results <- medical_diagnosis_example_r()
  
  # Generate data for additional analysis
  set.seed(42)
  X <- MASS::mvrnorm(1000, mu = c(0, 0), Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  y <- sample(0:1, 1000, replace = TRUE)
  
  # Feature independence analysis
  cat("\n5. Feature Independence Analysis:\n")
  independence_results <- analyze_feature_independence_r(X, y)
  
  # Decision boundaries
  cat("\n6. Decision Boundaries:\n")
  nb_model <- naive_bayes_scratch(X, y)
  plot_naive_bayes_decision_boundaries_r(X, y, nb_model)
  
  return(list(
    nb_results = nb_results,
    numerical_results = numerical_results,
    text_results = text_results,
    medical_results = medical_results,
    independence_results = independence_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
