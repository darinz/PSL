# Naive Bayes Classifier in R
library(e1071)
library(caret)
library(ggplot2)
library(gridExtra)
library(tm)
library(wordcloud)
library(pROC)

# Custom Naive Bayes implementation
naive_bayes_classifier <- function(X, y, variant = "gaussian", alpha = 1e-10) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Calculate prior probabilities
  priors <- rep(0, n_classes)
  for (i in 1:n_classes) {
    priors[i] <- sum(y == classes[i]) / n_samples
  }
  
  # Initialize parameters
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  variances <- matrix(0, nrow = n_classes, ncol = n_features)
  feature_probs <- matrix(0, nrow = n_classes, ncol = n_features)
  
  # Fit parameters for each class
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    
    if (variant == "gaussian") {
      # Calculate means and variances
      means[i,] <- colMeans(class_data)
      variances[i,] <- apply(class_data, 2, var)
      # Add regularization to prevent zero variance
      variances[i,] <- pmax(variances[i,], alpha)
    } else if (variant == "multinomial") {
      # Calculate feature probabilities with Laplace smoothing
      feature_counts <- colSums(class_data)
      total_count <- sum(feature_counts)
      feature_probs[i,] <- (feature_counts + alpha) / (total_count + n_features * alpha)
    } else if (variant == "bernoulli") {
      # Calculate probability of feature being present
      feature_present <- colSums(class_data > 0)
      n_class_samples <- sum(class_mask)
      feature_probs[i,] <- (feature_present + alpha) / (n_class_samples + 2 * alpha)
    }
  }
  
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    variances = variances,
    feature_probs = feature_probs,
    variant = variant,
    alpha = alpha
  ))
}

# Predict function
predict_naive_bayes <- function(model, X) {
  n_samples <- nrow(X)
  n_classes <- length(model$classes)
  
  log_probs <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Add log prior
    log_probs[, i] <- log(model$priors[i])
    
    if (model$variant == "gaussian") {
      log_probs[, i] <- log_probs[, i] + gaussian_log_likelihood(X, model$means[i,], model$variances[i,])
    } else if (model$variant == "multinomial") {
      log_probs[, i] <- log_probs[, i] + multinomial_log_likelihood(X, model$feature_probs[i,])
    } else if (model$variant == "bernoulli") {
      log_probs[, i] <- log_probs[, i] + bernoulli_log_likelihood(X, model$feature_probs[i,])
    }
  }
  
  # Return predicted classes
  predictions <- apply(log_probs, 1, which.max)
  return(model$classes[predictions])
}

# Predict probabilities
predict_proba_naive_bayes <- function(model, X) {
  n_samples <- nrow(X)
  n_classes <- length(model$classes)
  
  log_probs <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Add log prior
    log_probs[, i] <- log(model$priors[i])
    
    if (model$variant == "gaussian") {
      log_probs[, i] <- log_probs[, i] + gaussian_log_likelihood(X, model$means[i,], model$variances[i,])
    } else if (model$variant == "multinomial") {
      log_probs[, i] <- log_probs[, i] + multinomial_log_likelihood(X, model$feature_probs[i,])
    } else if (model$variant == "bernoulli") {
      log_probs[, i] <- log_probs[, i] + bernoulli_log_likelihood(X, model$feature_probs[i,])
    }
  }
  
  # Convert log probabilities to probabilities
  probs <- exp(log_probs - apply(log_probs, 1, max))
  probs <- probs / rowSums(probs)
  
  return(probs)
}

# Gaussian log-likelihood
gaussian_log_likelihood <- function(X, means, variances) {
  # Gaussian log-likelihood
  log_likelihood <- -0.5 * rowSums(
    log(2 * pi * variances) + 
    (X - matrix(means, nrow = nrow(X), ncol = length(means), byrow = TRUE))^2 / variances
  )
  return(log_likelihood)
}

# Multinomial log-likelihood
multinomial_log_likelihood <- function(X, feature_probs) {
  # Multinomial log-likelihood
  log_likelihood <- rowSums(X * log(feature_probs + 1e-10))
  return(log_likelihood)
}

# Bernoulli log-likelihood
bernoulli_log_likelihood <- function(X, feature_probs) {
  # Bernoulli log-likelihood
  X_binary <- (X > 0) * 1.0
  log_likelihood <- rowSums(
    X_binary * log(feature_probs + 1e-10) + 
    (1 - X_binary) * log(1 - feature_probs + 1e-10)
  )
  return(log_likelihood)
}

# Generate synthetic data
generate_synthetic_data <- function(n_samples = 1000, n_features = 2, n_classes = 3, seed = 42) {
  set.seed(seed)
  
  # Generate class means
  means <- matrix(rnorm(n_classes * n_features, 0, 2), nrow = n_classes)
  
  # Generate diagonal covariance matrices (independent features)
  covariances <- list()
  for (i in 1:n_classes) {
    # Create diagonal covariance matrix
    cov <- diag(runif(n_features, 0.5, 2.0))
    covariances[[i]] <- cov
  }
  
  # Generate samples
  X <- matrix(0, nrow = n_samples, ncol = n_features)
  y <- rep(0, n_samples)
  samples_per_class <- n_samples %/% n_classes
  
  for (i in 1:n_classes) {
    start_idx <- (i-1) * samples_per_class + 1
    end_idx <- i * samples_per_class
    class_samples <- MASS::mvrnorm(samples_per_class, means[i,], covariances[[i]])
    X[start_idx:end_idx,] <- class_samples
    y[start_idx:end_idx] <- i - 1
  }
  
  return(list(X = X, y = y))
}

# Demonstrate Naive Bayes
demonstrate_naive_bayes <- function() {
  # Generate data
  data <- generate_synthetic_data(n_samples = 900, n_features = 2, n_classes = 3)
  X <- data$X
  y <- data$y
  
  # Split data
  set.seed(42)
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Fit our implementation
  nb_scratch <- naive_bayes_classifier(X_train, y_train, variant = "gaussian")
  
  # Fit e1071 implementation
  nb_e1071 <- naiveBayes(X_train, factor(y_train))
  
  # Predict
  y_pred_scratch <- predict_naive_bayes(nb_scratch, X_test)
  y_pred_e1071 <- predict(nb_e1071, X_test)
  
  # Calculate accuracy
  accuracy_scratch <- mean(y_pred_scratch == y_test)
  accuracy_e1071 <- mean(y_pred_e1071 == y_test)
  
  cat("Accuracy Comparison:\n")
  cat("Our Implementation:", round(accuracy_scratch, 4), "\n")
  cat("e1071 Implementation:", round(accuracy_e1071, 4), "\n")
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  # Decision boundaries (simplified - using test points)
  x_min <- min(X[,1]) - 1
  x_max <- max(X[,1]) + 1
  y_min <- min(X[,2]) - 1
  y_max <- max(X[,2]) + 1
  
  grid_points <- expand.grid(
    x1 = seq(x_min, x_max, length.out = 50),
    x2 = seq(y_min, y_max, length.out = 50)
  )
  
  # Predict on grid
  grid_pred <- predict_naive_bayes(nb_scratch, as.matrix(grid_points))
  grid_points$prediction <- factor(grid_pred)
  
  # Plot original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.6) +
    labs(title = "Original Data", color = "Class") +
    theme_minimal()
  
  # Plot decision boundaries
  p2 <- ggplot() +
    geom_tile(data = grid_points, aes(x = x1, y = x2, fill = prediction), alpha = 0.3) +
    geom_point(data = df_original, aes(x = x1, y = x2, color = class), alpha = 0.6) +
    labs(title = "Decision Boundaries", color = "Class", fill = "Prediction") +
    theme_minimal()
  
  # Feature importance
  feature_importance <- apply(nb_scratch$means, 2, var) / colMeans(nb_scratch$variances)
  importance_df <- data.frame(
    feature = 1:length(feature_importance),
    importance = feature_importance
  )
  
  p3 <- ggplot(importance_df, aes(x = feature, y = importance)) +
    geom_bar(stat = "identity") +
    labs(title = "Feature Importance (Variance Ratio)", x = "Feature", y = "Importance") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(nb_scratch = nb_scratch, nb_e1071 = nb_e1071))
}

# Demonstrate numerical issues
demonstrate_numerical_issues_r <- function() {
  set.seed(42)
  
  # Generate data with one class far from others
  # Class 0: centered at (0, 0)
  X0 <- MASS::mvrnorm(100, mu = c(0, 0), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
  
  # Class 1: centered at (10, 10) - far from class 0
  X1 <- MASS::mvrnorm(100, mu = c(10, 10), Sigma = matrix(c(1, 0, 0, 1), nrow = 2))
  
  X <- rbind(X0, X1)
  y <- rep(c(0, 1), each = 100)
  
  # Test point far from both classes
  test_point <- matrix(c(20, 20), nrow = 1)
  
  # Fit Naive Bayes
  nb <- naive_bayes_classifier(X, y, variant = "gaussian")
  
  # Calculate probabilities using different methods
  cat("Numerical Stability Demonstration:\n")
  cat("-" * 50, "\n")
  
  # Method 1: Direct probability calculation (problematic)
  means <- nb$means
  variances <- nb$variances
  
  # Calculate Gaussian PDF directly
  pdf_values <- rep(0, length(nb$classes))
  for (i in 1:length(nb$classes)) {
    pdf <- 1.0
    for (j in 1:ncol(test_point)) {
      pdf <- pdf * dnorm(test_point[1, j], means[i, j], sqrt(variances[i, j]))
    }
    pdf_values[i] <- pdf
  }
  
  cat("Direct PDF values:", pdf_values, "\n")
  cat("Direct probabilities:", pdf_values / sum(pdf_values), "\n")
  
  # Method 2: Log-probability calculation (stable)
  log_probs <- predict_proba_naive_bayes(nb, test_point)
  probs <- exp(log_probs - max(log_probs))
  probs <- probs / sum(probs)
  
  cat("Log-probability approach:", probs, "\n")
  
  # Visualize the issue
  df_data <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  df_test <- data.frame(
    x1 = test_point[1,1],
    x2 = test_point[1,2],
    class = "Test"
  )
  
  # Plot data and test point
  p1 <- ggplot() +
    geom_point(data = df_data, aes(x = x1, y = x2, color = class), alpha = 0.7) +
    geom_point(data = df_test, aes(x = x1, y = x2), color = "red", size = 3, shape = 4) +
    labs(title = "Data and Test Point", color = "Class") +
    theme_minimal()
  
  # Plot PDF vs distance
  distances <- seq(0, 30, length.out = 100)
  pdf_at_distance <- dnorm(distances, 0, 1)
  log_pdf_at_distance <- dnorm(distances, 0, 1, log = TRUE)
  
  df_pdf <- data.frame(
    distance = distances,
    pdf = pdf_at_distance,
    log_pdf = log_pdf_at_distance
  )
  
  p2 <- ggplot(df_pdf, aes(x = distance)) +
    geom_line(aes(y = pdf), color = "blue", alpha = 0.7) +
    geom_line(aes(y = exp(log_pdf)), color = "red", linetype = "dashed", alpha = 0.7) +
    labs(title = "PDF vs Distance (Numerical Underflow)", 
         x = "Distance from Mean", y = "PDF Value") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(nb = nb, pdf_values = pdf_values, probs = probs))
}

# Safe prediction using log-probabilities
safe_naive_bayes_predict <- function(X, nb_model) {
  log_probs <- predict_proba_naive_bayes(nb_model, X)
  return(nb_model$classes[apply(log_probs, 1, which.max)])
}

# Regularized Naive Bayes
regularized_naive_bayes <- function(X, y, alpha = 1e-10) {
  nb <- naive_bayes_classifier(X, y, variant = "gaussian", alpha = alpha)
  return(nb)
}

# Truncated Naive Bayes (not recommended)
truncated_naive_bayes <- function(X, y, threshold = 1e-10) {
  nb <- naive_bayes_classifier(X, y, variant = "gaussian")
  
  # Override predict function to truncate probabilities
  predict_truncated <- function(X_new) {
    probs <- predict_proba_naive_bayes(nb, X_new)
    # Truncate very small probabilities
    probs <- pmax(probs, threshold)
    probs <- probs / rowSums(probs)
    return(nb$classes[apply(probs, 1, which.max)])
  }
  
  nb$predict <- predict_truncated
  return(nb)
}

# Text classification example
text_classification_example_r <- function() {
  # Sample text data
  texts <- c(
    "I love this movie, it's amazing!",
    "This is the worst film I've ever seen",
    "Great acting and wonderful story",
    "Terrible plot, boring characters",
    "Fantastic cinematography and direction",
    "Awful script and poor acting",
    "Beautiful and inspiring movie",
    "Disappointing and waste of time",
    "Excellent performance by all actors",
    "Horrible waste of money"
  )
  
  labels <- rep(c(1, 0), each = 5)  # 1: positive, 0: negative
  
  # Simple text vectorization (word presence)
  words <- unique(unlist(strsplit(tolower(paste(texts, collapse = " ")), "\\s+")))
  words <- words[!words %in% c("", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by")]
  
  # Create feature matrix
  X <- matrix(0, nrow = length(texts), ncol = length(words))
  colnames(X) <- words
  
  for (i in 1:length(texts)) {
    text_words <- strsplit(tolower(texts[i]), "\\s+")[[1]]
    for (j in 1:length(words)) {
      X[i, j] <- sum(text_words == words[j])
    }
  }
  
  # Split data
  set.seed(42)
  train_idx <- createDataPartition(labels, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- labels[train_idx]
  y_test <- labels[-train_idx]
  
  # Fit Multinomial Naive Bayes
  nb <- naive_bayes_classifier(X_train, y_train, variant = "multinomial", alpha = 1.0)
  
  # Predict
  y_pred <- predict_naive_bayes(nb, X_test)
  
  cat("Text Classification Results:\n")
  cat("-" * 40, "\n")
  cat("Accuracy:", round(mean(y_pred == y_test), 4), "\n")
  
  # Feature importance (most discriminative words)
  feature_importance <- abs(nb$feature_probs[2,] - nb$feature_probs[1,])
  
  # Get top discriminative words
  top_indices <- order(feature_importance, decreasing = TRUE)[1:10]
  
  cat("\nTop Discriminative Words:\n")
  for (i in 1:length(top_indices)) {
    idx <- top_indices[i]
    word <- words[idx]
    importance <- feature_importance[idx]
    pos_prob <- nb$feature_probs[2, idx]
    neg_prob <- nb$feature_probs[1, idx]
    cat(sprintf("%s: %.4f (pos: %.4f, neg: %.4f)\n", word, importance, pos_prob, neg_prob))
  }
  
  # Visualize feature importance
  importance_df <- data.frame(
    word = words[top_indices],
    importance = feature_importance[top_indices]
  )
  
  p <- ggplot(importance_df, aes(x = reorder(word, importance), y = importance)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Most Discriminative Words for Sentiment Analysis",
         x = "Word", y = "Discriminative Power") +
    theme_minimal()
  
  print(p)
  
  return(list(nb = nb, words = words))
}

# Medical diagnosis example
medical_diagnosis_example_r <- function() {
  set.seed(42)
  n_samples <- 200
  
  # Generate synthetic medical features
  # Feature 0: Age (normalized)
  age <- rnorm(n_samples, 0, 1)
  
  # Feature 1: Blood pressure (normalized)
  bp <- rnorm(n_samples, 0, 1)
  
  # Feature 2: Cholesterol level (normalized)
  cholesterol <- rnorm(n_samples, 0, 1)
  
  # Feature 3: BMI (normalized)
  bmi <- rnorm(n_samples, 0, 1)
  
  X <- cbind(age, bp, cholesterol, bmi)
  
  # Generate disease labels based on features
  # Higher values of features increase disease probability
  disease_prob <- 1 / (1 + exp(-(0.3*age + 0.5*bp + 0.4*cholesterol + 0.2*bmi)))
  y <- rbinom(n_samples, 1, disease_prob)
  
  # Split data
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Fit Gaussian Naive Bayes
  nb <- naive_bayes_classifier(X_train, y_train, variant = "gaussian")
  
  # Predict
  y_pred <- predict_naive_bayes(nb, X_test)
  y_proba <- predict_proba_naive_bayes(nb, X_test)
  
  cat("Medical Diagnosis Results:\n")
  cat("-" * 40, "\n")
  cat("Accuracy:", round(mean(y_pred == y_test), 4), "\n")
  
  # Feature importance analysis
  feature_names <- c("Age", "Blood Pressure", "Cholesterol", "BMI")
  feature_importance <- apply(nb$means, 2, var) / colMeans(nb$variances)
  
  cat("\nFeature Importance (Variance Ratio):\n")
  for (i in 1:length(feature_names)) {
    cat(sprintf("%s: %.4f\n", feature_names[i], feature_importance[i]))
  }
  
  # Visualize results
  df_data <- data.frame(
    age = X[,1],
    bp = X[,2],
    cholesterol = X[,3],
    bmi = X[,4],
    class = factor(y)
  )
  
  # Feature distributions by class
  p1 <- ggplot(df_data, aes(x = age, fill = class)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 20) +
    labs(title = "Age Distribution", x = "Age", y = "Frequency") +
    theme_minimal()
  
  p2 <- ggplot(df_data, aes(x = bp, fill = class)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 20) +
    labs(title = "Blood Pressure Distribution", x = "Blood Pressure", y = "Frequency") +
    theme_minimal()
  
  p3 <- ggplot(df_data, aes(x = cholesterol, fill = class)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 20) +
    labs(title = "Cholesterol Distribution", x = "Cholesterol", y = "Frequency") +
    theme_minimal()
  
  p4 <- ggplot(df_data, aes(x = bmi, fill = class)) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 20) +
    labs(title = "BMI Distribution", x = "BMI", y = "Frequency") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  # ROC curve for probability predictions
  roc_obj <- roc(y_test, y_proba[,2])
  auc_val <- auc(roc_obj)
  
  plot(roc_obj, main = paste("ROC Curve for Medical Diagnosis (AUC =", round(auc_val, 3), ")"),
       col = "darkorange", lwd = 2)
  abline(a = 0, b = 1, col = "navy", lwd = 2, lty = 2)
  grid()
  
  return(list(nb = nb, feature_importance = feature_importance))
}

# Compare Naive Bayes variants
compare_naive_bayes_variants_r <- function() {
  set.seed(42)
  
  # Generate different types of data
  # 1. Continuous data (Gaussian)
  data_gaussian <- generate_synthetic_data(n_samples = 300, n_features = 2, n_classes = 2)
  X_gaussian <- data_gaussian$X
  y_gaussian <- data_gaussian$y
  
  # 2. Count data (Multinomial)
  X_multinomial <- matrix(rpois(300 * 10, 5), nrow = 300)
  y_multinomial <- sample(0:1, 300, replace = TRUE)
  
  # 3. Binary data (Bernoulli)
  X_bernoulli <- matrix(rbinom(300 * 10, 1, 0.3), nrow = 300)
  y_bernoulli <- sample(0:1, 300, replace = TRUE)
  
  # Test different variants
  variants <- c("gaussian", "multinomial", "bernoulli")
  datasets <- list(
    list(X = X_gaussian, y = y_gaussian, name = "Gaussian"),
    list(X = X_multinomial, y = y_multinomial, name = "Multinomial"),
    list(X = X_bernoulli, y = y_bernoulli, name = "Bernoulli")
  )
  
  results <- data.frame(
    variant = character(),
    dataset = character(),
    accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in 1:length(variants)) {
    for (j in 1:length(datasets)) {
      variant <- variants[i]
      dataset <- datasets[[j]]
      
      # Split data
      train_idx <- createDataPartition(dataset$y, p = 0.7, list = FALSE)
      X_train <- dataset$X[train_idx,]
      X_test <- dataset$X[-train_idx,]
      y_train <- dataset$y[train_idx]
      y_test <- dataset$y[-train_idx]
      
      # Fit and predict
      nb <- naive_bayes_classifier(X_train, y_train, variant = variant)
      y_pred <- predict_naive_bayes(nb, X_test)
      accuracy <- mean(y_pred == y_test)
      
      results <- rbind(results, data.frame(
        variant = variant,
        dataset = dataset$name,
        accuracy = accuracy,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Display results
  cat("Naive Bayes Variants Comparison:\n")
  cat("-" * 50, "\n")
  for (i in 1:nrow(results)) {
    cat(sprintf("%s on %s: %.4f\n", results$variant[i], results$dataset[i], results$accuracy[i]))
  }
  
  # Visualize results
  p <- ggplot(results, aes(x = paste(variant, "on", dataset), y = accuracy)) +
    geom_bar(stat = "identity") +
    labs(title = "Naive Bayes Variants Performance Comparison",
         x = "Model-Dataset Combination", y = "Accuracy") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0, 1)
  
  print(p)
  
  return(results)
}

# Analyze feature independence
analyze_feature_independence_r <- function(X, y) {
  n_features <- ncol(X)
  correlations <- matrix(0, nrow = n_features, ncol = n_features)
  
  # Calculate correlations for each class
  classes <- unique(y)
  
  for (c in classes) {
    class_mask <- y == c
    class_data <- X[class_mask,, drop = FALSE]
    class_corr <- cor(class_data)
    correlations <- correlations + class_corr
  }
  
  correlations <- correlations / length(classes)
  
  # Visualize correlation matrix
  corr_df <- data.frame(
    Var1 = rep(1:n_features, each = n_features),
    Var2 = rep(1:n_features, times = n_features),
    Correlation = as.vector(correlations)
  )
  
  p <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                        midpoint = 0, limit = c(-1, 1)) +
    labs(title = "Feature Correlation Matrix (Averaged over Classes)",
         x = "Feature", y = "Feature") +
    theme_minimal() +
    theme(axis.text = element_text(size = 8))
  
  print(p)
  
  # Calculate average absolute correlation (excluding diagonal)
  mask <- !diag(n_features)
  avg_correlation <- mean(abs(correlations[mask]))
  
  cat("Average absolute correlation:", round(avg_correlation, 4), "\n")
  cat("Correlation interpretation:\n")
  if (avg_correlation < 0.1) {
    cat("Features are approximately independent (good for Naive Bayes)\n")
  } else if (avg_correlation < 0.3) {
    cat("Features have moderate correlation (acceptable for Naive Bayes)\n")
  } else {
    cat("Features are highly correlated (may affect Naive Bayes performance)\n")
  }
  
  return(list(correlations = correlations, avg_correlation = avg_correlation))
}

# Main function to demonstrate Naive Bayes implementation
main <- function() {
  cat("Naive Bayes Classifier Demonstration\n")
  cat("=", 50, "\n")
  
  # Basic demonstration
  cat("\n1. Basic Naive Bayes Demonstration:\n")
  nb_demo <- demonstrate_naive_bayes()
  
  # Numerical stability demonstration
  cat("\n2. Numerical Stability Issues:\n")
  nb_numerical <- demonstrate_numerical_issues_r()
  
  # Text classification example
  cat("\n3. Text Classification Example:\n")
  nb_text <- text_classification_example_r()
  
  # Medical diagnosis example
  cat("\n4. Medical Diagnosis Example:\n")
  nb_medical <- medical_diagnosis_example_r()
  
  # Compare variants
  cat("\n5. Naive Bayes Variants Comparison:\n")
  results <- compare_naive_bayes_variants_r()
  
  # Analyze feature independence
  cat("\n6. Feature Independence Analysis:\n")
  data_indep <- generate_synthetic_data(n_samples = 500, n_features = 4, n_classes = 3)
  correlations <- analyze_feature_independence_r(data_indep$X, data_indep$y)
  
  # Cross-validation comparison
  cat("\n7. Cross-Validation Comparison:\n")
  data_cv <- generate_synthetic_data(n_samples = 300, n_features = 2, n_classes = 2)
  
  # Our implementation
  nb_scratch <- naive_bayes_classifier(data_cv$X, data_cv$y, variant = "gaussian")
  
  # e1071 implementation
  nb_e1071 <- naiveBayes(data_cv$X, factor(data_cv$y))
  
  # Simple cross-validation
  set.seed(42)
  folds <- createFolds(data_cv$y, k = 5)
  cv_scratch <- numeric(5)
  cv_e1071 <- numeric(5)
  
  for (i in 1:5) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    X_train <- data_cv$X[train_idx,, drop = FALSE]
    X_test <- data_cv$X[test_idx,, drop = FALSE]
    y_train <- data_cv$y[train_idx]
    y_test <- data_cv$y[test_idx]
    
    # Our implementation
    nb_scratch_cv <- naive_bayes_classifier(X_train, y_train, variant = "gaussian")
    y_pred_scratch <- predict_naive_bayes(nb_scratch_cv, X_test)
    cv_scratch[i] <- mean(y_pred_scratch == y_test)
    
    # e1071 implementation
    nb_e1071_cv <- naiveBayes(X_train, factor(y_train))
    y_pred_e1071 <- predict(nb_e1071_cv, X_test)
    cv_e1071[i] <- mean(y_pred_e1071 == y_test)
  }
  
  cat("Our Implementation CV Score:", round(mean(cv_scratch), 4), 
      "(+/-", round(sd(cv_scratch) * 2, 4), ")\n")
  cat("e1071 Implementation CV Score:", round(mean(cv_e1071), 4), 
      "(+/-", round(sd(cv_e1071) * 2, 4), ")\n")
  
  return(list(
    nb_scratch = nb_scratch,
    nb_e1071 = nb_e1071,
    nb_text = nb_text,
    nb_medical = nb_medical,
    results = results,
    correlations = correlations,
    cv_scratch = cv_scratch,
    cv_e1071 = cv_e1071
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
