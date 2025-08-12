# R implementation of Quadratic Discriminant Analysis
library(MASS)
library(ggplot2)
library(caret)
library(e1071)
library(pROC)

# Create synthetic data for QDA demonstration
create_qda_demo_data <- function(n_samples = 1000, random_state = 42) {
  set.seed(random_state)
  
  # Three classes with different means and covariances
  means <- list(
    c(0, 0),
    c(3, 3),
    c(-2, 2)
  )
  
  covs <- list(
    matrix(c(1, 0.5, 0.5, 1), nrow = 2),      # Positive correlation
    matrix(c(1, -0.5, -0.5, 1), nrow = 2),    # Negative correlation
    matrix(c(0.5, 0, 0, 2), nrow = 2)         # Different variances
  )
  
  X_list <- list()
  y_list <- list()
  
  for (k in 1:3) {
    n_k <- n_samples %/% 3
    X_k <- mvrnorm(n_k, mu = means[[k]], Sigma = covs[[k]])
    X_list[[k]] <- X_k
    y_list[[k]] <- rep(k - 1, n_k)
  }
  
  X <- do.call(rbind, X_list)
  y <- factor(unlist(y_list))
  
  return(list(X = X, y = y))
}

# Custom QDA implementation
custom_qda <- function(X, y, regularization = 1e-6) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_features <- ncol(X)
  n_samples <- length(y)
  
  # Initialize storage
  priors <- numeric(n_classes)
  means <- matrix(0, nrow = n_classes, ncol = n_features)
  covariances <- array(0, dim = c(n_features, n_features, n_classes))
  
  # Estimate parameters for each class
  for (i in 1:n_classes) {
    k <- classes[i]
    class_mask <- y == k
    X_k <- X[class_mask, , drop = FALSE]
    n_k <- sum(class_mask)
    
    # Estimate prior
    priors[i] <- n_k / n_samples
    
    # Estimate mean
    means[i, ] <- colMeans(X_k)
    
    # Estimate covariance with regularization
    if (n_k > 1) {
      cov_k <- cov(X_k)
    } else {
      cov_k <- matrix(0, n_features, n_features)
    }
    covariances[, , i] <- cov_k + regularization * diag(n_features)
  }
  
  # Return model parameters
  return(list(
    classes = classes,
    priors = priors,
    means = means,
    covariances = covariances,
    n_features = n_features
  ))
}

# Predict function for custom QDA
predict_custom_qda <- function(model, X_new) {
  n_samples <- nrow(X_new)
  n_classes <- length(model$classes)
  
  # Compute discriminant function for each class
  discriminant_values <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Compute Mahalanobis distance
    diff <- t(X_new) - model$means[i, ]
    inv_cov <- solve(model$covariances[, , i])
    mahal_dist <- colSums(diff * (inv_cov %*% diff))
    
    # Compute log determinant
    log_det <- log(det(model$covariances[, , i]))
    
    # Compute prior term
    prior_term <- -2 * log(model$priors[i])
    
    # Combine all terms
    discriminant_values[, i] <- mahal_dist + log_det + prior_term
  }
  
  # Return predicted classes
  predicted_classes <- model$classes[apply(discriminant_values, 1, which.min)]
  return(predicted_classes)
}

# Predict probabilities for custom QDA
predict_proba_custom_qda <- function(model, X_new) {
  n_samples <- nrow(X_new)
  n_classes <- length(model$classes)
  
  # Compute discriminant function for each class
  discriminant_values <- matrix(0, nrow = n_samples, ncol = n_classes)
  
  for (i in 1:n_classes) {
    # Compute Mahalanobis distance
    diff <- t(X_new) - model$means[i, ]
    inv_cov <- solve(model$covariances[, , i])
    mahal_dist <- colSums(diff * (inv_cov %*% diff))
    
    # Compute log determinant
    log_det <- log(det(model$covariances[, , i]))
    
    # Compute prior term
    prior_term <- -2 * log(model$priors[i])
    
    # Combine all terms
    discriminant_values[, i] <- mahal_dist + log_det + prior_term
  }
  
  # Convert to probabilities using softmax
  # Subtract minimum for numerical stability
  discriminant_values <- discriminant_values - apply(discriminant_values, 1, min)
  exp_values <- exp(-0.5 * discriminant_values)
  probs <- exp_values / rowSums(exp_values)
  
  colnames(probs) <- model$classes
  return(probs)
}

# Function to plot QDA decision boundaries
plot_qda_decision_boundaries <- function(X, y, qda_model, title = "QDA Decision Boundaries") {
  # Create mesh grid
  x_range <- range(X[, 1])
  y_range <- range(X[, 2])
  x_grid <- seq(x_range[1] - 1, x_range[2] + 1, by = 0.02)
  y_grid <- seq(y_range[1] - 1, y_range[2] + 1, by = 0.02)
  grid_points <- expand.grid(x = x_grid, y = y_grid)
  
  # Predict on grid
  if (inherits(qda_model, "qda")) {
    # Use MASS qda
    predictions <- predict(qda_model, grid_points)
    Z <- as.numeric(predictions$class)
    Z_proba <- predictions$posterior[, 2]  # Probability of second class
  } else {
    # Use custom QDA
    Z <- as.numeric(predict_custom_qda(qda_model, as.matrix(grid_points)))
    Z_proba <- predict_proba_custom_qda(qda_model, as.matrix(grid_points))[, 2]
  }
  
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
  gridExtra::grid.arrange(p1, p2, ncol = 2)
}

# Function to compare QDA and LDA decision boundaries
compare_qda_lda_boundaries <- function(X_train, y_train, X_test, y_test) {
  # Fit both models
  qda_model <- qda(X_train, y_train)
  lda_model <- lda(X_train, y_train)
  
  # Create mesh grid
  x_range <- range(X_test[, 1])
  y_range <- range(X_test[, 2])
  x_grid <- seq(x_range[1] - 1, x_range[2] + 1, by = 0.02)
  y_grid <- seq(y_range[1] - 1, y_range[2] + 1, by = 0.02)
  grid_points <- expand.grid(x = x_grid, y = y_grid)
  
  # Predict on grid for both models
  qda_pred <- predict(qda_model, grid_points)
  lda_pred <- predict(lda_model, grid_points)
  
  # Create data frames
  grid_df <- data.frame(
    x = grid_points$x,
    y = grid_points$y,
    qda_class = as.numeric(qda_pred$class),
    lda_class = as.numeric(lda_pred$class),
    qda_proba = qda_pred$posterior[, 2],
    lda_proba = lda_pred$posterior[, 2]
  )
  
  data_df <- data.frame(
    x = X_test[, 1],
    y = X_test[, 2],
    class = factor(y_test)
  )
  
  # Create plots
  p1 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = qda_class), alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(title = "QDA Decision Boundaries", x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  p2 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = lda_class), alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(title = "LDA Decision Boundaries", x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  p3 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = qda_proba), alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.5) +
    labs(title = "QDA Posterior Probabilities", x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  p4 <- ggplot() +
    geom_contour_filled(data = grid_df, aes(x = x, y = y, z = lda_proba), alpha = 0.4) +
    geom_point(data = data_df, aes(x = x, y = y, color = class), alpha = 0.8) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.5) +
    labs(title = "LDA Posterior Probabilities", x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  # Combine plots
  gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
  
  # Print accuracies
  qda_accuracy <- mean(predict(qda_model, X_test)$class == y_test)
  lda_accuracy <- mean(predict(lda_model, X_test)$class == y_test)
  
  cat("Model Comparison:\n")
  cat("QDA Accuracy:", round(qda_accuracy, 3), "\n")
  cat("LDA Accuracy:", round(lda_accuracy, 3), "\n")
}

# Function to analyze QDA parameters
analyze_qda_parameters <- function(qda_model, feature_names = NULL) {
  if (is.null(feature_names)) {
    feature_names <- paste0("Feature_", 1:qda_model$n_features)
  }
  
  n_classes <- length(qda_model$classes)
  n_features <- length(feature_names)
  
  # Print parameter summary
  cat("QDA Model Parameters:\n")
  cat("=", 50, "\n")
  
  for (i in 1:n_classes) {
    k <- qda_model$classes[i]
    cat("\nClass", k, ":\n")
    cat("  Prior Probability:", round(qda_model$priors[i], 3), "\n")
    cat("  Mean Vector:", round(qda_model$means[i, ], 3), "\n")
    cat("  Covariance Matrix:\n")
    print(round(qda_model$covariances[, , i], 3))
    cat("  Log Determinant:", round(log(det(qda_model$covariances[, , i])), 3), "\n")
  }
  
  # Create visualizations
  # Class priors
  priors_df <- data.frame(
    class = factor(qda_model$classes),
    prior = qda_model$priors
  )
  
  p1 <- ggplot(priors_df, aes(x = class, y = prior)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Class Prior Probabilities", x = "Class", y = "Prior Probability") +
    theme_minimal()
  
  # Class means
  means_df <- data.frame(
    class = rep(qda_model$classes, each = n_features),
    feature = rep(feature_names, times = n_classes),
    mean = as.vector(t(qda_model$means))
  )
  
  p2 <- ggplot(means_df, aes(x = feature, y = mean, fill = factor(class))) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), alpha = 0.8) +
    labs(title = "Class Mean Vectors", x = "Feature", y = "Mean Value", fill = "Class") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Covariance matrices (heatmaps)
  cov_plots <- list()
  for (i in 1:n_classes) {
    cov_matrix <- qda_model$covariances[, , i]
    cov_df <- expand.grid(
      feature1 = feature_names,
      feature2 = feature_names
    )
    cov_df$value <- as.vector(cov_matrix)
    
    cov_plots[[i]] <- ggplot(cov_df, aes(x = feature1, y = feature2, fill = value)) +
      geom_tile() +
      geom_text(aes(label = round(value, 2)), size = 3) +
      scale_fill_viridis_c() +
      labs(title = paste("Covariance Matrix - Class", qda_model$classes[i]),
           x = "Feature", y = "Feature") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  
  # Combine plots
  if (n_classes == 2) {
    gridExtra::grid.arrange(p1, p2, cov_plots[[1]], cov_plots[[2]], ncol = 2, nrow = 2)
  } else {
    # For more classes, arrange differently
    gridExtra::grid.arrange(p1, p2, ncol = 2)
    for (i in 1:n_classes) {
      print(cov_plots[[i]])
    }
  }
}

# Function to analyze Mahalanobis distances
analyze_mahalanobis_distances <- function(X, y, qda_model) {
  n_classes <- length(qda_model$classes)
  
  # Create plots for each class
  dist_plots <- list()
  
  for (i in 1:n_classes) {
    k <- qda_model$classes[i]
    class_mask <- y == k
    X_k <- X[class_mask, , drop = FALSE]
    
    # Compute Mahalanobis distances
    diff <- t(X_k) - qda_model$means[i, ]
    inv_cov <- solve(qda_model$covariances[, , i])
    mahal_dist <- colSums(diff * (inv_cov %*% diff))
    
    # Create histogram
    dist_df <- data.frame(mahal_dist = mahal_dist)
    
    dist_plots[[i]] <- ggplot(dist_df, aes(x = mahal_dist)) +
      geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "black") +
      labs(title = paste("Mahalanobis Distances - Class", k),
           x = "Mahalanobis Distance", y = "Frequency") +
      theme_minimal()
    
    # Add theoretical chi-squared distribution
    df <- ncol(X)  # degrees of freedom = number of features
    x_chi2 <- seq(0, max(mahal_dist), length.out = 100)
    y_chi2 <- length(mahal_dist) * diff(x_chi2)[1] * dchisq(x_chi2, df)
    
    chi2_df <- data.frame(x = x_chi2, y = y_chi2)
    dist_plots[[i]] <- dist_plots[[i]] +
      geom_line(data = chi2_df, aes(x = x, y = y), color = "red", size = 1) +
      annotate("text", x = max(mahal_dist) * 0.7, y = max(y_chi2) * 0.8,
               label = paste("χ²(", df, ")"), color = "red")
  }
  
  # Combine plots
  if (n_classes == 1) {
    print(dist_plots[[1]])
  } else {
    do.call(gridExtra::grid.arrange, c(dist_plots, ncol = n_classes))
  }
}

# Function to test high-dimensional QDA
test_high_dimensional_qda <- function() {
  set.seed(42)
  
  # Generate high-dimensional data
  n_samples <- 200
  n_features <- 50
  n_classes <- 3
  
  # Create sparse covariance matrices
  means <- lapply(1:n_classes, function(k) rnorm(n_features))
  covs <- list()
  
  for (k in 1:n_classes) {
    # Create sparse precision matrix
    precision <- diag(n_features)
    for (i in 1:n_features) {
      for (j in (i+1):min(i+2, n_features)) {
        if (runif(1) < 0.3) {
          precision[i, j] <- precision[j, i] <- 0.5
        }
      }
    }
    covs[[k]] <- solve(precision)
  }
  
  # Generate data
  X_list <- list()
  y_list <- list()
  for (k in 1:n_classes) {
    n_k <- n_samples %/% n_classes
    X_k <- mvrnorm(n_k, mu = means[[k]], Sigma = covs[[k]])
    X_list[[k]] <- X_k
    y_list[[k]] <- rep(k - 1, n_k)
  }
  
  X <- do.call(rbind, X_list)
  y <- factor(unlist(y_list))
  
  # Split data
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  # Test different methods
  methods <- list(
    "MASS QDA" = function() qda(X_train, y_train),
    "Custom QDA" = function() custom_qda(X_train, y_train)
  )
  
  results <- list()
  for (name in names(methods)) {
    tryCatch({
      model <- methods[[name]]()
      if (name == "MASS QDA") {
        accuracy <- mean(predict(model, X_test)$class == y_test)
      } else {
        accuracy <- mean(predict_custom_qda(model, X_test) == y_test)
      }
      results[[name]] <- accuracy
      cat(name, ":", round(accuracy, 3), "\n")
    }, error = function(e) {
      cat(name, ": Failed -", e$message, "\n")
      results[[name]] <- 0
    })
  }
  
  return(results)
}

# Function for cross-validation
qda_cross_validation <- function(X, y, cv = 5) {
  # Use caret for cross-validation
  ctrl <- trainControl(method = "cv", number = cv)
  
  # Create data frame for training
  train_data <- data.frame(X = X, y = y)
  
  # Train QDA model
  qda_model <- train(y ~ ., data = train_data, method = "qda", trControl = ctrl)
  
  cat("QDA Cross-Validation Results:\n")
  cat("Mean CV Accuracy:", round(mean(qda_model$resample$Accuracy), 3), "\n")
  cat("CV Accuracy SD:", round(sd(qda_model$resample$Accuracy), 3), "\n")
  cat("Individual CV Scores:", round(qda_model$resample$Accuracy, 3), "\n")
  
  return(qda_model$resample$Accuracy)
}

# Function for Iris dataset example
qda_iris_example <- function() {
  # Load iris data
  data(iris)
  
  # Use only two classes for binary classification
  iris_binary <- iris[iris$Species != "virginica", ]
  iris_binary$Species <- droplevels(iris_binary$Species)
  
  # Split data
  train_index <- createDataPartition(iris_binary$Species, p = 0.7, list = FALSE)
  train_data <- iris_binary[train_index, ]
  test_data <- iris_binary[-train_index, ]
  
  # Fit QDA
  qda_model <- qda(Species ~ ., data = train_data)
  
  # Make predictions
  predictions <- predict(qda_model, test_data)
  accuracy <- mean(predictions$class == test_data$Species)
  
  cat("Iris Dataset QDA Accuracy:", round(accuracy, 3), "\n")
  
  # Confusion matrix
  cm <- confusionMatrix(predictions$class, test_data$Species)
  print(cm)
  
  return(list(model = qda_model, accuracy = accuracy))
}

# Function for credit risk assessment example
qda_credit_risk_example <- function() {
  set.seed(42)
  n_samples <- 1000
  
  # Generate synthetic credit data
  income <- rnorm(n_samples, 50000, 20000)
  credit_score <- rnorm(n_samples, 700, 100)
  debt_ratio <- rbeta(n_samples, 2, 5) * 2
  employment_years <- rexp(n_samples, 1/5)
  
  # Generate target based on features
  risk_score <- (0.3 * (income - 50000) / 20000 + 
                 0.4 * (credit_score - 700) / 100 + 
                 0.2 * (debt_ratio - 1) + 
                 0.1 * (employment_years - 5) / 5)
  
  risk_score <- risk_score + rnorm(n_samples, 0, 0.2)
  risk_level <- factor(ifelse(risk_score > 0, "High", "Low"))
  
  # Create data frame
  credit_data <- data.frame(
    income = income,
    credit_score = credit_score,
    debt_ratio = debt_ratio,
    employment_years = employment_years,
    risk_level = risk_level
  )
  
  # Split data
  train_index <- createDataPartition(credit_data$risk_level, p = 0.7, list = FALSE)
  train_data <- credit_data[train_index, ]
  test_data <- credit_data[-train_index, ]
  
  # Fit QDA
  qda_model <- qda(risk_level ~ ., data = train_data)
  
  # Make predictions
  predictions <- predict(qda_model, test_data)
  accuracy <- mean(predictions$class == test_data$risk_level)
  
  cat("Credit Risk Assessment Results:\n")
  cat("QDA Accuracy:", round(accuracy, 3), "\n")
  
  # Classification report
  cm <- confusionMatrix(predictions$class, test_data$risk_level)
  print(cm)
  
  # Feature analysis
  feature_names <- c("Income", "Credit Score", "Debt Ratio", "Employment Years")
  
  # Compare means between classes
  means_by_class <- aggregate(. ~ risk_level, data = train_data[, -5], FUN = mean)
  print("Feature Means by Class:")
  print(means_by_class)
  
  # Create visualization
  means_long <- tidyr::gather(means_by_class, feature, mean_value, -risk_level)
  
  p1 <- ggplot(means_long, aes(x = feature, y = mean_value, fill = risk_level)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), alpha = 0.8) +
    labs(title = "Feature Means by Class", x = "Features", y = "Mean Value", fill = "Risk Level") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p1)
  
  return(list(model = qda_model, accuracy = accuracy))
}

# Main function to demonstrate QDA implementation
main <- function() {
  cat("Quadratic Discriminant Analysis Demonstration\n")
  cat("=", 50, "\n")
  
  # Generate data
  data <- create_qda_demo_data()
  X <- data$X
  y <- data$y
  
  # Split data
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  # Fit QDA using MASS package
  qda_model <- qda(X_train, y_train)
  
  # Make predictions
  qda_predictions <- predict(qda_model, X_test)
  qda_pred_class <- qda_predictions$class
  qda_pred_proba <- qda_predictions$posterior
  
  # Evaluate performance
  accuracy <- mean(qda_pred_class == y_test)
  cat("QDA Accuracy:", round(accuracy, 3), "\n")
  
  # Confusion matrix
  confusion_matrix <- table(Predicted = qda_pred_class, Actual = y_test)
  cat("Confusion Matrix:\n")
  print(confusion_matrix)
  
  # Plot decision boundaries
  plot_qda_decision_boundaries(X_test, y_test, qda_model, "QDA")
  
  # Compare with LDA
  compare_qda_lda_boundaries(X_train, y_train, X_test, y_test)
  
  # Analyze parameters
  custom_model <- custom_qda(X_train, y_train)
  analyze_qda_parameters(custom_model)
  
  # Analyze Mahalanobis distances
  analyze_mahalanobis_distances(X_test, y_test, custom_model)
  
  # Test high-dimensional QDA
  cat("\nHigh-Dimensional QDA Test:\n")
  high_dim_results <- test_high_dimensional_qda()
  
  # Cross-validation
  cat("\nCross-Validation:\n")
  cv_scores <- qda_cross_validation(X, y, cv = 5)
  
  # Iris example
  cat("\nIris Dataset Example:\n")
  iris_result <- qda_iris_example()
  
  # Credit risk example
  cat("\nCredit Risk Assessment Example:\n")
  credit_result <- qda_credit_risk_example()
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
