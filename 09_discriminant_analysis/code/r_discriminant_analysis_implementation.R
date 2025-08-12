# Discriminant Analysis Implementation in R
# =======================================
#
# This script provides comprehensive implementations of discriminant analysis methods,
# including Bayes Classifier framework, QDA, LDA, Naive Bayes, and Fisher's Discriminant Analysis.

library(MASS)
library(ggplot2)
library(caret)
library(e1071)
library(klaR)

# Create synthetic Gaussian mixture data
create_gaussian_mixture_data <- function(n_samples = 1000, random_state = 42) {
  set.seed(random_state)
  
  # Three classes with different means and covariances
  means <- list(
    c(0, 0),
    c(3, 3),
    c(-2, 2)
  )
  
  covs <- list(
    matrix(c(1, 0.5, 0.5, 1), nrow = 2),
    matrix(c(1, -0.5, -0.5, 1), nrow = 2),
    matrix(c(0.5, 0, 0, 0.5), nrow = 2)
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

# Bayes Classifier Framework
BayesClassifier <- function() {
  # This is a conceptual framework - actual implementations are in specific methods
  list(
    fit = function(X, y) {
      # Base fitting function
      classes <- unique(y)
      n_classes <- length(classes)
      n_samples <- length(y)
      
      # Estimate class priors
      priors <- numeric(n_classes)
      for (i in 1:n_classes) {
        priors[i] <- sum(y == classes[i]) / n_samples
      }
      
      return(list(classes = classes, priors = priors))
    }
  )
}

# Quadratic Discriminant Analysis
QDA <- function(X, y) {
  # Use MASS package for QDA
  qda_model <- qda(X, y)
  return(qda_model)
}

# Linear Discriminant Analysis
LDA <- function(X, y) {
  # Use MASS package for LDA
  lda_model <- lda(X, y)
  return(lda_model)
}

# Gaussian Naive Bayes
GaussianNaiveBayes <- function(X, y) {
  # Use e1071 package for Naive Bayes
  nb_model <- naiveBayes(X, y)
  return(nb_model)
}

# Fisher's Discriminant Analysis
FisherDiscriminantAnalysis <- function(X, y, n_components = NULL) {
  # Use MASS package for LDA (which implements FDA)
  lda_model <- lda(X, y)
  
  # Extract discriminant functions
  n_classes <- length(unique(y))
  if (is.null(n_components)) {
    n_components <- min(n_classes - 1, ncol(X))
  }
  
  return(list(
    model = lda_model,
    n_components = n_components,
    eigenvalues = lda_model$svd^2,
    explained_variance_ratio = lda_model$svd^2 / sum(lda_model$svd^2)
  ))
}

# Visualization functions
plot_decision_boundaries <- function(X, y, model, title = "Decision Boundaries") {
  # Create grid for decision boundaries
  x_range <- range(X[, 1])
  y_range <- range(X[, 2])
  x_grid <- seq(x_range[1] - 0.5, x_range[2] + 0.5, length.out = 100)
  y_grid <- seq(y_range[1] - 0.5, y_range[2] + 0.5, length.out = 100)
  grid_points <- expand.grid(x = x_grid, y = y_grid)
  
  # Predict on grid
  if (inherits(model, "qda")) {
    grid_pred <- predict(model, grid_points)$class
  } else if (inherits(model, "lda")) {
    grid_pred <- predict(model, grid_points)$class
  } else {
    grid_pred <- predict(model, grid_points)
  }
  
  # Create plot
  p <- ggplot() +
    geom_tile(data = data.frame(grid_points, class = grid_pred), 
              aes(x = x, y = y, fill = class), alpha = 0.3) +
    geom_point(data = data.frame(X, class = y), 
               aes(x = X.1, y = X.2, color = class), size = 2) +
    labs(title = title, x = "Feature 1", y = "Feature 2") +
    theme_minimal()
  
  return(p)
}

# Model comparison function
compare_models <- function(X_train, y_train, X_test, y_test) {
  # Fit models
  qda_model <- QDA(X_train, y_train)
  lda_model <- LDA(X_train, y_train)
  nb_model <- GaussianNaiveBayes(X_train, y_train)
  
  # Make predictions
  qda_pred <- predict(qda_model, X_test)$class
  lda_pred <- predict(lda_model, X_test)$class
  nb_pred <- predict(nb_model, X_test)
  
  # Calculate accuracies
  qda_acc <- mean(qda_pred == y_test)
  lda_acc <- mean(lda_pred == y_test)
  nb_acc <- mean(nb_pred == y_test)
  
  # Results
  results <- data.frame(
    Model = c("QDA", "LDA", "Naive Bayes"),
    Accuracy = c(qda_acc, lda_acc, nb_acc)
  )
  
  print("Model Comparison Results:")
  print(results)
  
  return(list(
    results = results,
    models = list(qda = qda_model, lda = lda_model, nb = nb_model),
    predictions = list(qda = qda_pred, lda = lda_pred, nb = nb_pred)
  ))
}

# Regularized LDA
RegularizedLDA <- function(X, y, alpha = 0.1) {
  # Use shrinkage parameter in lda
  lda_model <- lda(X, y, method = "mle", nu = alpha)
  return(lda_model)
}

# Demonstration functions
demonstrate_bayes_classifier <- function() {
  # Create dataset
  data <- create_gaussian_mixture_data()
  X <- data$X
  y <- data$y
  
  # Split data
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  cat("Dataset shape:", nrow(X), "x", ncol(X), "\n")
  cat("Class distribution:\n")
  print(table(y) / length(y))
  
  return(list(
    X_train = X_train, X_test = X_test,
    y_train = y_train, y_test = y_test
  ))
}

demonstrate_qda <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Fit QDA
  qda_model <- QDA(data_split$X_train, data_split$y_train)
  
  # Make predictions
  qda_pred <- predict(qda_model, data_split$X_test)
  
  # Results
  accuracy <- mean(qda_pred$class == data_split$y_test)
  cat("QDA Results:\n")
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("\nClassification Report:\n")
  print(table(Predicted = qda_pred$class, Actual = data_split$y_test))
  
  # Plot decision boundaries
  p <- plot_decision_boundaries(data_split$X_test, data_split$y_test, 
                               qda_model, "QDA Decision Boundaries")
  print(p)
  
  return(qda_model)
}

demonstrate_lda <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Fit LDA
  lda_model <- LDA(data_split$X_train, data_split$y_train)
  
  # Make predictions
  lda_pred <- predict(lda_model, data_split$X_test)
  
  # Results
  accuracy <- mean(lda_pred$class == data_split$y_test)
  cat("LDA Results:\n")
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("\nClassification Report:\n")
  print(table(Predicted = lda_pred$class, Actual = data_split$y_test))
  
  # Plot decision boundaries
  p <- plot_decision_boundaries(data_split$X_test, data_split$y_test, 
                               lda_model, "LDA Decision Boundaries")
  print(p)
  
  return(lda_model)
}

demonstrate_naive_bayes <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Fit Naive Bayes
  nb_model <- GaussianNaiveBayes(data_split$X_train, data_split$y_train)
  
  # Make predictions
  nb_pred <- predict(nb_model, data_split$X_test)
  
  # Results
  accuracy <- mean(nb_pred == data_split$y_test)
  cat("Gaussian Naive Bayes Results:\n")
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("\nClassification Report:\n")
  print(table(Predicted = nb_pred, Actual = data_split$y_test))
  
  return(nb_model)
}

demonstrate_fda <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Apply FDA
  fda_model <- FisherDiscriminantAnalysis(data_split$X_train, data_split$y_train)
  
  # Transform data
  X_train_fda <- predict(fda_model$model, data_split$X_train)$x[, 1:fda_model$n_components]
  X_test_fda <- predict(fda_model$model, data_split$X_test)$x[, 1:fda_model$n_components]
  
  cat("FDA Results:\n")
  cat("Explained variance ratio:", round(fda_model$explained_variance_ratio, 4), "\n")
  cat("Eigenvalues:", round(fda_model$eigenvalues, 4), "\n")
  
  # Visualize FDA projection
  if (fda_model$n_components >= 2) {
    # Original data
    p1 <- ggplot(data.frame(data_split$X_test, class = data_split$y_test), 
                 aes(x = X.1, y = X.2, color = class)) +
      geom_point() +
      labs(title = "Original Data", x = "Feature 1", y = "Feature 2") +
      theme_minimal()
    
    # FDA projection
    p2 <- ggplot(data.frame(X_test_fda, class = data_split$y_test), 
                 aes(x = LD1, y = LD2, color = class)) +
      geom_point() +
      labs(title = "FDA Projection", x = "First Discriminant", y = "Second Discriminant") +
      theme_minimal()
    
    print(p1)
    print(p2)
  }
  
  # Apply LDA on FDA-transformed data
  lda_fda <- LDA(X_train_fda, data_split$y_train)
  lda_fda_pred <- predict(lda_fda, X_test_fda)
  fda_lda_accuracy <- mean(lda_fda_pred$class == data_split$y_test)
  
  cat("LDA on FDA-transformed data accuracy:", round(fda_lda_accuracy, 3), "\n")
  
  return(fda_model)
}

demonstrate_model_comparison <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Compare models
  comparison <- compare_models(data_split$X_train, data_split$y_train,
                              data_split$X_test, data_split$y_test)
  
  # Visualize results
  p <- ggplot(comparison$results, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.3f", Accuracy)), vjust = -0.5) +
    labs(title = "Model Accuracy Comparison", y = "Accuracy") +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p)
  
  return(comparison)
}

demonstrate_regularization <- function() {
  data_split <- demonstrate_bayes_classifier()
  
  # Test different regularization levels
  alphas <- c(0.1, 0.5, 0.9)
  
  for (alpha in alphas) {
    lda_reg <- RegularizedLDA(data_split$X_train, data_split$y_train, alpha)
    lda_reg_pred <- predict(lda_reg, data_split$X_test)
    accuracy <- mean(lda_reg_pred$class == data_split$y_test)
    cat("Regularized LDA (α =", alpha, ") accuracy:", round(accuracy, 3), "\n")
  }
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  cat("=== Bayes Classifier Framework ===\n")
  demonstrate_bayes_classifier()
  
  cat("\n=== QDA Demonstration ===\n")
  demonstrate_qda()
  
  cat("\n=== LDA Demonstration ===\n")
  demonstrate_lda()
  
  cat("\n=== Naive Bayes Demonstration ===\n")
  demonstrate_naive_bayes()
  
  cat("\n=== FDA Demonstration ===\n")
  demonstrate_fda()
  
  cat("\n=== Model Comparison ===\n")
  demonstrate_model_comparison()
  
  cat("\n=== Regularization ===\n")
  demonstrate_regularization()
}
