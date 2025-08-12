# Classification Implementation in R
# =================================
#
# This script provides comprehensive implementations of classification concepts,
# including data preprocessing, classification models, loss functions, optimization,
# Bayes optimal classifier, decision boundaries, evaluation metrics, and practical considerations.

library(ggplot2)
library(caret)
library(dplyr)
library(e1071)
library(randomForest)
library(pROC)
library(ROCR)

create_credit_dataset <- function(n_samples = 1000, random_state = 42) {
  """
  Create synthetic credit risk dataset.
  
  Parameters:
  -----------
  n_samples : integer
      Number of samples to generate
  random_state : integer
      Random seed for reproducibility
      
  Returns:
  --------
  data : list
      List containing X (features) and y (target)
  """
  set.seed(random_state)
  
  # Generate features
  income <- rnorm(n_samples, mean = 50000, sd = 20000)
  credit_score <- rnorm(n_samples, mean = 700, sd = 100)
  debt_ratio <- rbeta(n_samples, 2, 5) * 2
  employment_years <- rexp(n_samples, rate = 1/5)
  
  # Create feature matrix
  X <- data.frame(
    income = income,
    credit_score = credit_score,
    debt_ratio = debt_ratio,
    employment_years = employment_years
  )
  
  # Generate target
  risk_score <- 0.3 * (income - 50000) / 20000 + 
                0.4 * (credit_score - 700) / 100 + 
                0.2 * (debt_ratio - 1) + 
                0.1 * (employment_years - 5) / 5
  
  risk_score <- risk_score + rnorm(n_samples, 0, 0.2)
  y <- as.factor(ifelse(risk_score > 0, 1, 0))
  
  return(list(X = X, y = y))
}

demonstrate_data_preprocessing <- function() {
  """
  Demonstrate data preprocessing for classification.
  """
  # Create dataset
  data <- create_credit_dataset()
  X <- data$X
  y <- data$y
  
  # Split data
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  # Standardize features
  preprocess_params <- preProcess(X_train, method = c("center", "scale"))
  X_train_scaled <- predict(preprocess_params, X_train)
  X_test_scaled <- predict(preprocess_params, X_test)
  
  cat("Dataset shape:", nrow(X), "x", ncol(X), "\n")
  cat("Class distribution:\n")
  print(table(y) / length(y))
  
  list(X_train_scaled = X_train_scaled, 
       X_test_scaled = X_test_scaled, 
       y_train = y_train, 
       y_test = y_test)
}

ClassificationModels <- function() {
  """
  Collection of classification model implementations.
  """
  list()
}

linear_classifier <- function(X, w, b) {
  """
  Linear classifier: f(x) = sign(w^T x + b)
  
  Parameters:
  -----------
  X : data.frame
      Feature matrix
  w : numeric vector
      Weight vector
  b : numeric
      Bias term
      
  Returns:
  --------
  predictions : factor
      Binary predictions
  """
  scores <- as.matrix(X) %*% w + b
  return(as.factor(ifelse(scores > 0, 1, 0)))
}

logistic_classifier <- function(X, w, b) {
  """
  Logistic classifier: f(x) = 1 if P(Y=1|X) > 0.5
  
  Parameters:
  -----------
  X : data.frame
      Feature matrix
  w : numeric vector
      Weight vector
  b : numeric
      Bias term
      
  Returns:
  --------
  predictions : factor
      Binary predictions
  """
  scores <- as.matrix(X) %*% w + b
  probabilities <- 1 / (1 + exp(-scores))
  return(as.factor(ifelse(probabilities > 0.5, 1, 0)))
}

nearest_neighbor_classifier <- function(X_train, y_train, X_test, k = 1) {
  """
  k-NN classifier
  
  Parameters:
  -----------
  X_train : data.frame
      Training features
  y_train : factor
      Training labels
  X_test : data.frame
      Test features
  k : integer
      Number of neighbors
      
  Returns:
  --------
  predictions : factor
      Predictions
  """
  knn_model <- knn3(X_train, y_train, k = k)
  return(predict(knn_model, X_test, type = "class"))
}

decision_tree_classifier <- function(X_train, y_train, X_test, max_depth = 3) {
  """
  Decision tree classifier
  
  Parameters:
  -----------
  X_train : data.frame
      Training features
  y_train : factor
      Training labels
  X_test : data.frame
      Test features
  max_depth : integer
      Maximum tree depth
      
  Returns:
  --------
  predictions : factor
      Predictions
  """
  tree_model <- rpart(y_train ~ ., data = X_train, 
                      control = rpart.control(maxdepth = max_depth))
  return(predict(tree_model, X_test, type = "class"))
}

demonstrate_classification_models <- function() {
  """
  Demonstrate different classification models.
  """
  # Get preprocessed data
  data <- demonstrate_data_preprocessing()
  X_train_scaled <- data$X_train_scaled
  X_test_scaled <- data$X_test_scaled
  y_train <- data$y_train
  y_test <- data$y_test
  
  # Linear classifier
  w <- c(0.1, -0.2, 0.3, -0.1)
  b <- 0.5
  linear_predictions <- linear_classifier(X_test_scaled, w, b)
  
  # k-NN classifier
  knn_predictions <- nearest_neighbor_classifier(X_train_scaled, y_train, X_test_scaled, k = 5)
  
  # Decision tree classifier
  tree_predictions <- decision_tree_classifier(X_train_scaled, y_train, X_test_scaled)
  
  cat("Model Performance:\n")
  cat("Linear Classifier Accuracy:", mean(linear_predictions == y_test), "\n")
  cat("k-NN Classifier Accuracy:", mean(knn_predictions == y_test), "\n")
  cat("Decision Tree Accuracy:", mean(tree_predictions == y_test), "\n")
  
  list(linear_predictions = linear_predictions,
       knn_predictions = knn_predictions,
       tree_predictions = tree_predictions)
}

ClassificationLoss <- function() {
  """
  Collection of classification loss functions.
  """
  list()
}

zero_one_loss <- function(y_pred, y_true) {
  """
  0-1 Loss: L(f(x), y) = 0 if y = f(x), 1 otherwise
  
  Parameters:
  -----------
  y_pred : factor
      Predicted labels
  y_true : factor
      True labels
      
  Returns:
  --------
  loss : numeric
      Average loss
  """
  return(mean(y_pred != y_true))
}

hinge_loss <- function(scores, y_true) {
  """
  Hinge loss for SVM: L = max(0, 1 - y * score)
  
  Parameters:
  -----------
  scores : numeric vector
      Model scores
  y_true : factor
      True labels
      
  Returns:
  --------
  loss : numeric
      Average loss
  """
  y_true_binary <- 2 * as.numeric(as.character(y_true)) - 1  # Convert to {-1, 1}
  return(mean(pmax(0, 1 - y_true_binary * scores)))
}

logistic_loss <- function(scores, y_true) {
  """
  Logistic loss: L = -log(P(Y=y|X))
  
  Parameters:
  -----------
  scores : numeric vector
      Model scores
  y_true : factor
      True labels
      
  Returns:
  --------
  loss : numeric
      Average loss
  """
  probabilities <- 1 / (1 + exp(-scores))
  # Avoid log(0)
  probabilities <- pmax(pmin(probabilities, 1 - 1e-15), 1e-15)
  y_numeric <- as.numeric(as.character(y_true))
  return(-mean(y_numeric * log(probabilities) + 
               (1 - y_numeric) * log(1 - probabilities)))
}

demonstrate_loss_functions <- function() {
  """
  Demonstrate different loss functions.
  """
  # Get preprocessed data
  data <- demonstrate_data_preprocessing()
  X_test_scaled <- data$X_test_scaled
  y_test <- data$y_test
  
  # Calculate different losses
  w <- c(0.1, -0.2, 0.3, -0.1)
  b <- 0.5
  scores <- as.matrix(X_test_scaled) %*% w + b
  linear_predictions <- as.factor(ifelse(scores > 0, 1, 0))
  
  cat("Loss Function Values:\n")
  cat("0-1 Loss:", zero_one_loss(linear_predictions, y_test), "\n")
  cat("Hinge Loss:", hinge_loss(scores, y_test), "\n")
  cat("Logistic Loss:", logistic_loss(scores, y_test), "\n")
}

ClassificationOptimization <- function() {
  """
  Optimization methods for classification models.
  """
  list()
}

optimize_logistic_regression <- function(X_train, y_train, X_test, y_test) {
  """
  Optimize logistic regression using caret
  
  Parameters:
  -----------
  X_train : data.frame
      Training features
  y_train : factor
      Training labels
  X_test : data.frame
      Test features
  y_test : factor
      Test labels
      
  Returns:
  --------
  results : list
      Optimization results
  """
  # Train logistic regression
  lr_model <- train(X_train, y_train, method = "glm", 
                    trControl = trainControl(method = "cv", number = 5))
  
  # Predictions
  y_pred <- predict(lr_model, X_test)
  y_prob <- predict(lr_model, X_test, type = "prob")[, 2]
  
  # Evaluation
  accuracy <- mean(y_pred == y_test)
  cv_scores <- lr_model$results$Accuracy
  
  return(list(
    model = lr_model,
    predictions = y_pred,
    probabilities = y_prob,
    accuracy = accuracy,
    cv_mean = mean(cv_scores),
    cv_std = sd(cv_scores)
  ))
}

optimize_svm <- function(X_train, y_train, X_test, y_test) {
  """
  Optimize SVM classifier
  
  Parameters:
  -----------
  X_train : data.frame
      Training features
  y_train : factor
      Training labels
  X_test : data.frame
      Test features
  y_test : factor
      Test labels
      
  Returns:
  --------
  results : list
      Optimization results
  """
  # Train SVM
  svm_model <- train(X_train, y_train, method = "svmRadial",
                     trControl = trainControl(method = "cv", number = 5))
  
  y_pred <- predict(svm_model, X_test)
  y_prob <- predict(svm_model, X_test, type = "prob")[, 2]
  
  accuracy <- mean(y_pred == y_test)
  cv_scores <- svm_model$results$Accuracy
  
  return(list(
    model = svm_model,
    predictions = y_pred,
    probabilities = y_prob,
    accuracy = accuracy,
    cv_mean = mean(cv_scores),
    cv_std = sd(cv_scores)
  ))
}

optimize_random_forest <- function(X_train, y_train, X_test, y_test) {
  """
  Optimize random forest classifier
  
  Parameters:
  -----------
  X_train : data.frame
      Training features
  y_train : factor
      Training labels
  X_test : data.frame
      Test features
  y_test : factor
      Test labels
      
  Returns:
  --------
  results : list
      Optimization results
  """
  # Train random forest
  rf_model <- train(X_train, y_train, method = "rf",
                    trControl = trainControl(method = "cv", number = 5))
  
  y_pred <- predict(rf_model, X_test)
  y_prob <- predict(rf_model, X_test, type = "prob")[, 2]
  
  accuracy <- mean(y_pred == y_test)
  cv_scores <- rf_model$results$Accuracy
  
  return(list(
    model = rf_model,
    predictions = y_pred,
    probabilities = y_prob,
    accuracy = accuracy,
    cv_mean = mean(cv_scores),
    cv_std = sd(cv_scores)
  ))
}

demonstrate_optimization <- function() {
  """
  Demonstrate optimization of different classifiers.
  """
  # Get preprocessed data
  data <- demonstrate_data_preprocessing()
  X_train_scaled <- data$X_train_scaled
  X_test_scaled <- data$X_test_scaled
  y_train <- data$y_train
  y_test <- data$y_test
  
  # Optimize different classifiers
  lr_results <- optimize_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
  svm_results <- optimize_svm(X_train_scaled, y_train, X_test_scaled, y_test)
  rf_results <- optimize_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
  
  # Compare results
  cat("Model Comparison:\n")
  cat(sprintf("Logistic Regression: %.3f (CV: %.3f ± %.3f)\n", 
              lr_results$accuracy, lr_results$cv_mean, lr_results$cv_std))
  cat(sprintf("SVM: %.3f (CV: %.3f ± %.3f)\n", 
              svm_results$accuracy, svm_results$cv_mean, svm_results$cv_std))
  cat(sprintf("Random Forest: %.3f (CV: %.3f ± %.3f)\n", 
              rf_results$accuracy, rf_results$cv_mean, rf_results$cv_std))
  
  list(lr_results = lr_results, svm_results = svm_results, rf_results = rf_results)
}

BayesOptimalClassifier <- function() {
  """
  Bayes optimal classifier implementation.
  """
  list()
}

fit_bayes_optimal <- function(X, y) {
  """
  Estimate P(Y=1|X) using kernel density estimation
  
  Parameters:
  -----------
  X : data.frame
      Feature matrix
  y : factor
      Target labels
      
  Returns:
  --------
  classifier : list
      Fitted classifier object
  """
  # Separate data by class
  X_class_0 <- X[y == 0, ]
  X_class_1 <- X[y == 1, ]
  
  # Estimate class priors
  prior_0 <- nrow(X_class_0) / nrow(X)
  prior_1 <- nrow(X_class_1) / nrow(X)
  
  # For simplicity, use naive Bayes
  nb_model <- naiveBayes(y ~ ., data = data.frame(X, y = y))
  
  return(list(
    model = nb_model,
    prior_0 = prior_0,
    prior_1 = prior_1
  ))
}

predict_bayes_optimal <- function(classifier, X) {
  """
  Predict class labels using Bayes optimal rule
  
  Parameters:
  -----------
  classifier : list
      Fitted classifier object
  X : data.frame
      Feature matrix
      
  Returns:
  --------
  predictions : factor
      Binary predictions
  """
  probabilities <- predict(classifier$model, X, type = "raw")[, 2]
  return(as.factor(ifelse(probabilities >= 0.5, 1, 0)))
}

demonstrate_bayes_optimal <- function() {
  """
  Demonstrate Bayes optimal classifier.
  """
  # Get preprocessed data
  data <- demonstrate_data_preprocessing()
  X_train_scaled <- data$X_train_scaled
  X_test_scaled <- data$X_test_scaled
  y_train <- data$y_train
  y_test <- data$y_test
  
  # Fit Bayes optimal classifier
  bayes_classifier <- fit_bayes_optimal(X_train_scaled, y_train)
  bayes_predictions <- predict_bayes_optimal(bayes_classifier, X_test_scaled)
  
  cat("Bayes Optimal Classifier Accuracy:", mean(bayes_predictions == y_test), "\n")
  
  list(classifier = bayes_classifier, predictions = bayes_predictions)
}

create_multi_class_dataset <- function(n_samples = 1000, n_classes = 3) {
  """
  Create synthetic multi-class dataset
  
  Parameters:
  -----------
  n_samples : integer
      Number of samples
  n_classes : integer
      Number of classes
      
  Returns:
  --------
  data : list
      List containing X (features) and y (target)
  """
  set.seed(42)
  
  # Generate features from different Gaussian distributions
  X_class_0 <- MASS::mvrnorm(n_samples/3, mu = c(0, 0), 
                             Sigma = matrix(c(1, 0.5, 0.5, 1), 2, 2))
  X_class_1 <- MASS::mvrnorm(n_samples/3, mu = c(3, 3), 
                             Sigma = matrix(c(1, -0.5, -0.5, 1), 2, 2))
  X_class_2 <- MASS::mvrnorm(n_samples/3, mu = c(-2, 2), 
                             Sigma = matrix(c(0.5, 0, 0, 0.5), 2, 2))
  
  X <- rbind(X_class_0, X_class_1, X_class_2)
  X <- as.data.frame(X)
  colnames(X) <- c("feature1", "feature2")
  
  y <- factor(rep(0:2, each = n_samples/3))
  
  return(list(X = X, y = y))
}

demonstrate_multi_class <- function() {
  """
  Demonstrate multi-class classification.
  """
  # Multi-class example
  data <- create_multi_class_dataset()
  X <- data$X
  y <- data$y
  
  # Split data
  train_index <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  # Train naive Bayes
  nb_model <- naiveBayes(y_train ~ ., data = data.frame(X_train, y_train = y_train))
  multi_predictions <- predict(nb_model, X_test)
  
  cat("Multi-class Bayes Optimal Accuracy:", mean(multi_predictions == y_test), "\n")
  
  list(predictions = multi_predictions, model = nb_model)
}

plot_decision_boundaries <- function(X, y, classifiers, titles) {
  """
  Plot decision boundaries for different classifiers
  
  Parameters:
  -----------
  X : data.frame
      Feature matrix (2D)
  y : factor
      Target labels
  classifiers : list
      List of classifier objects
  titles : character vector
      List of titles for each classifier
  """
  # Create mesh grid
  x_min <- min(X[, 1]) - 1
  x_max <- max(X[, 1]) + 1
  y_min <- min(X[, 2]) - 1
  y_max <- max(X[, 2]) + 1
  
  xx <- seq(x_min, x_max, length.out = 100)
  yy <- seq(y_min, y_max, length.out = 100)
  grid <- expand.grid(xx, yy)
  colnames(grid) <- colnames(X)
  
  # Create plots
  plots <- list()
  
  for (i in seq_along(classifiers)) {
    classifier <- classifiers[[i]]
    title <- titles[i]
    
    # Fit classifier and predict
    if (inherits(classifier, "train")) {
      # Caret model
      Z <- predict(classifier, grid)
    } else {
      # Other model types
      Z <- predict(classifier, grid)
    }
    
    # Create plot
    plot_data <- data.frame(
      x = grid[, 1],
      y = grid[, 2],
      class = Z
    )
    
    p <- ggplot() +
      geom_tile(data = plot_data, aes(x = x, y = y, fill = class), alpha = 0.3) +
      geom_point(data = data.frame(X, class = y), 
                aes(x = X[, 1], y = X[, 2], color = class), alpha = 0.8) +
      labs(title = title, x = "Feature 1", y = "Feature 2") +
      theme_minimal()
    
    plots[[i]] <- p
  }
  
  # Combine plots
  do.call(grid.arrange, c(plots, ncol = 2))
}

create_2d_dataset <- function(n_samples = 300) {
  """
  Create 2D dataset for visualization
  
  Parameters:
  -----------
  n_samples : integer
      Number of samples
      
  Returns:
  --------
  data : list
      List containing X (features) and y (target)
  """
  set.seed(42)
  
  # Generate two classes with different distributions
  X_class_0 <- MASS::mvrnorm(n_samples/2, mu = c(0, 0), 
                             Sigma = matrix(c(1, 0.5, 0.5, 1), 2, 2))
  X_class_1 <- MASS::mvrnorm(n_samples/2, mu = c(2, 2), 
                             Sigma = matrix(c(1, -0.5, -0.5, 1), 2, 2))
  
  X <- rbind(X_class_0, X_class_1)
  X <- as.data.frame(X)
  colnames(X) <- c("feature1", "feature2")
  
  y <- factor(rep(0:1, each = n_samples/2))
  
  return(list(X = X, y = y))
}

demonstrate_decision_boundaries <- function() {
  """
  Demonstrate decision boundaries for different classifiers.
  """
  # Create dataset and classifiers
  data <- create_2d_dataset()
  X <- data$X
  y <- data$y
  
  # Train different classifiers
  lr_model <- train(X, y, method = "glm")
  svm_model <- train(X, y, method = "svmRadial")
  rf_model <- train(X, y, method = "rf")
  
  classifiers <- list(lr_model, svm_model, rf_model)
  titles <- c("Logistic Regression (Linear)",
              "SVM with RBF Kernel (Non-linear)",
              "Random Forest")
  
  # Plot decision boundaries
  plot_decision_boundaries(X, y, classifiers, titles)
}

ClassificationEvaluator <- function() {
  """
  Comprehensive evaluation of classification models.
  """
  list()
}

evaluate_classifier <- function(y_true, y_pred, y_prob = NULL) {
  """
  Comprehensive evaluation of a classifier
  
  Parameters:
  -----------
  y_true : factor
      True labels
  y_pred : factor
      Predicted labels
  y_prob : numeric vector, optional
      Predicted probabilities
      
  Returns:
  --------
  results : list
      Evaluation results
  """
  results <- list()
  
  # Basic metrics
  results$accuracy <- mean(y_pred == y_true)
  results$precision <- precision(y_true, y_pred)
  results$recall <- recall(y_true, y_pred)
  results$f1 <- F_meas(y_true, y_pred)
  
  # Confusion matrix
  results$confusion_matrix <- confusionMatrix(y_pred, y_true)
  
  # ROC AUC (if probabilities available)
  if (!is.null(y_prob)) {
    results$roc_auc <- auc(roc(y_true, y_prob))
  }
  
  return(results)
}

plot_confusion_matrix <- function(y_true, y_pred, title = "Confusion Matrix") {
  """
  Plot confusion matrix
  
  Parameters:
  -----------
  y_true : factor
      True labels
  y_pred : factor
      Predicted labels
  title : character
      Plot title
  """
  cm <- confusionMatrix(y_pred, y_true)
  
  # Create heatmap
  cm_data <- as.data.frame(cm$table)
  ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 4) +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    labs(title = title) +
    theme_minimal()
}

plot_roc_curve <- function(y_true, y_prob, title = "ROC Curve") {
  """
  Plot ROC curve
  
  Parameters:
  -----------
  y_true : factor
      True labels
  y_prob : numeric vector
      Predicted probabilities
  title : character
      Plot title
  """
  roc_obj <- roc(y_true, y_prob)
  auc_value <- auc(roc_obj)
  
  # Create ROC plot
  plot(roc_obj, main = title, col = "blue", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "gray")
  legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), 
         col = "blue", lwd = 2)
}

demonstrate_evaluation <- function() {
  """
  Demonstrate evaluation metrics for classification.
  """
  # Get optimized results
  results <- demonstrate_optimization()
  lr_results <- results$lr_results
  
  # Evaluate our classifiers
  evaluator <- ClassificationEvaluator()
  
  # Evaluate logistic regression
  lr_eval <- evaluate_classifier(lr_results$predictions, lr_results$probabilities)
  cat("Logistic Regression Results:\n")
  cat("Accuracy:", lr_eval$accuracy, "\n")
  cat("Precision:", lr_eval$precision, "\n")
  cat("Recall:", lr_eval$recall, "\n")
  cat("F1:", lr_eval$f1, "\n")
  
  # Plot confusion matrix
  plot_confusion_matrix(lr_results$predictions, "Logistic Regression")
  
  # Plot ROC curve
  plot_roc_curve(lr_results$predictions, lr_results$probabilities, "Logistic Regression ROC")
}

handle_class_imbalance <- function() {
  """
  Demonstrate handling of class imbalance.
  """
  # Create imbalanced dataset
  set.seed(42)
  n_samples <- 1000
  
  # 90% class 0, 10% class 1
  X_imb <- data.frame(
    feature1 = rnorm(n_samples),
    feature2 = rnorm(n_samples)
  )
  y_imb <- factor(sample(c(0, 1), size = n_samples, replace = TRUE, prob = c(0.9, 0.1)))
  
  # Add some signal
  X_imb[y_imb == 1, ] <- X_imb[y_imb == 1, ] + 1
  
  # Split data
  train_index <- createDataPartition(y_imb, p = 0.7, list = FALSE)
  X_train_imb <- X_imb[train_index, ]
  X_test_imb <- X_imb[-train_index, ]
  y_train_imb <- y_imb[train_index]
  y_test_imb <- y_imb[-train_index]
  
  # Standard classifiers
  lr_imb <- train(X_train_imb, y_train_imb, method = "glm")
  rf_imb <- train(X_train_imb, y_train_imb, method = "rf")
  
  lr_pred <- predict(lr_imb, X_test_imb)
  rf_pred <- predict(rf_imb, X_test_imb)
  
  cat("Imbalanced Dataset Results:\n")
  cat("Class distribution:", table(y_test_imb), "\n")
  cat("Logistic Regression Accuracy:", mean(lr_pred == y_test_imb), "\n")
  cat("Random Forest Accuracy:", mean(rf_pred == y_test_imb), "\n")
  
  # Handle imbalance with class weights
  lr_weighted <- train(X_train_imb, y_train_imb, method = "glm",
                       weights = ifelse(y_train_imb == 1, 9, 1))
  rf_weighted <- train(X_train_imb, y_train_imb, method = "rf",
                       weights = ifelse(y_train_imb == 1, 9, 1))
  
  lr_w_pred <- predict(lr_weighted, X_test_imb)
  rf_w_pred <- predict(rf_weighted, X_test_imb)
  
  cat("\nWith Class Weights:\n")
  cat("Logistic Regression F1:", F_meas(y_test_imb, lr_w_pred), "\n")
  cat("Random Forest F1:", F_meas(y_test_imb, rf_w_pred), "\n")
}

analyze_feature_importance <- function() {
  """
  Analyze feature importance in classification.
  """
  # Get preprocessed data
  data <- demonstrate_data_preprocessing()
  X_train_scaled <- data$X_train_scaled
  y_train <- data$y_train
  
  # Use our credit dataset
  feature_names <- c('Income', 'Credit_Score', 'Debt_Ratio', 'Employment_Years')
  
  # Random Forest feature importance
  rf <- train(X_train_scaled, y_train, method = "rf")
  importance <- varImp(rf)$importance
  
  # Plot feature importance
  importance_df <- data.frame(
    feature = feature_names,
    importance = importance$Overall
  )
  importance_df <- importance_df[order(-importance_df$importance), ]
  
  ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance (Random Forest)",
         x = "Features", y = "Importance") +
    theme_minimal()
  
  # Logistic regression coefficients
  lr <- train(X_train_scaled, y_train, method = "glm")
  coef_values <- coef(lr$finalModel)[-1]  # Exclude intercept
  
  coef_df <- data.frame(
    feature = feature_names,
    coefficient = coef_values
  )
  coef_df <- coef_df[order(-abs(coef_df$coefficient)), ]
  
  ggplot(coef_df, aes(x = reorder(feature, abs(coefficient)), y = coefficient)) +
    geom_bar(stat = "identity", fill = "darkred") +
    coord_flip() +
    labs(title = "Feature Coefficients (Logistic Regression)",
         x = "Features", y = "Coefficient") +
    theme_minimal()
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  cat("Demonstrating Classification Implementation...\n")
  
  # Data preprocessing
  cat("\n1. Data Preprocessing\n")
  data <- demonstrate_data_preprocessing()
  
  # Classification models
  cat("\n2. Classification Models\n")
  model_results <- demonstrate_classification_models()
  
  # Loss functions
  cat("\n3. Loss Functions\n")
  demonstrate_loss_functions()
  
  # Optimization
  cat("\n4. Model Optimization\n")
  optimization_results <- demonstrate_optimization()
  
  # Bayes optimal classifier
  cat("\n5. Bayes Optimal Classifier\n")
  bayes_results <- demonstrate_bayes_optimal()
  
  # Multi-class classification
  cat("\n6. Multi-class Classification\n")
  multi_results <- demonstrate_multi_class()
  
  # Decision boundaries
  cat("\n7. Decision Boundaries\n")
  demonstrate_decision_boundaries()
  
  # Evaluation metrics
  cat("\n8. Evaluation Metrics\n")
  demonstrate_evaluation()
  
  # Class imbalance
  cat("\n9. Class Imbalance Handling\n")
  handle_class_imbalance()
  
  # Feature importance
  cat("\n10. Feature Importance Analysis\n")
  analyze_feature_importance()
}
