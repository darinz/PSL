# Discriminant Analysis Summary Implementation in R
library(MASS)
library(ggplot2)
library(gridExtra)
library(caret)
library(e1071)
library(pROC)
library(dplyr)
library(tidyr)

# Compare computational complexity of discriminant analysis methods
compare_complexity_r <- function() {
  # Parameters
  p_values <- seq(10, 100, by = 10)  # Feature dimensions
  K <- 3  # Number of classes
  
  # Parameter counts
  qda_params <- K * p_values^2 + K * p_values + K
  lda_params <- p_values^2 + K * p_values + K
  nb_params <- 2 * K * p_values + K
  
  # Create data frame for plotting
  complexity_df <- data.frame(
    p = rep(p_values, 3),
    parameters = c(qda_params, lda_params, nb_params),
    method = rep(c("QDA", "LDA", "Naive Bayes"), each = length(p_values))
  )
  
  # Plotting
  p <- ggplot(complexity_df, aes(x = p, y = parameters, color = method, shape = method)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    scale_y_log10() +
    labs(title = "Parameter Complexity Comparison",
         x = "Number of Features (p)",
         y = "Number of Parameters (log scale)",
         color = "Method",
         shape = "Method") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p)
  
  # Print numerical results
  cat("Parameter Complexity Comparison:\n")
  cat("-" * 50, "\n")
  for (i in 1:length(p_values)) {
    cat(sprintf("p=%3d: QDA=%6d, LDA=%6d, NB=%6d\n", 
                p_values[i], qda_params[i], lda_params[i], nb_params[i]))
  }
  
  return(list(qda_params = qda_params, lda_params = lda_params, nb_params = nb_params))
}

# Comprehensive comparison of discriminant analysis methods
class DiscriminantAnalysisComparisonR {
  methods <- list(
    LDA = function() lda,
    QDA = function() qda,
    "Naive Bayes" = function() naiveBayes
  )
  
  results <- list()
  
  generate_data <- function(n_samples = 1000, n_features = 10, n_classes = 3, 
                           n_informative = 8, n_redundant = 2, random_state = 42) {
    set.seed(random_state)
    
    # Generate synthetic data
    X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
    
    # Create informative features
    informative_features <- 1:n_informative
    redundant_features <- (n_informative + 1):(n_informative + n_redundant)
    noise_features <- (n_informative + n_redundant + 1):n_features
    
    # Generate class labels based on informative features
    class_centers <- matrix(rnorm(n_classes * n_informative), nrow = n_classes)
    y <- rep(0, n_samples)
    
    for (i in 1:n_classes) {
      start_idx <- (i-1) * (n_samples %/% n_classes) + 1
      end_idx <- i * (n_samples %/% n_classes)
      y[start_idx:end_idx] <- i - 1
    }
    
    # Add class-specific patterns to informative features
    for (i in 1:n_classes) {
      class_mask <- y == (i-1)
      X[class_mask, informative_features] <- X[class_mask, informative_features] + 
        matrix(class_centers[i,], nrow = sum(class_mask), ncol = n_informative, byrow = TRUE)
    }
    
    # Add redundancy to redundant features
    if (n_redundant > 0) {
      X[, redundant_features] <- X[, informative_features[1:n_redundant]] + 
        matrix(rnorm(n_samples * n_redundant, 0, 0.1), nrow = n_samples)
    }
    
    # Scale features
    X <- scale(X)
    
    return(list(X = X, y = y))
  }
  
  compare_methods <- function(X, y, cv = 5) {
    # Split data for cross-validation
    set.seed(42)
    folds <- createFolds(y, k = cv)
    
    for (method_name in names(methods)) {
      scores <- numeric(cv)
      
      for (i in 1:cv) {
        train_idx <- unlist(folds[-i])
        test_idx <- folds[[i]]
        
        X_train <- X[train_idx,, drop = FALSE]
        X_test <- X[test_idx,, drop = FALSE]
        y_train <- y[train_idx]
        y_test <- y[test_idx]
        
        # Fit method
        if (method_name == "LDA") {
          model <- lda(X_train, factor(y_train))
          pred <- predict(model, X_test)$class
        } else if (method_name == "QDA") {
          model <- qda(X_train, factor(y_train))
          pred <- predict(model, X_test)$class
        } else if (method_name == "Naive Bayes") {
          model <- naiveBayes(X_train, factor(y_train))
          pred <- predict(model, X_test)
        }
        
        # Calculate accuracy
        scores[i] <- mean(pred == y_test)
      }
      
      results[[method_name]] <<- list(
        mean_score = mean(scores),
        std_score = sd(scores),
        scores = scores
      )
    }
    
    return(results)
  }
  
  visualize_results <- function() {
    methods_list <- names(results)
    means <- sapply(results, function(x) x$mean_score)
    stds <- sapply(results, function(x) x$std_score)
    
    # Create data frame for plotting
    plot_df <- data.frame(
      method = methods_list,
      mean_score = means,
      std_score = stds
    )
    
    # Bar plot
    p1 <- ggplot(plot_df, aes(x = method, y = mean_score, fill = method)) +
      geom_bar(stat = "identity", alpha = 0.7) +
      geom_errorbar(aes(ymin = mean_score - std_score, ymax = mean_score + std_score), 
                   width = 0.2) +
      labs(title = "Accuracy Comparison",
           y = "Cross-validation Accuracy",
           fill = "Method") +
      theme_minimal() +
      theme(legend.position = "none") +
      ylim(0, 1)
    
    # Box plot
    scores_data <- lapply(results, function(x) x$scores)
    scores_df <- data.frame(
      method = rep(names(scores_data), sapply(scores_data, length)),
      score = unlist(scores_data)
    )
    
    p2 <- ggplot(scores_df, aes(x = method, y = score, fill = method)) +
      geom_boxplot(alpha = 0.7) +
      labs(title = "Score Distribution",
           y = "Accuracy",
           fill = "Method") +
      theme_minimal() +
      theme(legend.position = "none")
    
    # Display plots
    grid.arrange(p1, p2, ncol = 2)
    
    # Print results
    cat("Method Comparison Results:\n")
    cat("-" * 50, "\n")
    for (method in methods_list) {
      result <- results[[method]]
      cat(sprintf("%-15s: %.4f ± %.4f\n", method, result$mean_score, result$std_score))
    }
  }
  
  analyze_decision_boundaries <- function(X, y) {
    # Use only first 2 features for visualization
    X_2d <- X[, 1:2, drop = FALSE]
    
    # Create mesh for decision boundaries
    x_min <- min(X_2d[, 1]) - 1
    x_max <- max(X_2d[, 1]) + 1
    y_min <- min(X_2d[, 2]) - 1
    y_max <- max(X_2d[, 2]) + 1
    
    grid_points <- expand.grid(
      x1 = seq(x_min, x_max, length.out = 50),
      x2 = seq(y_min, y_max, length.out = 50)
    )
    
    plots <- list()
    
    for (i, method_name in enumerate(names(methods))) {
      # Fit method
      if (method_name == "LDA") {
        model <- lda(X_2d, factor(y))
        pred <- predict(model, grid_points)$class
      } else if (method_name == "QDA") {
        model <- qda(X_2d, factor(y))
        pred <- predict(model, grid_points)$class
      } else if (method_name == "Naive Bayes") {
        model <- naiveBayes(X_2d, factor(y))
        pred <- predict(model, grid_points)
      }
      
      grid_points$prediction <- pred
      
      # Create plot
      p <- ggplot() +
        geom_tile(data = grid_points, aes(x = x1, y = x2, fill = factor(prediction)), alpha = 0.3) +
        geom_point(data = data.frame(x1 = X_2d[, 1], x2 = X_2d[, 2], class = factor(y)),
                  aes(x = x1, y = x2, color = class), alpha = 0.7) +
        labs(title = paste(method_name, "Decision Boundaries"),
             x = "Feature 1",
             y = "Feature 2",
             color = "Class",
             fill = "Prediction") +
        theme_minimal()
      
      plots[[i]] <- p
    }
    
    # Display plots
    do.call(grid.arrange, c(plots, ncol = 3))
  }
  
  parameter_efficiency_analysis <- function(X, y) {
    n_features <- ncol(X)
    n_classes <- length(unique(y))
    
    # Calculate parameter counts
    qda_params <- n_classes * n_features^2 + n_classes * n_features + n_classes
    lda_params <- n_features^2 + n_classes * n_features + n_classes
    nb_params <- 2 * n_classes * n_features + n_classes
    
    # Calculate decision parameters
    qda_decision_params <- n_classes * (n_features + 1)  # Quadratic terms
    lda_decision_params <- n_features + 1  # Linear terms
    nb_decision_params <- n_features + 1  # Linear in log space
    
    # Create comparison table
    comparison_data <- data.frame(
      Method = c("QDA", "LDA", "Naive Bayes"),
      Total_Parameters = c(qda_params, lda_params, nb_params),
      Decision_Parameters = c(qda_decision_params, lda_decision_params, nb_decision_params),
      Efficiency_Ratio = c(qda_decision_params/qda_params, lda_decision_params/lda_params, nb_decision_params/nb_params)
    )
    
    cat("Parameter Efficiency Analysis:\n")
    cat("-" * 60, "\n")
    print(comparison_data)
    
    # Visualize efficiency
    # Parameter counts
    plot_df1 <- data.frame(
      Method = rep(comparison_data$Method, 2),
      Parameters = c(comparison_data$Total_Parameters, comparison_data$Decision_Parameters),
      Type = rep(c("Total Parameters", "Decision Parameters"), each = 3)
    )
    
    p1 <- ggplot(plot_df1, aes(x = Method, y = Parameters, fill = Type)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
      scale_y_log10() +
      labs(title = "Parameter Count Comparison",
           y = "Number of Parameters (log scale)",
           fill = "Parameter Type") +
      theme_minimal() +
      theme(legend.position = "bottom")
    
    # Efficiency ratio
    p2 <- ggplot(comparison_data, aes(x = Method, y = Efficiency_Ratio, fill = Method)) +
      geom_bar(stat = "identity", alpha = 0.7) +
      labs(title = "Parameter Efficiency (Higher is Better)",
           y = "Efficiency Ratio",
           fill = "Method") +
      theme_minimal() +
      theme(legend.position = "none") +
      geom_text(aes(label = sprintf("%.3f", Efficiency_Ratio)), 
                vjust = -0.5, size = 3)
    
    grid.arrange(p1, p2, ncol = 2)
    
    return(comparison_data)
  }
}

# Demonstrate comprehensive comparison
demonstrate_comparison_r <- function() {
  # Create comparison object
  comparison <- DiscriminantAnalysisComparisonR()
  
  # Generate data
  data <- comparison$generate_data(n_samples = 1000, n_features = 10, n_classes = 3)
  X <- data$X
  y <- data$y
  
  # Compare methods
  results <- comparison$compare_methods(X, y)
  
  # Visualize results
  comparison$visualize_results()
  
  # Analyze decision boundaries
  comparison$analyze_decision_boundaries(X, y)
  
  # Parameter efficiency analysis
  efficiency_df <- comparison$parameter_efficiency_analysis(X, y)
  
  return(list(comparison = comparison, results = results, efficiency = efficiency_df))
}

# Analyze computational scalability
analyze_scalability_r <- function() {
  # Parameters
  n_samples_list <- c(100, 500, 1000, 2000, 5000)
  n_features <- 50
  n_classes <- 3
  
  methods <- list(
    LDA = function(X, y) lda(X, factor(y)),
    QDA = function(X, y) qda(X, factor(y)),
    "Naive Bayes" = function(X, y) naiveBayes(X, factor(y))
  )
  
  timing_results <- list()
  
  for (n_samples in n_samples_list) {
    # Generate data
    set.seed(42)
    X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
    y <- sample(0:(n_classes-1), n_samples, replace = TRUE)
    
    sample_timing <- list()
    
    for (method_name in names(methods)) {
      # Time fitting
      start_time <- Sys.time()
      model <- methods[[method_name]](X, y)
      fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      # Time prediction
      start_time <- Sys.time()
      if (method_name == "LDA") {
        pred <- predict(model, X)$class
      } else if (method_name == "QDA") {
        pred <- predict(model, X)$class
      } else if (method_name == "Naive Bayes") {
        pred <- predict(model, X)
      }
      pred_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      sample_timing[[method_name]] <- c(fit_time, pred_time)
    }
    
    timing_results[[as.character(n_samples)]] <- sample_timing
  }
  
  # Create plotting data
  plot_data <- data.frame(
    n_samples = rep(n_samples_list, each = length(methods) * 2),
    method = rep(rep(names(methods), each = 2), length(n_samples_list)),
    time_type = rep(c("Fit", "Predict"), length(methods) * length(n_samples_list)),
    time = unlist(lapply(timing_results, function(x) {
      unlist(lapply(x, function(y) y))
    }))
  )
  
  # Plotting
  p1 <- ggplot(subset(plot_data, time_type == "Fit"), 
               aes(x = n_samples, y = time, color = method, shape = method)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    scale_y_log10() +
    labs(title = "Fitting Time Scalability",
         x = "Number of Samples",
         y = "Fitting Time (seconds, log scale)",
         color = "Method",
         shape = "Method") +
    theme_minimal()
  
  p2 <- ggplot(subset(plot_data, time_type == "Predict"), 
               aes(x = n_samples, y = time, color = method, shape = method)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    scale_y_log10() +
    labs(title = "Prediction Time Scalability",
         x = "Number of Samples",
         y = "Prediction Time (seconds, log scale)",
         color = "Method",
         shape = "Method") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 2)
  
  # Print timing results
  cat("Scalability Analysis Results:\n")
  cat("-" * 60, "\n")
  cat(sprintf("%8s %15s %10s %10s\n", "Samples", "Method", "Fit (s)", "Pred (s)"))
  cat("-" * 60, "\n")
  
  for (n_samples in n_samples_list) {
    for (method_name in names(methods)) {
      fit_time <- timing_results[[as.character(n_samples)]][[method_name]][1]
      pred_time <- timing_results[[as.character(n_samples)]][[method_name]][2]
      cat(sprintf("%8d %15s %10.4f %10.4f\n", n_samples, method_name, fit_time, pred_time))
    }
  }
  
  return(timing_results)
}

# Binary LDA analysis
binary_lda_analysis_r <- function() {
  # Generate binary classification data
  set.seed(42)
  n_samples <- 1000
  n_features <- 2
  
  # Class 0: centered at (0, 0)
  X0 <- MASS::mvrnorm(n_samples/2, mu = c(0, 0), Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  
  # Class 1: centered at (2, 2)
  X1 <- MASS::mvrnorm(n_samples/2, mu = c(2, 2), Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  
  X <- rbind(X0, X1)
  y <- rep(c(0, 1), each = n_samples/2)
  
  # Fit LDA
  lda_model <- lda(X, factor(y))
  
  # Extract parameters
  means <- lda_model$means
  covariance <- lda_model$scaling %*% t(lda_model$scaling)
  priors <- lda_model$prior
  
  # Calculate decision boundary parameters
  beta <- solve(covariance) %*% (means[2,] - means[1,])
  beta0 <- -0.5 * (means[2,] %*% solve(covariance) %*% means[2,] - 
                   means[1,] %*% solve(covariance) %*% means[1,]) + log(priors[2]/priors[1])
  
  cat("Binary LDA Analysis:\n")
  cat("-" * 40, "\n")
  cat("Class means:\n")
  print(means)
  cat("Shared covariance:\n")
  print(covariance)
  cat("Class priors:", priors, "\n")
  cat("Decision boundary coefficient:", beta, "\n")
  cat("Decision boundary intercept:", beta0, "\n")
  
  # Visualize decision boundary
  # Create mesh for decision boundary
  x_min <- min(X[, 1]) - 1
  x_max <- max(X[, 1]) + 1
  y_min <- min(X[, 2]) - 1
  y_max <- max(X[, 2]) + 1
  
  grid_points <- expand.grid(
    x1 = seq(x_min, x_max, length.out = 100),
    x2 = seq(y_min, y_max, length.out = 100)
  )
  
  # Calculate decision function
  Z <- beta[1] * grid_points$x1 + beta[2] * grid_points$x2 + beta0
  grid_points$decision <- Z
  
  # Create plots
  p1 <- ggplot() +
    geom_contour(data = grid_points, aes(x = x1, y = x2, z = decision), 
                breaks = 0, color = "red", size = 1) +
    geom_point(data = data.frame(x1 = X[, 1], x2 = X[, 2], class = factor(y)),
              aes(x = x1, y = x2, color = class), alpha = 0.7) +
    labs(title = "Binary LDA Decision Boundary",
         x = "Feature 1",
         y = "Feature 2",
         color = "Class") +
    theme_minimal()
  
  # Parameter efficiency visualization
  total_params <- n_features^2 + 2*n_features + 1  # covariance + means + prior
  decision_params <- n_features + 1  # beta + beta0
  
  efficiency_data <- data.frame(
    Parameter_Type = c("Total Parameters", "Decision Parameters"),
    Count = c(total_params, decision_params)
  )
  
  p2 <- ggplot(efficiency_data, aes(x = Parameter_Type, y = Count, fill = Parameter_Type)) +
    geom_bar(stat = "identity", alpha = 0.7) +
    labs(title = "Parameter Efficiency in Binary LDA",
         y = "Number of Parameters",
         fill = "Parameter Type") +
    theme_minimal() +
    theme(legend.position = "none") +
    geom_text(aes(label = Count), vjust = -0.5, size = 4)
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(lda_model = lda_model, beta = beta, beta0 = beta0))
}

# Method selection guidelines
method_selection_guidelines_r <- function() {
  # Create different scenarios
  scenarios <- list(
    "Low-dimensional, normal data" = list(
      n_features = 2,
      n_samples = 500,
      n_classes = 3,
      n_informative = 2,
      n_redundant = 0,
      recommended = "QDA"
    ),
    "High-dimensional, normal data" = list(
      n_features = 50,
      n_samples = 1000,
      n_classes = 3,
      n_informative = 20,
      n_redundant = 30,
      recommended = "LDA"
    ),
    "Limited training data" = list(
      n_features = 10,
      n_samples = 100,
      n_classes = 2,
      n_informative = 8,
      n_redundant = 2,
      recommended = "Naive Bayes"
    )
  )
  
  results <- list()
  
  for (scenario_name in names(scenarios)) {
    params <- scenarios[[scenario_name]]
    cat("\n", scenario_name, ":\n", sep = "")
    cat("-" * 50, "\n")
    
    # Generate data
    set.seed(42)
    X <- matrix(rnorm(params$n_samples * params$n_features), 
               nrow = params$n_samples, ncol = params$n_features)
    
    # Create class labels
    y <- sample(0:(params$n_classes-1), params$n_samples, replace = TRUE)
    
    # Scale features
    X <- scale(X)
    
    # Compare methods
    methods <- list(
      LDA = function(X, y) lda(X, factor(y)),
      QDA = function(X, y) qda(X, factor(y)),
      "Naive Bayes" = function(X, y) naiveBayes(X, factor(y))
    )
    
    # Cross-validation
    set.seed(42)
    folds <- createFolds(y, k = 5)
    scenario_results <- list()
    
    for (method_name in names(methods)) {
      scores <- numeric(5)
      
      for (i in 1:5) {
        train_idx <- unlist(folds[-i])
        test_idx <- folds[[i]]
        
        X_train <- X[train_idx,, drop = FALSE]
        X_test <- X[test_idx,, drop = FALSE]
        y_train <- y[train_idx]
        y_test <- y[test_idx]
        
        # Fit method
        model <- methods[[method_name]](X_train, y_train)
        
        # Predict
        if (method_name == "LDA") {
          pred <- predict(model, X_test)$class
        } else if (method_name == "QDA") {
          pred <- predict(model, X_test)$class
        } else if (method_name == "Naive Bayes") {
          pred <- predict(model, X_test)
        }
        
        scores[i] <- mean(pred == y_test)
      }
      
      scenario_results[[method_name]] <- list(
        mean_score = mean(scores),
        std_score = sd(scores)
      )
    }
    
    # Print results
    cat("Recommended method:", params$recommended, "\n")
    cat("Performance comparison:\n")
    for (method_name in names(scenario_results)) {
      result <- scenario_results[[method_name]]
      marker <- ifelse(method_name == params$recommended, "★", " ")
      cat(sprintf("%s %-12s: %.4f ± %.4f\n", marker, method_name, 
                 result$mean_score, result$std_score))
    }
    
    results[[scenario_name]] <- scenario_results
  }
  
  return(results)
}

# Limitations analysis
limitations_analysis_r <- function() {
  limitations <- list(
    "Distributional Assumptions" = list(
      description = "Most methods assume normality",
      impact = "Performance degrades with non-normal data",
      mitigation = "Use non-parametric methods or data transformation"
    ),
    "Linear Decision Boundaries" = list(
      description = "LDA and FDA are limited to linear separators",
      impact = "Cannot capture complex non-linear relationships",
      mitigation = "Use QDA, kernel methods, or non-linear classifiers"
    ),
    "Parameter Inefficiency" = list(
      description = "Many parameters for simple decision rules",
      impact = "Computational cost and overfitting risk",
      mitigation = "Use direct methods like logistic regression"
    ),
    "Curse of Dimensionality" = list(
      description = "Performance degrades in high dimensions",
      impact = "Poor generalization with many features",
      mitigation = "Feature selection, regularization, or dimensionality reduction"
    ),
    "Feature Independence" = list(
      description = "Naive Bayes assumes independence",
      impact = "Performance loss with correlated features",
      mitigation = "Feature engineering or use other methods"
    )
  )
  
  cat("Limitations of Discriminant Analysis:\n")
  cat("=" * 80, "\n")
  
  for (limitation_name in names(limitations)) {
    details <- limitations[[limitation_name]]
    cat("\n", limitation_name, ":\n", sep = "")
    cat("  Description: ", details$description, "\n", sep = "")
    cat("  Impact: ", details$impact, "\n", sep = "")
    cat("  Mitigation: ", details$mitigation, "\n", sep = "")
  }
  
  # Demonstrate some limitations with examples
  cat("\n", "=" * 80, "\n", sep = "")
  cat("Demonstrating Limitations:\n")
  cat("=" * 80, "\n")
  
  # 1. Non-normal data example
  cat("\n1. Non-normal Data Example:\n")
  set.seed(42)
  X_nonnormal <- matrix(rexp(500 * 2, 1), nrow = 500, ncol = 2)
  y_nonnormal <- as.numeric(X_nonnormal[, 1] + X_nonnormal[, 2] > 2)
  
  lda_nonnormal <- lda(X_nonnormal, factor(y_nonnormal))
  nb_nonnormal <- naiveBayes(X_nonnormal, factor(y_nonnormal))
  
  # Cross-validation
  set.seed(42)
  folds <- createFolds(y_nonnormal, k = 5)
  
  lda_scores <- numeric(5)
  nb_scores <- numeric(5)
  
  for (i in 1:5) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    X_train <- X_nonnormal[train_idx,, drop = FALSE]
    X_test <- X_nonnormal[test_idx,, drop = FALSE]
    y_train <- y_nonnormal[train_idx]
    y_test <- y_nonnormal[test_idx]
    
    # LDA
    lda_model <- lda(X_train, factor(y_train))
    lda_pred <- predict(lda_model, X_test)$class
    lda_scores[i] <- mean(lda_pred == y_test)
    
    # Naive Bayes
    nb_model <- naiveBayes(X_train, factor(y_train))
    nb_pred <- predict(nb_model, X_test)
    nb_scores[i] <- mean(nb_pred == y_test)
  }
  
  cat("LDA accuracy on non-normal data:", mean(lda_scores), "\n")
  cat("Naive Bayes accuracy on non-normal data:", mean(nb_scores), "\n")
  
  # 2. High-dimensional data example
  cat("\n2. High-dimensional Data Example:\n")
  set.seed(42)
  X_highdim <- matrix(rnorm(100 * 100), nrow = 100, ncol = 100)
  y_highdim <- sample(0:1, 100, replace = TRUE)
  
  # Cross-validation
  set.seed(42)
  folds <- createFolds(y_highdim, k = 5)
  
  lda_scores_high <- numeric(5)
  nb_scores_high <- numeric(5)
  
  for (i in 1:5) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    X_train <- X_highdim[train_idx,, drop = FALSE]
    X_test <- X_highdim[test_idx,, drop = FALSE]
    y_train <- y_highdim[train_idx]
    y_test <- y_highdim[test_idx]
    
    # LDA
    lda_model <- lda(X_train, factor(y_train))
    lda_pred <- predict(lda_model, X_test)$class
    lda_scores_high[i] <- mean(lda_pred == y_test)
    
    # Naive Bayes
    nb_model <- naiveBayes(X_train, factor(y_train))
    nb_pred <- predict(nb_model, X_test)
    nb_scores_high[i] <- mean(nb_pred == y_test)
  }
  
  cat("LDA accuracy on high-dimensional data:", mean(lda_scores_high), "\n")
  cat("Naive Bayes accuracy on high-dimensional data:", mean(nb_scores_high), "\n")
  
  return(limitations)
}

# Main function to demonstrate summary analysis
main_r <- function() {
  cat("Discriminant Analysis Summary Demonstration\n")
  cat("=" * 60, "\n")
  
  # 1. Complexity comparison
  cat("\n1. Parameter Complexity Analysis:\n")
  complexity_results <- compare_complexity_r()
  
  # 2. Comprehensive comparison
  cat("\n2. Comprehensive Method Comparison:\n")
  comparison_results <- demonstrate_comparison_r()
  
  # 3. Scalability analysis
  cat("\n3. Computational Scalability Analysis:\n")
  timing_results <- analyze_scalability_r()
  
  # 4. Binary LDA analysis
  cat("\n4. Binary LDA Detailed Analysis:\n")
  binary_results <- binary_lda_analysis_r()
  
  # 5. Method selection guidelines
  cat("\n5. Method Selection Guidelines:\n")
  selection_results <- method_selection_guidelines_r()
  
  # 6. Limitations analysis
  cat("\n6. Limitations Analysis:\n")
  limitations <- limitations_analysis_r()
  
  return(list(
    complexity = complexity_results,
    comparison = comparison_results,
    timing = timing_results,
    binary_lda = binary_results,
    selection = selection_results,
    limitations = limitations
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
