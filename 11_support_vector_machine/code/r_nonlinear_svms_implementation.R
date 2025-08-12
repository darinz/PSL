# Support Vector Machines: Nonlinear SVMs Implementation

# Load required libraries
library(e1071)
library(ggplot2)
library(gridExtra)
library(kernlab)

# Generate XOR-like data
generate_xor_data <- function(n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  # Generate XOR pattern with noise
  X <- matrix(rnorm(n_samples * 2), ncol = 2) * 0.3
  y <- rep(1, n_samples)
  
  # XOR pattern: (0,0) and (1,1) are class -1, (0,1) and (1,0) are class 1
  for (i in 1:n_samples) {
    if ((X[i, 1] < 0 && X[i, 2] < 0) || (X[i, 1] > 0 && X[i, 2] > 0)) {
      y[i] <- -1
    } else {
      y[i] <- 1
    }
  }
  
  return(list(X = X, y = y))
}

# Generate non-linear data
generate_nonlinear_data <- function(data_type = "circles", n_samples = 100, random_state = 42) {
  set.seed(random_state)
  
  if (data_type == "circles") {
    # Generate circular data
    theta <- runif(n_samples, 0, 2 * pi)
    r <- runif(n_samples, 0.5, 1.5)
    
    # Inner circle (class 1)
    n_inner <- n_samples %/% 2
    inner_theta <- theta[1:n_inner]
    inner_r <- runif(n_inner, 0.5, 1.0)
    
    # Outer circle (class -1)
    outer_theta <- theta[(n_inner + 1):n_samples]
    outer_r <- runif(n_samples - n_inner, 1.2, 1.8)
    
    X_inner <- cbind(inner_r * cos(inner_theta), inner_r * sin(inner_theta))
    X_outer <- cbind(outer_r * cos(outer_theta), outer_r * sin(outer_theta))
    
    X <- rbind(X_inner, X_outer)
    y <- c(rep(1, n_inner), rep(-1, n_samples - n_inner))
    
  } else if (data_type == "moons") {
    # Generate moon-shaped data
    n_class1 <- n_samples %/% 2
    n_class2 <- n_samples - n_class1
    
    # Class 1: upper moon
    theta1 <- runif(n_class1, 0, pi)
    r1 <- runif(n_class1, 0.8, 1.2)
    X1 <- cbind(r1 * cos(theta1), r1 * sin(theta1) + 0.5)
    
    # Class 2: lower moon
    theta2 <- runif(n_class2, pi, 2 * pi)
    r2 <- runif(n_class2, 0.8, 1.2)
    X2 <- cbind(r2 * cos(theta2), r2 * sin(theta2) - 0.5)
    
    X <- rbind(X1, X2)
    y <- c(rep(1, n_class1), rep(-1, n_class2))
    
  } else if (data_type == "xor") {
    data <- generate_xor_data(n_samples, random_state)
    X <- data$X
    y <- data$y
  } else {
    stop("Unknown data type: ", data_type)
  }
  
  return(list(X = X, y = y))
}

# Visualize kernel SVM
visualize_kernel_svm <- function(X, y, model, title = "Kernel SVM") {
  # Create grid
  x_min <- min(X[, 1]) - 0.5
  x_max <- max(X[, 1]) + 0.5
  y_min <- min(X[, 2]) - 0.5
  y_max <- max(X[, 2]) + 0.5
  
  grid_x <- seq(x_min, x_max, length.out = 100)
  grid_y <- seq(y_min, y_max, length.out = 100)
  grid_data <- expand.grid(X1 = grid_x, X2 = grid_y)
  
  # Predict on grid
  grid_data$pred <- predict(model, grid_data)
  
  # Create plot
  p <- ggplot() +
    geom_contour(data = grid_data, aes(x = X1, y = X2, z = as.numeric(pred)), 
                 breaks = 0, color = "black", size = 1) +
    geom_point(data = data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y)), 
               aes(x = X1, y = X2, color = y), size = 3, alpha = 0.8) +
    scale_color_manual(values = c("-1" = "blue", "1" = "red")) +
    labs(title = title, x = "Feature 1", y = "Feature 2") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Highlight support vectors if available
  if (!is.null(model$index)) {
    support_vectors <- data.frame(X1 = X[model$index, 1], X2 = X[model$index, 2])
    p <- p + geom_point(data = support_vectors, aes(x = X1, y = X2), 
                        shape = 21, size = 4, fill = "transparent", color = "black")
  }
  
  return(p)
}

# Demonstrate kernel comparison
demonstrate_kernel_comparison <- function() {
  cat("=== Kernel Comparison ===\n\n")
  
  # Generate non-linear data
  data <- generate_nonlinear_data("circles", n_samples = 100)
  X <- data$X
  y <- data$y
  
  cat("Data shape:", dim(X), "\n")
  cat("Class distribution:", table(y), "\n")
  
  # Compare different kernels
  kernels <- c("linear", "polynomial", "radial", "sigmoid")
  models <- list()
  plots <- list()
  
  for (i in seq_along(kernels)) {
    kernel <- kernels[i]
    cat("Fitting SVM with", kernel, "kernel...\n")
    
    # Fit SVM
    model <- svm(X, y, kernel = kernel, scale = FALSE)
    models[[i]] <- model
    
    # Evaluate
    y_pred <- predict(model, X)
    accuracy <- mean(y_pred == y)
    n_sv <- length(model$index)
    
    cat("  Accuracy:", round(accuracy, 3), "Support vectors:", n_sv, "\n")
    
    # Create plot
    title <- paste(toupper(kernel), "Kernel\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_kernel_svm(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print summary
  cat("\nSummary:\n")
  for (i in seq_along(kernels)) {
    kernel <- kernels[i]
    model <- models[[i]]
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    cat(sprintf("%-12s kernel: Accuracy = %.3f, Support Vectors = %d\n", 
                toupper(kernel), accuracy, n_sv))
  }
  
  return(list(models = models, X = X, y = y, kernels = kernels))
}

# Demonstrate kernel functions
demonstrate_kernel_functions <- function() {
  cat("\n=== Kernel Functions Demonstration ===\n\n")
  
  # Generate sample data
  x1 <- matrix(c(1, 2), nrow = 1)
  x2 <- matrix(c(3, 4), nrow = 1)
  
  # Test different kernels
  kernels <- c("linear", "polynomial", "radial", "sigmoid")
  gamma_values <- c(0.1, 1.0, 10.0)
  
  cat("Kernel function values for x1 = [1, 2], x2 = [3, 4]:\n")
  cat("-" * 60, "\n")
  
  for (kernel in kernels) {
    cat("\n", toupper(kernel), "Kernel:\n")
    for (gamma in gamma_values) {
      # Create kernel matrix using kernlab
      if (kernel == "linear") {
        k_value <- as.numeric(x1 %*% t(x2))
      } else if (kernel == "polynomial") {
        k_value <- (gamma * as.numeric(x1 %*% t(x2)) + 1)^3
      } else if (kernel == "radial") {
        dist_sq <- sum((x1 - x2)^2)
        k_value <- exp(-gamma * dist_sq)
      } else if (kernel == "sigmoid") {
        k_value <- tanh(gamma * as.numeric(x1 %*% t(x2)) + 1)
      }
      cat(sprintf("  γ = %4.1f: K(x1, x2) = %.4f\n", gamma, k_value))
    }
  }
  
  # Visualize kernel functions
  x <- seq(-3, 3, length.out = 100)
  y <- rep(0, length(x))
  
  # Create data for plotting
  plot_data <- data.frame()
  
  for (kernel in kernels) {
    for (gamma in gamma_values) {
      k_values <- numeric(length(x))
      
      for (i in seq_along(x)) {
        x_point <- matrix(c(x[i], y[i]), nrow = 1)
        origin <- matrix(c(0, 0), nrow = 1)
        
        if (kernel == "linear") {
          k_values[i] <- as.numeric(x_point %*% t(origin))
        } else if (kernel == "polynomial") {
          k_values[i] <- (gamma * as.numeric(x_point %*% t(origin)) + 1)^3
        } else if (kernel == "radial") {
          dist_sq <- sum((x_point - origin)^2)
          k_values[i] <- exp(-gamma * dist_sq)
        } else if (kernel == "sigmoid") {
          k_values[i] <- tanh(gamma * as.numeric(x_point %*% t(origin)) + 1)
        }
      }
      
      plot_data <- rbind(plot_data, data.frame(
        x = x,
        k_value = k_values,
        kernel = kernel,
        gamma = gamma
      ))
    }
  }
  
  # Create plots
  p1 <- ggplot(subset(plot_data, kernel == "linear"), 
               aes(x = x, y = k_value, color = factor(gamma))) +
    geom_line(size = 1) +
    labs(title = "Linear Kernel", x = "Distance from origin", y = "Kernel value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p2 <- ggplot(subset(plot_data, kernel == "polynomial"), 
               aes(x = x, y = k_value, color = factor(gamma))) +
    geom_line(size = 1) +
    labs(title = "Polynomial Kernel", x = "Distance from origin", y = "Kernel value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p3 <- ggplot(subset(plot_data, kernel == "radial"), 
               aes(x = x, y = k_value, color = factor(gamma))) +
    geom_line(size = 1) +
    labs(title = "RBF Kernel", x = "Distance from origin", y = "Kernel value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  p4 <- ggplot(subset(plot_data, kernel == "sigmoid"), 
               aes(x = x, y = k_value, color = factor(gamma))) +
    geom_line(size = 1) +
    labs(title = "Sigmoid Kernel", x = "Distance from origin", y = "Kernel value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, p3, p4, ncol = 2)
}

# Demonstrate parameter effects
demonstrate_parameter_effects <- function() {
  cat("\n=== Parameter Effects ===\n\n")
  
  # Generate data
  data <- generate_nonlinear_data("circles", n_samples = 100)
  X <- data$X
  y <- data$y
  
  # Test different gamma values for RBF kernel
  gamma_values <- c(0.1, 1.0, 10.0, 100.0)
  models <- list()
  plots <- list()
  
  for (i in seq_along(gamma_values)) {
    gamma <- gamma_values[i]
    cat("Testing gamma =", gamma, "...\n")
    
    # Fit SVM
    model <- svm(X, y, kernel = "radial", gamma = gamma, scale = FALSE)
    models[[i]] <- model
    
    # Evaluate
    y_pred <- predict(model, X)
    accuracy <- mean(y_pred == y)
    n_sv <- length(model$index)
    
    cat("  Accuracy:", round(accuracy, 3), "Support vectors:", n_sv, "\n")
    
    # Create plot
    title <- paste("RBF Kernel, γ =", gamma, "\nAccuracy:", round(accuracy, 3), "SVs:", n_sv)
    plots[[i]] <- visualize_kernel_svm(X, y, model, title)
  }
  
  # Display plots
  do.call(grid.arrange, c(plots, ncol = 2))
  
  # Print analysis
  cat("\nRBF Kernel Parameter Analysis:\n")
  for (i in seq_along(gamma_values)) {
    gamma <- gamma_values[i]
    model <- models[[i]]
    accuracy <- mean(predict(model, X) == y)
    n_sv <- length(model$index)
    cat(sprintf("  γ = %6.1f: Accuracy = %.3f, Support Vectors = %d\n", gamma, accuracy, n_sv))
  }
  
  return(models)
}

# Demonstrate cross-validation
demonstrate_cross_validation <- function() {
  cat("\n=== Cross-Validation for Kernel Selection ===\n\n")
  
  # Generate data
  data <- generate_nonlinear_data("circles", n_samples = 200)
  X <- data$X
  y <- data$y
  
  # Create data frame for tuning
  df <- data.frame(X1 = X[, 1], X2 = X[, 2], y = factor(y))
  
  # Define parameter grids for different kernels
  cat("Testing different kernels with cross-validation...\n")
  
  # Linear kernel
  cat("Testing linear kernel...\n")
  linear_tuned <- tune(svm, y ~ ., data = df, 
                       ranges = list(cost = c(0.1, 1, 10, 100)),
                       kernel = "linear")
  
  # Polynomial kernel
  cat("Testing polynomial kernel...\n")
  poly_tuned <- tune(svm, y ~ ., data = df, 
                     ranges = list(cost = c(0.1, 1, 10), 
                                  degree = c(2, 3, 4), 
                                  gamma = c(0.1, 1, 10)),
                     kernel = "polynomial")
  
  # RBF kernel
  cat("Testing RBF kernel...\n")
  rbf_tuned <- tune(svm, y ~ ., data = df, 
                    ranges = list(cost = c(0.1, 1, 10, 100), 
                                 gamma = c(0.001, 0.01, 0.1, 1, 10)),
                    kernel = "radial")
  
  # Sigmoid kernel
  cat("Testing sigmoid kernel...\n")
  sigmoid_tuned <- tune(svm, y ~ ., data = df, 
                        ranges = list(cost = c(0.1, 1, 10), 
                                     gamma = c(0.1, 1, 10)),
                        kernel = "sigmoid")
  
  # Collect results
  best_scores <- list(
    linear = 1 - linear_tuned$best.performance,
    polynomial = 1 - poly_tuned$best.performance,
    radial = 1 - rbf_tuned$best.performance,
    sigmoid = 1 - sigmoid_tuned$best.performance
  )
  
  best_params <- list(
    linear = linear_tuned$best.parameters,
    polynomial = poly_tuned$best.parameters,
    radial = rbf_tuned$best.parameters,
    sigmoid = sigmoid_tuned$best.parameters
  )
  
  # Print results
  cat("\nResults:\n")
  for (kernel in names(best_scores)) {
    cat(sprintf("%-12s kernel: Best score = %.3f\n", kernel, best_scores[[kernel]]))
    cat("              Best parameters:", paste(names(best_params[[kernel]]), 
                                               best_params[[kernel]], sep = "=", collapse = ", "), "\n")
  }
  
  # Find best kernel
  best_kernel <- names(best_scores)[which.max(unlist(best_scores))]
  cat(sprintf("\nBest kernel: %s with score %.3f\n", toupper(best_kernel), best_scores[[best_kernel]]))
  
  # Plot results
  plot_data <- data.frame(
    kernel = names(best_scores),
    score = unlist(best_scores)
  )
  
  p <- ggplot(plot_data, aes(x = kernel, y = score, fill = kernel)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = sprintf("%.3f", score)), vjust = -0.5) +
    labs(title = "Kernel Performance Comparison", 
         x = "Kernel Type", y = "Cross-validation Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = "none") +
    ylim(0, 1)
  
  print(p)
  
  return(list(best_scores = best_scores, best_params = best_params))
}

# Demonstrate representer theorem
demonstrate_representer_theorem <- function() {
  cat("\n=== Representer Theorem Demonstration ===\n\n")
  
  # Generate data
  data <- generate_nonlinear_data("circles", n_samples = 50)
  X <- data$X
  y <- data$y
  
  # Fit kernel SVM
  model <- svm(X, y, kernel = "radial", gamma = 1, scale = FALSE)
  
  # Test points
  test_points <- matrix(c(0, 0, 1, 1, -1, -1, 0.5, -0.5), ncol = 2, byrow = TRUE)
  
  cat("Representer Theorem Verification:\n")
  cat("f(x) = Σ α_i K(x_i, x) + β_0\n")
  cat("-" * 50, "\n")
  
  for (i in 1:nrow(test_points)) {
    test_point <- test_points[i, ]
    prediction <- predict(model, matrix(test_point, nrow = 1))
    cat(sprintf("Test point %d [%.1f, %.1f]: f(x) = %.4f\n", 
                i, test_point[1], test_point[2], as.numeric(prediction)))
  }
  
  # Visualize the representer form
  cat(sprintf("\nNumber of support vectors: %d\n", length(model$index)))
  cat(sprintf("Total training points: %d\n", nrow(X)))
  cat(sprintf("Sparsity ratio: %.3f\n", length(model$index)/nrow(X)))
  
  return(model)
}

# Demonstrate advantages and limitations
demonstrate_advantages_limitations <- function() {
  cat("\n=== Advantages and Limitations ===\n\n")
  
  # Generate different types of data
  datasets <- list(
    "Linear" = generate_nonlinear_data("circles", n_samples = 100, random_state = 42),
    "Nonlinear" = generate_nonlinear_data("moons", n_samples = 100, random_state = 42),
    "XOR" = generate_nonlinear_data("xor", n_samples = 100, random_state = 42)
  )
  
  kernels <- c("linear", "radial")
  results <- list()
  
  for (name in names(datasets)) {
    cat(name, "Data:\n")
    data <- datasets[[name]]
    X <- data$X
    y <- data$y
    
    results[[name]] <- list()
    
    for (kernel in kernels) {
      # Fit SVM
      model <- svm(X, y, kernel = kernel, scale = FALSE)
      
      # Calculate metrics
      accuracy <- mean(predict(model, X) == y)
      n_sv <- length(model$index)
      
      results[[name]][[kernel]] <- list(
        accuracy = accuracy,
        n_support_vectors = n_sv
      )
      
      cat(sprintf("  %-8s kernel: Accuracy = %.3f, SVs = %d\n", 
                  toupper(kernel), accuracy, n_sv))
    }
  }
  
  # Create comparison plot
  plot_data <- data.frame()
  
  for (name in names(datasets)) {
    for (kernel in kernels) {
      result <- results[[name]][[kernel]]
      plot_data <- rbind(plot_data, data.frame(
        dataset = name,
        kernel = kernel,
        accuracy = result$accuracy,
        n_support_vectors = result$n_support_vectors
      ))
    }
  }
  
  # Accuracy comparison
  p1 <- ggplot(plot_data, aes(x = dataset, y = accuracy, fill = kernel)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Accuracy Comparison", x = "Dataset Type", y = "Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Support vectors comparison
  p2 <- ggplot(plot_data, aes(x = dataset, y = n_support_vectors, fill = kernel)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Support Vectors Comparison", 
         x = "Dataset Type", y = "Number of Support Vectors") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(results)
}

# Main function to demonstrate nonlinear SVM concepts
main_r <- function() {
  cat("Support Vector Machines: Nonlinear SVMs Implementation\n")
  cat("=" * 60, "\n")
  
  # 1. Kernel comparison
  cat("\n1. Kernel Comparison:\n")
  kernel_results <- demonstrate_kernel_comparison()
  
  # 2. Kernel functions demonstration
  cat("\n2. Kernel Functions Demonstration:\n")
  demonstrate_kernel_functions()
  
  # 3. Parameter effects
  cat("\n3. Parameter Effects:\n")
  parameter_results <- demonstrate_parameter_effects()
  
  # 4. Cross-validation for kernel selection
  cat("\n4. Cross-Validation for Kernel Selection:\n")
  cv_results <- demonstrate_cross_validation()
  
  # 5. Representer theorem demonstration
  cat("\n5. Representer Theorem Demonstration:\n")
  representer_model <- demonstrate_representer_theorem()
  
  # 6. Advantages and limitations
  cat("\n6. Advantages and Limitations:\n")
  advantages_results <- demonstrate_advantages_limitations()
  
  cat("\n=== Key Insights ===\n")
  cat("1. Kernel trick allows nonlinear classification without explicit feature transformation\n")
  cat("2. RBF kernel is most commonly used and works well for most problems\n")
  cat("3. Parameter γ controls the influence of each training point\n")
  cat("4. Cross-validation is essential for kernel and parameter selection\n")
  cat("5. Representer theorem ensures finite representation using support vectors\n")
  cat("6. Kernel SVM provides sparse solutions with only support vectors mattering\n")
  cat("7. Computational cost scales with number of support vectors\n")
  cat("8. Kernel selection depends on data characteristics and domain knowledge\n")
  
  return(list(
    kernel_results = kernel_results,
    parameter_results = parameter_results,
    cv_results = cv_results,
    representer_model = representer_model,
    advantages_results = advantages_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
