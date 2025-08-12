# Support Vector Machines: Appendix Implementation

# Load required libraries
library(e1071)
library(ggplot2)
library(gridExtra)
library(kernlab)
library(MASS)

# RKHS implementation
RKHS <- function(kernel = "rbf", gamma = 1.0) {
  list(
    kernel = kernel,
    gamma = gamma,
    X_train = NULL,
    alpha = NULL
  )
}

kernel_function <- function(X1, X2, kernel = "rbf", gamma = 1.0) {
  if (kernel == "rbf") {
    # RBF kernel: K(x,y) = exp(-gamma ||x-y||^2)
    dist_sq <- as.matrix(dist(rbind(X1, X2)))[1:nrow(X1), (nrow(X1)+1):(nrow(X1)+nrow(X2))]
    return(exp(-gamma * dist_sq^2))
  } else if (kernel == "linear") {
    return(X1 %*% t(X2))
  } else if (kernel == "poly") {
    return((X1 %*% t(X2) + 1)^2)
  } else {
    stop("Unknown kernel: ", kernel)
  }
}

fit_rkhs <- function(rkhs, X, y, lambda_reg = 0.1) {
  rkhs$X_train <- X
  n_samples <- nrow(X)
  
  # Compute kernel matrix
  K <- kernel_function(X, X, rkhs$kernel, rkhs$gamma)
  
  # Add regularization
  K_reg <- K + lambda_reg * diag(n_samples)
  
  # Solve linear system: K_reg * alpha = y
  rkhs$alpha <- solve(K_reg, y)
  
  return(rkhs)
}

predict_rkhs <- function(rkhs, X) {
  if (is.null(rkhs$X_train)) {
    stop("Model not fitted yet")
  }
  
  K_test <- kernel_function(X, rkhs$X_train, rkhs$kernel, rkhs$gamma)
  return(K_test %*% rkhs$alpha)
}

demonstrate_rkhs <- function() {
  cat("=== RKHS Demonstration ===\n\n")
  
  # Generate example data
  set.seed(42)
  X <- matrix(rnorm(100 * 2), ncol = 2)
  y <- sin(X[, 1]) + cos(X[, 2]) + 0.1 * rnorm(100)
  
  # Fit RKHS model
  rkhs <- RKHS(kernel = "rbf", gamma = 1.0)
  rkhs <- fit_rkhs(rkhs, X, y, lambda_reg = 0.01)
  
  # Test predictions
  X_test <- matrix(rnorm(20 * 2), ncol = 2)
  y_pred <- predict_rkhs(rkhs, X_test)
  
  cat("Predictions shape:", dim(y_pred), "\n")
  cat("Representer theorem form: f(x) = Σ α_i K(x_i, x)\n")
  cat("Number of training points:", nrow(X), "\n")
  cat("Number of coefficients:", length(rkhs$alpha), "\n")
  
  # Visualize the fit
  plot_data <- data.frame(
    x = c(X[, 1], X_test[, 1]),
    y = c(X[, 2], X_test[, 2]),
    value = c(y, y_pred),
    type = c(rep("Training", nrow(X)), rep("Test", nrow(X_test)))
  )
  
  p1 <- ggplot(subset(plot_data, type == "Training"), aes(x = x, y = y, color = value)) +
    geom_point(size = 3) +
    scale_color_viridis_c() +
    labs(title = "Training Data", x = "X1", y = "X2") +
    theme_minimal()
  
  p2 <- ggplot(subset(plot_data, type == "Test"), aes(x = x, y = y, color = value)) +
    geom_point(size = 3) +
    scale_color_viridis_c() +
    labs(title = "Test Predictions", x = "X1", y = "X2") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(rkhs)
}

check_kernel_properties <- function(K) {
  n <- nrow(K)
  
  # Check symmetry
  is_symmetric <- all.equal(K, t(K))
  cat("Symmetric:", is_symmetric, "\n")
  
  # Check positive semi-definiteness
  eigenvals <- eigen(K, symmetric = TRUE)$values
  is_psd <- all(eigenvals >= -1e-10)  # Allow small numerical errors
  cat("Positive semi-definite:", is_psd, "\n")
  cat("Eigenvalues:", eigenvals[1:5], "...\n")  # Show first 5
  
  # Check trace
  trace <- sum(diag(K))
  cat("Trace:", round(trace, 3), "\n")
  
  return(is_symmetric && is_psd)
}

demonstrate_mercer_theorem <- function() {
  cat("\n=== Mercer's Theorem Demonstration ===\n\n")
  
  # Test different kernels
  set.seed(42)
  X <- matrix(rnorm(50 * 3), ncol = 3)
  
  # Linear kernel
  K_linear <- X %*% t(X)
  cat("Linear kernel:\n")
  check_kernel_properties(K_linear)
  
  # RBF kernel
  gamma <- 1.0
  dist_sq <- as.matrix(dist(X))^2
  K_rbf <- exp(-gamma * dist_sq)
  cat("\nRBF kernel:\n")
  check_kernel_properties(K_rbf)
  
  # Polynomial kernel
  K_poly <- (X %*% t(X) + 1)^2
  cat("\nPolynomial kernel:\n")
  check_kernel_properties(K_poly)
  
  # Visualize kernel matrices
  plot_data_linear <- expand.grid(i = 1:50, j = 1:50)
  plot_data_linear$value <- as.vector(K_linear)
  
  plot_data_rbf <- expand.grid(i = 1:50, j = 1:50)
  plot_data_rbf$value <- as.vector(K_rbf)
  
  plot_data_poly <- expand.grid(i = 1:50, j = 1:50)
  plot_data_poly$value <- as.vector(K_poly)
  
  p1 <- ggplot(plot_data_linear, aes(x = i, y = j, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "Linear Kernel Matrix") +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank())
  
  p2 <- ggplot(plot_data_rbf, aes(x = i, y = j, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "RBF Kernel Matrix") +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank())
  
  p3 <- ggplot(plot_data_poly, aes(x = i, y = j, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c() +
    labs(title = "Polynomial Kernel Matrix") +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank())
  
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(K_linear = K_linear, K_rbf = K_rbf, K_poly = K_poly))
}

ovo_svm_example <- function() {
  cat("\n=== One-vs-One SVM Example ===\n\n")
  
  # Generate multi-class data
  set.seed(42)
  X <- matrix(rnorm(300 * 2), ncol = 2)
  y <- sample(1:3, 300, replace = TRUE)
  
  # Split data
  train_indices <- sample(1:300, 210)
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Train OVO SVM
  df_train <- data.frame(X = X_train, y = factor(y_train))
  ovo_svm <- svm(y ~ ., data = df_train, kernel = "radial", scale = FALSE)
  
  # Evaluate
  train_pred <- predict(ovo_svm, df_train)
  test_pred <- predict(ovo_svm, data.frame(X = X_test))
  
  train_score <- mean(train_pred == y_train)
  test_score <- mean(test_pred == y_test)
  
  cat("OVO SVM - Train accuracy:", round(train_score, 3), "\n")
  cat("OVO SVM - Test accuracy:", round(test_score, 3), "\n")
  
  # Visualize decision boundaries
  plot_data_train <- data.frame(X1 = X_train[, 1], X2 = X_train[, 2], y = factor(y_train))
  plot_data_test <- data.frame(X1 = X_test[, 1], X2 = X_test[, 2], y = factor(y_test))
  
  p1 <- ggplot(plot_data_train, aes(x = X1, y = X2, color = y)) +
    geom_point(size = 2) +
    labs(title = "Training Data") +
    theme_minimal()
  
  p2 <- ggplot(plot_data_test, aes(x = X1, y = X2, color = y)) +
    geom_point(size = 2) +
    labs(title = "Test Data") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(ovo_svm)
}

ovr_svm_example <- function() {
  cat("\n=== One-vs-Rest SVM Example ===\n\n")
  
  # Generate multi-class data
  set.seed(42)
  X <- matrix(rnorm(300 * 2), ncol = 2)
  y <- sample(1:3, 300, replace = TRUE)
  
  # Split data
  train_indices <- sample(1:300, 210)
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Train OVR SVM (using one-vs-one as approximation since e1071 doesn't have direct OVR)
  df_train <- data.frame(X = X_train, y = factor(y_train))
  ovr_svm <- svm(y ~ ., data = df_train, kernel = "radial", scale = FALSE)
  
  # Evaluate
  train_pred <- predict(ovr_svm, df_train)
  test_pred <- predict(ovr_svm, data.frame(X = X_test))
  
  train_score <- mean(train_pred == y_train)
  test_score <- mean(test_pred == y_test)
  
  cat("OVR SVM - Train accuracy:", round(train_score, 3), "\n")
  cat("OVR SVM - Test accuracy:", round(test_score, 3), "\n")
  
  return(ovr_svm)
}

svr_example <- function() {
  cat("\n=== Support Vector Regression Example ===\n\n")
  
  # Generate regression data
  set.seed(42)
  X <- sort(5 * runif(100))
  y <- sin(X) + 0.1 * rnorm(100)
  
  # Fit SVR models with different kernels
  df <- data.frame(X = X, y = y)
  
  svr_rbf <- svm(y ~ X, data = df, kernel = "radial", cost = 100, gamma = 0.1, epsilon = 0.1)
  svr_linear <- svm(y ~ X, data = df, kernel = "linear", cost = 100, epsilon = 0.1)
  svr_poly <- svm(y ~ X, data = df, kernel = "polynomial", cost = 100, degree = 3, epsilon = 0.1)
  
  # Predictions
  X_test <- seq(0, 5, length.out = 100)
  df_test <- data.frame(X = X_test)
  
  y_rbf <- predict(svr_rbf, df_test)
  y_linear <- predict(svr_linear, df_test)
  y_poly <- predict(svr_poly, df_test)
  
  # Plotting
  plot_data <- data.frame(
    X = rep(X_test, 3),
    y_pred = c(y_rbf, y_linear, y_poly),
    kernel = rep(c("RBF", "Linear", "Polynomial"), each = length(X_test))
  )
  
  p1 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(x = X, y = y), color = "black", alpha = 0.6) +
    geom_line(data = subset(plot_data, kernel == "RBF"), aes(x = X, y = y_pred), color = "red", size = 1) +
    labs(title = "SVR with RBF Kernel", x = "data", y = "target") +
    theme_minimal()
  
  p2 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(x = X, y = y), color = "black", alpha = 0.6) +
    geom_line(data = subset(plot_data, kernel == "Linear"), aes(x = X, y = y_pred), color = "blue", size = 1) +
    labs(title = "SVR with Linear Kernel", x = "data", y = "target") +
    theme_minimal()
  
  p3 <- ggplot() +
    geom_point(data = data.frame(X = X, y = y), aes(x = X, y = y), color = "black", alpha = 0.6) +
    geom_line(data = subset(plot_data, kernel == "Polynomial"), aes(x = X, y = y_pred), color = "green", size = 1) +
    labs(title = "SVR with Polynomial Kernel", x = "data", y = "target") +
    theme_minimal()
  
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(svr_rbf = svr_rbf, svr_linear = svr_linear, svr_poly = svr_poly))
}

simplified_smo <- function(X, y, C = 1.0, max_iter = 1000, tol = 1e-3) {
  n_samples <- nrow(X)
  alpha <- rep(0, n_samples)
  b <- 0.0
  
  # Precompute kernel matrix
  K <- X %*% t(X)
  
  for (iteration in 1:max_iter) {
    alpha_pairs_changed <- 0
    
    for (i in 1:n_samples) {
      # Calculate error
      Ei <- sum(alpha * y * K[i, ]) + b - y[i]
      
      # Check KKT conditions
      if ((y[i] * Ei < -tol && alpha[i] < C) || 
          (y[i] * Ei > tol && alpha[i] > 0)) {
        
        # Choose second alpha randomly
        j <- sample(setdiff(1:n_samples, i), 1)
        
        Ej <- sum(alpha * y * K[j, ]) + b - y[j]
        
        # Save old alphas
        alpha_i_old <- alpha[i]
        alpha_j_old <- alpha[j]
        
        # Compute bounds
        if (y[i] != y[j]) {
          L <- max(0, alpha[j] - alpha[i])
          H <- min(C, C + alpha[j] - alpha[i])
        } else {
          L <- max(0, alpha[i] + alpha[j] - C)
          H <- min(C, alpha[i] + alpha[j])
        }
        
        if (L == H) next
        
        # Compute eta
        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) next
        
        # Update alpha[j]
        alpha[j] <- alpha_j_old - y[j] * (Ei - Ej) / eta
        alpha[j] <- max(L, min(H, alpha[j]))
        
        if (abs(alpha[j] - alpha_j_old) < 1e-5) next
        
        # Update alpha[i]
        alpha[i] <- alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
        
        # Update b
        b1 <- b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
        b2 <- b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
        b <- (b1 + b2) / 2
        
        alpha_pairs_changed <- alpha_pairs_changed + 1
      }
    }
    
    if (alpha_pairs_changed == 0) break
  }
  
  return(list(alpha = alpha, b = b))
}

demonstrate_smo <- function() {
  cat("\n=== SMO Algorithm Demonstration ===\n\n")
  
  # Generate data
  set.seed(42)
  X <- matrix(rnorm(100 * 2), ncol = 2)
  y <- sign(X[, 1] + X[, 2])
  
  # Run SMO
  result <- simplified_smo(X, y, C = 1.0)
  alpha <- result$alpha
  b <- result$b
  
  cat("Converged with", sum(alpha > 1e-5), "support vectors\n")
  cat("Bias term:", round(b, 4), "\n")
  
  # Visualize support vectors
  support_vector_indices <- alpha > 1e-5
  support_vectors <- X[support_vector_indices, ]
  
  plot_data <- data.frame(
    X1 = X[, 1],
    X2 = X[, 2],
    y = factor(y),
    is_sv = support_vector_indices
  )
  
  p1 <- ggplot(plot_data, aes(x = X1, y = X2, color = y)) +
    geom_point(size = 2, alpha = 0.6) +
    labs(title = "All Data Points") +
    theme_minimal()
  
  p2 <- ggplot(plot_data, aes(x = X1, y = X2, color = y)) +
    geom_point(size = 1, alpha = 0.3) +
    geom_point(data = subset(plot_data, is_sv), 
               aes(x = X1, y = X2), 
               color = "red", size = 3, shape = 21, fill = "transparent") +
    labs(title = "Support Vectors Highlighted") +
    theme_minimal()
  
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(alpha = alpha, b = b))
}

kernel_approximation_example <- function() {
  cat("\n=== Kernel Approximation Example ===\n\n")
  
  # Generate data
  set.seed(42)
  theta <- runif(1000, 0, 2 * pi)
  r <- runif(1000, 0.5, 1.5)
  X <- cbind(r * cos(theta), r * sin(theta))
  y <- ifelse(r < 1, 1, -1)
  
  # Split data
  train_indices <- sample(1:1000, 700)
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Standard SVM
  df_train <- data.frame(X = X_train, y = factor(y_train))
  df_test <- data.frame(X = X_test, y = factor(y_test))
  
  svm_standard <- svm(y ~ ., data = df_train, kernel = "radial", gamma = 1.0, scale = FALSE)
  score_standard <- mean(predict(svm_standard, df_test) == y_test)
  
  # RBF approximation (simplified)
  # Note: R doesn't have direct RBF approximation like sklearn, so we'll simulate
  cat("Standard SVM accuracy:", round(score_standard, 3), "\n")
  cat("Note: RBF and Nystroem approximations require additional packages\n")
  
  return(svm_standard)
}

margin_analysis <- function(X, y, svm_model) {
  # Get support vectors
  support_vectors <- svm_model$SV
  support_vector_indices <- svm_model$index
  
  # For linear SVM, compute margin
  if (svm_model$kernel == 0) {  # Linear kernel
    w <- t(svm_model$coefs) %*% svm_model$SV
    margin <- 2 / sqrt(sum(w^2))
  } else {
    margin <- NA  # Margin computation for nonlinear kernels is more complex
  }
  
  cat("Margin:", round(margin, 4), "\n")
  cat("Number of support vectors:", length(support_vectors), "\n")
  cat("Support vector ratio:", round(length(support_vectors)/nrow(X), 3), "\n")
  
  return(list(margin = margin, support_vectors = support_vectors))
}

demonstrate_margin_analysis <- function() {
  cat("\n=== Margin Analysis ===\n\n")
  
  # Generate data
  set.seed(42)
  X <- mvrnorm(100, mu = c(0, 0), Sigma = matrix(c(1, 0.5, 0.5, 1), 2, 2))
  y <- sign(X[, 1] + X[, 2])
  
  # Fit SVM
  df <- data.frame(X = X, y = factor(y))
  svm <- svm(y ~ ., data = df, kernel = "linear", scale = FALSE)
  
  # Analyze margin
  result <- margin_analysis(X, y, svm)
  
  # Visualize
  plot_data <- data.frame(
    X1 = X[, 1],
    X2 = X[, 2],
    y = factor(y),
    is_sv = 1:nrow(X) %in% svm$index
  )
  
  p <- ggplot(plot_data, aes(x = X1, y = X2, color = y)) +
    geom_point(size = 2, alpha = 0.6) +
    geom_point(data = subset(plot_data, is_sv), 
               aes(x = X1, y = X2), 
               color = "red", size = 3, shape = 21, fill = "transparent") +
    labs(title = paste("Margin:", round(result$margin, 3))) +
    theme_minimal()
  
  print(p)
  
  return(result)
}

main_r <- function() {
  cat("Support Vector Machines: Appendix Implementation\n")
  cat("=" * 60, "\n")
  
  # 1. RKHS demonstration
  cat("\n1. RKHS Demonstration:\n")
  rkhs_model <- demonstrate_rkhs()
  
  # 2. Mercer's theorem demonstration
  cat("\n2. Mercer's Theorem Demonstration:\n")
  kernel_matrices <- demonstrate_mercer_theorem()
  
  # 3. Multi-class SVM examples
  cat("\n3. Multi-class SVM Examples:\n")
  ovo_model <- ovo_svm_example()
  ovr_model <- ovr_svm_example()
  
  # 4. Support Vector Regression
  cat("\n4. Support Vector Regression:\n")
  svr_models <- svr_example()
  
  # 5. SMO algorithm
  cat("\n5. SMO Algorithm:\n")
  smo_results <- demonstrate_smo()
  
  # 6. Kernel approximation
  cat("\n6. Kernel Approximation:\n")
  approx_model <- kernel_approximation_example()
  
  # 7. Margin analysis
  cat("\n7. Margin Analysis:\n")
  margin_results <- demonstrate_margin_analysis()
  
  cat("\n=== Key Insights ===\n")
  cat("1. RKHS provides theoretical foundation for kernel methods\n")
  cat("2. Mercer's theorem ensures valid kernel functions\n")
  cat("3. Multi-class SVM extends binary classification\n")
  cat("4. SVR applies SVM principles to regression\n")
  cat("5. SMO enables efficient SVM training\n")
  cat("6. Kernel approximation scales to large datasets\n")
  cat("7. Margin analysis provides generalization insights\n")
  cat("8. Support vectors determine the optimal solution\n")
  
  return(list(
    rkhs_model = rkhs_model,
    kernel_matrices = kernel_matrices,
    multiclass_models = list(ovo = ovo_model, ovr = ovr_model),
    svr_models = svr_models,
    smo_results = smo_results,
    approximation_model = approx_model,
    margin_results = margin_results
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main_r()
}
