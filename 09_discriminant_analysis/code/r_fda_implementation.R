# Fisher Discriminant Analysis in R
library(MASS)
library(ggplot2)
library(gridExtra)
library(caret)

# Custom FDA implementation
fisher_discriminant_analysis <- function(X, y, n_components = NULL) {
  # Get unique classes
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Set number of components
  if (is.null(n_components)) {
    n_components <- min(n_classes - 1, n_features)
  }
  
  # Calculate class means and counts
  class_means <- matrix(0, nrow = n_classes, ncol = n_features)
  class_counts <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_means[i,] <- colMeans(X[class_mask,, drop = FALSE])
    class_counts[i] <- sum(class_mask)
  }
  
  # Overall mean
  overall_mean <- colSums(class_means * class_counts) / sum(class_counts)
  
  # Between-class scatter matrix
  B <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    diff <- class_means[i,] - overall_mean
    B <- B + class_counts[i] * outer(diff, diff)
  }
  B <- B / (n_classes - 1)
  
  # Within-class scatter matrix
  W <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    diff <- sweep(class_data, 2, class_means[i,], "-")
    W <- W + t(diff) %*% diff
  }
  W <- W / (n_samples - n_classes)
  
  # Solve generalized eigenvalue problem
  tryCatch({
    W_inv <- solve(W)
    eigen_result <- eigen(W_inv %*% B)
    
    # Sort eigenvalues and eigenvectors
    idx <- order(eigen_result$values, decreasing = TRUE)
    eigenvals <- eigen_result$values[idx]
    eigenvecs <- eigen_result$vectors[, idx]
    
    # Select components
    scalings <- eigenvecs[, 1:n_components, drop = FALSE]
    explained_variance_ratio <- eigenvals[1:n_components]
    
  }, error = function(e) {
    # Handle singular matrix
    cat("Warning: Singular within-class scatter matrix. Using regularization.\n")
    W_reg <- W + 1e-6 * diag(n_features)
    W_inv <- solve(W_reg)
    eigen_result <- eigen(W_inv %*% B)
    
    idx <- order(eigen_result$values, decreasing = TRUE)
    eigenvals <- eigen_result$values[idx]
    eigenvecs <- eigen_result$vectors[, idx]
    
    scalings <- eigenvecs[, 1:n_components, drop = FALSE]
    explained_variance_ratio <- eigenvals[1:n_components]
  })
  
  return(list(
    scalings = scalings,
    explained_variance_ratio = explained_variance_ratio,
    classes = classes,
    n_components = n_components
  ))
}

# Transform function
transform_fda <- function(model, X) {
  return(X %*% model$scalings)
}

# Calculate separation criterion
calculate_separation <- function(X_proj, y) {
  classes <- unique(y)
  overall_mean <- mean(X_proj)
  
  # Between-class variance
  between_var <- 0
  for (c in classes) {
    class_mean <- mean(X_proj[y == c])
    n_class <- sum(y == c)
    between_var <- between_var + n_class * (class_mean - overall_mean)^2
  }
  
  # Within-class variance
  within_var <- 0
  for (c in classes) {
    class_data <- X_proj[y == c]
    class_mean <- mean(class_data)
    within_var <- within_var + sum((class_data - class_mean)^2)
  }
  
  return(ifelse(within_var > 0, between_var / within_var, 0))
}

# Generate toy data
generate_toy_data <- function(n_samples = 300, seed = 42) {
  set.seed(seed)
  
  # Generate 3 classes with different means
  n_per_class <- n_samples %/% 3
  
  # Class 0: centered at (0, 0)
  X0 <- MASS::mvrnorm(n_per_class, mu = c(0, 0), 
                      Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  
  # Class 1: centered at (3, 2)
  X1 <- MASS::mvrnorm(n_per_class, mu = c(3, 2), 
                      Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  
  # Class 2: centered at (1, 4)
  X2 <- MASS::mvrnorm(n_per_class, mu = c(1, 4), 
                      Sigma = matrix(c(1, 0.5, 0.5, 1), nrow = 2))
  
  X <- rbind(X0, X1, X2)
  y <- rep(c(0, 1, 2), each = n_per_class)
  
  return(list(X = X, y = y))
}

# Compare PCA and FDA
compare_pca_fda <- function() {
  # Generate data
  data <- generate_toy_data()
  X <- data$X
  y <- data$y
  
  # Apply PCA
  pca_result <- prcomp(X, center = TRUE, scale = FALSE)
  X_pca <- pca_result$x[, 1, drop = FALSE]
  
  # Apply FDA (via LDA)
  lda_result <- lda(X, y)
  X_fda <- predict(lda_result, X)$x
  
  # Calculate separations
  pca_separation <- calculate_separation(X_pca, y)
  fda_separation <- calculate_separation(X_fda, y)
  
  cat("Class Separation Analysis:\n")
  cat("-" * 40, "\n")
  cat("PCA Separation:", round(pca_separation, 4), "\n")
  cat("FDA Separation:", round(fda_separation, 4), "\n")
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  df_pca <- data.frame(
    x = X_pca,
    class = factor(y)
  )
  
  df_fda <- data.frame(
    x = X_fda,
    class = factor(y)
  )
  
  # Original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Original Data", color = "Class") +
    theme_minimal()
  
  # PCA projection
  p2 <- ggplot(df_pca, aes(x = x, y = 0, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "PCA Projection (1D)", color = "Class") +
    theme_minimal() +
    ylim(-0.1, 0.1)
  
  # FDA projection
  p3 <- ggplot(df_fda, aes(x = x, y = 0, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "FDA Projection (1D)", color = "Class") +
    theme_minimal() +
    ylim(-0.1, 0.1)
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(pca = pca_result, lda = lda_result))
}

# Demonstrate FDA
demonstrate_fda_r <- function() {
  # Generate synthetic data
  set.seed(42)
  n_samples <- 300
  n_features <- 4
  
  # Generate 3 classes
  n_per_class <- n_samples %/% 3
  
  # Class 0
  X0 <- MASS::mvrnorm(n_per_class, mu = c(0, 0, 0, 0), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  # Class 1
  X1 <- MASS::mvrnorm(n_per_class, mu = c(3, 2, 1, 0), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  # Class 2
  X2 <- MASS::mvrnorm(n_per_class, mu = c(1, 4, 2, 3), 
                      Sigma = matrix(c(1, 0.5, 0.3, 0.2,
                                      0.5, 1, 0.4, 0.3,
                                      0.3, 0.4, 1, 0.5,
                                      0.2, 0.3, 0.5, 1), nrow = 4))
  
  X <- rbind(X0, X1, X2)
  y <- rep(c(0, 1, 2), each = n_per_class)
  
  # Apply our FDA
  fda_model <- fisher_discriminant_analysis(X, y, n_components = 2)
  X_fda <- transform_fda(fda_model, X)
  
  # Apply MASS LDA
  lda_model <- lda(X, y)
  X_lda <- predict(lda_model, X)$x
  
  # Create visualizations
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  df_fda <- data.frame(
    x1 = X_fda[,1],
    x2 = X_fda[,2],
    class = factor(y)
  )
  
  df_lda <- data.frame(
    x1 = X_lda[,1],
    x2 = X_lda[,2],
    class = factor(y)
  )
  
  # Plot original data
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Original Data (First 2 Dimensions)", color = "Class") +
    theme_minimal()
  
  # Plot our FDA
  p2 <- ggplot(df_fda, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Our FDA Projection", color = "Class") +
    theme_minimal()
  
  # Plot MASS LDA
  p3 <- ggplot(df_lda, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "MASS LDA Projection", color = "Class") +
    theme_minimal()
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  # Print results
  cat("FDA Results:\n")
  cat("-" * 30, "\n")
  cat("Number of components:", fda_model$n_components, "\n")
  cat("Explained variance ratios:", fda_model$explained_variance_ratio, "\n")
  cat("Discriminant directions shape:", dim(fda_model$scalings), "\n")
  
  # Calculate separation
  separation_our <- calculate_separation(X_fda[,1], y)
  separation_mass <- calculate_separation(X_lda[,1], y)
  
  cat("\nSeparation Analysis:\n")
  cat("Our FDA:", round(separation_our, 4), "\n")
  cat("MASS LDA:", round(separation_mass, 4), "\n")
  
  return(list(fda_model = fda_model, lda_model = lda_model))
}

# Demonstrate overfitting
demonstrate_overfitting <- function() {
  set.seed(42)
  
  # Generate high-dimensional data with random features
  n_samples <- 20
  n_features <- 50  # Much larger than n_samples
  
  # Random features
  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples)
  
  # Binary labels
  y <- sample(0:1, n_samples, replace = TRUE)
  
  # Apply FDA
  fda_model <- fisher_discriminant_analysis(X, y, n_components = 1)
  X_fda <- transform_fda(fda_model, X)
  
  # Calculate separation
  separation <- calculate_separation(X_fda, y)
  
  cat("High-dimensional FDA Results:\n")
  cat("n_samples:", n_samples, "\n")
  cat("n_features:", n_features, "\n")
  cat("Separation:", round(separation, 4), "\n")
  cat("Perfect separation achieved:", separation > 100, "\n")
  
  # Visualize projection
  df_high <- data.frame(
    x = X_fda,
    class = factor(y)
  )
  
  p1 <- ggplot(df_high, aes(x = x, y = 0, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "FDA Projection (Random Features)", color = "Class") +
    theme_minimal() +
    ylim(-0.1, 0.1)
  
  # Compare with low-dimensional case
  X_low <- X[, 1:5]  # Use only first 5 features
  fda_low <- fisher_discriminant_analysis(X_low, y, n_components = 1)
  X_fda_low <- transform_fda(fda_low, X_low)
  separation_low <- calculate_separation(X_fda_low, y)
  
  df_low <- data.frame(
    x = X_fda_low,
    class = factor(y)
  )
  
  p2 <- ggplot(df_low, aes(x = x, y = 0, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = paste("FDA Projection (5 Features)\nSeparation:", round(separation_low, 4)), 
         color = "Class") +
    theme_minimal() +
    ylim(-0.1, 0.1)
  
  # Display plots
  grid.arrange(p1, p2, ncol = 2)
  
  return(list(separation = separation, separation_low = separation_low))
}

# Regularized FDA
regularized_fda <- function(X, y, alpha = 0.1, n_components = NULL) {
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  if (is.null(n_components)) {
    n_components <- min(n_classes - 1, n_features)
  }
  
  # Calculate scatter matrices
  B, W <- calculate_scatter_matrices(X, y)
  
  # Regularize W
  W_reg <- W + alpha * diag(n_features)
  
  # Solve eigenvalue problem
  W_inv <- solve(W_reg)
  eigen_result <- eigen(W_inv %*% B)
  
  # Sort and select
  idx <- order(eigen_result$values, decreasing = TRUE)
  eigenvals <- eigen_result$values[idx]
  eigenvecs <- eigen_result$vectors[, idx]
  
  scalings <- eigenvecs[, 1:n_components, drop = FALSE]
  
  return(list(scalings = scalings, eigenvals = eigenvals[1:n_components]))
}

# Calculate scatter matrices
calculate_scatter_matrices <- function(X, y) {
  classes <- unique(y)
  n_classes <- length(classes)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  
  # Class means and counts
  class_means <- matrix(0, nrow = n_classes, ncol = n_features)
  class_counts <- rep(0, n_classes)
  
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_means[i,] <- colMeans(X[class_mask,, drop = FALSE])
    class_counts[i] <- sum(class_mask)
  }
  
  overall_mean <- colSums(class_means * class_counts) / sum(class_counts)
  
  # Between-class scatter
  B <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    diff <- class_means[i,] - overall_mean
    B <- B + class_counts[i] * outer(diff, diff)
  }
  B <- B / (n_classes - 1)
  
  # Within-class scatter
  W <- matrix(0, nrow = n_features, ncol = n_features)
  for (i in 1:n_classes) {
    class_mask <- y == classes[i]
    class_data <- X[class_mask,, drop = FALSE]
    diff <- sweep(class_data, 2, class_means[i,], "-")
    W <- W + t(diff) %*% diff
  }
  W <- W / (n_samples - n_classes)
  
  return(list(B = B, W = W))
}

# FDA with feature selection
fda_with_feature_selection <- function(X, y, n_features = 10, n_components = NULL) {
  # Simple feature selection based on F-statistic
  f_scores <- rep(0, ncol(X))
  for (i in 1:ncol(X)) {
    # Calculate F-statistic for each feature
    feature_data <- X[, i]
    classes <- unique(y)
    
    # Between-group variance
    overall_mean <- mean(feature_data)
    between_var <- 0
    for (c in classes) {
      class_mean <- mean(feature_data[y == c])
      n_class <- sum(y == c)
      between_var <- between_var + n_class * (class_mean - overall_mean)^2
    }
    
    # Within-group variance
    within_var <- 0
    for (c in classes) {
      class_data <- feature_data[y == c]
      class_mean <- mean(class_data)
      within_var <- within_var + sum((class_data - class_mean)^2)
    }
    
    f_scores[i] <- between_var / within_var
  }
  
  # Select top features
  selected_features <- order(f_scores, decreasing = TRUE)[1:n_features]
  X_selected <- X[, selected_features, drop = FALSE]
  
  # Apply FDA
  fda_model <- fisher_discriminant_analysis(X_selected, y, n_components = n_components)
  X_fda <- transform_fda(fda_model, X_selected)
  
  return(list(X_fda = X_fda, fda_model = fda_model, selected_features = selected_features))
}

# Cross-validate FDA
cross_validate_fda <- function(X, y, n_splits = 5) {
  set.seed(42)
  folds <- createFolds(y, k = n_splits)
  separations <- numeric(n_splits)
  
  for (i in 1:n_splits) {
    train_idx <- unlist(folds[-i])
    test_idx <- folds[[i]]
    
    X_train <- X[train_idx,, drop = FALSE]
    X_test <- X[test_idx,, drop = FALSE]
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    # Fit FDA on training data
    fda_model <- fisher_discriminant_analysis(X_train, y_train)
    
    # Transform test data
    X_test_fda <- transform_fda(fda_model, X_test)
    
    # Calculate separation on test data
    separation <- calculate_separation(X_test_fda, y_test)
    separations[i] <- separation
  }
  
  return(list(mean_separation = mean(separations), std_separation = sd(separations)))
}

# Face recognition example (simplified)
face_recognition_fda <- function() {
  # Simulate face data (since we don't have the actual dataset)
  set.seed(42)
  n_samples <- 400
  n_features <- 4096  # 64x64 pixels
  
  # Generate synthetic face data
  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples)
  
  # Create 40 classes (persons)
  y <- rep(1:40, each = 10)
  
  # Split data
  train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X[train_idx,]
  X_test <- X[-train_idx,]
  y_train <- y[train_idx]
  y_test <- y[-train_idx]
  
  # Apply FDA
  fda_model <- fisher_discriminant_analysis(X_train, y_train, n_components = 39)
  X_train_fda <- transform_fda(fda_model, X_train)
  X_test_fda <- transform_fda(fda_model, X_test)
  
  # Simple classification (nearest neighbor)
  accuracy <- 0
  for (i in 1:nrow(X_test_fda)) {
    distances <- sqrt(colSums((t(X_train_fda) - X_test_fda[i,])^2))
    pred_class <- y_train[which.min(distances)]
    if (pred_class == y_test[i]) accuracy <- accuracy + 1
  }
  accuracy <- accuracy / length(y_test)
  
  cat("Face Recognition Accuracy:", round(accuracy, 4), "\n")
  
  return(list(fda_model = fda_model, accuracy = accuracy))
}

# Gene expression example
gene_expression_fda <- function() {
  set.seed(42)
  n_samples <- 100
  n_genes <- 1000
  
  # Generate data with some discriminative genes
  X <- matrix(rnorm(n_samples * n_genes), nrow = n_samples)
  
  # Add discriminative signal to first 50 genes
  X[1:50, 1:50] <- X[1:50, 1:50] + 2  # Class 0
  X[51:100, 1:50] <- X[51:100, 1:50] - 2  # Class 1
  
  y <- rep(c(0, 1), each = 50)
  
  # Apply FDA with feature selection
  result <- fda_with_feature_selection(X, y, n_features = 100, n_components = 1)
  
  # Visualize results
  df_original <- data.frame(
    x1 = X[,1],
    x2 = X[,2],
    class = factor(y)
  )
  
  df_fda <- data.frame(
    x = result$X_fda,
    class = factor(y)
  )
  
  # Original data (first 2 genes)
  p1 <- ggplot(df_original, aes(x = x1, y = x2, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "Original Data (First 2 Genes)", color = "Class") +
    theme_minimal()
  
  # Selected features
  selected_features <- result$selected_features
  feature_selection_df <- data.frame(
    gene_index = 1:100,
    selected = 1:100 %in% selected_features[1:100]
  )
  
  p2 <- ggplot(feature_selection_df, aes(x = gene_index, y = selected)) +
    geom_bar(stat = "identity") +
    labs(title = "Feature Selection", x = "Gene Index", y = "Selected") +
    theme_minimal()
  
  # FDA projection
  p3 <- ggplot(df_fda, aes(x = x, y = 0, color = class)) +
    geom_point(alpha = 0.7) +
    labs(title = "FDA Projection", color = "Class") +
    theme_minimal() +
    ylim(-0.1, 0.1)
  
  # Display plots
  grid.arrange(p1, p2, p3, ncol = 3)
  
  return(list(fda_model = result$fda_model, selected_features = selected_features))
}

# Plot FDA directions
plot_fda_directions <- function(X, y, fda_model, title = "FDA Discriminant Directions") {
  # Get discriminant directions
  directions <- fda_model$scalings
  explained_var <- fda_model$explained_variance_ratio
  
  # Create visualization
  par(mfrow = c(2, 2))
  
  # Original data
  plot(X[,1], X[,2], col = y + 1, pch = 16, main = "Original Data",
       xlab = "Feature 1", ylab = "Feature 2")
  
  # Explained variance ratio
  barplot(explained_var, main = "Explained Variance Ratio",
          xlab = "Component", ylab = "Variance Ratio")
  
  # First discriminant direction
  if (ncol(directions) >= 1) {
    barplot(directions[,1], main = "First Discriminant Direction",
            xlab = "Feature", ylab = "Coefficient")
  }
  
  # Second discriminant direction (if available)
  if (ncol(directions) >= 2) {
    barplot(directions[,2], main = "Second Discriminant Direction",
            xlab = "Feature", ylab = "Coefficient")
  }
  
  par(mfrow = c(1, 1))
}

# Main function to demonstrate FDA implementation
main <- function() {
  cat("Fisher Discriminant Analysis Demonstration\n")
  cat("=", 50, "\n")
  
  # Compare PCA vs FDA
  cat("\n1. PCA vs FDA Comparison:\n")
  comparison <- compare_pca_fda()
  
  # Demonstrate FDA from scratch
  cat("\n2. FDA Implementation from Scratch:\n")
  fda_demo <- demonstrate_fda_r()
  
  # Demonstrate overfitting
  cat("\n3. Overfitting Demonstration:\n")
  overfitting <- demonstrate_overfitting()
  
  # Face recognition example
  cat("\n4. Face Recognition Example:\n")
  face_result <- face_recognition_fda()
  
  # Gene expression example
  cat("\n5. Gene Expression Analysis:\n")
  gene_result <- gene_expression_fda()
  
  # Generate data for cross-validation
  data <- generate_toy_data(n_samples = 300)
  X <- data$X
  y <- data$y
  
  # Cross-validation
  cat("\n6. Cross-Validation:\n")
  cv_result <- cross_validate_fda(X, y, n_splits = 5)
  cat("Cross-validation separation:", round(cv_result$mean_separation, 4), 
      "(+/-", round(cv_result$std_separation, 4), ")\n")
  
  # Plot FDA directions
  cat("\n7. FDA Directions Analysis:\n")
  plot_fda_directions(X, y, fda_demo$fda_model)
  
  return(list(
    fda_model = fda_demo$fda_model,
    lda_model = fda_demo$lda_model,
    face_result = face_result,
    gene_result = gene_result
  ))
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}
