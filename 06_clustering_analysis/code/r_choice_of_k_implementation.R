# Choice of K Implementation in R
# ===============================
#
# This script provides comprehensive implementations of methods for determining
# the optimal number of clusters K, including gap statistics, silhouette analysis,
# and prediction strength.

library(stats)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)

gap_statistic <- function(X, K_range, B = 10, method = "uniform", random_state = 42) {
  """
  Compute gap statistic for determining optimal number of clusters.
  
  Parameters:
  -----------
  X : matrix
      Input data
  K_range : vector
      Range of K values to test
  B : integer, default=10
      Number of reference datasets
  method : character, default="uniform"
      Method for generating reference data: "uniform" or "pca"
  random_state : integer, default=42
      Random seed for reproducibility
  
  Returns:
  --------
  gap_scores : vector
      Gap statistic values for each K
  gap_errors : vector
      Standard errors for gap statistics
  """
  set.seed(random_state)
  n_samples <- nrow(X)
  n_features <- ncol(X)
  gap_scores <- numeric(length(K_range))
  gap_errors <- numeric(length(K_range))
  
  # Compute observed sum of squares for each K
  ss_obs <- numeric(length(K_range))
  for (i in seq_along(K_range)) {
    K <- K_range[i]
    km <- kmeans(X, centers = K, nstart = 10)
    ss_obs[i] <- km$tot.withinss
  }
  
  # Generate reference datasets
  if (method == "uniform") {
    # Uniform sampling over the range of observed data
    min_vals <- apply(X, 2, min)
    max_vals <- apply(X, 2, max)
    reference_data <- list()
    for (b in 1:B) {
      ref_sample <- matrix(runif(n_samples * n_features, min = min_vals, max = max_vals), 
                          ncol = n_features, byrow = TRUE)
      reference_data[[b]] <- ref_sample
    }
  } else if (method == "pca") {
    # PCA-based sampling
    pca_result <- prcomp(X, center = TRUE, scale. = TRUE)
    X_pca <- pca_result$x
    min_vals <- apply(X_pca, 2, min)
    max_vals <- apply(X_pca, 2, max)
    reference_data <- list()
    for (b in 1:B) {
      ref_sample_pca <- matrix(runif(n_samples * n_features, min = min_vals, max = max_vals), 
                              ncol = n_features, byrow = TRUE)
      ref_sample <- ref_sample_pca %*% t(pca_result$rotation) + 
                   matrix(pca_result$center, n_samples, n_features, byrow = TRUE)
      reference_data[[b]] <- ref_sample
    }
  }
  
  # Compute reference sum of squares for each K and each reference dataset
  ss_ref <- matrix(0, length(K_range), B)
  for (b in 1:B) {
    ref_data <- reference_data[[b]]
    for (i in seq_along(K_range)) {
      K <- K_range[i]
      km <- kmeans(ref_data, centers = K, nstart = 10)
      ss_ref[i, b] <- km$tot.withinss
    }
  }
  
  # Compute gap statistic
  for (i in seq_along(K_range)) {
    log_ss_ref <- log(ss_ref[i, ])
    log_ss_obs <- log(ss_obs[i])
    
    gap_scores[i] <- mean(log_ss_ref) - log_ss_obs
    
    # Standard error
    gap_errors[i] <- sd(log_ss_ref) * sqrt(1 + 1/B)
  }
  
  list(gap_scores = gap_scores, gap_errors = gap_errors)
}

find_optimal_k_gap <- function(gap_scores, gap_errors, K_range) {
  """
  Find optimal K using gap statistic with one-standard-error rule.
  
  Parameters:
  -----------
  gap_scores : vector
      Gap statistic values
  gap_errors : vector
      Standard errors for gap statistics
  K_range : vector
      Range of K values tested
  
  Returns:
  --------
  optimal_k : integer
      Optimal number of clusters
  """
  # Find K where gap(K) >= gap(K+1) - se(K+1)
  for (i in 1:(length(K_range) - 1)) {
    if (gap_scores[i] >= gap_scores[i + 1] - gap_errors[i + 1]) {
      return(K_range[i])
    }
  }
  
  # If no clear elbow, return K with maximum gap
  return(K_range[which.max(gap_scores)])
}

silhouette_analysis <- function(X, K_range, random_state = 42) {
  """
  Perform silhouette analysis for determining optimal K.
  
  Parameters:
  -----------
  X : matrix
      Input data
  K_range : vector
      Range of K values to test
  random_state : integer, default=42
      Random seed for reproducibility
  
  Returns:
  --------
  silhouette_scores : vector
      Average silhouette scores for each K
  silhouette_samples_dict : list
      Individual silhouette scores for each K
  """
  set.seed(random_state)
  silhouette_scores <- numeric(length(K_range))
  silhouette_samples_dict <- list()
  
  for (i in seq_along(K_range)) {
    K <- K_range[i]
    km <- kmeans(X, centers = K, nstart = 10)
    cluster_labels <- km$cluster
    
    # Compute silhouette scores
    sil_result <- silhouette(cluster_labels, dist(X))
    silhouette_scores[i] <- mean(sil_result[, 3])
    silhouette_samples_dict[[as.character(K)]] <- sil_result[, 3]
  }
  
  list(silhouette_scores = silhouette_scores, 
       silhouette_samples_dict = silhouette_samples_dict)
}

prediction_strength <- function(X, K, n_splits = 5, threshold = 0.8, random_state = 42) {
  """
  Compute prediction strength for a given K.
  
  Parameters:
  -----------
  X : matrix
      Input data
  K : integer
      Number of clusters
  n_splits : integer, default=5
      Number of data splits for averaging
  threshold : numeric, default=0.8
      Threshold for prediction strength
  random_state : integer, default=42
      Random seed for reproducibility
  
  Returns:
  --------
  ps_score : numeric
      Average prediction strength score
  """
  set.seed(random_state)
  n_samples <- nrow(X)
  ps_scores <- numeric(0)
  
  for (split in 1:n_splits) {
    # Split data randomly
    idx <- sample(n_samples)
    split_point <- n_samples %/% 2
    A <- X[idx[1:split_point], , drop = FALSE]
    B <- X[idx[(split_point + 1):n_samples], , drop = FALSE]
    
    # Cluster B
    km_B <- kmeans(B, centers = K, nstart = 10)
    labels_B <- km_B$cluster
    
    # Cluster A and predict labels for B
    km_A <- kmeans(A, centers = K, nstart = 10)
    labels_B_pred <- predict_kmeans(km_A, B)
    
    # Compute prediction strength for each cluster
    ps_j <- numeric(0)
    for (j in 1:K) {
      members <- which(labels_B == j)
      if (length(members) < 2) next
      
      # Count pairs that agree in both clusterings
      pairs <- combn(members, 2, simplify = FALSE)
      agree <- sum(sapply(pairs, function(pair) {
        labels_B_pred[pair[1]] == labels_B_pred[pair[2]]
      }))
      ps_j <- c(ps_j, agree / length(pairs))
    }
    
    if (length(ps_j) > 0) {
      ps_scores <- c(ps_scores, min(ps_j))
    }
  }
  
  if (length(ps_scores) > 0) mean(ps_scores) else 0.0
}

predict_kmeans <- function(km_model, X) {
  """
  Predict cluster labels for new data using fitted kmeans model.
  """
  distances <- sapply(1:km_model$centers, function(k) {
    rowSums((X - matrix(km_model$centers[k, ], nrow(X), ncol(X), byrow = TRUE))^2)
  })
  apply(distances, 1, which.min)
}

compute_prediction_strength_range <- function(X, K_range, n_splits = 5, threshold = 0.8, random_state = 42) {
  """
  Compute prediction strength for a range of K values.
  
  Parameters:
  -----------
  X : matrix
      Input data
  K_range : vector
      Range of K values to test
  n_splits : integer, default=5
      Number of data splits for averaging
  threshold : numeric, default=0.8
      Threshold for prediction strength
  random_state : integer, default=42
      Random seed for reproducibility
  
  Returns:
  --------
  ps_scores : vector
      Prediction strength scores for each K
  """
  ps_scores <- numeric(length(K_range))
  for (i in seq_along(K_range)) {
    K <- K_range[i]
    ps_scores[i] <- prediction_strength(X, K, n_splits, threshold, random_state)
  }
  ps_scores
}

find_optimal_k_prediction_strength <- function(ps_scores, K_range, threshold = 0.8) {
  """
  Find optimal K using prediction strength.
  
  Parameters:
  -----------
  ps_scores : vector
      Prediction strength scores
  K_range : vector
      Range of K values tested
  threshold : numeric, default=0.8
      Threshold for prediction strength
  
  Returns:
  --------
  optimal_k : integer
      Optimal number of clusters
  """
  # Find the largest K where PS(K) >= threshold
  for (i in length(K_range):1) {
    if (ps_scores[i] >= threshold) {
      return(K_range[i])
    }
  }
  
  K_range[1]  # Default to smallest K if none meet threshold
}

plot_gap_statistic <- function(K_range, gap_scores, gap_errors, optimal_k = NULL) {
  """
  Plot gap statistic results.
  
  Parameters:
  -----------
  K_range : vector
      Range of K values tested
  gap_scores : vector
      Gap statistic values
  gap_errors : vector
      Standard errors for gap statistics
  optimal_k : integer, optional
      Optimal K value to highlight
  """
  df <- data.frame(
    K = K_range,
    gap = gap_scores,
    error = gap_errors
  )
  
  p <- ggplot(df, aes(x = K, y = gap)) +
    geom_errorbar(aes(ymin = gap - error, ymax = gap + error), width = 0.2) +
    geom_point(size = 3) +
    labs(x = "Number of Clusters (K)", y = "Gap Statistic",
         title = "Gap Statistic for Optimal K Selection") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  if (!is.null(optimal_k)) {
    p <- p + geom_vline(xintercept = optimal_k, color = "red", linetype = "dashed",
                       aes(label = paste("Optimal K =", optimal_k)))
  }
  
  print(p)
}

plot_silhouette_analysis <- function(K_range, silhouette_scores, silhouette_samples_dict, optimal_k = NULL) {
  """
  Plot silhouette analysis results.
  
  Parameters:
  -----------
  K_range : vector
      Range of K values tested
  silhouette_scores : vector
      Average silhouette scores
  silhouette_samples_dict : list
      Individual silhouette scores for each K
  optimal_k : integer, optional
      Optimal K value to highlight
  """
  # Plot average silhouette scores
  df_scores <- data.frame(
    K = K_range,
    silhouette = silhouette_scores
  )
  
  p1 <- ggplot(df_scores, aes(x = K, y = silhouette)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    labs(x = "Number of Clusters (K)", y = "Average Silhouette Score",
         title = "Silhouette Analysis for Optimal K Selection") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  if (!is.null(optimal_k)) {
    p1 <- p1 + geom_vline(xintercept = optimal_k, color = "red", linetype = "dashed")
  }
  
  print(p1)
  
  # Plot silhouette distribution for optimal K (if provided)
  if (!is.null(optimal_k) && as.character(optimal_k) %in% names(silhouette_samples_dict)) {
    sample_silhouette_values <- silhouette_samples_dict[[as.character(optimal_k)]]
    
    # Create silhouette plot
    sil_result <- silhouette(rep(1:optimal_k, each = length(sample_silhouette_values) / optimal_k), 
                           dist(matrix(sample_silhouette_values, ncol = 1)))
    
    p2 <- fviz_silhouette(sil_result, palette = "viridis") +
      labs(title = paste("Silhouette Plot for K =", optimal_k)) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    print(p2)
  }
}

plot_prediction_strength <- function(K_range, ps_scores, threshold = 0.8, optimal_k = NULL) {
  """
  Plot prediction strength results.
  
  Parameters:
  -----------
  K_range : vector
      Range of K values tested
  ps_scores : vector
      Prediction strength scores
  threshold : numeric, default=0.8
      Threshold line
  optimal_k : integer, optional
      Optimal K value to highlight
  """
  df <- data.frame(
    K = K_range,
    prediction_strength = ps_scores
  )
  
  p <- ggplot(df, aes(x = K, y = prediction_strength)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    geom_hline(yintercept = threshold, color = "red", linetype = "dashed",
              aes(label = paste("Threshold =", threshold))) +
    labs(x = "Number of Clusters (K)", y = "Prediction Strength",
         title = "Prediction Strength for Optimal K Selection") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  if (!is.null(optimal_k)) {
    p <- p + geom_vline(xintercept = optimal_k, color = "red", linetype = "dashed")
  }
  
  print(p)
}

comprehensive_k_selection <- function(X, K_range, methods = c("gap", "silhouette", "prediction_strength"), 
                                   random_state = 42, threshold = 0.8) {
  """
  Perform comprehensive K selection using multiple methods.
  
  Parameters:
  -----------
  X : matrix
      Input data
  K_range : vector
      Range of K values to test
  methods : vector, default=c("gap", "silhouette", "prediction_strength")
      Methods to use for K selection
  random_state : integer, default=42
      Random seed for reproducibility
  threshold : numeric, default=0.8
      Threshold for prediction strength
  
  Returns:
  --------
  results : list
      Results from all methods including optimal K values
  """
  results <- list()
  
  if ("gap" %in% methods) {
    cat("Computing gap statistic...\n")
    gap_result <- gap_statistic(X, K_range, random_state = random_state)
    optimal_k_gap <- find_optimal_k_gap(gap_result$gap_scores, gap_result$gap_errors, K_range)
    results$gap <- list(
      scores = gap_result$gap_scores,
      errors = gap_result$gap_errors,
      optimal_k = optimal_k_gap
    )
    cat("Gap statistic optimal K:", optimal_k_gap, "\n")
  }
  
  if ("silhouette" %in% methods) {
    cat("Computing silhouette analysis...\n")
    sil_result <- silhouette_analysis(X, K_range, random_state = random_state)
    optimal_k_silhouette <- K_range[which.max(sil_result$silhouette_scores)]
    results$silhouette <- list(
      scores = sil_result$silhouette_scores,
      samples_dict = sil_result$silhouette_samples_dict,
      optimal_k = optimal_k_silhouette
    )
    cat("Silhouette optimal K:", optimal_k_silhouette, "\n")
  }
  
  if ("prediction_strength" %in% methods) {
    cat("Computing prediction strength...\n")
    ps_scores <- compute_prediction_strength_range(X, K_range, random_state = random_state)
    optimal_k_ps <- find_optimal_k_prediction_strength(ps_scores, K_range, threshold)
    results$prediction_strength <- list(
      scores = ps_scores,
      optimal_k = optimal_k_ps
    )
    cat("Prediction strength optimal K:", optimal_k_ps, "\n")
  }
  
  results
}

demonstrate_k_selection <- function() {
  """
  Demonstrate K selection methods with synthetic data.
  """
  cat("=== K Selection Methods Demonstration ===\n\n")
  
  # Generate synthetic data with known clusters
  set.seed(42)
  n_samples <- 300
  
  # Create three well-separated clusters
  cluster1 <- matrix(rnorm(n_samples/3 * 2, mean = c(0, 0), sd = 1), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/3 * 2, mean = c(6, 6), sd = 1), ncol = 2)
  cluster3 <- matrix(rnorm(n_samples/3 * 2, mean = c(3, 9), sd = 1), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Define K range to test
  K_range <- 2:10
  
  cat("Testing K values:", K_range, "\n")
  cat("Data shape:", dim(X), "\n")
  cat("True number of clusters: 3\n\n")
  
  # Perform comprehensive K selection
  results <- comprehensive_k_selection(X, K_range)
  
  # Plot results
  cat("\nGenerating plots...\n")
  
  if ("gap" %in% names(results)) {
    plot_gap_statistic(K_range, results$gap$scores, 
                      results$gap$errors, results$gap$optimal_k)
  }
  
  if ("silhouette" %in% names(results)) {
    plot_silhouette_analysis(K_range, results$silhouette$scores,
                           results$silhouette$samples_dict, 
                           results$silhouette$optimal_k)
  }
  
  if ("prediction_strength" %in% names(results)) {
    plot_prediction_strength(K_range, results$prediction_strength$scores,
                           optimal_k = results$prediction_strength$optimal_k)
  }
  
  # Summary
  cat("\n=== Summary ===\n")
  for (method in names(results)) {
    cat(method, ": Optimal K =", results[[method]]$optimal_k, "\n")
  }
  
  results
}

compare_methods_on_different_data <- function() {
  """
  Compare K selection methods on different types of data.
  """
  cat("=== Comparing K Selection Methods ===\n\n")
  
  set.seed(42)
  K_range <- 2:10
  
  # Test 1: Well-separated clusters
  cat("Test 1: Well-separated clusters\n")
  cluster1 <- matrix(rnorm(100 * 2, mean = c(0, 0), sd = 0.5), ncol = 2)
  cluster2 <- matrix(rnorm(100 * 2, mean = c(4, 4), sd = 0.5), ncol = 2)
  cluster3 <- matrix(rnorm(100 * 2, mean = c(0, 4), sd = 0.5), ncol = 2)
  X_well_separated <- rbind(cluster1, cluster2, cluster3)
  
  results_well <- comprehensive_k_selection(X_well_separated, K_range)
  
  # Test 2: Overlapping clusters
  cat("\nTest 2: Overlapping clusters\n")
  cluster1 <- matrix(rnorm(100 * 2, mean = c(0, 0), sd = 1.5), ncol = 2)
  cluster2 <- matrix(rnorm(100 * 2, mean = c(2, 2), sd = 1.5), ncol = 2)
  cluster3 <- matrix(rnorm(100 * 2, mean = c(0, 2), sd = 1.5), ncol = 2)
  X_overlapping <- rbind(cluster1, cluster2, cluster3)
  
  results_overlapping <- comprehensive_k_selection(X_overlapping, K_range)
  
  # Test 3: No clear structure
  cat("\nTest 3: No clear structure\n")
  X_no_structure <- matrix(rnorm(300 * 2), ncol = 2)
  
  results_no_structure <- comprehensive_k_selection(X_no_structure, K_range)
  
  # Summary comparison
  cat("\n=== Method Comparison Summary ===\n")
  datasets <- list(
    "Well-separated" = results_well,
    "Overlapping" = results_overlapping,
    "No structure" = results_no_structure
  )
  
  for (dataset_name in names(datasets)) {
    cat("\n", dataset_name, ":\n", sep = "")
    results <- datasets[[dataset_name]]
    for (method in names(results)) {
      cat("  ", method, ": K = ", results[[method]]$optimal_k, "\n", sep = "")
    }
  }
  
  datasets
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic K selection
  cat("=== BASIC K SELECTION DEMONSTRATION ===\n")
  results <- demonstrate_k_selection()
  
  # Compare methods on different data types
  cat("\n=== COMPARING METHODS ON DIFFERENT DATA TYPES ===\n")
  comparison_results <- compare_methods_on_different_data()
}
