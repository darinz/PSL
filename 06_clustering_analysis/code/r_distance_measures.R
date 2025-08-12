# Distance Measures and Multidimensional Scaling in R
# ===================================================
#
# This script provides a comprehensive implementation of distance measures
# for clustering analysis, including numerical, categorical, and text-based measures.

library(stats)
library(ggplot2)
library(dplyr)
library(proxy)
library(MASS)

DistanceMeasures <- setRefClass("DistanceMeasures",
  methods = list(
    
    euclidean_distance = function(x, z) {
      sqrt(sum((x - z)^2))
    },
    
    manhattan_distance = function(x, z) {
      sum(abs(x - z))
    },
    
    minkowski_distance = function(x, z, p = 2) {
      (sum(abs(x - z)^p))^(1/p)
    },
    
    chebyshev_distance = function(x, z) {
      max(abs(x - z))
    },
    
    hamming_distance = function(x, z) {
      sum(x != z)
    },
    
    jaccard_distance = function(set_a, set_b) {
      intersection <- length(intersect(set_a, set_b))
      union <- length(union(set_a, set_b))
      if (union == 0) return(0)
      1 - intersection / union
    },
    
    cosine_distance = function(x, z) {
      dot_product <- sum(x * z)
      norm_x <- sqrt(sum(x^2))
      norm_z <- sqrt(sum(z^2))
      if (norm_x == 0 || norm_z == 0) return(1)
      1 - dot_product / (norm_x * norm_z)
    },
    
    edit_distance = function(s, t) {
      # Simple implementation using adist
      adist(s, t)[1, 1]
    },
    
    compute_distance_matrix = function(X, metric = "euclidean") {
      if (metric == "euclidean") {
        as.matrix(dist(X, method = "euclidean"))
      } else if (metric == "manhattan") {
        as.matrix(dist(X, method = "manhattan"))
      } else if (metric == "cosine") {
        # Use proxy package for cosine distance
        as.matrix(proxy::dist(X, method = "cosine"))
      } else {
        stop(paste("Unsupported metric:", metric))
      }
    },
    
    classical_mds = function(D, k = NULL) {
      n <- nrow(D)
      if (is.null(k)) k <- n
      
      # Step 1: Double centering
      D_squared <- D^2
      H <- diag(n) - matrix(1, n, n) / n
      B <- -0.5 * H %*% D_squared %*% H
      
      # Step 2: Eigendecomposition
      eigen_decomp <- eigen(B)
      eigenvals <- eigen_decomp$values
      eigenvecs <- eigen_decomp$vectors
      
      # Sort in descending order
      idx <- order(eigenvals, decreasing = TRUE)
      eigenvals <- eigenvals[idx]
      eigenvecs <- eigenvecs[, idx]
      
      # Step 3: Reconstruction
      X_reconstructed <- eigenvecs[, 1:k] %*% diag(sqrt(eigenvals[1:k]))
      
      list(
        coordinates = X_reconstructed,
        eigenvalues = eigenvals,
        eigenvectors = eigenvecs
      )
    },
    
    plot_distance_comparison = function(X, metrics = c("euclidean", "manhattan", "cosine")) {
      plots <- list()
      
      for (i in seq_along(metrics)) {
        metric <- metrics[i]
        D <- compute_distance_matrix(X, metric)
        
        # Convert to long format for ggplot
        D_long <- as.data.frame(D) %>%
          mutate(row_id = row_number()) %>%
          tidyr::gather(key = "col_id", value = "distance", -row_id) %>%
          mutate(col_id = as.numeric(col_id))
        
        p <- ggplot(D_long, aes(x = col_id, y = row_id, fill = distance)) +
          geom_tile() +
          scale_fill_viridis_c() +
          labs(title = paste(toupper(metric), "Distance"),
               x = "Sample Index", y = "Sample Index") +
          theme_minimal() +
          theme(axis.text = element_text(size = 8))
        
        plots[[i]] <- p
      }
      
      # Combine plots
      do.call(gridExtra::grid.arrange, c(plots, ncol = length(metrics)))
    },
    
    analyze_distance_distributions = function(X, metrics = c("euclidean", "manhattan", "cosine")) {
      plots <- list()
      
      for (i in seq_along(metrics)) {
        metric <- metrics[i]
        D <- compute_distance_matrix(X, metric)
        
        # Get upper triangular part (excluding diagonal)
        distances <- D[upper.tri(D)]
        
        p <- ggplot(data.frame(distance = distances), aes(x = distance)) +
          geom_histogram(bins = 30, alpha = 0.7, fill = "steelblue", color = "black") +
          geom_vline(xintercept = mean(distances), color = "red", linestyle = "dashed") +
          labs(title = paste(toupper(metric), "Distance Distribution"),
               x = "Distance", y = "Frequency") +
          annotate("text", x = mean(distances), y = Inf, 
                   label = paste("Mean:", round(mean(distances), 3)),
                   vjust = 2, hjust = -0.1, color = "red") +
          theme_minimal()
        
        plots[[i]] <- p
      }
      
      # Combine plots
      do.call(gridExtra::grid.arrange, c(plots, ncol = length(metrics)))
    }
  )
)

# Example usage and demonstration
demonstrate_distance_measures <- function() {
  cat("=== Distance Measures Demonstration ===\n\n")
  
  dm <- DistanceMeasures$new()
  
  # Generate sample data
  set.seed(42)
  X <- matrix(rnorm(50 * 3), ncol = 3)  # 50 samples, 3 features
  
  # Numerical distance examples
  x1 <- X[1, ]
  x2 <- X[2, ]
  cat("Sample points:\n")
  cat("x1 =", paste(round(x1, 3), collapse = ", "), "\n")
  cat("x2 =", paste(round(x2, 3), collapse = ", "), "\n\n")
  
  cat("Distance measures:\n")
  cat("Euclidean distance:", round(dm$euclidean_distance(x1, x2), 4), "\n")
  cat("Manhattan distance:", round(dm$manhattan_distance(x1, x2), 4), "\n")
  cat("Minkowski distance (p=3):", round(dm$minkowski_distance(x1, x2, 3), 4), "\n")
  cat("Chebyshev distance:", round(dm$chebyshev_distance(x1, x2), 4), "\n")
  cat("Cosine distance:", round(dm$cosine_distance(x1, x2), 4), "\n\n")
  
  # Categorical distance examples
  set_a <- c("apple", "banana", "cherry", "date")
  set_b <- c("apple", "banana", "elderberry")
  cat("Set A:", paste(set_a, collapse = ", "), "\n")
  cat("Set B:", paste(set_b, collapse = ", "), "\n")
  cat("Jaccard distance:", round(dm$jaccard_distance(set_a, set_b), 4), "\n\n")
  
  # String distance examples
  s1 <- "karolin"
  s2 <- "kathrin"
  cat("String 1:", s1, "\n")
  cat("String 2:", s2, "\n")
  cat("Edit distance:", dm$edit_distance(s1, s2), "\n\n")
  
  # Distance matrix analysis
  cat("Computing distance matrices for", nrow(X), "samples...\n")
  dm$plot_distance_comparison(X)
  dm$analyze_distance_distributions(X)
  
  # MDS demonstration
  cat("\n=== Multidimensional Scaling Demo ===\n")
  D <- dm$compute_distance_matrix(X, "euclidean")
  mds_result <- dm$classical_mds(D, k = 2)
  
  cat("Original data shape:", dim(X), "\n")
  cat("MDS reconstructed shape:", dim(mds_result$coordinates), "\n")
  cat("Top 5 eigenvalues:", round(mds_result$eigenvalues[1:5], 4), "\n")
  
  # Plot MDS results
  par(mfrow = c(1, 2))
  
  plot(X[, 1], X[, 2], main = "Original Data (First 2 Dimensions)",
       xlab = "Feature 1", ylab = "Feature 2", pch = 19, col = "blue")
  
  plot(mds_result$coordinates[, 1], mds_result$coordinates[, 2],
       main = "MDS Reconstruction (2D)",
       xlab = "MDS Dimension 1", ylab = "MDS Dimension 2", 
       pch = 19, col = "red")
  
  par(mfrow = c(1, 1))
}

# Function to analyze distance properties
analyze_distance_properties <- function() {
  cat("=== Distance Measure Properties Analysis ===\n\n")
  
  dm <- DistanceMeasures$new()
  
  # Generate data with different characteristics
  set.seed(42)
  
  # Normal data
  X_normal <- matrix(rnorm(100 * 3), ncol = 3)
  
  # Data with outliers
  X_outliers <- matrix(rnorm(100 * 3), ncol = 3)
  X_outliers[1, ] <- c(10, 10, 10)  # Add outlier
  
  # High-dimensional data
  X_high_dim <- matrix(rnorm(50 * 20), ncol = 20)
  
  # Compare distance distributions
  datasets <- list(
    "Normal" = X_normal,
    "With Outliers" = X_outliers,
    "High Dimensional" = X_high_dim
  )
  
  metrics <- c("euclidean", "manhattan", "cosine")
  
  for (name in names(datasets)) {
    X <- datasets[[name]]
    cat(name, "Data (", nrow(X), "samples,", ncol(X), "features):\n")
    
    for (metric in metrics) {
      D <- dm$compute_distance_matrix(X, metric)
      distances <- D[upper.tri(D)]
      
      cat("  ", toupper(metric), ": mean=", round(mean(distances), 3),
          ", std=", round(sd(distances), 3),
          ", min=", round(min(distances), 3),
          ", max=", round(max(distances), 3), "\n", sep = "")
    }
    cat("\n")
  }
}

# Function to demonstrate MDS applications
demonstrate_mds_applications <- function() {
  cat("=== MDS Applications Demo ===\n\n")
  
  dm <- DistanceMeasures$new()
  
  # Generate data with known structure
  set.seed(42)
  
  # Create data with 3 clusters
  cluster1 <- matrix(rnorm(30 * 2), ncol = 2) + matrix(c(0, 0), 30, 2, byrow = TRUE)
  cluster2 <- matrix(rnorm(30 * 2), ncol = 2) + matrix(c(5, 5), 30, 2, byrow = TRUE)
  cluster3 <- matrix(rnorm(30 * 2), ncol = 2) + matrix(c(0, 5), 30, 2, byrow = TRUE)
  
  X_clustered <- rbind(cluster1, cluster2, cluster3)
  labels <- rep(c(0, 1, 2), each = 30)
  
  # Test different distance measures
  metrics <- c("euclidean", "manhattan", "cosine")
  
  par(mfrow = c(2, length(metrics)))
  
  for (metric in metrics) {
    D <- dm$compute_distance_matrix(X_clustered, metric)
    mds_result <- dm$classical_mds(D, k = 2)
    
    # Original data
    plot(X_clustered[, 1], X_clustered[, 2], 
         main = paste("Original Data -", toupper(metric)),
         xlab = "Feature 1", ylab = "Feature 2", 
         pch = 19, col = labels + 1)
    
    # MDS reconstruction
    plot(mds_result$coordinates[, 1], mds_result$coordinates[, 2],
         main = paste("MDS Reconstruction -", toupper(metric)),
         xlab = "MDS Dimension 1", ylab = "MDS Dimension 2",
         pch = 19, col = labels + 1)
    
    cat(toupper(metric), "MDS - Top 3 eigenvalues:", 
        round(mds_result$eigenvalues[1:3], 4), "\n")
  }
  
  par(mfrow = c(1, 1))
}

# Function to compare with built-in R functions
compare_with_builtin <- function() {
  cat("=== Comparison with Built-in R Functions ===\n\n")
  
  dm <- DistanceMeasures$new()
  
  # Generate sample data
  set.seed(42)
  X <- matrix(rnorm(20 * 3), ncol = 3)
  
  # Compare distance matrices
  cat("Comparing distance matrix computations:\n")
  
  # Euclidean distance
  D_custom <- dm$compute_distance_matrix(X, "euclidean")
  D_builtin <- as.matrix(dist(X, method = "euclidean"))
  
  cat("Euclidean distance - Max difference:", max(abs(D_custom - D_builtin)), "\n")
  
  # Manhattan distance
  D_custom <- dm$compute_distance_matrix(X, "manhattan")
  D_builtin <- as.matrix(dist(X, method = "manhattan"))
  
  cat("Manhattan distance - Max difference:", max(abs(D_custom - D_builtin)), "\n")
  
  # Compare MDS
  cat("\nComparing MDS implementations:\n")
  
  D <- dm$compute_distance_matrix(X, "euclidean")
  mds_custom <- dm$classical_mds(D, k = 2)
  mds_builtin <- cmdscale(D, k = 2)
  
  cat("MDS - Max difference:", max(abs(mds_custom$coordinates - mds_builtin)), "\n")
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic distance measures
  cat("=== BASIC DISTANCE MEASURES DEMONSTRATION ===\n")
  demonstrate_distance_measures()
  
  # Analyze distance properties
  cat("\n=== ANALYZING DISTANCE PROPERTIES ===\n")
  analyze_distance_properties()
  
  # Demonstrate MDS applications
  cat("\n=== MDS APPLICATIONS ===\n")
  demonstrate_mds_applications()
  
  # Compare with built-in functions
  cat("\n=== COMPARISON WITH BUILT-IN FUNCTIONS ===\n")
  compare_with_builtin()
}
