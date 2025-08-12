# K-means and K-medoids Implementation in R
# =========================================
#
# This script provides comprehensive implementations of K-means and K-medoids
# clustering algorithms with various enhancements and evaluation metrics.

library(stats)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)

KMeansClustering <- setRefClass("KMeansClustering",
  fields = list(
    n_clusters = "numeric",
    max_iter = "numeric",
    tol = "numeric",
    n_init = "numeric",
    labels = "numeric",
    cluster_centers = "matrix",
    inertia = "numeric",
    n_iter = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_clusters = 3, max_iter = 300, tol = 1e-4, n_init = 10) {
      n_clusters <<- n_clusters
      max_iter <<- max_iter
      tol <<- tol
      n_init <<- n_init
    },
    
    kmeans_plus_plus_init = function(X) {
      n <- nrow(X)
      p <- ncol(X)
      centroids <- matrix(0, n_clusters, p)
      
      # Choose first centroid randomly
      centroids[1, ] <- X[sample(n, 1), ]
      
      for (k in 2:n_clusters) {
        # Compute distances to nearest centroid
        distances <- sapply(1:n, function(i) {
          min(sapply(1:(k-1), function(j) {
            sum((X[i, ] - centroids[j, ])^2)
          }))
        })
        
        # Choose next centroid with probability proportional to distance^2
        probs <- distances / sum(distances)
        cumprobs <- cumsum(probs)
        r <- runif(1)
        idx <- which(cumprobs >= r)[1]
        centroids[k, ] <- X[idx, ]
      }
      
      centroids
    },
    
    kmeans_single_run = function(X, initial_centroids) {
      n <- nrow(X)
      p <- ncol(X)
      centroids <- initial_centroids
      
      for (iteration in 1:max_iter) {
        old_centroids <- centroids
        
        # Assignment step
        distances <- sapply(1:n_clusters, function(k) {
          rowSums((X - matrix(centroids[k, ], n, p, byrow = TRUE))^2)
        })
        labels <- apply(distances, 1, which.min)
        
        # Update step
        for (k in 1:n_clusters) {
          if (sum(labels == k) > 0) {
            centroids[k, ] <- colMeans(X[labels == k, , drop = FALSE])
          }
        }
        
        # Check convergence
        if (max(sqrt(rowSums((centroids - old_centroids)^2))) < tol) {
          break
        }
      }
      
      # Compute final inertia
      inertia <- sum(sapply(1:n_clusters, function(k) {
        if (sum(labels == k) > 0) {
          sum(rowSums((X[labels == k, , drop = FALSE] - 
                       matrix(centroids[k, ], sum(labels == k), p, byrow = TRUE))^2))
        } else {
          0
        }
      }))
      
      list(labels = labels, centroids = centroids, inertia = inertia, n_iter = iteration)
    },
    
    fit = function(X) {
      best_inertia <- Inf
      best_labels <- NULL
      best_centroids <- NULL
      best_n_iter <- 0
      
      for (init in 1:n_init) {
        # Initialize centroids
        centroids <- kmeans_plus_plus_init(X)
        
        # Run single K-means
        result <- kmeans_single_run(X, centroids)
        
        # Update best result
        if (result$inertia < best_inertia) {
          best_inertia <- result$inertia
          best_labels <- result$labels
          best_centroids <- result$centroids
          best_n_iter <- result$n_iter
        }
      }
      
      labels <<- best_labels
      cluster_centers <<- best_centroids
      inertia <<- best_inertia
      n_iter <<- best_n_iter
      
      invisible(.self)
    },
    
    predict = function(X) {
      distances <- sapply(1:n_clusters, function(k) {
        rowSums((X - matrix(cluster_centers[k, ], nrow(X), ncol(X), byrow = TRUE))^2)
      })
      apply(distances, 1, which.min)
    },
    
    plot_clusters = function(X, title = "K-means Clustering") {
      df <- data.frame(
        x = X[, 1],
        y = X[, 2],
        cluster = factor(labels)
      )
      
      centroids_df <- data.frame(
        x = cluster_centers[, 1],
        y = cluster_centers[, 2],
        cluster = factor(1:n_clusters)
      )
      
      ggplot() +
        geom_point(data = df, aes(x = x, y = y, color = cluster), 
                   alpha = 0.7, size = 2) +
        geom_point(data = centroids_df, aes(x = x, y = y), 
                   color = "red", shape = 4, size = 4, stroke = 2) +
        labs(title = title, x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
    },
    
    evaluate_clustering = function(X) {
      metrics <- list()
      
      # Inertia
      metrics$inertia <- inertia
      
      # Silhouette score
      if (length(unique(labels)) > 1) {
        metrics$silhouette <- mean(silhouette(labels, dist(X))[, 3])
      } else {
        metrics$silhouette <- 0
      }
      
      # Number of iterations
      metrics$n_iterations <- n_iter
      
      # Cluster sizes
      cluster_counts <- table(labels)
      metrics$cluster_sizes <- as.list(cluster_counts)
      
      metrics
    }
  )
)

KMedoidsClustering <- setRefClass("KMedoidsClustering",
  fields = list(
    n_clusters = "numeric",
    max_iter = "numeric",
    labels = "numeric",
    medoids = "numeric",
    inertia = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_clusters = 3, max_iter = 300) {
      n_clusters <<- n_clusters
      max_iter <<- max_iter
    },
    
    pam_swap_phase = function(D, labels, medoids) {
      n <- nrow(D)
      K <- length(medoids)
      improved <- TRUE
      
      while (improved) {
        improved <- FALSE
        
        for (k in 1:K) {
          current_medoid <- medoids[k]
          
          # Try swapping with each non-medoid point
          for (i in 1:n) {
            if (i %in% medoids) next
            
            # Temporarily swap
            temp_medoids <- medoids
            temp_medoids[k] <- i
            
            # Compute new assignments and cost
            temp_labels <- apply(D[, temp_medoids], 1, which.min)
            temp_cost <- sum(sapply(1:n, function(j) {
              D[j, temp_medoids[temp_labels[j]]]
            }))
            
            # Current cost
            current_cost <- sum(sapply(1:n, function(j) {
              D[j, medoids[labels[j]]]
            }))
            
            # If improvement, make swap permanent
            if (temp_cost < current_cost) {
              medoids <- temp_medoids
              labels <- temp_labels
              improved <- TRUE
              break
            }
          }
        }
      }
      
      list(labels = labels, medoids = medoids)
    },
    
    fit = function(D) {
      n <- nrow(D)
      
      # Initialize medoids randomly
      medoids <- sample(n, n_clusters)
      
      for (iteration in 1:max_iter) {
        old_medoids <- medoids
        
        # Assignment step
        labels <- apply(D[, medoids], 1, which.min)
        
        # Swap step
        result <- pam_swap_phase(D, labels, medoids)
        labels <- result$labels
        medoids <- result$medoids
        
        # Check convergence
        if (all(medoids == old_medoids)) break
      }
      
      # Compute final cost
      inertia <- sum(sapply(1:n, function(i) {
        D[i, medoids[labels[i]]]
      }))
      
      labels <<- labels
      medoids <<- medoids
      inertia <<- inertia
      
      invisible(.self)
    }
  )
)

# Utility functions
random_init <- function(X, K) {
  """Random initialization: randomly select K data points as centroids."""
  n <- nrow(X)
  indices <- sample(n, K, replace = FALSE)
  X[indices, ]
}

kmeans_plus_plus_init <- function(X, K) {
  """K-means++ initialization for better initial centroids."""
  n <- nrow(X)
  p <- ncol(X)
  centroids <- matrix(0, K, p)
  
  # Choose first centroid randomly
  centroids[1, ] <- X[sample(n, 1), ]
  
  for (k in 2:K) {
    # Compute distances to nearest centroid
    distances <- sapply(1:n, function(i) {
      min(sapply(1:(k-1), function(j) {
        sum((X[i, ] - centroids[j, ])^2)
      }))
    })
    
    # Choose next centroid with probability proportional to distance^2
    probs <- distances / sum(distances)
    cumprobs <- cumsum(probs)
    r <- runif(1)
    idx <- which(cumprobs >= r)[1]
    centroids[k, ] <- X[idx, ]
  }
  
  centroids
}

kmeans_multiple_runs <- function(X, K, n_runs = 10) {
  """Run K-means multiple times and return best clustering."""
  best_inertia <- Inf
  best_labels <- NULL
  best_centroids <- NULL
  
  for (run in 1:n_runs) {
    # Initialize centroids
    centroids <- kmeans_plus_plus_init(X, K)
    
    # Run K-means
    result <- kmeans_single_run(X, K, centroids)
    
    # Update best result
    if (result$inertia < best_inertia) {
      best_inertia <- result$inertia
      best_labels <- result$labels
      best_centroids <- result$centroids
    }
  }
  
  list(labels = best_labels, centroids = best_centroids, inertia = best_inertia)
}

kmeans_single_run <- function(X, K, initial_centroids) {
  """Single run of K-means algorithm."""
  n <- nrow(X)
  p <- ncol(X)
  centroids <- initial_centroids
  max_iter <- 300
  tol <- 1e-4
  
  for (iteration in 1:max_iter) {
    old_centroids <- centroids
    
    # Assignment step
    distances <- sapply(1:K, function(k) {
      rowSums((X - matrix(centroids[k, ], n, p, byrow = TRUE))^2)
    })
    labels <- apply(distances, 1, which.min)
    
    # Update step
    for (k in 1:K) {
      if (sum(labels == k) > 0) {
        centroids[k, ] <- colMeans(X[labels == k, , drop = FALSE])
      }
    }
    
    # Check convergence
    if (max(sqrt(rowSums((centroids - old_centroids)^2))) < tol) {
      break
    }
  }
  
  # Compute final inertia
  inertia <- sum(sapply(1:K, function(k) {
    if (sum(labels == k) > 0) {
      sum(rowSums((X[labels == k, , drop = FALSE] - 
                   matrix(centroids[k, ], sum(labels == k), p, byrow = TRUE))^2))
    } else {
      0
    }
  }))
  
  list(labels = labels, centroids = centroids, inertia = inertia)
}

kmeans_with_dimension_reduction <- function(X, K, method = "pca", d = NULL) {
  """K-means with dimension reduction preprocessing."""
  if (is.null(d)) {
    d <- min(K + 1, ncol(X))  # Rule of thumb
  }
  
  if (method == "pca") {
    # PCA using prcomp
    pca_result <- prcomp(X, center = TRUE, scale. = TRUE)
    X_reduced <- pca_result$x[, 1:d, drop = FALSE]
    reducer <- pca_result
  } else if (method == "random") {
    # Random projection (simplified)
    set.seed(42)
    R <- matrix(rnorm(ncol(X) * d, 0, 1/sqrt(d)), ncol(X), d)
    X_reduced <- X %*% R
    reducer <- list(R = R)
  } else {
    stop(paste("Unknown method:", method))
  }
  
  # Run K-means on reduced data
  result <- kmeans_multiple_runs(X_reduced, K)
  
  # Transform centroids back to original space
  if (method == "pca") {
    centroids <- result$centroids %*% t(pca_result$rotation[, 1:d, drop = FALSE])
    centroids <- sweep(centroids, 2, pca_result$center, "+")
  } else {
    centroids <- result$centroids %*% t(R)
  }
  
  list(labels = result$labels, centroids = centroids, 
       inertia = result$inertia, reducer = reducer)
}

manhattan_centroid <- function(X_cluster) {
  """Compute centroid for Manhattan distance (median)."""
  apply(X_cluster, 2, median)
}

cosine_centroid <- function(X_cluster) {
  """Compute centroid for cosine distance."""
  mean_vec <- colMeans(X_cluster)
  norm <- sqrt(sum(mean_vec^2))
  if (norm > 0) mean_vec / norm else mean_vec
}

mixed_distance <- function(x, y, weights = c(0.4, 0.6)) {
  """Mixed distance: L1 for numerical, Hamming for categorical."""
  numerical_dist <- sum(abs(x[1:2] - y[1:2]))  # First 2 features
  categorical_dist <- sum(x[3:length(x)] != y[3:length(y)])  # Remaining features
  weights[1] * numerical_dist + weights[2] * categorical_dist
}

mixed_centroid <- function(X_cluster) {
  """Compute centroid for mixed distance measure."""
  # Numerical features: median
  numerical_centroid <- apply(X_cluster[, 1:2, drop = FALSE], 2, median)
  
  # Categorical features: mode
  categorical_centroid <- sapply(3:ncol(X_cluster), function(j) {
    values <- X_cluster[, j]
    unique_vals <- unique(values)
    counts <- sapply(unique_vals, function(v) sum(values == v))
    unique_vals[which.max(counts)]
  })
  
  c(numerical_centroid, categorical_centroid)
}

pam_swap_phase <- function(D, labels, medoids) {
  """PAM swap phase: try swapping medoids with non-medoids."""
  n <- nrow(D)
  K <- length(medoids)
  improved <- TRUE
  
  while (improved) {
    improved <- FALSE
    
    for (k in 1:K) {
      current_medoid <- medoids[k]
      
      # Try swapping with each non-medoid point
      for (i in 1:n) {
        if (i %in% medoids) next
        
        # Temporarily swap
        temp_medoids <- medoids
        temp_medoids[k] <- i
        
        # Compute new assignments and cost
        temp_labels <- apply(D[, temp_medoids], 1, which.min)
        temp_cost <- sum(sapply(1:n, function(j) {
          D[j, temp_medoids[temp_labels[j]]]
        }))
        
        # Current cost
        current_cost <- sum(sapply(1:n, function(j) {
          D[j, medoids[labels[j]]]
        }))
        
        # If improvement, make swap permanent
        if (temp_cost < current_cost) {
          medoids <- temp_medoids
          labels <- temp_labels
          improved <- TRUE
          break
        }
      }
    }
  }
  
  list(labels = labels, medoids = medoids)
}

# Example usage and demonstration
demonstrate_kmeans <- function() {
  cat("=== K-means Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  n_samples <- 300
  
  # Create three clusters
  cluster1 <- matrix(rnorm(n_samples/3 * 2, mean = c(0, 0), sd = 1), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/3 * 2, mean = c(4, 4), sd = 1), ncol = 2)
  cluster3 <- matrix(rnorm(n_samples/3 * 2, mean = c(2, 6), sd = 1), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Test different numbers of clusters
  for (K in c(2, 3, 4, 5)) {
    cat("Testing K =", K, "clusters...\n")
    
    # Fit K-means
    kmeans <- KMeansClustering$new(n_clusters = K, n_init = 10)
    kmeans$fit(X)
    
    # Evaluate results
    metrics <- kmeans$evaluate_clustering(X)
    cat("  Inertia:", round(metrics$inertia, 2), "\n")
    cat("  Silhouette Score:", round(metrics$silhouette, 3), "\n")
    cat("  Iterations:", metrics$n_iterations, "\n")
    cat("  Cluster Sizes:", unlist(metrics$cluster_sizes), "\n\n")
    
    # Plot results
    print(kmeans$plot_clusters(X, paste("K-means with K=", K)))
  }
  
  # Compare with built-in kmeans
  cat("Comparing with built-in kmeans function...\n")
  builtin_kmeans <- kmeans(X, centers = 3, nstart = 10)
  cat("Built-in inertia:", round(builtin_kmeans$tot.withinss, 2), "\n")
  cat("Our inertia:", round(kmeans$inertia, 2), "\n")
}

demonstrate_kmedoids <- function() {
  cat("=== K-medoids Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  X <- matrix(rnorm(50 * 2), ncol = 2)
  
  # Compute distance matrix
  D <- as.matrix(dist(X))
  
  # Fit K-medoids
  kmedoids <- KMedoidsClustering$new(n_clusters = 3)
  kmedoids$fit(D)
  
  cat("Final cost:", round(kmedoids$inertia, 2), "\n")
  cat("Medoids:", kmedoids$medoids, "\n")
  cat("Cluster sizes:", table(kmedoids$labels), "\n")
  
  # Visualize results
  df <- data.frame(
    x = X[, 1],
    y = X[, 2],
    cluster = factor(kmedoids$labels)
  )
  
  centroids_df <- data.frame(
    x = X[kmedoids$medoids, 1],
    y = X[kmedoids$medoids, 2],
    cluster = factor(1:kmedoids$n_clusters)
  )
  
  p <- ggplot() +
    geom_point(data = df, aes(x = x, y = y, color = cluster), 
               alpha = 0.7, size = 2) +
    geom_point(data = centroids_df, aes(x = x, y = y), 
               color = "red", shape = 4, size = 4, stroke = 2) +
    labs(title = "K-medoids Clustering", x = "Feature 1", y = "Feature 2") +
    theme_minimal() +
    scale_color_viridis_d()
  
  print(p)
}

analyze_initialization_methods <- function() {
  cat("=== Initialization Method Comparison ===\n\n")
  
  set.seed(42)
  
  # Generate challenging data
  n_samples <- 200
  cluster1 <- matrix(rnorm(n_samples/2 * 2, mean = c(0, 0), sd = 0.5), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/2 * 2, mean = c(3, 3), sd = 0.5), ncol = 2)
  X <- rbind(cluster1, cluster2)
  
  # Test random initialization
  inertias_random <- sapply(1:20, function(i) {
    centroids <- random_init(X, 2)
    result <- kmeans_single_run(X, 2, centroids)
    result$inertia
  })
  
  # Test K-means++ initialization
  inertias_plus_plus <- sapply(1:20, function(i) {
    centroids <- kmeans_plus_plus_init(X, 2)
    result <- kmeans_single_run(X, 2, centroids)
    result$inertia
  })
  
  cat("Random initialization - Mean inertia:", round(mean(inertias_random), 2), "\n")
  cat("Random initialization - Std inertia:", round(sd(inertias_random), 2), "\n")
  cat("K-means++ initialization - Mean inertia:", round(mean(inertias_plus_plus), 2), "\n")
  cat("K-means++ initialization - Std inertia:", round(sd(inertias_plus_plus), 2), "\n")
  
  # Plot comparison
  df <- data.frame(
    method = rep(c("Random", "K-means++"), each = 20),
    inertia = c(inertias_random, inertias_plus_plus)
  )
  
  p <- ggplot(df, aes(x = method, y = inertia)) +
    geom_boxplot() +
    labs(title = "Initialization Method Comparison", 
         x = "Method", y = "Inertia") +
    theme_minimal()
  
  print(p)
}

demonstrate_dimension_reduction <- function() {
  cat("=== Dimension Reduction for K-means ===\n\n")
  
  set.seed(42)
  
  # Generate high-dimensional data
  n_samples <- 300
  n_features <- 20
  
  # Create 3 clusters in high-dimensional space
  cluster1 <- matrix(rnorm(n_samples/3 * n_features, mean = 0, sd = 1), ncol = n_features)
  cluster2 <- matrix(rnorm(n_samples/3 * n_features, mean = 3, sd = 1), ncol = n_features)
  cluster3 <- matrix(rnorm(n_samples/3 * n_features, mean = 6, sd = 1), ncol = n_features)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Compare different methods
  methods <- c("pca", "random")
  
  for (method in methods) {
    cat("Testing", toupper(method), "dimension reduction...\n")
    
    result <- kmeans_with_dimension_reduction(X, 3, method = method, d = 3)
    
    cat("  Reduced dimensions:", 3, "\n")
    cat("  Final inertia:", round(result$inertia, 2), "\n")
    if (method == "pca") {
      cat("  Explained variance:", round(sum(result$reducer$sdev[1:3]^2) / sum(result$reducer$sdev^2), 3), "\n")
    }
    cat("\n")
  }
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Demonstrate basic K-means
  cat("=== BASIC K-MEANS DEMONSTRATION ===\n")
  demonstrate_kmeans()
  
  # Demonstrate K-medoids
  cat("\n=== K-MEDOIDS DEMONSTRATION ===\n")
  demonstrate_kmedoids()
  
  # Analyze initialization methods
  cat("\n=== INITIALIZATION METHOD ANALYSIS ===\n")
  analyze_initialization_methods()
  
  # Demonstrate dimension reduction
  cat("\n=== DIMENSION REDUCTION DEMONSTRATION ===\n")
  demonstrate_dimension_reduction()
}
