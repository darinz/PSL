# Hierarchical Clustering Implementation in R
# ==========================================
#
# This script provides comprehensive implementations of hierarchical clustering
# methods, including various linkage criteria, dendrogram visualization, and
# comparison tools.

library(stats)
library(cluster)
library(dendextend)
library(ggplot2)
library(dplyr)

HierarchicalClustering <- setRefClass("HierarchicalClustering",
  fields = list(
    method = "character",
    metric = "character",
    linkage_matrix = "matrix",
    distance_matrix = "dist"
  ),
  
  methods = list(
    
    initialize = function(method = "complete", metric = "euclidean") {
      """
      Initialize hierarchical clustering.
      
      Parameters:
      -----------
      method : character, default="complete"
          Linkage method: "single", "complete", "average", "ward.D"
      metric : character, default="euclidean"
          Distance metric for computing pairwise distances
      """
      method <<- method
      metric <<- metric
    },
    
    fit = function(X) {
      """
      Fit hierarchical clustering to the data.
      
      Parameters:
      -----------
      X : matrix
          Training data
          
      Returns:
      --------
      self : object
          Returns self invisibly
      """
      # Compute distance matrix
      distance_matrix <<- dist(X, method = metric)
      
      # Perform hierarchical clustering
      hc <- hclust(distance_matrix, method = method)
      linkage_matrix <<- hc$merge
      
      invisible(.self)
    },
    
    get_clusters = function(n_clusters = NULL, height = NULL) {
      """
      Extract clusters from the dendrogram.
      
      Parameters:
      -----------
      n_clusters : integer, optional
          Number of clusters to extract
      height : numeric, optional
          Height at which to cut the dendrogram
          
      Returns:
      --------
      labels : vector
          Cluster labels for each sample
      """
      if (!is.null(n_clusters)) {
        cutree(hclust(distance_matrix, method = method), k = n_clusters)
      } else if (!is.null(height)) {
        cutree(hclust(distance_matrix, method = method), h = height)
      } else {
        stop("Must specify either n_clusters or height")
      }
    },
    
    plot_dendrogram = function(title = NULL) {
      """
      Plot the dendrogram.
      
      Parameters:
      -----------
      title : character, optional
          Title for the plot
      """
      hc <- hclust(distance_matrix, method = method)
      
      plot(hc, 
           main = title %||% paste("Hierarchical Clustering Dendrogram (", method, " linkage)"),
           xlab = "Sample Index", 
           ylab = "Distance",
           sub = "")
    },
    
    cophenetic_correlation = function() {
      """
      Compute cophenetic correlation coefficient.
      
      Returns:
      --------
      correlation : numeric
          Cophenetic correlation coefficient
      """
      hc <- hclust(distance_matrix, method = method)
      cor(distance_matrix, cophenetic(hc))
    },
    
    compare_linkage_methods = function(X, methods = c("single", "complete", "average", "ward.D")) {
      """
      Compare different linkage methods.
      
      Parameters:
      -----------
      X : matrix
          Input data
      methods : vector, default=c("single", "complete", "average", "ward.D")
          List of linkage methods to compare
          
      Returns:
      --------
      results : list
          List containing results for each method
      """
      results <- list()
      
      for (method in methods) {
        # Fit clustering
        hc_temp <- HierarchicalClustering$new(method = method)
        hc_temp$fit(X)
        
        # Compute cophenetic correlation
        cophenetic_corr <- hc_temp$cophenetic_correlation()
        
        # Compute silhouette scores for different K
        silhouette_scores <- numeric(9)  # K = 2 to 10
        for (k in 2:10) {
          labels <- hc_temp$get_clusters(n_clusters = k)
          if (length(unique(labels)) > 1) {
            silhouette_scores[k-1] <- mean(silhouette(labels, hc_temp$distance_matrix)[, 3])
          }
        }
        
        results[[method]] <- list(
          cophenetic_correlation = cophenetic_corr,
          silhouette_scores = silhouette_scores,
          hclust_obj = hclust(hc_temp$distance_matrix, method = method)
        )
      }
      
      results
    },
    
    plot_comparison = function(X, methods = c("single", "complete", "average", "ward.D")) {
      """
      Plot comparison of different linkage methods.
      
      Parameters:
      -----------
      X : matrix
          Input data
      methods : vector, default=c("single", "complete", "average", "ward.D")
          List of linkage methods to compare
      """
      results <- compare_linkage_methods(X, methods)
      
      # Plot dendrograms
      par(mfrow = c(2, 2))
      for (method in methods) {
        plot(results[[method]]$hclust_obj, 
             main = paste(toupper(method), "Linkage"),
             xlab = "Sample Index", 
             ylab = "Distance")
      }
      par(mfrow = c(1, 1))
      
      # Plot silhouette scores
      silhouette_data <- data.frame()
      for (method in methods) {
        scores <- results[[method]]$silhouette_scores
        silhouette_data <- rbind(silhouette_data, 
                                data.frame(
                                  K = 2:10,
                                  Score = scores,
                                  Method = method
                                ))
      }
      
      p <- ggplot(silhouette_data, aes(x = K, y = Score, color = Method)) +
        geom_line() +
        geom_point() +
        labs(title = "Silhouette Scores for Different Linkage Methods",
             x = "Number of Clusters (K)",
             y = "Silhouette Score") +
        theme_minimal() +
        scale_color_viridis_d()
      
      print(p)
      
      # Print cophenetic correlations
      cat("Cophenetic Correlation Coefficients:\n")
      for (method in methods) {
        cat(sprintf("%s: %.4f\n", toupper(method), results[[method]]$cophenetic_correlation))
      }
    }
  )
)

# Example usage and demonstration
demonstrate_hierarchical_clustering <- function() {
  """
  Demonstrate hierarchical clustering with various examples.
  """
  cat("=== Hierarchical Clustering Demonstration ===\n\n")
  
  # Generate sample data
  set.seed(42)
  n_samples <- 100
  
  # Create three well-separated clusters
  cluster1 <- matrix(rnorm(n_samples/3 * 2, mean = c(0, 0), sd = 1), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/3 * 2, mean = c(6, 6), sd = 1), ncol = 2)
  cluster3 <- matrix(c(rnorm(n_samples/3, mean = 3, sd = 1), 
                       rnorm(n_samples/3, mean = 9, sd = 1)), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  cat("Data shape:", dim(X), "\n")
  cat("Number of samples:", nrow(X), "\n")
  cat("Number of features:", ncol(X), "\n\n")
  
  # Initialize hierarchical clustering
  hc <- HierarchicalClustering$new(method = "complete")
  hc$fit(X)
  
  # Plot dendrogram
  cat("Plotting dendrogram...\n")
  hc$plot_dendrogram("Complete Linkage Dendrogram")
  
  # Extract clusters
  labels_3 <- hc$get_clusters(n_clusters = 3)
  labels_5 <- hc$get_clusters(n_clusters = 5)
  
  # Visualize cluster assignments
  par(mfrow = c(1, 2))
  
  plot(X[, 1], X[, 2], col = labels_3, pch = 19, 
       main = "3 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  plot(X[, 1], X[, 2], col = labels_5, pch = 19, 
       main = "5 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  par(mfrow = c(1, 1))
  
  # Compare linkage methods
  cat("\nComparing different linkage methods...\n")
  hc$plot_comparison(X)
  
  # Cophenetic correlation
  cat(sprintf("\nCophenetic correlation: %.4f\n", hc$cophenetic_correlation()))
  
  # Evaluate clustering quality
  silhouette_3 <- mean(silhouette(labels_3, hc$distance_matrix)[, 3])
  silhouette_5 <- mean(silhouette(labels_5, hc$distance_matrix)[, 3])
  cat(sprintf("Silhouette score (3 clusters): %.4f\n", silhouette_3))
  cat(sprintf("Silhouette score (5 clusters): %.4f\n", silhouette_5))
  
  list(hc = hc, X = X, labels_3 = labels_3, labels_5 = labels_5)
}

analyze_linkage_methods <- function() {
  """
  Analyze the behavior of different linkage methods on various data types.
  """
  cat("=== Linkage Methods Analysis ===\n\n")
  
  set.seed(42)
  
  # Test 1: Well-separated clusters
  cat("Test 1: Well-separated clusters\n")
  cluster1 <- matrix(rnorm(50 * 2, mean = c(0, 0), sd = 0.5), ncol = 2)
  cluster2 <- matrix(rnorm(50 * 2, mean = c(4, 4), sd = 0.5), ncol = 2)
  cluster3 <- matrix(rnorm(50 * 2, mean = c(0, 4), sd = 0.5), ncol = 2)
  X_well_separated <- rbind(cluster1, cluster2, cluster3)
  
  hc_well <- HierarchicalClustering$new(method = "complete")
  hc_well$fit(X_well_separated)
  cat(sprintf("Cophenetic correlation: %.4f\n", hc_well$cophenetic_correlation()))
  
  # Test 2: Overlapping clusters
  cat("\nTest 2: Overlapping clusters\n")
  cluster1 <- matrix(rnorm(50 * 2, mean = c(0, 0), sd = 1.5), ncol = 2)
  cluster2 <- matrix(rnorm(50 * 2, mean = c(2, 2), sd = 1.5), ncol = 2)
  cluster3 <- matrix(rnorm(50 * 2, mean = c(0, 2), sd = 1.5), ncol = 2)
  X_overlapping <- rbind(cluster1, cluster2, cluster3)
  
  hc_overlapping <- HierarchicalClustering$new(method = "complete")
  hc_overlapping$fit(X_overlapping)
  cat(sprintf("Cophenetic correlation: %.4f\n", hc_overlapping$cophenetic_correlation()))
  
  # Test 3: Chain-like structure
  cat("\nTest 3: Chain-like structure\n")
  t <- seq(0, 4*pi, length.out = 100)
  X_chain <- cbind(cos(t) + rnorm(100, 0, 0.1),
                   sin(t) + rnorm(100, 0, 0.1))
  
  hc_chain <- HierarchicalClustering$new(method = "single")
  hc_chain$fit(X_chain)
  cat(sprintf("Cophenetic correlation: %.4f\n", hc_chain$cophenetic_correlation()))
  
  # Visualize all test cases
  par(mfrow = c(2, 3))
  
  # Well-separated clusters
  plot(X_well_separated[, 1], X_well_separated[, 2], 
       main = "Well-separated Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  # Overlapping clusters
  plot(X_overlapping[, 1], X_overlapping[, 2], 
       main = "Overlapping Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  # Chain-like structure
  plot(X_chain[, 1], X_chain[, 2], 
       main = "Chain-like Structure", xlab = "Feature 1", ylab = "Feature 2")
  
  # Dendrograms
  plot(hclust(hc_well$distance_matrix, method = hc_well$method), 
       main = "Well-separated Dendrogram", xlab = "Sample Index", ylab = "Distance")
  
  plot(hclust(hc_overlapping$distance_matrix, method = hc_overlapping$method), 
       main = "Overlapping Dendrogram", xlab = "Sample Index", ylab = "Distance")
  
  plot(hclust(hc_chain$distance_matrix, method = hc_chain$method), 
       main = "Chain Dendrogram", xlab = "Sample Index", ylab = "Distance")
  
  par(mfrow = c(1, 1))
}

demonstrate_cluster_extraction <- function() {
  """
  Demonstrate different ways to extract clusters from hierarchical clustering.
  """
  cat("=== Cluster Extraction Demonstration ===\n\n")
  
  # Generate data
  set.seed(42)
  n_samples <- 80
  
  # Create clusters with different densities
  cluster1 <- matrix(rnorm(n_samples/4 * 2, mean = c(0, 0), sd = 0.8), ncol = 2)
  cluster2 <- matrix(rnorm(n_samples/4 * 2, mean = c(4, 0), sd = 0.8), ncol = 2)
  cluster3 <- matrix(rnorm(n_samples/2 * 2, mean = c(2, 4), sd = 1.2), ncol = 2)
  
  X <- rbind(cluster1, cluster2, cluster3)
  
  # Fit hierarchical clustering
  hc <- HierarchicalClustering$new(method = "ward.D")
  hc$fit(X)
  
  # Extract clusters at different levels
  labels_2 <- hc$get_clusters(n_clusters = 2)
  labels_3 <- hc$get_clusters(n_clusters = 3)
  labels_4 <- hc$get_clusters(n_clusters = 4)
  
  # Extract clusters at different heights
  height_1 <- 2.0
  height_2 <- 3.5
  labels_h1 <- hc$get_clusters(height = height_1)
  labels_h2 <- hc$get_clusters(height = height_2)
  
  # Visualize results
  par(mfrow = c(2, 3))
  
  # Number-based extraction
  plot(X[, 1], X[, 2], col = labels_2, pch = 19, 
       main = "2 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  plot(X[, 1], X[, 2], col = labels_3, pch = 19, 
       main = "3 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  plot(X[, 1], X[, 2], col = labels_4, pch = 19, 
       main = "4 Clusters", xlab = "Feature 1", ylab = "Feature 2")
  
  # Height-based extraction
  plot(X[, 1], X[, 2], col = labels_h1, pch = 19, 
       main = paste("Height =", height_1), xlab = "Feature 1", ylab = "Feature 2")
  
  plot(X[, 1], X[, 2], col = labels_h2, pch = 19, 
       main = paste("Height =", height_2), xlab = "Feature 1", ylab = "Feature 2")
  
  # Dendrogram with cut lines
  hc_obj <- hclust(hc$distance_matrix, method = hc$method)
  plot(hc_obj, main = "Dendrogram with Cut Lines", xlab = "Sample Index", ylab = "Distance")
  abline(h = height_1, col = "red", lty = 2, lwd = 2)
  abline(h = height_2, col = "orange", lty = 2, lwd = 2)
  legend("topright", legend = c(paste("Height =", height_1), paste("Height =", height_2)), 
         col = c("red", "orange"), lty = 2, lwd = 2)
  
  par(mfrow = c(1, 1))
  
  # Print cluster statistics
  cat("Cluster Statistics:\n")
  cat("2 clusters:", length(unique(labels_2)), "unique labels\n")
  cat("3 clusters:", length(unique(labels_3)), "unique labels\n")
  cat("4 clusters:", length(unique(labels_4)), "unique labels\n")
  cat("Height", height_1, ":", length(unique(labels_h1)), "unique labels\n")
  cat("Height", height_2, ":", length(unique(labels_h2)), "unique labels\n")
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Basic demonstration
  cat("=== BASIC HIERARCHICAL CLUSTERING DEMONSTRATION ===\n")
  results <- demonstrate_hierarchical_clustering()
  
  # Analyze linkage methods
  cat("\n=== LINKAGE METHODS ANALYSIS ===\n")
  analyze_linkage_methods()
  
  # Demonstrate cluster extraction
  cat("\n=== CLUSTER EXTRACTION DEMONSTRATION ===\n")
  demonstrate_cluster_extraction()
}
