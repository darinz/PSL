# Model-Based Clustering Implementation in R
# ==========================================
#
# This script provides comprehensive implementations of model-based clustering
# using Gaussian Mixture Models, including model selection, visualization,
# and analysis tools.

library(mclust)
library(ggplot2)
library(dplyr)
library(gridExtra)

ModelBasedClustering <- setRefClass("ModelBasedClustering",
  fields = list(
    n_components = "numeric",
    covariance_type = "character",
    gmm = "ANY",
    bic_scores = "numeric",
    aic_scores = "numeric"
  ),
  
  methods = list(
    
    initialize = function(n_components = 2, covariance_type = "VVV") {
      """
      Initialize model-based clustering.
      
      Parameters:
      -----------
      n_components : numeric, default=2
          Number of mixture components (clusters)
      covariance_type : character, default="VVV"
          Type of covariance parameters: "VVV", "VVI", "VII", etc.
      """
      n_components <<- n_components
      covariance_type <<- covariance_type
    },
    
    fit = function(X) {
      """
      Fit Gaussian Mixture Model to the data.
      
      Parameters:
      -----------
      X : matrix
          Training data
          
      Returns:
      --------
      self : object
          Returns self invisibly
      """
      # Fit Gaussian Mixture Model using mclust
      gmm <<- Mclust(X, G = n_components, modelNames = covariance_type)
      invisible(.self)
    },
    
    predict = function(X) {
      """
      Predict cluster labels.
      
      Parameters:
      -----------
      X : matrix
          Data to predict
          
      Returns:
      --------
      labels : vector
          Predicted cluster labels
      """
      predict(gmm, X)$classification
    },
    
    predict_proba = function(X) {
      """
      Predict cluster membership probabilities.
      
      Parameters:
      -----------
      X : matrix
          Data to predict
          
      Returns:
      --------
      probabilities : matrix
          Cluster membership probabilities
      """
      predict(gmm, X)$z
    },
    
    score = function(X) {
      """
      Compute log-likelihood.
      
      Parameters:
      -----------
      X : matrix
          Data to score
          
      Returns:
      --------
      log_likelihood : numeric
          Log-likelihood of the data
      """
      # Log-likelihood
      sum(logLik(gmm))
    },
    
    bic = function(X) {
      """
      Compute BIC score.
      
      Parameters:
      -----------
      X : matrix
          Data to score
          
      Returns:
      --------
      bic : numeric
          Bayesian Information Criterion score
      """
      BIC(gmm)
    },
    
    aic = function(X) {
      """
      Compute AIC score.
      
      Parameters:
      -----------
      X : matrix
          Data to score
          
      Returns:
      --------
      aic : numeric
          Akaike Information Criterion score
      """
      AIC(gmm)
    },
    
    plot_clusters = function(X, title = NULL) {
      """
      Visualize clustering results.
      
      Parameters:
      -----------
      X : matrix
          Data to visualize
      title : character, optional
          Title for the plot
      """
      labels <- predict(X)
      probas <- predict_proba(X)
      max_proba <- apply(probas, 1, max)
      
      # Create data frames for plotting
      df_clusters <- data.frame(
        x = X[, 1],
        y = X[, 2],
        cluster = factor(labels)
      )
      
      df_uncertainty <- data.frame(
        x = X[, 1],
        y = X[, 2],
        uncertainty = 1 - max_proba
      )
      
      # Plot cluster assignments
      p1 <- ggplot(df_clusters, aes(x = x, y = y, color = cluster)) +
        geom_point(alpha = 0.7) +
        labs(title = "Hard Cluster Assignments",
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
      
      # Plot uncertainty
      p2 <- ggplot(df_uncertainty, aes(x = x, y = y, color = uncertainty)) +
        geom_point(alpha = 0.7) +
        labs(title = "Cluster Assignment Uncertainty",
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_c()
      
      # Combine plots
      grid.arrange(p1, p2, ncol = 2, 
                   top = title %||% paste("GMM Clustering (K=", n_components, ")"))
    },
    
    plot_contours = function(X, title = NULL) {
      """
      Plot GMM contours and data.
      
      Parameters:
      -----------
      X : matrix
          Data to visualize
      title : character, optional
          Title for the plot
      """
      # Create grid for contour plot
      x_range <- range(X[, 1])
      y_range <- range(X[, 2])
      
      x_grid <- seq(x_range[1] - 1, x_range[2] + 1, length.out = 100)
      y_grid <- seq(y_range[1] - 1, y_range[2] + 1, length.out = 100)
      
      grid_points <- expand.grid(x = x_grid, y = y_grid)
      
      # Compute density
      density_values <- predict(gmm, as.matrix(grid_points))$z
      total_density <- rowSums(density_values)
      
      grid_points$density <- total_density
      
      # Plot
      p <- ggplot() +
        geom_contour(data = grid_points, aes(x = x, y = y, z = density), 
                     bins = 20, alpha = 0.6) +
        geom_contour_filled(data = grid_points, aes(x = x, y = y, z = density), 
                           alpha = 0.3) +
        geom_point(data = data.frame(x = X[, 1], y = X[, 2], 
                                    cluster = factor(predict(X))),
                  aes(x = x, y = y, color = cluster), alpha = 0.7) +
        geom_point(data = data.frame(x = gmm$parameters$mean[1, ], 
                                    y = gmm$parameters$mean[2, ]),
                  aes(x = x, y = y), color = "red", shape = 4, size = 3) +
        labs(title = title %||% paste("GMM Density Contours (K=", n_components, ")"),
             x = "Feature 1", y = "Feature 2") +
        theme_minimal() +
        scale_color_viridis_d()
      
      print(p)
    },
    
    model_selection = function(X, K_range = 1:10) {
      """
      Perform model selection using BIC and AIC.
      
      Parameters:
      -----------
      X : matrix
          Data for model selection
      K_range : vector, default=1:10
          Range of K values to test
          
      Returns:
      --------
      results : list
          List containing model selection results
      """
      bic_scores <- numeric(length(K_range))
      aic_scores <- numeric(length(K_range))
      log_likelihoods <- numeric(length(K_range))
      
      for (i in seq_along(K_range)) {
        k <- K_range[i]
        if (k == 1) {
          bic_scores[i] <- Inf
          aic_scores[i] <- Inf
          log_likelihoods[i] <- -Inf
          next
        }
        
        gmm_temp <- Mclust(X, G = k, modelNames = covariance_type)
        bic_scores[i] <- BIC(gmm_temp)
        aic_scores[i] <- AIC(gmm_temp)
        log_likelihoods[i] <- sum(logLik(gmm_temp))
      }
      
      # Create plots
      df_results <- data.frame(
        K = K_range,
        BIC = bic_scores,
        AIC = aic_scores,
        LogLik = log_likelihoods
      )
      
      p1 <- ggplot(df_results, aes(x = K, y = LogLik)) +
        geom_line() + geom_point() +
        labs(title = "Log-Likelihood", x = "Number of Components (K)", y = "Log-Likelihood") +
        theme_minimal()
      
      p2 <- ggplot(df_results, aes(x = K, y = BIC)) +
        geom_line(color = "red") + geom_point(color = "red") +
        labs(title = "BIC Score", x = "Number of Components (K)", y = "BIC") +
        theme_minimal()
      
      p3 <- ggplot(df_results, aes(x = K, y = AIC)) +
        geom_line(color = "green") + geom_point(color = "green") +
        labs(title = "AIC Score", x = "Number of Components (K)", y = "AIC") +
        theme_minimal()
      
      grid.arrange(p1, p2, p3, ncol = 3)
      
      # Find optimal K
      optimal_bic_k <- K_range[which.min(bic_scores)]
      optimal_aic_k <- K_range[which.min(aic_scores)]
      
      cat("Optimal K (BIC):", optimal_bic_k, "\n")
      cat("Optimal K (AIC):", optimal_aic_k, "\n")
      
      list(
        bic_scores = bic_scores,
        aic_scores = aic_scores,
        log_likelihoods = log_likelihoods,
        optimal_bic_k = optimal_bic_k,
        optimal_aic_k = optimal_aic_k
      )
    },
    
    analyze_components = function(X) {
      """
      Analyze component parameters and characteristics.
      
      Parameters:
      -----------
      X : matrix
          Data to analyze
          
      Returns:
      --------
      analysis : list
          List containing component analysis
      """
      labels <- predict(X)
      probas <- predict_proba(X)
      
      analysis <- list(
        component_sizes = numeric(n_components),
        component_weights = gmm$parameters$pro,
        component_means = gmm$parameters$mean,
        component_covariances = gmm$parameters$variance$sigma,
        component_uncertainty = numeric(n_components)
      )
      
      for (k in 1:n_components) {
        # Component size
        analysis$component_sizes[k] <- sum(labels == k)
        
        # Component uncertainty (average probability for assigned points)
        component_mask <- labels == k
        if (any(component_mask)) {
          avg_prob <- mean(probas[component_mask, k])
          analysis$component_uncertainty[k] <- avg_prob
        } else {
          analysis$component_uncertainty[k] <- 0.0
        }
      }
      
      analysis
    }
  )
)

# Load Old Faithful data
load_old_faithful_data <- function() {
  """
  Load and preprocess Old Faithful Geyser data.
  
  Returns:
  --------
  X : matrix
      Old Faithful data with duration and waiting time
  """
  # Use built-in faithful data
  data(faithful)
  as.matrix(faithful)
}

# Example usage and demonstration
demonstrate_model_based_clustering <- function() {
  """
  Demonstrate model-based clustering with Old Faithful data.
  """
  cat("=== Model-Based Clustering Demonstration ===\n\n")
  
  # Load data
  X <- load_old_faithful_data()
  cat("Dataset shape:", dim(X), "\n")
  cat("Features: Duration (minutes), Waiting time (minutes)\n")
  
  # Model selection
  cat("\nPerforming model selection...\n")
  mbc <- ModelBasedClustering$new()
  results <- mbc$model_selection(X, K_range = 1:8)
  
  # Fit optimal model
  optimal_k <- results$optimal_bic_k
  cat(sprintf("\nFitting optimal model with K=%d...\n", optimal_k))
  
  mbc_optimal <- ModelBasedClustering$new(n_components = optimal_k)
  mbc_optimal$fit(X)
  
  # Visualize results
  mbc_optimal$plot_clusters(X, sprintf("Old Faithful Data - %d Components", optimal_k))
  mbc_optimal$plot_contours(X, sprintf("Old Faithful Data - %d Components", optimal_k))
  
  # Compare different K values
  cat("\nComparing different numbers of components...\n")
  for (k in c(2, 3, 4)) {
    mbc_k <- ModelBasedClustering$new(n_components = k)
    mbc_k$fit(X)
    
    # Evaluate clustering
    labels <- mbc_k$predict(X)
    silhouette <- mean(silhouette(labels, dist(X))[, 3])
    bic <- mbc_k$bic(X)
    
    cat(sprintf("K=%d: Silhouette=%.3f, BIC=%.1f\n", k, silhouette, bic))
    
    # Plot
    mbc_k$plot_clusters(X, sprintf("Old Faithful Data - %d Components", k))
    mbc_k$plot_contours(X, sprintf("Old Faithful Data - %d Components", k))
  }
  
  # Analyze component parameters
  cat(sprintf("\nComponent parameters for K=%d:\n", optimal_k))
  analysis <- mbc_optimal$analyze_components(X)
  for (k in 1:optimal_k) {
    cat(sprintf("Component %d:\n", k))
    cat(sprintf("  Size: %d\n", analysis$component_sizes[k]))
    cat(sprintf("  Mixing weight: %.3f\n", analysis$component_weights[k]))
    cat(sprintf("  Mean: [%.2f, %.2f]\n", 
                analysis$component_means[1, k],
                analysis$component_means[2, k]))
    cat(sprintf("  Average assignment probability: %.3f\n", analysis$component_uncertainty[k]))
    cat("  Covariance:\n")
    print(analysis$component_covariances[, , k])
  }
  
  list(mbc_optimal = mbc_optimal, X = X, results = results)
}

compare_covariance_types <- function() {
  """
  Compare different covariance types for GMM.
  """
  cat("=== Covariance Type Comparison ===\n\n")
  
  # Load data
  X <- load_old_faithful_data()
  
  covariance_types <- c("VVV", "VVI", "VII", "EEE")
  results <- list()
  
  for (cov_type in covariance_types) {
    cat(sprintf("Testing %s covariance...\n", cov_type))
    
    # Model selection
    mbc <- ModelBasedClustering$new(covariance_type = cov_type)
    model_results <- mbc$model_selection(X, K_range = 2:6)
    
    # Fit optimal model
    optimal_k <- model_results$optimal_bic_k
    mbc_optimal <- ModelBasedClustering$new(n_components = optimal_k, covariance_type = cov_type)
    mbc_optimal$fit(X)
    
    # Evaluate
    labels <- mbc_optimal$predict(X)
    silhouette <- mean(silhouette(labels, dist(X))[, 3])
    bic <- mbc_optimal$bic(X)
    
    results[[cov_type]] <- list(
      optimal_k = optimal_k,
      silhouette = silhouette,
      bic = bic,
      log_likelihood = mbc_optimal$score(X)
    )
    
    cat(sprintf("  Optimal K: %d\n", optimal_k))
    cat(sprintf("  Silhouette: %.3f\n", silhouette))
    cat(sprintf("  BIC: %.1f\n", bic))
    cat(sprintf("  Log-likelihood: %.1f\n", mbc_optimal$score(X)))
    
    # Plot
    mbc_optimal$plot_clusters(X, sprintf("Old Faithful Data - %s Covariance", cov_type))
  }
  
  # Summary
  cat("\n=== Summary ===\n")
  for (cov_type in names(results)) {
    result <- results[[cov_type]]
    cat(sprintf("%s: K=%d, Silhouette=%.3f, BIC=%.1f\n", 
                cov_type, result$optimal_k, result$silhouette, result$bic))
  }
  
  results
}

demonstrate_uncertainty_analysis <- function() {
  """
  Demonstrate uncertainty analysis in model-based clustering.
  """
  cat("=== Uncertainty Analysis ===\n\n")
  
  # Load data
  X <- load_old_faithful_data()
  
  # Fit models with different K
  models <- list()
  for (k in c(2, 3, 4)) {
    mbc <- ModelBasedClustering$new(n_components = k)
    mbc$fit(X)
    models[[as.character(k)]] <- mbc
  }
  
  # Analyze uncertainty
  uncertainty_data <- data.frame()
  
  for (k in c(2, 3, 4)) {
    mbc <- models[[as.character(k)]]
    probas <- mbc$predict_proba(X)
    max_proba <- apply(probas, 1, max)
    uncertainty <- 1 - max_proba
    
    uncertainty_data <- rbind(uncertainty_data, 
                             data.frame(
                               K = k,
                               Uncertainty = uncertainty
                             ))
    
    cat(sprintf("K=%d: Mean uncertainty = %.3f\n", k, mean(uncertainty)))
  }
  
  # Plot uncertainty distributions
  p1 <- ggplot(uncertainty_data, aes(x = Uncertainty, fill = factor(K))) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 20) +
    labs(title = "Uncertainty Distribution by K", x = "Uncertainty", y = "Frequency") +
    theme_minimal() +
    scale_fill_viridis_d(name = "K")
  
  print(p1)
  
  # Plot uncertainty in data space
  mbc_3 <- models[["3"]]
  probas_3 <- mbc_3$predict_proba(X)
  max_proba_3 <- apply(probas_3, 1, max)
  uncertainty_3 <- 1 - max_proba_3
  
  p2 <- ggplot(data.frame(x = X[, 1], y = X[, 2], uncertainty = uncertainty_3), 
               aes(x = x, y = y, color = uncertainty)) +
    geom_point(alpha = 0.7) +
    labs(title = "Uncertainty in Data Space (K=3)", x = "Duration", y = "Waiting Time") +
    theme_minimal() +
    scale_color_viridis_c()
  
  print(p2)
  
  models
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Basic demonstration
  cat("=== BASIC MODEL-BASED CLUSTERING DEMONSTRATION ===\n")
  results <- demonstrate_model_based_clustering()
  
  # Compare covariance types
  cat("\n=== COVARIANCE TYPE COMPARISON ===\n")
  covariance_results <- compare_covariance_types()
  
  # Uncertainty analysis
  cat("\n=== UNCERTAINTY ANALYSIS ===\n")
  uncertainty_models <- demonstrate_uncertainty_analysis()
}
