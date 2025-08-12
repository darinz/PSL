# EM Algorithm Implementation in R
# ================================
#
# This script provides comprehensive implementations of the Expectation-Maximization
# algorithm, including basic EM, convergence monitoring, K-means comparison,
# and variational EM variants.

library(mixtools)
library(cluster)
library(ggplot2)
library(gridExtra)

EMAlgorithm <- function(max_iter=100, tol=1e-6) {
  """
  Create an EM algorithm object.
  
  Parameters:
  -----------
  max_iter : integer
      Maximum number of iterations
  tol : numeric
      Convergence tolerance
      
  Returns:
  --------
  em : list
      EM algorithm object
  """
  list(max_iter=max_iter, tol=tol)
}

fit_em <- function(em, x, initial_params=NULL) {
  """
  Fit the model using EM algorithm.
  
  Parameters:
  -----------
  em : list
      EM algorithm object
  x : numeric vector
      Training data
  initial_params : list, optional
      Initial parameters for the model
      
  Returns:
  --------
  result : list
      Fitted model results
  """
  n_samples <- length(x)
  
  # Initialize parameters
  if (is.null(initial_params)) {
    params <- list(
      means = sample(x, 2, replace=FALSE),
      variances = rep(var(x), 2),
      weights = rdirichlet(1, c(1, 1))[1,]
    )
  } else {
    params <- initial_params
  }
  
  log_likelihoods <- numeric(0)
  
  for (iteration in 1:em$max_iter) {
    # E-step: Compute responsibilities
    responsibilities <- matrix(0, n_samples, 2)
    
    for (k in 1:2) {
      responsibilities[, k] <- params$weights[k] * dnorm(x, params$means[k], sqrt(params$variances[k]))
    }
    
    # Normalize
    row_sums <- rowSums(responsibilities)
    responsibilities <- responsibilities / row_sums
    
    # M-step: Update parameters
    for (k in 1:2) {
      # Update weights
      params$weights[k] <- mean(responsibilities[, k])
      
      # Update means
      params$means[k] <- sum(responsibilities[, k] * x) / sum(responsibilities[, k])
      
      # Update variances
      params$variances[k] <- sum(responsibilities[, k] * (x - params$means[k])^2) / sum(responsibilities[, k])
    }
    
    # Compute log-likelihood
    likelihood <- rep(0, n_samples)
    for (k in 1:2) {
      likelihood <- likelihood + params$weights[k] * dnorm(x, params$means[k], sqrt(params$variances[k]))
    }
    log_likelihood <- sum(log(likelihood + 1e-10))
    log_likelihoods <- c(log_likelihoods, log_likelihood)
    
    # Check convergence
    if (length(log_likelihoods) > 1) {
      if (abs(log_likelihoods[length(log_likelihoods)] - log_likelihoods[length(log_likelihoods)-1]) < em$tol) {
        cat("Converged after", iteration, "iterations\n")
        break
      }
    }
  }
  
  list(params=params, log_likelihoods=log_likelihoods, responsibilities=responsibilities)
}

demonstrate_basic_em <- function() {
  """
  Demonstrate the basic EM algorithm.
  """
  set.seed(42)

  # Generate data
  n_samples <- 1000
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(0.6, 0.4))
  x <- numeric(n_samples)

  x[z == 0] <- rnorm(sum(z == 0), 0, 1)
  x[z == 1] <- rnorm(sum(z == 1), 4, 1.5)

  # Fit using EM
  em <- EMAlgorithm(max_iter=100, tol=1e-6)
  result <- fit_em(em, x)

  # Compare with mixtools
  fit <- normalmixEM(x, k=2, maxit=100, epsilon=1e-6)

  cat("EM Algorithm Results:\n")
  cat("Means:", result$params$means, "\n")
  cat("Variances:", result$params$variances, "\n")
  cat("Weights:", result$params$weights, "\n")

  cat("\nMixtools Results:\n")
  cat("Means:", fit$mu, "\n")
  cat("Variances:", fit$sigma^2, "\n")
  cat("Weights:", fit$lambda, "\n")

  # Plot convergence
  plot(result$log_likelihoods, type="l", main="EM Algorithm Convergence",
       xlab="Iteration", ylab="Log-Likelihood")
  grid()
  
  list(em_result=result, x=x, mixtools_fit=fit)
}

monitor_em_convergence <- function(x, n_components=2, n_runs=5) {
  """
  Monitor EM convergence across multiple runs.
  
  Parameters:
  -----------
  x : numeric vector
      Training data
  n_components : integer
      Number of mixture components
  n_runs : integer
      Number of runs to perform
      
  Returns:
  --------
  results : list
      List of results from each run
  """
  results <- list()
  
  for (run in 1:n_runs) {
    em <- EMAlgorithm(max_iter=200, tol=1e-8)
    result <- fit_em(em, x)
    
    results[[run]] <- list(
      run = run,
      final_ll = result$log_likelihoods[length(result$log_likelihoods)],
      iterations = length(result$log_likelihoods),
      params = result$params,
      log_likelihoods = result$log_likelihoods
    )
  }
  
  results
}

demonstrate_convergence_monitoring <- function() {
  """
  Demonstrate convergence monitoring across multiple runs.
  """
  set.seed(42)
  
  # Generate data
  n_samples <- 1000
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(0.6, 0.4))
  x <- numeric(n_samples)
  x[z == 0] <- rnorm(sum(z == 0), 0, 1)
  x[z == 1] <- rnorm(sum(z == 1), 4, 1.5)

  # Monitor convergence
  convergence_results <- monitor_em_convergence(x, n_components=2, n_runs=5)

  cat("Convergence Results:\n")
  for (result in convergence_results) {
    cat(sprintf("Run %d: Final LL = %.3f, Iterations = %d\n", 
                result$run, result$final_ll, result$iterations))
  }

  # Plot convergence for all runs
  plot(convergence_results[[1]]$log_likelihoods, type="l", 
       main="EM Algorithm Convergence (Multiple Runs)",
       xlab="Iteration", ylab="Log-Likelihood", col=1)
  
  for (i in 2:length(convergence_results)) {
    lines(convergence_results[[i]]$log_likelihoods, col=i)
  }
  
  legend("bottomright", legend=paste("Run", 1:length(convergence_results)), 
         col=1:length(convergence_results), lty=1)
  grid()
  
  convergence_results
}

compare_em_kmeans <- function(x, n_components=2) {
  """
  Compare EM algorithm with K-means.
  
  Parameters:
  -----------
  x : numeric vector
      Training data
  n_components : integer
      Number of components/clusters
      
  Returns:
  --------
  comparison : list
      Comparison results
  """
  # EM Algorithm
  em <- EMAlgorithm(max_iter=100, tol=1e-6)
  em_result <- fit_em(em, x)
  
  # K-means
  kmeans_result <- kmeans(x, centers=n_components, nstart=10)
  
  # Compare results
  cat("EM Algorithm Results:\n")
  cat("Means:", em_result$params$means, "\n")
  cat("Variances:", em_result$params$variances, "\n")
  cat("Weights:", em_result$params$weights, "\n")
  
  cat("\nK-means Results:\n")
  cat("Centers:", kmeans_result$centers, "\n")
  cat("Within SS:", kmeans_result$withinss, "\n")
  
  # Compare assignments
  em_labels <- apply(em_result$responsibilities, 1, which.max) - 1
  ari_score <- adjustedRandIndex(em_labels, kmeans_result$cluster)
  cat("Adjusted Rand Index:", round(ari_score, 3), "\n")
  
  list(em=em_result, kmeans=kmeans_result, em_labels=em_labels, kmeans_labels=kmeans_result$cluster)
}

demonstrate_em_kmeans_comparison <- function() {
  """
  Demonstrate comparison between EM algorithm and K-means.
  """
  set.seed(42)
  
  # Generate data
  n_samples <- 1000
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(0.6, 0.4))
  x <- numeric(n_samples)
  x[z == 0] <- rnorm(sum(z == 0), 0, 1)
  x[z == 1] <- rnorm(sum(z == 1), 4, 1.5)

  # Compare EM and K-means
  comparison <- compare_em_kmeans(x, n_components=2)

  # Visualize results
  par(mfrow=c(2, 2))

  # EM responsibilities
  plot(x, comparison$em$responsibilities[,1], col=comparison$em_labels+1, pch=16,
       main="EM Algorithm: Responsibilities", xlab="x", ylab="P(Z=1|x)")

  # K-means assignments
  plot(x, rep(0, length(x)), col=comparison$kmeans_labels, pch=16,
       main="K-means: Hard Assignments", xlab="x", ylab="Cluster")

  # Histogram comparison
  hist(x[comparison$em_labels == 0], breaks=30, col="red", alpha=0.7, 
       main="EM Algorithm Clusters", xlab="x", freq=FALSE)
  hist(x[comparison$em_labels == 1], breaks=30, col="blue", alpha=0.7, add=TRUE)

  hist(x[comparison$kmeans_labels == 0], breaks=30, col="red", alpha=0.7,
       main="K-means Clusters", xlab="x", freq=FALSE)
  hist(x[comparison$kmeans_labels == 1], breaks=30, col="blue", alpha=0.7, add=TRUE)
  
  comparison
}

VariationalEM <- function(max_iter=100, tol=1e-6) {
  """
  Create a variational EM algorithm object.
  
  Parameters:
  -----------
  max_iter : integer
      Maximum number of iterations
  tol : numeric
      Convergence tolerance
      
  Returns:
  --------
  vem : list
      Variational EM algorithm object
  """
  list(max_iter=max_iter, tol=tol)
}

fit_variational_em <- function(vem, x, initial_params=NULL) {
  """
  Fit using variational EM.
  
  Parameters:
  -----------
  vem : list
      Variational EM algorithm object
  x : numeric vector
      Training data
  initial_params : list, optional
      Initial parameters
      
  Returns:
  --------
  result : list
      Fitted model results
  """
  n_samples <- length(x)
  
  # Initialize parameters
  if (is.null(initial_params)) {
    params <- list(
      means = sample(x, 2, replace=FALSE),
      variances = rep(var(x), 2),
      weights = rdirichlet(1, c(1, 1))[1,]
    )
  } else {
    params <- initial_params
  }
  
  # Initialize variational distribution
  q <- matrix(0.5, n_samples, 2)
  free_energies <- numeric(0)
  
  for (iteration in 1:vem$max_iter) {
    # E-step: Update variational distribution
    for (i in 1:n_samples) {
      # Compute unnormalized responsibilities
      log_resp <- numeric(2)
      for (k in 1:2) {
        log_resp[k] <- log(params$weights[k]) + 
                      dnorm(x[i], params$means[k], sqrt(params$variances[k]), log=TRUE)
      }
      
      # Normalize using log-sum-exp trick
      max_log <- max(log_resp)
      exp_log_resp <- exp(log_resp - max_log)
      q[i, ] <- exp_log_resp / sum(exp_log_resp)
    }
    
    # M-step: Update parameters
    for (k in 1:2) {
      # Update weights
      params$weights[k] <- mean(q[, k])
      
      # Update means
      params$means[k] <- sum(q[, k] * x) / sum(q[, k])
      
      # Update variances
      params$variances[k] <- sum(q[, k] * (x - params$means[k])^2) / sum(q[, k])
    }
    
    # Compute free energy
    free_energy <- 0
    for (i in 1:n_samples) {
      for (k in 1:2) {
        if (q[i, k] > 0) {
          # Log-likelihood term
          log_likelihood <- log(params$weights[k]) + 
                           dnorm(x[i], params$means[k], sqrt(params$variances[k]), log=TRUE)
          
          # Entropy term
          entropy <- -log(q[i, k])
          
          free_energy <- free_energy + q[i, k] * (log_likelihood - entropy)
        }
      }
    }
    free_energies <- c(free_energies, free_energy)
    
    # Check convergence
    if (length(free_energies) > 1) {
      if (abs(free_energies[length(free_energies)] - free_energies[length(free_energies)-1]) < vem$tol) {
        cat("Converged after", iteration, "iterations\n")
        break
      }
    }
  }
  
  list(params=params, free_energies=free_energies, q=q)
}

FactorizedVariationalEM <- function(max_iter=100, tol=1e-6) {
  """
  Create a factorized variational EM algorithm object.
  
  Parameters:
  -----------
  max_iter : integer
      Maximum number of iterations
  tol : numeric
      Convergence tolerance
      
  Returns:
  --------
  fvem : list
      Factorized variational EM algorithm object
  """
  list(max_iter=max_iter, tol=tol)
}

fit_factorized_variational_em <- function(fvem, x, initial_params=NULL) {
  """
  Fit using factorized variational EM.
  
  Parameters:
  -----------
  fvem : list
      Factorized variational EM algorithm object
  x : numeric vector
      Training data
  initial_params : list, optional
      Initial parameters
      
  Returns:
  --------
  result : list
      Fitted model results
  """
  n_samples <- length(x)
  
  # Initialize parameters
  if (is.null(initial_params)) {
    params <- list(
      means = sample(x, 2, replace=FALSE),
      variances = rep(var(x), 2),
      weights = rdirichlet(1, c(1, 1))[1,]
    )
  } else {
    params <- initial_params
  }
  
  # Initialize factorized variational distribution
  q_factors <- matrix(0.5, n_samples, 2)
  free_energies <- numeric(0)
  
  for (iteration in 1:fvem$max_iter) {
    # Update each factor independently
    for (i in 1:n_samples) {
      # Compute expected log-likelihood for each component
      expected_log_likelihood <- numeric(2)
      
      for (k in 1:2) {
        # Prior term
        expected_log_likelihood[k] <- log(params$weights[k])
        
        # Likelihood term
        expected_log_likelihood[k] <- expected_log_likelihood[k] + 
                                    dnorm(x[i], params$means[k], sqrt(params$variances[k]), log=TRUE)
      }
      
      # Update factor using softmax
      max_log <- max(expected_log_likelihood)
      exp_log <- exp(expected_log_likelihood - max_log)
      q_factors[i, ] <- exp_log / sum(exp_log)
    }
    
    # Update parameters
    for (k in 1:2) {
      # Update weights
      params$weights[k] <- mean(q_factors[, k])
      
      # Update means
      params$means[k] <- sum(q_factors[, k] * x) / sum(q_factors[, k])
      
      # Update variances
      params$variances[k] <- sum(q_factors[, k] * (x - params$means[k])^2) / sum(q_factors[, k])
    }
    
    # Compute free energy
    free_energy <- 0
    for (i in 1:n_samples) {
      for (k in 1:2) {
        if (q_factors[i, k] > 0) {
          # Expected log-likelihood
          expected_ll <- log(params$weights[k]) + 
                        dnorm(x[i], params$means[k], sqrt(params$variances[k]), log=TRUE)
          
          # Entropy of factor
          entropy <- -log(q_factors[i, k])
          
          free_energy <- free_energy + q_factors[i, k] * (expected_ll - entropy)
        }
      }
    }
    free_energies <- c(free_energies, free_energy)
    
    # Check convergence
    if (length(free_energies) > 1) {
      if (abs(free_energies[length(free_energies)] - free_energies[length(free_energies)-1]) < fvem$tol) {
        cat("Converged after", iteration, "iterations\n")
        break
      }
    }
  }
  
  list(params=params, free_energies=free_energies, q_factors=q_factors)
}

demonstrate_variational_em <- function() {
  """
  Demonstrate variational EM algorithms.
  """
  set.seed(42)
  
  # Generate data
  n_samples <- 1000
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(0.6, 0.4))
  x <- numeric(n_samples)
  x[z == 0] <- rnorm(sum(z == 0), 0, 1)
  x[z == 1] <- rnorm(sum(z == 1), 4, 1.5)

  # Standard EM
  em <- EMAlgorithm(max_iter=100, tol=1e-6)
  em_result <- fit_em(em, x)

  # Variational EM
  vem <- VariationalEM(max_iter=100, tol=1e-6)
  vem_result <- fit_variational_em(vem, x)

  # Factorized Variational EM
  fvem <- FactorizedVariationalEM(max_iter=100, tol=1e-6)
  fvem_result <- fit_factorized_variational_em(fvem, x)

  cat("Standard EM Results:\n")
  cat("Means:", em_result$params$means, "\n")
  cat("Variances:", em_result$params$variances, "\n")
  cat("Weights:", em_result$params$weights, "\n")

  cat("\nVariational EM Results:\n")
  cat("Means:", vem_result$params$means, "\n")
  cat("Variances:", vem_result$params$variances, "\n")
  cat("Weights:", vem_result$params$weights, "\n")

  cat("\nFactorized Variational EM Results:\n")
  cat("Means:", fvem_result$params$means, "\n")
  cat("Variances:", fvem_result$params$variances, "\n")
  cat("Weights:", fvem_result$params$weights, "\n")

  # Plot free energy convergence
  plot(vem_result$free_energies, type="l", col="blue", 
       main="Variational EM: Free Energy Convergence",
       xlab="Iteration", ylab="Free Energy")
  lines(fvem_result$free_energies, col="red")
  legend("bottomright", legend=c("Variational EM", "Factorized VEM"), 
         col=c("blue", "red"), lty=1)
  grid()

  # Compare different EM variants
  em_variants <- list(
    "Standard EM" = em_result,
    "Variational EM" = vem_result,
    "Factorized VEM" = fvem_result
  )

  cat("\nComparison of EM Variants:\n")
  for (name in names(em_variants)) {
    variant <- em_variants[[name]]
    if ("log_likelihoods" %in% names(variant)) {
      cat(sprintf("%s: Final LL = %.3f\n", name, variant$log_likelihoods[length(variant$log_likelihoods)]))
    } else {
      cat(sprintf("%s: Final Free Energy = %.3f\n", name, variant$free_energies[length(variant$free_energies)]))
    }
  }
  
  list(em=em_result, vem=vem_result, fvem=fvem_result, x=x)
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  cat("Demonstrating EM Algorithm...\n")
  
  # Basic EM demonstration
  cat("\n1. Basic EM Algorithm\n")
  basic_result <- demonstrate_basic_em()
  
  # Convergence monitoring
  cat("\n2. Convergence Monitoring\n")
  convergence_results <- demonstrate_convergence_monitoring()
  
  # EM vs K-means comparison
  cat("\n3. EM vs K-means Comparison\n")
  comparison <- demonstrate_em_kmeans_comparison()
  
  # Variational EM demonstration
  cat("\n4. Variational EM\n")
  variational_result <- demonstrate_variational_em()
}
