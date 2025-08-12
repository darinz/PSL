# Mixture Models Implementation in R
# ==================================
#
# This script provides comprehensive implementations of mixture models,
# including visualization, KL divergence computation, and the EM algorithm.

library(mixtools)
library(ggplot2)
library(gridExtra)

visualize_mixture_model <- function() {
  """
  Visualize a two-component Gaussian mixture model.
  """
  # Set random seed for reproducibility
  set.seed(42)

  # Parameters for a two-component Gaussian mixture
  mu1 <- 0; sigma1 <- 1
  mu2 <- 4; sigma2 <- 1.5
  pi1 <- 0.6; pi2 <- 1 - pi1

  # Generate data
  n_samples <- 1000
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(pi1, pi2))
  x <- numeric(n_samples)

  x[z == 0] <- rnorm(sum(z == 0), mu1, sigma1)
  x[z == 1] <- rnorm(sum(z == 1), mu2, sigma2)

  # Plot the mixture
  par(mfrow=c(1, 2))

  # Histogram of data
  hist(x, breaks=50, freq=FALSE, col="skyblue", border="black",
       main="Histogram of Mixture Data", xlab="x", ylab="Density")

  # True mixture density
  x_range <- seq(-3, 8, length.out=1000)
  true_density <- pi1 * dnorm(x_range, mu1, sigma1) + pi2 * dnorm(x_range, mu2, sigma2)
  lines(x_range, true_density, col="red", lwd=2)

  # Individual components
  plot(x_range, pi1 * dnorm(x_range, mu1, sigma1), type="l", lty=2, col="blue",
       main="Mixture Components", xlab="x", ylab="Density",
       ylim=c(0, max(true_density)))
  lines(x_range, pi2 * dnorm(x_range, mu2, sigma2), lty=2, col="green")
  lines(x_range, true_density, col="red", lwd=2)
  legend("topright", legend=c(paste("Component 1 (π=", pi1, ")"), 
                             paste("Component 2 (π=", pi2, ")"), "Mixture"),
         col=c("blue", "green", "red"), lty=c(2, 2, 1), lwd=c(1, 1, 2))

  cat("Generated", n_samples, "samples from mixture model\n")
  cat("Component 1: μ=", mu1, ", σ=", sigma1, ", π=", pi1, "\n")
  cat("Component 2: μ=", mu2, ", σ=", sigma2, ", π=", pi2, "\n")
}

TwoComponentGaussianMixture <- function(mu1=0, mu2=4, sigma1=1, sigma2=1.5, pi=0.6) {
  """
  Create a two-component Gaussian mixture model.
  
  Parameters:
  -----------
  mu1, mu2 : numeric
      Means of the two Gaussian components
  sigma1, sigma2 : numeric
      Standard deviations of the two Gaussian components
  pi : numeric
      Mixing weight for the first component
      
  Returns:
  --------
  gmm : list
      List containing the mixture model parameters
  """
  list(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, pi=pi)
}

generate_data <- function(gmm, n_samples=1000) {
  """
  Generate data from the mixture model.
  
  Parameters:
  -----------
  gmm : list
      Mixture model parameters
  n_samples : integer
      Number of samples to generate
      
  Returns:
  --------
  data : list
      List containing generated data and latent assignments
  """
  z <- sample(c(0, 1), size=n_samples, replace=TRUE, prob=c(gmm$pi, 1-gmm$pi))
  x <- numeric(n_samples)
  
  x[z == 0] <- rnorm(sum(z == 0), gmm$mu1, gmm$sigma1)
  x[z == 1] <- rnorm(sum(z == 1), gmm$mu2, gmm$sigma2)
  
  list(x=x, z=z)
}

pdf_mixture <- function(x, gmm) {
  """
  Compute the probability density function.
  
  Parameters:
  -----------
  x : numeric vector
      Points at which to evaluate the PDF
  gmm : list
      Mixture model parameters
      
  Returns:
  --------
  density : numeric vector
      Probability density values
  """
  gmm$pi * dnorm(x, gmm$mu1, gmm$sigma1) + (1-gmm$pi) * dnorm(x, gmm$mu2, gmm$sigma2)
}

plot_mixture <- function(x, gmm, z=NULL) {
  """
  Plot the mixture model and data.
  
  Parameters:
  -----------
  x : numeric vector
      Data to plot
  gmm : list
      Mixture model parameters
  z : numeric vector, optional
      True component assignments
  """
  # Create data frame for plotting
  df <- data.frame(x=x)
  x_range <- seq(min(x)-1, max(x)+1, length.out=1000)
  df_density <- data.frame(
    x=x_range,
    component1=gmm$pi * dnorm(x_range, gmm$mu1, gmm$sigma1),
    component2=(1-gmm$pi) * dnorm(x_range, gmm$mu2, gmm$sigma2),
    mixture=pdf_mixture(x_range, gmm)
  )
  
  # Plot 1: Histogram with true density
  p1 <- ggplot(df, aes(x=x)) +
    geom_histogram(aes(y=..density..), bins=50, fill="skyblue", alpha=0.7) +
    geom_line(data=df_density, aes(x=x, y=mixture), color="red", size=1) +
    labs(title="Data and True Mixture Density", x="x", y="Density") +
    theme_minimal()
  
  # Plot 2: Components
  p2 <- ggplot(df_density, aes(x=x)) +
    geom_line(aes(y=component1), color="blue", linetype="dashed") +
    geom_line(aes(y=component2), color="green", linetype="dashed") +
    geom_line(aes(y=mixture), color="red", size=1) +
    labs(title="Mixture Components", x="x", y="Density") +
    theme_minimal()
  
  gridExtra::grid.arrange(p1, p2, ncol=2)
}

demonstrate_two_component_mixture <- function() {
  """
  Demonstrate the two-component Gaussian mixture model.
  """
  # Example usage
  set.seed(42)
  gmm <- TwoComponentGaussianMixture()
  data <- generate_data(gmm, 1000)
  plot_mixture(data$x, gmm, data$z)

  # Fit using mixtools
  fit <- normalmixEM(data$x, k=2)
  print("True parameters:")
  cat("μ1=", gmm$mu1, ", μ2=", gmm$mu2, ", σ1=", gmm$sigma1, ", σ2=", gmm$sigma2, ", π=", gmm$pi, "\n")
  print("Estimated parameters:")
  cat("μ1=", round(fit$mu[1], 3), ", μ2=", round(fit$mu[2], 3), "\n")
  cat("σ1=", round(fit$sigma[1], 3), ", σ2=", round(fit$sigma[2], 3), "\n")
  cat("π1=", round(fit$lambda[1], 3), ", π2=", round(fit$lambda[2], 3), "\n")
}

kl_divergence_discrete <- function(p, q) {
  """
  Compute KL divergence for discrete distributions.
  
  Parameters:
  -----------
  p, q : numeric vectors
      Probability mass functions
      
  Returns:
  --------
  kl_div : numeric
      KL divergence KL(p||q)
  """
  # Normalize probabilities
  p <- p / sum(p)
  q <- q / sum(q)
  
  # Add small epsilon to avoid log(0)
  epsilon <- 1e-10
  p <- p + epsilon
  q <- q + epsilon
  
  sum(p * log(p / q))
}

kl_divergence_gaussian <- function(mu1, sigma1, mu2, sigma2) {
  """
  Compute KL divergence between two Gaussian distributions.
  
  Parameters:
  -----------
  mu1, mu2 : numeric
      Means of the two Gaussians
  sigma1, sigma2 : numeric
      Standard deviations of the two Gaussians
      
  Returns:
  --------
  kl_div : numeric
      KL divergence KL(N(mu1,sigma1)||N(mu2,sigma2))
  """
  log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2 * sigma2^2) - 0.5
}

demonstrate_kl_divergence <- function() {
  """
  Demonstrate KL divergence computation.
  """
  # Example
  mu1 <- 0; sigma1 <- 1
  mu2 <- 1; sigma2 <- 1.5

  kl_analytical <- kl_divergence_gaussian(mu1, sigma1, mu2, sigma2)

  cat("KL divergence between N(", mu1, ",", sigma1, ") and N(", mu2, ",", sigma2, ")\n")
  cat("Analytical:", round(kl_analytical, 6), "\n")

  # Visualize
  x <- seq(-4, 6, length.out=1000)
  p1 <- dnorm(x, mu1, sigma1)
  p2 <- dnorm(x, mu2, sigma2)

  df <- data.frame(x=x, p1=p1, p2=p2)
  p <- ggplot(df, aes(x=x)) +
    geom_line(aes(y=p1), color="blue", size=1) +
    geom_line(aes(y=p2), color="red", size=1) +
    geom_ribbon(aes(ymin=pmin(p1, p2), ymax=pmax(p1, p2)), alpha=0.3, fill="gray") +
    labs(title=paste("Gaussian Distributions (KL divergence:", round(kl_analytical, 4), ")"),
         x="x", y="Density") +
    theme_minimal()
  
  print(p)
}

demonstrate_em_algorithm <- function() {
  """
  Demonstrate the EM algorithm for fitting Gaussian mixture models.
  """
  # Generate data
  set.seed(42)
  gmm <- TwoComponentGaussianMixture()
  data <- generate_data(gmm, 1000)

  # Fit using mixtools EM
  fit <- normalmixEM(data$x, k=2, maxit=100, epsilon=1e-6)

  # Compare results
  cat("True parameters:\n")
  cat("μ1=", gmm$mu1, ", μ2=", gmm$mu2, "\n")
  cat("σ1=", gmm$sigma1, ", σ2=", gmm$sigma2, "\n")
  cat("π=", gmm$pi, "\n")

  cat("\nEM estimated parameters:\n")
  cat("μ1=", round(fit$mu[1], 3), ", μ2=", round(fit$mu[2], 3), "\n")
  cat("σ1=", round(fit$sigma[1], 3), ", σ2=", round(fit$sigma[2], 3), "\n")
  cat("π1=", round(fit$lambda[1], 3), ", π2=", round(fit$lambda[2], 3), "\n")

  # Visualize results
  par(mfrow=c(1, 2))

  # Data and fitted mixture
  hist(data$x, breaks=50, freq=FALSE, col="skyblue", border="black",
       main="Data and Mixture Densities", xlab="x", ylab="Density")

  x_range <- seq(min(data$x)-1, max(data$x)+1, length.out=1000)
  true_density <- pdf_mixture(x_range, gmm)
  lines(x_range, true_density, col="red", lwd=2)

  fitted_density <- fit$lambda[1] * dnorm(x_range, fit$mu[1], fit$sigma[1]) +
                   fit$lambda[2] * dnorm(x_range, fit$mu[2], fit$sigma[2])
  lines(x_range, fitted_density, col="green", lty=2, lwd=2)

  legend("topright", legend=c("True Mixture", "Fitted Mixture"),
         col=c("red", "green"), lty=c(1, 2), lwd=2)

  # Responsibilities
  plot(data$x, fit$posterior[,1], pch=16, cex=0.5, col="blue",
       main="Responsibilities (Component 1)", xlab="x", ylab="P(Z=1|x)")
}

fit_multiple_initializations <- function(x, n_components=2, n_init=10) {
  """
  Fit GMM with multiple initializations and return the best result.
  
  Parameters:
  -----------
  x : numeric vector
      Training data
  n_components : integer
      Number of mixture components
  n_init : integer
      Number of different initializations to try
      
  Returns:
  --------
  best_fit : list
      Best fitted model
  best_log_likelihood : numeric
      Best log-likelihood achieved
  """
  best_log_likelihood <- -Inf
  best_fit <- NULL
  
  for (i in 1:n_init) {
    tryCatch({
      fit <- normalmixEM(x, k=n_components, maxit=100, epsilon=1e-6)
      log_likelihood <- sum(log(fit$lambda[1] * dnorm(x, fit$mu[1], fit$sigma[1]) +
                               fit$lambda[2] * dnorm(x, fit$mu[2], fit$sigma[2])))
      
      if (log_likelihood > best_log_likelihood) {
        best_log_likelihood <- log_likelihood
        best_fit <- fit
      }
    }, error = function(e) {
      # Skip failed fits
    })
  }
  
  list(fit=best_fit, log_likelihood=best_log_likelihood)
}

demonstrate_multiple_initializations <- function() {
  """
  Demonstrate multiple initializations for robust EM fitting.
  """
  set.seed(42)
  
  # Generate data
  gmm <- TwoComponentGaussianMixture()
  data <- generate_data(gmm, 1000)
  
  # Fit with multiple initializations
  result <- fit_multiple_initializations(data$x, n_components=2, n_init=10)
  cat("Best log-likelihood:", round(result$log_likelihood, 3), "\n")
  
  return(result)
}

# Main execution function
if (FALSE) {  # Set to TRUE to run demonstrations
  # Basic mixture model visualization
  cat("=== BASIC MIXTURE MODEL VISUALIZATION ===\n")
  visualize_mixture_model()
  
  # Two-component mixture demonstration
  cat("\n=== TWO-COMPONENT MIXTURE MODEL ===\n")
  demonstrate_two_component_mixture()
  
  # KL divergence demonstration
  cat("\n=== KL DIVERGENCE COMPUTATION ===\n")
  demonstrate_kl_divergence()
  
  # EM algorithm demonstration
  cat("\n=== EM ALGORITHM ===\n")
  demonstrate_em_algorithm()
  
  # Multiple initializations
  cat("\n=== MULTIPLE INITIALIZATIONS ===\n")
  result <- demonstrate_multiple_initializations()
}
